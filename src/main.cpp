#include "person/app.hpp"
#include "person/pipeline.hpp"
#include <csignal>
#include <fstream>
#include <iostream>
#include <mutex>
#include <pthread.h>
#include <sys/resource.h>
#include <thread>

namespace person {
void log(const std::string& message) {static std::mutex mutex; std::lock_guard<std::mutex> lock(mutex); std::cerr<<message<<'\n';}
namespace {
void help() {
    std::cout<<"person_detect --input PATH [options]\n"
      "  --model PATH        RK3568 model (default ../model/best.rknn relative to executable)\n"
      "  --workers N         fixed workers, default 1\n"
      "  --inflight N        total frame slots, default 2*workers+2\n"
      "  --preprocess MODE   auto | opencv | rga\n"
      "  --io MODE           auto | standard | bound\n"
      "  --conf FLOAT        default 0.25\n"
      "  --nms FLOAT         default 0.45\n"
      "  --jsonl PATH        default detections.jsonl\n"
      "  --annotated PATH    image output directory or annotated video file\n"
      "  --summary PATH      performance summary JSON\n"
      "  --warmup N          independent warmup runs per worker, default 3\n"
      "  --dump-first DIR    dump first frame input and raw outputs for reference validation\n";
}
size_t positive(const std::string& text) {
    size_t used=0; const auto value=std::stoull(text,&used);
    if(used!=text.size() || text.front()=='-' || !value || value>1024) throw std::invalid_argument("Invalid positive integer: "+text);
    return size_t(value);
}
float number(const std::string& text) {
    size_t used=0; float value=std::stof(text,&used);
    if(used!=text.size() || !std::isfinite(value)) throw std::invalid_argument("Invalid finite number: "+text);
    return value;
}
Options parse(int argc,char** argv) {
    Options o; o.model=std::filesystem::canonical("/proc/self/exe").parent_path()/"../model/best.rknn";
    for(int i=1;i<argc;++i) {
        const std::string key=argv[i];
        if(key=="--help" || key=="-h") {help(); std::exit(0);}
        if(i+1==argc) throw std::invalid_argument("Missing value for "+key);
        const std::string value=argv[++i];
        if(key=="--input") o.input=value;
        else if(key=="--model") o.model=value;
        else if(key=="--workers") o.workers=positive(value);
        else if(key=="--inflight") o.capacity=positive(value);
        else if(key=="--preprocess") o.preprocess=value;
        else if(key=="--io") o.io=value;
        else if(key=="--conf") o.confidence=number(value);
        else if(key=="--nms") o.nms=number(value);
        else if(key=="--jsonl") o.jsonl=value;
        else if(key=="--annotated") o.annotated=value;
        else if(key=="--summary") o.summary=value;
        else if(key=="--dump-first") o.dump=value;
        else if(key=="--warmup") {
            size_t used=0; o.warmup=std::stoi(value,&used);
            if(used!=value.size() || o.warmup<0 || o.warmup>1000) throw std::invalid_argument("Invalid warmup integer");
        }
        else throw std::invalid_argument("Unknown argument: "+key);
    }
    if(o.input.empty()) throw std::invalid_argument("--input is required");
    if(o.workers>32) throw std::invalid_argument("Workers must be between 1 and 32");
    if(!o.capacity) o.capacity=2*o.workers+2;
    if(o.preprocess!="auto" && o.preprocess!="opencv" && o.preprocess!="rga") throw std::invalid_argument("Invalid --preprocess mode");
    if(o.io!="auto" && o.io!="standard" && o.io!="bound") throw std::invalid_argument("Invalid --io mode");
    Postprocessor validation(o.confidence,o.nms);
    // Reject output aliases before a sink can truncate an input or model.
    auto absolute=[](const std::filesystem::path& p){return std::filesystem::weakly_canonical(std::filesystem::absolute(p));};
    std::vector<std::filesystem::path> outputs;
    for(const auto& p:{o.jsonl,o.summary,o.annotated}) if(!p.empty()) {
        if(absolute(p)==absolute(o.input) || absolute(p)==absolute(o.model)) throw std::invalid_argument("Output path aliases input/model: "+p.string());
        for(const auto& other:outputs) if(absolute(p)==absolute(other)) throw std::invalid_argument("Output paths must differ");
        outputs.push_back(p);
    }
    return o;
}
struct Worker {
    std::unique_ptr<InferenceEngine> engine; std::unique_ptr<Preprocessor> preprocess; Postprocessor post;
    explicit Worker(const Options& o,InferenceEngine* shared):engine(EngineFactory::create(o,shared)),
        preprocess(PreprocessorFactory::create(o.preprocess)),post(o.confidence,o.nms) {}
    FrameResult detect(Frame& f,const Options& o) {
        FrameResult result; auto start=Clock::now(); result.timing.read=f.read_ms; result.timing.queue=milliseconds(f.ready,start);
        const auto geometry=preprocess->process(f.image,engine->input()); auto prepared=Clock::now();
        const auto tensors=engine->infer(); const auto inferred=Clock::now();
        try {result.detections=post.decode(tensors,geometry); if(f.id==0 && !o.dump.empty()) engine->dump(o.dump,geometry);}
        catch(...) {engine->release_outputs(); throw;}
        engine->release_outputs(); auto finished=Clock::now();
        result.timing.preprocess=milliseconds(start,prepared); result.timing.inference=milliseconds(prepared,inferred);
        result.timing.postprocess=milliseconds(inferred,finished); result.detected=finished;
        result.preprocessing=preprocess->name(); result.io=engine->name(); return result;
    }
    void warmup(int count) {
        // Synthetic data does not determine auto RGA selection for the real input geometry.
        auto b=engine->input(); cv::Mat input(416,416,CV_8UC3,b.data,b.row_bytes); input.setTo(cv::Scalar(0,0,0));
        for(int i=0;i<count;++i) {const auto outputs=engine->infer(); post.decode(outputs,letterbox(416,416)); engine->release_outputs();}
    }
};
class SignalMask {
public:
    SignalMask() {
        sigset_t set; sigemptyset(&set); sigaddset(&set,SIGINT); sigaddset(&set,SIGTERM); sigaddset(&set,SIGUSR1);
        if(pthread_sigmask(SIG_BLOCK,&set,&previous_)!=0) throw std::runtime_error("Cannot block termination signals");
    }
    ~SignalMask(){pthread_sigmask(SIG_SETMASK,&previous_,nullptr);}
private: sigset_t previous_{};
};
class Signals {
public:
    explicit Signals(std::function<void(int)> cancel) {
        sigemptyset(&set_); sigaddset(&set_,SIGINT); sigaddset(&set_,SIGTERM); sigaddset(&set_,SIGUSR1);
        if(pthread_sigmask(SIG_BLOCK,&set_,&previous_)!=0) throw std::runtime_error("Cannot block termination signals");
        try {thread_=std::thread([this,cancel]{int signal=0; if(sigwait(&set_,&signal)==0 && signal!=SIGUSR1) cancel(signal);});}
        catch(...) {pthread_sigmask(SIG_SETMASK,&previous_,nullptr); throw;}
    }
    ~Signals(){if(thread_.joinable()){pthread_kill(thread_.native_handle(),SIGUSR1); thread_.join();} pthread_sigmask(SIG_SETMASK,&previous_,nullptr);}
private: sigset_t set_{},previous_{}; std::thread thread_;
};
}
int run(int argc,char** argv) {
    const auto options=parse(argc,argv); SignalMask mask; cv::setNumThreads(1);
    auto source=InputSourceFactory::create(options.input);
    std::vector<std::unique_ptr<Worker>> workers;
    for(size_t i=0;i<options.workers;++i) workers.push_back(std::make_unique<Worker>(options,i?workers[0]->engine.get():nullptr));
    for(auto& worker:workers) worker->warmup(options.warmup);
    OutputSink output(options,source->video(),source->fps());
    using Pipeline=OrderedPipeline<Frame,FrameResult>;
    std::vector<Pipeline::Worker> functions;
    std::mutex compute_mutex; Clock::time_point compute_begin{},compute_end{};
    for(auto& worker:workers) functions.emplace_back([&,w=worker.get()](Frame& f){
        auto begin=Clock::now(); auto result=w->detect(f,options); auto end=Clock::now();
        {std::lock_guard<std::mutex> lock(compute_mutex); if(compute_begin==Clock::time_point{} || begin<compute_begin) compute_begin=begin; compute_end=std::max(compute_end,end);}
        return result;
    });
    Pipeline pipeline(std::move(functions),options.capacity);
    Signals signals([&](int signal){pipeline.cancel("Interrupted by signal "+std::to_string(signal));});
    auto histograms=std::make_unique<std::array<Histogram,8>>(); uint64_t detections=0;
    const auto begin=Clock::now();
    const auto stats=pipeline.run([&](Frame& f,size_t id){return source->read(f,id);},[&](Frame& f,FrameResult& r){
        const auto writing=Clock::now(); r.timing.result_wait=milliseconds(r.detected,writing);
        output.write(f,r); const auto written=Clock::now();
        r.timing.output=milliseconds(writing,written); r.timing.latency=milliseconds(f.started,written);
        const std::array<double,8> values{r.timing.read,r.timing.queue,r.timing.preprocess,r.timing.inference,r.timing.postprocess,r.timing.result_wait,r.timing.output,r.timing.latency};
        for(size_t i=0;i<8;++i) (*histograms)[i].add(values[i]);
        detections+=r.detections.size();
    });
    output.finish(); const auto end=Clock::now(); const double elapsed=milliseconds(begin,end)/1000;
    const double compute_seconds=stats.completed?milliseconds(compute_begin,compute_end)/1000:0;
    struct rusage usage{}; getrusage(RUSAGE_SELF,&usage);
    std::ostringstream report; report.precision(9);
    report<<"{\"frames\":"<<stats.completed<<",\"submitted\":"<<stats.submitted<<",\"detections\":"<<detections
        <<",\"workers\":"<<options.workers<<",\"preprocess_requested\":"<<json_string(options.preprocess)<<",\"io_requested\":"<<json_string(options.io)
        <<",\"annotated\":"<<(!options.annotated.empty()?"true":"false")<<",\"total_seconds\":"<<elapsed
        <<",\"end_to_end_fps\":"<<(elapsed>0?stats.completed/elapsed:0)<<",\"detection_fps\":"<<(compute_seconds>0?stats.completed/compute_seconds:0)
        <<",\"rss_peak_kib\":"<<usage.ru_maxrss<<",\"capacity\":"<<stats.capacity<<",\"inflight_peak\":"<<stats.inflight_peak
        <<",\"task_queue_peak\":"<<stats.task_peak<<",\"result_queue_peak\":"<<stats.result_peak<<",\"quantile_resolution_ms\":1,\"stages_ms\":{";
    const char* names[]={"read","queue","preprocess","inference","postprocess","result_wait","output","latency"};
    for(size_t i=0;i<8;++i) {if(i) report<<','; const auto& h=(*histograms)[i]; report<<json_string(names[i])<<":{\"mean\":"<<h.mean()<<",\"p50\":"<<h.percentile(.5)<<",\"p95\":"<<h.percentile(.95)<<'}';}
    report<<"},\"worker_modes\":[";
    for(size_t i=0;i<workers.size();++i) {if(i) report<<','; report<<"{\"io\":"<<json_string(workers[i]->engine->name())<<",\"preprocess\":"<<json_string(workers[i]->preprocess->name())<<'}';}
    report<<"]}";
    if(!options.summary.empty()) {std::ofstream file(options.summary); file<<report.str()<<'\n'; if(!file) throw std::runtime_error("Cannot write performance summary");}
    std::cout<<report.str()<<'\n'; return 0;
}
}
int main(int argc,char** argv) {try{return person::run(argc,argv);} catch(const std::exception& e){std::cerr<<"ERROR: "<<e.what()<<'\n';return 1;}}
