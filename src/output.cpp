#include "person/app.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace person {
struct OutputSink::Impl {
    std::ofstream json; std::filesystem::path annotation; bool video{}; double fps{};
    cv::VideoWriter writer; cv::Size size; bool finished{};
};
OutputSink::OutputSink(const Options& options,bool video,double fps):impl_(std::make_unique<Impl>()) {
    impl_->json.open(options.jsonl); if(!impl_->json) throw std::runtime_error("Cannot open JSONL: "+options.jsonl.string());
    impl_->annotation=options.annotated; impl_->video=video; impl_->fps=fps;
    if(!options.annotated.empty() && !video) std::filesystem::create_directories(options.annotated);
}
OutputSink::~OutputSink()=default;
void OutputSink::write(Frame& f,FrameResult& result) {
    const auto begin=Clock::now();
    if(!impl_->annotation.empty()) {
        for(const auto& d:result.detections) {
            cv::rectangle(f.image,cv::Point(int(d.box.x1),int(d.box.y1)),cv::Point(int(d.box.x2),int(d.box.y2)),cv::Scalar(0,255,0),2);
            std::ostringstream label; label<<"person "<<std::fixed<<std::setprecision(2)<<d.score;
            cv::putText(f.image,label.str(),cv::Point(int(d.box.x1),std::max(15,int(d.box.y1)-5)),cv::FONT_HERSHEY_SIMPLEX,.5,cv::Scalar(0,255,0),1);
        }
        if(impl_->video) {
            if(!impl_->writer.isOpened()) {
                impl_->size=f.image.size();
                if(!impl_->writer.open(impl_->annotation.string(),cv::CAP_FFMPEG,cv::VideoWriter::fourcc('m','p','4','v'),impl_->fps,impl_->size,true))
                    throw std::runtime_error("Cannot open annotated video (FFmpeg MPEG-4 encoder required): "+impl_->annotation.string());
            }
            if(f.image.size()!=impl_->size) throw std::runtime_error("Video frame dimensions changed during encoding");
            impl_->writer.write(f.image);
        } else {
            std::ostringstream name; name<<std::setw(9)<<std::setfill('0')<<f.id<<"_"<<std::filesystem::path(f.source).stem().string()<<".jpg";
            if(!cv::imwrite((impl_->annotation/name.str()).string(),f.image)) throw std::runtime_error("Cannot save annotated image");
        }
    }
    // Submit the detection portion before recording time, so output_ms includes its stream write.
    std::ostringstream row; row<<std::setprecision(8)<<"{\"frame_id\":"<<f.id<<",\"source\":"<<json_string(f.source)
        <<",\"timestamp_ms\":"<<f.timestamp_ms<<",\"detections\":[";
    for(size_t i=0;i<result.detections.size();++i) {
        if(i) row<<',';
        const auto& d=result.detections[i];
        row<<"{\"class_id\":0,\"label\":\"person\",\"score\":"<<d.score<<",\"box\":["<<d.box.x1<<','<<d.box.y1<<','<<d.box.x2<<','<<d.box.y2<<"]}";
    }
    row<<"],\"preprocess\":"<<json_string(result.preprocessing)<<",\"io\":"<<json_string(result.io);
    impl_->json<<row.str();
    result.timing.output=milliseconds(begin,Clock::now()); result.timing.latency=milliseconds(f.started,Clock::now());
    const auto& t=result.timing;
    impl_->json<<",\"timing_ms\":{\"read\":"<<t.read<<",\"queue\":"<<t.queue<<",\"preprocess\":"<<t.preprocess
        <<",\"inference\":"<<t.inference<<",\"postprocess\":"<<t.postprocess<<",\"result_wait\":"<<t.result_wait<<",\"output\":"<<t.output<<",\"latency\":"<<t.latency<<"}}\n";
    if(!impl_->json) throw std::runtime_error("JSONL write failed");
}
void OutputSink::finish() {
    if(impl_->finished) return;
    impl_->writer.release(); impl_->json.flush();
    if(!impl_->json) throw std::runtime_error("JSONL final flush failed");
    if(impl_->video && !impl_->annotation.empty()) {
        std::error_code error;
        const auto bytes=std::filesystem::file_size(impl_->annotation,error);
        if(error || !bytes) throw std::runtime_error("Annotated video is missing or empty after finalization");
    }
    impl_->finished=true;
}
}
