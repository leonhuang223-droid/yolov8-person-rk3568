#include "person/app.hpp"
#include <rknn_api.h>
#include <array>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace person {
namespace {
void check(int ret,const std::string& operation) {
    if(ret!=RKNN_SUCC) throw std::runtime_error(operation+" failed: RKNN status "+std::to_string(ret));
}
struct Context {
    rknn_context value{};
    ~Context(){if(value) rknn_destroy(value);}
};
struct Memory {
    rknn_context context{}; rknn_tensor_mem* value{};
    ~Memory(){if(value) rknn_destroy_mem(context,value);}
};
class Engine final:public InferenceEngine {
public:
    Engine(const Options& options,InferenceEngine* shared,bool bound):bound_(bound) {
        if(shared) {
            check(rknn_dup_context(static_cast<rknn_context*>(shared->context_address()),&context_.value),"rknn_dup_context");
        } else {
            std::ifstream file(options.model,std::ios::binary|std::ios::ate);
            if(!file) throw std::runtime_error("Cannot read model: "+options.model.string());
            const auto size=file.tellg();
            if(size<=0 || uint64_t(size)>UINT32_MAX) throw std::runtime_error("Invalid model file size");
            std::vector<char> model(static_cast<size_t>(size)); file.seekg(0);
            if(!file.read(model.data(),model.size())) throw std::runtime_error("Truncated model file");
            check(rknn_init(&context_.value,model.data(),model.size(),0,nullptr),"rknn_init");
        }
        rknn_sdk_version version{};
        check(rknn_query(context_.value,RKNN_QUERY_SDK_VERSION,&version,sizeof(version)),"Query SDK version");
        if(!shared) log(std::string("RKNN Runtime ")+version.api_version+", driver "+version.drv_version);
        rknn_input_output_num number{};
        check(rknn_query(context_.value,RKNN_QUERY_IN_OUT_NUM,&number,sizeof(number)),"Query tensor count");
        if(number.n_input!=1 || number.n_output!=9) throw std::runtime_error("Model must have 1 input and 9 outputs");
        rknn_tensor_attr input{};
        check(rknn_query(context_.value,RKNN_QUERY_INPUT_ATTR,&input,sizeof(input)),"Query input");
        const bool nhwc=input.fmt==RKNN_TENSOR_NHWC && input.dims[0]==1 && input.dims[1]==416 && input.dims[2]==416 && input.dims[3]==3;
        const bool nchw=input.fmt==RKNN_TENSOR_NCHW && input.dims[0]==1 && input.dims[1]==3 && input.dims[2]==416 && input.dims[3]==416;
        if(input.n_dims!=4 || (!nhwc && !nchw) || input.type!=RKNN_TENSOR_INT8 || input.qnt_type!=RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC)
            throw std::runtime_error("Model input must be batch 1, 416x416x3, affine INT8");
        for(size_t i=0;i<9;++i) {
            auto& attr=attributes_[i]; attr.index=i;
            check(rknn_query(context_.value,RKNN_QUERY_OUTPUT_ATTR,&attr,sizeof(attr)),"Query output "+std::to_string(i));
            const uint32_t side=52>>(i/3), channels=i%3==0?64:1;
            if(attr.n_dims!=4 || attr.fmt!=RKNN_TENSOR_NCHW || attr.type!=RKNN_TENSOR_INT8 ||
               attr.qnt_type!=RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC || attr.dims[0]!=1 ||
               attr.dims[1]!=channels || attr.dims[2]!=side || attr.dims[3]!=side ||
               attr.n_elems!=channels*side*side || attr.size<attr.n_elems || !std::isfinite(attr.scale) || attr.scale<=0)
                throw std::runtime_error("Output "+std::to_string(i)+" must be INT8 NCHW DFL64 / person1 / score_sum1 at 52,26,13");
            storage_[i].resize(attr.size);
        }
        if(bound_) {
            native_.index=0;
            check(rknn_query(context_.value,RKNN_QUERY_NATIVE_NHWC_INPUT_ATTR,&native_,sizeof(native_)),"Query native NHWC input");
            if(native_.fmt!=RKNN_TENSOR_NHWC || native_.n_dims!=4 || native_.dims[0]!=1 || native_.dims[1]!=416 ||
               native_.dims[2]!=416 || native_.dims[3]!=3) throw std::runtime_error("Native bound input is not 416x416 RGB NHWC");
            const uint32_t stride=native_.w_stride?native_.w_stride:416;
            row_bytes_=size_t(stride)*3;
            if(stride<416 || native_.size_with_stride<row_bytes_*416) throw std::runtime_error("Invalid bound input stride or size_with_stride");
            memory_.context=context_.value;
            memory_.value=rknn_create_mem(context_.value,native_.size_with_stride);
            if(!memory_.value || !memory_.value->virt_addr || memory_.value->size<row_bytes_*416)
                throw std::runtime_error("rknn_create_mem returned invalid input memory");
            native_.type=RKNN_TENSOR_UINT8; native_.pass_through=0; native_.h_stride=416;
            check(rknn_set_io_mem(context_.value,memory_.value,&native_),"Bind UINT8 NHWC input");
        } else { row_bytes_=416*3; input_.resize(row_bytes_*416); }
    }
    ~Engine() override { if(outputs_active_) rknn_outputs_release(context_.value,9,outputs_.data()); }
    InputBuffer input() override {
        InputBuffer b;
        b.row_bytes=row_bytes_; b.data=bound_?static_cast<uint8_t*>(memory_.value->virt_addr):input_.data();
        b.fd=bound_?memory_.value->fd:-1; b.size=bound_?memory_.value->size:input_.size();
        if(bound_) {
            b.before_device=[this]{check(rknn_mem_sync(context_.value,memory_.value,RKNN_MEMORY_SYNC_TO_DEVICE),"Flush input before RGA");};
            b.after_device=[this]{check(rknn_mem_sync(context_.value,memory_.value,RKNN_MEMORY_SYNC_FROM_DEVICE),"Invalidate input after RGA");};
        }
        return b;
    }
    Outputs infer() override {
        if(outputs_active_) throw std::logic_error("Previous output tensors were not released");
        if(!bound_) {
            rknn_input input{}; input.index=0; input.buf=input_.data(); input.size=input_.size();
            input.type=RKNN_TENSOR_UINT8; input.fmt=RKNN_TENSOR_NHWC; input.pass_through=0;
            check(rknn_inputs_set(context_.value,1,&input),"Set RGB input");
        }
        check(rknn_run(context_.value,nullptr),"rknn_run");
        for(size_t i=0;i<9;++i) {
            outputs_[i]={}; outputs_[i].index=i; outputs_[i].want_float=0; outputs_[i].is_prealloc=1;
            outputs_[i].buf=storage_[i].data(); outputs_[i].size=storage_[i].size();
        }
        check(rknn_outputs_get(context_.value,9,outputs_.data(),nullptr),"Get preallocated outputs");
        outputs_active_=true; Outputs tensors;
        for(size_t i=0;i<9;++i) {
            const auto& a=attributes_[i];
            tensors[i]={storage_[i].data(),int(a.dims[1]),int(a.dims[2]),int(a.dims[3]),a.zp,a.scale};
        }
        return tensors;
    }
    void release_outputs() override {
        if(outputs_active_) { outputs_active_=false; check(rknn_outputs_release(context_.value,9,outputs_.data()),"Release outputs"); }
    }
    void* context_address() override {return &context_.value;}
    std::string name() const override {return bound_?"bound":"standard";}
    void dump(const std::filesystem::path& path,const Letterbox& t) override {
        std::filesystem::create_directories(path);
        std::ofstream meta(path/"metadata.json");
        meta.precision(10);
        meta<<"{\"width\":"<<t.original_width<<",\"height\":"<<t.original_height<<",\"outputs\":[";
        for(size_t i=0;i<9;++i) {
            if(i) meta<<',';
            const auto& a=attributes_[i];
            meta<<"{\"channels\":"<<a.dims[1]<<",\"height\":"<<a.dims[2]<<",\"width\":"<<a.dims[3]<<",\"zp\":"<<a.zp<<",\"scale\":"<<a.scale<<'}';
            std::ofstream f(path/("output_"+std::to_string(i)+".bin"),std::ios::binary);
            f.write(reinterpret_cast<const char*>(storage_[i].data()),a.n_elems);
            if(!f) throw std::runtime_error("Cannot write tensor dump");
        }
        meta<<"]}\n";
        auto b=input(); std::ofstream rgb(path/"input.rgb",std::ios::binary);
        for(int y=0;y<416;++y) rgb.write(reinterpret_cast<char*>(b.data+y*b.row_bytes),416*3);
        if(!meta || !rgb) throw std::runtime_error("Cannot write dump metadata/input");
    }
private:
    Context context_; Memory memory_; bool bound_{},outputs_active_{};
    size_t row_bytes_{}; rknn_tensor_attr native_{};
    std::vector<uint8_t> input_; std::array<std::vector<int8_t>,9> storage_;
    std::array<rknn_tensor_attr,9> attributes_{}; std::array<rknn_output,9> outputs_{};
};
}
std::unique_ptr<InferenceEngine> EngineFactory::create(const Options& options,InferenceEngine* shared) {
    if(options.io=="standard") return std::make_unique<Engine>(options,shared,false);
    if(options.io=="bound") return std::make_unique<Engine>(options,shared,true);
    // Establish the baseline first so model/API errors cannot be hidden by an auto fallback.
    auto baseline=std::make_unique<Engine>(options,shared,false);
    baseline->infer(); baseline->release_outputs();
    try {
        auto bound=std::make_unique<Engine>(options,baseline.get(),true);
        auto input=bound->input(); std::memset(input.data,0,input.size);
        bound->infer(); bound->release_outputs();
        return bound;
    }
    catch(const std::exception& e) {log(std::string("Bound input unavailable; use standard: ")+e.what()); return baseline;}
}
}
