#include "person/app.hpp"
#include <opencv2/imgproc.hpp>
#include <cstring>
#include <stdexcept>
#ifdef PERSON_ENABLE_RGA
#include <im2d.h>
#endif

namespace person {
namespace {
void validate(const cv::Mat& source,const InputBuffer& input) {
    if(source.empty() || source.type()!=CV_8UC3) throw std::runtime_error("Preprocessing needs a BGR UINT8 image");
    if(!input.data || input.row_bytes<416*3 || input.size<input.row_bytes*416) throw std::runtime_error("Invalid input buffer");
}
class OpenCVPreprocessor final:public Preprocessor {
public:
    Letterbox process(const cv::Mat& source,InputBuffer input) override {
        validate(source,input); const auto t=letterbox(source.cols,source.rows);
        cv::Mat destination(416,416,CV_8UC3,input.data,input.row_bytes);
        destination.setTo(cv::Scalar(0,0,0));
        cv::resize(source,resized_,cv::Size(t.width,t.height),0,0,cv::INTER_LINEAR);
        cv::Mat roi=destination(cv::Rect(t.left,t.top,t.width,t.height));
        cv::cvtColor(resized_,roi,cv::COLOR_BGR2RGB); return t;
    }
    std::string name() const override {return "opencv";}
private: cv::Mat resized_;
};
#ifdef PERSON_ENABLE_RGA
class RGAPreprocessor final:public Preprocessor {
public:
    Letterbox process(const cv::Mat& source,InputBuffer input) override {
        validate(source,input);
        if(source.step%3 || input.row_bytes%3) throw std::runtime_error("RGA RGB stride must be divisible by 3 bytes");
        const auto t=letterbox(source.cols,source.rows);
        // RGA2 RGB requires pixel stride aligned to 4; preserve the Python geometry rather than rounding rectangles.
        if((source.step/3)%4 || (input.row_bytes/3)%4) throw std::runtime_error("RGA2 RGB pixel stride must be aligned to 4");
        auto src=wrapbuffer_virtualaddr_t(source.data,source.cols,source.rows,source.step/3,source.rows,RK_FORMAT_BGR_888);
        auto dst=input.fd>=0 ? wrapbuffer_fd_t(input.fd,416,416,input.row_bytes/3,416,RK_FORMAT_RGB_888) :
                              wrapbuffer_virtualaddr_t(input.data,416,416,input.row_bytes/3,416,RK_FORMAT_RGB_888);
        const im_rect sr{0,0,source.cols,source.rows},dr{t.left,t.top,t.width,t.height};
        const rga_buffer_t empty{}; const im_rect rect{};
        auto status=imcheck_t(src,dst,empty,sr,dr,rect,IM_SYNC);
        if(status!=IM_STATUS_NOERROR && status!=IM_STATUS_SUCCESS) throw std::runtime_error(std::string("RGA compatibility: ")+imStrError_t(status));
        std::memset(input.data,0,input.size);
        if(input.before_device) input.before_device();
        status=improcess(src,dst,empty,sr,dr,rect,-1,nullptr,nullptr,IM_SYNC);
        if(status!=IM_STATUS_SUCCESS) throw std::runtime_error(std::string("RGA resize/color conversion: ")+imStrError_t(status));
        // CPU invalidation avoids a later Runtime input flush overwriting device-written pixels.
        if(input.after_device) input.after_device();
        return t;
    }
    std::string name() const override {return "rga";}
};
#endif
class AutoPreprocessor final:public Preprocessor {
public:
    AutoPreprocessor(){
#ifdef PERSON_ENABLE_RGA
        active_=std::make_unique<RGAPreprocessor>();
#else
        active_=std::make_unique<OpenCVPreprocessor>();
#endif
    }
    Letterbox process(const cv::Mat& source,InputBuffer input) override {
        try {return active_->process(source,input);}
        catch(const std::exception& e) {
            if(active_->name()!="rga") throw;
            log(std::string("RGA unavailable; use OpenCV for this worker: ")+e.what());
            active_=std::make_unique<OpenCVPreprocessor>(); return active_->process(source,input);
        }
    }
    std::string name() const override {return active_->name();}
private: std::unique_ptr<Preprocessor> active_;
};
}
std::unique_ptr<Preprocessor> PreprocessorFactory::create(const std::string& mode) {
    if(mode=="opencv") return std::make_unique<OpenCVPreprocessor>();
    if(mode=="auto") return std::make_unique<AutoPreprocessor>();
#ifdef PERSON_ENABLE_RGA
    if(mode=="rga") return std::make_unique<RGAPreprocessor>();
#endif
    throw std::runtime_error("RGA was not compiled in this build");
}
}
