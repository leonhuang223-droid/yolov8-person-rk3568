#include "person/app.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <algorithm>
#include <cctype>
#include <stdexcept>

namespace person {
namespace {
bool image_extension(const std::filesystem::path& path) {
    std::string extension=path.extension().string();
    std::transform(extension.begin(),extension.end(),extension.begin(),[](unsigned char c){return std::tolower(c);});
    return extension==".jpg" || extension==".jpeg" || extension==".png" || extension==".bmp" || extension==".webp" || extension==".tif" || extension==".tiff";
}
class ImageSource final:public InputSource {
public:
    explicit ImageSource(const std::filesystem::path& path) {
        if(std::filesystem::is_directory(path)) {
            for(const auto& entry:std::filesystem::directory_iterator(path))
                if(entry.is_regular_file() && image_extension(entry.path())) paths_.push_back(entry.path());
            std::sort(paths_.begin(),paths_.end());
        } else paths_.push_back(path);
        if(paths_.empty()) throw std::runtime_error("No supported images in directory: "+path.string());
    }
    bool read(Frame& f,uint64_t id) override {
        if(next_==paths_.size()) return false;
        f.started=Clock::now(); f.id=id; f.source=paths_[next_++].string(); f.timestamp_ms=0;
        // imread itself allocates; copy into the retained slot so ownership never aliases worker frames.
        cv::Mat decoded=cv::imread(f.source,cv::IMREAD_COLOR);
        if(decoded.empty()) throw std::runtime_error("Cannot decode image: "+f.source);
        decoded.copyTo(f.image); f.ready=Clock::now(); f.read_ms=milliseconds(f.started,f.ready); return true;
    }
    bool video() const override {return false;}
private: std::vector<std::filesystem::path> paths_; size_t next_{};
};
class VideoSource final:public InputSource {
public:
    explicit VideoSource(const std::filesystem::path& path):path_(path.string()) {
        if(!capture_.open(path_,cv::CAP_FFMPEG)) throw std::runtime_error("Cannot open local video using OpenCV FFmpeg: "+path_+"; check videoio/FFmpeg build and codec support");
        fps_=capture_.get(cv::CAP_PROP_FPS);
        if(!(std::isfinite(fps_) && fps_>0)) {fps_=25; log("Video FPS unavailable; use 25 FPS for annotation output");}
        frame_count_=capture_.get(cv::CAP_PROP_FRAME_COUNT);
        log("Video backend FFmpeg, FPS "+std::to_string(fps_));
    }
    bool read(Frame& f,uint64_t id) override {
        f.started=Clock::now();
        if(!capture_.read(f.image)) {
            // Report a short decode when the container advertises a reliable positive integer frame count.
            if(frame_count_>0 && std::isfinite(frame_count_) && id+1<frame_count_)
                throw std::runtime_error("Video decode ended before declared frame count ("+std::to_string(id)+" / "+std::to_string(frame_count_)+")");
            return false;
        }
        if(f.image.empty() || f.image.type()!=CV_8UC3) throw std::runtime_error("Video decoder returned an invalid BGR frame");
        f.id=id; f.source=path_;
        double timestamp=capture_.get(cv::CAP_PROP_POS_MSEC);
        f.timestamp_ms=timestamp>0?timestamp:id*1000/fps_;
        f.ready=Clock::now(); f.read_ms=milliseconds(f.started,f.ready); return true;
    }
    bool video() const override {return true;}
    double fps() const override {return fps_;}
private: cv::VideoCapture capture_; std::string path_; double fps_{},frame_count_{};
};
}
std::unique_ptr<InputSource> InputSourceFactory::create(const std::filesystem::path& input) {
    if(!std::filesystem::exists(input)) throw std::runtime_error("Input must be an existing local image, directory, or video: "+input.string());
    if(std::filesystem::is_directory(input) || image_extension(input)) return std::make_unique<ImageSource>(input);
    return std::make_unique<VideoSource>(input);
}
}
