#pragma once
#include "person/core.hpp"
#include <opencv2/core.hpp>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>

namespace person {
struct Options {
    std::filesystem::path input, model, jsonl{"detections.jsonl"}, annotated, summary, dump;
    size_t workers{1}, capacity{}; int warmup{3};
    std::string preprocess{"auto"}, io{"auto"};
    float confidence{.25f}, nms{.45f};
};
struct Frame {
    uint64_t id{}; std::string source; double timestamp_ms{}, read_ms{};
    cv::Mat image; Clock::time_point started{}, ready{};
};
struct FrameResult {
    std::vector<Detection> detections; Timings timing;
    Clock::time_point detected{};
    std::string preprocessing, io;
};
struct InputBuffer {
    uint8_t* data{}; int width{416}, height{416}; size_t row_bytes{};
    int fd{-1}; size_t size{};
    std::function<void()> before_device, after_device;
};
class InferenceEngine {
public:
    virtual ~InferenceEngine()=default;
    virtual InputBuffer input()=0;
    virtual Outputs infer()=0;
    virtual void release_outputs()=0;
    virtual void* context_address()=0;
    virtual std::string name() const=0;
    virtual void dump(const std::filesystem::path&,const Letterbox&)=0;
};
class EngineFactory {
public:
    static std::unique_ptr<InferenceEngine> create(const Options&,InferenceEngine* shared=nullptr);
};
class Preprocessor {
public:
    virtual ~Preprocessor()=default;
    virtual Letterbox process(const cv::Mat&,InputBuffer)=0;
    virtual std::string name() const=0;
};
class PreprocessorFactory {
public:
    static std::unique_ptr<Preprocessor> create(const std::string& mode);
};
class InputSource {
public:
    virtual ~InputSource()=default;
    virtual bool read(Frame& frame,uint64_t id)=0;
    virtual bool video() const=0;
    virtual double fps() const { return 0; }
};
class InputSourceFactory {
public:
    static std::unique_ptr<InputSource> create(const std::filesystem::path& input);
};
class OutputSink {
public:
    OutputSink(const Options&,bool video,double fps);
    ~OutputSink();
    void write(Frame&,FrameResult&);
    void finish();
private:
    struct Impl; std::unique_ptr<Impl> impl_;
};
void log(const std::string& message);
}
