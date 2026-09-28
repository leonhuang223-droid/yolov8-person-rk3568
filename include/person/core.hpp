#pragma once
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace person {
using Clock = std::chrono::steady_clock;
inline double milliseconds(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b-a).count();
}
struct Box { float x1{}, y1{}, x2{}, y2{}; };
struct Detection { Box box; float score{}; int class_id{0}; };
struct Letterbox {
    int original_width{}, original_height{}, width{}, height{}, left{}, top{};
    double scale{}, pad_x{}, pad_y{};
};
Letterbox letterbox(int width, int height, int target=416);
Box restore(Box box, const Letterbox& transform);
float iou(const Box& a, const Box& b);
struct Tensor {
    const int8_t* data{};
    int channels{}, height{}, width{}, zero_point{};
    float scale{};
};
using Outputs = std::array<Tensor,9>;
struct Timings { double read{}, queue{}, preprocess{}, inference{}, postprocess{}, result_wait{}, output{}, latency{}; };
class Postprocessor {
public:
    Postprocessor(float confidence=.25f, float nms=.45f);
    std::vector<Detection> decode(const Outputs& outputs, const Letterbox& transform);
private:
    struct Candidate { Box box; float score; size_t anchor; };
    float confidence_, nms_;
    std::vector<Candidate> candidates_;
    std::vector<uint8_t> suppressed_;
};
// Fixed storage: 1 ms bins up to 60 seconds. Quantiles report the upper bin edge.
class Histogram {
public:
    void add(double value);
    double percentile(double p) const;
    double mean() const { return count_ ? sum_/count_ : 0; }
    uint64_t count() const { return count_; }
private:
    std::array<uint64_t,60001> bins_{};
    uint64_t count_{}; double sum_{};
};
std::string json_string(const std::string& text);
}
