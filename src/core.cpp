#include "person/core.hpp"
#include <algorithm>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace person {
static int round_even(double x) {
    const int floor_value=static_cast<int>(std::floor(x));
    const double fraction=x-floor_value;
    return floor_value + (fraction>.5 || (fraction==.5 && (floor_value&1)));
}
Letterbox letterbox(int width, int height, int target) {
    if(width<=0 || height<=0 || target<=0) throw std::invalid_argument("Invalid image dimensions");
    Letterbox t;
    t.original_width=width; t.original_height=height;
    t.scale=std::min(double(target)/width,double(target)/height);
    t.width=round_even(width*t.scale); t.height=round_even(height*t.scale);
    t.pad_x=(target-t.width)/2.0; t.pad_y=(target-t.height)/2.0;
    t.left=static_cast<int>(std::floor(t.pad_x)); t.top=static_cast<int>(std::floor(t.pad_y));
    return t;
}
Box restore(Box b, const Letterbox& t) {
    auto x=[&](float v){return float(std::clamp((v-t.pad_x)/t.scale,0.0,double(t.original_width)));};
    auto y=[&](float v){return float(std::clamp((v-t.pad_y)/t.scale,0.0,double(t.original_height)));};
    return {x(b.x1),y(b.y1),x(b.x2),y(b.y2)};
}
float iou(const Box& a, const Box& b) {
    const float intersection=std::max(0.f,std::min(a.x2,b.x2)-std::max(a.x1,b.x1))*
                             std::max(0.f,std::min(a.y2,b.y2)-std::max(a.y1,b.y1));
    const float area_a=std::max(0.f,a.x2-a.x1)*std::max(0.f,a.y2-a.y1);
    const float area_b=std::max(0.f,b.x2-b.x1)*std::max(0.f,b.y2-b.y1);
    const float total=area_a+area_b-intersection;
    return total>0 ? intersection/total : 0;
}
Postprocessor::Postprocessor(float confidence,float nms):confidence_(confidence),nms_(nms) {
    if(!(confidence>0 && confidence<=1 && nms>=0 && nms<=1))
        throw std::invalid_argument("Confidence must be in (0,1], NMS in [0,1]");
    candidates_.reserve(3549); suppressed_.reserve(3549);
}
std::vector<Detection> Postprocessor::decode(const Outputs& outputs,const Letterbox& transform) {
    candidates_.clear(); size_t anchor=0;
    for(int branch=0;branch<3;++branch) {
        const Tensor& box=outputs[branch*3]; const Tensor& score=outputs[branch*3+1];
        const Tensor& sum=outputs[branch*3+2]; const int side=52>>branch;
        for(int i=0;i<3;++i) {
            const Tensor& t=outputs[branch*3+i];
            if(!t.data || t.height!=side || t.width!=side || t.channels!=(i==0?64:1) ||
               !(t.scale>0 && std::isfinite(t.scale))) throw std::runtime_error("Invalid INT8 YOLOv8 output contract");
        }
        (void)sum; // Single-class score_sum adds no information; classification is already sigmoid probability.
        const int cells=side*side; const float stride=416.f/side;
        // Compare quantized values, then check dequantized score to make float threshold boundaries exact.
        const double threshold=double(confidence_)/score.scale+score.zero_point;
        const double cutoff=std::floor(threshold);
        for(int p=0;p<cells;++p,++anchor) {
            const int q=score.data[p];
            if(q<cutoff) continue;
            const float probability=(q-score.zero_point)*score.scale;
            if(probability<confidence_) continue;
            float distance[4];
            for(int d=0;d<4;++d) {
                int maximum=-128;
                for(int k=0;k<16;++k) maximum=std::max(maximum,int(box.data[(d*16+k)*cells+p]));
                double denominator=0,numerator=0;
                for(int k=0;k<16;++k) {
                    const double e=std::exp(double(int(box.data[(d*16+k)*cells+p])-maximum)*box.scale);
                    denominator+=e; numerator+=k*e;
                }
                distance[d]=float(numerator/denominator)*stride;
            }
            const float cx=(p%side+.5f)*stride, cy=(p/side+.5f)*stride;
            candidates_.push_back({{cx-distance[0],cy-distance[1],cx+distance[2],cy+distance[3]},probability,anchor});
        }
    }
    std::sort(candidates_.begin(),candidates_.end(),[](const Candidate& a,const Candidate& b){
        return a.score==b.score ? a.anchor<b.anchor : a.score>b.score;
    });
    suppressed_.assign(candidates_.size(),0);
    std::vector<Detection> result; result.reserve(candidates_.size());
    for(size_t i=0;i<candidates_.size();++i) {
        if(suppressed_[i]) continue;
        const Box b=restore(candidates_[i].box,transform);
        if(b.x2>b.x1 && b.y2>b.y1) result.push_back({b,candidates_[i].score,0});
        for(size_t j=i+1;j<candidates_.size();++j)
            if(!suppressed_[j] && iou(candidates_[i].box,candidates_[j].box)>nms_) suppressed_[j]=1;
    }
    return result;
}
void Histogram::add(double value) {
    if(!std::isfinite(value) || value<0) return;
    ++bins_[size_t(std::min(60000.0,std::ceil(value)))]; ++count_; sum_+=value;
}
double Histogram::percentile(double p) const {
    if(!count_) return 0;
    const uint64_t wanted=std::max<uint64_t>(1,uint64_t(std::ceil(p*count_)));
    uint64_t seen=0;
    for(size_t i=0;i<bins_.size();++i) if((seen+=bins_[i])>=wanted) return double(i);
    return 60000;
}
std::string json_string(const std::string& text) {
    std::ostringstream out; out<<'"'; const char* hex="0123456789abcdef";
    for(unsigned char c:text) {
        if(c=='"' || c=='\\') out<<'\\'<<char(c);
        else if(c<32) out<<"\\u00"<<hex[c>>4]<<hex[c&15];
        else out<<char(c);
    }
    out<<'"'; return out.str();
}
}
