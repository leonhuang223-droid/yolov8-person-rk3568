#include "person/core.hpp"
#include <iostream>
#include <stdexcept>

using namespace person;
void require(bool condition,const char* message) {if(!condition) throw std::runtime_error(message);}
struct Fixture {
    std::array<std::vector<int8_t>,9> data; Outputs outputs;
    Fixture() {
        for(int i=0;i<9;++i) {
            int side=52>>(i/3),ch=i%3==0?64:1;
            data[i].assign(side*side*ch,i%3==0?-100:0);
            outputs[i]={data[i].data(),ch,side,side,0,i%3==0?100.f:.01f};
            if(i%3==0) for(int d=0;d<4;++d) for(int p=0;p<side*side;++p) data[i][(d*16+2)*side*side+p]=100;
        }
    }
};
int main() {
    try {
        Postprocessor post; Fixture f;
        require(post.decode(f.outputs,letterbox(416,416)).empty(),"Empty frame should yield no people");
        f.data[1][20*52+20]=24;
        require(post.decode(f.outputs,letterbox(416,416)).empty(),"Below threshold rejected");
        f.data[1][20*52+20]=25;
        auto result=post.decode(f.outputs,letterbox(416,416));
        require(result.size()==1 && std::abs(result[0].score-.25)<1e-6,"Threshold boundary retained without sigmoid");
        require(std::abs(result[0].box.x1-148)<1e-5,"Extreme logits must yield stable DFL distance");
        f.data[1][20*52+21]=26; f.data[1][40*52+40]=80;
        result=post.decode(f.outputs,letterbox(416,416));
        require(result.size()==2 && result[0].score>result[1].score,"NMS removes overlapping box and preserves separate person");
        Postprocessor high(.99f); f.outputs[1].scale=.001f;
        require(high.decode(f.outputs,letterbox(416,416)).empty(),"Threshold outside INT8 range must reject all scores");
        for(auto size:std::vector<std::pair<int,int>>{{1920,1080},{1080,1920},{641,479},{479,641},{831,417},{417,831}}) {
            auto t=letterbox(size.first,size.second);
            Box b{float(t.pad_x+10*t.scale),float(t.pad_y+20*t.scale),float(t.pad_x+100*t.scale),float(t.pad_y+120*t.scale)};
            auto restored=restore(b,t);
            require(std::abs(restored.x1-10)<.001 && std::abs(restored.y2-120)<.001,"Letterbox inverse mismatch");
            auto clipped=restore({-100,-100,10000,10000},t);
            require(clipped.x1==0 && clipped.y2==size.second,"Original image clipping failed");
        }
        auto half=letterbox(832,421);
        require(half.height==210 && half.top==103,"Python round-to-even geometry mismatch");
        Histogram histogram; histogram.add(.1);histogram.add(2.1);histogram.add(3.9);
        require(histogram.percentile(.5)==3 && histogram.percentile(.95)==4,"Quantile histogram mismatch");
        require(json_string("a\n\"b")=="\"a\\u000a\\\"b\"","JSON escaping mismatch");
        auto invalid=f.outputs; invalid[1].channels=80;
        bool threw=false;try{post.decode(invalid,letterbox(416,416));}catch(...){threw=true;}
        require(threw,"Multi-class output must fail validation");
        std::cout<<"core_tests passed\n";
    } catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
