#include "person/pipeline.hpp"
#include <chrono>
#include <iostream>
#include <stdexcept>

using namespace person;
void require(bool condition,const char* message){if(!condition) throw std::runtime_error(message);}
using Pipeline=OrderedPipeline<size_t,size_t>;
std::vector<Pipeline::Worker> workers(size_t n,bool failure=false) {
    std::vector<Pipeline::Worker> result;
    for(size_t i=0;i<n;++i) result.emplace_back([failure](size_t& value){
        if(failure && value==4) throw std::runtime_error("inference failed");
        std::this_thread::sleep_for(std::chrono::milliseconds(value%5==0?4:1)); return value*2;
    }); return result;
}
int main(){try {
    for(size_t frames:{0,1,2,19,101}) {
        Pipeline pipeline(workers(4),6); size_t next=0;
        auto stats=pipeline.run([&](size_t& task,size_t id){task=id;return id<frames;},[&](size_t& task,size_t& result){
            require(task==next && result==next*2,"Reordered or duplicate frame"); ++next;
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        });
        require(next==frames && stats.submitted==frames && stats.completed==frames,"EOF dropped frames");
        require(stats.inflight_peak<=6 && stats.task_peak<=6 && stats.result_peak<=6,"Unbounded queue");
        if(frames==101) require(stats.inflight_peak==6,"Backpressure capacity not exercised");
    }
    for(int failure=0;failure<3;++failure){
        Pipeline pipeline(workers(3,failure==0),4); bool threw=false;
        try {pipeline.run([&](size_t& value,size_t id){if(failure==1 && id==5) throw std::runtime_error("read failed");value=id;return id<100;},
            [&](size_t& value,size_t&){if(failure==2 && value==3) throw std::runtime_error("write failed");});}
        catch(...){threw=true;} require(threw,"Pipeline failed to surface an error");
    }
    Pipeline canceled(workers(2),3); std::thread interrupter([&]{std::this_thread::sleep_for(std::chrono::milliseconds(10));canceled.cancel("interrupt");});
    bool threw=false;try {canceled.run([](size_t& value,size_t id){value=id;return id<10000;},[](size_t&,size_t&){});}catch(...){threw=true;}
    interrupter.join(); require(threw,"Cancellation was ignored");
    Pipeline narrow(workers(4),1);size_t next=0;
    auto stats=narrow.run([](size_t& value,size_t id){value=id;return id<10;},[&](size_t& v,size_t&){require(v==next++,"Single-slot ordering failed");});
    require(stats.completed==10 && stats.inflight_peak==1,"Single-slot pipeline failed");
    std::cout<<"pipeline_tests passed\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
