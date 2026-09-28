#include "person/core.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

int main(int argc,char** argv){try{
    if(argc!=2) throw std::runtime_error("Usage: decode_dump DIRECTORY (contract.txt + output_0..8.bin)");
    std::filesystem::path directory=argv[1]; std::ifstream contract(directory/"contract.txt");
    int width,height; float confidence,nms;
    if(!(contract>>width>>height>>confidence>>nms)) throw std::runtime_error("Invalid dump contract");
    person::Outputs outputs; std::array<std::vector<int8_t>,9> data;
    for(size_t i=0;i<9;++i){auto& t=outputs[i];
        if(!(contract>>t.channels>>t.height>>t.width>>t.zero_point>>t.scale)) throw std::runtime_error("Invalid tensor metadata");
        if(t.channels<=0 || t.height<=0 || t.width<=0 || t.channels>64 || t.height>52 || t.width>52) throw std::runtime_error("Invalid tensor dimensions");
        data[i].resize(t.channels*t.height*t.width); t.data=data[i].data();
        std::ifstream file(directory/("output_"+std::to_string(i)+".bin"),std::ios::binary);
        if(!file.read(reinterpret_cast<char*>(data[i].data()),data[i].size())) throw std::runtime_error("Missing/short tensor dump");
    }
    auto detections=person::Postprocessor(confidence,nms).decode(outputs,person::letterbox(width,height));
    std::cout.precision(9);std::cout<<'[';
    for(size_t i=0;i<detections.size();++i){if(i)std::cout<<',';const auto& d=detections[i];
        std::cout<<"["<<d.box.x1<<','<<d.box.y1<<','<<d.box.x2<<','<<d.box.y2<<','<<d.score<<']';}
    std::cout<<"]\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
