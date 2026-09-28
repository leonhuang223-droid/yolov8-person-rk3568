// Host integration tests only. Never installed; it does not emulate NPU numerical inference.
#include <rknn_api.h>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>

namespace {
struct State {rknn_tensor_mem* input{}; const uint8_t* standard{};};
State* state(rknn_context c){return reinterpret_cast<State*>(c);}
std::atomic<size_t> calls{0};
int env_number(const char* name){auto value=std::getenv(name);return value?std::atoi(value):0;}
}
extern "C" {
int rknn_init(rknn_context* c,void*,uint32_t,uint32_t,rknn_init_extend*){*c=reinterpret_cast<rknn_context>(new State);return 0;}
int rknn_dup_context(rknn_context*,rknn_context* c){return rknn_init(c,nullptr,0,0,nullptr);}
int rknn_destroy(rknn_context c){delete state(c);return 0;}
int rknn_query(rknn_context,rknn_query_cmd cmd,void* out,uint32_t){
    if(cmd==RKNN_QUERY_SDK_VERSION){auto v=static_cast<rknn_sdk_version*>(out);std::strcpy(v->api_version,"MOCK-TEST-ONLY");std::strcpy(v->drv_version,"none");return 0;}
    if(cmd==RKNN_QUERY_IN_OUT_NUM){auto v=static_cast<rknn_input_output_num*>(out);v->n_input=1;v->n_output=env_number("MOCK_BAD_MODEL")?6:9;return 0;}
    auto a=static_cast<rknn_tensor_attr*>(out);auto index=a->index;*a={};a->index=index;a->n_dims=4;
    a->type=RKNN_TENSOR_INT8;a->qnt_type=RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC;a->scale=.01f;
    if(cmd==RKNN_QUERY_INPUT_ATTR || cmd==RKNN_QUERY_NATIVE_NHWC_INPUT_ATTR){
        if(cmd==RKNN_QUERY_NATIVE_NHWC_INPUT_ATTR && env_number("MOCK_NO_BOUND"))return -1;
        a->fmt=RKNN_TENSOR_NHWC;a->dims[0]=1;a->dims[1]=416;a->dims[2]=416;a->dims[3]=3;
        a->n_elems=a->size=416*416*3;a->w_stride=448;a->size_with_stride=448*416*3;return 0;
    }
    if(cmd==RKNN_QUERY_OUTPUT_ATTR){a->fmt=RKNN_TENSOR_NCHW;a->dims[0]=1;a->dims[1]=index%3==0?64:1;
        a->dims[2]=a->dims[3]=52>>(index/3);a->n_elems=a->size=a->dims[1]*a->dims[2]*a->dims[3];return 0;}
    return -1;
}
int rknn_inputs_set(rknn_context c,uint32_t n,rknn_input* input){
    if(n!=1 || input[0].index || input[0].type!=RKNN_TENSOR_UINT8 || input[0].fmt!=RKNN_TENSOR_NHWC || input[0].pass_through || input[0].size!=416*416*3)return -2;
    state(c)->standard=static_cast<uint8_t*>(input->buf);return 0;
}
int rknn_run(rknn_context c,rknn_run_extend*){
    if(!state(c)->input && !state(c)->standard)return -2;
    if(state(c)->input && env_number("MOCK_BOUND_FAIL"))return -4;
    const auto call=++calls;if(env_number("MOCK_FAIL_RUN") && call>=size_t(env_number("MOCK_FAIL_RUN")))return -3;
    if(env_number("MOCK_SLEEP_MS"))std::this_thread::sleep_for(std::chrono::milliseconds(env_number("MOCK_SLEEP_MS")));
    return 0;
}
int rknn_outputs_get(rknn_context,uint32_t n,rknn_output* outputs,rknn_output_extend*){
    if(n!=9)return -2;
    for(uint32_t i=0;i<n;++i){const auto& o=outputs[i];uint32_t side=52>>(i/3),ch=i%3==0?64:1;
        if(o.index!=i || o.want_float || !o.is_prealloc || !o.buf || o.size<side*side*ch)return -2;
        std::memset(o.buf,0,side*side*ch);
        if(i==1){auto p=static_cast<int8_t*>(o.buf);p[20*52+20]=80;p[40*52+40]=75;}
    }return 0;
}
int rknn_outputs_release(rknn_context,uint32_t n,rknn_output*){return n==9?0:-2;}
rknn_tensor_mem* rknn_create_mem(rknn_context,uint32_t size){auto m=new rknn_tensor_mem{};m->virt_addr=std::calloc(1,size);m->size=size;m->fd=-1;return m;}
int rknn_destroy_mem(rknn_context,rknn_tensor_mem* m){std::free(m->virt_addr);delete m;return 0;}
int rknn_set_io_mem(rknn_context c,rknn_tensor_mem* m,rknn_tensor_attr* a){
    if(a->index || a->type!=RKNN_TENSOR_UINT8 || a->fmt!=RKNN_TENSOR_NHWC || a->pass_through || a->w_stride!=448)return -2;
    state(c)->input=m;return 0;
}
int rknn_mem_sync(rknn_context,rknn_tensor_mem*,rknn_mem_sync_mode){return 0;}
}
