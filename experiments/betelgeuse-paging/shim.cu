// Experiment only: reuse the engine's arithmetic without changing its loading contract.
#include "../../native/exl3.cu"
#include <cuda_runtime.h>
#include <vector>
#include <cstdio>
#include <cstdlib>
using namespace alaska_exl3;
constexpr size_t T = 1048576;
#define CK(call) do { cudaError_t e=(call); if(e!=cudaSuccess) { fprintf(stderr,"%s: %s\n",#call,cudaGetErrorString(e)); std::abort(); } } while(0)
struct State {
 uint32_t *resident[3], *pool[3]; half *sg,*su,*sd,*vg,*vu,*vd,*xg,*xu,*xd;
 __nv_bfloat16 *x; int *picks,*ids,*count,*members; float *z,*y;
 cudaStream_t stream; cudaEvent_t start,end; cudaGraphExec_t graphs[2]{};
};
template<class V> void alloc(V*& p,size_t n) { CK(cudaMalloc(&p,n*sizeof(V))); CK(cudaMemset(p,0,n*sizeof(V))); }
void launch(State* s,bool pooled) {
 auto w=pooled?s->pool:s->resident; auto st=s->stream;
 rot_in_kernel<<<dim3(9,32,2),32,0,st>>>(s->x,4096,s->picks,s->sg,s->su,s->xg,s->xu,4096,9);
 exl3_grouped_8_4<<<dim3(9,4,8),128,0,st>>>(s->xg,s->xu,w[0],w[1],s->ids,s->count,s->members,s->z,4096,512,9,4,1,9,288);
 gateup_epilogue_kernel<<<dim3(9,4,1),32,0,st>>>(s->z,s->picks,s->vg,s->vu,s->sd,s->xd,9,512,4,9,10.f,0);
 exl3_grouped_8_4<<<dim3(9,32,1),128,0,st>>>(s->xd,s->xd,w[2],w[2],s->ids,s->count,s->members,s->z,512,4096,9,1,1,9,288);
 down_epilogue_kernel<<<dim3(9,32,1),32,0,st>>>(s->z,s->picks,s->vd,s->y,9,4096,1,9,0);
 CK(cudaGetLastError());
}
extern "C" unsigned char* probe_host_alloc(size_t bytes) {
 unsigned char* p; CK(cudaHostAlloc(&p,bytes,cudaHostAllocDefault)); return p;
}
extern "C" void probe_host_free(unsigned char* p) { CK(cudaFreeHost(p)); }
extern "C" State* probe_init() {
 State* s=new State; CK(cudaStreamCreate(&s->stream)); CK(cudaEventCreate(&s->start)); CK(cudaEventCreate(&s->end));
 for(int m=0;m<3;m++) { alloc(s->resident[m],288*T/4); alloc(s->pool[m],8*T/4); }
 auto transform=[](half*& dst,int width,int salt) { std::vector<half> v(288*width); for(int e=0;e<288;e++) for(int i=0;i<width;i++) v[e*width+i]=__float2half(0.025f+float((e*13+i*7+salt)%37)*0.001f); alloc(dst,v.size()); CK(cudaMemcpy(dst,v.data(),v.size()*2,cudaMemcpyHostToDevice)); };
 transform(s->sg,4096,1); transform(s->su,4096,2); transform(s->sd,512,3); transform(s->vg,512,4); transform(s->vu,512,5); transform(s->vd,4096,6);
 alloc(s->x,4096); std::vector<__nv_bfloat16> x(4096); for(int i=0;i<4096;i++) x[i]=__float2bfloat16(float((i*73)%997-498)/512.f); CK(cudaMemcpy(s->x,x.data(),8192,cudaMemcpyHostToDevice));
 alloc(s->picks,9); alloc(s->ids,9); alloc(s->count,1); alloc(s->members,9); int count=9, members[9]={0,1,2,3,4,5,6,7,8}; CK(cudaMemcpy(s->count,&count,4,cudaMemcpyHostToDevice)); CK(cudaMemcpy(s->members,members,36,cudaMemcpyHostToDevice));
 alloc(s->xg,9*4096); alloc(s->xu,9*4096); alloc(s->xd,9*512); alloc(s->z,2*4*9*4096); alloc(s->y,9*4096);
 return s;
}
extern "C" void probe_transforms(State* s,const unsigned char* bytes) {
 half* destinations[]={s->sg,s->su,s->sd,s->vg,s->vu,s->vd};
 const int widths[]={4096,4096,512,512,512,4096}; size_t offset=0;
 for(int i=0;i<6;i++) { size_t n=288*widths[i]*2; CK(cudaMemcpy(destinations[i],bytes+offset,n,cudaMemcpyHostToDevice)); offset+=n; }
}
extern "C" void probe_fill(State* s,int pooled,int slot,const unsigned char* bytes) {
 auto w=pooled?s->pool:s->resident;
 for(int m=0;m<3;m++) CK(cudaMemcpy(w[m]+size_t(slot)*T/4,bytes+m*T,T,cudaMemcpyHostToDevice));
}
extern "C" float probe_run(State* s,int pooled,int graph,const int* picks,const int* ids,float* output) {
 CK(cudaMemcpy(s->picks,picks,36,cudaMemcpyHostToDevice)); CK(cudaMemcpy(s->ids,ids,36,cudaMemcpyHostToDevice));
 if(graph && !s->graphs[pooled]) { cudaGraph_t g; CK(cudaStreamBeginCapture(s->stream,cudaStreamCaptureModeGlobal)); launch(s,pooled); CK(cudaStreamEndCapture(s->stream,&g)); CK(cudaGraphInstantiate(&s->graphs[pooled],g,0)); CK(cudaGraphDestroy(g)); }
 CK(cudaEventRecord(s->start,s->stream));
 if(graph) CK(cudaGraphLaunch(s->graphs[pooled],s->stream)); else launch(s,pooled);
 CK(cudaEventRecord(s->end,s->stream)); CK(cudaEventSynchronize(s->end)); float ms; CK(cudaEventElapsedTime(&ms,s->start,s->end));
 CK(cudaMemcpy(output,s->y,9*4096*4,cudaMemcpyDeviceToHost)); return ms;
}
extern "C" void probe_destroy(State* s) {
 for(auto g:s->graphs) if(g) CK(cudaGraphExecDestroy(g));
 for(int m=0;m<3;m++) { CK(cudaFree(s->resident[m])); CK(cudaFree(s->pool[m])); }
 void* pointers[]={(void*)s->sg,(void*)s->su,(void*)s->sd,(void*)s->vg,(void*)s->vu,(void*)s->vd,(void*)s->xg,(void*)s->xu,(void*)s->xd,(void*)s->x,(void*)s->picks,(void*)s->ids,(void*)s->count,(void*)s->members,(void*)s->z,(void*)s->y};
 for(void* p:pointers) CK(cudaFree(p));
 CK(cudaEventDestroy(s->start)); CK(cudaEventDestroy(s->end)); CK(cudaStreamDestroy(s->stream)); delete s;
}
