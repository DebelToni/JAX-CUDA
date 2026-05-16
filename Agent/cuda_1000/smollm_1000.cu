#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#define CUDA_CHECK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);}}while(0)
#define CUBLAS_CHECK(x) do{cublasStatus_t s=(x); if(s!=CUBLAS_STATUS_SUCCESS){fprintf(stderr,"CUBLAS %s:%d %d\n",__FILE__,__LINE__,(int)s); exit(1);}}while(0)
constexpr int D=576,V=49152,L=30,QH=9,KVH=3,HD=64,FF=1536,CTX=128,QKV=960;
struct W{__half *emb,*qkv[L],*o[L],*fc1[L],*fc2[L]; float *final,*rms1[L],*rms2[L];};
static void read_exact(std::ifstream& f, void* p, size_t n){f.read((char*)p,n); if(!f){std::cerr<<"read failed\n"; exit(1);}}
static std::vector<float> read_vec(std::ifstream& f){uint64_t n; read_exact(f,&n,8); std::vector<float> v(n); read_exact(f,v.data(),n*4); return v;}
static float* to_f(const std::vector<float>& h){float*d; CUDA_CHECK(cudaMalloc(&d,h.size()*4)); CUDA_CHECK(cudaMemcpy(d,h.data(),h.size()*4,cudaMemcpyHostToDevice)); return d;}
static __half* to_h(const std::vector<float>& v){std::vector<__half> h(v.size()); for(size_t i=0;i<v.size();++i) h[i]=__float2half(v[i]); __half*d; CUDA_CHECK(cudaMalloc(&d,h.size()*2)); CUDA_CHECK(cudaMemcpy(d,h.data(),h.size()*2,cudaMemcpyHostToDevice)); return d;}
static __half* to_h_t(const std::vector<float>& v,int K,int N){std::vector<__half> h((size_t)K*N); for(int k=0;k<K;++k)for(int n=0;n<N;++n)h[(size_t)n*K+k]=__float2half(v[(size_t)k*N+n]); __half*d; CUDA_CHECK(cudaMalloc(&d,h.size()*2)); CUDA_CHECK(cudaMemcpy(d,h.data(),h.size()*2,cudaMemcpyHostToDevice)); return d;}
__global__ void set_int(int*p,int v){*p=v;} __global__ void copy_int(const int*s,int*d,int i){d[i]=*s;}
__global__ void gather(const __half* emb,const int* tok,__half* x){int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<D)x[i]=emb[((int)*tok)*D+i];}
__global__ void rms(const __half*x,const float*w,__half*y,int n){__shared__ float ss[128]; int tid=threadIdx.x; int n2=n>>1; const __half2* x2=reinterpret_cast<const __half2*>(x); float sum=0; for(int i=tid;i<n2;i+=128){__half2 hv=x2[i]; float a=__half2float(hv.x), b=__half2float(hv.y); sum+=a*a+b*b;} ss[tid]=sum; __syncthreads(); for(int s=64;s;s>>=1){if(tid<s)ss[tid]+=ss[tid+s]; __syncthreads();} float sc=rsqrtf(ss[0]/n+1e-5f); for(int i=tid;i<n2;i+=128){int j=i<<1; y[j]=__float2half(__half2float(x[j])*sc*w[j]); y[j+1]=__float2half(__half2float(x[j+1])*sc*w[j+1]);}}
__global__ void addh(__half*x,const __half*y,int n){int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n)x[i]=__hadd(x[i],y[i]);}
__global__ void rope_cache(__half*qkv,__half*kcache,__half*vcache,const float*sin,const float*cos,int layer,int pos){
  int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<(QH+KVH)*HD){
    bool isq=idx<QH*HD; int base=isq?0:D; int local=isq?idx:idx-QH*HD; int d=local%HD; int pair=d<32?d+32:d-32;
    float x=__half2float(qkv[base+local]); float xp=__half2float(qkv[base+(local/HD)*HD+pair]); float rot=d<32?-xp:xp;
    __half val=__float2half(x*cos[pos*HD+d]+rot*sin[pos*HD+d]); qkv[base+local]=val;
    if(!isq) kcache[((layer*KVH+local/HD)*CTX+pos)*HD+d]=val;
  } else if(idx<(QH+2*KVH)*HD) {
    int j=idx-(QH+KVH)*HD;
    vcache[((layer*KVH+j/HD)*CTX+pos)*HD+j%HD]=qkv[D+KVH*HD+j];
  }
}
__global__ void attn(const __half*qkv,const __half*kcache,const __half*vcache,__half*out,int layer,int pos){
  int h=blockIdx.x,tid=threadIdx.x,kv=h/3; __shared__ float scores[CTX],red[64];
  for(int tt=tid; tt<CTX; tt+=64){
    float s=-1e20f;
    if(tt<=pos){
      const __half2* q2=reinterpret_cast<const __half2*>(qkv+h*HD);
      const __half2* k2=reinterpret_cast<const __half2*>(&kcache[((layer*KVH+kv)*CTX+tt)*HD]);
      __half2 acc2=__float2half2_rn(0.f);
      #pragma unroll
      for(int d=0;d<HD/2;++d) acc2=__hfma2(q2[d],k2[d],acc2);
      float acc=__half2float(acc2.x)+__half2float(acc2.y);
      s=acc*0.125f;
    }
    scores[tt]=s;
  }
  __syncthreads(); float mx=-1e20f; for(int t=tid;t<CTX;t+=64)mx=fmaxf(mx,scores[t]); red[tid]=mx; __syncthreads();
  for(int r=32;r;r>>=1){if(tid<r)red[tid]=fmaxf(red[tid],red[tid+r]); __syncthreads();} mx=red[0];
  float sum=0; for(int t=tid;t<CTX;t+=64){float e=exp2f((scores[t]-mx)*1.4426950408889634f); scores[t]=e; sum+=e;} red[tid]=sum; __syncthreads();
  for(int r=32;r;r>>=1){if(tid<r)red[tid]+=red[tid+r]; __syncthreads();} float inv=1.f/red[0];
  if(tid<HD){float acc=0; for(int t=0;t<=pos;++t)acc+=scores[t]*inv*__half2float(vcache[((layer*KVH+kv)*CTX+t)*HD+tid]); out[h*HD+tid]=__float2half(acc);}
}
__global__ void silu(__half*x){int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<FF){float u=__half2float(x[i]),v=__half2float(x[i+FF]); x[i]=__float2half(u/(1.f+expf(-u))*v);}}
__global__ void argmax(const float*logits,int*out){__shared__ float vals[256]; __shared__ int ids[256]; int tid=threadIdx.x; float best=-1e30f; int bid=0; for(int i=tid;i<V;i+=256){float v=logits[i]; if(v>best){best=v; bid=i;}} vals[tid]=best; ids[tid]=bid; __syncthreads(); for(int s=128;s;s>>=1){if(tid<s&&vals[tid+s]>vals[tid]){vals[tid]=vals[tid+s]; ids[tid]=ids[tid+s];} __syncthreads();} if(tid==0)*out=ids[0];}
__global__ void gemv_ht(const __half* __restrict__ Wt,const __half* __restrict__ x,__half* __restrict__ y,int K,int N){
  int lane=threadIdx.x&31, warp=threadIdx.x>>5, row=blockIdx.x*8+warp; if(row>=N) return;
  const __half2* w2=reinterpret_cast<const __half2*>(Wt+(size_t)row*K); const __half2* x2=reinterpret_cast<const __half2*>(x); int K2=K>>1; __half2 acc2=__float2half2_rn(0.f);
  for(int k=lane;k<K2;k+=32){acc2=__hfma2(w2[k],x2[k],acc2);}
  float acc=__half2float(acc2.x)+__half2float(acc2.y); for(int off=16;off;off>>=1) acc += __shfl_down_sync(0xffffffff, acc, off);
  if(lane==0) y[row]=__float2half(acc);
}
__device__ __forceinline__ unsigned int ord_float(float x){unsigned int u=__float_as_uint(x); return (u&0x80000000u)?~u:(u^0x80000000u);}
__global__ void reset_best(unsigned long long* best,int* out){*best=((unsigned long long)ord_float(-1.0e30f)<<32); *out=0;}
__global__ void logits_atomic(const __half* __restrict__ Wt,const __half* __restrict__ x,unsigned long long* __restrict__ best,int* __restrict__ out,int K,int N){
  int lane=threadIdx.x&31, warp=threadIdx.x>>5, row=blockIdx.x*32+warp;
  float acc=-1.0e30f;
  if(row<N){
    const __half2* w2=reinterpret_cast<const __half2*>(Wt+(size_t)row*K); const __half2* x2=reinterpret_cast<const __half2*>(x); int K2=K>>1; __half2 acc2=__float2half2_rn(0.f);
    for(int k=lane;k<K2;k+=32){acc2=__hfma2(w2[k],x2[k],acc2);}
    acc=__half2float(acc2.x)+__half2float(acc2.y); for(int off=16;off;off>>=1) acc += __shfl_down_sync(0xffffffff, acc, off);
  }
  __shared__ float sv[32]; __shared__ int si[32];
  if(lane==0){sv[warp]=acc; si[warp]=row;}
  __syncthreads();
  if(threadIdx.x==0){float bv=sv[0]; int bid=si[0]; for(int i=1;i<32;++i){if(sv[i]>bv){bv=sv[i]; bid=si[i];}} unsigned long long pack=((unsigned long long)ord_float(bv)<<32) | (unsigned int)(0xffffffffu-(unsigned int)bid); atomicMax(best,pack);}
}
__global__ void unpack_best(const unsigned long long* best,int* out){*out=(int)(0xffffffffu-(unsigned int)(*best));}
__global__ void gemv_ht_add(const __half* __restrict__ Wt,const __half* __restrict__ x,__half* __restrict__ y,int K,int N){
  int lane=threadIdx.x&31, warp=threadIdx.x>>5, row=blockIdx.x*8+warp; if(row>=N) return;
  const __half2* w2=reinterpret_cast<const __half2*>(Wt+(size_t)row*K); const __half2* x2=reinterpret_cast<const __half2*>(x); int K2=K>>1; __half2 acc2=__float2half2_rn(0.f);
  for(int k=lane;k<K2;k+=32){acc2=__hfma2(w2[k],x2[k],acc2);}
  float acc=__half2float(acc2.x)+__half2float(acc2.y); for(int off=16;off;off>>=1) acc += __shfl_down_sync(0xffffffff, acc, off);
  if(lane==0) y[row]=__float2half(__half2float(y[row])+acc);
}
__global__ void gemv_swiglu_add(const __half* __restrict__ Wt,const __half* __restrict__ x,__half* __restrict__ y,int N){
  int lane=threadIdx.x&31, warp=threadIdx.x>>5, row=blockIdx.x*8+warp; if(row>=N) return;
  const __half2* w2=reinterpret_cast<const __half2*>(Wt+(size_t)row*FF);
  float acc=0.f;
  for(int k2=lane;k2<FF/2;k2+=32){
    int k=k2<<1;
    float u0=__half2float(x[k]), v0=__half2float(x[k+FF]);
    float u1=__half2float(x[k+1]), v1=__half2float(x[k+1+FF]);
    __half2 xv=__halves2half2(__float2half(u0/(1.f+expf(-u0))*v0), __float2half(u1/(1.f+expf(-u1))*v1));
    __half2 ww=w2[k2];
    acc += __half2float(ww.x)*__half2float(xv.x)+__half2float(ww.y)*__half2float(xv.y);
  }
  for(int off=16;off;off>>=1) acc += __shfl_down_sync(0xffffffff, acc, off);
  if(lane==0) y[row]=__float2half(__half2float(y[row])+acc);
}
static void gemm_h(cublasHandle_t h,const __half*A,int K,int N,const __half*x,__half*y,cudaStream_t s){(void)h; gemv_ht<<<(N+7)/8,256,0,s>>>(A,x,y,K,N);}
static void gemm_h_add(cublasHandle_t h,const __half*A,int K,int N,const __half*x,__half*y,cudaStream_t s){(void)h; gemv_ht_add<<<(N+7)/8,256,0,s>>>(A,x,y,K,N);}
static void gemm_swiglu_add(cublasHandle_t h,const __half*A,const __half*x,__half*y,cudaStream_t s){(void)h; gemv_swiglu_add<<<(D+7)/8,256,0,s>>>(A,x,y,D);}
static void logits_h(cublasHandle_t h,const __half*E,const __half*x,float*y,int*yi,int*dtok,unsigned long long*best,cudaStream_t s){(void)h;(void)y;(void)yi; reset_best<<<1,1,0,s>>>(best,dtok); logits_atomic<<<(V+31)/32,1024,0,s>>>(E,x,best,dtok,D,V); unpack_best<<<1,1,0,s>>>(best,dtok);}
static void one_dev(cublasHandle_t h,const W&w,int*dtok,int pos,__half*k,__half*v,const float*sin,const float*cos,__half*x,__half*hb,__half*qkv,__half*attn_b,__half*ff,float*logits,int*logit_ids,unsigned long long*best,cudaStream_t s=0){gather<<<3,256,0,s>>>(w.emb,dtok,x); for(int l=0;l<L;++l){rms<<<1,128,0,s>>>(x,w.rms1[l],hb,D); gemm_h(h,w.qkv[l],D,QKV,hb,qkv,s); rope_cache<<<4,256,0,s>>>(qkv,k,v,sin,cos,l,pos); attn<<<QH,64,0,s>>>(qkv,k,v,attn_b,l,pos); gemm_h_add(h,w.o[l],D,D,attn_b,x,s); rms<<<1,128,0,s>>>(x,w.rms2[l],hb,D); gemm_h(h,w.fc1[l],D,FF*2,hb,ff,s); gemm_swiglu_add(h,w.fc2[l],ff,x,s);} rms<<<1,128,0,s>>>(x,w.final,hb,D); logits_h(h,w.emb,hb,logits,logit_ids,dtok,best,s);}
static int one(cublasHandle_t h,const W&w,int tok,int pos,__half*k,__half*v,const float*sin,const float*cos,__half*x,__half*hb,__half*qkv,__half*attn_b,__half*ff,float*logits,int*logit_ids,unsigned long long*best,int*dtok){set_int<<<1,1>>>(dtok,tok); one_dev(h,w,dtok,pos,k,v,sin,cos,x,hb,qkv,attn_b,ff,logits,logit_ids,best); int ht; CUDA_CHECK(cudaMemcpy(&ht,dtok,4,cudaMemcpyDeviceToHost)); return ht;}
int main(int argc,char**argv){std::string path=argc>1?argv[1]:"Agent/cuda/smollm135m_f32.bin"; bool graph=argc>2&&std::string(argv[2])=="--graph"; std::ifstream f(path,std::ios::binary); if(!f){std::cerr<<"missing "<<path<<"\n"; return 1;} char magic[9]; read_exact(f,magic,9); int cfg[7]; read_exact(f,cfg,28); int plen; read_exact(f,&plen,4); std::vector<int> prompt(plen); read_exact(f,prompt.data(),plen*4); W w; w.emb=to_h(read_vec(f)); w.final=to_f(read_vec(f)); for(int i=0;i<L;++i){w.rms1[i]=to_f(read_vec(f)); w.rms2[i]=to_f(read_vec(f)); w.qkv[i]=to_h_t(read_vec(f),D,QKV); w.o[i]=to_h_t(read_vec(f),D,D); w.fc1[i]=to_h_t(read_vec(f),D,FF*2); w.fc2[i]=to_h_t(read_vec(f),FF,D);} std::vector<float>hsin(CTX*HD),hcos(CTX*HD); for(int p=0;p<CTX;++p)for(int j=0;j<32;++j){float inv=powf(10000.f,-(2*j/(float)HD)); float a=p*inv; hsin[p*HD+j]=hsin[p*HD+j+32]=sinf(a); hcos[p*HD+j]=hcos[p*HD+j+32]=cosf(a);} float *sin=to_f(hsin),*cos=to_f(hcos),*logits; __half *k,*v,*k0,*v0,*x,*hb,*qkv,*attn_b,*ff; int*dtok; int*logit_ids; unsigned long long* best; CUDA_CHECK(cudaMalloc(&k,L*KVH*CTX*HD*2)); CUDA_CHECK(cudaMalloc(&v,L*KVH*CTX*HD*2)); CUDA_CHECK(cudaMalloc(&k0,L*KVH*CTX*HD*2)); CUDA_CHECK(cudaMalloc(&v0,L*KVH*CTX*HD*2)); CUDA_CHECK(cudaMalloc(&x,D*2)); CUDA_CHECK(cudaMalloc(&hb,D*2)); CUDA_CHECK(cudaMalloc(&qkv,QKV*2)); CUDA_CHECK(cudaMalloc(&attn_b,D*2)); CUDA_CHECK(cudaMalloc(&ff,FF*2*2)); CUDA_CHECK(cudaMalloc(&logits,V*4)); CUDA_CHECK(cudaMalloc(&logit_ids,((V+7)/8)*4)); CUDA_CHECK(cudaMalloc(&dtok,4)); CUDA_CHECK(cudaMalloc(&best,8)); CUDA_CHECK(cudaMemset(k,0,L*KVH*CTX*HD*2)); CUDA_CHECK(cudaMemset(v,0,L*KVH*CTX*HD*2)); cublasHandle_t handle; CUBLAS_CHECK(cublasCreate(&handle)); CUBLAS_CHECK(cublasSetPointerMode(handle,CUBLAS_POINTER_MODE_HOST)); int tok=prompt[0]; for(int p=0;p<plen;++p)tok=one(handle,w,prompt[p],p,k,v,sin,cos,x,hb,qkv,attn_b,ff,logits,logit_ids,best,dtok); CUDA_CHECK(cudaMemcpy(k0,k,L*KVH*CTX*HD*2,cudaMemcpyDeviceToDevice)); CUDA_CHECK(cudaMemcpy(v0,v,L*KVH*CTX*HD*2,cudaMemcpyDeviceToDevice)); int steps=100,last=prompt.back(); for(int i=0;i<steps;++i)last=one(handle,w,last,plen-1+i,k,v,sin,cos,x,hb,qkv,attn_b,ff,logits,logit_ids,best,dtok); int*dout; CUDA_CHECK(cudaMalloc(&dout,steps*4)); cudaStream_t stream; CUDA_CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking)); CUBLAS_CHECK(cublasSetStream(handle,stream)); cudaEvent_t st,en; CUDA_CHECK(cudaEventCreate(&st)); CUDA_CHECK(cudaEventCreate(&en)); cudaGraph_t g{}; cudaGraphExec_t ge{}; if(graph){CUDA_CHECK(cudaMemcpyAsync(k,k0,L*KVH*CTX*HD*2,cudaMemcpyDeviceToDevice,stream)); CUDA_CHECK(cudaMemcpyAsync(v,v0,L*KVH*CTX*HD*2,cudaMemcpyDeviceToDevice,stream)); set_int<<<1,1,0,stream>>>(dtok,prompt.back()); CUDA_CHECK(cudaStreamSynchronize(stream)); CUDA_CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal)); for(int i=0;i<steps;++i){one_dev(handle,w,dtok,plen-1+i,k,v,sin,cos,x,hb,qkv,attn_b,ff,logits,logit_ids,best,stream); copy_int<<<1,1,0,stream>>>(dtok,dout,i);} CUDA_CHECK(cudaStreamEndCapture(stream,&g)); CUDA_CHECK(cudaGraphInstantiate(&ge,g,nullptr,nullptr,0));} CUDA_CHECK(cudaMemcpyAsync(k,k0,L*KVH*CTX*HD*2,cudaMemcpyDeviceToDevice,stream)); CUDA_CHECK(cudaMemcpyAsync(v,v0,L*KVH*CTX*HD*2,cudaMemcpyDeviceToDevice,stream)); set_int<<<1,1,0,stream>>>(dtok,prompt.back()); CUDA_CHECK(cudaStreamSynchronize(stream)); CUDA_CHECK(cudaEventRecord(st,stream)); if(graph)CUDA_CHECK(cudaGraphLaunch(ge,stream)); else for(int i=0;i<steps;++i){one_dev(handle,w,dtok,plen-1+i,k,v,sin,cos,x,hb,qkv,attn_b,ff,logits,logit_ids,best,stream); copy_int<<<1,1,0,stream>>>(dtok,dout,i);} CUDA_CHECK(cudaEventRecord(en,stream)); CUDA_CHECK(cudaEventSynchronize(en)); float ms; CUDA_CHECK(cudaEventElapsedTime(&ms,st,en)); std::vector<int>out(steps); CUDA_CHECK(cudaMemcpy(out.data(),dout,steps*4,cudaMemcpyDeviceToHost)); std::cout<<"[cuda-half-perf] prompt="<<plen<<" gen="<<steps<<" decode_only="<<ms/1000.0<<"s tok/s="<<steps/(ms/1000.0)<<(graph?" graph=1":" graph=0")<<"\n[tokens]"; for(int t:out)std::cout<<' '<<t; std::cout<<"\n";}
