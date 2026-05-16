#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#define CUDA_CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)
#define CUBLAS_CHECK(x) do { cublasStatus_t s=(x); if(s!=CUBLAS_STATUS_SUCCESS){fprintf(stderr,"CUBLAS %s:%d %d\n",__FILE__,__LINE__,(int)s); exit(1);} } while(0)

constexpr int D=576, V=49152, L=30, QH=9, KVH=3, HD=64, FF=1536, CTX=128, QKV=960;

struct Weights {
  float *emb, *final_norm;
  __half *emb_h;
  float *rms1[L], *rms2[L];
  __half *qkv[L], *o[L], *fc1[L], *fc2[L];
};
static float *G_ALPHA=nullptr, *G_BETA=nullptr;
static bool G_DEVICE_SCALARS=false;

static void read_exact(std::ifstream& f, void* p, size_t n){ f.read((char*)p,n); if(!f) { std::cerr<<"read failed\n"; exit(1);} }
static std::vector<float> read_vec(std::ifstream& f){ uint64_t n; read_exact(f,&n,8); std::vector<float> v(n); read_exact(f,v.data(),n*4); return v; }
static float* to_dev(const std::vector<float>& h){ float* d; CUDA_CHECK(cudaMalloc(&d,h.size()*4)); CUDA_CHECK(cudaMemcpy(d,h.data(),h.size()*4,cudaMemcpyHostToDevice)); return d; }
static __half* to_dev_half(const std::vector<float>& v){
  std::vector<__half> h(v.size());
  for(size_t i=0;i<v.size();++i) h[i]=__float2half(v[i]);
  __half* d; CUDA_CHECK(cudaMalloc(&d,h.size()*sizeof(__half))); CUDA_CHECK(cudaMemcpy(d,h.data(),h.size()*sizeof(__half),cudaMemcpyHostToDevice)); return d;
}

__global__ void rope_kernel(float* qkv, const float* sin, const float* cos, int pos){
  int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx>= (QH+KVH)*HD) return;
  bool isq = idx < QH*HD;
  int base = isq ? 0 : D;
  int local = isq ? idx : idx - QH*HD;
  int head_dim = local % HD;
  int half = head_dim & 31;
  int pair = head_dim < 32 ? head_dim + 32 : head_dim - 32;
  float x = qkv[base + local];
  float xp = qkv[base + (local/HD)*HD + pair];
  float rot = head_dim < 32 ? -xp : xp;
  qkv[base + local] = x * cos[pos*HD + head_dim] + rot * sin[pos*HD + head_dim];
}

__global__ void rms_kernel(const float* x, const float* w, float* y, int n){
  __shared__ float ss[256];
  float sum=0;
  for(int i=threadIdx.x;i<n;i+=blockDim.x){ float v=x[i]; sum += v*v; }
  ss[threadIdx.x]=sum; __syncthreads();
  for(int s=blockDim.x/2;s>0;s>>=1){ if(threadIdx.x<s) ss[threadIdx.x]+=ss[threadIdx.x+s]; __syncthreads(); }
  float scale=rsqrtf(ss[0]/n + 1e-5f);
  for(int i=threadIdx.x;i<n;i+=blockDim.x) y[i]=x[i]*scale*w[i];
}

__global__ void add_kernel(float* x, const float* y, int n){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) x[i]+=y[i]; }
__global__ void gather_emb_kernel(const float* emb, const int* tok, float* x){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<D) x[i]=emb[((int)(*tok))*D+i]; }
__global__ void set_int_kernel(int* p, int v){ *p=v; }
__global__ void silu_kernel(float* x){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<FF){ float u=x[i], v=x[i+FF]; x[i]=u/(1.0f+expf(-u))*v; } }
__global__ void split_cache_v_kernel(const float* qkv, float* kcache, float* vcache, int layer, int pos){
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<KVH*HD){ kcache[((layer*KVH + i/HD)*CTX + pos)*HD + i%HD]=qkv[D+i]; vcache[((layer*KVH + i/HD)*CTX + pos)*HD + i%HD]=qkv[D+KVH*HD+i]; }
}

__global__ void attn_kernel(const float* qkv, const float* kcache, const float* vcache, float* out, int layer, int pos){
  int h=blockIdx.x, tid=threadIdx.x, kv=h/3;
  __shared__ float scores[CTX];
  float s=-1e20f;
  if(tid<CTX){
    if(tid<=pos){
      float acc=0; const float* q=qkv+h*HD; const float* k=&kcache[((layer*KVH+kv)*CTX+tid)*HD];
      #pragma unroll
      for(int d=0; d<HD; ++d) acc += q[d]*k[d];
      s = acc * 0.125f;
    }
    scores[tid]=s;
  }
  __syncthreads();
  float mx=-1e20f; for(int t=tid;t<CTX;t+=blockDim.x) mx=fmaxf(mx,scores[t]);
  __shared__ float red[128]; red[tid]=mx; __syncthreads();
  for(int r=64;r;r>>=1){ if(tid<r) red[tid]=fmaxf(red[tid],red[tid+r]); __syncthreads(); }
  mx=red[0];
  float sum=0; for(int t=tid;t<CTX;t+=blockDim.x){ float e=expf(scores[t]-mx); scores[t]=e; sum+=e; }
  red[tid]=sum; __syncthreads();
  for(int r=64;r;r>>=1){ if(tid<r) red[tid]+=red[tid+r]; __syncthreads(); }
  float inv=1.0f/red[0];
  if(tid<HD){ float acc=0; for(int t=0;t<=pos;++t) acc += scores[t]*inv*vcache[((layer*KVH+kv)*CTX+t)*HD+tid]; out[h*HD+tid]=acc; }
}

__global__ void gemv_t_kernel(const float* __restrict__ Wt, const float* __restrict__ x, float* __restrict__ y, int K){
  int row=blockIdx.x, tid=threadIdx.x;
  const float* w = Wt + (size_t)row*K;
  float acc=0.f;
  for(int k=tid;k<K;k+=blockDim.x) acc = fmaf(w[k], x[k], acc);
  __shared__ float red[256]; red[tid]=acc; __syncthreads();
  for(int s=blockDim.x/2;s;s>>=1){ if(tid<s) red[tid]+=red[tid+s]; __syncthreads(); }
  if(tid==0) y[row]=red[0];
}

__global__ void argmax_kernel(const float* logits, int* out){
  __shared__ float vals[256]; __shared__ int ids[256];
  int tid=threadIdx.x; float best=-1e30f; int bid=0;
  for(int i=tid;i<V;i+=blockDim.x){ float v=logits[i]; if(v>best){best=v; bid=i;} }
  vals[tid]=best; ids[tid]=bid; __syncthreads();
  for(int s=128;s;s>>=1){ if(tid<s && vals[tid+s]>vals[tid]){ vals[tid]=vals[tid+s]; ids[tid]=ids[tid+s]; } __syncthreads(); }
  if(tid==0) *out=ids[0];
}
__global__ void copy_int_kernel(const int* src, int* dst, int i){ dst[i] = *src; }
__global__ void cast_f32_f16_kernel(const float* x, __half* y, int n){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) y[i]=__float2half(x[i]); }

static void gemv_h_s(cublasHandle_t h, const __half* W, int K, int N, const float* x, __half* xh, float* y, cudaStream_t s){
  cast_f32_f16_kernel<<<(K+255)/256,256,0,s>>>(x,xh,K);
  const float a=1.f,b=0.f;
  CUBLAS_CHECK(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, N, 1, K,
    &a, W, CUDA_R_16F, N, xh, CUDA_R_16F, K, &b, y, CUDA_R_32F, N,
    CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}
static void logits_h_s(cublasHandle_t h, const __half* E, const float* x, __half* xh, float* y, cudaStream_t s){
  cast_f32_f16_kernel<<<(D+255)/256,256,0,s>>>(x,xh,D);
  const float a=1.f,b=0.f;
  CUBLAS_CHECK(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, V, 1, D,
    &a, E, CUDA_R_16F, D, xh, CUDA_R_16F, D, &b, y, CUDA_R_32F, V,
    CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

static void one_dev(cublasHandle_t h, const Weights& w, int* dtok, int pos, float* kcache, float* vcache, float* sin, float* cos, float* x, float* hbuf, float* qkv, float* attn, float* ff, float* logits, __half* xh, cudaStream_t s=0){
  gather_emb_kernel<<<3,256,0,s>>>(w.emb, dtok, x);
  for(int l=0;l<L;++l){
    rms_kernel<<<1,256,0,s>>>(x,w.rms1[l],hbuf,D);
    gemv_h_s(h,w.qkv[l],D,QKV,hbuf,xh,qkv,s);
    rope_kernel<<<4,256,0,s>>>(qkv,sin,cos,pos);
    split_cache_v_kernel<<<1,256,0,s>>>(qkv,kcache,vcache,l,pos);
    attn_kernel<<<QH,128,0,s>>>(qkv,kcache,vcache,attn,l,pos);
    gemv_h_s(h,w.o[l],D,D,attn,xh,hbuf,s); add_kernel<<<3,256,0,s>>>(x,hbuf,D);
    rms_kernel<<<1,256,0,s>>>(x,w.rms2[l],hbuf,D);
    gemv_h_s(h,w.fc1[l],D,FF*2,hbuf,xh,ff,s); silu_kernel<<<6,256,0,s>>>(ff);
    gemv_h_s(h,w.fc2[l],FF,D,ff,xh,hbuf,s); add_kernel<<<3,256,0,s>>>(x,hbuf,D);
  }
  rms_kernel<<<1,256,0,s>>>(x,w.final_norm,hbuf,D);
  logits_h_s(h,w.emb_h,hbuf,xh,logits,s);
  argmax_kernel<<<1,256,0,s>>>(logits,dtok);
}

static int one(cublasHandle_t h, const Weights& w, int tok, int pos, float* kcache, float* vcache, float* sin, float* cos, float* x, float* hbuf, float* qkv, float* attn, float* ff, float* logits, __half* xh, int* dtok){
  set_int_kernel<<<1,1>>>(dtok,tok);
  one_dev(h,w,dtok,pos,kcache,vcache,sin,cos,x,hbuf,qkv,attn,ff,logits,xh);
  int ht; CUDA_CHECK(cudaMemcpy(&ht,dtok,4,cudaMemcpyDeviceToHost)); return ht;
}

int main(int argc, char** argv){
  std::string path = argc>1 ? argv[1] : "Agent/cuda/smollm135m_f32.bin";
  bool use_graph = argc>2 && std::string(argv[2]) == "--graph";
  std::ifstream f(path, std::ios::binary); if(!f){ std::cerr<<"missing "<<path<<"\n"; return 1; }
  char magic[9]; read_exact(f,magic,9); int cfg[7]; read_exact(f,cfg,28); int plen; read_exact(f,&plen,4); std::vector<int> prompt(plen); read_exact(f,prompt.data(),plen*4);
  Weights w; { auto embv=read_vec(f); w.emb=to_dev(embv); w.emb_h=to_dev_half(embv); } w.final_norm=to_dev(read_vec(f));
  for(int i=0;i<L;++i){
    w.rms1[i]=to_dev(read_vec(f)); w.rms2[i]=to_dev(read_vec(f));
    w.qkv[i]=to_dev_half(read_vec(f));
    w.o[i]=to_dev_half(read_vec(f));
    w.fc1[i]=to_dev_half(read_vec(f));
    w.fc2[i]=to_dev_half(read_vec(f));
  }
  std::vector<float> hsin(CTX*HD), hcos(CTX*HD); for(int p=0;p<CTX;++p) for(int j=0;j<HD/2;++j){ float inv=powf(10000.0f,-(2*j/(float)HD)); float a=p*inv; hsin[p*HD+j]=hsin[p*HD+j+32]=sinf(a); hcos[p*HD+j]=hcos[p*HD+j+32]=cosf(a); }
  float *sin=to_dev(hsin), *cos=to_dev(hcos), *k,*v,*k0,*v0,*x,*hb,*qkv,*attn,*ff,*logits; __half* xh; int* dtok;
  CUDA_CHECK(cudaMalloc(&k,L*KVH*CTX*HD*4)); CUDA_CHECK(cudaMalloc(&v,L*KVH*CTX*HD*4)); CUDA_CHECK(cudaMalloc(&k0,L*KVH*CTX*HD*4)); CUDA_CHECK(cudaMalloc(&v0,L*KVH*CTX*HD*4));
  CUDA_CHECK(cudaMalloc(&x,D*4)); CUDA_CHECK(cudaMalloc(&hb,D*4)); CUDA_CHECK(cudaMalloc(&qkv,QKV*4)); CUDA_CHECK(cudaMalloc(&attn,D*4)); CUDA_CHECK(cudaMalloc(&ff,FF*2*4)); CUDA_CHECK(cudaMalloc(&logits,V*4)); CUDA_CHECK(cudaMalloc(&xh,FF*2*sizeof(__half))); CUDA_CHECK(cudaMalloc(&dtok,4));
  CUDA_CHECK(cudaMemset(k,0,L*KVH*CTX*HD*4)); CUDA_CHECK(cudaMemset(v,0,L*KVH*CTX*HD*4));
  cublasHandle_t handle; CUBLAS_CHECK(cublasCreate(&handle));
  CUBLAS_CHECK(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));
  int tok=prompt[0]; for(int p=0;p<plen;++p) tok=one(handle,w,prompt[p],p,k,v,sin,cos,x,hb,qkv,attn,ff,logits,xh,dtok);
  CUDA_CHECK(cudaMemcpy(k0,k,L*KVH*CTX*HD*4,cudaMemcpyDeviceToDevice)); CUDA_CHECK(cudaMemcpy(v0,v,L*KVH*CTX*HD*4,cudaMemcpyDeviceToDevice));
  int steps=100; int last=prompt.back(); for(int i=0;i<steps;++i) last=one(handle,w,last,plen-1+i,k,v,sin,cos,x,hb,qkv,attn,ff,logits,xh,dtok);
  int* dout; CUDA_CHECK(cudaMalloc(&dout, steps*4));

  cudaStream_t stream; CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  CUBLAS_CHECK(cublasSetStream(handle, stream));
  cudaEvent_t st,en; CUDA_CHECK(cudaEventCreate(&st)); CUDA_CHECK(cudaEventCreate(&en));

  cudaGraph_t graph{}; cudaGraphExec_t graph_exec{};
  if(use_graph){
    CUDA_CHECK(cudaMemcpyAsync(k,k0,L*KVH*CTX*HD*4,cudaMemcpyDeviceToDevice,stream));
    CUDA_CHECK(cudaMemcpyAsync(v,v0,L*KVH*CTX*HD*4,cudaMemcpyDeviceToDevice,stream));
    set_int_kernel<<<1,1,0,stream>>>(dtok,prompt.back());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    for(int i=0;i<steps;++i){
      one_dev(handle,w,dtok,plen-1+i,k,v,sin,cos,x,hb,qkv,attn,ff,logits,xh,stream);
      copy_int_kernel<<<1,1,0,stream>>>(dtok,dout,i);
    }
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
  }

  CUDA_CHECK(cudaMemcpyAsync(k,k0,L*KVH*CTX*HD*4,cudaMemcpyDeviceToDevice,stream));
  CUDA_CHECK(cudaMemcpyAsync(v,v0,L*KVH*CTX*HD*4,cudaMemcpyDeviceToDevice,stream));
  set_int_kernel<<<1,1,0,stream>>>(dtok,prompt.back());
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaEventRecord(st, stream));
  if(use_graph){
    CUDA_CHECK(cudaGraphLaunch(graph_exec, stream));
  } else {
    for(int i=0;i<steps;++i){
      one_dev(handle,w,dtok,plen-1+i,k,v,sin,cos,x,hb,qkv,attn,ff,logits,xh,stream);
      copy_int_kernel<<<1,1,0,stream>>>(dtok,dout,i);
    }
  }
  CUDA_CHECK(cudaEventRecord(en, stream)); CUDA_CHECK(cudaEventSynchronize(en)); float ms; CUDA_CHECK(cudaEventElapsedTime(&ms,st,en));
  std::vector<int> out(steps); CUDA_CHECK(cudaMemcpy(out.data(),dout,steps*4,cudaMemcpyDeviceToHost));
  std::cout << "[cuda-perf] prompt="<<plen<<" gen="<<steps<<" decode_only="<<(ms/1000.0)<<"s tok/s="<<(steps/(ms/1000.0))<<(use_graph?" graph=1":" graph=0")<<"\n";
  std::cout << "[tokens]"; for(int t:out) std::cout << ' ' << t; std::cout << "\n";
}
