#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cassert>
#include <cmath>

using torch::Tensor;
using bf16=__nv_bfloat16;
__device__ float tree(float x) {
  #pragma unroll
  for(int d=16;d;d>>=1) x=__fadd_rn(x,__shfl_xor_sync(0xffffffff,x,d));
  return x;
}
__device__ float norm(float x,float inv,float w) {
  return __bfloat162float(__float2bfloat16_rn(__fmul_rn(__fmul_rn(x,inv),w)));
}
template<int WARPS>
__global__ void qk_kernel(const bf16* q,const bf16* k,const bf16* qw,const bf16* kw,
 const int64_t* positions,const float* cache,bf16* qo,bf16* ko,int b,int64_t qs0,
 int64_t qs1,int64_t qs2,int64_t ks0,int64_t ks1,int64_t ks2,int64_t ps,
 int64_t qws,int64_t kws,int64_t cache_rows,float eps,bool native_graph) {
  int row=blockIdx.x*WARPS+threadIdx.x/32,lane=threadIdx.x%32;
  if(row>=b*24) return;
  int token=row/24,head=row%24;bool isq=head<16;head=isq?head:head-16;
  const bf16* x=(isq?q:k)+token*(isq?qs0:ks0)+head*(isq?qs1:ks1);
  const bf16* w=isq?qw:kw;int64_t xs=isq?qs2:ks2,ws=isq?qws:kws;
  bf16* out=(isq?qo:ko)+(token*(isq?16:8)+head)*128;
  int i=lane*2;
  float a=__bfloat162float(x[i*xs]),bb=__bfloat162float(x[(i+1)*xs]);
  float c=__bfloat162float(x[(i+64)*xs]),d=__bfloat162float(x[(i+65)*xs]);
  float sum;
  if(xs==1 && b==1 && native_graph) {
    float lo=tree(__fmaf_rn(a,a,__fmul_rn(bb,bb)));
    float hi=tree(__fmaf_rn(c,c,__fmul_rn(d,d)));
    sum=__fadd_rn(lo,hi);
  } else if(xs==1 && b==1) {
    int base=(lane%16)*8;
    float v0=__bfloat162float(x[base]),v1=__bfloat162float(x[base+1]);
    sum=__fmaf_rn(v0,v0,__fmul_rn(v1,v1));
    #pragma unroll
    for(int j=2;j<8;++j) { float v=__bfloat162float(x[base+j]);sum=__fmaf_rn(v,v,sum); }
    #pragma unroll
    for(int shift=8;shift;shift>>=1) sum=__fadd_rn(sum,__shfl_xor_sync(0xffffffff,sum,shift));
  } else if(xs==1) {
    int base=lane*4;
    float v0=__bfloat162float(x[base]),v1=__bfloat162float(x[base+1]);
    sum=__fmaf_rn(v0,v0,__fmul_rn(v1,v1));
    #pragma unroll
    for(int j=2;j<4;++j) { float v=__bfloat162float(x[base+j]);sum=__fmaf_rn(v,v,sum); }
    sum=tree(sum);
  } else {
    float u=__bfloat162float(x[lane*xs]),v=__bfloat162float(x[(lane+32)*xs]);
    float z=__bfloat162float(x[(lane+64)*xs]),t=__bfloat162float(x[(lane+96)*xs]);
    sum=__fadd_rn(__fmaf_rn(u,u,__fmul_rn(z,z)),__fmaf_rn(v,v,__fmul_rn(t,t)));
    #pragma unroll
    for(int shift=8;shift;shift>>=1) sum=__fadd_rn(sum,__shfl_xor_sync(0xffffffff,sum,shift));
    sum=__fadd_rn(sum,__shfl_xor_sync(0xffffffff,sum,16));
  }
  float variance=__fadd_rn(__fmul_rn(sum,0.0078125f),eps);
  float inv;asm("rsqrt.approx.ftz.f32 %0, %1;":"=f"(inv):"f"(variance));
  a=norm(a,inv,__bfloat162float(w[i*ws]));bb=norm(bb,inv,__bfloat162float(w[(i+1)*ws]));
  c=norm(c,inv,__bfloat162float(w[(i+64)*ws]));d=norm(d,inv,__bfloat162float(w[(i+65)*ws]));
  int64_t pos=positions[token*ps];
  assert(pos>=0 && pos<cache_rows);
  if(pos<0 || pos>=cache_rows) return;
  const float* cs=cache+pos*128;
  out[i]=__float2bfloat16_rn(__fmaf_rn(a,cs[i],-__fmul_rn(c,cs[i+64])));
  out[i+1]=__float2bfloat16_rn(__fmaf_rn(bb,cs[i+1],-__fmul_rn(d,cs[i+65])));
  out[i+64]=__float2bfloat16_rn(__fmaf_rn(c,cs[i],__fmul_rn(a,cs[i+64])));
  out[i+65]=__float2bfloat16_rn(__fmaf_rn(d,cs[i+1],__fmul_rn(bb,cs[i+65])));
}

std::vector<Tensor> fused(Tensor q,Tensor k,Tensor qw,Tensor kw,Tensor p,Tensor cache,double eps,int warps,bool native_graph) {
  TORCH_CHECK(q.is_cuda(),"Q must be CUDA");
  for(const auto& t:{q,k,qw,kw,p,cache}) {
    TORCH_CHECK(t.device()==q.device(),"all inputs must share CUDA device");
    for(auto s:t.strides()) TORCH_CHECK(s>0,"positive input strides required");
  }
  for(const auto& t:{q,k,qw,kw}) TORCH_CHECK(t.scalar_type()==at::kBFloat16,"Q/K/weights must be BF16");
  TORCH_CHECK(q.dim()==3 && q.size(1)==16 && q.size(2)==128,"Q shape must be [B,16,128]");
  TORCH_CHECK(k.dim()==3 && k.size(0)==q.size(0) && k.size(1)==8 && k.size(2)==128,"K shape must be [B,8,128]");
  TORCH_CHECK(q.size(0)>0 && q.size(0)<=65536,"B outside supported range");
  TORCH_CHECK(qw.dim()==1 && kw.dim()==1 && qw.numel()==128 && kw.numel()==128,"weights must be [128]");
  TORCH_CHECK(p.scalar_type()==at::kLong && p.dim()==1 && p.size(0)==q.size(0),"positions must be int64 [B]");
  TORCH_CHECK(cache.scalar_type()==at::kFloat && cache.dim()==3 && cache.size(0)>0 && cache.size(1)==1 && cache.size(2)==128 && cache.is_contiguous(),"cache must be contiguous float32 [P,1,128]");
  TORCH_CHECK(std::isfinite(eps) && eps>0 && std::isfinite(float(eps)) && float(eps)>0,"eps must be finite positive FP32");
  TORCH_CHECK(warps==1 || warps==4,"warps must be 1 or 4");
  c10::cuda::CUDAGuard guard(q.device());
  auto qo=torch::empty(q.sizes(),q.options()),ko=torch::empty(k.sizes(),k.options());
  auto stream=at::cuda::getCurrentCUDAStream(q.get_device());
  #define LAUNCH(W) qk_kernel<W><<<(q.size(0)*24+W-1)/W,W*32,0,stream>>>( \
    (bf16*)q.data_ptr(),(bf16*)k.data_ptr(),(bf16*)qw.data_ptr(),(bf16*)kw.data_ptr(), \
    p.data_ptr<int64_t>(),cache.data_ptr<float>(),(bf16*)qo.data_ptr(),(bf16*)ko.data_ptr(),q.size(0), \
    q.stride(0),q.stride(1),q.stride(2),k.stride(0),k.stride(1),k.stride(2),p.stride(0),qw.stride(0),kw.stride(0),cache.size(0),float(eps),native_graph)
  if(warps==1) { LAUNCH(1); } else { LAUNCH(4); }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {qo,ko};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) { m.def("fused",&fused); }
