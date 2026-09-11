#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <vector>

constexpr int THREADS = 256;

__device__ __forceinline__ float warp_sum(float value) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        value += __shfl_down_sync(0xffffffff, value, offset);
    return value;
}

// One CTA per row. Values stay in registers across the reduction; only warp
// partial sums and the final inverse RMS use shared memory.
template <typename T, int ITEMS>
__global__ void add_rms_kernel(const T* x, const T* r, const T* w,
                              T* y, T* residual_out, int hidden, float eps) {
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const int64_t base = static_cast<int64_t>(blockIdx.x) * hidden;
    float values[ITEMS];
    float sum = 0.f;
    #pragma unroll
    for (int i = 0; i < ITEMS; ++i) {
        int col = threadIdx.x + i * THREADS;
        float z = col < hidden ? float(x[base + col]) + float(r[base + col]) : 0.f;
        values[i] = z;
        sum += z * z;
    }
    sum = warp_sum(sum);
    __shared__ float partial[THREADS / 32];
    __shared__ float inv_rms;
    if (lane == 0) partial[warp] = sum;
    __syncthreads();
    if (warp == 0) {
        sum = lane < THREADS / 32 ? partial[lane] : 0.f;
        sum = warp_sum(sum);
        if (lane == 0) inv_rms = rsqrtf(sum / hidden + eps);
    }
    __syncthreads();
    #pragma unroll
    for (int i = 0; i < ITEMS; ++i) {
        int col = threadIdx.x + i * THREADS;
        if (col < hidden) {
            float z = values[i];
            residual_out[base + col] = T(z);
            // Match installed torch.compile: intermediate cast is eliminated.
            // Both normalization and weight multiply stay FP32 until final store.
            y[base + col] = T((z * inv_rms) * float(w[col]));
        }
    }
}

template <typename T, int N>
struct alignas(sizeof(T) * N) Pack { T data[N]; };

template <typename T>
__global__ void silu_mul_vec_kernel(const T* input, T* output, int count, int hidden) {
    constexpr int V = 4;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count / V) return;
    const int packs_per_row = hidden / V;
    const int row = i / packs_per_row;
    const int col = i % packs_per_row;
    using P = Pack<T, V>;
    const P gate = reinterpret_cast<const P*>(input)[row * 2 * packs_per_row + col];
    const P up = reinterpret_cast<const P*>(input)[row * 2 * packs_per_row + packs_per_row + col];
    P out;
    #pragma unroll
    for (int j = 0; j < V; ++j) {
        float g = float(gate.data[j]);
        float activated = g / (1.f + expf(-g));
        out.data[j] = T(activated * float(up.data[j]));
    }
    reinterpret_cast<P*>(output)[i] = out;
}

template <typename T>
__global__ void silu_mul_kernel(const T* input, T* output, int64_t count, int64_t hidden) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    int64_t row = i / hidden;
    int64_t col = i % hidden;
    float gate = float(input[row * (2 * hidden) + col]);
    float up = float(input[row * (2 * hidden) + hidden + col]);
    // Original compiled SiLU*up keeps intermediates in FP32.
    float activated = gate / (1.f + expf(-gate));
    output[i] = T(activated * up);
}

std::vector<torch::Tensor> add_rms_cuda(torch::Tensor x, torch::Tensor r, torch::Tensor w, double eps) {
    c10::cuda::CUDAGuard guard(x.device());
    auto y = torch::empty_like(x);
    auto residual = torch::empty_like(r);
    int hidden = x.size(-1);
    int64_t rows = x.numel() / hidden;
    if (rows == 0) return {y, residual};
    TORCH_CHECK(rows <= 2147483647, "too many rows");
    auto stream = at::cuda::getCurrentCUDAStream();
    #define LAUNCH_RMS(N) add_rms_kernel<scalar_t, N><<<rows, THREADS, 0, stream>>>( \
            x.data_ptr<scalar_t>(), r.data_ptr<scalar_t>(), w.data_ptr<scalar_t>(), \
            y.data_ptr<scalar_t>(), residual.data_ptr<scalar_t>(), hidden, float(eps))
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "add_rms_cuda", [&] {
        if (hidden <= 1024) { LAUNCH_RMS(4); }
        else if (hidden <= 2048) { LAUNCH_RMS(8); }
        else if (hidden <= 4096) { LAUNCH_RMS(16); }
        else { LAUNCH_RMS(32); }
    });
    #undef LAUNCH_RMS
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {y, residual};
}

torch::Tensor silu_cuda(torch::Tensor x) {
    c10::cuda::CUDAGuard guard(x.device());
    auto shape = x.sizes().vec();
    shape.back() /= 2;
    auto y = torch::empty(shape, x.options());
    int64_t count = y.numel();
    if (count == 0) return y;
    int64_t blocks = (count + THREADS - 1) / THREADS;
    TORCH_CHECK(blocks <= 2147483647, "too many elements");
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "silu_cuda", [&] {
        // Vector loads require both width and actual storage offsets aligned.
        if (shape.back() % 4 == 0 && count <= 1073741823 &&
            reinterpret_cast<uintptr_t>(x.data_ptr<scalar_t>()) % (sizeof(scalar_t) * 4) == 0) {
            silu_mul_vec_kernel<scalar_t><<<(count / 4 + THREADS - 1) / THREADS, THREADS, 0, stream>>>(
                x.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(), count, shape.back());
        } else {
            silu_mul_kernel<scalar_t><<<blocks, THREADS, 0, stream>>>(
                x.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(), count, shape.back());
        }
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}
