#include <torch/extension.h>
#include <cmath>
#include <vector>

std::vector<torch::Tensor> add_rms_cuda(torch::Tensor, torch::Tensor, torch::Tensor, double);
torch::Tensor silu_cuda(torch::Tensor);

static void check_input(const torch::Tensor& x) {
    TORCH_CHECK(x.is_cuda(), "input must be CUDA");
    TORCH_CHECK(x.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(x.dim() >= 1 && x.size(-1) > 0, "last dimension must be positive");
    TORCH_CHECK(x.scalar_type() == at::kFloat || x.scalar_type() == at::kHalf ||
                x.scalar_type() == at::kBFloat16, "supported dtypes: float32, float16, bfloat16");
}

std::vector<torch::Tensor> add_rms_norm(torch::Tensor x, torch::Tensor r, torch::Tensor w, double eps) {
    check_input(x);
    check_input(r);
    check_input(w);
    TORCH_CHECK(x.sizes() == r.sizes(), "residual shape mismatch");
    TORCH_CHECK(w.dim() == 1 && w.numel() == x.size(-1), "weight shape mismatch");
    TORCH_CHECK(x.device() == r.device() && x.device() == w.device(), "device mismatch");
    TORCH_CHECK(x.scalar_type() == r.scalar_type() && x.scalar_type() == w.scalar_type(), "dtype mismatch");
    TORCH_CHECK(std::isfinite(eps) && eps > 0, "eps must be finite and positive");
    TORCH_CHECK(x.size(-1) <= 8192, "CUDA RMSNorm supports hidden size <= 8192");
    return add_rms_cuda(x, r, w, eps);
}

torch::Tensor silu_and_mul(torch::Tensor x) {
    check_input(x);
    TORCH_CHECK(x.size(-1) % 2 == 0, "gate/up dimension must be even");
    return silu_cuda(x);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add_rms_norm", &add_rms_norm);
    m.def("silu_and_mul", &silu_and_mul);
}
