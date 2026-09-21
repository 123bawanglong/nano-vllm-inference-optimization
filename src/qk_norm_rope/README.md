# Bounded CUDA Q/K RMSNorm + RoPE prototype

`from src.qk_norm_rope import build, fused`; call `build()` before CUDA Graph
capture, then `fused(q,k,q_weight,k_weight,positions,cos_sin_cache,eps)`.
Optional `warps=1|4` selects heads per block; the frozen default is 4.
Outputs are new dense BF16 tensors. Inputs are never modified.

`reduction_mode='standalone'` (default) matches the isolated test's native compile
context. The model adapter **must** pass `reduction_mode='native_graph'` to match
the frozen model's compiled CUDA Graph context. These are explicit arithmetic
contracts selected by the caller, never selected from tensor values or reference
outputs. The full model compiles B8/B4/B2 before B1, and its B1 norm selects
`XBLOCK=1,num_warps=2` (two 64-element adjacent-pair trees). The isolated B1 norm
selects `XBLOCK=8,num_warps=2` (eight-value folds). Both preserve the same BF16
boundary; their FP32 addition orders differ and can round to different BF16
values. Actual model compiler logs, selected PTX, failing earlier model gate,
and captured intermediate regressions document this distinction.

The C++ binding validates CUDA device equality, dtype, shape, positive strides,
cache contiguity, finite positive epsilon, and supported launch configuration.
The kernel reads positions and asserts their bounds on device; valid positions
must lie in `[0, cache.size(0))`. This avoids a CPU synchronization during Graph
capture. It uses the current PyTorch CUDA stream and guards the CUDA device.

This is an inference experiment for Q16/K8 and head dimension 128. No backward,
partial RoPE, alternative dtype, KV-cache store, or model integration is supplied.
`CUDA_HOME=/usr/local/cuda-12.8`; the build cache lives under the Linux home
directory, and the target architecture defaults to 12.0 (RTX 5080).

The operator preserves the *native compiled* RMSNorm result rounding to BF16
before RoPE. It intentionally does not reproduce eager Python's intermediate
BF16 cast before multiplying by the RMS weight, which native compilation removes.
RMS products, sums, rsqrt, weighted result rounding, and RoPE FMA order follow
the inspected native compiler output. For contiguous B1 the selected norm folds
8 squares per lane; packed B>1 folds 4; the non-unit inner-stride specialization
uses another reduction tree. This exact contract depends on the frozen PyTorch/
Triton toolchain and compiler specialization. It is not a bitwise-equivalence
promise for all compiler histories, layouts, or inputs. Full-model correctness
must pass independently of the micro fixture gate.

Run `python scripts/full_project/kernel_validation.py --label final` after
sourcing `scripts/profiling/environment.sh`. Evidence and failed iterations are
in `results/full_project_20260919/kernel_validation_*.json` and `.log`.
The 138-row gate covers 9 real first-layer fixtures, 10 random packed seeds at
B1/2/4/8, noncontiguous zero/random/small/large/extreme inputs, positions
0/255/256/4095, both block configurations, Graph replay, and input immutability.
It additionally writes `kernel_model_fixture_validation_<label>.json`, comparing
native_graph mode's norm and RoPE results with actual saved native Graph
intermediates from all 28 layers; its fixture includes regression failures from
the earlier implementation. `kernel_model_diagnostic.py` performs independent
live native Graph comparisons across 2,380 layer/decode steps in five workloads.
