# Full-chain candidate discovery — fixed before new measurements

First identify cumulative time and high-frequency small kernels, inspect dependency/materialization opportunities across several model paths, then compare candidates and choose a prototype. This study follows earlier targeted measurements; it is not a historically blind discovery and does not predetermine a Norm/RoPE optimization.

Reuse exactly original7 cases/inputs: B1/4/8 × P64/256 × O256, plus B1/P2048/O32. BF16 Qwen3-0.6B, FlashAttention, CUDA Graph enabled, TP1,64 KV pages ×256 tokens; source hash must match frozen native baseline.

Three independent native timing processes (one warmup and two measured requests per case) and three independent Nsight Systems processes. Profiling captures every GPU kernel for one complete warmed repeat1 request in each of7 cases, with no kernel-name filter. Rotate case order by process index, same rotation in timing/profile. No model optimization, joint-compile adapter or handwritten kernel.

Requests include scheduler, model preparation, transformer layers, logits and sampling. Add NVTX around complete llm.step calls only. Native sampler synchronizes output before step ends. Verify actual GPU events fall in these ranges. Match generated token IDs against frozen baseline and verify no prefix reuse/preemption.

Analyze prefill, all decode, early decode1–32, late224–255 for O256; O32 control early1–15 and late16–31. Preserve middle decode in complete totals. Produce cumulative duration rank plus count/median/p95/max duration rank for all observed kernel names. Disambiguate semantic chain totals through validated structure; same-name norms must not all be called Q/K norm.

Report each of3 runs, then median/min/max across runs with raw samples and outliers. Percentages use summed GPU kernel durations unless explicitly labeled CPU wall-time. No slow-run deletion. Duration ranks do not establish achievable gains; low occupancy alone does not establish a bottleneck.

Candidate audit includes matrix products, residual AddRMS, Q/K Norm→RoPE(+cache), MLP projection/activation, and sampling. Distinguish measured time, source-confirmed materialization, theoretical ceilings, and unknown improvement potential. Do not reject GEMM solely because it is cuBLAS or treat already-fused primitives as unfused.

Deliver rankings, dependency/consumer table and cost/risk comparison. Select a tractable prototype if evidence supports one; otherwise retain alternatives. This stage does not implement a kernel or establish speedups. Any previous Norm/RoPE results must be labeled historical, not new discovery evidence.
