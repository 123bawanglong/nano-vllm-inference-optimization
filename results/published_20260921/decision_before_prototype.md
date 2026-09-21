# Decision recorded after NEW native NSYS/NCU, before variant evaluation

NSYS:661895 kernel events/1568 steps/0 unmatched; per-workload Decode matrix66.40–76.08%, Attention13.69–23.32%, Norm/RoPE3.12–3.72%, exactly56 Norm+56 RoPE calls/step.

Candidate decision: matrix and Attention dominate but replacing their kernels has larger shape/tiling/validation scope; AddRMS and SiLU-mul already fused. Choose Norm/RoPE for an explicit immediate-consumer fusion opportunity. This does not prove library kernels optimal or Norm/RoPE largest hotspot.

NCU native:22 passes/node, cold-cache replay, no clock lock; grids16/8/8/4, compute0.11–0.25%, DRAM1.04–4.52%, eligible warps0.04–0.07/scheduler, no-eligible93.24–95.71%. Roofline and memory data do not establish bandwidth saturation; tiny grid and dependency waits warrant reducing independent nodes/intermediate reads/writes, not a GEMM tiling scheme.

Hypothesis: one warp owns one head, four warps/block; keep normalized RoPE pairs in registers and fuse four nodes into one. Kernel specialization of shape/launch is not numerical equality. Evaluate intermediate-rounding restoration and native-reduction alignment as separate controlled numerical changes. Four warps is a selected layout, not proven globally optimal.

Actual runtime/prototype history is a controlled reproduction with prior knowledge; all new measurements have their own paths and hashes.
