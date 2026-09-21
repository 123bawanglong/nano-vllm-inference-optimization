# Whole-chain torch.compile experiment

Frozen design before measurements: 9 cases, B=1/2/4/8 x P=64/256 x O=256 plus B1/P2048/O32. Reuse old seven cases byte-for-byte; add B2 with fixed seeds. Native model, backend, Graph and cache capacity unchanged.

Compare native separately compiled Q/K norm and RoPE with one fullgraph torch.compile function calling the undecorated original module bodies. No handwritten CUDA/Triton and no KV store fusion. Joint path enabled only in decode; prefill stays native as a control.

Numerical gate is exact equality to native output for real operator fixtures and same-history model logits, not a tolerance chosen after observing custom errors. Also report absolute differences and finite checks. If exact equality fails, report a diagnostic experiment, not validated replacement performance; do not silently relax the gate.

Prepare real first-layer decode QKV fixtures for all9 cases. Microbenchmark both variants in CUDA Graph, with identical outputs and no destructive input mutation. Warmup20 calls; capture1000 chain invocations in one timing graph to avoid CPU submission bottlenecks; time10 paired samples with alternating order. Profiler records a separate one-chain graph and verifies kernel counts. This is a hot-cache microbenchmark, not a prediction of model latency.

Full model:10 paired independent-process runs, AB/BA alternation; each process has1 warmup and2 measured requests per case; same fixed input IDs and seeds. Rotate case order by pair identically within pair. Exclude initialization and compilation. Main scenarios are B1/P64 and B1/P256; other scenarios describe batch/context behavior and controls.

Record native and joint GPU timelines in separate processes, per case: prefill, early decode17–32, late decode225–240. Long-output32 control uses decode16–31 only. Trace kernel durations are diagnostic; formal performance is without profiler. Nsys known incompatible in this environment, so use torch.profiler, preserving raw compressed traces.

Full-model numerical check independently uses baseline-generated token histories at every step (teacher forcing), compares full logits and caches at chosen early/page-boundary positions. Do not confuse matching sampled text with numerical equivalence.

Report process-pair ratios with paired bootstrap intervals and all raw data. No removal of slow samples or extending sample count to obtain significance. Distinguish local-chain gains, whole-engine effects, and null results. No model-quality evaluation claim.
