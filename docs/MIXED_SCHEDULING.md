> 历史实验记录：以下性能数值未因发布重测或改写。原始实验目录与公开仓库目录的对应关系、可运行命令见 [REPRODUCE.md](REPRODUCE.md)。

# Decode-priority Chunked Prefill Mixed Scheduling

本目录是 `nano-vllm-cuda` 的独立副本，同时保留 Add+RMSNorm、SiluAndMul CUDA 融合。第二项优化只改变调度和 Engine 协调。完整实测数据由 `benchmarks/write_mixed_report.py` 生成到 `MIXED_REPORT.md`。

## 1. Baseline 原来怎么工作

已检查本机真实代码：Scheduler、LLMEngine、ModelRunner、Sequence、BlockManager、Attention、Context、采样器和 CUDA Graph 路径。

```text
Scheduler.schedule() -> (seqs, is_prefill)
    优先 waiting prefill，已有 chunked prefill
    有 prefill 就返回；否则 running decode
LLMEngine.step()
    ModelRunner.call("run", seqs, is_prefill) 一次
    Scheduler.postprocess(seqs, token_ids, is_prefill)
ModelRunner.run()
    prefill: prepare_prefill -> eager model -> logits -> sampler
    decode: prepare_decode -> CUDA Graph replay -> logits -> sampler
```

原实现已用 `num_cached_tokens` 表示已计算 KV 的 token 数，用 `num_scheduled_tokens` 表示本轮 chunk 长度。中间 chunk 的采样结果丢弃，最终 chunk 才产生 completion token。Decode 每个请求处理一个 token。BlockManager 保留完整序列的 block reservation，并复用已完成 block 的 prefix hash。

另外，原调度器只允许本轮第一个 prefill 请求切块：已选请求之后，若下一个请求完整放不进剩余预算就结束本轮。新版本允许最后一个请求填满剩余预算，因此纯 prefill 的组批形状也可能改变。

## 2. 原来的瓶颈

单个 `is_prefill` 只能描述一个同质 sub-batch；Engine 每轮也只执行一次 runner。waiting 中连续出现 prefill 时，正在 decode 的请求可能连续多个 iteration 得不到执行。已有 chunking 限制单轮 prefill 大小，但 prefill 优先使多个 chunk 仍可连续阻塞 decode。

## 3. 修改了哪些文件

相对上一个融合版本，生产代码只修改两个已有文件，新增一个文件：

| 文件 | 函数/结构 | 变化 |
|---|---|---|
| `nanovllm/engine/scheduler.py` | `schedule` | 先 decode，再用剩余联合预算和序列名额安排 FIFO prefill；返回明确的两组请求 |
| 同上 | `postprocess` | 按 sequence 阶段推进 KV、转入 running、追加 token、处理结束；校验重复/未调度请求 |
| 同上 | `preempt`、`_release_waiting_reservation`、`add` | 保持 running 尾部抢占；必要时回收 waiting 的部分 KV，保持 FIFO 位置；显式拒绝超过总 KV 容量的完整 reservation |
| `nanovllm/engine/llm_engine.py` | `step` | 一次 schedule，先执行 decode sub-batch，再执行 prefill sub-batch，分别 postprocess |
| 同上 | `generate` | 使用明确的 prefill/decode token 数和耗时；移除用正负 token 数判断阶段的逻辑 |
| `nanovllm/engine/schedule_output.py`（新增） | `ScheduleOutput`、`StepStats` | 独立保存两组请求及 token 数快照、阶段耗时、整轮耗时 |

新增测试 `tests/test_mixed_scheduler.py`、`tests/test_mixed_execution.py`。新增实验脚本 `benchmarks/bench_mixed.py`、`run_mixed_suite.py`、`write_mixed_report.py`。副本中的旧 `benchmarks/bench_engine.py` 适配新接口，专用 mixed 比较使用新脚本。新增 `fusion_baseline_manifest.json` 记录原融合版本哈希。

工作区外层新增 `scripts/prepare_mixed_copy.py` 和 `docs/superpowers/plans/2026-09-10-mixed-scheduling.md`。原仓库和原融合副本均未编辑。

ModelRunner、Sequence、BlockManager、Attention、Sampler、模型数学和 `fused_ops.cu/.cpp` 均沿用原融合版本。复制来的 `CUDA_FUSION.md`、`PERFORMANCE_REPORT.md` 只记录第一项融合实验，第二项实验以本文件和 `MIXED_REPORT.md` 为准。

## 4. 新 Scheduler 流程

假设 budget=1024，running 为 A/B/C，waiting 为尚未计算 3000 token 的 D，KV 容量和序列名额都足够：

```text
先给 A/B/C 各 1 个 decode token，并预留必要的新 KV block。
剩余 token budget = 1024 - 3 = 1021。
给 D 安排 min(3000, 1021) = 1021 个 prefill token。
ScheduleOutput(decode=(A,B,C), prefill=(D))
Engine: run(A/B/C, False) -> postprocess
        run(D, True)      -> postprocess
D.num_cached_tokens: 0 -> 1021；D 仍然 waiting，无 completion token。
```

同一个 scheduler iteration 内有两个顺序执行的 GPU sub-batch，并不要求两个 kernel 同时执行。若 decode 已用完 token 或 sequence 预算，prefill 留到以后。FIFO 队首因 KV 不可分配而等待时，不越过它去选后面的 prompt。

## 5. KV Cache 如何保持正确

- Decode 优先执行 `can_append`/`may_append`，先保护本轮 decode 所需的 slot/block，再尝试 prefill allocation。
- `num_cached_tokens` 只在本轮执行成功后增加 `num_scheduled_tokens`；随后将后者清零。采样产生的最新 token 尚未计算 KV，下轮 decode 才处理它。
- 中间 prefill chunk 保持 waiting，不追加 completion；最终 chunk 成功后才移动到 running。
- `block_table` 仍由原 BlockManager 管理。沿用完整序列 reservation，不把 chunking 等同于按 chunk 增量分配 KV。
- 完整 block 在 postprocess 中按旧实现计算 hash；prefix-cache 引用计数、防止过早复用、共享 block 释放逻辑不变。
- 内存紧张时沿用 running 尾部抢占；若部分 prefill 的 reservation 阻塞最后一个 decoder，可回收未选中的 waiting reservation，让它以后重算。保持其队列位置；不回收本轮已选请求。
- EOS/max_tokens 后 deallocate；本轮请求 ID 去重，并要求前一轮完成 postprocess 才能重新 schedule。

该优先级可能牺牲等待请求的 TTFT；KV 紧张时还可能丢弃已做的部分 prefill。持续 decode 流量占满预算时，不承诺 prefill 的等待时间上界。一次请求所需完整 reservation 超过总池容量时显式报错。

## 6. CUDA Graph 怎么处理

Decode 保留原 `prepare_decode -> graph.replay()`；prefill 保留动态 `prepare_prefill -> eager`。未改变 graph capture、bucket、padding、slot_mapping 或 Attention API。ModelRunner 的 `is_prefill` 仍描述一个同质 sub-batch，整个 iteration 的阶段由 ScheduleOutput 表达。

实验通过 graph proxy 计数真实的 `replay()` 调用，而非只观察开关。单卡完成 GPU 验证；多卡进程通信接口未改，做了 Sequence 序列化检查，但未在多卡硬件上运行。

## 7. 测试结果与复现

`python -m unittest discover -s tests -v` 覆盖：

- 保留的 7 项融合测试：数值/类型/shape、不改输入、真实 compiled baseline、回退开关、非默认 stream/CUDA Graph、非对齐 storage offset、非法输入、极值与相消。
- 13 项调度测试：纯 prefill/decode、3 decode+1021 prefill、3000/512 分块、FIFO 300/500/1000、联合预算、3/4/5/7/8/9 block 边界、EOS 释放、尾部抢占、部分 prefill reservation 压力、prefix reuse、重复调度/错误 postprocess、不可能容量、序列化、随机状态不变量（部分测试含多个场景）。
- 2 项执行测试：Engine 同轮先 decode 后 prefill；真实 GPU tensor 的跨物理 block slot_mapping、positions、cu_seqlens、context_lens。

CPU 调度边界测试 block_size=4；真实模型使用 FlashAttention 所支持的原配置 block_size=256。GPU metadata 测试仅构造 size=4 的位置 tensor，不以该尺寸启动 Attention。

端到端测试固定模型 Qwen3-0.6B、BF16、temperature=0.8、seed。原采样器不接受 temperature=0，因此采用：单请求原采样输出完全一致；混合请求按 request/position 进行 teacher forcing，保证历史相同，比较 16 个有效 logits 行。每行预设 relative L2<0.03 且 cosine>0.999，检查有限值并报告 max_abs/top1。3 种 budget 开 Graph，加 budget512 关 Graph。容差内一致不等于 bitwise 一致，也不承诺调度改变后随机生成的整段 token 一致。

在 WSL 中进入本目录，用已经安装依赖的 Python 运行：

```bash
src/nano-vllm-upstream/.venv/bin/python -m unittest discover -s tests -v
src/nano-vllm-upstream/.venv/bin/python benchmarks/run_mixed_suite.py --action correctness
src/nano-vllm-upstream/.venv/bin/python benchmarks/run_mixed_suite.py --action benchmark
src/nano-vllm-upstream/.venv/bin/python benchmarks/write_mixed_report.py
```

迁移电脑时复制整个本目录和相邻 `nano-vllm-cuda` 对照目录，安装同一依赖环境，修改脚本里的模型绝对路径。生产 `LLM` 使用方式不变：从本副本安装/导入 `nanovllm` 即启用新调度；`NANOVLLM_FUSED_OPS=all` 保留两项 CUDA 融合。

## 8. Benchmark 口径

真实数字见 `MIXED_REPORT.md`，原始记录见 `results_mixed/`。固定两版都启用 CUDA 融合和 Decode Graph；token budget=128/512/1024，max_num_seqs=16，max_model_len=4096。每组单独加载模型，完成预热；A/B 和 B/A 两轮，每轮 3 次，合计每种实现/预算/场景 6 次。没有计入加载、编译、Graph capture 或 warmup 时间；没有 logits hook 进入计时。

纯 decode 使用 4 条已产生首 token 的请求；纯 prefill 的 300/500/1000 token 请求各生成 1 token。混合场景先准备同样的 4 条 decode，再注入 64、3000 或 4×3000 token 新请求。旧请求共生成 32 token，新混合请求各生成 16 token；ignore_eos=True 保持工作量。不同案例/重复使用不同 prompt，避免 warmup 的 prefix-cache 命中。

**这里没有网络流式接口。** 内部 `times` 是 postprocess 的 token-ready 时刻；`step_times` 是包含它的整个 step 返回时刻。mixed 的 decode 后面还可能有 prefill，所以两者不能混为一谈。报告同时列出内部停顿和 step 边界停顿；后者也只是外层轮询可见的时刻，原 Engine 只返回已经结束的请求，不逐 token streaming。E2E 取已完成请求从 step 返回的时刻。

旧 decode 请求的首 gap 由新负载注入时刻计起，表示“注入后到下一 token”的延迟；后续 gap 是实际相邻 token 间隔。TTFT 只统计新请求。TPOT=(末 token ready-首 token ready)/(测量期生成数-1)，不含旧请求的首 gap，因此同时列最大 gap，避免均值掩盖阻塞。阶段 throughput 以同一 runner 包装范围（准备输入、模型、采样）计算；overall throughput=本次输出 token 数/整段 wall time。

## 9. 面试讲解（2～3 分钟）

我在 nano-vLLM 已有 CUDA 算子融合和 Chunked Prefill 的基础上，实现了 Decode-priority Mixed Scheduling。原调度器优先处理 waiting prefill，返回一个请求列表和全局 is_prefill 标记，Engine 每轮只运行一种阶段。长 prompt 即使切块，仍可能连续占用多轮，使已经在生成的请求停顿。

我的设计是在一次调度决策里先给 running 请求各分配一个 decode token，并保护它们需要的 KV block，再用剩余 token budget 和序列名额安排 FIFO prefill。比如预算 1024，有 3 个 decoder，就可以再执行 1021 个 prompt token。通过 ScheduleOutput 明确表达两组请求，Engine 先执行 decode，再执行 prefill；这样保留原 Decode CUDA Graph，Prefill 仍走 eager，不重写 Attention 或采样器。

正确性重点是状态和 KV 生命周期。中间 chunk 只推进 cached tokens，不追加生成 token，也不提前进入 running；最终 chunk 才转入 decode。分配和引用计数沿用 BlockManager，Decode 先预留资源，内存不足时保持原尾部抢占，并处理部分 prefill 占有 reservation 导致的无进展风险。测试覆盖 block 边界、prefix reuse、抢占、EOS 和随机状态不变量，还比较了真实模型在相同历史下的 logits。

性能部分我比较了三种预算、五种负载，并区分内部 token-ready 和整个 step 返回时间。具体数字引用本项目 MIXED_REPORT 的同一行，不能把调度收益说成单个 Attention kernel 加速。这个设计主要减少连续 prefill 对 decode 的阻塞，但两条路径仍然顺序执行；小预算可能增加调用和分块开销，新请求 TTFT、总吞吐也可能变差。KV 紧张时，优先保护 decode 还可能导致部分 prefill 重算，所以我把延迟改善和吞吐/TTFT 代价一起报告。
