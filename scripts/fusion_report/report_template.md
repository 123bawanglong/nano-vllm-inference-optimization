# Q/K RMSNorm + RoPE 算子融合实验

这次我想做的事情是：先从真实推理里找可以减少的开销，再写一个 CUDA kernel，把它接回 nano-vLLM，最后看整个推理有没有变快。不是先写一个孤立的算子，再只拿 microbenchmark 的结果说模型加速了。

本文按“截图 → 观察 → 诊断 → 下一步”的顺序整理实际实验。V0/V1/V2 是本文对原生实现、初版融合、正确性修正的叙述分组，不代表额外跑了三套完整 benchmark。全链路、正确性和 A/B 来自 **2026-09-19** 的冻结实验；Roofline 与配套 NCU 指标是 **2026-09-20** 对同一冻结实现的补采，用来补充解释，不冒充当时已经采到的证据。

> 图片说明：时间线与结果表是读取真实 NSYS/JSON/CSV 后生成的**数据查看器截图**，不是 Nsight GUI 截图；Roofline 图依据补采 NCU 的计数器及官方 section 公式重建，不是示意图。每节附原始报告或数据链接，可在 Nsight 中复查。

## 实验设置

| 项目 | 设置 |
|---|---|
| 模型 | nano-vLLM + Qwen3-0.6B，28 层 |
| Attention 形状 | Q 有 16 个 head，K/V 各 8 个 head，head_dim=128 |
| GPU | RTX 5080，16 GB，84 个 SM |
| 软件 | PyTorch 2.11.0+cu128、Triton 3.6.0、CUDA Toolkit 12.8 |
| 执行方式 | BF16，TP=1，原生 torch.compile、FlashAttention 和 CUDA Graph 均保留 |
| KV cache | 64 个 block，每 block 256 token，两个版本相同 |
| 测试输入 | 固定随机 token IDs、seed、采样设置；不是语义质量数据集 |
| 主要 workload | batch=1/4/8 × 输入=64/256、输出=256，共 6 组 |
| 补充 workload | batch=1、输入=2048、输出=32 |

一次输出 256 token 的请求，实际是 **1 次 Prefill + 255 次 Decode**。batch=8 时，一个 Decode step 同时推进 8 条序列，不是一次只生成整批中的一个 token。

## V0：先看原生实现，确定哪里值得动

### Nsight Systems：全局采集

![完整采集检查](fusion_experiment_assets/01_capture.png)

我先不按 RMSNorm、RoPE 的名字过滤，直接采完整的 7 个请求。CUDA Graph 用 `--cuda-graph-trace=node` 展开到单个操作；在 `llm.step()` 外加 NVTX 范围，名字包含 workload 和 step index。当前固定 workload 的 step 0 对应 Prefill，后面对应 Decode。

复核得到 **661,895 条 GPU kernel 事件、1,568 个模型 step、0 条未归属事件**，7 个请求输出与 baseline 相同。历史确认集另外做了 3 轮同规模完整采集。

这里的 66 万是执行事件数量，不是 66 万种算子。0 未归属指分析范围内提取的事件都能归到模型 step，也不表示 GPU 以外的所有开销都统计成了 kernel。

**诊断：先确认采集完整、请求执行正确，后面的排序才有数据基础。下一步分别按耗时和次数排。**

原始证据：[NSYS 报告](../results/full_project_20260919/discovery_recheck/native_1.nsys-rep)、[逐步分析](../results/full_project_20260919/discovery_recheck/analysis_1.json)、[NVTX 采集代码](../scripts/discovery/capture.py)。

### 累计耗时排名

![耗时排名](fusion_experiment_assets/02_rankings.png)

三轮确认里，各 workload 的矩阵投影合计占 Decode kernel 时间 **66.77%～75.18%**，Attention 占 **13.80%～22.74%**，它们才是主要耗时。

Q/K RMSNorm + RoPE 合计只有 **2.97%～3.81%**；另一次完整复核约为 **3.19%～3.83%**。所以这里还不能直接写“我发现 RMSNorm 和 RoPE 是最大的热点”。

这张表的分母是 **GPU kernel duration 总和**，不是用户等待请求完成的 wall time；后面计算端到端收益时要换成真正的请求计时。

### 调用次数排名

![调用次数排名](fusion_experiment_assets/02b_frequency.png)

再看次数，每个 Decode step 有 **56 次 Q/K RMSNorm、56 次 RoPE**，合起来 112 次。原因很直观：28 层，每层 Q/K 各做一次 Norm、一次 RoPE。

单次耗时很小，但是重复很多。这样我就有了第二个调查方向：除了研究最大的矩阵计算，还可以检查这批小 kernel 之间有没有多余的中间张量和执行边界。

**诊断：耗时排名告诉我谁是大头；次数排名让我注意到高频的小算子链。次数多只是线索，还要检查数据依赖。**

原始证据：[耗时排名](../results/discovery_20260918_234946/confirmation/ranking_by_time.csv)、[次数排名](../results/discovery_20260918_234946/confirmation/ranking_by_count.csv)、[多轮统计](../results/discovery_20260918_234946/report_metrics.json)。

### 时间线与源码：为什么是这四个算子

![真实 NSYS 局部事件顺序](fusion_experiment_assets/02c_sequence.png)

沿着第一层 Attention 看下去，能看见独立的 Q Norm、K Norm、Q RoPE、K RoPE，后面才是 KV cache 写入。实际数据流是：

```text
hidden_states → QKV 投影 → 拆成 Q、K、V
                            Q → RMSNorm → RoPE ─┐
                            K → RMSNorm → RoPE ─┼→ 原来的 Attention 路径
                            V ─────────────────┘
```

我回到源码确认，归一化后的 Q/K 紧接着由 RoPE 消费，没有别的分支要求保留 Norm 的中间输出。原生实际 trace 是四个 kernel；这里的 `torch.compile` 没有跨越这条链把四个操作合在一起，不能只看 Python 代码就猜它已经融合。

![候选算子的源码审查](fusion_experiment_assets/03_candidates.png)

同时检查其他候选：矩阵投影和 Attention 占比高，但替换范围大；Residual + RMSNorm、SiLU × up 已有融合；采样链仍有机会，但涉及词表归约和随机数语义，验证更复杂。

**诊断：我选择 Q RMSNorm、K RMSNorm、Q RoPE、K RoPE 做第一个原型，是因为边界明确、开销可测、实现范围可控。这是一个值得尝试的融合点，不是证明它收益最大。下一步用 NCU 看这条链的执行特征。**

原始证据：[多候选审查](discovery_source_audit.md)、[Attention 源码](../nanovllm/models/qwen3.py)。

## V0：结合 Roofline 和 NCU 分析融合机会

以下补采使用真实模型初始化后的第一层输入，**B1、Q16/K8、dim128**，不是所有 batch 的硬件计数器概括。NCU 为 node 级、kernel replay、冷缓存、未锁频；它会扰动运行，不能拿这里的时长替代正式 benchmark。

### Roofline

![原生四个 kernel 的实测计数器 Roofline](fusion_experiment_assets/roofline_native.png)

Roofline 横轴是计算强度，纵轴是浮点吞吐。这里内部主要用 FP32 算，所以看 **Single Precision Roofline**，不能因为输入是 BF16 就拿 Tensor Core 的峰值当它的计算上限。

这张图按 NCU section 的口径，用 `FADD + FMUL + 2×FFMA` 计算浮点操作吞吐，结合 DRAM、L2、L1 的对应流量指标得到不同层级的点；rsqrt、类型转换、地址计算和 shuffle 不包含在这个 FLOP 计数中。每个小图的 roof 使用该次采样对应频率，图中的标记是从原始计数器计算出来的。

@@ROOFLINE_NATIVE@@

我看到的不是“已经贴着带宽上限跑”，而是这些小 kernel 的浮点吞吐与对应 roof 还有明显距离。**低计算强度只说明数据搬运值得关注，不足以单独证明 DRAM 带宽已经成为瓶颈。**

**诊断：Roofline 不能单独解释这些微秒级 kernel。接着看 Speed of Light、grid 和调度情况，判断是不是工作量太小、可并行的 warp 不够。**

原始证据：[原生 Roofline NCU 报告](../results/roofline_supplement_20260920/native_roofline.ncu-rep)、[原始指标](../results/roofline_supplement_20260920/native_raw.csv)、[NCU 原始 section 公式](../results/roofline_supplement_20260920/SingleRoofline.section)。

### Speed of Light / Compute Workload Analysis

![补采 NCU 吞吐指标](fusion_experiment_assets/ncu_sol.png)

@@SOL_OBSERVATION@@

Memory 汇总指标并不等于 DRAM 带宽，Compute(SM) 汇总指标也不等于 FP32 FLOP 峰值利用率。这里要分别看细项，不能把“访存比计算的百分比高”直接翻译成“显存带宽打满”。

### Launch Statistics / Occupancy

![补采 NCU 启动与资源指标](fusion_experiment_assets/ncu_launch.png)

原生四个 kernel 的 grid 分别只有 **16、8、8、4 个 block**，而显卡有 84 个 SM。单独一次执行连每个 SM 分到一个 block 都做不到。

即使理论 occupancy 允许更多 warp 驻留，实际也没有那么多 block 可以派出去。因此“理论 occupancy 很高”不能说明 GPU 已经充分利用。此时我不会直接照搬 GEMM 的思路加 shared memory tile，因为还没有证据说明这里需要跨 head 的数据复用。

### Scheduler Statistics / Warp State Statistics

![补采 NCU 调度与等待指标](fusion_experiment_assets/ncu_scheduler.png)

@@SCHED_OBSERVATION@@

Long Scoreboard 提示存在对 L1TEX 相关访存结果的依赖等待，但**等待访存延迟不等于带宽饱和**。图中的等待数值保留导出指标的归一化口径，不能把它直接说成“占总运行时间多少百分比”。

这些 kernel 很短，跨 replay 的比率还容易受波动影响，所以我把它们用于辅助判断，不把某一个 stall 数字当成最终性能结论。关于短 kernel 和多 pass 指标误差，参见 [NVIDIA Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)。

### Memory Workload Analysis：我实际能消掉什么

![数据流与逻辑访存核算](fusion_experiment_assets/memory_plan.png)

原来 RMSNorm 把归一化后的 Q/K 写入 global memory，RoPE 随后再读出来。我想消掉的是这次中间输出的写回和重新读取：

```text
元素数：(16 + 8) × 128 = 3072
BF16 中间结果大小：3072 × 2 byte = 6144 byte = 6 KiB
一次写 + 一次读：12 KiB / 层 / token
28 层合计：336 KiB / token
```

这是按数据依赖计算的**逻辑访问量**，不是 NCU 测到的 DRAM 节省量。中间数据可能命中缓存，实际 DRAM 流量、事务数和访问指令数要分别看。

### 诊断：决定融合方案

把证据连起来，我的判断是：**这条链工作量很小，四个独立节点反复出现，Norm 输出又马上被 RoPE 使用。可以尝试把中间值留在寄存器里，减少执行节点和中间读写。**

原生四个节点 → 一个融合节点；每层减少 3 个，28 层每个 Decode step 理论上减少 **84 个 GPU kernel 节点**。因为 baseline 已开启 CUDA Graph，这不等于每步减少 84 次 CPU `cudaLaunchKernel` 调用。

这个判断还只是可验证的假设：究竟更快多少，要看实现后独立计时。也不能仅靠低吞吐就断言 CPU launch overhead 是唯一瓶颈。

## V1：实现 Q/K RMSNorm + RoPE 融合 kernel

### 线程组织和数据流

![实现中的数值边界，展示最终冻结代码](fusion_experiment_assets/06_kernel.png)

这里用最终冻结代码说明线程布局和数据流；初版失败及后来增加归约模式的过程在下一节展开。

我先固定一个 block 来看：默认 **128 个 thread，也就是 4 个 warp**，每个 warp 负责一个 head，各个 warp 的 head 独立，不需要共享归约结果。

head_dim=128，每个 warp 有 32 个 lane，所以 RoPE 主路径每个 lane 持有 4 个元素。`i=2×lane`，它处理 `i、i+1、i+64、i+65`，对应半分式 RoPE 的两对数据。RMSNorm 的归约顺序则按 native 编译特化组织，部分分支会再次读取用于求和的数据，不能说实现总是只读一遍。

warp 内通过 `__shfl_xor_sync` 交换部分和，得到该 head 的平方和；不需要跨 warp 合并，因此没有用户 shared memory 缓冲区，也不需要 `__syncthreads()`。

扩展到整个 grid：一个 token 有 16 个 Q head + 8 个 K head，一共 24 个 warp 的工作量。默认每 block 4 个 warp，所以 B1 是 **6 个 block**。这样把 Q 和 K 两支放到同一次 launch 中，仍保持它们各自归一化。

```text
读 BF16 Q/K → 转 FP32 → 平方和归约 → rsqrt → 乘权重
                      ↓
             显式保留 BF16 舍入
                      ↓
               FP32 RoPE 旋转
                      ↓
              写出 BF16 Q/K
```

**诊断：融合消掉的是中间 tensor 的全局读写和节点边界，不是把 RMSNorm 数学计算删掉；也不是使用 Tensor Core。**

代码：[冻结实验 kernel](../results/roofline_supplement_20260920/replay_source/src/qk_norm_rope/kernel.cu)、[Python 扩展入口](../src/qk_norm_rope/__init__.py)。截图展示的是原实验冻结源码。

### 正确性：局部通过后，模型为什么还失败

![第一版模型级失败](fusion_experiment_assets/07b_model_failure.png)

第一版局部测试已经通过，但接到真实模型时，B1 的 logits 仍然存在不一致，B4/B8 却能对齐。这说明只测单个随机输入、只看生成 token 是否相同，都不够。

我检查 native 实际生成的代码和 PTX，发现 B1 的独立测试与真实模型选择了不同的 RMSNorm 归约组织。浮点加法不满足结合律，求和顺序改变后，结果可能跨过 BF16 的舍入边界，再传到后续层。

另外，原来 Norm → RoPE 之间存在 BF16 舍入。融合之后即使中间值留在寄存器，也要显式保留这次舍入，不能直接全程 FP32 一算到底，再只在最后转 BF16。

**诊断：数学公式相同不保证逐元素完全相同。我要对齐实际 compiled native 的计算顺序和舍入边界，不能为了通过测试就临时放宽误差阈值。**

原始证据：[保留的失败版本](../results/full_project_20260919/attempt_1/)、[编译上下文诊断](../results/full_project_20260919/kernel_reduction_context_diagnosis.json)。

## V2：对齐真实模型的归约路径，再验证

### 数值修正与结果

![归约原因与复核](fusion_experiment_assets/07c_diagnosis.png)

我增加明确的 `native_graph` 模式，模型接入时按实际编译上下文使用对应归约顺序。它不根据输入数值或参考答案选结果。微基准也先初始化真实 engine，再调用模型的 Norm/RoPE 模块，避免比较对象不是模型实际执行的版本。

最终独立测试 **138 条**、捕获模型 fixture 检查 **294 条**通过；实际模型中间结果检查共 **2,380 行**，修复前有 8 行不一致，修复后为 0。

![完整模型正确性](fusion_experiment_assets/10_model_correctness.png)

7 个配置合计 **1,568 个模型 step 的输出 logits exact**。这里 Prefill 检查每条请求用于生成首 token 的最后输入位置，不是所有 prompt 位置；另在选定 step 检查全 28 层新写入的 KV 槽位。

模型对拍采用相同 token 历史；正式 A/B 另外检查正常采样输出。这个结论只覆盖固定模型、软件栈和已测输入，不是任意版本都逐位一致的保证。

### Roofline / NCU：融合后发生了什么

![融合 kernel 的实测计数器 Roofline](fusion_experiment_assets/roofline_fused.png)

@@ROOFLINE_FUSED@@

再对照前面的 NCU 资源表，融合后只有一个节点，但 grid 仍然很小，寄存器使用也增加了。不能写成“融合使 GPU 满载”或者“occupancy 提升就是所有加速的原因”。

单个融合节点承担了原来四个节点的工作。它的 replay 时长比原来某一个小节点长，并不表示整条链变慢；反过来，把四个冷缓存 replay 时长相加算出百分比，也不是热态加速比。要看独立的整链微基准。

原始证据：[融合 Roofline 报告](../results/roofline_supplement_20260920/fused_roofline.ncu-rep)、[补采数据汇总](../results/roofline_supplement_20260920/derived_metrics.json)。

### 局部 microbenchmark

![局部实测结果](fusion_experiment_assets/08_micro.png)

同一组真实输入，native 和 fused 交替测量。每张 CUDA Graph 放 1,000 次链调用，用 CUDA Event 测量后折算单次耗时，共 10 组成对测量；节点计数另开 profiler 检查。

@@MICRO_TABLE@@

9 组 fixture 中，原生 **4 个 kernel → 融合 1 个 kernel**，局部延迟下降 **61.9%～73.0%**。B2 是额外局部验证点，不是正式端到端新增 workload。

**诊断：融合链本身确实更快了。但微基准反复访问同一组小输入，缓存状态与整模型不同。下一步必须接回真实模型，看整体收益。**

原始证据：[micro.json](../results/full_project_20260919/micro.json)。

## V2 接入 nano-vLLM：确认不是只跑了一个 demo

![适配器代码](fusion_experiment_assets/09_integration.png)

在 engine 初始化和 CUDA Graph capture 之前安装 adapter。满足条件的 Decode 分支，在 QKV projection 后调用融合函数，然后进入原来的 Attention 和输出投影。

Prefill 仍走原实现；KV cache 写入、Attention、V 和采样都不在融合范围内。`cos_sin_cache` 是 RoPE 的三角函数缓存，不是 KV cache。

![真实 Decode Graph 节点变化](fusion_experiment_assets/09b_graph.png)

代表性真实 Decode step 的 kernel 数量从 **405 降到 321**，刚好少了 **84 个**；出现 28 次融合 kernel，独立 RoPE 节点消失，原来的 28 次 KV cache 写入仍在。

**诊断：这证明模型实际走到了新路径。现在才可以用端到端 A/B 回答“整个推理有没有变快”。**

原始证据：[adapter.py](../scripts/full_project/adapter.py)、[native trace 统计](../results/full_project_20260919/integration_native.json)、[fused trace 统计](../results/full_project_20260919/integration_fused.json)。

## 最终端到端 A/B

### 测试方法

固定 10 个独立进程对，native→fused 与 fused→native 交替；同一对的 workload 顺序相同，跨对轮转。每个进程先预热全部配置，每种配置测 2 次，一共 **20 个进程、280 个测量请求**。

正式计时不开 profiler，不包含模型加载、编译和初始化。计时范围是 engine 添加请求到生成完成，包含调度和采样，不是网络服务端延迟。机器是未锁频的桌面 GPU，启动前检查背景占用，保留全部样本，包括变慢的配对。

先平均每个进程同配置的两次请求，再计算每对 `(native - fused) / native`，报告 10 对的平均下降率和 pair 级 bootstrap 95% 区间。因此表中的配对下降率不一定等于两个总均值直接相除。

### 结果

![端到端成对实验](fusion_experiment_assets/11_ab.png)

@@AB_TABLE@@

六组输出 256 token 的 workload，平均配对延迟下降 **1.92%～3.60%**，每个配置的描述性 95% 区间为正。batch=1、输入 2048、输出 32 的区间跨 0，所以只能写**收益方向仍不确定**。

所有正式请求的正常采样输出 token 都与冻结 baseline 一致。

### 波动和显存

![保留所有配对与显存统计](fusion_experiment_assets/12_variation.png)

个别配对确实退化，没有把它们删掉。Prefill 没有修改，但测量也有波动，所以不能把端到端每一点变化都严格归因于融合。本实验没有对 Prefill 的变化做独立的因果消融。

两种实现的 peak allocated 都是约 **3021.20 MiB**，稳态 allocated 都是约 **2965.16 MiB**，reserved 范围也一样。**这次没有测到显存容量下降**；减少中间读写量，不等于峰值显存一定下降。

原始证据：[完整 A/B 统计](../results/full_project_20260919/ab_summary.json)。

### 最后我怎么解释这个结果

局部快了六七成，整个模型只快几个百分点，是合理的：这条链本来只占 Decode GPU kernel 时间的几个百分点，矩阵运算、Attention、调度和采样仍然要做。长输入、短输出时，没有优化的 Prefill 比重更高，收益也可能被稀释。

这次能够确认的是：**在固定 Qwen3-0.6B、RTX 5080、BF16 和已测 workload 下，通过全链路定位高频小算子链，手写 CUDA 将四个节点合成一个；正确性通过后接入真实推理，六组 Decode 较多的配置获得了可测的端到端延迟下降。**

不能把这个结果扩大成“所有模型都有同样提升”，也不能把 Norm/RoPE 说成整条推理最大的耗时。换模型、精度、head_dim、编译器版本或输入规模，都需要重新验证。

## 数据、图片与版本说明

这份文档参照用户的 GEMM 实验文档组织表达，但没有照搬 GEMM 的结论：本实验没有观察到同样的 shared bank conflict 问题，也没有实现 shared 双缓冲或 Tensor Core 优化。

2026-09-20 补采前，版本校验发现当前 `kernel.cu` 删去了若干注释和空行，哈希与旧记录不一致；对比确认非注释计算文本相同。为保持证据一致，补采使用独立目录中**哈希完全匹配旧实验**的源码，未覆盖当前学习代码，也没有修改旧实验 manifest。

- [本报告生成与核验脚本](../scripts/fusion_report/build_report.py)
- [Roofline 补采脚本](../scripts/fusion_report/collect_roofline.sh)
- [补采报告、CSV、冻结副本及版本说明](../results/roofline_supplement_20260920/)
- [完整旧实验流程与复现入口](PROJECT_WALKTHROUGH.md)
- [NVIDIA Roofline 与指标说明](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)

分享时请把本 MD 与同目录 `fusion_experiment_assets` 一起带上。图片和原始结果链接分开保存；只有 MD 和图片也可以离线阅读正文，但复查 `.ncu-rep`、JSON 等需要保留项目结果目录。
