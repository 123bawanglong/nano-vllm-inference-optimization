> 历史实验记录：以下性能数值未因发布重测或改写。原始实验目录与公开仓库目录的对应关系、可运行命令见 [REPRODUCE.md](REPRODUCE.md)。

# CUDA 融合实现与性能实测

测量日期：2026-09-09。结论：已完成 CUDA Add+RMSNorm、SiluAndMul 与独立副本集成。在本次 Qwen3-0.6B 测试中，关闭 CUDA Graph 时有小幅收益；开启 CUDA Graph 时整体更慢。不能声称这两个手写 CUDA kernel 全面优于原项目的 torch.compile。

## 环境与基线

- GPU：NVIDIA GeForce RTX 5080；PyTorch：2.11.0+cu128；CUDA runtime：12.8。
- CUDA Toolkit 12.8，WSL Ubuntu-24.04；Qwen3-0.6B，BF16，单卡。
- 原始提交：`bb823b3e06983d71485a8e1f23715ebd87d98ef8`，26 个原跟踪文件 SHA256 校验一致。
- 原目录 `src/nano-vllm-upstream`；修改只在本工作区的 `nano-vllm-cuda` 副本。
- 主要 baseline 为原项目的 `@torch.compile` 实现。未编译 eager 只作额外参考。
- 两边同一 Python 环境、权重、固定请求 token IDs、输出长度、seed、GPU 内存比例 0.6。
- max_num_seqs=16，max_num_batched_tokens=4096，max_model_len=1024。

## 正确性

- `tests/test_cuda_fused.py`：7 个测试方法通过，包括 FP32/FP16/BF16 数学结果、双输出、输入不修改、非整齐/空形状、零值与抵消、非法输入、非默认 stream、CUDA Graph 修改输入后重放、非对齐 storage offset、模块开关和 fallback。
- 原编译算子对照：FP16/BF16 的 Add+RMSNorm 与 SiluAndMul 相对 L2 误差均低于 5e-4。
- 模型校验：2 条请求、32 个 prompt tokens、4 个固定 teacher-forced step，覆盖两种 Graph 模式、单项消融和逆序复测。每步后续输入固定，避免随机采样混入误差。
- 所有记录中最大 logits 绝对误差：0.171875；最大相对 L2：0.018593；最小余弦相似度：0.999713。
- 已测位置的最低 top-1 一致率：100.0%。这只覆盖上述输入，不代表任意文本、长上下文或随机生成序列逐位一致，也不是 perplexity/任务质量评估。
- 预设模型阈值为每步相对 L2 < 0.02、余弦相似度 > 0.999；最终通过，未放宽阈值。

## 模型原版与 CUDA 副本

每种配置在两个独立进程中各运行 5 次，第二组反转 A/B 顺序。下表将 10 次样本合并取中位数。输入已 tokenized；计时包括请求添加、step、每步 GPU 同步和输出 detokenization，不包括模型加载、编译、预热或文本 tokenization。因此这是固定离线 workload 的测量，不是在线 serving 的 TTFT/TPOT/SLO benchmark。输出吞吐=生成 token 数/总耗时。

| CUDA Graph | 请求数 × prompt 长度 → 输出长度/请求 | 原版耗时 ms | CUDA 耗时 ms | 耗时变化 | 原版输出 tok/s | CUDA 输出 tok/s |
|---|---|---:|---:|---:|---:|---:|
| 开启 | 1 × 128 → 64 | 209.16 | 224.24 | +7.21% | 306.0 | 285.4 |
| 开启 | 8 × 256 → 64 | 256.62 | 262.01 | +2.10% | 1995.1 | 1954.2 |
| 关闭 | 1 × 128 → 64 | 856.35 | 809.16 | -5.51% | 74.7 | 79.1 |
| 关闭 | 8 × 256 → 64 | 968.54 | 909.46 | -6.10% | 528.6 | 563.0 |

正数表示变慢，负数表示变快。各独立进程对的耗时变化如下；幅度随进程变化，不能把小差异视为稳定硬件指标。

| Graph | batch | 第一对进程 | 逆序进程对 |
|---|---:|---:|---:|
| 1 | 1 | +9.78% | +2.77% |
| 1 | 8 | +3.67% | +0.81% |
| 0 | 1 | -5.20% | -1.05% |
| 0 | 8 | -8.68% | -4.34% |

## 单项消融（Graph 开启）

以下仅一次独立进程、每进程 5 次；用于观察组成，不足以确认小幅度差异。

| 方案 | batch=1 耗时 ms | batch=8 耗时 ms |
|---|---:|---:|
| 原版 | 207.64 | 254.14 |
| 仅 CUDA Add+RMSNorm | 215.90 | 264.80 |
| 仅 CUDA SiluAndMul | 218.37 | 257.33 |
| 两者都启用 | 227.95 | 263.46 |

## BF16 算子设备时间

单位 μs/次，CUDA Graph 重放下的 CUDA Event 时间。每个 Graph 捕获 64 次调用，每轮重放 10 次，5 轮中位数，原版/CUDA/eager 交替顺序。模型 hidden=1024，MLP intermediate=3072。低精度 eager 舍入语义与 compiled 有轻微差别。

| 算子 | token 数 | 原 torch.compile | CUDA | 未编译 eager | 相对 compiled 加速比 |
|---|---:|---:|---:|---:|---:|
| add_rms | 1 | 1.070 | 1.713 | 14.178 | 0.62× |
| silu | 1 | 0.830 | 1.105 | 2.628 | 0.75× |
| add_rms | 8 | 1.094 | 1.745 | 15.842 | 0.63× |
| silu | 8 | 1.227 | 1.185 | 3.392 | 1.04× |
| add_rms | 32 | 1.391 | 1.833 | 17.457 | 0.76× |
| silu | 32 | 1.835 | 1.539 | 3.981 | 1.19× |
| add_rms | 128 | 1.345 | 1.990 | 17.755 | 0.68× |
| silu | 128 | 1.537 | 1.596 | 4.768 | 0.96× |
| add_rms | 512 | 2.468 | 3.895 | 26.336 | 0.63× |
| silu | 512 | 4.304 | 4.020 | 12.038 | 1.07× |
| add_rms | 2048 | 6.353 | 7.534 | 60.438 | 0.84× |
| silu | 2048 | 15.024 | 12.080 | 41.508 | 1.24× |

加速比 >1 才表示 CUDA 更快。FP16 全部数据、5 轮原始样本和带 Python 提交的计时见 `results/microbench.json`。这里复用同一输入，可能命中 L2；不能据此计算实际 HBM 带宽。

## Profiler：kernel 数量

BF16，在 token 数 1 和 512 上得到相同的 kernel 数量：

| 算子 | 未编译 eager | 原 torch.compile | CUDA |
|---|---:|---:|---:|
| add_rms | 11 | 1 | 1 |
| silu | 2 | 1 | 1 |

## 如何解释结果

- 相对未编译 eager，融合减少中间 Tensor、全局内存读写以及 kernel launch；kernel 数量已被 profiler 验证。没有采集 Nsight 的 HBM 字节数或 occupancy，不能把这些量化为实测。
- 相对原项目的 torch.compile，双方已经都是单 kernel；不能再用“减少 launch 数”解释收益。当前 CUDA Add+RMSNorm 设备时间更慢，block 内同步、归约布局和访存指令都可能影响差距，尚未用 Nsight 单独归因。
- 向量化 SiluAndMul 在较大 token 数下设备时间更好，但小 token 数不保证获益。
- Graph 关闭时，单算子带 Python 提交的计时显示直接 CUDA 调用通常开销较小，模型也出现小幅改善。它与减少框架调度开销的解释一致，但不是 CPU 各阶段的严格归因实验。
- Graph 开启后，CPU 逐算子提交影响大幅减少，Add+RMSNorm kernel 的劣势更明显。当前副本是可运行的 CUDA 融合实验实现，不能推荐为此模型所有运行模式的默认性能替代。

## 数值修正和淘汰实验

- v1：标量 SiLU 与 block 归约；低精度中间舍入按 eager 写法保留。仅微基准，后续发现需要对齐 compiled。
- v2：尝试 warp-per-row RMSNorm，实测比 v1 更慢；模型相对误差超预设阈值，淘汰。
- 最终：回到较快的 block 归约，保留 vector4 SiLU，按实际生成的 Triton 代码消除中间低精度舍入。所有最终数据对应此版本。参考代码保存在 `results/compiled_reference_*.txt`。
- 原 FP32 Add+RMSNorm 有输入原地修改与输出别名；模型集成保留原 fallback。独立 FP32 CUDA 数学接口是 out-of-place，不声称复现原 FP32 别名。

## 显存与边界

KV dtype、容量算法和调度保持原版。以下是 Graph 开启首轮进程在完成 benchmark 后的PyTorch allocated bytes，包含模型与 KV；不是峰值显存，也不能作为算子临时显存的消融：

| 方案 | allocated MiB | KV blocks |
|---|---:|---:|
| 原版 | 8173.2 | 250 |
| CUDA | 8173.2 | 250 |

测试 GPU 同时承担桌面显示；未锁频、未隔离桌面负载。测量只适用于这些 shape、当前软件版本和硬件。没有验证多卡、训练、任意模型质量、长上下文性能或在线服务 SLO。

运行方法、代码入口和开关见 `CUDA_FUSION.md`。全部原始日志与 JSON 位于 `results/`。
