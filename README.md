# nano-vLLM：Q/K RMSNorm + RoPE CUDA 融合

基于 [nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm) 的推理优化实验：**全链路 profiling → 候选比较 → 手写 CUDA → 数值对齐 → 接入模型 → 端到端 A/B 验证**。

技术栈：C++ / CUDA / PyTorch / Triton / Nsight Systems / Nsight Compute。

## 实现与验证

1. **先定位，再选择。** Nsight Systems 以 CUDA Graph node 粒度采集，用 NVTX 区分 Prefill 与每个 Decode step；按累计耗时和调用次数分别排序，并审查矩阵计算、Attention、归一化、MLP 与采样等候选。
2. **形成可验证的融合假设。** 矩阵计算和 Attention 是主要耗时；Q/K Norm→RoPE 在七种场景中占 Decode GPU kernel 累计时间的 **3.12%～3.72%**，每步有 **112 个独立节点**。结合源码确认中间读写可消除，选择范围可控的融合原型。
3. **实现并解释。** NCU 的 Roofline、访存、调度和 SASS 分析辅助设计：Q=16 heads、K=8 heads、D=128 特化；每 warp 处理一个 head，每 block 四个 warp，把四个 kernel 合为一个。低吞吐或 Long Scoreboard 高本身不能证明 DRAM 带宽饱和。
4. **解决数值问题。** BF16 输入/输出、FP32 中间计算；保留 Norm→RoPE 的 BF16 舍入边界，并对齐实际编译上下文的归约顺序。V1→V2→V3 的严格一致 logits step 从 **7/1568 → 1056/1568 → 1568/1568**。
5. **接入真实推理再验证。** 在模型创建及 CUDA Graph capture 前安装 opt-in adapter，仅替换符合条件的 Decode 路径；Prefill、Attention、采样和 KV cache 写入保留原实现。

> 本项目融合的是 **Q RMSNorm、K RMSNorm、Q RoPE、K RoPE**；KV cache 写入仍是独立 kernel。这里的精确一致性限于冻结环境和已测试输入，不是跨编译器、跨模型的普遍保证。

## 实验结果

RTX 5080 · Qwen3-0.6B · BF16 · TP=1 · CUDA Graph · PyTorch 2.11.0+cu128 · CUDA Toolkit 12.8。以下为 **2026-09-21** 的完整复核数据。

| 检查 | 结果 |
|---|---|
| 全链路采集 | 661,895 条 GPU kernel 事件、1,568 个 step、0 条未归属事件 |
| 独立算子 / 模型中间结果 | 138 / 294 组 fixture 检查通过；2,380 组在线中间结果严格一致 |
| 模型数值 | 相同 token 历史下 1,568 个 step 的完整 logits 哈希一致；选定位置、全部 28 层的 KV 检查通过 |
| 实际 Decode 图 | 单步 kernel 数 **405 → 321**，28 个融合节点；28 次 KV 写入保留 |
| 局部微基准 | 9 组真实形状下，Norm＋RoPE 整段延迟下降 **67.24%～73.08%** |
| 端到端 A/B | 10 组配对、20 个独立进程、280 次测量请求，生成 token 全部一致 |

端到端延迟包含请求入队、Prefill、Decode、调度和采样；不含模型加载、首次编译、tokenizer 和网络服务。

| Batch / 输入 / 输出 | 原生 ms | 融合 ms | 平均配对延迟下降 | 描述性 95% 区间 |
|---|---:|---:|---:|---|
| 1 / 64 / 256 | 833.54 | 830.80 | 0.31% | [-0.92, 1.49]% |
| 1 / 256 / 256 | 856.03 | 838.92 | 1.90% | [-0.79, 4.14]% |
| 4 / 64 / 256 | 903.85 | 876.43 | 2.95% | [0.46, 5.16]% |
| 4 / 256 / 256 | 951.75 | 915.67 | 3.76% | [1.96, 5.94]% |
| 8 / 64 / 256 | 938.97 | 906.18 | 3.47% | [2.42, 4.48]% |
| 8 / 256 / 256 | 1000.73 | 982.27 | 1.83% | [0.83, 2.84]% |
| 1 / 2048 / 32 | 145.46 | 142.33 | 2.09% | [0.54, 3.91]% |

五种配置的区间为正；两种 B=1、输出256的配置区间跨零，收益方向尚不明确。百分比按每对进程的相对差再平均，不等于表中两个均值直接相除。所有轮次保留；桌面 GPU 未锁频，区间不是七种配置的联合显著性结论。PyTorch allocated 显存未观察到下降。

## 分析过程与原生截图

[完整实验文档](docs/EXPERIMENT.md) 包含 Nsight Systems、NCU Roofline / Speed Of Light / Compute / Memory / Scheduler / Warp State / Source 的原生截图，以及数值和性能结果。

![Nsight Systems 累计耗时排序](docs/images/image-20260921181349154.png)

![融合前 NCU Roofline](docs/images/image-20260921174852958.png)

## 代码入口

| 文件 | 用途 |
|---|---|
| [kernel.cu](src/qk_norm_rope/kernel.cu) | CUDA 融合、warp 归约、BF16 舍入和输入检查 |
| [Python binding](src/qk_norm_rope/__init__.py) | JIT 编译、两种明确的归约上下文 |
| [adapter.py](scripts/full_project/adapter.py) | 模型 Decode 接入与原生回退 |
| [qwen3.py](nanovllm/models/qwen3.py) | 原生 QKV→Norm→RoPE→Attention 调用链 |
| [capture.py](scripts/discovery/capture.py) / [analyze.py](scripts/discovery/analyze.py) | 全链路 NVTX 采集和分阶段热点排名 |
| [kernel_validation.py](scripts/full_project/kernel_validation.py) | 独立算子、实际模型 fixture、Graph 和输入不变性检查 |
| [study.py](scripts/full_project/study.py) / [analyze_ab.py](scripts/full_project/analyze_ab.py) | 模型数值、微基准、集成节点统计、配对 A/B |
| [实验数据](results/published_20260921) | 已冻结 JSON、CSV、NCU 导出及失败版本结果 |

## 运行

需要 Linux / WSL、兼容的 NVIDIA GPU、CUDA Toolkit、PyTorch、Triton、FlashAttention 和 Ninja。当前默认编译目标为 RTX 5080 的 SM120；先阅读 [复现说明](docs/REPRODUCE.md)，尤其是冻结环境与 bitwise 对齐的限制。

```bash
git clone https://github.com/123bawanglong/nano-vllm-inference-optimization.git
cd nano-vllm-inference-optimization
# 先安装与 GPU 匹配的 PyTorch、Triton 和 FlashAttention。
python -m pip install ninja
python -m pip install -e . --no-deps
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=12.0
export PYTHONPATH="$PWD"
python -m unittest discover -s tests -v
python scripts/full_project/kernel_validation.py --label local
```

模型接入须在创建 `LLM` 之前调用 `scripts.full_project.adapter.install()`；普通 `LLM` 默认仍走原生实现。验证命令不下载模型；仓库包含两份经过选择的数值 fixture，不包含模型权重、二进制、安装包或原始 profiler 大报告。

## 来源与范围

原生框架来自 nano-vLLM 提交 `bb823b3e06983d71485a8e1f23715ebd87d98ef8`，保留 [MIT LICENSE](LICENSE)。本项目是框架之上的 CUDA 融合及实验扩展，不是从零实现框架。基础模型保留上游布局以维持冻结源码校验；自定义 CUDA 位于 `src/`，配套工具位于 `scripts/`。

仓库当前版本替换了早期 Add+RMSNorm / SiLU+Mul 与混合调度实验；旧内容仍可从 Git 历史查看。历史数据中记录的本机绝对路径用于来源追踪，不代表新机器上的路径。
