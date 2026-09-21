# nano-vLLM Q/K RMSNorm + RoPE 融合

面向 Qwen3-0.6B 小 batch Decode 的 CUDA 算子融合实验：从全链路 profiling 选择候选，将 Q/K RMSNorm 和 Q/K RoPE 四个 kernel 合为一个，再接入 nano-vLLM 验证数值和端到端收益。

**范围：** BF16、Q=16 heads、K=8 heads、head_dim=128；每 warp 处理一个 head，每 block 4 个 warp。仅替换符合条件的 Decode 路径，KV cache 写入仍独立执行。

## 实验思路

1. **发现候选：** Nsight Systems 按累计时间和调用次数分别排名。矩阵计算与 Attention 占主要耗时，Norm→RoPE 占 Decode kernel 累计时间的 3.12%～3.72%，每步重复 112 个节点。
2. **确认可省的开销：** 时间线与源码确认四个独立节点及中间张量传递；结合 NCU Roofline、访存、调度和 SASS 分析，设计寄存器内融合。
3. **对齐数值：** FP32 计算、保留 Norm→RoPE 的 BF16 舍入边界，并对齐原生 Graph 的归约顺序；严格一致 logits 从 7/1568 → 1056/1568 → 1568/1568 step。
4. **验证收益：** 独立算子检查 → 局部微基准 → 模型接入 → 10 组配对 A/B；profiler 采集与正式计时分开。

## 结果

RTX 5080 · Qwen3-0.6B · BF16 · TP=1 · CUDA Graph · PyTorch 2.11.0+cu128 · CUDA 12.8。

| 项目 | 2026-09-21 测量结果 |
|---|---|
| 全链路采集 | 661,895 个 kernel、1,568 个 step、0 个未归属事件 |
| 正确性 | 138 组算子 + 294 组模型 fixture；1,568 个 step 的 logits 严格一致 |
| Decode 图节点 | 405 → 321；每层 4 个 Norm/RoPE 节点 → 1 个 |
| 局部整段延迟 | 9 组形状下降 67.24%～73.08% |
| 端到端延迟 | 7 种配置平均配对下降 0.31%～3.76%；5 种配置的描述性 95% 区间为正 |

两种 B=1、输出 256 的配置区间跨零；未观察到 PyTorch allocated 显存下降。端到端统计包含调度、Prefill、Decode 和采样，不包含模型加载、编译、tokenizer 或网络。桌面 GPU 未锁频；精确数值一致限于已测环境与输入。

[实验报告与原生截图](docs/EXPERIMENT.md) · [原始结果](results/published_20260921)

## 代码入口

| 位置 | 内容 |
|---|---|
| [kernel.cu](src/qk_norm_rope/kernel.cu) | CUDA 融合、归约和 BF16 舍入 |
| [adapter.py](src/qk_norm_rope/adapter.py) | Decode 接入，条件不满足时回退原生实现 |
| [qwen3.py](nanovllm/models/qwen3.py) | 原生 QKV→Norm→RoPE→Attention 调用链 |
| [experiment.py](scripts/experiment.py) | 统一实验入口、独立进程、输出隔离 |
| [discovery](scripts/discovery) | NVTX 全链路采集、Prefill/Decode 排名 |
| [full_project](scripts/full_project) | 数值检查、微基准、模型 A/B 和配对统计 |
| [tests](tests) | workload、统计口径、实验入口回归测试 |

## 复现

使用 Linux / WSL 和已安装 CUDA Toolkit 的 NVIDIA GPU 环境，先安装与 GPU 匹配的 PyTorch、Triton、FlashAttention。当前数值契约在上述 RTX 5080 环境验证，默认编译目标为 SM120。模型权重需自行准备为本地 Qwen3-0.6B 目录。

```bash
python -m pip install ninja
python -m pip install -e . --no-deps
# 激活安装了上述依赖的 Python 环境，并设置 CUDA_HOME / PATH。
python -m unittest discover -s tests -v

# 不加载模型：验证仓库内两份数值 fixture。
python -m scripts.experiment kernel --out results/runs/kernel-check

# 新实验：冻结当前代码、模型和输入，采集 3 轮 baseline。
python -m scripts.experiment prepare --model /path/to/Qwen3-0.6B --out results/runs/run1
python -m scripts.experiment profile --tool systems --out results/runs/run1
python -m scripts.experiment profile --tool compute --out results/runs/run1
python -m scripts.experiment validate --out results/runs/run1
python -m scripts.experiment micro --out results/runs/run1
python -m scripts.experiment ab --out results/runs/run1
python -m scripts.experiment analyze --out results/runs/run1
```

所有命令在仓库根目录执行，使用当前 Python；Nsight 工具默认从 PATH 查找，也可传 `--nsys /path/to/nsys`、`--ncu /path/to/ncu`。`--dry-run` 只显示命令；日志位于输出目录的 `logs/`。入口拒绝覆盖已存在的阶段日志，失败结果保留，重试请使用新实验目录。

固定 workload 为 B=1/4/8 × 输入64/256 × 输出256，另加 B1/输入2048/输出32；KV cache 为 64 页 × 每页256 token。A/B 前必须通过算子、模型数值、接入节点和微基准检查，10 组进程配对交替顺序执行。

模型接入在创建 `LLM` 前调用 `src.qk_norm_rope.adapter.install()`；默认 `LLM` 保持原生路径。[算子接口与数值契约](src/qk_norm_rope/README.md) 说明两种归约上下文。

## 数据与来源

`results/published_20260921/` 是历史冻结结果，新测量写入 `results/runs/`，不会改写原有 manifest 或截图。历史测量源码可从提交 [`23d2240`](https://github.com/123bawanglong/nano-vllm-inference-optimization/tree/23d2240) 追溯；本次入口整理不代表重新测得以上性能。

原生框架来自 [nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm) 提交 `bb823b3e06983d71485a8e1f23715ebd87d98ef8`，保留 [MIT LICENSE](LICENSE) 和上游布局；自定义实现位于 `src/`。仓库包含两份数值 fixture，不包含模型权重、构建产物和原始 profiler 大报告。早期实验保留在 Git 历史。
