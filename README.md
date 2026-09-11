# nano-vLLM Inference Optimization

基于 [nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm) 的 LLM 推理优化项目：**CUDA 算子融合 + Decode-priority Chunked Prefill Mixed Scheduling**。

技术栈：Python / PyTorch / C++ / CUDA / Triton / FlashAttention。

本项目保留上游、仅算子融合、算子融合加混合调度三份实现，用于分别验证两项优化。它是基于上游框架的实验扩展，不是从零实现的推理框架。

## 实现内容

**CUDA 算子融合**

- Add+RMSNorm：一个 thread block 处理一行，FP32 中间计算，warp shuffle 和 shared memory 完成分层归约。
- SiLU+Mul：融合激活与逐元素乘法，使用 `Pack<T,4>` 向量化访存，未对齐时走标量回退。
- 通过 PyTorch C++/CUDA Extension 加载，支持 FP16/BF16，保留不支持输入的原实现回退，以及当前 CUDA stream/Graph 行为。

**Decode-priority 混合调度**

- 每轮先给 running 请求安排 Decode，并预留 KV block；再利用剩余 token budget 与请求名额安排 FIFO Prefill。
- 长 Prompt 分块执行，中间 chunk 只推进 KV，不追加 completion token。
- `ScheduleOutput` 表达 Decode/Prefill 两组请求；Engine 在同一步里先执行 Decode，再执行 Prefill。
- Decode 保留 CUDA Graph replay；Prefill 保留 eager 路径。两组顺序执行，不是统一 tensor batch 或 GPU 并发执行。
- 覆盖 Prefix Cache 复用、block 边界、抢占、EOS 释放和部分 Prefill reservation 的回收。

## 实测结果

环境：RTX 5080 单卡、Qwen3-0.6B、BF16、PyTorch 2.11.0+cu128、WSL Ubuntu 24.04。下列结果是固定实验负载，不代表在线服务 SLO。

| 实验 | 对照 | 结果 |
|---|---|---|
| SiLU+Mul，2048 tokens，intermediate=3072 | 原 `torch.compile` → CUDA | 15.024 → 12.080 μs，约 **1.24×** 加速 |
| 模型离线总耗时，CUDA Graph 关闭 | 上游 → 仅 CUDA 融合 | 两组已测负载耗时降低 **5.5%～6.1%** |
| 模型离线总耗时，CUDA Graph 开启 | 上游 → 仅 CUDA 融合 | 耗时增加 **2.1%～7.2%**，没有获得收益 |
| 3000-token Prompt + 4 条 Decode，budget=512 | 仅融合 → 融合+混合调度 | step 边界最大停顿 **94.47 → 20.78 ms**，降低 **78.0%** |
| 同一混合场景的新请求首 token 延迟 | 仅融合 → 融合+混合调度 | **90.05 → 107.54 ms** |
| 同一混合场景的输出吞吐 | 仅融合 → 融合+混合调度 | **710.03 → 708.86 token/s**，基本持平 |

**边界：**原 `torch.compile` 已将两组算子分别融合成一个 kernel，手写 Add+RMSNorm 的设备时间仍较慢。混合调度减少连续 Prefill 引起的长停顿，但可能增加 TTFT、平均 token 间隔或降低总吞吐。Engine 没有逐 token 网络 streaming；step 边界时间不是客户端流式延迟。

详细方法、重复测量、失败实验和误差范围见 [融合报告](docs/FUSION_REPORT.md)、[混合调度报告](docs/MIXED_REPORT.md)。

## 代码入口

```text
src/
  nano-vllm-upstream/       上游提交的原始快照
  nano-vllm-cuda/           仅 CUDA 融合的对照实现
  nano-vllm-cuda-mixed/     CUDA 融合 + 混合调度，默认安装版本
tests/                     22 项单元及 GPU 测试
scripts/
  fusion/                  原版 vs CUDA 融合的测试脚本
  mixed/                   融合版 vs 混合调度版的测试脚本
  smoke.py                 可配置模型路径的推理验证
docs/                      实现说明、复现步骤、性能报告与来源记录
results/fusion/             融合实验 JSON 与生成的 compiled kernel 参考
results/mixed/              混合调度实验 JSON
```

直接查看：[CUDA kernels](src/nano-vllm-cuda-mixed/nanovllm/csrc/fused_ops.cu) · [Python/CUDA 接入](src/nano-vllm-cuda-mixed/nanovllm/layers/cuda_fused.py) · [Scheduler](src/nano-vllm-cuda-mixed/nanovllm/engine/scheduler.py) · [Engine](src/nano-vllm-cuda-mixed/nanovllm/engine/llm_engine.py) · [ScheduleOutput](src/nano-vllm-cuda-mixed/nanovllm/engine/schedule_output.py)。

## 安装与运行

需要 Linux/WSL、受支持的 NVIDIA GPU、匹配的 PyTorch/CUDA Toolkit 和 FlashAttention 环境。Python 3.10–3.12；实验使用 Python 3.12。首次运行会编译 CUDA Extension，需要 `nvcc`、C++ 编译器和 Ninja。

```bash
git clone https://github.com/123bawanglong/nano-vllm-inference-optimization.git
cd nano-vllm-inference-optimization

# 先准备与本机 GPU 匹配的 PyTorch、Triton 和 FlashAttention 环境。
python -m pip install ninja
python -m pip install -e . --no-deps

export NANOVLLM_MODEL=/absolute/path/to/Qwen3-0.6B
python scripts/smoke.py --model "$NANOVLLM_MODEL"
python -m unittest discover -s tests -v
```

`pip install -e .` 默认安装 `src/nano-vllm-cuda-mixed/nanovllm`。为了明确控制依赖，上例使用 `--no-deps`；依赖和环境说明见 [复现指南](docs/REPRODUCE.md)。不包含模型权重或预编译二进制。

## 测试与复现

22 项测试涵盖算子数值误差、类型/布局回退、非默认 stream、CUDA Graph、调度预算、FIFO、Prefix Cache、抢占、EOS 和 KV block 边界。

完整模型对照使用请求/生成位置对齐的 teacher forcing，验证 4 组配置共 64 个有效 logits 行，均满足预设 relative L2<0.03、cosine>0.999；单请求原采样输出一致。不是任意文本质量或所有随机输出的一致性证明。

混合调度 benchmark 包含 5 类负载、3 种预算，每组 A/B、B/A 两种进程顺序共 6 次。命令见 [REPRODUCE.md](docs/REPRODUCE.md)。历史 JSON 中的绝对导入路径已改为仓库相对路径，性能数值保持不变。

## 来源与许可

上游：[GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)，基于提交 [`bb823b3e06983d71485a8e1f23715ebd87d98ef8`](https://github.com/GeeeekExplorer/nano-vllm/commit/bb823b3e06983d71485a8e1f23715ebd87d98ef8)。保留原 MIT LICENSE 和作者信息；具体修改范围见 [PROVENANCE.md](docs/PROVENANCE.md)。
