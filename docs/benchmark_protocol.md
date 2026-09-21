# Stage 0 baseline protocol

本协议定义融合实现之前的原生基线，用于固定输入、运行环境和测量口径。第三方 fork 的性能结果不计入本项目测量。

## 固定对象

- Upstream commit: `bb823b3e06983d71485a8e1f23715ebd87d98ef8`。
- 原生 nano-vLLM Qwen3 路径、FlashAttention、BF16、TP=1；decode CUDA Graph 开启，prefill 沿用原生 eager/torch.compile 路径。
- KV page=256 tokens，固定 64 pages；max_model_len=4096，max_num_seqs=8，max_num_batched_tokens=4096，gpu_memory_utilization=0.75。
- 唯一推理引擎补丁：尊重显式指定的 KV 容量；默认自动容量公式不变。容量不足报错，禁止静默缩小实验。模型层、RMSNorm、RoPE、Attention 和 cache kernel 均为 upstream。
- `baseline_manifest.json` 保存源码 SHA256、模型权重/config/tokenizer JSON 的 SHA256、输入文件 SHA256；`environment.json` 保存实际导入路径和依赖版本。
- `baseline.patch` 只包含受跟踪的 model_runner 改动；新增 helper 和工具源码保存在仓库中，并对参与运行的源码单独 hash。

## 输入与运行次数

主场景：B=1，P=64/256，O=256。对照场景：B=4/8，同样 P/O。额外 prefill 对照：B=1，P=2048，O=32。

输入是固定种子的合成 token ID，不是自然语言质量测试。每个场景 1 次预热和 2 次正式测量，输入不同以避免重复 prefix cache 命中；3 个独立 Python 进程读取同一份输入。temperature=1，ignore_eos=True，固定输出长度。原生 sampler 不使用 temperature=0。

初始化、模型加载、初次编译和 Graph capture 单独记录，不计入热推理。所有进程按相同场景顺序执行。每个场景预热结束后立即测量两次；这是首轮 baseline 的波动刻画，不是正式 custom A/B 证据。

## 计时范围

- `e2e_ms`：已 tokenized 请求入队开始，经过全部 `llm.step()`，到 CUDA synchronize 返回。包含请求构造、scheduler、模型执行、sampling、结果 token ID 收集；不含 tokenizer、detokenizer、进程启动及模型初始化，也不代表网络服务端到端延迟。
- `output_tokens_per_s = B * O / e2e_seconds`，分母包含 prefill。
- `decode_step_ms`：decode 的完整 `llm.step()` host wall-clock 平均时间。原生 sampler 的 `.tolist()` 每步同步。包含准备输入、Graph replay、logits/sampling 和 scheduler 开销，**不是某一个 CUDA kernel 的耗时**。
- `prefill_step_ms`：一次 prefill step 的 wall-clock 时间。prefill 会生成第一个输出 token；后续恰好 O-1 次 decode。
- `engine_first_token_ms`：入队开始到首次 step 完成；这里只针对同长度、同时到达且单次完成 prefill 的批次，不等价于一般在线请求 TTFT。
- 保留每一步时间和完整输出 token ID。正式比较必须保持相同测量工具和范围。

## 完整性与正确性边界

每次请求检查：prefill tokens=B*P、decode tokens=B*(O-1)、每条输出长度=O、prefill step=1、decode steps=O-1、结束时 KV blocks 全部释放。额外 host guard 检查 prefix cache hit=0、preemption=0。未达条件立即失败，不接受隐藏的 workload 变化。

独立 eager 进程记录第一层 B=1/4/8 的输入 shape/stride/storage_offset、slot_mapping、KV cache 布局，以及 prefill/两个 decode step 的全量 logits；要求有限值并保存到本地 `.pt`。该进程的时延不用于性能表。eager 布局记录不单独证明 CUDA Graph 数值等价。正式 Graph 基线通过成功捕获和执行相应 batch 的原生 graph 验证可运行性。

这些是原版可运行性和后续对照数据，不是 fused kernel 正确性证明。后续 custom 需要独立算子数学参考、cache 内容检查和受控相同 token 历史的逐层/模型比较；不能仅靠生成文本或 top-1 相同验收。

## 统计和可重复性

先对每个进程同一场景的两次测量取中位数，再报告三个进程中位数及 min/max；不把同一进程内几百个 decode step 当独立实验样本。n=3 只描述当前机器波动，不宣称置信区间、显著加速或跨硬件泛化。

GPU 同时承担 Windows 显示任务，未锁频；保存进程前后 GPU 状态快照，但快照不能排除运行中干扰。微小差异需要后续交错配对 baseline/custom、多轮独立进程检验。每次新运行使用新目录，已有结果禁止覆盖。

## 重跑

在 Windows 工作区根目录 PowerShell 执行：

```powershell
wsl -d Ubuntu-24.04 -- bash nano-vllm-fusion-learning/scripts/run_baseline.sh
```

脚本复用 `/home/xietaibo/nano-vllm/.venv/bin/python`，显式 PYTHONPATH 指向独立 worktree，独立 compiler cache，不安装或修改依赖。当前绝对模型和 Python 路径写在工具中；迁移机器时先更新路径，再建立新 manifest。

结果在 `results/baseline_<timestamp>/`；汇总使用：

```powershell
wsl -d Ubuntu-24.04 -- /home/xietaibo/nano-vllm/.venv/bin/python nano-vllm-fusion-learning/scripts/summarize_baseline.py nano-vllm-fusion-learning/results/baseline_<timestamp>
```
