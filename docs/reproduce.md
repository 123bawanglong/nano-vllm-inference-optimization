# 复现实验

使用 Linux / WSL、RTX 5080、CUDA Toolkit 12.8、PyTorch 2.11.0+cu128，以及匹配的 Triton、FlashAttention、Transformers 和 Ninja。准备本地 Qwen3-0.6B 权重，激活对应 Python 环境，并设置 `CUDA_HOME` 和 `PATH`。精确数值一致不保证跨编译器版本成立。

在仓库根目录运行：

```bash
python -m pip install -e . --no-deps
python -m unittest discover -s tests -v
python -m scripts.experiment kernel --out results/kernel-check
python -m scripts.experiment prepare --model /path/to/Qwen3-0.6B --out results/run1
python -m scripts.experiment profile --tool systems --out results/run1
python -m scripts.experiment profile --tool compute --out results/run1
python -m scripts.experiment validate --out results/run1
python -m scripts.experiment micro --out results/run1
python -m scripts.experiment ab --out results/run1
python -m scripts.experiment analyze --out results/run1
```

入口使用当前 Python；Nsight 工具从 PATH 查找，或传入 `--nsys /path/to/nsys`、`--ncu /path/to/ncu`。加 `--dry-run` 可查看命令。每阶段单独记录 `logs/`，已有输出不覆盖；失败后保留日志，使用新目录重试。

## 固定配置和统计口径

- Qwen3-0.6B、BF16、TP=1、CUDA Graph；Q16/K8/D128。仅融合 Decode 的 Q/K RMSNorm 和 RoPE，保留 Prefill、Attention 和 KV cache 写入。
- 7种请求：B1/4/8 × 输入64/256 × 输出256，加 B1/输入2048/输出32；KV cache 固定64页，每页256 token。输入使用固定种子，避免前缀复用及抢占。
- `prepare` 冻结当前源码、模型和输入，执行3个原生进程；每种配置先预热1次，再测2次。新运行不能沿用旧 manifest。
- Systems 采集3轮完整请求，以 NVTX 标记每个 step，展开 CUDA Graph node，不设 kernel 名称过滤。分别统计 Prefill、全部 Decode、早期 Decode 和晚期 Decode。
- 输出256时，早期为 step 1–32、晚期为224–255；输出32时，分别为1–15、16–31。累计时间包含中间 step，分阶段百分比的分母是 GPU kernel 耗时之和。
- 按累计时间及调用次数分别排序，再结合源码判断依赖和中间读写。NCU 的 Roofline、访存和 stall 用于解释假设；profiler 耗时不充当正式加速比。
- `validate` 依次验证两份 fixture、原生/融合模型的同历史 logits 与选定 KV 位置、真实图节点替换；`micro` 检查9组形状的数值和4→1节点。
- A/B 在上述门禁通过后执行10组独立进程配对，交替 native/fused 先后顺序，并轮换 case 顺序；每 case 每进程2次正式测量，共280次请求。保留全部轮次。
- 统计先取每进程两次测量的均值，再计算10组配对相对变化；95%区间使用20,000次配对 bootstrap，为逐配置描述性区间，不是七种配置的联合显著性结论。桌面GPU不锁频。
- 端到端计时含调度、Prefill、Decode、采样，不含模型加载、首次编译、tokenizer 和网络。

仓库中保留的是历史结果及固定数值输入；新实验写入被 Git 忽略的 `results/`。[数据来源与完整历史结果](results/README.md)。
