# Q/K RMSNorm + RoPE 实验协议

## 范围

- 模型为 Qwen3-0.6B，BF16，head_dim=128，Q16/K8；仅替换 Decode 的 Q/K RMSNorm→RoPE，保留 Prefill、KV cache 写入、Attention 和采样路径。
- 基线模型源码与 baseline manifest 固定。适配器在模型初始化、CUDA Graph capture 前安装；不支持的形状使用原实现。
- 全链路 profiling 不按目标 kernel 名过滤。分别统计累计耗时和调用次数，并结合源码比较矩阵计算、Attention、归一化、MLP 和采样候选。该分析是已有定向实验后的完整复核，不作为首次盲选目标的记录。

## 正确性

- 以实际 compiled native 输出为参照，要求逐元素相等；保留 Norm→RoPE 的 BF16 舍入边界。
- 数值失败单独记录，检查归约顺序、FMA 与舍入路径，不在看到结果后放宽阈值。
- 独立算子覆盖真实输入、随机输入、非连续 stride、position 边界、CUDA Graph 和输入不变性。
- 模型检查覆盖七种配置的全部 step logits 哈希；在选定早晚期及页边界检查全部层的新写 KV，并验证正常采样的完整输出。

## 性能

- 局部微基准使用真实首层 QKV fixture，B=1/2/4/8；CUDA Graph 内重复调用，十组交替顺序的配对测量。
- 正式性能测试以正确性通过为前提：十个独立进程对，奇数对 native→fused，偶数对 fused→native。
- 每个进程先预热全部七种配置，每种配置测两个请求；配对内顺序一致，跨对轮转。
- 保留所有轮次、背景 GPU 快照及原始结果。主指标为每种配置的配对端到端延迟下降率，补充 Decode、Prefill、吞吐和 PyTorch allocated/reserved 显存。
- 报告全部十对结果及 bootstrap 区间，不预设正收益。
- profiling 与正式计时分开；NCU 冷缓存 replay 时间只用于结构分析。

## 输出

运行结果写入 `results/full_project_20260919/`；发布测量位于 `results/published_20260921/`。已有实验 manifest 不覆盖。实验说明与原生截图见 [EXPERIMENT.md](EXPERIMENT.md)。
