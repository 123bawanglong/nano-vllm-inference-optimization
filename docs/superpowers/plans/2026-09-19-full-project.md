# 完整 CUDA 融合项目复现计划

> 执行方式：subagent-driven-development。用户已批准完整流程并明确要求本轮实现 CUDA kernel、运行实验、生成含实际截图的 Markdown。此前“用户亲自写 kernel”仅适用于旧阶段。

## 范围与预先约定

- 复用并复核已完成的、无目标 kernel 过滤的 Nsight Systems 全链路发现（7 配置、3 轮确认），如实标明来源日期，不伪装成未知目标的首次发现。
- 原型：Qwen3-0.6B，BF16，head_dim=128，Q16/K8，decode 的 Q/K RMSNorm→RoPE；保留 prefill、KV store、Attention、采样原路径。
- 原始模型源码和 baseline manifest 不变。使用显式进程内适配器，在 CUDA Graph capture 前安装自写扩展；不支持的形状走原路径。
- 正确性门槛：先要求与实际 compiled native 输出逐元素相等。必须保留 Norm→RoPE 的 BF16 舍入，不以删除舍入换取速度。若失败，记录并诊断，不能事后悄悄放宽阈值。
- 微基准：真实首层 QKV fixture，B1/2/4/8；Graph 内重复调用，10 组交替顺序成对测量。计时与 profiler 分开。
- 模型检查：7 种配置，所有 step logits hash，对选定早/晚期及页边界检查全层新写 KV；native reference 历史固定。另验证正常采样完整输出。
- 正式性能：正确性过关后固定 10 个独立进程对，奇数 native→fused，偶数 fused→native；每进程先预热全部7配置，再各测2请求。配对中配置顺序相同、跨对轮转。不删慢样本，保留背景 GPU 快照。主指标是每配置配对端到端延迟下降率，辅以 decode、prefill、throughput 和 PyTorch peak allocated/reserved。报告全部10对及 bootstrap CI。
- NCU 的冷缓存 replay 时间只用于结构分析，不冒充无 profiler 热态延迟。

## 执行任务

1. [进行中] 核验全链路发现、源码审查和 NCU 原始证据，形成真实截图。
2. [进行中] CUDA 实现与边界测试；生成数值日志；独立规格和代码审查。
3. [待做] 微基准、kernel 数验证、NCU 指标对照。
4. [待做] 集成开关、模型级同历史正确性、正式成对 A/B。
5. [待做] 根据结果解释适用范围及退化；不保证正收益。
6. [待做] 生成口语化第一人称项目文档，每个数据→决策节点附真实截图及原始数据链接；验证图片和链接。

产物：`results/full_project_20260919/`、`src/qk_norm_rope/`、`scripts/full_project/`、`docs/PROJECT_WALKTHROUGH.md`、`docs/project_walkthrough_assets/`。
