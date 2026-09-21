# 实验依据

本目录是2026-09-21实验的精选原始文件，迁移时保持字节不变；不是本次仓库整理重新测得的结果。`SHA256SUMS` 同时记录这里的数据和 `tests/fixtures/` 两份数值输入的哈希。

| 文件 | 内容 |
|---|---|
| `discovery_summary.json` | 全链路统计与归属 |
| `ab_summary.json` | 七种配置、10组配对、区间和逐对结果 |
| `micro.json` | 九组局部微基准及数值检查 |
| `numerical_v1_*`、`numerical_v2_*`、`numerical_fused.json` | 失败版本与最终版本数值结果 |
| `integration_native.json`、`integration_fused.json` | 真实模型节点对比 |
| `kernel_validation_final.json`、`kernel_model_fixture_validation_final.json` | 算子与模型 fixture 检查 |
| `workloads.json` | 固定输入及随机种子 |

[完整原始结果、20个A/B进程文件、idle日志、NCU导出及SASS](https://github.com/123bawanglong/nano-vllm-norm-rope-fusion/tree/2d18b908c635f63f2b6d99fa850607a6355052bc/results/published_20260921) 固定指向清理前提交，保留在Git历史；不会随当前目录精简而丢失。

历史JSON中的源码路径和哈希属于当时的实验身份，不对应当前整理后的目录。新实验需重新冻结当前源码，不能拿历史manifest验证当前版本。[历史测量源码](https://github.com/123bawanglong/nano-vllm-norm-rope-fusion/tree/23d2240)。
