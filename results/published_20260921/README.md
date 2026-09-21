# 2026-09-21 冻结实验数据

此目录为本地 `results/closed_loop_20260921` 及其 `replay/results/full_project_20260919` 中经过选择的原始小型结果，JSON/CSV/SASS 字节保持不变。目录名中的 20260919 是脚本固定输出名，不代表沿用 9 月 19 日的性能测量。

- `discovery_summary.json`、`ranking_by_*`：全链路归属与排名。
- `native_*`、`final_*`：NCU 指标导出与 SASS；原始报告未上传。
- `numerical_v1_*`、`numerical_v2_*`、`numerical_fused.json`：失败版本与最终版本模型 gate。
- `kernel_validation_final.json`、`kernel_model_fixture_validation_final.json`：算子和模型 fixture gate。
- `kernel_model_diagnostic_replay_v*.json`：实际模型中间结果诊断。
- `integration_*.json`、`micro.json`：真实图节点与局部微基准。
- `ab_??_*.json`、`ab_summary.json`：所有 A/B 进程结果、逐步时间、输出 token、配对统计及输入哈希。
- `manifest.json`、`workloads.json`：冻结代码和输入定义。

GPU UUID、本机路径等来源字段保留作实验身份核验；不含认证凭据。该目录只读，不用于新实验输出。
