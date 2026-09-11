# 来源与修改范围

本项目基于 GeeeekExplorer/nano-vllm 的 MIT 许可代码，作者 Xingkai Yu。上游提交为 `bb823b3e06983d71485a8e1f23715ebd87d98ef8`，上游许可文本保留在仓库根目录及各源码快照内。

`src/nano-vllm-upstream/` 保留该提交在原实验环境中的 26 个跟踪文件，`upstream_manifest.json` 可验证其字节哈希。上游 README、配置和示例中的原路径属于原始快照，没有为适配本项目而改写。

`src/nano-vllm-cuda/` 相对上游的生产代码变动：

- 修改 `layers/layernorm.py` 和 `layers/activation.py`，接入有条件的 CUDA dispatch 和 fallback。
- 新增 `layers/cuda_fused.py`、`csrc/fused_ops.cpp`、`csrc/fused_ops.cu`。

`src/nano-vllm-cuda-mixed/` 相对融合版的生产代码变动：

- 修改 `engine/scheduler.py` 和 `engine/llm_engine.py`。
- 新增 `engine/schedule_output.py`。
- ModelRunner、Sequence、BlockManager、Attention、采样器、模型计算和 CUDA kernels 沿用融合版本。

公开仓库对生产源码做了目录整理，没有重新修改调度或 kernel 数学逻辑。`source_manifest.json` 记录发布时源码哈希。测试和 benchmark 仅做源码/模型/结果路径适配；模型目录由 `NANOVLLM_MODEL` 指定。root `pyproject.toml` 定义默认安装 mixed 版本，并保留上游作者、许可证及来源链接。

报告与 JSON 是原实验的历史记录。公开前仅替换个人机器的路径元数据，未改性能数值；未提交模型权重、二进制、logits tensor、个人学习配置或无关练习。
