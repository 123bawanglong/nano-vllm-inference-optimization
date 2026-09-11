# 复现指南

在仓库根目录执行下列命令。公开目录结构与早期本机实验不同：历史报告里的 `nano-vllm-cuda`、`nano-vllm-cuda-mixed` 对应 `src/` 下同名目录；`benchmarks/` 对应 `scripts/fusion/` 或 `scripts/mixed/`；旧脚本名 `write_mixed_report.py` 未作为公开入口保留。以本文件命令为准。

## 环境

原测量环境是 Python 3.12、PyTorch 2.11.0+cu128、CUDA Toolkit 12.8、RTX 5080、Qwen3-0.6B BF16。安装 `pyproject.toml` 中的 torch、triton、transformers、flash-attn、xxhash，并准备 ninja。FlashAttention 的安装与 GPU/PyTorch/CUDA 组合有关，本仓库没有声称任意版本组合可用。

```bash
python -m pip install ninja
python -m pip install -e . --no-deps
export NANOVLLM_MODEL=/absolute/path/to/Qwen3-0.6B
export NANOVLLM_FUSED_OPS=all
```

模型路径必须是已下载的本地模型目录。执行脚本时使用当前 Python 环境；无硬编码的原电脑模型路径。环境变量 `NANOVLLM_FUSED_OPS` 可设为 `none`、`add_rms`、`silu` 或 `all`；benchmark 会按对照方案设置该变量。

## 单元测试与推理

```bash
python -m unittest discover -s tests -v
python scripts/smoke.py --model "$NANOVLLM_MODEL"
```

测试显式导入 `src/nano-vllm-cuda-mixed`。融合数值测试从随仓库附带的 `src/nano-vllm-upstream` 读取参考实现，不依赖另一台电脑的源码位置。

## CUDA 融合对比

```bash
# GPU 算子正确性与微基准
python scripts/fusion/run_suite.py --stage micro
# 完整模型对比，包括 Graph ON/OFF、单项消融、正反运行顺序和 logits 校验
python scripts/fusion/run_suite.py --stage engine
```

也可用 `--stage all` 一次运行。上游与仅融合版分别从两个独立源目录导入，每个后端单独进程运行。结果写入 `results/fusion/`；复跑会覆盖同名历史记录，若要保留历史测量，请先复制该结果目录。

## 混合调度对比

```bash
# 先做完整模型正确性：budget128/512/1024 + Graph ON，budget512 + Graph OFF
python scripts/mixed/run_mixed_suite.py --action correctness
# 正确性通过后，再运行 5 类负载、3 种 budget、A/B 与 B/A 顺序
python scripts/mixed/run_mixed_suite.py --action benchmark
```

结果写入 `results/mixed/`。两个版本都启用 CUDA 融合，比较只改变调度方案。逐请求记录内部 token-ready 时间、所在 step 的返回时间、TTFT、TPOT、完成时间，并计数真实 Graph replay。日志和 `.pt` logits 不提交 Git；JSON 可审阅。

可设置 `NANOVLLM_RESULTS_DIR=/absolute/path/to/new-results`，让 mixed 脚本和 suite 写入另一目录，保留仓库里的历史结果。正确性和性能测试须使用同一个结果目录。

单独运行示例：

```bash
python scripts/mixed/bench_mixed.py --backend mixed --budget 512 --graph 1 --action correctness
python scripts/mixed/bench_mixed.py --backend fusion --budget 512 --graph 1 --action correctness
```

每种 backend 必须在独立进程运行，避免同名 Python 模块缓存污染。请勿同时运行多份 GPU benchmark。`scripts/` 通过文件位置定位源码，root 安装的默认 mixed 版本不会覆盖指定的 baseline 导入。

## 指标口径

混合调度的两组模型执行是顺序的。Decoder 内部产生 token 后，该 step 可能还需执行 Prefill；因此内部 token-ready 时间不能替代 step 返回时间。原 Engine 仅返回完成请求，不提供逐 token 网络 streaming。旧请求的首 gap 从新负载注入时刻起算，后续才是相邻 token 的时间间隔。

历史报告中的 `summary.json` 为当时结果汇总，不会随脚本复跑自动刷新。复跑所得逐次 JSON 是新测量的事实依据，不应混用历史汇总和新样本。本文档没有把硬件/软件环境差异下的测量当作同一次实验。
