# 复现与证据范围

## 先运行不依赖模型权重的检查

使用 README 的安装命令，在仓库根目录运行：

```bash
export PYTHONPATH="$PWD"
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=12.0
python -m unittest discover -s tests -v
python scripts/full_project/kernel_validation.py --label local
```

第二条命令会 JIT 编译 `src/qk_norm_rope/kernel.cu`，验证 138 组独立算子条件和 294 组真实模型中间结果，覆盖非连续输入、边界 position、输入不变性、两种 block 配置和 CUDA Graph replay。fixture 用 `weights_only=True` 加载。输出在 `results/full_project_20260919/kernel*_local.json`，不覆盖发布数据。

两份 `.pt` 是经过选择的输入/输出数值 fixture，约 12.2 MiB，不是完整模型 checkpoint。其 SHA256 可由发布的 kernel validation JSON 和实验 manifest 交叉核验。比较 native 编译路径的严格一致性依赖 PyTorch/Triton 编译上下文；其他版本如果不一致，应记录并重新诊断，不能直接放宽 gate 宣称复现成功。

## 接入模型的入口

在仓库根目录运行 Python；确保 `PYTHONPATH` 包含仓库，使用支持 Q16/K8/D128 的 Qwen3-0.6B：

```python
from scripts.full_project.adapter import install
from nanovllm import LLM, SamplingParams

install()  # 必须在创建模型和 capture CUDA Graph 之前；会先编译扩展
llm = LLM(
    "/absolute/path/to/Qwen3-0.6B",
    enforce_eager=False,
    tensor_parallel_size=1,
    max_num_seqs=8,
    max_num_batched_tokens=4096,
    max_model_len=4096,
    kvcache_block_size=256,
    num_kvcache_blocks=64,
    gpu_memory_utilization=0.75,
)
outputs = llm.generate(
    ["Explain CUDA kernel fusion."],
    SamplingParams(temperature=1.0, max_tokens=32, ignore_eos=True),
)
print(outputs[0]["text"])
```

不调用 `install()` 就是原生对照。adapter 是实验接入，不是通用推理插件：固定 Q16/K8/D128、BF16、相同 norm epsilon；不符合条件的路径回退。`native_graph` 模式针对本次真实模型编译上下文；普通 eager 示例或不同编译历史不自动继承模型 logits 严格一致性结论。

## 完整实验复现：先建新目录

保留冻结脚本的原始源码和 SHA，避免改写历史证据。历史脚本使用以下本机路径：

| 配置 | 位置 |
|---|---|
| Python | `/home/xietaibo/nano-vllm/.venv/bin/python`，在 `scripts/profiling/environment.sh` |
| 模型 | `/home/xietaibo/models/Qwen3-0.6B`，在 `benchmarks/baseline.py` |
| CUDA | `/usr/local/cuda-12.8` |
| Nsight Systems | `scripts/full_project/collect.sh` 和 `scripts/discovery/capture.py` 中的 2026.5.1 路径 |

在原冻结环境中，使用空的新目录运行：

```bash
python scripts/full_project/create_reproduction.py /absolute/path/to/new-run
cd /absolute/path/to/new-run
bash scripts/full_project/reproduce.sh
```

该流程依次执行数值 fixture gate、无 kernel 名过滤的全链路采集、模型准备、模型正确性与局部微基准、NCU、10 组配对 A/B；包含多轮完整推理，耗时明显长于 smoke test。已有 manifest 会阻止覆盖。

换机器时必须先配置自己的 Python、模型与 profiler 路径，重新冻结 baseline、输入及源码 manifest；不能改路径后继续声称与历史源码哈希一致。本仓库的完整实验脚本是冻结环境的复现记录，不是跨机器自动安装器。模型文件也需通过 baseline manifest 的哈希检查。

## 如何复核发布数据

[published_20260921](../results/published_20260921) 保留原始小型 JSON 和 CSV。`ab_01_native.json` 至 `ab_10_fused.json` 是 20 个进程的 280 次测量请求；`ab_summary.json` 保留配对统计方法、区间及输入哈希。

可把整份 `results/published_20260921` 复制到新的临时目录，并在 Python 中将 `scripts.full_project.analyze_ab.OUT` 指向副本，再调用 `main()`，重新计算 A/B 汇总。不要原地覆盖发布记录。

`manifest.json` 的运行源码及协议哈希对应[测量源码归档提交](https://github.com/123bawanglong/nano-vllm-inference-optimization/tree/23d2240f0df79550235ec7fe64f1412036a1b918)。当前版本将协议整理为 `docs/experiment_protocol.md`，并更新了准备、运行脚本的协议路径；核心 CUDA 和 baseline 源码保持不变。重新运行会生成新的源码与协议哈希，历史 manifest 不作改写。历史 JSON 内路径是采集时路径；发布数据统一放入 `published_20260921`。

NCU 原始报告、Nsight Systems 原始报告/SQLite、模型权重和完整 logits 张量未上传。可查阅原生截图、NCU 导出表、SASS 和排名 CSV；重新在 GUI 中交互分析需自行重跑采集。原生截图内容未重绘；实验文档的少量解释性文字经整理，不改变测量数据。

## 性能边界

- workload 是固定种子的合成 token ID，主实验 B=1/4/8 × P=64/256 × O=256，加 B1/P2048/O32；微基准额外含 B2。
- Prefill 生成首个输出 token；后续是 O−1 个 Decode step。O256 的早期步为 1–32，晚期为 224–255；中间步仍计入完整 Decode。
- CUDA Graph 保持开启，融合减少图内节点，不是每个节点都对应一次独立 CPU launch。
- profiling 与正式计时分开；NCU replay 下的时间不能直接拿来算热推理加速比。
- exact logits、局部延迟与端到端延迟是独立验证维度。局部下降约 70% 不代表整模型加速约 70%。

更详细的计时定义见 [baseline protocol](benchmark_protocol.md)。
