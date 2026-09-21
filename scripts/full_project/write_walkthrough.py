"""Write the final narrative from verified artifacts, never from guessed outcomes."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/full_project_20260919'
def read(name):return json.loads((OUT/name).read_text(encoding='utf-8'))

def main():
    numerical=read('numerical_fused.json');micro=read('micro.json');ab=read('ab_summary.json')
    assert numerical['complete'] and numerical['exact_gate'] and micro['complete'] and ab['all_tokens_exact']
    reductions=[100*(1-r['median_us']['fused']/r['median_us']['native']) for r in micro['rows']]
    integration=read('integration_fused.json');native_integration=read('integration_native.json')
    steps=sum(r['step_count'] for r in numerical['rows'])
    text=f'''# 我怎样把 Q/K RMSNorm → RoPE 融合接进 nano-vLLM：一次完整实验

这份文档按“我先看到了什么，再据此做什么”的顺序写。源码、原始数据和截图都留在项目里，可以顺着链接复查。

先交代结果：我完成了手写 CUDA 原型和真实模型接入。局部从 **4 个 kernel 变成 1 个**，9 组真实输入的局部延迟下降 **{min(reductions):.1f}%～{max(reductions):.1f}%**；模型级 {steps:,} 个 step 的 logits 逐元素一致，选定 step 的全层新写 KV 也一致。最终端到端结果见第 9 节，不能把局部百分比直接写成模型加速比。

这里优化的是 **Decode 的 Q/K RMSNorm＋RoPE**。KV cache 写入仍然保留原实现，不把没有实现的 KV 融合写进项目成果。

## 0. 我先把对照条件固定下来

我想回答的具体问题是：“在这台机器、这个模型和这些请求上，删除 Norm 与 RoPE 之间的执行边界，能不能在保证结果正确的情况下缩短推理时间？”

所以先固定模型和运行方式，避免今天测的是一种条件、明天换了条件却把差异都算到 kernel 上。

| 项目 | 固定设置 |
|---|---|
| 模型 | Qwen3-0.6B 文本模型，28 层，Q16/K8，head_dim=128 |
| 推理框架 | nano-vLLM，基于 bb823b3e06983d71485a8e1f23715ebd87d98ef8 的独立 worktree |
| 设备 | RTX 5080，16 GB；同一张卡还负责 Windows 显示 |
| 软件 | PyTorch 2.11.0+cu128、Triton 3.6.0、CUDA Toolkit 12.8、FlashAttention 2.8.3.post1 |
| 精度和执行 | BF16，TP=1，CUDA Graph 开启；原有 torch.compile 保留 |
| KV cache | 固定 64 个 block，每 block 256 token；native/fused 一样 |
| 采样 | temperature=1，固定随机 seed，ignore_eos，输出长度固定 |
| 输入 | 固定随机 token ID；这是可复现的合成 workload，不是问答数据集或模型质量 benchmark |

| batch | 每条输入 token | 每条输出 token | 用途 |
|---:|---:|---:|---|
| 1 / 4 / 8 | 64 / 256 | 256 | 6 个组合，观察小 batch 及上下文变化 |
| 1 | 2048 | 32 | 补一个 Prefill 占比更高的对照 |

注意，输出 256 token 的请求包括 **1 次 Prefill 产生首 token＋255 次 Decode**，不是整条请求都属于 Decode。这里的端到端是引擎 `add_request → 生成完成`，包含调度和采样，排除模型加载、编译、HTTP 网络和 tokenizer；`engine_first_token_ms` 也只是引擎内首 token 延迟。

运行协议在 [预先写好的计划](superpowers/plans/2026-09-19-full-project.md)，冻结条件在 [manifest](../results/full_project_20260919/manifest.json)。固定 KV 容量是为了使对比公平，不是说真实部署只能用这么小的 cache。

**证据怎么读：** 历史的 3 轮全链路确认采集是已有发现依据，本轮又重采 1 轮完整请求复核。本文按分析逻辑组织，不假装我们从未知道 Norm→RoPE 这个候选。除第 4 节标明来源的 Nsight 软件截图外，其余是浏览器对真实 JSON/CSV/源码数据页的截图，页眉明确写“非 Nsight 软件界面”；不是手工画出的假 profiler 截图。各页脚列了原始文件和 SHA256。

## 1. 我先抓完整链路，不先过滤想优化的名字

一开始就只抓 RMSNorm 和 RoPE，会很容易得出“这两个算子值得优化”的循环结论。所以我用 Nsight Systems 抓完整请求，CUDA Graph 展开到 node，按 NVTX 把 Prefill、每个 Decode step 标出来。

本轮复核抓到了 **661,895 条 GPU kernel 事件，1,568 个模型 step，0 条未归属事件**；7 个请求的输出与冻结 baseline 一致。历史确认集有 3 轮相同规模的完整采集。到这里我才愿意相信后面的排序有完整数据基础。

![完整采集的数据质量检查](project_walkthrough_assets/01_capture.png)

原始证据：[本轮 nsys 报告](../results/full_project_20260919/discovery_recheck/native_1.nsys-rep)、[SQLite](../results/full_project_20260919/discovery_recheck/native_1.sqlite)、[逐 step 分析](../results/full_project_20260919/discovery_recheck/analysis_1.json)。这里没有按目标 kernel 名称过滤。

**我因此进入下一步：先算谁总共花了时间，再算谁被反复调用。**

## 2. 两种排序给了我不同的信息

先按累计时间看，历史 3 轮确认里，全部矩阵投影占 Decode GPU kernel 时间的 **66.77%～75.18%**，Attention 占 **13.80%～22.74%**。这两类才是大头。

Q/K Norm＋RoPE 只有 **2.97%～3.81%**；本轮独立复核是约 **3.19%～3.83%**。所以我不能说“profiling 发现它们是主要热点”。准确的说法是：“发现了一条每层反复执行、可能存在融合机会的小算子链。”

![累计耗时排序](project_walkthrough_assets/02_rankings.png)

然后我换成调用次数排序：Q/K Norm 每 step 共 56 次，RoPE 也有 56 次，加起来 **112 次**。单个很短，但每层都重复。

![调用频率排序](project_walkthrough_assets/02b_frequency.png)

这里的百分比分母是 **GPU kernel duration 总和**，不是请求 wall time。单次 median/p95 也是该角色下单个 kernel 的统计，不是整条链的耗时。桌面 GPU 上的长尾原样保留，不把每个长尾都解释成算法瓶颈。

原始证据：[累计时间排名](../results/discovery_20260918_234946/confirmation/ranking_by_time.csv)、[调用次数排名](../results/discovery_20260918_234946/confirmation/ranking_by_count.csv)、[三轮角色统计](../results/discovery_20260918_234946/report_metrics.json)。

**这两张表还不能直接决定优化谁。我的下一步是：回到源码，检查这些短 kernel 之间有没有可以消掉的数据搬运和执行边界。**

## 3. 我比较多个候选，再选择一个可验证的原型

我先沿着一次 Attention 调用看：QKV 投影输出被拆成 Q、K、V；Q/K 各过一次 RMSNorm，再各过 RoPE，然后进入 Attention。实际 trace 也能看到两次 Norm、两次 RoPE、一次 cache store 分开执行。

![真实 trace 的局部顺序](project_walkthrough_assets/02c_sequence.png)

继续看消费者：归一化后的 Q/K 紧接着被 RoPE 使用，中间没有别的分支必须保留这个张量。也就是说，这里确实有“写出去、又读回来”的边界。

但我没有只看这条链。矩阵投影、Attention、Residual＋RMSNorm、SiLU×up、采样链、Prefill 最后一个 Norm 都一起检查了。

![多候选审查及源码](project_walkthrough_assets/03_candidates.png)

审查后的判断是：矩阵和 Attention 值得研究，但替换和调优范围比较大；Residual＋RMSNorm、SiLU×up 已有融合，不能把它们现有的效果算成我的新收益；采样链仍然是有竞争力的候选，但要处理大词表归约和随机数语义。

我最后选择 Norm→RoPE，并不是证明它收益最大，而是因为 **有可重复测量的开销、明确的中间消费者、相对小的实现范围和能验证的融合假设**。这是一个适合先验证的工程选择。

源码审查：[完整候选记录](discovery_source_audit.md)、[Qwen3Attention](../nanovllm/models/qwen3.py)、[Norm](../nanovllm/layers/layernorm.py)、[RoPE](../nanovllm/layers/rotary_embedding.py)。

## 4. 再用 NCU 看它为什么有优化空间

Nsight Systems 帮我回答“在哪里花时间”；接下来 Nsight Compute 帮我看具体 kernel 的 grid、block、寄存器、occupancy 和访存指标。

下面这张是之前这份原生链报告在 Nsight Compute 中的真实截图，沿用用户提供的截图，**不是本轮重新截到的 GUI**。它有 5 行，因为包含 **2 个 Norm＋2 个 RoPE＋1 个 KV store**。本轮原型只融合前 4 行。

![原生链 Nsight Compute 界面，沿用此前截图](project_walkthrough_assets/04_native_ncu.png)

最值得留意的是 Small Grid：图里选中的 K Norm 只有 8 个 block，而 GPU 有 84 个 SM。它的工作量很小，无法靠这一次 launch 让整张卡忙起来。其他几个 kernel 也很短，链却分成了多个节点。

我不会从“Memory Throughput 很低”就推出“显存带宽打满了”；也不会把 Estimated Speedup 的 90% 多当成实现后的收益。NCU 的这些估计是诊断提示，实际性能还要独立测。

**我据此提出假设：把归一化后的值留在寄存器里继续做 RoPE，减少中间张量读写和 Graph 节点，可能比逐个小 kernel 调 block 更直接。**

按张量大小估算，Q16/K8、dim128、BF16 的 Norm 中间输出读＋写为每层每 token **12 KiB**，28 层是 **336 KiB/token**。这只是逻辑 global-memory 访问量，不能直接叫“节省了同等数量的 DRAM 流量”，因为缓存也可能命中。

原始报告：[native NCU](../results/profiling_20260918/native_qk_rope_cache_graph.ncu-rep)。该报告用 cold-cache kernel replay，里面的微秒数不能直接和无 profiler 的热态微基准相减。

## 5. 我怎样把融合假设写成 CUDA

我的第一版只做一件事：合并 Q/K Norm 和 RoPE，输出新的 dense Q/K；V、KV store、Attention 都保留原路径。

Host 侧 C++ 检查形状、dtype、device、stride 等条件，分配输出，取 PyTorch 当前 CUDA stream 发射 kernel。Device 侧，一个 warp 负责一个 head，warp 内处理 128 个元素并用 shuffle 交换归约结果；默认一个 block 放 4 个 warp，各 head 独立。整张 grid 覆盖这批 token 的 Q16＋K8 个 head。

这里不需要 head 之间共享数据，所以没有 shared memory 和 block 级 `__syncthreads()`；归约所需的数据交换在 warp 内完成。减少节点不保证 occupancy 更高，occupancy 也不是本项目最终优化目标。

有一个非常容易漏掉的细节：原来的两个 kernel 之间有一次 **BF16 落地舍入**。我虽然不再把这个中间结果写到 global memory，但仍在寄存器里显式做 BF16 舍入，再转成 FP32 继续旋转。否则就把“融合”和“改变数值计算路径”混到一起了。

![CUDA 实现中的数值边界](project_walkthrough_assets/06_kernel.png)

FP32 用在平方、归约、rsqrt 和旋转的中间计算中，输入和输出仍然是 BF16。这不是量化项目。原生 torch.compile 已经消除了 Python Norm 内部某个更早的 cast，所以比较对象必须是 **实际 compiled native**，不能只照 Python 表面顺序猜。

源码：[CUDA kernel](../src/qk_norm_rope/kernel.cu)、[加载入口](../src/qk_norm_rope/__init__.py)、[数值和适用范围说明](../src/qk_norm_rope/README.md)。默认 warps=4；1/4 warp 的探索性测量噪声较大，没有据此声称某个 block 配置更快。

## 6. 正确性检查确实抓到了问题

我先跑局部测试：真实首层输入、多个随机 seed、B1/2/4/8、非连续 stride、零/很小/很大值、position 0/255/256/4095、Graph replay、输入是否被改写和参数拒绝检查。

![局部正确性检查](project_walkthrough_assets/07_kernel_validation.png)

最开始确实出现过单元素偏差。检查生成代码后发现，native Triton 会按输入布局和编译特化选择不同的求和顺序。浮点加法不满足结合律，FMA 也会改变舍入次数；同一个公式并不自动代表逐位相等。我保留了失败日志，逐项对齐了归约、FMA 和 BF16 边界。

然后，一个更有价值的失败出现在真实模型：第一版局部测试已经通过，但 B1 的模型 logits 仍不完全一致，B4/B8 却一致。

![第一版的模型级失败记录](project_walkthrough_assets/07b_model_failure.png)

这让我不能直接拿局部 exact 就宣称项目正确。检查模型实际生成代码后确认：B1 的独立脚本选了 XBLOCK=8，每 block 处理多个 head；真实模型选了 XBLOCK=1、num_warps=2，每个 head 一个 block。两条求和路径并不相同。

我给实现增加了显式的 `native_graph` 归约模式，模型接入使用它；独立编译的单测保留自己的明确模式。这是按实际执行路径选择计算顺序，不是根据某个输入的参考答案选择结果。最终微基准也先初始化真实 native engine，再使用模型本身的 Norm/RoPE 模块，避免继续比较另一种编译特化。

![归约差异的定位和逐层复核](project_walkthrough_assets/07c_diagnosis.png)

这次补充验证检查了 5 种 workload、每种 17 个 Decode step、每 step 28 层，共 2,380 行实际模型中间结果；修复前有 8 行不一致，修复后为 0。另有 294 组捕获 fixture 的 Norm、RoPE 和 Graph 对拍全部通过。[原因和编译器证据](../results/full_project_20260919/kernel_reduction_context_diagnosis.json) 可以复查。

第一版的 [源码快照和失败结果](../results/full_project_20260919/attempt_1/) 完整保留，正式 A/B 没有使用这一版。

最终实现的边界说明和修正依据见 [kernel README](../src/qk_norm_rope/README.md) 及结果目录里的诊断记录。这里坚持 exact gate，没有为了得到加速结论而事后改成“误差差不多就行”。但 exact 也只覆盖固定软件栈和已测输入，不是跨 GPU、跨版本的数学等价证明。

## 7. 局部快了多少，节点是不是真的少了？

正确性过关后，我把同一组真实 QKV 输入分别交给 native 和 CUDA 原型。每个 CUDA Graph 包含 1,000 次链调用，用 CUDA Event 测总时间后折算；做 10 组成对测量并交替顺序。Profiler 的节点检查与计时分开。

![真实 fixture 的微基准](project_walkthrough_assets/08_micro.png)

9 个 fixture 的结果是 **4→1 个 GPU kernel**，Q/K 与 native exact，局部延迟下降 **{min(reductions):.1f}%～{max(reductions):.1f}%**。B2 是补充的局部验证点；正式模型 A/B 仍然是固定的原始 7 个 workload。

这里反复访问同一组小输入，缓存状态与完整模型不完全相同。它回答的是“这条融合链是否更快”，不是“整个模型会快同样的百分比”。

我也重新采了 native/fused 的 NCU 报告，确认 launch 和资源结构发生了变化：

![本轮 NCU 导出指标对照](project_walkthrough_assets/05_ncu_comparison.png)

原始证据：[micro.json](../results/full_project_20260919/micro.json)、[native NCU](../results/full_project_20260919/native_qk_rope.ncu-rep)、[fused NCU](../results/full_project_20260919/fused_qk_rope.ncu-rep)。NCU 仍使用冷缓存 replay，只作结构和资源诊断。

**到这里，证据支持“原型本身正确且局部有收益”。下一步才是看接进模型后是否成立。**

## 8. 接进 nano-vLLM，再检查整个模型

接入时，我在 engine 初始化和 CUDA Graph capture **之前**安装显式适配器。满足条件的 Decode 调用自写扩展；Prefill 和不支持的形状保留 native 路径。原始模型源码没有被直接改写，baseline 的源文件 hash 仍然能够复核。

![模型接入代码](project_walkthrough_assets/09_integration.png)

为了排除“代码写了，但 Graph 还在跑旧路径”，我抓了一个真实模型的 warmed Decode step。native 是 **{native_integration['counts']['total']}** 个 GPU kernel，fused 是 **{integration['counts']['total']}** 个；出现 **28** 个自写融合 kernel，独立 RoPE 从 56 个变成 0，KV store 仍为 28 个。整个 step 少了 **84=28×(4−1)** 个节点。

![真实模型 Graph 的节点计数](project_walkthrough_assets/09b_graph.png)

这里已经启用 CUDA Graph，因此不能把它说成每 step 少了 84 次 CPU CUDA launch 调用；准确的是减少了 **Graph 内的 GPU kernel 节点**。

数值验证使用 native 的 token 历史，保证每一步输入一致，再比较所有 step 的 logits。额外挑选早期、晚期和 KV 页边界，检查全 28 层新写入的 K/V。这样可以避免“前面采样走了另一条路，后面自然不同”混淆问题。

这里比较每个生成 step 的完整词表 logits；Prefill 比较的是每条请求最后一个输入位置的 logits，并不是所有 prompt 位置的 logits。KV 检查覆盖选定 step 新写入的槽位，不把它说成逐步扫描整个 cache。

![模型级正确性证据](project_walkthrough_assets/10_model_correctness.png)

最终 7 个配置、**{steps:,} 个 step** 的 logits 都 exact，选定位置的全层新写 KV 也 exact。正常采样是否一致则在正式 A/B 中另外检查，不能把 teacher forcing 下历史相同当成正常生成结果相同。

证据：[模型数值检查](../results/full_project_20260919/numerical_fused.json)、[native 模型 trace](../results/full_project_20260919/integration_native.json)、[fused 模型 trace](../results/full_project_20260919/integration_fused.json)。

## 9. 正式端到端 A/B：我不只挑一遍好看的数字

正式测试前，我已固定 **10 个独立进程对**：奇数对 native→fused，偶数对 fused→native；同一对 workload 顺序一致，跨对轮转。每个进程先预热全部配置，然后每种配置测两个固定请求。总共 **20 个进程、280 个测量请求**，不删慢样本。

启动前检查背景 GPU 使用率，同时保留运行前后的 GPU 状态。机器没有锁频，还是桌面 GPU，所以这些检查只能减轻干扰，不能证明完全没有干扰。正式计时不开 NCU/NSYS，也不混入编译和初始化时间。

每对的延迟下降率是 `(native − fused) / native`；先把一个进程里同配置的两次请求取均值，再得到 10 个配对下降率。下面报告它们的平均值和 pair 级 bootstrap 95% 区间；区间是每个配置的描述性估计，不是 7 个配置同时成立的保证。

![端到端成对结果](project_walkthrough_assets/11_ab.png)

| workload | native 平均 ms | fused 平均 ms | 配对延迟下降 | 95% 区间 | 吞吐增幅 |
|---|---:|---:|---:|---:|---:|
'''
    for r in ab['rows']:
        text+=f"| {r['case']} | {r['native_e2e_ms']:.2f} | {r['fused_e2e_ms']:.2f} | {r['e2e_reduction_pct']:.2f}% | [{r['ci95'][0]:.2f}, {r['ci95'][1]:.2f}]% | {r['throughput_gain_pct']:.2f}% |\n"
    positive=[r['case'] for r in ab['rows'] if r['ci95'][0]>0]
    uncertain=[r['case'] for r in ab['rows'] if r['ci95'][0]<=0<=r['ci95'][1]]
    negative=[r['case'] for r in ab['rows'] if r['ci95'][1]<0]
    text+='\n就这次数据而言：\n\n'
    text+=f"- 区间整体为正：{', '.join(positive) if positive else '没有配置'}。\n"
    text+=f"- 区间跨过 0、方向仍不确定：{', '.join(uncertain) if uncertain else '没有配置'}。\n"
    text+=f"- 区间整体为负：{', '.join(negative) if negative else '没有配置'}。\n"
    text+='''
我还把每一对结果放出来，负数就是这一对发生了退化，不隐藏它：

![各对结果和显存统计](project_walkthrough_assets/12_variation.png)

正式测量的所有正常采样输出 token 都与冻结 baseline 一致。显存只报告 PyTorch 的 allocated/reserved；模型权重和固定 KV cache 仍是主要容量，不能仅凭省掉中间张量就宣称显存大幅下降。若差异很小，就应该如实写很小。

__MEMORY_RESULT__

原始证据：[A/B 汇总和输入 hashes](../results/full_project_20260919/ab_summary.json)，同目录 `ab_01_native.json` 到 `ab_10_fused.json` 保存各请求耗时、每步耗时、输出 token 和 GPU 快照。

## 10. 我怎样解释收益、退化和适用范围

先把因果链说完整：原来每层四个小节点，融合后变成一个；保留中间舍入，减少 normalized Q/K 的逻辑写回再读取；局部正确性和微基准支持这一变化有效；真实模型 trace 证明替换确实发生；最后以独立端到端 A/B 判断整体收益。

局部提升很大，整体提升可能很小，这并不矛盾。Norm→RoPE 本来只占完整 Decode kernel 时间的约几个百分点，矩阵乘、Attention、LM head、采样和 CPU 调度仍然要跑。可以用“原占比 × 局部节省比例”理解大致量级，但 GPU 时间占比不能直接套成端到端预测，也不能把所有节点都当成相互独立的费用。

Prefill 本轮没有替换，所以它的变化不能算作融合算法收益。长输入、短输出的 workload 中，没优化的 Prefill 更重要，整体收益可能被稀释。早晚 Decode 的上下文长度也不同；其描述性数据保存在 A/B 汇总中，不能只拿最快的一小段代表整条请求。

__PHASE_RESULT__

如果某些配置的区间跨 0，我会写“当前测量还不能稳定区分”，而不是挑一次最快结果。即使区间为正，也只支持这台 RTX 5080、这个软件栈、Qwen3-0.6B 和这些 batch/长度条件；不外推到大模型、长上下文、高并发在线 serving。

实现也有边界：这是 BF16、完整 RoPE、Q16/K8、head_dim128 的推理原型，不支持 backward，不是通用 RMSNorm/RoPE 库。精确对齐依赖 native 编译特化；升级 PyTorch/Triton、换模型或换布局，需要重新跑正确性和性能门槛。新模型可能有其他热点，不能照搬这个优化优先级。

这次项目能够站得住的描述是：**我通过全链路 profiling 识别了高频小算子链，结合源码和 NCU 提出融合假设；手写 CUDA 并保留数值边界，接入真实 CUDA Graph 推理路径；用数值、局部和成对端到端三层证据验证结果，同时报告失败迭代、波动与适用条件。** 具体端到端百分比引用上表，不把它改写成所有场景都成立的结论。

## 11. 复查与复现入口

从项目根目录，在同一套 WSL Ubuntu-24.04 环境中运行。Python、模型和 CUDA 路径由 `scripts/profiling/environment.sh` 等入口固定。下面的准备脚本会创建一个全新的实验目录，复制源码、固定 workload、原生输出参考及两类输入 fixture，**不复制本轮测量结果，也不覆盖当前目录**。

```bash
# 1. 准备新的实验快照。目标目录必须尚不存在。
source scripts/profiling/environment.sh
"$PYTHON" scripts/full_project/create_reproduction.py /home/xietaibo/nano-vllm-fusion-replay

# 2. 在新目录完整重跑，任何数值或版本门槛失败都会停止。
cd /home/xietaibo/nano-vllm-fusion-replay
bash scripts/full_project/reproduce.sh
```

`reproduce.sh` 依次运行 `kernel_validation.py --label final`（生成两个 final 数值门槛）、全链路采集、prepare、micro/模型/trace 验证、NCU、10 对 A/B 和统计。准备脚本包含模型中间结果 fixture，避免只有首层输入、却缺失模型回归样本的情况。新目录的两个完整模型数值过程和所有 A/B 请求都会重新执行；复制的历史参考仅用来核对一致性。

本次已核验准备脚本的目录结构和 baseline 源文件 hashes；上述“完整重跑”入口没有再重复执行第二套 10 对 A/B。本文展示的是本轮原始目录里的实际实验。

| 想看什么 | 入口 |
|---|---|
| 原始 baseline 和固定 workloads | [baseline 目录](../results/baseline_20260918_170614/) |
| 多候选源码审查 | [discovery_source_audit.md](discovery_source_audit.md) |
| 手写 kernel | [kernel.cu](../src/qk_norm_rope/kernel.cu) |
| 模型适配开关 | [adapter.py](../scripts/full_project/adapter.py) |
| 数值、微基准、trace、A/B 驱动 | [study.py](../scripts/full_project/study.py) |
| 统计口径 | [analyze_ab.py](../scripts/full_project/analyze_ab.py) |
| 所有本轮结果及失败版本 | [full_project_20260919](../results/full_project_20260919/) |
| 截图的真实数据页和生成脚本 | [图片目录](project_walkthrough_assets/)、[evidence.py](../scripts/full_project/evidence.py) |

阅读或分享这份 Markdown 时，请把同目录的 `project_walkthrough_assets` 一起带上；只复制 MD 会丢失截图。
'''
    def interval(values):
        return f'{values[0]:,.2f}' if values[0]==values[1] else f'{values[0]:,.2f}～{values[1]:,.2f}'
    memory_text='本次实际显存统计：'
    for variant in ('native','fused'):
        m=ab['memory'][variant]
        memory_text+=f"**{variant}** peak allocated {interval(m['peak_allocated_MiB'])} MiB，稳态 allocated {interval(m['steady_allocated_MiB'])} MiB，reserved {interval(m['peak_reserved_MiB'])} MiB；"
    if ab['memory']['native']==ab['memory']['fused']:
        memory_text+='两种实现的这些统计范围相同，因此本次**没有测到这些显存容量指标的下降**。'
    memory_text+='减少中间张量的读写量，和降低整段推理的峰值显存容量，需要分别测量。'
    decode=[r['decode_reduction_pct'] for r in ab['rows'] if r['case'].endswith('_o256')]
    prefill=[r['prefill_reduction_pct'] for r in ab['rows']]
    phase_text=f'具体看这次对照，Prefill 的平均配对延迟下降率范围为 {min(prefill):.2f}%～{max(prefill):.2f}%，而 Decode 较多的六种请求，Decode 平均步延迟下降率范围为 {min(decode):.2f}%～{max(decode):.2f}%。'
    phase_text+='Prefill 原路径的波动提醒我：不能把端到端每一点变化都精确归因于融合；它也可能包含测量干扰或执行上下文影响，本实验没有对 Prefill 变化做单独归因消融。'
    text=text.replace('__MEMORY_RESULT__',memory_text).replace('__PHASE_RESULT__',phase_text)
    dest=ROOT/'docs/PROJECT_WALKTHROUGH.md'
    dest.write_text(text,encoding='utf-8')
    print(dest)

if __name__=='__main__':main()
