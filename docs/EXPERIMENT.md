# nano-vLLM：从热点发现到融合验证

**环境与 workload：** RTX 5080、Qwen3-0.6B、BF16、CUDA Graph；B=1/4/8 × 输入64/256 × 输出256，另加 B1/输入2048/输出32；KV=64页×256 token。

本文记录 2026-09-21 的测量；图片为当时工具与运行结果的原生截图，数据见 [冻结结果](../results/published_20260921)。

## 1．Nsight Systems分析热点算子

Nsight Systems：累计耗时前三项是两类矩阵计算和一类 Attention；另有累计占比较小、调用频繁的归一化 kernel。

![image-20260921181349154](images/image-20260921181349154.png)

再按调用次数排序：归一化 kernel 调用频繁，先列为候选；Q/K 归属和融合条件仍需时间线与源码确认。

![image-20260921181431507](images/image-20260921181431507.png)


Decode 时间线与源码确认：Q Norm、K Norm、Q RoPE、K RoPE 分开连续执行，可尝试融合。

![image-20260921183540731](images/image-20260921183540731.png)


|候选|判断|
|---|---|
|GEMM、Attention|分别占 Decode GPU kernel 累计时间约66.40%～76.08%、13.69%～23.32%；实现复杂度较高，本轮暂缓，不代表没有优化空间|
|Q/K Norm→RoPE|占 Decode GPU kernel 累计时间3.12%～3.72%；每步112个节点，存在可省掉的中间读写|
|Add+Norm、SiLU×up、采样|部分计算已由 torch.compile 融合；需分别审查剩余边界，不能认定整个采样链只有一个 kernel|

## 2．NCU分析

Roofline：点远低于上界；计算强度低，还不能据此断言 DRAM 带宽打满。

![image-20260921174852958](images/image-20260921174852958.png)


Speed of Light：四个节点 Compute仅0.11%～0.25%、DRAM仅1.04%～4.52%，先关注小任务的开销。

![image-20260921174903132](images/image-20260921174903132.png)


Compute Workload Analysis：Q Norm 的 Executed IPC Active约0.08，没有计算管线饱和的证据。

![image-20260921174941710](images/image-20260921174941710.png)


Memory Workload Analysis：结合源码，Norm中间结果每层每token逻辑上写6 KiB再读6 KiB；融合可省这段传递。

![image-20260921175005623](images/image-20260921175005623.png)


Scheduler Statistics：Q Norm 的 Eligible约0.04、No Eligible约95.71%，可发射工作很少。

![image-20260921175046966](images/image-20260921175046966.png)


Warp State Statistics：存在 Long Scoreboard 访存依赖等待；需结合指令和数据依赖判断哪些读取可省掉，不能单凭 stall 决定融合。

![image-20260921175100817](images/image-20260921175100817.png)


Source / SASS：实际指令包含SHFL.BFLY归约、MUFU.RSQ和BF16转换，融合时要保留对应数值语义。

![image-20260921175150662](images/image-20260921175150662.png)


## 3．融合算子

优化方案：Q16/K8/D128特化，每warp一个head、每block四个warp，在寄存器里接着算RoPE，把四个节点合成一个。

Kernel Specialization负责固定形状和执行布局；归约顺序、FMA与BF16舍入需要另外对齐。

![image-20260921183946242](images/image-20260921183946242.png)


|版本|本次改动|严格一致的logits step|
|---|---|---|
|V1|Norm后继续用FP32|7/1568|
|V2|恢复Norm→RoPE间的BF16舍入|1056/1568|
|V3|对齐batch=1原生Graph的归约顺序|1568/1568|

实测发现：只保留BF16存储和FP32计算仍不够；恢复舍入边界并对齐归约后，logits和选定KV位置才全部通过。

![image-20260921184121401](images/image-20260921184121401.png)


BF16输入→FP32 Norm→寄存器内BF16舍入→FP32 RoPE→BF16输出

## 4．接入模型再测收益

真实Decode接入验证：每step的GPU节点 **405→321**，28个融合节点生效；Prefill和28次KV cache写入保持原实现。

9组局部微基准数值对齐，Norm＋RoPE 整段延迟下降67.24%～73.08%；不是整模型的加速幅度。

![image-20260921184312895](images/image-20260921184312895.png)


融合后Roofline仍远低于上界；采用独立计时验证收益，不用NCU的3.328 μs冷重放时间计算加速比。

![image-20260921184932098](images/image-20260921184932098.png)


端到端：固定10组配对、280次请求、全部保留

![image-20260921185015602](images/image-20260921185015602.png)


|B / 输入 / 输出|原生ms|融合ms|配对延迟下降|95%区间|
|---|---:|---:|---:|---|
|1 / 64 / 256|833.54|830.80|0.31%|[-0.92, 1.49]%|
|1 / 256 / 256|856.03|838.92|1.90%|[-0.79, 4.14]%|
|4 / 64 / 256|903.85|876.43|2.95%|[0.46, 5.16]%|
|4 / 256 / 256|951.75|915.67|3.76%|[1.96, 5.94]%|
|8 / 64 / 256|938.97|906.18|3.47%|[2.42, 4.48]%|
|8 / 256 / 256|1000.73|982.27|1.83%|[0.83, 2.84]%|
|1 / 2048 / 32|145.46|142.33|2.09%|[0.54, 3.91]%|

## 5．融合后 NCU 原图复核

Speed of Light：融合后Compute约0.26%、DRAM约3.03%，仍未接近整卡上限。

![image-20260921185251509](images/image-20260921185251509.png)


Compute Workload Analysis：复核融合后的指令和管线压力，不能把四个原节点与一个融合节点的吞吐比当成端到端加速比。

![image-20260921185238948](images/image-20260921185238948.png)


Memory Workload Analysis：中间全局数组被消除；逻辑字节节省不等于同量DRAM流量节省。

![image-20260921185222818](images/image-20260921185222818.png)


Scheduler Statistics：融合后Eligible约0.05，仍是小任务，收益不等于GPU已经满载。

![image-20260921185208074](images/image-20260921185208074.png)


Warp State Statistics：融合后仍有依赖等待；保留这一限制，不以所有 stall 均下降作为成功条件。

![image-20260921185149330](images/image-20260921185149330.png)


Source / SASS：源码保留BF16舍入

![image-20260921185125015](images/image-20260921185125015.png)
