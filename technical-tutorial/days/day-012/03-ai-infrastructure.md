# AI Infrastructure 第 12 章：并行策略不是名字选择，而是张量轴选择

## 1. 从一个真实任务开始

上一章讲了编译和算子融合，它优化的是单个 rank 内部的执行效率。现在假设单卡 kernel 已经足够高效，模型、batch 或序列仍继续变大。FSDP 能切模型状态，checkpoint 能切 activation 峰值，但某些单层矩阵乘法、attention 或渲染批次仍然大到单个 GPU 放不下或算不快。

真实任务是训练一个大规模 CTGS/NeRF hybrid 或 Transformer-based reconstruction prior。你有 64 张 GPU。若只用 data parallel，每张卡仍要运行完整大层；若只用 FSDP，参数状态能分片，但某个层的计算仍可能需要完整 activation；若只用 pipeline，层间有 bubble；若只用 tensor parallel，每层通信密集。问题不是“哪个并行技术先进”，而是模型的哪个轴需要被切。

今天的任务是把 data、tensor、pipeline、sequence/context parallel 放到同一张知识树里。每种并行策略移动的张量不同，通信模式不同，适合的瓶颈也不同。

## 2. 最直接的办法，以及它为什么不够

最直接的方法是增加 data-parallel ranks。它提高吞吐，但每个 rank 仍保存或临时 all-gather 同样的大层，单样本内的 compute 不会变小。遇到单层显存或单层算力瓶颈时，data parallel 不够。

第二个方法是把模型按层切成 pipeline stages。它减少每卡层数，却引入 microbatch 调度和 pipeline bubble。若模型层数不够多，或某些 stage 特别慢，GPU 利用率会不均衡。

第三个方法是把所有矩阵都 tensor parallel。它能切单层 compute，但每层之间要 all-reduce、all-gather 或 reduce-scatter。跨节点网络慢时，tensor parallel 会被通信拖住。

因此需要先做瓶颈诊断：是模型状态太大、单层参数太大、activation 太大、sequence/context 太长，还是层数太多？并行策略应对准具体轴。

## 3. 关键想法是怎样被引出来的

把训练张量看成多维对象。Batch 轴可以给 data parallel；参数和 optimizer state 可以给 FSDP/ZeRO；矩阵乘法的 hidden dimension 可以给 tensor parallel；layer 轴可以给 pipeline parallel；sequence 或 ray/context 轴可以给 sequence/context parallel。

Tensor parallel 的核心是把一个层内部的矩阵乘法切开。PyTorch 的 Tensor Parallel 教程展示了如何在大规模 Transformer 中结合 TP 和 FSDP；NVIDIA Megatron 文档也把 TP 定义为在 GPU 间切分单层参数张量。[PyTorch Tensor Parallel Tutorial](https://docs.pytorch.org/tutorials/intermediate/TP_tutorial.html) [Megatron Core Parallelism Guide](https://docs.nvidia.com/megatron-core/developer-guide/0.16.0/user-guide/parallelism-guide.html)

Pipeline parallel 则把 forward pass 表示为层序列，不同 GPU 持有不同层。DeepSpeed pipeline 文档强调它依赖层间简单接口，并用 microbatches 填充流水线。[DeepSpeed Pipeline Parallelism](https://deepspeed.readthedocs.io/en/latest/pipeline.html)

关键想法是：这些策略不是互斥标签，而是可组合的并行维度。组合后形成 2D、3D 或更多维并行，但每多一个维度，通信顺序和调试难度也上升。

## 4. 一步一步建立正式模型

先看 data parallel。全局 batch 大小为

\[
B_{\rm global}=P_{\rm dp}B_{\rm local}.
\]

每个 data-parallel rank 计算 local gradient，再 all-reduce 或 reduce-scatter。它切 batch，不切单个样本内部计算。

Tensor parallel 以线性层为例：

\[
Y=XW.
\]

若按列切 \(W\)：

\[
W=[W_1,W_2],
\]

则

\[
Y=[XW_1,\ XW_2].
\]

每个 GPU 产生输出的一部分，后续若需要完整 \(Y\)，要 all-gather。若按行切 \(W\)，同时切 \(X\)：

\[
X=[X_1,X_2],
\qquad
W=
\begin{bmatrix}
W_1\\
W_2
\end{bmatrix},
\]

则

\[
Y=X_1W_1+X_2W_2.
\]

此时需要 all-reduce 求和。Megatron 风格会把 column-parallel 和 row-parallel 层配对，减少不必要同步。

Pipeline parallel 把 \(L\) 层切成 \(P_{\rm pp}\) 段。若 microbatch 数为 \(M\)，简单 GPipe 式流水线 bubble 比例近似为

\[
\frac{P_{\rm pp}-1}{M+P_{\rm pp}-1}.
\]

\(M\) 越大，bubble 越小，但 activation 和调度开销也会变大。

Sequence/context parallel 切的是长序列或上下文维度。对 attention，序列长度 \(S\) 带来

\[
O(S^2)
\]

的 attention score。把 \(S\) 切到多个 GPU 可减少每卡 activation，但需要跨 shard 汇总 key/value 或 attention 结果。它适合上下文很长、hidden dimension 已经不再是唯一瓶颈的场景。

## 5. 跟着一个完整例子走到底

假设有 64 张 GPU。选择

\[
P_{\rm dp}=4,\quad P_{\rm tp}=4,\quad P_{\rm pp}=4.
\]

总并行度为

\[
4\times4\times4=64.
\]

这意味着 4 个 data-parallel replicas；每个 replica 内有 4 个 pipeline stages；每个 stage 内的单层再用 4-way tensor parallel。

若全局 batch 为

\[
B_{\rm global}=128,
\]

则每个 data-parallel replica 处理

\[
32
\]

个 samples。若 pipeline microbatch 数为

\[
M=8,
\]

每个 microbatch 为

\[
4
\]

个 samples。Pipeline bubble 约为

\[
\frac{4-1}{8+4-1}
=
\frac{3}{11}
\approx27.3\%.
\]

若把 microbatch 增到 16，bubble 变成

\[
\frac{3}{19}\approx15.8\%.
\]

但 microbatch 更小或更多会影响 kernel efficiency、activation memory 和 optimizer step 频率。

现在看一个线性层。hidden size 为 8192，batch-token 合并维度为 4096。完整矩阵乘法为

\[
4096\times8192
\]

乘

\[
8192\times8192.
\]

4-way tensor parallel 后，每个 GPU 只持有一部分 \(W\)，单卡矩阵宽度约为 2048。计算和参数显存下降，但层边界需要通信。若这些 4 张 GPU 在同一 NVLink 节点内，通信可接受；若跨节点，成本可能压过收益。

这个例子说明并行布局首先是拓扑布局。TP 通常放在最快互联内，PP 跨节点也可行，DP 最适合跨更多节点扩展。

## 6. 回到真实系统：程序实际上怎样工作

真实训练平台会把所有 GPU 组织成多个 process groups：data-parallel group、tensor-parallel group、pipeline group、sequence/context group。每个 collective 必须在正确 group 上以一致顺序执行。日志里应记录 rank 到并行坐标的映射，例如 \((dp,tp,pp)\)。

对 CTGS，parallelism 不一定完全模仿 Transformer。Rays 或 views 可切 batch；Gaussian 参数可按空间 shard 或 FSDP shard；MLP hidden dimension 可 tensor parallel；多阶段 pipeline 可把 projection、network、compositing 和 loss 分到不同阶段，但只有当 stage 计算足够平衡才值得。

选择策略时，先用 profiler 量化：model state memory、activation memory、largest layer compute、communication bandwidth、pipeline stage imbalance、kernel efficiency。然后从最少维度开始。能用 FSDP+DP 解决，就不要过早引入 TP/PP；单层太大时再加 TP；层数多且显存分布合适时再加 PP。

Checkpoint、compile 和 parallelism 也会相互影响。Pipeline 需要保存或重算跨 stage activations；TP 会改变 kernel shape 和编译 specialization；FSDP 会改变 all-gather 时机。组合策略必须整体测试。

## 7. 容易走错的岔路

第一个误区是把 3D parallelism 当成固定配方。不同模型和硬件拓扑需要不同切法。错误的 TP 跨节点会非常慢。

第二个误区是只看每卡显存。某种切分可能让显存够了，但通信进入 critical path，step time 变差。

第三个误区是 pipeline stage 不均衡。最慢 stage 决定吞吐，其他 stage 等待。切层时要按实际 compute，而不是按层数平均。

第四个误区是忘记 optimizer 和 checkpoint 格式。多维并行下，state dict 可能同时按 DP、TP、PP、FSDP 分片；保存和恢复必须知道完整坐标。

最后，动态图结构会让并行坐标复杂化。CTGS 的 densification/pruning 若改变参数形状，必须同步所有相关 parallel groups。

## 8. 本章落点、验证与下一章

本章把并行策略解释为张量轴选择：DP 切 batch，FSDP/ZeRO 切状态，TP 切层内矩阵维度，PP 切层序列，sequence/context parallel 切长上下文。组合策略必须同时考虑显存、计算、通信和硬件拓扑。

在 Infrastructure 与 CTGS 项目中，本章对应 process group 设计、rank mapping、FSDP+TP 组合、pipeline microbatch、ray/view batch 切分、动态 Gaussian 参数同步和 checkpoint shard schema。

本章的 90 分钟验证是设计一个 64 GPU 布局：取 \(P_{\rm dp}=4,P_{\rm tp}=4,P_{\rm pp}=4\)，计算 global batch 128 下每个 DP replica 和 microbatch 的大小，并比较 \(M=8\) 与 \(M=16\) 的 pipeline bubble。预期是 microbatch 增加会降低 bubble，但可能增加调度和 activation 压力。

下一章将进入容错、检查点与可观测性。并行维度越多，失败点越多；训练平台必须能保存、恢复、定位慢 rank、检测通信 hang，并把系统状态变成可诊断数据。
