# AI Infrastructure 第 10 章：activation checkpointing 为什么是在显存和计算之间重新定价

## 1. 从一个真实任务开始

上一章用 FSDP/ZeRO 切掉了 DDP 中重复保存的模型状态。状态分片后，训练仍可能 OOM，因为 forward 中产生的 activations 还要留到 backward 使用。对 Transformer、NeRF、CTGS ray marching 或大分辨率 differentiable rendering，activations 可能比参数更大。

今天的真实任务是训练一个 CTGS 模型：每个 batch 有大量 rays，每条 ray 上采样多个点，MLP 和 Gaussian compositing 会生成许多中间 tensor。参数已经通过 FSDP 分片，但 peak memory 仍在 forward 末尾达到最高，因为 autograd 保存了 backward 需要的中间结果。此时继续切参数没有用，瓶颈变成 activation residency。

activation checkpointing 的目标不是减少数学计算量，而是减少 forward 期间保存的中间 tensor。它把一部分 forward 结果丢掉，等 backward 真正需要时重新执行相应 forward 片段。PyTorch 文档也把 checkpointing 描述为通过重跑 forward segment 来用计算换显存。[torch.utils.checkpoint](https://docs.pytorch.org/docs/stable/checkpoint.html)

## 2. 最直接的办法，以及它为什么不够

最直接的办法是让 autograd 保存所有中间结果。它最快，因为 backward 直接读取 forward 保存的 activations。但深模型中，保存量随层数、batch size、sequence length、ray samples 或图像分辨率增长。显存爆掉时，速度没有意义。

第二个办法是减 batch size。它能线性减少一部分 activation，但也改变吞吐、梯度噪声和硬件利用率。对 distributed training，batch 太小会让每步通信和 kernel launch overhead 占比变大。

第三个办法是手写 `detach` 或禁用梯度。它能省显存，但也会截断梯度，改变训练目标。checkpointing 要求 backward 数学结果仍等价于原模型，只是中间值来源从“保存”变成“重算”。

因此关键问题是：哪些中间值值得保存，哪些可以丢掉并重算；丢掉后 backward 需要多付多少计算；这个交换是否降低了总体 step time 或至少让训练可运行。

## 3. 关键想法是怎样被引出来的

反向传播本质上需要每个算子的局部输入或输出。例如

\[
y=f(x),
\]

若 loss 为 \(L\)，backward 要计算

\[
\frac{\partial L}{\partial x}
=
\frac{\partial L}{\partial y}
\frac{\partial f}{\partial x}.
\]

\(\partial f/\partial x\) 往往需要 forward 时的 \(x\) 或 \(y\)。普通 autograd 选择保存它们。checkpointing 的观察是：如果一个连续 segment 的输入仍被保存，那么 backward 时可以重新执行该 segment 的 forward，恢复内部 activations，再正常计算梯度。

因此 checkpoint 不是“保存所有层”，而是保存 segment 边界。边界之间的内部状态在 forward 后释放。Backward 到达该 segment 时，先从边界输入重算内部 forward，再继续反向。PyTorch 2025 的 activation checkpointing 文章还讨论了 selective checkpointing 和 memory budget API，核心仍是围绕保存张量和重计算张量之间的预算选择。[Current and New Activation Checkpointing Techniques in PyTorch](https://pytorch.org/blog/activation-checkpointing-techniques/)

## 4. 一步一步建立正式模型

设网络由 \(L\) 个相同层组成，每层 activation 大小为 \(A\)，计算时间为 \(C\)。普通训练需要在 forward 后保存近似

\[
M_{\rm act}^{\rm full}
\approx
L A
\]

的 activations，forward 计算为

\[
T_f=LC.
\]

若把 \(L\) 层分成 \(K\) 个 segments，每个 segment 长度为

\[
s=\frac{L}{K}.
\]

checkpoint 只保存 segment 边界，activation memory 近似为

\[
M_{\rm act}^{\rm ckpt}
\approx
K A
+sA.
\]

第一项是边界 activations，第二项是 backward 重算某个 segment 时的临时内部 activations。这个粗略公式忽略不同层大小，却能说明选择：segment 太短，边界多；segment 太长，重算峰值大。

计算时间会增加。普通 forward 运行一次；checkpoint 的 forward 也运行一次，但 backward 时每个 checkpointed segment 还要额外 forward 一次。因此额外计算约为

\[
T_{\rm extra}
\approx
LC,
\]

如果所有层都 checkpoint，总训练 step 的 forward-like compute 接近翻倍。实际增加小于或大于这个估计，取决于 backward 本身成本、kernel fusion、通信 overlap 和是否只 checkpoint 部分层。

有随机操作时，还要处理 RNG state。若 checkpoint segment 内有 dropout，重算 forward 必须使用和原 forward 相同的随机 mask，否则梯度对应的不是同一个计算图。框架通常保存和恢复 RNG state，但这也有开销。

## 5. 跟着一个完整例子走到底

考虑 12 层网络，每层 activation 为 200 MB。普通保存需要

\[
12\times200=2400\ {\rm MB}.
\]

若每层 forward 需要 2 ms，forward 时间为

\[
12\times2=24\ {\rm ms}.
\]

现在分成 \(K=4\) 个 segments，每段 \(s=3\) 层。保存四个边界 activation 约为

\[
4\times200=800\ {\rm MB}.
\]

Backward 重算某段时，临时内部最多约 3 层：

\[
3\times200=600\ {\rm MB}.
\]

peak activation 粗略估计为

\[
800+600=1400\ {\rm MB},
\]

比 2400 MB 少 1000 MB。代价是 backward 时四个 segments 各重算 3 层，总额外 forward 仍是 12 层，即约 24 ms。

若改成 \(K=2\)，边界只有 400 MB，但每段 6 层，重算临时 1200 MB，peak 约 1600 MB；若 \(K=12\)，边界 2400 MB，几乎没有省。这个例子说明 checkpoint 粒度不是越细越好，而是在边界保存和段内重算峰值之间找平衡。

对 CTGS ray marching，segment 可以不是神经网络层，而是 ray samples 的 chunk。保存所有 sample 的 density、color、alpha 和 transmittance 会很贵；可以保存 chunk 边界 transmittance，在 backward 时重算 chunk 内部 compositing。数学上仍要保证重算的 sampling order、随机扰动和 occupancy mask 与 forward 一致。

## 6. 回到真实系统：程序实际上怎样工作

工程实现 checkpointing 前，应先用 profiler 确认 peak memory 的主要来源。如果 activations 不是主因，checkpoint 只会让训练变慢。若 activations 主导，再决定 checkpoint 哪些模块：大 activation、小参数、forward 计算相对便宜、且没有复杂副作用的模块最适合。

PyTorch 中常见选择是对 Transformer block、MLP block 或 rendering chunk 使用 `torch.utils.checkpoint.checkpoint`。新的非 reentrant checkpoint 通常更易与 autograd 功能组合，但具体行为需要按文档和版本确认。含有 dropout、随机采样、global counters、缓存写入或自定义 CUDA op 的模块要格外小心，因为重算 forward 不应改变模型状态。

与 FSDP 组合时，checkpoint 会改变参数 all-gather 的时机。Backward 重算 segment 需要再次访问参数，如果 FSDP 已经 reshard，就可能触发额外 all-gather。为了避免通信放大，需要配合 activation checkpoint policy、prefetch 和 wrap 粒度一起调。

对 CTGS，最实用的 checkpointing 往往是 chunked rendering。把 rays 或 samples 分块，保存必要边界量，重算局部 density/color。需要记录 occupancy grid、ray sorting、random jitter 和 densification 状态的版本号，确保 backward 重算看到的是 forward 同一个结构。

## 7. 容易走错的岔路

第一个误区是认为 checkpointing 总能提速。它通常降低显存、增加计算；只有当省下显存允许更大 batch 或更好并行时，端到端吞吐才可能变高。

第二个误区是 checkpoint 含副作用函数。重算 forward 如果再次更新统计量、缓存、随机状态或全局计数器，训练就不再等价。

第三个误区是只看 allocated memory，不看 peak memory。checkpoint 主要降低 peak activation；如果 allocator fragmentation 或 persistent buffers 占主导，效果会被掩盖。

第四个误区是把所有模块无脑 checkpoint。很小的 layer 省不了多少显存，却增加调度和重算开销。应按 activation size 与 compute cost 做选择。

最后，分布式训练中 checkpoint 可能改变通信重叠。重算推迟了梯度产生时间，可能让 DDP bucket 或 FSDP reduce-scatter overlap 变差。必须用 timeline 验证，而不是只看 OOM 是否消失。

## 8. 本章落点、验证与下一章

本章把 activation memory 从“必须保存”改写成“可保存也可重算”的调度问题。checkpoint 保存 segment 边界，释放内部 activations，在 backward 时重跑 forward segment，以额外计算换显存。

在 Infrastructure 和 CTGS 项目中，本章对应 activation profiler、checkpoint policy、FSDP 组合策略、chunked rendering、随机状态一致性和 timeline 复核。它不是模型并行的替代，而是显存预算的一项局部交易。

本章的 60 到 90 分钟验证是构造 12 层 MLP 或 Transformer block，测量无 checkpoint、4 段 checkpoint、2 段 checkpoint 的 peak memory 和 step time。预期是 checkpoint 显著降低 activation peak，但 step time 增加；过细或过粗的 segment 都不是最优。

下一章将进入编译栈与算子融合。checkpointing 改变了哪些中间值保存，而编译器进一步决定哪些算子被融合、哪些临时 tensor 根本不落显存，以及动态 shape 如何限制这些优化。
