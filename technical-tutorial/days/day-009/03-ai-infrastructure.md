# AI Infrastructure 第 9 章：当 DDP 仍然 OOM 时，模型状态到底应该怎么切

## 1. 从一个真实任务开始

上一章用 DDP 解决了多卡训练中“每张卡各自更新导致模型分叉”的问题。DDP 的基本假设是每张 GPU 都保存一份完整模型、完整梯度和完整 optimizer state，只把不同样本分给不同 rank。这个假设在中小模型上很舒服，但 CTGS 或大规模 neural field 一旦包含大量 Gaussians、MLP、grid features、per-Gaussian optimizer state 和高分辨率训练 batch，单卡显存会先被模型状态吃掉。

今天的真实任务是：四张 24 GB GPU，模型前向反向的临时 activation 还能通过 batch size 调小勉强塞下，但 Adam 的 parameters、gradients、moments 和 master weights 加起来已经超过单卡显存。继续用 DDP 没有意义，因为 DDP 增加 GPU 数量只增加吞吐，不减少每张卡持有的模型状态。

需要的系统问题变成：哪些状态必须在计算某一层时完整出现，哪些状态只在 optimizer step 时需要，哪些状态可以在 rank 之间分片保存，并在需要时通过 collective 临时重建。

## 2. 最直接的办法，以及它为什么不够

最直接的办法是继续减 batch size。若 OOM 来自 activations，减 batch size 或 gradient accumulation 有效；但若 OOM 来自 model states，batch size 降到 1 仍然放不下。此时问题不在样本数，而在每张卡复制了相同的参数、梯度和 optimizer state。

第二个直接办法是把一部分 tensor 放到 CPU。CPU offload 能缓解显存，但 PCIe 或 NVLink 传输会进入 critical path。如果每层 forward 前都从 CPU 拉参数，GPU 可能长期等数据。offload 是内存层次策略，不是并行策略本身。

第三个直接办法是手写 model parallel，把模型切成几段放到不同 GPU。它能减少每卡参数，但会改变模型代码、增加 activation 传输，并且对不规则 Gaussian 参数集合很麻烦。我们需要先弄清楚：在保持数据并行训练语义的前提下，能不能只消除 DDP 的冗余状态。

## 3. 关键想法是怎样被引出来的

DDP 的冗余来自一个事实：每个 rank 都保存同一份模型状态。以 Adam 为例，一个参数不仅有低精度训练权重，还有梯度、FP32 master weight、一阶动量和二阶动量。optimizer state 只在更新时使用，梯度只在 backward 后到更新前使用，完整参数只在某个 module 的 forward/backward 计算窗口内使用。它们的生命周期并不相同。

ZeRO 和 FSDP 的关键想法是按生命周期切分冗余。ZeRO Stage 1 先切 optimizer states；Stage 2 进一步切 gradients；Stage 3 或 FSDP full shard 连 parameters 也切。计算某一层需要完整参数时，rank 之间 all-gather；反向得到梯度后，不再 all-reduce 得到每张卡完整梯度，而是 reduce-scatter，让每张卡只保留自己负责的梯度 shard。

这仍然是数据并行，因为每个 rank 处理不同样本，数学上仍在求全局 batch 的平均梯度。不同的是，模型状态不再完整复制。ZeRO 原论文把这种目标表述为消除 data-parallel training 中的内存冗余，同时尽量保持通信量和计算粒度。[ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054) PyTorch FSDP 文档也把 FSDP 描述为跨 data-parallel workers 分片 module parameters 的 wrapper，并提供不同 sharding strategy 与 CPU offload 选项。[PyTorch FullyShardedDataParallel](https://docs.pytorch.org/docs/2.12/fsdp.html)

## 4. 一步一步建立正式模型

设模型有 \(N\) 个参数，使用 mixed precision Adam。为了看清数量级，按每个参数估算：

\[
\text{fp16 parameter}=2\ {\rm bytes},
\]

\[
\text{fp16 gradient}=2\ {\rm bytes},
\]

\[
\text{fp32 master weight}=4\ {\rm bytes},
\]

\[
\text{Adam first and second moments}=8\ {\rm bytes}.
\]

DDP 每个 rank 复制全部状态，因此每卡模型状态约为

\[
M_{\rm DDP}
=
N(2+2+4+8)
=16N\ {\rm bytes}.
\]

若有 \(P\) 张 GPU，ZeRO-1 只把 optimizer states 分片，参数和梯度仍复制：

\[
M_{\rm Z1}
=
N(2+2)+\frac{N(4+8)}{P}.
\]

ZeRO-2 再把 gradients 分片：

\[
M_{\rm Z2}
=
2N+\frac{2N+12N}{P}.
\]

ZeRO-3 或 FSDP full shard 让 parameters、gradients 和 optimizer states 都分片。忽略某一层 all-gather 的峰值临时缓冲，稳态模型状态约为

\[
M_{\rm Z3}
=
\frac{16N}{P}.
\]

代价是通信形式改变。DDP backward 对完整 gradient 做 all-reduce；FSDP 在 forward 前对当前 module 参数做 all-gather，backward 后对梯度做 reduce-scatter。若某层参数 shard 为 \(W_\ell/P\)，forward 需要临时形成

\[
W_\ell
\]

的完整参数，计算后释放；backward 也需要在合适时机重建或保留参数，以便计算输入梯度和参数梯度。

因此 FSDP 的设计变量包括 wrap 粒度、prefetch、reshard timing、mixed precision、activation checkpointing 和 CPU/NVMe offload。粗粒度 wrap 通信少但峰值显存高；细粒度 wrap 峰值低但 collective 变多。

## 5. 跟着一个完整例子走到底

考虑一个 \(N=10^9\) 参数的模型，四张 GPU，即

\[
P=4.
\]

DDP 每卡模型状态为

\[
16N=16\ {\rm GB}.
\]

这还没有算 activations、temporary buffers、fragmentation 和 CUDA workspace。若每卡 24 GB，看似还有 8 GB，但真实训练很容易 OOM。

ZeRO-1 每卡为

\[
N(2+2)+\frac{N(4+8)}{4}
=4\ {\rm GB}+3\ {\rm GB}
=7\ {\rm GB}.
\]

ZeRO-2 每卡为

\[
2N+\frac{14N}{4}
=2\ {\rm GB}+3.5\ {\rm GB}
=5.5\ {\rm GB}.
\]

ZeRO-3 稳态为

\[
\frac{16N}{4}=4\ {\rm GB}.
\]

数字说明了为什么 Stage 3 很有吸引力。但还要看一个 module 的 all-gather 峰值。若某个大层参数为 1 GB，FSDP 计算它时每张卡需要临时 all-gather 额外约

\[
1-\frac14=0.75\ {\rm GB}
\]

的参数。如果 wrap 太粗，一次 all-gather 可能是多个大层，峰值又回来了。若 wrap 太细，通信 latency 会变多。

现在比较 DDP 和 FSDP 的一步训练。DDP 的状态流是：所有 rank 从完整参数 forward，backward 得到完整 gradients，all-reduce 后每个 rank 用完整 optimizer state 更新完整参数。FSDP 的状态流是：rank 持有参数 shard；进入某个 module 前 all-gather 完整参数；计算后释放非本 rank shard；backward 后 reduce-scatter 得到梯度 shard；optimizer 只更新本 rank 负责的参数和 Adam states。最终所有 shards 合起来仍是同一个全局模型。

## 6. 回到真实系统：程序实际上怎样工作

实际使用 FSDP 时，第一步不是盲目打开 full shard，而是做显存剖面。把显存分成 parameters、gradients、optimizer states、activations、temporary buffers 和 allocator fragmentation。若 activations 最大，应优先 checkpointing 或减 microbatch；若 optimizer states 最大，ZeRO-1/2 就可能足够；若 parameters 本身也大，才需要 full shard。

PyTorch FSDP 通常围绕 module wrap。每个 wrapped module 有 flatten 或原始参数策略、mixed precision policy、sharding strategy、prefetch 策略和 state dict 策略。训练循环看似仍是 forward、loss、backward、optimizer step，但内部插入了 all-gather、reduce-scatter 和 reshard。checkpoint 也不再只是保存一个完整 `state_dict`，而要选择 full、sharded 或 local state dict，并保证恢复时 optimizer state 与 parameter shard 对齐。

对 CTGS，一个额外难点是参数集合会变。densification 新增 Gaussian，pruning 删除 Gaussian，排序和 compaction 改变 parameter layout。FSDP 假设 wrapped module 的参数结构在多数 step 内稳定；频繁结构编辑会破坏 sharding metadata 和 optimizer state 对齐。因此应把结构编辑做成同步事务：先在所有 ranks 上基于全局统计决定新增/删除，再暂停或重建相关 shard 和 optimizer state，之后继续训练。

并行策略选择也要分清轴。Data parallel 切 batch；FSDP/ZeRO 切模型状态但仍沿 batch 并行；tensor parallel 切矩阵乘法内部维度；pipeline parallel 切层；sequence/context parallel 切序列长度。选哪一种，不看名字新旧，而看瓶颈究竟是状态显存、activation 显存、单层计算太大、pipeline 空泡，还是通信拓扑。

## 7. 容易走错的岔路

第一个误区是把 FSDP 当作“无代价显存扩展”。它节省稳态显存，但引入 all-gather 和 reduce-scatter。网络慢、wrap 不当或参数太碎时，step time 可能显著变差。

第二个误区是忽略 activation。FSDP 切的是模型状态；长射线 batch、体渲染采样或大分辨率 feature map 的 activations 仍可能占主导。此时需要 activation checkpointing、chunked rendering 或 recomputation。

第三个误区是 checkpoint 保存错格式。把每个 rank 的 local shard 当作完整模型保存，恢复时会缺参数；把 full state dict 每步聚到 rank 0，又可能导致 CPU 或 GPU 内存峰值过高。训练平台必须明确 checkpoint 类型和恢复流程。

第四个误区是把 tensor parallel、pipeline parallel 和 FSDP 混为一谈。tensor parallel 改变单层算子执行，pipeline parallel 改变层间调度，FSDP 改变状态驻留。它们可组合，但每组合一次都会增加 collective 顺序和 failure surface。

最后，动态图结构与 sharding 的关系不能靠运气。CTGS 的 Gaussian 增删若在不同 rank 上不一致，会让参数 shard 含义不同，后续通信数值仍能完成但模型已经逻辑损坏。

## 8. 本章落点、验证与下一章

本章从 DDP 的模型状态冗余推导出 ZeRO/FSDP。DDP 每卡约持有 \(16N\) bytes 的 mixed-precision Adam 状态；ZeRO-1 切 optimizer states，ZeRO-2 再切 gradients，ZeRO-3/FSDP full shard 连 parameters 也切，并通过 all-gather 与 reduce-scatter 在计算窗口内重建需要的状态。

在基础设施项目中，本章对应显存剖面、FSDP wrap 策略、sharded checkpoint、通信 timeline 和动态参数集合事务。在 CTGS 中，它直接决定大规模 Gaussian 或多材料神经表示能否在有限显存内训练。

本章的 90 分钟验证是用一个小 Transformer 或 MLP 统计 DDP 与 FSDP 的显存：记录 parameters、gradients、optimizer states 和 peak memory；改变 wrap 粒度，观察峰值显存与 all-gather 次数。预期是 full shard 显著降低稳态模型状态，但 wrap 过细会让通信次数增加，wrap 过粗会抬高峰值。

下一章将研究 activation checkpointing 与重计算。状态分片之后，很多训练任务的最大显存重新变成 activations；届时要决定哪些中间结果保存，哪些在 backward 时重新计算，以及这怎样改变 step time。
