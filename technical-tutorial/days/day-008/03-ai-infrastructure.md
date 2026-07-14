# AI Infrastructure 第 8 章：四张 GPU 为什么不能各自训练四个逐渐分叉的模型

## 1. 从一个真实任务开始

上一章把单张 GPU 上稳定的训练 step捕获为 CUDA Graph。现在 CTGS 模型和 projection batches已经大到需要四张 GPU。最容易想到的扩展是每张卡保存一份模型，各自处理不同 views，从而把每 step的样本数扩大四倍。

但四份模型从相同参数开始并不够。每张卡看到不同数据，因此 backward产生不同 gradients。若各自执行 optimizer step，参数在第一步后就会分叉；后续 collective即使交换部分结果，也不再等价于训练一个模型。

今天的任务是建立 Distributed Data Parallel 的数学与系统链条：每个 rank计算局部梯度，collective将它们归约成同一个全局平均梯度，然后所有 rank执行相同更新。真正的性能问题则是：如何让这次通信尽量藏在 backward计算之后，而不是成为每步末尾的一堵墙。

## 2. 最直接的办法，以及它为什么不够

最直接的正确实现是等整个 backward结束，把所有 parameter gradients拼成一个大 buffer，做一次 all-reduce，然后再 optimizer step。它保证了参数一致，却让计算和通信完全串行。

假设 backward需要 40 ms，gradient all-reduce需要 15 ms，optimizer需要 5 ms。总 step至少是

\[
40+15+5=60\ {\rm ms}.
\]

GPU 在最后 15 ms可能只运行通信 kernels，前面已经算完的高层 gradients则一直等待。

另一个直接方法是每得到一个很小的 gradient就立即 all-reduce。这样最早开始通信，却会产生上千个 collectives。每次 collective有固定 latency，网络和 NCCL无法在极小 messages上达到峰值 bandwidth。需要的抽象不是“一个 tensor一次通信”，而是把 gradients按产生顺序聚成适当大小的 buckets。

## 3. 关键想法是怎样被引出来的

设共有 \(P\) 个 ranks，第 \(r\) 个 rank处理 local mini-batch \(\mathcal B_r\)，local loss为

\[
L_r(\theta)
=
\frac{1}{|\mathcal B_r|}
\sum_{z\in\mathcal B_r}
\ell(\theta;z).
\]

若每个 local batch大小相同，全局 batch loss为

\[
L(\theta)
=
\frac1P\sum_{r=1}^{P}L_r(\theta).
\]

微分与有限求和可交换，因此

\[
\nabla_\theta L
=
\frac1P
\sum_{r=1}^{P}
\nabla_\theta L_r.
\]

这正是 data parallel需要的通信：不是交换 activations，也不是轮流传完整模型，而是对每个 parameter的 local gradient求和再除以 \(P\)。只要所有 rank从同一参数开始、使用同一个平均 gradient和相同 optimizer state，它们更新后仍完全一致。

系统上，backward按计算图反向顺序逐层产生 gradients。某一 bucket的所有 gradients ready后，就可以在独立 CUDA stream上启动 asynchronous all-reduce；backward继续计算更早层。通信与计算由此重叠。

## 4. 一步一步建立正式模型

第 \(r\) 个 rank对 parameter vector的 local gradient记为 \(g_r\)。all-reduce sum使每个 rank得到

\[
g_{\rm sum}
=
\sum_{r=1}^{P}g_r.
\]

框架再除以 world size：

\[
\bar g
=
\frac{g_{\rm sum}}{P}.
\]

SGD更新为

\[
\theta_{t+1}
=
\theta_t-\eta\bar g.
\]

这等价于在 global batch上做一次梯度下降，但不自动保证与单卡完全相同：随机数、BatchNorm统计、floating-point reduction order和不等长 batches仍会带来差异。

为了理解通信成本，考虑 ring all-reduce。每个 rank先通过 reduce-scatter获得结果的 \(1/P\) shard，再通过 all-gather收集全部 shards。若 gradient总大小为 \(S\) bytes，每个阶段有 \(P-1\) 步，每步传输 \(S/P\)。每个 rank发送的数据量为

\[
V_{\rm rank}
=
2\frac{P-1}{P}S.
\]

用每步 latency \(\alpha\) 和有效 bandwidth \(\beta\) 表示，时间近似为

\[
T_{\rm ring}
\approx
2(P-1)\alpha
+
2\frac{P-1}{P}\frac{S}{\beta}.
\]

大消息主要受 bandwidth项控制，小消息主要受 latency项控制。这就是 bucket既不能太大也不能太小的原因。

设 backward总时间为 \(T_b\)，所有通信总时间为 \(T_c\)，其中成功重叠的部分为 \(T_o\)。暴露在 step critical path上的时间为

\[
T_{\rm step}
\approx
T_b+T_c-T_o+T_{\rm opt}.
\]

最后产生的 bucket几乎无法与后续 backward重叠，称为 exposed tail。bucket order、size和参数 ready order共同决定 \(T_o\)。

## 5. 跟着一个完整例子走到底

考虑四张 GPU，即

\[
P=4,
\]

模型的全部 gradients为

\[
S=1\ {\rm GB}.
\]

ring all-reduce中，每个 rank发送的数据量为

\[
2\frac{4-1}{4}\times1
=1.5\ {\rm GB}.
\]

若有效 link bandwidth为 100 GB/s，忽略 latency时通信下界约为

\[
\frac{1.5}{100}\ {\rm s}
=15\ {\rm ms}.
\]

现在 backward为 40 ms，optimizer为 5 ms。若整个 gradient在 backward结束后一次归约：

\[
T_{\rm step}
=40+15+5
=60\ {\rm ms}.
\]

把 gradients按反向产生顺序分成四个 256 MB buckets。假设前三个 bucket的 12 ms通信都能与剩余 backward完全重叠，最后一个 bucket仍暴露 3 ms，则

\[
T_o=12\ {\rm ms},
\]

\[
T_{\rm step}
\approx40+15-12+5
=48\ {\rm ms}.
\]

数学结果没有改变，但 step从 60 ms降到 48 ms。若把 bucket再切成 1 MB，理论上更早启动，实际每个 collective的 latency累计，\(T_c\) 可能从 15 ms升到 30 ms；若只用一个 1 GB bucket，\(T_c\) 仍是 15 ms，却完全没有 overlap。

再看梯度数值。四个 ranks对一个 scalar parameter得到

\[
g_0=1,\quad g_1=3,\quad g_2=2,\quad g_3=6.
\]

all-reduce average为

\[
\bar g=\frac{1+3+2+6}{4}=3.
\]

若初始参数 \(\theta=10\)，learning rate \(\eta=0.1\)，所有 ranks都更新为

\[
\theta'=10-0.1\times3=9.7.
\]

若不归约，它们会分别变成 9.9、9.7、9.8和 9.4；下一步已经不再训练同一个模型。

## 6. 回到真实系统：程序实际上怎样工作

一个 DDP step的实际结构是：

~~~text
one process owns one GPU
→ DistributedSampler gives each rank different samples
→ replicated model runs local forward
→ local backward marks gradients ready
→ reducer packs ready gradients into buckets
→ NCCL all-reduce runs asynchronously
→ each rank receives identical averaged gradients
→ each rank performs the same optimizer update
~~~

通常每张 GPU使用一个 process，因为这使 device context、failure和collective ordering更清楚。初始化时还需要同步 model parameters；之后参数一致性依赖所有 ranks执行相同 collectives与 optimizer steps。

PyTorch DDP本身复制 model，并在 backward中同步 gradients；它不会自动切分输入，数据分片由用户或 `DistributedSampler`负责。文档也提供 `bucket_cap_mb`、`gradient_as_bucket_view`和 `static_graph` 等与 reducer行为相关的选项。[PyTorch DistributedDataParallel 文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)

DDP解决计算吞吐，却不减少每张卡保存的 model states。以 mixed-precision Adam为例，每个 parameter可能对应低精度参数、FP32 master parameter、gradient以及一阶和二阶 moments。若模型状态本身放不下，需要 ZeRO/FSDP一类 sharding：stage 1分 optimizer states，stage 2再分 gradients，stage 3进一步分 parameters，并在计算前按需 all-gather。

对 CTGS，有一个额外一致性约束：densification、pruning和Gaussian排序会改变 parameter集合。若每个 rank根据自己的 local views独立新增 Gaussians，参数 shapes与collective顺序会立刻分叉。结构编辑必须由一致的全局统计决定，或由一个 rank决策后 broadcast，并在所有 ranks以同一顺序修改 optimizer state。

CUDA Graph与 DDP 可以结合，但所有 ranks必须 capture/replay一致的 collectives。任何 rank跳过一个 bucket、提前退出或改变 graph topology，其他 ranks都会等待一个永远不会到来的 collective。

## 7. 容易走错的岔路

把 loss在每个 rank先除以 world size、DDP又平均 gradient，可能造成重复缩放。必须确认框架 reducer是 sum还是 average，以及 local loss采用 sum还是 mean。

只看 GPU utilization无法判断通信隐藏得好不好。NCCL kernels本身也让 GPU显示 busy；应在 timeline上检查 backward compute与collectives是否重叠，以及最后 exposed tail有多长。

盲目增大 global batch会改变 optimization。DDP在数学上等价于更大 batch，不保证同一 learning rate、warmup和regularization仍合适。

设置 `find_unused_parameters=True` 可以处理动态未使用参数，却增加 graph traversal与同步复杂度。若模型拓扑稳定，应明确 unused pattern或使用 static graph，而不是长期依靠隐式搜索。

最后，DDP不是显存扩容器。每卡仍保存完整 parameters和多数 states；遇到 model-state OOM时应考虑 sharding、activation checkpointing或model parallel，而不是继续增加 data-parallel ranks。

## 8. 本章落点、验证与下一章

本章从 global-batch gradient推导了 DDP：各 rank计算 local gradient，all-reduce得到同一个平均值，所有模型副本因此保持同步。ring all-reduce每 rank传输 \(2(P-1)S/P\)，gradient buckets则在 message latency与communication-backward overlap之间取平衡。

在训练基础设施中，本章对应 process group、distributed sampler、DDP reducer、NCCL timeline和state sharding边界。在 CTGS 中，它还要求 densification与pruning成为跨 rank一致的结构事务。

本章的 90 分钟验证是用两张或四张 GPU训练一个小模型：记录单卡 global-batch gradient与DDP averaged gradient的最大误差，再分别设置很小、适中和单一大 bucket，用 profiler测量 all-reduce数量、总通信时间和 exposed tail。预期是数值在容差内一致，极小 bucket受 latency拖累，极大 bucket没有 overlap，适中 bucket的 step time最低。

下一章将专门研究通信与计算重叠及并行策略选择：当 gradient all-reduce已经优化后，什么时候该使用 FSDP/ZeRO、tensor parallel、pipeline parallel或sequence/context parallel，以及它们分别移动哪一种 tensor。
