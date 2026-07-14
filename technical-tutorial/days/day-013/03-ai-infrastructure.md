# AI Infrastructure：容错检查点、弹性恢复与训练可观测性

## 1. 从一个真实任务开始

今天的 infrastructure 任务是让一个长时间、多 GPU 的训练作业真正可运行。你可能在训练一个 CTGS 表示、一个 3D reconstruction 模型、一个大规模 diffusion 或 transformer，也可能在做需要几天才能收敛的消融实验。前面章节已经讲过数据并行、tensor parallel、pipeline parallel、ZeRO/FSDP、显存分解和通信瓶颈；这些知识让训练能扩到更多 GPU。今天要处理的是另一个现实：GPU 越多、训练越长，失败就越不是异常，而是系统的常态。

在小实验里，训练失败通常只是重新跑一次。在 256 或 1024 张 GPU 上跑几天时，节点重启、网络抖动、NCCL hang、磁盘抖动、数据读取错误、OOM、preemption 都会出现。没有容错设计的训练系统，本质上是在赌整个集群在你的训练窗口内不出事。这个赌注通常不值得。

## 2. 最直接的办法，以及它为什么不够

最直接的做法是在 rank 0 上每隔一段时间执行一次 `torch.save(model.state_dict())`。这在单卡或简单 DDP 中看起来够用，因为模型参数集中在每个进程里，优化器也相对简单。失败后加载模型，继续训练，似乎问题解决了。

它在现代训练系统里不够。FSDP 或 ZeRO 会把参数、梯度和 optimizer state 分片到不同 rank；tensor parallel 会让一个层的权重本身分布在多个设备上；pipeline parallel 会让不同 stage 拥有不同子图；dataloader 有自己的 epoch、shuffle seed 和消费位置；混合精度 scaler、learning-rate scheduler、随机数状态也会影响后续轨迹。只保存 rank 0 的模型，恢复后可能得到一个“能跑但不是同一次训练”的状态。更糟的是，如果没有观测指标，训练卡住时你甚至不知道是数据读取慢、all-reduce 慢、某个 rank OOM，还是 NCCL 在等待已经死亡的 peer。

## 3. 关键想法是怎样被引出来的

从这个失败中引出的关键抽象是训练状态，而不是模型权重。训练状态包括模型分片、optimizer 分片、scheduler、AMP scaler、RNG、dataloader 位置、global step、parallel topology、代码版本、配置文件和数据版本。检查点不是一个文件，而是一个分布式状态快照；恢复也不是简单 load，而是把所有 rank 重新放回同一个逻辑时间点。

因此，容错系统需要两个契约。第一个是 checkpoint contract：什么时候保存、保存哪些 state、写入是否原子、是否能跨 rank 一致恢复。第二个是 observability contract：训练系统必须持续暴露 step time、samples/sec、GPU utilization、显存、通信时间、loss、gradient norm、data loader latency 和 rank heartbeat。PyTorch 的 [Distributed Checkpoint](https://docs.pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html) 和 `torchrun` fault tolerance 教程，以及 NVIDIA [NCCL environment variable](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html) 文档，都在不同层面服务于这两个契约。

## 4. 一步一步建立正式模型

先把 checkpoint 间隔看成一个优化问题。设每次保存 checkpoint 需要 `C` 秒，平均故障间隔是 `MTBF`，故障率近似为

\[
\lambda=\frac{1}{\mathrm{MTBF}}.
\]

如果每 `T` 秒保存一次 checkpoint，那么保存本身带来的时间开销比例约为

\[
\frac{C}{T}.
\]

另一方面，故障发生时平均会丢掉半个间隔的工作，也就是 `T/2`。故障率是 `lambda`，所以期望丢失工作比例近似为

\[
\frac{\lambda T}{2}.
\]

总开销可以写成

\[
H(T)=\frac{C}{T}+\frac{\lambda T}{2}.
\]

这个式子告诉你 checkpoint 不能太频繁，也不能太稀疏。对 `T` 求最小值，得到近似最优间隔

\[
T^\*=\sqrt{\frac{2C}{\lambda}}.
\]

恢复契约可以写得更直接。若最近一次成功 checkpoint 在时间 `t_c`，恢复后的训练状态应该满足

\[
S_{\mathrm{resume}}=S(t_c),
\]

然后最多重放 `T` 秒内的数据或 step。这里的 `S` 必须是完整训练状态，而不是模型参数的子集。如果 RNG、dataloader 或 optimizer moment 没有恢复，loss 曲线可能继续下降，但实验语义已经变了。

## 5. 跟着一个完整例子走到底

假设一个 CTGS 训练作业每次完整分布式 checkpoint 需要 120 秒。集群在这种作业规模下的平均故障间隔约为 10 小时，也就是 36000 秒。故障率为

\[
\lambda=\frac{1}{36000}.
\]

近似最优 checkpoint 间隔是

\[
T^\*=\sqrt{2\times120\times36000}\approx2939\text{ 秒},
\]

约等于 49 分钟。此时 checkpoint 写入开销约为

\[
\frac{120}{2939}\approx4.1\%.
\]

故障造成的期望丢失工作也约为

\[
\frac{2939}{2\times36000}\approx4.1\%.
\]

总开销约 8.2%。这个数字看起来不小，但它比一次 30 小时训练在第 28 小时失败后从头重跑要便宜得多。它还提供了一个工程判断：如果你把 checkpoint 写入从 120 秒优化到 30 秒，最优间隔会缩短，容错损失也会下降；如果存储系统太慢，过于频繁的 checkpoint 反而会拖垮训练吞吐。

## 6. 回到真实系统：程序实际上怎样工作

真实实现中，checkpoint 应该由训练框架的分布式接口管理，而不是每个 rank 自己随手写文件。PyTorch Distributed Checkpoint 的价值在于它理解分片 state，可以让每个 rank 写出自己的 shard，并在恢复时按当前并行拓扑重新加载。对于 FSDP、tensor parallel 或 CTGS 这种带有动态 Gaussian 数量变化的训练，checkpoint 还要保存结构性状态，例如当前 Gaussian 数量、densification/pruning schedule、optimizer 对每个 Gaussian 参数的 moment、相机或投影 batch 的采样状态。

可观测性方面，训练 loop 每一步至少要记录三个层次。第一层是任务层：loss、validation loss、PSNR 或 CT projection residual、任务指标。第二层是性能层：step time、data time、forward/backward time、optimizer time、all-reduce 或 all-gather time、checkpoint time。第三层是系统层：GPU utilization、显存峰值、CPU memory、I/O throughput、NCCL error、rank heartbeat。没有这些指标，系统出问题时你会只能猜；有这些指标，你可以区分“模型变慢”和“数据加载变慢”，区分“通信瓶颈”和“某个 rank 卡住”。

对你的 infrastructure 项目来说，最务实的设计是把 checkpoint 和 observability 当作训练主循环的一部分，而不是 debug 辅助功能。每次实验目录应该有 config、代码版本、data manifest、periodic checkpoints、restore test log、metrics log 和 failure log。一个可靠系统甚至应该定期做恢复演练：启动训练，保存 checkpoint，强制杀掉进程，再从 checkpoint 恢复并确认 global step、loss、optimizer state 和 dataloader 位置一致。

## 7. 容易走错的岔路

第一个岔路是只保存 rank 0。只要状态被分片，这就不是 checkpoint，而是一个不完整快照。第二个岔路是忘记 RNG 和 dataloader。你可能能恢复训练，但不能恢复实验；shuffle 顺序改变后，细粒度对比和复现实验都会失效。第三个岔路是非原子写入。训练在写到一半时失败，留下一个看似存在但实际损坏的 checkpoint，恢复时会制造更难定位的问题。

第四个岔路是没有 per-rank 可观测性。只看 rank 0 的日志会遮住很多分布式故障，因为真正卡住的常常是另一个 rank。第五个岔路是 checkpoint 过于频繁。容错不是越密越好，存储带宽、训练吞吐和失败率之间必须平衡。第六个岔路是从来不测试恢复。没有 restore test 的 checkpoint，只能算是写文件习惯，不能算是容错能力。

## 8. 本章落点、验证与下一章

今天的落点是：大规模训练系统的基本单位不是 model，而是完整 training state；可靠训练的基本能力不是“保存权重”，而是“在故障后回到同一个逻辑时间点，并知道训练为何变慢或卡住”。checkpoint 解决可恢复性，observability 解决可解释性，两者缺一不可。

验证练习用 60 到 90 分钟完成：为你现有的一个训练脚本写一份 `failure_recovery_contract.md`。列出必须保存的 state，估算一次 checkpoint 的耗时 `C`，假设一个 MTBF，按本章公式计算推荐 checkpoint 间隔；然后列出至少 10 个每 step 或每 N step 应记录的指标。预期结果是你能判断现在的训练脚本在节点失败后会丢多少工作，以及缺哪几个状态会导致恢复不等价。

下一章自然会进入“完整训练平台设计”。因为单个作业有了 checkpoint 和 metrics 之后，下一步就是让调度器、实验管理、数据版本、资源隔离和自动恢复共同工作，形成可以长期支撑 CTGS 或大模型实验的训练平台。

