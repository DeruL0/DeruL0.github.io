# AI Infrastructure 第 4 章：反向传播为什么要保存中间结果，以及 checkpointing 在交换什么

## 1. 从一个真实任务开始

上一章优化了 kernel 的数据复用，但训练 CTGS、NeRF 或大型网络时，程序常在算力尚未跑满前先触发 out-of-memory。减小 batch 可以运行，却降低吞吐和统计稳定性；换更大显存 GPU 只是延后问题。真正需要回答的是：一次训练 step 的显存究竟由谁占用，哪些数据必须保存到 backward，哪些可以在需要时重新计算？

今天的任务是把 autograd graph 看成带生命周期的数据流。forward 产生 activation，backward 按逆序消费它们。activation checkpointing 刻意不保存一部分中间结果，反向到达该区域时重新执行 forward，用额外计算换取更低峰值显存。

## 2. 最直接的办法，以及它为什么不够

最直接的节省方法是在 forward 完成后删除所有中间 tensors，只保留 loss。问题是 backward 的局部导数通常依赖 forward 值。例如 ReLU backward 需要知道哪些输入为正，矩阵乘对权重的梯度需要输入 activation，体渲染 backward 需要 transmittance 或能重建它的量。删除后无法直接计算梯度。

另一个直接办法是把所有中间结果都保存。这保证 backward 快速，却让 activation memory 随层数、分辨率、batch 和 ray samples 增长。尤其是高分辨率 feature maps、per-ray samples 或每个 Gaussian 的中间排序数据，可能远大于参数本身。

所以显存管理不能只有“保存或不保存”两种选择。关键是保存少量边界状态，把计算图分成 segments；backward 进入某个 segment 时，从最近 checkpoint 重放该段 forward，恢复局部需要的 activations，用完立即释放。

## 3. 关键想法是怎样被引出来的

设 forward 是链

\[
h_i=f_i(h_{i-1}),
\qquad i=1,\ldots,N.
\]

普通 reverse-mode AD 保存 \(h_0,h_1,\ldots,h_N\)，然后从 \(h_N\) 向前计算 gradients。checkpointing 只保存某些 \(h_k\)。当需要处理 \(f_{k+1},\ldots,f_{k+s}\) 的 backward 时，从 \(h_k\) 重放这一段 forward，临时恢复中间 tensors。

这是一种生命周期变换：原来 activation 从 forward 产生后一直活到对应 backward；checkpointing 让它在 forward 后立即死亡，随后在 backward 附近重新出生并很快释放。

因此它交换的不是抽象的“空间与时间”，而是具体的 activation bytes 与重复 kernel work。若模型瓶颈本来在计算，重放代价明显；若训练受显存限制、减 batch 会严重降低利用率，checkpointing 可能反而提高端到端吞吐。

## 4. 一步一步建立正式模型

训练峰值显存可粗略拆成

\[
M_{\rm peak}
\approx
M_{\rm param}
+M_{\rm grad}
+M_{\rm optimizer}
+M_{\rm activation}
+M_{\rm temporary}
+M_{\rm allocator}.
\]

checkpointing 主要改变 \(M_{\rm activation}\)，不会自动减少参数、gradient 和 Adam moments。mixed precision、optimizer sharding 和 activation checkpointing 针对的是不同项。

先假设 \(N\) 层每层 activation 大小都为 \(S\)，忽略其他内存。全部保存需要

\[
M_{\rm save-all}\approx NS.
\]

每隔 \(K\) 层保存一个 checkpoint，大约需要 \(N/K\) 个长期 activation。反向重放一个 segment 时，最多临时产生约 \(K\) 个 activations，因此理想化峰值为

\[
M_{\rm ckpt}
\approx
S\left(\frac{N}{K}+K\right).
\]

令两项平衡，\(K\approx\sqrt N\)，得到

\[
M_{\rm ckpt}\approx2S\sqrt N.
\]

这比 \(NS\) 从线性增长降为平方根级的理想模型。真实网络各层大小不同，最优分段应按 bytes 和 compute cost，而不是机械按层数。

计算代价来自重放。若每个未保存 segment 在 backward 中重算一次，额外 forward work 接近一次原 forward。因为训练本来还包含 backward，step time 通常不会简单翻倍，但增加多少取决于 forward/backward 比例和 kernel overlap。

autograd 本身也只保存 backward 真正需要的 tensors。自定义 operation 应在 `save_for_backward` 中保存最小充分信息。例如一个可从输出恢复 mask 的函数，不应同时保存完整输入、输出和临时 workspace。checkpointing 不能弥补 backward API 无节制保存数据。

## 5. 跟着一个完整例子走到底

考虑 16 个顺序 blocks，每个 block 输出 256 MiB activation，计算成本近似相同。忽略参数和临时 workspace，全部保存需要

\[
16\times256\ \mathrm{MiB}
=4096\ \mathrm{MiB}.
\]

现在每 4 个 blocks 设置边界 checkpoint。长期保存约 4 个 segment boundaries，大小约

\[
4\times256=1024\ \mathrm{MiB}.
\]

backward 处理一个 segment 时，从其入口重放 4 个 blocks，局部最多产生约

\[
4\times256=1024\ \mathrm{MiB}
\]

的临时 activations。因此理想化峰值约为 2 GiB，而不是 4 GiB。具体框架还会保留 segment 输入、输出、gradients 和 temporary buffers，所以实测不会恰好减半，但数量级和峰值位置可由 timeline 验证。

forward 首次执行 16 个 blocks；backward 期间四个 segments 各重放一次，总共额外执行约 16 个 forward blocks。若原 forward 占 step compute 的三分之一、backward 占三分之二，额外一次 forward 可能让纯计算时间从 3 个单位增至 4 个单位，约增加三分之一，而不是翻倍。

若显存节省允许 batch 从 1 增到 2，并让 GPU 从许多小 kernels 变成更饱满的 kernels，实际 samples/s 仍可能提高。正确评估指标因此是吞吐与峰值显存的联合变化，而不是只看单 step 延迟。

## 6. 回到真实系统：程序实际上怎样工作

PyTorch checkpoint 通常在原 forward 中只保留 segment 输入和必要状态，backward 时重新调用该函数。被 checkpoint 的函数必须在重放时产生与原 forward 一致的计算；random dropout、随机采样和全局可变状态需要保存/恢复 RNG 或显式传入，否则 backward 使用的是另一条函数路径。

分析顺序应先做 memory snapshot 或 profiler，确认峰值来自 activations，而不是 optimizer state、CUDA caching allocator 或某个自定义 op 的 temporary workspace。然后按高 bytes、可重算、边界清晰的 blocks 选择 checkpoint，而不是给整个模型一键开启。

对 CTGS，rasterizer backward 可能保存每 pixel 的 contributor list、累计 transmittance 和排序结果。三种选择需要分别比较：保存全部、重算 binning/sorting、保存压缩后的最小状态。sorting 重算成本可能比普通 MLP block 高，不能只按 tensor bytes 决策。

对 ray-based model，samples、densities 和 features 的尺寸与 rays × samples 成正比。checkpoint ray marcher 可降低保存量，但若采样本身含随机扰动，必须保证重放的 sample positions 一致。

## 7. 容易走错的岔路

看到 OOM 就启用 checkpointing，可能掩盖真正的 graph retention。把带 graph 的 loss 存进 Python list、忘记 `detach`，会让多个 steps 的图都不能释放；checkpointing 只能减少每张图，不能修复无限累计。

以为 checkpoint 后所有显存都按比例下降也不准确。参数、gradients、optimizer states 和不可重算 workspace 保持不变；activation 只占小部分时收益有限。

重放函数包含 in-place mutation、I/O 或依赖全局计数器时，第二次 forward 可能与第一次不同。checkpoint segment 应尽量是纯函数式数据变换。

分段越细并非越省。保存太多 boundaries 会让 \(N/K\) 项上升；分段太大则重放局部峰值 \(K\) 项上升。还要考虑各层 activation 大小不均匀。

最后，只比较 `max_memory_allocated` 也不够。allocator reserved memory、fragmentation 和库 workspace 可能决定能否启动下一个 kernel；同时记录 allocated、reserved 和 timeline 才能解释 OOM。

## 8. 本章落点、验证与下一章

本章把 autograd 显存解释为 activation 生命周期问题。普通 reverse mode 让中间结果从 forward 活到对应 backward；checkpointing 只保存 segment boundaries，在反向附近重放局部 forward，使理想 activation 峰值从 \(NS\) 变为约 \(S(N/K+K)\)。它只改变 activation 项，并以额外计算为代价。

在 CTGS/NeRF 项目中，本章对应 renderer saved tensors、ray samples 和网络 blocks 的重算策略；在训练平台中，应把 peak allocated/reserved memory、checkpoint policy 和 throughput 一同纳入实验记录。

本章的 75 分钟验证是构造 16 个大 activation blocks，分别运行全部保存和每 4 blocks checkpoint。用 profiler 记录 forward/backward timeline、peak allocated memory 和 step time。预期是 checkpoint 版本在 backward 中出现重复 forward kernels，activation 峰值显著下降，step time 上升；若保留每步 loss graph 到列表中，两种版本的显存都会逐步增长，从而区分 checkpoint 与 graph leak。

下一章会继续处理训练数值和吞吐：mixed precision 怎样减少 activation 与带宽、FP16/BF16 的动态范围为何不同，以及 loss scaling 如何避免小 gradient 下溢而不掩盖真正的 overflow。

