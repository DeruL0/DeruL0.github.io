# AI Infrastructure 第 7 章：CUDA Graphs 为什么要求下一次训练看起来像上一次

## 1. 从一个真实任务开始

上一章用 compiler fusion 减少了 kernels 和中间流量，但一次训练 step 仍可能包含数百个必须独立执行的 kernels、memcpy 和 library calls。每个 GPU operation 都要经过 host runtime/driver setup；当单个 kernel 只有几微秒时，CPU launch overhead 会在时间线上形成明显空隙。

今天的任务是把一个稳定训练 step 的 GPU 工作捕获为 CUDA Graph，实例化一次后重复 replay。CPU 不再逐个提交全部 operations，而是用一次 graph launch 提交整张依赖图。

CUDA Graphs 不会自动融合 kernels，也不会改变数学计算。它优化的是 work submission。代价是 replay 的 graph structure、kernel arguments 所指向的 memory addresses 和依赖关系必须与 capture 时兼容。

## 2. 最直接的办法，以及它为什么不够

最直接的加速是把整个训练 loop 包进 capture。若其中包含 Python 根据 loss 决定分支、每 step 创建新 tensors、动态改变 Gaussian count 或调用 CPU synchronization，capture 会失败或 replay 错误工作。

另一种做法是 capture 一次后继续把新 batch tensor 重新分配并传入 model。graph kernel nodes 仍保存 capture 时的 pointer arguments；Python variable 名字变了，不会自动修改 graph 中的设备地址。replay 可能继续读取旧 batch。

所以 CUDA Graph 不是记录“高层代码模板”，而是记录一组具体 GPU operations 与 dependencies。要更新数据，通常把新内容复制到地址稳定的 static input buffers；要改变结构，则使用 graph update、bucketed graphs 或重新 capture。

## 3. 关键想法是怎样被引出来的

普通 stream submission 按时间逐个发出 operations：

~~~text
host launch A
host launch B
host library call
host launch C
~~~

CUDA Graph 把这些 operations 先定义成 nodes 与 edges，随后实例化为 executable graph。replay 时，driver 已经知道拓扑和大部分 setup，只需提交整个 executable。

这要求区分两类“动态”：tensor 内容可以改变，只要写入同一地址；graph 结构、launch dimensions、pointer identity 和某些 library state 若改变，旧 executable 可能不再有效。

因此最适合 capture 的区域是 shapes 稳定、控制流稳定、allocation 已完成的 steady-state step。densification、checkpoint save、validation 和日志等低频事件留在 graph 外。

## 4. 一步一步建立正式模型

设一个 step 有 \(K\) 个 GPU operations，第 \(k\) 个实际 device time 为 \(t_k\)，每次 host submission 平均 overhead 为 \(h\)。忽略重叠时，普通提交时间近似为

\[
T_{\rm stream}
\approx
\sum_{k=1}^{K}t_k+Kh.
\]

graph replay 的一次提交 overhead 为 \(h_g\)：

\[
T_{\rm graph}
\approx
\sum_{k=1}^{K}t_k+h_g.
\]

理论节省主要是

\[
\Delta T
\approx
Kh-h_g.
\]

若 kernels 本身很长，\(\sum t_k\gg Kh\)，收益有限；若大量小 kernels 由 CPU launch 限制，收益显著。

capture 后 executable node 保存 kernel function、grid/block dimensions、arguments 和 dependency edges。若 argument 是 device pointer \(p\)，replay 使用的仍是该地址。因此输入更新应写入 \(p_{\rm static}\) 对应的 storage，而不是把 Python reference 指向新 tensor。

对动态尺寸，可建立 buckets。假设最大 active Gaussian count 为 \(N_b\)，static buffers 分配为

\[
[N_b,\ldots],
\]

实际 count \(N\le N_b\) 通过 device scalar 传入，kernel 内只处理

\[
i<N.
\]

这样同一 graph 支持一个范围，但会占用 bucket capacity，并可能让某些 libraries 仍按 \(N_b\) 工作。多个 buckets 在内存、capture 数量和过度计算之间权衡。

graph capture 也形成 memory lifetime。capture 区域中的 allocations 可能来自 graph-aware/private pool；只要 executable 仍会 replay，对应 storage 和 tensor references 就不能被回收或换址。

## 5. 跟着一个完整例子走到底

设一个 CTGS steady step 包含 200 个小 kernels，每个平均 device time 为

\[
5\ \mu s.
\]

总 GPU work 为

\[
200\times5\ \mu s
=1.0\ \mathrm{ms}.
\]

若每个 host launch overhead 平均为

\[
h=4\ \mu s,
\]

普通提交额外需要

\[
200\times4\ \mu s
=0.8\ \mathrm{ms}.
\]

总时间近似

\[
T_{\rm stream}=1.8\ \mathrm{ms}.
\]

假设 graph replay submission 为 10 \(\mu s\)，则

\[
T_{\rm graph}
\approx1.01\ \mathrm{ms},
\]

理想 speedup 约为

\[
\frac{1.8}{1.01}
\approx1.78.
\]

现在 capture 时 Gaussian count 是 500,000。下一次 densification 后变成 610,000，原来 buffers 和 launch 范围不够，不能直接 replay。

一种策略是建立 640,000 bucket。positions、gradients 和 optimizer states 都预分配固定地址，device scalar `active_count` 从 500,000 更新为 610,000。kernels 按 \(i<active\_count\) 屏蔽多余 lanes，同一 graph 可继续 replay。

若 sorting temporary storage 或 rasterizer topology 必须随 count 重新分配，bucket 仍不够。此时结束当前 executable，在结构编辑后 warm up 新 shapes 并重新 capture。densification 频率低于普通 steps，因此 recapture cost 可被后续多次 replay 摊薄。

每个新 batch 也不替换 static input tensor，而是执行

~~~text
copy new batch into static_input
→ update active_count and scalar hyperparameters in place
→ launch graph
→ consume static_output
~~~

从输入到 replay 的地址契约由此闭合。

## 6. 回到真实系统：程序实际上怎样工作

一个稳健流程是：

~~~text
allocate long-lived static inputs/outputs
→ run warmup on a side stream
→ ensure lazy library initialization is complete
→ begin stream capture
→ execute one steady forward/backward/update step
→ end capture and instantiate executable
→ copy fresh data into static buffers
→ replay many times
→ recapture at explicit structural boundaries
~~~

warmup 不是为了性能数字，而是让 cuBLAS/cuDNN、自定义 op caches 和 allocator 在 capture 前完成一次性初始化。capture 期间禁止未支持的 CPU synchronization、device-to-host scalar extraction 和非 graph-safe allocation。

多 stream capture 必须用 events 表达 fork/join，并最终回到 origin stream。没有加入 capture dependency 的旁路 stream work 不会自动成为 graph 一部分。

NCCL collectives 可以进入 CUDA Graph，但所有 ranks 必须以一致顺序 capture/replay 兼容的 collectives，且通信库与版本必须支持。一个 rank 走不同 branch 会导致 collective mismatch 或 deadlock。

NVIDIA CUDA Programming Guide 将 graph 定义为与执行分离的一组 operations/dependencies，并说明 stream capture、instantiation 与 replay 的正式 API 语义。[CUDA Graphs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html)

## 7. 容易走错的岔路

graph replay 更快不表示 GPU compute 更快。若 step 被一个 50 ms kernel 主导，减少几十微秒 launch overhead 几乎没有价值。

修改 Python tensor variable 不会改变已捕获 pointer。必须 in-place copy 到 static storage，或使用明确 graph update 机制。

为了固定 shape 把所有 buckets 设得极大，会浪费显存和计算；bucket policy 应来自真实 count distribution。

capture 中使用 item、打印 GPU scalar 或同步计时，会引入 host dependency 并破坏 capture。profiling 应使用 graph-compatible events 或在 replay 外测量。

随机操作也不能默认安全。RNG state 必须由 graph-aware 机制更新；否则每次 replay 可能重复完全相同随机数，或触碰未捕获 state。

最后，重新 capture 不一定是失败。对低频结构变化，显式 recapture 往往比把所有动态行为塞进一个复杂 graph 更清楚、更可靠。

## 8. 本章落点、验证与下一章

本章把 CUDA Graphs 定位为 GPU work-submission 优化。capture 固定 operations、dependencies 与地址，instantiation 预处理执行信息，replay 用一次 host launch 提交整图。收益约来自 \(Kh-h_g\)，因此主要适合大量小 kernels 的稳定区域。

在 CTGS 中，本章对应 densification 之间的 steady training steps、static renderer buffers 和 count buckets；在训练平台中，应记录 capture time、replay time、graph memory pool、recapture events 与 bucket utilization。

本章的 90 分钟验证是构造 200 个小 pointwise kernels，比较普通 stream 与 graph replay 的 CPU/GPU timeline；随后分别替换 input tensor、in-place copy 到 static input，并改变 active count。预期是替换 reference 后 replay 仍读取旧地址，in-place copy 才更新结果；超出 bucket 时必须切换 graph 或 recapture。

下一章会进入多 GPU 分布式训练：data parallel 为什么需要梯度 all-reduce，bucketization 怎样让通信与 backward 重叠，以及参数、gradient 和 optimizer states 的 sharding 分别解决哪一部分显存。

