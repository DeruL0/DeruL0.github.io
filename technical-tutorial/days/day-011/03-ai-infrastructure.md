# AI Infrastructure 第 11 章：为什么少几个 kernel launch 往往比少几行 Python 更重要

## 1. 从一个真实任务开始

上一章用 activation checkpointing 在显存和重计算之间做交易。现在显存能放下了，训练或推理仍然慢。Profiler 显示 GPU 上有大量很短的 kernels：elementwise add、multiply、activation、mask、reshape、small reductions。每个 kernel 本身只跑几十微秒，数据却反复从 HBM 读出写回，CPU 还要不断 launch。

真实任务是优化 CTGS 训练中的一段 ray compositing 或 MLP 后处理。代码在 PyTorch eager mode 中很清楚：先算 density，再 clamp，再算 alpha，再乘 transmittance，再累积 color。每一步都是一个 tensor op。可读性很好，但 GPU 看到的是一串小 kernels 和一堆中间 tensor。

今天的主题是编译栈与算子融合。目标不是把 Python 写得更“底层”，而是让系统捕获计算图，识别哪些操作可以合并，生成一个更少读写、更少 launch、布局更合适的 kernel。

## 2. 最直接的办法，以及它为什么不够

最直接的优化是手写 CUDA kernel。它能非常快，但维护成本高。每改一次模型、shape 或数据布局，都要改 C++/CUDA、编译扩展、处理 autograd 和多平台兼容。对研究迭代很快的 CTGS 或 neural rendering，手写所有融合 kernel 不现实。

第二个办法是只依赖 PyTorch eager。它最灵活，debug 也方便，但每个 op 都单独调度，graph-level optimization 很少。像

\[
y=\operatorname{relu}(x+a)\cdot b
\]

这样简单的表达式，eager 可能产生 add、relu、mul 三个 kernels 和两个中间 tensor。

第三个办法是只调 batch size 或使用更快 GPU。这不会改变程序的 memory traffic 和 launch overhead。若瓶颈是 bandwidth 和调度碎片，硬件提升会被低效执行吃掉。

因此需要编译器在保持高层模型表达的同时，把 Python tensor 程序变成更适合硬件的执行图。PyTorch 2.x 的 `torch.compile` 就是这条路线：捕获 Python 中的 tensor operations，遇到无法追踪的部分产生 graph break，再把可编译图交给后端优化。[Introduction to torch.compile](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)

## 3. 关键想法是怎样被引出来的

GPU kernel 的成本不只来自算术，还来自 launch、global memory 读写和中间 tensor 分配。若三个 elementwise ops 分开执行，第一次写出的中间 tensor 马上被第二次读，第二次又写一个中间 tensor。数据在 HBM 中来回移动，算术却很少。

算子融合的关键想法是：只要多个 ops 对同一个 index 独立计算，并且没有需要全局同步的依赖，就可以把它们合成一个 kernel。对每个元素 \(i\)，直接计算

\[
y_i=\max(x_i+a_i,0)\,b_i.
\]

这样中间值留在寄存器里，不落到 HBM。编译器做的事情是先捕获 graph，再分析依赖、shape、layout 和 dtype，最后生成目标后端代码。PyTorch 的现代路径大致包含 Dynamo 捕获、AOTAutograd 处理前后向图、Inductor 做优化和代码生成；GPU 端常生成 Triton 或相关后端 kernel。

关键抽象不是“自动加速”，而是 graph。没有稳定 graph，编译器不知道可优化的边界；有 graph break，就会回到 eager islands，融合机会随之消失。

## 4. 一步一步建立正式模型

考虑三个 elementwise operations：

\[
t_1=x+a,
\]

\[
t_2=\operatorname{relu}(t_1),
\]

\[
y=t_2\cdot b.
\]

若每个 tensor 有 \(N\) 个 FP32 元素，unfused 执行的粗略 global memory traffic 为：add 读 \(x,a\) 写 \(t_1\)，relu 读 \(t_1\) 写 \(t_2\)，mul 读 \(t_2,b\) 写 \(y\)。元素访问总数约为

\[
3N+2N+3N=8N.
\]

字节数是

\[
32N\ {\rm bytes}.
\]

融合后，每个元素只读 \(x,a,b\)，写 \(y\)：

\[
4N
\]

次元素访问，即

\[
16N\ {\rm bytes}.
\]

理论上 memory traffic 减半，kernel launch 从 3 次变 1 次。若这个表达式是 bandwidth-bound，速度提升可能接近这个比例；若是 compute-bound 或受 occupancy 限制，收益会不同。

编译器还会做 shape specialization。若某个维度固定为

\[
N=1024,
\]

kernel 可以选择特定 tile size 和 vectorization；若 shape 每步变化，编译器可能生成多个 specialized kernels，或退回动态图处理。动态 shape 不是不能编译，但会增加 guard、recompile 和优化难度。

Backward 也需要 graph。AOTAutograd 可以从 forward graph 生成 backward graph，再让 Inductor 融合 backward 中的 elementwise 和 reduction 周边操作。对训练系统来说，forward 加速但 backward graph break 仍会留下大量 overhead。

## 5. 跟着一个完整例子走到底

设

\[
N=10^8
\]

个 FP32 元素。Unfused memory traffic 约为

\[
8N\times4
=
3.2\ {\rm GB}.
\]

Fused 后为

\[
4N\times4
=
1.6\ {\rm GB}.
\]

若有效带宽为 800 GB/s，单看 memory 下界，unfused 为

\[
\frac{3.2}{800}\ {\rm s}=4\ {\rm ms},
\]

fused 为

\[
\frac{1.6}{800}\ {\rm s}=2\ {\rm ms}.
\]

还没有算 kernel launch overhead。若每次 launch 约 10 微秒，三个 kernels 是 30 微秒，一个 kernel 是 10 微秒。对大 tensor，bandwidth 主导；对很多小 tensor，launch overhead 会更明显。

现在把例子放回 CTGS。若每条 ray 的 compositing 写成多个 PyTorch ops，alpha、weight、transmittance、color accumulation 都可能产生中间 tensor。编译或手写融合 kernel 可以让每个 ray chunk 在一个 kernel 内完成更多工作，减少 HBM 读写。但如果代码中间调用 Python list append、根据 tensor value 分支、打印 debug 信息或使用无法追踪的自定义对象，就会产生 graph break，融合边界被切断。

这个例子说明编译优化不是魔法。它需要代码把 tensor computation 暴露成稳定图，才有机会融合。

## 6. 回到真实系统：程序实际上怎样工作

实际启用 `torch.compile` 前，应先用 profiler 找到 kernel 数量、graph breaks、memory bandwidth 和 recompilation。只看 wall time 不够，因为第一次 compile 会有开销；要区分 compile time 和 steady-state runtime。

对 CTGS 或可微渲染，最适合编译的是 shape 稳定、control flow 简单、tensor ops 密集的模块，例如 MLP block、ray chunk compositing、loss 后处理和部分 regularizer。不适合一开始编译整个训练 loop，尤其包含 densification、pruning、logging、checkpoint、数据加载和可变结构编辑的部分。

自定义 CUDA/Triton kernels 仍然有位置。当编译器不能理解稀疏数据结构、复杂 ray traversal 或特殊 memory layout 时，手写 kernel 更可靠。编译器和手写 kernel 的边界应清楚：高层图负责组织，低层 kernel 负责热路径。

与 FSDP、checkpointing 和 DDP 组合时，要注意 graph breaks 与 collective。分布式 collective、动态参数增删和重计算都可能限制编译区域。应从单卡单模块编译验证数值，再扩展到分布式训练。

## 7. 容易走错的岔路

第一个误区是把 `torch.compile` 当成无条件加速开关。若代码 graph break 很多、shape 频繁变化或模型主要瓶颈在自定义 kernel，收益可能很小。

第二个误区是忽略 recompile。动态 shape 或 Python guard 失效会触发重新编译，导致周期性卡顿。训练日志应记录 compile 次数和 graph break 原因。

第三个误区是为了编译牺牲正确性。把 tensor-dependent branch 改成固定分支可能提高速度，却改变模型逻辑。优化必须先有数值一致性测试。

第四个误区是只看 kernel 数减少。融合过度可能增加寄存器压力，降低 occupancy，反而变慢。需要同时看 memory traffic、occupancy、launch count 和实际 wall time。

最后，不要把编译器当作替代算法设计。低效采样、过多无效 rays、错误数据布局和不必要的 precision 仍要从算法和数据结构层面解决。

## 8. 本章落点、验证与下一章

本章把训练执行从 eager kernels 推进到 graph capture 与 operator fusion。融合把中间 tensor 留在寄存器或共享内存中，减少 HBM traffic 和 launch overhead；编译栈依赖稳定图、shape guard 和后端代码生成。

在 Infrastructure 与 CTGS 项目中，本章对应 `torch.compile` 试验边界、graph break 诊断、ray compositing 热路径、Triton/CUDA 自定义 kernel 边界，以及分布式训练中的编译安全区。

本章的 60 到 90 分钟验证是写一个 `relu((x+a))*b` microbenchmark，比较 eager 与 `torch.compile` 后的 kernel 数、显存带宽和运行时间；再加入一个 tensor-dependent Python branch 观察 graph break。预期是稳定 elementwise 图能融合并减少 kernel 数，动态分支会切断优化。

下一章将进入大模型并行策略。编译和融合解决单个 rank 内的执行效率；当模型或 batch 继续扩大，仍需要在 data、tensor、pipeline、sequence/context 等轴上拆分工作，并分析每种拆分移动了什么数据。
