# AI Infrastructure 第 2 章：怎样判断一个 GPU kernel 到底在等计算还是等数据

## 1. 从一个真实任务开始

上一章解释了 GPU 依靠大量 warp 隐藏延迟，也用 arithmetic intensity 初步区分计算瓶颈和带宽瓶颈。现在把问题放回一次真实的 CTGS 训练：总 step time 是 120 ms，profiler 显示一个逐元素更新 kernel 占了 18 ms，一个 projector 占了 31 ms。我们需要决定下一周应该优化什么。

如果逐元素 kernel 已经接近显存带宽上限，继续减少几条乘法指令几乎没有意义；如果 projector 的算术单元长期空闲是因为访问分散，单纯换用 Tensor Core 也不会解决问题。Infrastructure 的工作不是看到“慢”就尝试优化开关，而是把每个 kernel 放入一个可证伪的性能模型：它移动了多少数据，执行了多少运算，硬件能提供哪些上限，实测结果距离哪个上限最近？

今天的输出不是一句“memory-bound”标签，而是一套从时间线、工作量估算、Roofline 到访存事务的完整诊断方法。

## 2. 最直接的办法，以及它为什么不够

最直接的判断来自监控面板。GPU utilization 为 100%，于是我们说 GPU 已经跑满；显存占用只有一半，于是我们说显存不是问题；换一张峰值 TFLOP/s 更高的卡速度没有同比提升，于是又怀疑框架开销。

这些指标都可能是真的，却没有回答 kernel 在做什么。utilization 通常只表示采样窗口内 GPU 上存在活动，低效率的 warp、等待 memory dependency 的 warp 和真正持续执行 Tensor Core 的 warp 都能让设备显示繁忙。显存容量表示能放下多少数据，显存带宽表示每秒能搬多少数据，两者也不是同一件事。

另一个常见做法是先调整 block size。某个配置快了 8%，我们便保留它；换一个输入尺寸后提升消失。原因是 block size 同时影响 occupancy、register pressure、memory access pattern 和 block 数量。没有上层模型时，这只是局部搜索，无法说明为何有效，也无法判断还有多少空间。

所以必须把“硬件很忙”拆成可计算的资源使用：算术工作量 \(F\)、从某一级存储移动的数据量 \(Q\)、执行时间 \(T\)，以及对应硬件的峰值吞吐和带宽。

## 3. 关键想法是怎样被引出来的

一个 kernel 完成前，至少要等两件事：所需运算被执行，所需数据被送到执行单元。假设二者可以理想重叠，那么耗时不可能小于两项下界中的较大者：

\[
T
\ge
\max\left(
\frac{F}{P_{\max}},
\frac{Q}{B_{\max}}
\right).
\]

这里 \(P_{\max}\) 是目标数据类型和指令路径的峰值计算吞吐，\(B_{\max}\) 是所讨论存储层级的带宽。把 \(F/Q\) 记作 arithmetic intensity \(I\)，就得到 Roofline：

\[
\frac{F}{T}
\le
\min(P_{\max},B_{\max}I).
\]

这条折线把优化方向分成两侧。低 \(I\) 区域受斜线 \(B_{\max}I\) 限制，减少字节或增加数据复用最有价值；高 \(I\) 区域碰到水平的计算上限，指令吞吐、Tensor Core、occupancy 和依赖链更重要。

但 \(Q\) 不能只按源代码中的数组元素计算。一个 warp 发出 32 个 load，memory subsystem 按对齐的 sector 或 cache line 传输。相邻线程访问相邻地址时，请求能够合并；跨大步长访问时，为了取得同样 128 bytes 有效数据，硬件可能传输数倍的事务字节。于是需要区分：

```text
算法需要的 useful bytes
→ warp 产生的 memory requests
→ 合并后的 sectors / cache lines
→ L2 与 DRAM 实际传输的 bytes
```

Roofline 告诉我们上限在哪里，coalescing 和 cache 指标解释为什么实际 \(Q\) 比算法直觉更大。

## 4. 一步一步建立正式模型

仍从最简单的 kernel 开始：

\[
y_i=ax_i+b.
\]

对 float32，每个元素执行一次乘法和一次加法，约为 2 FLOPs；忽略被缓存的常数 \(a,b\)，至少读取 \(x_i\) 的 4 bytes 并写出 \(y_i\) 的 4 bytes。因此

\[
F=2N,
\qquad
Q_{\rm useful}=8N,
\qquad
I=0.25\ \mathrm{FLOP/byte}.
\]

若 GPU 的持续 DRAM 带宽约为 \(1\ \mathrm{TB/s}\)，带宽屋顶只有

\[
B_{\max}I
=
1000\times0.25
=250\ \mathrm{GFLOP/s}.
\]

即使 FP32 峰值为几十 TFLOP/s，这个 kernel 也不可能靠计算单元达到那个数字。真正有意义的实测量是

\[
B_{\rm achieved}
=
\frac{Q_{\rm transferred}}{T},
\]

而不是只报告 achieved FLOP/s。

现在检查 warp 地址。连续访问时，lane \(l\) 读取

\[
\operatorname{addr}(l)
=
\operatorname{base}+4l.
\]

32 个 lane 总共请求连续 128 bytes。若起始地址满足对齐，硬件只需少数连续 memory sectors。若改为 stride 32 个 float：

\[
\operatorname{addr}(l)
=
\operatorname{base}+4\times32l,
\]

相邻 lane 相距 128 bytes。每个 lane 可能落入不同 sector，取得 128 bytes 有效数据却触发大约 32 个独立 sector。以 32-byte sector 为直观模型，事务字节可从约 128 增加到约 1024，load efficiency 只有八分之一。具体 sector 和 cache 行为随架构及缓存命中变化，但“logical bytes 不等于 transferred bytes”这一点不变。

对真实 kernel，分析需要形成一组相互校验的量：

\[
P_{\rm achieved}=F/T,
\qquad
B_{\rm achieved}=Q/T,
\qquad
I=F/Q.
\]

同时查看 DRAM bytes、L2 hit rate、sectors per request、active warps、issue rate 和主要 stall reasons。若 DRAM 带宽接近持续上限而计算吞吐很低，结论是外存带宽限制；若 DRAM 不高但 long scoreboard stall 很高，可能是访问延迟大且并行度不足；若计算管线接近上限，才进入指令与计算优化。

Roofline 也可以分层。一个 tile 从 DRAM 读入一次、从 shared memory 被复用几十次时，相对于 DRAM 的 intensity 很高，相对于 shared memory 的访问压力仍可能很大。只画一条 HBM roof 会漏掉 L2、shared memory 或 register bank 的瓶颈。

## 5. 跟着一个完整例子走到底

设 \(N=2^{28}=268{,}435{,}456\)，执行 \(y=ax+b\)。最小数据流量是

\[
Q=8N
\approx2.147\ \mathrm{GB}.
\]

总运算量为

\[
F=2N
\approx0.537\ \mathrm{GFLOP}.
\]

在持续带宽 \(1\ \mathrm{TB/s}\) 的理想设备上，memory lower bound 约为

\[
T_{\rm memory}
=
\frac{2.147\ \mathrm{GB}}
{1000\ \mathrm{GB/s}}
\approx2.15\ \mathrm{ms}.
\]

若 FP32 峰值按 \(60\ \mathrm{TFLOP/s}\) 估计，compute lower bound 只有

\[
T_{\rm compute}
=
\frac{0.537\ \mathrm{GFLOP}}
{60{,}000\ \mathrm{GFLOP/s}}
\approx0.009\ \mathrm{ms}.
\]

二者相差两百多倍，所以优化乘加指令不会改变主导项。假设实测连续版本为 2.8 ms，则按 useful bytes 计算的有效带宽为

\[
2.147/0.0028
\approx767\ \mathrm{GB/s}.
\]

它已经接近假设的持续上限。这个 2.8 ms 不是“只用了很少 TFLOP/s”的失败，而是一个接近带宽屋顶的合理结果。

现在把输入改成大 stride。若 profiler 显示 DRAM 实际传输增加到约 17 GB，哪怕数学工作量完全相同，带宽下界也会上升到约 17 ms。此时重新布局为 structure-of-arrays、让 warp 访问连续索引，或者先做转置，比减少一条算术指令重要得多。

最后考虑三次独立逐元素操作。每次都读一个 4-byte 输入并写一个 4-byte 输出，总流量约为

\[
3\times8N=24N
\approx6.442\ \mathrm{GB}.
\]

融合后中间值留在 register，只读原始输入并写最终输出，回到 \(8N\approx2.147\ \mathrm{GB}\)。在完全带宽受限且 register pressure 没有恶化的理想情况下，融合上限接近三倍。这就是为什么 `torch.compile` 可能让 FLOP 数不变的图明显加速。

## 6. 回到真实系统：程序实际上怎样工作

诊断顺序应先宽后窄。先用 PyTorch profiler 或 Nsight Systems 看整步时间线，确认 CPU launch gap、同步、数据加载和 GPU kernels 各占多少；再对真正占主导的 kernel 使用 Nsight Compute，检查 memory workload、scheduler 和 Roofline section。没有必要对一个只占 step time 1% 的 kernel 做两天 micro-optimization。

对每个候选 kernel，记录一份与输入共同变化的实验元数据：tensor shape、dtype、Gaussian count、ray 数、每条 ray 平均 sample 数、GPU 型号、warmup 次数和同步位置。CUDA 是异步提交的；若计时前后没有 event 或显式同步，测到的可能只是 launch 时间。

在 CT projector 中，算法 useful bytes 可以从“每条射线采多少点、每个点读取多少体素”估算。三线性插值逻辑上读取八个邻居，但 cache 和 texture path 可能复用；相邻 ray 是否访问相近体素，会决定实际 L2/DRAM 流量。ray 长度差异还会导致 warp 内循环次数不同。

在 Gaussian rasterizer 中，binning、sorting 和 blending 应分别分析。sorting 可能受比较与临时存储影响，tile blending 可能同时受 overdraw、数据读取和原子操作影响。把整个 renderer 合成一个平均 FLOP/byte，往往会掩盖真正瓶颈。

NVIDIA 的 CUDA Best Practices Guide 对 global memory coalescing 和实际带宽计算给出了架构化说明；Nsight Compute 则提供 memory sectors、throughput 与 Roofline 数据。它们的作用是帮助验证本章模型，而不是替代工作量估算。[CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)

## 7. 容易走错的岔路

“GPU utilization 100%”看似已经给出结论，实际只排除了设备完全空闲。它不能区分高效计算、低效访存和大量等待。

按源码中的数组元素统计 \(Q\) 也容易低估瓶颈。未合并访问、cache miss、write allocate 和中间临时量都会让实际事务字节高于 useful bytes。反过来，cache 复用也可能让 DRAM bytes 低于所有逻辑 load 之和，所以最好同时报告两者。

把产品页峰值带宽直接当作 \(B_{\max}\) 会让所有 kernel 看起来都离屋顶很远。更实际的屋顶来自同机型、同数据宽度的大型拷贝或官方持续性能测量；ECC、功耗、频率和访问模式都会影响可达值。

融合也不是越多越好。融合后 register 使用增加，occupancy 下降，甚至发生 spill；原本可以并行调度的阶段也可能被绑在一起。正确问题是“减少的 global traffic 是否大于增加的执行与资源代价”。

最后，microbenchmark 加速不等于训练加速。若 kernel 只占总时间很小一部分，或优化后触发了新的同步和显存峰值，端到端 step time 可能没有变化。每次局部优化都必须回到整步时间线验证。

## 8. 本章落点、验证与下一章

本章把“GPU 可能受带宽限制”变成了一个可以计算和测量的判断。\(F\)、\(Q\) 和 \(T\) 给出 achieved compute 与 bandwidth；Roofline 说明性能上限由哪类资源决定；warp 地址和 memory sectors 解释算法字节怎样膨胀成实际传输；分层 Roofline 则提醒我们瓶颈也可能位于 L2 或 shared memory。

对 CTGS 项目，这套方法应直接用于 projector、tile rasterization、mask/opacity 更新和 optimizer kernel。对 AI Infrastructure，它应成为性能回归表的基础字段，而不是只存一个总 step time。

本章的 75 分钟验证是写三个大数组实验：连续的 \(y=ax+b\)、大 stride 读取、以及三个逐元素操作的独立与融合版本。完成 warmup 后用 CUDA events 计时，并记录 profiler 的 DRAM bytes。预期是：连续版本接近带宽屋顶；stride 版本的 logical FLOPs 不变但事务字节和耗时上升；融合版本的 kernel 数与中间写回减少，速度上限接近流量减少比例，但不一定精确达到三倍。

下一章会追问如何主动提高 arithmetic intensity。矩阵乘和局部邻域算子为什么要分 tile，shared memory 如何把一次 DRAM 读取转化为多次复用，tile 尺寸又为什么会同时受到 register、occupancy、bank conflict 和同步成本约束。

