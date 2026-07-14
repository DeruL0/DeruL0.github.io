# AI Infrastructure 第 5 章：混合精度为什么既能加速，也会让训练悄悄停住

## 1. 从一个真实任务开始

上一章通过 checkpointing 降低了 activation peak，但训练仍要在 GPU 上读写大量 tensors，并执行矩阵乘、卷积和自定义 rasterization。现代 Tensor Core 对低精度输入吞吐更高，FP16/BF16 activation 也只占 FP32 一半字节。于是我们希望用 mixed precision 同时提高计算吞吐、降低带宽与显存。

真实任务不是把模型整体转换成 half。我们需要决定哪些 operation 使用低精度输入，哪些 reduction 保持 FP32 accumulation，参数更新保存什么精度，以及梯度小到 FP16 无法表示时怎样继续训练。正确的 mixed precision 是一套逐操作数值策略，而不是单一 dtype 开关。

## 2. 最直接的办法，以及它为什么不够

最直接的实现是把 parameters、activations、loss 和 optimizer states 全部 cast 到 FP16。它通常很快暴露两个问题。

第一，数值范围有限。CT projection sum、exponential、large loss 或未归一化坐标可能超过 FP16 最大有限值 65504，变成 infinity。第二，小 gradients 可能低于 FP16 subnormal 范围，直接变成零。

即使 gradient 能表示，直接用 FP16 保存 parameter 也会丢失小更新。权重接近 1 时，FP16 相邻可表示数间距约为 \(2^{-10}\approx9.77\times10^{-4}\)。学习率乘 gradient 若只有 \(10^{-7}\)，从 1 中减去它再 round 到 FP16，权重完全不变。

所以不能用“所有值都占 16 bits”作为目标。低精度适合大规模乘加的输入和部分 activation，关键 reduction、loss、unscale 和 optimizer update 需要更高精度。

## 3. 关键想法是怎样被引出来的

浮点格式由 sign、exponent 和 significand 共同决定。更多 exponent bits 扩大动态范围，更多 fraction bits 提高相对精度。

FP16 有 5 个 exponent bits 和 10 个 fraction bits，精度较好但范围窄。BF16 有 8 个 exponent bits、7 个 fraction bits，范围接近 FP32，但有效数字更少。于是：

- FP16 容易 overflow/underflow，但在相同数量级附近比 BF16 更精细；
- BF16 更能容纳大 activation 和 gradient，却不适合直接积累微小 parameter update；
- FP32 accumulation 可以把大量低精度乘积的和保存在更宽范围和更高精度中。

loss scaling 专门解决 FP16 gradient underflow。反向传播前把 loss 乘一个大尺度 \(S\)，所有 gradients 也被乘 \(S\)，从而进入 FP16 可表示区间；optimizer 前再在 FP32 中除以 \(S\)。在没有 overflow 时，这不会改变数学 gradient。

## 4. 一步一步建立正式模型

FP16 的最大有限值约为

\[
65504,
\]

最小 normal 正数为

\[
2^{-14}\approx6.10\times10^{-5},
\]

最小 subnormal 约为

\[
2^{-24}\approx5.96\times10^{-8}.
\]

低于 subnormal 的值会舍入为零，许多 GPU 路径还可能对 subnormal 使用 flush-to-zero。

BF16 与 FP32 同样使用 8 exponent bits，normal 动态范围约从 \(10^{-38}\) 到 \(10^{38}\)，但 BF16 只有 7 fraction bits。在值 1 附近，BF16 间距约为

\[
2^{-7}=0.0078125.
\]

因此 BF16 的优势主要是 range 与吞吐，不是 parameter update 精度。

设原始 loss 为 \(L\)，scale 后

\[
L'=SL.
\]

链式法则给出

\[
\nabla_\theta L'
=
S\nabla_\theta L.
\]

backward 生成 scaled gradient \(g'=Sg\)。在 optimizer 前执行

\[
g=\frac{g'}{S},
\]

然后做 gradient clipping、finite check 和更新。若在 unscale 前 clipping，阈值也被 scale 改变，结果不再等价。

dynamic loss scaling 从较大 \(S\) 开始。若任何 gradient 出现 inf/NaN，本 step 不更新参数并减小 \(S\)；连续若干 steps 都 finite 时再尝试增大。它在“避免 underflow”与“避免 overflow”之间自动寻找窗口。

混合精度矩阵乘可概念化为

\[
C_{\rm FP32}
=
\operatorname{accum}_{\rm FP32}
(A_{\rm FP16/BF16}B_{\rm FP16/BF16}),
\]

输出随后按需要存为低精度。softmax normalization、variance、norm、指数和大 reduction 常保留 FP32，因为小相对误差会在这些操作中放大。

## 5. 跟着一个完整例子走到底

设某参数真实 gradient 为

\[
g=2\times10^{-9}.
\]

它小于 FP16 最小 subnormal，直接 cast 后得到 0，参数永远收不到这部分信号。

取 loss scale

\[
S=2^{15}=32768.
\]

scaled gradient 为

\[
g'=Sg
=
32768\times2\times10^{-9}
=6.5536\times10^{-5}.
\]

它已经进入 FP16 normal 范围，可以在 backward 中传播。随后转到 FP32 并 unscale：

\[
\frac{6.5536\times10^{-5}}{32768}
\approx2\times10^{-9}.
\]

原 gradient 得以恢复。

现在设 parameter \(w=1\)、gradient \(10^{-4}\)、learning rate \(10^{-3}\)。更新量为

\[
\Delta w=10^{-7}.
\]

若直接在 FP16 weight 上执行

\[
w_{\rm new}=1-10^{-7},
\]

结果仍 round 为 1，因为更新远小于 1 附近约 \(9.77\times10^{-4}\) 的间距。若 optimizer 保持 FP32 parameter 或 master copy，多个微小更新可以真实累积；forward 时再为矩阵乘生成低精度视图。

最后，若某 activation 为 \(10^5\)，FP16 已 overflow，loss scaling 对 forward overflow 无能为力，因为它只放大 loss/gradient。此时应改用 BF16、归一化输入或让该 operation 留在 FP32。一个完整 mixed-precision step 因而同时依赖 autocast policy、FP32 accumulation、loss scaling 和高精度 update。

## 6. 回到真实系统：程序实际上怎样工作

PyTorch AMP 一类系统通常让 autocast 根据 operation 选择 dtype：GEMM/conv 使用 FP16 或 BF16 输入，某些 reduction 与 numerically sensitive operations 自动升到 FP32。model parameters 可保持 FP32，低精度输入由 autocast 产生；具体 optimizer 与分布式方案也可能维护显式 master weights。

执行顺序应是：

~~~text
autocast forward
→ FP32 loss where needed
→ scale loss
→ backward creates scaled gradients
→ unscale gradients
→ finite check and gradient clipping
→ optimizer step in adequate precision
→ update dynamic scale
~~~

自定义 CUDA op 不会自动拥有正确策略。必须说明输入 dtype、accumulator dtype 和 output dtype，并对 FP16/BF16/FP32 做 reference tests。CT projector 沿 ray 累积许多正值，输入体可低精度，但 accumulator 常应 FP32；Gaussian covariance determinant、transmittance 和 opacity product 也要专门检查 range。

性能评估要确认 Tensor Core 路径真的启用，shape/alignment 合适，并记录 conversion kernels。若大量 casts 把每个小 op 包围，mixed precision 可能增加 launch 与带宽。

## 7. 容易走错的岔路

loss scaling 能修复所有 FP16 数值问题是错误的。它只移动 backward gradient 的尺度，不能修复 forward activation overflow、指数溢出或不稳定除法。

BF16 range 接近 FP32也不代表精度接近。它在 1 附近只有约三位十进制有效数字，长 reduction 和小 update 仍需 FP32。

看到 loss 正常下降也不能排除局部 gradient underflow。某些小尺度 Gaussian、稀有材料或深层参数可能已经长期为零，应记录 per-module gradient finite/zero statistics。

在 unscale 前做 gradient clipping 会把 scale 混入阈值；在检测 inf 后仍执行 optimizer step，则可能把一次 overflow 传播到所有 states。

最后，mixed precision 的显存节省不会覆盖 FP32 optimizer states。若 Adam moments 和 master parameters 主导显存，还需要 optimizer sharding、8-bit states 或其他策略。

## 8. 本章落点、验证与下一章

本章把 mixed precision 拆成了四个职责：低精度输入减少带宽并使用高吞吐硬件，FP32 accumulation 保证 reduction，loss scaling 把小 gradients 移入 FP16 可表示范围，高精度 parameters/optimizer states 累积微小更新。FP16 与 BF16 的主要差异是 precision-range tradeoff。

在 CTGS 项目中，本章直接约束 projector accumulation、Gaussian covariance、opacity compositing 和 optimizer update；在训练基础设施中，dtype policy、loss scale、overflow count 和 Tensor Core utilization 都应进入实验日志。

本章的 75 分钟验证是训练一个小网络或 Gaussian toy model，分别使用 FP32、纯 FP16、AMP-FP16 和 AMP-BF16。人工加入 \(10^5\) activation 与 \(2\times10^{-9}\) gradient 路径，记录 finite ratio、zero-gradient ratio、loss scale 和吞吐。预期是纯 FP16 同时出现 forward overflow 与小 gradient 消失；FP16 loss scaling只修复后者；BF16避免大范围 overflow，但小 parameter update 仍需要 FP32 累积。

下一章会研究编译器如何把 eager tensor program 变成更少、更大的 kernels：graph capture、shape guards、operator fusion 和 recompilation 为什么能减少 launch/中间流量，又为什么会在动态 Gaussian 数量和数据依赖控制流下 graph break。

