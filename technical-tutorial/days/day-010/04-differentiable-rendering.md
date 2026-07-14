# 可微三维渲染第 10 章：可见性为什么是渲染反传中最硬的不连续

## 1. 从一个真实任务开始

上一章把颜色拆成材质、光照和 visibility。现在问题集中到 visibility 本身：一个三角形、Gaussian 或遮挡物稍微移动，某个 pixel 可能从看见它变成完全看不见；shadow ray 也可能从无遮挡变成被挡。图像值发生跳变，但普通反向传播喜欢连续函数。

真实任务是用 silhouette loss 优化一个 mesh 或 Gaussian surface。预测轮廓比真实轮廓小一圈，希望梯度把边界向外推。标准 rasterizer 中，一个 pixel 要么被三角形覆盖，要么不覆盖；如果当前三角形边界还没覆盖到目标 silhouette 的 pixel，这些 pixel 对三角形顶点没有梯度。模型不知道应该往哪里长。

今天的任务是理解 visibility discontinuity：不连续不是 bug，而是 z-buffer、rasterization 和 ray occlusion 的数学性质。可微渲染必须选择一种近似或补充梯度，让优化能跨过边界。

## 2. 最直接的办法，以及它为什么不够

最直接的方法是对普通 rasterizer 做 autograd。三角形内部颜色对 vertex attributes、texture 和 shading 参数可微；但覆盖关系由离散测试决定。一个 pixel 的 coverage 是 0 或 1，边界移动只在跨过 pixel center 的瞬间改变结果。几乎处处梯度为零，在跨越点梯度不存在。

第二个方法是提高分辨率或开 MSAA。它让边界更细，视觉更平滑，但没有从根本上给未覆盖 pixel 提供方向。优化仍可能在 silhouette 外没有有效梯度。

第三个方法是只优化可见 surface 的颜色或材质，忽略几何 visibility。这样 photometric loss 能下降，但几何边界、遮挡和阴影位置不会被正确更新。上一章的 relighting 需要 visibility 正确，因此不能绕开它。

需要的抽象是把 hard visibility 替换、近似或补充为对边界位置有梯度的函数。不同方法在偏差、稳定性和速度上取不同折中。

## 3. 关键想法是怎样被引出来的

一个 silhouette pixel 的 hard coverage 可写成指示函数：

\[
C(x)=\mathbf 1[d(x,\triangle)\le 0],
\]

其中 \(d\) 是带符号距离。指示函数对 \(d\) 的导数在普通意义下几乎处处为零，在边界处是分布意义上的 delta。普通 autograd 不会自动给出可用的边界梯度。

软化方法把指示函数替换成连续函数，例如 sigmoid：

\[
\widetilde C(x)
=
\sigma\left(-\frac{d(x,\triangle)}{\tau}\right).
\]

\(\tau\) 控制边界软化宽度。大 \(\tau\) 给远处 pixel 也提供梯度，但 bias 大；小 \(\tau\) 更接近真实 rasterizer，但梯度范围窄。

Soft Rasterizer 把三角形对 pixel 的贡献看成概率并聚合，从而让 silhouette 对远近边界都有梯度。[Soft Rasterizer](https://arxiv.org/abs/1901.05567) DIB-R 用局部插值和全局距离聚合来处理 foreground/background 梯度。[DIB-R](https://arxiv.org/abs/1908.01210) nvdiffrast 则把 rasterization、interpolation、texture 和 antialiasing 分成模块，并通过 antialiasing 处理可见性相关梯度。[nvdiffrast](https://nvlabs.github.io/nvdiffrast/)

## 4. 一步一步建立正式模型

先看一维边界。预测区间为 \([a,b]\)，目标区间为 \([a^\star,b^\star]\)。Hard coverage 是

\[
C(x;a,b)=\mathbf 1[a\le x\le b].
\]

若目标右边界 \(b^\star>b\)，在 \(b<x<b^\star\) 的 pixels 上，预测 coverage 为 0、目标为 1。但对 \(b\) 的普通导数为零，因为这些 pixels 还没被覆盖。

用 soft right boundary 表示：

\[
\widetilde C_{\rm right}(x;b)
=
\sigma\left(\frac{b-x}{\tau}\right).
\]

它对 \(b\) 的导数为

\[
\frac{\partial \widetilde C_{\rm right}}{\partial b}
=
\frac1{\tau}
\sigma\left(\frac{b-x}{\tau}\right)
\left[
1-\sigma\left(\frac{b-x}{\tau}\right)
\right].
\]

当 \(x\) 离边界不太远时，导数非零，loss 可以推动 \(b\) 向外扩张。这个公式说明 soft visibility 的作用：它不是让前向渲染永远模糊，而是在优化时给边界附近提供连续梯度。

深度可见性还有另一个不连续。两个 surface 在同一 pixel 上比较 depth：

\[
z_1<z_2
\]

时显示 surface 1；稍微移动后若

\[
z_2<z_1,
\]

显示 surface 2。Soft z-buffer 用 softmax 近似最近深度：

\[
w_i
=
\frac{\exp(-z_i/\tau_z)}
\sum_k\exp(-z_k/\tau_z),
\qquad
C=\sum_i w_i c_i.
\]

\(\tau_z\) 越小越接近 hard z-buffer，越大则多个表面混合。训练时可以使用 soft depth 获得梯度，评估时仍用 hard rasterization 检查真实渲染质量。

## 5. 跟着一个完整例子走到底

继续一维边界。目标区间右边界为

\[
b^\star=1.0,
\]

当前预测右边界为

\[
b=0.8.
\]

取一个目标内但预测外的 pixel：

\[
x=0.9.
\]

Hard coverage 下

\[
C(x; b)=0,
\qquad
C^\star(x)=1.
\]

loss 为

\[
L=(C-C^\star)^2=1.
\]

但只要 \(b<0.9\)，\(C\) 都是 0，对 \(b\) 的梯度为 0，优化器不会把边界推向目标。

现在设

\[
\tau=0.1.
\]

Soft coverage 为

\[
\widetilde C
=
\sigma\left(\frac{0.8-0.9}{0.1}\right)
=
\sigma(-1)
\approx0.269.
\]

loss 为

\[
\widetilde L=(0.269-1)^2\approx0.534.
\]

导数为

\[
\frac{\partial \widetilde C}{\partial b}
=
\frac1{0.1}\times0.269\times0.731
\approx1.966.
\]

因此

\[
\frac{\partial \widetilde L}{\partial b}
=
2(0.269-1)\times1.966
\approx-2.875.
\]

梯度下降会增加 \(b\)，把右边界向外推。这个例子展示了 soft visibility 如何把“目标边界外没有梯度”的 hard 问题转成可优化的连续问题。

若 \(\tau=0.01\)，同一 pixel 的 sigmoid 输入为 \(-10\)，梯度几乎为零；若 \(\tau=1\)，梯度范围大，但预测会过度模糊。温度是 bias 与优化范围之间的工程旋钮。

## 6. 回到真实系统：程序实际上怎样工作

可微渲染系统通常分两层。训练 forward 可以使用 soft coverage、soft depth、analytic antialiasing 或 stochastic visibility estimator；验证和最终渲染则用真实 rasterizer 或 ray tracer。这样能避免把 soft approximation 的模糊当成最终质量。

Gaussian splatting 的 visibility 也有类似问题。Alpha compositing 比 hard z-buffer 更连续，但 densification、排序、遮挡和 depth order 仍会产生不连续或近似梯度。对表面约束和 relighting，必须检查 normal、depth 和 silhouette buffer 是否一致，而不只看 RGB loss。

对阴影和 ray visibility，路径空间中的遮挡边界更难。移动一个遮挡物会改变 shadow ray 是否命中，梯度集中在 shadow boundary。常见处理包括软阴影近似、面积光源采样、visibility smoothing、边界采样和重参数化估计。无论采用哪种方法，都要知道它引入了哪种 bias。

在 CTGS 项目中，可见性不只用于 RGB 渲染。若把 Gaussian field 提取成 surface 并用 silhouette 或 depth loss 调几何，soft visibility 能帮助边界收敛；但 CT 的 X-ray 投影本身是穿透积分，不是可见表面遮挡。不要把光学 visibility 的梯度问题错误套到 CT attenuation projection 上。

## 7. 容易走错的岔路

第一个误区是认为可微渲染就等于处处有正确梯度。Hard visibility 本来不可微；任何可用梯度都是近似、重参数化或分布意义下的处理。

第二个误区是把 soft rendering 的前向结果当成真实图像。Soft coverage 训练有用，但最终资产仍要用 hard rasterization、真实 z-buffer 或 ray tracing 评估。

第三个误区是温度固定不变。早期需要较大 \(\tau\) 获取远距离梯度，后期需要较小 \(\tau\) 减少 bias。调度常比单个温度更稳。

第四个误区是只用 silhouette loss。Silhouette 给边界梯度，但不约束内部 depth、normal 和 topology。一个错误凹凸面也可以有正确轮廓。

最后，visibility gradient 不能替代几何先验。边界梯度可能很噪，必须配合 surface smoothness、normal consistency、multi-view depth 或物理约束。

## 8. 本章落点、验证与下一章

本章解释了 visibility discontinuity 为什么让普通反传失效：coverage 和 z-buffer 是离散选择，边界外 pixel 对几何没有普通梯度。Soft coverage、soft z-buffer、analytic antialiasing 和边界梯度方法都是为了给边界附近提供可优化信号。

在可微渲染和 CTGS 可视化项目中，本章对应 silhouette loss、depth/normal supervision、Gaussian alpha sorting、soft-to-hard validation 和 relighting 中的 shadow visibility。要始终区分训练近似和最终渲染语义。

本章的 60 到 90 分钟验证是实现一维边界例子：设 \(b=0.8\)、\(b^\star=1.0\)、\(x=0.9\)，比较 hard coverage 与 \(\tau=0.1\)、\(\tau=0.01\) 的 soft coverage 梯度。预期 hard 梯度为零，\(\tau=0.1\) 能推动边界，\(\tau=0.01\) 梯度几乎消失。

下一章将进入语义和可编辑表示。可见性梯度让几何边界能被优化，但用户真正想编辑的是部件、材料和功能区域；下一步需要把连续几何表示连接到可命名、可选择、可约束的语义结构。
