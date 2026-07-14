# 可微三维渲染第 3 章：体渲染为什么连续，却仍然会失去梯度

## 1. 从一个真实任务开始

上一章看到 hard surface visibility 会在轮廓跨过 pixel 时跳变，因此一种自然路线是把表面改写成连续体密度。NeRF、体素辐射场和许多 Gaussian 方法都让一条相机射线上的多个位置连续贡献颜色，从而避免“只取最近三角形”的硬选择。

现在的任务是用多视角图像训练一个体表示。输入为射线、目标像素以及沿射线采样得到的密度和颜色；输出为渲染颜色和对密度、颜色、位置参数的梯度。体渲染公式是连续可导的，但训练仍会出现一种新失败：靠近相机的样本变得过度不透明后，后面的样本几乎收不到梯度。

今天要把这种 saturation 从公式中直接推出来，并区分 RGB 体合成与 X 射线衰减模型，避免在 CTGS 中错误复用可见光 compositing。

## 2. 最直接的办法，以及它为什么不够

最直接的体表示是把射线上所有样本颜色做加权平均。若某条 ray 经过红色表面后又经过蓝色背景，简单平均会让两个物体同时透出，无法表达前方物体遮挡后方物体。

另一个直接方法是选择密度最大的 sample 作为表面。这又退回 hard argmax：最大值索引改变时图像跳变，其他 sample 没有梯度。

我们需要一种连续机制，同时表达“前方密度会减少后方可见性”。物理上，光在介质中传播时，尚未被吸收的比例随累计密度指数下降。这个剩余比例就是 transmittance。它让每个 sample 的颜色贡献既取决于自身密度，也取决于前面所有 sample 已经挡掉多少光。

## 3. 关键想法是怎样被引出来的

把射线切成许多小段。第 \(i\) 段密度为 \(\sigma_i\)，长度为 \(\Delta_i\)。这一段吸收光的概率写成

\[
\alpha_i=1-e^{-\sigma_i\Delta_i}.
\]

但第 \(i\) 段只有在光穿过前面所有区段后才可见。到达它之前的 transmittance 为

\[
T_i
=
\prod_{j<i}(1-\alpha_j)
=
\exp\left(-\sum_{j<i}\sigma_j\Delta_j\right).
\]

所以实际颜色权重不是单独的 \(\alpha_i\)，而是 \(T_i\alpha_i\)。这套乘性结构连续地表达了 visibility：没有离散最近面，却仍然让前方样本控制后方贡献。

问题也从同一结构出现。若前方累计密度很大，\(T_i\) 接近零，后方颜色、密度和位置的所有梯度都会一起变小。体渲染消除了硬跳变，却用指数衰减换来 gradient starvation。

## 4. 一步一步建立正式模型

离散射线颜色写成

\[
C
=
\sum_{i=1}^{N}T_i\alpha_i c_i
+T_{N+1}c_{\rm bg},
\]

其中

\[
T_{N+1}=\prod_{j=1}^{N}(1-\alpha_j)
\]

是穿过所有 samples 后到达背景的比例。所有权重之和为 1，因此这是一个有顺序的前向合成。

颜色导数最直接：

\[
\frac{\partial C}{\partial c_i}
=T_i\alpha_i.
\]

一个 sample 若自身几乎透明或前方已经不透明，它的颜色都难以学习。

密度导数还包含它对后方遮挡的影响。把第 \(k\) 个 sample 之后的颜色，按“已经到达 \(k+1\)”为条件归一化记为

\[
R_{k+1}
=
\sum_{i>k}
\left[
\prod_{k<j<i}(1-\alpha_j)
\right]
\alpha_i c_i
+
\left[
\prod_{j>k}(1-\alpha_j)
\right]c_{\rm bg}.
\]

于是从第 \(k\) 段开始的局部颜色为

\[
\alpha_k c_k+(1-\alpha_k)R_{k+1}.
\]

因为

\[
\frac{\partial\alpha_k}{\partial\sigma_k}
=
\Delta_k(1-\alpha_k),
\]

可以得到

\[
\frac{\partial C}{\partial\sigma_k}
=
T_k\Delta_k(1-\alpha_k)
(c_k-R_{k+1}).
\]

这个式子给出三个梯度门。\(T_k\) 要求前方仍有光，\(1-\alpha_k\) 要求当前段没有完全饱和，\(c_k-R_{k+1}\) 要求当前颜色与被替代的后方颜色不同。若当前 sample 与后方颜色相同，改变遮挡也不会改变像素。

对像素损失 \(L(C,C^\ast)\)，再乘

\[
\frac{\partial L}{\partial\sigma_k}
=
\frac{\partial L}{\partial C}
\frac{\partial C}{\partial\sigma_k}.
\]

因此 loss 有误差不代表每个 sample 都能得到有效梯度；渲染 Jacobian 可能已经把它压到接近零。

## 5. 跟着一个完整例子走到底

一条射线有两个 samples，\(\Delta_1=\Delta_2=1\)。设

\[
\sigma_1=2,
\quad c_1=0.2,
\qquad
\sigma_2=0.5,
\quad c_2=0.9,
\]

背景为 0。两段 opacity 为

\[
\alpha_1=1-e^{-2}\approx0.8647,
\]

\[
\alpha_2=1-e^{-0.5}\approx0.3935.
\]

第二段之前的 transmittance 是

\[
T_2=1-\alpha_1=e^{-2}\approx0.1353.
\]

所以渲染颜色为

\[
C
=
1\times0.8647\times0.2
+0.1353\times0.3935\times0.9
\approx0.2209.
\]

第二段之后只有黑背景，因此

\[
R_2=\alpha_2c_2\approx0.3541.
\]

第一段密度导数为

\[
\frac{\partial C}{\partial\sigma_1}
=
1\times1\times e^{-2}
(0.2-0.3541)
\approx-0.0209.
\]

它为负，因为增加前方的暗灰密度，会遮住后方更亮的颜色。第二段密度导数为

\[
\frac{\partial C}{\partial\sigma_2}
=
0.1353\times e^{-0.5}\times(0.9-0)
\approx0.0739.
\]

现在把第一段密度从 2 增到 8。第二段前的 transmittance 变为

\[
e^{-8}\approx0.000335.
\]

无论第二段颜色或密度多错，它的导数都会再缩小约 400 倍。连续公式没有不连续，却形成了几乎不可穿透的梯度屏障。

## 6. 回到真实系统：程序实际上怎样工作

前向实现通常逐 ray 保持累计 transmittance：

```text
T = 1
C = 0
for samples front-to-back:
    alpha = 1 - exp(-sigma * delta)
    weight = T * alpha
    C += weight * color
    T *= 1 - alpha
    optionally stop when T is very small
C += T * background
```

backward 需要保存或重算 \(T_i\)、\(\alpha_i\) 及后方累积颜色。early termination 能省计算，但阈值之后的 samples 被完全裁掉，梯度也变为零；训练和推理可以使用不同阈值，或对阈值做误差分析。

NeRF 常通过分层采样把更多 samples 放到高权重区域，但若粗网络过早形成错误不透明层，细网络仍可能看不到后方。常见缓解包括密度初始化、coarse-to-fine positional encoding、占用网格更新节奏和额外 sparsity/opacity regularization；这些方法都在控制 transmittance 的形成过程。

对 CTGS，必须区分两种测量。可见光相机的前向颜色需要有序 transmittance；理想 CT 负对数投影则是

\[
p=\int\mu(t)\,dt
\approx\sum_i\mu_i\Delta_i,
\]

所有位置线性相加，不应使用 front-to-back alpha occlusion。若直接拟合原始强度，则

\[
I=I_0e^{-p},
\]

会因高衰减产生小强度和噪声问题，但这仍不同于 RGB 颜色遮挡。正确物理模型决定正确梯度结构。

## 7. 容易走错的岔路

“体渲染可导，所以不会有梯度问题”忽略了 Jacobian 的数值尺度。导数存在与导数足够大是两回事，指数 transmittance 很容易让后方梯度低于浮点和优化噪声。

把所有初始密度设得很大看起来能快速形成实体，实际可能让错误前层遮住真实结构。较低初始 opacity 往往更利于多视角共同塑形，但太低又会使颜色和几何信号弱。

early termination 的阈值只影响性能这一说法也不完整。训练时一旦停止遍历，后方参数就没有梯度；阈值是优化模型的一部分。

另一个错误是把 \(\alpha\) 当作与采样间距无关的参数。\(\alpha=1-e^{-\sigma\Delta}\)，改变 sample spacing 而不调整密度含义，会改变成像结果。

最后，CT 的指数透射与 RGB alpha compositing 都含指数和乘积，不代表物理相同。CT detector 测总衰减后的光子强度，不会把某个位置的“颜色”按可见性选出来。

## 8. 本章落点、验证与下一章

本章建立了离散体渲染的完整梯度链。\(T_i\alpha_i\) 连续表达前向可见性；密度导数由前方 transmittance、当前剩余透射以及当前与后方颜色差共同决定。体表示消除了 hard surface 的跳变，却会在高 opacity 前层之后产生 gradient starvation。

在 NeRF 与可微体渲染中，本章对应 ray marcher、compositor 和 density initialization。在 CTGS 中，它给出一个明确判断：RGB 训练需要有序 transmittance，负对数 CT 前向应使用沿线加和，不能因代码复用而混淆。

本章的 60 分钟验证是实现两 sample 例子，用 autograd 与正文解析式比较 \(\partial C/\partial\sigma_1\) 和 \(\partial C/\partial\sigma_2\)。把 \(\sigma_1\) 从 0 扫到 10，绘制第二段梯度。预期曲线按前方 transmittance 指数下降；关闭 early termination 时梯度虽小但非零，开启阈值后会在某点突然变为零。

下一章会把连续体表示进一步离散成可实时光栅化的 3D Gaussian：三维协方差怎样投影成屏幕椭圆，Gaussian footprint 怎样提供位置和尺度梯度，以及 alpha compositing、排序与 tile binning 在哪里重新引入离散事件。

