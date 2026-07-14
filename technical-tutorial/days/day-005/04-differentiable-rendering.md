# 可微三维渲染第 5 章：为什么只优化固定数量的 Gaussian 无法学出复杂场景

## 1. 从一个真实任务开始

上一章已经把一个三维 Gaussian 投影成屏幕椭圆，并让中心、尺度、旋转、opacity 和颜色都能通过图像 loss 更新。现在训练一个真实场景时，固定 Gaussian 集合会出现两种相反失败：初始化稀疏的区域没有 primitive，gradient 无法凭空创建新 Gaussian；纹理复杂区域只有一个过大的 Gaussian，它同时覆盖多个不同细节，连续参数怎么调都无法分开。

真实任务因此不仅是优化参数，还要改变表示结构。训练期间需要在欠拟合区域 clone 或 split Gaussians，在长期无贡献区域 prune，并把 optimizer state、显存预算和 rasterization cost 与这些结构变化一起管理。

这类操作称为 densification 与 pruning。它们不是普通 gradient step，而是在连续优化之间插入离散模型编辑。

## 2. 最直接的办法，以及它为什么不够

最直接的 densification 是每隔若干 steps 给每个 Gaussian 复制一个副本。覆盖能力会增长，但数量指数膨胀，大量副本位于已经拟合良好的区域，sorting、tile duplication 和 optimizer states 很快耗尽显存。

另一个直接办法是只看 position gradient：gradient 大就 split。问题是 gradient 大可能来自相机误差、曝光变化、遮挡排序或暂时噪声；一个只在单视角出现一次的大 gradient 不应立即改变结构。反过来，一个过大的 Gaussian 可能在不同 pixels 上梯度互相抵消，平均 position gradient 不大，却仍然模糊。

pruning 若只看当前 opacity 也不可靠。新 Gaussian 的 opacity 可能尚未建立，一个低 opacity Gaussian 在许多 views 上累积的总贡献仍可能重要；一个高 opacity 但只覆盖屏幕外或被完全遮挡的 Gaussian则没有价值。

所以结构决策需要时间累计、可见性归一化、尺度与贡献联合判断，并受全局 primitive budget 约束。

## 3. 关键想法是怎样被引出来的

连续参数优化和离散结构优化应工作在两个时间尺度上。每个普通 step 更新参数；每隔一段稳定窗口，系统汇总统计并执行一次结构编辑。编辑后让参数重新适应，再进行下一次统计。

clone 与 split 解决不同问题。小 Gaussian 的位置 gradient 大，通常表示附近缺少足够 primitives，可以在邻近位置 clone；大 Gaussian 的 gradient 大，表示一个 footprint 正试图解释多个细节，应沿主轴 split 成更小 children。prune 则清除低 opacity、低累计贡献、长期不可见或异常过大的 primitives。

关键不是某个固定阈值，而是每项统计对应一个可解释问题：

- gradient：现有参数是否持续想向不同位置移动；
- scale：一个 primitive 是否承担过大空间范围；
- visibility：统计是否来自足够多的有效观测；
- contribution：它是否实际影响 loss 所见的 pixels；
- budget：新增表示能力是否值得增加渲染与优化成本。

## 4. 一步一步建立正式模型

对 Gaussian \(i\)，在训练窗口内记录它投影中心 \(m_{i,t}\) 的 screen-space gradient。用可见 indicator \(v_{i,t}\) 归一化：

\[
g_i
=
\frac{
\sum_t v_{i,t}
\left\|
\frac{\partial L_t}{\partial m_{i,t}}
\right\|_2
}{
\sum_t v_{i,t}+\epsilon
}.
\]

这样频繁可见的 Gaussian 不会仅因累计次数多而自动超过阈值。还可记录最大 projected radius、world-space principal scale 和累计 alpha contribution：

\[
c_i
=
\sum_{t,p}
T_{i,t,p}\alpha_{i,t,p}.
\]

一个简化决策为：

\[
g_i>\tau_g,\quad s_i\le\tau_s
\quad\Rightarrow\quad
\text{clone},
\]

\[
g_i>\tau_g,\quad s_i>\tau_s
\quad\Rightarrow\quad
\text{split}.
\]

split 时对 covariance 做 eigendecomposition 或直接使用参数化 rotation 的主轴。设最大标准差方向为 \(e_1\)、尺度为 \(s_1\)，children centers 可初始化为

\[
\mu_\pm
=
\mu\pm\delta s_1e_1.
\]

children covariance 在主轴上缩小，其他方向可保持或一并缩放。结构编辑应尽量保持初始渲染连续。若两个 children 暂时重合并沿同一 ray compositing，希望合成 opacity 等于 parent \(o\)，可令

\[
1-(1-o_c)^2=o,
\]

所以

\[
o_c=1-\sqrt{1-o}.
\]

实际 children 被分开后，这只是局部近似；还需通过后续训练调节。

pruning 可结合条件：

\[
o_i<\tau_o
\quad\land\quad
c_i<\tau_c,
\]

或长期不可见、屏幕 footprint 异常大、world scale 超出场景范围。使用 AND 而不是单一 opacity threshold，可以减少误删刚初始化但有潜在贡献的 primitives。

结构变化还涉及 optimizer state。Adam 为每个参数保存一、二阶 moments。clone/split 后必须定义 children 是继承、缩放还是重置 moments；prune 时同步删除所有 parameter groups 与 states。否则 tensor 索引错位会产生隐蔽错误。

## 5. 跟着一个完整例子走到底

设一个 Gaussian 在 100 个有效 views 的归一化 screen gradient 为

\[
g=0.8,
\]

阈值为 \(\tau_g=0.2\)。其最大主轴标准差为

\[
s_1=0.12,
\]

而 split scale threshold 为 \(\tau_s=0.08\)。它既高 gradient 又过大，因此选择 split，而不是 clone。

令主轴为 \(e_1=(1,0,0)\)，取 \(\delta=0.5\)。两个 children centers 为

\[
\mu_\pm
=
\mu\pm0.06e_1.
\]

把主轴尺度除以 1.6：

\[
s_{1,\rm child}
=
\frac{0.12}{1.6}
=0.075.
\]

parent opacity 为 \(o=0.6\)。若希望 children 初始重合时合成 opacity 近似不变，则

\[
o_c
=
1-\sqrt{0.4}
\approx0.3675.
\]

因此初始结构从一个宽 footprint 变成两个中心分离、主轴更窄的 footprints，二者有能力分别靠向两个图像细节。若直接给每个 child opacity 0.6，重合处合成 opacity 会变成

\[
1-(1-0.6)^2=0.84,
\]

结构编辑瞬间就会让画面过度不透明。

训练 500 steps 后，假设正向 child 的累计贡献保持 0.12，负向 child 因被遮挡而 opacity 降到 0.01、归一化贡献降到 \(5\times10^{-4}\)。若 \(\tau_o=0.02\)、\(\tau_c=10^{-3}\)，后者同时满足两个 prune 条件，被删除；前者保留。这个例子从统计、split 初始化、opacity 守恒到后续 prune 走完了一次结构周期。

对 CTGS，若 Gaussian 表示归一化衰减密度并在投影中线性积分，更自然的守恒量不是 RGB opacity，而是总 attenuation mass。parent 权重 \(a\) split 后可先令两个 children 各为 \(a/2\)，使全空间积分严格保持，再根据投影 loss 优化。

## 6. 回到真实系统：程序实际上怎样工作

训练循环应明确区分统计窗口和编辑点：

~~~text
ordinary optimization steps
→ accumulate visibility, gradients, radius and contribution
→ synchronize statistics
→ select clone / split / prune candidates under budget
→ rebuild parameter tensors and optimizer states
→ reset or decay statistics
→ stabilization steps before next edit
~~~

所有统计必须与 Gaussian stable ID 对齐，不能只依赖会在 prune 后改变的 tensor index。结构编辑后 tile buffers、visibility arrays 和 optimizer state shapes 都要重建；分布式训练还需要所有 ranks 对同一结构达成一致。

预算控制可按 projected cost 而不只是 Gaussian count。一个覆盖 200 tiles 的大 Gaussian 可能比几十个小 Gaussian 更贵；记录 tile duplication、average contributors/pixel 和 early-termination depth，才能把质量决策与 renderer cost 对应。

原始 3D Gaussian Splatting 的 adaptive density control 使用位置梯度、尺度与 opacity 执行 clone/split/prune，证明了结构优化是表示能力的重要组成。[原始项目](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) 后续系统会使用误差图、层次结构或预算感知策略，但仍需回答本章的守恒与状态迁移问题。

## 7. 容易走错的岔路

gradient threshold 越低不等于质量越高。它会让噪声、相机误差和曝光残差不断生成 Gaussians，训练 loss 下降但几何和显存失控。

split 时复制 parent opacity 会造成局部密度突然增加；复制 Adam moments 也可能让 children 带着不合适的历史速度向同一方向移动。结构编辑必须定义守恒量与 state policy。

只 prune 低 opacity 会保留大量高 opacity、长期不可见的 primitives；只按 contribution prune 又可能删除只在少数关键视角可见的结构。统计窗口和任务权重需要匹配数据分布。

频繁每 step densify 看起来响应更快，却让 optimizer 永远在变化的参数空间中追赶。结构更新应比连续优化慢，并留稳定期。

最后，RGB 表示中的 opacity 守恒不能直接用于 CTGS。X 射线负对数投影线性累加衰减质量，深度合成规则不同；错误守恒会在 split 时改变物理投影。

## 8. 本章落点、验证与下一章

本章把 Gaussian 训练从固定参数优化扩展为结构与参数交替优化。可见性归一化 gradient 判断表示压力，scale 区分 clone 与 split，opacity/contribution 共同决定 prune；结构编辑还必须处理 opacity或衰减质量守恒、optimizer state 迁移和全局预算。

在 3DGS 项目中，本章对应 density-control scheduler、statistics buffers 和 parameter-state rebuild。在 CTGS 中，应把 RGB opacity 规则替换为 attenuation-mass conservation，并分别报告 Gaussian count、投影误差和 raster/projector cost。

本章的 90 分钟验证是实现单 Gaussian split toy：用正文的 \(o=0.6\) 比较“复制 opacity 0.6”和“child opacity 0.3675”的重合合成结果，再把 centers 分开观察 footprint。随后在小场景记录每次 densification 前后的 loss jump、Gaussian count、tile duplicates 和 optimizer-state shapes。预期是守恒初始化显著减小结构编辑瞬间的图像跳变。

下一章会处理 Gaussian 数量增长后的部署问题：screen-space 抗锯齿、层次 LOD 与压缩怎样控制远近尺度，为什么简单缩小 Gaussian 会产生闪烁，以及 rate-distortion 视角如何在图像质量、显存和帧时间之间选择表示。

