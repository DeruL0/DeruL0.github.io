# 可微三维渲染第 2 章：一个轮廓移动时，梯度为什么会在像素之间消失

## 1. 从一个真实任务开始

上一章把渲染器写成 \(I=R(\theta)\)，并指出矩阵变换、着色和连续体积分通常可以沿计算图反向传播，真正困难的是可见性。现在考虑一个具体任务：目标图像中有一个零件轮廓，当前 mesh 渲染出的轮廓偏左，我们希望通过 silhouette loss 调整顶点或相机，使两条轮廓重合。

输入是目标 mask、当前三角网格和相机；输出是顶点位置或相机参数的更新。这个任务看起来比材质逆渲染更简单，因为 mask 只有前景和背景，没有复杂光照。但恰恰是在二值轮廓上，传统 rasterization 的离散性表现得最清楚：顶点连续移动，像素覆盖却只在轮廓跨过 pixel sample 时改变。

今天要建立的不是某个 soft rasterizer 的使用说明，而是可见性梯度的来源。只有先知道真实导数缺了哪一项，才能理解 soft rasterization、edge sampling 和 Gaussian footprint 分别近似了什么。

## 2. 最直接的办法，以及它为什么不够

最直接的实现是用普通 rasterizer 生成 hard mask，然后让 autograd 对顶点坐标反向传播。三角形内部的 barycentric interpolation 对顶点和属性有明确关系，因此我们很容易期待覆盖 mask 也能自动求导。

但设轮廓位于屏幕坐标 \(a=0.40\)，相邻 pixel centers 在 \(0.35\) 和 \(0.45\)。只要 \(a\) 在这两个采样点之间移动而没有越过任何一个中心，前景像素集合完全不变，离散 loss 也是常数。autograd 看到的是一个由比较操作生成的固定 mask，于是

\[
\frac{\partial L}{\partial a}=0.
\]

当轮廓最终跨过 \(0.45\) 时，一个像素突然翻转，loss 跳变。梯度不是平滑地指向目标，而是长时间为零、在跳变点又没有普通意义下的有限导数。

加大有限差分步长似乎能看见变化：让轮廓一次跨过几个像素，比较两次 loss。但这个结果依赖步长和分辨率；它把一段区间内所有变化压成平均值，也无法为百万顶点高效计算。因此困难不在反向传播框架，而在 hard visibility 本身不是普通的逐像素光滑函数。

## 3. 关键想法是怎样被引出来的

离散像素让我们误以为“轮廓处只有一条线，面积为零，所以它对图像积分没有贡献”。实际情况相反：当轮廓移动一个很小距离 \(da\) 时，它会扫过一条宽度为 \(da\) 的窄带。窄带面积是一阶小量，正好构成 loss 的一阶变化。

因此，对一个随参数移动的可见区域求导，除了区域内部颜色和材质的普通导数，还必须包含边界运动项。边界本身虽是低维集合，它的法向速度乘上前景与背景的损失差，会产生有限梯度。

这给出三种有明确含义的路线。第一种把硬阶跃替换成有宽度的连续过渡，使普通自动微分能在轮廓附近看到梯度；这就是 soft rasterization。第二种保留 hard forward image，专门在几何边界上积分或采样缺失的边界项；edge sampling 属于这一类。第三种改变表示，让 primitive 本来就对邻域有连续 footprint，例如体密度或 Gaussian splatting。三者不是互相替代的 API，而是对同一个边界导数采取了不同近似。

## 4. 一步一步建立正式模型

先用一维轮廓把问题写清。令位置 \(x<a\) 为前景，位置 \(x>a\) 为背景。前景颜色为 \(c_f\)，背景颜色为 \(c_b\)，则

\[
I(x;a)
=
H(a-x)c_f
+
[1-H(a-x)]c_b,
\]

其中 \(H\) 是 Heaviside step function。若对连续图像域上的损失积分：

\[
L(a)
=
\int
\ell(I(x;a),I^\ast(x))\,dx,
\]

当 \(a\) 增加 \(da\) 时，区间 \([a,a+da]\) 从背景变成前景。因此不用形式化 delta function，也能直接得到边界导数：

\[
\frac{dL}{da}
=
\ell(c_f,I^\ast(a))
-
\ell(c_b,I^\ast(a)).
\]

这个式子说明梯度只需要问：在当前边界处，把一小条背景换成前景会让损失增加还是减少？它正是 hard rasterization 的逐像素 autograd 没有表示出来的项。

二维情况下，可见区域记为 \(\Omega(\theta)\)，边界为 \(\partial\Omega\)。参数变化使边界具有法向速度 \(v_n\)。损失导数包含

\[
\frac{dL}{d\theta}
=
\int_{\Omega}
\frac{\partial \ell}{\partial\theta}\,dA
+
\int_{\partial\Omega}
(\ell_f-\ell_b)
v_n\,ds.
\]

第一项是可见区域内部的普通 shading derivative，第二项是轮廓或遮挡边界移动造成的 visibility derivative。即使第一项为零，第二项仍可推动几何。

soft rasterization 通过把 \(H\) 换成 sigmoid 近似：

\[
H_\tau(a-x)
=
\sigma\!\left(\frac{a-x}{\tau}\right).
\]

它的导数为

\[
\frac{\partial H_\tau}{\partial a}
=
\frac{1}{\tau}
\sigma\!\left(\frac{a-x}{\tau}\right)
\left[
1-
\sigma\!\left(\frac{a-x}{\tau}\right)
\right].
\]

\(\tau\) 决定梯度支持域。较大 \(\tau\) 让远离当前轮廓的像素也能提供方向，但前向轮廓变软、梯度偏差更大；较小 \(\tau\) 更接近 hard edge，却只在极窄区域有梯度，低分辨率下又容易错过 sample。这是 bias 与 optimization range 的权衡，不存在“永远越小越准确”的单调结论。

edge sampling 则不修改 \(H\)。它寻找投影后的三角形边和遮挡边界，在边界两侧评估 radiance 或 loss jump，再乘边界对参数的法向运动，估计上式第二个积分。前向图像可以保持硬而准确，代价是必须可靠识别和采样可见边界，Monte Carlo 版本还会带来方差。

## 5. 跟着一个完整例子走到底

在区间 \([0,1]\) 上，目标前景为 \(x<0.70\)，当前前景为 \(x<0.40\)。使用每点 \(\tfrac12(I-I^\ast)^2\) 的连续积分。两者不一致的区间是 \([0.40,0.70]\)，其中当前为背景 0、目标为前景 1，所以

\[
L(0.40)
=
\int_{0.40}^{0.70}
\frac12(0-1)^2\,dx
=0.15.
\]

当前边界 \(a=0.40\) 处，若向右移动一小段，错误背景会被正确前景替换。边界公式给出

\[
\frac{dL}{da}
=
\frac12(1-1)^2
-
\frac12(0-1)^2
=-0.5.
\]

取学习率 \(\eta=0.2\)，梯度下降更新为

\[
a_{\rm new}
=
0.40-0.2(-0.5)
=0.50.
\]

新错误区间长度为 \(0.20\)，所以新损失为 \(0.10\)。边界梯度给出了正确方向，也准确描述了移动一小段时损失下降的速率。

现在只在 pixel centers \(0.1,0.3,0.5,0.7,0.9\) 采样 hard mask。\(a=0.40\) 附近无论移动 \(0.01\) 还是 \(0.05\)，只要不越过 \(0.5\)，覆盖集合不变，离散 loss 的局部梯度为零。连续图像有明确的 \(-0.5\) 导数，point-sampled raster 却把边界扫过的面积漏掉了。

若改用 \(\tau=0.05\) 的 sigmoid，在 \(x=0.5\) 处

\[
H_\tau(0.40-0.50)
=
\sigma(-2)
\approx0.119.
\]

它对边界位置的导数约为

\[
\frac{0.119(1-0.119)}{0.05}
\approx2.10.
\]

因为目标在该点为前景，平方损失的该点梯度为 \((0.119-1)\times2.10<0\)，会把 \(a\) 向右推。这个数值不等于连续积分的 \(-0.5\)，因为我们现在使用有限像素和带温度的近似；但它恢复了可用于优化的方向。随着像素积分、采样密度和 \(\tau\) 合理配合，soft transition 会近似边界 delta 的作用。

## 6. 回到真实系统：程序实际上怎样工作

可微 mesh renderer 应明确区分两条反向路径。三角形内部的属性插值、纹理和 shading 走常规链式法则；silhouette 与 occlusion boundary 需要 antialias/soft coverage 或显式 edge estimator。若只对 rasterizer 输出的 barycentric coordinates 求导，内部属性可能有梯度，轮廓几何仍然不动。

一个可检查的执行结构是：

```text
vertices and camera
→ projection and triangle setup
→ hard or soft coverage
→ shading / compositing
→ image and silhouette losses
→ interior gradient + visibility gradient
→ accumulate to vertices and camera
```

NVIDIA 的 nvdiffrast 把 point-sampled rasterization 与后续 antialiasing 分开；其 antialiasing 阶段利用轮廓信息产生与可见性有关的顶点梯度。这种接口正好体现了“内部插值导数”和“边界导数”不是同一件事。[nvdiffrast documentation](https://nvlabs.github.io/nvdiffrast/)

对 3D Gaussian Splatting，二维 Gaussian footprint 本身就是连续 coverage，所以中心、尺度和协方差在局部能从邻近 pixels 得到梯度。但 tile culling、深度排序交换和 pruning 仍是离散事件。opacity 过低或 Gaussian 过窄时，梯度支持域仍会消失；过宽则用外观模糊换取优化范围。

对 CTGS 还要做一个重要区分。理想负对数 CT 的测量是沿射线累加衰减，前后物体都贡献，不存在 RGB 渲染中的“最近表面遮住后面”这一硬可见性。若模型在 log domain 对 Gaussian line integral 做加和，深度排序可能根本不应进入物理前向。把 RGB alpha compositing 的 visibility 规则直接搬到 CT，会人为引入排序不连续和错误的遮挡关系。

## 7. 容易走错的岔路

把 \(\tau\) 调得越来越小看似能逼近真实硬边界，但优化梯度会集中到越来越窄的区域。当前轮廓离目标较远时，所有 target pixels 可能落在支持域之外，训练反而停住。实践中常使用 coarse-to-fine 或温度退火：先扩大捕获范围，再逐步提高边界精度。

有限差分与自动微分不一致也不总是 backward 写错。在可见性跳变点，普通导数可能不存在；不同 \(\varepsilon\) 跨过不同数量的像素，会得到不同平均斜率。梯度检查应同时避开离散事件测试连续路径，并为边界项设计专门的积分或统计验证。

“hard forward、soft backward”可以给出有用优化方向，但它优化的不是严格同一个函数。这个 straight-through 风格的偏差必须被明确记录，不能因为 loss 下降就把梯度称为精确导数。

只用 silhouette loss 也无法唯一恢复三维。相机焦距、物体尺度和深度可以互相补偿，多种三维形状能产生同一轮廓。可见性梯度解决的是“怎样移动”，不是“答案是否唯一”。

最后，不能把 RGB 的遮挡问题原样套到 CT。CT 物理中的所有沿线衰减贡献与表面渲染中的最近可见点是不同 forward model；错误的可见性抽象会让一个数值上平滑的 renderer 优化错误目标。

## 8. 本章落点、验证与下一章

本章解释了 hard rasterization 为什么让轮廓梯度在像素之间消失。连续可见区域的导数包含边界运动项：前景与背景的 loss jump 乘以边界法向速度。point-sampled mask 漏掉了这项；soft rasterization 用有限宽度近似边界 delta，edge sampling 直接估计边界积分，Gaussian 和 volume representation 则通过连续 footprint 改变问题形态。

在 mesh inverse rendering 中，这一章对应 silhouette antialiasing 和 edge gradient；在 3DGS 中，它对应 Gaussian 尺度、tile 边界和排序事件；在 CTGS 中，它还帮助删除不属于 X 射线物理的 RGB visibility 规则。

本章的 60 分钟验证是实现正文的一维区间例子。把边界 \(a\) 从 0 到 1 扫描，分别绘制 hard point-sampled loss、continuous analytic loss 和三种 \(\tau\) 的 soft loss及其梯度。预期结果是：hard 离散曲线呈台阶并在台阶间梯度为零；continuous loss 在目标附近分段线性；soft loss 消除台阶，较大 \(\tau\) 支持域更宽但边界偏差更大，较小 \(\tau\) 更尖锐却更容易漏过像素。

下一章会沿“改变表示”这条路线继续：体渲染的 transmittance 怎样把沿线密度变成像素，梯度为什么会因早期不透明而饱和，以及 Gaussian/NeRF 的连续积分究竟解决了哪些可见性问题、又引入了哪些新的不可辨识性。

