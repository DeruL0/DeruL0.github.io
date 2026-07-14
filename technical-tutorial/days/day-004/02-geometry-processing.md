# 几何处理第 4 章：怎样去掉扫描噪声而不把真实棱边磨圆

## 1. 从一个真实任务开始

上一章用隐式曲率流稳定地平滑 CT 网格，也亲眼看到稳定算法仍会缩小体积和圆化特征。现在考虑一个带机械加工平面、孔口和锐利折线的零件。Marching Cubes 在每个平面上产生细碎法向噪声，但相邻平面之间的折角是真实设计特征。我们的目标是让同一平面内的法向一致，同时不让信息跨过折线传播。

这项工作与普通图像去噪相似：同一物体区域中的相邻像素应互相支持，跨越强边缘的像素不应平均。网格上要处理的主要信号不是颜色，而是 face normal 和 vertex position；连接关系又是不规则的三角网格。因此需要先在法向空间判断“哪些邻面属于同一局部表面”，再把去噪后的法向转换回一致的顶点位置。

## 2. 最直接的办法，以及它为什么不够

最直接的方法仍是 Laplacian smoothing。它根据顶点位置差移动顶点，不知道某条边是噪声还是真实 crease。只要折线处曲率大，它就会被当成应消除的高频。

另一个直接方法是只平均 face normals，不移动 vertices。这样光照会看起来更平滑，实际三角形仍然起伏；几何测量、碰撞和后续布尔运算完全没有改善。若把平均后的法向仅写入 vertex normal，甚至会用 shading 掩盖真实网格误差。

所以任务必须分成两步：先构造保特征的目标法向，再求一组顶点，使三角形尽量符合这些目标法向，同时通过 data term 限制偏离原始扫描。

## 3. 关键想法是怎样被引出来的

同一平面上的相邻 face normals 即使有噪声，夹角通常较小；真实折线两侧的 normals 差异大。于是邻面权重不应只由空间距离决定，还要由法向相似度决定。

这就是 bilateral filtering 的网格版本。空间 kernel 保证只看局部邻域，range kernel 根据 normal difference 阻止信息跨越特征。一个邻面必须同时“离得近”和“朝向相似”才有高权重。

过滤后的 normal 只是方向约束，不直接给出 vertex coordinates。第二个关键抽象是把“一个 face 的边应位于该 face 平面内”写成 least-squares energy。若目标 normal 为 \(\widehat n_f\)，face 内任意 edge \(e_{ij}=v_j-v_i\) 应满足

\[
\widehat n_f^{\mathsf T}e_{ij}=0.
\]

把所有 faces 的这类约束与原始位置保真一起求解，才能把 normal 去噪落回几何。

## 4. 一步一步建立正式模型

对 face \(f\)，中心为 \(c_f\)、面积为 \(A_f\)、单位法向为 \(n_f\)。设 \(N(f)\) 是若干环邻域。空间权重写成

\[
w_s(f,g)
=
\exp\left(
-\frac{\|c_f-c_g\|^2}{2\sigma_s^2}
\right),
\]

法向相似权重写成

\[
w_n(f,g)
=
\exp\left(
-\frac{\|n_f-n_g\|^2}{2\sigma_n^2}
\right).
\]

面积较大的 face 通常应贡献更多，因此未归一化目标为

\[
m_f
=
\sum_{g\in N(f)}
A_g w_s(f,g)w_n(f,g)n_g.
\]

最终 target normal 是

\[
\widehat n_f
=
\frac{m_f}{\|m_f\|}.
\]

\(\sigma_s\) 控制空间平滑尺度，\(\sigma_n\) 控制可跨越的角度差。对单位 normals，\(\|n_f-n_g\|^2=2-2n_f^{\mathsf T}n_g\)，所以 range weight 与夹角直接相关。

接着恢复 vertices。对每个 face 的三条 edges，构造法向偏离能量

\[
E_{\rm normal}(V)
=
\frac12
\sum_{f}A_f
\sum_{(i,j)\in E(f)}
\left[
\widehat n_f^{\mathsf T}
(v_j-v_i)
\right]^2.
\]

若目标 normals 暂时固定，这对 vertex coordinates 是二次能量。加入原始位置 \(V^0\) 的质量加权保真项：

\[
E(V)
=
E_{\rm normal}(V)
+
\frac{\lambda}{2}
\sum_i m_i\|v_i-v_i^0\|^2.
\]

令梯度为零得到稀疏线性系统。实际算法可以交替执行：根据当前几何过滤 normals，再固定 target normals 更新 vertices。每轮都检查三角形翻转和数据偏移。

这类方法仍不能凭空知道 CAD 特征。\(\sigma_n\) 只是根据现有 normal jump 推断 crease；当噪声角度与真实小折角相近时，还需要 feature-edge 标记、曲率尺度或外部设计信息。

## 5. 跟着一个完整例子走到底

考虑中心 face 的 normal

\[
n_0=(0,0,1).
\]

它有两个同距离邻面。第一个只受小噪声影响：

\[
n_1=(0,0.100,0.995),
\]

第二个位于真实直角折线另一侧：

\[
n_2=(0,1,0).
\]

忽略面积和空间权重差异，取 \(\sigma_n=0.3\)。中心与第一个 normal 的平方距离约为 \(0.01\)，所以

\[
w_n(0,1)
\approx
\exp\left(-\frac{0.01}{0.18}\right)
\approx0.946.
\]

中心与折线另一侧的平方距离为 2，因此

\[
w_n(0,2)
=
\exp\left(-\frac{2}{0.18}\right)
\approx1.5\times10^{-5}.
\]

过滤结果几乎只由 \(n_0\) 和 \(n_1\) 构成：

\[
m_0
\approx
(0,0,1)+0.946(0,0.100,0.995),
\]

归一化后

\[
\widehat n_0
\approx
(0,0.049,0.999).
\]

小法向噪声被平均，直角另一侧几乎没有穿过 range kernel。

现在取该 face 中一条噪声 edge

\[
e=(1,0,0.1).
\]

它沿目标 normal 的分量为

\[
\widehat n_0^{\mathsf T}e
\approx0.0999.
\]

若固定 edge 起点，并暂时只满足这个平面约束，最小修正是从终点减去该法向分量：

\[
e'
=
e-0.0999\widehat n_0
\approx
(1,-0.0049,0.0002).
\]

高度噪声几乎被消除，修正主要沿 normal 发生。真实全局 least-squares 会同时考虑相邻 faces 和 data term，因此不会独立移动一条 edge，但每个方程的几何含义与这个例子相同。

## 6. 回到真实系统：程序实际上怎样工作

管线应明确区分 face signal 与 vertex geometry：

```text
ValidatedMesh
→ face centers / areas / normals
→ face adjacency and feature masks
→ bilateral normal filter
→ sparse vertex reconstruction
→ flip, volume, distance and feature diagnostics
```

normal filter 可以在 CPU 邻接结构上实现，也可把 face adjacency 压缩后在 GPU 上并行。每次迭代都必须重新归一化 normals；退化 face 面积接近零时不应参与可靠方向估计。

vertex reconstruction 的矩阵通常半正定，需要 data attachment、固定点或去除刚体自由度才能唯一。对 CT 网格，data term 可以不是简单的 \(\|V-V^0\|^2\)，而是 point-to-plane distance 到原始等值面或体数据的梯度方向，这样更符合提取来源。

参数应按物理尺度记录。\(\sigma_s\) 用毫米或平均边长倍数，\(\sigma_n\) 对应可跨越角度。只写“迭代 10 次、sigma=0.3”无法在不同分辨率网格上复现。

## 7. 容易走错的岔路

range weight 很小会保护大折角，但也可能把强噪声误当成特征，使孤立尖刺无法被平均。空间尺度、法向尺度和迭代次数必须联合选择。

只过滤 vertex normals 是渲染优化，不是几何去噪。除非最终目标仅是视觉 shading，否则必须更新 positions 并重新计算真实 face normals。

把所有大 normal jump 标记成永久 feature edge 也可能锁住 Marching Cubes 台阶。真实特征应在多尺度或外部测量中保持稳定，而单尺度噪声 jump 往往会随分辨率变化。

位置重建时去掉 data term，会得到符合目标 normals 但整体漂移、缩放或塌陷的网格。normal constraints 不决定绝对位置。

最后，视觉上锐利不代表尺寸准确。保特征算法可能保护了折角方向，却移动了折线位置；必须单独测量 feature curve displacement。

## 8. 本章落点、验证与下一章

本章把“保特征平滑”拆成了两个可解释步骤。bilateral normal filter 用空间邻近和法向相似共同决定信息传播，阻止大折角两侧互相平均；vertex reconstruction 再通过 edge 与目标 normal 正交的二次能量，把过滤结果落实到坐标，同时用 data term 防止漂移。

在 CT mesh 项目中，本章对应 normal denoiser、feature mask 和 sparse vertex solver；在 STL 查看器中，应同时显示原始/过滤 normals、feature edges 和几何位移，而不只是最终 shaded surface。

本章的 75 分钟验证是复现三个 normals 的权重计算，再在一个由两个平面以 90 度相交的带噪网格上比较 implicit Laplacian smoothing 与 bilateral-normal denoising。预期是前者同时圆化折线，后者在合适 \(\sigma_n\) 下压低平面内法向方差并保持折角；把 \(\sigma_n\) 增大后，折线会开始被跨越。

下一章会进入重网格化：即使几何已经去噪，三角形仍可能细长、密度不均或与曲率尺度不匹配。edge split、collapse、flip 和 tangential relaxation 怎样在控制误差的同时改善离散质量，将成为下一问题。

