# 几何处理第 7 章：为什么任何 UV 展开都必须决定牺牲哪一种长度

## 1. 从一个真实任务开始

上一章把 CT mesh 简化到可交互规模。现在我们希望给表面绘制材料标签、缺陷热图、照片纹理，或者让工程师在二维界面中编辑一块曲面。GPU texture 是二维数组，因此必须为每个 surface vertex 或 corner 分配二维坐标 \((u,v)\)。

真实任务是把三维三角曲面切成若干 charts，分别铺到平面，再打包成 UV atlas。输出不能有大面积重叠或翻转，还要控制纹理拉伸。闭合球面、带孔零件和高 genus 表面无法在不切开的情况下整体一一映射到平面，因此 seam 设计本来就是问题的一部分。

## 2. 最直接的办法，以及它为什么不够

最直接的方法是丢掉一个坐标，例如使用 \((x,y)\) 作为 UV。对接近水平的平面有效；竖直 faces 会被压成线，前后表面还会重叠。

另一种方法是把每个 triangle 独立展开。每个三角形都能无失真放到平面，但相邻 faces 不再共享 UV edge，texture 在每条边都断开，atlas 还需要大量 padding。

所以 parameterization 必须在两个目标之间平衡：同一 chart 内保持邻接连续，同时允许少量 seams 解除拓扑阻碍与过大失真。没有一种映射能对一般曲面同时保持所有长度、角度和面积。

## 3. 关键想法是怎样被引出来的

把三维 surface chart 映射到二维，局部 Jacobian \(J\) 描述两个切向方向怎样被拉伸。它的 singular values

\[
\sigma_1\ge\sigma_2>0
\]

是局部最大和最小 stretch。

若

\[
\sigma_1=\sigma_2,
\]

局部角度保持，映射 conformal，但面积可能统一缩放。若

\[
\sigma_1\sigma_2=1,
\]

局部面积保持，但两个方向可以不同程度拉伸。一般曲面无法让两者处处同时满足。

parameterization algorithm 的本质就是选择 distortion energy、boundary/seam constraints 与 injectivity条件。术语“UV 展开”背后实际是一个离散 PDE 或优化问题。

## 4. 一步一步建立正式模型

先考虑拓扑为 disk 的 triangle mesh。每个 vertex 有二维未知量

\[
u_i=(u_i,v_i)^{\mathsf T}.
\]

最简单的 harmonic parameterization 最小化 Dirichlet energy：

\[
E(u)
=
\frac12
\sum_{(i,j)\in E}
w_{ij}\|u_i-u_j\|_2^2.
\]

使用 cotangent weights：

\[
w_{ij}
=
\frac12
(\cot\alpha_{ij}+\cot\beta_{ij}).
\]

固定 boundary UV 后，内部 vertex 满足离散 Laplace equation：

\[
\sum_{j\in N(i)}
w_{ij}(u_i-u_j)=0.
\]

写成矩阵分块：

\[
L_{II}u_I
=
-L_{IB}u_B.
\]

对 \(u\) 和 \(v\) 两个坐标分别解同一个 sparse system。这个方程表示每个内部 UV 是邻居的加权平衡点。

若 boundary 映射到 convex polygon 且使用非负 barycentric weights，可以获得不翻转的经典保证；cotangent weights 在钝角三角形上可能为负，因此实际仍需检查 signed UV triangle area：

\[
A_f^{UV}
=
\frac12
\det
\begin{bmatrix}
u_j-u_i & u_k-u_i
\end{bmatrix}.
\]

若 \(A_f^{UV}\le0\)，face 翻转或退化。

harmonic energy 不直接保证 conformal。LSCM 一类方法最小化 Cauchy-Riemann 残差，优先保持角度；ARAP parameterization 则交替估计每个 triangle 的最佳局部 rotation 与全局 UV。不同方法对应不同 distortion preference。

对闭合或高 genus mesh，先建立 cut graph，把表面切成 disk-like charts。atlas 再对 charts 做 rotate、scale 和 packing，并为 texture filtering 留 padding。seam 越少，单 chart distortion 可能越高；chart 越碎，连续性和 packing efficiency 越差。

## 5. 跟着一个完整例子走到底

继续使用早期章节的五顶点凸起网格。中心为

\[
v_0=(0,0,0.5),
\]

boundary vertices 为

\[
v_1=(1,0,0),\quad
v_2=(0,1,0),\quad
v_3=(-1,0,0),\quad
v_4=(0,-1,0).
\]

四个 faces 围绕中心。把 boundary 固定到同样的二维 diamond：

\[
u_1=(1,0),\quad
u_2=(0,1),\quad
u_3=(-1,0),\quad
u_4=(0,-1).
\]

由于几何和权重对称，中心 harmonic equation 给出

\[
u_0
=
\frac{\sum_{j=1}^{4}w_{0j}u_j}
{\sum_{j=1}^{4}w_{0j}}
=(0,0).
\]

所以四个 UV triangles 都有正面积

\[
A_f^{UV}=0.5,
\]

atlas 总面积为 2。

三维中每个 triangle 面积在前面算过：

\[
A_f^{3D}
=
\frac{\sqrt{1.5}}{2}
\approx0.6124.
\]

总三维面积约 2.449，映射总面积为 2，整体面积比例约

\[
\frac{2}{2.449}
\approx0.8165.
\]

角度也发生变化。中心处指向 \(v_1,v_2\) 的三维 edges dot product 为 0.25，长度都为 \(\sqrt{1.25}\)，所以夹角为

\[
\arccos(0.2)
\approx78.46^\circ.
\]

UV 中对应两条 edges 正交，夹角是 \(90^\circ\)。这个极小例子说明：即使映射对称、无翻转且视觉合理，面积和角度仍不可能都原样保留。

若把这个 patch 的 boundary 映射成长宽比极端的矩形，内部 harmonic 解仍存在，但 singular-value ratio 会增大，texture 被明显拉伸。boundary choice 本身就是 distortion 的来源。

## 6. 回到真实系统：程序实际上怎样工作

完整 UV pipeline 通常为：

~~~text
validated oriented mesh
→ detect boundaries, features and material seams
→ choose cut graph / chart segmentation
→ parameterize each disk-like chart
→ detect flips and measure angle/area stretch
→ split or reparameterize bad charts
→ pack charts with padding
→ duplicate seam vertices for render buffers
→ bake textures or scalar fields
~~~

几何处理 mesh 与 render mesh 的 vertex identity 会在 seam 处不同。三维中一个 vertex 可能对应多个 UV corners，因此不要强迫每个 geometric vertex 只有一组 UV。half-edge/corner attributes 更自然。

atlas padding 必须考虑 mipmaps。只给 base level 留一 pixel 空隙，低 mip 中相邻 charts 会互相污染。bake 时还应做 gutter dilation。

质量指标至少包括 flipped faces、angle distortion、area stretch、chart count、seam length、packing utilization 和 texel density。对 CT defect heatmap，面积 distortion 会改变每单位表面的 texel sampling；对照片纹理，angle distortion更容易产生视觉拉伸。

## 7. 容易走错的岔路

“conformal”不等于没有 distortion。它局部保角，却允许面积随位置大幅变化；同样大小的缺陷在 atlas 中可能占不同 texel 数。

把 boundary 固定成圆最容易实现，但对狭长或带凹口 chart 可能产生严重拉伸。boundary shape 应与 chart 几何匹配，或使用自由边界方法。

只检查 UV triangle 是否翻转不足以判断质量。全部面积为正的 atlas 仍可能有极端 stretch 和 chart overlap。

减少 seam length 也不总是更好。强行用一个巨大 chart 覆盖高曲率闭合区域会把 distortion推到不可接受；适当 seam 是解除几何冲突的必要代价。

最后，UV 上的欧氏距离不是三维 surface distance。二维编辑工具若需要物理尺度，应使用 per-chart metric 或把操作映射回 surface 验证。

## 8. 本章落点、验证与下一章

本章把 UV 展开解释为局部 Jacobian distortion 与全局 cut/atlas 问题。harmonic parameterization 通过 cotangent Laplacian求内部 UV，signed area 检查翻转；singular values 区分角度与面积 distortion；seams 则让非 disk 拓扑能够被铺平。

在 CT mesh 项目中，本章对应缺陷热图、材料标签和二维标注 atlas。在 STL/DirectX 项目中，它对应 render-vertex seam duplication、texture baking 与 mip-safe packing。

本章的 90 分钟验证是实现五顶点 patch 的 harmonic UV，复现中心 \((0,0)\)、总面积比例 0.8165 和中心角度从约 \(78.46^\circ\) 变为 \(90^\circ\)。随后对一个闭合 sphere 人工切 seam 再参数化，记录 seam length、flip count、angle 与 area distortion；不切 seam 时应无法得到全局一一平面映射。

下一章会从 mesh 转向 point samples 与隐式曲面：当 CT segmentation、扫描点云或 Gaussian centers 没有可靠 connectivity 时，Poisson surface reconstruction 怎样从 oriented normals 建立 indicator field，并从等值面重新获得 watertight mesh。

