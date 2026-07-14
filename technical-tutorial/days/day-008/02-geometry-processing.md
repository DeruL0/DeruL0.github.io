# 几何处理第 8 章：没有三角形连接关系时，怎样从法线恢复一个完整曲面

## 1. 从一个真实任务开始

上一章把已有 triangle mesh 展开成 UV atlas。但 CT segmentation 的边界采样、激光扫描和 Gaussian centers常只提供一组点

\[
\{p_i\}_{i=1}^{N},
\]

以及估计的 normals

\[
\{n_i\}_{i=1}^{N},
\]

并没有说明哪些三点应该连成一个 face。真实任务是把这组带噪、密度不均且有局部缺口的 oriented points 转成 watertight mesh，供体积测量、布尔运算与 DirectX 渲染使用。

我们希望算法利用全部样本形成一个全局一致的 inside/outside，而不是在每个点附近独立拼小三角形。Poisson surface reconstruction 的出发点正是：一个闭合曲面的 outward normals，可以被看成某个体积指示函数梯度的边界信号。

## 2. 最直接的办法，以及它为什么不够

最直接的方法是对每个点找 \(k\) 个 nearest neighbors，在局部切平面里做 Delaunay triangulation。平坦、均匀采样的小 patch 上可行，但两个彼此接近的薄壁可能被错误连接；孔洞附近的邻居只分布在一侧，局部三角化还会产生长而翻转的 faces。

另一种方法是先把点栅格化为 occupancy，再做 Marching Cubes。若只把采样点所在 cells标为 occupied，得到的是许多离散小块；必须先决定哪里是 inside。点本身只说明 surface 位置，没有直接给出整个 volume 的符号。

因此困难不只是补 connectivity，而是从局部边界观测恢复一个全局 scalar field。只要这个 field 的某个等值面代表原曲面，连接关系就能由标准 isosurface extraction统一产生。

## 3. 关键想法是怎样被引出来的

设实体区域为 \(\Omega\)，定义 indicator function

\[
\chi(x)=
\begin{cases}
1,&x\in\Omega,\\
0,&x\notin\Omega.
\end{cases}
\]

\(\chi\) 在实体内部和外部都是常数，只有穿过边界时才跳变。因此它的 gradient在普通区域为零，在 surface附近沿法线集中。若使用 outward normal convention，\(\nabla\chi\) 与 outward normal方向相反；符号可通过整体取负调整，重要的是所有 normals orientation一致。

输入 oriented points可以被 splat 成一个连续 vector field \(V(x)\)，使它近似 \(\nabla\chi(x)\)。于是重建问题变成寻找一个 scalar function，使其 gradient最接近 \(V\)：

\[
\min_{\chi}
\int
\|\nabla\chi(x)-V(x)\|_2^2\,dx.
\]

这一步把成千上万个互不连接的局部 normals，转成了一个整体可积的 gradient-field fitting问题。

## 4. 一步一步建立正式模型

为了得到求解方程，对能量作变分。令 \(\chi\) 发生一个小扰动 \(\epsilon\varphi\)。一阶变化为

\[
\frac{d}{d\epsilon}
E(\chi+\epsilon\varphi)
\bigg|_{\epsilon=0}
=
2\int
(\nabla\chi-V)^{\mathsf T}\nabla\varphi\,dx.
\]

通过分部积分并忽略适当边界项：

\[
\delta E
=
-2\int
\varphi\,
\nabla\cdot(\nabla\chi-V)\,dx.
\]

要让任意 \(\varphi\) 下的一阶变化为零，必须满足

\[
\Delta\chi
=
\nabla\cdot V.
\]

这就是空间 Poisson equation。左侧 \(\Delta\chi\) 是 indicator 的 Laplacian，右侧是由 oriented samples构造的 vector-field divergence。

实际算法不会在无限连续空间直接求解。它建立 adaptive octree，以局部 basis functions \(\phi_j(x)\) 表示

\[
\chi(x)=\sum_j c_j\phi_j(x).
\]

把 weak form投影到每个 test function后得到 sparse linear system

\[
Lc=b,
\]

其中

\[
L_{jk}
=
\int
\nabla\phi_j^{\mathsf T}\nabla\phi_k\,dx,
\qquad
b_j
=
\int
V^{\mathsf T}\nabla\phi_j\,dx.
\]

求得 \(c\) 后，用样本处 field values 的统计量选择 isovalue \(\tau\)，再提取

\[
\mathcal S=\{x\mid\chi(x)=\tau\}.
\]

原始 Poisson formulation偏向全局平滑，可能让 surface偏离 samples。Screened Poisson额外加入点值约束：

\[
E_{\rm screened}
=
\int\|\nabla\chi-V\|^2dx
+
\lambda
\sum_i
w_i\left(\chi(p_i)-\tau\right)^2.
\]

\(\lambda\) 增大时 surface更贴近样本，但噪声也更容易进入几何；它控制的是 gradient consistency 与 interpolation之间的平衡。

## 5. 跟着一个完整例子走到底

先用一维例子看清整个因果链。真实实体是区间

\[
\Omega=[1,3],
\]

其 indicator为区间内 1、外部 0。它在 \(x=1\) 从 0 跳到 1，在 \(x=3\) 从 1 跳到 0，所以 distributional derivative为

\[
\chi'(x)=\delta(x-1)-\delta(x-3).
\]

区间左端 outward normal是 \(-1\)，右端是 \(+1\)。因此 \(\chi'\) 等于 outward-normal impulses 的负值，这就是前面提到的 convention。若输入点为

\[
(p_1,n_1)=(1,-1),
\qquad
(p_2,n_2)=(3,+1),
\]

则令

\[
V(x)=-n_1\delta(x-1)-n_2\delta(x-3)
=\delta(x-1)-\delta(x-3).
\]

现在最小化

\[
\int|\chi'(x)-V(x)|^2dx
\]

的理想解满足 \(\chi'=V\)。从左侧取 \(\chi=0\) 并积分：经过 \(x=1\) 后函数增加 1，在 \(1<x<3\) 保持 1；经过 \(x=3\) 后下降 1，重新回到 0。于是完整 inside interval被恢复出来。

离散到五个 cell boundaries，可把 derivative samples近似写成

\[
V=[0,\ 1,\ 0,\ -1,\ 0].
\]

从左向右累积得到

\[
\chi=[0,\ 1,\ 1,\ 0],
\]

选择 \(\tau=0.5\) 后，两个 crossing正好位于区间左右边界。这个简单例子已经包含三维算法的全部结构：oriented samples形成 vector field，求一个最接近它的 scalar potential，再提取等值面。

若把右端 normal错误地也设为 \(-1\)，则两个 impulses同号，积分后 \(\chi\) 不会回到外部基线。三维中，一簇翻转 normals同样会迫使 field形成错误壳层或拓扑。因此 normal orientation不是可选清洗步骤，而是 Poisson模型成立的前提。

## 6. 回到真实系统：程序实际上怎样工作

完整 pipeline 通常为：

~~~text
raw points
→ remove outliers and estimate sampling density
→ estimate local normals by PCA
→ orient normals consistently
→ splat normals into an adaptive vector field
→ solve the sparse Poisson system
→ choose an isovalue and extract the mesh
→ trim low-support regions
→ validate components, normals, topology and distance
~~~

局部 PCA 只给出 normal axis，\(n\) 与 \(-n\) 都是 eigenvector。可从已知 sensor viewpoints定向；没有 viewpoints时，可在 neighborhood graph上做 minimum spanning tree propagation，再通过全局 inside/outside检查修正。

octree depth决定最小可表达尺度，但不是越深越好。采样间距之外的高分辨率主要拟合噪声并增加 memory。每个 vertex还应保存 density/support estimate，便于删除由 Poisson天然闭合的、但输入点几乎没有支持的薄盖。

Poisson方法倾向 watertight，这是体积测量的优点，也是开放扫描的风险。扫描对象若本来只有一块开曲面，算法会在缺口处补盖；不能因为输出 manifold就认定补出的区域真实。

经典 Poisson reconstruction把所有 oriented points作为一个全局空间问题，并利用局部 basis hierarchy形成 sparse system；Screened Poisson进一步增加样本约束。[Poisson Surface Reconstruction 原论文](https://diglib.eg.org/items/245d6ba1-0209-4365-a4a4-16ce46421620)

对 CT segmentation，可直接从 scalar volume gradient生成 oriented samples，或者跳过点云直接提取原始 isosurface。Poisson更适合融合多个稀疏表面来源、Gaussian samples或存在局部孔洞的扫描；若原 voxel field完整，先丢成点云再重建可能反而损失信息。

## 7. 容易走错的岔路

把 Poisson当成“自动补洞”工具会掩盖它补出了什么。它求的是最符合全局 gradient field的闭合 level set，并不知道缺口处的真实形状。

只提高 octree depth无法恢复采样中不存在的细节。它会让 solver更贵，并把 position与normal noise变成细小褶皱。

normal未统一定向时直接求解，正负 contributions会互相抵消或制造双层 surface。先检查 neighborhood normal dot products与全局朝向。

用最近点 Chamfer评价重建不足以发现错误桥接和 genus。还要报告 normal consistency、component count、watertightness、self-intersection与已知体积误差。

最后，Screened Poisson 的高 interpolation weight并不总更准确。若点云含 outliers或 anisotropic noise，过度贴点会牺牲本来有价值的全局平滑。

## 8. 本章落点、验证与下一章

本章从 indicator function推导了 Poisson surface reconstruction。oriented points近似边界 gradient，最小化 \(\|\nabla\chi-V\|^2\) 的 Euler-Lagrange equation给出 \(\Delta\chi=\nabla\cdot V\)；adaptive basis把它变成 sparse system，等值面再提供统一 connectivity。

在几何和 STL 项目中，本章对应 point preprocessing、normal orientation、octree solve、isosurface extraction和 support-aware trimming。在 CTGS 中，它可把有物理或几何意义的 oriented Gaussian samples转换为可测 mesh，但不能替代对 field 与 surface含义的标定。

本章的 90 分钟验证是先复现区间 \([1,3]\) 的一维 discrete example，再对一个球面均匀采样点、删除顶部 15% 样本并运行 Poisson reconstruction。预期是输出仍然闭合，但顶部由算法补出；翻转 20% normals后会出现明显 field冲突或额外壳层。记录 Hausdorff distance、watertightness和低支持区域，而不只看截图。

下一章会进入鲁棒 predicates 与 mesh Boolean。Poisson可以给出 watertight triangles，却不能保证后续 triangle-triangle intersection、inside test和拓扑更新在浮点误差下可靠；布尔运算需要把数值判定与拓扑构造一起设计。
