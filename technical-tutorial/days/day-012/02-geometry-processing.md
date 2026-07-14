# 几何处理第 12 章：Laplacian 谱为什么能把形状拆成低频和高频

## 1. 从一个真实任务开始

上一章用拓扑量判断 mesh 是否是合理的流形、是否有洞和 handle。拓扑告诉我们连接方式，但不告诉我们形状上的频率：哪些是整体弯曲，哪些是局部噪声，哪些是多尺度特征。平滑、分割、对应和形状检索都需要这种频率视角。

真实任务是处理一个 CT 提取的多孔材料表面。你想保留大孔结构和整体曲面趋势，去掉 marching cubes 的 stair-step 噪声；或者比较两个弯曲姿态下的同一零件，要求描述符对刚体位姿和轻微弯曲稳定。普通坐标频率不适合曲面，因为曲面没有统一平面坐标系。

谱几何的关键是用曲面自己的 Laplace-Beltrami operator 定义频率。低特征值对应缓慢变化的模式，高特征值对应快速振荡的模式。Shape-DNA 等方法把 Laplace-Beltrami 谱作为形状指纹，用于比较曲面和实体。[Laplace-Beltrami spectra as Shape-DNA](https://www.sciencedirect.com/science/article/abs/pii/S0010448505001867)

## 2. 最直接的办法，以及它为什么不够

最直接的平滑方法是把每个顶点移动到邻居平均值。它能去噪，但也会收缩形状、抹掉边界和薄结构。没有频率解释时，你很难控制“去掉多小的细节，保留多大的结构”。

第二个方法是在三维坐标轴上做 Fourier 分析。曲面不是规则网格，顶点分布不均，坐标轴方向还会随姿态变化。一个弯曲但近似等距的薄片，欧氏坐标变了，曲面内在结构却没有变。

第三个方法是只用局部曲率。曲率能描述局部细节，但缺少全局多尺度基。孔洞、handle 和整体形状变化往往需要把全局和局部放在同一个框架里。

因此需要曲面内在的差分算子。Laplacian 正好度量函数在曲面上的局部平均偏离，它的特征函数给出曲面自己的振动模式。

## 3. 关键想法是怎样被引出来的

在平面上，Fourier basis 是 Laplacian 的特征函数。曲面上没有全局正交坐标，但有 Laplace-Beltrami operator \(\Delta\)。求解

\[
-\Delta \phi_k=\lambda_k\phi_k
\]

得到的 \(\phi_k\) 就是曲面上的频率基。小 \(\lambda_k\) 变化慢，大 \(\lambda_k\) 变化快。任意标量函数 \(f\) 可以展开为

\[
f=\sum_k a_k\phi_k.
\]

这把 mesh smoothing、shape descriptor 和多尺度分析放到同一个语言里：保留低频系数是平滑；使用若干特征值是全局形状指纹；观察热扩散则是不同时间尺度下的局部几何。

对离散 mesh，连续 operator 要变成矩阵。最常用的三角网格离散是 cotangent Laplacian，它把几何角度写进边权重，因此比简单 graph Laplacian 更接近曲面内在几何。

## 4. 一步一步建立正式模型

设 mesh 顶点上有标量函数 \(f\)。cotangent Laplacian 在顶点 \(i\) 上可写为

\[
(Lf)_i=
\sum_{j\in N(i)}
w_{ij}(f_i-f_j),
\]

其中

\[
w_{ij}=\frac12(\cot\alpha_{ij}+\cot\beta_{ij}).
\]

\(\alpha_{ij}\) 和 \(\beta_{ij}\) 是边 \((i,j)\) 对面的两个角。这个权重来自三角形有限元离散，反映了局部几何，而不只是邻接关系。

还需要 mass matrix \(M\)，因为顶点代表的面积不同。离散特征问题写成

\[
L\phi_k=\lambda_k M\phi_k.
\]

特征函数按 \(M\)-inner product 正交：

\[
\phi_i^{\mathsf T}M\phi_j=\delta_{ij}.
\]

若坐标函数为

\[
X=(x,y,z),
\]

低通平滑可以对每个坐标分量做谱截断：

\[
X_{\rm smooth}
=
\sum_{k=0}^{K} a_k\phi_k.
\]

热扩散则使用 heat kernel。时间 \(t\) 下，频率 \(k\) 的衰减为

\[
e^{-\lambda_k t}.
\]

因此

\[
f(t)=\sum_k e^{-\lambda_k t}a_k\phi_k.
\]

大 \(\lambda_k\) 高频随 \(t\) 快速衰减，小 \(\lambda_k\) 低频保留更久。这给多尺度平滑一个明确的物理解释。

## 5. 跟着一个完整例子走到底

用一条四点链近似理解谱分解。顶点为 \(1,2,3,4\)，简单 graph Laplacian 为

\[
L=
\begin{bmatrix}
1&-1&0&0\\
-1&2&-1&0\\
0&-1&2&-1\\
0&0&-1&1
\end{bmatrix}.
\]

考虑一个带噪信号

\[
f=[1,\ 1.2,\ -0.8,\ -1]^{\mathsf T}.
\]

它的大尺度结构是左边为正、右边为负；中间的 \(1.2\) 和 \(-0.8\) 包含局部噪声。Laplacian 的低频特征向量变化慢，高频特征向量在相邻点间快速变号。若只保留前两个低频模式，信号会接近

\[
f_{\rm low}\approx[1.05,\ 0.8,\ -0.8,\ -1.05]^{\mathsf T},
\]

整体正负结构保留，但局部尖跳减弱。

这个一维例子对应 mesh 上的坐标平滑：低频保留整体形状，高频对应局部皱褶和噪声。若把顶点坐标的所有高频都删掉，薄孔边缘也会被抹掉；若保留太多高频，marching cubes 噪声仍在。谱方法给了一个明确的尺度旋钮 \(K\) 或 \(t\)。

在真实三角网格中，不能用简单链矩阵替代 cotangent Laplacian，但思路一致：解 \(L\phi=\lambda M\phi\)，再决定保留或衰减哪些模式。

## 6. 回到真实系统：程序实际上怎样工作

工程谱处理的第一步是准备高质量 mesh。非流形、自交和极差三角形会让 cotangent weights 异常，上一章的 topology repair 和 remeshing 是前置条件。然后构造 sparse \(L\) 和 \(M\)，用 eigensolver 求前 \(K\) 个特征对。

对大 mesh，完整特征分解不可行。通常只求低频前几十到几百个特征，或用 Chebyshev polynomial、heat method、multigrid 等近似。若只是平滑，不必显式求所有特征；可以解隐式 Laplacian smoothing 系统。

在 STL/CT 项目中，谱方法适合做多尺度平滑、形状签名、对称性分析和粗分割。它不适合直接处理拓扑错误，也不应盲目用于保护尖锐制造边。对 sharp features，应结合 feature-aware weights 或先做局部分区。

对 CTGS，Gaussian surface 的谱分析可以用于导出 mesh 的多尺度诊断，或把 surface regularization 写成 Laplacian energy。若直接对 Gaussian centers 建 graph Laplacian，需要确认边权代表真实表面邻接，而不是空间近邻误连。

## 7. 容易走错的岔路

第一个误区是把谱低频等同于真实结构。真实细裂纹和薄壁也可能是高频，不能一概删除。

第二个误区是忽略 mesh 质量。cotangent weights 对 skinny triangles 敏感；坏网格会让谱分析产生数值伪影。

第三个误区是只比较特征值。特征值是全局描述符，不能定位局部差异；需要特征函数、HKS/WKS 或局部谱特征才能做局部分析。

第四个误区是认为谱方法自动对所有变形不变。Laplace-Beltrami 谱对等距变形稳定，但拉伸、拓扑变化和边界条件改变都会影响谱。

最后，边界条件必须明确。开放 surface 使用 Neumann、Dirichlet 或补洞后的 closed surface，会得到不同谱。

## 8. 本章落点、验证与下一章

本章把几何处理推进到谱视角：Laplacian 的特征函数提供曲面内在频率基，低频描述整体结构，高频描述局部变化；cotangent Laplacian 和 mass matrix 把连续 Laplace-Beltrami 离散到三角网格。

在几何、STL 和 CTGS 项目中，本章对应多尺度平滑、shape descriptor、surface regularization、特征保留策略和大 mesh eigensolver 选择。

本章的 60 到 90 分钟验证是构造四点链 Laplacian，求它的前两个低频模式，用它们近似信号 \(f=[1,1.2,-0.8,-1]\)。预期低频近似保留左正右负的大结构，同时削弱局部尖跳；再比较保留模式数对平滑程度的影响。

下一章将进入高性能几何内核。谱方法和修复算法都依赖稀疏矩阵、邻接遍历和空间查询；要在大型 CT mesh 上实用，必须把这些操作做成稳定、并行、缓存友好的几何内核。
