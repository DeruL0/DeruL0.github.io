# 可微三维渲染第 7 章：图像拟合正确的 Gaussian 为什么仍可能没有正确表面

## 1. 从一个真实任务开始

上一章解决了 Gaussian 场景在不同分辨率下的抗锯齿与 LOD。现在项目需要的不只是新视角图像，而是可测量 mesh：CT 缺陷要算孔径和体积，机器人要做碰撞，DirectX viewer 需要实体切片与 CAD 对比。

一个 RGB 3DGS 可以用许多半透明 Gaussians 共同产生正确像素，却把密度分散在真实表面前后。scale、opacity 和 view-dependent color 还能互相补偿。训练 PSNR 很高不代表存在唯一、薄而连续的 surface。

今天的任务是建立从 Gaussian 参数到几何表面的明确桥梁：定义可提取的 scalar field，用 field gradient 给出 normals，用 surface-alignment 与 depth-normal consistency减少漂浮自由度，最后从等值面提取 mesh。

## 2. 最直接的办法，以及它为什么不够

最直接的方法是把 Gaussian centers 当 point cloud，按中心做 Poisson reconstruction。问题是中心未必落在表面上；大 Gaussian 的中心可能位于实体内部，背景 floaters 也会进入点云。没有可靠 oriented normals 时，重建器只能猜测内外。

另一个方法是把 opacity 大于阈值的 Gaussians 直接连成 mesh。opacity 是 raster compositing 参数，受 footprint、排序和训练视角影响，不是统一的 volumetric density sample。相邻 Gaussians 也没有天然 triangle connectivity。

所以必须先回答“表面是什么”。对于物理 density 或 attenuation field，可以用固定等值面定义；对于 RGB Gaussians，则要构造一个 surrogate field 并通过几何约束让它与可见 surface 对齐。

## 3. 关键想法是怎样被引出来的

把 Gaussians 视为连续 basis functions。第 \(i\) 个 Gaussian 贡献

\[
\rho_i(x)
=
a_i
\exp\left[
-\frac12
(x-\mu_i)^{\mathsf T}
\Sigma_i^{-1}
(x-\mu_i)
\right].
\]

总 scalar field 为

\[
\rho(x)=\sum_i\rho_i(x).
\]

选择 threshold \(\tau\) 后，surface 定义为

\[
\mathcal S_\tau
=
\{x\mid \rho(x)=\tau\}.
\]

这让 mesh extraction 变成标准 implicit-surface 问题。但 threshold 只有在 \(a_i\) 与 field normalization具有一致意义时才可解释。CTGS 的 attenuation density 更接近物理 field；RGB 3DGS 的 opacity 需要重新标定或专门设计 surrogate density。

为了让 Gaussians 表达薄表面而非体积云，每个 covariance 应有两个切向大尺度和一个法向小尺度，最小轴朝向 surface normal。多视角 rendered depth 与这些 normals 还应彼此一致。

## 4. 一步一步建立正式模型

field gradient 可由每个 Gaussian 解析求出：

\[
\nabla\rho_i(x)
=
-\rho_i(x)
\Sigma_i^{-1}(x-\mu_i).
\]

所以

\[
\nabla\rho(x)
=
-\sum_i
\rho_i(x)
\Sigma_i^{-1}(x-\mu_i).
\]

在 regular level set 上，surface normal 与 field gradient 平行：

\[
n(x)
=
\frac{\nabla\rho(x)}
{\|\nabla\rho(x)\|_2}.
\]

分母接近零时，level set normal不稳定，通常表示多个 contributions 抵消或 threshold 落在平坦区域。

对 Gaussian covariance

\[
\Sigma_i
=
R_i
\operatorname{diag}
(s_{t1,i}^2,s_{t2,i}^2,s_{n,i}^2)
R_i^{\mathsf T},
\]

希望

\[
s_{n,i}\ll s_{t1,i},s_{t2,i}.
\]

可加入薄片正则：

\[
L_{\rm thin}
=
\sum_i
\frac{s_{n,i}^2}
{s_{t1,i}^2+s_{t2,i}^2+\epsilon}.
\]

把最小轴方向记为 Gaussian normal \(n_i=R_ie_n\)。对一个 pixel，alpha weights 为 \(w_i=T_i\alpha_i\)，可构造 weighted normal：

\[
n_G(p)
=
\operatorname{normalize}
\left(
\sum_iw_i(p)n_i
\right).
\]

渲染 depth 可写成

\[
D(p)
=
\frac{\sum_iw_i(p)z_i}
{\sum_iw_i(p)+\epsilon}.
\]

将相邻 pixels 的 depth backproject 成 camera-space points \(X(u,v)\)，由

\[
n_D
\propto
\partial_uX\times\partial_vX
\]

得到 depth normal。consistency loss 为

\[
L_{\rm normal}
=
\sum_p
\left[
1-|n_G(p)^{\mathsf T}n_D(p)|
\right].
\]

绝对值消除 normal sign ambiguity；若需要明确内外朝向，则应由 camera visibility或 CT segmentation统一 orientation，而不是永远取绝对值。

训练完成后，在 adaptive grid 或 octree 上采样 \(\rho(x)-\tau\)，找 sign changes，并使用 Marching Cubes 或 dual contouring 提取 mesh。随后执行 topology diagnostics、component filtering、normal orientation 与 reference projection检查。

## 5. 跟着一个完整例子走到底

考虑一个中心在原点的各向异性 Gaussian：

\[
\mu=(0,0,0),
\qquad
\Sigma=
\operatorname{diag}(1,1,0.04),
\qquad
a=1.
\]

选择 threshold

\[
\tau=e^{-1/2}\approx0.607.
\]

等值条件

\[
\rho(x,y,z)=e^{-1/2}
\]

等价于

\[
x^2+y^2+\frac{z^2}{0.04}=1.
\]

因此 extracted isosurface 是半轴

\[
(1,1,0.2)
\]

的扁 ellipsoid。Gaussian 的最小尺度沿 \(z\)，表面在上下方向很薄。

取顶点

\[
q=(0,0,0.2).
\]

因为

\[
\Sigma^{-1}q=(0,0,5),
\]

field gradient 为

\[
\nabla\rho(q)
=
-\tau(0,0,5),
\]

归一化 normal 指向 \(-z\)。若采用外向 convention，需根据 inside/outside 定义翻转符号。这个 normal不是从三角形近似得到，而是 field 的解析 normal。

现在在 \(z=1.5\) 处加入一个 amplitude 0.3 的小 Gaussian。若 extraction threshold 降到 0.2，它的 peak 大于 threshold，会产生第二个 disconnected isosurface，也就是 floater component。仅凭 RGB image loss，这个 floater 可能在训练 views 中被遮挡而不受惩罚。

工程上可以通过 visibility/contribution pruning、free-space loss、multi-view depth consistency 或 connected-component policy 删除它。但简单提高 \(\tau\) 也可能同时删除真实低密度薄结构，因此 threshold 与几何约束必须联合验证。

## 6. 回到真实系统：程序实际上怎样工作

可测几何 pipeline 应把 radiance rendering 与 surface extraction分开：

~~~text
Gaussian radiance/attenuation parameters
→ define calibrated scalar field rho(x)
→ surface-alignment and depth-normal losses
→ train / refine Gaussians
→ adaptive field sampling near level set
→ Marching Cubes or dual contouring
→ orient, repair and simplify mesh
→ compare mesh against depth/CT/reference geometry
~~~

field sampling若遍历整个高分辨率 dense grid，会消耗巨大内存。可用 Gaussian bounds 建 occupancy grid，只在可能跨越 \(\tau\) 的 cells 细分；但 bounds 必须保守，避免漏掉低 amplitude contributions 叠加形成的 surface。

RGB 3DGS 的 opacity 不应直接当作 CT attenuation。CTGS 若每个 Gaussian 表示线性衰减 density，则 \(\rho(x)\) 与 line integral有物理联系，threshold 可按材料/segmentation任务校准。RGB 场景则常需要 depth、normal、SDF 或 surface-alignment regularization建立 geometry meaning。

SuGaR 显式鼓励 Gaussians 与场景表面对齐并提取 mesh；2D Gaussian Splatting则直接使用嵌入三维空间的 oriented disks强化几何表达。这些方法修改的正是“volumetric ellipsoid 可以漂浮”这一自由度。[SuGaR](https://openaccess.thecvf.com/content/CVPR2024/papers/Guedon_SuGaR_Surface-Aligned_Gaussian_Splatting_for_Efficient_3D_Mesh_Reconstruction_and_CVPR_2024_paper.pdf)；[2DGS 官方实现](https://github.com/hbb1/2d-gaussian-splatting)

## 7. 容易走错的岔路

选择一个让 mesh 看起来最完整的 threshold 不等于几何正确。threshold 会改变壁厚、component 数和孔径，必须通过 reference或物理标定选择。

让所有 Gaussians 极薄也不保证形成连续 surface。它们可能朝向不一致、彼此错层或留下孔洞；还需要 normal、depth 与 neighborhood consistency。

从 rendered depth 计算 normal 时，遮挡边界的 finite difference会跨越前后表面，产生错误 normal。应使用 edge-aware mask 或置信度。

Marching Cubes 输出 watertight mesh 也不证明原 field 正确。它只忠实提取给定 scalar field；floaters、错误桥接和 oversmoothing仍会成为合法 triangles。

最后，image PSNR、mesh Chamfer 和 topology 是不同指标。两张图相近的模型可以有不同薄壁与孔洞，低 Chamfer 也可能掩盖 genus错误。

## 8. 本章落点、验证与下一章

本章把 Gaussian 表示连接到显式 surface。连续 field \(\rho(x)\) 的 level set定义几何，解析 gradient给出 normals；薄片 covariance、Gaussian-normal 与 depth-normal consistency限制漂浮自由度；adaptive sampling和等值面算法最终产生 mesh。

在 3DGS 项目中，本章对应 surface regularization、field evaluator 与 mesh extractor。在 CTGS 中，attenuation density使 field更有物理意义，但 threshold、材料界面和 topology仍需 CT reference验证。

本章的 90 分钟验证是实现单 anisotropic Gaussian field，复现半轴 \((1,1,0.2)\) 的等值面与顶点 \(q=(0,0,0.2)\) 的 normal；再加入远处 amplitude 0.3 Gaussian，扫描 threshold并记录 connected components。预期是 threshold 低于 0.3 后出现 floater component，从而验证 image-invisible density仍会进入 mesh。

下一章会进入动态与 4D Gaussian：当 scene 随时间变形时，逐帧独立重建为什么会产生 identity flicker，canonical space、deformation field 与 temporal regularization怎样维持跨帧对应。
