# 可微三维渲染第 9 章：为什么颜色不能同时扮演材质、光照和阴影

## 1. 从一个真实任务开始

上一章用 canonical space 和 deformation field 让动态 Gaussian 在时间上保持身份。现在场景不再只是“从新视角看起来对”，还要能换光照、编辑材质或把 CTGS 提取的表面放进一个真实渲染器中。此时最常见的失败是：训练出的颜色已经把材质、环境光、阴影、曝光和相机白平衡全部烘焙进去。

真实任务是扫描一个金属和塑料混合的小零件。多视角照片中，金属边缘有高光，塑料上有柔和阴影。普通 3DGS 可以为每个 Gaussian 学一个 view-dependent color 或 spherical harmonics color，在训练视角上很像原图。但如果把光源移到另一侧，阴影仍留在原位置；如果把金属改成更粗糙，原来的高光方向仍可能被颜色场保留。它不是一个可重光照的资产，而是一组被相机和光照条件锁住的辐射值。

今天的任务是把颜色拆成几何、材质、入射光、可见性和相机响应。逆渲染不是为了让公式更物理，而是为了让“换光照”和“改材质”这样的操作有稳定的对象可改。

## 2. 最直接的办法，以及它为什么不够

最直接的方法是继续优化每个 Gaussian 的 RGB 或 SH coefficients。模型看到某个像素偏暗，就降低对应 Gaussian 的颜色；看到某个视角高光，就给 view-dependent 分量更强的响应。训练 loss 会下降，但原因不唯一：暗可以来自低 albedo、弱光、阴影、法线背光、曝光低或透明遮挡。

这种混合在新视角合成里尚可接受，因为目标只是复现已观察光场。一旦要 relighting，它会失败。把灯移到右侧后，阴影应该跟随几何和可见性移动；若阴影已经写进 albedo，灯怎么移动都不会改变暗斑。高光也类似：若 view-dependent color 直接拟合高光，它能插值训练视角，但无法根据 roughness 和 half-vector 产生正确新高光。

第二个直接办法是把所有外观交给一个神经网络 \(c(x,\omega_o,t)\)。它表达能力更强，但分解更弱。网络可以用复杂函数同时记住相机、光照和材质，不会自动给出可编辑 BRDF。需要的抽象是 rendering equation 中的物理因子，而不是更大的颜色函数。

## 3. 关键想法是怎样被引出来的

一张图像上的颜色来自一条光路：光从环境或光源出发，被其他几何遮挡或反射，抵达表面点，再由表面 BRDF 按观察方向反射到相机。若我们只学习最终 RGB，就把这条链条压成一个数。逆渲染要把链条拆开。

对一个表面点，最小分解至少包括 normal \(n\)、albedo 或 base color \(\rho\)、roughness \(\alpha\)、incoming illumination \(L_i\)、visibility \(V\) 和相机曝光。Gaussian 表示没有天然三角形法线，因此还要从 covariance、depth map 或局部 surface regularization 中得到稳定 normal。材质和光照分解只有在 normal 与几何比较可靠时才有意义。

近年的 relightable Gaussian 方法正沿这个方向扩展。Relightable 3D Gaussian 给每个 Gaussian 增加 normal、BRDF 和 incident light，并用物理可微渲染与点式 ray tracing 处理可见性；CVPR 2025 的 SVG-IR 进一步强调空间变化的 Gaussian normal 和 material 属性。这些工作改变的不是“3DGS 很快”这一点，而是把 3DGS 从 radiance cache 推向可编辑的 inverse-rendering asset。[Relightable 3D Gaussian](https://nju-3dv.github.io/projects/Relightable3DGaussian/) [SVG-IR](https://openaccess.thecvf.com/content/CVPR2025/papers/Sun_SVG-IR_Spatially-Varying_Gaussian_Splatting_for_Inverse_Rendering_CVPR_2025_paper.pdf)

## 4. 一步一步建立正式模型

先写最核心的表面反射。对表面点 \(x\)、观察方向 \(\omega_o\)，出射辐射为

\[
L_o(x,\omega_o)
=
\int_{\Omega^+}
f_r(x,\omega_i,\omega_o;\rho,\alpha)
L_i(x,\omega_i)
V(x,\omega_i)
\max(0,n\cdot\omega_i)
d\omega_i.
\]

这个公式的每一项都有工程含义。\(f_r\) 是 BRDF，说明材料如何反射；\(L_i\) 是从方向 \(\omega_i\) 来的光；\(V\) 是该方向是否被遮挡；\(\max(0,n\cdot\omega_i)\) 是斜照射的投影面积。若任何一项被固定到颜色里，relighting 就失去可控性。

最简单的 Lambertian BRDF 为

\[
f_r=\frac{\rho}{\pi}.
\]

在单一方向光 \(L\) 下，且 visibility 为 1，颜色变成

\[
L_o
=
\frac{\rho}{\pi}
L
\max(0,n\cdot l).
\]

这一步说明 albedo 与光照的乘法歧义：如果只观察到 \(L_o\)，把 \(\rho\) 乘 2、把 \(L\) 除以 2，图像不变。因此逆渲染需要多光照、多视角、已知曝光、材质先验或环境光约束来破除歧义。

对 Gaussian splatting，某个 pixel 的最终颜色仍通过 alpha compositing 得到。第 \(i\) 个 Gaussian 先由材质模型算出 shading color

\[
s_i
=
\mathcal S(n_i,\rho_i,\alpha_i,L,V_i,\omega_o),
\]

再按屏幕顺序合成：

\[
C
=
\sum_i
T_i \alpha_i s_i,
\qquad
T_i
=
\prod_{k<i}(1-\alpha_k).
\]

训练 objective 不再只比较 RGB：

\[
L
=
L_{\rm photo}
+\lambda_n L_{\rm normal}
+\lambda_m L_{\rm material}
+\lambda_l L_{\rm light}
+\lambda_v L_{\rm visibility}.
\]

这些项不是越多越好；它们分别约束 normal 平滑、材质低频、光照表示容量和 visibility 一致性，防止所有因素互相冒充。

## 5. 跟着一个完整例子走到底

考虑一个小的 Lambertian patch，albedo 为

\[
\rho=0.8,
\]

方向光强度为

\[
L=10,
\]

法线与光方向夹角满足

\[
n\cdot l=0.5.
\]

没有遮挡时，出射辐射为

\[
L_o
=
\frac{0.8}{\pi}\times 10\times 0.5
\approx1.273.
\]

如果该方向一半被遮挡，取

\[
V=0.5,
\]

则

\[
L_o
=
\frac{0.8}{\pi}\times 10\times 0.5\times0.5
\approx0.637.
\]

普通颜色场如果只看到第二个数字，可能把 Gaussian 颜色学成 0.637。现在把光强改成 \(L=20\)，真实未遮挡部分应翻倍，遮挡仍由 \(V\) 决定：

\[
L_o'
=
\frac{0.8}{\pi}\times 20\times 0.5\times0.5
\approx1.273.
\]

若模型把 0.637 当成 albedo 或 baked color，换光后仍输出接近 0.637，relighting 失败。反过来，若它估计出 \(\rho=0.8\)、\(V=0.5\)、\(n\cdot l=0.5\)，则可以按新光照重新计算。

再看高光。粗糙金属与光滑金属在同一视角下可能有相近平均亮度，但 roughness 决定高光宽度。若用 view-dependent SH 拟合，模型可能在训练视角重现高光，却不知道 roughness；换观察方向后，高光不会按照 half-vector 移动。BRDF 参数化的价值就在于让这种移动由几何关系决定。

## 6. 回到真实系统：程序实际上怎样工作

一个可重光照 Gaussian pipeline 通常先得到较稳定的几何：depth、normal 或 surface proxy。然后为每个 Gaussian 或每个局部 patch 维护 base color、roughness、metallic 或更简单的 diffuse/specular 参数。光照可用 environment map、spherical harmonics、spherical Gaussians 或 per-point incident radiance 表示；visibility 可由 shadow ray、BVH baking、screen-space approximation 或 learned visibility 估计。

训练时通常先优化 geometry 与 radiance，使视角重建稳定；再逐步引入 PBR shading，避免一开始 normal、material 和 light 全部乱动。若有多光照数据，分解会稳得多；只有单一未知光照时，必须依赖材质平滑、灰世界、已知相机响应或标定球这类先验。

对 CTGS，材质一词要谨慎。CT attenuation field 的“材料”首先是 X-ray attenuation 或密度相关属性，不等于 RGB albedo。若把 CTGS 输出放入可视化渲染器，应该明确区分物理重建的 attenuation/material label 与可视化使用的 surface shader。真正项目中可以把 CT 重建得到的材料标签映射到 PBR 参数，但不能把照片中的阴影当作 CT 材料。

在引擎落地时，Gaussian inverse rendering 的结果最好导出为可诊断 buffers：normal、albedo、roughness、visibility、direct light、indirect light 和最终 composite。只看 final RGB，无法判断模型是否真的分解成功。

## 7. 容易走错的岔路

第一个误区是相信 photometric loss 会自动分解。图像误差只监督乘积和合成结果，不监督因子。没有先验或额外观测时，albedo-light-shadow 的分解本来就不唯一。

第二个误区是从不稳定 geometry 估计材质。normal 错会直接污染 BRDF；模型可能用 roughness 或 albedo 补偿 normal 错误，最后所有 buffer 都看似合理但不可编辑。

第三个误区是把 shadow baked 到 albedo。阴影在训练图里是低频暗区，很容易被材质平滑项接收。验证 relighting 时必须移动光源，检查阴影是否随几何和 visibility 移动。

第四个误区是把每个 Gaussian 独立估材质。真实材质通常在表面区域内连续；完全独立的 per-Gaussian BRDF 会吸收噪声和视角误差。需要局部共享、表面参数化或空间正则。

最后，过强的物理模型也会失败。若真实图像包含 auto exposure、tone mapping、sensor saturation 或未建模间接光，硬套简单直接光模型会把相机误差推给材质。相机响应也是 inverse rendering 的一部分。

## 8. 本章落点、验证与下一章

本章从 baked color 的 relighting 失败推导出材质、光照、visibility 和相机响应的分解。Rendering equation 告诉我们，最终像素是 BRDF、入射光、遮挡和几何投影的组合；Gaussian renderer 可以先按每个 Gaussian 计算 PBR shading，再通过 alpha compositing 合成。

在可微渲染项目中，本章对应 normal estimation、per-Gaussian material、environment lighting、visibility baking、buffer supervision 和 relighting evaluation。在 CTGS 中，它提醒你把 X-ray attenuation 的材料含义与 RGB/PBR 可视化材料分开。

本章的 60 到 90 分钟验证是实现 Lambertian patch 例子：固定 \(\rho=0.8\)、\(n\cdot l=0.5\)、\(V=0.5\)，分别计算 \(L=10\) 和 \(L=20\) 的输出；再训练一个只有 baked color 的小模型和一个显式 \(\rho,V\) 的小模型，比较换光后的预测。预期是 baked model 不能正确翻倍，分解 model 可以按新光照更新。

下一章将处理可微 visibility。材质与光照分解后，最不平滑、也最容易产生错误梯度的是遮挡：一点点几何移动可能让光线从可见变成不可见，普通连续反向传播不能直接解释这种跳变。
