# DirectX 12 第 7 章：为什么实时材质必须同时解释粗糙度、视角和能量

## 1. 从一个真实任务开始

上一章已经让正确 LOD 的几何按需进入 GPU。现在 STL/CT viewer 要区分塑料、氧化金属、抛光钢和树脂；游戏引擎还需要同一材质在太阳、室内灯和环境光下保持一致。简单调一个高光强度可以在某张图上好看，却会在换灯或换视角后失真。

今天的任务是建立实时 physically based rendering 的核心材质模型。输入是 surface normal、view/light directions、base color、roughness、metallic 与光照；输出是符合 reciprocity、近似能量守恒并随视角变化的 reflected radiance。

microfacet BRDF 把粗糙表面看成大量微小镜面。normal distribution、Fresnel 与 masking-shadowing分别描述微表面朝向、界面反射比例和互相遮挡，三者共同形成高光。

## 2. 最直接的办法，以及它为什么不够

最直接的 Blinn-Phong 高光写成

\[
(n^{\mathsf T}h)^s
\]

再乘一个 artist-selected intensity。指数 \(s\) 控制高光宽度，但 intensity 与宽度没有必然能量关系；多个 lights 叠加时可能反射超过入射能量，材质参数也难以从测量迁移。

另一个直接方法是给金属和塑料使用同一 diffuse 加白色 specular。真实金属在可见光中几乎没有独立 diffuse lobe，反射颜色主要来自 wavelength-dependent Fresnel；dielectric 的 normal-incidence reflectance 通常较低，剩余能量才进入介质形成 diffuse。

所以需要 BRDF 把入射方向、出射方向和材质参数放入统一函数，并明确 diffuse 与 specular 如何分配能量。

## 3. 关键想法是怎样被引出来的

rendering equation 在一个 surface point 写成

\[
L_o(v)
=
\int_{\Omega^+}
f_r(l,v)
L_i(l)
(n^{\mathsf T}l)
d\omega_l.
\]

\(f_r\) 是 BRDF，表示单位入射 irradiance向 view direction反射多少 radiance。实时 direct light 将积分离散为若干 lights，environment lighting则用预积分近似。

microfacet 模型假设只有法向恰好接近 half vector

\[
h=\frac{l+v}{\|l+v\|}
\]

的微镜面能把 \(l\) 反射到 \(v\)。有多少这类 facets由 normal distribution \(D\) 决定；每个 facet反射比例由 Fresnel \(F\) 决定；facet是否被邻居遮挡由 geometry term \(G\) 决定。

## 4. 一步一步建立正式模型

常用 Cook-Torrance specular BRDF 为

\[
f_s(l,v)
=
\frac{
D(h)F(v,h)G(l,v)
}{
4(n^{\mathsf T}l)(n^{\mathsf T}v)
}.
\]

分母把 microfacet 与宏观 surface projected area联系起来。实现中所有 dot products 应 clamp 到非负，并单独处理 grazing numerical stability。

以 GGX normal distribution 为例。令 perceptual roughness 为 \(r\)，常用映射

\[
\alpha=r^2.
\]

则

\[
D_{\rm GGX}(h)
=
\frac{\alpha^2}
{
\pi
\left[
(n^{\mathsf T}h)^2(\alpha^2-1)+1
\right]^2
}.
\]

roughness 小时 distribution集中，高光尖；roughness 大时微表面方向分散，高光宽。

Schlick Fresnel approximation 为

\[
F(v,h)
=
F_0+(1-F_0)(1-v^{\mathsf T}h)^5.
\]

\(F_0\) 是 normal-incidence reflectance。dielectric 常使用约 0.04 的 RGB 值；metallic workflow让金属的 \(F_0\) 接近 base color。

Smith masking-shadowing写成

\[
G(l,v)=G_1(l)G_1(v),
\]

其中一种 GGX 形式为

\[
G_1(x)
=
\frac{
2(n^{\mathsf T}x)
}{
(n^{\mathsf T}x)
+
\sqrt{
\alpha^2+
(1-\alpha^2)(n^{\mathsf T}x)^2
}
}.
\]

metallic 参数 \(m\in[0,1]\) 常用来混合

\[
F_0
=
(1-m)0.04+m\,c_{\rm base}.
\]

dielectric diffuse 可近似写为

\[
f_d
=
(1-m)
(1-F)
\frac{c_{\rm base}}{\pi}.
\]

于是

\[
f_r=f_d+f_s.
\]

这里的 \((1-F)\) 近似把已进入 specular reflection 的能量从 diffuse中移除。实际引擎会使用多散射补偿、Disney参数化或预积分修正，但核心分工不变。

## 5. 跟着一个完整例子走到底

考虑 grayscale dielectric，base color 为 0.8，metallic \(m=0\)，roughness

\[
r=0.5,
\qquad
\alpha=r^2=0.25.
\]

view 正对 surface：

\[
n^{\mathsf T}v=1.
\]

light 与 normal夹角 60 度：

\[
n^{\mathsf T}l=0.5.
\]

half vector位于二者中间，所以

\[
n^{\mathsf T}h
=
v^{\mathsf T}h
=
\cos30^\circ
\approx0.866.
\]

GGX distribution 为

\[
D
=
\frac{0.25^2}
{
\pi
\left[
0.866^2(0.25^2-1)+1
\right]^2
}
\approx0.226.
\]

Fresnel 为

\[
F
=
0.04+0.96(1-0.866)^5
\approx0.04004.
\]

view 方向的 \(G_1(v)=1\)。light方向：

\[
G_1(l)
\approx
\frac{1}
{0.5+\sqrt{0.25^2+(1-0.25^2)0.5^2}}
\approx0.957.
\]

所以 specular BRDF 为

\[
f_s
\approx
\frac{0.226\times0.04004\times0.957}
{4\times0.5\times1}
\approx0.00433.
\]

diffuse BRDF 约为

\[
f_d
\approx
(1-0.04004)\frac{0.8}{\pi}
\approx0.2445.
\]

总 BRDF 约 0.2488。若 direct light radiance 为 1，最终还要乘 cosine：

\[
L_o
\approx
0.2488\times0.5
\approx0.1244.
\]

这个例子展示了高光并不是任意加上去的白点；roughness决定 \(D\)，view/light决定 \(F,G\)，剩余能量进入 diffuse。

## 6. 回到真实系统：程序实际上怎样工作

DX12 renderer 中，material通常保存：

~~~text
base-color texture / factor
normal map
roughness
metallic
ambient occlusion
optional emissive
~~~

base color texture从 sRGB decode 到 linear space 后参与 BRDF，最终 tone mapping 后再编码到 display。roughness/metallic/AO 是线性数据，不能按 sRGB解释。

normal map需要 tangent frame。mesh simplification、UV seam 和 mirrored UV会影响 tangent handedness；错误 TBN 会让高光沿 seam 翻转。几何 normal、shading normal 和 double-sided policy也要明确。

direct lighting pass计算 BRDF并乘 light visibility。shadow map或 ray tracing visibility不属于 BRDF本身；把 shadow直接烘进 base color会破坏换灯一致性。

environment lighting通常使用 split-sum approximation：diffuse irradiance map近似半球积分，prefiltered specular environment按 roughness选择 mip，BRDF LUT近似 Fresnel/geometry积分。它是对 rendering equation的预积分，不是额外 artist glow。

在 deferred renderer 中，G-buffer至少要能恢复 normal、base color、roughness、metallic和 depth；格式精度与带宽需权衡。Forward+ 则先按 screen tiles组织 light lists，再在 material shader中直接评估 BRDF。

## 7. 容易走错的岔路

roughness 为零会让 GGX接近 delta distribution，有限 precision和有限 pixel sampling下容易闪烁。实际系统通常设置小正下界或使用 specular antialiasing。

把 metallic 当作“金属感强度”随意设中间值，对真实纯材料并不总合理。中间值更多用于混合像素、污渍或艺术工作流。

base color已经包含光照或阴影时，再进入 PBR会重复照明。材质输入应尽量是 intrinsic reflectance。

normal map只改变 shading normal，不改变 silhouette和真实遮挡。强法线起伏在轮廓处仍是平的，需要 displacement或几何。

最后，PBR参数一致不代表结果自动写实。light intensity、exposure、tone mapping、color management和 shadow都必须使用一致单位与空间。

## 8. 本章落点、验证与下一章

本章从 rendering equation推导了实时 microfacet BRDF。GGX \(D\) 描述微表面方向，Schlick \(F\) 描述视角相关界面反射，Smith \(G\) 描述 masking-shadowing；metallic workflow再分配 diffuse与specular能量。

在 STL/DirectX viewer中，本章对应 material buffers、BRDF shader、normal mapping与 environment maps。对 CT表面，它能把材料或分割标签映射为稳定可比较的可视化材质，而不是依赖固定 Phong高光。

本章的 90 分钟验证是实现单 direct light的 GGX shader，复现正文约 \(f_s=0.00433\)、\(f_d=0.2445\) 的标量结果，再扫描 roughness 0.1 到 1 和 view angle。预期是低 roughness高光更尖、grazing angle Fresnel增大；若跳过 sRGB decode，base color与能量会明显错误。

下一章会处理 light visibility与组织：shadow map的 depth comparison为什么产生 acne和Peter Panning，cascaded shadows怎样覆盖大场景，Forward+/clustered lighting又怎样让大量 lights不再逐像素全遍历。

