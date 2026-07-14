# DirectX 12 第 9 章：DXR 为什么不是替代光栅化，而是给光栅化补上射线查询

## 1. 从一个真实任务开始

上一章用 shadow maps 和 clustered lighting 组织了实时光照。它们适合大量直接光和相机可见表面的 shading，但仍有一个边界：shadow map 只能从光源视角保存有限深度，screen-space reflection 只能反射屏幕上已有内容，ambient occlusion 也常被局部近似限制。只要需要查询“沿任意方向第一块几何在哪里”，光栅化的屏幕投影就不够。

今天的真实任务是在 DX12 viewer 或游戏引擎中加入两类效果：接触阴影和反射。接触阴影需要从 shading point 向光源发一条短 ray，判断中间是否有遮挡；反射需要沿反射方向找最近命中，即使命中的物体当前不在屏幕上。我们不想把整个 renderer 改成离线路径追踪，而是保留 G-buffer、PBR、shadow map 和 temporal pipeline，只在光栅化缺信息的地方调用 DXR。

因此 DXR 的入口不是“画三角形的另一种方式”，而是让 GPU 能高效遍历一份三维场景加速结构，并把 ray 命中交给可编程 shader。

## 2. 最直接的办法，以及它为什么不够

最直接的反射方案是 screen-space reflection。它从当前 pixel 的 depth buffer 出发，在屏幕空间 marching，遇到已有 depth 就认为命中。SSR 很快，也自然继承 G-buffer 材质。但如果反射物体在屏幕外、被前景遮挡、或 depth buffer 没有背面信息，ray 就找不到真实命中。水面边缘和金属边缘会出现断裂。

最直接的软阴影方案是提高 shadow map 分辨率或加 PCF。它能改善滤波，但仍依赖 light-space rasterization。非常近的接触阴影、细小遮挡物和屏幕外遮挡者仍容易出错。把所有光都做高分辨率 shadow atlas 又会爆内存和 draw cost。

另一个直接办法是在 CPU 上建 BVH 并做 raycast，再把结果传给 GPU。这适合编辑器拾取，不适合每帧每像素。实时 shading 需要 ray traversal 与 shading 数据都在 GPU 上，且与 DX12 资源、descriptor、command queue 和 barrier 模型一致。

## 3. 关键想法是怎样被引出来的

DXR 引入两个核心对象。第一是 acceleration structure。底层 BLAS 描述 mesh geometry，例如 vertex buffer 和 index buffer；顶层 TLAS 描述实例，每个 instance 引用某个 BLAS，并带 transform、instance id 和 hit group 索引。这样静态网格可以构建一次 BLAS，多次以不同 transform 出现在 TLAS 中。

第二是 shader table。光栅化 draw call 可以按 material 分批，因为你知道当前 draw 的 mesh 是什么；ray tracing 不知道 ray 会击中哪个物体，所以需要在命中后根据 instance 和 geometry 找到对应 shader record。shader table 把 ray-generation、miss、hit group 与 local root arguments 组织起来。

Microsoft 对 DXR 的介绍把 acceleration structure、`DispatchRays`、新的 HLSL shader 类型和 raytracing pipeline state 列为 DX12 中的新增概念，并说明 acceleration structure 使用两级层次；官方 functional spec 进一步规定了 shader table record、stride 和 indexing 规则。[Announcing Microsoft DirectX Raytracing](https://devblogs.microsoft.com/directx/announcing-microsoft-directx-raytracing/) [DXR Functional Spec](https://microsoft.github.io/DirectX-Specs/d3d/Raytracing.html)

## 4. 一步一步建立正式模型

一条 ray 由 origin、direction 和参数范围定义：

\[
r(t)=o+t d,
\qquad
t_{\min}\le t\le t_{\max}.
\]

shadow ray 的目标不是颜色，而是 visibility。若从表面点 \(x\) 到光源方向 \(l\) 的距离为 \(D\)，可设

\[
o=x+\epsilon n,\qquad d=l,\qquad t_{\max}=D-\epsilon.
\]

如果 traversal 在这个范围内找到任何不透明命中，则

\[
V(x,l)=0;
\]

否则

\[
V(x,l)=1.
\]

反射 ray 则需要最近命中。给定 view direction \(v\) 和 normal \(n\)，反射方向为

\[
d_{\rm refl}=v-2(n\cdot v)n.
\]

DXR traversal 在 TLAS 中先测试 instance bounds，再进入对应 BLAS 测试 triangles。命中后，closest-hit shader 可根据 barycentric coordinates 取顶点属性：

\[
P=(1-\beta-\gamma)P_0+\beta P_1+\gamma P_2.
\]

在 API 层，`DispatchRays` 描述了 ray-generation、miss、hit group 和 callable shader table 的 GPU 地址、stride 和 size。shader table record 至少包含 shader identifier，后面跟 local root arguments。若 indexing 错或 record stale，ray 仍会 dispatch，但命中执行的 shader 或资源就是错的。

实时混合渲染还必须定义预算。若每 pixel 发一条 reflection ray，1080p 每帧约

\[
1920\times1080\approx2.07\times10^6
\]

条 ray。若再加多光源阴影和多 bounce，成本会立刻超过实时预算。因此常见策略是只对高 roughness 以下的材质发 reflection ray，只对近距离 contact shadow 发短 ray，并把结果交给 temporal accumulation 和 denoiser。

## 5. 跟着一个完整例子走到底

考虑一个地面点

\[
x=(0,0,0),
\]

法线

\[
n=(0,1,0),
\]

点光源在

\[
L=(0,10,0).
\]

从表面到光源的方向为

\[
l=(0,1,0),
\]

距离

\[
D=10.
\]

为避免自相交，ray origin 取

\[
o=x+0.01n=(0,0.01,0),
\]

参数范围为

\[
0\le t\le9.99.
\]

如果 TLAS 中有一个小盒子在 \(y=3\) 处被 ray 命中，则 shadow ray 不需要知道盒子的材质，只要 any-hit 或 closest-hit 确认它不透明，就返回

\[
V=0.
\]

若没有命中，则

\[
V=1.
\]

再看反射。设相机方向从表面指向相机为

\[
v=\frac{1}{\sqrt2}(0,1,1),
\]

则按公式

\[
d_{\rm refl}
=
v-2(n\cdot v)n
=
\frac{1}{\sqrt2}(0,-1,1).
\]

这条 ray 指向屏幕中未必可见的位置。SSR 如果当前 depth buffer 没有该方向上的物体，就会 miss；DXR 则在 TLAS/BLAS 中遍历完整场景。命中一个金属球后，closest-hit shader 读出其 material id，返回反射颜色或 hit information，再由主光栅化 pass 的 PBR shader 混合。

一个最小 DXR frame 的工程顺序是：构建或更新 BLAS；构建 TLAS；绑定 raytracing pipeline state 和 global root signature；填充 shader table；把 G-buffer、TLAS、输出 UAV 和材质表放入 descriptor；调用 `DispatchRays`；最后把 ray result 输入 denoising 或 composite pass。每一步都对应明确资源状态，不能只把它当成一个 shader 函数调用。

## 6. 回到真实系统：程序实际上怎样工作

在 DX12 引擎中，BLAS 构建通常放在 asset load 或 mesh 更新时。静态 mesh 可用 build 后 compaction 降低显存；skinned mesh 或变形 mesh 可能需要 refit 或 rebuild。TLAS 每帧根据可见实例、transform 和 instance mask 更新。instance mask 能让不同 ray types 查询不同对象，例如 reflection 忽略 invisible helper geometry，shadow 查询 alpha-tested foliage 时走 any-hit。

Shader Binding Table 是最容易出错的部分之一。每个 record 的 shader identifier、local root arguments、stride 对齐和 hit group index 都必须与 pipeline state 一致。材质表最好通过 bindless descriptor 或 structured buffer 间接索引，避免为每个物体塞过大的 local root arguments。

混合渲染常把 DXR 做成独立 passes。G-buffer pass 仍由 raster 生成 position、normal、roughness、material id。DXR reflection pass 根据这些 buffer 发 ray，输出 hit radiance 和 hit distance。Denoiser 使用 normal、depth、motion vector 和 history 做时域累积。最后 composite pass 按 roughness、Fresnel 和 confidence 混合 SSR、DXR 和 fallback environment。

对 STL/CT viewer，DXR 可以先从 debug 功能开始：pick ray、ambient occlusion 或单光源 contact shadow。不要一开始做 full path tracing。CT 模型常有大量细三角形，BLAS build cost 和显存可能成为瓶颈；需要 mesh simplification、instance organization 和 build flags 的取舍。

## 7. 容易走错的岔路

第一个误区是认为 DXR 会自动给出电影级路径追踪。DXR 只提供加速结构遍历和 ray shader 机制；采样策略、材质模型、denoising、temporal stability 和预算仍由引擎负责。

第二个误区是忽略 acceleration structure 更新成本。静态场景很适合 BLAS compaction；大量 deforming geometry 每帧 rebuild 会吞掉预算。需要区分 static、rigid dynamic、skinned 和 procedural geometry。

第三个误区是 shader table 当成普通常量 buffer。它的 record stride、alignment、shader identifier 和 local root 参数必须严格匹配。错位不会总是立刻崩溃，可能表现为随机材质或偶发 miss。

第四个误区是每个 pixel 发太多 ray，然后指望 denoiser 修复。denoiser 能降噪，但不能恢复系统性错误，例如法线错、motion vector 错、history rejection 错或 hit distance 不稳定。

最后，DXR 不是 SSR、shadow map 和 probe 的简单替代。实时引擎通常混合使用：SSR 处理便宜且稳定的屏幕内反射，DXR 补屏幕外或关键材质，probe 提供远场 fallback。预算来自分层组合，而不是单一算法。

## 8. 本章落点、验证与下一章

本章把 DXR 放回实时引擎的位置：它给光栅化补上任意方向的三维 ray query。BLAS 描述 mesh，TLAS 描述实例，shader table 把命中对象映射到 shader record；`DispatchRays` 运行 ray-generation、miss、any-hit 和 closest-hit shader。实时效果必须通过 ray budget、denoising 和 raster hybrid 组织。

在 DirectX 12 项目中，本章对应 acceleration-structure builder、SBT layout、raytracing PSO、reflection/contact-shadow passes、descriptor 管理和 denoiser 输入。在 CT/STL viewer 中，它适合先做拾取、AO 或接触阴影，再逐步扩展到反射。

本章的 90 分钟验证是实现一个最小 DXR shadow ray pass：用一个平面、一个盒子和一盏点光源构建 BLAS/TLAS，从 G-buffer 表面点向光源发 shadow ray，输出黑白 visibility buffer。预期是盒子下方出现接触阴影；移动盒子只需更新 TLAS transform 或相关 BLAS，visibility buffer 跟随变化。

下一章将进入资源驻留与流式加载。DXR、virtual geometry、高分辨率 CT mesh 和材质贴图都会让 GPU memory 成为帧级资源调度问题；引擎必须知道哪些资源本帧常驻，哪些可以异步加载、压缩或降级。
