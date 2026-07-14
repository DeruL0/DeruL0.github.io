# DirectX 12 第 8 章：一张阴影图为什么无法同时照顾脚边和地平线

## 1. 从一个真实任务开始

上一章建立了 GGX PBR，但 BRDF只回答“光到达表面后怎样反射”，没有回答光是否被遮挡。现在 STL/CT viewer需要一盏方向光展示细小孔洞和接触关系，游戏场景还包含数百个 local lights。我们既要稳定的远近阴影，也不能让每个 pixel遍历全部 lights。

今天的任务分成一条连续的实时光照链。方向光先从 light view渲染 depth shadow map；camera shading时把 world point变换到 light space并比较 depth。由于单张固定分辨率 shadow map覆盖整个 camera frustum时，近处 pixel获得的 texel太少，因此把 frustum切成 cascades。对于大量 point/spot lights，再用 screen tiles与depth slices建立 clustered light lists。

这些结构都在解决同一个系统问题：visibility和lighting work必须按屏幕上真正需要的空间范围分配，而不是对整个世界均匀支付成本。

## 2. 最直接的办法，以及它为什么不够

最直接的 shadow map用一张 2048×2048 depth texture覆盖 camera可见的全部世界区域。若覆盖宽度是 1000 m，一个 texel约代表 0.49 m；脚边 1 cm的零件特征不可能得到清晰阴影。若只紧密覆盖近处，远方则完全没有 shadow。

shadow comparison本身也有离散误差。surface在生成 shadow map与camera pass中以不同 sample位置和斜率栅格化，同一表面可能满足

\[
z_{\rm receiver}>z_{\rm shadow},
\]

于是错误地遮挡自己，形成 acne。增加 depth bias可消除 acne，但 bias过大时阴影与物体分离，形成 Peter Panning。

对 local lights，最直接的 forward shader循环所有 \(L\) 盏灯，成本近似 pixels乘以 \(L\)。即使一盏灯只影响屏幕角落，它也在所有 pixels上被测试。需要先做空间 culling，再让 pixel只访问所在区域的 light list。

## 3. 关键想法是怎样被引出来的

shadow map把 visibility变成从 light viewpoint看到的最近 depth。world point \(x\)经 light view-projection矩阵变换：

\[
q=VP_{\rm light}
\begin{bmatrix}
x\\1
\end{bmatrix}.
\]

透视除法后得到 shadow texture coordinates与 receiver depth：

\[
(u_s,v_s,z_s)
=
\left(
\frac{q_x}{q_w},
\frac{q_y}{q_w},
\frac{q_z}{q_w}
\right).
\]

若

\[
z_s-b
\le
D_{\rm shadow}(u_s,v_s),
\]

point被认为可见；\(b\) 是处理数值与斜率误差的 bias。

一张 map的问题不是 comparison逻辑错误，而是 world-space texel footprint随覆盖范围过大。camera近处需要高密度 shadow texels，远处可接受低密度。于是按照 camera depth切分 frustum，每段分别拟合一个 light-space orthographic projection，这就是 Cascaded Shadow Maps。

大量 lights也使用相似思想：先把 camera frustum划成 clusters，只把与 cluster bounding volume相交的 lights写入列表，shading时不再遍历全局 lights。

## 4. 一步一步建立正式模型

常用 cascade split把 uniform与logarithmic分割混合。camera near/far planes为 \(n,f\)，共 \(N\) 个 cascades，第 \(i\) 个 split为

\[
C_i
=
\lambda
\left[
n\left(\frac fn\right)^{i/N}
\right]
+
(1-\lambda)
\left[
n+(f-n)\frac iN
\right],
\qquad i=1,\ldots,N.
\]

当 \(\lambda\) 接近 1，更多 cascades集中在近处；接近 0则近似均匀深度。每个 subfrustum的八个 corners变换到 light space，求包围盒并建立 orthographic projection。

为了减小 camera微动时 shadow texels在世界中滑动，应把 light-space projection center或bounds量化到 shadow texel increments。cascade之间还可设置 overlap band并混合两个 visibility结果，避免分辨率突变形成 seam。

depth bias常由 constant与slope部分组成：

\[
b
=
b_0+b_s(1-n^{\mathsf T}l).
\]

表面越接近 grazing angle，栅格化depth变化越快，所需 slope bias越大。但这只是近似；normal offset、receiver-plane bias和更紧的 light near/far范围也可减少错误。

对 clustered lighting，屏幕先按 \(T_x\times T_y\) pixels划 tiles，再把 view-space depth按 logarithmic boundaries切成 \(N_z\) slices：

\[
z_k
=
n\left(\frac fn\right)^{k/N_z}.
\]

每个 cluster对应一个三维 frustum cell。compute pass测试 light sphere/cone与cluster bounds相交，输出 compact light indices。pixel shader根据 screen position与depth算出 cluster id，只遍历该 cluster的 lights。

## 5. 跟着一个完整例子走到底

设 camera范围为

\[
n=0.1\ {\rm m},
\qquad
f=1000\ {\rm m},
\]

使用四个 cascades和

\[
\lambda=0.9.
\]

纯 logarithmic splits为

\[
1,\ 10,\ 100,\ 1000.
\]

纯 uniform splits约为

\[
250.075,\ 500.05,\ 750.025,\ 1000.
\]

按混合公式，第一个 split为

\[
C_1
=0.9(1)+0.1(250.075)
=25.9075.
\]

第二个为

\[
C_2
=0.9(10)+0.1(500.05)
=59.005.
\]

第三个为

\[
C_3
=0.9(100)+0.1(750.025)
=165.0025.
\]

最后一个为

\[
C_4=1000.
\]

于是四段约为 0.1–25.9 m、25.9–59.0 m、59.0–165.0 m和165–1000 m。近处三个 cascades覆盖较短距离，把更多 2048² texels用于 camera附近；最后一张承担遥远区域。

再看 clustered lights。1920×1080 framebuffer使用 16×16 tiles，需要

\[
\left\lceil\frac{1920}{16}\right\rceil
\times
\left\lceil\frac{1080}{16}\right\rceil
=120\times68
=8160
\]

个 screen tiles。若 depth再分 24 slices，总 cluster数为

\[
8160\times24=195840.
\]

场景有 500 盏灯，但某个近处 cluster只与 12 盏相交，则该 cluster内的 pixel只评估 12 次 BRDF，而不是 500 次。若平均每 pixel从500次降到12次，light-loop work约减少

\[
\frac{500}{12}\approx41.7
\]

倍；代价是 compute culling、list memory与overflow管理。

## 6. 回到真实系统：程序实际上怎样工作

DX12 frame中的关键 passes可组织为：

~~~text
camera depth prepass
→ compute cascade splits and stable light matrices
→ render directional-light shadow depth array
→ build depth bounds per screen tile
→ cull local lights into clustered lists
→ material pass evaluates GGX using shadow visibility and local list
→ blend cascade transitions and tone-map
~~~

shadow depth array需要 DSV views用于逐 cascade写入，以及 SRV用于 shading读取。资源从 depth-write切换到 shader-resource时必须有正确 barriers；若多个 cascades并行记录 command lists，还要保证 descriptors与per-cascade constants生命周期稳定。

比较采样通常用 comparison sampler和PCF。PCF不是模糊原始 depth再比较，而是在多个邻域位置分别 comparison后平均可见比例。kernel越大，邻近 texels与当前 receiver plane不匹配的问题越明显，bias策略也更困难。

Microsoft 的 CSM说明把流程明确分为 frustum partition、每段 orthographic projection、分别渲染 shadow maps和camera pass选择对应 cascade，也讨论了 perspective aliasing、cascade blend、PCF与bias问题。[Microsoft Cascaded Shadow Maps](https://learn.microsoft.com/en-us/windows/win32/dxtecharts/cascaded-shadow-maps)

clustered-light buffers通常包含每 cluster的 offset/count和一个全局 packed index array。必须检测 capacity overflow；静默截断会让 lights随camera变化突然消失。可先做 count pass与prefix sum精确分配，或使用保守固定上限并记录overflow telemetry。

对 STL/CT viewer，方向光CSM可能只需2到3 cascades，因为场景尺度可控；大量local lights也未必需要 clustered pipeline。引擎架构应让 simple forward path与clustered path共享 BRDF与material数据，而不是为了“现代”强制支付复杂度。

## 7. 容易走错的岔路

增加 shadow map分辨率不能根治 perspective aliasing。若一张 map仍覆盖极大深度范围，近处与远处分配比例依然错误，只是总体更贵。

每帧紧密重算 cascade bounds会让分辨率利用率提高，却可能随camera轻微旋转发生尺度变化和shimmering。stable CSM常主动保留一些空白换取时间稳定性。

把 bias调到“当前场景不 acne”后写死会在不同尺度、light angle和cascade depth range下失效。bias应与 light-space depth precision、slope和world texel size联系。

Forward+只有 screen tiles，没有 depth slices。灯光在屏幕上覆盖很大但只存在于远处时，tile list仍会包含它；clustered lighting利用 depth进一步拒绝。相反，深度分片也增加内存和culling成本，不是所有场景都需要。

最后，cluster list正确不代表阴影正确。local light是否投射 shadow是另一项昂贵选择，通常需要 shadow atlas、cache或ray-traced visibility；不能把“灯被分到 cluster”误认为已经计算遮挡。

## 8. 本章落点、验证与下一章

本章把 PBR前缺失的 visibility与light organization补上。shadow map从light viewpoint保存最近 depth；bias处理离散自遮挡但会引入分离。CSM按camera depth重新分配shadow texels，clustered lighting则按screen tile和depth slice重新分配light-loop work。

在 DirectX 12 引擎中，本章对应 shadow depth resources、cascade matrices、comparison samplers、PCF、cluster culling compute pass和packed light lists。在 CT/STL viewer中，应根据真实场景尺度选择较简版本，同时保留相同的可见性诊断视图。

本章的 90 分钟验证是实现四 cascade debug view，复现正文 splits \(25.9075,59.005,165.0025,1000\)，并用不同颜色显示每个 cascade；移动camera观察是否 shimmer。随后建立 16×16×24 clusters，显示每 cluster light count heatmap。预期是近处阴影明显比单张全范围map清晰，pixel只遍历局部light list，overflow counter始终为零。

下一章将进入 DXR与混合渲染。shadow maps和screen-space light culling仍受栅格化可见性限制；ray tracing可以直接查询任意方向遮挡与反射，但必须通过 BLAS/TLAS、Shader Binding Table、denoising和raster hybrid budget才能进入实时引擎。
