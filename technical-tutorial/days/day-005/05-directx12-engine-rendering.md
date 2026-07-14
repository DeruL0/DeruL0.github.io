# DirectX 12 第 5 章：为什么高模需要先切成 meshlets 才能真正由 GPU 驱动

## 1. 从一个真实任务开始

上一章已经把 object-level culling 和 draw command generation 移到 GPU。现在一个可见的工业 STL 仍可能包含一百二十万个 triangles。只要 object bounding box 进入视锥，传统 indexed draw 就把整个 mesh 交给 vertex/triangle pipeline；背面、遮挡区和屏幕上很小的局部仍会消耗几何处理。

今天的任务是把一个高模预处理成许多小型 triangle clusters，也就是 meshlets。每个 meshlet 有紧凑的局部 vertex/primitive 列表和独立 bounds，GPU 可以在簇级做 frustum、backface cone 和 occlusion culling；通过 DirectX 12 mesh shader pipeline，存活簇直接输出 vertices 与 primitives。

## 2. 最直接的办法，以及它为什么不够

最直接的办法是把整个 mesh 的 triangles 继续交给一个 indirect indexed draw。它已经消除了 CPU draw-call 瓶颈，却没有更细粒度的几何可见性。一个细长零件只露出 5%，vertex shader 仍处理全部 indexed vertices。

另一个办法是预先把 mesh 切成几千个独立 submeshes，并为每个 submesh 生成一个 DrawIndexed command。这能 cull 更细，但 argument buffer、draw setup 和状态管理重新增长；cluster 之间的 vertex duplication 和 CPU 资产结构也可能失控。

mesh shader 的关键不是“换一个新 shader stage”，而是让一个 threadgroup 以 meshlet 为自然工作单元：协作读取小簇、决定是否可见，并一次输出有限数量的 vertices 与 primitives。簇足够小，可以被独立剔除；又足够大，避免每 triangle 一个 command。

## 3. 关键想法是怎样被引出来的

一个 meshlet 保存两级索引。第一层是本簇使用的 global vertex IDs；第二层是每个 triangle 对这个小列表的 local indices。若一个 meshlet 少于 256 个 local vertices，triangle local index 可用 8 bits 表示，数据更紧凑。

每个 meshlet 还预计算：

~~~text
bounding sphere or box
normal cone
optional LOD error
material or section metadata
~~~

bounding volume 用于 frustum/occlusion culling；normal cone 汇总所有 triangle normals，若从当前视角看整个 cone 都背向相机，整簇可一次删除。这样 geometry pipeline 不必先运行每个 triangle 才知道它不可见。

DirectX mesh pipeline 将传统 input assembler、vertex shader 和 geometry shader 的固定顺序替换为可选 amplification shader、mesh shader 和 pixel shader。amplification shader 可生成或筛选 mesh shader workgroups；mesh shader 明确声明本组输出多少 vertices 与 primitives。

## 4. 一步一步建立正式模型

设 meshlet \(m\) 的 global vertex index 列表为

\[
V_m=[i_0,i_1,\ldots,i_{n_v-1}],
\]

local triangle 列表为

\[
P_m=[(a_0,b_0,c_0),\ldots],
\qquad
a_k,b_k,c_k<n_v.
\]

构建器限制

\[
n_v\le V_{\max},
\qquad
n_p\le P_{\max},
\]

使一个 mesh shader group 能在 API 与硬件输出上限内处理。实际常选几十个 vertices、约百个 triangles，具体目标应按硬件和资产测试。

frustum culling 与上一章相同。meshlet bounding sphere 中心 \(c_m\)、半径 \(r_m\)，若对任一 plane 有

\[
n_k^{\mathsf T}c_m+d_k<-r_m,
\]

整簇在视锥外。

normal cone 由轴 \(a\) 和 half-angle \(\theta\) 描述，所有 triangle normals 都在 cone 内。设从 meshlet 指向相机的单位方向为 \(v\)。cone 内 normal 与 \(v\) 的最大可能 dot product 为

\[
\max n^{\mathsf T}v
=
(a^{\mathsf T}v)\cos\theta
+
\sqrt{1-(a^{\mathsf T}v)^2}\sin\theta.
\]

若这个上界仍小于等于零，所有 triangles 都背向相机，meshlet 可保守 cull。若 meshlet 跨过尖锐折角，normal cone 很宽，这项测试会失去剔除能力；构建时应避免把朝向差异过大的 faces 强塞进同一簇。

通过测试后，mesh shader 读取 \(V_m\) 对应的 positions/attributes，执行变换，并把 local triangles 写入 primitive output。它必须先调用相应输出计数接口，且不超过 PSO 与 shader 声明的上限。被 cull 的 meshlet 输出零 primitives。

## 5. 跟着一个完整例子走到底

设一个高模有 120,000 triangles。离线 builder 把它切成 1,000 个 meshlets，每个平均 120 triangles，并为每簇保存约 64 个 unique vertex IDs、local primitive indices、bounding sphere 与 normal cone。

上一章的 object-level culling 判断整个高模可见，因此传统 indirect indexed draw 会提交全部

\[
120{,}000
\]

triangles。

mesh pipeline 中，amplification shader 为 1,000 个 meshlets 启动并行测试。假设：

- 700 个 meshlets 在视锥外；
- 180 个虽在视锥内，但 normal cone 证明全部背向；
- 40 个被 HZB 保守判定遮挡；
- 最后 80 个存活。

mesh shaders 最终输出约

\[
80\times120=9{,}600
\]

triangles，只是原几何的 8%。CPU 仍只提交一次 mesh dispatch 或 indirect mesh dispatch，没有为 80 个簇分别记录 draw。

看其中一个 meshlet。它的 local triangle \((3,7,11)\) 表示从 local vertex list 取得 global IDs

\[
(V_m[3],V_m[7],V_m[11]).
\]

mesh shader 只需为本簇 unique vertices 计算一次变换，再让多个 local triangles 复用输出。与把 120 个 triangles 存成完全独立 vertices 相比，局部索引保留了簇内复用。

若 normal cone 很宽导致 180 个背面 meshlets 无法剔除，问题可能在资产分簇，而不是 shader。重新构建时优先让相似 normals 进入同一 meshlet，可提高 cone culling，同时要避免过多 vertex duplication。这就是 asset preprocessing 与 runtime performance 的直接连接。

## 6. 回到真实系统：程序实际上怎样工作

完整管线分为离线与运行时两部分：

~~~text
offline meshlet builder
→ cluster triangles under vertex/primitive limits
→ generate local indices
→ compute bounds, normal cone and optional LOD error
→ serialize GPU-ready buffers

runtime
→ object-level GPU culling
→ meshlet amplification/culling
→ mesh shader emits visible geometry
→ pixel shader and normal render targets
~~~

meshlet buffers通常通过 SRV 读取，而不是传统 IA vertex/index binding。PSO 需要 mesh shader bytecode，并与传统 vertex shader pipeline 分开管理。frame graph 仍负责 depth/HZB 输入、visible meshlet lists 和资源状态。

对 STL，导入后应先完成 topology validation、去噪和 remeshing，再构建 meshlets。若原始 triangle soup 有重复 vertices 和随机 winding，meshlet builder 会得到差 bounds、宽 normal cone 和高 duplication。

Microsoft 的 DirectX mesh shader specification 定义 amplification/mesh shader 执行模型、payload、输出和限制，是实现 DX12 路径时的权威接口参考。[DirectX Mesh Shader Specification](https://microsoft.github.io/DirectX-Specs/d3d/MeshShader.html)

## 7. 容易走错的岔路

meshlet 越小不一定越好。小簇 culling 更精细，却增加 bounds、dispatch、payload 和 duplicated vertices；过大簇则剔除粗糙、threadgroup 工作不均。需要以实际场景测量。

只按 triangle adjacency 贪心聚类，可能把尖锐折角两侧放在一起，使 normal cone 接近半球，backface cone culling 几乎失效。构建 cost 应同时考虑 vertex reuse、空间紧凑和 normal coherence。

把 mesh shader 当作更快的 vertex shader 也会失望。若场景没有高几何密度、没有簇级 culling 或输出很小，传统 pipeline 可能更简单且同样快。

HZB culling 必须保守。meshlet bounds 过紧但计算错误会直接丢失 geometry；过松只损失性能。正确性优先于剔除率。

最后，meshlets 本身不是 LOD。它们只是可独立处理的 clusters；若远处仍输出每个存活 meshlet 的全部 triangles，几何总量仍可能过高。需要层次 LOD 或 virtual geometry 进一步选择分辨率。

## 8. 本章落点、验证与下一章

本章把 GPU-driven rendering 从 object 级推进到 geometry cluster 级。meshlet 的局部 vertex/primitive lists 提供紧凑复用，bounding volume 与 normal cone 允许整簇 culling，amplification/mesh shader 则在 GPU 上生成存活 primitives。性能取决于离线分簇与运行时剔除共同设计。

在 STL/DirectX 项目中，本章对应 meshlet asset builder、mesh-shader PSO 和 cluster diagnostics；在 CT surface viewer 中，它适合高分辨率等值面和分块更新后的局部 geometry。

本章的 90 分钟验证是把一个高模切成 meshlets，统计每簇 vertex/triangle 数、vertex duplication 和 normal-cone angle；在 DX12 中显示 object-visible、frustum-surviving、cone-surviving 和最终输出 primitive 数。预期是背面/局部不可见时输出 primitives 显著下降；故意只按 adjacency 聚类后，normal cone 变宽并降低剔除率。

下一章会继续到虚拟几何和层次 LOD：怎样把 meshlets 组织成 error-bounded hierarchy，按屏幕误差选择不同分辨率，并通过 streaming 只让当前需要的 clusters 驻留显存。

