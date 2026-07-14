# DirectX 12 第 11 章：ECS 和任务图为什么要接在资源 streaming 后面

## 1. 从一个真实任务开始

上一章解决了资源是否 resident、何时上传、何时回收。现在问题变成：谁发起这些操作？一个 CT/STL viewer 或游戏引擎中，模型有 transform、mesh、material、BLAS、LOD、selection state、测量标注和 streaming 状态。相机移动时，有些对象要更新 bounds，有些要加载 LOD，有些要重建 BLAS，有些要写入 TLAS，有些只要渲染。

真实任务是组织一个可以交互浏览大型 CT mesh 的 DX12 应用。若每个对象都是一个巨大的 C++ class，里面包含渲染、物理、UI、资源和加载逻辑，很快会出现互相调用、锁、生命周期不清和难以并行的问题。资源 streaming 已经让数据跨越 CPU IO、GPU copy、graphics queue 和 fence；再用临时脚本式调用会失控。

今天的主题是 ECS、任务图和资产管线。ECS 负责把对象状态拆成可查询的数据；任务图负责把每帧工作按依赖并行调度；资产管线负责把源文件变成运行时可加载的 cooked resources。

## 2. 最直接的办法，以及它为什么不够

最直接的架构是面向对象层级：`MeshObject` 继承 `SceneObject`，再派生出 `CTObject`、`AnimatedObject`、`SelectableObject`。每个对象自己更新、自己加载、自己渲染。小程序里直观，但一旦需要批量 culling、批量上传、并行更新和统一渲染排序，就会被虚函数、指针跳转和混合职责拖慢。

第二个办法是把所有逻辑塞进 render loop：遍历对象，if 判断是否有 mesh、是否 selected、是否需要 BLAS。它初期简单，但状态越来越多时，循环变成条件墙，任何系统都可能修改任何对象。

第三个办法是只建一个任务线程池。线程池能并行执行函数，却不知道哪些任务读写哪些数据、哪些任务必须先完成、哪些资源还被 GPU 使用。没有依赖图，parallelism 会变成 race condition。

因此需要两个分离但配合的抽象：ECS 描述“数据在哪里，哪些实体有这些数据”；任务图描述“本帧哪些工作能并行，哪些工作依赖哪些结果”。

## 3. 关键想法是怎样被引出来的

ECS 把对象身份和数据拆开。Entity 是一个 id；Component 是纯数据，例如 `Transform`、`MeshRef`、`MaterialRef`、`Bounds`、`GpuResidentState`；System 是对拥有某些组件的实体集合执行的逻辑。EnTT 文档中核心 registry 就围绕 entity 和 component storage 组织，Flecs 也把 queries 作为查找匹配组件实体的核心接口。[EnTT ECS](https://github.com/skypjack/entt/wiki/Entity-Component-System) [Flecs Queries](https://www.flecs.dev/flecs/md_docs_2Queries.html)

任务图则把每帧工作变成有向无环图。若任务 \(B\) 需要任务 \(A\) 的输出，就有边

\[
A\rightarrow B.
\]

调度器可以并行执行没有依赖关系的任务。Unreal 的 Tasks System 文档也把它描述为支持构建和运行带依赖的 DAG 的异步任务系统；Taskflow 等 C++ 库同样以 task graph 表达并行模式。[Unreal Tasks System](https://dev.epicgames.com/documentation/unreal-engine/tasks-systems-in-unreal-engine) [Taskflow](https://taskflow.github.io/)

资产管线则把 runtime 之前的混乱源文件变成可预测数据：导入、清洗、生成 LOD、构建 meshlet 或 BLAS 数据、压缩贴图、生成 metadata，最后写成 runtime package。这样 runtime 不需要理解所有源格式和昂贵处理。

## 4. 一步一步建立正式模型

先定义实体和组件。实体集合为

\[
\mathcal E=\{e_1,e_2,\ldots,e_N\}.
\]

组件类型 \(C\) 对一部分实体有值：

\[
C:e\mapsto c_e.
\]

一个系统需要组件集合

\[
\mathcal Q=\{C_1,C_2,\ldots,C_m\}.
\]

它的查询结果是

\[
\operatorname{Query}(\mathcal Q)
=
\{e\in\mathcal E\mid e\text{ has all }C_1,\ldots,C_m\}.
\]

例如 render extraction system 查询 `Transform`、`MeshRef`、`MaterialRef` 和 `GpuResidentState`，只处理已经 resident 的可渲染实体。

任务图定义为

\[
G=(T,D),
\]

其中 \(T\) 是 tasks，\(D\) 是 dependencies。任务 \(t\) 有成本

\[
c(t).
\]

若无限多 worker，frame time 仍不可能小于 critical path：

\[
T_{\rm frame}\ge
\max_{\pi\in{\rm paths}(G)}
\sum_{t\in\pi}c(t).
\]

这说明任务图优化的目标不是把所有工作拆到最碎，而是减少关键路径和避免不必要依赖。

资产管线可建模为 source 到 cooked asset 的函数：

\[
\text{CookedAsset}
=
\operatorname{Cook}(\text{SourceAsset},\text{ImportSettings},\text{TargetPlatform}).
\]

runtime 只依赖 cooked asset 的稳定 schema，例如 vertex buffer layout、LOD table、material descriptors、BLAS build info 和 streaming chunks。

## 5. 跟着一个完整例子走到底

考虑三个实体。实体 \(e_1\) 是 CT 零件，拥有 `Transform`、`MeshRef`、`MaterialRef`、`Bounds`、`ResidentLOD`。实体 \(e_2\) 是测量标注，只拥有 `Transform`、`Annotation`。实体 \(e_3\) 是光源，拥有 `Transform`、`Light`。

渲染系统查询

\[
\{\text{Transform},\text{MeshRef},\text{MaterialRef},\text{ResidentLOD}\}.
\]

结果只包含

\[
\{e_1\}.
\]

光照系统查询

\[
\{\text{Transform},\text{Light}\}
\]

结果只包含

\[
\{e_3\}.
\]

这比在对象继承树里问“你是不是可渲染对象”更直接，因为系统只关心数据组合。

现在构造一帧任务图。`UpdateTransforms` 需要 3 ms；`CullVisible` 需要 2 ms；`PrepareUploads` 需要 2 ms；`BuildCommandLists` 需要 2 ms；`Render` 需要 8 ms。`UpdateTransforms` 和 `PrepareUploads` 可以并行；`CullVisible` 依赖 `UpdateTransforms`；`BuildCommandLists` 依赖 `CullVisible` 和 `PrepareUploads`；`Render` 依赖 `BuildCommandLists`。

串行时间为

\[
3+2+2+2+8=17\ {\rm ms}.
\]

并行后，关键路径是

\[
UpdateTransforms(3)
\rightarrow
CullVisible(2)
\rightarrow
BuildCommandLists(2)
\rightarrow
Render(8),
\]

即

\[
15\ {\rm ms}.
\]

`PrepareUploads` 的 2 ms 被隐藏在前面任务旁边。若上传任务变成 6 ms，关键路径会改为

\[
PrepareUploads(6)
\rightarrow
BuildCommandLists(2)
\rightarrow
Render(8)
=16\ {\rm ms}.
\]

这个例子说明任务图的性能由依赖和关键路径决定，不是由任务数量本身决定。

## 6. 回到真实系统：程序实际上怎样工作

一个实际 DX12 viewer 可以把 runtime 分成 world、asset、render 和 job 四层。World 层保存 ECS registry；asset 层处理 source import、cooked package、streaming request 和 residency；render 层从 ECS 提取可见对象，生成 render items、TLAS instances 和 command lists；job 层执行 CPU 并行任务并通过 fences 与 GPU 队列同步。

资产管线应在离线或后台阶段生成 LOD、bounding volumes、meshlet、collision proxy、thumbnail、material metadata 和可选 BLAS build data。运行时加载的是 chunk，不是原始 STL 或 OBJ 的任意文本解析结果。这样 streaming manager 才能按上一章的预算模型选择资源。

ECS 的组件应避免直接持有复杂 GPU 对象。更稳妥的是持有 handles，例如 `MeshHandle`、`MaterialHandle`、`ResidentLOD`。真实 D3D12 resource 由 resource manager 拥有，生命周期由 fence 和 residency 控制。否则 entity 删除可能错误释放仍被 GPU 使用的资源。

对 CT/STL 工具，ECS 不必过度复杂。少量组件就够：transform、mesh asset、material、selection、measurement、streaming state、BLAS state。价值在于让 culling、selection、render extraction、measurement update 和 streaming 请求各自成为清楚的系统。

## 7. 容易走错的岔路

第一个误区是为了 ECS 把所有东西拆成过小组件。组件太碎会让查询和同步复杂化。拆分应围绕系统访问模式，而不是形式主义。

第二个误区是在组件里放所有行为。ECS 的价值是数据和系统分离；如果每个组件又开始调用渲染和加载逻辑，就回到了混合对象。

第三个误区是任务越多越好。过细任务会增加调度开销，并让依赖图难以理解。任务粒度应大到足以覆盖调度成本，小到能暴露关键并行性。

第四个误区是忽略 GPU 异步。CPU 任务完成不代表 GPU 使用完成；资源销毁、LOD 切换和 descriptor 更新仍要遵守 fence。

最后，资产管线不能只服务渲染。测量、选择、碰撞、BLAS、缩略图和元数据都应从同一个 cooked asset schema 派生，否则各系统会用不同版本的几何。

## 8. 本章落点、验证与下一章

本章把上一章的资源 streaming 接到引擎架构。ECS 用 entity id 和组件数据组织场景；系统通过查询处理匹配实体；任务图用 DAG 表达每帧依赖和并行；资产管线把源文件变成 runtime 可加载、可 streaming、可渲染的 cooked assets。

在 DirectX 12 和 CT/STL viewer 项目中，本章对应 ECS registry、component schema、render extraction、streaming requests、job graph、asset cook step 和 GPU resource handle 生命周期。

本章的 90 分钟验证是实现一个小型 ECS 数据表：三个实体、六种组件，写两个 query，确认 render system 只选中 mesh 实体、light system 只选中光源实体。再画出正文五个任务的 DAG，计算串行 17 ms 与并行关键路径 15 ms。预期是任务图能隐藏无依赖上传工作，但关键路径仍限制帧时间。

下一章将进入时域渲染与重建。ECS 和任务图组织了一帧内的工作；实时渲染还会跨帧累积 history、motion vector、TAA、denoising 和 temporal upsampling，下一步需要把时间维度纳入渲染质量与稳定性。
