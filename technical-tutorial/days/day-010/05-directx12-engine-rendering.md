# DirectX 12 第 10 章：资源不 resident 时，GPU 看到的不是慢资源，而是不可访问资源

## 1. 从一个真实任务开始

上一章把 DXR 加入实时管线，BLAS、TLAS、G-buffer、材质表和 denoiser history 都开始占用显存。现在场景换成高分辨率 CT/STL viewer 或大型游戏关卡：一个零件 mesh 有几千万三角形，多层 LOD、贴图、BLAS、历史 buffer 和调试 buffer 同时存在。即使 GPU 算力够，显存预算也可能先爆。

真实任务是让 viewer 在用户缩放、旋转和切换数据集时不卡顿。近处需要高精度 mesh 和高分辨率材质；远处或不可见部分可以降级、卸载或异步加载。DirectX 12 不再把这些细节全部交给驱动。应用必须知道资源在哪个 heap、是否 resident、什么时候上传、什么时候释放，以及 GPU 是否还在使用。

今天的主题是 residency 和 streaming。资源不是“存在于程序里就能被 GPU 访问”。在 D3D12 中，GPU 可访问内存、heap、resource state、descriptor 和同步 fence 都要共同正确，资源才真正可用。

## 2. 最直接的办法，以及它为什么不够

最直接的做法是启动时把所有 mesh、textures、BLAS 和 buffers 都加载到 GPU。小 demo 可行，但真实数据会超过 video memory budget。超过预算后，系统可能触发 paging、stutter，甚至 allocation failure。对 DXR，BLAS 占用也可能和 mesh vertex/index buffer 同时存在，显存翻倍得很快。

第二个办法是每帧临时创建和销毁资源。它让内存看似按需使用，却会造成 heap 碎片、同步等待和 CPU overhead。GPU 可能还在上一帧使用一个 resource，CPU 就想释放或复用它；没有 fence 保护就会产生未定义行为。

第三个办法是只按 CPU 侧可见性加载资源。CPU frustum culling 不知道未来几帧相机运动、GPU 队列延迟和 streaming IO 时间。等物体进入视野再加载，用户已经看到空洞或低清占位。

因此需要把资源管理建成一个时间系统：预测需求、异步加载、分配 heap、上传数据、转换状态、使其 resident、绑定 descriptor，并在 GPU 不再使用后回收。

## 3. 关键想法是怎样被引出来的

D3D12 把资源分成 committed、placed 和 reserved 等形式。Committed resource 自带 heap，简单但分配粒度粗；placed resource 放在显式 heap 的 offset 上，适合 suballocation；reserved resource 只保留虚拟地址空间，通过 tile mapping 映射物理内存，适合大型稀疏资源。

Residency 的关键是：一个对象 resident 时，GPU 才能访问它对应的物理内存。Microsoft 文档明确把 memory management 和 residency 作为 D3D12 程序需要显式处理的内容，并提供 `MakeResident`、`Evict`、budget 查询等接口。[Memory Management in Direct3D 12](https://learn.microsoft.com/en-us/windows/win32/direct3d12/memory-management) [Residency](https://learn.microsoft.com/en-us/windows/win32/direct3d12/residency)

Streaming 则把资源按优先级和 mip/LOD 分层。近处物体需要高 mip 和高 LOD；远处物体只需要低 mip 或简化 mesh；不可见但可能很快出现的物体要预取。DirectStorage 进一步降低大量小 IO 请求的 CPU 开销，目标是让高速存储更适合游戏式 asset streaming。[DirectStorage Overview](https://learn.microsoft.com/en-us/gaming/gdk/docs/features/console/storage/directstorage/directstorage-overview)

## 4. 一步一步建立正式模型

先定义每帧显存预算。适配器给出当前预算

\[
B_{\rm budget},
\]

应用当前使用量为

\[
B_{\rm used}.
\]

为了避免刚好顶到预算造成抖动，设置安全余量

\[
B_{\rm target}
=
\rho B_{\rm budget},
\qquad
0<\rho<1.
\]

资源 \(i\) 有大小

\[
s_i,
\]

优先级

\[
p_i,
\]

以及预计下一次使用时间

\[
\Delta t_i.
\]

一个简单 streaming 决策是保留高价值资源，使

\[
\sum_{i\in R}s_i\le B_{\rm target}.
\]

当需要释放空间时，优先 evict 低优先级、远期才用或可快速重建的资源。真实系统不会每帧解完整 knapsack，但这个模型说明 residency manager 的核心不是 LRU 一个规则，而是大小、重要性和未来使用的折中。

上传路径也有模型。CPU 把数据写入 upload heap，GPU copy 到 default heap：

\[
\text{disk/system memory}
\rightarrow
\text{upload heap}
\rightarrow
\text{default heap resource}.
\]

copy 完成前，default resource 不能被 graphics/compute 使用；使用完前，heap 空间不能被复用。同步由 fence value 表示：

\[
\text{resource free only if } F_{\rm completed}\ge F_{\rm lastUse}.
\]

这条不等式是资源回收的安全条件。没有它，内存管理看似正确，GPU 实际可能还在读旧数据。

## 5. 跟着一个完整例子走到底

设 GPU 当前 video memory budget 为

\[
B_{\rm budget}=8\ {\rm GB},
\]

应用选择

\[
\rho=0.85,
\]

则目标使用量为

\[
B_{\rm target}=6.8\ {\rm GB}.
\]

当前 resident 资源为：高精度 CT mesh 3.0 GB，BLAS 1.5 GB，textures 1.0 GB，G-buffer 和 history 1.0 GB，调试 buffers 0.6 GB，总计

\[
B_{\rm used}=7.1\ {\rm GB}.
\]

这已经超过目标 0.3 GB。现在用户打开第二个零件，需要加载 1.2 GB 的 mesh LOD0 和 0.6 GB 的 BLAS。若直接加载，总量变为

\[
7.1+1.8=8.9\ {\rm GB},
\]

超过预算。

streaming manager 先选择第二零件的 LOD2：mesh 0.25 GB，BLAS 0.15 GB，总计 0.4 GB。同时 evict 调试 buffers 0.6 GB，并把第一个零件远处部分从 LOD0 降到 LOD1，释放 0.5 GB。新使用量为

\[
7.1-0.6-0.5+0.4=6.4\ {\rm GB}.
\]

这样低于目标。随后后台逐步加载 LOD1 或 LOD0；只有当相机靠近且预算允许，才把高精度资源 make resident 并切换 descriptor。若 copy queue 上传 LOD0 的 fence value 是 120，graphics queue 只有在等待或确认

\[
F_{\rm copy}\ge120
\]

后才能绑定它；旧 LOD 的 heap 空间也必须等 graphics fence 完成后才能回收。

这个例子说明 streaming 的输出不是“资源加载成功”四个字，而是每帧的 LOD 选择、预算、resident set 和 fence 安全状态。

## 6. 回到真实系统：程序实际上怎样工作

一个可维护的 DX12 resource system 通常包含 allocator、upload manager、residency manager、descriptor manager 和 streaming scheduler。allocator 负责 heap suballocation；upload manager 维护 ring buffer 和 copy queue；residency manager 跟踪 budget、priority 和 `MakeResident`/`Evict`；descriptor manager 保证 shader 看到的 SRV/UAV/CBV 指向当前 resident resource；scheduler 根据相机和任务预测未来需要。

Placed resources 能减少 committed resource 的碎片和创建成本，但需要自己处理 alignment、heap type 和 lifetime。D3D12 Memory Allocator 这类库的价值就在于把 suballocation 的繁琐规则封装起来，但它不替你决定哪些资源应该 resident。

对 DXR，BLAS/TLAS 也要纳入 streaming。低 LOD mesh 应有对应低成本 BLAS；高 LOD BLAS build 可能放到 async compute 或 loading phase。切换 LOD 时，TLAS instance 指向的 acceleration structure、材质表和 vertex/index buffers 必须同步更新。

对 CT/STL viewer，资源调度可以更领域化。用户当前截面、相机距离和测量工具决定需要高精度的区域；远处可以用 decimated mesh 或 point proxy。若要浏览巨大体数据，reserved/tiled resources 或 bricked volume streaming 会比一次性上传整个 volume 更合理。

## 7. 容易走错的岔路

第一个误区是把显存超预算当作“只是变慢”。在显式 API 中，资源 residency 和同步错误可能直接导致访问失败、device removed 或严重 stutter。预算是运行时约束，不是建议。

第二个误区是只管理 textures，不管理 geometry 和 acceleration structures。DXR 场景中 BLAS/TLAS、scratch buffers 和 compaction 后 buffers 都可能很大。

第三个误区是释放 CPU handle 就认为 GPU 不再使用。GPU 队列是异步的，必须用 fence 证明最后一次使用已经完成。

第四个误区是 streaming 只按距离。材质重要性、遮挡、用户交互焦点、测量工具和即将到来的相机运动都可能改变优先级。

最后，不要在切换 LOD 时忽略 temporal stability。mesh、normal、material 和 BLAS 同时跳变会造成闪烁；需要 hysteresis、渐进切换或 history invalidation。

## 8. 本章落点、验证与下一章

本章把 DX12 资源从静态对象推进到运行时 resident set。GPU 只有在资源所在内存 resident、状态正确、descriptor 有效且同步完成时才能安全访问。Streaming 是在预算内选择合适 LOD、异步上传并用 fence 管理生命周期。

在 DirectX 12 和 CT/STL viewer 项目中，本章对应 heap allocator、upload ring、residency budget、LOD mesh/BLAS、descriptor 切换、copy queue 和 fence-based reclamation。它是大型模型和 DXR 场景能否稳定运行的基础。

本章的 90 分钟验证是实现一个资源预算模拟器：输入若干 mesh、texture、BLAS 的大小和优先级，设 \(B_{\rm budget}=8\) GB、\(\rho=0.85\)，模拟加载第二零件时的 LOD 选择和 evict 决策。预期是系统先加载低 LOD、释放低优先级资源，并且任何资源回收都等待对应 fence 完成。

下一章将进入 ECS、任务系统和资产管线。资源 streaming 解决了“什么数据在 GPU 上”，但大型引擎还需要决定谁发起加载、谁拥有组件状态、哪些任务能并行，以及资产从磁盘到运行时对象的完整生命周期。
