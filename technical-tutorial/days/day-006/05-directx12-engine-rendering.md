# DirectX 12 第 6 章：虚拟几何怎样让“看得见多少”决定“显存里放多少”

## 1. 从一个真实任务开始

上一章把高模切成 meshlets，并在 GPU 上做簇级 culling。但一个大型工业装配、城市或超高分辨率 CT 表面可能包含数亿 triangles。即使只绘制可见 meshlets，完整最高精度数据仍放不进显存；远处 meshlets 虽然可见，输出全部细节也只会落在少数 pixels 中。

今天的任务是建立层次化虚拟几何。离线阶段把 meshlets 组织成从粗到细的 hierarchy，每个 node 保存几何近似和误差；运行时根据 screen-space error 选择一组 nodes，形成当前视角需要的几何 cut。只有 cut 引用的数据 pages 需要驻留 GPU，缺失细节则暂时回退到已驻留 ancestor。

这样，几何复杂度由屏幕需求和显存预算共同决定，而不再等于资产最高 triangle count。

## 2. 最直接的办法，以及它为什么不够

最直接的 LOD 是为整个 object 制作 high、medium、low 三个 meshes，按距离切换。它实现简单，但一个细长物体可能一端靠近、一端很远，整个 object 只能选择一个 level；切换点还会产生明显 popping。

另一个直接办法是始终保留最高 LOD 在显存，只在 shader 中跳过远处 triangles。这减少 raster work，却没有减少 storage 和 transfer；GPU 仍需访问或至少保留完整数据。

streaming 若只在需要时才请求 fine page，又没有 fallback，会在上传完成前出现孔洞。若 CPU 每帧等待 copy queue 完成，异步 streaming 又变成卡顿。

所以系统需要三件东西同时成立：可局部选择的误差层次、虚拟 page 到物理 memory 的映射，以及永远可用的 coarse fallback。

## 3. 关键想法是怎样被引出来的

meshlet hierarchy 的每个 parent 是一组 children 的简化近似。parent 记录 object-space geometric error \(\epsilon\)：真实 children surface 与 parent approximation 的最大或保守距离。

观察距离越远，同样 \(\epsilon\) 投影到屏幕越小。runtime 只要把 object-space error 转成 pixel error，就能判断 parent 是否足够；若足够，children 无需访问。

层次选择结果不是任意 nodes 集合，而是 tree cut：从 root 到每个 leaf 的路径恰好选择一个 node。这样 surface 被覆盖一次，不会同时画 parent 与 children，也不会漏掉整个 branch。

streaming 把 node data 分组到固定大小 pages。GPU 使用 stable virtual page ID，page table 决定它当前映射到哪个 physical slot。缺页时 selector 仍可选择 resident ancestor，request system 在后台上传 fine page，完成后下一帧再细化。

## 4. 一步一步建立正式模型

设 camera vertical field of view 为 \(\phi\)，viewport 高度为 \(H\) pixels。焦距的 pixel 表示为

\[
f_{\rm px}
=
\frac{H}{2\tan(\phi/2)}.
\]

node 的 object-space error 为 \(\epsilon\)，到 camera 的保守深度为 \(z\)。小角度近似下，screen error 为

\[
e_{\rm px}
\approx
\frac{f_{\rm px}\epsilon}{z}.
\]

若

\[
e_{\rm px}\le\tau,
\]

parent approximation 满足 pixel threshold，可停止展开；否则访问 children。bounds 应使用 node 最近可能深度，而不是 center depth，以免低估靠近相机部分的误差。

一个合法 cut \(\mathcal C\) 需要满足：对每条 root-to-leaf path，恰有一个 node 位于 \(\mathcal C\)。总预计 geometry work 可写成

\[
T(\mathcal C)
\approx
\sum_{n\in\mathcal C}
t_n,
\]

resident bytes 为

\[
B(\mathcal C)
=
\left|
\bigcup_{n\in\mathcal C}
\operatorname{pages}(n)
\right|.
\]

当预算不足时，不可能只按 error 独立展开所有 nodes。系统需要在 error benefit、visible area、page reuse 和 transfer cost之间排序 refinement requests。

page table 维护

\[
\operatorname{virtualPageID}
\longrightarrow
\operatorname{physicalSlot}.
\]

missing entry 指向 null 或 coarse fallback。上传链为：

~~~text
GPU/CPU selection emits page requests
→ deduplicate and prioritize
→ allocate physical slots under budget
→ copy queue uploads page data
→ signal upload fence
→ graphics queue waits before first use
→ publish page-table mapping
~~~

eviction 也受 fence 约束。某 physical slot 只有在所有引用它的 submitted frames 完成后才能复用；仅因本帧 cut 不需要它，不代表 GPU 已停止读取。

DX12 可用 placed resources 加自定义 geometry page table，也可在适合的资源布局上利用 reserved/tiled resources。reserved resource 先保留 virtual address range，再按 tile 映射 backing heap；但 hierarchy、error selection 和 fallback 仍由引擎负责，API 不会自动生成虚拟几何。

## 5. 跟着一个完整例子走到底

设 hierarchy 有四级，每级 object-space error 分别为

\[
40,\ 10,\ 2.5,\ 0.6\ \mathrm{mm}.
\]

camera 对应

\[
f_{\rm px}=1000\ \mathrm{pixels},
\]

物体距离

\[
z=5000\ \mathrm{mm},
\]

threshold 为 \(\tau=1\) pixel。

各级 screen errors 为

\[
8,\ 2,\ 0.5,\ 0.12\ \mathrm{pixels}.
\]

root 与第二级误差超标，必须展开；第三级误差 0.5 pixel 已满足，因此 cut 选择第三级 nodes，不访问 leaves。

假设当前可见区域选择 64 个第三级 clusters，每 16 个打包进一个 64 KiB page，共需要 4 pages，即

\[
4\times64\ \mathrm{KiB}
=256\ \mathrm{KiB}.
\]

最高精度 leaves 可能需要 32 pages，但远景完全不必驻留。

现在相机移动到

\[
z=1000\ \mathrm{mm}.
\]

四级 screen errors 变为

\[
40,\ 10,\ 2.5,\ 0.6\ \mathrm{pixels}.
\]

第三级不再满足 1 pixel threshold，需要 leaves。selector 发现 leaf pages missing，于是本帧仍绘制 resident 第三级 parents，同时产生 32 个 page requests。streamer 按可见面积和距离排序，在 copy queue 上传预算允许的 pages；fence 完成并发布 mapping 后，相应 branches 在后续帧展开为 leaves。

整个过程中没有等待 CPU 同步，也没有孔洞。质量逐步提高的代价是短暂使用 2.5 pixel error 的 parent，而不是访问未驻留 memory。

## 6. 回到真实系统：程序实际上怎样工作

离线构建器需要生成：

~~~text
leaf meshlets
→ cluster groups
→ simplify groups into parent geometry
→ compute conservative geometric error
→ preserve shared boundaries
→ assign hierarchy links
→ pack nodes into streaming pages
→ serialize root/fallback pages separately
~~~

parent simplification必须处理 cluster boundaries。相邻 branches 在不同 LOD 时若边界顶点不一致，会产生 cracks。可固定 shared boundaries、使用 compatible refinement、添加 transition geometry，或设计 hierarchy 使 cut 保持无缝。

runtime 模块可分为 `HierarchySelector`、`PageRequestBuffer`、`GeometryPageCache`、`UploadScheduler`、`ResidencyTracker` 和 `PageTable`。每帧记录 selected nodes、missing pages、uploaded bytes、fallback count、visible triangles、budget pressure 与 eviction age。

root 或 coarse levels 应常驻，保证任何资产始终有可绘制表示。page cache eviction 可以结合 last-used frame 和 refinement benefit，但必须等相关 fence 完成。高速相机运动还需要预取相邻空间 pages，减少持续 fallback。

Microsoft 的 Direct3D 12 memory-management documentation 区分 committed、placed 与 reserved resources，并说明 reserved resource 可先拥有 virtual address、再映射 backing memory；这是实现稀疏资源的 API 基础之一。[D3D12 Memory Management Strategies](https://learn.microsoft.com/en-us/windows/win32/direct3d12/memory-management-strategies)

对 CT volume 与 surface，可共享同一种“需求驱动 residency”思想，但 page 内容不同。3D texture bricks 可使用 tiled resources；meshlet geometry 常通过自定义 buffer pages 与 indirection 访问。

## 7. 容易走错的岔路

只按 node center distance 计算 screen error，会低估巨大 bounds 中靠近相机的部分。应使用保守最近深度或 projected bounds。

每个 node 独立选 LOD 会破坏 tree cut，可能同时绘制 parent 和 child，或漏掉 branch。选择器必须维持层次覆盖不变量。

page 上传完成前就更新 page table，会让 graphics queue 读取尚未写完的数据。mapping 发布必须位于 copy fence 之后。

只看显存容量、不看 residency budget 和 in-flight frames，也会导致抖动。频繁 evict/reload 同一 pages 的 thrashing 可能比使用较粗 LOD 更慢。

把 reserved resources 当成自动 streaming 系统同样错误。它提供 virtual-to-physical mapping能力，不负责决定哪些 geometry 有价值，也不生成 fallback。

最后，triangle count 下降不保证画面稳定。LOD error metric、boundary cracks 和 temporal hysteresis 都必须单独验证。

## 8. 本章落点、验证与下一章

本章把 meshlets 组织成 error-bounded hierarchy，并用 \(e_{\rm px}\approx f_{\rm px}\epsilon/z\) 决定展开深度。合法 tree cut 保证曲面覆盖，virtual page table 与 fence-aware cache 让只有当前需要的 clusters 驻留；missing fine pages 由 resident ancestor 提供无孔洞 fallback。

在 STL/DirectX 项目中，本章对应多级 QEM 资产构建、GPU hierarchy selection、geometry page cache 和 streaming diagnostics。在 CT viewer 中，同一框架可扩展到 surface meshlets 与 volume bricks。

本章的 90 分钟验证是为四级 toy hierarchy 实现 screen-error selection，复现 5 m 时选择第三级、1 m 时请求 leaves 的结果。再给 page cache 设置只能容纳一半 leaf pages 的预算，记录 fallback 与 eviction。预期是始终存在 coarse geometry、没有缺页访问；去掉 fence-delayed reuse 后，GPU validation 或画面应暴露 slot 被提前覆盖的问题。

下一章会从“几何已经以正确分辨率到达 GPU”进入 shading：PBR 为什么把材质分成可测参数，microfacet BRDF 中的 normal distribution、Fresnel 与 masking-shadowing怎样共同决定实时高光。

