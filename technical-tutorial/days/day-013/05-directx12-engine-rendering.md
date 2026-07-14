# DirectX 12：PIX 性能分析、瓶颈定位与验证实验

## 1. 从一个真实任务开始

今天的 DirectX 12 任务是分析一个真实慢帧。你的 viewer 或小型引擎已经能加载 CT mesh、Gaussian point cloud、STL 模型或实时渲染场景，但某个典型视角下帧时间从 12 ms 变成 24 ms。用户看到的是卡顿；工程上你看到的是一堆可能原因：CPU 提交慢、GPU raster pass 慢、compute pass 慢、barrier 过多、descriptor 管理混乱、显存 residency 抖动、TAA 或 denoiser 太贵，或者 present 阶段阻塞。

前面章节讲过 DX12 的资源、命令队列、同步、PSO、descriptor heap、GPU-driven rendering、ray tracing 和 temporal rendering。今天的位置在知识树中属于“从会写 renderer 到会诊断 renderer”。DX12 给了你很多显式控制权，也把错误归因的责任交给你。PIX 的价值就在这里：它不是一个抽象性能建议器，而是把 CPU、GPU、D3D12 调用、资源状态、barrier、marker 和 shader 执行放到同一条证据链中。Microsoft 的 [PIX overview](https://learn.microsoft.com/en-us/windows/win32/direct3dtools/pix/articles/general/pix-overview)、[GPU captures](https://learn.microsoft.com/en-us/windows/win32/direct3dtools/pix/articles/gpu-captures/pix-gpu-captures) 和 [Timing Captures](https://devblogs.microsoft.com/pix/timing-captures/) 文档可以作为这条线的官方参考。

## 2. 最直接的办法，以及它为什么不够

最直接的办法是猜。帧慢了，就先把 shader 简化一点；还慢，就降低分辨率；再慢，就减少 draw call；或者看到某个 pass 名字很重，就先优化它。这个办法偶尔会碰巧有效，因为某些瓶颈很明显。但它在 DX12 中经常失败，因为 CPU 和 GPU 是并行工作的，多个 queue 可能重叠，frame latency 会隐藏或放大某些等待，单个 pass 的耗时也可能不是最终瓶颈。

比如你花两小时把 CPU command recording 从 6 ms 优化到 3 ms，但 GPU critical path 仍然是 22 ms，最终帧时间不会明显变化。反过来，如果 GPU 只用 9 ms，而 CPU submit 需要 18 ms，你去优化像素 shader 也不会解决卡顿。没有 capture 和 controlled experiment，你优化的可能只是看起来可疑的地方，而不是决定帧时间的地方。

## 3. 关键想法是怎样被引出来的

从这个失败中引出的关键抽象是性能证据链。一次可信的优化必须从观测开始：先用 timing capture 判断 CPU、GPU、present、queue overlap 的总体关系；再用 GPU capture 进入具体 frame，看每个 pass 的 marker、draw/dispatch、barrier 和资源状态；然后提出一个瓶颈假设；最后做一个只改变一个变量的 A/B 实验。

这条证据链要求 renderer 自己配合。没有 marker 的 frame capture 就像没有函数名的 profiler，你只能看到命令，难以知道它们属于 shadow、G-buffer、Gaussian splat、TAA、postprocess 还是 UI。一个成熟 DX12 renderer 应该在命令列表中用 PIX event 标记 pass、subpass 和关键资源转换。这样 capture 不是事后补救，而是引擎架构的一部分。

## 4. 一步一步建立正式模型

先把帧时间拆成几个竞争的路径。一个简化模型是

\[
T_{\mathrm{frame}}\approx
\max(T_{\mathrm{CPUsubmit}},T_{\mathrm{GPUgraphics}},T_{\mathrm{copy exposed}},T_{\mathrm{present}}).
\]

这个式子的重点是 `max`。帧时间通常由关键路径决定，而不是所有工作简单相加。copy queue 的上传如果完全被 graphics queue 覆盖，它不一定暴露到帧时间；但如果 graphics queue 等它的 fence，它就会变成 exposed cost。

GPU graphics critical path 可以继续拆成 pass：

\[
T_{\mathrm{GPUgraphics}}=
T_{\mathrm{depth}}+T_{\mathrm{raster}}+T_{\mathrm{ray}}+
T_{\mathrm{post}}+T_{\mathrm{barrier}}+\cdots
\]

这个加法只对同一 queue 上串行执行的关键路径成立。如果有 async compute 或 copy overlap，就必须看依赖关系，而不是盲目相加。对单个 pass，可以用带宽和算力做粗判断。若一个 pass 读写 `bytes` 字节，用时 `t`，有效带宽为

\[
B=\frac{\mathrm{bytes}}{t}.
\]

如果 `B` 接近硬件可达到带宽，同时 shader arithmetic 不重，pass 可能是 memory bandwidth bound。若带宽不高但 ALU 指令、occupancy、wave stalls 显示压力很大，可能是 compute bound 或 latency bound。PIX 的 counter 和 shader 分析就是为了把这种猜测变成证据。

## 5. 跟着一个完整例子走到底

假设你的 CT/DirectX viewer 在一个复杂模型上帧时间是 24 ms。Timing capture 显示 CPU submit 是 6 ms，GPU graphics queue 的关键路径是 22 ms，copy queue 大部分被覆盖，present 没有明显阻塞。根据

\[
T_{\mathrm{frame}}\approx\max(6,22,\text{hidden},\text{small})=22\text{ ms}
\]

可以先判断瓶颈在 GPU，而不是 CPU。

进入 GPU capture 后，你用 marker 看到这个 frame 大致由几个 pass 组成：depth prepass 2 ms，main raster 6 ms，DXR picking 或 AO pass 10 ms，TAA/postprocess 2 ms，barrier 和 resolve 约 2 ms。现在如果你先优化 CPU submit，从 6 ms 到 3 ms，frame 仍接近 22 ms，因为关键路径没动。若你把 DXR pass 的 sample 数减半，使它从 10 ms 降到 6 ms，同时把不必要的 UAV barrier 合并，让 barrier cost 从 2 ms 降到 1 ms，则 GPU critical path 变成约

\[
2+6+6+2+1=17\text{ ms}.
\]

这时帧时间才会接近 17 ms。这个例子说明性能分析的关键不是“哪个模块看起来贵”，而是“哪个模块在关键路径上，且有可控变量能改变它”。

## 6. 回到真实系统：程序实际上怎样工作

在真实 DX12 工程中，你应该先把 capture 友好性做进 renderer。每个 pass 开始和结束写 PIX marker；每个资源状态转换尽量通过统一 helper 记录来源和目标 state；每个 frame 保存关键 configuration，例如分辨率、sample count、mesh 数量、Gaussian 数量、LOD 设置、TAA 状态和 camera。这样当某个 capture 显示慢帧时，你能复现同一场景，而不是只得到一次偶然记录。

PIX GPU capture 会记录 D3D12 调用和 GPU work，并允许你检查 draw/dispatch、资源、pipeline state、barrier 和 shader。Timing capture 更适合看长时间运行中的 CPU/GPU 时间分布、线程行为和队列关系。一个务实流程是：先用 Timing capture 确认慢帧是否稳定、CPU/GPU 谁是瓶颈；再抓一个代表性 GPU capture；最后在代码中做小步 A/B，例如关闭某个 pass、减少 sample count、改变 barrier 策略、替换 shader 分支、改变 descriptor 更新方式。每次只改变一个因素，才能把结果归因。

对你的 STL/DirectX 或 CT viewer 项目来说，PIX 分析还可以连接几何和 CTGS 线。大 mesh 渲染慢，可能来自 index buffer cache、overdraw、material 分支或 LOD 缺失；Gaussian splat 渲染慢，可能来自排序、blend、tile binning 或 memory bandwidth；体数据切片慢，可能来自 3D texture bandwidth 或采样模式。只有 capture 能告诉你瓶颈在哪一层，而不是靠场景类型猜。

## 7. 容易走错的岔路

第一个岔路是没有 marker。没有 marker 的 capture 仍然有数据，但解释成本会急剧上升。第二个岔路是在 debug build 或非代表性场景里做性能结论。debug layer、validation、窗口大小、camera、asset 数量都会改变瓶颈。第三个岔路是混淆 CPU time 和 GPU time。CPU 函数返回快，不代表 GPU work 便宜；CPU 阻塞等待 fence，也不一定说明 CPU 代码本身慢。

第四个岔路是平均很多帧后忽略慢帧来源。实时渲染中的体验常由 percentile 和 spike 决定，平均 16 ms 但每秒一次 80 ms 仍然会卡。第五个岔路是修非瓶颈。把一个 1 ms pass 优化到 0.5 ms 没错，但如果关键路径上另一个 pass 是 12 ms，用户感知可能几乎不变。

## 8. 本章落点、验证与下一章

今天的落点是：DX12 性能优化不是猜测，而是证据链。先用 Timing capture 判断 CPU/GPU/queue/present 的关系，再用 GPU capture 定位具体 pass、barrier、资源和 shader，最后用受控 A/B 实验证明某个修改确实缩短关键路径。没有这条链，优化很容易变成随机改代码。

验证练习用 60 到 90 分钟完成：在你的 renderer 或一个 DX12 sample 中给至少五个 pass 加 PIX marker，抓一次代表性 capture，写下 `T_CPUsubmit`、`T_GPUgraphics`、最贵的三个 pass、一个 barrier 或 resource state 可疑点，以及一个只改变单一变量的 A/B 实验。预期结果是你能明确说出当前慢帧是 CPU-bound、GPU-bound、present-bound 还是同步/上传暴露，并能提出一个可验证的优化假设。

下一章自然会进入“把性能分析结果反馈到引擎架构”。因为单次 capture 能定位瓶颈，但长期项目需要把 marker、frame graph、resource lifetime、async queue 和性能回归测试整合进 renderer，使慢帧在开发阶段就被发现。

