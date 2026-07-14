# DirectX 12 第 12 章：时域渲染为什么既能补细节，也会制造拖影

## 1. 从一个真实任务开始

上一章用 ECS 和任务图组织了一帧内的工作。现在进入跨帧问题。实时渲染为了省成本，常常一帧只采样一次或低分辨率渲染，然后借助历史帧补充信息。TAA、temporal upsampling、ray tracing denoising 和 SSR accumulation 都依赖同一件事：把上一帧的信息投影到当前帧。

真实任务是在 DX12 CT/STL viewer 中渲染细网格和 DXR 接触阴影。单帧采样会闪烁，低分辨率渲染会锯齿；如果累积历史，边缘稳定得多。但当用户快速旋转模型，历史颜色可能来自错误位置，形成拖影；新暴露区域没有历史，出现 ghost 或模糊。时域方法不是免费超分，而是用 motion vector、depth、history validation 和 rejection 在旧信息和新信息之间做选择。

今天的主题是 temporal reconstruction：如何用 motion vector 找到上一帧对应像素，如何融合历史，为什么 disocclusion 和错误 motion 会破坏结果。AMD FSR2 文档也明确把 motion vectors 视为 temporal upscaling/antialiasing 的关键输入。[AMD FSR2 Temporal Upscaling](https://gpuopen.com/manuals/fidelityfx_sdk/techniques/super-resolution-temporal/)

## 2. 最直接的办法，以及它为什么不够

最直接的抗锯齿是提高分辨率或 MSAA。它能改善几何边缘，但对 shader aliasing、ray tracing noise 和低分辨率 upscaling 成本很高。复杂场景里，硬算更多 samples 不一定可接受。

第二个办法是简单平均多帧：

\[
C_t=\frac12C_t^{\rm current}+\frac12C_{t-1}.
\]

如果相机和物体不动，它能降噪；一旦移动，上一帧同一 pixel 对应的是世界中另一个点，平均会产生拖影。

第三个办法是总是相信 motion vector。它能把上一帧像素搬到当前帧，但 motion vector 错、depth mismatch、disocclusion、透明物体和反射都会让历史无效。需要 history validation，而不是盲目累积。

因此 temporal reconstruction 的核心不是简单“用上一帧”，而是判断上一帧哪个样本仍然代表当前像素，以及它应占多少权重。

## 3. 关键想法是怎样被引出来的

对当前 pixel \(p_t\)，motion vector \(v_t(p_t)\) 指向上一帧对应位置：

\[
p_{t-1}=p_t+v_t(p_t).
\]

从上一帧 history buffer 采样

\[
H_{t-1}(p_{t-1})
\]

后，与当前帧颜色 \(C_t\) 融合：

\[
H_t(p_t)
=
(1-\alpha)C_t(p_t)
+\alpha H_{t-1}(p_{t-1}).
\]

\(\alpha\) 越大，历史越稳定但越容易拖影；越小，响应快但噪声和锯齿多。关键是 \(\alpha\) 不应固定，它应根据 depth、normal、velocity、neighborhood color 和 disocclusion 判断动态调整。

Temporal Anti-Aliasing 通常还配合 subpixel jitter。每帧 projection matrix 轻微偏移，当前帧采样不同子像素位置；多帧累积近似 supersampling。Temporal AA survey 将 TAA 描述为时间摊销的 supersampling，这正是它能用多帧补细节的原因。[A Survey of Temporal Antialiasing Techniques](https://behindthepixels.io/assets/files/TemporalAA.pdf)

## 4. 一步一步建立正式模型

先从 reproject 开始。当前 clip-space position 来自当前 frame 的 depth。通过 inverse current view-projection 得到 world position：

\[
x=
(VP_t)^{-1}p_t.
\]

再用上一帧 view-projection 投到上一帧屏幕：

\[
p_{t-1}
=
VP_{t-1}x.
\]

这等价于由几何计算 camera motion vector。动态物体还需要对象上一帧 transform，否则只用 camera motion 会错。

历史有效性首先看深度。上一帧采样位置的 depth 为 \(z_{t-1}^{\rm hist}\)，当前点投到上一帧的 depth 为 \(z_{t-1}^{\rm reproj}\)。若

\[
|z_{t-1}^{\rm hist}-z_{t-1}^{\rm reproj}|>\epsilon_z,
\]

说明上一帧那里不是同一个 surface，应降低或拒绝 history。

融合可写成

\[
H_t=(1-\alpha_t)C_t+\alpha_t H_{t-1}.
\]

为了避免历史颜色超出当前邻域，常做 neighborhood clamp。设当前 3x3 邻域颜色最小最大为

\[
C_{\min},\quad C_{\max}.
\]

先把 history 限制到范围内：

\[
\widetilde H_{t-1}
=
\operatorname{clamp}(H_{t-1},C_{\min},C_{\max}).
\]

再融合。这个步骤不能保证完美，但能减少明显 ghost，因为历史颜色不能偏离当前局部颜色太远。

## 5. 跟着一个完整例子走到底

考虑当前 pixel 颜色为

\[
C_t=0.8.
\]

motion vector 指向上一帧位置，采样到 history

\[
H_{t-1}=0.2.
\]

如果直接用

\[
\alpha=0.9,
\]

融合为

\[
H_t=0.1\times0.8+0.9\times0.2=0.26.
\]

结果明显偏暗，说明历史可能来自旧背景或 disoccluded 区域。

现在做 depth validation。若当前点投到上一帧的 depth 为

\[
z_{t-1}^{\rm reproj}=0.4,
\]

历史位置 depth 为

\[
z_{t-1}^{\rm hist}=0.9,
\]

差值为

\[
0.5.
\]

若阈值

\[
\epsilon_z=0.05,
\]

则 history 被拒绝，取

\[
\alpha_t=0.
\]

输出为

\[
H_t=0.8.
\]

另一个情况是 depth 通过，但 history 颜色超出当前邻域。若当前邻域颜色范围为

\[
C_{\min}=0.6,\qquad C_{\max}=0.9,
\]

则

\[
\widetilde H_{t-1}=\operatorname{clamp}(0.2,0.6,0.9)=0.6.
\]

再用 \(\alpha=0.9\) 融合：

\[
H_t=0.1\times0.8+0.9\times0.6=0.62.
\]

仍偏向历史，但不再把旧背景的 0.2 拖进当前物体。这就是 temporal reconstruction 中 validation 和 clamping 的实际作用。

## 6. 回到真实系统：程序实际上怎样工作

DX12 temporal pipeline 通常需要当前 color、depth、motion vector、reactive mask、exposure、history color、history depth 和可选 normal/roughness。Render pass 先用 jittered projection 生成当前帧；TAA/upsampler pass 根据 motion vector 采样 history，做 depth/normal validation、neighborhood clamp、disocclusion rejection，再输出新的 history。

Motion vector 的正确性是系统条件。Skinned mesh、camera cut、LOD switch、particle、transparent surface、reflection 和 procedural animation 都可能产生错误 motion。错误 motion 不只是局部模糊，它会把历史投到错误对象上。

DXR denoising 也依赖 temporal accumulation。单帧 1 spp ray tracing 噪声很大，需要跨帧累积。但 ray hit 可能来自反射、阴影或间接光，motion vector 和 G-buffer surface 不总能代表 hit point。denoiser 常需要 hit distance、normal、roughness 和 history confidence。

对 CT/STL viewer，模型旋转和 clipping plane 变化会制造大量 disocclusion。若用户拖动切面，上一帧 history 多半无效。应在交互强变化时降低 \(\alpha\)，甚至重置 history，避免拖影误导测量。

## 7. 容易走错的岔路

第一个误区是把 TAA 当成普通 blur。好的 temporal reconstruction 依赖 motion vector、jitter、validation 和 history management；简单平均只会拖影。

第二个误区是 motion vector 只处理相机。动态物体、skinned mesh、LOD 切换和 procedural deformation 都需要正确 motion。

第三个误区是 history 永远越多越好。高 \(\alpha\) 稳定但响应慢，容易 ghost；低 \(\alpha\) 响应快但噪声多。权重应随信任度变化。

第四个误区是忽略 disocclusion。新暴露区域没有历史，强行采样上一帧背景会产生最明显拖影。

最后，temporal upscaling 不应改变测量语义。CT/STL viewer 中，截图可用 temporal smoothing，但几何测量和像素读取应基于真实几何或明确标记的重建图像。

## 8. 本章落点、验证与下一章

本章把实时渲染从单帧推进到跨帧重建。Motion vector 把当前 pixel 映射到上一帧，history blending 用旧样本补细节和降噪；depth/normal validation、neighborhood clamp 和 disocclusion rejection 防止错误历史造成拖影。

在 DirectX 12 和 CT/STL viewer 项目中，本章对应 motion vector pass、jittered projection、history buffers、TAA/FSR-like upsampling、DXR denoising history、camera cut reset 和交互时 history confidence。

本章的 60 到 90 分钟验证是实现正文一维颜色例子：当前 \(C_t=0.8\)、history \(0.2\)、\(\alpha=0.9\)，先计算错误融合 0.26；再用 depth mismatch 拒绝 history 得到 0.8；最后用 neighborhood clamp 把 history 限制到 0.6 后得到 0.62。预期是 validation 比固定平均更能避免拖影。

下一章将进入性能分析与 PIX。时域技术、DXR、streaming 和任务图叠在一起后，肉眼很难判断瓶颈；需要用 PIX/Profiler 把 CPU、GPU、copy queue、barrier、occupancy 和 memory bandwidth 全部量化。
