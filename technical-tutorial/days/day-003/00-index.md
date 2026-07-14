# 第 3 日：从局部算子进入完整系统

日期：2026-06-30

第 2 日已经让每条线获得了一个可用的局部算子：FBP 的滤波反投影、网格 Laplacian、GPU Roofline、可见性边界梯度，以及 DX12 的资源绑定。今天继续追问这些算子放进真实三维系统后会暴露什么新问题。

## 阅读入口

1. [CT：从平行束 FBP 到圆轨迹 cone-beam FDK](01-ct.md)
2. [几何处理：隐式曲率流、稳定性与收缩](02-geometry-processing.md)
3. [AI Infrastructure：tiling 与 shared memory 为什么提高算术强度](03-ai-infrastructure.md)
4. [可微三维渲染：体渲染的 transmittance 与梯度饱和](04-differentiable-rendering.md)
5. [DirectX 12：frame graph 怎样推导依赖与资源寿命](05-directx12-engine-rendering.md)

五篇依然是五条独立知识线。每篇都从上一章结尾的未解问题开始，并在本章内部完成一条从输入到输出的推导。

