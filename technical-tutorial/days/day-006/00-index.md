# 第 6 日：从确定性模型进入统计、层次与编译系统

日期：2026-07-04

第 5 日分别改变了重建先验、网格采样、数值精度、Gaussian 数量和几何工作单元。今天进一步建立这些变化背后的系统模型：光子计数的统计似然、误差驱动的网格简化、动态图的编译边界、多尺度 Gaussian 的采样理论，以及虚拟几何的层次选择与驻留。

## 阅读入口

1. [CT：Poisson 光子统计与低剂量重建](01-ct.md)
2. [几何处理：Quadric Error Metrics 与网格简化](02-geometry-processing.md)
3. [AI Infrastructure：graph capture、guards、fusion 与 graph break](03-ai-infrastructure.md)
4. [可微三维渲染：抗锯齿、LOD 与 rate-distortion](04-differentiable-rendering.md)
5. [DirectX 12：层次化虚拟几何与 GPU residency](05-directx12-engine-rendering.md)

五篇保持独立，不把术语压缩成摘要；每篇都从一个具体失败推导出今天需要的模型。

