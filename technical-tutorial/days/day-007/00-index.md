# 第 7 日：当模型必须面对材料、表面与动态执行

日期：2026-07-05

第 6 日建立了 Poisson 统计、QEM、编译图、Gaussian 多尺度过滤与虚拟几何。今天继续处理这些模型留下的真实系统问题：多能 X 射线不再满足单一线积分、闭合曲面无法无失真铺平、GPU launch 图要求稳定地址、Gaussian 外观不自动等于可测表面，以及实时材质需要满足能量与视角规律。

## 阅读入口

1. [CT：多能谱、束硬化、散射与多材料](01-ct.md)
2. [几何处理：曲面参数化、UV atlas 与失真](02-geometry-processing.md)
3. [AI Infrastructure：CUDA Graphs 的 capture、replay 与静态地址](03-ai-infrastructure.md)
4. [可微三维渲染：Gaussian 场的表面约束与 mesh 提取](04-differentiable-rendering.md)
5. [DirectX 12：microfacet PBR 与实时材质模型](05-directx12-engine-rendering.md)

五条线保持独立；每篇都从上一章的下一问题开始，并通过一个连续算例走到工程实现。

