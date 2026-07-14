# 第 9 日：当模型本身也要被估计

日期：2026-07-08

第 8 日分别处理了金属删失观测、Poisson 曲面重建、DDP 梯度同步、动态 Gaussian 的 canonical space，以及 shadow map 与 clustered lighting。今天五条线各自推进到一个共同但不混合的问题：有些东西不能再被当成固定背景。CT 中几何参数会错；几何处理中浮点判定会错；训练系统里模型状态不能全量复制；可微渲染中颜色不能继续把材质和光照混在一起；实时渲染中栅格化可见性不能回答所有射线问题。

## 阅读入口

1. [CT：几何标定、投影矩阵与联合重建](01-ct.md)
2. [几何处理：鲁棒 predicate、精确判定与 mesh Boolean](02-geometry-processing.md)
3. [AI Infrastructure：FSDP、ZeRO 与并行策略选择](03-ai-infrastructure.md)
4. [可微三维渲染：材质、光照与逆渲染分解](04-differentiable-rendering.md)
5. [DirectX 12：DXR、BLAS/TLAS 与混合光追管线](05-directx12-engine-rendering.md)

五条线仍然彼此独立。每章都从一个真实工作开始，先展示直接做法在哪里失败，再把必须引入的数学或系统抽象推出来，最后落回项目实现和可验证练习。
