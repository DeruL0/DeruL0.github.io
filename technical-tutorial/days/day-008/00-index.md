# 第 8 日：当观测、对应关系与可见性不再完整

日期：2026-07-06

第 7 日分别建立了多能 CT、UV 参数化、CUDA Graphs、Gaussian 表面场与 microfacet PBR。今天沿五条线各自继续：金属让部分 CT 射线变成失真或删失观测；无连接点云需要从有向法线重建整体曲面；多 GPU 训练必须让各副本看到同一个平均梯度；动态 Gaussian 需要跨时间保持身份；实时光照则必须先解决阴影分辨率与大量光源的组织。

## 阅读入口

1. [CT：金属伪影、删失射线与模型驱动 MAR](01-ct.md)
2. [几何处理：从有向点样本推导 Poisson 曲面重建](02-geometry-processing.md)
3. [AI Infrastructure：DDP、ring all-reduce 与梯度 bucket](03-ai-infrastructure.md)
4. [可微三维渲染：canonical space、deformation field 与 4D Gaussian](04-differentiable-rendering.md)
5. [DirectX 12：shadow map、级联阴影与 clustered lighting](05-directx12-engine-rendering.md)

五条线彼此独立。每章都从一个具体工作开始，经历直接方法的失败、模型推导和连续算例，最后回到对应项目的实现边界。
