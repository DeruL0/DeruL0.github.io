# 第 5 日：先验、结构变化与层次化工作单元

日期：2026-07-03

第 4 日分别建立了迭代 CT、保特征去噪、activation checkpointing、3D Gaussian 投影和 GPU-driven draw。今天继续处理这些系统中无法由连续参数调节解决的问题：怎样选择边缘先验、怎样改变网格采样、怎样改变数值精度、怎样增删 Gaussian，以及怎样把高模拆成 GPU 可独立处理的小簇。

## 阅读入口

1. [CT：Total Variation、有限角与零空间](01-ct.md)
2. [几何处理：edge split/collapse/flip 与各向同性重网格化](02-geometry-processing.md)
3. [AI Infrastructure：FP16、BF16、混合精度与 loss scaling](03-ai-infrastructure.md)
4. [可微三维渲染：Gaussian densification 与 pruning](04-differentiable-rendering.md)
5. [DirectX 12：mesh shader、meshlet 与簇级剔除](05-directx12-engine-rendering.md)

五篇依然彼此独立；每篇的练习只验证正文中已经完整推导的结论。

