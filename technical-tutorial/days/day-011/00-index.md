# 第 11 日：当先验、拓扑、编译和语义成为系统的一部分

日期：2026-07-11

第 10 日把五条线推进到时间、显存和可见性：CT 进入运动补偿，几何进入配准，训练系统进入 activation 重计算，可微渲染进入 visibility 梯度，DX12 进入资源驻留。今天继续分线推进：CT 开始把学习先验放进迭代结构；几何开始用拓扑量检查和修复 mesh；训练系统开始把 eager 执行编译成更少、更大的 kernel；可微渲染开始给 Gaussian 或 radiance field 加语义和可编辑对象；DX12/引擎线则把资源、任务和对象生命周期组织成 ECS、任务图与资产管线。

## 阅读入口

1. [CT：深度展开、Plug-and-Play 与物理一致学习先验](01-ct.md)
2. [几何处理：拓扑分析、洞、handle 与 mesh 修复](02-geometry-processing.md)
3. [AI Infrastructure：torch.compile、图捕获与算子融合](03-ai-infrastructure.md)
4. [可微三维渲染：语义特征、实例分组与可编辑表示](04-differentiable-rendering.md)
5. [DirectX 12：ECS、任务图与资产管线](05-directx12-engine-rendering.md)

五条线继续保持独立。今天每章都围绕同一个写作要求展开：不是列方法，而是先说明前一章模型在哪个真实工作中不够，再推导出新抽象为什么必须进入系统。
