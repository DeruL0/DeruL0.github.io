# 第 10 日：当时间、显存和可见性进入模型

日期：2026-07-10

第 9 日把五条线都推进到“模型本身也要被估计”：CT 的几何参数、几何处理的判定符号、训练系统的模型状态、可微渲染的材质光照分解，以及 DXR 的射线查询。今天继续各自深入，不把它们合成一个项目：CT 中扫描过程有时间，几何处理中两个形状先要配准，训练系统中 activation 可以保存也可以重算，可微渲染中 visibility 是不连续的，DX12 中资源能不能被 GPU 访问本身就是运行时状态。

## 阅读入口

1. [CT：运动补偿、4D 状态与时间相关投影](01-ct.md)
2. [几何处理：刚体 ICP、对应关系与非刚性配准](02-geometry-processing.md)
3. [AI Infrastructure：activation checkpointing 与重计算调度](03-ai-infrastructure.md)
4. [可微三维渲染：可见性不连续、软化与边界梯度](04-differentiable-rendering.md)
5. [DirectX 12：资源驻留、heap、streaming 与显存预算](05-directx12-engine-rendering.md)

每章仍按固定八段因果链写作。今天的重点不是介绍名词，而是解释为什么原来的静态假设失败，以及新的时间、对应、重计算、可见性和 residency 抽象怎样被迫出现。
