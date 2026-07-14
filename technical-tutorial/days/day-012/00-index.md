# 第 12 日：当质量、尺度和时间稳定性要被量化

日期：2026-07-12

第 11 日把五条线推进到学习先验、拓扑修复、编译融合、语义编辑和引擎任务组织。今天继续分别推进：CT 不再只问图像是否清楚，而要问具体检测任务是否可靠；几何不再只看连通性，而要用 Laplacian 谱描述形状频率；AI Infrastructure 不再只优化单个 rank，而要选择 data/tensor/pipeline/sequence 等并行轴；可微渲染不再只追求最高画质，而要在码率、显存和失真之间取舍；DX12 则把渲染从单帧推进到跨帧的 history、motion vector 和 temporal reconstruction。

## 阅读入口

1. [CT：不确定性、任务型质量评价与检测可靠性](01-ct.md)
2. [几何处理：谱几何、Laplace-Beltrami 与多尺度形状分析](02-geometry-processing.md)
3. [AI Infrastructure：data/tensor/pipeline/sequence 并行策略](03-ai-infrastructure.md)
4. [可微三维渲染：率失真、压缩和部署](04-differentiable-rendering.md)
5. [DirectX 12：时域渲染、motion vector 与 temporal reconstruction](05-directx12-engine-rendering.md)

五条线仍保持独立。今天的共同背景只是“必须量化取舍”：CT 量化任务风险，几何量化形状频率，训练系统量化通信与显存轴，渲染表示量化码率失真，实时渲染量化跨帧稳定性。
