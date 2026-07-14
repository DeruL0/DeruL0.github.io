# 第 4 日：当正确模型遇到规模、噪声与表示选择

日期：2026-07-01

第 3 日把五条线推进到真实系统后，新的限制已经明确：FDK 无法补全缺失数据，曲率流会损失特征，训练会被 activation memory 限制，连续体表示需要更高效的 primitive，而 frame graph 仍无法消除海量 CPU draw decisions。今天分别解决这些问题。

## 阅读入口

1. [CT：从 FDK 残差到迭代重建与正则化](01-ct.md)
2. [几何处理：从法向滤波到保特征网格去噪](02-geometry-processing.md)
3. [AI Infrastructure：autograd activation 生命周期与 checkpointing](03-ai-infrastructure.md)
4. [可微三维渲染：3D Gaussian 怎样投影成屏幕椭圆](04-differentiable-rendering.md)
5. [DirectX 12：GPU-driven culling 与 ExecuteIndirect](05-directx12-engine-rendering.md)

每篇都保持固定八段叙事，并给出可以在 30 到 90 分钟内复现正文结论的验证任务。

