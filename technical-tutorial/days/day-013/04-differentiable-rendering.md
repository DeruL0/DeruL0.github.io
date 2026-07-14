# 可微三维渲染：CTGS 端到端系统设计

## 1. 从一个真实任务开始

今天的可微三维渲染任务是把 CTGS 从一个训练脚本推进成完整系统。输入不再是一组已经整理好的 toy projections，而是工业 CT 或 micro-CT 的 raw/corrected projection、扫描几何、采集协议和任务需求；输出也不只是一个能渲染的 Gaussian cloud，而是可验证的重建、可解释的误差报告、可导出的表面或材料表示，以及能被 viewer 或后续几何处理使用的 artifact。

前面章节已经讲过 differentiable rendering 的投影、体渲染、Gaussian 表示、densification、正则化、任务损失和压缩。今天的位置在知识树中更靠系统层：当一种可训练 3D 表示要承担 CT reconstruction 或 CT analysis 的任务时，它必须尊重 CT 的物理链，同时具备机器学习系统的训练、验证、部署和追溯能力。3D Gaussian Splatting 的官方项目展示了 Gaussian 表示在实时新视角合成中的工程优势；医学和 CT 方向的 Gaussian 表示研究则说明这种表示正在被迁移到 sparse-view CT、CBCT 和可训练体表示场景中。但 CTGS 若要真正服务你的项目，关键不是“把 3DGS 换成 CT 数据”，而是建立端到端 contract。

## 2. 最直接的办法，以及它为什么不够

最直接的做法是把 CT projections 当作训练图片，把 Gaussian 参数当作可优化变量，定义一个 projection loss，然后训练到 loss 下降。训练完成后导出 Gaussian，或者从 Gaussian 采样成 volume，再做 surface extraction。这个流程可以很快做出可视化结果，也能在论文图里显示结构变清楚。

它不够的原因是 CT 的错误常常不是最后一步才出现。raw counts 需要 dark/flat correction；beam hardening、scatter、detector saturation 会改变 projection 的物理含义；geometry 中一个小的 center offset 会让 Gaussian 学到错误的结构；训练集 projections 上的 loss 可以下降，但 held-out angles 上可能过拟合；Gaussian 的 opacity 或 density 参数若没有单位和正性约束，很容易从“衰减表示”滑向“为了匹配投影的任意渲染表示”。最后，即使训练图像很好看，如果没有任务指标和 artifact provenance，也很难把它接入 CT 测量或 DirectX viewer。

## 3. 关键想法是怎样被引出来的

从这个失败中引出的关键抽象是 typed artifact DAG。CTGS 系统不是一个函数 `train(projections) -> gaussians`，而是一张有类型、有 hash、有依赖关系的 artifact graph。raw counts、corrected counts、geometry、scan recipe、initial volume、Gaussian set、training config、checkpoints、validation report、mesh、compressed package、viewer scene 都是节点。每个节点的产生方式、输入 hash、单位和有效性条件都必须记录。

这个抽象的好处是把系统错误局部化。如果 geometry 更新了，下游训练结果和 validation report 自动失效；如果只改 viewer shader，不应该让 training checkpoint 失效；如果 corrected projections 重新生成，Gaussian training 必须重跑；如果只改报告模板，原始重建不需要重算。对 CTGS 这种跨 CT 物理、可微渲染、训练系统和实时显示的项目来说，artifact DAG 是避免混乱的最低成本结构。

## 4. 一步一步建立正式模型

先把 Gaussian 表示写成衰减场，而不是普通 radiance field。一个简化形式是

\[
\mu_G(x)=\sum_j \alpha_j
\exp\left(-\frac{1}{2}(x-\mu_j)^T\Sigma_j^{-1}(x-\mu_j)\right),
\]

其中 `mu_G(x)` 表示位置 `x` 的线性衰减系数，`alpha_j` 是第 `j` 个 Gaussian 的强度，`mu_j` 是中心，`Sigma_j` 是协方差。把它写成衰减场是为了保持 CT 语义：投影不是相机颜色，而是射线积分后的 X 射线透过强度。

给定一条射线 `r_i`，理想情况下它的线积分是

\[
p_i(G)=\int_{r_i}\mu_G(x)\,ds.
\]

若入射强度为 `I_0`，预测 detector count 的期望可以写成

\[
\lambda_i(G)=I_{0,i}\exp(-p_i(G)).
\]

由于 detector counts 更接近 Poisson 观测，projection loss 可以使用负对数似然的形式：

\[
\mathcal{D}_{\mathrm{count}}(G)=\sum_i \left[\lambda_i(G)-n_i\log\lambda_i(G)\right],
\]

这里省略了与 `G` 无关的常数项。这个 loss 的意义是让训练尊重 counts 的统计结构，而不是只对 log projection 做无差别 L2。

完整目标还需要正则化和任务约束：

\[
\mathcal{L}(G)=\mathcal{D}_{\mathrm{count}}(G)+
\beta R_{\mathrm{density}}(G)+
\gamma R_{\mathrm{geometry}}(G)+
\eta \mathcal{L}_{\mathrm{task}}(G).
\]

`R_density` 防止密度无界或漂浮噪声，`R_geometry` 约束 Gaussian 的尺度、各向异性和空间连通性，`L_task` 则把壁厚、孔隙或表面误差等任务目标接入训练。这个式子不是为了把所有东西塞进一个万能 loss，而是提醒你：CTGS 的系统目标应同时包含物理一致性、表示稳定性和任务有效性。

## 5. 跟着一个完整例子走到底

假设你有 180 个投影，每个投影是 512×512。你把其中 150 个角度用于训练，30 个角度保留为 validation。初始 Gaussian 来自一个低分辨率 FDK volume 的采样。训练开始时，归一化 count residual 在训练集上是 1.00，在 validation 上是 1.05。训练 5000 step 后，训练 residual 降到 0.20，validation 降到 0.24，这说明模型不仅记住训练角度，也改善了未见角度的解释。

现在进入 densification。你允许高 residual 区域分裂 Gaussian，让训练 residual 进一步降到 0.12。可是 validation residual 升到 0.40。这个现象在视觉上可能仍然好看，因为训练角度的渲染更锐利；但从 CT 系统角度看，这是过拟合或 geometry/regularization 错误的信号。artifact DAG 中的 validation report 会阻止这个 checkpoint 成为可发布结果。

再看部署包。假设最终有效 Gaussian 数量是 300000，每个 Gaussian 部署时保存 position、scale、orientation、attenuation 和少量 material/tag 信息，平均 48 bytes，则主表示大小约为

\[
300000\times48=14.4\text{ MB}.
\]

这个数值说明 Gaussian package 可以比 dense volume 更适合交互式 viewer。但它也说明 metadata 不能丢：没有 geometry、单位、训练 projection split、validation report 和压缩参数，这 14.4 MB 只是一个漂亮资产，不是 CTGS 系统产物。

## 6. 回到真实系统：程序实际上怎样工作

真实 CTGS pipeline 可以按 artifact DAG 实现。第一层是 acquisition artifacts：raw counts、dark/flat frames、scan recipe、geometry calibration。第二层是 preprocessing artifacts：corrected counts、bad pixel mask、log projection 或 count-domain target。第三层是 training artifacts：initial Gaussian set、training config、optimizer state、densification/pruning schedule、checkpoints。第四层是 validation artifacts：held-out projection residual、task metric、uncertainty note、failure cases。第五层是 export artifacts：mesh、compressed Gaussian package、viewer scene、report。

对你的项目来说，这个结构可以同时连接 CT Framework、SAD-GS/CTGS、geometry processing 和 DirectX viewer。CT Framework 提供物理链和 reconstruction baseline；CTGS 训练模块优化 Gaussian 衰减场；geometry 模块从 Gaussian 或 volume 中提取 surface、厚度和缺陷；DirectX 12 viewer 显示 Gaussian 或 mesh，并读取同一份 validation metadata。这样，系统不是几个孤立 demo，而是一条能解释“这个结果从哪里来、为什么可信、怎样被显示”的生产链。

## 7. 容易走错的岔路

第一个岔路是只优化训练 projections。CT inverse problem 本来就容易在 sparse-view 或 noisy setting 下不适定；如果没有 held-out angle、task metric 或 phantom validation，模型会学到对训练角度有利的 hallucination。第二个岔路是隐藏 preprocessing。CTGS 结果差，可能不是 Gaussian 表示差，而是 dark/flat correction、geometry 或 log transform 错；如果 preprocessing 不作为 artifact 记录，错误会被训练过程掩盖。

第三个岔路是混淆 radiance 和 attenuation。3DGS 的原始语境是相机成像和颜色混合；CTGS 必须回到射线衰减和 counts 统计。第四个岔路是导出 viewer package 时丢掉单位和验证报告。一个实时显示资产如果不能告诉你体素尺寸、坐标系、扫描几何和误差指标，就不能作为 CT 分析结果。

## 8. 本章落点、验证与下一章

今天的落点是：CTGS 的端到端系统不是“训练一个 Gaussian cloud”，而是“用 typed artifact DAG 管住 CT 物理、可微训练、验证、导出和显示”。这个结构让每个结果有来源，每个下游 artifact 有失效规则，每次模型改动都能回到任务指标上判断。

验证练习用 60 到 90 分钟完成：为你现有或计划中的 CTGS pipeline 画一张 artifact DAG，不需要画得漂亮，但每个节点必须写清楚输入、输出、单位、hash 或版本、失效条件。然后选一个节点，例如 geometry calibration，说明它变化后哪些下游结果必须重算。预期结果是你能发现当前 pipeline 中哪些文件是“隐式依赖”，也就是实际影响结果但没有被记录的东西。

下一章自然会进入“CTGS release gate 和部署形态”。因为端到端 DAG 解决了结构问题，下一步要回答每次训练结束后怎样自动判断：这个 Gaussian representation 是否可以进入报告、viewer、几何测量或下一轮实验。

