# 可微三维渲染第 12 章：3DGS 部署为什么要用率失真思维

## 1. 从一个真实任务开始

上一章给 Gaussian 增加语义和实例特征，让用户能选择、编辑和导出对象。表示更可用了，也更重了。一个高质量 3DGS 场景可能包含数百万 Gaussians，每个 Gaussian 有 position、scale、rotation、opacity、SH color、semantic feature、material 参数和实例 id。训练机能渲染，不代表浏览器、移动 GPU 或嵌入式 viewer 能加载。

真实任务是把 CTGS/3DGS 结果部署到一个交互 viewer。用户希望几秒内打开场景，显存不超过预算，帧率稳定，还要保留测量和可编辑属性。最直接的高质量模型太大；简单删点会丢细节；统一量化会让某些材料边界出错。部署不再是导出文件，而是在 bitrate、显存、渲染时间和失真之间做系统取舍。

今天的主题是率失真。压缩不是最后 zip 一下，而是在训练或后处理时明确优化“用多少 bits 换多少质量”。RDO-Gaussian 等工作把 3D Gaussian 表示学习写成 rate-distortion optimization，结合 pruning 和 entropy-constrained vector quantization 来控制码率和失真。[RDO-Gaussian](https://rdogaussian.github.io/)

## 2. 最直接的办法，以及它为什么不够

最直接的方法是按 opacity 或 size 删除低贡献 Gaussians。它能快速减小模型，但贡献不是全局固定的。一个 Gaussian 在大部分视角看不见，却可能决定某个近距离测量边界；一个低 opacity Gaussian 可能是半透明或细结构的一部分。

第二个方法是统一把所有属性量化到 8 bit。位置、scale、rotation、SH coefficients 和语义 feature 的误差敏感性不同。位置量化误差会移动几何，颜色量化误差影响外观，语义 feature 量化误差可能影响选择和编辑。统一 bit-width 很粗糙。

第三个方法是只压缩磁盘文件，运行时解压成完整 float32。它减少存储，不减少显存和带宽。部署真正关心的是 runtime resident memory、GPU bandwidth 和 shader 解码成本。

因此需要把 rate 放进优化目标。每个 Gaussian 和每类属性都应根据对任务和视觉失真的贡献分配 bits，而不是平均对待。

## 3. 关键想法是怎样被引出来的

Rate-distortion 的基本形式是

\[
\min_{\theta}
D(\theta)+\lambda R(\theta).
\]

\(D\) 是失真，例如渲染图像误差、depth/normal 误差、语义选择误差或 CT 测量误差；\(R\) 是码率或存储成本；\(\lambda\) 控制质量和大小的交换。大 \(\lambda\) 更重视压缩，小 \(\lambda\) 更重视质量。

对 3DGS，\(\theta\) 包括 Gaussian 数量和属性。Rate 可由 Gaussian count、量化 code length、entropy model 或实际文件大小估计。Distortion 不能只用 RGB PSNR；如果模型用于 CTGS 测量，还要包括 density、surface、材料 label 或 task metric。

LightGaussian 一类方法从 pruning 和 recovery 出发，寻找对重建贡献小的 Gaussians 并恢复质量。[LightGaussian](https://arxiv.org/html/2311.17245v5) RDO-Gaussian 则直接把 compact Gaussian learning 写成端到端 rate-distortion objective，说明 3DGS 部署已经从经验剪枝走向明确优化。

## 4. 一步一步建立正式模型

设场景有 \(N\) 个 Gaussians。第 \(i\) 个 Gaussian 是否保留由

\[
m_i\in\{0,1\}
\]

表示。保留后的渲染失真为

\[
D_{\rm img}
=
\sum_v
\ell\left(
\mathcal R_v(\{m_iG_i\}),
I_v
\right).
\]

存储成本可粗略写成

\[
R
=
\sum_i m_i
\left(
b_{\mu}+b_{\Sigma}+b_{\alpha}+b_c+b_f
\right),
\]

其中各项分别是位置、协方差、opacity、颜色和 feature 的 bit 数。若有 entropy coding，则更准确地写为

\[
R\approx
\sum_j -\log_2 p(q_j),
\]

\(q_j\) 是量化后的符号。

完整目标为

\[
\min_{m,q}
D_{\rm img}
+\lambda R
+\eta D_{\rm task}.
\]

\(D_{\rm task}\) 是任务失真，例如 CTGS 的孔隙率误差、材料边界误差或语义选择错误。这个项很重要，因为视觉上可接受的压缩可能破坏测量。

量化可写为

\[
\widehat a
=
\Delta\cdot
\operatorname{round}\left(\frac{a}{\Delta}\right).
\]

训练时 round 不可微，常用 straight-through estimator 或软量化近似。部署时则使用真正的整数或 codebook。

## 5. 跟着一个完整例子走到底

设一个小场景有 100 万个 Gaussians。每个 Gaussian 未压缩属性为 64 bytes，总大小为

\[
64\ {\rm MB}.
\]

现在考虑两种压缩方案。方案 A 删除 50% Gaussians，剩余仍用 64 bytes：

\[
R_A=0.5\times64=32\ {\rm MB}.
\]

方案 B 只删除 20%，但把每个剩余 Gaussian 压到 24 bytes：

\[
R_B=0.8\times24=19.2\ {\rm MB}.
\]

若方案 A 的图像失真为

\[
D_A=0.010,
\]

方案 B 的图像失真为

\[
D_B=0.014.
\]

取

\[
\lambda=0.001
\]

且把 MB 直接作为 rate 单位，则目标为

\[
J_A=0.010+0.001\times32=0.042,
\]

\[
J_B=0.014+0.001\times19.2=0.0332.
\]

在这个 \(\lambda\) 下，方案 B 更好，因为它用少量失真换来更多码率下降。若任务失真 \(D_{\rm task}\) 显示方案 B 把材料边界量化坏了，则需要把任务项加入比较，而不能只看 RGB。

这个例子说明部署决策应是连续 trade-off，而不是“删一半点”或“统一 8 bit”的固定规则。

## 6. 回到真实系统：程序实际上怎样工作

一个部署 pipeline 通常包含 importance scoring、pruning、attribute quantization、entropy coding、LOD 分层、runtime decode 和质量回归测试。训练阶段可加入 rate penalty；后处理阶段可做 Pareto sweep，生成多个质量档位。

对 web 或移动 viewer，位置可局部坐标量化，rotation/scale 可用紧凑参数，SH 可降阶或按区域共享，semantic features 可 PCA、codebook 或按需加载。不要默认所有属性都 resident。上一章的语义 feature 可能只在编辑模式加载，普通浏览只加载 RGB 和几何。

对 CTGS，压缩策略必须保护物理任务。Gaussian attenuation、材料 label 和几何 surface 的误差比 RGB 误差更关键。一个只优化 novel-view PSNR 的压缩器，可能破坏孔径、体积或缺陷边界。应保存压缩前后的 projection residual、surface distance 和任务指标。

运行时还要考虑解码成本。极致 entropy coding 若需要 CPU 解码大量小块，可能造成加载卡顿；GPU shader 内解码若分支复杂，也会降低帧率。Rate-distortion 部署应同时看 size、quality、decode time 和 render time。

## 7. 容易走错的岔路

第一个误区是把文件大小当成唯一 rate。部署还关心显存、带宽、解码缓存和 GPU attribute fetch。

第二个误区是只用 RGB PSNR 评价压缩。CTGS 和可编辑场景还需要 geometry、depth、semantic、material 和任务指标。

第三个误区是过度依赖 opacity 剪枝。低 opacity 不等于低任务贡献，尤其在边界、半透明和细结构处。

第四个误区是所有区域同等压缩。用户关注区域、测量 ROI、材料边界和近景需要更多 bits；背景可更激进压缩。

最后，压缩后不做交互测试。平均指标好不代表相机快速移动、LOD 切换和编辑选择时稳定。

## 8. 本章落点、验证与下一章

本章把 3DGS/CTGS 部署写成 rate-distortion problem：目标不是盲目删点，而是在图像、几何、语义和任务失真与存储、显存、带宽之间取舍。Pruning、quantization、entropy coding 和 LOD 都应服务这个目标。

在可微渲染和 CTGS 项目中，本章对应 Gaussian importance、attribute bit allocation、semantic feature 按需加载、压缩前后 task metric、runtime decode 和多档 LOD。

本章的 60 到 90 分钟验证是复现 100 万 Gaussian 的方案比较：计算方案 A 和 B 的 rate-distortion objective，并改变 \(\lambda\) 找到两者切换点。预期是 \(\lambda\) 越大越偏向更小码率；加入任务失真后，视觉更优方案可能被否决。

下一章将进入部署与完整 CTGS 系统设计。压缩只是部署的一部分；完整系统还要组织训练、导出、验证、viewer、交互编辑和任务报告的端到端边界。
