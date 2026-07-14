# 可微三维渲染第 6 章：为什么训练清晰的 Gaussian 场景在缩小时会闪烁

## 1. 从一个真实任务开始

上一章通过 densification 让 Gaussian 数量适应场景复杂度。训练视角上图像已经清晰，但把相机拉远、改变分辨率或缩小 viewport 时，细节会闪烁、忽隐忽现，帧时间也没有按屏幕尺寸明显下降。原因是同一批高频 Gaussians 仍被逐个投影和点采样。

今天的任务有三个相关目标。第一，pixel 应测量一个有限面积，而不是只取中心点，必须做 screen-space filtering；第二，远处多个 subpixel Gaussians 应由较粗 parent 近似，避免继续处理所有 children；第三，存储与帧时间都有预算，需要用 rate-distortion 语言决定保留多少 primitives 和多少 bits。

抗锯齿、LOD 与压缩不是三个互不相关的部署技巧。它们都在回答：当观察尺度和资源预算改变时，怎样删除不可分辨的高频，同时控制图像误差。

## 2. 最直接的办法，以及它为什么不够

最直接的抗锯齿方法是给二维 covariance 设置最小值。Gaussian 不会小于一个 pixel，闪烁会减少。但如果只把 footprint 放大而不调整 amplitude，同一个 Gaussian 的总贡献随分辨率改变；画面会变亮或变得过度不透明。

另一种直接方法是相机拉远后随机丢弃 Gaussians。数量下降了，但被删和保留的 primitives 在相邻帧之间变化，产生 temporal popping，也无法保证颜色、opacity 或几何 moments 守恒。

压缩若只把所有参数统一量化到 8 bits，也会让不同参数承受不同比例误差。position 在薄结构处非常敏感，颜色或高阶 spherical harmonics 在某些场景可能更可压缩。资源决策必须与最终渲染 distortion 联系，而不是只看 byte count。

## 3. 关键想法是怎样被引出来的

pixel 的理想测量是连续图像与 pixel reconstruction filter 的积分。若 screen-space Gaussian 比一个 pixel 小很多，直接在 pixel center 评估相当于欠采样；相机稍微移动，中心可能从 Gaussian 峰值跳到尾部，形成闪烁。

Gaussian 的重要优势是 convolution 仍为 Gaussian。将 projected covariance \(\Sigma\) 与近似 pixel filter covariance \(\Sigma_p\) 卷积，得到

\[
\Sigma_f=\Sigma+\Sigma_p.
\]

footprint 变宽后还要缩放 peak，使积分质量近似不变。

LOD 则把一组 children 的低频效果预先聚合成 parent。远处只选择 parent，近处展开 children。parent 应匹配 children 的总权重、一阶 mean 和二阶 covariance，而不只是把中心做平均。

## 4. 一步一步建立正式模型

上一章的二维 Gaussian 为

\[
g(p)
=
\exp\left[
-\frac12(p-m)^{\mathsf T}
\Sigma^{-1}
(p-m)
\right].
\]

用归一化 Gaussian 近似 pixel filter，其 covariance 为 \(\Sigma_p\)。卷积后：

\[
\Sigma_f=\Sigma+\Sigma_p.
\]

若希望未归一化 footprint 的二维积分保持不变，filtered kernel 可写成

\[
g_f(p)
=
\sqrt{
\frac{\det\Sigma}
{\det\Sigma_f}
}
\exp\left[
-\frac12(p-m)^{\mathsf T}
\Sigma_f^{-1}
(p-m)
\right].
\]

前面的 determinant ratio 降低 peak，同时扩大 support。对 alpha compositing，这仍是近似，因为多个 alpha 的组合是非线性的；但它比单纯 clamp covariance 更明确地控制单 primitive 的积分贡献。

一个单位宽 box pixel 在单轴上的 variance 是

\[
\sigma_p^2=\frac1{12}.
\]

实际 filter 可以根据 reconstruction kernel、resolution scale 和 camera footprint 选择，不必固定为 box。

现在构造 hierarchy。children \(i\) 有权重 \(w_i\)、mean \(\mu_i\) 和 covariance \(\Sigma_i\)。moment-matched parent 的总权重为

\[
W=\sum_iw_i,
\]

mean 为

\[
\mu_p
=
\frac1W\sum_iw_i\mu_i,
\]

covariance 为

\[
\Sigma_p
=
\frac1W
\sum_iw_i
\left[
\Sigma_i+
(\mu_i-\mu_p)(\mu_i-\mu_p)^{\mathsf T}
\right].
\]

第二项保留 children centers 的空间分散。只平均 covariance 会让 parent 太窄。

runtime 根据 projected error 或 footprint 选择 parent/children。若 cluster 的 projected radius 和 approximation error 都低于阈值，就使用 parent；否则展开。为了避免阈值附近来回切换，可使用 hysteresis 或 cross-fade。

压缩与 LOD 可以统一写成 rate-distortion objective：

\[
J
=
D
+\lambda_B B
+\lambda_T T,
\]

其中 \(D\) 是渲染 distortion，\(B\) 是 storage/transfer bytes，\(T\) 是 frame time 或 projected work。不同参数的量化 bits、prune 与 hierarchy level 都是在最小化这项权衡。

## 5. 跟着一个完整例子走到底

考虑一个已经投影到屏幕的 subpixel Gaussian：

\[
\Sigma
=
\begin{bmatrix}
0.16&0\\
0&0.04
\end{bmatrix}
\ \mathrm{pixel}^2.
\]

它的标准差只有 0.4 和 0.2 pixels，直接 point sample 很容易随相机移动闪烁。用 box pixel 的 Gaussian variance 近似：

\[
\Sigma_p
=
\frac1{12}I
\approx
\begin{bmatrix}
0.0833&0\\
0&0.0833
\end{bmatrix}.
\]

filtered covariance 为

\[
\Sigma_f
\approx
\begin{bmatrix}
0.2433&0\\
0&0.1233
\end{bmatrix}.
\]

标准差变成约 0.493 和 0.351 pixels。peak correction 是

\[
\sqrt{
\frac{0.16\times0.04}
{0.2433\times0.1233}
}
\approx0.462.
\]

如果只扩大 covariance 而保持 peak 1，总积分会增加约 \(1/0.462\) 倍；加入 correction 后，贡献被摊到更大 footprint，中心变低但总量近似保持。

再看 LOD。两个等权 children 的 screen means 为

\[
\mu_1=(-0.4,0),
\qquad
\mu_2=(0.4,0),
\]

各自 covariance 为

\[
\Sigma_1=\Sigma_2=0.04I.
\]

parent mean 是零。沿 \(x\) 方向，children center spread variance 为 \(0.4^2=0.16\)，所以 parent covariance 是

\[
\Sigma_p
=
\begin{bmatrix}
0.04+0.16&0\\
0&0.04
\end{bmatrix}
=
\begin{bmatrix}
0.20&0\\
0&0.04
\end{bmatrix}.
\]

当这组 children 投影小于 pixel 且误差阈值允许时，绘制一个 parent 比绘制两个独立 splats 更稳定。相机靠近后再展开 children，恢复可分辨结构。

## 6. 回到真实系统：程序实际上怎样工作

部署管线可以分成离线 hierarchy/quantization 与运行时 selection/filtering：

~~~text
trained Gaussian set
→ build spatial clusters
→ moment-match parent Gaussians
→ estimate parent rendering error
→ quantize parameters under bit budget
→ serialize hierarchy

runtime
→ frustum and projected-error selection
→ choose parent or children with hysteresis
→ add screen-space pixel filter
→ tile binning and compositing
→ collect distortion and frame-time statistics
~~~

anti-aliasing filter 必须跟 viewport resolution、camera intrinsics 和 render scale 一致。先以半分辨率渲染再放大时，pixel footprint 应按低分辨率定义，而不是沿用训练相机的一个 pixel。

hierarchy 构建要保留 stable IDs 与 parent-child ranges，便于 GPU 连续访问。runtime 不能递归执行大量不规则 CPU calls；可在 GPU 上按 projected error 生成 active list，再进入 tile binning。

量化应按 sensitivity 分配 bits。position、log-scale、rotation、opacity、color/SH 的误差传播不同；可以用 calibration views 测量参数 block quantization 对 image loss 的变化，再决定 codebook 或 bit width。

现代 alias-free Gaussian 方法明确把 3D smoothing 与 2D screen-space filtering结合，说明仅靠原始最小 footprint 无法覆盖 zoom-in/zoom-out 的采样问题。[Mip-Splatting, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Yu_Mip-Splatting_Alias-free_3D_Gaussian_Splatting_CVPR_2024_paper.pdf)

对 CTGS，detector pixel 本身也有有限面积。正确 projector 应积分 Gaussian 对 detector bin 的 footprint，或使用与 detector response 匹配的 filter。LOD parent 必须优先守恒 attenuation mass 与投影误差，而不是 RGB alpha。

## 7. 容易走错的岔路

把所有 Gaussian 最小半径设为 1 pixel 可以减少闪烁，却会在高分辨率近景中过度模糊。filter 应随 projected footprint 和输出采样率变化。

只做 2D filter 不能修复训练阶段已经学到的超高频 3D 表示。换分辨率时，world-space scale 与 screen filter 都可能需要约束。

moment-matched parent 保留前两阶 moments，不保证 alpha compositing 与 view-dependent color 完全相同。parent error 必须在多视角上测量，不能仅比较参数。

LOD threshold 没有 hysteresis 会在相机轻微抖动时 parent/children 反复切换。即使单帧误差小，temporal popping 仍明显。

最后，压缩率高不等于部署更快。复杂 entropy decode、随机内存访问或过多 codebook lookup 可能让 bytes 下降但 frame time 上升，所以 objective 中需要同时包含 \(B\) 与 \(T\)。

## 8. 本章落点、验证与下一章

本章从 pixel integration 推出了 Gaussian 抗锯齿：screen covariance 与 pixel filter covariance 相加，determinant ratio 近似保持 footprint 积分。moment matching 把 children 聚合为 parent，使远处只处理低频表示；rate-distortion objective 再把图像误差、存储与帧时间放入同一选择。

在 3DGS 项目中，本章对应 pixel filter、Gaussian hierarchy、active-level selection 和 parameter quantization。在 CTGS 中，filter 与 hierarchy 应以 detector integration和 attenuation mass 为准，而不是照搬 RGB opacity。

本章的 90 分钟验证是实现正文的 subpixel Gaussian filter，比较无 filter、只扩大 covariance 和带 determinant correction 三种渲染，在相机横向移动 0.1 pixel 时记录总能量与 temporal variance。预期是无 filter 闪烁最大，只扩大 covariance 会改变总能量，带 correction 的 temporal variance下降且积分更稳定。随后用两个 children 构造 moment-matched parent并比较远景误差。

下一章会转向几何约束与表面提取：当 Gaussian 场景需要可测量 mesh、碰撞或 CT 表面时，density/opacity 如何定义 surface，surface-aligned Gaussian 和 depth-normal consistency又怎样减少“图像正确但几何漂浮”的自由度。

