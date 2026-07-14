# 可微三维渲染第 4 章：一个三维 Gaussian 怎样变成屏幕上的椭圆

## 1. 从一个真实任务开始

上一章用连续体渲染表达了可见性，却要沿每条 ray 采许多 points。为了实时训练和显示，我们希望使用数量有限、具有空间范围的 primitives：每个 primitive 能投影到一片 pixels，位置和尺度的小变化又能连续改变 footprint。3D Gaussian Splatting 选择各向异性三维 Gaussian，并把它们投影成屏幕椭圆后按 tile 光栅化。

今天的输入是 Gaussian 的三维中心、旋转、尺度、opacity 和颜色，以及相机参数；输出是二维均值、二维 covariance、pixel weights 和最终颜色。必须理解从三维 covariance 到二维 ellipse 的推导，才能判断尺度初始化、数值稳定和 gradient 是否正确，而不是把 rasterizer 当成黑盒。

## 2. 最直接的办法，以及它为什么不够

最直接的点云渲染把每个三维点投到一个 pixel。它速度快，但点的位置在 pixel centers 之间移动时覆盖集合跳变，远处点之间出现孔洞，单个点也没有可训练的空间尺度。

把每个点画成固定半径圆盘可以填洞，却忽略透视与三维形状。同一个 world-space primitive 远近应有不同屏幕尺寸；一个沿某方向拉长的三维结构，换相机后应呈现不同椭圆。

因此 primitive 必须在三维中拥有 covariance，投影时由相机 Jacobian 把局部不确定性传播到二维。这样屏幕 footprint 不是经验半径，而是三维尺度、旋转、深度和相机焦距共同决定的结果。

## 3. 关键想法是怎样被引出来的

三维 Gaussian 可看作中心附近一小团椭球密度。对中心附近的点，透视投影虽然是非线性的，但可以用一阶 Taylor expansion 近似：

\[
\pi(\mu+\delta)
\approx
\pi(\mu)+J\delta.
\]

线性变换会把 Gaussian 仍然变成 Gaussian，covariance 按

\[
\Sigma' = J\Sigma J^{\mathsf T}
\]

传播。因此只要计算相机坐标系中的三维 covariance 与投影 Jacobian，就能得到屏幕椭圆。

这种一阶近似适用于 Gaussian 相对深度不太大、没有跨过相机平面时。Gaussian 尺度过大或太靠近相机，透视畸变不再由单个 ellipse 精确描述，所以实现需要 near-plane culling、尺度约束和 covariance regularization。

## 4. 一步一步建立正式模型

用中心 \(\mu\in\mathbb R^3\) 和正半定 covariance \(\Sigma\) 定义未归一化 Gaussian：

\[
G(\mathbf x)
=
\exp\left[
-\frac12
(\mathbf x-\mu)^{\mathsf T}
\Sigma^{-1}
(\mathbf x-\mu)
\right].
\]

为了保证训练中 \(\Sigma\) 不变成负尺度，常把它参数化为

\[
\Sigma
=
R\operatorname{diag}(s_x^2,s_y^2,s_z^2)R^{\mathsf T},
\]

其中 \(R\) 来自归一化 quaternion，尺度 \(s_i>0\) 可由指数或其他正值映射产生。

设 world-to-camera 线性部分为 \(W\)，则相机空间中心和 covariance 为

\[
\mu_c=W\mu+t,
\qquad
\Sigma_c=W\Sigma W^{\mathsf T}.
\]

对相机坐标 \((x,y,z)\)，针孔投影为

\[
u=f_x\frac{x}{z}+c_x,
\qquad
v=f_y\frac{y}{z}+c_y.
\]

在 Gaussian 中心处的 Jacobian 是

\[
J=
\begin{bmatrix}
f_x/z & 0 & -f_x x/z^2\\
0 & f_y/z & -f_y y/z^2
\end{bmatrix}.
\]

于是二维 covariance 近似为

\[
\Sigma_{2D}
=
J\Sigma_cJ^{\mathsf T}
+\sigma_{\rm pix}^2I.
\]

最后一项是最小 pixel footprint 或低通项，防止投影 covariance 接近奇异并减少亚像素跳变。

屏幕中心为 \(m=\pi(\mu_c)\)。pixel \(p\) 的 footprint 为

\[
g(p)
=
\exp\left[
-\frac12(p-m)^{\mathsf T}
\Sigma_{2D}^{-1}(p-m)
\right].
\]

若 Gaussian opacity 参数为 \(o\)，其 pixel alpha 常写成

\[
\alpha(p)=\operatorname{clamp}(o\,g(p),0,\alpha_{\max}).
\]

对按深度排序的 Gaussians 做 front-to-back 合成：

\[
C(p)
=
\sum_i T_i(p)\alpha_i(p)c_i,
\qquad
T_i(p)=\prod_{j<i}[1-\alpha_j(p)].
\]

这与上一章的离散体合成同构，但 primitive footprint 已经由解析 ellipse 给出，不需要沿 ray 均匀采许多 points。

## 5. 跟着一个完整例子走到底

设相机焦距 \(f_x=f_y=100\) pixels，principal point 为零。一个 Gaussian 位于相机坐标

\[
\mu_c=(0,0,2),
\]

三维 covariance 为

\[
\Sigma_c
=
\operatorname{diag}(0.01,0.0025,0.04).
\]

因为中心位于光轴上，投影 Jacobian 中与 \(z\) 扰动相关的第三列为零：

\[
J=
\begin{bmatrix}
50&0&0\\
0&50&0
\end{bmatrix}.
\]

忽略最小 pixel 项，二维 covariance 为

\[
\Sigma_{2D}
=
J\Sigma_cJ^{\mathsf T}
=
\begin{bmatrix}
25&0\\
0&6.25
\end{bmatrix}.
\]

所以屏幕标准差分别为 5 pixels 和 2.5 pixels。三维中 \(x\) 方向尺度是 \(y\) 的两倍，投影后确实成为横向更宽的椭圆。

取 pixel \(p=(5,0)\)，它距离中心一个横向标准差：

\[
g(p)
=
\exp\left(-\frac12\frac{5^2}{25}\right)
=e^{-1/2}
\approx0.607.
\]

若 opacity \(o=0.8\)，该 pixel 的 alpha 为

\[
\alpha_1\approx0.8\times0.607=0.485.
\]

设它颜色为红色，后方另一个蓝色 Gaussian 在同 pixel 的 alpha 为 0.3。合成颜色为

\[
C
=
0.485\,c_{\rm red}
+(1-0.485)0.3\,c_{\rm blue}.
\]

因此红色权重为 0.485，蓝色权重为约 0.155，剩余背景 transmittance 为

\[
(1-0.485)(1-0.3)
\approx0.361.
\]

从三维尺度到 ellipse，再到一个 pixel 的 Gaussian weight、alpha 与最终颜色，整条前向路径由此闭合。

## 6. 回到真实系统：程序实际上怎样工作

典型 rasterizer 先在每帧把 Gaussian 变换到相机空间，计算二维 mean/covariance 和 ellipse bounding box，剔除 near-plane 外或屏幕外 primitives。随后根据 bounding box 把 Gaussian 复制到覆盖的 screen tiles，按 tile 和深度排序，再由每个 tile 的 pixel threads front-to-back 合成。

```text
3D parameters
→ camera transform and covariance projection
→ ellipse bounds and tile duplication
→ sort by tile/depth
→ pixel Gaussian evaluation
→ alpha compositing and early termination
→ backward to color, opacity, mean, scale and rotation
```

covariance 必须保持正定，二维逆矩阵需要 determinant 下界；ellipse bounds 应基于选定置信半径，例如若干标准差。过小 bounds 会截断贡献并产生边界梯度错误，过大 bounds 会增加 tile list 和 overdraw。

原始 3D Gaussian Splatting 工作把各向异性 Gaussian、tile-based rasterization 与交替 densification 组合成实时新视角合成系统。[项目与论文](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) 后续方法会修改抗锯齿、层次表示或压缩，但二维 covariance 投影仍是理解许多变体的基础。

对 CTGS，不能直接把 RGB alpha compositing 当作 X 射线 forward。三维 Gaussian 沿 X 射线的 line integral 也有解析 Gaussian 形式，负对数投影应累加各 Gaussian 的衰减贡献；screen ellipse 可以用于 footprint culling 和加速，但深度排序与颜色遮挡未必属于 CT 物理。

## 7. 容易走错的岔路

直接优化 covariance 的九个矩阵元素看起来最自由，却可能产生非对称或非正定矩阵，使 ellipse inverse 和梯度失效。rotation 加正尺度的参数化是在表达合法域。

给二维 covariance 加很大的 \(\sigma_{\rm pix}^2I\) 会稳定训练，却让所有 Gaussian 保持模糊。最小 footprint 是采样保护，不应替代正确三维尺度。

按 Gaussian 中心深度排序并不等于所有像素处的严格顺序。大型重叠 ellipsoids 可能在空间中交叉，单一 center depth 只是近似；排序交换也会带来不连续。

把 opacity clamp 当成纯数值细节也不准确。接近 clamp 后梯度会改变或消失，并影响 densification 判断。应记录饱和 Gaussian 比例。

最后，RGB 渲染变好不保证三维 covariance 物理正确。尺度、opacity 和颜色可以互相补偿；多视角、几何先验和 densification policy 共同决定可辨识性。

## 8. 本章落点、验证与下一章

本章从固定大小 point sprite 推导到真正的三维 Gaussian splat。三维 covariance 经相机变换和透视 Jacobian传播为二维 covariance，screen-space Mahalanobis distance 产生连续 ellipse footprint，再通过 opacity 与有序 alpha compositing形成像素。这个模型把位置、尺度和旋转都接入可微前向。

在 3DGS 项目中，本章对应 covariance projection、tile bounds、sorting 和 compositor。在 CTGS 中，Gaussian 几何可复用，但观测模型应改为 X 射线 line integral，而不是 RGB visibility。

本章的 90 分钟验证是实现单 Gaussian 的 CPU reference projector，复现正文中 \(5\times2.5\) pixel 标准差与 \(g(5,0)\approx0.607\)。旋转三维 covariance 并移动 Gaussian 深度，绘制二维 ellipse；预期屏幕尺度与 \(1/z\) 近似成比例，off-axis 时 Jacobian 的第三列会让深度方差进入二维 covariance。再用有限差分检查二维 mean 和 scale 的梯度。

下一章会处理固定 Gaussian 集合无法覆盖复杂场景的问题：什么时候应 split、clone 或 prune，梯度大小为什么不能单独决定 densification，以及结构变化如何与 optimizer state、显存和实时 rasterization 成本共同控制。
