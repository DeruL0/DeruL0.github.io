# 可微三维渲染第 8 章：动态 Gaussian 为什么需要一个跨时间保持身份的参考空间

## 1. 从一个真实任务开始

上一章从静态 Gaussian field提取了可测 surface。现在输入变成同步多视角视频或 time-resolved CT：物体会转动、弯曲，甚至局部出现与消失。我们希望在任意时刻和任意视角渲染场景，同时追踪“这一时刻的 Gaussian 与上一时刻哪一个是同一材料点”。

真实任务是一段旋转并轻微变形的零件序列。若每帧独立训练一套 3D Gaussians，每帧图像都可能很好，但第 27 帧的第 1000 个 Gaussian与第 28 帧的第 1000 个没有任何关系。centers、scales和colors会重新分配，播放时出现 flicker，也无法计算稳定 velocity或把同一 defect跨时间追踪。

今天的任务是建立一个共享 canonical representation，再用时间条件 deformation将其映射到每个观测时刻。这样动态场景不再是许多无关的三维快照，而是同一组 primitives沿时间形成的轨迹。

## 2. 最直接的办法，以及它为什么不够

最直接的方法是为 \(T\) 帧各保存 \(N\) 个 Gaussians。存储从 \(O(N)\) 增长为 \(O(TN)\)，训练也不能利用相邻帧的重复结构。更严重的是，每帧优化的 permutation symmetry不同，跨帧 identity没有定义。

另一种方法是给每个 Gaussian只学习恒定速度：

\[
\mu_i(t)=\mu_i^0+t v_i.
\]

它适合短时间刚性平移，却不能表示旋转、周期运动、局部非刚性变形或速度变化。直接给每个时刻自由 offset又回到了逐帧独立参数，只是换了存储形式。

需要的关键不是“时间作为第四个输入”本身，而是把静态外观、长期身份和时变运动分开：canonical Gaussians保存对象是谁，deformation field说明它在时刻 \(t\) 到了哪里、朝向和尺度怎样变化。

## 3. 关键想法是怎样被引出来的

为每个 Gaussian定义 canonical attributes：

\[
G_i^0
=
\left(
\mu_i^0,\Sigma_i^0,\alpha_i^0,c_i^0
\right).
\]

时间条件 deformation field \(F_\theta\)接收 canonical position或feature与时间：

\[
\Delta G_i(t)
=
F_\theta(\mu_i^0,f_i,t).
\]

当前时刻的参数由 canonical state和deformation组合得到：

\[
G_i(t)
=
\mathcal D\left(G_i^0,\Delta G_i(t)\right).
\]

最简单的 center mapping是

\[
\mu_i(t)
=
\mu_i^0+\Delta\mu_i(t).
\]

同一个 index \(i\) 在全部时刻保留，因此 correspondence由表示结构给出，而不是事后最近邻匹配。时间正则也可以直接作用于同一 primitive的轨迹。

## 4. 一步一步建立正式模型

若 deformation是连续空间映射

\[
\phi_t:\mathbb R^3\rightarrow\mathbb R^3,
\]

则 center为

\[
\mu_i(t)=\phi_t(\mu_i^0).
\]

局部形状应随 mapping Jacobian变换。记

\[
J_i(t)
=
\frac{\partial\phi_t}{\partial x}
\bigg|_{\mu_i^0},
\]

一阶近似下 covariance为

\[
\Sigma_i(t)
=
J_i(t)\Sigma_i^0J_i(t)^{\mathsf T}.
\]

实际 4DGS常让网络直接预测 translation、rotation和scale residual，以保证 covariance保持正定；Jacobian公式说明了这些更新在几何上应表达什么。

给定时刻 \(t\) 和 camera \(C_t\)，differentiable splatting renderer产生

\[
\widehat I_t
=
\mathcal R\left(\{G_i(t)\},C_t\right).
\]

photometric objective为

\[
L_{\rm photo}
=
\sum_t
\ell(\widehat I_t,I_t).
\]

只用图像 loss时，deformation可以抖动并由 colors、opacity补偿。为了让轨迹平滑，可对离散帧使用速度与加速度：

\[
v_i^t
=
\frac{\mu_i^{t+1}-\mu_i^t}{\Delta t},
\]

\[
L_{\rm accel}
=
\sum_{i,t}
\left\|
\mu_i^{t+1}-2\mu_i^t+\mu_i^{t-1}
\right\|_2^2.
\]

对局部近似刚性区域，还可保持邻接距离：

\[
L_{\rm rigid}
=
\sum_t
\sum_{(i,j)\in E}
\left(
\|\mu_i(t)-\mu_j(t)\|_2
-
\|\mu_i^0-\mu_j^0\|_2
\right)^2.
\]

这些 regularizers不是假设所有物体静止，而是阻止没有图像证据支持的高频 identity交换。对突然运动或真实拓扑变化，权重必须降低或引入 scene flow、visibility与birth/death机制。

## 5. 跟着一个完整例子走到底

考虑一个 canonical Gaussian：

\[
\mu^0=(1,0,0)^{\mathsf T},
\qquad
\Sigma^0
=
\operatorname{diag}(0.04,0.01,0.01).
\]

物体在 \(t\in[0,1]\) 内绕 \(z\) 轴旋转 90 度。令

\[
\theta(t)=\frac{\pi}{2}t,
\]

\[
\phi_t(x)=R_z(\theta(t))x.
\]

在 \(t=0.5\) 时，旋转角为 45 度，center变为

\[
\mu(0.5)
=
R_z(45^\circ)(1,0,0)^{\mathsf T}
=
\left(
\frac{\sqrt2}{2},
\frac{\sqrt2}{2},
0
\right)^{\mathsf T}.
\]

因为这是刚性旋转，Jacobian就是

\[
J=R_z(45^\circ).
\]

所以 covariance为

\[
\Sigma(0.5)
=
R_z(45^\circ)
\Sigma^0
R_z(45^\circ)^{\mathsf T}
=
\begin{bmatrix}
0.025&0.015&0\\
0.015&0.025&0\\
0&0&0.01
\end{bmatrix}.
\]

原来沿 \(x\) 的长轴跟随物体转到 45 度，而不是只有 center移动、椭球仍朝旧方向。若 renderer只更新 \(\mu\) 不更新 \(\Sigma\)，Gaussian footprint会在运动中看似滑动或挤压。

再取三个时刻 \(t=0,0.5,1\)。center依次为

\[
(1,0,0),\quad
(0.707,0.707,0),\quad
(0,1,0).
\]

它的速度方向随时间旋转，因此恒定线性速度模型会预测中点 \((0.5,0.5,0)\)，偏离真实圆弧。canonical deformation使用同一 Gaussian identity，并能通过非线性 \(\phi_t\)表达正确轨迹。

## 6. 回到真实系统：程序实际上怎样工作

一个 canonical dynamic-GS pipeline为：

~~~text
synchronized images and calibrated cameras
→ initialize canonical Gaussians
→ encode time and canonical features
→ predict translation / rotation / scale / appearance residuals
→ render deformed Gaussians at each timestamp
→ optimize photometric, motion and geometry losses
→ prune or densify with cross-time consistency
→ render, track or extract surfaces at arbitrary time
~~~

time encoding可以是 Fourier features、temporal grid、4D planes或learned embeddings。deformation MLP适合平滑共享运动；explicit trajectories更容易解释；native 4D Gaussian则直接在 \(x,y,z,t\) 中定义 covariance，通过时间切片得到某一时刻的 3D Gaussian。它们对运动先验、存储与突变处理各有不同偏向。

CVPR 2024 的 4D-GS使用 3D Gaussians、分解的 4D neural voxels和轻量 MLP预测时间相关 deformation，目标正是避免逐帧独立 3DGS，同时保持实时渲染。[4D Gaussian Splatting](https://openaccess.thecvf.com/content/CVPR2024/html/Wu_4D_Gaussian_Splatting_for_Real-Time_Dynamic_Scene_Rendering_CVPR_2024_paper.html)

对动态 CTGS，观测模型比 RGB更严格。第 \(j\) 条 X-ray在 acquisition time \(t_j\) 测得：

\[
\lambda_j
=
b_j
\exp\left[
-\int_{\text{ray}_j}
\mu(x,t_j)\,ds
\right]+r_j.
\]

因此 motion field必须在每条 projection的真实时间点评估。若把整个 rotation的 projections都当成同一个静态 volume，运动会被 reconstruction吸收成 blur和double edges。canonical attenuation field加 deformation可以把不同时间的 measurements拉回同一参考空间。

densification也不能逐帧各做各的。新增 primitive需要定义 canonical identity及其整个时间轨迹；只在某一帧新增会造成 temporal popping。可依据跨时间累计 gradient或visibility决定结构编辑。

## 7. 容易走错的岔路

canonical space不一定等于第一帧。第一帧可能遮挡严重或处在极端形变；canonical是优化坐标系，可以选择中性 pose或隐式学习。

过强 temporal smoothness会抹掉碰撞、快速运动和真实拓扑变化。正则化表达的是先验，不能凌驾于同步多视角证据。

只正则 center trajectory不够。rotation、scale、opacity和appearance仍可能逐帧抖动并补偿图像；要检查每类 attribute的时间频谱与物理含义。

单目视频中的 motion与depth高度歧义。模型可以用沿 view direction的变形解释同一 optical flow；需要多视角、depth、scene flow、rigidity或其他监督降低歧义。

最后，把 time直接拼到 color MLP可能得到随时间变化的正确像素，却没有正确几何运动。必须单独评价 trajectory、depth、surface consistency与novel-time interpolation。

## 8. 本章落点、验证与下一章

本章把动态 Gaussian建立为 canonical primitives与时间 deformation的组合。canonical index提供跨帧身份，\(\phi_t\)移动 centers，Jacobian或显式 rotation/scale更新 covariance；photometric loss之外的 acceleration与local-rigidity约束抑制无依据抖动。

在动态 3DGS 中，本章对应 temporal encoder、deformation network、attribute update和cross-time densification。在 4D CTGS 中，它进一步对应 acquisition timestamp、motion-compensated projector和canonical attenuation field。

本章的 90 分钟验证是实现正文绕 \(z\) 轴的单 Gaussian deformation，检查 \(t=0.5\) 的 center与 covariance矩阵；随后对 20 个时间点渲染，并比较“只移动 center”和“同步旋转 covariance”的 footprints。预期是前者的椭球方向滞留，后者沿圆弧保持刚性；线性轨迹在中点出现可测位置误差。

下一章将进入材质与光照的可训练分解。动态对应建立后，若场景还要 relighting，就必须把 geometry、BRDF、environment illumination与曝光从 view-dependent color中拆开，否则时间变化和光照变化仍会相互冒充。
