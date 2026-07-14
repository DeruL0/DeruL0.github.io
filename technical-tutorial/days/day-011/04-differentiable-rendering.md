# 可微三维渲染第 11 章：可编辑三维表示为什么需要语义，而不只是颜色和密度

## 1. 从一个真实任务开始

上一章解决了 visibility 的梯度问题，让几何边界能在 silhouette、depth 和 shadow 监督下移动。现在用户的问题更高一层：不是“让这个像素变对”，而是“选中这个零件”“把金属螺钉变暗”“删除这个支架”“只导出孔隙连通区域”。颜色、密度和几何本身不能回答对象身份。

真实任务是一个 CTGS 或 3DGS viewer：场景中有塑料外壳、金属螺钉、孔隙、支撑结构和背景。用户希望点击或用文本选中“螺钉”，修改其材质或单独导出 mesh。若每个 Gaussian 只有 position、scale、opacity 和 color，系统只能按空间框或颜色阈值选择；同色不同物体会混在一起，被遮挡但同一物体的部分也不会自动相连。

今天的任务是把语义特征、实例分组和编辑操作放进可微三维表示。语义不是贴在最终截图上的标签，而是随三维 primitive 一起被渲染、监督和一致化的属性。

## 2. 最直接的办法，以及它为什么不够

最直接的方法是在每个训练视角上跑 2D segmentation，然后把 mask 贴回图像。它能选中当前视角的对象，但换视角时 mask 需要重算，遮挡区域没有三维身份；同一个物体在不同视角的 mask id 也不一定一致。

第二个方法是根据颜色或空间距离聚类 Gaussians。它能处理简单物体，却会把颜色相近的不同部件合并，或者把一个颜色变化大的物体拆开。语义对象不等同于 RGB cluster。

第三个方法是训练完 radiance field 后再做点云分割。这把语义放到后处理阶段，不能在训练中利用多视角一致性，也不能让编辑操作直接作用于渲染表示。

因此需要把语义或实例特征作为 primitive attribute，让它像颜色一样参与 differentiable rendering。这样每个视角的 2D 监督都能反向影响同一个三维对象身份。

## 3. 关键想法是怎样被引出来的

对每个 Gaussian \(i\)，原本有

\[
G_i=(\mu_i,\Sigma_i,\alpha_i,c_i).
\]

现在增加一个语义特征或 identity embedding：

\[
G_i=(\mu_i,\Sigma_i,\alpha_i,c_i,f_i).
\]

渲染 RGB 时使用 \(c_i\)，渲染 semantic feature map 时使用 \(f_i\)。同一个 alpha compositing 权重把几何可见性和语义投影联系起来：

\[
F(p)
=
\frac{\sum_i T_i(p)\alpha_i(p)f_i}
{\sum_i T_i(p)\alpha_i(p)+\epsilon}.
\]

若有 2D foundation model 给出每个 pixel 的特征、mask 或类别概率，就可以让 \(F(p)\) 与 2D feature 对齐。这样不同视角看到的同一个 Gaussian 会被拉向一致语义。

近年的方法正是沿这条路走。LERF 把语言嵌入蒸馏进 radiance field，使 3D 位置可被自然语言查询。[LERF](https://www.lerf.io/) Feature 3DGS 给 3D Gaussians 增加高维语义特征，用 2D foundation model 蒸馏，实现分割和语言引导编辑。[Feature 3DGS](https://feature-3dgs.github.io/) Gaussian Grouping 则给 Gaussian 增加 identity encoding，用 2D masks 和 3D consistency 形成实例级分组与局部编辑。[Gaussian Grouping](https://arxiv.org/abs/2312.00732)

## 4. 一步一步建立正式模型

先定义 feature rendering。第 \(i\) 个 Gaussian 对 pixel \(p\) 的可见权重为

\[
w_i(p)=T_i(p)\alpha_i(p).
\]

RGB 渲染是

\[
C(p)=\sum_i w_i(p)c_i.
\]

语义特征渲染使用同一组权重：

\[
F(p)=\frac{\sum_i w_i(p)f_i}{\sum_i w_i(p)+\epsilon}.
\]

若 2D 模型给出目标特征 \(F^\star(p)\)，可用 cosine loss：

\[
L_{\rm feat}
=
\sum_p
\left[
1-
\frac{F(p)^{\mathsf T}F^\star(p)}
{\|F(p)\|\|F^\star(p)\|+\epsilon}
\right].
\]

若监督是 2D mask \(m_k(p)\)，可为每个 Gaussian 学 instance logits \(s_i\)，渲染成概率：

\[
P_k(p)
=
\sum_i w_i(p)\operatorname{softmax}(s_i)_k.
\]

再用 cross entropy 与 mask 对齐。为了让同一物体的 Gaussians 聚在一起，可加入空间一致性：

\[
L_{\rm smooth}
=
\sum_{(i,j)\in E}
\omega_{ij}\|f_i-f_j\|^2,
\]

其中 \(E\) 是邻近 Gaussian 图。编辑时，用户查询一个文本 embedding \(e\)，每个 Gaussian 的相关性可写为

\[
r_i=
\frac{f_i^{\mathsf T}e}{\|f_i\|\|e\|+\epsilon}.
\]

选择

\[
\mathcal S=\{i\mid r_i>\tau\}
\]

后，颜色、材质、opacity、导出或删除操作都作用于这组三维 primitives，而不是某一张 2D 图上的 mask。

## 5. 跟着一个完整例子走到底

设一个 pixel 由两个 Gaussian 贡献。第一个属于塑料壳，语义特征为

\[
f_1=(1,0),
\]

第二个属于金属螺钉，语义特征为

\[
f_2=(0,1).
\]

可见权重为

\[
w_1=0.7,\qquad w_2=0.3.
\]

渲染出的 feature 是

\[
F(p)
=
\frac{0.7(1,0)+0.3(0,1)}{0.7+0.3}
=(0.7,0.3).
\]

若用户查询“metal”，文本 embedding 简化为

\[
e=(0,1).
\]

pixel feature 与查询的点积为

\[
F(p)^{\mathsf T}e=0.3.
\]

这个 pixel 主要是塑料，所以相关性不高。但对单个 Gaussian 来看，

\[
f_2^{\mathsf T}e=1,
\]

因此编辑系统会选择第二个 Gaussian，而不是把整个 mixed pixel 都改成金属。若只基于 2D mask，边界混合像素很难处理；primitive-level feature 可以把编辑落回三维对象。

现在执行“把金属螺钉变暗”。选择集合为

\[
\mathcal S=\{2\}.
\]

只更新

\[
c_2\leftarrow0.5c_2,
\]

不改变 \(c_1\)。换一个视角后，只要同一个 Gaussian 可见，编辑仍然保持一致。这就是三维语义优于逐帧 2D mask 的地方。

## 6. 回到真实系统：程序实际上怎样工作

一个语义 3DGS pipeline 通常先训练或加载 radiance Gaussians，再加入 feature head 或 identity embedding。训练数据来自 2D segmentation、SAM masks、CLIP/LSeg features、人工标签或 CT 材料标签。渲染时同时输出 RGB、feature map、instance probability、depth 和 alpha。

多视角一致性是核心。2D foundation model 的结果会有噪声，同一物体在不同视角可能被分成不同 mask。三维 feature 通过共享 Gaussian identity 把这些监督融合；空间 smoothness 和 mask consistency 防止特征在同一物体内部破碎。

对 CTGS，语义来源可能比 RGB 更可靠：材料 label、孔隙连通分量、CAD part id 或 defect annotation 都可以成为 Gaussian feature。需要明确区分 semantic feature 与物理 attenuation。一个 Gaussian 可以同时有 X-ray attenuation、可视化颜色、材料类别和实例 id，这些属性服务不同任务。

编辑系统要保留 provenance。若用户删除一个语义组，应记录选择条件、阈值、被影响的 Gaussian ids、后续 hole filling 或 inpainting 策略。否则编辑后的场景无法追踪哪些结构来自原始测量，哪些来自后处理。

## 7. 容易走错的岔路

第一个误区是把 2D mask 当成 3D truth。2D 模型会受视角、遮挡和纹理影响；三维语义必须通过多视角一致性和几何约束过滤噪声。

第二个误区是只在 pixel 上存语义。Pixel feature 换视角就变；primitive feature 才能支持一致编辑和导出。

第三个误区是把语义与材质混同。金属材质、螺钉实例、缺陷区域和 X-ray attenuation 是不同属性。编辑一个不应无意改变另一个。

第四个误区是阈值选择不记录。语义查询常是连续相关性，阈值会影响边界。工程系统应显示不确定区域，并保存选择参数。

最后，语言查询不等于可靠计量。对工业 CT，语义编辑可以辅助选择区域，但尺寸、孔隙率和缺陷判断仍要回到物理标定和几何测量。

## 8. 本章落点、验证与下一章

本章把可微三维表示从颜色、密度和 visibility 推进到语义特征与实例分组。每个 Gaussian 增加 \(f_i\) 或 identity embedding，通过同一 alpha compositing 权重渲染 feature map；2D masks、语言特征或材料标签反向监督三维 primitive；编辑操作作用于选中的 primitive 集合。

在可微渲染和 CTGS 项目中，本章对应 feature rendering、mask/CLIP/SAM 蒸馏、材料与实例属性分离、语义选择、局部编辑和 provenance 记录。

本章的 60 到 90 分钟验证是实现两个 Gaussian 的 feature compositing 例子：设置 \(f_1=(1,0)\)、\(f_2=(0,1)\)、权重 \(0.7,0.3\)，计算 pixel feature，并用 query \(e=(0,1)\) 选择金属 Gaussian。预期 pixel 相关性只有 0.3，但 primitive 2 的相关性为 1，因此编辑应落在 primitive 2 而不是整个 pixel。

下一章将进入率失真和部署。语义、材质和 feature 让表示更可用，但也增加存储、带宽和渲染成本；部署时必须决定哪些属性保留、量化、压缩或按需加载。
