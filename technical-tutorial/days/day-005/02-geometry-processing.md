# 几何处理第 5 章：怎样改变三角形连接而不改变原来的曲面

## 1. 从一个真实任务开始

上一章已经把 CT 表面上的法向噪声压低，并尽量保护真实棱边。但网格本身仍可能很差：Marching Cubes 生成的三角形受 voxel grid 控制，局部有细长三角形、过密平坦区和过稀高曲率区。这样的网格会降低曲率计算、有限元求解、碰撞检测和实时渲染的效率。

今天的任务不是继续移动同一批 vertices，而是改变采样本身：长边需要 split，过短边需要 collapse，不合理的对角线需要 flip，剩余 vertices 再沿曲面切向重新分布。目标是获得边长接近目标尺度、三角形角度更均匀、同时仍贴近原始表面的网格。

这叫 remeshing。它与 smoothing 的差别在于拓扑连接和顶点数量会改变，但通常希望物体的几何形状与大尺度拓扑保持不变。

## 2. 最直接的办法，以及它为什么不够

最直接的做法是把每个三角形均匀细分。它能消除过长边，却让原本已经过密的平面继续增大数据量，短边和细长三角形也不会自动消失。

另一个直接办法是只删除小三角形。若直接移除 face，会产生孔洞；若把短边两端合并，却不检查邻域，可能把两个本不应连接的表面粘在一起、翻转相邻 triangles，或者改变 genus。

所以重网格化不能由单一操作完成。split 解决采样不足，collapse 解决冗余，flip 改善连接质量，relaxation 改善点分布。每个操作还必须通过拓扑合法性、几何误差和 feature constraints 三类检查。

## 3. 关键想法是怎样被引出来的

各向同性 remeshing 的关键抽象是目标边长场 \(h(\mathbf x)\)。常数 \(h\) 希望全表面近似均匀采样；随曲率变化的 \(h\) 可以让高曲率区域更密、平坦区域更稀。局部操作都围绕当前 edge length 与目标值的比例作决定。

但边长只是采样目标，不是几何误差。一个平面上的长边可安全 collapse，薄壁两侧即使空间接近也不能合并。于是每次候选操作都要回答：

1. 局部 connectivity 是否仍为 manifold？
2. 新 triangles 是否翻转或退化？
3. 新位置到原始曲面的误差是否在容许范围？
4. 是否跨越边界、材料 seam 或 feature edge？

这使 remeshing 成为“局部修改加全局不变量”的过程，而不是简单按长度删点。

## 4. 一步一步建立正式模型

对 edge \(e=(i,j)\)，长度为

\[
\ell_{ij}=\|v_i-v_j\|.
\]

若目标边长为 \(h\)，常见的滞回阈值是

\[
\ell_{ij}>\frac43h
\quad\Rightarrow\quad
\text{split candidate},
\]

\[
\ell_{ij}<\frac45h
\quad\Rightarrow\quad
\text{collapse candidate}.
\]

两个阈值之间保留缓冲，避免同一 edge 在相邻迭代中反复 split/collapse。具体常数可以调整，但必须保持 hysteresis。

split 通常在 midpoint 或投影到 reference surface 后的位置插入 vertex，并把 incident faces 分开。collapse 把 \(i,j\) 合并为一个 vertex \(q\)。拓扑上要满足 link condition：两个端点的一环交集应与该 edge 的 link 一致，否则 collapse 可能生成非流形邻域或改变局部拓扑。

几何上可检查所有新 face 的有向面积。若旧法向为 \(n_f\)，新三角形两边为 \(a,b\)，至少要求

\[
n_f^{\mathsf T}(a\times b)>0
\]

且面积高于阈值，防止翻转和退化。候选位置 \(q\) 还应投影到 reference surface，或使局部 quadric error 不超过上限。

edge flip 不改变 vertex 数，只把两个相邻 triangles 的公共对角线换成另一条。各向同性网格内部 vertex 的理想 valence 约为 6，边界 vertex 约为 4。可定义局部 valence energy

\[
E_{\rm val}
=
\sum_{v\in\mathcal N}
(\operatorname{valence}(v)-v^\ast)^2,
\]

仅在 flip 后能量下降且 triangles 合法时接受。

最后做 tangential relaxation。令邻居 centroid 为 \(\bar v_i\)，vertex normal 为 \(n_i\)。直接 Laplacian 位移会沿 normal 改变形状；只保留切向分量：

\[
d_i
=
(I-n_in_i^{\mathsf T})(\bar v_i-v_i).
\]

移动后再投影回 reference surface。这样主要改善采样分布，而不是继续平滑几何。

## 5. 跟着一个完整例子走到底

设目标边长 \(h=1\)。一个三角形顶点为

\[
A=(0,0),\quad
B=(1.8,0),\quad
C=(0.9,0.8).
\]

底边长度为

\[
\ell_{AB}=1.8>\frac43,
\]

所以它是 split candidate。在 midpoint

\[
M=(0.9,0)
\]

插入新 vertex，原三角形变成 \(AMC\) 和 \(MCB\)。新边长度为

\[
|AM|=|MB|=0.9,\qquad |MC|=0.8,
\]

而

\[
|AC|=|BC|\approx1.204.
\]

所有边都进入目标附近，且两个新 triangles 朝向保持一致。

假设相邻区域还有 vertex

\[
D=(0.9,0.3),
\]

于是 \(|MD|=0.3<0.8h\)，成为 collapse candidate。候选位置可取中点

\[
Q=(0.9,0.15).
\]

程序不会立即接受，而是先检查 \(M,D\) 的一环 link、删除重复 faces、验证所有新三角形面积与朝向，并把 \(Q\) 投影回 reference surface。只有这些条件都通过，短边才真正消失。

再看一个 flip 的局部 valence 例子。旧对角线两端 valence 为 8、8，另外两个对角顶点为 4、4。以理想 valence 6 计算，旧能量为

\[
(8-6)^2+(8-6)^2+(4-6)^2+(4-6)^2=16.
\]

flip 后旧端点各减 1、对角点各加 1，变成 7、7、5、5，能量为

\[
4\times(1^2)=4.
\]

若新对角线不破坏 feature、三角形质量也合法，这次 flip 明显改善连接。

最后，某 vertex 到邻居 centroid 的位移为 \((0.2,0.1,0.3)\)，局部 normal 为 \((0,0,1)\)。切向投影得到

\[
d=(0.2,0.1,0),
\]

因此 vertex 在表面内重排，不直接沿 normal 抬高 0.3。split、collapse、flip 与 relaxation 至此形成一轮完整输入到输出。

## 6. 回到真实系统：程序实际上怎样工作

典型 isotropic remeshing 迭代按固定顺序执行：

~~~text
mark protected boundaries and features
→ split long edges
→ collapse short edges
→ flip edges that improve valence/quality
→ tangential relaxation
→ project to reference surface
→ validate topology and geometric error
~~~

edge 操作会使 half-edge handles 失效，因此数据结构必须提供安全的局部更新和 stable property transfer。normal、UV、material、CT confidence 和 feature labels 不能只复制一个端点；每种属性都需要明确插值或禁止跨 seam collapse。

reference surface 可以是原始 mesh、隐式 signed distance field 或 CT volume 的等值面。投影到已经反复 remesh 的当前表面会累积漂移；保留独立 reference 才能控制长期误差。

对高曲率 adaptive sizing，可令

\[
h(\mathbf x)
=
\operatorname{clamp}
\left(
\frac{c}{\sqrt{|\kappa(\mathbf x)|+\epsilon}},
h_{\min},h_{\max}
\right),
\]

但曲率本身受噪声影响。应先完成可靠去噪，并平滑 sizing field，避免相邻区域目标长度突变。

## 7. 容易走错的岔路

只检查 edge length 就 collapse 看似局部合理，却可能跨过薄壁或粘合相邻壳层。topological link 与 reference-surface error 都不可省略。

把新 vertex 永远放在 midpoint 对平面足够，在曲面上会离开原表面。插值后应投影，或使用局部曲面参数。

edge flip 只追求 valence 也可能制造极小角、穿过 feature edge 或破坏 Delaunay 性质。valence 是质量代理，不是唯一标准。

tangential smoothing 不是绝对不改形。离散 normal 和 reference projection都有误差，多轮后仍会漂移，因此要记录 Hausdorff distance 和 volume。

最后，三角形更均匀不等于测量更准确。若原网格在缺陷附近已欠采样，先 collapse 再谈误差上限会永久删除结构。target size 必须由任务分辨率约束。

## 8. 本章落点、验证与下一章

本章把网格质量改善写成四种互补局部操作。split 补充长边采样，collapse 删除短边冗余，flip 改善 connectivity，tangential relaxation 重排 vertices；link condition、face orientation、reference error 和 feature masks 保证这些操作不会随意改变曲面。

在 CT mesh 项目中，本章对应 remeshing kernel、reference-surface projector 和 property transfer；在 STL/DirectX 项目中，它能把不规则提取网格转成更适合曲率、碰撞和 GPU cluster 构建的输入。

本章的 90 分钟验证是对带噪后已去噪的 sphere 或机械平面网格执行五轮 isotropic remeshing，记录 edge-length histogram、最小角、vertex valence、Hausdorff distance 和封闭体积。预期是边长集中到目标 \(h\) 附近、极小角减少，而 reference error 保持在设定阈值内；关闭 link condition 后，应能构造出非流形 collapse 反例。

下一章会从“保持几何误差的同时主动减少三角形数”进入 mesh simplification。Quadric Error Metrics 怎样为每次 edge collapse 估计平面偏差，priority queue 又怎样组织全局最便宜的 collapse，将成为下一问题。

