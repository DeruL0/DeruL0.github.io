# 几何处理第 6 章：怎样用一个二次型决定哪条边最值得被删除

## 1. 从一个真实任务开始

上一章通过 split、collapse、flip 和 tangential relaxation 改善了三角形质量，但目标边长固定后，triangle count 不一定显著下降。现在我们有一个两百万 triangles 的 CT 表面，希望生成二十万 triangles 的交互版本，同时让孔口、平面和锐利棱边的几何偏差可量化。

这不是各向同性 remeshing，而是 simplification：主动减少 vertices 与 faces，并尽量把删除造成的曲面误差控制在最小。核心问题变成：在所有可 collapse edges 中，下一条应该选哪一条？候选新 vertex 应放在哪里？怎样让选择不仅依据 edge length，还依据它对原表面的破坏？

Quadric Error Metrics，简称 QEM，把一个 vertex 偏离邻近 face planes 的平方距离累计成 \(4\times4\) 矩阵，使每次 collapse 的误差和最优位置都能快速计算。

## 2. 最直接的办法，以及它为什么不够

最直接的简化是不断 collapse 最短 edge。短边通常表示局部过密，但长度不等于形状重要性。锐利折线附近可能有很短的 edges，平坦大面上则可能有较长 edges；按长度优先会先破坏棱边，却保留平面上的冗余。

另一种做法是 collapse 后计算新 mesh 到完整原 mesh 的 Hausdorff distance，再选择误差最小者。它概念正确，却要求为大量候选重复做最近点查询，成本太高。

我们需要一个局部、可累积、能在 edge collapse 后快速更新的误差代理。face plane 正好提供这种结构：如果新 vertex 仍接近被删除邻域原来的所有 planes，局部表面通常不会偏离太多。

## 3. 关键想法是怎样被引出来的

一个归一化平面写成

\[
ax+by+cz+d=0,
\qquad
a^2+b^2+c^2=1.
\]

点 \((x,y,z)\) 到该平面的有符号距离就是左侧。平方距离可写成齐次向量的二次型。令

\[
\bar v=
\begin{bmatrix}
x&y&z&1
\end{bmatrix}^{\mathsf T},
\qquad
p=
\begin{bmatrix}
a&b&c&d
\end{bmatrix}^{\mathsf T}.
\]

则

\[
(p^{\mathsf T}\bar v)^2
=
\bar v^{\mathsf T}(pp^{\mathsf T})\bar v.
\]

因此每个 plane 对应一个 quadric \(K_p=pp^{\mathsf T}\)。多个 planes 的平方距离可以直接把 matrices 相加。一个 vertex 周围所有 incident face planes 的总误差只需一个矩阵 \(Q\) 表示。

当 edge 两端合并时，把两个 vertex quadrics 相加，就得到新 vertex 应尽量满足的全部旧 planes。这使误差可以随 collapse 局部传播，而不必每次回到完整原 mesh。

## 4. 一步一步建立正式模型

对 vertex \(v_i\)，初始 quadric 为相邻 faces 的 plane quadrics 之和：

\[
Q_i
=
\sum_{f\ni i}
w_f p_fp_f^{\mathsf T}.
\]

\(w_f\) 可取 face area，使大面贡献更稳定；边界和 feature planes 还可额外加权，阻止 collapse 穿过重要线。

对候选 edge \((i,j)\)，合并后的 quadric 为

\[
Q_{ij}=Q_i+Q_j.
\]

若候选位置为 \(\bar q=[q_x,q_y,q_z,1]^{\mathsf T}\)，collapse cost 是

\[
E_{ij}(q)
=
\bar q^{\mathsf T}Q_{ij}\bar q.
\]

把 \(Q_{ij}\) 分块：

\[
Q_{ij}
=
\begin{bmatrix}
A&b\\
b^{\mathsf T}&c
\end{bmatrix},
\]

其中 \(A\in\mathbb R^{3\times3}\)。误差对 \(q\) 的梯度为

\[
\nabla E(q)=2Aq+2b.
\]

若 \(A\) 可逆，最优位置满足

\[
Aq^\ast=-b.
\]

若 \(A\) 奇异，说明 planes 没有在所有方向约束位置，例如完全平面区域允许沿平面自由移动。此时可比较两个端点、中点或在 edge segment 上求受限最优位置，选择 cost 最小者。

所有 edges 的当前 cost 放入 min-priority queue。每次取最小候选，仍要执行上一章的 link condition、face flip、feature 与 boundary 检查；接受后合并 quadrics，局部重算相邻 edge costs。由于旧 queue entries 可能失效，工程上常用 version 或 lazy deletion 验证候选仍然当前。

## 5. 跟着一个完整例子走到底

考虑一个 corner 邻域，新 vertex 希望接近三个 planes：

\[
x=0,\qquad z=0,\qquad y=1.
\]

对应齐次 plane vectors 为

\[
p_x=(1,0,0,0)^{\mathsf T},
\]

\[
p_z=(0,0,1,0)^{\mathsf T},
\]

\[
p_y=(0,1,0,-1)^{\mathsf T}.
\]

把三个 plane quadrics 相加。对任意 \(q=(x,y,z)\)，二次误差直接化为

\[
E(q)
=
x^2+z^2+(y-1)^2.
\]

因此

\[
A=I,
\qquad
b=
\begin{bmatrix}
0\\-1\\0
\end{bmatrix}.
\]

解

\[
Aq^\ast=-b
\]

得到

\[
q^\ast=(0,1,0),
\qquad
E(q^\ast)=0.
\]

假设 edge 两端是

\[
u=(0,0,0),
\qquad
v=(0.2,1.6,0.1).
\]

简单 midpoint 为

\[
m=(0.1,0.8,0.05).
\]

它的 quadric error 是

\[
E(m)
=
0.1^2+0.05^2+(0.8-1)^2
=0.0525.
\]

QEM 候选 \(q^\ast\) 同时满足三个局部 planes，误差为零，因此在几何代理上优于 midpoint。程序随后检查 \(q^\ast\) 是否导致相邻 triangles 翻转、是否越过 feature segment，以及 collapse 是否满足 link condition。只有这些检查通过，它才进入 mesh。

接受后，新 vertex quadric 保存

\[
Q_{\rm new}=Q_u+Q_v.
\]

它会在后续 collapse 中继续携带已经删除 faces 的 plane 信息。这就是误差“记忆”随简化传播的方式。

## 6. 回到真实系统：程序实际上怎样工作

QEM simplifier 的数据流可以写成：

~~~text
validated manifold mesh
→ compute face planes
→ accumulate vertex quadrics
→ evaluate every legal edge collapse
→ push costs into min-priority queue
→ pop cheapest valid candidate
→ topology / flip / feature / error checks
→ collapse and merge quadrics
→ update local candidates
→ stop at target count or error threshold
~~~

停止条件不应只有 triangle count。若 queue 最小 cost 已超过物理允许误差，即使尚未达到目标数量，也应停止；反过来，单纯达到 10% triangles 不说明关键孔径保持。

feature edge 可通过额外 constraint planes 或禁止跨 feature collapse 保护。边界 vertex 也可添加垂直于相邻 face、穿过 boundary edge 的约束 plane，防止边界向内漂移。

属性简化需要额外处理。UV seam、material boundary、CT label 与 normal discontinuity 可能要求复制 vertices 或在 cost 中加入 attribute quadric。只优化 position error 后再平均所有属性，会破坏语义边界。

最终仍要对简化结果做真实几何评估：sample-to-reference distance、Hausdorff approximation、normal deviation、volume、feature curve displacement 和任务尺寸。QEM cost 是高效代理，不是最终质量证明。

## 7. 容易走错的岔路

QEM cost 很小不代表 topology 一定安全。二次型只衡量 plane deviation，不会阻止非流形 collapse、孔洞闭合或 triangle flip；拓扑检查仍是独立条件。

平面区域的 \(A\) 奇异不是数值 bug，而是位置沿平面欠约束。强行加大 diagonal 虽能求逆，却会引入 arbitrary bias；比较端点、中点或受限 edge optimum 更可解释。

只对初始 faces 建 quadrics却在 collapse 后重新丢弃历史，会低估已删除区域的累计误差。新 vertex 应继承两端 quadrics 之和。

按固定 triangle ratio 比较不同模型也不公平。复杂曲率和孔洞结构需要更多 triangles；应同时报告误差阈值与任务指标。

最后，QEM 偏好接近 planes，不直接保证 silhouette。对查看器而言，轮廓边缘的一个小三维误差可能造成明显屏幕偏差，需要对 boundary/silhouette 加权或使用 view-dependent metric。

## 8. 本章落点、验证与下一章

本章把 mesh simplification 变成了可排序的局部优化。每个 face plane 产生 \(pp^{\mathsf T}\)，vertex quadric 累积邻域平方距离，edge collapse 通过 \(Q_i+Q_j\) 计算误差并求最优位置；priority queue 让系统总是优先执行当前最便宜的合法 collapse。

在 CT mesh 管线中，本章对应离线 LOD 生成、误差阈值和 feature-preserving collapse。在 STL/DirectX 项目中，它可为后续 meshlet hierarchy 生成多个几何层级，但每级仍需实际距离与体积验证。

本章的 90 分钟验证是实现 plane quadric 和 edge candidate cost，先复现正文中 midpoint cost 0.0525 与最优点 cost 0，再对一个平面加圆柱孔的网格简化到 50%、20% 和 10%。预期是平面区域快速减少，孔口曲率区域保留更多 vertices；关闭 feature constraints 后，孔口边缘会更早漂移。

下一章会进入 surface parameterization：当几何简化后需要纹理、材料图或二维编辑时，怎样把曲面切开并映射到平面，角度、面积与 seam distortion 为什么不可能同时为零，将成为下一问题。

