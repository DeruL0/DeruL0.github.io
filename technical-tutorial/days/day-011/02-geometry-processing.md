# 几何处理第 11 章：拓扑修复为什么不能只靠补洞按钮

## 1. 从一个真实任务开始

上一章把扫描 mesh 和 CAD mesh 配准到同一坐标系。配准之后，下一步常常是测体积、做布尔、生成 STL、进入 DirectX viewer 或送去仿真。此时你会发现一个看似很小的问题会让整条管线失败：mesh 不是一个干净的二维流形。

真实任务是修复一个从 CT segmentation 提取的零件表面。它看起来闭合，但 Boolean 报错；切片软件说有 non-manifold edge；体积计算结果不稳定；某些法线朝内，某些孔洞被错误补上。你不能只按一个“repair all”按钮，因为有些洞是真实孔隙，有些洞是分割断裂；有些 handle 是真实结构，有些 handle 是噪声桥接。拓扑修复首先要诊断结构，再决定是否修改。

今天的目标是建立 mesh topology 的最小工作模型：边的邻接、顶点的一环、边界环、Euler characteristic、genus 和连通分量。几何坐标告诉你形状在哪里，拓扑量告诉你它连接成了什么。

## 2. 最直接的办法，以及它为什么不够

最直接的方法是把所有边界环都补上。对 3D 打印外壳，这可能有效；对多孔材料，它会把真实孔隙封死。另一个直接方法是删除所有小连通分量。它能清理噪声，也可能删掉真实颗粒或薄片。

还有一种方法是只看渲染结果。GPU rasterizer 能画出很多坏 mesh，因为它只需要三角形列表。体积、布尔、法线传播和有限元却需要一致拓扑。一个 bow-tie vertex 在屏幕上几乎看不出来，但它的邻域不是一张盘，后续算法无法定义 inside/outside。

因此修复不是美化三角形，而是把 mesh 的组合结构改回算法假设的类型。CGAL Polygon Mesh Processing 文档把 polygon mesh repair 明确分成 orientation、hole filling、degeneracy removal、boundary stitching 等工具；这些工具对应不同拓扑和几何缺陷，不能混成一个黑箱动作。[CGAL Polygon Mesh Processing](https://doc.cgal.org/latest/Polygon_mesh_processing/index.html)

## 3. 关键想法是怎样被引出来的

一个封闭三角形网格若是二维流形，每条 edge 应恰好有两个 incident faces；若有一条 incident face，它是边界；若超过两个，它是 non-manifold edge。对每个 vertex，周围 faces 应形成一个连续 fan；若形成两个或更多分离 fan，就是 non-manifold vertex。

这类规则只依赖 connectivity，不依赖坐标。它们是拓扑诊断的第一层。第二层是全局不变量。对一个三角网格，Euler characteristic 是

\[
\chi=V-E+F.
\]

对于一个 connected、orientable、带 \(b\) 个边界环的 surface，

\[
\chi=2-2g-b,
\]

其中 \(g\) 是 genus，也就是 handle 数。这个公式把局部计数变成全局拓扑判断：补一个洞、打通一个 handle、分裂一个连通分量，都会改变 \(\chi\)、\(b\) 或 \(g\)。

关键想法是：修复操作必须说明它打算改变哪个拓扑量。补洞会减少 \(b\)；切断噪声桥可能减少 \(g\)；拆 non-manifold vertex 可能增加连通分量。没有这个诊断，所谓修复只是任意改几何。

## 4. 一步一步建立正式模型

先从边邻接开始。对每条无向边 \(e=(i,j)\)，统计 incident face 数：

\[
d(e)=|\{f\mid e\subset f\}|.
\]

若

\[
d(e)=2,
\]

它在封闭流形内部；若

\[
d(e)=1,
\]

它属于 boundary；若

\[
d(e)>2,
\]

它是 non-manifold edge。所有 \(d(e)=1\) 的边按端点连接起来形成 boundary loops。

再看整体计数。设 mesh 有 \(V\) 个顶点、\(E\) 条无向边、\(F\) 个 faces。Euler characteristic 为

\[
\chi=V-E+F.
\]

若 mesh 有 \(C\) 个 connected components，每个 component 都 orientable，则对第 \(c\) 个 component 有

\[
\chi_c=2-2g_c-b_c.
\]

所以

\[
g_c=\frac{2-b_c-\chi_c}{2}.
\]

这个公式只在 orientable manifold component 上使用。若 mesh 有 non-manifold edges、self-intersections 或 orientation 冲突，先不能相信 genus 结果。正确顺序是：局部流形检查，orientation 检查，自交检查，边界环统计，最后才解释 Euler characteristic。

修复操作也可以形式化。Hole filling 选择一个 boundary loop \(L\)，添加 faces 使其变成内部边界。若该 loop 是一个 disk-like hole，则补洞会让

\[
b\leftarrow b-1,
\qquad
\chi\leftarrow \chi+1.
\]

若补的是物体真实通孔的一端，拓扑解释就错了。数学操作相同，工程含义完全不同。

## 5. 跟着一个完整例子走到底

先看一个立方体壳。它有

\[
V=8,\qquad E=12,\qquad F=6.
\]

因此

\[
\chi=8-12+6=2.
\]

它是 connected、closed、orientable，并且

\[
b=0.
\]

由

\[
g=\frac{2-b-\chi}{2}
\]

得到

\[
g=\frac{2-0-2}{2}=0.
\]

这就是球面拓扑。

现在删除顶部一个 face。顶点和边仍为

\[
V=8,\qquad E=12,
\]

faces 变成

\[
F=5.
\]

所以

\[
\chi=8-12+5=1.
\]

顶部形成一个 boundary loop，因此

\[
b=1.
\]

代入公式：

\[
g=\frac{2-1-1}{2}=0.
\]

拓扑诊断告诉我们：这不是 handle，而是一个 disk-like boundary hole。如果任务是重建封闭立方体，补这个洞合理；如果任务是保留一个真实开口，补洞就是错误。

再看 non-manifold edge。假设某条边被三个三角形共享，则

\[
d(e)=3.
\]

此时它不是普通边界，也不是普通内部边。不能用 Euler characteristic 直接解释 genus，因为局部邻域已经不是二维 manifold。修复通常要复制该 edge 或相关 vertices，把三个 fan 拆成多个 manifold patches，再根据任务决定是否焊接或保留分离。

## 6. 回到真实系统：程序实际上怎样工作

工程 mesh repair pipeline 应先构造 edge-face adjacency 和 vertex one-ring。第一步报告 connected components、boundary edge count、boundary loops、non-manifold edges、non-manifold vertices、degenerate faces、duplicate vertices、orientation conflicts 和 self-intersections。第二步才选择操作：删除退化 face、合并重复点、统一 orientation、拆 non-manifold fan、填小洞、保留语义孔洞、重新三角化 patch。

对 CT segmentation，洞的语义尤其重要。材料内部孔隙是真实结构，外表面分割断裂是缺陷，扫描噪声造成的小孔可能要修。不能只按 hole area 阈值处理；应结合体素标签、局部厚度、连通到外部的路径和项目目标。

在 STL/DirectX viewer 中，topology diagnostic 应作为调试视图显示。边界边用一种颜色，non-manifold edge 用另一种颜色，self-intersection 用标记。这样用户能看到“修复”改动了哪里，而不是只拿到一个不可解释的新 mesh。

对 CTGS，Gaussian surface extraction 后的 mesh 可能拓扑不稳定。densification 与 opacity threshold 会改变连通分量和孔洞。导出前应扫描多个 threshold 下的 \(\chi\)、component count 和 boundary loops，避免单一阈值产生偶然 topology。

## 7. 容易走错的岔路

第一个误区是把 watertight 等同于正确。把所有洞补上会得到 watertight mesh，但也可能封掉真实通道或孔隙。

第二个误区是只看 Euler characteristic。若 mesh 非流形或自交，\(\chi\) 仍能算出一个数，但它不再对应可靠的 genus。

第三个误区是把小组件全部删掉。小组件可能是噪声，也可能是真实碎屑、颗粒或薄壁。删除规则要和物理尺度和任务目标绑定。

第四个误区是先平滑再修拓扑。强平滑可能把相邻边界粘在一起，制造假 handle；也可能把细孔封掉，让后续诊断失真。

最后，自动修复后不复查。每一次 topology operation 都应重新统计 adjacency 和 Euler quantities，因为修复一个 non-manifold edge 可能引入新的 boundary loop。

## 8. 本章落点、验证与下一章

本章把 mesh 修复从“补洞按钮”改成 topology-aware pipeline。边 incident face 数检测局部流形性；boundary loops 解释开口；Euler characteristic 和 genus 描述全局结构；修复操作必须说明它改变了哪个拓扑量。

在几何、STL 和 CTGS 项目中，本章对应 mesh diagnostic、boundary/non-manifold 可视化、hole filling policy、真实孔隙保护、以及导出前 topology regression check。

本章的 60 到 90 分钟验证是构造一个立方体 mesh，分别计算完整立方体和删除顶面后的 \(V,E,F,\chi,b,g\)。预期完整立方体为 \(\chi=2,b=0,g=0\)，删顶面后为 \(\chi=1,b=1,g=0\)。再人为让一条边连接第三个三角形，检查 \(d(e)=3\) 并禁止直接解释 genus。

下一章将进入谱几何与多尺度。拓扑告诉我们表面连通方式是否正确，但还不能描述形状上的低频与高频结构；谱方法会把 Laplacian 的特征函数作为几何上的频率基底，用于平滑、分割和多尺度分析。
