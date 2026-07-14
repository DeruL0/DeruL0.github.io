# 几何处理第 9 章：mesh Boolean 为什么首先是判定问题，而不是切三角形问题

## 1. 从一个真实任务开始

上一章用 Poisson reconstruction 从有向点样本得到 watertight mesh。接下来真实工作往往是布尔运算：把 CT 分割出的孔隙从实体中扣掉，把 STL 修复后的零件与夹具求交，或者把一个测量区域裁剪到工程坐标盒内。输入是两个 triangle meshes，输出应是新的 triangle mesh，并且拓扑、边界和内外分类都一致。

如果只看界面，Boolean 像是“把三角形相交处切开，然后保留里面或外面的面”。但真正困难在于每一个局部决定都依赖几何判定：点在三角形哪侧，线段是否相交，交点落在边内还是边外，两个面是否共面，某个小面片在另一个封闭体内部还是外部。任何一个判定错一次，后续拓扑就会接错，结果出现裂缝、翻转面、非流形边或丢失薄片。

今天的任务是从 Boolean 的失败推导出 robust predicates。谓词不是为了数学洁癖，而是为了让 mesh surgery 的离散拓扑选择建立在可靠的符号判定上。

## 2. 最直接的办法，以及它为什么不够

最直接的方法是用 double 计算三角形-三角形交点，并给所有比较加一个 epsilon。例如若点到平面的有符号距离小于 \(10^{-8}\)，就认为在平面上；若线段参数 \(t\) 位于 \([-10^{-8},1+10^{-8}]\)，就认为交在线段内。

这种做法在普通尺度下看起来有效，但 epsilon 没有统一物理意义。模型单位是米、毫米或纳米时，同一个 \(10^{-8}\) 代表完全不同的容差。更糟的是，同一个点可能在一个测试中被认为在左边，在另一个相邻测试中被认为共面。Boolean 需要全局一致的拓扑，而不是每个局部函数各自“差不多”。

另一个直接方法是先把所有坐标 snap 到网格。它能减少近似共面情况，却会移动输入几何。薄壁、小孔和 CT 产生的细小缺陷可能正好被 snap 消掉。若输出用于尺寸测量，几何变形本身就是错误。

因此 Boolean 的核心失败不是“交点算得不够准”单独造成的，而是符号判定和拓扑构造不一致造成的。需要把判断左/右/共面、内/外/边界的 predicates 从坐标构造中分离出来，并尽量保证符号正确。

## 3. 关键想法是怎样被引出来的

很多几何关系最终只需要一个符号。二维中，点 \(c\) 在有向边 \(ab\) 的左侧、右侧还是共线，由一个 determinant 的符号决定。三维中，点 \(d\) 在有向三角形 \(abc\) 的哪一侧，也由 determinant 的符号决定。只要符号可靠，拓扑选择就可以一致；交点坐标稍有误差，反而常可通过后续投影或重构处理。

这就是 predicate 的意义。它回答离散问题，例如 `orient3d(a,b,c,d)` 的结果是正、负还是零；construction 才负责生成交点坐标、切分边和写入新顶点。强行用一个浮点 epsilon 同时处理 predicate 和 construction，会把“应该选择哪种拓扑”与“交点坐标怎样表示”混在一起。

Shewchuk 的 robust predicates 工作正是围绕 orientation 和 incircle 这类 determinant 符号展开：先用普通浮点快速估计，若结果离零足够远就返回；若接近零，再使用自适应精度保证符号正确。它比全程任意精度更快，也比普通 double 更可靠。[Fast Robust Predicates for Computational Geometry](https://www.cs.cmu.edu/~quake/robust.html)

## 4. 一步一步建立正式模型

先建立二维方向判定。三点 \(a,b,c\) 的有向面积可写为

\[
\operatorname{orient2d}(a,b,c)
=
\det
\begin{bmatrix}
b_x-a_x & b_y-a_y\\
c_x-a_x & c_y-a_y
\end{bmatrix}.
\]

如果结果为正，\(c\) 在有向边 \(a\to b\) 左侧；为负则在右侧；为零则共线。这个 determinant 不只是面积公式，它给出了拓扑分支的符号。

三维中，四点的有向体积为

\[
\operatorname{orient3d}(a,b,c,d)
=
\det
\begin{bmatrix}
a_x-d_x & a_y-d_y & a_z-d_z\\
b_x-d_x & b_y-d_y & b_z-d_z\\
c_x-d_x & c_y-d_y & c_z-d_z
\end{bmatrix}.
\]

它判断 \(d\) 位于有向平面 \(abc\) 的哪一侧。三角形相交测试会反复使用这种 side-of-plane 符号：若一条线段的两个端点在三角形平面两侧，则线段可能穿过该平面；再用 barycentric 或 edge orientation 判断穿过点是否落在三角形内部。

Boolean 可抽象为三个阶段。第一阶段构造 arrangement：把两个 mesh 的三角形沿所有交线切开，使最终小面片之间只相交于共享顶点、共享边或完全不交。第二阶段分类：对每个小面片判断它在另一个实体的 inside、outside 还是 boundary。第三阶段选择：并集、交集和差集只是在分类标签上选择不同面片并决定法线方向。

用集合写就是：

\[
A\cup B:\quad
\text{keep faces whose cell is inside } A \text{ or inside } B,
\]

\[
A\cap B:\quad
\text{keep faces whose cell is inside } A \text{ and inside } B,
\]

\[
A\setminus B:\quad
\text{keep faces inside } A \text{ and outside } B,
\]

但这些集合公式只有在 arrangement 和 inside test 都一致时才成立。否则一个面片可能同时被相邻区域认为内外不一致，拓扑就会破裂。

## 5. 跟着一个完整例子走到底

用二维多边形交集看完整流程，因为它保留了 Boolean 的关键结构。设正方形

\[
A=[0,2]\times[0,2],
\]

另一个矩形

\[
B=[1,3]\times[-1,1].
\]

两者交集应为

\[
[1,2]\times[0,1].
\]

先找边交点。A 的底边从

\[
p_0=(0,0)
\quad\text{到}\quad
p_1=(2,0),
\]

B 的左边从

\[
q_0=(1,-1)
\quad\text{到}\quad
q_1=(1,1).
\]

两条线段相交。用参数形式

\[
p(t)=p_0+t(p_1-p_0),
\qquad
q(u)=q_0+u(q_1-q_0).
\]

解得

\[
t=\frac12,\qquad u=\frac12,
\]

交点为

\[
(1,0).
\]

接下来判断 A 的右边 \(x=2\) 与 B 的上边 \(y=1\) 的交点，得到

\[
(2,1).
\]

arrangement 阶段把 A 的边切成若干小段，例如底边被切成 \((0,0)\to(1,0)\) 和 \((1,0)\to(2,0)\)。分类阶段取每段中点。底边右半段中点 \((1.5,0)\) 位于 B 内部，也在 A 边界上，因此属于交集边界；底边左半段中点 \((0.5,0)\) 不在 B 内部，不属于交集边界。

再看一个 predicate。判断点 \((1,0)\) 是否在 A 底边上，可计算

\[
\operatorname{orient2d}\big((0,0),(2,0),(1,0)\big)=0.
\]

判断点 \((1,1)\) 相对同一有向边，则

\[
\operatorname{orient2d}\big((0,0),(2,0),(1,1)\big)=2>0.
\]

如果浮点误差让第一个零被判成正，算法会认为交点不在边上，底边不会被正确切开；如果相邻测试又把它当成共线，就会出现一个拓扑上没有对应边的孤立交点。这个小例子展示了 Boolean 为什么依赖 consistent predicates，而不仅是“算出交点”。

## 6. 回到真实系统：程序实际上怎样工作

工程 Boolean 一般会明确分层。宽相位先用 AABB tree 找可能相交的三角形对；窄相位用 robust predicates 判断真正相交关系；构造阶段把交线插入三角形并局部重三角化；分类阶段在 arrangement 的 cells 或面片上做 inside/outside；最后清理重复顶点、合并共面片、检测非流形边和退化三角形。

数据结构上，半边结构适合维护拓扑邻接；空间索引适合减少三角形对数量；exact predicate kernel 负责所有符号判定；construction kernel 可使用 double、rational 或 filtered exact constructions，取决于输出精度要求。许多成熟库，例如 CGAL 的 exact predicates/exact constructions 或 exact predicates/inexact constructions kernel，就是围绕这条边界设计的。

对 STL/DirectX 项目，Boolean 后的 mesh 不能只检查“有没有崩溃”。需要报告顶点数、三角形数、boundary edge count、nonmanifold edge count、self-intersection count、法线一致性和体积变化。渲染器能画出来不说明几何可用于制造或测量。

对 CT segmentation，输入表面常带 voxel stair-step、薄壁和小孔。若先过强平滑再 Boolean，可能把真实缺陷抹掉；若完全不修复，Boolean 可能被大量近共面小三角形拖慢。更实际的流程是先做尺度受控的 remeshing 和退化清理，再用 robust Boolean，并把被删除或修改的小特征记录下来。

## 7. 容易走错的岔路

第一个误区是认为 double 足够，因为坐标看起来不大。错误往往发生在 determinant 接近零时，与绝对坐标范围和局部几何退化都有关；薄三角形、共面接触和长短边混合都会让普通浮点符号不可靠。

第二个误区是把 epsilon 调大。它能减少一部分抖动，却会把真实的小间隙、小重叠和薄片吞掉。更重要的是，不同函数里的 epsilon 很难保持拓扑一致。

第三个误区是只修交点坐标，不修谓词。Boolean 的核心分支是符号决定的；交点坐标更精确但 side test 错了，面片仍会被分类到错误区域。

第四个误区是把 watertight 输入等同于 Boolean 输出 watertight。两个 watertight mesh 相交后，如果 arrangement 切分不一致，输出照样会有裂缝。拓扑正确性必须在输出上重新验证。

最后，完全 exact construction 也不是免费答案。它可能带来巨大表达式和性能成本。工业程序常采用 filtered predicates、局部 exact construction、最后投影或 snap rounding 的组合，但前提是每一步的误差边界被记录和验证。

## 8. 本章落点、验证与下一章

本章把 mesh Boolean 拆成 arrangement、classification 和 selection，并说明这些阶段首先依赖 robust predicates。orientation determinant 的符号决定拓扑分支；自适应精度谓词在接近退化时保证符号正确，避免 epsilon 造成的不一致。

在几何处理和 STL 项目中，本章对应 Boolean kernel、AABB broad phase、exact predicate policy、半边拓扑更新和输出几何验证。对 CTGS 提取出的 mesh，本章说明了为什么“看起来闭合”的表面还需要布尔级别的鲁棒性检查。

本章的 90 分钟验证是实现二维矩形交集的 arrangement：求出 \((1,0)\) 和 \((2,1)\) 两个交点，切分边，用点在多边形内测试选择交集边界。然后把其中一条边改成几乎共线，比较固定 epsilon 与精确 `orient2d` 的分类差异。预期是 epsilon 版本会在某些容差下丢边或重复边，精确谓词版本的拓扑选择稳定。

下一章将进入配准和形变。Boolean 假设两个 mesh 已在同一坐标系中；但 CT、扫描点云和 CAD 往往存在位姿差异或非刚性形变，下一步需要估计把一个几何对象对齐到另一个几何对象的变换。
