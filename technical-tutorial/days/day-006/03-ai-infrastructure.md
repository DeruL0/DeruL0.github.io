# AI Infrastructure 第 6 章：编译器为什么能融合算子，却不能随意编译所有 Python

## 1. 从一个真实任务开始

上一章用 mixed precision 减少了 tensor bytes，但一次 CTGS step 仍可能由数百个小 operations 组成。Python eager mode 每遇到一个 tensor operation 就进行 dispatch、shape/dtype 检查并提交 kernel；多个逐元素 operations 之间还会把中间 tensor 写回 global memory。

今天的真实任务是让训练框架一次看到较大的计算区域，把逐操作程序转换为计算图，跨节点做 fusion、buffer reuse 和 code generation。理想结果是更少的 kernel launches、更少的中间读写和更大的优化窗口。

困难在于训练程序不是固定数学表达式。Gaussian count 会 densify/prune，tensor shapes 会变化，Python 可能根据数据执行不同分支，自定义 CUDA op 还可能对编译器不透明。编译系统必须说明：这份生成代码在什么条件下仍然等价？条件不成立时是重新编译、回到 eager，还是在 graph boundary 分段执行？

## 2. 最直接的办法，以及它为什么不够

最直接的想法是记录第一次运行的所有 operations，以后无条件重放。若输入 shape、dtype、stride 和控制流永远不变，这可以工作。

但第一次训练时 Gaussian 数量为 500,000，densification 后变成 620,000。按旧 shape 生成的 indexing 和 launch dimensions 可能越界或漏算。若第一次 Python 分支走了 A，下一次数据要求 B，无条件重放 A 会产生错误结果。

另一种极端是为每个可能条件生成完全动态代码。它牺牲了许多静态优化，并且 data-dependent loop trip count、Python side effect 和动态内存结构未必能被目标后端表达。

所以编译器需要在 specialization 与 generality 之间选择。它捕获一段可表示的图，同时生成 guards，声明这段图只对哪些输入与程序状态有效。遇到无法捕获的语义时产生 graph break，让该段回到 Python/eager。

## 3. 关键想法是怎样被引出来的

计算图把 operations 作为 nodes，把 tensor dependencies 作为 edges。只要一组 nodes 没有不可观察的中间副作用，编译器可以改变执行边界而保持最终结果。

例如

\[
t_1=x+\alpha,
\qquad
t_2=\max(t_1,0),
\qquad
y=\beta t_2
\]

在 eager 中可能是三个 kernels；在图中可合成一个 loop，每个 element 从读取 \(x_i\) 到写出 \(y_i\) 都在 register 中完成。

guards 则是编译正确性的契约。编译器可能要求：

\[
\operatorname{dtype}(x)=\mathrm{FP16},
\]

\[
\operatorname{stride}(x)\text{ contiguous},
\]

\[
\operatorname{shape}_1(x)=128.
\]

再次调用时 guards 全部满足，便复用 compiled artifact；任一失败则需要选择已有其他 specialization、重新编译或 fallback。dynamic shapes 把某些尺寸从具体常数改成 symbolic variables，但符号仍可能受范围、stride 和 branch 条件约束。

## 4. 一步一步建立正式模型

把捕获程序表示为图

\[
\mathcal G=(V,E),
\]

其中 node \(v\in V\) 是 operation，edge 表示数据依赖。若两个相邻 pointwise nodes 具有相同 iteration domain 且中间结果没有外部使用，compiler 可执行 fusion：

\[
v_1\rightarrow v_2\rightarrow v_3
\quad\Longrightarrow\quad
v_{\rm fused}.
\]

对 \(N\) 个 FP32 elements，三次独立 pointwise kernels 至少产生约

\[
3\times(4N+4N)=24N
\]

bytes 的 global traffic。融合后只读取输入、写最终输出：

\[
Q_{\rm fused}\approx8N.
\]

这与前面 Roofline 章节一致：compiler 的价值不是减少数学 FLOPs，而是改变 kernel boundaries 和 memory lifetime。

编译 artifact 可抽象为

\[
(C,G),
\]

其中 \(C\) 是生成代码，\(G\) 是 guard predicate。输入状态 \(s\) 到来时：

\[
G(s)=\text{true}
\quad\Rightarrow\quad
\text{execute }C,
\]

\[
G(s)=\text{false}
\quad\Rightarrow\quad
\text{lookup/recompile/fallback}.
\]

若尺寸 \(N\) 被 specialized 为 500,000，新的 620,000 会 guard failure。若 \(N\) 被表示为 symbolic dimension，生成 kernel 可在 runtime 读取 \(N\)，但某些优化仍可能加入条件，例如 \(N\) 是 16 的倍数或不超过某上限。

graph break 把程序切成

\[
\mathcal G_1
\rightarrow
\text{eager region}
\rightarrow
\mathcal G_2.
\]

每个边界会失去跨区 fusion，并可能引入同步。尤其是从 GPU tensor 取 scalar 回 Python 做 branch，CPU 必须等 GPU 结果，成本不仅是少一次 fusion，还可能是 device synchronization。

training compiler 还要处理 backward。常见路线是先捕获 forward 与其 backward graph，再联合决定 saved tensors、recomputation 和 fusion。若一个 custom op 只注册 forward 而没有清楚的 backward、shape、alias 和 mutation metadata，编译器只能把它视为 opaque boundary。

## 5. 跟着一个完整例子走到底

考虑大 tensor：

\[
y=\operatorname{ReLU}(x+1)\times0.5,
\]

其中 \(N=2^{26}\) 个 FP32 elements，约 67 million。

eager 三 kernel 版本的最小 traffic 为

\[
24N
\approx1.61\ \mathrm{GB}.
\]

融合版本为

\[
8N
\approx0.54\ \mathrm{GB}.
\]

在 900 GB/s 的有效带宽下，只按 DRAM lower bound 估计：

\[
T_{\rm eager}
\gtrsim
\frac{1.61}{900}
\approx1.79\ \mathrm{ms},
\]

\[
T_{\rm fused}
\gtrsim
\frac{0.54}{900}
\approx0.60\ \mathrm{ms}.
\]

实际还包括 launch、cache 和 codegen 差异，但约三倍 traffic reduction 给出了优化上限来源。

现在把它放入动态 Gaussian pipeline。第一次调用 positions shape 是

\[
[500000,3].
\]

若 compiler 生成 exact-shape guard，第二次 densification 后 shape

\[
[620000,3]
\]

会触发 recompilation。若每 100 steps 都改变 count，compile latency 可能反复进入训练时间线。将第 0 维建模为 dynamic 可复用同一 artifact，但 renderer 中按 count 分配 temporary buffers、自定义 sorting op 或 data-dependent output size 仍可能造成新 guards 或 graph break。

再加入 Python 控制流：

~~~python
if opacity.sum().item() > threshold:
    prune()
~~~

item 操作需要把 GPU scalar 带回 Python，branch 结果依赖运行数据。编译器无法仅凭第一次路径证明以后相同，因而可能在此 break，并产生同步。把 pruning 改为固定 schedule，在编译区域外集中执行结构编辑，可让普通 optimization steps 保持稳定大图。

## 6. 回到真实系统：程序实际上怎样工作

实际优化顺序应先测 compile coverage，而不是只调用 compile API：

~~~text
profile eager baseline
→ enable compiler and include warmup
→ inspect graph breaks and guard failures
→ identify recompilation causes
→ make stable dimensions symbolic where profitable
→ move rare structural edits outside hot compiled region
→ register custom op metadata/decompositions
→ remeasure steady-state and end-to-end time
~~~

compile time 必须单独报告。短实验可能全部时间都在 tracing/codegen，长期训练才有机会 amortize。保存 artifact cache 也需要包含 device、dtype、shape assumptions 和 code version，不能把旧 binary 无条件复用。

对 CTGS，densification/pruning 本来就是低频结构事件。可以让若干百个普通 steps 使用稳定 compiled graph，在编辑点离开图、重建参数和 optimizer states，再进入新的 specialization。这样不会强迫 compiler 在一个 artifact 中支持任意结构变化。

自定义 projector/rasterizer 应注册输出 shape、dtype、device、alias/mutation 和 backward contract。若 compiler不能看进 kernel，至少能安全安排它前后的 pointwise fusion；若提供 decomposition，可能获得更大优化空间，但也要避免生成比专用 CUDA kernel 更慢的通用实现。

当前 PyTorch 官方编译模型把 graph breaks、guards、recompilations 与 dynamic shapes明确列为理解 compile 行为的核心概念；具体版本接口应以官方文档为准。[PyTorch torch.compile Programming Model](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/compile/programming_model.html)

## 7. 容易走错的岔路

首次 compiled run 比 eager 慢并不说明编译无效，因为其中包含 tracing 与 code generation；反过来，只报告预热后单个 microbenchmark 也会隐藏现实中的 recompilation。

把所有 dimensions 标记 dynamic 不一定更好。更通用的 kernel 可能失去 specialization、vectorization 或 memory planning，runtime guards 也不会全部消失。

graph break 数量少不等于代价小。一个位于大 fusion chain 中央、并触发 GPU-to-CPU scalar readback 的 break，可能比多个无同步的小边界更严重。

看到 kernel 数下降也不能直接推断更快。过度 fusion 会增加 register pressure、降低 occupancy 或重复计算；仍需回到 profiler 和 Roofline。

最后，compiler 与 CUDA Graphs 不是同一件事。compiler 改变 operations 与 kernels；CUDA Graphs主要重放一组已经确定的 GPU launches。前者可以为后者创造稳定区域，但两者解决的成本层次不同。

## 8. 本章落点、验证与下一章

本章把 tensor compiler 解释为带 guards 的图转换系统。graph capture 提供跨 operation 优化窗口，fusion 减少 kernel launches 和中间 traffic；guards 保证 specialization 仍然正确，dynamic shapes 扩大复用范围，graph breaks 则保留无法捕获的 Python 语义。性能取决于 steady-state reuse，而不只取决于一次 codegen。

在 CTGS 中，本章对应普通 optimization steps 的稳定编译区、densification 边界和 custom renderer contract；在训练平台中，应记录 compile time、graph count、break reasons、guard failures、recompilation count 与 steady-state step time。

本章的 90 分钟验证是对三操作 pointwise chain 比较 eager 与 compiled profiler，确认 kernel 和 DRAM traffic 是否减少；再依次输入不同 Gaussian-count shapes，并加入一次 data-dependent item branch。预期是静态 shape 触发 specialization/recompile，dynamic shape 减少部分 recompilation，而 item branch 造成 graph break 或同步。删除该 branch 后稳态时间应改善。

下一章会在稳定 compiled region 上继续降低 CPU launch 成本：CUDA Graphs 怎样捕获并重放整组 kernels，为什么它要求稳定 memory addresses 和控制流，以及动态 shape、allocator 与 NCCL collective 如何决定能否安全 capture。

