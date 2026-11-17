# FP16/FP32 训练的数值边界

状态：实验数学说明与 CPU 单元测试，2026-09-27 更新数值门槛与块交换边界。
范围：冻结普通 Linear 的局部精度域、FP32 LoRA、FP32 残差流。
本轮未加载底模、未运行训练任务或 CUDA kernel，不认证显存、吞吐、收敛或生成质量。

代码入口：[`islands.py`](../../library/training/adaptive_runtime/islands.py)、
[`training_precision.py`](../../library/training/adaptive_runtime/training_precision.py)。
测试：[`test_adaptive_precision_math.py`](../../tests/test_adaptive_precision_math.py)、
[`test_adaptive_precision_aliases.py`](../../tests/test_adaptive_precision_aliases.py)、
[`test_adaptive_precision_scale_feasibility.py`](../../tests/test_adaptive_precision_scale_feasibility.py)、
[`test_adaptive_precision_comparison_math.py`](../../tests/test_adaptive_precision_comparison_math.py)、
[`test_adaptive_precision_swap_cpu.py`](../../tests/test_adaptive_precision_swap_cpu.py)。
使用边界见[实验说明](../experimental/adaptive-runtime.md)，配置不漂移的证明见
[静态契约报告](../findings/adaptive_precision_oom_static_proof_20260924.md)。

## 1. 三种比较对象

必须区分以下对象，不能将 FP32 运行直接称为精确实数计算：

1. **理想实数图**：把已加载权重、输入的数值视为固定实数，不发生运算舍入。
2. **FP32 参考图**：同一来源权重提升到 FP32 后实际运行，仍存在舍入、backend 差异。
3. **混合精度图**：部分冻结 Linear 使用 FP16，输出提升 FP32；adapter 保持 FP32。

BF16 来源权重提升 FP32 只能无损保留其已有数值，不能恢复预训练保存前的信息。
以下一般误差界相对理想图推导；若两条有限精度路径对理想图的界为 `E_mix`、
`E_32`，由三角不等式只能推出两路径差异 `<= E_mix + E_32`。
不能直接假设包含离散舍入的 FP32 图处处可微或满足光滑 Lipschitz 条件。

CPU 测试优先使用二进制有理数，使指定的 FP32 加乘可精确表示；一般性的递推式
则用 Python `Fraction` 检查代数，避免让公式测试本身的浮点舍入掩盖错误。

## 2. FP32 边界保留什么

单个 island 的实际形态为：

```text
y_hat = cast32(linear16(cast16(x), W16, b16))
```

`cast32` 阻止后续残差流继续以 FP16 存储，但无法逆转输入、权重和 Linear 输出
已有的舍入、下溢或溢出。即便 GEMM 内部采用更高精度累加，也不能据此假定最终
FP16 输出、输入 cast 或反向 cast 不会出问题。本说明不假定 CPU 和 CUDA 的累加
实现相同，且不将 GPU TF32、reduced-precision reduction 当作已被这些测试覆盖。

可精确复现的例子：`W=1+2^-12, x=1`。权重转为 FP16 后为 `1`，返回的 FP32
结果仍为 `1`，不是 `1+2^-12`。另一个例子 `W=2, x=32768` 的理想输出为
`65536`，FP16 输出溢出，提升 FP32 后仍是 Inf。

**输出相同也不保证梯度相同**：令 `W=[1, 1+2^-12]`、`x=[1,0]`。
FP32 与混合路径输出都为 `1`，但 loss 等于输出时，输入梯度第二分量分别为
`1+2^-12` 和 `1`。因此输出误差、输入梯度误差和 LoRA 梯度必须分开检查。

## 3. 前向误差传播

设固定参数下 `x_(i+1)=F_i(x_i)`、`xhat_(i+1)=Fhat_i(xhat_i)`，选定一致的
向量范数及诱导矩阵范数。假设在两条轨迹经过的整个域内：

```text
||F_i(u)-F_i(v)|| <= L_i ||u-v||
||Fhat_i(u)-F_i(u)|| <= epsilon_i
```

第二条包含权重/输入 cast、kernel 误差及输出 cast；不是单个样本测得的误差。
把 `F_i(xhat_i)` 加入并减去，得到：

```text
E_i = ||xhat_i-x_i||
E_(i+1) <= L_i E_i + epsilon_i
E_n <= E_0 product(j=0..n-1, L_j)
       + sum(i=0..n-1, epsilon_i product(j=i+1..n-1, L_j))
```

空乘积为 1。残差块 `F_i(x)=x+R_i(x)` 可使用 `L_i <= 1+L_Ri`，但恒等分支
并不保证误差被衰减。对 attention、多分支和多个条件输入，需要把完整依赖纳入
状态，不能把 Linear 列表顺序直接当作串联图。若存在 overflow，有限的
`epsilon_i` 假设已失效，不能继续套用有限误差界。

测试以首层 `1+2^-12` 和后续 FP32 增益 `32` 构造了等号案例：局部绝对误差
`2^-12` 在最终输出变成 `2^-7`。这说明敏感性与下游放大有关，不能只按本层
输出相对误差排序。真实 DiT 的统一 `L_i`、`epsilon_i` 目前未建立。

## 4. 反向误差不是前向界的重复

PyTorch 对浮点 cast 传播梯度，不使用离散 rounding 函数几乎处处为零的导数。
因此不能先把 `round` 当数学函数求导，再声称模型无法训练。以下用理想块的
Jacobian `J_i(x)` 和数值反向误差 `r_i` 描述实际 autograd：

```text
g_i    = J_i(x_i)^T g_(i+1)
ghat_i = J_i(xhat_i)^T ghat_(i+1) + r_i
||r_i|| <= eta_i
```

`eta_i` 必须包含低精度权重、激活、cast、乘加及重算对反向的偏差，且覆盖实际
数值 cotangent `ghat_(i+1)` 的范围；不能把一次前向误差当作它的替代。
再假设 `||J_i(xhat_i)|| <= L_i` 且
`||J_i(u)-J_i(v)|| <= H_i ||u-v||`，令 `D_i=||ghat_i-g_i||`，有：

```text
D_i <= L_i D_(i+1) + H_i E_i ||g_(i+1)|| + eta_i
```

推导是将差分拆成 `J_i(xhat_i)^T (ghat-g)`、
`(J_i(xhat_i)-J_i(x_i))^T g` 与 `r_i` 三项。这样不会漏掉前向状态改变对
Jacobian 的影响。终点也必须给界：若 loss 梯度是 `L_loss`-Lipschitz，
则 `D_n <= L_loss E_n + eta_loss`；MSE 的梯度常数还取决于 mean 的规约元素数。

测试中的理想块为 `x^2` 与 `3x`，第一块输入限制在 `[-2,2]`，所以可取
`L_0=4, H_0=2`；第二块 `L_1=3, H_1=0`。使用显式扰动和精确分数逐项验证
前后向递推。这只验证条件论证，不是给真实 DiT 提供已测的 Jacobian 常数。

## 5. FP32 LoRA 仍会接收误差

固定一个样本，用列向量表示，关闭 dropout 和其它路由：

```text
y = W x + s B A x
g = d(loss)/dy
d(loss)/dB = s g (Ax)^T
d(loss)/dA = s B^T g x^T
d(loss)/dx = W^T g + s A^T B^T g
```

批量训练相应求和，`s` 包括 alpha/rank 和 multiplier。基础 `W` 冻结仅表示
不更新 W，并没有消除 `W^T g` 这条输入梯度路径。测试调用实际 `LoRAModule`，
使用精确可表示的 W/A/B/x/g 验证非零 B 和初始化 `B=0` 两种情形。

`B=0` 时 `grad A=0`，但 `grad B` 通常非零。这是合法零梯度，不能当作下溢或
“没有梯度”；指标需要同时保留参考梯度量级及绝对误差。

更直接的反例是在 FP32 LoRA 后放一个权重 `1+2^-12` 的冻结 Linear。下游改为
FP16 后其传回的 g 改变，即使 LoRA 参数、计算及梯度存储均为 FP32，`grad B`
仍与 FP32 对照相差 `2^-12`。因此不能只检查 adapter dtype 就宣称训练等价。

固定 A/B 的状态重放可隔离此误差；若比较独立更新后的两条轨迹，还要加入 A/B
自身差异，不能将全部偏差归因于某一个基础层。

## 6. Loss scaling 的有效范围

理想情况下，`grad(S loss)/S = grad(loss)`；有限精度下却是“先缩放、再量化、
再 unscale”，不能随意交换顺序。FP16 round-to-nearest-even、保留 subnormal
时，最小正数为 `2^-24`，`2^-25` 恰好舍入到零，最大有限值为 `65504`。
若理想中间梯度非零幅值落在 `[m,M]`，仅就该次 cast 的下溢/溢出考虑，保守要求：

```text
S*m > 2^-25
S*M <= 65504
```

这些条件不是全程安全证明：不同中间量、累加、误差反馈可能进一步限制 S。
若硬件 flush-to-zero，不能套用 subnormal 阈值；必须按实际 kernel 行为验证。
当前 CPU 测试不证明 CUDA 的 FTZ 行为。

三项测试界定其作用：

- 对 `2^-26` 的反向梯度，S=1 时变零，S=1024 后 unscale 可精确恢复；
  S=`2^44` 又在 FP16 边界溢出。提高 scale 不是无限改善。
- 若前向 activation 已将 `2^-26` 舍入到零，下游参数梯度在 S=1/1024 下都为零，
  scaling 无法重建消失的 activation。
- 前向已产生 Inf 时，缩放 loss 也无法恢复有限值。

所以诊断需分别标记前向范围失败、反向下溢/溢出、有限但不准确，不能全部交给
GradScaler。测试不进行优化器参数更新，不改变生产“跳步即停止”等策略。

### 6.1 统一 scale 可能没有可行值

上一节的 `65504` 是保守界。对 IEEE binary16、round-to-nearest-even、保留
subnormal 的单次转换，正数保持非零且有限的精确输入范围是：

```text
2^-25 < value < 65520
```

`65520` 是溢出的舍入边界，不是最大有限值；`65519` 仍会舍入到 `65504`，
`65520` 则舍入为 Inf。这些端点有 CPU cast 测试，并未认证 CUDA kernel 的行为。

在固定精度计划、固定状态下，假设各反向 cast 前的非零理想 cotangent 幅值范围
为 `[m_i,M_i]`，缩放只将其乘以同一个正数 S。仅就这些 cast 而言，必要可行区间为：

```text
I_i = (2^-25 / m_i, 65520 / M_i)
I = intersection(i, I_i)
```

参考值本来就为零的项不提供非零下限；不能将合法零梯度当作下溢。实际计算还受
缩放自身、GEMM、累加和其它 cast 约束；有限非零也不保证准确，因此 `I` 非空
不是全程安全证明。若 `I` 已空，在上述固定 cotangent 模型中继续改变统一 S
不能同时满足这些必要条件，应该回到精度候选，而非只反复降低 scale。

具体取两个独立反向边界：`m=2^-30`、`M=2^20`，必要条件为：

```text
S > 32
S < 65520 / 2^20 < 1
```

两者对所有正数 S 都无交集。此结论不依赖 `S>=1`；PyTorch GradScaler 的 scale
本身允许降到 1 以下，不能把本项目对初始 scale 的配置限制当作其数学性质。
CPU 夹具包含 `S=2^-5` 的诊断点，不意味着放开实际训练配置。

### 6.2 定点提升与静默下溢的 CPU 证据

夹具每个分支都为“FP32 冻结 identity Linear + FP32 LoRA，再接一个独立 identity
Linear”。用实际 `install_training_precision()` 安装精度域，实际 `LoRAModule`
取 `A=1,B=0,alpha/rank=1`，无 bias/dropout，输入均为 1。loss 是两路输出分别
乘以 `2^-30`、`2^20` 后求和；所有前向输出均为 1，loss 有限。它是受控线性
目标，不是 DiT 的 flow-matching loss，不能从该例推出真实层名单。

上游基础 Linear 也保持 FP32，否则它自身的低精度反向边界会引入另一个瓶颈。
两路下游均为 FP16 时，小梯度下溢或大梯度溢出；只提升大梯度分支的下游 Linear
到 FP32，保留小梯度分支的下游 FP16，在 `S=64/1024` 下，两路 `grad B` 和
输入梯度与全 FP32 对照逐位一致，`grad A=0`。直接执行和 non-reentrant CPU
checkpoint 路径均有对照。在固定 `S=1024` 时，只提升小梯度分支仍不能修复
大分支溢出。这里没有更新参数，也没有测量真实 GPU 性能或激活显存。

另取大分支梯度 `M=2^10`，则 `32<S<65520/1024<64` 非空，但没有 `2^k`
形式的值。只有在初始 scale 为 2 的整数次幂、且增长/回退乘以 2/0.5 时，
“可达 scale 没有可行值”的推断才适用，不能声称所有 GradScaler 配置都如此。
非整数次幂 scale 也不是充分修复，例如 S=48：

```text
48 * 2^-30 = (3/4) * 2^-24
cast16(48 * 2^-30) = 2^-24
unscale 的理想结果 / 原梯度 = (2^-24 / 48) / 2^-30 = 4/3
```

梯度虽然非零且有限，相对误差仍为 `1/3`；实际 CPU unscale 的 FP32 舍入用
显式容差核验，分数式本身用 `Fraction` 精确验证。在 S=2 时小梯度已变为零，
所有参数梯度仍有限。真实 CPU GradScaler 允许通过 step gate，且不降低 scale；
测试把 optimizer 的 step 替换为只记一次事件的无更新函数，再调用 scaler.update，
遵循其调用顺序并验证参数未改变，并未执行优化器更新。

对自动分配的意义是联合检查 scale 区间、可达 scale 和实际梯度误差，再选择需要
提升的候选边界。提升移除了对应 FP16 cast 的范围约束，不代表 FP32 没有范围限制，
也不保证其它层的 cotangent 不变；更换计划后仍需重新验证完整路径。该例提供
选择性 FP32 有效的具体见证，不将其推广成误差单调性或真实 DiT 的认证。

## 7. 为什么不能按 FP32 层数做单调搜索

两个平行 Linear 的权重取 `1+d` 和 `1-d`，`d=2^-12`，输入均为 1，输出在
FP32 相加。FP16 在 1 下方的间距为 `2^-11`，故 `1-d` 恰处于 tie，RN-even
选 1；`1+d` 也舍入到 1。因此：

```text
两支 FP32:             (1+d)+(1-d) = 2
两支 FP16 -> FP32:     1+1         = 2
只提升第一支:          (1+d)+1     = 2+d
只提升第二支:          1+(1-d)     = 2-d
```

所有结果都有限，但单独提升任何一支反而增加组合误差。此反例不声称真实 DiT
依赖这种抵消，只否定“更多 FP32 必然更准”的普遍命题。基于绝对误差和固定
Lipschitz 常数的保守上界可以单调下降，实际误差仍可能非单调，两者不矛盾。

同样，某一次局部 VJP 也不能证明最坏方向安全：矩阵误差 `diag(0,1000)` 在
cotangent `(1,0)` 上的 VJP 为零，而谱范数为 1000。随机方向能提供统计观测，
不能在没有概率假设和覆盖论证时将一次观测宣称为谱范数上界。

## 8. 对自动分配实现的约束

- 硬件决定可尝试 dtype，不决定敏感层名单；禁止用本说明推导三个模型的通用名单。
- 先固定权重、adapter、输入、timestep、mask 和 RNG，再比较候选组合。
- 局部 VJP/前向误差只用于生成候选；整网输入梯度、LoRA 梯度和绝对误差仍需独立验证。
- 搜索必须有预算，保存失败候选；不能按 FP32 层数作精度单调二分。
- scaling 不能替代 FP32 promotion；FP32 promotion 也不能替代 scaling 的范围诊断。
- 统一 scale 的必要区间冲突时重新评估精度候选；仅 finite 检查无法检测静默下溢。
- 理论界缺失、误差超阈值或真实模型验证未执行时，继续标记“未认证”，不自动写推荐配置。

以上约束指导后续混合精度实现，不增加 OOM 编排、性能承诺或训练任务授权。

### 8.1 当前比较指标的范围和数值域

局部校准的 `precision.py::_measure` 对每个输出/梯度张量单独计算，再取最大
relative-L2 和最小 cosine。独立训练探针的
[`compare_training.py`](../../bench/adaptive_runtime/compare_training.py) 则分别比较
prediction、input gradient、拼接意义下的全部 adapter gradient；最后一组先汇总
平方和、点积再计算指标。逐参数最坏误差列表只是报告项，没有加入该探针的全局门槛。
这些是现有局部/实验诊断，不等同于训练入口已经具备自动精度认证。

两条路径都先提升到 FP64 再计算范数，避免先在 FP32 中平方溢出。bench 的加载器
只接受非空、有限 FP32 张量，单个 case 文件至多 512MiB；按未压缩 FP32 数据计算，
元素总数不超过 `2^27`，元素绝对值小于 `2^128`。于是实数算术下平方和小于
`2^283`、差平方和小于 `2^285`，计算 cosine 分母用的两个平方和乘积小于 `2^566`，
远低于 FP64 约 `2^1024` 的上界。最小 FP32 subnormal 为 `2^-149`，其平方
`2^-298` 也在 FP64 正规数范围内。该边界只排除这条受限捕获路径的范围溢出，
不是任意 FP64 helper 输入或规约舍入误差的保证。

CPU 测试覆盖零、最小 subnormal、最小正规数、最大有限 FP32，以及最大正负数
互相比较；loader 对 NaN/Inf、FP64、空张量的拒绝在临时 safetensors 上验证。
比较还拒绝 key/shape 改变。零参考和零候选的 cosine=1 是项目约定，不能据此
声称数学上零向量存在方向；零参考而候选为 1 的对照会被拒绝。

### 8.2 固定分母下限为何会误拒绝相同小梯度（2026-09-30 已修复）

早期实现曾把 cosine 分母截断到固定下限 `1e-30`。对于非零向量，数学 cosine 应对共同的正数缩放不变，固定分母下限会破坏这一性质：

```text
x = y = [2^-60]
dot(x,y) = ||x|| ||y|| = 2^-120
数学 cosine = 1
带 1e-30 固定下限时的失真值 = 2^-120 / 1e-30，约 7.52e-7
relative-L2 = 0
```

这是固定下限造成的失真，而非 FP64 下溢，曾导致完全相同的小梯度捕获未通过 bench 比较，以及 `candidate="fp32"` 的局部自比较误标 `safe=false`（当时以 2 个 `strict xfail` 记录）。

现行实现（2026-09-30 起）：`vector_metrics` 统一在 FP64 中计算，不再对 cosine 分母施加固定 floor，明确区分真正零范数与非零小范数；当参考范数为零而候选非零时，`relative-L2` 返回 `null` 并附原因，由比较器 fail-closed，同时保留 `absolute-L2` 与 `reference-L2`（不能把任意分母下限当作经过任务验收的绝对容差）。原 2 个 `strict xfail` 已随修复移除。

### 8.3 全局梯度通过不保证 Adam 更新接近

取两个独立参数的参考梯度 `g=(1,m)`、候选 `ghat=(1,0)`，`m=2^-26`。
prediction 与 input gradient 完全相同，则理想的全局指标为：

```text
relative-L2 = m / sqrt(1+m^2) < 2e-8
cosine = 1 / sqrt(1+m^2) > 0.999
```

真实 `compare_case()` 返回全局门槛通过，同时在最坏参数列表中报告小参数
relative-L2=1、cosine=0、absolute-L2=`2^-26`。这是现有 aggregate 合同的局限，
不是该合同声称了逐参数通过；测试保留这一反例，不把返回 true 当作质量认证。

进一步考虑第一步、初始一二阶矩均为零、带 bias correction、无 weight decay 的
Adam。对非负标量梯度，小参数的更新幅度为：

```text
u(g) = learning_rate * g / (g + epsilon)
u(m) - u(0) = learning_rate * m / (m + epsilon)
epsilon = 1e-8 < m  =>  u(m)-u(0) > learning_rate/2
```

因此全局梯度相对误差小于 `2e-8`，仍可伴随某一坐标超过半个 learning rate 的
更新差异。测试用 `Fraction` 精确验证该不等式，不运行 optimizer.step；不将
第一步公式外推为任意已有 optimizer 状态的界，也不声称真实 DiT 已出现此反例。

后续精度验收需要同时考察全局、逐参数/分组绝对与相对误差，以及给定 optimizer
状态下的更新敏感性。不能机械地要求所有微小或合法零梯度满足统一相对门槛；
如何组合绝对容差、参考量级和更新误差，仍需任务相关验收。本轮只明确现有证据
不能推出的结论，没有新增生产门槛、推荐名单或 OOM 策略。

## 9. 共享参数的精度约束

当前 island 将普通 Linear 的参数本身转为目标 dtype，而不是给每次调用生成独立
权重副本。设 `P(m)` 是模块 m 使用的 Parameter 对象集合，计划给 m 的存储 dtype
为 `d_m`。若两个模块使用同一个 Parameter 对象 p，则：

```text
p in P(m) intersect P(n)
dtype(p) = d_m and dtype(p) = d_n  =>  d_m = d_n
```

因此，只要沿共享参数边相连，整个连通分量的存储 dtype 就必须一致。这是当前
原地转换表示的必要条件，不是数值安全的充分条件。若希望共享权重的不同调用使用
不同计算精度，需要另行设计显式 cast/副本、身份及生命周期；顺序 `.to()` 不能
同时满足两种存储 dtype，也不能悄悄解除原有共享关系。

当前边界分两层：

- `install_training_precision()` 使用 `named_parameters(remove_duplicate=False)`
  检查 Parameter 身份重复，在冻结权重、转换 dtype 和安装 wrapper 之前拒绝。
  即使所有共享者要求相同 dtype，当前实验训练合同也保守拒绝，并未承诺支持。
- `install_precision_islands()` 只检查模块对象是否重复。两个不同 Linear 共享同一
  weight 时，冲突 dtype 不会被前置拒绝；后一次转换会改变共享权重，令另一模块的
  `_adaptive_dtype` 与权重不符。只转换一个消费者也会影响未分配的共享消费者。
  这是底层安装器缺少检查，不是实际训练入口缺少同一保护。

新增 CPU 测试覆盖共享 weight/bias、冻结前后、全 FP16/混合/全 FP32 的训练拒绝，
并检查 dtype、数值、参数身份、`requires_grad`、forward 和 manifest 均未改变。
相同数值但对象与存储独立的参数可分别选择精度，小型精确夹具的前后向通过。
底层已有的非法 dtype、可训练权重、非有限值、溢出和模块别名校验，也检查不会
在第二单元失败前把第一单元部分安装。

底层 shared-Parameter 缺口用 3 项 `strict xfail` 单独记录：两种转换顺序和未分配
消费者。它们要求安装前拒绝并保持模型不变，当前因“没有抛出异常”而预期失败；
意外通过会让测试失败，意外的 RuntimeError/AssertionError 不由 xfail 隐藏。
本轮仅增加测试和论证，不修改底层实现，也不将这些 xfail 计作通过。

此论证针对同一个 Parameter 对象。不同 Parameter 对象共享底层 storage、参数与
buffer 共享等情况不在这次证明范围内；不能据此声称已有全面的存储别名检查。
没有证据表明 Anima、Krea-2 或 Z-Image 实际底模包含这里构造的参数共享情形。

### 9.1 块交换保持的是已选定的权重表示

令 `W_D = cast_D(W_source)` 为安装精度域后的冻结权重，CPU master 捕获为 C，
恢复为 R。若块交换不改变数值计划，需要满足：

```text
R_D(C(W_D)) = W_D
恢复 dtype = D = island._adaptive_dtype
Parameter 对象身份保持；底层 storage 地址允许改变
```

这里比较的是已经选定的 `W_D`，不是要求 FP16 安装恢复源 FP32 权重的精度。
同样，dtype manifest 只验证 dtype 身份，不包含权重值；测试另行逐元素比较 master
和恢复权重，不能把 hash 不变单独当作数值不变的证据。

当前普通非量化权重的
[`_capture_cpu_master`](../../library/runtime/block_swap_masters.py) 中，策略名
`transfer_dtype="bf16"` 表示原生表示路径，并没有调用 `.to(torch.bfloat16)`。
FP16 权重捕获为 FP16 master，FP32 权重捕获为 FP32 master；只有另外的 FP8/int8
分支进行有损表示转换。现有混合精度训练合同仍拒绝这两个有损选项。

这一区别有明确反例：`1+2^-10` 能被 FP16 精确表示、`1+2^-20` 能被 FP32 精确
表示，但两者转为 BF16 都舍入到 1。若把上述策略名误解为强制 BF16 压缩，再提升
回执行 dtype，数值恒等式即失败。CPU 测试用这两个值核验当前捕获/恢复仍逐位保真，
并检查 `stored_bytes == source_bytes`。

[`ModelOffloader`](../../library/runtime/offloading.py) 保存每个权重安装后的执行
dtype，恢复时写回 `Parameter.data`，不会替换普通 Parameter 对象。一个 block
内 master dtype 混合时，CPU packing 不建立单一 dtype slab；同 dtype block
可以打包。此处说的是 CPU master 布局，不是对 CUDA slab 恢复路径的认证。

新增 CPU 夹具使用实际训练精度安装器、真实 LoRAModule 和 ModelOffloader，包含
6 个双 Linear 残差块、swap=4、non-reentrant checkpoint、原生传输与 foreach 设置。
分别验证块内 FP16/FP32 混合，以及块间交错精度、交换源与目标 dtype 不同的布局。
每种布局运行三轮无参数更新的前后向，调用真实 CPU fallback 和 CPU worker future，
核对每轮 4 次前向/4 次反向交换；两布局合计 48 次恢复。测试结束会等待全部
future、关闭线程池并移除 backward hooks，不留下后台任务。

输出、输入梯度和全部 FP32 LoRA 参数梯度都与独立构造的同精度无交换对照逐位
一致；冻结权重及 adapter 数值、Parameter 身份、state keys 和 dtype manifest
均不变。这里没有比较混合精度与全 FP32 的质量，也没有执行 optimizer.step。

对普通冻结权重，逻辑 payload 字节数为 `2*N_fp16 + 4*N_fp32`。该夹具两类
各 24 个元素，所以为 144 bytes，且与 master 统计一致。不能把它当作实际峰值：
GPU/CPU allocator、活跃与暂存权重、FP32 激活、checkpoint 重算、梯度、optimizer
和传输缓冲都要另计，CPU 测试也没有 GPU 常驻量。当前结果只补上权重表示与
CPU 恢复边界，不认证 PCIe/CUDA stream、速度、不 OOM，或任何三模型双卡组合。

## 10. 可复现验证

```bash
CUDA_VISIBLE_DEVICES= timeout 60 .venv/bin/python -m pytest -q \
  tests/test_adaptive_precision_math.py \
  tests/test_adaptive_precision_aliases.py \
  tests/test_adaptive_precision_scale_feasibility.py \
  tests/test_adaptive_precision_comparison_math.py \
  tests/test_adaptive_precision_swap_cpu.py \
  tests/test_block_swap_master_binding.py \
  tests/test_adaptive_probe_scaling.py \
  tests/test_adaptive_training_precision.py \
  tests/test_adaptive_training_replay.py \
  tests/test_adaptive_runtime.py \
  -k 'not real_accelerate_scaler_clip_and_resume'
```

上述测试全部运行在 CPU 上（额外拦截 CUDA 初始化，排除需真实 GPU 的 `real_accelerate_scaler_clip_and_resume`），无需下载底模或执行真实 DiT forward：

- 覆盖范围：前后向误差界与非单调性反例（`math`）、共享参数拒绝（`aliases`）、GradScaler 可行域与静默下溢（`scale_feasibility`）、FP64 比较指标与无固定 cosine floor 边界（`comparison_math`）、以及块交换混合精度 CPU master 逐位恢复（`swap_cpu` / `master_binding`）。
- 已知保留的 `strict xfail`：仅第 9 节底层 `install_precision_islands()` 跨模块共享同一 `Parameter` 对象的 3 项前置检查缺口（训练入口 `install_training_precision()` 已前置拒绝）；第 8.2 节的 2 项比较指标 `xfail` 已于 2026-09-30 修复并转正。
