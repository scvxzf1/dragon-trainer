# Anima 混合 NF4 的 autocast 输出与激活显存

状态：实验。定向验收与三组真实短程对照通过；训练显存下降，VAE主导的全程峰值基本不变。

本轮只修算法实现，不改变用户配置、49 个 NF4 MLP 的量化策略、rank32、
swap12、BF16 compute、full checkpoint 或 1024×1024 分辨率。

## 根因

bitsandbytes 0.49.2 的 `Linear4bit.forward` 在 BF16 运算后将输出转换回输入
dtype。Anima 的归一化/调制可以提供 FP32 输入；普通 `nn.Linear` 在 BF16
autocast 中返回 BF16，而原 NF4 分支返回 FP32。

LoRA 根据基座输出 dtype 决定计算 dtype，因此该 FP32 输出会使 LoRA 分支及
MLP 残差继续使用 FP32，并使随后 block 的 checkpoint 保存 FP32 输入。
原始首步 CUDA 分配快照发现 27 份 32 MiB 的 checkpoint 边界激活，共864 MiB。
相同形状 `[1,1,64,64,2048]` 使用 BF16 时每份16 MiB，仅这一项就相差432 MiB。
旧日志的 `after_forward allocated - model storage` 差值也为432.045 MiB。

快照还发现 MLP LoRA 加法和 bnb FP32 输出的临时分配；bnb 本身只保存 packed
权重引用及 QuantState，没有把49份完整反量化 BF16 权重长久留在计算图里。
因此本轮不采用只能节省单层几十 MiB、且可能拖慢 GEMM 的分片反量化方案。

## 实现

`library/anima/mixed_nf4/linear.py::AnimaMixedNF4Linear` 继承 bnb Linear4bit。
仅 CUDA BF16 autocast 开启、compute dtype为BF16、输入为FP32/FP16时，在调用
原 forward 前把输入转成BF16，避免先分配大FP32输出再转回BF16。
`runtime.py::restore_linear` 使用此类；没有新配置或全局 bnb patch。

CPU、FP64输入、其他compute dtype、FP16 autocast和禁用autocast继续走原语义。
原packed Parameter、QuantState、自定义反向与稳定换层缓冲保持不变。
LoRA可训练参数仍为FP32，本次改变的是在既有autocast下的计算/输出dtype。

先前已存在的 `loss.detach()`、预览生命周期和 VAE 解码前 CPU park/restore
在全部新对照中共同启用，不计入本次收益。原2026-10-02的历史峰值不能直接作为
本次修复前基线。

## 定向验收

独立验收：CPU三文件44 passed / 61 skipped；CMP90HX三文件104 passed / 1 skipped。
Ruff通过，Pyright 0 errors / 0 warnings。

- 量化小夹具的输出、输入梯度与全部plain LoRA梯度，对stock bnb显式BF16输入逐位一致。
- 对同一packed权重反量化后的普通Linear，前向输出逐位一致；dense反向采用明确容差。
- 验证4032/4200 token原生5D布局、非连续输入、checkpoint开关、真实Anima小模型、
  无bias、状态保存、LoRA目标识别、无autocast及嵌套禁用语义。
- 换层回归包含foreach/slab、checkpoint开关、训练→采样→训练及动态换层。

复现：

```bash
CUDA_VISIBLE_DEVICES=GPU-806debd9-d254-da57-d515-ccae15b4ac6a \
timeout 60 .venv/bin/python -m pytest -q \
  tests/test_anima_mixed_nf4_autocast.py tests/test_anima_mixed_nf4.py \
  tests/test_block_swap_mixed_nf4.py
```

## 真实短程对照与证据

诊断目录：`output/anima-nf4-algorithm-20261003/controlled-ab/`。
隔离wrapper比较stock bnb、修复NF4和BF16基座，各4步、第2/4步采样、相同caption
衍生prompt和Euler8步，加载同一LoRA并在loop_ready重置seed20261002。
原500步配置及产物不变；这轮短测不证明长期收敛或样图质量等价。

`run_probe.py` 仅在stock诊断进程把实验子类替换成原Linear4bit；不覆盖项目源码。
记录源码指纹、初始LoRA哈希、逐步loss、block/MLP输入dtype、分阶段显存及样图。
诊断包含CUDA同步，不作为吞吐基准。采样峰值可能由VAE主导，必须与训练峰值分开看。

三组全部exit0；每组4步、第2/4步采样并继续训练，最终840个LoRA张量全部finite。
三组初始LoRA SHA256相同：
`e61092f16d65e50b0ad6f76a4a799a1d5f2c3f98003f9bd0990d5f90ccf9f4e2`。
运行前后及三组之间的训练源码指纹一致。行为配置没有差异；模型路径及输出、日志、
诊断记录路径和名称按实验目录隔离，stock/fixed使用同一个量化文件。

CUDA allocated实测（GiB；四步最大值，普通步末采用第2步）：

| 指标 | 旧NF4实现 | 修复NF4实现 | BF16基座 |
| --- | ---: | ---: | ---: |
| 前向峰值 | 3.7504 | 3.0711 | 3.6448 |
| 反向/训练峰值 | 4.0670 | 3.4439 | 4.0176 |
| 普通步末 | 2.3166 | 2.2716 | 2.8453 |
| 含采样的全程峰值 | 4.7009 | 4.6988 | 4.6988 |
| 首次forward各block输入总逻辑MiB | 880 | 448 | 448 |
| BF16 / FP32 block输入数 | 1 / 27 | 28 / 0 | 28 / 0 |

修复比旧NF4训练峰值减少638 MiB（15.32%），比同参数BF16基座减少587.46 MiB
（14.28%）。前向峰值下降695.63 MiB，除了432 MiB边界激活外，也消除了LoRA/MLP
临时FP32分配。普通步末下降46 MiB；两者不是同一口径。
含采样全程仅下降约2.15 MiB，修复后与BF16一样由VAE解码约4.70 GiB主导。
因此量化已在训练阶段产生实际收益，但这份带采样任务的最低显存门槛仍未明显下降。

数值：旧/新四步loss分别为
`[0.22462502, 0.15667361, 0.16949117, 0.46406353]`和
`[0.22490203, 0.15681422, 0.16955733, 0.46474159]`，最大绝对变化约0.0006781。
最终保存LoRA有280/840个张量逐位一致，max_abs=9.15527e-5、整体relative-L2=2.28809e-4。
这是移除意外FP32计算后的舍入变化；不能把短测的接近程度外推为长期训练质量等价。
完整模型compile、其他adapter变体和其他硬件未在本轮验收。

可复核本地产物（未入库）：

- `output/anima-nf4-algorithm-20261003/controlled-ab/metrics.json`
- 运行命令与退出状态：`output/anima-nf4-algorithm-20261003/controlled-ab/status.json`
- 阶段记录：`output/anima-nf4-algorithm-20261003/controlled-ab/stock.phase.jsonl`、
  `output/anima-nf4-algorithm-20261003/controlled-ab/fixed.phase.jsonl`、
  `output/anima-nf4-algorithm-20261003/controlled-ab/bf16.phase.jsonl`
- 独立分析脚本：`output/anima-nf4-algorithm-20261003/controlled-ab/analyze.py`

原始归因快照另存于 `output/anima-nf4-algorithm-20261003/current-profile-005942/`。
其采样history达到100000条上限，早期记录截断；不能按完整时间线解释整个采样。
首步训练history未截断；该运行后半段有其他GPU任务，所以不使用其耗时比较速度。
