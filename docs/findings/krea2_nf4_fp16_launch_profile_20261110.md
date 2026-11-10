# Krea-2 NF4 + 内层 fp16：kernel launch 规模与 GPU 利用率

状态：实验
适用版本：2026-11-10 `dev`（含 `9d530588`）
相关代码：`bench/krea2_nf4_fp16_grad/launch_count.py`、
`library/quantization/nf4.py`、`library/models/krea2_raw/attention_backend.py`
入口命令：`CUDA_VISIBLE_DEVICES=1 .venv/bin/python bench/krea2_nf4_fp16_grad/launch_count.py --img 256 --steps 2`

## 结论

**单步约 21,200 次 CUDA kernel launch，GPU 实际忙时占墙钟 84.8%（低于 90%），
空泡 424 ms/step。**

**重要修正**：本文最初基于 `nvidia-smi dmon` 的 `sm% = 100` 得出「利用率饱和、
launch 不是瓶颈」的结论，该结论**已撤回**。`nvidia-smi` 的 `utilization.gpu`
定义为「采样窗口内有一个以上 kernel 在执行的时间占比」，采样粒度 1 秒——它只能
证明「没有彻底空闲」，不能证明「SM 排满」。改用 nsys 按真实时间线计算后才得到
84.8% 这个可用数字。

`torch.profiler` 同一路径报出的 4,923 ms/step 也是错的：它是真实值的 2.08 倍，
`key_averages()` 的 `self_device_time_total` 在此路径重复累加。CUPTI 记录
（nsys）是权威口径。

## 测量方法

| 指标 | 工具 | 说明 |
|---|---|---|
| launch 次数 | `torch.profiler`（`cudaLaunchKernel` 计数） | CPU 侧 API 调用，口径干净 |
| 利用率 | `nvidia-smi dmon -s u` | 每秒采样，不叠加 profiler 开销 |

**关键**：两者必须分开测。profiler 自身有开销，混在一起测得的利用率不可信。
利用率测量用 `dir1_e2e.py --repeat 12` 跑纯训练，另起 dmon 采样 25 秒。

## 数据

### launch 规模（img=256，2 步，warmup 1）

```
CUDA launch (cudaLaunchKernel)   42,416    → 21,208 / step
GPU kernel 实例                  100,826    → 50,413 / step
不同 kernel 种类                 192
kernel GPU 时间合计              9,846 ms   → 4,923 ms / step
```

dir1 相比 bf16 基线增加约 **27%** 的 launch（16,696 → 21,208/step），
换来 GEMM 时间从 4,919 ms/step 降到 1,452 ms/step。

### GPU 利用率：三种口径的差异

`nvidia-smi dmon`（25 秒采样）：

```
gpu   sm%    mem%
  0   69-78    28-34     ← GTX 1050，他人负载，非本次训练
  1    100     23-46     ← 训练卡（V100）
  2     0       0
```

卡号映射已核对：`CUDA_DEVICE_ORDER=PCI_BUS_ID` + `CUDA_VISIBLE_DEVICES=1`
解析到 `04:00.0` 的 V100，与 dmon 的 index 1 一致。

**这个 100% 不能用来判断饱和**，见上面「重要修正」。

nsys 按真实时间线计算（`/tmp/prof/step.nsys-rep`，恰好一步）：

```
kernel 实例数          21,917
时间总和（非 union）    2,361.2 ms
union 忙时              2,361.2 ms
重叠                    0.0 ms      ← 完全单流
kernel 时间线跨度       2,649.1 ms
实测墙钟                2,785.1 ms

GPU 忙 / 墙钟          84.8%
空泡                    424 ms (15.2%)
```

`union == 总和` 且 `重叠 == 0` 证明这条路径是纯单流，不存在并发掩盖。

**GPU 实际忙时 84.8%，低于 90%。**

### 强制同步点

```
aten::_local_scalar_dense   786 次 / step
```

每次 `.item()` / 标量提取都是一次 GPU→CPU 同步。786 次/step 是个不小的数，
但在 2.6–2.9 s 的步长上仍未造成利用率损失——说明它们大多发生在步末的
loss 汇总/日志路径，而不是散布在 28 个 block 的 forward 里。

调用栈未能采集（此 torch 版本的 `evt.stack` 为空），所以**无法证伪**「同步点
集中在步末」这个推断，它只是与 100% 利用率一致的解释。

## 空泡来源未定，有两种可能

```
21,917 kernel / 2,649 ms ≈ 每 121 μs 一个
424 ms / 21,917 ≈ 平均每个间隙 19 μs
```

19 μs 的间隙有两种成因，优化方向完全不同：

- **CPU 分发跟不上** → 优化 launch 次数有效
- **kernel 尾部排空（tail effect）** → 优化 launch 无用，需改 kernel 形状

**本报告尚未区分这两者。** 分辨方法是比对 CPU 侧 `cudaLaunchKernel` 时刻与
GPU 侧 kernel 起始时刻：若 GPU 在等 CPU，间隙落在 CPU 侧；若 CPU 早已入队，
间隙是 GPU 自身的排空。

**因此「launch 数量是不是瓶颈」仍未定论**，只能确定 GPU 有 15.2% 的时间空着。

## 对优化的含义

**不能断定「不该围绕 launch 做优化」**——原始版本这么写是基于错误的 100%
口径。现在只确定 424 ms 空泡存在，成因待归因。在归因完成前，减少 launch
的收益上限是那 424 ms 的一部分，而不是全部。

**该看的是 GEMM 效率。** kernel 时间分布里 `aten::mm` 占 1,452 ms/step，
是绝对大头；elementwise 和 copy 加起来约 500 ms/step。原 findings
`krea2_nf4_fp16_inner_path_20261109.md` 里 elementwise 占 2.065 s / 11.06 s
的观察，在 256 分辨率下同样成立。

**CUDA graph 的收益预期要放低。** 既然 launch 不是瓶颈，用 graph 压缩
launch 只能省下 CPU 侧那 2–4%，不会带来数量级改善。

## 未验证

- **424 ms 空泡的归因未做**（CPU 分发 vs GPU 尾部排空），因此 launch 是否为
  瓶颈仍未定论
- 调用栈未采集，786 次同步点的分布位置未定位
- 只在 img=256 测过；1024 分辨率下 launch 数量会随 token 数增长，空泡比例是否
  同比例变化未测
- 只在单卡 V100 测过，未验证多卡或 3080 等小显存卡
- `sm%` 与真实忙时是两个不同指标，本文已说明前者不可用于饱和度判断
