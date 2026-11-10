# Krea-2 NF4 + 内层 fp16：kernel launch 规模与 GPU 利用率

状态：实验
适用版本：2026-11-10 `dev`（含 `9d530588`）
相关代码：`bench/krea2_nf4_fp16_grad/launch_count.py`、
`library/quantization/nf4.py`、`library/models/krea2_raw/attention_backend.py`
入口命令：`CUDA_VISIBLE_DEVICES=1 .venv/bin/python bench/krea2_nf4_fp16_grad/launch_count.py --img 256 --steps 2`

## 结论

**单步 21,208 次 CUDA kernel launch，但 GPU 利用率实测 100%，不是瓶颈。**

launch 数量偏高（典型大模型 3k–8k/step），但在这台 V100 上并没有转化成
利用率损失。原因是这些 launch 被 GPU 队列吸收得很好：SM 始终有活干。

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

### GPU 利用率（dmon，25 秒采样）

```
gpu   sm%    mem%
  0   69-78    28-34     ← GTX 1050，他人负载，非本次训练
  1    100     23-46     ← 训练卡（V100），稳定饱和
  2     0       0
```

卡号映射已经核对：`CUDA_DEVICE_ORDER=PCI_BUS_ID` + `CUDA_VISIBLE_DEVICES=1`
解析到 `04:00.0` 的 V100，与 dmon 的 index 1 一致。

**SM 利用率 100%，不是低于 90%。**

### 强制同步点

```
aten::_local_scalar_dense   786 次 / step
```

每次 `.item()` / 标量提取都是一次 GPU→CPU 同步。786 次/step 是个不小的数，
但在 2.6–2.9 s 的步长上仍未造成利用率损失——说明它们大多发生在步末的
loss 汇总/日志路径，而不是散布在 28 个 block 的 forward 里。

调用栈未能采集（此 torch 版本的 `evt.stack` 为空），所以**无法证伪**「同步点
集中在步末」这个推断，它只是与 100% 利用率一致的解释。

## 为什么 21k launch 没有拖垮利用率

拉平看：21,208 次 launch / 2.6 s ≈ **每 123 μs 一次**。CPU 侧单次分发 2–5 μs，
占比 2–4%，队列不会饿死 GPU。

真正的风险不在数量，而在两类结构性分布：
- 若某个同步点插在 block 循环内部，每次都要排空队列
- 若 kernel 小到 GPU 调度器来不及填满 SM

本次两项都没触发，所以 100%。

## 对优化的含义

**不该围绕「减少 launch 次数」做优化。** 21k 这个数字看起来吓人，但实测
证明它不是瓶颈。把力气花在它上面不会换来步时下降。

**该看的是 GEMM 效率。** kernel 时间分布里 `aten::mm` 占 1,452 ms/step，
是绝对大头；elementwise 和 copy 加起来约 500 ms/step。原 findings
`krea2_nf4_fp16_inner_path_20261109.md` 里 elementwise 占 2.065 s / 11.06 s
的观察，在 256 分辨率下同样成立。

**CUDA graph 的收益预期要放低。** 既然 launch 不是瓶颈，用 graph 压缩
launch 只能省下 CPU 侧那 2–4%，不会带来数量级改善。

## 未验证

- 调用栈未采集，786 次同步点的分布位置未定位
- 只在 img=256 测过；1024 分辨率下 launch 数量会随 token 数增长，
  但利用率是否同样饱和未测
- 只在单卡 V100 测过，未验证多卡或 3080 等小显存卡
- `nvidia-smi dmon` 的 `sm%` 是采样窗口内的瞬时占用率，不是精确的指令发射率
