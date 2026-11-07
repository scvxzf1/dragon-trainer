状态：阶段 7 已完成（Inductor mode 消融）
日期：2026-08-09
原始摘要：`krea2_3080_speed_stage7.json`

# Krea-2 compile mode 消融：阶段 7

## 假设

`torch.compile(mode="reduce-overhead")` 会开启 CUDA Graph，理论上可减少 per-block Python/
kernel launch 开销。但 Krea-2 训练每个 block 用 non-reentrant checkpoint，backward 会
重新调用同一 `_forward`，输入/输出地址契约不一定满足 CUDA Graph。

## 前置探针

PG199 上构造小型 NF4 `SingleStreamBlock`，开启标准 checkpoint，编译 `_forward`，
执行 forward + backward。默认 Inductor mode 已在阶段 3 通过；`reduce-overhead`
在 checkpoint recompute 失败：

```text
RuntimeError: accessing tensor output of CUDAGraphs that has been overwritten
by a subsequent run
```

错误指向 modulation `vec + self.lin` 的 graph output，并要求在编译外 clone 或每轮调
`torch.compiler.cudagraph_mark_step_begin()`。这会改变 checkpoint wrapper 和训练编排契约，
为尚未实测的 launch 收益引入这类修复不值得。

## 判定与修复

**REJECT**。Krea-2 `compile_blocks` 现只接受 `mode=None` 或 `mode="default"`：

- `reduce-overhead`：已证 checkpoint/CUDA Graph 运行时冲突。
- `max-autotune`、`max-autotune-no-cudagraphs`：未验证编译时间、显存和 NF4 数学，
  同样显式拒绝，而不是默认转发。

`configs/methods/krea2_lora.toml` 注释已更新：固定长度 opt-in 必须使用
`compile_dynamic_seq=false`、`compile_block_scope="resident"` 和 default mode。
