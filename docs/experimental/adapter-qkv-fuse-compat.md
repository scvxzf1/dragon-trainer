# Adapter QKV Fuse/Defuse 兼容边界

状态：实验，2026-11-19。QKV 保存/加载主链已统一，旧变体实现仍作为内部兼容 shim 保留。

## 唯一入口

`networks/attn_fuse.py` 是运行期 fused projection 与磁盘 split projection 的唯一编排入口：

- `defuse_qkv_state_dict()` 负责保存侧分派；标准 LoRA、LoHa、LoKr、VeRA、GLoRA 的矩阵
  运算仍由各自 variant 模块实现。
- `refuse_qkv_state_dict()` 负责加载侧顺序；它依次处理 Stacked Experts、Hydra、Chimera
  和普通 split LoRA，最后返回运行期 `qkv_proj`/`kv_proj` 键。
- `ATTN_FUSE_SPECS` 是 q/k/v 与 k/v 投影顺序的单一事实源，禁止在 variant save/load
  文件中重新声明投影列表。

## 旧 helper 审计结论

`networks/lora_anima/loading.py` 中的以下函数不是可删除死代码：

| helper | 当前角色 | 生产调用者 |
| --- | --- | --- |
| `_refuse_split_stacked_experts_keys` | 独立 A 的 stacked expert split → fused | `networks.attn_fuse.refuse_qkv_state_dict` |
| `_refuse_split_hydra_keys` | Hydra split → fused，并恢复共享 router/scale | `networks.attn_fuse.refuse_qkv_state_dict` |
| `_refuse_split_chimera_keys` | Chimera 双池 split → fused | `networks.attn_fuse.refuse_qkv_state_dict` |
| `_refuse_unfused_attn_lora_keys` | 普通 LoRA/DoRA split → fused | `networks.attn_fuse.refuse_qkv_state_dict` |

源码搜索确认它们没有第二个生产调用入口；测试对 helper 的直接调用只覆盖历史布局、
缺失分量和坏形状的兼容行为。删除这些函数会删除 facade 实际委托的变体数学，而不是
清理重复路径，因此当前保留。新增变体必须接入 facade 并补 save/load round-trip，不能
从业务入口直接调用这些私有 helper。

## 验收边界

CPU 回归覆盖标准 LoRA、LoHa、LoKr、VeRA、GLoRA、Chimera、Hydra 与 Stacked Experts
的关键布局；未覆盖的变体必须 fail-closed，不得静默丢弃 adapter keys。涉及
ComfyUI `_vendor/` 的 live `networks/` 改动仍需执行 `python tasks.py vendor-sync`。
