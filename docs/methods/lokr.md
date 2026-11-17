# LoKr

状态：稳定
适用版本：当前 main
入口命令：`python tasks.py lora-gui lokr`（兼容别名：`make lora-gui GUI_PRESETS=lokr`）/ WebUI 选择 `lokr`（`configs/gui-methods/lokr.toml`）
相关代码：`networks/plugins/lokr/`

LoKr 把适配器增量参数化成 Kronecker 积：

```text
delta_W = kron(w1, w2) * (network_alpha / network_dim)
```

`lokr_factor` 控制 Kronecker 维度拆分。`lokr_decompose_w2=true` 时，较大的
`w2` 会再拆成低秩对 `lokr_w2_a / lokr_w2_b`，此时 `network_dim` 才控制这个
二次低秩。

## Dragon Trainer 语义说明

这里的 `lokr_full_factor` 是**显式全因子声明 + 旧哨兵迁移开关**，不是
“从低秩默认突然切到全因子默认”。

| 配置 | 实际布局 |
| --- | --- |
| `lokr_decompose_w2=false`（默认） | 已是完整 `lokr_w1` + 完整 `lokr_w2` |
| `lokr_full_factor=true` | 强制保持完整因子，并拒绝 `lokr_decompose_w2=true` |
| `lokr_decompose_w2=true` | 才把大的 `w2` 再拆低秩 |

所以：

- 默认 LoKr 训练本来就是 full-w2 容量
- 打开 `lokr_full_factor=true` 不会突然改训练动力学
- 它的主要价值是：正规化配置、写入/识别 `ss_lokr_full_factor`、彻底弃用
  `network_dim=114514`

## 推荐写法

```toml
network_dim = 32
network_alpha = 32
use_lokr = true
lokr_factor = 8
lokr_decompose_w2 = false
lokr_full_factor = true
```

这样会得到完整的 `lokr_w1` / `lokr_w2`，同时训练缩放保持 `32 / 32 = 1`。

不要再用 `network_dim = 114514` 当全因子哨兵。那个值虽然也能表达“别拆”，
但会把缩放改成 `network_alpha / 114514`。当 `network_alpha = 32` 时，输出和
初始梯度大约被压到 1/3578.6。

全因子 LoKr 仍然有 Kronecker 结构。它能达到两个因子允许的最大秩，但不是
对基底权重做无限制全量微调。

## 旧状态续训

历史训练状态如果用过哨兵，续训时不能直接改缩放，否则会出现训练跳变。
续训时必须显式打开兼容开关：

```toml
lokr_allow_legacy_dim = true
```

训练器会保留历史缩放并发出警告。新训练会拒绝该哨兵，并引导使用
`lokr_full_factor`。

## 保存与加载

- 保存时会写 `ss_lokr_full_factor`
- 以**实际模块布局**为准：完整 `lokr_w2` 会 stamp 为 `true`
- 加载时优先读 stamp；旧无 stamp 的 full-w2 权重会推断为 full_factor
- stamp=`true` 但权重却是 `lokr_w2_a/b` 时会直接拒绝
