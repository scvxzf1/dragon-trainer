# N:M 与 W4A16 的结构关系

状态：实验
适用版本：当前 main；可运行边界以 `tasks.py --help` 和实时源码为准
入口命令：`python tasks.py exp-nm-w4a16`（转发至 `bench.nm_w4a16`，用于 2:4 INT4/BF16 静态开销核算或与 dense BF16 的显式 GPU 探针对照；兼容别名：`make exp-nm-w4a16`）

`N:M` 约束权重位置；`W4A16` 约束数值表示。二者是正交轴：在 2:4 中每四个
权重最多保留两个，保留值使用 uniform INT4，激活和累加仍使用 BF16/FP32。存储
压缩与 Tensor Core 实际加速必须分别核算；“4 bit 权重”不代表乘法指令自动为 INT4。

训练还要计算 `dX = dY W`。仅前向合法的行向 2:4 mask 不足以复用压缩布局；需要
同时沿转置 reduction 维度满足 2:4，或者为反向另建可用表示。本实验选择每个 4×4
tile 行列均为二度的平衡 mask，因此静态上可复用 forward/backward 的稀疏结构。

与 dense BF16 相比，若不计 scale、padding 和缓存，2:4 INT4 payload 是 3 bit/原始
权重，而 BF16 是 16 bit；带每组 `G` 个权重的 FP32 scale 后为 `3+32/G` bit。理论
前向与 `dX` 乘加量均减半，但端到端训练速度还受解码、稀疏元数据、反向、LoRA、
激活与访存约束，不能把存储比当作速度比。
