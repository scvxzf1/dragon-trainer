# Soft Tokens Contrastive 兼容入口

状态：已落地为 Soft Tokens 的可选训练目标，默认关闭。

这是旧 `soft_tokens_contrastive` 提案路径的兼容入口。当前可维护说明集中在：

- [experimental/soft_tokens.md](../experimental/soft_tokens.md) 的 “Contrastive objective” 小节
- `configs/methods/soft_tokens.toml`
- `networks/methods/soft_tokens.py`
- `library/training/losses.py::_soft_tokens_contrastive_loss`

当前实现要点：

- B=1 场景下不使用 batch peers，而是从磁盘读取其他样本的 cached text embedding 作为 negatives。
- `contrastive_objective=infonce` 保留为可选目标。
- 负样本来源复用 `library/datasets/base.py::setup_contrastive_negatives` 和 `library/datasets/identity_pairs.py::IdentityPairSampler`。
- 该目标只影响训练期辅助损失，不改变 checkpoint 的推理参数结构。

如果需要 soft-rank/listwise rank 方向，看 [soft_tokens_softrank.md](soft_tokens_softrank.md)。如果继续研究 AGSM 风格的有界双 token 对齐，看 [soft_tokens_agsm.md](soft_tokens_agsm.md)。
