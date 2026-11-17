# Soft Tokens Soft-Rank 兼容入口

状态：已落地为 Soft Tokens 的可选 listwise rank 目标；当前 `configs/methods/soft_tokens.toml` 默认使用 `contrastive_objective=softrank`。

这是旧 `soft_tokens_softrank` 提案路径的兼容入口。当前可维护说明集中在：

- [experimental/soft_tokens.md](../experimental/soft_tokens.md) 的 “Soft-rank objective” 小节
- `configs/methods/soft_tokens.toml`
- `networks/methods/soft_tokens.py::softrank_loss`

当前实现要点：

- `contrastive_objective=softrank` 使用 differentiable listwise rank，让 matched caption 在候选集合中排名靠前。
- `softrank_method` 支持 `neuralsort` 和 `softsort`。
- `softrank_softness` 控制 rank relaxation 的 softness，不复用 InfoNCE 的 `contrastive_tau`。
- dual bank 通过 `dual_bank=true` 启用，ψ⁺/ψ⁻ 走同一 Soft Tokens adapter 的分支选择。

InfoNCE 兼容入口见 [soft_tokens_contrastive.md](soft_tokens_contrastive.md)。AGSM 后续研究入口见 [soft_tokens_agsm.md](soft_tokens_agsm.md)。
