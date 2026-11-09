# Findings 索引

状态：索引
适用版本：当前文档树

这里放审计、实验结论、失败路径、性能记录和阶段性整理报告。

## Anima 权重静态分析

| 文档 | 说明 |
| --- | --- |
| [分类文件清单](anima_weight_analysis_20261001/README.md) | 2026-10-01 人物与画风调查产物、图表、统计和验证记录 |
| [人物与画风汇总](anima_weight_analysis_20261001/summary.md) | 15 文件、5,700 模块；共同目标对照与异常样本敏感性检查 |
| [画风完整报告](anima_weight_analysis_20261001/style/report.md) | 10 文件的有效 ΔW、模块分布与层热点 |
| [人物完整报告](anima_weight_analysis_20261001/character/report.md) | 5 文件的逐模型差异、集中度及人物与画风对照 |

## 数据集、存储与传输

| 文档 | 说明 |
| --- | --- |
| [image_compression_bandwidth_audit_20261102.md](image_compression_bandwidth_audit_20261102.md) | 2026-11-02：Lepton / LatentStore / JXL-Chrome155 / GACT / INSTANT / dreaMLearning 6 篇材料核验（2 个 arXiv 编号指错论文、1 篇查无此文、2 篇指标转述失真）+ 本机实测压缩率到有效带宽换算（传统 JPEG q95/q92 重压缩 2.2~2.9×、JXL 有损 5.0×、降分辨率 13~20×），含等画质交叉、4:2:0 隐性降质、1-5/5-10/10-15MB 分档的 SSIM+PSNR+ΔE76 与编解码 CPU/内存占用+ JXL 钩子在当前代码中失效的证据 |
| [image_compression_bench_data_20261102.md](image_compression_bench_data_20261102.md) | 2026-11-02：上述基准的纯数据记录（分档汇总、逐图像素指标与编解码耗时、CPU%/峰值 RSS），原始 JSONL 与采集脚本归档在 `image_compression_bench_data_20261102/raw/` |
| [image_storage_pipeline_recommendation_20261102.md](image_storage_pipeline_recommendation_20261102.md) | 2026-11-02：在 128 核 / 6 GB/s 盘 / 500 GB/s 内存假设下的容量模型与决策建议（单图 CPU 秒成本、转码 vs 传输总时间矩阵、latent 缓存后的磁盘占用、内存带宽核算、盈亏平衡链路速度） |

## 维护和整理

| 文档 | 说明 |
| --- | --- |
| [uncommitted_audit_closure_20260913.md](uncommitted_audit_closure_20260913.md) | 未提交更新审计修正、本地提交序列、验证证据与未关闭风险 |
| [test_suite_audit_20260831.md](test_suite_audit_20260831.md) | 2026-08-31 测试集规模、分层、重复、source-probe 与默认门禁审计 |
| [test_script_audit_20260809.md](test_script_audit_20260809.md) | 2026-08-09 测试脚本审计、精简结果与 probe 保护边界 |
| [documentation_consolidation_20260706.md](documentation_consolidation_20260706.md) | 2026-07-06 文档库合并整理报告 |
| [project_cleanup_checkpoint_20260705.md](project_cleanup_checkpoint_20260705.md) | 项目清理检查点 |
| [project_cleanup_sustained_goal_20260706.md](project_cleanup_sustained_goal_20260706.md) | 持续清理目标最新记录 |

## WebUI 和配置

| 文档 | 说明 |
| --- | --- |
| [dragon_next_stage_audit_verification_20260924.md](dragon_next_stage_audit_verification_20260924.md) | Dragon Next 当前只读盘点、验证阻塞、阶段性审计与功能验收矩阵 |
| [dragon_next_s0_baseline_20260924.md](dragon_next_s0_baseline_20260924.md) | Dragon Next S0 只读门禁、四视口基线、主题/缩放与静态包哈希溯源结果 |
| [dragon_next_s1_shell_audit_20260925.md](dragon_next_s1_shell_audit_20260925.md) | Dragon Next S1 mock E2E 进度、壳层回归补测与未关闭风险 |
| [dragon_next_s2_isolation_20260925.md](dragon_next_s2_isolation_20260925.md) | Dragon Next S2 配置、数据集、模型与设置的隔离写入验收 |
| [dragon_next_s3_queue_monitor_history_20260925.md](dragon_next_s3_queue_monitor_history_20260925.md) | Dragon Next S3 队列、监控、历史与续训控制面隔离验证报告 |
| [dragon_next_s4_image_mask_caption_20260925.md](dragon_next_s4_image_mask_caption_20260925.md) | Dragon Next S4 图片工作台、蒙版批量应用与打标候选/TXT commit 隔离验收 |
| [dragon_next_s5_release_candidate_20260925.md](dragon_next_s5_release_candidate_20260925.md) | Dragon Next S5 隔离发布候选、深链、原子替换与回退验证记录 |
| [webui_frontend_audit_20260913.md](webui_frontend_audit_20260913.md) | 2026-09-13 Classic/Dragon/Next 前端当前工作树审计：静态/Next 门禁、cache token、配置库键盘排序与文档风险分级 |
| [captioning_uiux_external_review_20260829.md](captioning_uiux_external_review_20260829.md) | Captioning 13 界面外部视觉评审 API 探针、失败边界与后续裁决要求 |
| [captioning_uiux_iteration_20260829.md](captioning_uiux_iteration_20260829.md) | Captioning/打标工作台 13 界面两轮 UI/UX 优化与验收记录 |
| [tagging-port-spec.md](tagging-port-spec.md) | 打标能力净室移植行为契约：caption 还原点（Phase 1）与 caption 写入语义（Phase 2），含审计→修复→独立复核闭环（BAC / FAC / T9-1 全量状态）、captions_json 写回 fail-closed 设计决策与已知限制 |
| [dragon_frontend_performance_20260826.md](dragon_frontend_performance_20260826.md) | Dragon 五个核心路由的按需加载与运行时优化历史基线（不含后续 tagging 工作台） |
| [dragon_frontend_uiux_review_20260829.md](dragon_frontend_uiux_review_20260829.md) | Dragon 13 页面双视口截图评审、外部视觉模型意见与实施顺序 |
| [dragon_responsive_hardcoding_audit_20260824.md](dragon_responsive_hardcoding_audit_20260824.md) | Dragon 视口硬编码解耦、浏览器降级和响应式验证记录 |
| [dragon_classic_feature_parity_audit_20260815.md](dragon_classic_feature_parity_audit_20260815.md) | Classic、现有 Dragon 与 React 新前端的逐功能对照审计及迁移证据 |
| [dragon_dataset_subset_field_audit_20260815.md](dragon_dataset_subset_field_audit_20260815.md) | Dragon 数据集 subset 字段与后端契约对照审计 |
| [dragon_next_audit_20260819.md](dragon_next_audit_20260819.md) | Dragon React 前端对 live backend 的阶段性审计 |
| [dragon_next_stage1_queue_monitor_history_20260819.md](dragon_next_stage1_queue_monitor_history_20260819.md) | Dragon React 队列、监控和历史页面阶段 1 集成记录 |
| [dragon_ui_pr1_integration_20260814.md](dragon_ui_pr1_integration_20260814.md) | Dragon UI PR #1 的分阶段集成、测试门禁、调试与发布记录 |
| [webui_backend_audit_20260815.md](webui_backend_audit_20260815.md) | WebUI 后端安全、并发、状态机和测试覆盖审计 |
| [webui_frontend_p0_p2_fix_20260726.md](webui_frontend_p0_p2_fix_20260726.md) | 2026-07-26 WebUI 前端 P0–P2 审核修复记录（no-undef / 搜索 debounce / WS / dashboard） |
| [webui_dataset_cross_group_drag_fix_20260727.md](webui_dataset_cross_group_drag_fix_20260727.md) | 2026-07-27 数据集/文件分组跨组拖动静默失败修复 |
| [webui_frontend_visual_audit_20260530.md](webui_frontend_visual_audit_20260530.md) | WebUI 视觉和交互审计 |
| [webui_god_files_refactor_20260607.md](webui_god_files_refactor_20260607.md) | WebUI 上帝文件治理合并记录 |
| [training_history_detail_performance.md](training_history_detail_performance.md) | 训练历史详情性能记录 |
| [ui_scale_independent_settings.md](ui_scale_independent_settings.md) | UI 缩放独立设置结论 |
| [uncommitted_remote_review_20260810.md](uncommitted_remote_review_20260810.md) | 2026-08-10 未提交改动与远端同步审计 |

## Runtime 和能力边界

- [fp16/E3 窗口事务实施记录](fp16_transaction_implementation_20261107.md)：统计事务、多卡一致性与可观测性的实现边界（含未验项）。
- [Qwen-Image 2.1 上的 Block Swap 与自动水位（CMP 50HX 第 12 轮）](block_swap_qwen_image_21_hot_50hx_20261019.md)：第四族、也是四族收口的一轮。固定 `swap=30` 每步 60 任务、24.38 GiB，自动晋升 `30 → 14`（free=7.00 GiB、margin=0.39 GiB、block=416.0 MiB，可晋升 16 块但受 `current-2` 契约限制）后降到每步 28 任务、11.38 GiB（−53.3%），步跨度 7848 → 3877 ms（−50.6%）；每块 416.0 MiB 与头部推算逐字节一致。**修正 P0' 结论**：`exposed_wait` 小不等于搬运不构成瓶颈——baseline 的 DMA 忙时间（8362 ms）甚至超过步跨度（7848 ms），步时约一半由搬运量间接决定。另修正第 9 轮的 qwen 载荷数字，并把 `duplicate_restores` 判定为跨相位误报。含 tokenizer 定位、`timestep_sampling` 漏配、错误 checkpoint 三个实测坑。
- [Z-Image 上的 Block Swap 与自动水位（CMP 50HX 第 11 轮）](block_swap_zimage_hot_50hx_20261019.md)：给探针加 `--family` 族预设（各族 TE 缓存入口、缓存后缀、运行时键；目录型组件原地引用），并在第三族真机验收。日志 `10.11 GiB across 30 blocks` 与头部推算一致；固定 swap=28 每步 56 任务、20260 MB（预测 20262）、3.117 GB/s（天花板 99.3%）、DMA 占步时约 93%。**非 Anima 族的正向晋升首次实测**：`28 → 13`（free=6.15 GiB、margin=0.78 GiB、block=345.0 MiB，可晋升 15 块），步时 7.69/8.12 → 5.30/5.38 s（−30%），每步少搬 10853 MB，一次性晋升约 1.6 s 且一步回本。含三个实测坑与未验证边界。
- [Krea-2-Raw 上的 Block Swap 与自动水位（CMP 50HX 第 10 轮）](block_swap_krea2_hot_50hx_20261019.md)：第二个模型族的真机验收。日志 `22.64 GiB across 28 blocks` 与只读头部推算的 22.64 GiB/28 块/828.1 MiB 相差 0.011%；每步 52 任务（=2S）、45147 MB（预测 45153，−0.013%）、有效带宽 3.114 GB/s（天花板 99.2%）、DMA 占步时约 96%。无余量时调优器正确保持 `blocks_to_swap=26`（free=0.67 GiB 小于 1.17 GiB 余量），且单块 828 MiB > 空闲 686 MiB 说明该卡上 Krea-2 的晋升结构性不可行。含原生分辨率 OOM、缓存 identity、`image_dir` glob、`save_every_n_steps=0` 零除四个实测坑与未验证边界。
- [Block Swap 各族分块载荷与晋升算术（CMP 50HX 第 9 轮）](block_swap_family_payload_50hx_20261019.md)：只读 safetensors 头部算出四族真实分块载荷（anima 132.0 MiB/块、z_image 345.1、krea2_raw 828.1、qwen_image_2_1 416.0，均为均匀块）与 `swap` 上界（26/28/26/30）；Anima 档与真机 profiler 的 138.4 MB/任务逐字一致，四族总量与文件大小交叉吻合。**第 12 轮更正**：初版 qwen 行（60 块/上界 58/648.2 MiB/37.98 GiB）取错了权重文件，指向了会被 `_validate_header` 拒绝的 60 块 fp8 架构，改用本族真实 32 块 bf16 后为上列数值并已真机逐字节验证。推论：非 Anima 族「省一块」值 0.23~0.55 s/步（Anima 的 2.6~6.3 倍）。含工具自身抓到的 `llm_adapter.blocks` 同号 bug 与未验证边界；跨族真机的缺口在探针（TE 缓存已由 `tasks.py preprocess-te` 按 `model_family` 分派）。
- [INT8 单字节编码的误差上限（CMP 50HX 第 4/5 轮）](int8_encoding_floor_50hx_20261019.md)：per-row 对称 int8 已贴均匀量化噪声地板，离群裁剪/scale 搜索的上限只有 ≤2.01%（3A 否定）；固定 NF8 码本在真实 Anima 行再降 13.7%~64.3% MSE，但在高斯型行回归 70%~477%，必须按行择优；1.09% loss 偏移尚未归因。含 15 项单测与未验证边界。
- [Block Swap 进程内水位调优（CMP 50HX 第 3/7/8 轮）](block_swap_auto_tuner_50hx_20261019.md)：`--block_swap_auto` 在第一个优化器步后按空闲显存晋升常驻块。512 默认模式 24 → 2（H2D 6644 → 554 MB/步、任务窗口 2118 → ~720 ms）；256 加停止条件 `stop_ratio=0.5` 按设计停在 13（省约 1.45 GB 显存、步时 +0.4 s）；1024 默认模式 24 → 2，DMA 占步时降到约 5%，绑定约束变成晋升上限（保留 2 个换出块）而非显存。含 21 项单测与未验证边界。
- [Block Swap 恢复链已到链路极限（CMP 50HX 第 2 轮）](block_swap_restore_link_limit_50hx_20261019.md)：纯拷贝微基准测出本机链路天花板 3.14 GB/s（pinned，分块粒度无关）；训练内 profiler 加字节计量后实测每步 3321.9 MB / 24 任务，foreach 达 3.02 GB/s、slab 达 3.12 GB/s（96% / 99% 链路），A/B 中 `slab` 优于 `foreach`（DMA +3.3%、步时 −4%），该默认值随后由上游 `e8ad25b3` 定为 `slab`；据此确认后续杠杆只剩减少搬运字节（auto swap / int8）。（该测量为 Gen1×16 时期；当前实测 ×8、H2D 1.58 GB/s，同日 17:40 另有 3.14 GB/s 记录）
- [Block Swap 宏观流水线基线（CMP 50HX）](block_swap_pipeline_baseline_50hx_20261019.md)：同一工作区 swap=0 vs swap=12 A/B（swap 代价 ≈ +0.45~0.49 s/步），profiler 实测 24 任务/步(=2S)、重复恢复 0 次、exposed_wait ≈ 1.6 ms/步、单块 H2D 有效带宽 ≈1.3~2.3 GB/s；据此否定"折返点重复搬运"前提，把 P0 修正为压缩串行 DMA 链。含复现命令、边界与未证明项。
- [模型身份缓存重建 · CMP 50HX 真机热测试](cache_identity_hot_50hx_20261018.md)：三条真机热测试（VAE latent / Anima TE / 真实 train.py 启动）。覆盖身份一致复用、路径变化重建、就地 mtime 变化重建、旧缓存默认复用与严格模式重建、换 VAE 丢弃陈旧分辨率 key、消费端匹配接受/不匹配拒绝，以及训练启动路径上的 fail-closed 拒绝与重建恢复（干净启动 10 步、0.81~0.88 s/步、max allocated 4.44 GiB）。另记录本轮暴露的两个无关阻塞：flash 可用性未按设备能力判断（已加守卫并补测试）、int8 block-swap 参数与 Anima/Krea-2 enable_block_swap 签名不匹配（未修，blocks_to_swap>0 会 TypeError）。数据集为 2 图 / 512×512，非吞吐基准。
- [MOSS-Image BF16 配置 CMP 50HX 十步基准](moss_image_bf16_baseline10_50hx_20261017.md)：256×256、22-token、plain LoRA rank32、swap20；10 次有效更新，后 9 步平均 11.3386 秒，PyTorch 峰值 allocated/reserved 5.8498/5.9277 GiB。
- [MOSS-Image 分辨率 / 图像 token 预算档位](moss_image_token_budget_tiers_20261017.md)：1024/768/512 三档与文本上限档的 bucket 表、塌缩陷阱、数据短边契约、attention 成本模型和热测试协议；静态契约已验收。
- [MOSS-Image 档位热测试：CMP 50HX 十二行实测](moss_image_tier_hot_50hx_20261018.md)：7 行协议矩阵 + 3 行 1024 档补充短测 + 2 行 int8/fused 配对，各 10 步。短字幕档 5.67/3.44/3.04 s/步（1024/768/512），面积收益次线性（≈2.1 s 固定项 + 0.00084 s/token）；4096-token 字幕仅 512 档装得下（7.21 s/步），1024 档首步 OOM、768 档第 3 步 OOM，`tecap1024` 让 1024 档可跑（7.23 s/步）。1024 档文本天花板在 cap2048（9.37 s/步、9777 MiB、余量<8%）与 cap3072（首步 OOM）之间；cap4096 用 swap28 可跑完但慢到 14.12 s/步。10 步窗口内无热降频证据；含被作废首轮矩阵根因与 `moss_image_te_v2` token-cap 缓存身份修复。质量对照未做。
- [MOSS-Image 块交换传输诊断：int8 的 0.94 s 与 3.5 GiB 归属](moss_image_int8_blockswap_diagnosis_20261018.md)：int8 传输净赚 1.222×（2.9312→2.3986 s/步）但 reserved 5.93→9.16 GiB；隔离基准 + SM 饱和实验定位 DMA 队头阻塞，落地 `int8_restore_mode=fused`（单 kernel 反量化 + 静态 staging）后 1.9740 s/步（相对 bf16 1.485×）、reserved 7.21 GiB；并裁定功耗/3080/段管理/fp8 等既有推断；`ANIMA_BLOCK_SWAP_INT8_STAGING_RING` 三档 staging 实测给出显存与速度的二选一；最终组合 int8 fused + `blocks_to_swap=7` 实测 **0.8793 s/步（相对 bf16/swap20 3.33×）**，代价是 int8 量化使 loss 系统性偏移 0.3%~2.0%；双流解耦实验否证了「0.47 s 可回收」；weight-tying 探针测出 swap=0 算力硬底 **0.8044 s/步**，据此否决「常驻块 INT8 驻留」改造，并用 `ABC_INT8_NAMES` 证明 loss 偏差只随量化字节比例缩放、不存在敏感层。
- [MOSS-Image A/B 对比与合并建议（krea2-webui vs KohakuMossTrainer）](moss_vs_kohakumoss_trainer_ab_20261019.md)：汇总两侧基线、性能/显存与功能两条主线，候选统一编号为 P-/F- 两组共 41 条。**三条纠偏**：①A 侧生产在 sm75 上实为 fp16（`resolve_mixed_precision` 在 GPU major<8 时就地改写 bf16→fp16），**不存在「bf16 环境退化」**，bf16 是策略已规避的死路；②PCIe ×8 的代价按档位分化而非统一 +52%——默认 1024 档仅 +1.2%、长字幕 0%，但 256×256 +96%、512-short +89%、768-short +67%（步时 ≈ max(总 DMA, 总计算)）；③`library/training/moss_compat.py:84-97` 的必须关闭项为 39 项，A 的 `--ema*` 是死代码。**最高风险**：fp16 数值稳定性未通过（FP16 臂 5 步全 NaN、scaler 65536→2048、第 6 步 CUDA 失败 + Xid 31 MMU Fault，根因未定位；与生产 10 步干净矛盾未解释），是能否长训练的前提。**合并顺序**：P0 = `P-C9` 误差门禁 + `P-C1` 数值治理 + 三条零风险测试（`F-E6`/`F-E5`/`F-B1`）；P1 = `P-C2` elementwise 融合 kernel（目标 −20%）、`P-C4` swap 膝点重标定、`P-C3` ×8→×16 纯诊断；P2 = `F-C2` 放开 `loss_type`、`F-A1` LoHa/LoKr/GLoRA 上 MOSS、`F-D4` counter-based 确定性采样。另含 10 GB sm75 不可行清单 12 项与 12 条未验证声明。
- [MOSS-Image 512 幅面 A/B 换卡后复测汇总（三档硬件收口）](moss_image_512_ab_remeasure_20261020.md)：同卡、同数据、同幅面下两侧端到端首次同时可测。**A 2.4181 s/步 / 8992 MiB（swap=8 为最小可行档，配置默认 20 贵 2.42×）对 B 12.0330 s/步 / 2499 MiB（全 eager）**——但这是**不对等显存预算**下的比值：A peak alloc 8992.2 MiB 对 B 2499.0 MiB（**3.598×**，A 多用 6493 MiB），故 4.98× 衡量的是「A 用满显存换来的速度」，不等于同显存下的实现效率差；档位也不对齐（B 的 `low` 不是其「刚好装下」档，`balanced` 估算 7.41 GiB 才算）。**等显存口径（§5.1，纯算术推导、两侧均未实测）**：A 显存严格线性 `alloc = 10720.3 − 216.02 × swap`，降到 2499 MiB 需 swap≈38 ⇒ 每步 17.2 GB、预测 10.98 s，对 B 的 12.033 s **≈1.10×**，优势几乎消失、两侧都撞 PCIe 带宽；swap=38 与 B `balanced` 已另开 t29/t30 补测、由 t31 定稿。两侧步时都与同一条链路的搬运字节高度一致（A 3.624 GB/步为字节模型量、B 18.646 GB/步为计数器读数，均为十进制 GB；字节比 5.15 ≈ 步时比 4.98；步均有效 H2D 1.4987 对 1.5496 GB/s 十进制，非探针读数），故 B 的 4.98× 主要是搬运字节问题——但**不是「kernel 收益上界为 0」**：按 H2D 下界 11.764 s 算，非 H2D 项只剩约 0.27 s/步（≈2.24%），属条件推算而非收益实测；已付代价是 1 张卡 + 11.2 小时（Xid 13→154）。链路 ×4→×8 使 swap 斜率 559.51→285.64 ms/块（1.9588×，对上探针 H2D 比 1.955×；探针值 1.476/1.559 实为 **GiB/s** = 1.585/1.674 GB/s，见正文 §1.4），swap=8 步时 4.6175→2.4173 s。B 全 eager 可测但 39/60 步 NaN、有效更新仅 35%，四种 FP16_FIX 无一解决（none 被配置拒、weights 改权重不可比、online 与 fp32_residual 实测失败）。**收口 2026-10-17 的 11.3386 s/步：同一张物理卡（UUID 相同）、零残差，差异全部来自 ×4 链路**，据此修正既有 A/B 汇总 §7-4 的卡归因。1024 档不外推：该档实测为计算主导（int8 把搬运减半而步时不变）。
- [MOSS-Image A/B 六轴最终对比（E7 定稿）](moss_six_axis_final_20261020.md)：把 t29/t30/t34/t35 四轮热测与 t28/t31/t32/t33 四轮审计收口成**六轴最终判决**（CPU 内存 / 拟合速度 / 有效步数 / NaN 处理 / 步时 / 显存，每轴固定「定义→两侧实测→证据路径→结论行」）。**判决**：在 CMP 50HX / sm75 / 512×512/batch1 / fp16 这一格训练选 A——同显存预算步时少 8.98%（B/A=1.0986×），三计数可用、512 档 0 NaN 跑完 112 步；B 两轮被自家 `NonFiniteGuard` 中止（86/64 attempted，候选 25/20），价值在机制与门禁而非这一格性能。**口径**：所有成本称「按各侧判据计的候选成本」（A 8/336 抽样 vs B 全量三判据，不同源），S1/S2 × seed_wise/pooled 四栏并列 10.76/10.78/9.90/9.93×，不作实现效率或收敛排名；跨侧 loss 绝对值与跨 scope 内存差均不给。含 **11 条结论修正**与 **14 条未测/不可比清单**、7 条合并建议（风险/工作量/是否立即）。
- [MOSS-Image「等显存预算」A/B 对比（三口径定稿）](moss_equal_vram_comparison_20261020.md)：**近等显存** 2.453/2.440 GiB 但**工作量未对齐**，全文只是吞吐对照、不能隔离实现效率（模型拆分仅为算术推导，不作因果）。**B/A 步时比 = 1.0986×**：A 步时少 8.98%（1−A/B）、A 步均吞吐高 9.86%（B/A−1）。B 的 21/60 是 `finite_loss_steps` 统计量，非完整权重验证。
- [历史 MOSS 测试文档实测回审（2026-10-20）](moss_historical_measurements_audit_20261020.md)：用 2026-10-20 实测回审 6 篇 findings + 1 篇配置文档的 8 条断言，**2 已推翻 / 5 需加注 / 1 成立**，历史正文一字未改。**推翻一**：11.3386 s/步 与今日数据是**同一张卡**（`GPU-bfda6d91`，同 UUID），3.87× 全来自链路宽度 ×4→×16——今日同卡 ×4 下得 11.3443 s（与原值差 0.06%），且幅面不同（今日 512 幅面、基线 256×256）、精度不同（今日 fp16、基线 bf16）而步时相同，正好同时排除「换卡」与「算力差异」（swap20 每步固定搬 9.06 GB，步时由搬运决定）；A/B 汇总 §7-4 的「另一块物理卡」归因作废。**推翻二**：「3.14 GB/s 就是这块卡的实际上限」不成立——链路宽度是协商结果不是卡的常量，同机实测 ×4 0.755 / ×8 1.476 / ×16 2.90 GiB/s（每档 ×1.96），且该文全文无链路标签、3.14 本身无落盘（可查值 2.796–2.908，`h2d.json`）。**需加注 5 条**：fp16 NaN 是 `GPU-46c7d0db` 单卡病理（五事件逐条落卡、fp32 臂同样 NaN），不得推广为「sm75 上 fp16 不稳」——据此 P-C1 从「最高优先阻塞」降为「待硬件样本前不阻塞」；int8 1.222× 与十二行矩阵均为 ×16+swap20 协议内数，不与 ×8 并排；preflight 与 README 索引给出逐行改法（本任务未落）。**成立 1 条**：面积收益次线性（固定项+线性）与「文本长度是显存主变量」被今日 swap 扫描独立复现（0.1300+0.28564·swap，四点残差 ≤0.12%），只有系数限于 ×16。P1 据此重排：`P-C3`（×8→×16）升为仅次于 P0 的性能首项，`P-C4` 视为已完成一轮（最小可行 swap=8、20 贵 2.42×）。
- [MOSS-Image A/B 块交换标度律：216 MiB/块 · 0.4530 GB/块·步](moss_block_swap_law_20261021.md)：把 A 的 `blocks_to_swap` 与 B 的 `STREAM_RESIDENT` 放在同一坐标上，**两侧显存律斜率逐字节一致**——A 216.012695 MiB/块（四段差分全同）、B 216.000000（四点全同），每块每步流量同为 2×216 MiB = 0.4530 GB（A 0.453010 / B 0.452985）。**全部差异落在截距**：非块常驻足迹 A 1647.7368 MiB 对 B 2660.4000 MiB，差 **1012.66 MiB ≈ 4.69 个块**——这解释了「同显存下 B 必须多流 5 个块」（A swap=8 流 8 块 / B ring=29 流 13 块，alloc 相差仅 67.8 MiB）。步时双侧线性拟合 R² 均为 0.9999992（A `t = 0.138346 + bytes/1.591300`，5 点；B `t = 1.354351 + bytes/1.771613`，4 点）。**修正既有文档两处**：①`moss_equal_vram_comparison_20261020.md` §6.1 的 `1.5724 GB/s` 是 `reference_gbs` 外部假设值而非 B 的反推带宽，故「A 等效带宽比 B 高 1.2%」不成立——B 四点实测边际带宽 **高 11.33%**，方向相反；②B 已不是「单点无法拟合」，§6.2(b) 同字节归因由「固定项差 +0.3699（80.6%）/带宽差 +0.0889」改为「**+1.2160（+262.7%）/ −0.7533（−162.7%）**」，即 B 固定开销更大但搬运更高效、两项部分抵消。含等显存差距顺序分解（多流 5 块 +1.4232 s / 带宽 −0.3766 s / 固定项 +1.2160 s = 2.2625 s 对实测 2.2631 s）与 10 条未测边界（逐档主机内存两侧均无数据）。
- [跨项目 fp16 / NaN 保护对照（2026-11-07）](fp16_nan_cross_project_comparison_20261107.md)：以 lulynx-trainer 的 `guide/fp16_nan_protection_design_2026-11-06.md` 十节口径为纲，做**代码级**双向对照（41 条差异：A 缺失 9 / B 语义一致 15 / C 冲突 3 / D 仅命名 10 / E 反向优势 4），每条带双方 `file:line` 与五级证据强度。**关键前提**：参照实现 5 条非标准路由与其 fp16 缓存量程拒收**零 pytest 覆盖**，故「抄来即用」项数为 0，移植须自建验收。**最重要的三条**：①A1 我方非有限 loss **仍执行 backward**（`loop.py:1109` 只算布尔、`:1126` 无条件反传），白跑反传且 NaN 梯度灌进累积器；②A9 `batch_step.py:101` 清洗判据只 `torch.isnan`，**±inf 不触发清洗**，与 A8 构成「fp16 TE 缓存写 Inf → 嵌入 Inf → 不清洗 → 下游 NaN」因果链；③A6 若照搬「不计入」而不做连续复位，会比参照实现**更松**，故须「区分成因 + 复位 `consecutive_discards` + `scaler_backoff` 窗口总量上限」三件套。**明确不移植**：9 路由/7 seam 框架、SafeGuard 恢复层、手动缩放纪律全量（会引入第二个 scale authority，违反 D1）、interval 扫描与地址表 kernel。另顺带发现我方文档债 F1–F5（F5 已修）。证据附录收录 captain 实测「13 文件 223 passed / 0 xfailed」的精确清单与命令，并注明 CUDA/DDP/真实训练三条证据链**不得并入冒充同一份输出**。
- [fp16 保护与 NaN 后处理：静态等效判定与待热测（2026-10-21）](fp16_nan_static_equivalence_20261021.md)：把「数学等效」拆成 E1–E5 可判定判据（L0 位级 / L1 语义 / L2 计数），24 个 CPU 用例（2026-11-07 起 24 passed / 0 xfailed，真 `GradScaler("cpu")`；E3/E5 规格已转正）与 10 条改坏验证 M1–M10；记录缺陷 D1（debug 梯度检查早于 unscale/clip，本分支未修复）、D2（默认路径无有限性检查）、D3（adaptive 把跳步当致命）及 E3 六条接缝。**2026-11-06 补记**：在 2× V100 上以生产 `_run_step` 复现 E2 跳步语义与 E5/D1，并归因「调换检查顺序修不好 D1、缺的是 scaler 所有权判定」；H1/H2/H8 合成探针通过（`bench/fp16_precision/run_cuda_probe.py`），实测 torch 2.10.0+cu129（`pyproject` 要求 ≥2.12），H3–H7 仍待热测。
- [公开训练器 DDP / TP / PP 实现调研](multigpu_trainers_ddp_tp_pp_survey_20261004.md)：sd-scripts/Musubi/OneTrainer 数据并行、diffusion-pipe 真实 DP+PP、TorchTitan LoRA TP 与 Megatron 调度；固定源码版本与全模型接入建议。
- [Anima 混合 NF4 autocast 与激活显存](anima_mixed_nf4_autocast_90hx_20261003.md)：修复 NF4 输出意外回升 FP32、LoRA 与 checkpoint 激活膨胀；保持配置和量化策略不变。
- [扩散混合量化基座与后训练线上调研](diffusion_mixed_quant_post_training_survey_20261002.md)：训练器静态混合、EfficientDM/BitsFusion量化后训练、MixDQ/ViDiT-Q敏感度搜索、Nunchaku训练边界及sm86 INT8候选。
- [INT8 训练本地与 musubi 对比](int8_training_local_vs_musubi_20261002.md)：3080 W8A8/W8A16历史证据、真实INT8前向、Triton融合与反向精度差异；原始日志缺失和数值门槛边界。
- [ConvRot采样生命周期修复与90HX复测](convrot_lifecycle_fix_90hx_20261003.md)：完成step图释放、offloader显式CPU park/restore、共享模型族兼容与真实VAE短测；不含500步长测。
- [ConvRot INT8 CMP90HX重新验证](convrot_90hx_validation_20261002.md)：同预算BF16最快，Triton前向改善但未晋级；Flash反向重复噪声、两处采样生命周期问题及500步条件跳过。
- [Anima 混合 NF4 异型换层与显存追踪](anima_mixed_nf4_swap_memory_90hx_20261002.md)：稳定打包/dense 缓冲、采样池复用和重配置回滚；真实8步对照训练峰值略降但全程峰值未降，另有两步同初始LoRA数值对照。
- [公网 AMP FP16 训练实现调研](public_amp_fp16_training_survey_20261002.md)：Diffusers Flux LoRA、Wan FP32 保护、OneTrainer 组件精度及其他训练器边界；公开源码依据，非热测或质量认证。
- [公开训练器 OOM 回退源码调研](public_oom_fallback_survey_20261004.md)：Accelerate/Trainer 缩 batch、Lightning 启动校准、扩散与 LLM 训练器 checkpoint 边界；固定源码版本，区分启动重试与中途恢复。
- [Anima 中途 OOM 恢复与独立预览交付](anima_committed_oom_preview_delivery_20261004.md)：完整 sampler/预取、optimizer/scaler/RNG 与设备布局，真实 CUDA OOM 新进程恢复对照及独立预览 CPU 回退；受限实验、默认关闭。
- [Anima 原生 FP16 启动 OOM 静态验收](anima_native_fp16_startup_oom_static_20261002.md)：冻结精度与源身份、新进程递增 swap、首次 optimizer 开始后停止；限 CPU 验证。
- [Anima 原生 FP16 正式入口](anima_native_fp16_product_entry_90hx_20261002.md)：配置、自动路由、轻量报告，以及生产保存/续训到 step5 的 scale 和真实 AdamW 验证。
- [Anima FP16/BF16 200步配对热训](anima_fp16_bf16_ab200_90hx_20261002.md)：两路各200有效更新、零跳步，100/200步共16张采样图、240次共同BF16固定验证；未观察到FP16明显劣化，单seed弱改善信号和FP16外部监控汇总缺失如实记录。
- [Krea-2 native FP16 CMP 90HX 三步 smoke](krea2_native_fp16_smoke_90hx_20261002.md)：官方非量化 plain-LoRA 配置，BF16/FP16 各 3 次有效更新、零跳步、588 个初始 LoRA hash 匹配并保存 state；非质量或长期稳定性认证。
- [Anima旧FP16集中验收](anima_native_fp16_acceptance_90hx_20261001.md)：12格finite且有效更新，独立FP64证据审核通过；严格0/12通过，梯度漂移明显。质量标准已预登记，质量未运行，结束05/04与scale扩张。
- [Anima旧FP16 AMP补充对照](anima_legacy_fp16_comparison_90hx_20261001.md)：旧路径scale1024五步有效、零跳步，2.694s/步、GPU峰3524MiB；其后严格矩阵见上项，长期质量和Turing未验收。
- [Anima Linear精度交付批次2](anima_precision_delivery_batch2_90hx_20261001.md)：两样本×三sigma×初始/非零AdamW状态，05/04及控制各12/12通过；原四组真实五步与独立CPU验收通过，但候选显存峰值更高；旧FP16补测见上项。
- [Anima Linear精度交付批次1](anima_precision_delivery_batch1_90hx_20261001.md)：有界12候选及同状态实际AdamW更新对照；10个固定案例数值通过，05/04进入保留集，理论权重减少不等于显存或速度收益。
- [Qwen Image 2.1 Edit 50 对数据热训练](qwen_edit_hot_training_20260928.md)：Qwen3-VL 低显存缓存、32px 对齐和编译缓存路径修复；CMP 90HX swap24+compile 完成 50 步并通过权重 CPU 严格重载。
- [预处理 Auto Batch 实测与消融](preprocess_auto_batch_20260906.md)：从1起步、预测上拉、OOM退避、吞吐收敛，以及真实模型/缓存验收边界。
- [AUTO 块交换真实 GPU 热测与消融](auto_block_swap_hot_20260908.md)：Krea-2 真实缓存、显存竞争、OOM 回退、安全余量和速度排序的时变边界。
- [全程动态 AUTO 块交换验收](auto_block_swap_dynamic_20260908.md)：运行时显存压力退避、释放后再探索、A/B/A 决策、NF4 存储复用和策略消融。
- [Anima / Z-Image AUTO 块交换热测](auto_block_swap_families_20260909.md)：Anima 真实 startup AUTO 通过；Z-Image 因系统换页保护拒绝。
- [FP16/FP32 与 OOM 重试静态契约论证](adaptive_precision_oom_static_proof_20260924.md)：精度请求与重试关联的不变量证明；不含热测。
- [动态 OOM 上限与递增重试的静态数学论证](adaptive_oom_bounds_math_20260930.md)：结构/用户/host 上限归纳、指数尝试界、CPU 预算模型及不完备搜索反例；仅 CPU 验证。
- [Anima 自适应精度首阶段 CMP 90HX 探针](anima_adaptive_precision_90hx_20260930.md)：disposable capture/replay 与首轮混合精度候选；FP16/BF16 梯度偏差 fail-closed，仅全 FP32 线性控制组通过。
- [Anima FP16 梯度误差定位](anima_adaptive_precision_diagnostics_90hx_20260930.md)：Luna 分支与 scale 热测；三个分支均影响上游梯度，scale 有帮助但六个候选仍未达标。
- [Anima 局部 VJP 与 cotangent 定位](anima_local_vjp_90hx_20260930.md)：固定 sigma/初始状态、8 层 base-output cotangent；局部 cast 误差与整网轨迹偏移分离，MLP2/GELU 尚待验证。
- [Anima MLP2/GELU 与投影拆分](anima_mlp_path_90hx_20260930.md)：真实 GELU 两侧 cotangent、MLP1/MLP2 单独 FP16 干预；两种拆分均不达标，控制通过，实际误差非单调。
- [Anima 同前向 FP32 base-dx 反事实](anima_backward_counterfactual_90hx_20260930.md)：两种 FP16 prediction 逐位一致，FP32 base-dx 未改善大梯度漂移；仅用于收窄因果定位。
- [Anima 残差流边界 cotangent](anima_block_boundaries_90hx_20260930.md)：29个采样边界与FP32重放控制；反向首次超2%在boundary14，定位范围不等于因果归因。
- [Anima block14 完整形状受控VJP](anima_block14_vjp_90hx_20260930.md)：八组真实4200-token对照及完整梯度独立复算；固定其他因素换cotangent的差异最大，仍非根因或生产认证。
- [Anima 上游15/17/27完整VJP](anima_upstream_vjp_90hx_20260930.md)：两次多块捕获、24组完整矩阵；block17的激活变化与cotangent对照影响接近，下一步拆查其内部路径。
- [Anima block17 内部残差阶段VJP](anima_block17_paths_90hx_20261001.md)：复用捕获的分析热测与CPU独立复算；固定参考cotangent时自注意力阶段对输入轨迹的对照最大，尚未定位具体算子。
- [Anima 自注意力归一化与投影/核心拆分](anima_attention_split_90hx_20261001.md)：两批分析热测与独立CPU复算；归一化输入对照较小、core传回K梯度变化最大，仍未区分QK-norm/RoPE/attention。
- [Anima 隔离QK-norm与attention cotangent](anima_qknorm_90hx_20261001.md)：隔离dp重现、四格解析验收及Luna热测完成；显存峰值1500MiB，换attention传回梯度的K对照大于固定梯度换norm输入，尚非根因或生产认证。
- [Anima post-RoPE Q/K/V八格对照](anima_qkv_matrix_90hx_20261001.md)：固定du真实激活替换、端点/重放与CPU统计审计通过；K输入对照最大，仍不等于误差源头或安全层名单。
- [Anima K方向插值与V背景](anima_k_interpolation_90hx_20261001.md)：固定Q/du的五点×两V背景；采样梯度响应非线性，独立CPU审计通过，外层热测超时退出124如实保留。
- [Anima K插值梯度head/token分布](anima_k_distribution_90hx_20261001.md)：已有梯度的CPU定位，端点误差约99%集中于top1% tokens；独立统计复算通过，尚非误差源头或精度放行。
- [Anima K激活漂移与梯度分布重合](anima_k_activation_distribution_90hx_20261001.md)：CPU真实norm/RoPE重建及独立NumPy核验；最大token同为3343，激活top42覆盖约93.1% fused梯度误差，仍需局部干预。
- [Anima 局部K替换](anima_k_local_90hx_20261001.md)：head/token真实VJP干预通过，top42距full响应残差约12%--15%，低误差对照响应小；尚非精度放行，无新梯度张量独立审计。
- [Adaptive runtime 实测与失败记录](adaptive_runtime_20260921.md)：自适应运行时实测、失败路径与能力边界。
- [Adaptive training 实测结论](adaptive_training_20260922.md)：自适应训练实测与训练运行时结论。
- [Qwen Image 2.1 CMP 90HX 热测](qwen_image_2_1_90hx_hot_test_20260926.md)：full checkpoint + swap24 + per-block compile 的 60-step 单图短测和硬件遥测。
- [Qwen Image 2.1 Flash CMP 90HX 热测](qwen_image_2_1_flash_90hx_20260927.md)：左填充 mask 正确性、Flash/torch A/B/A 与 60-step 复测、缓存候选未采纳；24/32-head 探针差异、GPU wait 未测及 PCIe 传输预算修正。
- [Qwen Image 2.1 INT8 packed block-swap CMP 90HX 热测](qwen_image_2_1_int8_swap_90hx_20260927.md)：INT8 packed swap 与 BF16 各 60-step 单轮对照，步时约快 3.7%；因顺序运行和温频差异，仍需交替复测及质量验收。
- [Qwen Image 2.1 阶段 3 开发验收](qwen_image_2_1_stage3_acceptance_20260928.md)：MLP/QKV 融合与投影保留已集成，独立数值与完整模型短测通过，确定性诊断下恢复首步梯度逐位一致，性能未验收、默认关闭。
- [Qwen Image 2.1 LoKr/LoHa 90HX 热测](qwen_image_2_1_adapter_variants_90hx_20260929.md)：LoKr 编译预算修复、两步 smoke 与 20 步热测、plain LoRA 对照及后端支持边界。
- [Qwen Image 2.1 NF4 eager Edit smoke](qwen_nf4_edit_eager_20261005.md)：50 对真实 Edit 数据在 90HX 上完成全量 NF4 eager 单步、缓存重建、LoRA 更新与 checkpoint 保存。
- [Qwen Image 2.1 NF4 compile 数值定位](qwen_nf4_compile_90hx_20261005.md)：strict/native norm+SiLU 联合策略与两个真实 bucket 独立逐位一致；默认关闭的受限入口通过真实两步 AdamW、预览与完整状态续训短测，记录 Rich 日志消费 Python RNG 的边界，普通 NF4 compile gate 保留。
- [Qwen Image 2.1 独立 T2I/Edit 推理](qwen_standalone_inference_20261005.md)：真实 NF4 + plain LoRA 的 T2I、单/四参考 Edit 出图，自动 TE offload、严格加载和专属 latent 再解码像素一致。
- [Qwen Image 2.1 编辑采样多参考图热测](qwen_edit_sampling_hot_20260928.md)：CMP 90HX 10 GiB 上单/双/四参考图与 CFG 编辑采样、采样后恢复训练和低显存 Qwen3-VL 缓存验收。
- [Qwen 编辑采样块交换初始化 A/B](qwen_sampling_prepare_ab_90hx_20261002.md)：去掉逐预测重复 prepare；90HX 两张 28 步样张耗时约减少 45.4%，A/B/A 复跑、数值一致性、异常恢复与梯度更新验收。
- [Qwen 编辑采样 DiT K/V A/B/C](qwen_sampling_kv_abc_90hx_20261002.md)：文本/参考图前缀缓存已接入；90HX 对 B 仅约0.17%耗时差且增加显存，BF16数值门槛未通过，训练预览保留 B 默认。
- [多模型采样生命周期验收](sampling_batch_lifecycle_20261002.md)：三族训练预览去除每 prompt 初始化并覆盖入口异常恢复；Anima tiled 使用一次 inference scope；CPU/Gloo 和小型真实 CUDA/slab 验收通过，完整模型性能未测。
- [Qwen 编辑 LoRA 1024 双图采样](qwen_edit_1024_dual_sampling_90hx_20261003.md)：step400、B默认、90HX两张1024方图各28步；记录去噪/解码、编译计数、显存与每秒硬件遥测，限单轮实测。

| 文档 | 说明 |
| --- | --- |
| [backend_multi_model_audit_20260810.md](backend_multi_model_audit_20260810.md) | 2026-08-10 后端多模型兼容审计：Anima/Krea-2 主链、P1 风险、测试缺口与 registry 路线 |
| [runtime_support_matrix_20260704.md](runtime_support_matrix_20260704.md) | compile / checkpoint / block swap 组合矩阵审计 |
| [v100_flash_attention_support.md](v100_flash_attention_support.md) | V100 FlashAttention 目标仓库对照、移植状态与生产边界 |
| [adapter_registry_capabilities_audit_20260704.md](adapter_registry_capabilities_audit_20260704.md) | Adapter registry、merge、推理加载和续训能力边界审计 |
| [anima_int8_base_linear_audit.md](anima_int8_base_linear_audit.md) | Anima int8 base linear 审计（存储/传输 int8；非 ConvRot） |
| [../experimental/convrot_int8_training.md](../experimental/convrot_int8_training.md) | ConvRot int8 训练探索（W8A16/W8A8）；与上条区分 |

## 性能、显存和训练报告

| 文档 | 说明 |
| --- | --- |
| [anima_1mp_speed_50hx_20261022.md](anima_1mp_speed_50hx_20261022.md) | Anima 1MP（1024×1024）/ bs1 / 完整检查点在 CMP 50HX 的速度基线：swap=0 两轮复测 **2.363 / 2.400 s/步**（差 1.6%，SM 1850~1905 MHz、显存峰值 6074 MiB、49% 利用率、≈56 W），swap=4 观测 21.07 s/步但夹着外部算力解锁动作，不算干净 A/B；附 fp16 tensor 85 TFLOPS（理论 99.38）的算力锁判定、gen1×8 链路与 120 W 功耗墙 |
| [anima_1mp_speed_v100_20261025.md](anima_1mp_speed_v100_20261025.md) | 与 50HX 同口径（1MP/bs1/完整检查点/20 步）的 V100 跨卡对照：swap=0 **1.961 s/步**（复现 1.972，差 0.56%），比 50HX 最好档快 16%、默认档快 22%；两卡 MFU 近同（28.4% vs 27.9%）⇒ 差异来自带宽/时间占比而非算力；块交换在 V100（Gen3×16）只多 2%（50HX 同配置 21.07 s/步）；fp16 96.39 TFLOPS 无算力锁；300 W + SM 1530 MHz 锁定步时仅 +2.8% 但功耗 −22%、J/图 −20%、峰值温度 65→56 ℃，解除时钟锁只买回 0.8% ⇒ 瓶颈是功耗墙而非锁 |
| [anima_dual_gpu_parallel_probe_20260904.md](anima_dual_gpu_parallel_probe_20260904.md) | Anima 异构双卡 PP2、TP2、TP2 INT8 通信的 BS=1 性能、显存、数值与同参数图片对比 |
| [multigpu_v100_nvlink_baseline_20261107.md](multigpu_v100_nvlink_baseline_20261107.md) | 2×V100-SXM2-16GB NVLink 的 NCCL 集合通信/P2P/H2D 实测与 FP16/BF16 算力对照；BF16 在 SM70 上为仿真路径 |
| [lulynx_multigpu_comparison_20261107.md](lulynx_multigpu_comparison_20261107.md) | Lulynx 多卡设计与本项目的源码对照：机制差异与正确性条件、P0-0 窗口/scaler 一致性优先项、控制面 P0 清单、FSDP2/TP 实验候选、基线口径纠偏与许可证边界 |
| [multigpu_reference_validation_20261107.md](multigpu_reference_validation_20261107.md) | **多卡参考点集成验收**：P0-0..P0-5 / P1-1..P1-3 / EXP-FSDP2-BASE / EXP-TP-COLROW 的源码位置、验证层级、命令与退出码、逐点裁定；RUN-A/B/C = 50 次有效更新 + 判别实验 20 次的真实账本；v1 偏差逐项现状、4096 单 shape 候选为何未进 L3。**§12–§17 增量**：取数顺序三层缺陷（`SeedableRandomSampler` 的 `len(dataset)`、`shuffle_buckets()` 真凶、loader 硬编码 `shuffle=True`）与 `ANIMA_MULTIGPU_PINNED_ORDER` 门控及其覆盖缺口；四个 S-d/S-g 数字（1.219e-02 / 1.193e-02 / 3.434e-03 / 3.106e-03）判为**取数错位、不可解释为协议差异**，P0-5/P1-2 的 V-REAL 数值面由 failed 改记 **unverified** 并留 t35 结论占位；F-T31-3（合同「关 shuffle」只能靠 env 门控、默认 shuffle 下跨 world_size 不等价）；F-T33-2（verify 全绿却漏掉阻断回归的盲区与新规）；captain 侧流程缺陷账；F-T3-1 旧结论降级为「分片算术已验证、样本级对齐未验证」。**§18–§23 收口（2026-11-08）**：按契约修订 v2（C1–C4）裁定 **P0-5/P1-2 的 V-REAL 数值面通过**，附旧数字作废清单与替代数字、合同阈值来源与失效教训（`≤1e-6` 只出现在契约一处、无推导无实测）、`pinned_order` 正名为产品特性「可复现的确定性数据顺序」、captain 判决失误清单 |
| [z_image_170hx_100step_20260905.md](z_image_170hx_100step_20260905.md) | Z-Image 在 CMP 170HX 上的 100-step BF16 Flash varlen + full checkpoint + swap8 + fused AdamW 真机测试：运行时 PASS，单图过拟合在 step 100 出现明显曝光和细节退化 |
| [lycoris_4_fused_kernel_audit_20260902.md](lycoris_4_fused_kernel_audit_20260902.md) | LyCORIS 4.0.0 Triton/TileLang fused kernel 发布、性能、精度与 release 宣传审计，以及对本项目 LoKr/LoHa 的借鉴优先级 |
| [lokr_fused_backward_stage1_20260902.md](lokr_fused_backward_stage1_20260902.md) | 基于 LyCORIS 审计方向独立实现 LoKr `grad_w1` Triton reduction；RTX 3080 组件验证与 CMP 170HX 3-seed x 50-step 端到端热测、Nsight、resume/compile/swap gate |
| [krea2_adapter_variants_170hx_300step_20260904.md](krea2_adapter_variants_170hx_300step_20260904.md) | Krea-2 在 CMP 170HX 上的 adapter 变体 smoke 与部分 300-step 训练：Flash/checkpoint/swap/compile/fused 组合、显存、loss、checkpoint 身份和固定 prompt 预览对比 |
| [krea2_nf4_adapter_repair_20260905.md](krea2_nf4_adapter_repair_20260905.md) | DoRA / OrthoLoRA / ReFT 的 NF4 数值、checkpoint/compile、保存重载与 CMP 170HX 短训修复验证 |
| [convrot_longrun_bf16_w8a8_w8a16_20260727.md](convrot_longrun_bf16_w8a8_w8a16_20260727.md) | RTX 3080 上 BF16/W8A8/W8A16 三组 1710-step 长训审计（速度、显存、loss、样图及最终保存回归修复） |
| [loha_hot_test_20260725.md](loha_hot_test_20260725.md) | LoHa 在 RTX 3080 10GB 上的 12-step 热测与检查点验证 |
| [training_profiling_hot_test_20260629.md](training_profiling_hot_test_20260629.md) | 训练 profiling 热测记录 |
| [anima_lokr_blockswap_oom_report.md](anima_lokr_blockswap_oom_report.md) | LoKr 16G block swap OOM 报告 |
| [lokr_anima_shaojianV1_run_report.md](lokr_anima_shaojianV1_run_report.md) | lokr-anima-shaojianV1 运行报告 |
| [anima_fp8_blockswap_transfer_report.md](anima_fp8_blockswap_transfer_report.md) | FP8 block swap transfer 最终报告 |
| [blockswap_baseline_20260806.md](blockswap_baseline_20260806.md) | 块交换优化基线测量（计算 vs 传输，RTX 3080 / CMP 90HX，标准参考） |
| [krea2_90hx_nsight_tiles.md](krea2_90hx_nsight_tiles.md) | Krea-2 CMP 90HX Nsight tile/cache 验证：尾 tile 主因 REJECT、CUTLASS/cuDNN/FA2 实际 tile、NCU 权限边界 |
| [krea2_nf4_ablation_findings.md](krea2_nf4_ablation_findings.md) | Krea-2 NF4 × {完整检查点, 块交换} 消融矩阵（5 格六维指标：显存/内存/速度/loss/数学实现/数学偏移，PG199 1024×1024 30 步） |
| [krea2_nf4_self_contained.md](krea2_nf4_self_contained.md) | Krea-2 自包含 NF4 v2：无重新量化构建、版本化严格加载、v1 兼容与 PG199/RTX 3080 训练验证 |
| [krea2_nf4_correction_pg199.md](krea2_nf4_correction_pg199.md) | Krea-2 NF4 激活加权 rank-16 回补实测：层级误差下降但端到端收益不足，暂不生产化 |
| [krea2_nf4_downstream_correction_90hx.md](krea2_nf4_downstream_correction_90hx.md) | Krea-2 NF4 下游 velocity 目标低秩回补的 90HX 校准与 held-out 验证 |
| [krea2_nf4_correction_rollout_3080.md](krea2_nf4_correction_rollout_3080.md) | Krea-2 NF4 rank-16 回补的 RTX 3080 完整 rollout 否定结果 |
| [krea2_nf4_fp16_grad_underflow_20261108.md](krea2_nf4_fp16_grad_underflow_20261108.md) | Krea-2 NF4 fp16 compute 的 35.65% 梯度偏差溯源：真实逐层 dY rms 6.5e-9~1.7e-7 全在 fp16 下溢区、loss-scale 严格逐 k 减半、归约开关仅二阶、多步 LoRA 轨迹门槛 FAIL → 维持 BF16 契约 |
| [krea2_nf4_h2d_bottleneck_findings.md](krea2_nf4_h2d_bottleneck_findings.md) | Krea-2 方向 B 前置诊断：NF4+swap H2D 搬运占比双口径实测 0% + 理论上界 2% → NOT_WORTH 归档（含口径缺陷诚实记录 + 与消融矩阵矛盾核验） |
| [krea2_3080_speed_stage_summary.md](krea2_3080_speed_stage_summary.md) | Krea-2 3080/PG199 速度优化 12 阶段汇总：瓶颈定位（硬件大矩阵 3.6x）、compile `-19.1%`、选择性 checkpoint 仅 PG199 可行、FP16/Inductor mode/降 rank 三项否决、mm+attention 占墙钟 89% 的天花板、Flash varlen 显式 opt-in；逐阶段原文已归档 |
| [krea2_3080_speed_final.md](krea2_3080_speed_final.md) | Krea-2 RTX 3080 速度研究最终审计：根因、生产建议、可选后端、否决路径与证据边界总表 |
| [krea2_3080_speed_comparison_extended.md](krea2_3080_speed_comparison_extended.md) | Krea-2 PG199/RTX 3080 扩展速度矩阵：step、it/min、显存、冷/热稳态、swap、checkpoint、compile 与 Flash 对比 |
| [anima_perband_dynamic_seq_20260830.md](anima_perband_dynamic_seq_20260830.md) | Anima per-band dynamic-seq 移植、64GB CMP 170HX 60-step union/per-band A/B 与默认关闭结论 |
| [anima_pp_microbatch_scan_20261109.md](anima_pp_microbatch_scan_20261109.md) | Anima PP v0 在 2×V100 上的 microbatch 扫描（M=4/8/16/32）：每样本 1.788→1.641→1.543→1.519 s、加速比 1.520×→1.656×→1.761×→1.789×、峰值显存 5.54→5.69→5.98→6.57 GB。**拐点在 M=16 与 M=32 之间**，判据是边际性价比（1.01 → 0.33 → 0.041 s/GB）而非绝对速度；M=16 相对 M=4 每样本快 13.7% 只多 0.44 GB。达成率稳定在 1F1B 天花板 `M/(M+1)×2` 的 92–95%，说明实现未随 M 劣化；**显存不是限制**（M=32 仅 6.57 GB），限制是收益递减。含一个反直觉观察：M 从 4 到 8 显存只涨 0.145 GB，因为 gradient checkpointing + 1F1B 让同时存活的激活数有界 |
| [nsys_ncu_pytorch_cupti_20261109.md](nsys_ncu_pytorch_cupti_20261109.md) | **nsys 在 PyTorch 上拿不到 CUPTI 活动表的根因与修复**：`libtorch_cuda.so` 硬依赖 `libcupti.so.12`、经 `DT_RUNPATH` 解析到 venv pip 包，CUPTI 是进程级单例且在 `import torch` 时即被占用；因 `DT_RUNPATH` 优先级低于 `LD_LIBRARY_PATH`，用 shim 指向 nsys 自带 CUPTI 12.9 即可共用同一份。含每 rank 单独起 nsys + `--capture-range=cudaProfilerApi` 的多卡配方、6 个踩坑记录（包 torchrun 只抓父进程、`--trace=nccl` 非法、含空格路径、`demangledName` 是整数 id 需 join `StringIds`）、区间并集与「GPU 忙须减去 NCCL」两个分析口径，以及 2 rank PP 的空泡实测（rank1 真空闲 15.4% 集中为步末 1.027 s 排空，rank0 同期在 NCCL 内阻塞 1.049 s，两者是同一空泡两端）。**本地设施路径非仓库依赖** |

## 方法和研究结论

| 文档 | 说明 |
| --- | --- |
| [selfflow.md](selfflow.md) | Self-Flow rep-loss 在冻结 Anima backbone 上的否定结果 |
| [mod_guidance_quality_tag_axis.md](mod_guidance_quality_tag_axis.md) | Mod-guidance quality tag 轴分析 |
| [channel_stats_content_independence.md](channel_stats_content_independence.md) | channel stats 与 content independence 分析 |
| [asymflow_parameterization.md](asymflow_parameterization.md) | Anima velocity / sigma 参数化记录 |
| [l2p_pixel_transfer.md](l2p_pixel_transfer.md) | L2P pixel transfer 调研 |
| [krea2_raw_migration_stage0_findings.md](krea2_raw_migration_stage0_findings.md) | Krea-2-Raw 迁移阶段 0：R1/R2/R4/R8 定论 + VAE 互逆基准 + DiT key 清单 |
| [krea2_raw_migration_stage1_findings.md](krea2_raw_migration_stage1_findings.md) | Krea-2-Raw 迁移阶段 1：Qwen3-VL 文本链路 + 12 层 MFA + R1 padding 契约 (mask 屏蔽非 zero-sink) |
| [krea2_raw_migration_stage2_findings.md](krea2_raw_migration_stage2_findings.md) | Krea-2-Raw 迁移阶段 2：DiT 本体移植 + 加载器 + 单 latent forward 基准 |
| [krea2_raw_migration_stage3_findings.md](krea2_raw_migration_stage3_findings.md) | Krea-2-Raw 迁移阶段 3：LoRA 注入点 spec + family-aware target + attach+forward 真火测试 |
| [krea2_raw_migration_stage4_findings.md](krea2_raw_migration_stage4_findings.md) | Krea-2-Raw 迁移阶段 4：训练串通 + forward_for_loss 承重接口 + 单 prompt 过拟合 loss 下降 |
| [krea2_raw_migration_stage5_findings.md](krea2_raw_migration_stage5_findings.md) | Krea-2-Raw 迁移阶段 5：推理串通 + flow-matching Euler ODE + mu shift + CFG 采样 + VAE decode 出图 |
| [krea2_raw_migration_stage6_findings.md](krea2_raw_migration_stage6_findings.md) | Krea-2-Raw 迁移阶段 6：块交换 (ModelOffloader 复用) + 检查点 save/load round-trip + 256×256 净负发现 |
| [gsq_rco_quant_research_20261026.md](gsq_rco_quant_research_20261026.md) | 静态研究记录：arXiv 2411.17525 实为 Linearity Theorem/HIGGS 而非 "GSQ-RCO"（GSQ=2604.18556、RCO=2605.00649 另属两篇，本仓零实现）、可迁移组件拆解、量化×CNS×交互项 2×2 归因框架与 CNS 归因矩阵、Q0–Q5 判定（Q0/Q3/Q5 后半不成立，Q1/Q2 部分成立，Q4/Q5 前半无实测路径）、引用禁用口径清单与最小实测计划（未执行） |

## Agent Audit 2026-06-22

| 文档 | 说明 |
| --- | --- |
| [agent_audit_20260622/00_INDEX.md](agent_audit_20260622/00_INDEX.md) | Agent audit 总索引 |
| [agent_audit_20260622/01_architecture_map.md](agent_audit_20260622/01_architecture_map.md) | 架构地图 |
| [agent_audit_20260622/02_invariants_risk_audit.md](agent_audit_20260622/02_invariants_risk_audit.md) | 不变量和风险审计 |
| [agent_audit_20260622/03_test_coverage_map.md](agent_audit_20260622/03_test_coverage_map.md) | 测试覆盖地图 |
| [agent_audit_20260622/04_webui_maintenance_ux_audit.md](agent_audit_20260622/04_webui_maintenance_ux_audit.md) | WebUI 维护和 UX 审计 |
| [agent_audit_20260622/05_config_method_matrix.md](agent_audit_20260622/05_config_method_matrix.md) | 配置和方法矩阵 |

配套截图和图表在 [assets/](assets/)。
