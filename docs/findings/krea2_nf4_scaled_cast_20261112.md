# Krea-2 NF4：真实训练单步与 scaled-cast 融合候选

状态：热测补全（2026-11-14）；候选与原实现逐位等价已证，轨迹级差异归因为基线噪声，收益约2.3%／省430MiB，是否合入 dev 由用户决定  
日期：2026-11-12  
基线：`6610f710`，分支 `codex/krea2-nf4-perf-research`  
前置：[旧 84.8% 测量口径审计](krea2_nf4_profile_audit_20261112.md)

## 真实配置基线（本轮 L3 单步，不是长期 bench）

复用 `train.py` 的 `--profile_steps 1-1`，1 步预热后采样 1 步。真实缓存单图
768×1344（4032 image tokens），batch1、rank16/alpha8、NF4 自包含权重、外层 BF16、
`nf4_dir1=true`、`krea2_attn_fp16=true`、torch attention、full checkpoint、swap20、
fused AdamW。GPU 为物理 2 V100-SXM2-16GB。样本及缓存复制到新路径，原数据只读。

| 指标 | 实测 |
| --- | ---: |
| 生产 `_run_microbatch` NVTX 窗口 | 10834.792 ms |
| kernel union | 10764.617 ms |
| kernel 忙时 / 窗口 | **99.352%** |
| kernel 实例数 | 19294 |
| memcpy union（与计算大部重叠） | 1776.241 ms |
| kernel 或 memcpy union | 10786.446 ms |
| 三类 MAGMA kernel 时间合计 | 1417.948 ms |

这不代表 SM occupancy，也不含数据读取与 step 收尾。**旧 256² 探针的 15.2% 空档
不能当作本配置的优化空间**；当前窗口的 kernel 未覆盖时间只有约 70ms。
GEMM/attention 占主要时间；本轮只选择一个局部候选：NF4 dir1 backward 的缩放+cast 链。

本地产物根：`output/nf4-perf-single-step-20261112/`。
基线文件为 `baseline-v4.nsys-rep`、`baseline-v4.sqlite`、`baseline-v4-analysis.json`、
`baseline-v4.log` 与 `baseline-v4.toml`。导出/分析均 exit 0，真实 CUPTI kernel 表非空。

## 唯一候选与 L2 检查

- 新增 `library/quantization/_scaled_cast.py`：把
  `torch.ldexp(x.float(), shift).to(target_dtype)` 的 BF16→FP16 / FP16→BF16 链融合。
- 只替换 `library/quantization/nf4.py::_Dir1MatMul4Bit.backward` 的两处转换。
- 保留原 amax/frexp/clamp、权重与 `compute_dtype=bfloat16`；不改变公开配置。
- uint16 访问 BF16 存储，避免 V100 原生 BF16 指令；FP32 缩放使用精确 2 的整数次幂
  与 `mul.rn.f32`；有限值、Inf、signed zero 要求逐位一致，NaN 只要求类别一致。
- 非连续、其他 dtype、空张量及 grad-enabled 高阶路径回退原 Torch 表达式。

L2：`scaled_cast_probe.py`，gate exit 0，GPU 2；65536 个 16-bit 位模式 × 13 档
shift × 2 个方向，**全部零差异**（NaN 按类别）。峰值 allocated **270.001 MiB**。

| 单算子，shape=4608×6144 | Torch 中位数 | 融合中位数 |
| --- | ---: | ---: |
| BF16 → scaled FP16 | 0.744448 ms | 0.232448 ms |
| FP16 → unscaled BF16 | 0.747520 ms | 0.239616 ms |

这是单算子结果，**不是整步 3 倍加速**。原始输出 `scaled-cast-probe.json` 与 `.log`。

L1：定向 CPU pytest **11 passed / 1 skipped / 12 deselected**，exit 0；
14 条 `torch.jit.script_method` 弃用警告。随后空闲 GPU 的接线及边界回归已完成：
**27 passed**、exit0、峰值 allocated **18.422 MiB**；覆盖负视图、高阶回退、尾块和
完整 NF4 backward。该 L2 仍不替代下面的多步模型数值门。

## 同条件整步 A/B（第一对完成，重复组中断）

物理 GPU1，reference-0 与 candidate-0 均完成6步并 exit0，首步不计入性能统计：

| 第一对结果 | reference | candidate |
| --- | ---: | ---: |
| 后5步中位数 | 11.079563s | 10.778868s |
| peak allocated | 6841.355MiB | 6409.354MiB |

延迟下降约 **2.71%**、峰值少 **432.000MiB**，尚不能由一对短窗口外推长期收益。
第一对末步保存的392个参数/392个梯度张量（各48168960元素）比较：

- 参数 rel-L2 **0.0121788**、cos **0.9999258**；不是逐位相同。
- 梯度 rel-L2 **0.295309**、cos **0.962066**，**不满足2%/0.999门槛**。
- 六步 loss 最大相对差 **0.0000805329（0.008053%）**；不能以loss近似掩盖梯度差异。
- 所有被比较参数和梯度有限。原始汇总：`first-pair-numerical.json`。
- 不能仅凭算子位等价，猜测多步差异一定是生产非确定性；需要参考 A/A 或同状态
  单步数值对照。当前候选**暂不采纳，不部署**。

第二候选重复组 candidate-1 在首步后收到 SIGTERM，经 train.py 的 signal handler
变成 KeyboardInterrupt，exit130；没有出现 kernel 异常，也不是600s超时。其输出保留。
新目录 candidate-1-retry 尚未开始训练：gate exit4，两张V100又各占约14.25GB。
reference-1 尚未执行，不能把不完整顺序称为已完成ABBA。GPU资源阻塞时，本轮按
未通过模型数值门收口并撤回接线；不是认定kernel误算，也不以loss接近批准候选。

### 2026-11-14 热测补全（用户腾空 GPU 后授权）

四组六步真实训练全部完成（同物理 GPU1、同 seed42、同缓存与配置；各次独立过
gpu-gate，600s 上限，均 exit0）：

| 运行 | 后5步中位数 | peak allocated |
| --- | ---: | ---: |
| reference-0 / reference-1 | 11.0796s / 11.0579s | 6841.355MiB |
| candidate-0 / candidate-1-retry | 10.7789s / 10.8519s | 6409.354 / 6415.479MiB |

汇总：reference 中位 **11.0687s** vs candidate **10.8154s** = 延迟降 **2.29%**、
吞吐升 **2.34%**；峰值显存少约 **430MiB**。早期中断的 candidate-1 只剩预热步，
不纳入统计；完整原始数据在 `abba-attribution.json`。

四组末态对照（392参数+392梯度张量，各48168960元素，全部有限）：

| 对照 | 梯度 rel-L2 | 梯度 cos | 参数 rel-L2 |
| --- | ---: | ---: | ---: |
| A/A（reference-0 ↔ reference-1） | **27.84%** | 0.9647 | 0.555% |
| B/B（candidate-0 ↔ candidate-1） | 43.12% | 0.9058 | 1.254% |
| A/B 第一对 | 29.53% | 0.9621 | 1.218% |
| A/B 第二对 | 39.35% | 0.9230 | 0.560% |

**归因结论**：A/A 基线自身的跨进程发散（27.84%）与 A/B 差异（29.53/39.35%）
同一量级，四组互相重叠——之前"候选引入29.5%差异"的判定不成立，轨迹级验收
对任何实现（含基线本身）都不满足 2%/0.999 门槛。按独立审查预设规则转入同状态
对照。

**影子逐位校验（同状态对照的落地）**：`shadow_verify.py` 包住真实训练中的
`scaled_cast`，每次调用同时算候选与原实现并逐位比对。2 步真实训练：
**904/904 次调用 bit_equal，0 失配，0 回退**，实际观测 shift 共26档
（[-25,25]；全域 [-126,126] 已由算子级 65536 位模式覆盖）。即候选在真实生产
输入上与原实现**逐位等价**；末态差异只能来自生产路径固有的跨进程/原子归约
非确定性（A/A 已证），不是候选改变数学。原始数据：`shadow-result.json`。

**结论**：数值证据链闭合——算子全域逐位等价 + 真实权重层级等价 + 真实输入
影子逐位等价；A/A 证明轨迹级差异是基线噪声。候选收益：延迟 -2.29%、吞吐
+2.34%、显存 -430MiB（V100、NF4 双开、768×1344 单图六步口径，非长期 bench）。

### 2026-11-14 nsys 复测（应用户要求旧数字作废、全部重测）

同日、同方法（1 预热 + 1 采样步、capture-range=cudaProfilerApi、只读分析）重采。
**reference-v6（原实现路径，经 production_probe reference 臂）exit0**：

| 指标 | 11-12 基线（作废留存） | 11-14 reference-v6 |
| --- | ---: | ---: |
| NVTX 步窗口 | 10834.792 ms | 11021.361 ms |
| kernel union 忙时 | 10764.617 ms | 10955.325 ms |
| **kernel 忙时占比** | 99.352% | **99.400%** |
| kernel 数 | 19294 | 19294 |

两次独立采集的忙时占比一致（±0.05 个百分点），口径可复现；11-14 数字为当前参照。
热点构成（v6）：cutlass f16 GEMM 2040.9+1422.6ms、fmha backward 1767.4ms、
fmha forward 1128.5ms、magma_sgemmEx 746.1ms——大矩阵与 attention 占绝对主导。

**候选剖面未能直接采集**：nsys 2025.1.3 CUPTI 附加 + 候选 Triton kernel 的组合
在本机稳定挂起（首步内 0 GPU/近 0 CPU，`futex_do_wait`），四次尝试
（v5/v5b/v5c/v7）同签名；同期无 nsys 的候选运行已成功 5 次（ABBA×4 + 影子×1），
reference 剖面正常完成——变量锁定为候选 kernel × CUPTI。挂起与清理记录保留在
`candidate-v5*.log`、`candidate-v7*.log`。因此**候选版本的忙时占比与 kernel 数
只有 L0 投影**：融合把 528 次 3-kernel 链各减为 1，预计 kernel 数 ≈19294-1056，
忙时占比仍 ~99.4%（饱和型），步窗口按实测步时缩短约 2.3%；此为投影，
不得当作测量结论。

### 2026-11-14 合并 dev（用户批准）

本验证链已 cherry-pick 进本地 `dev`（`4f8ef803`/`20813e42`/`ed16cca5`，基于
origin/dev=6610f710，领先 3 提交）。dev checkout 复跑接线回归：
**27 passed、exit0、峰值 allocated 18.551 MiB**（物理 GPU2，gate exit0）。
未推送；推送与主树接入另行由用户决定。

### 当前状态与保留资产

- 研究分支 `codex/krea2-nf4-perf-research` 工作树当前**已重新应用候选补丁**
  （`library/quantization/nf4.py` + `library/quantization/_scaled_cast.py` +
  `tests/test_nf4_scaled_cast.py`）；dev/main 与原工作树未触碰。
- [候选补丁](../../bench/krea2_nf4_perf/scaled_cast_candidate.patch) 保留，
  可在任意实验树 `git apply` 复现；production_probe 在补丁未应用时拒绝 candidate
  臂，防止误测成相同实现。
- 影子校验入口：[shadow_verify.py](../../bench/krea2_nf4_perf/shadow_verify.py)；
  ABBA 归因入口：[compare_abba.py](../../bench/krea2_nf4_perf/compare_abba.py)。
- 是否把候选合入 dev（改动 `library/quantization/nf4.py` 一处接线 + 新增私有
  helper 与测试）属用户决定；未推送、未合入、未部署到原工作树。

## 失败、修正与边界

1. 首次采样 exit 1：测试 dataset 含不支持的 `shuffle_caption`；同时有空 log_with、
   未知 checkpointing_steps 警告。已用当前 sanitizer 校验新配置，旧日志保留。
2. 60s 与获准120s的两次采样均 exit137，无报告。启动仍为未加载的约26GB BF16基座
   计算完整 SHA256。测试配置改为指向实际自包含 NF4 后，实际加载权重不变，采样成功。
   不把启动哈希问题当作训练 step 瓶颈，也没有修改生产 metadata 契约。
3. 原路径 `library/training/cli.py` 不存在；实际入口为 `cli_entry.py`，按 traceback 定位。
4. 独立子代理直连调用因空 provider/显式路由校验被拒，未启动；改用继承式子代理成功。
   独立审查发现 negative-bit view 会在 dtype view 时报错；已让 value.is_neg() 或
   shift.is_neg() 回退 Torch，并新增负视图/1025元素尾块/storage_offset=1 用例。
   复核 **PASS（L0，限内部调用契约与当前依赖环境）**，没有剩余具体代码阻断；
   新增 GPU 用例随后纳入27项回归并通过。当前环境明确依赖 Triton，不宣称缺 Triton
   时仍能回退。独立复核第一对模型结果确认：当前不能批准生产接线，应先做参考A/A
   与同状态对照，不能只凭loss差小或算子位等价归因。
5. 旧 GPU 回归命令在 gate exit3 时提前退出，pytest 没运行；随后整模型 gate exit4。
6. 第一对模型级参数/梯度比较已完成且由独立审查以另一种FP64归约复现，未发现
   错键、shape错配或分母错误。后续热测补全：A/A 证明轨迹级门槛对基线本身也不
   成立，影子校验 904/904 逐位等价（见上文）。长期热态、训练质量、其他GPU/软件
   版本未验证；未推送、未合入 dev、未部署到原工作树。
7. 2026-11-12 重复组 candidate-1 曾在预热步后收到 SIGTERM（train.py signal
   handler 转 KeyboardInterrupt，exit130），原因未定位；2026-11-14 同配置补跑
   candidate-1-retry 全程正常（exit0），未复现中断。

## 复现入口

- [算子数值及 A/B 探针](../../bench/krea2_nf4_perf/scaled_cast_probe.py)
- [生产计时/数值采集包装](../../bench/krea2_nf4_perf/production_probe.py)
- [影子逐位校验](../../bench/krea2_nf4_perf/shadow_verify.py)
- [ABBA 归因汇总](../../bench/krea2_nf4_perf/compare_abba.py)
- [nsys 只读分析](../../bench/krea2_nf4_perf/analyze_nsys.py)
- [候选源码及定向测试补丁](../../bench/krea2_nf4_perf/scaled_cast_candidate.patch)
