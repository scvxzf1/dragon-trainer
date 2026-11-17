# ConvRot：训练采样内存生命周期修复与90HX复测

状态：两处生产修复、CPU/CUDA回归、9/9真实模型采样矩阵和9/9确定性replay对照均已完成。用户明确排除500步长测；本轮不扩展精度候选筛选或质量认证。

## 修复内容

1. `library/training/loop.py::_run_step` 在反向、optimizer、scheduler、zero_grad和`after_microstep`回调结束后返回`loss.detach()`。回调仍得到原loss，外层日志保留数值，采样期间不再保留已完成step的loss计算图。不修改ConvRot autograd ctx，也不无条件清空其他方法stash。
2. `library/runtime/block_swap_lifecycle.py` 实现offloader-owned CPU park/restore。等待所有futures与CUDA传输，解绑mixed slots，将冻结载荷重绑CPU master（包含NF4 quant_state），释放GPU slab/mixed缓存；保持managed payload placement保护、CPU master与可训练Parameter身份。恢复重建canonical resident/swapped布局，保留换层数和forward_only状态；失败重新park并传播错误，park状态禁止误forward/reconfigure。
3. 共享live preview decode在加载VAE前park、finally恢复；结束/中断deferred cleanup复用同一入口，删除session末尾重复的普通CPU迁移/decode。block集合解析覆盖Anima/Krea `blocks`、Z-Image `layers`和Qwen `transformer_blocks`，歧义或数量不匹配明确失败。正常VAE OOM仍可留latent稍后解码，训练布局恢复失败不会吞掉继续训练。

修复主体放在独立生命周期模块；现有大文件只加入detach或API调用编排。没有改训练精度默认、量化规则、功耗/风扇/时钟，也没有删除旧实验结果。

## 独立审阅及定向验证

独立审阅发现并修复两项问题：共享入口最初写死`dit.blocks`；新增GPU测试最初直接调用子层闭包，绕过block backward hook。原GPU测试在第一次park前失败，正确调用`block(...)`/`checkpoint(block,...)`后恢复对照通过。独立审阅者以临时正确harness验证4/4 CUDA组合通过；主代理完成对应源码测试修正，最终测试28通过，其中4个真实CUDA组合。跨族CPU测试覆盖三种block集合。

CPU关联集合90通过/14 CUDA跳过；另一组loop/master/mixed/tiled回归38通过/12 CUDA跳过。deferred cleanup接入后追加CPU回归72通过/4 CUDA跳过。新生命周期及验证模块Ruff通过、Pyright无错误。独立审阅后修正由主代理复核与测试；会话上限导致没有新的第二轮独立审阅，不标称全量独立最终验收。

证据目录（本地未入库产物）：`output/convrot-lifecycle-fix-20261003/summary.md`。replay工具先后修正了标量step hash和预览分支缺少BF16 autocast的问题，失败日志均保留；它们未在生产用例中出现。初次生产用例错误地设置`save_every_n_steps=0`，触发modulo-by-zero；保留失败目录，retry1用合法保存间隔。因磁盘空间有限，后续未启动用例只保存最终adapter/optimizer state，仍有第4/8步采样，没有删除任何已产生结果。

## 真实模型复测协议

指定CMP90HX UUID `GPU-806debd9-d254-da57-d515-ccae15b4ac6a`。沿用Anima base-v1.0、现有60张1024数据/cache、plain FP32 LoRA rank/alpha32、BF16 compute、AdamW LR2e-5、Flash/full checkpoint、compile off、foreach BF16 transfer。BF16/W8A16/W8A8 × swap0/12/26，各新进程8有效更新，第4/8步Euler8采样一张1024图。

`bench/convrot_90hx/production_lifecycle.py`包裹真实train CLI，记录完成step图释放、offloader park、模型CPU迁移、VAE加载/解码和训练恢复。诊断同步不是吞吐benchmark；不把日志的recent秒/步用于新的速度排名。全流程峰值与旧生产基线比较，不把两个局部释放量简单相加。

| 模式 / swap | 旧全流程 allocated GiB | 修复后 allocated GiB | 修复后 reserved GiB | 实际decode前 allocated GiB |
|---|---:|---:|---:|---:|
| bf16 / 0 | 5.5655 | 5.6114 | 5.8535 | 0.3306 |
| bf16 / 12 | 4.6998 | 4.7014 | 5.3125 | 0.3306 |
| bf16 / 26 | 4.6998 | 4.7014 | 5.3145 | 0.3306 |
| w8a16 / 0 | 6.2531 | 4.7015 | 5.3340 | 0.3318 |
| w8a16 / 12 | 5.6791 | 4.7007 | 5.3164 | 0.3306 |
| w8a16 / 26 | 4.8581 | 4.7011 | 5.3203 | 0.3306 |
| w8a8 / 0 | 未测 | 4.8650 | 5.3281 | 0.3318 |
| w8a8 / 12 | 5.6791 | 4.6995 | 5.3164 | 0.3306 |
| w8a8 / 26 | 未测 | 4.7011 | 5.3203 | 0.3306 |

9/9用例均有8次已释放loss图的完成step、两张1024PNG、第4/8步实际VAE设备加载和恢复、392个optimizer states step8、560个导出weight tensors且最终/optimizer张量有限。采样前managed GPU payload均为0；第5步确认采样后继续训练。

W8A16 swap0相对旧峰值下降约24.8%（1.55GiB），swap12下降约17.2%（0.98GiB）；修复后swap0比本轮BF16峰值低约16.2%（0.91GiB）。BF16旧/新swap0差约0.046GiB，不解释为算法收益。换层较多时VAE decode约4.70GiB形成全流程峰值，不能再按基座字节数线性推断节省量。

阶段表中`vae_decode_before_load`只是函数入口；只有发生`vae_device_transition`到CUDA才证明实际加载。final/no-latent调用的额外入口记录不参与两次采样的CPU驻留判断。生产审计：`output/convrot-lifecycle-fix-20261003/production-audit.json`。

## 确定性恢复对照

`bench/convrot_90hx/lifecycle_replay.py`在每个mode/swap组合执行两个新进程的固定4更新序列：无预览控制、step2插入一次推理+真实VAE解码。显式使用deterministic Flash，与默认吞吐栈分开；预览RNG隔离、采样后下一次backward、Parameter身份和CPUpark载荷驻留均核验。

生产9格验证live入口；deferred cleanup的后续接入另以同一replay实际GPU/VAE路径和CPU回归补测。

保存每个完整参数、梯度、AdamW exp_avg/exp_avg_sq/step的dtype/shape/SHA256，以及CPU/CUDA RNG hash，不重复存巨大数值矩阵。它验证插入preview生命周期是否改变同一模式的更新；不表示量化与BF16数值等价，也不证明生产dataloader精确resume游标。

9/9模式/swap对照全部通过：392个参数、全部梯度、AdamW状态的dtype/shape/SHA256、CPU/CUDA RNG和四次更新loss/输入均与无预览控制完全一致，optimizer step均为4。真实VAE解码PNG成功，park时managed GPU payload归零，恢复后下一backward正常。replay审计：`output/convrot-lifecycle-fix-20261003/replay-audit.json`。

该对照使用确定性Flash和固定输入序列，不能当作默认非确定性Flash的逐位认证或量化对BF16质量认证。

## 范围边界

本轮针对plain LoRA生产loop与shared VAE decode。其他adapter若自己stash计算图，仍需其独立cleanup契约；没有无条件清空它们。Z-Image/Qwen的共享入口通过CPU布局/异常回归，不代表这两族完整模型GPU验收。mixed NF4经小模型CUDA换层测试，完整9格生产矩阵仅BF16和ConvRot INT8。

旧[S0–S5筛选报告](convrot_90hx_validation_20261002.md)和原数值门槛保留。N1仍待完整18样本确定性复核，N2未晋级；本轮没有启动3seed×20轨迹或500步训练。修复显存生命周期不自动等于证明量化训练质量或速度更好。
