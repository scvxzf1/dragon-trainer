# Krea-2 PP WebUI 验收（2026-11-17）

状态：验收记录 / 阶段快照

## 范围与环境

在 `origin/dev` 的 `1e613eff` 基线上继续 WebUI 接线与验收。
原 20112 服务来自 `krea2-webui-dev-merge`，与最初聊天的旧前端 checkout 不同。
本次在 `codex/krea2-pp-webui-acceptance` 独立工作树、20113 隔离服务验证，
配置、设置、历史、队列与输出均使用独立临时目录，没有重启 20112 服务。

浏览器使用全新 headless Chrome 会话，真实 HTTP API；未 mock 保存或启动接口。
交互验收使用桌面视口 1440×1000；另保存 390×844 的手机布局截图，
后者不计作手机交互或浮层定位验收。
本机证据保存在 `/tmp/krea2-pp-webui-acceptance/`，截图与用户输入路径不纳入 Git。

## 修正的实际问题

- 分布式设置单字段 PUT 原来会用默认值覆盖其他键；现在合并原 `[distributed]`，
  修改微批或模式不会重置交换数、GPU 白名单和其他设置。
- 卡片原来显示本机缺省值，运行却读取配置；现在显示草稿/配置优先值，
  重复选择模式不覆写当前 PP 参数。
- 详情诊断原来只读取已保存配置；现在接受相关标量的白名单 `config_overrides`，
  读取后复制配置，并随当前草稿重新查询。文件路径与数据集结构不能作为覆盖项。
- 预检保留具体跨族拒绝原因；Krea-2 微批错误附有效 batch/微批建议，
  WebUI 按即将 materialize 的 `train_batch_size` 判断，runtime 仍按实际 latent batch 判断。
- 外置配置的 Web 显示路径必须解析为实际绝对路径再传递给训练 CLI，
  否则 Web 预检可读取、子进程却会报 `configs/imported/config.runtime.toml not found`。
- 冻结 Krea-2 PP runtime 在完整缓存、无采样和无 CMMD 时实际跳过 VAE 加载。
  预检现只对这种已逐图验证缓存和 identity 的直训路径将缺失 VAE 改为信息；
  原配置的预处理要求、缺失/无效缓存拒绝、严格 identity 和其他模型族规则保持。

## 七项验收

| # | 项目 | 证据 |
| --- | --- | --- |
| 1 | 字段显示 | 桌面分布式详情可见“PP 块交换”，截图已保存 |
| 2 | 保存往返 | 设 4，保存本机缺省值与当前训练配置，刷新仍为 4；后端读回为 4 |
| 3 | 0 可设回 | 4→0 保存成功，刷新为 0；额外验证修改微批不重置 swap |
| 4 | 预检透传 | 实际接口结果包含 `pipeline_parallel_blocks_to_swap=4`；诊断也随草稿更新 |
| 5 | WebUI 到 runtime | 独立冻结 runtime 的 1 步真实训练验收，结果与日志见下节 |
| 6 | 跨族守卫 | 界面切换至 Anima、保留 swap=4，详情显示 `only implemented for krea2/krea2_raw` |
| 7 | 微批建议 | batch=2、microbatches=4 的实际预检拒绝，并建议 `--pipeline_parallel_microbatches=2` |

`browser-report.json` 记录 1–4、6–7 的真实请求与结果，页面 `pageerror` 为 0。
`runtime-start-report.json` 保存实际启动请求与确认弹窗的预检结果。
最终构建重新运行六项交互仍全部通过，合并真实训练结果后，
`acceptance-final.json` 记录 7/7 PASS。

最后定向检查：后端 87 passed，前端 14 passed，类型检查、`web-next-build`
与 `git diff --check` 通过。独立验收另复核了外置路径、runtime-only VAE
豁免、严格缓存身份及真实任务的 returncode=0。

## 真实训练边界与证据

使用已有 896×1200 的两张图片及 latent/text 缓存、真实 Krea-2 12.82B 权重，
FP16、plain LoRA rank 16、PP=2、microbatches=2、每 stage 交换 4 块，
`max_train_steps=1`。关闭采样和 CMMD，不重建缓存、不下载模型。

普通导入配置依然被缺失 VAE 拒绝，因此启动选择的是有合法目录结构的
`config.runtime.toml`，通过 WebUI 确认弹窗调用真实 `/api/training/start`。
这次验收针对完整缓存的直训与参数传递，不证明原始图片预处理或长期数值收敛。

实际任务 `20261117-211028-training-imported-config.runtime` 正常完成：

```text
[pp] rank=0 swapping blocks [1, 2, 3, 4] to cpu
[pp] rank=1 swapping blocks [14, 15, 16, 17] to cpu
[pp] step 1 wall=23.32s peak=14.36GiB
[pp] step 1 wall=23.30s peak=14.55GiB
pp step 1/1 (effective 1, skipped 0) loss=0.180718 lr=0.0002 scale=65536.0
PP bundle saved
model saved.
训练完成
```

这些步时是该次 1 步验收读数，不是热稳态基准。
最终日志和指标保存于本机 `runtime-result.json` 与 `runtime-logs.txt`；
API 指标也记录 step=1、loss=0.180718。
原始 20112 服务未替换；该端口仍需在部署本次版本后重新检查。
