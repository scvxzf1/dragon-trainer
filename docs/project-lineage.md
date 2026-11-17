# 项目血缘与上游政策

状态：稳定，仓库关系变化时同步更新
相关文件：`LICENSE`、`NOTICE`、`README.md`、`AGENTS.md`

这篇回答一个迟早会被人问的问题：Dragon Trainer 和 [sorryhyun/anima_lora](https://github.com/sorryhyun/anima_lora) 到底什么关系？

既不是「fork」，也不是「没关系」。是血缘：从那套代码起步，现在是自己在走。

## 一句话

起点是上游的，这棵树是自己的。发行、架构、待办都自己扛，但代码血缘、包名兼容和版权声明不抹。

## 三件「独立」的事

1. 发行独立。`dev` 进 `main` 由本仓决定，不看上游发不发版、发到第几版。
2. 架构独立。多模型族 registry、Next React 工作台、打标和预处理链路、块交换与编译策略，都是在这棵树上长出来的，上游没有这些。
3. 债务独立。上游没修的问题不是我们的待办，我们欠的账也不指望上游来还。

## 三件「不独立」的事

说独立是为了讲清谁负责，不是为了让谁消失。

1. 起点代码是上游的，`LICENSE` 和 `NOTICE` 里的原作者版权声明留着。删掉才是真难看，而且违法。
2. `anima_lora` 包名、`ANIMA_*` 环境变量、`.anima-webui-settings.toml` 这些名字沿用上游。改它们会砸掉用户已经跑起来的配置，还会撕碎 ComfyUI 那侧的对接。
3. 上游还是补丁和思路的正当来源，搬的时候标出处。

## 时间线

数字会涨，命令不会说谎。对不上就自己跑一遍。

| 时间 | 发生了什么 | 怎么验 |
| --- | --- | --- |
| 2026-03-27 | 共同起点，上游那次 `Initial extraction of standalone Anima LoRA training pipeline` | `git log --reverse --date=short --format='%ad %s'`，看第一行 |
| 2026-05-27 | 上游作者的提交最后一次进这棵树 | `git log -1 --before='2026-05-27 23:59'` |
| 2026-05-28 起 | 之后全由本仓维护者持续演进提交 | `git rev-list --count --since=2026-05-28 HEAD` |
| 2026-08-08 | Krea-2 Raw 落地 | `git log --diff-filter=A -- library/models/krea2_raw` |
| 2026-08-25 | Z-Image 落地 | `git log --diff-filter=A -- library/models/z_image` |
| 2026-09-27 | Qwen-Image 2.1 落地 | `git log --diff-filter=A -- library/models/qwen_image_2_1` |
| 2026-10-17 | MOSS-Image 落地 | `git log --diff-filter=A -- library/models/moss_image` |
| 2026-10-19 | 仓库从 `anima_lora_webui` 改名 `dragon-trainer` | GitHub 旧地址自动重定向 |

顺带一提，`web/frontend-next/src` 底下的整套 React 工作台源码也全在这条时间线里。

## 上游政策

- **不跟 merge-base。** 别指望这棵树的 `main` 能和上游 fast-forward，也别为了「保持同步」做整树合并。
- **按需搬。** 上游某个改动确实有用就单独 cherry-pick，或者照着重写一遍，别整树拖进来。
- **搬了要写。** 提交信息带 `port(upstream): <原提交或主题>`；行为有差异的再往 `docs/findings/` 记一笔。
- **issue 不回流。** 我们的 bug 自己修，不丢给上游；上游的 bug 也不是我们的 KPI。
- **remote 认 URL。** 指上游的 remote 统一叫 `upstream`，只 fetch 不 push；推拉本仓前用 `git remote -v` 核对指向 `github.com/scvxzf1/dragon-trainer` 的 remote 名称（常见为 `origin`）。

## 不能动的兼容面

改名是叙事，不是重构。下面这些动一下就会砸用户现场，没讨论过别碰：

- Python 包名 `anima_lora` 和 CLI 兼容层；
- `ANIMA_*` 环境变量、`.anima-webui-settings.toml`；
- `configs/` 目录结构，以及各种 cache、sidecar 的命名；
- 已发布 adapter 的 metadata key，比如 `ss_use_moe_style` 那一组。

## 法律面

- 代码走 MIT，见 [LICENSE](../LICENSE)。上游的版权行必须在，不是可选项。
- Anima 权重走 CircleStone NCL，不在 MIT 覆盖范围内，[NOTICE](../NOTICE) 写得比我清楚。
- 其他借来的工程，列在根 README 的「致谢与来源」。

## 这篇不干什么

它解释关系，不评价谁写得好；不吹「完全重写」，也不把上游的功劳算自己头上。真要比较，看提交和代码，别信形容词。
