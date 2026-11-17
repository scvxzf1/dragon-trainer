# Git 同步规则

状态：稳定

默认目标仓库：`github.com/scvxzf1/dragon-trainer`（2026-10-19 由 `anima_lora_webui` 改名，旧地址自动重定向）

分支职责：`dev` 用于日常开发和集成，`main` 用于发布。未指定分支的开发推送默认进入
`dev`；只有明确发布时才更新 `main`。

本规则按目标仓库 URL 和分支识别线上主线，不要求本机 remote 必须叫 `webui` 或 `origin`。

## 同步前

```bash
git status --short --branch
git remote -v
```

从输出中找到 URL 匹配默认目标的 remote，以下记作 `<target-remote>`。个人 fork、私有镜像和上游参考仓都不能替代默认目标。

```bash
git fetch <target-remote> --prune
git switch dev
git pull --rebase <target-remote> dev
git rev-list --left-right --count HEAD...<target-remote>/dev
git log --oneline <target-remote>/dev..HEAD
```

若目标分支也有本地没有的提交，先说明差异并选择正常合并方式，不要强推覆盖。发布前还要
对 `main` 重复比较，并确认 `dev` 已通过相关测试。

## 首次建立 dev

在已同步的 `main` 上执行：

```bash
git switch main
git pull --ff-only
git switch -c dev
git push -u <target-remote> dev:dev
```

若线上已经存在 `dev`，改用 `git switch --track <target-remote>/dev`，不要用本地分支覆盖线上历史。

## 开发分支推送

有目标仓写权限时，完成开发改动并确认 staged 内容后执行：

```bash
git push <target-remote> dev:dev
```

发布到主线时使用：

```bash
git switch main
git pull --ff-only
git merge --ff-only <target-remote>/dev
git push <target-remote> main:main
```

没有写权限时，把功能分支推到个人 fork 并向默认目标创建 PR。个人 fork 仍是 fork，不因本地 remote 名称而成为线上主仓。

## 安全边界

- 未跟踪文件默认不提交；推送前检查实际 staged 内容和相关测试。含本机配置、模型路径、训练
  历史、队列、日志或导入 prompt 时，不要使用 `git add -A`，只暂存明确的路径白名单。
- 本机 WebUI 覆盖使用已忽略的 `.anima-webui-settings.toml` 或外置配置根；不要把绝对路径写入
  受版本控制的默认配置。
- `private`、个人 fork 和上游参考仓只有在用户明确点名时才执行写操作。
- fetch/push 尽量使用同一协议和 credential helper；不要把 PAT、cookie 或私钥写进 remote URL、
  文档或日志。
- `git reset --hard`、`git clean -fd`、工作区丢弃和任何 force push 都需先说明影响并取得明确确认。
- 共享分支撤销提交优先 `git revert`；确需改写历史时只用 `--force-with-lease`。
- 完成后汇报目标仓库、remote、分支、提交 hash，以及剩余未提交或未跟踪改动。
