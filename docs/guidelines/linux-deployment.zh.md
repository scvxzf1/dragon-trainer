# Linux 部署与启动指南

状态：稳定

本文面向准备在 GitHub 上拉取并运行本项目的 Linux 用户，覆盖从系统依赖、Python 环境、模型下载、WebUI 启动到常见故障排查的完整流程。

当前项目推荐使用 `uv` 管理 Python 3.13 虚拟环境。不要直接把系统 Python 或 Conda base 环境混进训练环境里，否则很容易出现 `toml`、`torch`、`flash-attn` 等依赖版本不一致的问题。


## 1. 适用环境

推荐环境：

- Linux x86_64
- NVIDIA 显卡
- 已安装可用的 NVIDIA 驱动，`nvidia-smi` 能正常显示显卡
- Python 环境由 `uv sync` 自动创建
- 网络可访问 GitHub、PyPI、PyTorch wheel 源和 Hugging Face

项目的 `pyproject.toml` 在 Linux 上默认解析：

- Python `3.13.*`
- PyTorch `2.12`
- CUDA `13.0` 对应 wheel
- Flash Attention 2 预编译 wheel
- bitsandbytes 预编译 wheel，可用于 `AdamW8bit`

不要把环境解析到 CUDA `13.2` 的 PyTorch / Flash Attention 轨道：bitsandbytes 当前没有 CUDA 13.2 二进制包，`AdamW8bit` 会不可用。如果你的驱动太旧，不支持 CUDA 13.x 运行时，请先升级 NVIDIA 驱动。


## 2. 安装系统依赖

Ubuntu / Debian 系建议先安装这些基础工具：

```bash
sudo apt update
sudo apt install -y \
  git git-lfs curl wget build-essential \
  python3 python3-venv python3-pip \
  libgl1 libglib2.0-0
```

启用 Git LFS：

```bash
git lfs install
```

检查显卡驱动：

```bash
nvidia-smi
```

如果这一步失败，先修复 NVIDIA 驱动，再继续部署。


## 3. 安装 uv

推荐使用官方安装脚本：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

让当前终端加载 `uv`：

```bash
source "$HOME/.local/bin/env"
```

验证：

```bash
uv --version
```

如果提示找不到 `uv`，重新打开终端，或把下面内容加入 `~/.bashrc`：

```bash
export PATH="$HOME/.local/bin:$PATH"
```


## 4. 克隆项目

```bash
git clone <你的仓库地址> dragon-trainer
cd dragon-trainer
```

（若沿用旧目录名如 `anima_lora` 或 `anima_lora_webui`，项目内相对路径行为完全兼容。）

如果仓库内使用了 Git LFS 文件，执行：

```bash
git lfs pull
```


## 5. 创建 Python 环境

在项目根目录执行：

```bash
uv sync
```

完成后会生成项目内虚拟环境：

```text
.venv/
```

验证 Python 和关键依赖：

```bash
.venv/bin/python --version
.venv/bin/python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
.venv/bin/python -c "import toml, aiohttp, accelerate; print('ok')"
```

正常情况下，`torch.cuda.is_available()` 应输出 `True`。


## 6. 登录 Hugging Face

模型下载依赖 Hugging Face CLI。先登录：

```bash
.venv/bin/hf auth login
```

如果你使用的是无图形界面的服务器，可以在浏览器里生成 token，然后粘贴到终端。


## 7. 下载模型

下载项目默认需要的模型：

```bash
.venv/bin/python tasks.py download-models
```

或者在已激活项目虚拟环境后使用 Makefile：

```bash
source .venv/bin/activate
make download-models
```

下载完成后，默认模型路径应类似：

```text
models/diffusion_models/anima-base-v1.0.safetensors
models/text_encoders/qwen_3_06b_base.safetensors
models/vae/qwen_image_vae.safetensors
models/sam3/
models/mit/
models/pe/
```

基础模型路径由 `configs/base.toml` 管理：

```toml
pretrained_model_name_or_path = "models/diffusion_models/anima-base-v1.0.safetensors"
qwen3 = "models/text_encoders/qwen_3_06b_base.safetensors"
vae = "models/vae/qwen_image_vae.safetensors"
```

如果你手动放置模型，请保持这些路径一致，或在 WebUI / TOML 中修改为你的实际路径。


## 8. 准备训练数据

默认源图目录是：

```text
image_dataset/
```

推荐结构：

```text
image_dataset/
├── 0001.png
├── 0001.txt
├── 0002.jpg
├── 0002.txt
└── ...
```

要求：

- 图片支持常见格式，如 `.png`、`.jpg`、`.jpeg`、`.webp`
- 每张图片建议有同名 `.txt` caption 文件
- `source_image_dir` 必须真实存在

项目会基于 `source_image_dir` 自动派生并创建：

```text
post_image_dataset/resized/
post_image_dataset/lora/
```

它们分别对应：

- `resized_image_dir`：预处理后的缩放图像目录
- `lora_cache_dir`：VAE latent、文本编码等缓存目录


## 9. 启动 WebUI

推荐启动方式：

```bash
.venv/bin/python tasks.py web --host 127.0.0.1 --port 20102
```

然后浏览器打开：

```text
http://127.0.0.1:20102/
```

也可以直接使用模块入口：

```bash
.venv/bin/python -m web --host 127.0.0.1 --port 20102
```

或在已激活项目虚拟环境后使用 Makefile：

```bash
source .venv/bin/activate
make web ARGS="--host 127.0.0.1 --port 20102"
```

首次启动加载后端依赖时可能需要约 10–40 秒。访问服务根路径会进入 Next WebUI：

```text
http://127.0.0.1:20102/
http://127.0.0.1:20102/next
```

历史 `/?ui=dragon` 和 `/?ui=classic` 地址目前会重定向到 Next，并不启动旧界面。用户使用、开发和验证说明见 [Next 前端 README](../../web/frontend-next/README.md)。

项目根目录的 Linux 一键脚本也可以启动 WebUI：

```bash
./launch_webui_app.sh
```

该脚本默认使用 `127.0.0.1:20103`，不会自动打开浏览器，终端会保留输出。它只把“端口正在监听”作为便捷判断，不验证监听进程是不是 WebUI，也不等同于 HTTP 健康检查。可只对该脚本覆盖地址：

```bash
ANIMA_WEB_HOST=127.0.0.1 ANIMA_WEB_PORT=20102 ./launch_webui_app.sh
```

如果你需要让局域网其他机器访问：

```bash
export ANIMA_WEBUI_TOKEN='替换为足够长的随机令牌'
.venv/bin/python tasks.py web --host 0.0.0.0 --port 20102
```

然后使用服务器 IP 访问：

```text
http://服务器IP:20102/?token=你的令牌
```

注意：非 loopback 地址必须提供 `--token` 或环境变量 `ANIMA_WEBUI_TOKEN`，否则服务会拒绝启动。正确的 query token 会建立认证 cookie；也可以使用 `Authorization: Bearer` 或 `X-Anima-Token`。不要把真实 token 写入仓库、共享文档或带密钥的远程 URL。`0.0.0.0` 会暴露服务到当前网络，请同时确认防火墙和访问环境安全。


## 10. WebUI 推荐使用流程

首次使用建议按这个顺序：

1. 打开 WebUI。
2. 在 Next「模型」工作区检查模型路径是否存在。
3. 进入 Next「数据集蓝图」工作区（`/next/datasets`），新建或选择一份数据集预设，在子集卡片中设置原始图片目录 `source_dir`、缩放图目录 `image_dir`（如 `post_image_dataset/resized`）和缓存目录 `cache_dir`（如 `post_image_dataset/lora`），点击保存。
4. 进入 Next「训练」工作区（`/next/training`），在数据集与子集卡片中绑定该数据集预设（或直接在当前配置的子集卡片中核对目录），保存当前训练配置。
5. 如果提示需要预处理，先执行预处理。
6. 预处理完成后启动训练。
7. 在 Next「历史」和训练监控中查看日志、loss 曲线和任务配置快照。
8. 在 Next「预览」工作区查看训练样张和推理预览图。

训练任务快照中会记录：

- 源图像目录
- 缩放图像目录
- LoRA 缓存目录
- 输出目录
- 样张目录
- 日志文件
- 指标文件
- TOML 配置快照


## 11. CLI 训练流程

如果不使用 WebUI，也可以走命令行。

完整预处理：

```bash
.venv/bin/python tasks.py preprocess
```

训练默认 LoRA：

```bash
.venv/bin/python tasks.py lora
```

使用指定 preset：

```bash
PRESET=low_vram .venv/bin/python tasks.py lora
```

从 WebUI 配置变体训练：

```bash
GUI_PRESETS=lokr .venv/bin/python tasks.py lora-gui
```

或：

```bash
.venv/bin/python tasks.py lora-gui lokr
```

追加训练参数：

```bash
.venv/bin/python tasks.py lora-gui lora --network_dim 32 --max_train_epochs 10
```

推理测试：

```bash
.venv/bin/python tasks.py test
```

合并最新 LoRA：

```bash
.venv/bin/python tasks.py merge
```

查看所有任务：

```bash
.venv/bin/python tasks.py --help
```


## 12. 后台运行 WebUI

简单后台运行：

```bash
mkdir -p logs
nohup .venv/bin/python tasks.py web --host 127.0.0.1 --port 20102 \
  > logs/webui-20102.log 2>&1 &
```

查看日志：

```bash
tail -f logs/webui-20102.log
```

查看端口：

```bash
ss -ltnp | grep 20102
```

停止服务：

```bash
pkill -f "python -m web"
```

如果你只想停止某个端口，可以先查 PID：

```bash
ss -ltnp | grep 20102
```

然后：

```bash
kill <PID>
```


## 13. systemd 服务示例

如果希望服务器开机自动启动，可以创建 systemd 服务。

把路径替换为你的项目真实路径：

```ini
[Unit]
Description=Dragon Trainer
After=network.target

[Service]
Type=simple
WorkingDirectory=/home/USER/dragon-trainer
ExecStart=/home/USER/dragon-trainer/.venv/bin/python tasks.py web --host 127.0.0.1 --port 20102
Restart=on-failure
RestartSec=5
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
```

保存为：

```text
/etc/systemd/system/dragon-trainer-webui.service
```

（若升级前已配置 `anima-lora-webui.service`，沿用原服务名即可。）

启用并启动：

```bash
sudo systemctl daemon-reload
sudo systemctl enable dragon-trainer-webui
sudo systemctl start dragon-trainer-webui
```

查看状态：

```bash
systemctl status dragon-trainer-webui
```

查看日志：

```bash
journalctl -u dragon-trainer-webui -f
```


## 14. 目录说明

常用目录：

```text
configs/                    配置文件
configs/base.toml           全局基础配置
configs/presets.toml        训练预设
configs/gui-methods/        WebUI 内置训练方法配置
configs/imported/           导入或另存的用户配置
configs/web-training-history/ WebUI 训练任务历史

image_dataset/              默认源图像目录
post_image_dataset/resized/ 自动派生的缩放图像目录
post_image_dataset/lora/    自动派生的 LoRA 缓存目录

models/                     模型权重目录
output/ckpt/                默认训练输出目录
output/ckpt/sample/         默认训练中样张目录
output/tests/               默认推理预览目录
```

这些目录通常不建议提交到 GitHub：

- `.venv/`
- `models/`
- `output/`
- `image_dataset/`
- `post_image_dataset/`（含 `resized/`、`lora/`、`masks/` 等预处理缓存；旧版 `*_resized/`、`*_lora_cache/` 同理）
- `.env`
- 大型 `.safetensors` / `.ckpt` / `.pt` 文件

项目 `.gitignore` 已经覆盖了大部分运行产物。


## 15. 常见问题

### 15.1 ModuleNotFoundError: No module named 'toml'

原因通常是没有使用项目虚拟环境。

错误示例：

```bash
python tasks.py web
```

推荐：

```bash
.venv/bin/python tasks.py web
```

或先进入环境：

```bash
source .venv/bin/activate
python tasks.py web
```


### 15.2 torch.cuda.is_available() 是 False

先检查：

```bash
nvidia-smi
.venv/bin/python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

常见原因：

- NVIDIA 驱动不可用
- 驱动版本过旧
- 没有通过 `uv sync` 安装项目依赖
- 当前 shell 使用了错误 Python


### 15.3 端口已被占用

检查端口：

```bash
ss -ltnp | grep 20102
```

换端口启动：

```bash
.venv/bin/python tasks.py web --host 127.0.0.1 --port 20104
```


### 15.4 训练提示 No data found

通常是训练实际读取的图像目录为空。

检查：

```bash
ls image_dataset
ls post_image_dataset/resized
ls post_image_dataset/lora
```

处理方式：

1. 在 `/next/datasets` 的数据集子集卡片中确认 `source_dir` 指向真实源图目录，并设置对应的 `image_dir`（如 `post_image_dataset/resized`）与 `cache_dir`（如 `post_image_dataset/lora`），保存数据集蓝图。
2. 在 `/next/training` 的数据集与子集卡片中确认已绑定该数据集蓝图并保存配置。
3. 先运行预处理。
4. 再启动训练。


### 15.5 训练很久没有预览图

训练中样张不会自动强制开启。需要在 WebUI 的「训练中预览图」配置里设置：

- `sample_prompts`
- `sample_every_n_epochs` 或 `sample_every_n_steps`

推荐提示词文件：

```text
configs/sample_prompts.txt
```

一行一个提示词。未设置采样频率时，不会生成训练中样张。
多 GPU 训练会按提示词行将采样任务分配给各个 rank，再由主 rank 统一解码为
PNG；只有一行提示词时，仅一张 GPU 执行采样，其余 rank 等待。


### 15.6 Hugging Face 下载失败

检查：

```bash
.venv/bin/hf auth whoami
```

重新登录：

```bash
.venv/bin/hf auth login
```

如果网络环境无法直连 Hugging Face，需要自行配置代理。


### 15.7 Next WebUI 无法加载

确认服务终端已完成启动，访问 `/` 或 `/next`，并检查浏览器开发者工具中的网络错误。
Next 工作台提供返回训练配置的恢复入口；部署、静态构建和隔离验收步骤见
[Next 前端 README](../../web/frontend-next/README.md)。旧 `ui=dragon` / `ui=classic`
查询参数只会重定向到 Next，不应作为故障回退或排查入口。旧界面背景记录见
[已退役 Dragon UI 历史记录](../../_archive/docs/features/dragon-ui.md)。


## 16. 更新项目

拉取最新代码：

```bash
git pull
```

同步依赖：

```bash
uv sync
```

如果 WebUI 正在运行，重启它：

```bash
pkill -f "python -m web"
.venv/bin/python tasks.py web --host 127.0.0.1 --port 20102
```


## 17. 最小可用命令清单

新机器从零开始：

```bash
git clone <你的仓库地址> dragon-trainer
cd dragon-trainer
uv sync
.venv/bin/hf auth login
.venv/bin/python tasks.py download-models
mkdir -p image_dataset
.venv/bin/python tasks.py web --host 127.0.0.1 --port 20102
```

已有项目日常启动：

```bash
cd dragon-trainer
.venv/bin/python tasks.py web --host 127.0.0.1 --port 20102
```

CLI 预处理和训练：

```bash
.venv/bin/python tasks.py preprocess
.venv/bin/python tasks.py lora-gui lora
```
