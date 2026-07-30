# DemoVLA on OpenPI

本仓库基于 [Physical Intelligence OpenPI](https://github.com/Physical-Intelligence/openpi)，当前主线是在
π0.5 LIBERO policy 上加入可解释的 interaction memory：

- 从 base RGB 和 wrist RGB 的视觉 patch 中提取 4 个 interaction tokens；
- 在 action expert 的多个深度层进行 gated residual injection；
- 使用 attention diversity 和 memory diversity 抑制 query collapse；
- 支持按 episode/replan 保存带位置的 patch attention 可视化；
- 使用固定的 episode/replan flow noise 做可复现 rollout 对照。

mask、bbox、crop 和 point 条件已经退出当前主线，只保留历史代码与实验记录。

## 运行要求

推荐环境：

- Ubuntu 20.04 或 22.04，x86_64；
- Python 3.11（仓库通过 `.python-version` 固定）；
- `uv` 和 Git；
- NVIDIA GPU、支持 CUDA 12 的驱动；训练前应能正常运行 `nvidia-smi`；
- 至少 16 GB 内存；完整训练、数据和多个 checkpoint 建议准备充足磁盘空间；
- headless LIBERO 评估需要 EGL。

Ubuntu 系统依赖：

```bash
sudo apt-get update
sudo apt-get install -y \
  git git-lfs curl unzip ffmpeg \
  libgl1 libegl1 libglib2.0-0
git lfs install
```

安装 `uv`：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

如果安装目录尚未进入 PATH，可重新打开终端或执行 `uv tool update-shell`。

## Clone 后快速开始

```bash
git clone <your-repository-url>
cd openpi
bash scripts/setup_demovla.sh
```

GitHub HTTPS 较慢而 SSH 可用时：

```bash
OPENPI_LIBERO_REPO_URL=git@github.com:Lifelong-Robot-Learning/LIBERO.git \
bash scripts/setup_demovla.sh
```

如果需要从 RLDS/TFRecord 转换训练数据，安装额外的 TensorFlow/TFDS 依赖：

```bash
bash scripts/setup_demovla.sh --with-rlds
```

脚本会：

1. 只初始化 LIBERO 子模块，不下载 ALOHA；
2. 使用 Python 3.11 和 `uv.lock` 创建 `.venv`；
3. 安装训练、评估与开发依赖；
4. 检查 JAX、PyTorch、LeRobot、MuJoCo、Robosuite 和 LIBERO。

随时可以重新检查环境：

```bash
uv run python scripts/check_demovla_env.py
```

严格检查训练所需的 GPU、数据、norm stats 和本地 checkpoint：

```bash
uv run python scripts/check_demovla_env.py \
  --require-gpu \
  --require-data \
  --require-norm-stats \
  --require-checkpoint
```

使用默认 GCS checkpoint 而不是本地副本时，省略 `--require-checkpoint`。

## 本地路径配置

配置不再写死开发者的 home 目录。默认值和可选环境变量如下：

| 资源 | 默认值 | 环境变量 |
|---|---|---|
| LIBERO Object LeRobot 数据 | `data/lerobot/local/libero` | `OPENPI_LIBERO_DATA_ROOT` |
| π0.5 LIBERO 初始化权重 | `gs://openpi-assets/checkpoints/pi05_libero` | `OPENPI_PI05_LIBERO_CHECKPOINT` |
| π0.5 base 权重 | `gs://openpi-assets/checkpoints/pi05_base` | `OPENPI_PI05_BASE_CHECKPOINT` |
| LIBERO checkout | 自动识别 `third_party/libero` 和旧的嵌套布局 | `OPENPI_LIBERO_ROOT` |

使用本地 checkpoint 时：

```bash
export OPENPI_PI05_LIBERO_CHECKPOINT=/absolute/path/to/pi05_libero
```

该目录应包含：

```text
pi05_libero/
├── params/
│   └── _METADATA
└── assets/
    └── physical-intelligence/libero/norm_stats.json
```

可以复制 `.env.example` 保存个人路径；`.env` 已被 git 忽略：

```bash
cp .env.example .env
# 编辑 .env 后
set -a
source .env
set +a
```

## 准备 LIBERO Object 数据

当前主配置使用官方 `openvla/modified_libero_rlds` 中的
`libero_object_no_noops`，包含两路 256×256 RGB：

- `image`：第三人称 agent view；
- `wrist_image`：腕部视角。

模型输入阶段会按官方 π0.5 流程统一缩放到 224×224。

只下载 Object 子任务：

```bash
mkdir -p data/modified_libero_rlds
env -u HF_ENDPOINT \
uvx --from huggingface_hub hf download openvla/modified_libero_rlds \
  --repo-type dataset \
  --include "libero_object_no_noops/**" \
  --local-dir data/modified_libero_rlds
```

如果所在网络需要 Hugging Face 镜像，请先确认该镜像完整包含 Object 的 32 个 TFRecord 分片。曾经设置
`HF_ENDPOINT=https://hf-mirror.com` 时，可以对单条命令使用 `env -u HF_ENDPOINT` 强制访问官方端点。

转换为本地 LeRobot v2.1 数据：

```bash
uv run python examples/libero/convert_libero_data_to_lerobot.py \
  --data-dir data/modified_libero_rlds
```

默认输出：

```text
data/lerobot/local/libero
```

脚本默认拒绝覆盖已有目录。确认需要重建时显式增加 `--overwrite`。转换完成后检查：

```bash
uv run python scripts/check_demovla_env.py --require-data
```

当前 Object 数据的预期规模是 454 episodes、66,984 frames、10 tasks 和 454 个 Parquet 文件。

## 计算 normalization statistics

当前推荐训练配置：

```text
demovla_libero_sparse_deep_diverse
```

对新转换的数据计算 Object-only norm stats：

```bash
JAX_PLATFORMS=cpu \
uv run python scripts/compute_norm_stats.py \
  --config-name demovla_libero_sparse_deep_diverse
```

输出文件：

```text
assets/demovla_libero_sparse_deep_diverse/local/libero/norm_stats.json
```

训练配置会自动从这里读取 `state` 和 `actions` 统计。图像不参与 norm stats 计算。

## 训练

八卡训练示例：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python scripts/train.py demovla_libero_sparse_deep_diverse \
  --exp-name object_256_diverse_v1 \
  --batch-size 128 \
  --fsdp-devices 4
```

这里 `batch-size=128` 是 global batch size。8 张可见卡和 `fsdp-devices=4` 会形成两个 data-parallel
group，每组使用 4 卡 FSDP。首次运行如果未设置本地 checkpoint 环境变量，会从 GCS 下载官方
`pi05_libero` 权重。

单卡 smoke test 可以降低 batch 和 worker：

```bash
CUDA_VISIBLE_DEVICES=0 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python scripts/train.py demovla_libero_sparse_deep_diverse \
  --exp-name smoke \
  --batch-size 8 \
  --num-workers 0 \
  --num-train-steps 10 \
  --save-interval 10 \
  --no-wandb-enabled
```

checkpoint 输出：

```text
checkpoints/demovla_libero_sparse_deep_diverse/<exp-name>/<step>
```

## 评估

评估使用两个终端。

终端 1：启动单卡 policy server：

```bash
CUDA_VISIBLE_DEVICES=0 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python scripts/serve_policy.py \
  policy:checkpoint \
  --policy.config demovla_libero_sparse_deep_diverse \
  --policy.dir checkpoints/demovla_libero_sparse_deep_diverse/<exp-name>/<step>
```

终端 2：运行 LIBERO Object rollout：

```bash
MUJOCO_GL=egl \
MUJOCO_EGL_DEVICE_ID=0 \
uv run python examples/libero/main.py \
  --args.task-suite-name libero_object \
  --args.object-condition none \
  --args.num-trials-per-task 1 \
  --args.seed 7 \
  --args.policy-noise-seed 0 \
  --args.video-out-path data/libero/videos/pilot
```

先用每任务 1 次的 pilot 排查运行问题，再把 `--args.num-trials-per-task` 提高到 50 做完整评估。正式对照应固定：

- benchmark task 和 init state；
- `seed`；
- `policy-noise-seed`；
- `replan-steps`；
- diagnostics on/off 状态。

评估器会自动识别标准子模块布局 `third_party/libero` 和旧布局
`third_party/libero/LIBERO`，无需再手动 `uv pip install -e LIBERO` 或设置 `PYTHONPATH`。

### Interaction patch 可视化

启动 server 时增加：

```bash
--interaction-diagnostics
```

rollout 时增加：

```bash
--args.visualize-interaction-patches
```

每个 episode 的可视化保存在该 episode 对应的 `interaction_patches` 目录中。

## 常见问题

### Hugging Face 下载报 `Distant resource does not seem to be on huggingface.co`

检查：

```bash
echo "${HF_ENDPOINT:-unset}"
```

若指向不完整镜像，对下载命令使用：

```bash
env -u HF_ENDPOINT <hf-download-command>
```

### `ModuleNotFoundError: No module named 'libero'`

确认子模块存在：

```bash
git submodule update --init third_party/libero
uv run python scripts/check_demovla_env.py
```

本项目的 evaluator 会直接加载 vendored checkout，不依赖 LIBERO 上游有问题的 editable package discovery。

### Clone 卡在 ALOHA 或 LIBERO

不要使用 `git clone --recurse-submodules`。主仓库 clone 完成后，只拉取 LIBERO：

```bash
git config submodule.third_party/aloha.update none
git config submodule.third_party/libero.url git@github.com:Lifelong-Robot-Learning/LIBERO.git
git submodule update --init third_party/libero
```

### EGL 初始化失败

确认 `nvidia-smi` 正常，且 `MUJOCO_EGL_DEVICE_ID` 是当前进程可见 GPU 的索引。桌面环境可尝试：

```bash
MUJOCO_GL=glx uv run python examples/libero/main.py --help
```

### `FileNotFoundError: norm_stats.json`

为正在使用的训练配置重新运行 `scripts/compute_norm_stats.py`，并确认 `--config-name` 完全一致。

### uv 缓存不可写

```bash
export UV_CACHE_DIR=/tmp/openpi-uv-cache
export HF_DATASETS_CACHE=/tmp/openpi-hf-datasets
```

### 显存不足

先降低 global batch、关闭 EMA、减少 worker；多卡机器可调整 `--fsdp-devices`，该值必须整除可见设备数。

## 开发检查

```bash
uv run ruff check src/openpi scripts examples/libero
uv run pytest -q src/openpi/models/demovla_test.py
```

## 文档

- [DemoVLA 架构、训练与可视化](docs/demovla.md)
- [Sparse-deep 训练报告](docs/demovla_sparse_deep_training_report.md)
- [Diversity 10k 训练与评估报告](docs/demovla_sparse_deep_diversity_10k.md)
- [历史 object-condition 实验](docs/pi05_object_mask_todo.md)

## 致谢与许可

本项目建立在 OpenPI、LIBERO、LeRobot、MuJoCo、Robosuite、JAX、Flax 和 PyTorch 之上。上游模型、数据与代码
遵循各自许可证；本仓库遵循根目录 [LICENSE](LICENSE)。
