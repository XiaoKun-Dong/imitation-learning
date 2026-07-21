# π0.5 2D Object-Centric LIBERO

本项目基于 [Physical Intelligence OpenPI](https://github.com/Physical-Intelligence/openpi)，研究如何在不修改
π0.5 图像/语言主干的前提下，为 action expert 增加显式目标物条件，并验证它在目标位移、干扰物增加等
场景中的鲁棒性。

当前主线是 **2D object-centric**：使用 simulator 提供的目标实例分割，构造
`target_mask + target_bbox + target_crop`，编码为空间 object token，再由 action expert token 通过
cross-attention 读取。Point cloud 暂未引入；已有 `target_point` 单点 3D 路线仅保留为附加对照。

> 当前 mask 来自 LIBERO 的 GT instance segmentation，因此实验首先衡量 oracle 2D object condition 的价值，
> 不代表真实分割模型存在误检、漏检和延迟时的最终性能。

## 方法概览

```text
RGB + wrist RGB + language + proprioception
                    |
                    v
             π0.5 frozen backbone

target mask + bbox + masked RGB crop
                    |
                    v
          16x16 object patch tokens
                    |
                    v
action expert tokens -- cross-attention --> conditioned action tokens
                    |
                    v
              flow-matching actions
```

当前 2D 配置具有以下约束：

- 从官方 `pi05_libero` checkpoint 初始化。
- 冻结所有 legacy policy 参数，只训练 `object_condition_*`。
- cross-attention output projection 使用 zero initialization。
- object residual scale 固定为 `0.1`。
- 训练和推理均不向 2D 模型传入 depth 或 `target_point`。

## 环境要求

- Linux；当前开发环境为 Ubuntu 20.04/22.04。
- Python 3.11。
- NVIDIA GPU。π0.5 推理通常需要至少 8GB 显存；本项目的 adapter-only 单卡训练使用
  `batch_size=1` 且关闭 EMA。
- NVIDIA driver 可被 `nvidia-smi` 正常识别。JAX CUDA 运行库由 `uv` 环境安装，通常不需要单独安装系统 CUDA toolkit。
- Git、Git LFS、curl、unzip、ffmpeg 和 ImageMagick/MagickWand。

Ubuntu 系统依赖：

```bash
sudo apt-get update
sudo apt-get install -y \
  git git-lfs curl unzip ffmpeg \
  libgl1 libglib2.0-0 libmagickwand-dev
```

安装 [uv](https://docs.astral.sh/uv/getting-started/installation/) 后初始化仓库：

```bash
git submodule update --init --recursive
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

验证基础环境：

```bash
uv run python -c "import jax; print(jax.devices())"
uv run python -c "from libero.libero import get_libero_path; print(get_libero_path('datasets'))"
uv run python -c "from wand.api import library; print('MagickWand OK')"
```

如果 Codex/容器环境无法写入默认 uv cache，可临时使用：

```bash
export UV_CACHE_DIR=/tmp/uv-cache
export XDG_CACHE_HOME=/tmp/xdg-cache
```

## 目录结构

```text
src/openpi/models/pi0.py                 object encoder 与 cross-attention
src/openpi/models/model.py               Observation 与 object 字段预处理
src/openpi/training/config.py            数据和训练配置
src/openpi/policies/libero_policy.py      LIBERO policy 输入输出映射
examples/libero/main.py                   LIBERO/LIBERO-P 在线评测
examples/libero/run_libero_plus.py        隔离加载 LIBERO-P
examples/libero/convert_libero_data_to_lerobot.py
                                         HDF5 -> LeRobot + object condition
docs/pi05_object_mask_todo.md             实验记录与后续计划
```

## Checkpoint 准备

当前配置默认使用：

```text
/home/dongxiaokun/baseck/pi05_libero/
├── params/
└── assets/
```

官方 checkpoint 地址：

```text
gs://openpi-assets/checkpoints/pi05_libero
```

也可以让 OpenPI 直接读取 GCS 路径，或者将 checkpoint 下载到本地后修改
[`src/openpi/training/config.py`](src/openpi/training/config.py) 中的 `assets_dir` 和 `CheckpointWeightLoader`。

注意 checkpoint 层级：

- policy server 的 `--policy.dir` 指向包含 `params/` 和 `assets/` 的 checkpoint 根目录。
- `CheckpointWeightLoader` 指向具体的 `params/` 目录。
- Orbax 目录中应存在 `_METADATA`；若缺失，通常是路径多一层或少一层。

## LIBERO 训练数据

原始 LIBERO Object demonstrations 默认放置在：

```text
third_party/libero/LIBERO/libero/datasets/libero_object/*_demo.hdf5
```

转换为带 2D/单点 3D 标注的本地 LeRobot 数据集：

```bash
uv run python examples/libero/convert_libero_data_to_lerobot.py \
  --repo-name local/libero_object_mask \
  --output-root data/lerobot \
  --debug-overlay-dir data/libero_mask_debug_full \
  --debug-frames-per-task 3
```

输出位置：

```text
data/lerobot/local/libero_object_mask
```

当前完整数据包含：

- 500 episodes
- 74,507 frames
- 10 个 LIBERO Object 任务
- 每帧 `target_mask`、`target_bbox`、`target_crop` 和 `target_point`

2D 训练配置只 repack 前三个字段，即使数据文件包含 `target_point` 也不会传入模型。

## 训练

当前主配置：

```text
pi05_libero_object_2d_cross_attention
```

它复用官方 `pi05_libero` norm stats，因此使用当前数据和机器人动作定义时不需要重新计算。如果数据或动作空间发生变化，
再运行：

```bash
uv run python scripts/compute_norm_stats.py \
  --config-name pi05_libero_object_2d_cross_attention
```

单卡 1k-step pilot：

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python scripts/train.py pi05_libero_object_2d_cross_attention \
  --exp-name 2d_pilot_1k \
  --batch-size 1 \
  --num-workers 4 \
  --num-train-steps 1000 \
  --save-interval 250 \
  --ema-decay None \
  --no-wandb-enabled \
  --overwrite
```

checkpoint 将写入：

```text
checkpoints/pi05_libero_object_2d_cross_attention/2d_pilot_1k/{250,500,750,999}
```

显存不足时优先保持 `batch-size=1`、`ema-decay=None`，并减少 `num-workers`。adapter-only 训练不应通过
解冻 backbone 来换取短期 loss 下降，否则难以与官方成熟策略公平比较。

## 启动 Policy Server

### 2D object-centric

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python scripts/serve_policy.py \
  policy:checkpoint \
  --policy.config pi05_libero_object_2d_cross_attention \
  --policy.dir checkpoints/pi05_libero_object_2d_cross_attention/2d_pilot_1k/999
```

### 官方 π0.5 LIBERO baseline

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python scripts/serve_policy.py \
  policy:checkpoint \
  --policy.config pi05_libero \
  --policy.dir /home/dongxiaokun/baseck/pi05_libero
```

服务默认监听：

```text
ws://0.0.0.0:8000
```

同一时间只启动一个占用 8000 端口的 server。

## LIBERO-P 配置

LIBERO-P 用于测试目标位移、干扰物、相机、光照等分布扰动。它在独立 Python 路径中加载，不替换训练使用的
原始 LIBERO package。

首次安装：

```bash
bash examples/libero/setup_libero_plus.sh
```

脚本会：

- clone `sylvestf/LIBERO-plus`
- 安装 `scikit-image` 和 `wand`
- 检查系统 MagickWand
- 断点续传约 6.4GB 的 `assets.zip`
- 处理上游 zip 中异常的目录前缀并放置 assets

解压后的 assets 约占 9.5GB。确认评测可以启动后，可删除压缩包回收空间：

```bash
rm third_party/libero-plus/assets.zip
```

## 评测

评测需要两个终端：终端 1 启动 policy server，终端 2 运行 simulator client。

### 目标位移

从 LIBERO-P `_level*` 中确定性打乱并抽取 20 个多目标任务：

```bash
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 \
uv run python examples/libero/run_libero_plus.py \
  --args.task-suite-name libero_object \
  --args.object-condition 2d \
  --args.task-category "Objects Layout" \
  --args.task-name-contains _level \
  --args.shuffle-tasks \
  --args.max-tasks 20 \
  --args.num-trials-per-task 1 \
  --args.seed 7 \
  --args.video-out-path data/libero_plus/videos/target_displacement_2d_seed7
```

### 增加干扰物

```bash
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 \
uv run python examples/libero/run_libero_plus.py \
  --args.task-suite-name libero_object \
  --args.object-condition 2d \
  --args.task-category "Objects Layout" \
  --args.task-name-contains _add_ \
  --args.shuffle-tasks \
  --args.max-tasks 20 \
  --args.num-trials-per-task 1 \
  --args.seed 7 \
  --args.video-out-path data/libero_plus/videos/object_distractors_2d_seed7
```

评测 Base 时保持 task 过滤、seed、replan steps 和任务数完全一致，只修改：

```bash
--args.object-condition none
--args.video-out-path <新的输出目录>
```

建议至少运行 seed `7、42、123`。`seed` 同时控制任务打乱和 simulator 随机状态，因此 Base 与 2D 必须成对使用
相同 seed。

## Mask 在线更新

mask 不是 episode 开始时固定一次。每当 action plan 执行完、client 请求新的 action chunk 时，评测器会：

1. 从当前 observation 读取实时 instance segmentation。
2. 根据目标实例生成 mask、bbox 和 masked RGB crop。
3. 将当前 2D 条件发送给 policy server。
4. 执行 `replan_steps` 个动作后再次更新。

因此更新频率由 `--args.replan-steps` 控制，而不是每个 simulator step 都请求一次模型。

## 指标与结果文件

每次评测会在视频目录生成：

```text
metrics.jsonl
rollout_task_XXXX_..._success.mp4
rollout_task_XXXX_..._failure.mp4
```

核心指标：

- `success`：任务是否完成。
- `target_grasped`：是否曾抓住目标物。
- `wrong_object_grasped`：是否抓住非目标物。
- `steps`：episode 执行步数。
- `difficulty_level`：LIBERO-P 综合难度。

目标位移强度由任务名中的 `level1` 至 `level5` 表示，不应与 JSON 的综合 `difficulty_level` 混用。

主要实验应报告 success rate、target-grasp failure、post-grasp failure、wrong-object rate、共同成功任务步数，
并使用严格配对的 task ID 和 seed。当前实验只能解释已见物体和已见技能下的扰动鲁棒性，不能宣称未见物体、
新任务或新机器人上的通用泛化。

## 测试

运行 object-condition 相关测试：

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run pytest -q src/openpi/models/model_test.py \
  -k "object_condition or target or freeze_filter"
```

代码检查：

```bash
uv run ruff check src/openpi examples/libero
```

## 常见问题

### `ModuleNotFoundError: No module named 'libero'`

```bash
git submodule update --init --recursive
GIT_LFS_SKIP_SMUDGE=1 uv sync
uv pip install -e third_party/libero/LIBERO
```

### `MagickWand shared library not found`

```bash
sudo apt-get install -y libmagickwand-dev
uv run python -c "from wand.api import library; print('MagickWand OK')"
```

### `torch.load` 报 `weights_only` 错误

`examples/libero/run_libero_plus.py` 已为可信的 LIBERO-P init-state 文件设置兼容加载。请使用该 wrapper，
不要直接运行 LIBERO-P 自带 evaluator。

### EGL 初始化失败

先确认 `nvidia-smi` 正常，然后尝试：

```bash
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 <评测命令>
```

多卡机器需要把 `MUJOCO_EGL_DEVICE_ID` 改成实际可见设备序号。桌面环境也可以尝试 `MUJOCO_GL=glx`。

### `FileNotFoundError: _METADATA`

检查 `--policy.dir` 与 `CheckpointWeightLoader` 的目录层级。policy 根目录通常包含 `params/`，而 loader 通常直接
指向 `params/`。

### 单卡训练显存不足

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python scripts/train.py <config> \
  --batch-size 1 \
  --num-workers 0 \
  --ema-decay None
```

## 实验路线

当前优先级：

1. 证明 2D object-centric 在目标位移或干扰物任务上的稳定优势。
2. 确认原始 LIBERO Object 性能不被 residual 分支破坏。
3. 完成 mask+bbox+crop、mask+bbox、crop-only 和 mask corruption 消融。
4. 只有在明确 2D 表示的失败边界后，再恢复 point cloud 工作。

详细状态与历史结果见 [docs/pi05_object_mask_todo.md](docs/pi05_object_mask_todo.md)。

## 致谢与许可

本项目建立在 OpenPI、LIBERO、LIBERO-P、JAX、Flax、LeRobot、MuJoCo 和 Robosuite 之上。上游模型与代码的
版权和许可归各自作者所有；本仓库继续遵循根目录 [LICENSE](LICENSE) 中的许可条款。
