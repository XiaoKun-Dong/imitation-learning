# LIBERO evaluation

本目录提供当前 OpenPI/DemoVLA 的 LIBERO client。它与仓库根目录共用 Python 3.11 `uv` 环境，不再创建旧的
Python 3.8 client virtualenv。

完整安装、数据转换、训练和评估流程见根目录 [README](../../README.md)。

## 环境

从仓库根目录执行：

```bash
bash scripts/setup_demovla.sh
uv run python scripts/check_demovla_env.py
```

`main.py` 会自动识别：

```text
third_party/libero
third_party/libero/LIBERO
```

并在 `.cache/libero/config.yaml` 创建非交互式、随项目路径更新的 LIBERO 配置，因此无需手动安装
LIBERO editable package，也无需设置 `PYTHONPATH`。

## 运行

先启动 policy server：

```bash
CUDA_VISIBLE_DEVICES=0 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python scripts/serve_policy.py \
  policy:checkpoint \
  --policy.config demovla_libero_sparse_deep_diverse \
  --policy.dir checkpoints/demovla_libero_sparse_deep_diverse/<exp-name>/<step>
```

再运行 client：

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

正式评估时将每任务 rollout 数提高到 50：

```bash
--args.num-trials-per-task 50
```

## 图形后端

headless NVIDIA 机器推荐 EGL：

```bash
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0
```

桌面机器可以使用：

```bash
MUJOCO_GL=glx
```

`MUJOCO_EGL_DEVICE_ID` 是 `CUDA_VISIBLE_DEVICES` 过滤之后的可见设备索引。

## LIBERO-plus

LIBERO-plus 保持为隔离的可选评估环境：

```bash
bash examples/libero/setup_libero_plus.sh
uv run python examples/libero/run_libero_plus.py --help
```

它不会替换原始 LIBERO checkout。
