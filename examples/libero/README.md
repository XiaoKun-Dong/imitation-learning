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

普通评估默认遵循官方 OpenPI 口径：`OffScreenRenderEnv`、256 像素环境渲染、224 像素模型输入、
图像旋转 180 度、等待 10 步、每 5 步 replan，并使用 policy server 的正常随机数流。只有显式传入
`--args.policy-noise-seed` 时才固定每次 replan 的 flow noise；该选项用于配对消融，不作为官方最终成绩口径。

## Dynamic gate 退化检查

大规模评估前，先把 dynamic checkpoint 与两种基线做每任务 3 个 rollout 的配对检查：

```bash
bash examples/libero/eval_demovla_regression_pilot.sh \
  checkpoints/demovla_libero_sparse_deep_dynamic_gate/dynamic_gate_v1/29999 \
  /home/dongxiaokun/baseck/pi05_libero \
  0 8000 3 7
```

三组分别是正常 dynamic gate、同一 checkpoint 关闭 interaction injection、官方 `pi05_libero` checkpoint。
它们使用完全相同的初始状态和 flow noise。该 pilot 只用于发现明显退化；通过后，最终 50-rollout 官方口径
应去掉 `--args.policy-noise-seed`，单独运行正常 dynamic gate。

## Dynamic gate 配对消融

以下脚本依次评估同一个 dynamic-gate checkpoint 的正常推理、逐层平均 gate、关闭 interaction injection，
再评估单独训练的 static-deep checkpoint。四组使用相同的 LIBERO initial state 和
`(noise_seed, task, episode, replan)` flow noise：

```bash
bash examples/libero/eval_demovla_gate_ablation.sh \
  /path/to/dynamic/checkpoint/29999 \
  /path/to/static-deep/checkpoint/29999 \
  0 8000 50 7
```

默认逐层平均 gate 来自 dynamic 29999 日志：`0.0211 0.0242 0.0277`。可用训练末段多个日志点的均值覆盖：

```bash
LAYER_MEAN_GATES="<layer4> <layer9> <layer14>" \
bash examples/libero/eval_demovla_gate_ablation.sh DYNAMIC_CKPT STATIC_CKPT
```

脚本最后输出成功率以及相对 dynamic 的逐 episode 配对结果和 McNemar exact p-value。static-deep
checkpoint 必须使用与 dynamic 相同的数据、norm stats、训练步数和初始化权重；否则该组只能作为历史参考，
不能单独归因于 gate 结构。

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
