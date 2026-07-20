# pi0.5 增加 3D 目标物条件工程 TODO

## 当前路线

在原 2D object-centric 路线基础上，加入由 depth 提取的目标物 3D 点：

```text
RGB 图像 + 语言指令 + 本体状态
        +
目标物 2D mask / bbox / crop
        +
目标物 3D point [x, y, z]
        |
        v
轻量 object encoder
        |
        v
只注入 pi0.5 action expert
```

当前阶段先采用轻量 3D object-centric 表示：不引入完整点云、不改 PaliGemma 图像/语言主干，只把 depth 派生出的目标点作为 object encoder 的额外几何特征。目标是在 `libero_object` 上验证 2D 目标物显式条件 + 3D 位置信息是否提升目标定位、相似物区分和接近抓取阶段的动作稳定性。

## 已完成

### 模型与输入

- [x] 在 JAX `Observation` 中增加可选字段：`target_mask`、`target_bbox`、`target_crop`、`target_point`。
- [x] `Observation.from_dict()` 已支持从数据字典读取目标物字段。
- [x] 训练时将 `target_crop` 从 `uint8` 转为 `float32`，避免 jaxtyping 类型错误。
- [x] JAX `preprocess_observation()` 已保留目标物字段。
- [x] `target_mask` 和 `target_crop` 会 resize 到模型图像坐标系。
- [x] `target_bbox` 会按 RGB 图像相同的 resize-with-pad 几何关系同步变换。
- [x] 明确并固定 `target_bbox` 约定：绝对像素坐标 `[x1, y1, x2, y2)`，x2/y2 为 exclusive，不是 0-1 归一化坐标。
- [x] `target_point` 表示 depth 派生的 3D 目标点 `[x, y, z]`，不随图像 resize 变换。
- [x] `pi0.py` 中已将原来的 6 维全局均值 object condition 替换为轻量空间 object encoder。
- [x] object embedding 只注入 action expert，不改 PaliGemma 图像/语言主干。
- [x] 增加 mask dropout，使训练在 mask 缺失或轻微错误时更稳。
- [x] 给 `target_mask`、`target_bbox`、`target_crop`、`target_point` 增加 shape、dtype、resize 和 bbox 约定单元测试。

### LIBERO 数据与转换

- [x] `LiberoInputs` 已支持透传 `target_mask`、`target_bbox`、`target_crop`、`target_point`。
- [x] LIBERO Object eval 已接入 `SegmentationRenderEnv`，在线推理时可生成目标物 mask/bbox/crop。
- [x] 本地 LIBERO Object HDF5 转 LeRobot 脚本已支持重放 state 并生成 GT 目标物 mask。
- [x] 转换脚本默认将 LeRobot 数据写入项目内 `data/lerobot/`。
- [x] 给转换脚本增加可选 debug 输出：每个任务保存若干 overlay 图。
- [x] 修正 `target_crop` 语义：保留原图空间位置，不再把 crop 贴到左上角。
- [x] 已完成 1 个文件、1 条 demo 的 smoke test，成功写出带 mask 的 LeRobot 数据。
- [x] 已生成 mask overlay 可视化图，输出目录：`data/libero_mask_debug/`。
- [x] 使用修正后的 `target_crop` 重新转换完整 `local/libero_object_mask` 数据集。
- [x] 已完成完整 `libero_object` HDF5 转换，生成本地 `local/libero_object_mask` 数据集。
- [x] 已抽样检查完整数据集 mask overlay，输出目录：`data/libero_mask_debug_full/`。
- [x] 完整扫描 `local/libero_object_mask`，确认 mask、bbox、crop 标定一致。

### 训练配置与验证

- [x] `pyproject.toml` 已接入 LIBERO 相关依赖和本地 editable source。
- [x] 新增 `pi05_libero_object_mask` 训练配置。
- [x] 增加 object-condition ablation 训练配置：
  - `pi05_libero_object_none`：无 object input
  - `pi05_libero_object_bbox_only`：bbox only
  - `pi05_libero_object_mask_only`：mask only
  - `pi05_libero_object_crop_only`：crop only
  - `pi05_libero_object_point_only`：point only
  - `pi05_libero_object_mask_point`：mask + point
  - `pi05_libero_object_mask`：mask + bbox + crop + point
- [x] 增加冻结配置：冻结 VLA 图像/语言主干，只训练 action expert、action adapter 和 object encoder。
- [x] 修正 `pi05_base` 加载新增 object encoder 时的权重子集校验问题。
- [x] 计算 `pi05_libero_object_mask` norm stats，并随 smoke checkpoint 保存。
- [x] 单卡 smoke 训练使用 `--ema-decay None` 规避 EMA 额外显存占用。
- [x] 跑通 `pi05_libero_object_mask` 小步数训练，确认 batch、JIT、loss step 和 checkpoint 正常。

## 已验证命令

重新转换完整数据：

```bash
uv run python examples/libero/convert_libero_data_to_lerobot.py \
  --repo-name local/libero_object_mask \
  --output-root data/lerobot \
  --debug-overlay-dir data/libero_mask_debug_full \
  --debug-frames-per-task 3
```

计算 norm stats：

```bash
XDG_CACHE_HOME=/tmp/xdg-cache UV_CACHE_DIR=/tmp/uv-cache \
uv run python scripts/compute_norm_stats.py \
  --config-name pi05_libero_object_mask
```

单卡 smoke 训练：

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XDG_CACHE_HOME=/tmp/xdg-cache UV_CACHE_DIR=/tmp/uv-cache \
uv run python scripts/train.py pi05_libero_object_mask \
  --exp-name smoke_object_mask \
  --num-train-steps 2 \
  --save-interval 1 \
  --batch-size 1 \
  --num-workers 0 \
  --ema-decay None \
  --no-wandb-enabled \
  --overwrite
```

## 待做实验

### 主实验组

- [ ] 训练 `pi05_libero_object_none`：同一份 `libero_object` 数据，无 object input。
- [ ] 训练 `pi05_libero_object_bbox_only`：只用 bbox。
- [ ] 训练 `pi05_libero_object_mask_only`：只用 mask。
- [ ] 训练 `pi05_libero_object_crop_only`：只用 crop。
- [ ] 训练 `pi05_libero_object_point_only`：只用 point。
- [ ] 训练 `pi05_libero_object_mask_point`：mask + point。
- [ ] 训练 `pi05_libero_object_mask`：mask + bbox + crop + point。
- [ ] 每组使用同一个 `pi05_base` 初始化、相同训练步数、相同冻结策略。

### 评估指标

- [ ] 在 `libero_object` 上评估 5 组 ablation。
- [ ] 统计 success rate。
- [ ] 额外统计 wrong-object rate，重点看相似物体区分是否改善。
- [ ] 检查失败样例中的 mask 是否遮挡、漏检或目标物不可见。
- [ ] 对比训练前后 action 稳定性，尤其是接近抓取阶段的末端轨迹。
- [ ] 可选：把原始 `pi05_libero` 作为外部参考 baseline；主 ablation baseline 使用 `pi05_libero_object_none`。

### Quick 5k 单卡验证

- [x] 使用相同 seed 和 30 条配对 rollout 评估 `pi05_libero_object_none` 与 `pi05_libero_object_mask`。
- [x] `object_none`：success `0/30`，目标抓取 `1/30`，误抓 `0/30`。
- [x] `object_mask`：success `0/30`，目标抓取 `0/30`，误抓 `0/30`。
- [x] 当前 quick run 每组仅训练 `5k steps x batch 1 = 5k` 样本；相比标准 `30k x batch 256`
  的约 768 万样本少约 1536 倍。本轮只能说明两组均未学出可评估策略，不能据此判断 3D 条件无效。
- [x] 从已完成 LIBERO 微调的 `pi05_libero` checkpoint 初始化，完成低预算 object-condition 增量训练和配对评估。

官方 checkpoint 初始化的 5k 单卡实验（每组 100 条配对 rollout）：

- `object_none`：success `69/100`，目标抓取 `84/100`，误抓 `2/100`。
- `object_mask`：success `69/100`，目标抓取 `85/100`，误抓 `1/100`。
- success 配对差异：3D 改善 21 条、退化 21 条、相同 58 条，exact McNemar `p=1.0`，无净提升。
- 3D 组在 salad dressing（+3）、bbq sauce（+2）上提升，在 tomato sauce（-2）、milk（-2）、
  orange juice（-2）上退化，存在任务异质性。
- 两组均明显低于未增量训练的官方 `pi05_libero`（快速评估 `29/30`）。当前 batch 1 全量 action expert
  增量训练破坏了成熟策略，是比 object condition 差异更强的影响因素。下一轮应冻结全部官方参数，仅训练零初始化的
  object encoder / residual 分支。
- [x] 新增 `pi05_libero_object_adapter`：从官方 `pi05_libero` 初始化，冻结全部 legacy 参数，仅训练
  `object_condition_proj_in/out`，并复用本地 `pi05_libero_object_mask` norm stats。
- [x] 训练并评估 `pi05_libero_object_adapter`；主 baseline 使用未增量训练的官方 `pi05_libero`。

Object adapter only 结果：

- 1k/2k/3k/4k/5k 各 10 条初筛中，1k 与 2k 最好（均为 success `6/10`、抓取 `9/10`、误抓 0），
  因 1k 对官方策略扰动更小而选其做 100 条复测。
- adapter 1k 复测：success `70/100`（Wilson 95% CI `60.4%-78.1%`），目标抓取 `88/100`，
  抓取后失败 `18/88`，误抓 `1/100`；tomato sauce 仅 success `1/10`、抓取 `1/10`。
- 与官方已有的相同前 30 条 rollout 配对：官方 success `29/30`，adapter success `19/30`；adapter 改善 0 条、
  退化 10 条、相同 20 条。冻结 legacy 参数仍不足以保持性能，无约束 object residual 本身会显著扰动成熟策略。
- [ ] 下一轮给 object residual 增加幅度约束（固定小 scale 或有界 gate），并考虑加入保持官方动作输出的蒸馏损失。

## 3D 升级收尾

当前已从纯 2D object encoder 升级为轻量 3D object encoder。下面是还需要补齐的数据侧和验证工作。

### 数据侧

- [x] 在 LIBERO 转换脚本中从 depth + mask/bbox 提取 `target_point`。
- [x] 明确 `target_point` 坐标系：agentview 相机系 `[x, y, z]`。
- [x] 扫描完整 `local/libero_object_mask`，确认 `target_point` 无 NaN/Inf，尺度分布合理。
- [x] 重新转换完整数据集，保证每条样本包含 `target_point`。
- [x] 重新跑 `pi05_libero_object_mask` smoke 训练，确认新增 3D 条件 batch/JIT/checkpoint 正常。

2026-07-16 数据扫描：500 episodes / 74,507 frames 均包含 `target_point`，无 NaN/Inf；其中 1,260
帧（1.69%）因目标不可见、bbox 为空而使用零点回退。首次转换直接使用了 MuJoCo `[0, 1]` depth buffer，
导致非零 `z` 集中在 `0.9875-0.9914`，不满足米制尺度要求。转换脚本已改为先调用 robosuite
`get_real_depth_map`，需要重新转换后再次扫描并勾选以上两项。

2026-07-16 修复后复检通过：500 episodes / 74,507 frames 全部包含 `float32[3] target_point`，
无 NaN/Inf。1,260 帧（1.69%）使用零点回退，并与空 bbox 逐帧完全对应。非零 `z` 范围为
`0.8445-1.2347 m`，中位数 `1.0198 m`；各任务深度范围连续且尺度合理。

2026-07-16 3D smoke 训练通过：batch 中 `target_point` shape 为 `(1, 3)`，完成 2 个训练 step，
并成功写出 step 1 的 `params`、`train_state` 和 Orbax metadata。

### 暂不做的输入升级

- [ ] 暂不生成完整 3D point cloud。
- [ ] 暂不做 3D object token。
- [ ] 暂不引入 DP3 / point-wise 3D encoder。

### 暂不做的结构升级

- [ ] 暂不做 mask-to-visual-token spatial attention bias。
- [ ] 暂不把 `target_mask / target_crop` 切成 patch-level object tokens。
- [ ] 暂不让 action expert 通过 cross-attention 显式读取 object patch tokens。
- [ ] 暂不改 pi0.5 的 PaliGemma 图像/语言主干 token 流。
- [ ] 暂不替换当前轻量 MLP object encoder；如果 MLP 不够，再考虑小 CNN、共享 SigLIP crop encoder，或 object patch token encoder。

### 暂不做的 ManiFlow 风格 action backbone 升级

- [ ] 暂不把 pi0.5 action expert 改成 DiT-X。
- [ ] 暂不增加 adaptive cross-attention。
- [ ] 暂不增加 AdaLN-Zero 的 scale / shift / gate 调制。
- [ ] 暂不在 object condition 注入处做 zero-init gate；后续如果 object embedding 过强或训练不稳，再考虑 gate 从 0 初始化。
- [ ] 暂不改变 pi0.5 现有 flow matching 训练目标。
- [ ] 暂不加入 ManiFlow 风格 consistency flow training。
- [ ] 暂不为了 1-2 step action generation 改采样器；当前先沿用 pi0.5 原始 `sample_actions` 流程。

### 后续可借鉴的 ManiFlow 启发

- [ ] 如果全局 object embedding 收益有限，优先尝试保留 `mask/crop` 空间结构的 object patch tokens。
- [ ] 如果 object input 对不同动作阶段影响不稳定，考虑用 timestep + proprioception 生成 gate，动态控制 object condition 强度。
- [ ] 如果少步推理质量成为瓶颈，再研究 consistency flow training，而不是在当前 2D mask 主实验里同步修改训练目标。
- [ ] 如果要写后续工作，可以把当前方法定位为 ManiFlow/DiT-X 风格多模态 action conditioning 的轻量前置验证。
