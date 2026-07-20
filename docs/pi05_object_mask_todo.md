# pi0.5 增加 2D 目标物 Mask 工程 TODO

## 当前路线

继续沿用 2D object-centric 路线：

```text
RGB 图像 + 语言指令 + 本体状态
        +
目标物 2D mask / bbox / crop
        |
        v
轻量 object encoder
        |
        v
只注入 pi0.5 action expert
```

当前阶段不引入 depth、不做点云、不升级到 3D object-centric。目标是先在 `libero_object` 上验证 2D 目标物显式条件是否提升目标定位和相似物区分能力。

## 已完成

### 模型与输入

- [x] 在 JAX `Observation` 中增加可选字段：`target_mask`、`target_bbox`、`target_crop`。
- [x] `Observation.from_dict()` 已支持从数据字典读取目标物字段。
- [x] 训练时将 `target_crop` 从 `uint8` 转为 `float32`，避免 jaxtyping 类型错误。
- [x] JAX `preprocess_observation()` 已保留目标物字段。
- [x] `target_mask` 和 `target_crop` 会 resize 到模型图像坐标系。
- [x] `target_bbox` 会按 RGB 图像相同的 resize-with-pad 几何关系同步变换。
- [x] 明确并固定 `target_bbox` 约定：绝对像素坐标 `[x1, y1, x2, y2)`，x2/y2 为 exclusive，不是 0-1 归一化坐标。
- [x] `pi0.py` 中已将原来的 6 维全局均值 object condition 替换为轻量空间 object encoder。
- [x] object embedding 只注入 action expert，不改 PaliGemma 图像/语言主干。
- [x] 增加 mask dropout，使训练在 mask 缺失或轻微错误时更稳。
- [x] 给 `target_mask`、`target_bbox`、`target_crop` 增加 shape、dtype、resize 和 bbox 约定单元测试。

### LIBERO 数据与转换

- [x] `LiberoInputs` 已支持透传 `target_mask`、`target_bbox`、`target_crop`。
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
  - `pi05_libero_object_mask`：mask + bbox + crop
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
- [ ] 训练 `pi05_libero_object_mask`：mask + bbox + crop。
- [ ] 每组使用同一个 `pi05_base` 初始化、相同训练步数、相同冻结策略。

### 评估指标

- [ ] 在 `libero_object` 上评估 5 组 ablation。
- [ ] 统计 success rate。
- [ ] 额外统计 wrong-object rate，重点看相似物体区分是否改善。
- [ ] 检查失败样例中的 mask 是否遮挡、漏检或目标物不可见。
- [ ] 对比训练前后 action 稳定性，尤其是接近抓取阶段的末端轨迹。
- [ ] 可选：把原始 `pi05_libero` 作为外部参考 baseline；主 ablation baseline 使用 `pi05_libero_object_none`。

## 暂缓升级

当前论文主线先保持轻量 2D object encoder，不把 ManiFlow 风格结构混入第一轮实验。下面内容作为后续 backlog，只有在 5 组 ablation 证明 2D object input 有稳定收益后再逐步加入。

### 暂不做的输入升级

- [ ] 暂不加入 depth。
- [ ] 暂不生成 3D point cloud。
- [ ] 暂不做 3D object token。
- [ ] 暂不引入 DP3 / point-wise 3D encoder。
- [ ] 暂不把 LIBERO 深度图升级为 3D object-centric 表示。

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
