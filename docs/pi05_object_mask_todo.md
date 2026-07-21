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
2D patch encoder + 单点 geometry encoder
        |
        v
通过 cross-attention 只注入 pi0.5 action expert token
```

当前实现是“2D object tokens + 单个 3D 代表点”，不是 point-cloud 模型：不改 PaliGemma 图像/语言主干，
把 mask/crop 下采样成 `16x16` patch token，并将 depth 派生的目标点 `[x,y,z]` 与 bbox、mask moments
组成一个 geometry token。action expert 的噪声动作 token 通过多头 cross-attention 读取这些 object token。

这里的 `target_point` 是 mask 内深度中位数配合 bbox 中心反投影得到的相机系代表点，只表达目标的大致中心位置；
它不保留目标表面形状、点级局部几何、朝向和遮挡结构。因此代码中的 `object_condition="3d"` 是历史 CLI 名称，
准确含义应理解为 `2D + single-point 3D condition`。

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
- [x] `pi0.py` 中已将原来的全局均值 object condition 替换为 `16x16` 空间 patch encoder。
- [x] bbox + mask moments + `target_point` 被编码为一个 geometry token。
- [x] action expert token 通过 8-head cross-attention 读取 object patch/geometry token，不改 PaliGemma 图像/语言主干。
- [x] cross-attention output projection 使用 zero init，初始时严格保持 legacy policy 输出不变。
- [x] object residual 使用固定 `0.1` scale 限幅。
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
- [x] `pi05_libero_object_mask` 增加冻结配置：冻结 VLA 图像/语言主干，训练 action expert 和 object encoder。
- [x] `pi05_libero_object_cross_attention` 冻结全部 legacy 参数，只训练 `object_condition_*` cross-attention/encoder 参数。
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

## 单点 3D 升级收尾

当前已从纯 2D object encoder 升级为带单点几何条件的 object encoder，但尚未引入 point cloud。

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

### 2026-07-21 代码复核

- [x] 确认训练和在线评测都使用同一套 `get_real_depth_map + mask + bbox center` 单点提取语义。
- [x] 确认当前数据 schema 只有 `target_point: float32[3]`，没有 `target_points: float32[N,3/6]`。
- [x] 确认当前 geometry encoder 只接收 bbox、mask moments 和单点 `[x,y,z]`。
- [x] 确认当前 cross-attention 的 key/value token 是 2D crop/mask patch token + 1 个 geometry token，
  不是 point-cloud token。
- [x] 确认 `pi05_libero_object_cross_attention` 只训练 `object_condition_*`，官方 pi0.5 LIBERO 参数保持冻结。
- [ ] `target_point` 当前用 bbox 中心作为 `(u,v)`，而不是 mask 像素的深度加权 3D centroid；需要作为单点 ablation 的已知近似记录。
- [ ] 相机系单点没有外参变换；Camera Viewpoints 扰动下不能直接解释为相机不变 3D 表示。

### LIBERO-P 初步验证

- [x] 接入 LIBERO-P task category、difficulty、task-id、名称过滤和确定性随机抽样。
- [x] 在 20 个 `add_*` 干扰物任务上完成配对评测：Base `18/20`，单点 3D `17/20`，未显示收益。
- [x] 在 20 个多目标、多位移等级 `_level*` 任务上完成配对评测：Base `15/20`，单点 3D `16/20`。
- [x] 目标位移组共同成功的 14 项中，单点 3D 有 12 项步数更少，平均少 `12.1` 步。
- [x] 训练集首帧目标位置近似固定：5/10 个任务无可测变化，其余主要为毫米级，最大范围约 `2 cm`；
  LIBERO-P Level 2-5 中位位移约 `4.5-9.7 cm`，可作为目标位置分布偏移测试。
- [ ] 使用 seed `7/42/123` 扩展到至少 60 个严格配对回合，并按位移 Level 1-5 报告结果。
- [ ] 当前结果只支持“已见物体/已见任务下的位置扰动鲁棒性”解释，不能宣称未见物体或新技能泛化。

## 当前优先级：先验证 2D Object-Centric

Point cloud 与单点 3D 升级暂时冻结。当前主问题不是“更多 3D 几何是否更强”，而是先证明在成熟
`pi05_libero` 策略上加入显式 2D 目标条件是否带来稳定、可复现、可解释的收益。

### 公平训练配置

- [x] 新增 `pi05_libero_object_2d_cross_attention`。
- [x] 从同一个官方 `pi05_libero` checkpoint 初始化，而不是从未做 LIBERO 微调的 `pi05_base` 初始化。
- [x] 复用官方 `physical-intelligence/libero` norm stats。
- [x] 冻结全部 legacy policy 参数，只训练 `object_condition_*` encoder/cross-attention。
- [x] 2D 训练输入严格限制为 `target_mask + target_bbox + target_crop`，不 repack `target_point`。
- [x] 在线评测增加 `--args.object-condition 2d`，不向 server 发送 `target_point`。
- [ ] Base 与 2D 使用完全相同的 task IDs、rollout seed、replan steps 和 episode horizon。

旧的 `pi05_libero_object_mask_only/crop_only/bbox_only` 从 `pi05_base` 初始化并解冻 action expert，不能直接与
官方 `pi05_libero` 组成公平主实验；它们只保留为早期探索配置，不用于核心结论。

### 2D 优点假设

- [ ] **目标定位**：目标发生平面位移时，mask/bbox 提供明确的新位置，降低搜索和接近阶段失败。
- [ ] **抗干扰物**：增加相似或无关物体时，显式 mask/crop 降低 wrong-object grasp。
- [ ] **动作效率**：共同成功任务中减少接近目标前的无效动作和总完成步数。
- [ ] **位移退化更慢**：随 LIBERO-P target displacement Level 1-5 增大，2D 的成功率下降斜率小于 Base。
- [ ] **无扰动不退化**：原始 LIBERO Object 上性能应接近官方 Base，证明 residual 没有破坏成熟策略。

### 最小证据链

- [ ] 训练 `pi05_libero_object_2d_cross_attention`，先保存 `250/500/750/999` 四个 checkpoint。
- [ ] 用原始 LIBERO Object 小样本筛选 checkpoint，淘汰明显破坏 Base 行为的版本。
- [ ] 主评测 A：LIBERO-P `_level*` 目标位移，覆盖多目标和 Level 1-5。
- [ ] 主评测 B：LIBERO-P `add_*` 干扰物增加，覆盖多目标而非连续单任务。
- [ ] 保真评测：原始 LIBERO Object，确认 2D 模型相对官方 Base 的性能保持。
- [ ] 至少使用 seed `7/42/123`，每组累计不少于 60 个严格配对回合。
- [ ] 报告 success、target grasp、post-grasp failure、wrong-object grasp 和成功步数。
- [ ] 使用配对 task-level bootstrap CI 或 exact McNemar，而不是只比较点估计。

### 必要消融

- [ ] 2D combined：mask + bbox + crop，作为当前主模型。
- [ ] Mask + bbox：去掉 RGB crop，判断收益来自位置还是目标外观。
- [ ] Crop only：判断目标外观提示是否足够。
- [ ] Oracle mask corruption：平移/膨胀/随机漏检，测试对分割误差的敏感性。
- [ ] Single-point 3D 只作为附加参考，不作为当前主线，也不继续升级结构。

### 成功判据

- [ ] 原始 LIBERO Object 不出现明显性能退化。
- [ ] 在至少两个 seed 上，目标位移或干扰物任务的配对成功率方向一致。
- [ ] wrong-object grasp 或 target-grasp failure 至少一项稳定下降。
- [ ] 共同成功任务的完成步数不显著变差。
- [ ] 优势能够对应到预先声明的失败类型，而不是只依赖少数偶然 rollout。

## Point Cloud 下一阶段（冻结）

只有在上述 2D 证据链完成、且能明确指出 2D 表示的失败边界后，才恢复以下工作。

### 数据表示

- [ ] 从 agentview metric depth + target mask 反投影目标表面所有有效像素，而不是只取一个 bbox 中心点。
- [ ] 固定采样 `N=128` 或 `N=256` 个点，优先使用 farthest-point sampling；点不足时 padding 并提供 validity mask。
- [ ] 第一版点特征使用相机系 `XYZ`；第二版 ablation 再加入对应像素 `RGB`，形成 `XYZRGB`。
- [ ] 对点云做以目标 centroid 为中心的局部归一化，同时单独保留全局 centroid，避免丢失绝对抓取位置。
- [ ] 明确相机系与机器人 base/world 系两套方案；若测试 Camera Viewpoints，优先用相机外参转换到机器人 base 系。
- [ ] LeRobot schema 增加 `target_points: float32[N,3/6]` 和 `target_points_mask: bool[N]`。
- [ ] 重新转换数据并扫描 NaN/Inf、有效点数、空间范围、padding 比例和训练/评测坐标系一致性。

### 模型结构

- [ ] 增加轻量 PointNet/point-wise MLP：逐点编码后保留 `K` 个 point token，不只做全局 max pooling。
- [ ] 将 point token 与现有 2D patch token、geometry token 拼接，复用 action-token cross-attention。
- [ ] 为 2D patch、point、geometry 增加 modality/type embedding，避免三种 token 语义混淆。
- [ ] 保持 output projection zero init 和 residual scale，确保从官方 checkpoint 初始化时仍是严格 no-op。
- [ ] 第一阶段继续冻结 legacy policy，只训练 point encoder 与 object cross-attention；确认稳定后再考虑解冻 action expert 顶层。

### Point Cloud 对照实验

- [ ] Base：官方 `pi05_libero`，无 object condition。
- [ ] 2D：mask/crop patch + bbox/mask moments，不使用任何 depth。
- [ ] Single-point 3D：当前实现，2D + centroid point。
- [ ] Point cloud XYZ：2D + `N x XYZ`。
- [ ] Point cloud XYZRGB：2D + `N x XYZRGB`。
- [ ] 所有组共享 checkpoint、norm stats、训练样本数、seed、冻结策略和评测 task IDs。
- [ ] 主评测使用 LIBERO-P target displacement Level 1-5；辅助评测使用 Objects Layout distractors 和 Camera Viewpoints。
- [ ] 除 success 外报告 target-grasp、post-grasp failure、wrong-object grasp、成功步数和按位移等级退化曲线。

### 当前仍不修改

- [ ] 暂不改 pi0.5 的 PaliGemma 图像/语言主干 token 流。
- [ ] 暂不做 mask-to-PaliGemma-visual-token spatial attention bias。
- [ ] 暂不引入 DP3 的完整 policy/action backbone；先只借鉴 point encoder。

### 暂不做的 ManiFlow 风格 action backbone 升级

- [ ] 暂不把 pi0.5 action expert 改成 DiT-X。
- [ ] 暂不增加 adaptive cross-attention。
- [ ] 暂不增加 AdaLN-Zero 的 scale / shift / gate 调制。
- [x] object cross-attention output projection 已 zero init，并使用固定 `0.1` residual scale。
- [ ] 暂不增加可学习 AdaLN-Zero gate；若固定 scale 仍不稳，再考虑 gate 从 0 初始化。
- [ ] 暂不改变 pi0.5 现有 flow matching 训练目标。
- [ ] 暂不加入 ManiFlow 风格 consistency flow training。
- [ ] 暂不为了 1-2 step action generation 改采样器；当前先沿用 pi0.5 原始 `sample_actions` 流程。

### 后续可借鉴的 ManiFlow 启发

- [x] 已保留 `mask/crop` 的 `16x16` 空间 patch token，并由 action token cross-attention 读取。
- [ ] 如果 object input 对不同动作阶段影响不稳定，考虑用 timestep + proprioception 生成 gate，动态控制 object condition 强度。
- [ ] 如果少步推理质量成为瓶颈，再研究 consistency flow training，而不是在当前 2D mask 主实验里同步修改训练目标。
- [ ] 如果要写后续工作，可以把当前方法定位为 ManiFlow/DiT-X 风格多模态 action conditioning 的轻量前置验证。
