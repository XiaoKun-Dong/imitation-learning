# pi0.5 Object Condition Step 1 执行计划

## 结论与动机

当前主线是纯 2D object-centric，不引入 point cloud，也不向 2D 配置传递 `target_point`：

```text
target_mask + target_bbox + target_crop
        |
        v
16x16 object patch tokens + geometry token
        |
        v
action expert tokens cross-attend object tokens
```

`pi05_libero_object_2d_cross_attention/2d_syncaug_bs8_5k/4999` 已完成 LIBERO-P 目标位置扰动正式评测。
三组各包含 50 个任务、seed `7/42/123`，共 450 个严格配对 episode：

| 模型 | Success | Target grasp | Grasp failure | Post-grasp failure | Wrong-object |
|---|---:|---:|---:|---:|---:|
| Official `pi05_libero` | 82.7% | 87.3% | 12.7% | 4.7% | 7.3% |
| Frozen-none | 82.7% | 87.3% | 12.7% | 4.7% | 6.7% |
| Object-2D | 80.0% | 89.3% | 10.7% | 9.3% | 8.7% |

Object-2D 相对 Official 的 success 配对结果为 `9 wins / 13 losses / 128 ties`，exact McNemar
`p=0.5235`，不能证明端到端成功率提升。但它将抓取率提高 2 个百分点；双方都成功的 111 个 episode 中，
Object-2D 平均少用 9.3 步，说明显式目标条件已被模型使用。主要回退来自 post-grasp failure 从 4.7%
增至 9.3%，表明固定强度、单次注入的 object residual 可能干扰成熟策略的搬运和放置阶段。

Step 1 不增加模态、不修改数据 schema，先解决两个最直接的问题：object patch 缺少内部交互，以及 object
residual 缺少自适应强度控制。

## 本轮目标

1. 为 object tokens 增加 2 层轻量 self-attention encoder，建模目标形状、边界及 patch 间空间关系。
2. 将固定 residual 注入改为有界 learnable gate，使模型能够减弱不必要的 object condition 干扰。
3. 保持官方 PaliGemma、action expert 和 flow-matching 目标不变，继续从官方 `pi05_libero` 初始化。
4. 用同一套 LIBERO-P 配对协议验证：抓取收益能否保留，同时 post-grasp failure 回落。

修改后的结构：

```text
[RGB, mask, x, y] object patches
        |
        v
Linear + swish
        |
        + geometry token
        |
        v
2-layer masked object self-attention encoder
        |
        v
action-token cross-attention
        |
        v
residual_scale * sigmoid(learnable_gate) * delta
```

## 非目标

- 不引入 DINO、SAM 或 GroundingDINO。
- 不引入 depth point、point cloud、force 或 tactile。
- 不改变 LeRobot dataset schema 和同步增强管线。
- 不把 object tokens 拼入 PaliGemma prefix。
- 不修改 Gemma/action expert 每层内部 attention。
- 不改变 pi0.5 flow-matching loss、采样器或 action horizon。
- 本轮不同时解冻 legacy policy 参数，避免结构变化与微调范围混杂。

## 模型改动

### 1. 配置

在 `src/openpi/models/pi0_config.py` 的 `Pi0Config` 增加：

```python
object_condition_encoder_layers: int = 0
object_condition_encoder_mlp_ratio: int = 4
object_condition_use_gate: bool = False
object_condition_gate_init: float = -4.0
```

为保持现有 Object-2D 4999 checkpoint 的架构和输出不变，公共默认值关闭 encoder/gate；独立
`pi05_libero_object_2d_step1` 配置显式设置 `encoder_layers=2`、`use_gate=True`。这也允许后续做
self-attention-only 和 gate-only 消融。

继续复用：

```python
object_condition_num_heads: int = 8
object_condition_residual_scale: float = 0.1
```

`sigmoid(-4) ~= 0.018`，初始有效上限约为 `0.1 * 0.018 = 0.0018`。该设置用于保护成熟策略，
但训练时必须记录 gate；若 5k steps 内 gate 和 output projection 都几乎不更新，应将 gate init 作为消融变量，
而不是直接增加训练步数掩盖优化问题。

### 2. Object self-attention encoder

在 `Pi0.__init__()` 中新增且统一使用 `object_condition_` 前缀：

```text
object_condition_encoder_norms
object_condition_encoder_qkv_projs
object_condition_encoder_out_projs
object_condition_encoder_mlp_in
object_condition_encoder_mlp_out
object_condition_gate
```

每层采用 pre-norm residual block：

```text
x = x + SelfAttention(LN(x), key_mask=object_token_mask)
x = x + MLP(LN(x))
x = x * object_token_mask
```

约束：

- patch valid：mask patch 非空或对应 crop patch 非零；
- invalid key 的 attention logit 置为大负值；
- 每个 block 后清零 invalid token，避免 xy coordinate 或 MLP bias 激活 padding；
- geometry token 保留，但 `has_condition=False` 时整个 object residual 必须为零；
- 避免对全 invalid mask 做 softmax 产生 NaN。

### 3. Learnable gate

将 cross-attention输出从：

```python
action_tokens + residual_scale * delta
```

改为：

```python
gate = jax.nn.sigmoid(self.object_condition_gate.value)
effective_delta = has_condition * residual_scale * gate * delta
return action_tokens + effective_delta
```

保留 `residual_scale` 作为硬上限。第一版 gate 使用全局标量，保持可解释性；暂不引入 timestep、proprioception
或 token-wise 动态 gate。训练日志至少记录：raw gate、sigmoid gate、effective scale、object residual RMS。

### 4. Checkpoint 与冻结策略

- 所有新增参数必须匹配 `PathRegex("object_condition_.*")`。
- Step 1 从官方 `/home/dongxiaokun/baseck/pi05_libero` 初始化，而不是从已训练的 4999 继续训练。
- legacy 参数保持冻结，只训练所有 `object_condition_*`。
- 新旧 object checkpoint 结构不同；加载官方 legacy checkpoint 应初始化缺失 object 参数，加载旧 object checkpoint
  应明确报结构不兼容或通过专用迁移逻辑处理，不允许静默丢参数。

## 实现清单

- [x] 在 `Pi0Config` 增加 encoder layers、MLP ratio、gate 开关和 gate init。
- [x] 在 `Pi0.__init__()` 创建 2 层 object encoder 参数。
- [x] 实现 `_encode_object_tokens(tokens, token_mask)`。
- [x] 在 patch token 与 geometry token concat 后调用 encoder。
- [x] 复用并验证 `has_condition`，确保空条件严格 no-op。
- [x] 在 object cross-attention residual 上加入 bounded learnable gate。
- [x] 将 gate、effective scale、object delta/residual RMS 和 action token RMS 接入训练 metrics。
- [x] 检查参数路径全部以 `object_condition_` 开头。
- [x] 增加独立 `pi05_libero_object_2d_step1` 配置，未覆盖当前配置。

参数审计（abstract model）：现有 2D adapter 为 `4.219M / 12 tensors`，Step 1 为
`29.411M / 37 tensors`，新增约 25.19M 参数，且全部位于 `object_condition_*`。相对完整 pi0.5 仍较小，
但服务器 smoke 需要重新确认 batch 8 的显存余量。

建议新配置名：

```text
pi05_libero_object_2d_step1
```

## 测试清单

### Forward 与 shape

- [ ] 2D `target_mask + target_bbox + target_crop` 可完成 loss 和 sample forward。
- [x] object encoder 前后 token shape、dtype 保持一致。
- [ ] Pi0 与 Pi0.5 已有无条件路径不回退。
- [ ] batch size 1 与多 batch 均可 JIT。

### Mask 与缺失条件

- [x] invalid patch 不因 xy coordinate 或 bias 变成有效 token。
- [x] 全空 mask/crop/bbox 时无 NaN。
- [x] `has_condition=False` 时 object residual 严格为零。
- [x] object-condition dropout 后 mask/crop/bbox 被同步置空。
- [x] dropout 样本输出不受 geometry token 常量影响。

### Gate

- [x] gate 初始化满足 `sigmoid(gate) ~= 0.018`。
- [x] 默认 effective scale 约为 `0.0018`。
- [x] gate 始终有界，最大注入幅度不超过 residual scale。
- [x] gate 与 output projection 在非零 residual 分支下具有有限且非零梯度。

### Freeze 与 checkpoint

- [x] trainable filter 只选择 `object_condition_*`。
- [x] 所有 encoder/gate 参数均被选中，无漏训参数。
- [x] 官方 checkpoint 可加载并初始化新增参数。
- [x] 保存、恢复 Step 1 object 参数后输出一致。

P0 gate 审计（2026-07-23）：原实现启用 gate 时错误地以 `sigmoid(gate)` 覆盖固定 residual scale，使初始
effective scale 为 `0.01799`，而不是计划中的 `0.001799`。现已修复为
`residual_scale * sigmoid(gate)`。修复前训练的 `step1_gate_5k` checkpoint 对应旧公式，不能直接作为修复后
模型的正式结果；后续 Step 1 需从官方 checkpoint 重新训练。

建议测试命令：

```bash
uv run pytest -q src/openpi/models/model_test.py \
  -k "object_condition or target or freeze_filter or gate"
```

## 训练计划

### Smoke

- [x] batch 1、2 steps，验证 batch/JIT/loss/checkpoint。
- [x] 检查 gate、residual RMS、grad norm 非 NaN。
- [ ] 检查官方参数在 step 前后逐项不变。

P0 smoke 输出位于 `checkpoints/pi05_libero_object_2d_step1/p0_metrics_smoke/1`。两步的
effective scale 均约 `0.0018`，grad norm 分别约 `0.0018/0.0019`，所有新增指标均为有限值。

### 主训练

与当前 2D 模型保持一致：

```text
初始化：official pi05_libero
数据：local/libero_object_mask 的纯 2D 字段
norm stats：official physical-intelligence/libero
batch size：8
steps：5000
seed、增强、优化器、LR schedule：与 2d_syncaug_bs8_5k 相同
EMA：保持相同设置
```

保存至少 `1000/2000/3000/4000/4999`。checkpoint 选择只使用 `_sample2` 开发集，不查看最终 `_sample1`
结果。除 success/grasp/wrong-object 外，选模时加入 post-grasp failure，防止再次选择“更容易抓取但更难放置”的版本。

## 实验矩阵

### 主对比

```text
A. Official pi05_libero
B. Frozen-none（Step 1 checkpoint，不发送 object condition）
C. Current Object-2D MLP + fixed scale（现有 4999）
D. Step 1 Object-2D self-attn + learnable gate
```

所有组共享 task IDs、seed、replan steps、episode horizon 和 norm stats。

### 诊断消融

仅当 D 相对 C 有正信号后再训练：

```text
C1. current MLP + fixed scale
C2. current MLP + learnable gate only
C3. self-attention + fixed scale only
D.  self-attention + learnable gate
```

### 评测

1. LIBERO-P `_sample2` 小开发集：选择 checkpoint。
2. LIBERO-P `_sample1` 50 tasks x seeds `7/42/123`：正式目标位移评测。
3. 原始 LIBERO Object：检查成熟策略保真。
4. `add_*` distractor：次级评测 wrong-object robustness。
5. 原三组失败并集 overlay：检查 mask 选错、丢失、漂移与 post-grasp 失败。

报告：success、target grasp、grasp failure、post-grasp failure、wrong-object、共同成功步数、配对
wins/losses/ties、exact McNemar；位置扰动按任务名中的 `level1-level5` 分层，而不是混用 metadata
`difficulty_level`。

## 成功判据

Step 1 进入下一阶段至少满足：

- [ ] Frozen-none 与 Official 保持配对等价，排除加载和配置偏差。
- [ ] 相对 Current Object-2D，post-grasp failure 明显下降。
- [ ] target grasp 的现有正信号不被完全牺牲。
- [ ] success 配对 wins 不少于 losses，且至少两个 seed 方向一致。
- [ ] Level 3-5 不再出现当前 Level 4 的集中回退。
- [ ] 共同成功任务步数不显著变差。
- [ ] mask overlay 失败诊断未发现系统性 GT mask 映射错误。

如果 self-attention + gate 仍无收益，不直接引入更多模态。下一步优先评估：解冻 action expert 末层或 LoRA、
加入保持官方动作输出的蒸馏约束、使用 timestep + proprioception 生成动态 gate。只有明确证明 2D 表示受限后，
才恢复 point cloud 路线。
