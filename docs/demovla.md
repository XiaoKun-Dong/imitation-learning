# DemoVLA: Self-Grounded Interaction Memory for Precise VLA Control

## 1. 文档状态

- 工作名称：DemoVLA
- 基础模型：OpenPI `pi0.5`
- 当前阶段：LIBERO 仿真因果定位；Stage A fixed-0.03 已完成全套件配对评测，正在验证关键 replan 的内容特异作用
- 仿真结果总览：[DemoVLA 仿真实验结果总览与证据审计](demovla_simulation_results_status_20260831_zh.md)
- 阶段 A/B1/S1：[固定门控预热、readout 与语义排序实验](demovla_stage_a_causal_evaluation_zh.md)
- 局部行为因果：[关键 Replan 因果定位](demovla_key_replan_causal_progress_zh.md)
- 最新因果测评：[DemoVLA × Norm Stats 严格因果测评报告](demovla_norm_stats_causal_evaluation_20260824.md)
- Diversity 结果：[DemoVLA Sparse-Deep Diversity 10k 训练与评估报告](demovla_sparse_deep_diversity_10k.md)
- 无 diversity 基线：[DemoVLA Sparse-Deep 30k 训练报告](demovla_sparse_deep_training_report.md)
- 核心约束：
  - 不引入 SAM、DINO、GroundingDINO 等额外视觉模型。
  - 不要求 mask、bbox、crop、point 或 affordance 人工标注。
  - 推理输入保持为原始 RGB、语言指令和机器人状态。
  - 保留原始 VLM prefix 到 action expert 的信息通路。
  - 新增模块应支持从现有 `pi0.5` checkpoint 稳定初始化。

### 1.1 已完成

- 4-query、语言条件、空间与相机感知的 Interaction Token Extractor。
- 在 Action Expert 第 4、9、14 层执行共享 adapter、独立 gate 的 sparse-deep injection。
- Interaction memory 每次 replan 只提取一次，并在完整 flow denoising loop 中缓存。
- Attention diversity loss 和 memory diversity loss，以及对应训练日志。
- 以 replan 为时间单位的 query × camera patch attention 可视化。
- Policy server、websocket diagnostics、LIBERO rollout、视频和 episode metrics 链路。
- Dynamic gate 已训练到 29999 step，并通过固定 flow noise 的逐 episode 配对消融。
- Dynamic gate 在官方口径 LIBERO Object 评估中取得 495/500，成功率 99.0%。
- Stage A `4999` 在 fixed gate `0.03` 下取得 395/400；相对 injection off 为
  `+2.25pp`、McNemar `p=0.02246`，相对 matched zero-memory 为 `+1.00pp`、
  `p=0.2891`，因此后者尚未建立总体行为显著性。
- 关键 replan 已得到一个严格共享前缀的局部内容因果样本；跨 episode/跨 server
  稳健复现仍未完成。
- `demovla_libero_sparse_deep_diverse` adapter-only 训练到 10k。
- 10-task、每任务 1 次、seed 7 的 `libero_object` pilot：9/10 成功，0 次目标
  抓取失败，0 次错物抓取，1 次 post-grasp failure。
- 10-task、每任务 5 次的正式评估：36/50 成功，成功率 72.0%，Wilson 95% CI
  为 58.3%～82.5%。
- 历史评估中与无 diversity 10k 对齐前 5 个 initial states：36/50 对 35/50，
  McNemar `p=1.0`；但当时没有逐 replan 固定 flow noise，需要按新确定性协议
  重跑后才能作为严格因果比较。

### 1.2 Dynamic Gate LIBERO 正式评估

最终使用 checkpoint：

```text
demovla_libero_sparse_deep_dynamic_gate/dynamic_gate_v1/29999
```

评估遵循 OpenPI 官方 LIBERO Object rollout 设置：

```text
OffScreenRenderEnv
environment render size = 256
model input size = 224
rotate agentview and wrist images by 180 degrees
wait steps = 10
replan steps = 5
trials per task = 50
interaction diagnostics = off
flow noise = stateful policy RNG
```

在 10 个任务、每任务 50 次、共 500 episodes 上：

| 任务 | 成功数 | 成功率 | 失败 episode |
|---|---:|---:|---|
| Alphabet soup | 47/50 | 94.0% | 1, 27, 34 |
| Cream cheese | 50/50 | 100.0% | - |
| Salad dressing | 50/50 | 100.0% | - |
| BBQ sauce | 50/50 | 100.0% | - |
| Ketchup | 49/50 | 98.0% | 48 |
| Tomato sauce | 49/50 | 98.0% | 6 |
| Butter | 50/50 | 100.0% | - |
| Milk | 50/50 | 100.0% | - |
| Chocolate pudding | 50/50 | 100.0% | - |
| Orange juice | 50/50 | 100.0% | - |
| **总计** | **495/500** | **99.0%** | **5** |

成功 episode 的平均完成步数为 `149.4`，中位数为 `145`，范围为
`118～273`。5 个失败 episode 均运行到 `290` step 上限，没有评测脚本提前
异常。总体成功率的 Wilson 95% CI 为 `97.68%～99.57%`。官方公开的
`pi0.5 @ 30k` LIBERO Object 结果为 98.2%；当前差异不足以宣称显著提升，
但可以确认 dynamic gate 没有造成 rollout 性能退化。

大规模评估前完成了两阶段退化检查：

1. 固定逐 replan flow noise 的 30-episode 配对：dynamic 与官方 pi0.5 均为
   `29/30`，逐 episode 结果完全相同，McNemar `p=1`。
2. 官方 stateful RNG 的 100-episode dynamic-only 评估：`98/100`；仅 tomato
   sauce 的 episode 4、6 失败。

同一个 dynamic checkpoint 的固定-noise 注入消融结果为：

```text
dynamic       29/30  96.67%
injection_off 13/30  43.33%
pi05_official 29/30  96.67%
```

`injection_off` 相对官方参考的配对差异显著，McNemar `p=0.000145`。后续
[DemoVLA × Norm Stats 严格因果测评](demovla_norm_stats_causal_evaluation_20260824.md)
确认该对照同时受到 local norm stats 影响：100-episode 固定-noise 矩阵中，
`off + official stats` 为 `99/100`，`off + local stats` 为 `42/100`，
`dynamic + official stats` 为 `19/100`，`dynamic + local stats` 为 `100/100`。
因此原结果证明的是 dynamic injection 对当前 local-stats checkpoint 的必要性和补偿
作用，不能单独解释为 interaction 架构相对正确配置的官方 pi0.5 backbone 的纯增益。

此前 gate 机制消融的 10-episode pilot 为：

```text
dynamic      10/10
layer_mean   10/10
injection_off 3/10
static_deep   0/10
```

因此正式结果证明的是完整 dynamic-gate DemoVLA 有效且不退化；由于
`dynamic` 与 `layer_mean` 在小样本中均触及天花板，尚不能单独证明动态门优于
逐层平均固定门。

### 1.3 尚未完成

- 在第二个独立 episode 或独立 server 轨迹上复现 integrated-auto 的局部内容翻转。
- 将 B1、S1 和 lockstep 原始输出从评测节点恢复到可审计归档。
- 将 diverse 10k 扩展至每任务 10 次，缩小置信区间。
- 若保留 3k checkpoint，则完成 3k/10k 对比；同时补齐 single-shot、vanilla
  和 parameter-matched baseline 的相同 initial states 对比。
- Attention grounding、query 分工和局部 collapse 的定量分析。
- Layer 9 单层注入、query 数量、空间编码与参数共享消融。
- Distractor、目标位置、背景、相机、遮挡和光照鲁棒性实验。

## 2. 研究动机

现有 VLA 通常让 action expert 直接读取完整的视觉语言 prefix。完整 prefix 提供了丰富的全局上下文，但也包含背景、干扰物和与当前交互无关的视觉信息。对于抓取、插入、旋转和精确放置等任务，action expert 需要快速定位少量与动作直接相关的区域和关系。

一种直接方案是使用外部分割模型生成目标 mask，再将 mask、crop 或目标特征输入 action expert。但这种方案存在明显的系统成本：

- 需要额外部署强视觉模型。
- 增加推理延迟、显存占用和工程依赖。
- 分割错误会直接传播到动作策略。
- 即使 GT mask 实验有效，也不能保证实际分割器的收益能够覆盖计算成本。

DemoVLA 因此研究以下问题：

> 能否只使用 VLA 自身的视觉语言表示，学习少量任务相关的 interaction tokens，并让 action expert 在不同网络深度迭代读取这些 token，从而提高精细操作能力？

## 3. 核心假设

1. 原始 VLM prefix 中已经包含目标物体、目标区域和交互关系所需的信息，问题主要在于 action expert 缺少一个稀疏、明确的检索瓶颈。
2. interaction tokens 不应替代完整 prefix，而应作为动作相关信息的快捷通路。
3. 在 action expert 入口只融合一次条件可能会发生信息稀释；在少数中间层重复读取同一份 interaction memory 更合理。
4. 不同深度的 action hidden states 表示不同阶段的运动信息，因此它们应使用当前 hidden state 作为 Query，重新检索 interaction memory。
5. 新分支必须接近 no-op 初始化，避免破坏预训练 `pi0.5` 的动作分布。

## 4. 总体架构

```text
RGB + Language + Robot State
             |
             v
    SigLIP + PaliGemma Prefix
             |
       +-----+------------------------------+
       |                                    |
       v                                    v
Full Prefix KV Cache             Contextualized Visual Tokens
       |                                    |
       |                          Interaction Token Extractor
       |                                    |
       |                        Sparse Interaction Memory
       |                       (4-16 tokens, per replan)
       |                                    |
       v                                    |
Noisy Action Embeddings                     |
       |                                    |
       v                                    |
Action Expert Layers 0 ... k  <--------------+
       |
       +-- Gated Cross-Attention Injection #1
       |
Action Expert Layers k+1 ... m
       |
       +-- Gated Cross-Attention Injection #2
       |
Action Expert Layers m+1 ... n
       |
       +-- Gated Cross-Attention Injection #3
       |
       v
Flow Velocity -> Action Chunk
```

action expert 始终保留两条观测路径：

```text
原始路径：Action Expert <- Full VLM Prefix KV
增强路径：Action Hidden <- Sparse Interaction Memory
```

## 5. Interaction Token Extractor

### 5.1 输入

提取器只使用当前模型已有的信息：

- contextualized visual patch tokens；
- contextualized language tokens；
- patch 对应的二维位置编码；
- 可选的相机 ID embedding；
- 可选的机器人 state token。

不读取外部目标条件。

### 5.2 Learnable Queries

第一版建议使用 4 个 learnable queries：

```text
q_interaction_0
q_interaction_1
q_interaction_2
q_interaction_3
```

这些 query 不施加人工语义监督。可以在分析时将其解释为 object、goal、contact 和 context，但实现中不应假设固定角色，避免错误限制模型。

### 5.3 两阶段检索

先让 query 读取语言，再读取空间视觉 token：

```python
queries = queries + text_cross_attention(
    query=layer_norm(queries),
    key_value=language_tokens,
)

interaction_tokens = queries + visual_cross_attention(
    query=layer_norm(queries),
    key_value=visual_tokens + spatial_position_embedding,
)
```

推荐增加一个轻量 FFN：

```python
interaction_tokens = interaction_tokens + ffn(layer_norm(interaction_tokens))
```

### 5.4 空间信息

interaction tokens 必须能够区分相同物体在不同位置的情况。每个视觉 patch 至少加入：

```text
normalized x/y position
camera identity
```

不使用随 viewport 宽度变化的动态编码。位置编码应与 SigLIP patch grid 一一对应。

### 5.5 Token 数量

第一轮建议：

```python
num_interaction_tokens = 4
```

后续消融：

```text
1 / 4 / 8 / 16 tokens
```

token 太少可能丢失 source-target 关系，太多则会削弱稀疏瓶颈并增加深层 cross-attention 成本。

## 6. Sparse Deep Injection

### 6.1 单层更新

设第 `l` 层 action hidden states 为 `h_l`，interaction memory 为 `z_int`：

```text
delta_l = CrossAttention(
    Q = LN(h_l),
    K = z_int,
    V = z_int
)

h_l' = h_l + sigmoid(g_l) * W_out(delta_l)
```

其中：

- `z_int` 在每次 policy replan 中只计算一次，并在完整 action flow 循环中复用。
- `g_l` 是注入层独立的可学习 gate。
- `W_out` 使用小值或零值初始化。
- 更新只作用于 action token，不修改 prefix token。

### 6.2 注入层

若 action expert 深度为 18 层，首选配置：

```python
interaction_injection_layers = (4, 9, 14)
```

选择中前、中间和中后层，而不是只在 embedding 后注入。

需要保留以下消融：

```text
single-shot: action expert 之前注入一次
sparse-deep: 在 3 个中间层注入
every-layer: 每层注入
```

### 6.3 参数共享

第一版建议三个注入位置共享：

- interaction K projection；
- interaction V projection；
- action Q projection；
- output projection。

scalar-gate 基线中每层只保留独立 gate：

```python
interaction_layer_gates[layer_index]
```

这样可以：

- 控制参数量；
- 避免不同 adapter 学出完全不一致的 memory space；
- 通过 gate 大小分析不同深度对 interaction memory 的依赖。

如果共享参数明显限制性能，再消融独立 Q projection 或独立 adapter。

### 6.4 Dynamic Gate

dynamic gate 保持 interaction cross-attention 不变，只把每层全局标量 gate
替换为 layer-wise、action-slot-wise、flow-time-aware 的小型控制器：

```text
gate_input = concat(
    LayerNorm(action_hidden),
    action_slot_embedding,
    pi0.5_flow_time_embedding,
    injection_layer_embedding,
)
gate = sigmoid(Linear(SiLU(Linear(gate_input))))
hidden = hidden + gate * interaction_delta
```

对于默认的 3 个注入层和 `action_horizon=10`，每层输出
`[batch, 10, 1]`，而不是一个标量。interaction memory 仍在每个 replan
只计算一次；动态变化只发生在 Action Expert 的读取和注入强度上。

初始化继续满足 base-model no-op：

```text
interaction output projection kernel/bias = 0
dynamic gate output kernel = 0
dynamic gate output bias = -4
```

因此初始 gate 约为 `0.018`，且实际注入严格为零。

### 6.5 与 Flow Timestep 的边界

DemoVLA 不修改 `pi0.5` 原始 flow timestep 编码和 AdaRMS 路径。
Interaction Token Extractor 仍不读取 flow timestep；dynamic gate 仅复用
`embed_suffix()` 已经生成的 `adarms_cond`，让 Action Expert 对固定 interaction
memory 的读取强度能够随去噪阶段变化。

## 7. Attention 方向

DemoVLA 使用严格单向的信息流：

```text
VLM Prefix -> Interaction Extractor
Interaction Memory -> Action Expert
VLM Prefix -> Action Expert
```

禁止：

```text
Action Expert -> VLM Prefix
Action Expert -> Interaction Extractor
```

这样可以避免 action 信息泄漏到视觉语言表征，并保持 prefix KV cache 可复用。

## 8. 训练流程

### 8.1 基础目标

基础版本只使用原始 flow-matching action loss。检测到 query collapse 后，
diverse 配置加入 attention 和 memory 两个轻量正则：

```text
L_total = L_action
        + lambda_attn * L_attention_diversity
        + lambda_memory * L_memory_diversity
```

不引入：

- mask loss；
- bbox loss；
- affordance heatmap loss；
- object classification loss；
- 3D reconstruction loss。

这样可以直接回答 interaction tokens 是否能由动作监督自发形成。

### 8.2 Query Diversity Regularization

对 head-mean visual attention 和最终 interaction memory 分别做 L2
归一化。对于任意 query pair，仅惩罚 cosine similarity 超过 margin 的部分：

```text
L_pair(x) = mean_{i != j} relu(cos(x_i, x_j) - margin)^2

L_attention_diversity = L_pair(visual_attention)
L_memory_diversity = L_pair(interaction_memory)
```

margin 默认设为 `0.5`，允许多个 query 对目标物体保留必要重叠，只抑制高度
相似的塌缩。基础 sparse-deep 配置保持两个权重为 `0`，diverse 配置启用：

```text
lambda_attn   = 1e-3
lambda_memory = 1e-4
```

训练日志额外保存：

```text
demovla_flow_loss
demovla_attention_diversity_loss
demovla_attention_pair_cosine_mean
demovla_attention_pair_cosine_max
demovla_memory_diversity_loss
demovla_memory_pair_cosine_mean
demovla_memory_pair_cosine_max
demovla_diversity_regularization
```

`loss_action_first/middle/last` 仍只表示 action flow loss，不包含 diversity
regularization，便于与旧实验直接比较。

### 8.3 参数训练策略

建议分两阶段验证：

#### Adapter-only

冻结原始 `pi0.5`，只训练：

```text
interaction_token_extractor_*
interaction_cross_attention_*
interaction_layer_gates_*
```

用于验证新模块能否在不修改基础策略的情况下带来收益。

#### Partial fine-tuning

在 adapter-only 有正收益后，解冻 action expert，保持 VLM backbone 冻结。

不建议第一轮直接全量 fine-tuning，否则相同训练数据上的提升可能来自基础模型适配，而不是 interaction architecture。

## 9. 推理与缓存

推理时：

1. 运行一次 VLM prefix forward。
2. 保存完整 prefix KV cache。
3. 缓存 contextualized prefix hidden states。
4. 从当前观测提取一次 interaction memory。
5. action expert 在指定层读取 interaction memory。

```python
prefix_hidden, prefix_kv = encode_prefix(observation)
interaction_memory = extract_interaction_tokens(prefix_hidden)

for flow_step in range(num_steps):
    actions = action_expert(
        noisy_actions=actions,
        prefix_kv=prefix_kv,
        interaction_memory=interaction_memory,
        injection_layers=(4, 9, 14),
)
```

禁止在 flow loop 中重复运行 SigLIP、VLM prefix 或 Interaction Token
Extractor。新观测只在下一次 replan 时重新编码。

### 9.1 确定性 Rollout Noise

配对消融可以让每次 replan 使用以下稳定标识生成显式 Gaussian flow noise：

```text
(policy_noise_seed, benchmark_task_id, episode_idx, replan_idx)
```

显式启用时：

```text
policy_noise_seed = 0
```

相同 checkpoint、observation 和四元组会获得逐元素一致的 noise；episode
执行长短不会改变后续 episode 的 noise 序列。官方性能评估默认不传该参数，
使用 `None` 和 policy server 的 stateful RNG：

```bash
--args.policy-noise-seed None
```

请求通过保留字段 `_openpi_flow_noise_seed` 传递 seed components。
`Policy.infer()` 在 observation transforms 前移除该字段，根据模型自身的
`action_horizon` 和 `action_dim` 生成 noise，因此不会污染模型输入，也不需要
在 client 硬编码 action shape。

Diagnostics on/off 现在都使用标准 `sample_actions()` 生成动作。开启 diagnostics
时，attention 由额外的 diagnostics-only forward 计算，不再切换到另一条动作
采样图。因此 diagnostics 会增加分析模式的计算量，但不会有意改变 action；
正式性能评估仍应关闭 diagnostics。

## 10. 与现有工作的区别

### ACoT-VLA

ACoT 的 IAR 从每层 VLM KV 中提取全局 latent action prior，并在 action expert 前融合一次。DemoVLA：

- 保留视觉 patch 的空间结构；
- 使用语言条件检索少量 interaction tokens；
- 在 action expert 多个深度迭代读取；
- 不生成额外 coarse trajectory。

### AffordanceVLA

AffordanceVLA 使用 Which2Act、Where2Act、How2Act 和多阶段 affordance supervision。DemoVLA：

- 不使用 affordance annotation；
- 不使用外部生成或分割模型；
- 不显式预测 mask、heatmap 或 3D shape；
- 研究 action loss 能否形成内部 interaction bottleneck。

### PALM

PALM 面向长时程 affordance 和 progress estimation。DemoVLA 首先聚焦短动作块中的精细控制和抗干扰能力，不预测 subtask progress。

### LiLo-VLA

LiLo-VLA 使用外部物体位姿、motion planning、技能拆分和恢复模块。DemoVLA 保持端到端 VLA 输入输出接口，不依赖外部规划系统。

## 11. 工程修改位置

### `src/openpi/models/pi0.py`

保持原版 `pi0.5` 的 flow timestep、prefix cache 和 Action Expert 行为，不在
该文件中加入 DemoVLA 专用逻辑。当前实现通过独立模型文件继承 `Pi0`，避免
DemoVLA 实验污染基础模型。

### `src/openpi/models/demovla.py`

已实现：

```text
DemoVLAConfig
language-conditioned interaction queries
spatial and camera-aware visual retrieval
single-shot / sparse-deep injection
shared adapter with per-layer gates
optional layer/slot/flow-time-aware dynamic gate
attention and memory diversity losses
training auxiliary metrics
replan-level interaction diagnostics
```

主要配置项位于 `DemoVLAConfig`：

```python
use_interaction_memory: bool = True
num_interaction_tokens: int = 4
interaction_num_heads: int = 8
interaction_injection_mode: Literal["single_shot", "sparse_deep"] = "sparse_deep"
interaction_injection_layers: tuple[int, ...] = (4, 9, 14)
interaction_gate_init: float = -4.0
interaction_gate_mode: Literal["scalar", "dynamic"] = "scalar"
interaction_dynamic_gate_hidden_dim: int = 256
interaction_dynamic_gate_embedding_dim: int = 64
interaction_adapter_share_weights: bool = True
interaction_attention_diversity_weight: float = 0.0
interaction_attention_diversity_margin: float = 0.5
interaction_memory_diversity_weight: float = 0.0
interaction_memory_diversity_margin: float = 0.5
```

### `src/openpi/models/gemma.py`

已为 Transformer block loop 增加通用 `layer_adapter` hook：

```python
layer_adapter: Callable | None
```

DemoVLA 在指定层通过该 hook 更新 Action Expert token group，并可返回逐层
adapter diagnostics。`gemma.py` 不依赖 DemoVLA 模块，也不理解 interaction
memory 的具体语义。

### `scripts/train.py`

已支持检测 `compute_loss_with_aux()`，将 total loss 用于反向传播，并分别记录
flow loss、diversity loss 和 query cosine 指标。Action first/middle/last 会减去
diversity regularization，保持为纯 action flow loss。

### `src/openpi/training/weight_loaders.py`

`CheckpointWeightLoader` 已支持配置 `missing_regex`，允许从基础 `pi0.5`
checkpoint 加载已有参数，并为新增的 `demovla_.*` 参数保留模型初始化值。

### `src/openpi/training/config.py`

已新增：

- adapter-only freeze filter；
- DemoVLA 训练配置；
- single-shot 和 sparse-deep 消融配置；
- `demovla_libero_sparse_deep_diverse` 在 sparse-deep 基础上启用
  attention/memory diversity regularization。
- `demovla_libero_sparse_deep_dynamic_gate` 在相同 diversity 设置上启用
  dynamic gate，并复用相同数据的 norm stats。

当前 diverse 配置使用：

```text
attention diversity weight = 1e-3, margin = 0.5
memory diversity weight    = 1e-4, margin = 0.5
```

两个正则只惩罚 query pair cosine 超过 margin 的部分，允许不同 query 对目标
区域保留必要的重叠，而不是强制完全正交。

八卡训练示例：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
uv run python scripts/train.py demovla_libero_sparse_deep_diverse \
  --exp-name diverse_v1 \
  --fsdp-devices 4 \
  --batch-size 128
```

这里的 `batch_size=128` 是 global batch size。八张可见卡和
`--fsdp-devices 4` 形成两个 data-parallel group，每组进行四卡 FSDP。

dynamic gate 训练时将配置名替换为：

```text
demovla_libero_sparse_deep_dynamic_gate
```

### 推理、诊断与可视化

以下文件已完成端到端诊断链路：

```text
src/openpi/models/demovla_visualization.py
src/openpi/policies/policy.py
src/openpi/policies/libero_policy.py
scripts/serve_policy.py
examples/libero/main.py
```

Policy server 可以选择返回 interaction diagnostics；LIBERO rollout 按 episode
保存视频、`metrics.jsonl` 和每次 replan 的 query × camera patch attention 图。

## 12. 实验矩阵

### 12.1 必需基线

| ID | 模型 | 完整 Prefix | Interaction Memory | 注入方式 |
|---|---|---:|---:|---|
| B0 | Vanilla pi0.5 | 是 | 否 | 无 |
| B1 | Parameter-matched MLP | 是 | 否 | MLP adapter |
| B2 | Global pooled prefix | 是 | 是 | single-shot |
| B3 | ACoT-style layer pooling | 是 | 是 | single-shot |
| M1 | DemoVLA | 是 | 是 | single-shot |
| M2 | DemoVLA | 是 | 是 | sparse-deep |
| M3 | DemoVLA bottleneck-only | 否 | 是 | sparse-deep |

### 12.2 关键消融

```text
interaction token 数量：1 / 4 / 8 / 16
注入层数：1 / 3 / every layer
共享 adapter / 独立 adapter
有 / 无二维位置编码
有 / 无语言条件检索
adapter-only / action expert fine-tuning
```

### 12.3 鲁棒性评估

即使不使用外部 mask，也应构造视觉干扰：

```text
背景替换
增加同类别 distractor
目标位置变化
相机轻微偏移
光照变化
局部遮挡
纹理变化
```

重点报告：

- 平均任务成功率；
- 抓取或接触成功率；
- 放置终点误差；
- 干扰物选择错误率；
- 不同扰动强度下的性能曲线；
- 参数增量；
- 单步延迟和完整 action chunk 延迟；
- 训练显存和推理显存。

## 13. 可解释性分析

虽然不监督 mask，仍可可视化每个 interaction query 对视觉 patch 的 attention map。

需要观察：

- query 是否集中在被操作物体；
- source 和 goal 是否由不同 query 覆盖；
- action expert 不同层是否读取不同 query；
- distractor 出现后 attention 是否保持稳定。

这些可视化只用于分析，不应被表述为显式 object segmentation。

当前调试接口可以为每次 policy inference 导出：

```text
interaction_visual_attention
    [interaction_queries, camera_views, image_patches]
interaction_top_patch_view_indices
interaction_top_patch_xy
interaction_top_patch_weights
interaction_camera_mask
```

rollout 可视化以 replan 为时间单位。每次 policy inference 直接导出本次
观测对应的 attention，并生成一张 query × camera 网格图：

- 红色热力图表示完整 patch attention；
- 黄色框表示全相机联合 top-k patch；
- 框标签为 patch 排名和 attention weight；
- 标题包含 query 编号、相机名、replan 编号和对应环境步；
- 相邻图片之间机器人已经执行了 `replan_steps` 个动作，因此使用不同观测。

启动带诊断输出的 policy server：

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/serve_policy.py \
  --env LIBERO \
  --interaction-diagnostics \
  policy:checkpoint \
  --policy.config demovla_libero_sparse_deep_diverse \
  --policy.dir checkpoints/demovla_libero_sparse_deep_diverse/<exp>/<step>
```

运行 LIBERO rollout 并保存 patch 可视化：

```bash
MUJOCO_GL=egl \
MUJOCO_EGL_DEVICE_ID=0 \
CUDA_VISIBLE_DEVICES=0 \
uv run python examples/libero/main.py \
  --args.task-suite-name libero_object \
  --args.num-trials-per-task 1 \
  --args.seed 7 \
  --args.policy-noise-seed 0 \
  --args.visualize-interaction-patches \
  --args.interaction-visualizations-per-episode -1 \
  --args.video-out-path data/libero/videos/demovla_diverse_debug
```

图片保存在：

```text
data/libero/videos/demovla_diverse_debug/interaction_patches/
└── task_0001_<task_name>_episode_00/
    ├── interaction_replan_000_step_0000.png
    ├── interaction_replan_001_step_0005.png
    └── interaction_replan_002_step_0010.png
```

每个 episode 使用一个独立文件夹，文件夹内按 replan 编号保存图片，不会为
每个 replan 创建额外子目录。

`--args.interaction-visualizations-per-episode -1` 表示保存全部 replan；设为正数
时只保存 episode 开头指定数量的 replan。

## 14. 主要风险

### 风险 1：Interaction branch 与原始 prefix 冗余

action expert 本来就能读取完整 prefix。必须通过 parameter-matched baseline 和 global pooling baseline 证明收益来自稀疏、任务相关的空间检索。

### 风险 2：Query collapse

多个 query 可能关注同一区域。当前 diverse 配置使用带 margin 的 attention
diversity 和 memory diversity 正则，并持续记录 pairwise cosine；需要结合
rollout 成功率确认去塌缩没有迫使 query 转向无关背景。

### 风险 3：缺少显式监督导致定位不稳定

动作 loss 可能只学习数据集偏差。需要通过目标位置变化、背景变化和 distractor 测试判断是否真正形成 interaction grounding。

### 风险 4：深层注入破坏预训练策略

使用接近零的 gate 初始化，并确保关闭 DemoVLA 时输出严格退化为原始 `pi0.5`。

### 风险 5：修改 Gemma 增加维护成本

当前只在 Gemma layer loop 中保留一个与 DemoVLA 解耦的通用
`layer_adapter` hook。后续需要持续用原始模型测试确认 `layer_adapter=None`
时行为不变，避免 Gemma 接口继续扩张。

## 15. 分阶段实施

### Phase 0：清理接口（已完成）

- [x] DemoVLA 移入独立 `demovla.py`，不修改原始 `pi0.py` flow timestep 路径。
- [x] 新增 `use_interaction_memory` 开关。
- [x] 新 checkpoint 参数通过 `missing_regex` 与基础 `pi0.5` 权重兼容。
- [x] 取消 Interaction Extractor 对 flow timestep 的额外 conditioning。

### Phase 1：Single-Shot Baseline（部分完成）

- [x] 从模型内部 visual/language tokens 提取 interaction tokens。
- [x] 在 Action Expert 输入前完成 single-shot gated cross-attention。
- [x] 完成 single-shot 训练 checkpoint。
- [ ] 使用当前数据与 seeds 重跑 Vanilla、global pooling 和 parameter-matched
  MLP 对照。

### Phase 2：Sparse Deep Injection（已完成）

- [x] 在 Gemma layer loop 中加入通用 adapter hook。
- [x] 在 Action Expert 第 4、9、14 层注入共享 interaction adapter。
- [x] 三层使用独立 gate，并记录 layer-wise gate。
- [x] 验证 interaction memory 在 flow loop 外缓存，每次 replan 重新提取。

### Phase 3：解释性（部分完成）

- [x] 导出每个 query、每个 camera 的 visual patch attention 和 top-k patch。
- [x] 可视化时间单位从 flow timestep 改为 policy replan。
- [x] 每个 episode 使用一个文件夹保存全部 replan 图片。
- [x] 10k pilot 已生成 364 张诊断图。
- [ ] 完成 target/goal coverage、attention temporal movement 和 query-pair
  分工的定量分析。

### Phase 4：Query Diversity（实现、10k 训练与 50-episode 评估已完成）

- [x] 加入带 margin 的 attention diversity loss。
- [x] 加入权重更小的 memory diversity loss。
- [x] 记录 query pair cosine 与 replan attention top-k。
- [x] 完成 `demovla_libero_sparse_deep_diverse` 10k adapter-only 训练。
- [x] 平均 attention cosine 从 `0.9956` 降至 `0.2007`，平均 memory cosine
  从 `0.9715` 降至 `0.0010`。
- [x] 完成 seed 7、10 tasks × 1 trial pilot，结果为 9/10。
- [x] 完成 10 tasks × 5 trials 正式评估：36/50，成功率 72.0%。
- [x] 历史 initial-state 对齐结果为 36/50 对 35/50，McNemar `p=1.0`；
  已确认旧协议未逐 replan 固定 flow noise，不能作为严格因果消融。
- [x] 新增由 `(policy_seed, task, episode, replan)` 驱动的 stateless flow
  noise，并让 diagnostics on/off 共用标准 action sampler。
- [ ] 扩展至每任务 10 次，并完成目标选择错误率与背景误关注率分析。

### Phase 5：Dynamic Gate（训练与正式评估已完成）

- [x] 保留 scalar gate 基线，新增独立 `interaction_gate_mode="dynamic"`。
- [x] 将 action hidden、action-slot embedding、原始 pi0.5 flow-time embedding
  和 injection-layer embedding 拼接后输入两层 Gate MLP。
- [x] dynamic gate 输出形状为每层 `[batch, action_horizon, 1]`。
- [x] 保持 output projection 和 Gate MLP 最后一层零 kernel 初始化，step 0
  严格恢复 base pi0.5。
- [x] 返回 layer-wise gate、injection ratio 和 interaction attention entropy。
- [x] 将训练 flow time 分为 5 个区间，记录
  `layer × flow-bin × action-slot` gate 均值，并将细粒度指标写入
  JSONL/WandB。
- [x] 新增 `demovla_libero_sparse_deep_dynamic_gate` adapter-only 配置。
- [x] 完成 29999-step dynamic gate 训练。
- [x] 完成 30-episode 官方 pi0.5 / dynamic / injection-off 固定-noise 配对检查。
- [x] 完成官方 stateful RNG 的 100-episode 中等规模检查：98/100。
- [x] 完成 LIBERO Object 每任务 50 次正式评估：495/500，成功率 99.0%。
- [ ] 绘制 layer × flow-time × action-slot gate heatmap。
- [ ] 在失败更集中的任务上扩大 dynamic / layer-mean 固定-noise 配对消融。

### Phase 6：扩展评估、鲁棒性和论文级消融

- [ ] 将 diverse 10k 扩展到 100 episodes，与无 diversity 10k 完全对齐。
- [ ] 若 3k checkpoint 可用，则与 10k 对比；补齐 single-shot、vanilla 和
  parameter-matched adapter 的相同 initial states rollout。
- [ ] 增加 query cosine p95、固定 pair matrix 和 layer-wise injection ratio。
- [ ] 分析 BBQ sauce、ketchup、chocolate pudding 和 orange juice 的目标获取、
  错物抓取与 post-grasp failure。
- [ ] 完成 token 数量、Layer 9 单层注入、参数共享和训练策略消融。
- [ ] 增加 distractor、位置、背景、相机、遮挡和光照鲁棒性评估。
- [ ] 与 ACoT-style pooling 对齐参数和训练预算。
- [ ] 汇总成功率置信区间、计算成本、grounding 和失败案例。

### Phase 7：Kuavo 真机迁移（进行中）

- [x] 确认第一版数据为 Kuavo 5W、右臂、Leju claw、10 Hz。
- [x] 确认数据包含 111 episodes、101390 frames、8 维 state/action、头部与右腕相机。
- [x] 确认采集 prompt 为 `Pick and Place`；第一版保持不变。
- [x] 增加 LeRobot v3 本地数据兼容读取。
- [x] 增加 Kuavo 右臂输入输出 transform 和独立 norm stats 配置。
- [x] 增加从 pi0.5 base 初始化、联合训练 LoRA 与 DemoVLA dynamic gate 的配置。
- [x] 使用全部 101390 帧计算 `kuavo_right` norm stats。
- [ ] 启动第一版 30k 训练。
- [ ] 完成 action chunk 离线回放、关节范围检查和真机低速安全测试。

第一版保持数据集中的原始指令 `Pick and Place`，使用以下环境变量：

```bash
export OPENPI_KUAVO_RIGHT_DATA_ROOT=/home/dongxiaokun/小件钢圈上料/lerobot
export OPENPI_PI05_BASE_CHECKPOINT=/home/dongxiaokun/baseck/pi05_base
export OPENPI_LEROBOT_V3_SRC=/home/dongxiaokun/LeTools-Learning/kuavo_model/external_models/openpi/third_party/try
```

先计算独立 norm stats：

```bash
.venv/bin/python scripts/compute_norm_stats.py \
  --config-name demovla_kuavo_right_dynamic_gate
```

输出必须位于：

```text
assets/demovla_kuavo_right_dynamic_gate/kuavo_right/norm_stats.json
```

然后使用 8 卡启动训练：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
.venv/bin/python scripts/train.py \
  demovla_kuavo_right_dynamic_gate \
  --exp-name kuavo_right_prompt_unchanged_v1 \
  --overwrite
```

该配置从 `pi0.5 base` 初始化，action horizon 为 10；前 7 维绝对关节目标在
训练前变换为相对当前 state 的 delta，夹爪第 8 维保持绝对量。8 维 state 经
归一化后作为 pi0.5 的离散前缀 token 输入。冻结 dense LLM 参数，训练 LoRA、
action/time projection 与完整 DemoVLA dynamic-gate 模块。

## 16. 成功标准

DemoVLA 值得继续扩展，需要至少满足：

1. 相同数据和训练预算下，sparse-deep injection 稳定优于 Vanilla 和 single-shot。
2. parameter-matched MLP 不能解释主要增益。
3. 在 distractor、目标位置变化或遮挡设置下，性能下降小于 Vanilla。
4. 额外推理成本显著低于增加独立视觉 backbone。
5. 关闭 gate 后能够恢复原始 `pi0.5` 行为。
6. interaction attention 可视化与任务相关区域存在稳定对应，而不是只依赖固定背景。

## 17. 最小结论

DemoVLA 不尝试通过外部模型告诉策略“目标物体在哪里”，而是让策略从自身的视觉语言表示中形成一个稀疏 interaction memory。完整 prefix 继续提供全局上下文，action expert 则在少数关键深度使用当前动作 hidden state 反复检索这份 memory。

最终目标是：

> 在不增加外部感知模型、不增加 affordance 标注和不切断原始 VLM 信息通路的前提下，为 action expert 提供更直接、更稳定的交互相关视觉信息。
