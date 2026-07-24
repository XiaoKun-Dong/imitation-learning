# DemoVLA: Self-Grounded Interaction Memory for Precise VLA Control

## 1. 文档状态

- 工作名称：DemoVLA
- 基础模型：OpenPI `pi0.5`
- 当前阶段：架构设计
- 核心约束：
  - 不引入 SAM、DINO、GroundingDINO 等额外视觉模型。
  - 不要求 mask、bbox、crop、point 或 affordance 人工标注。
  - 推理输入保持为原始 RGB、语言指令和机器人状态。
  - 保留原始 VLM prefix 到 action expert 的信息通路。
  - 新增模块应支持从现有 `pi0.5` checkpoint 稳定初始化。

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
       |                           Interaction Token Extractor
       |                                    |
       |                        Sparse Interaction Memory
       |                           (4-16 tokens, cached)
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

- `z_int` 在一次 policy inference 中只计算一次。
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

每层只保留独立 gate：

```python
interaction_layer_gates[layer_index]
```

这样可以：

- 控制参数量；
- 避免不同 adapter 学出完全不一致的 memory space；
- 通过 gate 大小分析不同深度对 interaction memory 的依赖。

如果共享参数明显限制性能，再消融独立 Q projection 或独立 adapter。

### 6.4 Flow Timestep

`pi0.5` 使用 flow timestep 调制 action expert。深层注入时，Query 已来自 timestep-conditioned action hidden states，因此 interaction retrieval 会间接感知当前 denoising 阶段。

第一版不额外将 timestep 拼入 interaction token。后续可验证：

```text
implicit time conditioning through action hidden
vs.
explicit time embedding in cross-attention query
```

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

第一版只使用原始 flow-matching action loss：

```text
L_total = L_action
```

不引入：

- mask loss；
- bbox loss；
- affordance heatmap loss；
- object classification loss；
- 3D reconstruction loss。

这样可以直接回答 interaction tokens 是否能由动作监督自发形成。

### 8.2 可选正则

如果出现 query collapse，再逐项加入：

```text
L_diversity: 不同 query 的 attention map 保持差异
L_decorrelation: interaction token 特征去相关
L_consistency: 背景增强前后的 interaction token 保持一致
```

不要在第一轮同时加入全部正则，否则无法确定增益来源。

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
3. 从 prefix hidden states 提取一次 interaction memory。
4. 在全部 flow denoising steps 中复用 prefix KV 和 interaction memory。
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

禁止在 flow loop 中重复运行 interaction extractor。

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

新增：

```text
InteractionTokenExtractor
InteractionCrossAttentionAdapter
interaction memory extraction
sparse deep injection hook
```

现有 `_embed_object_condition()` 和 `_cross_attend_object_condition()` 不作为最终模型主路径。它们可以保留为 oracle object-condition ablation。

### `src/openpi/models/gemma.py`

需要为 Transformer block loop 增加可选的 action-token adapter hook，推荐接口：

```python
interaction_memory: Array | None
interaction_injection_layers: tuple[int, ...]
interaction_adapter: Callable | None
```

hook 只更新 action expert 对应的 token group。

不要让 `gemma.py` 直接依赖 DemoVLA 具体模块；Gemma 只负责在指定层调用通用 hook。

### `src/openpi/models/pi0_config.py`

建议配置项：

```python
use_interaction_memory: bool = False
num_interaction_tokens: int = 4
interaction_num_heads: int = 8
interaction_injection_layers: tuple[int, ...] = (4, 9, 14)
interaction_gate_init: float = -4.0
interaction_adapter_share_weights: bool = True
```

### `src/openpi/training/config.py`

新增：

- adapter-only freeze filter；
- DemoVLA 训练配置；
- single-shot 和 sparse-deep 消融配置。

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
- flow denoising 不同阶段的读取权重是否变化；
- distractor 出现后 attention 是否保持稳定。

这些可视化只用于分析，不应被表述为显式 object segmentation。

## 14. 主要风险

### 风险 1：Interaction branch 与原始 prefix 冗余

action expert 本来就能读取完整 prefix。必须通过 parameter-matched baseline 和 global pooling baseline 证明收益来自稀疏、任务相关的空间检索。

### 风险 2：Query collapse

多个 query 可能关注同一区域。先观察 attention，再决定是否加入 diversity 或 decorrelation regularization。

### 风险 3：缺少显式监督导致定位不稳定

动作 loss 可能只学习数据集偏差。需要通过目标位置变化、背景变化和 distractor 测试判断是否真正形成 interaction grounding。

### 风险 4：深层注入破坏预训练策略

使用接近零的 gate 初始化，并确保关闭 DemoVLA 时输出严格退化为原始 `pi0.5`。

### 风险 5：修改 Gemma 增加维护成本

先完成 single-shot 版本和正向结果，再实现通用 layer hook。不要在尚未验证 interaction token 有效前大规模修改 Transformer 内部。

## 15. 分阶段实施

### Phase 0：清理接口

- 将现有外部 `target_*` 路径标记为 oracle ablation。
- 新增 `use_interaction_memory` 开关。
- 确保关闭开关时 checkpoint、loss 和 sampling 行为不变。

### Phase 1：Single-Shot Baseline

- 从模型内部 visual/language tokens 提取 interaction tokens。
- 在 action expert 输入前做一次 gated cross-attention。
- 完成 Vanilla、global pooling、parameter-matched MLP 对照。

### Phase 2：Sparse Deep Injection

- 在 Gemma layer loop 中加入通用 adapter hook。
- 在 3 个 action expert 层注入共享 interaction adapter。
- 验证 interaction memory 在 flow loop 外缓存。

### Phase 3：鲁棒性和解释性

- 增加 clutter、位置变化、遮挡和相机偏移评估。
- 导出 interaction attention map 和各层 gate。
- 判断是否需要 query diversity regularization。

### Phase 4：论文级消融

- 完成 token 数量、注入深度、参数共享和训练策略消融。
- 与 ACoT-style pooling 对齐参数和训练预算。
- 汇总性能、计算成本和失败案例。

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
