# DemoVLA 新权重 Memory Overlay 与后续因果推进建议

## 1. 本轮可视化设置

本轮没有复用旧图，而是重新加载阶段 A 最终权重并在线重放环境：

| 项目 | 设置 |
|---|---|
| Checkpoint | `stage_a_fixed005_seed42_v2/4999` |
| Memory | correct memory |
| 推理 gate | 三层固定 `0.03` |
| 环境 seed | 7 |
| Flow-noise seed | 0，按 task/episode/replan stateless 配对 |
| Overlay | 每个 replan、每个 memory slot、每个有效相机 |
| 原始数据 | 每个 replan 同时保存 `.npz` attention |

选择的是正式评测中 `correct-0.03` 成功而 `zero-0.03` 失败的配对样本：

1. LIBERO-Object：`pick up the alphabet soup and place it in the basket`，episode 2；
2. LIBERO-10：`put both moka pots on the stove`，episode 3。

两次新权重重放均成功。清晰版分别保存 29 和 78 个 replans，共 107 张 overlay，并在图顶端写入任务提示、replan index 和 environment step。

本地完整图与 raw attention：

```text
artifacts/demovla_stage_a_memory_overlay_step4999_fixed003_seed7_v2/
  interaction_patches/
```

服务器完整 rollout、视频与日志：

```text
data/libero/videos/demovla_stage_a_memory_overlay_step4999_fixed003_seed7_v2/
```

## 2. Overlay 初步观测

### 2.1 Memory extractor 没有明显 slot collapse

| 指标 | 单物体 | 长程双物体 |
|---|---:|---:|
| Replans | 29 | 78 |
| Visual-attention 归一化熵 | 0.7720 | 0.7891 |
| Top-4 patch attention mass | 0.2460 | 0.2164 |
| Slot 间 attention cosine | 0.1423 | 0.1655 |
| 同一 slot 相邻 replan cosine | 0.6502 | 0.7027 |

解释：

- slot 间 cosine 仅约 `0.14–0.17`，四个 memory slot 没有复制成同一种视觉注意模式；
- 相邻 replan cosine 约 `0.65–0.70`，attention 会随场景和执行阶段变化，而不是固定背景模板；
- 视觉 attention 熵约 `0.77–0.79`，既不是单 patch 极端塌缩，也不是完全均匀；
- 不同 slot 的 base/wrist camera mass 存在稳定差异，说明已经出现一定跨视角分工。

图上仍能看到一部分 top patch 落在机器人、本体边缘或背景，而不是任务对象。这说明 memory extractor 有结构，但尚不能仅凭热力图断言它实现了可靠的目标 grounding。

### 2.2 训练日志中的低 entropy 属于另一个接口

阶段 A 最终记录的 `attention_entropy≈0.0014` 来自 action expert 读取四个 memory slot 时的 action-to-memory attention，不是本轮渲染的 memory-query-to-image-patch attention。

因此当前更可能的风险是：

```text
视觉 memory slots 本身有差异
        ↓
action expert 在每层/动作位置几乎只读取一个 slot
        ↓
有效内容没有稳定转化成可显著的行为收益
```

下一项最重要的可视化不是再画更多 extractor heatmap，而是导出每个注入层、flow time 和 action position 的 action-to-memory 权重，并与本轮 slot-to-patch attention 相乘：

```text
effective_patch_attention
  = action_to_memory_attention × memory_to_patch_attention
```

这才是“动作专家最终从 observation 的哪里取得了有效残差”的端到端 overlay。

## 3. DemoVLA 更可能有优势的任务类型

根据当前结构和已有结果，优先级从高到低为：

1. **长程、多物体、顺序敏感任务。** Memory 每次 replan 重建，能够随着第一个子目标完成而转向第二个对象或目标区域；当前 fixed-0.03 相对 off 的主要收益也集中在 LIBERO-10。
2. **目标/干扰物外观相近的任务。** 任务提示先更新 interaction query，再选择视觉 patch，理论上适合需要语言消歧的目标选择。
3. **跨视角遮挡任务。** Base view 提供全局关系，wrist view 提供抓取和接触细节；四个 slot 已表现出不同的相机质量分配。
4. **同场景不同 goal 的任务。** 主观测几乎相同、语言目标不同，最适合检验 task-conditioned memory 是否真的改变动作。

不应把简单单物体任务作为主要优势场景：官方 π0.5 在这些任务上接近天花板，DemoVLA 即使有效也很难产生足够配对翻转。

## 4. 当前证据能支持什么

已经支持：

1. 冻结主干后，新增支路能学到降低 flow loss 的残差；
2. correct memory 在 1,024 个配对样本上显著优于 shuffled/zero/off；
3. fixed-0.03 行为上显著优于 off：`+2.25pp`，McNemar `p=0.02246`；
4. correct-0.03 相对 zero-0.03 为 `+1.00pp`，6 比 2 个有利翻转；
5. memory extractor 的视觉 attention 具有 slot 差异和时序变化。

尚未支持：

1. correct-0.03 相对 zero-0.03 的行为差异尚不显著，`p=0.2891`；
2. overlay 只能说明模型关注了哪些 patch，不能证明这些 patch 是行为收益的原因；
3. 尚未证明 attention 随任务提示中的目标词发生特异变化；
4. 尚未证明正确 memory 在关键 replan 上具有“必要性”和“充分性”；
5. 尚未证明 DemoVLA 在共同初始化、共同训练预算下显著优于 parameter-matched control。

因此论文目前可以声称 loss-level 的样本相关因果作用和 fixed-0.03 相对 off 的行为作用，不能声称已经建立强行为内容因果或显著优于官方 π0.5。

## 5. 如何建立更强的因果链

### 5.1 特异性：Prompt counterfactual

固定 observation、action target、flow time 和 noise，仅替换任务提示：

- correct prompt；
- 目标名词互换；
- 同场景错误 goal；
- unrelated prompt；
- blank prompt。

同时比较 memory cosine、有效 patch attention、flow loss 和 action delta。正确 prompt 应稳定优于错误 prompt，并使 attention 转移到正确目标；否则 language-conditioned memory 的主张不成立。

### 5.2 必要性：时序与局部 knockout

在同一个 rollout 中只关闭特定范围：

- reach 前；
- grasp 附近；
- transport；
- placement；
- 指定注入层；
- 指定 memory slot。

如果 DemoVLA 对长程任务确实有优势，关闭 memory 的损害应集中在目标切换、遮挡或放置对齐等特定阶段，而不是所有 replan 均匀下降。

### 5.3 充分性：Memory transplant / rescue

对 batch-1 rollout 缓存并移植 memory：

- 当前样本 correct memory；
- 同任务其他 episode memory；
- 同场景错误目标 memory；
- 不同任务 memory；
- zero memory。

最强证据是：off/zero 条件失败时，注入对应 replan 的 correct cached memory 能恢复成功，而错误任务 memory 不能恢复。

### 5.4 剂量反应与中介分析

在完全相同 memory 内容下扫描 gate `0/0.01/0.02/0.03/0.04/0.05`，记录：

- flow-loss 改善；
- action delta；
- 实际 injection ratio；
- paired rollout 成功；
- 目标 patch attention 与 action delta 的相关性。

若从 zero 到适中 gate 逐步改善、过强 gate 再下降，能够把当前 `0.03` 最优从经验现象提升为可解释的剂量反应证据。

## 6. 现在是否修改 memory 注入方式

**当前不建议先重写 memory extractor。** Overlay 和量化结果表明 extractor 已形成非平凡、随时间变化且 slot 间不同的内容。现在直接加入 FocusVLA 式 top-k 或更多 object tokens，会同时改变内容、容量和优化难度，反而难以定位行为因果不足的原因。

应先检查 action-to-memory readout。如果确认几乎所有 layer/action token/flow time 都选择同一 slot，优先修改的是注入/readout：

1. 导出并监督每层、每动作位置的 slot 使用率；
2. 对 residual 做 RMS normalization，将“方向”和“强度”解耦；
3. 使用 CAC-VLA 风格的 context gate，但限制在有界范围并初始化为复现 fixed-0.03；
4. gate 输入读取 `hidden + retrieved memory + flow time + layer embedding`，而不是只学习全局开关；
5. 分层控制 gate，先保持主干冻结，避免再次通过关闭支路获得低损失；
6. 加入 correct-vs-shuffled ranking loss，让 gate/readout 对正确内容而非固定偏置负责。

只有 prompt counterfactual 显示 extractor 对目标词不敏感、目标区域覆盖率低时，才进入 FocusVLA 式 patch sparsification；只有 memory 与未来动作阶段缺乏一致性时，才引入 CAC-VLA 式 latent-action alignment。这样每次结构修改都有明确失败证据对应。

## 7. 推荐执行顺序

1. **已完成：** 新权重逐 replan memory-to-patch overlay 和 raw attention 保存；
2. **下一步：** action-to-memory × memory-to-patch 端到端 overlay；
3. 在固定 observation 上完成 prompt swap / target noun swap 的配对 flow-loss 扫描；
4. 在 LIBERO-10、多物体 correct-only 样本上做阶段性 memory knockout；
5. 根据 readout 结果决定是否实现 residual normalization + bounded context gate；
6. 结构确定后再做 3 seeds、共同初始化和 parameter-matched control 训练。

这条路线先区分“memory 没提好”和“memory 提好了但没注入好”，可以避免在证据不足时同时修改 extractor、gate 和 adapter，导致新结果仍无法归因。

## 8. 端到端 Action-to-Memory 诊断结果

在阶段 A 最终 checkpoint、fixed gate `0.03` 上，进一步导出了：

```text
action-to-memory attention:
  [replan, flow step, injection layer, head, action position, memory slot]

effective patch attention:
  mean_head(action-to-memory) × memory-to-patch
```

正常 rollout 的 action 仍由原始 sampler graph 产生；diagnostics 使用相同 RNG/noise 独立重放 denoising 轨迹，不改变被执行的 action。

长程任务 `put both moka pots on the stove` 的 82 个 replans 显示：

| 指标 | 结果 |
|---|---:|
| q0 平均读取质量 | 0.01% |
| q1 平均读取质量 | 37.47% |
| q2 平均读取质量 | 12.51% |
| q3 平均读取质量 | 50.01% |
| q3 成为 head-mean 最大项的比例 | 99.98% |
| Per-head attention entropy | 0.0024 |
| 三个注入层 effective-attention cosine | 1.0000 |
| fixed gate | 0.0300 |
| 实际 injection ratio | 2.96% |

八个 attention head 几乎形成固定分工：

- head 0/1/4/5 几乎只读 q3；
- head 2/3/6 几乎只读 q1；
- head 7 几乎只读 q2；
- q0 基本不被任何 head 使用。

该路由在 layer 4/9/14、flow time 和 action position 上几乎不变，因此三层 effective overlay 基本一致。单物体任务也得到同样的 `q1:q2:q3≈3:1:4` 配比。

这修正了此前对低 entropy 的解释：不是整个模块只使用一个 query，而是每个 attention head 饱和为近乎固定的单 query 路由。多头合计使用三个 query，但没有形成预期的 layer/time/action-conditioned memory retrieval。

完整端到端图与 raw arrays：

```text
artifacts/demovla_stage_a_effective_overlay_step4999_fixed003_seed7/
```

## 9. Memory-only Prompt Counterfactual

为排除主 VLM prompt 通路的混杂，正式实验固定正确 action prefix/KV cache，只让 memory extractor 读取以下提示：

1. 正确任务提示；
2. 同一 LIBERO 10-task group 内的下一个错误任务提示；
3. blank prompt。

三种条件共享 1,024 个 observation、state、action target、flow time 和 Gaussian noise，gate 均为 `0.03`。

| Memory prompt | Flow loss | Memory cosine | Visual-attention cosine | Top patch 改变率 |
|---|---:|---:|---:|---:|
| Correct | **0.00346511** | 1.0000 | 1.0000 | 0.00% |
| Same-suite wrong task | 0.00346951 | 0.8718 | 0.7323 | 60.06% |
| Blank | 0.00348827 | 0.7572 | 0.6951 | 64.21% |

配对统计：

| Contrast | Correct − counterfactual | 95% CI | Correct lower | Counterfactual lower | Sign p |
|---|---:|---:|---:|---:|---:|
| Correct vs wrong-task memory prompt | −0.00000440 | [−0.00001366, +0.00000486] | 534 | 490 | 0.179 |
| Correct vs blank-memory prompt | **−0.00002316** | **[−0.00003435, −0.00001198]** | 606 | 418 | **4.64e-9** |

Wrong-task prompt 使 memory 表示和选中 patch 明显变化，但没有显著改变动作 flow loss。即使在 Object 子集内将目标物体提示循环替换，均值差也只有 `−1.27e-6`，95% CI 跨零。Blank prompt 会造成小但显著的损失退化，说明语言信息并非完全无用，但当前支路主要利用的是一般性语言条件，而不是稳定的任务特异目标语义。

正式输出：

```text
outputs/demovla_stage_a/stage_a_fixed005_seed42_v2/
  prompt_memory_only_counterfactual_formal_20260826/
```

## 10. 更新后的结构判断

当前证据已经较明确地区分了两个接口：

```text
memory extractor:
  对 prompt 敏感，slot 视觉注意不同，并随 replan 变化

action readout / injection:
  per-head 路由近乎固定，跨层 effective attention 重复，
  对 wrong-task memory 的 task-specific 变化不敏感
```

因此下一版不应优先增加 query 数或直接对视觉 attention 做 top-k。更合理的是先修复 action readout：

1. 对 action query 和 memory key 做归一化，并使用有界可学习 temperature，避免 softmax 饱和为固定 head→slot 映射；
2. 给注入层加入 readout layer embedding，或使用轻量 layer-specific query projection，使 layer 4/9/14 不再完全重复；
3. 对 residual 做 RMS normalization，再以 bounded gate 控制强度，分离残差方向与幅度；
4. gate 初始化复现 fixed-0.03，范围优先设为 `0.005–0.05`，主干继续冻结；
5. 加入 memory-only wrong-prompt ranking loss：主 action prefix 保持正确，只要求正确任务 memory 比同场景错误任务 memory 获得更低 flow loss；
6. 暂不对每个样本强制 query usage 均匀，因为 per-head query 专门化本身可以合理；重点约束读取随 prompt/layer/time 产生条件变化。

建议把下一阶段拆成两个可归因步骤：

### B1：Readout recovery

- 冻结 π0.5 主干和当前 memory extractor；
- 固定 gate `0.03`；
- 只训练 action-to-memory query/key/value/output projection 和 layer conditioning；
- 加入 correct-vs-wrong-prompt memory ranking；
- 训练 1k–2k steps。

验收标准是 wrong-task memory prompt 显著劣于 correct，同时三个注入层 effective attention 不再完全一致。

### B2：Bounded dynamic gate

只有 B1 通过后才恢复动态 gate，并初始化为 B1 的 fixed-0.03。动态门的目标应是选择何时使用已经具有任务特异作用的 residual，而不是再次承担修复 memory 语义的任务。
