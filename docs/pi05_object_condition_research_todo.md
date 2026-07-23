# pi0.5 Stage-Adaptive Object Conditioning 研究 TODO

## 研究定位

本项目不将“向 pi0.5 注入 mask / bbox / crop”本身作为核心创新。上述内容属于目标对象先验的输入形式和工程实现，
cross-attention 也不是新的通用结构。

本项目聚焦的问题是：

> 如何让预训练 VLA 显式利用目标对象先验完成对象消歧，同时避免条件分支持续干扰已经成熟的操作策略。

当前实验揭示了一个明确冲突：显式对象条件能够改变模型行为并提高目标抓取率，但没有提高最终成功率，且增加了
抓取后的搬运和放置失败。因此，问题不是模型能否读取对象条件，而是对象条件应该在什么阶段生效、以多大强度
生效，以及如何保持预训练策略原有能力。

## 当前实验结论

LIBERO-P 三组 seed、共 450 个严格配对 episode 的结果：

| 模型 | Success | Target grasp | Grasp failure | Post-grasp failure | Wrong-object |
|---|---:|---:|---:|---:|---:|
| Official `pi05_libero` | 82.7% | 87.3% | 12.7% | 4.7% | 7.3% |
| Object-2D | 80.0% | 89.3% | 10.7% | 9.3% | 8.7% |

配对 success 为 `9 wins / 13 losses / 128 ties`，exact McNemar `p=0.5235`。目前可以支持：

- [x] 显式对象条件被模型实际使用，目标抓取率提高 2 个百分点。
- [x] 固定注入会产生负迁移，主要表现为 post-grasp failure 增加。
- [x] 冻结 legacy 参数不能保证行为保持，新增 residual 仍会改变 action token 分布。
- [ ] 尚未证明对象条件能够显著提升端到端任务成功率。
- [ ] 尚未证明阶段自适应门控或策略保持约束有效。

因此，当前成果属于“发现并量化研究问题”，还不能表述为“已经解决该问题”。

## 拟解决的问题

### 1. 目标对象消歧

语言指令和全局图像上下文可能不足以稳定区分相似对象。目标是降低错误对象接近、抓取和操作的比例，并提高
目标对象命中率。

### 2. 条件注入的阶段冲突

目标对象先验在搜索、接近和抓取阶段可能有益，但在抓取后的搬运与放置阶段可能成为多余干扰。固定强度、全程
生效的 object residual 无法表达这种阶段差异。

### 3. 预训练策略能力保持

即使冻结原模型参数，新增分支仍会改变原始策略输出。需要对 conditioned policy 的行为偏移进行显式控制，而不是
仅依赖参数冻结、zero init 或较小 residual scale。

## 候选创新点

### 创新点 A：阶段自适应对象条件

将当前固定残差：

```text
action_tokens <- action_tokens + constant_scale * object_delta
```

改为由动作生成状态决定注入强度：

```text
target mask / bbox / crop
        |
        v
object representation
        |
        v
gate(action token, proprioception, flow timestep)
        |
        v
action_tokens + residual_scale * gate * object_delta
```

方法目标是让对象条件在目标定位和抓取阶段增强，在抓取完成后自动衰减，而不是简单学习一个全局平均强度。

候选名称：

> Stage-Adaptive Object Conditioning for Pretrained VLA

### 创新点 B：带策略保持约束的条件适配

在 flow-matching 目标之外加入行为保持约束：

```text
L = L_flow + lambda_obj * L_object + lambda_keep * D(policy_cond, policy_base)
```

优先在以下样本或阶段约束 conditioned policy 接近 frozen base policy：

- object condition dropout 样本；
- 抓取后或目标条件不再重要的阶段；
- 对象条件置信度较低的样本；
- base policy 已能稳定成功的样本。

该方向要解决的是“冻结参数不等于保持行为”的问题。

### 创新点 C：将对象条件转化为动作级中间表征

参考 ACoT-VLA 的 action-centric reasoning 思路，不直接长期向最终 action token 注入原始对象特征，而是先预测
粗粒度动作、关键点轨迹或阶段表征，再由最终 action expert 使用该中间表征。

```text
object condition -> coarse action / phase representation -> final action expert
```

该方向的假设是：动作级中间表征与最终控制目标更对齐，比原始视觉几何 residual 更容易被成熟 action expert 使用。
第一阶段只作为候选路线，不与动态门控同时大规模修改，避免无法归因。

## 工程 TODO

### P0：校正当前基线

- [x] 审计 gate 实现，确保有效注入为 `residual_scale * sigmoid(gate) * delta`，并验证初始值约为 `0.0018`。
- [x] 记录 raw gate、sigmoid gate、effective scale、object delta RMS、object residual RMS 和 action token RMS。
- [x] 验证 `has_condition=False` 和 object dropout 时 residual 严格为零。
- [x] 完成 gate 与 output projection 的有限梯度、参数保存和恢复一致性测试。
- [x] 用相同 checkpoint、任务顺序和 seed 复现 Official 与 Object-2D 基线。

P0 审计记录（2026-07-23）：

- 修复前代码在启用 gate 时用 `sigmoid(gate)` 覆盖了 `object_condition_residual_scale`，初始 effective scale
  实际约为 `0.01799`，是计划值 `0.001799` 的 10 倍。
- 修复后 effective scale 为 `object_condition_residual_scale * sigmoid(gate)`，且 `0.1` 重新成为硬上限。
- 训练日志新增 object delta、实际 residual 和注入前 action token 的 RMS，便于区分“大特征、小门控”和
  “分支未学习”。
- `step1_gate_5k` 等修复前 gate checkpoint 是在旧公式下训练的。修复后直接加载会改变其有效注入强度，
  不能与旧评测混作同一模型；正式 Step 1 需要从 Official checkpoint 重新训练。
- Official 与无 gate 的 Object-2D 正式配对基线不受本次公式修复影响，继续使用本文“当前实验结论”中的
  固定任务顺序和 seeds `7/42/123` 结果。
- 已完成 `batch=1, 2 steps` 全链路 smoke，loss、grad norm、gate、effective scale 和三类 RMS 均为有限值，
  并成功保存 checkpoint：
  `checkpoints/pi05_libero_object_2d_step1/p0_metrics_smoke/1`。

### P1：先验证注入强度，而不是继续堆叠编码器

- [x] 新增 `pi05_libero_object_2d_step1_gate0`：保留 zero-init output projection，将 gate init 设为 `0`，
  初始 effective scale 为 `0.05`，且不覆盖旧 Step 1 配置。
- [x] 增加 policy-server 推理 scale override，使同一个 checkpoint 可以在不改变权重的情况下扫描注入强度。
- [x] 使用 Step 99 固定权重评估 scale：`0`、`0.001`、`0.003`、`0.01`、`0.03`、`0.05`。
- [ ] 分别统计 pre-grasp、grasp 和 post-grasp 阶段的 residual RMS。
- [ ] 绘制 target-grasp gain 与 post-grasp regression 的关系曲线。
- [ ] 判断退化来自注入幅度、注入时机，还是 object representation 本身。

P1 固定权重协议：

- checkpoint 固定为 `checkpoints/gate0_pilot_100/99`；
- object encoder、cross-attention、gate 和所有 legacy policy 参数保持完全相同；
- 仅通过 `--policy.object-condition-scale` 覆盖最终 applied residual scale；
- scale `0` 是同 checkpoint 的严格无注入对照，可隔离“训练得到的新增参数”和“实际 residual 注入”；
- 每个 scale 使用完全相同的 task IDs、episode seed、replan steps 和 horizon；
- 当前 scale `0.05` 的 seed 7 初测为 success `16/20`、target grasp `18/20`、post-grasp failure `2/20`、
  wrong-object `2/20`，仅作为曲线上的一个点，不据此选择强度。

P1 固定权重结果（seed 7，同一批 20 个 task）：

| Applied scale | Success | Target grasp | Grasp failure | Post-grasp failure | Wrong-object | 成功平均步数 |
|---:|---:|---:|---:|---:|---:|---:|
| `0` | 16/20 | 17/20 | 3/20 | 1/20 | 2/20 | 156.3 |
| `0.001` | 17/20 | 17/20 | 3/20 | 0/20 | 2/20 | 163.1 |
| `0.003` | 15/20 | 17/20 | 3/20 | 2/20 | 2/20 | 155.7 |
| `0.01` | 14/20 | 18/20 | 2/20 | 4/20 | 1/20 | 146.0 |
| `0.03` | 16/20 | 18/20 | 2/20 | 2/20 | 2/20 | 149.1 |
| `0.05` | 16/20 | 18/20 | 2/20 | 2/20 | 2/20 | 157.8 |

相对 scale `0` 的 success 严格配对结果：

```text
0.001: 1 win  / 0 losses / 19 ties, exact McNemar p=1.0
0.003: 0 wins / 1 loss   / 19 ties, exact McNemar p=1.0
0.01:  0 wins / 2 losses / 18 ties, exact McNemar p=0.5
0.03:  1 win  / 1 loss   / 18 ties, exact McNemar p=1.0
0.05:  1 win  / 1 loss   / 18 ties, exact McNemar p=1.0
```

P1 初步结论：

- 静态注入强度确实改变行为，但 success 对 scale 呈非单调响应，没有稳定最优点。
- `0.001` 的 success 最高，但 target grasp 与 scale `0` 相同，不能证明对象对齐改善。
- `0.01` 及以上总体上提高 target grasp，但收益被 post-grasp failure 抵消；`0.01` 的冲突最明显。
- 两个主要 wrong-object episode 在大多数 scale 下持续存在，单纯调节强度没有解决对象消歧。
- 只有 5/20 个任务随 scale 改变，且所有差异均不显著；当前结果用于诊断，不用于宣称性能提升。
- 静态全程注入无法同时满足抓取前增强和抓取后能力保持，后续方法重点应转向阶段自适应，而不是继续搜索
  单一全局 scale。

### P2：实现动态门控

- [x] Gate-V0：全局标量 gate，已通过 P1 固定权重 scale sweep 完成最小对照。
- [ ] Gate-V1：`gate(action_token, flow_timestep)`。
- [x] Gate-V2：实现 per-action-token `gate(action_token, proprioception, flow_timestep)`。
- [ ] 比较 scalar、per-token 和 per-layer gate；首轮优先 per-token 单点注入。
- [x] 检查门控是否在抓取后自然下降：当前训练将 gate 整体推高，rollout 未显示抓取后保护，假设未通过。
- [ ] 增加 gate 饱和、方差和阶段分布日志（mean/min/max/std 已完成，rollout 阶段分布待补）。

P2 实现协议：

```text
固定：Official policy + Step 99 object encoder/cross-attention/output projection
训练：object_condition_dynamic_gate_* only
输入：action token + proprioception + flow timestep
输出：per-action-token gate
初始化：sigmoid(-3.4761) ~= 0.03
初始 applied scale：0.1 * 0.03 ~= 0.003
```

- dynamic gate output projection 使用 zero init，初始化时所有 token 精确退化为 P1 的固定 scale `0.003`；
- gate 在每个 denoising flow step 重新计算，可依赖当前 noisy action 和 flow timestep；
- Step 99 object adapter 与全部 legacy policy 参数冻结，避免重新训练 encoder 导致变量混杂；
- 训练配置为 `pi05_libero_object_2d_dynamic_gate`，从
  `checkpoints/gate0_pilot_100/99/params` 加载；
- 已完成 `batch=1, 2 steps` smoke：初始 effective scale `0.003`，第二步 gate std 变为非零，
  loss、grad norm、gate 和 residual RMS 均为有限值；checkpoint 位于
  `checkpoints/pi05_libero_object_2d_dynamic_gate/p2_dynamic_gate_smoke/1`。

P2 训练后不能只根据 loss 或 gate mean 选 checkpoint，至少要求：

- gate std 明显大于零，证明不是退化为另一个全局标量；
- gate min/max 不快速饱和到 `0/1`；
- gate 与 proprioception、flow timestep 或 action token 的变化存在可重复关系；
- rollout 中 pre-grasp 的 applied residual 高于 post-grasp，且 post-grasp failure 相比固定 scale 回落；
- conditioned rollout 相比 scale `0` 提高 target grasp 或降低 wrong-object，同时不损害最终 success。

P2 训练与评测结果（seed 7，同一批 20 个 task）：

训练到 Step 499 时，object adapter 的 `delta_rms` 稳定在约 `0.076`，证明冻结有效；但 gate mean 从 `0.03`
上升至 `0.857`，min/max 为 `0.682/0.930`，effective scale 上升至 `0.0857`。gate std 为 `0.056`，
说明它不是严格的全局标量，但所有 token 的 gate 都处于较高区间，整体行为接近“全程打开”。

| 模型 | Success | Target grasp | Grasp failure | Post-grasp failure | Wrong-object | 成功平均步数 |
|---|---:|---:|---:|---:|---:|---:|
| Fixed scale `0` | 16/20 | 17/20 | 3/20 | 1/20 | 2/20 | 156.3 |
| Fixed scale `0.001` | 17/20 | 17/20 | 3/20 | 0/20 | 2/20 | 163.1 |
| Fixed scale `0.003` | 15/20 | 17/20 | 3/20 | 2/20 | 2/20 | 155.7 |
| Fixed scale `0.05` | 16/20 | 18/20 | 2/20 | 2/20 | 2/20 | 157.8 |
| Dynamic Gate Step 100 | 14/20 | 18/20 | 2/20 | 4/20 | 2/20 | 150.6 |
| Dynamic Gate Step 499 | 15/20 | 18/20 | 2/20 | 3/20 | 1/20 | 158.0 |

严格配对：

```text
Dynamic-100 vs Fixed-0:    0 wins / 2 losses / 18 ties, exact McNemar p=0.5
Dynamic-499 vs Fixed-0:    2 wins / 3 losses / 15 ties, exact McNemar p=1.0
Dynamic-499 vs Fixed-0.05: 1 win  / 2 losses / 17 ties, exact McNemar p=1.0
Dynamic-499 vs Dynamic-100: 2 wins / 1 loss   / 17 ties, exact McNemar p=1.0
```

Dynamic-499 相对 Fixed-0 的行为变化集中在 5 个任务：

- 改善：chocolate pudding 从 post-grasp failure 变为成功；
- 改善：salad dressing level5 sample4 从 wrong-object 变为成功；
- 退化：alphabet soup level1 从成功变为 post-grasp failure；
- 退化：salad dressing level5 sample1 从成功变为 post-grasp failure；
- 退化：orange juice level4 从成功变为 post-grasp failure。

P2 结论：

- 动态 gate 能读取 action/state/time 并产生差异，也出现了一个真实 wrong-object 修复案例；
- 仅使用 flow-matching loss 时，优化会把 gate 整体推向开启，而不是学习抓取后衰减；
- object grounding 的局部收益被新增 post-grasp failure 抵消，最终 success 低于无注入对照；
- 当前证据不支持“无额外约束的动态 gate”解决阶段冲突，应进入 P3，为策略偏移加入显式保持约束；
- P2 结果属于失败但有效的机制验证，不应通过继续训练、挑单个任务或扩大 gate 网络掩盖。

### P3：加入策略保持约束

- [ ] 冻结一份 base policy，生成同一 batch 的 base action/velocity prediction。
- [ ] 实现 conditioned 与 base prediction 的 consistency loss。
- [ ] 先在 condition-dropout 样本使用 consistency loss，避免过度限制对象条件收益。
- [ ] 再评估阶段加权 consistency：pre-grasp 弱约束、post-grasp 强约束。
- [ ] 消融 `lambda_keep`，报告对象命中率与原策略保持之间的权衡。

### P4：验证 ACoT-lite 动作级中间表征

- [ ] 从 action chunk 下采样或平滑得到 coarse-action 监督信号。
- [ ] 训练 object-aware coarse-action head，并先只作为 auxiliary loss 使用。
- [ ] 让 final action expert cross-attend coarse-action tokens，而不是直接读取 raw object residual。
- [ ] 比较 `Object -> Expert` 与 `Object -> Coarse -> Expert`。
- [ ] 仅在动态门控路线结论清楚后推进，避免多个变量同时变化。

## 实验矩阵

| 实验 | Object encoder | 注入方式 | 策略保持 | 目的 |
|---|---|---|---|---|
| Base | 无 | 无 | 不适用 | 官方基线 |
| Fixed-2D | 现有 | 固定 residual | 无 | 复现当前退化 |
| Scalar-Gate | 现有 | 全局 gate | 无 | 判断平均降幅是否足够 |
| Dynamic-Gate | 现有 | state/timestep-aware gate | 无 | 验证阶段自适应假设 |
| Dynamic-Gate-Keep | 现有 | 动态 gate | consistency | 验证能力保持 |
| ACoT-lite | object-to-coarse | coarse-to-expert | 可选 | 验证动作级中间表征 |

所有实验应保持数据、初始化、训练预算、任务顺序和评测 seed 一致。结构消融与训练目标消融分开进行。

## 评价指标

主结果不能只报告 success rate，至少联合报告：

- [ ] 最终任务成功率 `success rate`；
- [ ] 目标对象命中率 `target grasp rate`；
- [ ] 错误对象率 `wrong-object rate`；
- [ ] 抓取前失败率 `grasp failure rate`；
- [ ] 抓取后失败率 `post-grasp failure rate`；
- [ ] 成功 episode 的平均步数；
- [ ] paired wins / losses / ties；
- [ ] exact McNemar 检验与置信区间；
- [ ] 按任务、目标类别和阶段拆分的结果。

## 创新成立标准

只有满足以下条件，才能主张方法解决了问题：

- [ ] 相比 Official，wrong-object 或 grasp failure 有稳定下降；
- [ ] 相比 Fixed-2D，post-grasp failure 明显回落；
- [ ] 最终 success rate 超过 Official，或至少在统计上证明非劣且对象消歧显著改善；
- [ ] 多个 seed 和任务类别上趋势一致，不依赖少数对象类别；
- [ ] 消融证明收益来自阶段自适应或策略保持，而不是参数量或额外训练预算；
- [ ] gate 的阶段行为与研究假设一致，并有定量分析支撑。

如果动态 gate 最终收敛到接近零且模型恢复 Official 水平，只能说明它学会关闭有害分支，不能证明对象条件带来收益。

## 论文式贡献表述（待实验验证）

> 本工作识别并量化了预训练 VLA 中目标对象对齐与策略能力保持之间的冲突：静态对象条件能够改善目标抓取，
> 但持续干预动作表示会损害抓取后的长时程操作。为此，我们提出阶段自适应对象条件机制，根据动作生成状态
> 调节目标先验对 action expert 的影响，并结合策略保持约束控制 conditioned policy 对预训练策略的偏移，
> 以同时改善对象消歧与端到端操作成功率。

在完成“创新成立标准”之前，文档、汇报和论文中应将以上内容表述为研究假设或候选贡献，而不是已验证结论。
