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

- [ ] 审计 gate 实现，确保有效注入为 `residual_scale * sigmoid(gate) * delta`，并验证初始值约为 `0.0018`。
- [ ] 记录 raw gate、sigmoid gate、effective scale、object residual RMS 和 action token RMS。
- [ ] 验证 `has_condition=False` 和 object dropout 时 residual 严格为零。
- [ ] 完成 gate 与 output projection 的梯度、checkpoint 保存和恢复测试。
- [ ] 用相同 checkpoint、任务顺序和 seed 复现 Official 与 Object-2D 基线。

### P1：先验证注入强度，而不是继续堆叠编码器

- [ ] 评估固定 scale：`0`、`0.001`、`0.003`、`0.01`、`0.03`、`0.1`。
- [ ] 分别统计 pre-grasp、grasp 和 post-grasp 阶段的 residual RMS。
- [ ] 绘制 target-grasp gain 与 post-grasp regression 的关系曲线。
- [ ] 判断退化来自注入幅度、注入时机，还是 object representation 本身。

### P2：实现动态门控

- [ ] Gate-V0：全局标量 gate，作为最小对照，不作为最终方法。
- [ ] Gate-V1：`gate(action_token, flow_timestep)`。
- [ ] Gate-V2：`gate(action_token, proprioception, flow_timestep)`。
- [ ] 比较 scalar、per-token 和 per-layer gate；首轮优先 per-token 单点注入。
- [ ] 检查门控是否在抓取后自然下降，而不是仅将全局注入压到接近零。
- [ ] 增加 gate 饱和、方差和阶段分布日志。

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
