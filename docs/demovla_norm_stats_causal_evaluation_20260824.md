# DemoVLA × Norm Stats 严格因果测评报告

日期：2026-08-24  
范围：LIBERO Object，DemoVLA dynamic gate checkpoint `29999`

## 1. 结论

本次四卡测评表明，当前 DemoVLA checkpoint 的高成功率来自**模型与训练时
归一化统计的匹配组合**，不能把此前 `dynamic 29/30` 对
`injection_off 13/30` 直接解释为 interaction injection 相对官方 π0.5 的纯架构增益。

在最终 400-episode、逐 episode 固定 flow noise 的 `2 × 2` 因果矩阵中：

| 推理配置 | Official stats | Local training stats |
|---|---:|---:|
| 同一 DemoVLA checkpoint，关闭 injection | **99/100 (99%)** | 42/100 (42%) |
| 同一 DemoVLA checkpoint，开启 dynamic injection | 19/100 (19%) | **100/100 (100%)** |

两组匹配配置都接近满分，两组交叉配置则显著退化。这说明 dynamic branch 已经学习到
与 local stats 配套的补偿或控制映射；它不是可在任意归一化统计下直接替换官方
backbone 的独立增益模块。

因此，现阶段可支持的结论是：

- `dynamic checkpoint + local stats` 是有效的完整部署系统，正式 stateful-RNG
  结果 `495/500` 仍然有效。
- dynamic injection 对当前 local-stats checkpoint 是必要的：同一 stats 下开启后
  提升 `+58` 个百分点，精确 McNemar `p=6.94e-18`。
- 不能据此宣称 dynamic gate 优于经过正确 official stats 配置的 π0.5；严格的
  架构增益需要同数据、同 stats、同训练预算的重新训练对照。

## 2. 测评问题与设计

目标是拆开两个同时变化的因素：

1. DemoVLA interaction injection：`off` / `dynamic`；
2. 归一化统计：官方 π0.5 stats / 当前 DemoVLA 训练生成的 local stats。

所有正式因果条件均加载同一个 checkpoint：

```text
checkpoints/demovla_libero_sparse_deep_dynamic_gate/dynamic_gate_v1/29999
```

只在服务端显式切换 injection 和 norm stats，从而消除 checkpoint 权重差异。两份
stats 的 SHA256 为：

```text
official: b3a44bb2810436fb62917decaea58bd4d9110255df527dea21e8fd40c960bd84
local:    d9af3b6037f473806e807b51b3cb68859c8c2152c08d2ac00dbd9c9ba08ce221
```

两份统计并非小幅数值扰动。逐字段最大绝对差如下：

| 字段 | action | state |
|---|---:|---:|
| mean | 0.202 | 0.561 |
| std | 0.067 | 0.594 |
| q01 | 0.209 | 1.314 |
| q99 | 0.225 | 2.124 |

## 3. 协议

- 环境：`libero_object`，10 个任务；
- 正式矩阵：每任务 10 次，共 `100 episodes/condition`、400 episodes；
- environment seed：`7`；
- flow noise seed：`0`，每个 episode/replan 确定性复位；
- 输入分辨率：`224`；replan steps：`5`；初始 wait steps：`10`；
- interaction diagnostics：关闭；
- 统计：Wilson 95% CI、配对精确 McNemar 检验；
- 模型 × stats 交互：difference-in-differences（DID），按任务分层 bootstrap
  `10,000` 次。

固定 flow noise 的目的不是复现官方 stateful-RNG 分数，而是让四个条件共享同一批
初始状态和同一批采样噪声，使逐 episode 配对差异具有因果可解释性。

## 4. 前置等价性审计

先以 3 trials/task 检查官方 π0.5 checkpoint 与“DemoVLA checkpoint + injection
off”的近似等价性：

| 条件 | 成功率 | Wilson 95% CI |
|---|---:|---:|
| π0.5 + official stats | 28/30 (93.33%) | 78.68%–98.15% |
| DemoVLA off + official stats | 29/30 (96.67%) | 83.33%–99.41% |
| π0.5 + local stats | 12/30 (40.00%) | 24.59%–57.68% |
| DemoVLA off + local stats | 14/30 (46.67%) | 30.23%–63.86% |

同一 stats 下的 checkpoint 间差异均不显著：official stats 为 `+3.33pp`
（McNemar `p=1`），local stats 为 `+6.67pp`（`p=0.625`）。但是逐 episode
结果并非完全相同，所以正式矩阵不再混用两个 checkpoint。

进一步按实际推理精度把参数转换为 `bfloat16` 后，两个 checkpoint 的 51 个
非 DemoVLA tensor 数量、名称和内容全部一致，`mismatched_tensor_count=0`。剩余
rollout 差异可来自推理图或闭环数值扰动，而不是 backbone 参数发生了训练漂移。

## 5. 正式结果

### 5.1 总体结果

| 条件 | 成功数 | 成功率 | Wilson 95% CI |
|---|---:|---:|---:|
| DemoVLA off + official stats | 99/100 | 99.00% | 94.55%–99.82% |
| DemoVLA off + local stats | 42/100 | 42.00% | 32.80%–51.79% |
| Dynamic + official stats | 19/100 | 19.00% | 12.51%–27.78% |
| Dynamic + local stats | 100/100 | 100.00% | 96.30%–100.00% |

### 5.2 逐任务结果

每格为成功次数/10。

| 任务 | Off + official | Off + local | Dynamic + official | Dynamic + local |
|---|---:|---:|---:|---:|
| Alphabet soup | 9 | 9 | 0 | 10 |
| Cream cheese | 10 | 4 | 0 | 10 |
| Salad dressing | 10 | 9 | 6 | 10 |
| BBQ sauce | 10 | 0 | 5 | 10 |
| Ketchup | 10 | 7 | 2 | 10 |
| Tomato sauce | 10 | 4 | 5 | 10 |
| Butter | 10 | 2 | 1 | 10 |
| Milk | 10 | 4 | 0 | 10 |
| Chocolate pudding | 10 | 0 | 0 | 10 |
| Orange juice | 10 | 3 | 0 | 10 |

### 5.3 配对对比

| Contender − reference | 成功率差 | reference-only | contender-only | 精确 McNemar p |
|---|---:|---:|---:|---:|
| Dynamic official − Off official | −80pp | 80 | 0 | `1.65e-24` |
| Dynamic local − Off local | +58pp | 0 | 58 | `6.94e-18` |
| Off local − Off official | −57pp | 58 | 1 | `2.08e-16` |
| Dynamic local − Dynamic official | +81pp | 0 | 81 | `8.27e-25` |

模型 × stats 的 DID 为 `+138pp`，按任务分层 bootstrap 95% CI 为
`+129pp～+148pp`。DID 是两个成功率差之差，因此其取值可以超过 100pp；这里的
大正交互再次说明两者强耦合。

## 6. 对当前结论的修订

此前 `dynamic 29/30` 对 `injection_off 13/30` 的测试，两边都使用 local stats。
它能够证明 injection 对当前完整 checkpoint 的必要性，但不能隔离下列解释：

- interaction memory 提供了额外任务信息；
- dynamic branch 学会补偿 local normalization 带来的输入/输出坐标变化；
- 两种机制同时存在。

本次矩阵明确支持第二项至少占据主导作用，因为关闭注入并恢复 official stats 后达到
`99/100`，而开启注入却改用 official stats 时只有 `19/100`。所以后续论文或项目
文档应使用“匹配系统有效”“对 local stats 有显著补偿”，不应使用“已证明纯架构
优于官方 π0.5”的措辞。

## 7. 优化与下一轮实验建议

1. **把 norm stats 当作 checkpoint 的组成部分。** 服务端必须记录 stats 路径和
   SHA256；部署时发现 checkpoint/stats 指纹不匹配应拒绝启动，而不是静默运行。
2. **建立同训练条件架构对照。** 用完全相同的数据、norm stats、初始化、step、
   optimizer 和 seed 分别训练 π0.5 control 与 DemoVLA，唯一变量是 interaction
   branch。优先做 3 seeds，而不是继续放大单 seed 的 crossed-stats 诊断。
3. **保留当前 `2 × 2` 矩阵作为回归门禁。** 先跑 3 trials/task pilot；匹配条件
   低于阈值或 stats 指纹变化时才触发 10 trials/task 正式评估。
4. **动态门机制仍需单独对照。** 当前结果证明 dynamic 完整配置可用，但没有解决
   dynamic gate 与 `layer_mean` 在同 stats、同训练预算下谁更优的问题。
5. **扩展到 LIBERO Goal/Spatial/Long。** Object 已接近天花板，后续架构差异应在
   更长时序和空间组合任务上验证，并报告每类失败原因。

## 8. 与 CAC-VLA / FocusVLA 的关系

[CAC-VLA](https://arxiv.org/abs/2607.04816) 与当前 DemoVLA 最接近：两者都在
VLM 与连续 action expert 之间加入带 gate 的条件接口。CAC-VLA 进一步用未来 action
segment 编码出的 coarse-to-fine latent action 监督 VLM query；这提示下一版 DemoVLA
不应只依赖端到端 flow-matching 信号，可在 interaction queries 上增加**仅训练时使用**
的 latent-action alignment，再通过 context gate 注入 expert。对应消融必须包含
`无 latent supervision / 有 supervision`，并保持 stats、数据和训练预算一致。

[FocusVLA](https://arxiv.org/abs/2603.28740) 关注视觉信息利用：用 Modality Cascaded
Attention 限制 shortcut，再用 Focus Attention 动态选择任务相关 patch 并抑制无关
视觉噪声。它面向 autoregressive VLA，不能把论文分数直接与本项目的 π0.5 flow
expert 横比；但“先验证视觉利用瓶颈，再引入稀疏选择”的原则可直接用于 DemoVLA。
建议在不破坏完整 prefix 通路的前提下，只对 interaction-memory 支路加入可诊断的
patch top-k/soft sparsification，并报告选中 patch 比例、attention entropy、目标区域
覆盖率和跨相机分布。

优先级上，应先完成 matched-stats 的纯架构对照，再评估上述两类改动。否则 latent
action 或视觉聚焦模块的收益仍可能与 normalization 变化混杂。

## 9. 局限

- 本次只覆盖 LIBERO Object 和一个 environment seed；固定噪声结果用于因果配对，
  不应替代官方 stateful-RNG 报分。
- 每条件 100 episodes 足以确认当前巨大效应，但不足以分辨两个 99% 左右系统间的
  小差异。
- 两个 crossed-stats 条件是机制诊断，不是推荐部署配置。
- 本次没有重新训练 control，因此仍不能量化 interaction memory 的纯训练增益。

## 10. 可复现资产

- 四卡 launcher：`examples/libero/eval_demovla_norm_matrix_4gpu.sh`
- 汇总与统计：`examples/libero/summarize_norm_matrix.py`
- 参数审计：`examples/libero/audit_demovla_base_params.py`
- 等价性审计：`outputs/demovla_norm_matrix_20260824/equivalence/`
- 30-episode pilot：`outputs/demovla_norm_matrix_20260824/causal_pilot/`
- 100-episode 正式矩阵：`outputs/demovla_norm_matrix_20260824/causal_10x10/`

本地归档包含 manifest、stats 哈希、汇总 JSON/Markdown、参数审计结果和每个条件的
原始 `metrics.jsonl`。rollout 视频保留在四卡节点的对应评测目录中。
