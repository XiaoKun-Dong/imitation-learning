# DemoVLA 仿真实验结果总览与证据审计

实验数据截止：2026-08-31（Asia/Shanghai）  
实际最新入档实验：2026-08-26  
范围：仅 LIBERO 仿真；不包含真机实验，也不包含 2026-09-01 当天生成的实验

## 1. 当前结论

DemoVLA 当前有两条需要分开叙述的仿真结果。

第一条是部署性能。Dynamic-gate checkpoint `dynamic_gate_v1/29999` 与训练时 local
norm stats 配套时，在 LIBERO Object 官方 stateful-RNG 口径下达到 `495/500`
（99.0%）。但 `2 × 2` norm-stats 因果矩阵表明，dynamic branch 与 local stats
强耦合：匹配组合接近满分，交叉组合明显退化。因此 `495/500` 可以作为完整系统的
有效成绩，不能作为 interaction 架构相对正确配置的官方 π0.5 的纯增益证据。

第二条是机制研究。当前最适合继续做因果实验的 checkpoint 是 Stage A
`stage_a_fixed005_seed42_v2/4999`，推理时使用 fixed gate `0.03`。它在冻结官方
π0.5 主干的条件下建立了显著的 loss-level 样本相关 memory 作用，并在完整 LIBERO
40-task 配对评测中达到 `395/400`。它相对 injection-off 的优势显著，但相对同 gate
zero-memory 和官方 π0.5 的差异均不显著；因此还不能宣称正确 memory 内容带来了稳定的
整体成功率增益。

关键 replan 实验已经出现一个严格共享前缀下的局部内容因果样本：LIBERO-10 task 9、
episode 8 的同服务器 integrated-auto 实验在 replan 21 只改变一次 memory，得到
correct/transplant 成功而 zero/wrong-prompt/wrong-phase 失败。它是当前最强的局部行为
证据，但尚未在第二个独立 episode 或独立服务器轨迹上复现，不能升级为总体行为结论。

## 2. 结果台账

| 实验 | 严格设置 | 主要结果 | 当前判定 |
|---|---|---|---|
| Dynamic-gate 正式评估 | LIBERO Object，500 episodes，stateful RNG | `495/500`（99.0%） | 完整匹配部署配置有效 |
| Dynamic × norm stats | 同一 checkpoint，100 episodes/condition，固定 flow noise | off+official `99/100`；off+local `42/100`；dynamic+official `19/100`；dynamic+local `100/100` | dynamic 与 local stats 强耦合，旧 injection-off 对照有混杂 |
| Stage A 配对 flow loss | 1,024 个共享 observation/action/time/noise 的样本 | correct `0.00344828`；shuffled `0.00355646`；差值 `-0.00010818`，`p=1.73e-18` | 已建立 loss-level 样本相关因果作用 |
| Stage A 闭环 | 40 tasks × 10 trials，固定 noise，official stats | correct-0.03 `395/400`；off `386/400`；zero-0.03 `391/400`；official π0.5 `394/400` | 对 off 显著；对 matched zero 和官方基线不显著 |
| Memory-only prompt 反事实 | 正确 action prefix，只替换 memory prompt，1,024 样本 | wrong-task penalty `4.40e-6`，`p=0.179`；blank penalty `23.16e-6`，`p=4.64e-9` | 使用一般语言信息，但任务特异语义较弱 |
| B1-Control readout-only | fixed-0.03；冻结 backbone/extractor/gate | correct `391/400`，zero `387/400`，`p=0.4545` | loss 语义敏感性未转成显著闭环收益 |
| B1-Recovery | Q/K norm、layer conditioning、temperature、ranking 的组合改动 | entropy 上升，但 shuffled 区分和 wrong-prompt 敏感性下降 | 组合方案失败，不能归因到单个组件 |
| S1-Semantic | 相对 S1-Control 只增加同图像 wrong-prompt ranking | wrong-prompt effect 未增强，shuffled advantage 反而下降 | 未通过离线门槛，未启动 400-episode rollout |
| Replan 10，5 个 server launch | 严格 lockstep，共享 zero 前缀 | one-shot correct 相对 zero：2 正翻转、1 反翻转、2 同成败 | 局部作用不具跨 launch 稳健性 |
| Replan 11 负对照 | 同一 lockstep 协议 | correct、wrong-prompt、transplant、wrong-phase 脉冲全部成功 | 证明仅 correct-vs-zero 翻转可能只是通用 residual 脉冲 |
| Episode 8 integrated-auto | 同 server 先现场确认 correct 成功/zero 失败，再自动选最早候选 r21 | one-shot correct/transplant 成功；zero/wrong-prompt/wrong-phase 失败 | 当前最强局部内容证据，待独立复现 |

表中的配对显著性、翻转数和具体协议分别来自
[Stage A 因果评测](demovla_stage_a_causal_evaluation_zh.md)、
[norm-stats 因果矩阵](demovla_norm_stats_causal_evaluation_20260824.md)和
[关键 replan 因果定位](demovla_key_replan_causal_progress_zh.md)。不同表使用的 suite、
checkpoint、norm stats 和 RNG 口径不同，不能把成功率直接横向排名。

## 3. 当前可以与不可以陈述的结论

可以陈述：

1. DemoVLA dynamic checkpoint 与对应 local stats 构成一个有效的 LIBERO Object
   部署系统，成绩为 `495/500`。
2. 冻结主干的 Stage A 支路可训练，并在共享样本和噪声的配对实验中显著依赖正确样本的
   interaction memory。
3. Stage A fixed-0.03 相对关闭支路有显著闭环优势，但该对照不能单独隔离“非零
   residual”与“正确 memory 内容”。
4. Memory extractor 会随 prompt 和视觉输入变化；主要瓶颈更接近 action readout 的
   内容选择和注入时机，而不是 extractor 完全没有信息。
5. 在至少一个严格 lockstep 局部状态中，正确/成功轨迹 memory 的单次脉冲足以改变长期
   任务结果，且错误提示和错误阶段 memory 没有产生同样效果。

暂时不可以陈述：

1. DemoVLA 已经显著优于正确配置的官方 π0.5。
2. Dynamic gate 的纯架构收益已经被 norm stats 解耦证明。
3. Correct memory 相对 matched zero-memory 已在总体闭环成功率上达到显著优势。
4. Wrong-prompt ranking、高 attention entropy 或更大的 gate 会自动提高行为表现。
5. 单个 replan 的正向结果已经具有跨 episode、跨 server 的稳健性。

## 4. 当前主线选择

用途不同，推荐对象也不同：

| 用途 | 推荐对象 | 原因 |
|---|---|---|
| 复现 LIBERO Object 部署成绩 | Dynamic `dynamic_gate_v1/29999` + local stats | 已有 `495/500` 正式成绩；必须绑定 stats 指纹 |
| 继续做可归因的机制实验 | Stage A `stage_a_fixed005_seed42_v2/4999` + fixed gate `0.03` + official stats | 主干冻结、干预接口完整，已有 loss 与闭环配对基线 |
| 下一轮训练初始化 | 暂不切到 B1-Recovery 或 S1-Semantic | 两者均未通过预设机制/行为门槛 |

下一轮不宜先扩大训练规模或同时修改 extractor、readout 和 gate。优先级应为：

1. 在第二个独立 episode 上复现 episode 8 的 integrated-auto 内容翻转；
2. 所有候选点继续只根据 zero-vs-correct 动作差预注册，不读取 intervention outcome；
3. 若内容特异翻转可复现，再训练 key-state/counterfactual-selective readout 或
   event-triggered bounded gate；
4. 最终架构比较使用共同初始化、数据、stats、预算和至少 3 个训练 seed 的
   parameter-matched control。

## 5. 本地证据与归档缺口

当前 checkout 中可以直接复核的原始结果包括：

- `outputs/demovla_norm_matrix_20260824/`：完整 norm-stats 指标、汇总和参数审计；
- `outputs/demovla_stage_a/.../causal_loss_formal_20260825/`：Stage A 1k/3k/4999
  的逐样本配对 loss；
- `outputs/demovla_stage_a/.../prompt_memory_only_counterfactual_formal_20260826/`：
  正确、suite-swap 与 blank memory prompt 的逐样本结果；
- `artifacts/demovla_stage_a_effective_overlay_step4999_fixed003_seed7/`：端到端有效
  attention 图和数组；
- `artifacts/demovla_key_replan_donors_step4999_fixed003_seed7/`：关键 replan donor
  memory 资产。

以下目录被实验文档引用，但不在当前本地工作区中：

```text
outputs/demovla_b1_eval/
outputs/demovla_s1_eval/
outputs/demovla_key_replan/
data/libero/videos/demovla_stage_a_causal_step4999_formal10_seed7/
data/libero/videos/full_matched/
```

这不否定已经记录的结果，但意味着当前 checkout 无法独立重算 B1/S1 的逐样本统计，
也无法重新审计 lockstep 的 `branch_audit.npz` 和闭环视频。论文定稿前应从评测节点或
归档恢复这些目录，并至少保存：checkpoint/config、norm-stats SHA256、git commit、完整
命令、环境与 flow-noise seed、server/GPU 标识、逐 episode metrics、统计汇总和失败视频。

## 6. 文档入口

- [DemoVLA 总体设计与历史结果](demovla.md)
- [Stage A、B1、S1 因果评测](demovla_stage_a_causal_evaluation_zh.md)
- [Memory overlay 与 prompt 反事实](demovla_memory_overlay_and_causal_next_steps_zh.md)
- [关键 replan 因果定位](demovla_key_replan_causal_progress_zh.md)
- [Dynamic × norm stats 因果矩阵](demovla_norm_stats_causal_evaluation_20260824.md)
- [Dynamic gate collapse 与恢复](demovla_gate_collapse_and_recovery_zh.md)

