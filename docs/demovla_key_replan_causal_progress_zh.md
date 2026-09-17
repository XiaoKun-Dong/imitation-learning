# DemoVLA 关键 Replan 因果定位：进度、证据边界与后续计划

更新时间：2026-08-26（Asia/Shanghai）

## 0. 当前状态摘要

本文按实验发生顺序保留了 replan 10、replan 11 和独立 episode 8 的完整推进记录。
截至本轮仿真数据截止点，当前状态不是“仍在采集 replan 10”，而是：

1. replan 10 已完成预设的 5 个独立 policy-server launch；one-shot correct 相对
   zero 为 2 次正翻转、1 次反翻转、2 次同成败，不具备跨 launch 稳健性；
2. replan 11 已确定为通用 nonzero-memory residual 脉冲的负对照，不能作为内容特异
   证据；
3. 独立 episode 8 的 integrated-auto 协议在同一 server 生命周期内现场确认
   correct 成功/zero 失败，并自动选择最早候选 replan 21；one-shot correct 与
   transplant 成功，zero、wrong-prompt 和 wrong-phase 失败；
4. episode 8/replan 21 是当前最强的局部内容因果结果，但仍需在第二个独立 episode
   或独立 server 轨迹上复现，才能进入论文主证据表。

简要结论和其他仿真实验之间的关系见
[DemoVLA 仿真实验结果总览与证据审计](demovla_simulation_results_status_20260831_zh.md)。

## 1. 当前目标

本阶段不再通过增加训练步数或组合修改结构来追求平均 loss，而是回答一个更窄、也更关键的问题：

> 在闭环成败翻转附近，正确 interaction memory 是否会改变关键动作决策；这种改变由哪一注入层传递，并且是否足以改变任务结果？

因果链按证据强度分为四级：

1. 同一 observation、同一 flow noise 下，替换 memory 会改变动作；
2. success-trajectory memory transplant 能把 zero-memory 动作拉回 correct 动作；
3. 单层 knockout 能定位传递该动作差异的层；
4. 在严格共享的失败前缀上，仅改变关键 replan 的 memory 条件即可翻转闭环成败。

只有第 4 级成立，才能写成局部行为级 memory 内容因果证据。第 1–3 级只能证明动作级作用和候选机制，不能单独证明成功率提升。

## 2. 固定实验对象

| 项目 | 设置 |
|---|---|
| Checkpoint | Stage A `stage_a_fixed005_seed42_v2/4999` |
| 推理 gate | 第 4、9、14 层均固定为 `0.03` |
| 主干和 stats | 官方 `pi05_libero` |
| 关键任务 | `libero_10` task 9，episode 6 |
| 环境 seed | 7 |
| Flow noise | 按 task、episode、replan 构造的 stateless seed |
| Replan 间隔 | 5 个环境步 |

选择该 episode 的原因是正式 400-episode 配对结果中，它属于 `correct-0.03` 成功、`zero-0.03` 失败的翻转样本。新采集的完整轨迹也现场复现了这一成败关系：correct 在 439 步成功，zero 到 530 步仍失败。

## 3. 已实现的因果干预接口

### 3.1 Memory source 解耦

已增加两种只改变 memory 路径、不改变 action VLM prefix 的协议字段：

- `INTERACTION_MEMORY_PROMPT_KEY`：action prefix 继续使用正确提示，仅让 memory extractor 看到错误提示；
- `INTERACTION_MEMORY_OVERRIDE_KEY`：绕过 extractor，直接注入给定的 interaction memory tensor。

因此 `wrong_prompt` 和 `transplant` 不会把提示词变化泄漏到基础策略的 action prefix，满足单因素要求。

### 3.2 干预条件

当前支持：

- `correct`：正确 memory，三层 gate `0.03`；
- `zero`：memory 置零，gate 保持 `0.03`；
- `wrong_prompt`：同图像、同状态、正确 action prompt，只替换 memory prompt；
- `transplant`：注入成功轨迹同 phase/replan 的 memory；
- `knockout_layer4/9/14`：只把一层 gate 精确置零，其余两层保持 `0.03`；
- `one_shot_correct/transplant`：只在指定 replan 使用 correct 或 transplant，随后回到 zero。

固定 gate override 已允许精确值 `0`，所以 layer knockout 不再用近零近似。

### 3.3 代码与验证

主要实现：

```text
packages/openpi-client/src/openpi_client/base_policy.py
src/openpi/models/demovla.py
src/openpi/policies/policy.py
examples/libero/run_demovla_key_replan.py
examples/libero/eval_demovla_key_replan_5gpu.sh
examples/libero/run_demovla_branch_from_replan.py
examples/libero/eval_demovla_branch_5gpu.sh
examples/libero/run_demovla_lockstep_branch.py
examples/libero/eval_demovla_lockstep_branch_5gpu.sh
scripts/analyze_demovla_key_replans.py
```

相关 targeted tests、Ruff、Python compile、shell syntax 和 `git diff --check` 均已通过。

## 4. 关键 Replan 的预注册规则

候选点只根据 zero-memory 失败轨迹相对 correct 动作的差异选择，不能利用 transplant、wrong-prompt 或 knockout 的结果反向挑点，以避免结果泄漏。

主指标为接下来实际执行的 5 步动作差异：

```text
executed_l2 = ||a_zero[0:5] - a_correct[0:5]||_2
```

稳健阈值为：

```text
median(executed_l2) + 3 × scaled_MAD(executed_l2)
```

zero 失败轨迹统计：

- median：`0.012687`
- scaled MAD：`0.008007`
- 候选阈值：`0.036708`

## 5. 已完成的动作级结果

### 5.1 全轨迹均值

同一 observation 和同一 flow noise 下，相对 correct 的 executed action L2：

| 条件 | Correct 成功轨迹 | Zero 失败轨迹 |
|---|---:|---:|
| Zero memory | 0.020132 | 0.016754 |
| Wrong prompt memory | 0.004468 | 0.003891 |
| Success-memory transplant | 0.006871 | 0.004612 |
| Knockout layer 4 | 0.010971 | 0.009607 |
| Knockout layer 9 | 0.008379 | 0.007245 |
| Knockout layer 14 | 0.006373 | 0.005870 |

这说明 nonzero memory 的存在会系统性改变动作；但 wrong prompt 仍非常接近 correct，任务提示语义特异性依旧较弱。Layer 4/9 knockout 的动作影响大于 layer 14，早中层 readout 是更值得优先修改的候选。

### 5.2 Zero 失败轨迹候选点

| 排名 | Replan | 环境步 | Zero L2 | Wrong L2 | Transplant rescue | KO4 | KO9 | KO14 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 74 | 370 | 0.071408 | 0.007472 | 0.959 | 0.027613 | 0.026050 | 0.013968 |
| 2 | 94 | 470 | 0.050885 | — | — | 0.018997 | 0.023133 | 0.006864 |
| 3 | 83 | 415 | 0.049413 | 0.004056 | 0.786 | 0.033549 | 0.020680 | 0.006343 |
| 4 | 10 | 50 | 0.045152 | 0.004621 | 0.970 | 0.021497 | 0.016726 | 0.009896 |
| 5 | 11 | 55 | 0.042124 | 0.003609 | 0.947 | 0.027579 | 0.016796 | 0.004595 |
| 6 | 82 | 410 | 0.038971 | 0.003869 | 0.761 | 0.028456 | 0.015306 | 0.005064 |

`transplant rescue` 衡量 transplant 动作相对 zero 动作向 correct 动作回归的比例。Replan 10、11、74 的比例均超过 `0.94`，构成较强的动作级 memory transplant 证据。

当前可以支持：

1. correct memory 相对 zero memory 会改变关键动作；
2. 在若干关键状态，成功轨迹 memory 能把动作方向拉回 correct；
3. layer 4/9 比 layer 14 更可能承载关键作用。

当前仍不能支持：

1. wrong-task memory 会稳定破坏动作；
2. 某个单层对任务成功具有必要性；
3. transplant 已经改变闭环任务成败。

## 6. 闭环分支的复现性审计

### 6.1 直接 simulator-state restore 无效

从 replan 74 和 replan 10 保存的 MuJoCo `sim_state` 直接恢复后，分别出现“所有条件均失败”和“所有条件均成功”。原因是 `sim_state` 不包含 controller、observer 等完整历史状态。因此这两轮不能用于行为级因果结论。

### 6.2 跨进程 full-prefix replay 仍处于成功边界

随后改为从 episode 初始状态重放完整 zero 前缀，再在 replan 10 切换条件。结果如下：

| 条件 | 结果 | 结束步 |
|---|---:|---:|
| Zero | 成功 | 505 |
| Correct | 成功 | 418 |
| Wrong prompt | 成功 | 430 |
| Transplant | 成功 | 423 |
| One-shot correct | 成功 | 505 |
| One-shot transplant | 成功 | 488 |
| KO layer 4 | 成功 | 465 |
| KO layer 9 | 失败 | 520 |
| KO layer 14 | 成功 | 427 |

原始 zero 轨迹失败，而本轮 zero 成功，所以不能把 correct/transplant 的成功解释为失败翻转。KO9 的单次失败也只能作为候选线索，不能据此宣称 layer 9 必要。

审计发现，原始 correct/zero 独立轨迹在 replan 0 的图像、腕部图像、机器人状态，以及除 transplant 外所有条件的动作输出都逐元素一致；长时闭环差异来自任务位于数值敏感的成功边界，微小差异会在数百步后放大。历史同一 task/episode 也曾出现 398–453 步之间的不同成功轨迹。因此，不能继续把跨进程单次成败当作局部反事实证据。

## 7. 单进程严格 Lockstep 结果

已新增单进程 lockstep 分支评测：

1. 为 9 个条件分别创建独立 LIBERO 环境；
2. 每个环境使用相同 seed 和同一 episode 初始状态；
3. 在 branch replan 前只查询一次 zero policy，把完全相同的动作逐步施加到所有环境；
4. 每一步比较完整 simulator state，任何非零差异都立即终止并判为无效；
5. 在 branch observation 再次验证图像与机器人状态逐元素一致；
6. 只有到预注册的 replan 10 才分别应用 correct、wrong、transplant、one-shot 和 layer knockout。

首轮有效结果目录：

```text
outputs/demovla_key_replan/
  stagea4999_libero10_task9_ep6_lockstep_r010_v2/
```

一致性验收全部通过：

| 校验项 | 结果 |
|---|---:|
| `initialization_max_abs_diff` | `0.0` |
| `prefix_max_abs_diff` | `0.0` |
| `branch_image_equal` | `true` |
| `branch_state_equal` | `true` |

因此，所有条件在 replan 10 之前具有逐元素一致的 50 步 zero-memory 前缀；分支时看到的图像和机器人状态也逐元素一致。

### 7.1 闭环结果

| 条件 | 结果 | 结束步 | 解释 |
|---|---:|---:|---|
| Zero | 失败 | 520 | 共同失败基线 |
| Correct（持续） | 失败 | 520 | 持续注入不必然有利 |
| Wrong prompt（持续） | 失败 | 520 | 尚不能区分语义作用 |
| Transplant（持续） | 未完成 | — | replan 86 后 donor 用尽，不计成败 |
| **One-shot correct** | **成功** | **493** | 仅 replan 10 使用 correct，随后回到 zero |
| **One-shot transplant** | **成功** | **406** | 仅 replan 10 移植成功轨迹 memory，随后回到 zero |
| Knockout layer 4（持续） | 成功 | 456 | 候选层效应，尚需复现 |
| Knockout layer 9（持续） | 失败 | 520 | 与 zero 同结果 |
| Knockout layer 14（持续） | 成功 | 507 | 候选层效应，尚需复现 |

### 7.2 分支动作审计

在 replan 10 的同一 observation、同一 noise 上，相对 zero 的前 5 步动作 L2：

| 条件 | 相对 Zero | 相对 Correct |
|---|---:|---:|
| Correct | 0.043031 | 0 |
| Wrong prompt | 0.044311 | 0.003705 |
| Transplant | 0.043649 | 0.002272 |
| KO layer 4 | 0.025554 | 0.020312 |
| KO layer 9 | 0.027893 | 0.015919 |
| KO layer 14 | 0.035118 | 0.010444 |

`one_shot_correct` 与 `correct` 的完整 action chunk 逐元素一致；`one_shot_transplant` 与 `transplant` 的完整 action chunk 也逐元素一致。由此可排除 one-shot 实现选择了不同采样噪声或动作图的可能。

### 7.3 当前证据解释

首轮结果首次建立了如下局部行为因果链：

```text
完全相同的 zero 前缀和关键状态
→ 只在 replan 10 改变一次 memory 条件
→ 后续重新使用 zero 策略
→ zero 失败，而 correct/transplant one-shot 成功
```

这支持“replan 10 的 interaction memory 脉冲足以改变该 rollout 的长期结果”。持续 correct 失败而 one-shot correct 成功，说明作用并非“memory 越多越好”，更像一次关键纠偏；这也为后续 event-triggered gate 或 phase-selective injection 提供了直接依据。

但是该结果仍只有一个独立 launch，而且 wrong prompt 在该点产生的动作非常接近 correct。当前可以证明 nonzero/成功轨迹 memory 的局部行为作用，尚不能证明任务语义正确性本身是成功的必要条件。必须先做独立复现和 one-shot wrong-memory 对照。

### 7.4 独立服务器进程复现审计

第二次独立 launch（`lockstep_r010_v3`）仍通过所有组内一致性校验，但成败模式没有完整复现：

| 条件 | v2 | v3 |
|---|---:|---:|
| Zero | 失败 | 成功（453 步） |
| Correct（持续） | 失败 | 成功（416 步） |
| Wrong prompt（持续） | 失败 | 失败 |
| One-shot correct | 成功 | 失败 |
| One-shot transplant | 成功 | 成功（488 步） |
| KO layer 4 | 成功 | 成功 |
| KO layer 9 | 失败 | 失败 |
| KO layer 14 | 成功 | 成功 |

v2/v3 的初始图像和初始机器人状态逐元素一致，但经过 50 步 zero 前缀后，分支机器人状态最大绝对差达到 `1.005e-3`，分支图像也不同。对应的 zero/correct/transplant 前 5 步 action chunk 跨 launch L2 分别为 `0.00889/0.00841/0.00898`。

因此，lockstep 协议解决了单次 launch 内的反事实配对，却还没有解决不同 policy-server 进程之间的 GPU 数值差异及其闭环放大。v2 的 one-shot 翻转是有效的单次局部因果观测，但尚不具备跨 launch 稳健性。稳定信号目前只有：

- `one_shot_transplant` 两轮都成功；
- KO layer 4/14 两轮都成功；
- KO layer 9 两轮都失败；
- 但 zero 基线一败一成，不能据此计算稳定相对效应。

随后完成的下一项审计是在同一组 policy-server 进程上连续运行 3 个全新 LIBERO 客户端，并保存全部 zero-prefix action chunks。结果见下一节：同服务器三次逐元素一致，而更换服务器后闭环轨迹发生变化，主要不稳定性因此被定位到跨服务器数值执行路径及其闭环放大。

### 7.5 同一服务器三次重复与内容对照

输出目录：

```text
outputs/demovla_key_replan/
  stagea4999_libero10_task9_ep6_lockstep_r010_repeat3_v1/
```

同一组 policy-server 进程连续运行 3 个全新 LIBERO 客户端。三轮的全部 `branch_audit.npz` 数组逐元素一致，包括 10 个 zero-prefix action chunks、分支 observation、机器人状态和所有条件的 action chunk。三轮闭环结果和结束步也完全一致：

| 条件 | 成功次数 | 固定结束步 |
|---|---:|---:|
| Zero | 0/3 | 520 |
| Correct（持续） | 0/3 | 520 |
| Wrong prompt（持续） | 0/3 | 520 |
| **One-shot correct** | **3/3** | **481** |
| One-shot wrong-prompt | 0/3 | 520 |
| One-shot transplant | 0/3 | 520 |
| One-shot wrong-phase（r74 → r10） | 0/3 | 520 |
| KO layer 4（持续） | 3/3 | 460 |
| KO layer 9（持续） | 0/3 | 520 |
| KO layer 14（持续） | 0/3 | 520 |

持续 transplant 在 donor replan 86 用尽，仍不计为完整失败 rollout；表中的 one-shot transplant 不存在 donor 用尽问题，是有效的失败对照。

分支点相对 zero 的前 5 步动作 L2：

| 条件 | 相对 Zero | 相对 Correct | 闭环结果 |
|---|---:|---:|---:|
| Correct / one-shot correct | 0.042781 | 0 | 成功（one-shot） |
| Wrong prompt / one-shot wrong-prompt | 0.043961 | 0.003190 | 失败 |
| Transplant / one-shot transplant | 0.043463 | 0.003220 | 失败 |
| One-shot wrong-phase | 0.039909 | 0.005039 | 失败 |
| KO layer 4 | 0.025363 | 0.020885 | 成功 |
| KO layer 9 | 0.027788 | 0.016448 | 失败 |
| KO layer 14 | 0.035407 | 0.009383 | 失败 |

这组三次重复表明，在固定数值执行路径下，replan 10 是可稳定复现的内容敏感决策边界：只注入一次由当前图像和正确任务提示提取的 memory 可以成功；同图像 wrong-prompt memory、成功轨迹 transplant memory，以及来自错误阶段的 memory 均不能成功。特别是 wrong-prompt/transplant 动作与 correct 很接近，却仍产生相反闭环结果，说明不能用动作 L2 大小代替行为验证。

需要注意，三次共享同一服务器进程，因此它们证明的是客户端/环境重建可重复性，不是三个独立数值随机样本。跨服务器稳健性仍按独立 server launch 统计。

### 7.6 五个独立服务器进程的稳健性

为避免只报告正向 launch，replan 10 在达到预先设定的 5 个独立 policy-server 进程后停止追加。每个进程内都通过零差异共同前缀校验；同服务器三重复只计作一个独立进程。配对翻转以 zero 为 reference：

| 条件 | 成功服务器数 | 相对 Zero 正翻转 | 反翻转 | 同成败 |
|---|---:|---:|---:|---:|
| Zero | 3/5 | — | — | — |
| One-shot correct | 4/5 | 2 | 1 | 2 |
| One-shot transplant | 4/5 | 1 | 0 | 4 |
| KO layer 4（持续） | 4/5 | 2 | 1 | 2 |
| KO layer 9（持续） | 0/5 | 0 | 3 | 2 |
| KO layer 14（持续） | 4/5 | 1 | 0 | 4 |
| Correct（持续） | 1/5 | 0 | 2 | 3 |
| Wrong prompt（持续） | 0/5 | 0 | 3 | 2 |

新增 one-shot 内容对照只在后 3 个独立服务器中存在：

| 条件 | 成功服务器数 | 相对 Zero 正翻转 | 反翻转 | 同成败 |
|---|---:|---:|---:|---:|
| One-shot wrong-prompt | 2/3 | 0 | 0 | 3 |
| One-shot wrong-phase | 1/3 | 0 | 1 | 2 |

五服务器结果的正确解释是：

1. replan 10 的 one-shot correct 在部分数值轨迹上足以把失败变成成功，但也出现过一次反翻转；2 比 1 个 discordant launch 的样本量不足，不能宣称跨服务器稳健优势；
2. one-shot wrong-prompt 在已测 3 个服务器上始终与 zero 同成败，说明它没有显示独立行为收益；但只有一个服务器出现 correct 成功、wrong-prompt 失败的直接内容翻转，语义内容因果仍需独立 episode；
3. one-shot transplant 没有出现相对 zero 的反翻转，但只多救回 1/5 个服务器；它也不是稳定替代当前 correct memory 的方案；
4. 持续 correct 仅 1/5 成功，明显弱于 one-shot correct 的 4/5。这是目前最一致的结构信号：问题不在“memory 完全无效”，而在持续、无时机选择的注入会把关键纠偏与后续有害扰动混在一起；
5. 不同服务器进程产生不同的闭环数值轨迹，单 episode 的行为干预必须按 paired flips 和独立服务器/episode 复现报告，不能把同服务器重复当独立统计量。

replan 10 到此停止追加实验。下一步按预注册候选顺序进入 replan 11，并优先寻找第二个独立 episode，而不是继续围绕同一个成功边界增加样本。

### 7.7 相邻候选 Replan 11：通用脉冲负对照

replan 11 使用与 replan 10 相同的 checkpoint、服务器、环境、zero-prefix 和干预集合，在同一组服务器内连续重复 3 次。三轮全部 audit 数组逐元素一致，所有共同前缀校验通过。

| 条件 | 成功次数 | 固定结束步 |
|---|---:|---:|
| Zero | 0/3 | 520 |
| Correct（持续） | 0/3 | 520 |
| Wrong prompt（持续） | 0/3 | 520 |
| **One-shot correct** | **3/3** | **466** |
| **One-shot wrong-prompt** | **3/3** | **455** |
| **One-shot transplant** | **3/3** | **467** |
| **One-shot wrong-phase** | **3/3** | **437** |
| KO layer 4/9/14（持续） | 0/3 | 520 |

分支点动作 L2：

| 条件 | 相对 Zero | 相对 Correct |
|---|---:|---:|
| Correct | 0.042143 | 0 |
| Wrong prompt | 0.041201 | 0.003094 |
| Transplant | 0.041904 | 0.003143 |
| Wrong phase | 0.040654 | 0.006640 |
| KO layer 4 | 0.016981 | 0.027765 |
| KO layer 9 | 0.025683 | 0.017313 |
| KO layer 14 | 0.041677 | 0.003285 |

replan 11 是重要阴性机制结果：它能稳定证明“一次 nonzero memory 路径产生的动作扰动足以改变闭环结果”，但 correct、wrong-prompt、transplant 和明显错误阶段 memory 全部成功，所以不能证明正确 memory 内容具有特异性。该点不能作为语义因果主结果，应作为“仅有 correct-vs-zero 翻转仍可能是通用残差效应”的负对照。

结合 r10/r11，候选筛选不能只看 `||a_correct-a_zero||` 大小；两点差异都约 `0.042`，但内容特异性不同。后续 key-state 训练或门控必须加入 counterfactual selectivity：不仅放大 correct-vs-zero，还要要求 correct 相对 wrong-prompt/wrong-phase 的关键动作或价值更优。

### 7.8 独立 Episode 8 的现场翻转与候选

正式评测翻转列表中的同一 task、episode 8 已重新采集，现场再次得到：

| 轨迹 | 结果 | 步数 | Replans |
|---|---:|---:|---:|
| Correct 0.03 | 成功 | 425 | 84 |
| Zero 0.03 | 失败 | 530 | 104 |

输出目录：

```text
outputs/demovla_key_replan/stagea4999_libero10_task9_ep8_v1/
```

zero 失败轨迹的稳健阈值为 `0.057848`（median `0.020999`，scaled MAD `0.012283`）。候选如下：

| 排名 | Replan | 环境步 | Zero L2 | Wrong L2 | Transplant rescue | KO4 | KO9 | KO14 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 99 | 495 | 4.007523 | 1.958822 | — | 4.007645 | 3.265238 | 1.578573 |
| 2 | 84 | 420 | 0.136865 | 0.006789 | — | 0.071315 | 0.052844 | 0.024648 |
| 3 | 93 | 465 | 0.092656 | 0.004007 | — | 0.053475 | 0.026802 | 0.022060 |
| 4 | **73** | **365** | **0.065357** | **0.009349** | **0.933** | **0.036323** | **0.024137** | **0.012335** |
| 5 | 90 | 450 | 0.058384 | 0.007263 | — | 0.033334 | 0.019514 | 0.012797 |

按预注册规则选择 replan 73：它是阈值以上最早的候选，仍剩 155 个控制步，而且成功 donor 尚未耗尽，可以执行严格 transplant。replan 99 的巨大差异发生在极晚期且无对应成功 donor，不优先。

episode 8 / replan 73 的第一组 lockstep 在另一服务器进程上重复 3 次，三轮逐元素一致，但 zero 已经 3/3 成功（493 步），其余所有有效 one-shot 和 knockout 也成功。因此该组没有失败可救回，不提供行为翻转证据。它再次说明 trace 和 branch 分别重启服务器会改变数值轨迹。

为此，评测协议已进一步收紧为同一服务器生命周期：

```text
启动一次 5 个 policy servers
→ 采集 correct 完整轨迹
→ 采集 zero 完整轨迹
→ 自动检查 correct=success 且 zero=failure
→ 只有通过检查才立即在同一 servers 上执行 lockstep branch
```

第一次合并协议尝试中，correct 在 433 步成功，但 zero 在 424 步也成功，因此按停止规则中止分支。脚本现已加入自动 eligibility gate；后续若不满足现场翻转，只保存 `post_branch_status.json`，不会再启动昂贵的 11 环境分支。

随后继续执行合并协议尝试。所有尝试（包括 zero 成功而跳过的轮次）均保留，避免只报告被筛中的服务器轨迹。

第二次合并协议通过现场翻转门槛：correct 433 步成功、zero 530 步失败。但固定使用前一服务器得到的 r73 后，严格 lockstep 中所有条件均失败。结论是 r73 的大动作差发生得太晚，step 365 时失败已不可逆；它是失败症状，不是可救回原因。

重新分析这次同服务器 zero 轨迹后，候选变为：

| 排名 | Replan | 环境步 | Zero L2 | Wrong L2 | Transplant rescue | KO4 | KO9 | KO14 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 45 | 225 | 2.002840 | 0.006343 | −0.001 | 2.010848 | 1.997331 | 0.012094 |
| 2 | 84 | 420 | 0.162815 | 0.005206 | 0.847 | 0.083484 | 0.080867 | 0.024789 |
| 3 | **23** | **115** | **0.055334** | **0.003193** | **0.935** | **0.039499** | **0.023074** | **0.010662** |

r45 的约 `2.0` 差异来自夹爪动作符号翻转；KO4/KO9 随 zero 翻转，KO14 仍接近 correct，提示早中层可能控制关键 grasp/release 决策。r23 是阈值以上最早候选，且 success-memory transplant 能把动作拉回 correct。

候选位置本身也随服务器数值轨迹变化，因此协议现已支持 `POST_BRANCH_REPLAN=auto`：完成同服务器 trace 后，由固定的 `median + 3×scaled_MAD` 规则自动选择最早候选，再执行 branch。选择过程只读取 zero-vs-correct 动作差，不读取任何 wrong/transplant/knockout 的闭环结果。

第三次合并协议（`stagea4999_libero10_task9_ep8_integrated_auto_v4`）完成了首个完整自动闭环：同服务器 trace 为 correct 436 步成功、zero 530 步失败；固定规则自动选择最早候选 replan 21（step 105）；branch 的初始化、共享前缀、图像和状态校验全部为零差异。

| 条件 | 结果 | 结束步 |
|---|---:|---:|
| Zero | 失败 | 520 |
| Correct（持续） | 成功 | 419 |
| Wrong prompt（持续） | 成功 | 417 |
| **One-shot correct** | **成功** | **493** |
| **One-shot transplant** | **成功** | **429** |
| One-shot wrong-prompt | 失败 | 520 |
| One-shot wrong-phase | 失败 | 520 |
| KO layer 4/9/14（持续） | 失败 | 520 |

分支点相对 zero 的前 5 步动作 L2 为：correct `0.05782`、wrong-prompt `0.05860`、transplant `0.05806`、wrong-phase `0.06816`。Wrong-prompt 和 transplant 动作都很接近 correct（分别相差 `0.00293/0.00383`），但只有 one-shot transplant 与 correct 一样成功；wrong-prompt 失败。说明动作 L2 只能定位敏感状态，行为验证才能区分 memory 内容。

这是目前最强的局部内容因果证据：在独立 episode、同服务器现场翻转、自动预注册最早候选、完全相同 zero 前缀下，只改变一次 memory 内容即可得到 correct/transplant 成功而 zero/wrong-prompt/wrong-phase 失败。它仍需在另一个独立 episode 或独立服务器上复现后再作为论文主结论，但已经足以把下一轮训练目标从“增强全局注入”收窄为“增强早期关键状态的 counterfactual-selective readout”。

行为结果只有同时满足以下条件才进入论文最终证据表：

- `initialization_max_abs_diff == 0`；
- `prefix_max_abs_diff == 0`；
- `branch_image_equal == true`；
- `branch_state_equal == true`；
- zero 与某个单因素条件在共同前缀后出现可重复的成败差。

## 8. 后续计划与停止规则

### P1：复现 replan 10 lockstep 翻转

- replan 10 的 5 个独立 server launch 已完成，one-shot correct 相对 zero 为
  2 正翻转、1 反翻转、2 同成败，结论是不具跨 launch 稳健性；
- 同一 policy-server 内 3 次客户端重复逐元素一致，说明环境重建本身可重复，主要
  差异来自跨 server 数值轨迹；
- replan 11 已确认是通用 nonzero-memory 脉冲点，不是内容特异点；
- 独立 episode 8 的 integrated-auto r21 已得到首个完整内容特异翻转；
- 下一验收目标是在第二个独立 episode 或独立 server 轨迹上复现 episode 8 式结果；
- 后续继续报告所有 paired flips，不筛掉 zero 已成功或发生反翻转的阴性尝试；
- 若某层 knockout 独立导致失败，需同时在多个候选点和至少另一个 episode 上复现，
  确认不是单轨迹偶然。

### P2：补充 one-shot 内容对照

在复现现有协议后，加入但不改变其他条件：

- `one_shot_wrong_prompt`：只在 replan 10 让 memory extractor 看到错误任务提示；
- `one_shot_wrong_phase`：只在 replan 10 注入同一成功轨迹中远离当前阶段的 memory；
- 如有跨任务、同维度 donor，再加入 `one_shot_wrong_task_memory`。

若 correct/transplant 成功而 wrong-prompt/wrong-phase 失败，才支持“memory 内容正确性”而不只是“非零残差脉冲”的行为因果作用。

### P3：扩展到独立翻转和 matched control

至少加入：

- 另一个现场重新采集到的 correct-only 翻转；
- 一个 correct/zero 都成功的 matched control；
- 一个 correct/zero 都失败的 matched control（若能找到）；
- 多个 flow-noise seed，报告条件成功概率而不是只报告单次视频。

候选点必须在查看 intervention outcome 前由 zero-vs-correct action gap 预注册。

### P4：根据因果定位决定结构，不提前改 extractor

- 若 KO4/KO9 的关键动作和行为作用稳定、KO14 弱：把单因素结构实验限定为早中层注入重分配，优先测试移除 layer 14 或把固定总注入预算转给 layer 4/9；
- 若 transplant 有效而 wrong-prompt 仍接近 correct：主要瓶颈是 memory 语义可辨识性，训练目标改为关键状态上的同图像 hard-negative，而不是全时段平均 ranking；
- 若 correct/zero 动作差明显但闭环始终不翻转：memory 影响的不是任务瓶颈动作，应做 phase/key-state weighting，而不是继续增大 gate；
- 若所有严格实验都不能稳定复现行为作用：论文结论退回 loss-level sample-specific causality，不宣称 behavior-level content causality。

### P5：最小可发表证据包

建议论文最终至少包含：

1. Stage A 1,024-sample correct/off/zero/shuffled 配对 flow-loss；
2. 400-episode correct-0.03 vs zero-0.03 与官方 π0.5 对比；
3. 关键 replan 的同状态动作差分、transplant rescue 和 layer knockout；
4. 至少两个独立 episode 上通过共同前缀校验的行为干预复现；
5. 阴性结果：wrong-prompt ranking 和高 entropy readout 并未自动增强因果作用。

在 P1–P3 没有通过前，不启动新的大规模训练，也不恢复 dynamic gate。这样可以保证下一次结构修改由已定位的行为瓶颈驱动，而不是再次做多因素试错。
