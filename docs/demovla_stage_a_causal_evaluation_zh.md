# DemoVLA 阶段 A：固定门控预热与因果评测报告

## 1. 实验目标

本实验验证：在冻结官方 π0.5-LIBERO 主干后，仅训练 DemoVLA 交互记忆支路，是否能够形成样本相关、对动作预测有利的残差。

阶段 A 只回答新增支路有没有因果作用，不用于宣称 DemoVLA 已经在同预算端到端训练中优于 π0.5。后者仍需共同初始化、数据、统计量、训练步数和随机种子严格匹配的对照实验。

## 2. 训练设置与产物

| 项目 | 设置 |
|---|---|
| 配置 | `demovla_libero_full_stage_a_fixed_gate` |
| 实验名 | `stage_a_fixed005_seed42_v2` |
| 数据 | 完整 LIBERO 40 tasks |
| 主干与 stats | 官方 `pi05_libero` checkpoint |
| 可训练参数 | 除 gate 外的 `demovla_*` 参数 |
| 主干 | 全部冻结 |
| 注入层 | action expert 第 4、9、14 层 |
| Gate | 三层固定 `0.05`，运行时 BF16 值 `0.050293` |
| 训练步数 | 5,000 |
| Global batch | 128 |
| Seed | 42 |

最终检查点：

```text
checkpoints/demovla_libero_full_stage_a_fixed_gate/
  stage_a_fixed005_seed42_v2/4999
```

训练于 2026-08-25 19:05（Asia/Shanghai）完成，最终 checkpoint 的异步保存和元数据写入均成功。

## 3. 最终训练诊断

| 指标 | Step 0 | Step 4999 |
|---|---:|---:|
| Flow loss | 0.004090 | 0.0031 |
| Gate | 0.050293 | 0.050293 |
| Ungated delta ratio | 0.003468 | 1.0815 |
| Gated injection ratio | 0.000174 | 0.0544 |
| Output projection norm | 1.026 | 4.9688 |
| DemoVLA grad norm | 0.000629 | 0.0018 |
| Attention entropy | 约 1.371 | 0.0014 |

训练稳定、gate 冻结正确，memory extractor、injection adapter 和 output projection 在训练结束时均保持非零梯度。实际注入比例最终约为 `5.44%`，高于预先建议的 `0.1%–1%` 诊断区间；同时 attention entropy 接近零。因此不能仅根据训练 loss 宣布机制有效，必须依赖以下干预实验。

## 4. 配对 flow-loss 因果扫描

### 4.1 干预定义

对 `1k/3k/4999` 三个检查点分别评测 1,024 个相同样本（128 batches × batch 8）。每个 batch 的四个条件严格共享 observation、action、flow time 和 Gaussian flow noise：

1. `correct`：正确样本的交互记忆，训练得到的固定 gate；
2. `off`：完全绕过交互记忆注入；
3. `shuffled`：memory 提取后做确定性的半 batch 循环移位，VLM prefix 保持不变；
4. `zero`：将提取后的 memory 置零，但保留同一网络和固定 gate。

在线 rollout 的 batch size 为 1，直接做 batch shuffle 会退化成恒等变换，所以 shuffled-memory 只在 batch 配对 flow-loss 中使用，不伪造单样本 shuffled 行为结果。

正式结果目录：

```text
outputs/demovla_stage_a/stage_a_fixed005_seed42_v2/
  causal_loss_formal_20260825/
```

### 4.2 均值结果

| Checkpoint | Correct | Off | Shuffled | Zero |
|---:|---:|---:|---:|---:|
| 1k | 0.00348016 | 0.00363693 | 0.00349763 | 0.00363584 |
| 3k | 0.00346992 | 0.00363752 | 0.00353182 | 0.00363724 |
| 4999 | **0.00344828** | 0.00363875 | 0.00355646 | 0.00363893 |

### 4.3 配对差值

差值定义为 `correct − comparator`，负值表示正确 memory 的 flow loss 更低。

| Checkpoint | Correct − Off | Correct − Shuffled | Correct − Zero |
|---:|---:|---:|---:|
| 1k | −0.00015676 | −0.00001747 | −0.00015568 |
| 3k | −0.00016760 | −0.00006190 | −0.00016732 |
| 4999 | **−0.00019047** | **−0.00010818** | **−0.00019065** |

最终检查点的配对统计：

| Contrast | 95% normal CI | Correct 更低 | Comparator 更低 | Exact sign p |
|---|---:|---:|---:|---:|
| Correct vs Off | [−0.00027197, −0.00010897] | 586 | 438 | `4.20e-6` |
| Correct vs Shuffled | [−0.00015028, −0.00006609] | 652 | 372 | `1.73e-18` |
| Correct vs Zero | [−0.00027224, −0.00010906] | 584 | 440 | `7.62e-6` |

相对均值上，最终 correct 相比 off 降低约 `5.23%`，相比 shuffled 降低约 `3.04%`。更重要的是，`Correct − Shuffled` 差距从 1k 的 `−1.75e-5` 持续扩大到 3k 的 `−6.19e-5` 和最终的 `−1.08e-4`，说明阶段 A 学到的不只是固定残差偏置，而是逐步增强的样本相关 memory 作用。

`off` 与 `zero` 的均值几乎完全一致，说明在 memory 内容被移除后，剩余 projection bias 没有形成可观测收益。

上述显著性以样本为单位计算，连续数据可能存在时间相关性，因此 p 值作为强诊断证据，而不是替代独立 rollout 的最终行为结论。

## 5. 完整 LIBERO 行为扫描

最终 `4999` checkpoint 使用相同环境初始状态和逐 replan stateless flow noise 执行以下条件：

- correct memory + fixed gate 0.05；
- injection off；
- zero memory；
- fixed gate 0.01；
- fixed gate 0.03。

### 5.1 1 trial/task pilot

| 条件 | Spatial | Object | Goal | Long | 总计 |
|---|---:|---:|---:|---:|---:|
| Correct + fixed 0.05 | 10/10 | 10/10 | 10/10 | 10/10 | **40/40** |
| Injection off | 10/10 | 10/10 | 10/10 | 10/10 | **40/40** |
| Zero memory | 10/10 | 10/10 | 10/10 | 9/10 | **39/40** |
| Fixed 0.01 | 10/10 | 10/10 | 10/10 | 10/10 | **40/40** |
| Fixed 0.03 | 10/10 | 10/10 | 10/10 | 10/10 | **40/40** |

Correct 相对 off 没有配对翻转；相对 zero 有 1 个 correct-only 成功、0 个 zero-only 成功，发生在长程任务 `put both the alphabet soup and the tomato sauce in the basket`。总体差值为 `+2.5pp`，但 McNemar `p=1`，不能视为显著行为优势。

pilot 输出目录：

```text
data/libero/videos/demovla_stage_a_causal_step4999_pilot1_seed7/
```

该 pilot 存在明显天花板效应，因此进一步执行了每个任务 10 trials、每个条件 400 episodes 的正式评测：

```text
data/libero/videos/demovla_stage_a_causal_step4999_formal10_seed7/
```

正式评测比较 `correct fixed-0.05 / off / zero / fixed-0.03`，并直接复用相同 seed、flow-noise seed 和官方 stats 下已经完成的官方 π0.5 SFT 结果 `394/400` 作为参照。

### 5.2 10 trials/task 正式结果

四个条件均完成 40 tasks × 10 trials，共 1,600 episodes：

| 条件 | Spatial | Object | Goal | Long | 总计 |
|---|---:|---:|---:|---:|---:|
| Correct + fixed 0.05 | 99/100 | 97/100 | 95/100 | 97/100 | **388/400（97.00%）** |
| Injection off | 97/100 | 97/100 | 98/100 | 94/100 | **386/400（96.50%）** |
| Zero memory | 96/100 | 98/100 | 98/100 | 91/100 | **383/400（95.75%）** |
| Correct + fixed 0.03 | 99/100 | 99/100 | 98/100 | 99/100 | **395/400（98.75%）** |
| Zero memory + fixed 0.03 | 99/100 | 98/100 | 98/100 | 96/100 | **391/400（97.75%）** |
| 官方 π0.5 SFT | 100/100 | 100/100 | 99/100 | 95/100 | **394/400（98.50%）** |

其中官方 π0.5 SFT 来自既有严格匹配评测目录：

```text
data/libero/videos/full_matched/
  official_pi05_sft_full_matched_seed42_v1_pilot10/
```

### 5.3 配对行为统计

所有比较均按相同 task、trial 和环境 seed 配对；`Reference only` 表示仅参考条件成功，`Contender only` 表示仅 DemoVLA 条件成功。

| Contender vs Reference | 成功率差 | Reference only | Contender only | Exact McNemar p |
|---|---:|---:|---:|---:|
| Correct 0.05 vs Off | +0.50pp | 8 | 10 | 0.8145 |
| Correct 0.05 vs Zero | +1.25pp | 8 | 13 | 0.3833 |
| Correct 0.03 vs Off | **+2.25pp** | 2 | 11 | **0.02246** |
| Correct 0.03 vs Zero 0.03 | +1.00pp | 2 | 6 | 0.2891 |
| Correct 0.05 vs 官方 π0.5 SFT | −1.50pp | 10 | 4 | 0.1796 |
| Correct 0.03 vs 官方 π0.5 SFT | +0.25pp | 3 | 4 | 1.0000 |

固定 0.05 在行为端相对 off 和 zero 均为正向，但没有达到统计显著；这与最终约 `5.44%` 的注入比例一致，说明该 gate 很可能偏大。将同一个最终检查点的 gate 降到 0.03 后，总成功率提高到 `98.75%`，相对 off 的配对优势达到显著水平，并与官方 π0.5 SFT 持平。

收益主要集中在 LIBERO-10 长程任务：fixed-0.03 相对 off 为 `+5pp`，相对官方 π0.5 SFT 为 `+4pp`。不过单 suite 的翻转数量仍少，对官方基线的差异不显著，因此不能把 `395/400` 解读为已经证明优于官方 π0.5。

为排除 memory 内容和注入幅度同时变化的混杂，额外补跑了严格匹配的 `correct-0.03 vs zero-0.03`：两者使用同一 checkpoint、gate、task/trial、环境 seed、flow-noise seed 和官方 stats，只在 memory 提取后将后者置零。Correct 总体高 `1.00pp`，配对翻转为 6 个 correct-only、2 个 zero-only，但 McNemar `p=0.2891`，尚未达到行为端显著。长程任务为 `+3pp`、3 比 0 个翻转，McNemar `p=0.25`。

因此，匹配干预排除了“correct-0.03 的收益完全只是 gate 幅度差异”这一明显混杂，并给出方向一致的行为信号；但 400 episodes 的翻转数仍不足以作显著的行为内容因果声明。样本相关 memory 的直接显著证据仍主要来自第 4 节的 shuffled-memory flow-loss 扫描。

## 6. 验收结论与下一步

本轮阶段 A 训练和正式评测已经完成，结论如下：

1. 冻结主干和固定 gate 成功阻断了旧训练中的 gate-collapse 绕过路径；
2. 1k、3k 和最终检查点均满足 correct flow loss 低于 off/zero；
3. correct 相对 shuffled 的优势随训练推进持续扩大，支持样本相关 memory 的因果作用；
4. 最终检查点仍保持有效梯度，支路没有发生参数意义上的死亡。
5. 行为端 fixed-0.05 未显著优于 off/zero，但 fixed-0.03 相对 off 获得 `+2.25pp`、McNemar `p=0.02246`；
6. 严格匹配的 correct-0.03 相对 zero-0.03 为 `+1.00pp`，方向为正但不显著（McNemar `p=0.2891`）；
7. fixed-0.03 达到 `395/400`，与官方 π0.5 SFT 的 `394/400` 无显著差异，证明冻结主干新增支路至少没有牺牲基线能力，但尚不能宣称优于官方基线。

阶段 A 已证明支路可训练且在 loss 端使用样本相关 memory，也确定了 `0.05` 偏强、`0.03` 更合适。建议阶段 B 将 gate 的初始化/目标有效幅度设在 `0.03` 附近，并加入 residual normalization 或注入比例约束，避免再次增长到过强区域。

在论文中可以陈述“阶段 A 建立了 loss-level 的样本相关因果作用，fixed-0.03 在严格配对 rollout 中显著优于关闭支路；相对相同 gate 的 zero-memory 获得方向一致但不显著的行为提升”。暂不应陈述“行为端已经显著证明正确 memory 内容的因果作用”“动态门控已解决”或“DemoVLA 显著优于官方 π0.5”。

## 7. 阶段 B1：readout-only 恢复实验

### 7.1 实验问题与严格控制

阶段 A 的可视化和提示词反事实实验显示：memory extractor 会随图像和任务提示发生明显变化，但 action-to-memory readout 长期呈现近乎固定的 head-to-slot 路由。因此，阶段 B1 检验以下问题：

> 在不改变 π0.5 主干、memory extractor 和 gate 的情况下，仅优化 action readout，能否恢复任务特异的 memory 因果作用？

两组都从阶段 A 最终 checkpoint `4999` 初始化，并使用相同完整 LIBERO 数据、官方 stats、seed 42、global batch 128、1,200 steps 和 fixed gate `0.03`。主干、memory extractor 和 scalar gate 全部冻结。

| 分支 | 可训练部分 | 额外改动 |
|---|---|---|
| B1-Control | 原始 action query/key/value/output readout | 无 |
| B1-Recovery | 同一 readout | Q/K RMS normalization、逐层 query conditioning、有界温度、batch-shuffled memory ranking |

最终 checkpoint：

```text
checkpoints/demovla_libero_full_b1_readout_control/
  b1_control_g003_seed42_v1/1199
checkpoints/demovla_libero_full_b1_readout_recovery/
  b1_recovery_g003_seed42_v1/1199
```

末 10 个训练日志点中，Control 和 Recovery 的 flow loss 分别为 `0.002930` 和 `0.002957`，平均注入比例均约 `0.0546`。Recovery 的 action-to-memory attention entropy 从 `0.00131` 增加到 `0.00774`，但仍处于低熵区间。因此，entropy 上升只能说明路由变平，不能直接解释为 memory 使用更有效。

### 7.2 1,024 样本 memory 干预

两组评测均显式固定 gate `0.03`，关闭训练期 ranking 辅助损失，并共享 observation、action、flow time 和 Gaussian noise。

| 分支 | Correct | Off | Shuffled | Zero |
|---|---:|---:|---:|---:|
| Control | 0.00346442 | 0.00363987 | 0.00355809 | 0.00363939 |
| Recovery | 0.00346346 | 0.00363579 | 0.00350417 | 0.00363615 |

以 `comparator − correct` 表示正确 memory 的正向优势：

| 分支 | Off penalty | Shuffled penalty | Zero penalty |
|---|---:|---:|---:|
| Control | 0.00017545 | **0.00009368** | 0.00017497 |
| Recovery | 0.00017233 | **0.00004070** | 0.00017268 |

Recovery 相比 Control 的 shuffled-memory 优势减少 `5.297e-5`，配对差中差的 95% normal CI 为 `[-8.272e-5, -2.323e-5]`，exact sign `p=2.39e-11`。这说明 Recovery 虽然仍依赖“存在某种非零 memory”，但区分正确 memory 与其他样本 memory 的能力显著减弱。

### 7.3 memory-only 任务提示反事实

该实验始终向 action VLM prefix 提供正确提示，只改变 memory extractor 看到的提示；因此反事实影响只能通过 DemoVLA memory 路径传到 action prediction。

| 分支 | 条件 | Flow loss | Correct − counterfactual | 95% CI | Sign p |
|---|---|---:|---:|---:|---:|
| Control | suite-swap | 0.00347818 | **−0.00001479** | [−0.00002754, −0.00000204] | **0.002972** |
| Control | blank | 0.00348346 | −0.00002007 | [−0.00003604, −0.00000411] | `1.36e-5` |
| Recovery | suite-swap | 0.00346919 | −0.00000443 | [−0.00001069, +0.00000184] | 0.179 |
| Recovery | blank | 0.00347688 | −0.00001211 | [−0.00002289, −0.00000134] | `9.20e-5` |

Control 的 wrong-task prompt penalty 相比阶段 A 最终 checkpoint 的约 `4.40e-6` 增大到 `14.79e-6`，并从不显著变为显著；Recovery 则仍为 `4.43e-6`、`p=0.179`。这构成如下机制证据：

1. memory extractor 已包含可供动作端利用的任务相关信息；
2. 在冻结主干、extractor 和 gate 后，仅优化原始 readout 即可使动作 flow loss 对 memory 分支中的任务提示产生显著响应；
3. 阶段 A 的主要可恢复瓶颈位于 memory-to-action readout，而不是必须先重做 extractor；
4. 更高的 attention entropy 不等价于更强的任务语义因果作用。

上述结果支持“readout-only adaptation 建立了 loss-level 的任务特异 memory 依赖”，但尚不能证明 DemoVLA 提升闭环成功率、优于官方 π0.5，或能跨 seed/未见任务稳定泛化。总体 wrong-task 效应约占 correct flow loss 的 `0.43%`，且四个 suite 单独统计时置信区间仍跨零，论文中必须保留这一效应量和统计边界。

### 7.4 当前决策

B1-Recovery 作为组合改动失败：它增加了 entropy，却同时降低 shuffled-memory 区分能力和任务提示敏感性。由于 Q/K normalization、layer conditioning、temperature 和 ranking 同时改变，本轮不能把失败归因到单一组件。

下一步优先对 `B1-Control/1199` 执行 fixed-0.03 的 correct-memory 与 zero-memory 严格配对闭环评测。只有当相同 task/trial、环境 seed 和 flow-noise seed 下出现稳定的成功率翻转，才能把当前 loss-level 因果证据推进为 behavior-level memory 内容因果证据。Recovery 暂不投入完整 rollout；后续应拆成单因素消融，并把跨样本 shuffled ranking 改为同图像、错误任务提示生成的 memory ranking。

正式离线输出：

```text
outputs/demovla_b1_eval/control_memory_1024/
outputs/demovla_b1_eval/recovery_memory_1024/
outputs/demovla_b1_eval/control_prompt_1024/
outputs/demovla_b1_eval/recovery_prompt_1024/
```

### 7.5 B1-Control 闭环行为结果

基于第 7.3 节的 loss-level 正向结果，对 `B1-Control/1199` 执行 fixed-0.03 闭环 rollout，并与 zero-memory fixed-0.03 做严格配对。两者共享 task/trial、环境 seed、flow-noise seed、官方 stats 和同一 checkpoint，只在 memory 提取后是否置零上不同。

| 条件 | Spatial | Object | Goal | Long | 总计 |
|---|---:|---:|---:|---:|---:|
| B1-Control correct memory + fixed 0.03 | 99/100 | 99/100 | 98/100 | 95/100 | **391/400（97.75%）** |
| B1-Control zero memory + fixed 0.03 | 99/100 | 96/100 | 99/100 | 93/100 | **387/400（96.75%）** |

配对统计：

| Contrast | 成功率差 | Reference only | Contender only | Exact McNemar p |
|---|---:|---:|---:|---:|
| B1-Control correct 0.03 vs zero 0.03 | +1.00pp | 6 | 10 | 0.4545 |
| B1-Control correct 0.03 vs Stage A correct 0.03 | −1.00pp | 8 | 4 | 0.2891 |
| B1-Control correct 0.03 vs 官方 π0.5 SFT | −0.75pp | 6 | 3 | 0.5078 |

该结果说明：B1-Control 的 memory-only wrong-prompt loss 敏感性没有转化为稳定的闭环成功率提升。它相对 zero-memory 方向为正，但翻转数不足、统计不显著；相对阶段 A 最优 fixed-0.03 和官方 π0.5 SFT 也没有优势。因此，当前最强行为 checkpoint 仍是阶段 A `4999` 在推理时使用 fixed gate `0.03`，而不是 B1-Control 或 B1-Recovery。

## 8. 当前已有结论与 S1 设计依据

截至 2026-08-26，DemoVLA 的证据边界可以归纳为四层。

第一，结构支路确实可训练。阶段 A 在冻结官方 π0.5-LIBERO 主干后，仅训练 DemoVLA 交互记忆支路，得到稳定非零梯度、非零 residual 和显著的配对 flow-loss 改善。`correct` 显著优于 `off/zero/shuffled`，其中 `correct − shuffled` 从 1k 到 4999 持续扩大。这可以支持“样本相关 memory 支路在 loss 层面具有因果作用”。

第二，行为端已有正向但仍不充分。阶段 A `4999` 使用 fixed gate `0.03` 达到 `395/400`，相对 injection off 为 `+2.25pp` 且 McNemar `p=0.02246`；但相对同 gate 的 zero-memory 只有 `+1.00pp`、`p=0.2891`，相对官方 π0.5 SFT `394/400` 也无显著差异。因此目前不能宣称 DemoVLA 显著优于官方 π0.5，也不能宣称闭环行为已经显著证明“正确 memory 内容”本身的必要性。

第三，瓶颈主要在 memory-to-action readout，而不是 extractor 完全无效。Overlay 显示 memory slot 具有非冗余的视觉关注，并随 replan 和任务提示变化；memory-only prompt 反事实中，blank prompt 会显著恶化 flow loss。但阶段 A 对 wrong-task memory prompt 不显著，说明 action 端尚未稳定利用目标特异语义。B1-Control 通过 readout-only 训练把 wrong-task prompt penalty 提升到显著水平，进一步支持“extractor 中有可用语义，readout 是可恢复瓶颈”。

第四，不能把 attention entropy 当成有效性的替代指标。B1-Recovery 引入 Q/K normalization、逐层 query conditioning、有界 temperature 和 batch-shuffled ranking 后，attention entropy 上升，但 shuffled-memory 区分能力和 wrong-prompt 敏感性反而下降。这个组合改动不能继续作为主线，应拆成单因素实验。

由此，下一步不应直接重写 memory extractor、增加 query 数量，或恢复动态 gate。更干净的路线是回到阶段 A `4999`，围绕“如何让正确 memory 改变关键动作决策”做单因素实验：

| 实验 | 固定项 | 唯一变化 | 验收标准 |
|---|---|---|---|
| S1-Control | Stage A `4999`、fixed gate `0.03`、冻结主干/extractor/gate、只训练原始 readout | 标准 readout-only continuation | 不显著损害 correct flow 和闭环成功率 |
| S1-Semantic | 与 S1-Control 完全相同 | 加入同图像 wrong-prompt memory ranking | wrong-prompt memory 显著劣于 correct，且 correct-vs-zero/shuffled 的优势不下降 |

S1 的关键不是追求更低训练 loss，而是建立更强的内容因果链：

```text
同一图像 + 同一 action prefix
只替换 memory extractor 看到的任务提示
如果 wrong-prompt memory 使 flow loss 或关键动作决策变差，
则差异只能经由 DemoVLA memory 支路产生。
```

若 S1-Semantic 通过离线验收，再投入闭环行为评测；若仍只在 loss 端有效而行为端无翻转，则下一步应转向“关键 replan/失败状态”的 memory 移植与 knockout，而不是继续扩大整体训练规模。

## 9. S1 同图像错误提示排序实验结果

### 9.1 严格匹配设置

S1-Control 与 S1-Semantic 均从阶段 A `4999` 初始化，使用相同完整 LIBERO 数据、官方 stats、seed 42、global batch 128、1,200 steps 和 fixed gate `0.03`。两组都冻结主干、memory extractor 和 gate，只训练原始 memory-to-action readout。唯一变量是 S1-Semantic 增加同图像 wrong-prompt memory ranking；action prefix 始终使用正确任务提示。

最终 checkpoint：

```text
checkpoints/demovla_libero_full_s1_semantic_control/
  s1_control_g003_seed42_v1/1199
checkpoints/demovla_libero_full_s1_semantic_ranking/
  s1_semantic_g003_seed42_v1/1199
```

两组均正常完成，最终 gate 为 BF16 表示下的 `0.0302734`，未出现 NaN、OOM 或 checkpoint 保存错误。

### 9.2 1,024 样本 memory 干预

评测固定 gate `0.03`、关闭训练期 ranking loss，并共享数据顺序、observation、action、flow time、Gaussian noise 和 seed。

| 分支 | Correct | Off | Shuffled | Zero |
|---|---:|---:|---:|---:|
| S1-Control | 0.00346110 | 0.00363875 | 0.00355796 | 0.00363896 |
| S1-Semantic | 0.00346270 | 0.00363663 | 0.00354774 | 0.00363781 |

以 `comparator − correct` 表示正确 memory 的正向优势：

| 分支 | Off penalty | Shuffled penalty | Zero penalty |
|---|---:|---:|---:|
| S1-Control | 0.00017765 | **0.00009685** | 0.00017786 |
| S1-Semantic | 0.00017393 | **0.00008503** | 0.00017511 |

两组 correct flow 的差异小于 `0.05%`，因此 S1-Semantic 没有破坏基础拟合能力；但其 shuffled-memory 优势相对 Control 减少 `1.182e-5`，逐样本差中差的 95% normal CI 为 `[-1.850e-5, -5.135e-6]`，exact sign `p=0.0362`。下降幅度约为 Control shuffled advantage 的 `12.2%`，虽未超过预设的 `20%` 容忍上限，但方向显著为负。

### 9.3 1,024 样本 memory-only 提示反事实

| 分支 | Correct | Suite-swap | Suite-swap penalty | 95% CI | Sign p |
|---|---:|---:|---:|---:|---:|
| S1-Control | 0.00346328 | 0.00347593 | 0.00001266 | [−2.575e-5, +0.044e-5] | 0.02249 |
| S1-Semantic | 0.00346222 | 0.00347491 | 0.00001269 | [−2.323e-5, −0.214e-5] | 0.000818 |

表中的置信区间对应 `correct − suite-swap`。S1-Semantic 的 suite-swap 效应在本轮样本上显著，但效应量没有相对 Control 增强：两组 penalty 的逐样本差中差仅为 `+2.93e-8`，95% CI 为 `[-5.093e-6, +5.151e-6]`，exact sign `p=0.512`。它也明显低于预设的目标效应量 `2.0e-5`。

Blank-prompt penalty 从 Control 的 `1.784e-5` 下降到 Semantic 的 `1.608e-5`；差中差为 `−1.755e-6`，95% CI 跨零。视觉 memory 本身的任务提示敏感性保持不变：suite-swap 的 memory cosine 为 `0.8718`、top-patch changed rate 为 `60.06%`，说明唯一训练变量作用在 readout，而不是冻结的 extractor。

### 9.4 验收结论与后续方向

S1-Semantic **未通过核心验收标准**。它满足“correct flow 不恶化超过 1%”，也没有让 shuffled advantage 下降超过 20%；但它没有提高 wrong-prompt penalty，反而显著削弱了部分 shuffled-memory 区分能力。因此本轮不启动 400-episode 闭环 rollout，避免把计算投入到一个没有显示机制增益的 checkpoint。

该阴性结果收窄了问题范围：对全时段、全样本平均施加 wrong-prompt ranking，主要让 readout满足局部 margin，并没有使正确 memory 更强地控制动作。下一轮不应仅增大 ranking weight 或训练步数；优先方向应是：

1. 在阶段 A `4999` 上先定位 success/failure 翻转附近的关键 replan 和动作维度；
2. 对这些状态执行同图像 correct/wrong/zero memory 的动作轨迹差分、memory transplant 与 layer knockout；
3. 只在已确认 memory 能影响关键动作的时刻做 key-state weighting，保持其余架构和训练配置不变；
4. 以“关键动作差分增大且 correct flow 不退化”为离线门槛，再决定是否进行闭环评测。

正式输出：

```text
outputs/demovla_s1_eval/control_memory_1024/
outputs/demovla_s1_eval/semantic_memory_1024/
outputs/demovla_s1_eval/control_prompt_1024/
outputs/demovla_s1_eval/semantic_prompt_1024/
```
