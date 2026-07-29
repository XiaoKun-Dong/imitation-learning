# DemoVLA Sparse-Deep Diversity 10k 训练与评估报告

## 1. 实验状态

本报告记录 DemoVLA sparse-deep diversity 版本截至 10k steps 的训练结果、
诊断 pilot，以及 `libero_object` 上 50 episodes 的正式 rollout 评估。

- 训练配置：`demovla_libero_sparse_deep_diverse`
- Checkpoint：`/home/dongxiaokun/checkpoints/deep/10000`
- 基础模型：冻结的 `pi0.5`
- 训练方式：adapter-only
- Global batch size：128
- Interaction queries：4
- 注入层：`(4, 9, 14)`
- Gate 初始化：`sigmoid(-4) ≈ 0.018`
- Attention diversity：weight `1e-3`，margin `0.5`
- Memory diversity：weight `1e-4`，margin `0.5`
- 优化器：AdamW，gradient clipping norm `1.0`
- 学习率：500 steps warmup，峰值 `1e-4`，10k steps 衰减至 `1e-5`
- EMA：关闭

当前结论分为两级：

1. **已经确认**：训练数值稳定；平均意义上的 query collapse 已解除；10k
   checkpoint 在 50 episodes 中成功 36 次，成功率为 72.0%。
2. **已经确认**：与无 diversity 10k 的相同 50 个 initial-state slots 配对时，
   结果为 36/50 对 35/50，当前没有显著成功率提升证据。
3. **尚未确认**：扩大到 100 episodes 后的稳定结果、相对 single-shot/vanilla
   的收益，以及 query 分化是否稳定对应目标物体、目标容器和接触区域。

旧版无 diversity、训练到 30k 的结果保留在
[demovla_sparse_deep_training_report.md](demovla_sparse_deep_training_report.md)，
用于后续对照。

## 2. 模型与训练目标

每次 policy replan 中，Interaction Token Extractor 从当前观测的语言和视觉
patch 中提取 4 个 interaction tokens：

```text
language-conditioned learnable queries
        |
        v
cross-attend language tokens
        |
        v
cross-attend spatial + camera-aware visual patches
        |
        v
interaction memory [B, 4, 1024]
```

同一份 interaction memory 在完整 flow denoising loop 中缓存，并由 Action
Expert 的第 4、9、14 层通过共享 cross-attention adapter 读取。三个注入位置
共享 Q/K/V/output projection，但各自具有独立 gate。

总目标为：

\[
L_{\mathrm{total}} =
L_{\mathrm{flow}} +
10^{-3} L_{\mathrm{attention\ diversity}} +
10^{-4} L_{\mathrm{memory\ diversity}}
\]

两个 diversity loss 都只惩罚 query pair cosine similarity 超过 `0.5` 的部分：

\[
L_{\mathrm{pair}} =
\operatorname{mean}_{i \ne j}
\left[
\operatorname{ReLU}
\left(
\cos(x_i,x_j)-0.5
\right)
\right]^2
\]

因此正则用于解除高度相似的 query collapse，而不是强制所有 query 完全正交。

## 3. 训练结果

### 3.1 关键训练点

下表直接整理训练日志中已经保存的 0、1k、2k、3k 和 10k 指标。日志只打印
四位小数，因此非常小的正则项和学习率可能显示为 `0.0000`。

| Step | Total loss | Flow loss | Action first | Action middle | Action last |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.0618 | 0.0615 | 0.0687 | 0.0542 | 0.0736 |
| 1,000 | 0.0331 | 0.0330 | 0.0395 | 0.0299 | 0.0378 |
| 2,000 | 0.0257 | 0.0257 | 0.0319 | 0.0230 | 0.0290 |
| 3,000 | 0.0240 | 0.0240 | 0.0294 | 0.0216 | 0.0273 |
| 10,000 | **0.0199** | **0.0199** | **0.0247** | **0.0179** | **0.0227** |

从 Step 0 到 Step 10k：

- flow loss 从 `0.0615` 降到 `0.0199`，下降约 **67.6%**；
- first/middle/last 三段 action loss 均持续下降；
- 10k 相比 3k 的 flow loss 继续下降约 **17.1%**；
- 没有观察到 loss 发散、NaN 或梯度爆炸。

Action chunk 的首步和末步仍比中间步困难。10k 时相对于 middle：

- first 高约 38.0%；
- last 高约 26.8%。

首步误差仍应作为后续闭环稳定性分析的重点。

### 3.2 Attention diversity

| Step | Diversity loss | Pair cosine mean | Pair cosine max |
|---:|---:|---:|---:|
| 0 | 0.2456 | 0.9956 | 0.9995 |
| 1,000 | 0.0098 | 0.2466 | 0.7587 |
| 2,000 | 0.0010 | 0.1641 | 0.8030 |
| 3,000 | 0.0019 | 0.1805 | 0.8761 |
| 10,000 | **0.0009** | **0.2007** | **0.7982** |

Step 0 时四个 query 的 attention 几乎完全重合，确认初始版本存在明显 query
collapse。到 1k 时平均 cosine 已快速下降到 `0.2466`，10k 保持在 `0.2007`。

结论：

- 平均 attention collapse 已解除；
- attention mean 在 2k 以后维持在约 `0.16～0.20`，没有重新整体塌缩；
- max 仍可达到约 `0.80`，说明个别 batch 样本中的某一对 query 仍可能关注
  相似区域。

当前 max 是整个 batch 中的极端值。Global batch size 为 128、query 数量为 4
时，每步共有约 `128 × 6 = 768` 个非重复 query pair；因此应增加 p95 和固定
query-pair 矩阵，避免只根据单个最大值判断。

### 3.3 Memory diversity

| Step | Diversity loss | Pair cosine mean | Pair cosine max |
|---:|---:|---:|---:|
| 0 | 0.2225 | 0.9715 | 0.9911 |
| 1,000 | 0.1212 | 0.3280 | 0.9839 |
| 2,000 | 0.0226 | -0.0189 | 0.9505 |
| 3,000 | 0.0141 | -0.1277 | 0.9678 |
| 10,000 | **0.0089** | **0.0010** | **0.9658** |

10k 时 interaction memory 的平均 pair cosine 为 `0.0010`，整体接近正交。
这说明四个 memory token 已经形成显著不同的表示。

`memory_pair_cosine_max=0.9658` 表明仍有少量高度相似 pair，但同时：

- mean 接近 0；
- diversity loss 已从 `0.2225` 降到 `0.0089`；
- attention 和 memory 的平均相似度都没有反弹。

因此当前证据更符合“少量困难观测中的局部重合”，而不是整体 query
collapse。是否存在固定的两个 query 局部塌缩，需要用 per-pair 指标和 rollout
可视化继续确认。

### 3.4 Diversity 正则强度

各关键点的实际正则贡献约为：

| Step | Weighted attention | Weighted memory | Total regularization |
|---:|---:|---:|---:|
| 0 | 0.0002456 | 0.0000223 | 0.0002679 |
| 1,000 | 0.0000098 | 0.0000121 | 0.0000219 |
| 2,000 | 0.0000010 | 0.0000023 | 0.0000033 |
| 3,000 | 0.0000019 | 0.0000014 | 0.0000033 |
| 10,000 | 0.0000009 | 0.0000009 | 0.0000018 |

10k 时正则仅占 flow loss 约 **0.009%**。它已经从训练初期的“解除 collapse”
转变为防止 query 重新高度重合的轻量约束，没有主导 action optimization。

### 3.5 梯度、投影与 Gate

| Step | Adapter grad | Extractor grad | Output proj grad | Output proj norm | Update norm |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.0021 | 0.0002 | 0.0020 | 0.0002 | 0.0004 |
| 1,000 | 0.0510 | 0.0040 | 0.0379 | 2.8633 | 0.0799 |
| 2,000 | 0.0741 | 0.0119 | 0.0373 | 4.2111 | 0.0972 |
| 3,000 | 0.0806 | 0.0214 | 0.0313 | 5.1873 | 0.0880 |
| 10,000 | 0.0603 | 0.0255 | 0.0207 | 6.9504 | 0.0110 |

重要现象：

- Step 0 的 extractor gradient 已非零，说明 diversity loss 可以绕过零初始化
  output projection，直接训练 Interaction Token Extractor。
- output projection norm 从 `0.0002` 增长到 `6.9504`，注入通路已经建立。
- 10k 时 update norm 降至 `0.0110`，符合学习率衰减后的稳定收敛状态。
- 日志中的 `learning_rate=0.0000` 是四位小数显示造成的；训练配置的末端学习率
  为 `1e-5`，非严格的零。

10k 三层 gate：

| 注入层 | Gate | Raw gate |
|---:|---:|---:|
| Layer 4 | 0.0179 | -4.0035 |
| Layer 9 | **0.0190** | **-3.9459** |
| Layer 14 | 0.0177 | -4.0146 |

Layer 9 的 gate 最大，Layer 14 最小，延续了无 diversity sparse-deep 实验中
“中层更偏好 interaction injection”的趋势。不过 gate 差异仍然较小，在记录
layer-wise injection ratio 之前，不能将 gate 直接等价为实际层贡献。

## 4. 10k Rollout 评估

### 4.1 评估协议

- Task suite：`libero_object`
- 任务数：10
- 正式评估：每任务 5 trials，共 50 episodes
- Seed：7
- Object condition：`none`
- Replan interval：5 environment steps
- Policy GPU：0
- Checkpoint：`/home/dongxiaokun/checkpoints/deep/10000`
- 正式评估关闭 Interaction diagnostics，避免诊断开销影响评估吞吐

正式评估输出目录：

```text
data/libero/videos/demovla_sparse_deep_diverse_10k_eval5/
```

目录中包含 50 个 rollout 视频和 50 条 `metrics.jsonl` episode 记录。

正式评估之前还运行了每任务 1 trial 的诊断 pilot：

```text
data/libero/videos/demovla_sparse_deep_diverse_10k_pilot/
```

pilot 启用了 Interaction diagnostics，生成 10 个视频和 364 张以 replan 为
单位的 attention 图片。正式成功率只使用 `eval5` 的 50 episodes，不把 pilot
重复计入。

### 4.2 整体结果

| 指标 | 结果 |
|---|---:|
| Episodes | 50 |
| Successes | 36 |
| Success rate | **72.0%** |
| Wilson 95% CI | **58.3%～82.5%** |
| 未抓到正确目标 | 6 |
| 其中抓错物体 | 2 |
| 正确抓取后的失败 | 8 |
| Mean steps | 204.0 |
| Mean successful-episode steps | 170.5 |

14 次失败可以分成两个互斥阶段：

```text
正确目标抓取前失败：6 / 50 = 12%
正确目标抓取后失败：8 / 50 = 16%
```

`wrong_object_grasped=2` 是前一类的子集，不能与 6 和 8 再次相加。结果说明
当前失败同时来自目标获取和 post-grasp transport/place，而不是单一阶段。

### 4.3 逐任务结果

| Task | Success | Rate | 未抓到目标 | 抓错物体 | Post-grasp fail | Mean steps |
|---|---:|---:|---:|---:|---:|---:|
| Alphabet soup | 4/5 | 80% | 0 | 0 | 1 | 238.2 |
| Cream cheese | 5/5 | 100% | 0 | 0 | 0 | 175.0 |
| Salad dressing | 4/5 | 80% | 0 | 0 | 1 | 149.4 |
| BBQ sauce | 2/5 | 40% | 2 | 1 | 1 | 246.2 |
| Ketchup | 2/5 | 40% | 1 | 0 | 2 | 245.6 |
| Tomato sauce | 4/5 | 80% | 1 | 0 | 0 | 158.2 |
| Butter | 4/5 | 80% | 0 | 0 | 1 | 212.8 |
| Milk | 5/5 | 100% | 0 | 0 | 0 | 151.6 |
| Chocolate pudding | 3/5 | 60% | 0 | 1 | 2 | 228.0 |
| Orange juice | 3/5 | 60% | 2 | 0 | 0 | 234.6 |

最稳定的是 cream cheese 和 milk，均为 5/5。BBQ sauce 与 ketchup 只有 2/5，
是当前优先分析的两个任务；两者分别包含目标获取失败和 post-grasp failure，
需要逐视频区分。

### 4.4 与无 Diversity 10k 的对齐比较

已有无 diversity sparse-deep 10k 评估包含相同 10 个任务、seed 7、
`object_condition=none`，每任务 10 trials。取其中 episode 0～4，可以与当前
diverse eval5 对齐相同 benchmark initial states：

| 模型 | Episodes | Success | 未抓到目标 | 抓错物体 | Post-grasp fail |
|---|---:|---:|---:|---:|---:|
| Sparse-deep 10k，无 diversity | 50 | 35/50（70%） | 7 | 4 | 8 |
| Sparse-deep diversity 10k | 50 | **36/50（72%）** | **6** | **2** | 8 |

逐任务成功次数：

| Task | 无 diversity | Diversity | Difference |
|---|---:|---:|---:|
| Alphabet soup | 4/5 | 4/5 | 0 |
| Cream cheese | 4/5 | 5/5 | +1 |
| Salad dressing | 3/5 | 4/5 | +1 |
| BBQ sauce | 3/5 | 2/5 | -1 |
| Ketchup | 1/5 | 2/5 | +1 |
| Tomato sauce | 5/5 | 4/5 | -1 |
| Butter | 3/5 | 4/5 | +1 |
| Milk | 3/5 | 5/5 | +2 |
| Chocolate pudding | 5/5 | 3/5 | -2 |
| Orange juice | 4/5 | 3/5 | -1 |

50 个对齐 episode 的配对结果：

```text
两个模型都成功：27
仅 diversity 成功：9
仅无 diversity 成功：8
两个模型都失败：6
```

McNemar 双侧精确检验为 `p=1.0`。成功率只提高 2 个百分点，不能拒绝“两者
闭环成功率相同”的假设。抓错物体从 4 次降至 2 次是值得继续观察的方向，但
样本量不足以证明它是稳定收益。

无 diversity 10k 的完整 100-episode 结果为 70/100，与当前 72% 同样接近，
但两者 trial 数不同，因此主要结论仍以前 5 个 initial states 的配对比较为准。

需要注意：上述两轮历史评估只对齐了 initial states，没有将 flow noise 固定到
`(task, episode, replan)`。旧 policy server 使用跨请求累积的 RNG，episode
长度不同会让后续 noise 序列错位。因此该 McNemar 结果是历史 rollout 的描述性
比较，不能作为严格的固定-noise因果消融。

当前代码已经加入：

```text
(policy_noise_seed, benchmark_task_id, episode_idx, replan_idx)
```

驱动的 stateless flow noise。后续 checkpoint、gate 和 injection-layer 对比
必须使用同一个 `policy_noise_seed` 重跑。

### 4.5 Pilot 与正式评估的差异

诊断 pilot 为 9/10，而正式评估为 36/50。pilot 中唯一失败的 cream cheese
在正式评估中反而达到 5/5；BBQ sauce、ketchup、chocolate pudding 和 orange
juice 的多 initial-state 结果更弱。

这证明每任务单 trial 会明显高估或低估任务稳定性。后续不能再使用 pilot
成功率选择 checkpoint。

### 4.6 当前结果边界

正式评估已经覆盖 50 episodes，但仍有以下限制：

- 每个任务只有 5 trials，逐任务成功率仍以 20 个百分点为最小变化单位；
- 72% 的 Wilson 95% 区间仍较宽，为 58.3%～82.5%；
- 尚未与 single-shot、vanilla 和 parameter-matched baseline 进行相同
  initial states 的充分对比；
- 尚未对 364 张 attention 图进行定量 grounding 审核。

因此当前可以确认 diversity 解除了表示塌缩，但不能宣称它显著提高了闭环
成功率。

## 5. 当前已确认结论

1. Diversity loss 在 Step 0 即获得有效梯度，并快速解除初始 attention 和
   memory collapse。
2. 10k 时 attention pair mean 为 `0.2007`，memory pair mean 为 `0.0010`，
   平均 query 分化保持稳定。
3. 正则项在 10k 只占 flow loss 约 `0.009%`，没有压制主任务学习。
4. Flow loss 从 `0.0615` 降至 `0.0199`，训练数值稳定。
5. Sparse-deep output projection 和三层 gate 均获得有效更新，Layer 9 gate
   相对最大。
6. 10k checkpoint、norm stats、policy server、LIBERO client 和可视化链路均已
   端到端验证。
7. Seed 7 的正式评估完成 50 episodes，成功 36 次，成功率为 72.0%，Wilson
   95% CI 为 58.3%～82.5%。
8. 历史评估与无 diversity 10k 对齐前 5 个 initial states 后，结果为 36/50
   对 35/50，McNemar `p=1.0`；但旧协议没有固定逐 replan flow noise，需要
   按新协议重跑才能得到严格因果结论。
9. Diversity 版本的错物抓取为 2 次，无 diversity 对齐子集为 4 次；这是待扩大
   样本验证的趋势，不是已经确认的提升。

## 6. 后续计划

### P0：扩大评估并补齐关键基线

每任务 5 次的正式评估已经完成。若要缩小置信区间并与已有无 diversity
100-episode 结果完全对齐，下一步扩展到每任务 10 次：

```text
已完成：10 tasks × 5 trials  = 50 episodes
下一步：10 tasks × 10 trials = 100 episodes
```

所有 checkpoint 必须共享：

- 完全相同的 task IDs；
- 完全相同的 initial-state/seed 集合；
- 完全相同的 `policy_noise_seed`，使每个 replan 的 flow noise 对齐；
- `object_condition=none`；
- 相同 replan interval；
- 相同环境最大步数和成功判定。

下一轮优先对比：

1. diverse sparse-deep 10k 的 episode 5～9；
2. single-shot 10k；
3. vanilla `pi0.5` 或 parameter-matched adapter；
4. diverse sparse-deep 3k（如果 checkpoint 仍保留）。

报告整体成功率、逐任务成功率、置信区间、目标抓取失败率、错物抓取率、
post-grasp failure rate 和成功 episode 步数。

### P1：完成 grounding 与 query 分工分析

使用已经生成的 364 张 replan 图检查：

- 每个 query 是否随观测和机器人运动改变关注区域；
- 是否有 query 稳定关注目标物体、篮子、机械臂接触区域或上下文；
- 是否存在固定 query 长期只关注背景；
- 高 memory cosine max 是否来自固定 query pair。

正式 eval5 关闭了 diagnostics，因此正式失败 episode 没有对应 attention 图。
需要优先对 BBQ sauce、ketchup、chocolate pudding 和 orange juice 的失败
initial states 重新运行诊断 rollout，避免用 pilot 的不同轨迹解释正式失败。

新增定量诊断：

```text
attention pair cosine p50 / p90 / p95 / max
memory pair cosine p50 / p90 / p95 / max
4 × 4 query-pair cosine matrix
per-query attention entropy
top-k patch temporal displacement
top-k overlap across adjacent replans
```

如果使用 LIBERO segmentation 计算 target/basket attention coverage，只能作为
评估标注，不能输入模型。

### P2：定位目标获取与 Post-grasp Failure

14 个失败 episode 均运行到 290 steps。优先检查：

```text
BBQ sauce：episode 1 post-grasp，episode 2 pre-target，episode 4 wrong-object
Ketchup：episode 0/4 post-grasp，episode 1 pre-target
Chocolate pudding：episode 1 wrong-object，episode 4 post-grasp
Orange juice：episode 0/4 pre-target
```

目标获取失败重点检查：

- attention 是否被同类或高对比度 distractor 吸引；
- approach trajectory 是否到达目标但没有形成有效夹持；
- top-k patch 是否持续停留在背景或错误物体。

Post-grasp failure 重点检查：

- 是否抓取不牢或夹爪提前松开；
- transport trajectory 是否碰撞；
- basket attention 是否在抓取后消失；
- action chunk 首步是否发生明显跳变；
- 是否到达篮子但 release timing 错误。

后续训练日志拆分：

```text
translation / rotation / gripper loss
first-step translation / rotation / gripper loss
```

### P3：完成结构消融

按优先级进行：

1. 三层 `(4, 9, 14)` 对比单层 `(9,)`；
2. diversity on/off；
3. interaction token 数量 `1 / 4 / 8`；
4. 有/无二维位置与 camera embedding；
5. 共享 adapter 对比 layer-specific adapter。

同时增加：

```text
layer-wise injection delta norm
layer-wise action hidden norm
layer-wise injection ratio
action-to-memory attention received by each query
```

### P4：鲁棒性与阶段条件

完成标准成功率对比后，再评估：

```text
目标位置变化
同类 distractor
背景变化
相机偏移
遮挡
光照和纹理变化
```

当前不加入 flow-timestep conditioning，也不立即加入 episode progress
embedding。只有当可视化证明 query 对不同任务阶段缺少响应时，才考虑从环境或
轨迹数据提供可靠的 episode progress 信号，并单独做消融。

## 7. Checkpoint 选择原则

现阶段保留 10k checkpoint，但不根据最低训练 loss 直接认定它是最终模型。
最终选择应按以下顺序：

1. 闭环成功率和置信区间；
2. 目标抓取、错物抓取和 post-grasp failure 分解；
3. 不同 seeds 和扰动下的稳定性；
4. attention grounding 与 query 分工；
5. 推理延迟、显存和参数增量；
6. 最后才参考离线 flow loss。

当前阶段的准确表述是：

> DemoVLA sparse-deep diversity 10k 已经完成稳定训练和 50-episode 正式评估；
> query 的平均塌缩已解除，但与无 diversity 10k 的成功率差异不显著。下一阶段
> 需要扩大样本、补齐基线、分析失败轨迹和 grounding，确认它是否带来目标选择
> 或鲁棒性收益。
