# DemoVLA Sparse-Deep 训练报告

> 本文记录无 diversity 的 30k sparse-deep 基线。带 attention/memory
> diversity 的 10k 结果与 pilot rollout 见
> [DemoVLA Sparse-Deep Diversity 10k 训练与评估报告](demovla_sparse_deep_diversity_10k.md)。

## 1. 实验概况

本报告分析 DemoVLA sparse-deep adapter-only 实验的训练结果。

- 训练配置：`demovla_libero_sparse_deep`
- 基础模型：冻结的 `pi0.5`
- 可训练参数：名称匹配 `demovla_.*` 的 interaction-memory 模块
- Interaction token 数量：4
- 注入方式：sparse-deep
- 注入层索引：`(4, 9, 14)`，即第 5、10、15 个 Action Expert block 后
- 三个注入位置共享 Q/K/V/output projection，使用独立 gate
- Global batch size：128
- 优化器：AdamW，gradient clipping norm 为 1.0
- 学习率：500 steps warmup，峰值 `1e-4`，10k steps 衰减至 `1e-5`
- 训练步数：30,000
- EMA：关闭
- 日志间隔：1,000 steps
- 训练目标：原始 flow-matching action loss
- 外部目标监督：无 mask、bbox、crop、point 或 affordance loss

数据来源：

- `checkpoints/demovla_sparse_deep/metrics.jsonl`
- `checkpoints/demovla_libero_single_shot/demovla_single_shot/metrics.jsonl`

日志统计需要注意：

- Step 0 基本只反映第一个 batch。
- 后续日志中的 loss、gradient norm 和 update norm 是对应日志区间的均值。
- learning rate、gate 和 parameter norm 使用日志边界处的最新值。
- 因此 Step 0 不适合与后续 1,000-step 区间均值做严格等价比较。

## 2. 核心结论

Sparse-deep 实验在优化层面训练成功：

1. 总 loss 从 `0.06158` 下降到 `0.01675`，降幅约 72.8%。
2. 整个训练过程中没有 loss 发散、梯度爆炸、NaN 或参数范数异常。
3. 零初始化 output projection 按预期工作：早期先建立输出通路，随后 interaction extractor 开始获得梯度。
4. 在可比的约 10k steps 位置，sparse-deep 的离线 action loss 比 single-shot 低约 29.2%。
5. 三层 gate 出现分化：Layer 9 增强最多，Layer 4 基本不变，Layer 14 被轻微抑制。
6. Action chunk 的首步和末步仍明显难于中间步，尤其是首步。
7. 训练日志只能证明离线动作拟合改善，不能单独证明 interaction token 获得了正确 grounding，也不能证明闭环执行成功率提高。

## 3. Loss 收敛

### 3.1 关键训练点

| Step | Loss | Action first | Action middle | Action last | Output projection norm |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.061584 | 0.068689 | 0.054255 | 0.073711 | 0.000196 |
| 1,000 | 0.033082 | 0.039631 | 0.029949 | 0.037736 | 2.842715 |
| 2,000 | 0.024401 | 0.030472 | 0.021843 | 0.027497 | 4.295527 |
| 3,000 | 0.022420 | 0.027435 | 0.020169 | 0.025440 | 5.304646 |
| 5,000 | 0.020439 | 0.025164 | 0.018326 | 0.023182 | 6.479970 |
| 10,000 | 0.018566 | 0.022942 | 0.016704 | 0.021183 | 7.126440 |
| 15,000 | 0.018109 | 0.022437 | 0.016206 | 0.020785 | 7.267690 |
| 20,000 | 0.017714 | 0.022123 | 0.015866 | 0.020172 | 7.407580 |
| 25,000 | 0.017242 | 0.021589 | 0.015313 | 0.019805 | 7.538500 |
| 29,999 | **0.016753** | **0.020990** | **0.014932** | **0.019357** | **7.666603** |

最终 loss 是日志中的最低值，没有出现明显的训练 loss 反弹。

### 3.2 分阶段分析

#### 0～2k：快速建立注入通路

```text
Step 0:     0.061584
Step 1,000: 0.033082
Step 2,000: 0.024401
```

这一阶段下降最快。由于 action output projection 为零初始化，Step 0 时 extractor gradient 为 0 是预期现象：

```text
extractor_grad_norm:
Step 0       0.000000
Step 1,000   0.004275
Step 2,000   0.018013
```

output projection 离开零点后，梯度能够正常传入 interaction extractor，说明新增分支的梯度链路有效。

#### 2k～10k：主要收敛阶段

```text
Step 2,000:  0.024401
Step 10,000: 0.018566
```

这一阶段 loss 继续下降约 23.9%。学习率从接近 `1e-4` 逐步衰减至 `1e-5`，没有出现明显震荡。

#### 10k～30k：长尾优化阶段

```text
Step 10,000: 0.018566
Step 20,000: 0.017714
Step 29,999: 0.016753
```

10k 到 30k 继续改善约 9.8%，说明训练尚未完全进入严格平台期，但边际收益已经明显减小。

仅根据训练 loss：

- 10k 已完成大部分离线优化。
- 20k～30k 仍有稳定的小幅收益。
- 没有直接证据表明 30k 发生了训练集层面的过拟合。
- 是否出现闭环或泛化过拟合，需要通过 rollout 评估确定。

## 4. Action horizon 分析

最终三个位置的 loss 为：

```text
first  = 0.020990
middle = 0.014932
last   = 0.019357
```

相对于 middle：

- first 高约 40.6%；
- last 高约 29.6%。

从 Step 0 到 Step 29,999 的降幅：

| Horizon 位置 | 降幅 |
|---|---:|
| First | 69.4% |
| Middle | 72.5% |
| Last | 73.7% |

模型最容易拟合 action chunk 的中间部分，首步最困难。首步误差值得重点关注，因为每次重新规划得到的第一个动作会直接影响：

- 当前状态与新 action chunk 的连续性；
- 机械臂执行抖动；
- 接触前的微小修正；
- 闭环状态是否偏离专家分布。

后续日志建议把 action loss 进一步拆分为：

```text
translation loss
rotation loss
gripper loss
first-step translation/rotation/gripper loss
```

## 5. 梯度与更新稳定性

最终训练指标：

```text
grad_norm                       = 0.049111
demovla_extractor_grad_norm     = 0.038367
demovla_injection_grad_norm     = 0.029080
demovla_output_proj_grad_norm   = 0.017322
update_norm                     = 0.010630
```

训练中 global gradient norm 大致处于：

```text
0.034 ～ 0.051
```

它远低于配置的 gradient clipping norm `1.0`，因此训练基本没有依赖梯度裁剪维持稳定。

10k 以后 extractor gradient 逐渐增大，但学习率已经固定为 `1e-5`，update norm 稳定在约 `0.0106～0.0109`。这不属于训练发散，更可能表示 interaction extractor 在低学习率阶段继续调整 memory 表示。

## 6. 参数范数

DemoVLA adapter 总参数范数：

```text
160.0378 → 164.8154
```

总增幅约 3%，没有整体参数膨胀。

共享 action output projection 的参数范数：

```text
0.0002 → 7.6666
```

该值在整个训练过程中单调增长。这符合零初始化设计，但也表明 output projection 承担了建立注入通路的主要工作。

实际注入由下式决定：

\[
\Delta h_l =
\operatorname{sigmoid}(g_l)
W_o\left(
\operatorname{Attention}(h_l,z_{\mathrm{int}})
\right)
\]

仅观察 `output_proj_param_norm` 无法判断真实注入强度。下一版日志应直接记录：

\[
r_l = \frac{\|\Delta h_l\|_2}{\|h_l\|_2}
\]

建议分别记录：

```text
demovla_injection_delta_norm_layer_4
demovla_injection_delta_norm_layer_9
demovla_injection_delta_norm_layer_14
demovla_hidden_norm_layer_4
demovla_hidden_norm_layer_9
demovla_hidden_norm_layer_14
demovla_injection_ratio_layer_4
demovla_injection_ratio_layer_9
demovla_injection_ratio_layer_14
```

## 7. 三层 Gate 分化

| 注入层 | 初始 gate | 最终 gate | 最终 raw gate | 相对变化 |
|---:|---:|---:|---:|---:|
| Layer 4 | 0.017986 | 0.018054 | -3.996178 | +0.38% |
| Layer 9 | 0.017986 | **0.019474** | **-3.918987** | **+8.27%** |
| Layer 14 | 0.017986 | 0.017568 | -4.023940 | -2.32% |

模型表现出如下倾向：

```text
Layer 9  >  Layer 4  >  Layer 14
```

可能的解释：

- Layer 4 较早，action hidden state 仍偏向输入动作和 flow timestep 表示。
- Layer 9 位于中层，可能最适合将 interaction memory 融入动作推理。
- Layer 14 接近输出侧，额外视觉语言修正可能干扰已经形成的 velocity 表示。

不过三个 gate 绝对值仍然接近，且三个位置共享 projection。在缺少 layer-wise delta norm 和 attention 分布时，不能仅根据 gate 判定实际贡献。

建议的下一项注入层消融：

```python
interaction_injection_layers = (9,)
```

并与 `(4, 9, 14)` 使用相同数据、随机种子和训练配置对比。

## 8. 与 Single-Shot 的离线比较

在接近 10k steps 的位置：

| 模型 | Step | Loss | First | Middle | Last | Output proj norm | Grad norm |
|---|---:|---:|---:|---:|---:|---:|---:|
| Single-shot | 9,999 | 0.026214 | 0.031587 | 0.023585 | 0.031561 | 10.441308 | 0.103996 |
| Sparse-deep | 10,000 | **0.018566** | **0.022942** | **0.016704** | **0.021183** | **7.126440** | **0.035349** |

在训练条件一致的前提下，sparse-deep 的离线 loss 比 single-shot 低约：

\[
29.2\%
\]

同时 sparse-deep 的 output projection norm 和 gradient norm 更低。这个结果说明，多层读取 interaction memory 比仅在 Action Expert 入口融合一次更容易优化当前动作目标。

该结论仍有以下限制：

- 两次实验必须确认使用完全相同的数据版本、数据顺序、global batch size、norm stats、基础 checkpoint 和随机种子。
- 更低的专家轨迹 action MSE 不等价于更高的闭环成功率。
- 三次深层注入可能降低训练分布上的误差，同时增加策略偏离专家轨迹后的敏感性。

## 9. 当前日志无法验证的内容

当前 metrics 不能判断：

1. Interaction queries 是否关注被操作物体和目标区域。
2. 四个 queries 是否产生角色分工。
3. 是否出现 query collapse。
4. Interaction memory 是否只学习了数据集背景或相机捷径。
5. Layer 9 的实际注入 activation 是否最大。
6. 训练 loss 的下降是否转化为 LIBERO rollout 成功率。
7. 30k checkpoint 的泛化是否优于 10k 或 20k。

需要增加的表示分析指标：

```text
interaction token norm
interaction token pairwise cosine similarity
interaction memory effective rank
action-to-memory attention entropy
attention received by each memory token
visual attention overlap between queries
layer-wise injection ratio
```

## 10. Checkpoint 评估建议

优先评估三个 checkpoint：

| Checkpoint | 目的 |
|---:|---|
| 10k | 学习率刚到最低值，主要优化已经完成 |
| 20k | 中间长尾阶段，检查额外训练是否改善闭环 |
| 29,999 | 最低训练 loss，检查是否存在闭环过拟合 |

评估要求：

- 每个任务至少 10 次 rollout，避免单次 trial 的高方差。
- 所有 checkpoint 使用相同 seed 集合。
- 同时报告整体成功率和逐任务成功率。
- 记录抓取成功、物体选择错误、放置失败和超时等失败类型。
- 保留失败视频，观察首步抖动、接触阶段偏移和末端动作累积误差。

结果解释：

- 如果 30k 持续优于 20k 和 10k，长尾训练有效。
- 如果 10k 或 20k 优于 30k，说明后期存在闭环或泛化层面的过拟合。
- 如果 sparse-deep loss 更低但成功率低于 single-shot，说明当前 action loss 与闭环控制目标存在偏差，或深层注入破坏了预训练策略分布。
- 如果 Layer 9 单层注入接近或优于三层注入，说明共享 memory 的重复注入不是必要条件。

## 11. 最终判断

本次 sparse-deep 训练数值健康、优化充分，且离线 action loss 明显优于 single-shot。当前没有证据表明训练本身失败，主要不确定性已经从“是否收敛”转移到以下问题：

1. interaction memory 是否学习了真正的视觉语言 grounding；
2. 三层深层注入是否改善闭环控制而非只改善动作拟合；
3. 10k～30k 的长尾收益能否转化为 rollout 成功率；
4. Layer 9 是否可以作为更精简、更稳定的单层深层注入位置。

因此下一步不建议仅根据最低训练 loss 选择 29,999，而应通过 10k、20k、29,999 的多次 rollout 结果选择最终 checkpoint。
