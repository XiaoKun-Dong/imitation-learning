# DemoVLA 门控塌陷诊断与恢复方向

## 1. 文档目的

本文整理完整 LIBERO 训练中观察到的 DemoVLA 动态门控塌陷问题，并区分以下三个问题：

1. 动态门控是否被优化器关闭；
2. 输出投影和交互记忆支路是否仍然学到了非零信号；
3. 如何让交互记忆对动作生成产生可验证、不可替代的作用。

本文的统计快照取自 `full_matched_seed42_v1` 在 step 23900 左右的训练状态。最终结论仍需等待 30000 step 检查点及固定噪声配对评测。

## 2. 当前训练设置

当前实验配置为 `demovla_libero_full_matched`：

- 数据：完整 LIBERO 40 tasks；
- 初始化：`pi05_base/params`；
- global batch size：32；
- 训练步数：30000；
- 主干和 DemoVLA：同时全参数训练；
- 注入层：动作专家第 4、9、14 层；
- 门控：动态 sigmoid gate；
- gate 初始 logit：`-3.0`，对应初始概率约 `0.0474`；
- 输出投影：全零初始化；
- 辅助损失：查询注意力多样性和记忆词元多样性。

相关实现位置：

- 完整训练配置：`src/openpi/training/config.py`；
- 动态门控和残差注入：`src/openpi/models/demovla.py`；
- 训练指标记录：`scripts/train.py`。

## 3. 观测结果

### 3.1 训练本身保持稳定

截至 step 23900：

- loss 从 step 0 的 `0.09372` 降至 `0.01988`；
- 最近 20 个日志点的 loss 均值约为 `0.01926`；
- 未出现 NaN、OOM、掉卡或检查点保存错误；
- 10k 和 20k 检查点均完整保存；
- 四张 GPU 保持接近 100% 利用率。

因此，当前问题不是数值发散或系统故障，而是模型选择了不使用新增交互记忆支路的低损失解。

### 3.2 动态 gate 明显塌陷

| Step | Gate mean | Raw gate mean | 实际 injection ratio |
|---:|---:|---:|---:|
| 0 | 0.04743 | -3.000 | 0 |
| 100 | 0.04744 | -3.000 | 3.72e-5 |
| 500 | 0.04802 | -2.987 | 1.24e-3 |
| 1000 | 0.04854 | -2.976 | 3.94e-3 |
| 2000 | 0.04784 | -2.991 | 1.86e-3 |
| 5000 | 0.002240 | -6.485 | 1.06e-5 |
| 10000 | 9.42e-5 | -10.647 | 1.01e-6 |
| 15000 | 4.17e-5 | -11.639 | 3.19e-7 |
| 20000 | 3.49e-5 | -11.768 | 3.39e-7 |
| 23900 | 2.76e-5 | -11.972 | 3.09e-7 |

门控在前 2000 step 保持约 0.048，随后快速下降。到 step 23900，实际注入量仅约为动作隐藏状态范数的 `3e-7`，在功能上已经接近关闭。

### 3.3 输出投影没有参数塌陷

输出投影由全零初始化逐步增长：

| Step | Output-proj param norm | Output-proj grad norm | Output-proj update norm |
|---:|---:|---:|---:|
| 0 | 5.09e-6 | 1.85e-2 | 5.09e-6 |
| 100 | 0.01199 | 1.13e-2 | 1.30e-4 |
| 500 | 0.08397 | 7.61e-3 | 5.48e-4 |
| 1000 | 0.14536 | 1.18e-2 | 1.07e-3 |
| 2000 | 0.24582 | 3.36e-3 | 2.10e-3 |
| 5000 | 0.66900 | 2.77e-5 | 3.77e-3 |
| 10000 | 0.86381 | 1.50e-6 | 1.53e-3 |
| 20000 | 0.92228 | 4.75e-7 | 5.78e-4 |
| 23900 | 0.93237 | 3.28e-7 | 3.90e-4 |

输出投影的参数范数没有缩回零，而是增长到约 0.93。用

```text
injection_ratio / gate_mean
```

粗略估算未门控的 `delta/hidden` 比例，step 23900 仍约为 1.12%。该比值是两个均值之比，只适合作为量级诊断，后续应增加直接的 `ungated_delta_ratio` 指标。

因此目前更准确的判断是：

- 输出投影学到了非零残差；
- 交互支路并非参数意义上的全零；
- 动态 gate 将这部分残差几乎完全切断；
- 非零输出是否具有任务信息，仍需通过强制门控和打乱记忆实验验证。

## 4. 主要问题

### 4.1 主干与新增支路存在优化竞争

当前训练同时更新整个 π0.5 主干和 DemoVLA。主干能够不依赖交互记忆直接降低 flow loss，因此优化器最容易找到的解是：

```text
主干适应完整 LIBERO + 动态 gate 接近零
```

该解在训练损失上完全合法，但无法证明交互记忆机制有效。

### 4.2 零输出投影造成双重冷启动

当前初始状态同时满足：

- gate 约为 0.047；
- action output projection 为零。

因为 gated delta 在初始时严格为零，第一步主要只有输出投影获得任务梯度，gate 和上游记忆提取器几乎无法从动作损失中学习。等输出投影形成非零映射时，庞大的主干已经开始适应任务，新增支路失去竞争优势。

### 4.3 Gate 与输出投影存在尺度不可辨识性

残差的实际形式为：

```text
gate * output_projection(attended_memory)
```

模型可以同时缩小 gate、增大输出投影权重，而不保持任一参数的可解释尺度。当前已经出现 output-proj norm 增长、gate 快速下降的趋势。

参数范数并不等同于功能作用。判断支路是否有效必须看实际注入比例、正确记忆相对错误记忆的作用，以及行为消融结果。

### 4.4 现有 diversity loss 不能阻止支路关闭

现有多样性损失只在查询间余弦相似度超过 margin 时产生惩罚。查询在训练早期分化后，该损失很快接近零。它能够降低查询坍缩风险，但不约束：

- gate 是否接近零；
- residual delta 是否接近零；
- 正确交互记忆是否改善动作预测；
- 动作专家是否在行为上依赖该支路。

### 4.5 单纯增大 gate 初始化不能根治问题

把初始 logit 从 `-3` 提高到 `-2` 或 `-1` 只能推迟塌陷。只要主干和 gate 继续自由竞争，优化器仍可以把 raw gate 推向很大的负值；模型也可以通过缩小 delta 绕过对 gate 的约束。

## 5. 首要验证实验

### 5.1 最终检查点门控扫描

在相同环境初始状态和相同逐 replan flow-noise 下，对 30000 step 检查点测试：

1. 正常动态 gate；
2. injection off；
3. 每层固定 gate = 0.01；
4. 每层固定 gate = 0.03；
5. 每层固定 gate = 0.05。

解释标准：

- 固定 gate 优于 off：输出投影和记忆内容可能有效，主要问题在 gate；
- 固定 gate 与 off 相同：记忆支路没有形成可观察作用；
- 固定 gate 差于 off：当前残差可能是噪声或方向未对齐；
- 动态与 off 相同而固定 gate 更好：可以直接进入 gate 恢复训练。

### 5.2 正确记忆与打乱记忆比较

在相同 batch 或相同 rollout 条件下比较：

- 当前样本的正确交互记忆；
- batch 内随机置换的其他样本记忆；
- 全零记忆；
- 正常记忆但关闭注入。

只有正确记忆稳定优于 shuffled memory，才能说明支路使用的是样本相关交互信息，而不是额外参数容量或固定偏置。

### 5.3 增加直接诊断指标

后续训练应直接记录：

```text
ungated_delta_ratio = ||delta|| / ||hidden||
gated_delta_ratio   = ||gate * delta|| / ||hidden||
correct_vs_shuffled_flow_loss_gap
per-layer gate / delta / injection ratio
per-flow-bin gate / delta / injection ratio
```

避免继续用 `injection_ratio / gate_mean` 近似未门控残差。

## 6. 推荐恢复方案

### 6.1 阶段 A：冻结主干，固定门控预热

从已经适配完整 LIBERO 的检查点构造主干，冻结所有非 DemoVLA 参数，仅训练交互记忆模块。

建议设置：

```text
训练参数：仅 demovla_.*
门控：固定，不训练
固定 gate：根据门控扫描选择，优先尝试 0.03 或 0.05
训练步数：2000–5000
学习率：1e-4
warmup：约 500 steps
```

目标是让交互记忆在主干不能继续独立适应的条件下学会降低动作损失。

如果从当前检查点继续，应先通过固定 gate 的单 batch loss 扫描选取不会使 flow loss 突增的门控值，而不是直接强制较大 gate。

### 6.2 阶段 B：恢复动态门控并设置下限

将纯 sigmoid：

```python
gate = sigmoid(raw_gate)
```

改为有界且带下限的形式：

```python
gate = gate_min + (gate_max - gate_min) * sigmoid(raw_gate)
```

建议初始搜索范围：

```text
gate_min = 0.01
gate_max = 0.10
```

动态门控应初始化为复现阶段 A 的固定门控，然后继续保持主干冻结训练 2000–3000 steps，使 gate 学习层、动作位置和 flow time 的差异，而不是首先学习关闭整条支路。

### 6.3 阶段 C：小学习率解冻动作专家

交互支路稳定后，只解冻动作专家，暂不解冻 VLM 主干，并使用分组学习率：

```text
DemoVLA 参数学习率：5e-5
动作专家学习率：5e-6
VLM 主干：冻结
```

如果后续确实需要解冻 VLM，其学习率应进一步降低，并持续监控实际注入比例和 off/shuffled 差距。

## 7. 建议的结构修改

### 7.1 消除 gate 与 delta 的尺度耦合

在乘 gate 前对残差进行 RMS 归一化：

```python
delta = output_proj(attended_memory)
delta = delta / (rms(delta) + eps)
delta = delta * rms(hidden)
hidden = hidden + residual_scale * gate * delta
```

建议初始范围：

```text
residual_scale = 0.01
gate_min = 0.02
gate_max = 0.10
```

这样可以分离三个角色：

- output projection 学习残差方向；
- dynamic gate 学习何时、在哪一层和哪个动作位置使用记忆；
- residual scale 控制总体注入强度。

### 7.2 改善输出投影初始化

当前全零输出投影能够保证严格 no-op，但会阻断上游模块的初始任务梯度。可考虑两种方案：

1. **小随机初始化**：使用约 `1e-3` 尺度初始化输出投影，同时采用较小固定 gate；优点是所有模块从第一步就有梯度，缺点是不再是数学上的严格 no-op。
2. **ReZero 形式**：输出投影正常初始化，在残差外增加零初始化标量；能够保持初始 no-op，但仍需分阶段开放参数，避免标量自身重新塌到零。

对当前完整 LIBERO 实验，更推荐小随机初始化加固定门控预热。

## 8. 建议的训练约束

### 8.1 Correct-vs-shuffled 排序损失

使用相同的 imitation action，不引入掩码、边界框或外部教师：

```text
L_correct  = 使用正确交互记忆的 flow loss
L_shuffled = 使用 batch 内随机置换记忆的 flow loss

L_rank = max(0, margin + L_correct - L_shuffled)
```

建议初值：

```text
margin = 0.001
rank loss weight = 0.05–0.10
```

该约束直接要求正确记忆比错误记忆更有用，比单纯提高 gate 更接近目标。

### 8.2 实际注入量下限

可以对可微的实际注入比例增加 hinge 约束：

```text
L_use = max(0, ratio_min - mean(||gate * delta|| / ||hidden||))^2
```

建议从以下量级开始搜索：

```text
ratio_min = 1e-3
```

不能只约束 gate mean，否则模型可能提高 gate、同时把 delta 压到零。当前用于日志的 injection ratio 已经 `stop_gradient`，训练约束需要单独计算可微版本。

## 9. 不建议单独采用的做法

以下方法可以作为消融，但不应作为唯一修复：

- 只把 gate 初始化从 `-3` 改为 `-1`；
- 只增加 gate 均值正则；
- 只增大 diversity loss；
- 直接强制很大的固定 gate；
- 在未做固定 gate 扫描前从当前检查点盲目续训；
- 仅凭 output-proj 参数范数判断记忆支路有效。

## 10. 有效工作的验收标准

下一轮训练不应只报告 gate mean。至少需要同时满足：

1. 实际 gated injection ratio 稳定在可检测量级，初步目标为 `1e-3–1e-2`；
2. DemoVLA 参数梯度不长期降至 `1e-6` 以下；
3. 正确记忆的 flow loss 稳定低于 shuffled memory；
4. 固定噪声行为评测中，正常模式显著优于 injection off；
5. 正常模式显著优于 shuffled memory，而不仅仅优于全零记忆；
6. 查询注意力随任务、目标和相机视角产生可复现的差异；
7. 与官方 π0.5 SFT 的完整 suite 对比不出现系统性性能退化。

## 11. 推荐执行顺序

1. 保留并完成当前 30k 训练，将其作为全参数联合训练基线；
2. 完成官方 π0.5 SFT 与 DemoVLA 的自动 pilot 对比；
3. 对最终检查点执行 off/dynamic/fixed-gate 门控扫描；
4. 执行正确记忆、shuffled memory 和 zero memory 对比；
5. 根据结果判断采用 gate-only 恢复，还是重新训练整个 DemoVLA 支路；
6. 进行固定门控的 adapter-only 预热；
7. 加入 gate floor、残差归一化和 correct-vs-shuffled 约束；
8. 最后才以较低学习率解冻动作专家进行联合微调。

## 12. 当前阶段的论文表述边界

在最终配对评测完成前，不应把当前完整 LIBERO 训练作为“交互记忆被有效使用”的证据。当前可确认的是：

- 模型整体训练稳定；
- 输出投影学到了非零参数；
- 动态 gate 在全参数联合训练中发生明显塌陷；
- 最终有效性必须由固定 gate、off 和 shuffled-memory 行为实验决定。

如果最终动态模式与 injection off 几乎无差异，应将该结果记录为全参数联合训练下的机制失效案例，并使用分阶段训练结果支持后续方法结论。
