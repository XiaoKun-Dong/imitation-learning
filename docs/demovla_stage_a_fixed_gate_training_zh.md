# DemoVLA 阶段 A：冻结主干的固定门控预热

## 1. 目的与结论边界

阶段 A 的目标不是直接证明 DemoVLA 优于 π0.5，而是在排除主干优化竞争后，验证交互记忆支路能否学到稳定、样本相关且有利于动作预测的残差。

本阶段使用已经完成 LIBERO SFT 的官方 π0.5 作为固定主干，只训练 DemoVLA 参数，并固定三层 gate。该实验能够支持“DemoVLA 是可插拔且具有因果作用的交互支路”；不能单独支持“从 base 训练的完整 DemoVLA 方法优于同预算 π0.5”。后者必须另做严格匹配的端到端对照。

## 2. 进入阶段 A 的依据

完整 LIBERO 全参数训练 `full_matched_seed42_v1/29999` 出现了明确的门控塌陷：最终 gate 约为 `2.5e-5`，而 output-projection 参数范数仍约为 `0.94`。这说明模型通过关闭 gate 绕过了交互支路，并非整个输出投影缩回全零。

对最终检查点执行了相同环境初始状态、相同逐 replan flow noise 的 40-task 门控扫描：

| 条件 | Spatial | Object | Goal | Long | 总计 |
|---|---:|---:|---:|---:|---:|
| Dynamic | 9/10 | 10/10 | 10/10 | 10/10 | 39/40 |
| Injection off | 9/10 | 10/10 | 10/10 | 10/10 | 39/40 |
| Fixed 0.01 | 9/10 | 10/10 | 10/10 | 10/10 | 39/40 |
| Fixed 0.03 | 9/10 | 10/10 | 10/10 | 10/10 | 39/40 |
| Fixed 0.05 | 10/10 | 10/10 | 9/10 | 10/10 | 39/40 |

`fixed=0.05` 相对 off 有一个 episode 由失败变成功，也有一个由成功变失败，McNemar `p=1`。因此旧残差不是单向有害，但也没有稳定收益。阶段 A 不复用旧 DemoVLA 权重，而是从官方 π0.5-LIBERO 主干重新初始化交互支路。

## 3. 训练配置

配置名：

```text
demovla_libero_full_stage_a_fixed_gate
```

核心设置：

| 项目 | 设置 |
|---|---|
| 数据 | `physical-intelligence/libero` 完整 40 tasks |
| 主干初始化 | 官方 `pi05_libero/params` |
| Norm stats | 同一个官方 `pi05_libero` checkpoint 的 assets |
| 冻结参数 | 所有非 `demovla_*` 参数 |
| 额外冻结 | `demovla_interaction_gates` |
| 可训练参数 | 除 gate 外的全部 `demovla_*` 参数 |
| 注入层 | action expert 第 4、9、14 层 |
| Gate | 每层固定为 `0.05` |
| Out-projection 初始化 | 高斯小随机初始化，标准差 `1e-3` |
| Global batch | 128，四卡每卡 32 |
| 学习率 | warmup 500，峰值 `1e-4`，5k 内衰减至 `1e-5` |
| 训练步数 | 5,000 |
| EMA | 关闭 |
| 保存间隔 | 1,000 steps |
| W&B | 开启，project 为 `demovla` |

选择固定 `0.05` 的原因：它处于原始初始化和 CAC-VLA 论文观测门值的相近量级；在旧检查点扫描中没有造成总体成功率下降；同时比 `0.01/0.03` 提供更大的初始任务梯度。该值只用于预热，不代表最终动态门的最优取值。

## 4. 相对旧训练的关键修改

### 4.1 阻断主干绕过路径

旧完整训练同时更新 π0.5 主干和 DemoVLA，主干可以独立降低 flow loss，最终选择关闭 gate。阶段 A 冻结主干，使任务损失的任何进一步改善只能来自 DemoVLA 支路。

### 4.2 固定 gate

gate 参数存在于 checkpoint 中，但被明确排除出优化器。训练过程中三层 gate 必须始终保持 `0.05`，防止支路尚未对齐时再次被优化器关闭。

### 4.3 避免零投影冷启动

旧模型的 action output-projection 全零初始化，第一步几乎只有 output-projection 获得任务梯度。阶段 A 使用 `1e-3` 小随机初始化，使 memory extractor、cross-attention 和 output-projection 从第一步就能共同获得任务梯度。

### 4.4 直接记录未门控残差

新增指标：

```text
demovla_adapter_ungated_delta_ratio_mean
demovla_adapter_injection_ratio_mean
demovla_adapter_ungated_delta_ratio_layer_{4,9,14}_mean
demovla_adapter_injection_ratio_layer_{4,9,14}_mean
```

其中：

```text
ungated_delta_ratio = ||delta|| / ||hidden||
injection_ratio     = ||0.05 * delta|| / ||hidden||
```

两者应近似满足 `injection_ratio ≈ 0.05 × ungated_delta_ratio`。该指标替代此前用两个均值相除得到的粗略估计。

## 5. 启动命令

四卡启动脚本：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash scripts/train_demovla_libero_stage_a_4gpu.sh stage_a_fixed005_seed42_v2
```

断点恢复：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash scripts/train_demovla_libero_stage_a_4gpu.sh stage_a_fixed005_seed42_v2 --resume
```

启动脚本会在训练前完成：

1. 完整数据集结构、episode 数量、parquet 数量和总帧数校验；
2. 官方 stats 的 SHA256 记录，以及它与本地精确重算 stats 的差值记录；
3. checkpoint、数据根目录、固定 gate 和可训练范围记录；
4. 四卡数量检查。

这里有意锁定官方 checkpoint stats，因此审计允许其与本地完整数据精确重算值不同，但不会隐藏差异。审计 JSON 中应满足 `passed=true`、`stats_match_required=false`；`stats_matched=false` 和逐字段差值会被完整保留。这样可以保证阶段 A 的 off 模式恢复官方主干坐标系，同时避免把两份 stats 错误标记为相同。

审计文件写入：

```text
outputs/demovla_stage_a/stage_a_fixed005_seed42_v2/
```

检查点写入：

```text
checkpoints/demovla_libero_full_stage_a_fixed_gate/stage_a_fixed005_seed42_v2/
```

## 6. 训练期间监控与停止条件

每 100 step 检查：

- flow loss 是否稳定下降；
- gate 是否严格保持 `0.05`；
- `ungated_delta_ratio` 是否从小随机初始化逐渐形成稳定量级；
- `injection_ratio` 是否进入可检测范围；
- DemoVLA extractor、injection 和 output-projection 的梯度是否持续非零；
- 是否出现 NaN、显存异常或数据读取错误。

建议阶段 A 末期目标：

```text
gated injection ratio：优先达到 1e-3 左右或更高
gate：严格保持 0.05
DemoVLA gradient norm：不长期低于 1e-6
```

不能只凭 injection ratio 宣布成功；最终仍须满足正确 memory 相对 shuffled memory 的任务损失和行为优势。

提前停止条件：

- 连续多个日志区间出现 NaN；
- gate 偏离 `0.05`，说明冻结过滤器失效；
- DemoVLA 全部梯度长期为零；
- 注入比例快速扩大并伴随 flow loss 持续恶化。

## 7. 阶段 A 的因果验收

至少在 step 1k、3k、5k 对同一个 checkpoint 执行以下干预：

1. correct memory + fixed gate；
2. injection off；
3. shuffled memory；
4. zero memory。

使用相同样本、环境初始状态和 flow noise。关键差值为：

```text
Correct − Off      ：整条 DemoVLA 支路的总作用
Correct − Shuffled ：样本相关 memory 内容的作用
Correct − Zero     ：memory 内容相对固定偏置的作用
```

进入阶段 B 前应满足：

1. correct-memory flow loss 稳定低于 shuffled memory；
2. 正常注入在配对行为评测中不劣于 off，并出现可复现的 contender-only 成功；
3. correct 明显优于 shuffled/zero，而不是只依赖固定残差偏置；
4. output-projection、memory extractor 和 query attention 均保持有效梯度；
5. 至少一个中间 checkpoint 和最终 checkpoint 通过上述验收。

若 correct、shuffled、zero 三者没有差别，则不能进入动态门阶段，应先修改 memory 学习约束，例如加入 correct-vs-shuffled 排序损失。

## 8. 阶段 B 预留接口

阶段 B 将参考 CAC-VLA，把当前按层固定标量门替换为读取 `hidden + retrieved delta` 的上下文通道门，并设置：

```text
gate_min = 0.01
gate_max = 0.10
```

动态门初始化时必须复现阶段 A 的有效固定门值，主干继续冻结。只有阶段 A 已证明 memory 内容具有因果作用后，动态门才有明确的学习目标：选择何时、在哪一层、哪些通道使用有效残差，而不是再次学习关闭支路。

## 9. 论文表述

阶段 A 成功后可表述为：

> 在冻结且 stats 匹配的 π0.5-LIBERO 主干上，DemoVLA 交互记忆支路通过固定门控预热形成了样本相关的动作残差；同检查点的 off、shuffled 和 zero-memory 干预验证了该支路的因果作用。

不能表述为：

> DemoVLA 在同预算端到端训练中优于 π0.5。

后一个结论仍需要从同一个 `pi05_base` 或共同 matched backbone 出发，加入 parameter-matched adapter，并严格匹配数据、stats、训练步数、seed 和计算预算。

## 10. 当前执行状态

正式运行：

```text
exp_name: stage_a_fixed005_seed42_v2
remote PID: 1622243
W&B run: je4xa65e
start time: 2026-08-25 16:11 Asia/Shanghai
```

step 0 已成功完成并写入 `metrics.jsonl`：

| 指标 | Step 0 |
|---|---:|
| Loss | 0.004358 |
| Flow loss | 0.004090 |
| Gate | 0.050293 |
| Ungated delta ratio | 3.468e-3 |
| Gated injection ratio | 1.744e-4 |
| DemoVLA grad norm | 6.288e-4 |
| Extractor grad norm | 2.057e-4 |
| Injection grad norm | 5.942e-4 |
| Output-projection grad norm | 5.939e-4 |
| Output-projection param norm | 1.026 |

gate 配置值为 `0.05`；冻结参数转换为 bfloat16 后 raw logit 从 `-2.94444` 量化为 `-2.9375`，对应运行时 gate `0.050293`。这是固定精度量化，不是 gate 被优化器更新。三个注入层的 gate 完全一致且标准差为零。

初始化阶段检查结果：

- 官方 stats 已由训练数据管线实际加载；
- 数据审计 `passed=true`；
- 官方 stats 与本地精确重算 stats 的不一致被显式保留为 `stats_matched=false`；
- GPU 0–3 均进入 100% 利用率；
- 未出现 NaN 或 OOM；
- 初始预计训练时间约 2 小时 48 分。

此前同名后缀 `v1` 在完成首个训练 step 后因控制台无法格式化字符串辅助指标而退出，未生成有效 checkpoint。日志格式化已增加数值/文本兼容测试，正式结果只使用 `v2`。
