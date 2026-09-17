# Expansion 统计与审计实现说明

适用协议：`demovla_lvr_content_expansion_v1`  
记录时间：2026-09-02（e01/e02 已结束，e03--e30 尚未产生汇总）

本文件不修改冻结的主终点、纳入规则或检验，只明确
`expansion_manifest.json` 已注册分析的机械实现。e01/e02 均为 correct/shuffled
同成功，因此下列实现选择未根据 discordant 方向或显著性作调整。

## 有效 pair

只有 `intervention_reached=true` 且四项审计满足以下条件的注册点进入 causal pair
分母：

```text
initialization_max_abs_diff == 0
prefix_max_abs_diff == 0
branch_image_equal == true
branch_state_equal == true
```

`intervention_not_reached`、协议无效和基础设施失败均保留并逐点报告，但不伪装成
已执行干预的 pair。Pilot 四点不并入 expansion 主统计。

## 二元闭环统计

对 correct--comparator 定义差值：

```text
paired success-rate difference = mean(correct_success - comparator_success)
```

- discordant counts 同时报告 `correct-only` 与 `comparator-only`；
- p 值使用 discordant pair 上零假设概率 0.5 的双侧 exact binomial，即 exact
  McNemar；无 discordant pair 时 p=1；
- 95% CI 使用对注册有效 pair 做 100,000 次 paired percentile bootstrap，固定
  NumPy RNG seed `20260902`；该区间是效应量不确定性描述，不替代 exact p 值。

主比较为 shuffled；zero 与 injection_off 使用同一实现。Wrong prompt 只作诊断。

## 局部动作与轨迹

- action deviation 使用实际执行的五个 branch actions 的 L2；预注册非零阈值为
  `executed_l2 > 1e-6`；
- 轨迹分叉从第 `branch_replan * 5` 个环境动作后状态开始，在 correct 与 comparator
  的共同长度内计算 simulator state 最大绝对差；阈值为 `> 1e-6`；
- 成功后长度不同不外推或补齐状态。

## 完整性检查

自动分析必须逐点验证 manifest 的 task/episode/replan/donor SHA、四项 lockstep
审计、五路 summary/trajectory/replans/video、trajectory 长度、replan 行数和视频帧数。
任何不一致都使分析命令以非零状态退出，同时保留原始运行。

