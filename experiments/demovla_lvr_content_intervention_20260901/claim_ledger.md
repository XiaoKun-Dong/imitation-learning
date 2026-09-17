# DemoVLA LVR content intervention：冻结结论台账

协议：`demovla_lvr_content_pilot_v1`  
冻结时间：2026-09-01 18:15:43（Asia/Shanghai）  
范围：仅 LIBERO 仿真；不含此前同日的 identical-object 压力测试

## 实验前允许陈述

1. Stage A 在 1,024 个配对样本上建立了 loss-level 的 sample-specific memory
   sensitivity；correct flow loss 显著低于 shuffled/zero/off。
2. Stage A fixed-0.03 在 400-episode 闭环中显著优于 injection off，但相对 matched
   zero-memory 的差异不显著。
3. Dynamic `495/500` 是 checkpoint 与 local norm stats 的匹配部署结果，不是相对正确
   配置官方 π0.5 的纯架构增益。
4. 现有 prompt counterfactual 支持“视觉主导、弱语言条件化”，不支持强 task-semantic
   grounding。
5. Episode 8/replan 21 是探索性局部内容因果案例，尚不能推广到总体行为。

## 实验前禁止陈述

- Correct memory 稳定提升总体成功率。
- Memory 内容已被证明在总体上导致任务成功。
- Attention heatmap 证明目标 grounding、query 角色或行为因果。
- DemoVLA 显著优于正确配置的官方 π0.5。
- Pilot 中的任何单点翻转代表总体效应。

## 本轮唯一主问题

从逐元素一致的 correct-memory 前缀出发，只在预注册 replan 替换一次 memory 内容，
`correct` 是否比来自另一冻结样本的 `shuffled` 更经常完成任务？

主比较为 `correct vs shuffled`；`correct vs zero` 和 `correct vs injection_off` 为必要
次级比较，`wrong_prompt` 为诊断条件。四点 pilot 只验证协议和归档完整性，不做显著性
声明。无论正例、负例、同成败或基础设施失败，均不得从结果表中删除。

## 扩展决策

只有至少 3/4 pilot 点到达 branch、通过 initialization/prefix/image/state 四项一致性
检查，并完整生成所有条件的 summary、trajectory 和 video，才冻结新的 20–50 点独立
列表。扩展抽样不能使用 pilot 的 intervention outcome 作为选择依据。
