# AMENDMENT 002：计划干预点未到达时保留完整运行

时间：2026-09-02（Asia/Shanghai）  
适用协议：`demovla_lvr_content_pilot_v1`

## 触发原因

在启动 p03/p04 之前进行协议一致性复核时发现：若共享的 correct-memory
prefix 在预注册 `branch_replan` 前已经成功完成任务，runner 会因为不存在
branch payload 而报错。这不符合冻结计划中“保留失败与未到达干预点，不替换
样本”的规则。

p01_retry01 与 p02 均已到达干预点并完整结束，因此该边界修复不改变它们的
任何结果。修复发生在查看 p03/p04 outcome 之前。

## 唯一修改

当五个锁步环境均在预注册干预点前结束时：

- 保留五个条件的 rollout 视频、trajectory 与 replan 记录；
- 写出 `prefix_audit.npz`；
- 写出 `summary.json`，明确标记
  `status=intervention_not_reached` 和 `intervention_reached=false`；
- branch 图像、状态与动作差异字段写为 `null`，不伪造干预结果；
- 不补跑、不更换 episode、不提前或推后 branch。

正常到达干预点的运行增加 `status=completed` 与
`intervention_reached=true`，其算法与产物定义均不变。

Runner SHA256：

```text
before: 0e59e4441341f67a38970c5dfba04d4f23b3bd6e36cdebee9cdbfc6d37f7f0df
after:  64e5e1472ebbe79154af2d771ec87d8c8d29544fd1d4ecc72df16695d28ccc8d
```

