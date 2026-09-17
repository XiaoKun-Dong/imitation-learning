# DemoVLA memory-content lockstep pilot 汇总

协议：`demovla_lvr_content_pilot_v1`  
范围：LIBERO 仿真；pilot 仅验证协议和归档，不做显著性声明

## 决策门

结果为 **PASS**：p01、p02、p04 共 3/4 点到达预注册 branch，且
initialization、prefix、branch image、branch state 四项审计全部通过；五个条件均有
summary、trajectory、replan 日志和视频。p03 在 r72 前由共享 correct prefix 完成，
按冻结规则保留为 `intervention_not_reached`，未替换样本或移动 branch。

## 逐点结果

| 点 | task/episode | branch | 状态 | correct | shuffled | zero | off | wrong prompt | shuffled action L2 |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| p01_retry01 | 8/9 | r18 | completed | 1 | 1 | 1 | 1 | 1 | 0.005128 |
| p02 | 1/5 | r46 | completed | 1 | 1 | 1 | 1 | 1 | 0.011277 |
| p03 | 5/2 | r72 | intervention_not_reached | 1 | 1 | 1 | 1 | 1 | N/A |
| p04 | 3/6 | r13 | completed | 0 | 1 | 0 | 0 | 0 | 0.008145 |

p03 的五个“1”只描述共享前缀成功，不能计入任何 intervention 比较。

在 3 个实际到达 branch 的点上，correct vs shuffled 的 discordant pair 为：
`correct-only=0`、`shuffled-only=1`。唯一翻转方向与主假设相反；因此试点不能支持
“correct memory 更常成功”，只表明协议可复现且单次 content pulse 可能造成异质的
闭环后果。Correct vs zero 和 correct vs off 在这 3 点上均无成功结果翻转。

## 完整性审计

- p01/p02/p04：`initialization_max_abs_diff=0`、`prefix_max_abs_diff=0`、
  `branch_image_equal=true`、`branch_state_equal=true`。
- p03：前缀两项 diff 均为 0；branch 字段按规则为 null。
- 四点所有视频帧数均与对应 condition 的实际执行步数一致。
- 四个 shuffled donor 的运行时 SHA256 均与冻结 manifest 一致。
- 首次 p01 launcher 基础设施失败完整保留；有效同参数重跑命名为 p01_retry01。

