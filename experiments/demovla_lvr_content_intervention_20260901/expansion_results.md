# DemoVLA closed-loop content intervention: 30 点扩展结果

注册点: 30; 有效干预 pair: 20; 干预点前完成: 10。

## 配对闭环统计

| comparator | N | correct/comparator success | correct-only | comparator-only | paired diff | 95% bootstrap CI | exact McNemar p | action nonzero | trajectory diverged |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| shuffled | 20 | 19/18 | 1 | 0 | 0.050 | [0.000, 0.150] | 1 | 20 | 20 |
| zero | 20 | 19/19 | 0 | 0 | 0.000 | [0.000, 0.000] | 1 | 20 | 20 |
| injection_off | 20 | 19/18 | 1 | 0 | 0.050 | [0.000, 0.150] | 1 | 20 | 20 |
| wrong_prompt | 20 | 19/18 | 1 | 0 | 0.050 | [0.000, 0.150] | 1 | 20 | 20 |

## 逐点结果

| point | task/episode | phase | branch | status | correct | shuffled | zero | off | wrong prompt |
|---|---:|---|---:|---|---:|---:|---:|---:|---:|
| e01 | 1/3 | early | r12 | completed | 1 | 1 | 1 | 1 | 1 |
| e02 | 2/6 | early | r13 | completed | 1 | 1 | 1 | 1 | 1 |
| e03 | 3/0 | early | r24 | completed | 1 | 1 | 1 | 1 | 1 |
| e04 | 4/8 | early | r21 | completed | 1 | 1 | 1 | 1 | 1 |
| e05 | 5/3 | early | r15 | completed | 1 | 1 | 1 | 1 | 1 |
| e06 | 6/2 | early | r14 | completed | 1 | 1 | 1 | 1 | 1 |
| e07 | 7/8 | early | r23 | completed | 1 | 1 | 1 | 1 | 1 |
| e08 | 8/0 | early | r24 | completed | 1 | 1 | 1 | 1 | 1 |
| e09 | 9/7 | early | r15 | completed | 1 | 0 | 1 | 0 | 0 |
| e10 | 10/6 | early | r15 | completed | 1 | 1 | 1 | 1 | 1 |
| e11 | 1/1 | middle | r50 | completed | 1 | 1 | 1 | 1 | 1 |
| e12 | 2/0 | middle | r37 | completed | 1 | 1 | 1 | 1 | 1 |
| e13 | 3/2 | middle | r45 | completed | 1 | 1 | 1 | 1 | 1 |
| e14 | 4/7 | middle | r41 | completed | 1 | 1 | 1 | 1 | 1 |
| e15 | 5/4 | middle | r37 | completed | 1 | 1 | 1 | 1 | 1 |
| e16 | 6/0 | middle | r41 | intervention_not_reached | 1 | 1 | 1 | 1 | 1 |
| e17 | 7/4 | middle | r38 | completed | 1 | 1 | 1 | 1 | 1 |
| e18 | 8/2 | middle | r43 | completed | 1 | 1 | 1 | 1 | 1 |
| e19 | 9/2 | middle | r36 | completed | 0 | 0 | 0 | 0 | 0 |
| e20 | 10/1 | middle | r49 | completed | 1 | 1 | 1 | 1 | 1 |
| e21 | 1/2 | late | r79 | intervention_not_reached | 1 | 1 | 1 | 1 | 1 |
| e22 | 2/2 | late | r70 | intervention_not_reached | 1 | 1 | 1 | 1 | 1 |
| e23 | 3/4 | late | r71 | intervention_not_reached | 1 | 1 | 1 | 1 | 1 |
| e24 | 4/4 | late | r70 | intervention_not_reached | 1 | 1 | 1 | 1 | 1 |
| e25 | 5/9 | late | r69 | intervention_not_reached | 1 | 1 | 1 | 1 | 1 |
| e26 | 6/7 | late | r73 | intervention_not_reached | 1 | 1 | 1 | 1 | 1 |
| e27 | 7/5 | late | r72 | intervention_not_reached | 1 | 1 | 1 | 1 | 1 |
| e28 | 8/1 | late | r64 | intervention_not_reached | 1 | 1 | 1 | 1 | 1 |
| e29 | 9/9 | late | r71 | completed | 1 | 1 | 1 | 1 | 1 |
| e30 | 10/3 | late | r66 | intervention_not_reached | 1 | 1 | 1 | 1 | 1 |

注: `intervention_not_reached` 行的五路结果来自完全相同的共享前缀, 不进入干预比较分母。
Pilot 四点未并入本表统计。CI 为固定 seed 的配对 percentile bootstrap; p 值为双侧 exact McNemar。
