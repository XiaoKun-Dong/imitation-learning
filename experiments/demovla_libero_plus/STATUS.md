# DemoVLA LIBERO-Plus 实验状态

更新时间：2026-09-03（50-task pilot 已完成）

今天的运行全部标记为 infrastructure check 或 pilot，不进入论文正式统计。

## 已完成

### 首次启动检查（无有效 rollout）

输出目录：`pilot_runs/stagea4999_fixed003_tasks10_seed7`

三个 client 都在产生 `metrics.jsonl` 前退出。定位到两个运行时问题：远端 `run_libero_plus.py` 没有显式设置 `OPENPI_LIBERO_ROOT`，以及多 GPU client 的 `MUJOCO_EGL_DEVICE_ID` 错误地统一设为 0。两处已经修复；该目录没有实验结果，只作为失败审计记录保留。

### 1-task infrastructure check

输出目录：`pilot_runs/stagea4999_fixed003_task1_seed7_infra_v2`

- LIBERO-Plus `libero_object` suite：2518 tasks；
- 冻结任务：benchmark task 1840；
- `correct / zero_memory / injection_off` 均为 1/1；
- 三条件均生成 metrics、视频和配对 summary；
- 无 traceback。

### 10-task migration smoke

输出目录：`pilot_runs/stagea4999_fixed003_tasks10_seed7_v3`

固定选择 10 个不同目标的 `level1 sample1`：

| Condition | Success |
|---|---:|
| correct | 10/10 |
| zero_memory | 10/10 |
| injection_off | 10/10 |

三组没有 discordant pair。该结果只说明迁移和配对链路正确，同时显示 level1 存在天花板效应，不能支持条件优劣结论。

### 50-task × seed 7 pilot

输出目录：`pilot_runs/stagea4999_fixed003_tasks50_seed7_v1`

- 任务：10 个目标 × `level1`–`level5` × `sample1`；
- 条件：correct fixed-0.03、zero-memory fixed-0.03、injection-off；
- 环境 seed：7；flow-noise seed：0；
- GPU：0、1、2；端口：8400、8401、8402；
- 用途：检查难度分层、天花板/地板效应和失败分类，不进入正式统计。

实验已完整结束，三组各有 50 条 metrics，无 traceback，配对 task key 和 flow-noise seed 完全一致。

| Condition | Success | Failure | Grasp failure | Post-grasp failure | Wrong-object grasp |
|---|---:|---:|---:|---:|---:|
| correct | 38/50 | 12 | 7 | 5 | 4 |
| zero_memory | 39/50 | 11 | 5 | 6 | 3 |
| injection_off | 38/50 | 12 | 8 | 4 | 3 |

配对结果：

| Comparison | Difference | Correct-only | Comparator-only | 95% cluster bootstrap CI | McNemar p |
|---|---:|---:|---:|---:|---:|
| correct vs zero_memory | -2pp | 3 | 4 | [-12pp, +8pp] | 1.0 |
| correct vs injection_off | 0pp | 2 | 2 | [-8pp, +8pp] | 1.0 |

按任务名称中的目标位移 level：

| Level | correct | zero_memory | injection_off |
|---|---:|---:|---:|
| level1 | 10/10 | 9/10 | 10/10 |
| level2 | 8/10 | 9/10 | 9/10 |
| level3 | 8/10 | 8/10 | 7/10 |
| level4 | 7/10 | 7/10 | 8/10 |
| level5 | 5/10 | 6/10 | 4/10 |

成功率随位移 level 增大总体下降，说明该面板成功避开了纯天花板效应。但没有条件形成一致优势：correct–zero 为 3 个正向和 4 个反向翻转，correct–off 为 2 对 2。correct–zero 的 7 个翻转中有 6 个两边都抓到目标，差异主要发生在运输/放置阶段；tomato-sauce level5 是唯一明显的目标选择差异，correct 抓错且失败而 zero 成功。

因此该 pilot 只支持 LIBERO-Plus 对干预具有足够敏感性以及不同 memory 条件会产生行为分歧，不支持 correct memory 提高 OOD 成功率。

## 后续

冻结的正式主实验仍为相同 50-task 面板 × seeds `7/42/123`，并在非 2026-09-03 的独立输出目录运行。正式解释优先比较 `correct` 与 `zero_memory`，`correct` 与 `injection_off` 作为路径级对照。由于 pilot 效应接近零，正式实验的目的应写成估计效应和检验跨 seed 稳定性，不能改成寻找显著正向结果。
