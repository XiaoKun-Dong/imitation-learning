# DemoVLA LIBERO-Plus 实验计划

## 目标

将后续仿真评测从标准 LIBERO 切换到 LIBERO-Plus，检验 DemoVLA interaction memory 在目标位置变化和干扰物增加时是否仍有可归因的闭环作用。不包含真机实验。

主 checkpoint 固定为 `stage_a_fixed005_seed42_v2/4999`，推理时三层 gate 均固定为 `0.03`。主对照为同一 checkpoint 的：

- `correct`：正确 memory；
- `zero_memory`：保持 gate 和网络路径不变，只把 memory 置零；
- `injection_off`：关闭整条注入路径。

三组必须共享 LIBERO-Plus task、initial state、environment seed 和 stateless flow-noise key。

## 数据划分与证据边界

2026-09-03 当天运行只作为迁移 smoke/pilot，不进入论文正式统计，也不用于挑选有利 task。正式任务面板和分析代码在查看 pilot outcome 前冻结。

第一主面板是 LIBERO-Plus `Objects Layout` 中的 target-displacement 子集：10 个 LIBERO Object 目标，每个目标固定选 `level1` 到 `level5` 的 `sample1`，共 50 个任务。先运行 10-task × 1-seed smoke，确认环境、任务和三条件配对完整；再运行冻结的 50 tasks × 3 seeds。

第二面板是同类别的 `add_*` 干扰物任务。它应在第一面板完成后独立冻结，不根据第一面板的成功或失败挑任务。Camera Viewpoints 仅作为后续辅助泛化测试。

## 主要指标

1. 三条件总体成功率；
2. `correct` 对 `zero_memory`、`injection_off` 的逐 task/seed 配对翻转；
3. McNemar exact p-value；
4. 以 benchmark task 为 cluster 的成功率差 bootstrap 95% CI；
5. 按任务名称中的 `level1`–`level5` 分层的成功率；
6. grasp failure、post-grasp failure 和 wrong-object grasp 作为失败分类。

`correct > injection_off` 只能证明完整 memory 注入路径的行为作用；只有 `correct > zero_memory` 才更接近样本相关内容作用。任何单一 task 的正向或反向翻转都只作为案例，不单独形成结论。

## 运行

迁移 smoke（不计入正式结果）：

```bash
LIBERO_PLUS_SEEDS=7 \
bash examples/libero/eval_demovla_stage_a_libero_plus.sh \
  checkpoints/demovla_libero_full_stage_a_fixed_gate/stage_a_fixed005_seed42_v2/4999 \
  experiments/demovla_libero_plus/pilot_runs/stagea4999_fixed003_tasks10_seed7 \
  8400 10
```

冻结主实验：

```bash
LIBERO_PLUS_SEEDS="7 42 123" \
bash examples/libero/eval_demovla_stage_a_libero_plus.sh \
  checkpoints/demovla_libero_full_stage_a_fixed_gate/stage_a_fixed005_seed42_v2/4999 \
  experiments/demovla_libero_plus/formal_runs/stagea4999_fixed003_target_displacement_v1 \
  8400 50
```

每个输出目录包含不可覆盖的 `manifest.txt`、三条件逐 seed 的 `metrics.jsonl`/视频/日志，以及自动生成的 `summary.md`。

## 决策规则

- smoke 中任何条件不足 10 条、任务 ID 不匹配、server ablation 不匹配或配对 key 不一致：停止正式实验并修复基础设施；
- 正式主面板优先解释 `correct` 对 `zero_memory` 的配对结果；
- 若只有 `correct > off` 而没有 `correct > zero_memory`，论文维持“memory pathway 有效，但内容级成功率收益未建立”；
- 若不同 seed 或 level 方向明显相反，报告异质性，不合并成单向收益叙述。
