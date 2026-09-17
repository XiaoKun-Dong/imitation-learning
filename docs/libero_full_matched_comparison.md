# Full LIBERO π0.5 × DemoVLA 严格匹配实验

> 2026-08-24 执行决策：第一轮改为以官方 `pi05_libero` SFT checkpoint 为参照，
> 暂不重训 matched π0.5 control。DemoVLA 使用当前完整数据重新计算的 stats；官方
> checkpoint 保留其随权重发布的 stats。因此本轮属于官方外部基线对比，尚不能替代
> 同初始化、同数据、同训练预算的架构因果对比。

## 1. 目标

在标准 Full LIBERO 40-task 数据上重新训练 π0.5 control 与 DemoVLA，建立不受
Object-only normalization 混杂影响的主对比。标准训练集由以下四个 suite 组成：

- LIBERO-Spatial；
- LIBERO-Object；
- LIBERO-Goal；
- LIBERO-10（Long-horizon）。

数据源为 `physical-intelligence/libero`，预期规模为 1693 episodes、273465 frames、
40 tasks。现有 Object-only 数据保留在 `data/lerobot/local/libero`，不覆盖、不参与本次
训练。

## 2. 严格匹配约束

| 项目 | π0.5 control | DemoVLA |
|---|---|---|
| 配置 | `pi05_libero_full_matched` | `demovla_libero_full_matched` |
| 初始化 | `pi05_base/params` | 同一 `pi05_base/params`，新增模块零影响初始化 |
| 数据 | `physical-intelligence/libero` | 相同 |
| Norm stats | 当前 40-task 数据精确重算 stats | 相同文件 |
| Action horizon | 10 | 10 |
| Global batch | 32 | 32 |
| Train steps | 30000 | 30000 |
| Seed | 42 | 42 |
| LR | warmup 10000，之后 `5e-5` | 相同 |
| Optimizer | AdamW，gradient clip 1.0 | 相同 |
| EMA | 0.999 | 0.999 |
| 训练参数 | 全模型 | 全模型及新增 DemoVLA 参数 |
| 并行 | 四卡 FSDP | 四卡 FSDP |
| XLA allocator | 95% 单池预分配 | 相同 |

四卡真实 smoke 表明 global batch 128 和 64 在 4×RTX 4090（48 GiB）上会分别于首个
反向步骤额外申请 17.88 GiB 和 12.04 GiB，并因峰值/显存碎片化 OOM。因此两组同时固定
为 global batch 32（每卡 8）。30k steps 共处理 960000 个样本，约为 3.51 个 full-data
epoch。关闭预分配时 batch 32 仍因 10.78 GiB 连续块碎片化失败；统一设置
`XLA_PYTHON_CLIENT_PREALLOCATE=true`、`XLA_PYTHON_CLIENT_MEM_FRACTION=0.95` 后真实前向、
反向和优化器更新通过。不改变全模型训练、FSDP 分片、优化器或其他变量。

两者的实验方法差异是 DemoVLA interaction-memory 架构、dynamic gate 及方法内
diversity regularization。后续 diversity/gate 消融应独立训练，不能用推理时关闭
injection 代替训练对照。

## 3. 数据与统计门禁

训练前必须通过 `scripts/verify_libero_full_dataset.py`：

1. `meta/info.json` 必须为 LeRobot v2.0，且规模严格等于 1693/273465/40；
2. `tasks.jsonl`、`episodes.jsonl` 记录数与 metadata 一致；
3. 必须存在 1693 个 Parquet，逐文件读取 `state/actions` 后总行数为 273465；
4. 不允许残留 `.incomplete` 文件；
5. 独立重算 state 和 10-step action-chunk 的 mean/std/q01/q99，并与训练 stats 比较；
6. 记录训练 stats 的 SHA256 和逐字段最大绝对差。

后续 matched architecture 对照的两边统一使用：

```text
assets/libero_full_matched/physical-intelligence/libero/norm_stats.json
```

其 SHA256 为：

```text
44af18e5ac3a8cb142670a3d21d59d55174c844cc8ac920e67d6ff667ec9174e
```

下载完成后的审计发现，`pi05_libero` checkpoint 自带 stats（SHA256
`b3a44bb2810436fb62917decaea58bd4d9110255df527dea21e8fd40c960bd84`）与当前发布的
40-task Parquet 并非同一精确版本/口径。最大差包括 `state.std=0.02223` 和
`actions.mean=0.08490`，明显超过浮点或直方图近似误差。因此未来 matched control 对比
从当前 273465 帧精确重算，并对 control 与 DemoVLA 使用同一新文件。第一轮官方 SFT
对比则必须保留官方 checkpoint 自带 stats，否则会破坏官方权重的输入/输出标定。原文件
不覆盖，差异记录保存在 `outputs/libero_full_matched/official_stats_mismatch_audit.json`。

不得为 DemoVLA 单独切换 Object-only 或其他 stats。

## 4. 训练入口

```bash
bash scripts/train_libero_full_matched_4gpu.sh \
  control full_matched_seed42_v1

bash scripts/train_libero_full_matched_4gpu.sh \
  demovla full_matched_seed42_v1
```

四卡只能运行一个 FSDP 作业。该双训练入口保留给后续严格 matched 实验；每个入口都会
先重新执行数据与 stats
门禁，并将审计结果写入：

```text
outputs/libero_full_matched/full_matched_seed42_v1/<target>_dataset_and_stats_audit.json
```

实际文件分别带 `control_` 和 `demovla_` 前缀，并同时保存对应的
`*_training_contract.txt`，用于确认两次作业使用了同一数据路径、初始化 checkpoint
和 stats 哈希。

原远端顺序监督脚本设计为先训练 DemoVLA、再训练 control。根据第一轮采用官方 SFT 的
决策，DemoVLA 启动后已停止该调度器；训练子进程不受影响，matched control 不会启动。
当前实际训练命令等价于：

```bash
nohup bash scripts/train_libero_full_matched_4gpu.sh \
  demovla full_matched_seed42_v1 \
  > outputs/libero_full_matched/full_matched_seed42_v1/demovla_train.log 2>&1 &
```

历史启动状态仍保存在 `sequence_status.txt`；正式训练日志为 `demovla_train.log`。

当前后台监督器等待 DemoVLA 训练完成，再自动执行官方 SFT 与 DemoVLA 的
`10 trials/task` pilot 和配对汇总：

```bash
nohup bash scripts/run_libero_official_sft_posteval.sh \
  full_matched_seed42_v1 10 7 3324864 \
  > outputs/libero_full_matched/full_matched_seed42_v1/official_sft_posteval_supervisor.log 2>&1 &
```

它在 DemoVLA 训练 PID 结束后校验双方 checkpoint 的 params 与 stats，再开始占用 GPU；
任一 checkpoint 不完整都会记录 blocked 并退出。

## 5. 后续评测门禁

训练完成后，两个 checkpoint 各自使用随 checkpoint 保存的 stats：官方 SFT 使用官方
bundled stats（SHA256 `b3a44b...bd84`），DemoVLA 使用当前完整数据重算 stats（SHA256
`44af18...174e`）。双方按
相同 environment seed 和固定逐-replan flow noise 依次评测：

```text
libero_spatial
libero_object
libero_goal
libero_10
```

先执行 `10 trials/task` pilot；只有双方均无加载或系统性失败后，才扩展至
`50 trials/task` 正式测评。报告总体与逐-suite Wilson 95% CI、逐 episode 精确
McNemar 检验，并保留 stateful-RNG 分数作为单独的官方口径复核。

四 suite 并行入口：

```bash
bash examples/libero/eval_libero_full_4gpu.sh \
  official_pi05_sft_full_matched_seed42_v1_pilot10 \
  pi05_libero \
  /home/dongxiaokun/baseck/pi05_libero \
  10 7

bash examples/libero/eval_libero_full_4gpu.sh \
  demovla_full_matched_seed42_v1_pilot10 \
  demovla_libero_full_matched \
  checkpoints/demovla_libero_full_matched/full_matched_seed42_v1/29999 \
  10 7
```

第二个作业完成后生成配对比较：

```bash
.venv/bin/python examples/libero/summarize_full_suite_eval.py \
  data/libero/videos/full_matched/demovla_full_matched_seed42_v1_pilot10 \
  --reference-root \
  data/libero/videos/full_matched/official_pi05_sft_full_matched_seed42_v1_pilot10
```

## 6. 实际执行状态（2026-08-24）

- 本地下载完成并同步远端：1693 Parquet、273465 rows、40 tasks、0 incomplete；
- 本地到远端 `rsync --checksum --dry-run`：0 differences；
- 本地与远端独立 stats audit：全部字段最大绝对差均为 0；
- DemoVLA 四卡 smoke：global batch 32 + 95% XLA 预分配下真实前向、反向和更新通过；
- 正式 DemoVLA 作业于 `2026-08-24T13:48:03+08:00` 启动；随后按决策停止顺序调度器，
  训练 PID 继续运行且 matched control 不再自动启动；
- 正式 DemoVLA step 0：loss `0.0937204`、grad norm `0.992046`、adapter grad norm
  `0.0185122`，均为有限值；
- 正式 DemoVLA step 100：loss `0.0753884`、grad norm `0.580744`、adapter grad norm
  `0.0112620`、dynamic gate mean `0.0474415`，训练稳定且新增模块获得非零梯度；
- 当前训练未开启 W&B（`wandb_enabled=False`）；全部标量持续保存在 `metrics.jsonl`，
  可在训练后生成曲线和文档；
- 稳态初始吞吐约 `2.9 s/step`，DemoVLA 30k 预计约 24–25 小时，checkpoint I/O
  另计。

远端状态与日志：

```text
outputs/libero_full_matched/full_matched_seed42_v1/sequence_status.txt
outputs/libero_full_matched/full_matched_seed42_v1/demovla_train.log
outputs/libero_full_matched/full_matched_seed42_v1/official_sft_posteval_status.txt
checkpoints/demovla_libero_full_matched/full_matched_seed42_v1/metrics.jsonl
```
