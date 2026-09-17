# DemoVLA 迁移到 RoboCasa365：适用性、环境与实验计划

更新时间：2026-08-27

## 结论

RoboCasa365 适合验证 DemoVLA，但它更适合作为“长时程、阶段化、跨场景泛化”的主实验，而不是直接替换 LIBERO 后立刻进行全量训练。

其优势是：复合任务具有明确的子任务、原子技能、阶段（pick / place / navigate）和自然语言逐帧标注；target split 使用与 pretrain split 不重合的厨房和物体。因此可以把 memory 是否在关键 replan 改变动作，与任务成功率、阶段切换和跨场景泛化联系起来。

主要风险是域迁移和控制接口混杂：RoboCasa 的移动操作、三相机输入、长 horizon 和 12 维控制并不等同于 LIBERO。若直接把 LIBERO 上的 pi0.5 / DemoVLA 权重放入 RoboCasa，失败不能归因于 memory。RoboCasa 已提供在 300 个 human pretrain tasks 上训练 75000 steps 的官方 pi0.5 checkpoint，应优先用它建立可用基座，再研究 DemoVLA 的增量因果作用。

## 当前远端环境

- 服务器：`dongxiaokun@192.168.20.68`
- RoboCasa：`/home/dongxiaokun/robocasa`
  - 版本：1.0.1
  - Git：`a07e365c958c4216cd6bbd5f30b47f09a65c6f00`
  - 含资产约 23 GB、123434 个资产文件
- robosuite：`/home/dongxiaokun/robosuite`
  - 版本：1.5.2
  - Git：`5ce6643f3092639d08f7b0f90ed1c6a84f50552c`
- Conda：`/home/dongxiaokun/miniforge3/envs/robocasa`
  - Python 3.11
  - PyTorch 2.7.1+cu126
  - MuJoCo 3.3.1
  - NumPy 2.2.5
- GPU：8 张 RTX 4090，每张约 48 GB

已完成 `OpenDrawer` 的 EGL 无头测试：场景创建、观测渲染、reset 和 100-step 均成功；首次 reset 15.84 秒，单环境约 44.28 FPS。

注意：从 `/home/dongxiaokun` 直接启动 Python 会让 editable 仓库根目录被识别为同名 namespace package。运行 RoboCasa 命令时应先切到其他目录，例如：

```bash
cd /home/dongxiaokun/imitation-learning/openpi
MUJOCO_GL=egl conda run -n robocasa python ...
```

当前警告不阻塞 PandaOmron：`robosuite_models` 和 `mink` 仅在使用相应额外机器人或 IK 控制器时再安装；MimicGen 仅在需要其生成数据时安装。

## 资产不等于训练数据

已上传的是仓库和仿真资产，不包含 RoboCasa 的 LeRobot 演示数据。不要直接下载完整 2200+ 小时数据；先下载目标任务的小规模子集，验证数据字段、相机、动作、统计量、训练吞吐和评测闭环，再扩大范围。

## 官方 pi0.5 基线

RoboCasa leaderboard 已提供一组可复现的 pi0.5 提交：

- RoboCasa 版本：1.0.0
- 代码提交：`ca4c6d710db75e276bc7c866a57bd7e4aee5b6e8`
- 数据：`pretrain_human300`
- batch size：64
- 训练步数：75000
- Atomic-Seen：39.6%
- Composite-Seen：7.1%
- Composite-Unseen：1.2%
- checkpoint：`robocasa/robocasa365_checkpoints/pi05_pretrain_human300/multitask_learning/75000`

已核对该提交中的源码：它包含 `LeRobotRobocasaDataConfig`、RoboCasa policy transform 和 `pi05_pretrain_human300` 配置；模型使用 `Pi0Config(pi05=True, max_token_len=200)`，从 `pi05_base` 初始化，学习率峰值为 `2.5e-5`，配置 batch size 为 64。

这改变了首选路径：不需要先从 pi0.5 base 重训完整的 300-task 基座。应先下载官方 75k checkpoint 和其 norm stats，复现官方评测，再从该 checkpoint 分叉训练 DemoVLA。

版本必须严格区分：上述公开数字来自 RoboCasa 1.0.0，而当前远端安装的是 1.0.1。由于 1.0.1 调整了任务 horizon，当前环境中重测得到的数字不能直接与 1.0.0 leaderboard 数字比较。需要维护两条结果：

1. 在 1.0.0 上复现官方 checkpoint，作为外部复现结果；
2. 在 1.0.1 上用完全相同的 seeds、任务、horizon 和 evaluator 重测 official pi0.5 与 DemoVLA，作为主论文的严格配对结果。

## 推荐的推进顺序

### R0：接口闭环

1. 保留两个隔离环境：RoboCasa conda 环境负责仿真评测，当前 OpenPi `.venv` 负责策略训练与服务。
2. 在当前 DemoVLA/OpenPi 中增加 RoboCasa 数据转换、policy transform 和 websocket evaluator，不直接覆盖现有 LIBERO 实现。
3. 固定机器人、三相机、图像尺寸、动作顺序、action horizon、replan 周期和归一化统计量。
4. 用少量演示跑通 dataset sample -> train step -> policy server -> RoboCasa rollout。

### R1：官方基线复现

下载官方 pi0.5-75k checkpoint 及 norm stats。先在少量 atomic 和 composite tasks 上验证输入、动作与 evaluator，再分别执行 1.0.0 复现和 1.0.1 基线重测。

这一阶段只回答官方 checkpoint 是否被正确复现，不声称 memory 的贡献。

### R2：严格匹配的架构对比

从同一个官方 RoboCasa pi0.5-75k checkpoint 分叉：

- zero-additional-training baseline：官方 75k checkpoint 原样评测；
- continued-SFT baseline：使用与 DemoVLA 相同的额外数据、步数和优化预算继续训练 pi0.5；
- DemoVLA：相同数据、batch、步数、学习率预算、统计量和评测 seeds；
- 参数量控制：增加无 memory 或随机 memory 的等参数 adapter；
- 训练至少使用多个 seed，报告 task-level 和 seed-level 区间。

### R3：因果干预

在同一初始状态、同一任务提示、同一 replan 上做配对干预：

- correct memory；
- zero / off memory；
- wrong memory；
- 来自另一个 episode 或阶段的 memory transplant；
- injection layer knockout；
- 固定 gate 与 dynamic gate 扫描。

同时测量动作分布变化、关键物体/夹爪相关性、阶段完成率和最终成功率。只有当正确 memory 在需要记忆的关键阶段产生方向一致的动作变化，并且 wrong / transplant memory 导致可预测的退化，才构成比总成功率更强的因果证据。

## 首轮任务选择原则

优先 composite-seen 和 composite-unseen 中具备以下特征的任务：

- 相似物体或多个候选目标；
- 需要跨阶段保持目标身份；
- pick/place/navigate 间存在明确 replan；
- 基座存在可重复的局部失败，而不是完全不会控制机器人。

atomic task 用作控制组：若 DemoVLA 只在需要跨阶段记忆的 composite task 上增益，而在短程 atomic task 上增益很小，将更支持架构归纳偏置与任务需求匹配的主张。

## 公平比较边界

官方 RoboCasa OpenPi fork 的策略说明页仍主要描述 pi0，但 leaderboard 提交、代码提交和 checkpoint 已经覆盖 pi0.5。因此应以提交 `ca4c6d7` 的实际源码和公开 checkpoint 为准，把当前 DemoVLA 改动迁移到该基线，或把其中 RoboCasa 数据和评测实现移植到当前 DemoVLA 分支。无论选择哪条路径，最终 baseline 与 DemoVLA 必须来自同一个 pi0.5-75k checkpoint。

8 张约 48 GB GPU 的总显存充足，但官方文档给出的 OpenPi 单卡建议为至少 100 GB；当前多卡训练能否直接适配 RoboCasa 数据管线需要在 R0 中实测，不能仅按总显存相加。

## 下一项实际工作

先下载官方 pi0.5-75k checkpoint 和配套 norm stats，在少量任务上完成 checkpoint 加载与 rollout。随后比较官方 fork 与当前 DemoVLA 分支的 RoboCasa transform 差异，再决定采用“向官方 fork 移植 DemoVLA”还是“向当前分支移植 RoboCasa adapter/evaluator”。完成官方基线复现之后才下载训练数据和启动 DemoVLA 训练。

## 官方资料

- RoboCasa 仓库：https://github.com/robocasa/robocasa
- 安装：https://robocasa.ai/docs/introduction/installation.html
- 数据概览：https://robocasa.ai/docs/build/html/datasets/datasets_overview.html
- 数据使用：https://robocasa.ai/docs/build/html/datasets/using_datasets.html
- Benchmark 概览：https://robocasa.ai/docs/build/html/benchmarking/benchmarking_overview.html
- 策略算法：https://github.com/robocasa/robocasa/blob/main/docs/benchmarking/policy_learning_algorithms.md
- pi0.5 代码提交：https://github.com/robocasa-benchmark/openpi/tree/ca4c6d710db75e276bc7c866a57bd7e4aee5b6e8
- pi0.5-75k checkpoint：https://huggingface.co/robocasa/robocasa365_checkpoints/tree/main/pi05_pretrain_human300/multitask_learning/75000
