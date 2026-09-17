# Kuavo 真机 DemoVLA attention 可视化

该链路兼容 `kuavo_deploy.PolicyClient` 的 ZMQ 协议。GPU 推理机每次收到新的
action-chunk 请求时，会使用同一帧头部相机和右腕相机观测保存一张 DemoVLA
query attention 图。渲染只记录诊断结果，不改变动作采样路径。

## 1. 启动推理和可视化服务

在 `openpi` 目录运行：

```bash
CUDA_VISIBLE_DEVICES=0 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
.venv/bin/python scripts/serve_kuavo_attention_policy.py \
  --checkpoint-dir /home/dongxiaokun/checkpoints/xiaojianshangliao_dynamic_gate_v3_retry1/29999 \
  --config-name demovla_kuavo_right_dynamic_gate \
  --output-dir outputs/kuavo_attention \
  --port 5555 \
  --top-k 4
```

服务使用现有 checkpoint 中的 `kuavo_right/norm_stats.json`。每次推理额外执行
一次 diagnostics-only prefix forward，因此延迟会高于关闭 diagnostics 的正式控制。

## 2. 启动真机客户端

保持 `configs/deploy/deploy.yaml` 中：

```yaml
inference:
  policy_type: client
  task_prompt: "Pick and Place"
```

然后按照现有 Kuavo 部署流程运行 `run` 或 `go_run`。开始运动前应先完成 action
chunk 离线检查、关节范围检查和低速测试；attention diagnostics 会增加推理延迟，
不要直接沿用未重新验证的异步队列或控制频率设置。

## 3. 查看结果

每个客户端 `reset` 对应一个 episode：

```text
outputs/kuavo_attention/
├── latest.png
└── episode_0000/
    ├── kuavo_attention_replan_000_step_0000.png
    ├── kuavo_attention_replan_001_step_0001.png
    └── ...
```

- 每一行对应一个 interaction query；
- 每一列对应一个有效相机视角；
- 红色表示完整 patch attention；
- 黄色框表示跨相机联合 top-k patch；
- `latest.png` 始终是最近一次 replan，适合在图片查看器中自动刷新。

服务沿用现有基于 `torch.load` 的 Kuavo ZMQ 协议，只应暴露在可信网络中。

## 4. 使用训练集离线查看

不连接机器人时，可以直接从 LeRobot v3 训练视频中抽帧。以下示例展示 episode
0 中第一次夹爪闭合前的 frame 35，以及第一次张开前的 frame 72：

```bash
CUDA_VISIBLE_DEVICES=0 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
.venv/bin/python scripts/render_kuavo_dataset_attention.py \
  --episode-index 0 \
  --frame-indices 35 72 \
  --output-dir outputs/kuavo_dataset_attention
```

脚本使用 FFmpeg 解码数据集中的 AV1 视频。Attention 来自完整的
diagnostics-only prefix forward；为减少无关计算，动作 flow sampler 使用两步，
不会改变所导出的 interaction attention。
