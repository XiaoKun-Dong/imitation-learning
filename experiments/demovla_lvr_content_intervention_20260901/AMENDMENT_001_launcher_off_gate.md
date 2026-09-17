# AMENDMENT 001：injection-off server 启动参数修复

时间：2026-09-01 18:20（Asia/Shanghai）  
适用协议：`demovla_lvr_content_pilot_v1`

## 触发原因

p01 首次启动在任何 LIBERO rollout、policy client 或 branch intervention 运行前失败。
`injection_off` server 被 launcher 同时传入：

```text
--interaction-ablation off
--interaction-layer-mean-gates 0.03 0.03 0.03
```

模型按既有校验拒绝该无意义组合：off 模式不能再覆盖 fixed gate。失败日志完整保存在：

```text
experiments/demovla_lvr_content_intervention_20260901/runs/p01/
experiments/demovla_lvr_content_intervention_20260901/runs/p01_launcher.log
```

p02 在该次 shell 启动错误中没有创建输出目录，也没有启动 server。

## 唯一修改

`eval_demovla_content_lockstep_4gpu.sh` 仅在 ablation 不是 `off` 时传入 fixed gate：

```text
correct:     layer_mean + [0.03, 0.03, 0.03]
zero:        zero_memory + [0.03, 0.03, 0.03]
injection_off: off，不传 gate override
```

runner、checkpoint、stats、采样点、donor、条件定义、顺序、seed、终点和停止规则均未改变。

Launcher SHA256：

```text
before: 013bc69b60e75c049ba3e7a07f68ad05125c39ff3be3b9b8ed12a6d8936f51c2
after:  2412482fc36009ebd71c28c986c6ec8a3842f676aff5b859b99a01bb8985d390
```

## 重跑规则

保留首次基础设施失败，不覆盖。p01 使用原注册参数写入 `p01_retry01`；p02 仍使用 `p02`。
该修复不接触任何 intervention outcome，因此不触发重新抽样或结论修改。
