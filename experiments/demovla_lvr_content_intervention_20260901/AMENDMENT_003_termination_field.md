# AMENDMENT 003：扩展实验显式记录终止类型

时间：2026-09-02 14:21（Asia/Shanghai）  
适用协议：`demovla_lvr_content_pilot_v1` 的后续独立扩展

## 修改

在启动任何扩展点之前，为每个 condition 的 summary 增加：

```text
termination = success | timeout
```

该字段完全由既有 `success` 与固定 520-step 上限确定，不改变 rollout、动作、
干预、停止条件或统计定义。试点旧 summary 未回写；其终止类型可由原始
`success` 与 `final_env_step` 无歧义恢复。

Runner SHA256：

```text
before: 64e5e1472ebbe79154af2d771ec87d8c8d29544fd1d4ecc72df16695d28ccc8d
after:  2b50ca4fe3c747f18e4b7226e024f85b5f1ad561b8dfe09fd6fdda74f1c26e3b
```

