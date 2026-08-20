# ArmHack Stand Step Stabilize 20260815

本目录保存 2026-08-15 完成的最新 ArmHack Stand 训练产物及复现快照。

## 模型身份

- run：`2026-08-15_04-42-42_armhack_stand_step_stabilize_from_twostep25_gpu0_20260815`
- checkpoint：`model_1999.pt`
- 训练完成时间：2026-08-15 06:21:43（Asia/Shanghai）
- actor：96 维观测、29 维动作
- 控制频率：50 Hz
- 动作缩放：`q_target = q_default + 0.25 * action`
- Stand command：固定 `[0, 0, 0]`

`policy.onnx` 和 `policy.pt` 均由本目录的 `model_1999.pt` 离线导出；部署参数与关节顺序记录在
`policy.deploy.json`。完整 SHA-256 见 `SHA256SUMS`。

## 文件

- `model_1999.pt`：完整 RSL-RL/PPOAMP checkpoint，包含 actor、critic、优化器和 AMP 状态。
- `policy.onnx`：仅 actor 的 ONNX 部署模型。
- `policy.pt`：仅 actor 的 TorchScript 部署模型。
- `policy.deploy.json`：96→29 接口、关节顺序、默认姿态、PD 和时间步合同。
- `training_snapshot/agent.yaml`：训练时实例化的 agent 配置。
- `training_snapshot/env.yaml`：训练时实例化的完整环境配置。
- 当前分支提交了对应的最新训练源码；`training_snapshot` 额外保存训练时实际实例化的参数。

## 重要限制

这是按完成时间认定的最新训练 checkpoint，不等同于已经通过真机安全认证的发布模型。
本次只完成离线导出、模型结构检查和 PyTorch/ONNX 数值一致性检查；没有把它替换成默认真机模型。
真机测试必须显式指定模型路径，并继续使用吊架、急停、限幅和状态超时保护。
