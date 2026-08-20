# ArmHack Stand MuJoCo Validation

本文记录 Stand ONNX 在 MuJoCo 中的可视化与稳定性测试入口。

## 当前模型

- ONNX: `legged_lab/deployment/armhack_stand/stand.onnx`
- SHA-256: `0801f6463211503b69a231855f7488180713eef8b9c1705d6dce818d7605b8ce`
- 本机 MuJoCo Python: `/home/hecggdz/miniconda3/envs/env_leglab/bin/python`

## 保存的命令

直接运行：

```bash
cd /home/hecggdz/ARM-HACK/LocomotionHEC/legged_lab
bash scripts/val_mujoco_g1_armhack_stand_5traj_random.sh
```

等价展开命令：

```bash
cd /home/hecggdz/ARM-HACK/LocomotionHEC/legged_lab
POLICY_PATH="$PWD/deployment/armhack_stand/stand.onnx" \
MODE=randomized_trajectory ITEM=5 PAYLOAD_KG=1.0 \
JOINT_RANDOM_ENABLE=True JOINT_RANDOM_SEED=20260718 \
JOINT_POS_NOISE_RAD=0.03 JOINT_VEL_NOISE_RAD_PER_S=0.10 \
NON_ARM_JOINT_TARGET_NOISE_ENABLE=True NON_ARM_JOINT_TARGET_NOISE_SEED=20260719 \
NON_ARM_JOINT_TARGET_NOISE_RAD=0.02 \
USE_GLFW=True REAL_TIME=True \
bash scripts/val_mujoco_g1_armhack_stand.sh
```

## 轨迹语义

`MODE=randomized_trajectory ITEM=5` 现在表示连续播放前 5 条 randomized trajectory，并保留官方 sequence 中 2 秒 smooth bridge。脚本会从：

`Reference Data/ArmHack/StandPerturb/TestData/ArmOnly/sequences/randomized_trajectories_arm_only_sequence_seed20260715_50hz.csv`

截取到第 5 条轨迹结束，生成：

`deployment/armhack_stand/generated_mujoco_sequences/randomized_trajectories_first_05_seed20260715_50hz.csv`

如果只想跑第 5 条单独轨迹，使用：

```bash
MODE=randomized_trajectory TRAJECTORY_INDEX=5 bash scripts/val_mujoco_g1_armhack_stand.sh
```

## 关节随机扰动

Stand 验证入口默认开启初始关节随机扰动，用来做稳定性测试：

- `JOINT_RANDOM_ENABLE=True`
- `JOINT_RANDOM_SEED=20260718`
- `JOINT_POS_NOISE_RAD=0.03`
- `JOINT_VEL_NOISE_RAD_PER_S=0.10`

扰动施加在 29 个 policy joints 的初始 `qpos/qvel` 上，位置会按 MuJoCo XML 的关节限位裁剪；不会扰动 floating base。每次报告 JSON 中会记录 `initial_joint_randomization`，包括 seed、噪声上限和每个关节实际扰动。

此外，Stand 验证入口默认持续给双臂以外的 15 个腰腿关节加入目标位置噪声：

- `NON_ARM_JOINT_TARGET_NOISE_ENABLE=True`
- `NON_ARM_JOINT_TARGET_NOISE_SEED=20260719`
- `NON_ARM_JOINT_TARGET_NOISE_RAD=0.02`

这项噪声在每个 policy 控制周期重新采样，先叠加到非双臂 raw action，再通过 `target = default + action * action_scale` 进入 MuJoCo PD 控制。报告 JSON 中的 `policy_application` 会记录网络原始输出、实际执行 action、噪声和 `data.ctrl` 的统计，用于确认策略输出确实进入 MuJoCo 控制链路。

## 代码改动摘要

- `unitree_sim2sim2real/deploy/deploy_mujoco/deploy_mujoco_g1_amp.py`
  - 支持 `policy.pt` 和 `stand.onnx` 自动推理。
  - 增加初始关节随机扰动和非双臂关节持续目标噪声，并写入 metrics/report。
  - 增加 `policy_application` 诊断，确认 policy action 到 MuJoCo `data.ctrl` 的链路。
- `scripts/sim2sim_g1_amp_mujoco.sh`
  - 增加 `POLICY_RUNTIME=auto` 和 ONNX runtime 依赖检查。
  - 增加 `JOINT_RANDOM_*` 与 `NON_ARM_JOINT_TARGET_NOISE_*` 环境变量。
  - 对 GLFW viewer 在报告写完后的 shutdown segfault/abort 降级为 warning。
- `legged_lab/scripts/val_mujoco_g1_armhack_stand.sh`
  - 本机默认 Python 指向 `env_leglab`。
  - 支持直接使用部署 ONNX，跳过缺失 checkpoint 的导出流程。
  - `MODE=randomized_trajectory ITEM=N` 改为播放前 N 条 randomized trajectories。
  - `TRAJECTORY_INDEX=N` 用于只播放第 N 条。
- `legged_lab/scripts/val_mujoco_g1_armhack_stand_5traj_random.sh`
  - 保存本次 5 条轨迹稳定性可视化测试命令。

## 输出

报告默认写到：

`deployment/armhack_stand/Test Reports/StandArmOnlyMuJoCo/`

其中包含 Markdown 报告、JSON metrics、逐帧 trace CSV 和 torso 6D 曲线 PNG。
