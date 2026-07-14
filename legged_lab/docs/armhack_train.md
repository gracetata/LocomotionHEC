# ArmHack 训练实现、数据路径与验证指南

## 1. 当前状态

截至 2026-07-14，两个 ArmHack 任务均已完成数据整理、相对路径修复、本机环境验证和真实训练冒烟测试：

```text
LeggedLab-Isaac-AMP-G1-StandPerturb-v0
LeggedLab-Isaac-AMP-G1-WalkPerturbFinetune-v0
```

最终验证结果：

- 静态检查 `9 passed`；
- Stand：`2 env × 1 iteration` 正常完成并生成 `model_0.pt`；
- Walk：`2 env × 1 iteration` 正常完成并生成 `model_0.pt`；
- 两个最终日志均没有 traceback、`FileNotFoundError`、Isaac/Python 版本错误或 Nav2 数据缺失错误；
- 训练不再依赖 `/home/hecggdz/...` 等机器绝对路径。

## 2. Reference Data 目录

两个任务的姿态参考数据统一保存在：

```text
legged_lab/Reference Data/ArmHack/
├── README.md
├── StandPerturb/
│   ├── raw/
│   │   └── g1_full_body_motion_sdk_50hz.csv
│   └── g1_arm_trajectory_named_50hz.csv
└── WalkPerturbFinetune/
    └── g1_arm_pose_set.json
```

代码中的相对根路径是：

```python
Path("Reference Data") / "ArmHack"
```

运行时以 `legged_lab` 项目目录为锚点解析，因此从不同用户名或不同仓库父目录运行都不会依赖固定主目录。

### 2.1 Stand 原始数据和训练数据

原始文件来自：

```text
/home/user/Workspace/whole_body_joints_20260708_143133.csv
```

仓库保存副本：

```text
Reference Data/ArmHack/StandPerturb/raw/g1_full_body_motion_sdk_50hz.csv
```

原始文件共 20,122 帧，SHA-256：

```text
b43256da27b11a593fc244ab2dd7fb899490a575d7749ed858ac342e3a208c50
```

原始 `q0..q28` 顺序是：

```text
q0..q14  : 左右腿和腰部
q15..q21 : 左臂 shoulder pitch/roll/yaw、elbow、wrist roll/pitch/yaw
q22..q28 : 右臂 shoulder pitch/roll/yaw、elbow、wrist roll/pitch/yaw
```

源 CSV 有 1,346 行自然语言字段包含未引用逗号。为避免把容错逻辑留在训练关键路径中，转换脚本：

```text
scripts/tools/extract_armhack_stand_arm_csv.py
```

会读取每行最后 29 个 SDK q 值，只提取 `q15..q28`，生成严格的具名训练 CSV：

```text
Reference Data/ArmHack/StandPerturb/g1_arm_trajectory_named_50hz.csv
```

该文件仍为 20,122 帧，只含 15 列：`time_s + 14 个具名手臂关节`，SHA-256：

```text
afe3819937ecfa19fae835b8cc77038378ec40a821acd0fdf2feef0054583601
```

重新生成命令：

```bash
cd /home/user/Workspace/Humanoid/Locomotion/G1-Locomotion
conda run -n env_isaaclab \
  python legged_lab/scripts/tools/extract_armhack_stand_arm_csv.py
```

Stand 训练只读取规范的 14 关节文件，不会把 CSV 下肢姿态覆盖到机器人上。

### 2.2 Walk 三组双臂姿态

文件：

```text
Reference Data/ArmHack/WalkPerturbFinetune/g1_arm_pose_set.json
```

每只手臂的 7 关节顺序为：

```text
shoulder_pitch, shoulder_roll, shoulder_yaw, elbow,
wrist_roll, wrist_pitch, wrist_yaw
```

数据包含：

```text
pos1_back
left  = [0.91, 0.52, 0.11, 0.01, -0.12, -1.03, 0.01]
right = [0.91, -0.52, -0.11, 0.01, 0.12, -1.03, -0.01]

pos2_down
left  = [0.2504, 0.2650, -0.0919, 0.8356, 0.0031, 0.0104, -0.0102]
right = [0.2504, -0.2650, 0.0919, 0.8356, -0.0031, 0.0104, 0.0102]

pos3_front
left  = [0.27, 0.79, -0.22, -0.49, 0.85, 0.40, 0.05]
right = [0.27, -0.79, 0.22, -0.49, -0.85, 0.40, -0.05]
```

`reference_data.py` 会验证 JSON 单位为弧度、关节顺序正确、每侧恰好 7 个值、pose 名不重复、所有值有限，然后转换成环境使用的左右交错顺序。

## 3. 两个任务的实现

### 3.1 共同 action 路径

policy 输入仍是原 G1 AMP 的 96 维：

```text
base_ang_vel 3 + projected_gravity 3 + velocity_command 3
+ joint_pos 29 + joint_vel 29 + previous_action 29 = 96
```

actor 输出保持 29 维：

```text
policy action 29
  -> 保留 15 个腿部和腰部 action
  -> 用 CSV 或 JSON pose 覆盖 14 个手臂 action
  -> JointPositionAction
```

因此网络和部署接口维度不变，已有 G1/Nav2 policy checkpoint 可以继续作为初始化策略。

### 3.2 StandPerturb

配置：

```text
速度命令：vx=0、vy=0、wz=0
手臂来源：g1_arm_trajectory_named_50hz.csv
CSV 起点：每个环境 reset 时随机选择
CSV 循环：关闭，结束后保持最后一帧
AMP style reward：0.0
task/style lerp：1.0
RSI：关闭
```

行走、速度跟踪、步态和摆腿奖励被关闭，主要优化躯干 upright、高度、竖直速度、角速度、脚滑和摔倒终止。

### 3.3 WalkPerturbFinetune

每个环境 reset 时从三组 JSON pose 中随机选择一组。判别器只看 root 和 15 个腿/腰关节，torque、acc、action-rate 也只约束腿和腰。

速度命令现在使用自包含的 `UniformVelocityCommandCfg`：

```text
vx ∈ [-0.10, 0.30] m/s
vy ∈ [-0.12, 0.12] m/s
wz ∈ [-0.30, 0.30] rad/s
重采样周期：2.0 s
站立环境比例：2%
```

这样不再依赖外部 Nav2 recorded CSV。它保留标准三维速度命令接口，可接收 Nav2 的 `cmd_vel`，但当前训练分布应准确称为均匀速度命令，而不是真实 Nav2 recorded distribution。

## 4. 代码位置

| 作用 | 路径 |
|---|---|
| Reference Data 路径和 JSON 校验 | `source/legged_lab/legged_lab/tasks/locomotion/amp/config/g1_perturb/reference_data.py` |
| 29DoF action 覆盖、CSV 读取 | `source/legged_lab/legged_lab/envs/g1_perturb_env.py` |
| Stand 配置 | `source/legged_lab/legged_lab/tasks/locomotion/amp/config/g1_perturb/g1_stand_perturb_env_cfg.py` |
| Walk 配置 | `source/legged_lab/legged_lab/tasks/locomotion/amp/config/g1_perturb/g1_walk_perturb_env_cfg.py` |
| Gym task 注册 | `source/legged_lab/legged_lab/tasks/locomotion/amp/config/g1_perturb/__init__.py` |
| Runner 配置 | `source/legged_lab/legged_lab/tasks/locomotion/amp/config/g1_perturb/agents/rsl_rl_ppo_cfg.py` |
| 全身 CSV 转 14 手臂 CSV | `scripts/tools/extract_armhack_stand_arm_csv.py` |
| CSV 全身回放 | `scripts/tools/visualize_g1_csv_full_body_motion.py` |
| 通用训练脚本 | `scripts/train_g1_amp.sh` |
| Python 训练入口 | `scripts/rsl_rl/train.py` |
| policy-only checkpoint 加载 | `../rsl_rl/rsl_rl/runners/amp_runner.py` |
| 无仿真静态检查 | `source/legged_lab/test/test_g1_perturb_static.py` |

执行链：

```text
train_g1_amp.sh
  -> scripts/rsl_rl/train.py
  -> Gym task registry
  -> Stand/Walk env cfg + runner cfg
  -> G1PerturbAmpEnv
  -> RslRlVecEnvWrapper
  -> AMPRunner / PPOAMP.learn()
```

## 5. 必须使用的本机环境

进入环境：

```bash
source /home/user/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
cd /home/user/Workspace/Humanoid/Locomotion/G1-Locomotion/legged_lab
```

已验证版本：

```text
Python:     /home/user/anaconda3/envs/env_isaaclab/bin/python
IsaacLab:  0.54.2
rsl-rl-lib: 3.2.0
PyTorch:   2.7.0+cu128
GPU:       NVIDIA GeForce RTX 4090
CUDA:      available
```

不要使用旧的 `isaaclab` 环境：其中 `rsl-rl-lib=2.3.1`，低于训练入口要求的 `3.0.1`。

`train_g1_amp.sh` 的默认环境已修正为 `env_isaaclab`，默认 Conda 根目录为 `${HOME}/anaconda3`。已激活该环境后，无需再手写 `ISAACLAB_PYTHON`。

## 6. 训练命令

### 6.1 Stand 最小测试

```bash
TASK=LeggedLab-Isaac-AMP-G1-StandPerturb-v0 \
RUN_NAME=armhack_stand_smoke \
NUM_ENVS=2 \
MAX_ITERATIONS=1 \
HEADLESS=True \
QUIET_TERMINAL=False \
RANDOMIZATION_STRENGTH=0 \
bash scripts/train_g1_amp.sh
```

脚本识别到 Stand task 后会自动设置：

```text
RSI_ENABLE=False
STYLE_REWARD_SCALE=0.0
TASK_STYLE_LERP=1.0
```

### 6.2 Stand 正式训练

```bash
TASK=LeggedLab-Isaac-AMP-G1-StandPerturb-v0 \
RUN_NAME=stand_arm_trajectory_reference \
STYLE_REWARD_SCALE=0.0 \
TASK_STYLE_LERP=1.0 \
RSI_ENABLE=False \
NUM_ENVS=4096 \
MAX_ITERATIONS=8000 \
HEADLESS=True \
QUIET_TERMINAL=False \
bash scripts/train_g1_amp.sh
```

### 6.3 Walk 最小测试

```bash
TASK=LeggedLab-Isaac-AMP-G1-WalkPerturbFinetune-v0 \
RUN_NAME=armhack_walk_smoke \
STYLE_REWARD_SCALE=5.0 \
TASK_STYLE_LERP=0.4 \
RSI_ENABLE=False \
NUM_ENVS=2 \
MAX_ITERATIONS=1 \
HEADLESS=True \
QUIET_TERMINAL=False \
RANDOMIZATION_STRENGTH=0 \
bash scripts/train_g1_amp.sh
```

### 6.4 Walk 正式训练

```bash
TASK=LeggedLab-Isaac-AMP-G1-WalkPerturbFinetune-v0 \
RUN_NAME=walk_arm_pose_reference \
STYLE_REWARD_SCALE=5.0 \
TASK_STYLE_LERP=0.4 \
RSI_ENABLE=False \
NUM_ENVS=4096 \
MAX_ITERATIONS=4000 \
HEADLESS=True \
QUIET_TERMINAL=False \
bash scripts/train_g1_amp.sh
```

若从旧 Nav2/G1 policy 续训，还需要提供：

```text
RESUME=True
LOAD_RUN=<旧 run 目录名>
CHECKPOINT=<model_xxx.pt>
```

Walk runner 已默认 `load_policy_only=True`，会跳过旧 discriminator 和 optimizer，避免判别器观测维度变化导致 checkpoint 加载失败。

## 7. 最终测试记录

静态验证：

```text
bash -n scripts/train_g1_amp.sh
python -m compileall（ArmHack 环境、配置、转换和回放脚本）
python -m pytest -q source/legged_lab/test/test_g1_perturb_static.py
结果：9 passed
```

Stand 最终真实训练：

```text
run: logs/rsl_rl/g1_stand_perturb/2026-07-14_15-09-05_armhack_stand_final_smoke
结果: Learning iteration 0/1，Training time 1.50 s
数据: Reference Data/ArmHack/StandPerturb/g1_arm_trajectory_named_50hz.csv
产物: model_0.pt
```

Walk 最终真实训练：

```text
run: logs/rsl_rl/g1_amp/2026-07-14_15-09-39_armhack_walk_final_smoke
结果: Learning iteration 0/1，Training time 1.65 s
命令: UniformVelocityCommand
手臂: pose_set，三组姿态
产物: model_0.pt
```

结论：当前两个任务在指定 `env_isaaclab` 环境下均可从 task 注册、数据加载、Isaac Sim 创建、rollout 到一次 PPOAMP 更新完整运行，无已知启动阻塞。
