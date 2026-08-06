# G1 极鲁棒全身站立恢复策略

## 1. 任务目标与边界

本任务训练一个“受扰恢复并持续站立”的 G1 策略：机器人以零速度指令运行，初始关节姿态、关节速度、基座姿态和基座速度都带随机偏差；每个 episode 内，躯干、骨盆、双臂和双腿还会持续受到随机外力、外力矩和速度冲击。策略需要允许必要的小幅恢复踏步，最终回到接近资产默认关节姿态的稳定站立状态。

“极鲁棒”在这里表示覆盖本文明确列出的有限训练分布，不表示可以从任意物理上不可恢复的姿态或无限大外力下站起。完整训练后仍需在独立随机种子、扩大扰动范围、MuJoCo 和真机安全架上分别验收。

本任务与 ArmHack Stand 的关键区别如下：

- 环境入口是标准 `legged_lab.envs:ManagerBasedAmpEnv`，不是 `G1PerturbAmpEnv`。
- Actor 输入保持原 G1 Stand 的 96 维观测，输出保持 29 维全身关节动作。
- 29 维网络输出直接送入所有腿、腰、肩、肘和腕关节；没有双臂动作劫持、CSV 轨迹或 `compose_action`。
- 策略只看到当前本体状态和零速度指令，不知道下一次外力的部位、方向、大小或时间。

## 2. 代码位置

| 内容 | 路径 |
| --- | --- |
| 任务注册 | `source/legged_lab/legged_lab/tasks/locomotion/amp/config/g1_extreme_stand_recovery/__init__.py` |
| 环境、奖励和随机化配置 | `source/legged_lab/legged_lab/tasks/locomotion/amp/config/g1_extreme_stand_recovery/g1_extreme_stand_recovery_env_cfg.py` |
| 默认姿态恢复奖励 | `source/legged_lab/legged_lab/tasks/locomotion/amp/config/g1_extreme_stand_recovery/rewards.py` |
| PPO-AMP 配置 | `source/legged_lab/legged_lab/tasks/locomotion/amp/config/g1_extreme_stand_recovery/agents/rsl_rl_ppo_cfg.py` |
| 训练启动器 | `scripts/extreme_stand_recovery/train_g1_extreme_stand_recovery.sh` |
| 默认姿态强化续训启动器（Pose V2） | `scripts/extreme_stand_recovery/train_g1_extreme_stand_recovery_pose_v2.sh` |
| 抗高频抖动与窄高斯姿态续训启动器（Pose V3） | `scripts/extreme_stand_recovery/train_g1_extreme_stand_recovery_anti_jitter_v3.sh` |
| 平滑 action 与力矩续训启动器（Smooth-Torque V4） | `scripts/extreme_stand_recovery/train_g1_extreme_stand_recovery_smooth_torque_v4.sh` |
| Isaac Sim 可视化 | `scripts/extreme_stand_recovery/vis_g1_extreme_stand_recovery.sh` |
| 最终模型 TorchScript/ONNX 导出 | `../scripts/export_g1_extreme_stand_recovery.sh` |
| MuJoCo 单项测试、完整套件与 GUI 可视化 | `../scripts/sim2sim_g1_extreme_stand_recovery_mujoco.sh` |
| MuJoCo 汇总报告生成器 | `../scripts/summarize_g1_extreme_stand_recovery_mujoco.py` |
| 随机全身姿态恢复测试 | `../scripts/test_g1_extreme_stand_random_pose_recovery_mujoco.sh` |
| MuJoCo 兼容入口 | `scripts/extreme_stand_recovery/val_mujoco_g1_extreme_stand_recovery.sh` |
| MuJoCo 扰动器 | `../unitree_sim2sim2real/deploy/deploy_mujoco/extreme_stand_recovery.py` |
| 真机 ONNX 部署入口 | `../scripts/deploy_real_g1_extreme_stand_recovery_onnx.sh` |
| 复用的 AMP 真机执行器链 | `../scripts/deploy_real_g1_amp_onnx.sh`、`../unitree_sim2sim2real/deploy/deploy_real/deploy_real_g1_amp.py` |
| 静态合同测试 | `source/legged_lab/test/test_g1_extreme_stand_recovery_static.py` |

训练任务 ID：

```text
LeggedLab-Isaac-AMP-G1-ExtremeStandRecovery-v0
```

可视化任务 ID：

```text
LeggedLab-Isaac-AMP-G1-ExtremeStandRecovery-Play-v0
```

## 3. 输入与输出

Actor 输入为 96 维：

| 分量 | 维数 |
| --- | ---: |
| 基座角速度 | 3 |
| 机体坐标系投影重力 | 3 |
| 速度指令，固定为 `[0, 0, 0]` | 3 |
| 29 个关节相对默认姿态的位置 | 29 |
| 29 个关节速度 | 29 |
| 上一时刻 29 维动作 | 29 |
| 合计 | 96 |

Actor 输出为 29 维原始动作，按 `G1_LOCOMOTION_JOINT_NAMES` 的 Lab/deployment 顺序解释，并通过 `default_joint_pos + action_scale * action` 形成全身关节位置目标。双臂也是 Actor 输出的一部分，不在环境或 MuJoCo 里另行覆盖。

## 4. 训练随机化

### 4.1 初始状态

默认每次 reset 都从资产默认状态出发并叠加独立均匀噪声：

| 对象 | 默认范围 |
| --- | --- |
| 腿部关节位置 | `±0.25 rad` |
| 腰部关节位置 | `±0.35 rad` |
| 双臂关节位置 | `±0.60 rad` |
| 腿/腰/臂关节速度 | 分别为 `±1.0/±1.25/±1.50 rad/s` |
| 基座 x、y、z 偏移 | `±0.15/±0.15/±0.08 m` |
| 基座 roll、pitch、yaw | `±0.25/±0.25/±0.30 rad` |
| 基座线速度 | x、y 为 `±0.50 m/s`，z 为 `±0.35 m/s` |
| 基座角速度 | roll、pitch 为 `±0.80 rad/s`，yaw 为 `±0.60 rad/s` |

关节位置必须采用 `reset_joints_by_offset` 的加性噪声。不能用乘性 `reset_joints_by_scale` 代替，否则默认值为零的关节不会获得任何位置扰动。

### 4.2 episode 内外力

外力事件在每个并行环境中使用独立时间钟，策略不知道未来扰动：

| 部位 | 力 | 力矩 | 重采样间隔 |
| --- | --- | --- | --- |
| `torso_link` | 每轴 `±35 N` | 每轴 `±5 Nm` | `2.0–5.0 s` |
| `pelvis` | 每轴 `±30 N` | 每轴 `±4 Nm` | `2.5–5.5 s` |
| 左右肩、肘 | 每轴 `±12 N` | 每轴 `±2 Nm` | `1.5–4.5 s` |
| 左右髋、膝 | 每轴 `±12 N` | 每轴 `±2 Nm` | `2.0–5.0 s` |

此外每 `3.0–6.0 s` 还会直接叠加一次随机基座速度冲击，覆盖三轴线速度和三轴角速度。训练同时随机化地面摩擦、躯干质量、骨盆/躯干质心、左右连杆质量、全身执行器增益、关节摩擦和关节 armature。

### 4.3 奖励

主要奖励和惩罚如下：

| 奖励项 | 权重 | 目的 |
| --- | ---: | --- |
| `alive` | `+1.0` | 持续存活 |
| `default_joint_pose_exp` | `+5.0`，`std=0.25 rad` | 29 个关节在广义坐标下恢复到默认站立姿态 |
| `default_leg_joint_pose_exp` | `+3.0`，`std=0.18 rad` | 单独约束 12 个腿部关节，避免腿部误差被 29 维平均稀释 |
| `default_key_body_pose_gaussian` | `+8.0`，`variance=4e-4 m²`（σ=2 cm） | 只有关键点非常接近默认笛卡尔姿态时才给出高奖励 |
| `default_feet_distance_l2` | `-8.0` | 双脚平面距离偏离资产默认距离时作对称二次惩罚 |
| `default_feet_distance_gaussian` | `+3.0`，`variance=1e-4 m²`（σ=1 cm） | 在默认双脚距离处增加陡峭奖励峰 |
| `joint_jerk_l2` | `-1e-8` | 惩罚 29 关节加速度在相邻 50 Hz 控制步之间的变化，抑制高频两帧振荡 |
| `track_torso_lin_vel_xy_exp` | `+1.5` | 躯干水平速度回到零 |
| `track_torso_yaw_rate_exp` | `+0.75` | 躯干 yaw 角速度回到零 |
| `double_support` | `+0.30` | 稳态时保持双足支撑 |
| `flat_orientation_l2` | `-2.0` | 基座保持水平 |
| `torso_roll_pitch_l2` | `-4.0` | 强化躯干姿态稳定 |
| `torso_height_band_l2` | `-0.80` | 躯干高度回到目标区间 |
| `root_xy_position_l2` | `-0.35` | 允许恢复踏步，但抑制长期漂移 |
| `feet_slide` | `-0.20` | 抑制接触脚滑动 |
| `termination_penalty` | `-1000.0` | 严厉惩罚非超时摔倒 |

行走奖励、步态时序奖励和 Arm style prior 都已关闭。不能把 `root_xy_position_l2` 设置得过大，否则策略可能因为害怕恢复踏步而更容易摔倒。

Pose V2 的约束针对旧模型“能站住，但会选择明显宽于默认值的支撑姿态”这一漏洞。Pose V3 继续保留全身和腿部广义坐标奖励作为远离目标时的平滑恢复梯度，但把最终姿态一致性改成小方差高斯核。关键刚体为 `torso_link`、左右膝、左右踝、左右肘和左右腕，共 9 个。笛卡尔位置相对 root 表示并消除 root yaw，因此不会要求机器人固定在某个世界坐标或绝对朝向，但会约束由关节和 root roll/pitch 造成的身体几何偏差：

```text
variance_cart = 0.0004 m²
r_cart = 8.0 × exp(-0.5 × mean_i(||p_i - p_i_default||²) / variance_cart)
```

当关键点 RMS 误差分别为 2 cm、5 cm、10 cm 时，未乘权重的高斯核约为 `0.607`、`0.044`、`3.7e-6`；因此只有接近默认几何姿态才能获得高分，不再像旧 `std=0.12 m` 宽核那样容忍明显偏差。

双脚距离奖励以启动时从资产默认姿态缓存的左右踝平面距离 `d_default` 为目标，不硬编码某个数值。V3 同时使用宽范围二次整形项和默认距离处的窄高斯峰：

```text
r_feet_penalty = -8.0 × (||p_left,xy - p_right,xy|| - d_default)²
variance_feet = 0.0001 m²
r_feet_peak = 3.0 × exp(-0.5 × (d_feet - d_default)² / variance_feet)
```

所以双脚过近和过远都会受到惩罚；这不是“越近越好”的奖励。默认参考在施加 reset 噪声前缓存，策略观测仍保持 96 维，不包含未来扰动或未来目标。

Jerk 不是当前加速度本身，而是相邻策略控制步的加速度差分。环境以 `step_dt=0.02 s` 计算：

```text
joint_jerk_t = (joint_acc_t - joint_acc_(t-1)) / 0.02
r_jerk = -1e-8 × mean_j(joint_jerk_t,j²)
```

该奖励项具有每个环境独立的上一帧加速度缓存。reset 后第一帧的 jerk 强制为 0，避免把随机初始姿态/速度写入造成的状态跳变错误地惩罚到策略；从第二个控制步开始才计算真实 jerk。

### 4.4 Smooth-Torque V4：针对真机持续振荡

此前并非只约束位置。Pose V3 已经包含：

- `dof_torques_l2=-2e-6`：实际执行力矩幅值；
- `dof_acc_l2=-1e-7`：关节加速度；
- `joint_jerk_l2=-1e-8`：关节 jerk；
- `action_rate_l2=-0.01`：相邻两帧 action 差。

但是 2026-07-31 的 `360 N × 0.20 s` MuJoCo 侧推诊断显示，恢复段 actor action 高频能量放大约 `123×`、action 变化率放大约 `684×`、PD 力矩高频放大约 `155×`，而力矩饱和率只有约 `0.10%`。这说明仅限制力矩幅值和一阶 action 差不够：策略仍可产生正负交替的二帧振荡，并通过 PD 链形成快速变化的实际力矩。

V4 保留全部默认姿态、关键点、足距、存活、躯干稳定和摔倒奖励，同时使用以下平滑项：

| 项目 | V3 权重 | V4 权重 | 作用 |
| --- | ---: | ---: | --- |
| `dof_torques_l2` | `-2e-6` | `-1e-5` | 直接限制29关节实际执行力矩幅值 |
| `joint_torque_rate_l2` | `0` | `-2e-7` | 惩罚实际执行力矩在相邻50 Hz控制步之间的变化率 |
| `dof_acc_l2` | `-1e-7` | `-5e-7` | 加强关节加速度惩罚 |
| `joint_jerk_l2` | `-1e-8` | `-5e-8` | 将 jerk 惩罚提高5倍 |
| `action_rate_l2` | `-0.01` | `-0.05` | 将相邻 action 差惩罚提高5倍 |
| `action_second_difference_l2` | `0` | `-0.10` | 惩罚 action 二阶差分，直接打击两帧交替和高频曲率 |

新增项定义为：

```text
delta_action_t = action_t - action_(t-1)
action_second_difference_t = delta_action_t - delta_action_(t-1)

joint_torque_rate_t = (applied_torque_t - applied_torque_(t-1)) / 0.02
```

`action_second_difference_l2` 对29维离散二阶差分求平方和；`joint_torque_rate_l2` 对29个 Isaac 实际施加力矩的导数求均方。两项均维护每个并行环境独立的历史缓存，并在 reset 后第一帧返回0，避免把随机重置的不连续状态算成策略抖动。这里约束的是策略输出及其最终产生的实际力矩，不是简单给 ONNX 输出增加部署端低通滤波，因此不会改变 `96 -> 29` 输入输出合同。

为避免策略只在默认站姿附近学会平滑，V4 的训练分布同时扩大为：

- 腿、腰、双臂初始位置噪声：`±0.30/±0.40/±0.65 rad`；
- 躯干外力/力矩：每轴 `±45 N/±6 Nm`；
- 骨盆外力/力矩：每轴 `±40 N/±5 Nm`；
- 肩肘、髋膝外力/力矩：每轴 `±15 N/±2.5 Nm`。

基座姿态、速度冲击、摩擦、质量、质心、执行器增益、关节摩擦和 armature 随机化继续保留。惩罚系数采用渐进增强而不是无限增大；若平滑项压过恢复奖励，策略可能选择少动但摔倒，因此正式训练必须同时观察 episode length、摔倒率和恢复姿态奖励。

## 5. 环境与完整训练

在每个新终端中都先执行：

```bash
cd /home/user/Workspace/Humanoid/Locomotion/G1-Locomotion
conda activate env_isaaclab
cd legged_lab
```

默认基础模型是：

```text
/home/user/Workspace/Humanoid/Locomotion/G1-Locomotion/checkpoint/stand/model_2999.pt
```

该文件来自多 GPU 服务器时可能带有 `cuda:1` tensor storage。训练脚本会在实验日志目录生成 CPU-portable 副本，不修改源模型，从而能在本机单卡 `cuda:0` 上加载。

默认完整训练为 4096 个环境、5000 次迭代：

```bash
QUIET_TERMINAL=False \
RUN_NAME=g1_extreme_stand_recovery_full_20260719 \
bash scripts/extreme_stand_recovery/train_g1_extreme_stand_recovery.sh
```

指定另一张 GPU：

```bash
DEVICE=cuda:1 QUIET_TERMINAL=False \
RUN_NAME=g1_extreme_stand_recovery_full_gpu1 \
bash scripts/extreme_stand_recovery/train_g1_extreme_stand_recovery.sh
```

换基础模型时，必须显式关闭当前固定 SHA 检查，或同步给出新 SHA：

```bash
BASE_CHECKPOINT=/absolute/path/to/model_x.pt \
VERIFY_BASE_SHA256=False \
RUN_NAME=g1_extreme_stand_recovery_from_model_x \
bash scripts/extreme_stand_recovery/train_g1_extreme_stand_recovery.sh
```

快速训练 smoke：

```bash
NUM_ENVS=32 MAX_ITERATIONS=1 HEADLESS=True QUIET_TERMINAL=False \
RUN_NAME=smoke_g1_extreme_stand_recovery \
bash scripts/extreme_stand_recovery/train_g1_extreme_stand_recovery.sh
```

输出位置：

```text
logs/rsl_rl/g1_extreme_stand_recovery/<时间_运行名>/
ExtremeStandRecovery Checkpoints/<时间_运行名>/
```

两个目录中的 `model_*.pt` 是同一训练阶段的日志 checkpoint 和专用副本；不要与 `ArmHack Checkpoints/StandPerturb` 混用。

### 5.1 本次完整训练最终模型

本次 5000 次迭代训练已经完成，最终模型不是旧的 ArmHack Stand `model_2999.pt`，而是：

```text
/home/user/Workspace/Humanoid/Locomotion/G1-Locomotion/legged_lab/ExtremeStandRecovery Checkpoints/2026-07-19_14-34-18_g1_extreme_stand_recovery_full_20260719_1433/model_4999.pt
```

SHA256：

```text
16af8b298fe4789194b6f798ee5591a3cc61edab307724a82906cc5e9a038fe7
```

### 5.2 从 `model_4999.pt` 进行默认姿态强化续训（Pose V2）

Pose V2 在上述最终 `model_4999.pt` 基础上以 policy-only 方式继续训练。默认使用 4096 个环境、3000 次迭代和 `3e-5` 学习率；基础模型路径和 SHA256 已在专用脚本中固定，防止误续训其他同名 checkpoint。每个新终端执行：

```bash
cd /home/user/Workspace/Humanoid/Locomotion/G1-Locomotion
conda activate env_isaaclab
cd legged_lab

QUIET_TERMINAL=False \
RUN_NAME=g1_extreme_stand_recovery_pose_v2_from_model4999_20260720 \
bash scripts/extreme_stand_recovery/train_g1_extreme_stand_recovery_pose_v2.sh
```

只做训练链 smoke：

```bash
NUM_ENVS=32 MAX_ITERATIONS=1 HEADLESS=True QUIET_TERMINAL=False \
RUN_NAME=smoke_g1_extreme_stand_pose_v2_from_model4999 \
bash scripts/extreme_stand_recovery/train_g1_extreme_stand_recovery_pose_v2.sh
```

2026-07-20 已完成一次真实的 32 环境、1 次 PPO 迭代 smoke。它成功载入原 `model_4999.pt`，Reward Manager 中显示全部 22 项奖励，其中新增/增强项为 `+5.0/+3.0/+2.5/-8.0`，完成 rollout、反向传播和 checkpoint 保存，产物为：

```text
logs/rsl_rl/g1_extreme_stand_recovery/2026-07-20_12-27-36_smoke_g1_extreme_stand_pose_v2_from_model4999_20260720_retry2/model_0.pt
ExtremeStandRecovery Checkpoints/2026-07-20_12-27-36_smoke_g1_extreme_stand_pose_v2_from_model4999_20260720_retry2/model_0.pt
```

smoke 只证明训练管线和新增奖励能够正确运行，不代表新策略已经收敛，也不能代替 Pose V2 完整训练后的 MuJoCo 与真机验收。

本次 Pose V2 完整续训已于 2026-07-20 完成 3000 次迭代，运行信息为：

```text
systemd 用户服务：g1-extreme-stand-pose-v2-20260720.service
训练日志：logs/monitoring/g1_extreme_stand_pose_v2_from_model4999_full_20260720.log
运行目录：logs/rsl_rl/g1_extreme_stand_recovery/2026-07-20_12-30-10_g1_extreme_stand_recovery_pose_v2_from_model4999_full_20260720/
checkpoint 目录：ExtremeStandRecovery Checkpoints/2026-07-20_12-30-10_g1_extreme_stand_recovery_pose_v2_from_model4999_full_20260720/
```

最终 checkpoint 和 SHA256：

```text
/home/user/Workspace/Humanoid/Locomotion/G1-Locomotion/legged_lab/ExtremeStandRecovery Checkpoints/2026-07-20_12-30-10_g1_extreme_stand_recovery_pose_v2_from_model4999_full_20260720/model_2999.pt
SHA256: ce7017ff810c5f24c533c1fac3b3fe8e539c712df8e64463076e557fb2df6264
```

运行目录与专用 checkpoint 目录中的 `model_2999.pt` 已校验为完全相同。最终训练日志为 Mean reward `97.29`、Mean episode length `956.71`、`time_out=95.00%`、`bad_orientation=5.03%`、`base_height=0.05%`。这些是训练分布内的最终 rollout 指标，不能替代下面独立的 MuJoCo 验收。

### 5.3 从 Pose V2 `model_2999.pt` 进行抗抖动续训（Pose V3）

Pose V3 专门处理多初始姿态 MuJoCo 长时测试发现的 25 Hz 两帧振荡，并进一步收紧默认几何姿态。它以 policy-only 方式载入 Pose V2 最终模型，固定校验以下基础 checkpoint：

```text
/home/user/Workspace/Humanoid/Locomotion/G1-Locomotion/legged_lab/ExtremeStandRecovery Checkpoints/2026-07-20_12-30-10_g1_extreme_stand_recovery_pose_v2_from_model4999_full_20260720/model_2999.pt
SHA256: ce7017ff810c5f24c533c1fac3b3fe8e539c712df8e64463076e557fb2df6264
```

完整续训默认使用 4096 个环境、3000 次迭代、`2e-5` 学习率：

```bash
cd /home/user/Workspace/Humanoid/Locomotion/G1-Locomotion
conda activate env_isaaclab
cd legged_lab

QUIET_TERMINAL=False \
RUN_NAME=g1_extreme_stand_recovery_anti_jitter_v3_from_pose_v2_model2999 \
bash scripts/extreme_stand_recovery/train_g1_extreme_stand_recovery_anti_jitter_v3.sh
```

只验证 Isaac/PPO/奖励链路：

```bash
cd /home/user/Workspace/Humanoid/Locomotion/G1-Locomotion/legged_lab
conda activate env_isaaclab

NUM_ENVS=8 MAX_ITERATIONS=1 HEADLESS=True QUIET_TERMINAL=False \
RUN_NAME=smoke_g1_extreme_stand_anti_jitter_v3 \
bash scripts/extreme_stand_recovery/train_g1_extreme_stand_recovery_anti_jitter_v3.sh
```

脚本允许显式覆盖 `JOINT_JERK_PENALTY`、`DEFAULT_CARTESIAN_POSE_WEIGHT`、`DEFAULT_CARTESIAN_POSE_VARIANCE`、`DEFAULT_FEET_DISTANCE_PENALTY`、`DEFAULT_FEET_GAUSSIAN_WEIGHT` 和 `DEFAULT_FEET_GAUSSIAN_VARIANCE`，并拒绝零值或负值。历史 Pose V2 启动器会显式关闭 V3 的三项新奖励，因此仍可复现旧实验，不会悄悄混入新目标。

2026-07-23 已完成两级真实 Isaac smoke：

1. `8 env × 1 iteration` 成功载入 Pose V2 `model_2999.pt`，Reward Manager 显示 25 项，其中 `default_key_body_pose_gaussian=+8.0`、`default_feet_distance_gaussian=+3.0`、`joint_jerk_l2=-1e-8`；完成 rollout、PPO 反向传播和 checkpoint 保存：

   ```text
   logs/rsl_rl/g1_extreme_stand_recovery/2026-07-23_10-50-06_smoke_g1_extreme_stand_anti_jitter_v3_20260723/model_0.pt
   ExtremeStandRecovery Checkpoints/2026-07-23_10-50-06_smoke_g1_extreme_stand_anti_jitter_v3_20260723/model_0.pt
   ```

2. `64 env × 45 iterations` 量级 smoke 覆盖了完整 20 秒 episode。最后一轮 Mean reward 为 `8.25`、Mean episode length 为 `561.62`，`Episode_Reward/joint_jerk_l2=-3.1970`、`default_feet_distance_gaussian=+0.2807`、`default_key_body_pose_gaussian=+0.0010`。这说明 jerk 项是有实际优化压力但没有让奖励数值溢出或阻止训练改善；关键点窄高斯对旧策略几乎不给分，符合“只有非常接近默认姿态才得到高奖励”的设计。产物为：

   ```text
   logs/rsl_rl/g1_extreme_stand_recovery/2026-07-23_10-50-50_smoke_scale_g1_extreme_stand_anti_jitter_v3_20260723/model_44.pt
   ExtremeStandRecovery Checkpoints/2026-07-23_10-50-50_smoke_scale_g1_extreme_stand_anti_jitter_v3_20260723/model_44.pt
   ```

这两个 smoke 只验证代码、奖励量级和 PPO 管线，不是可部署模型，也不能证明 25 Hz 振荡已经消失。正式 V3 训练完成后必须重新进行多初始姿态、每组至少 40 秒、排除前 10 秒恢复段的 MuJoCo 高频测试。

### 5.4 从 Anti-Jitter V3 `model_2999.pt` 进行平滑力矩续训（V4）

V4 固定从已完成的 V3 最终模型继续训练：

```text
/home/user/Workspace/Humanoid/Locomotion/G1-Locomotion/legged_lab/ExtremeStandRecovery Checkpoints/2026-07-24_10-33-45_g1_extreme_stand_recovery_anti_jitter_v3_resume1400_to2999_full_20260724/model_2999.pt
SHA256: e2c694d2d7710315f41f1c6c75849ffb95b53d0fb29e612aa211e1525a7cb1e4
```

完整续训命令：

```bash
cd /home/user/Workspace/Humanoid/Locomotion/G1-Locomotion
conda activate env_isaaclab
cd legged_lab

QUIET_TERMINAL=False \
RUN_NAME=g1_extreme_stand_recovery_smooth_torque_v4_from_v3_model2999 \
bash scripts/extreme_stand_recovery/train_g1_extreme_stand_recovery_smooth_torque_v4.sh
```

默认使用 `4096 env × 3000 iterations`、学习率 `1e-5`。新脚本会校验 V3 checkpoint 的 SHA256，并以 policy-only 方式载入，保留 96 维输入、29 维全身输出、`action_scale=0.25` 以及所有 V3 站立奖励。

只做最小 smoke：

```bash
cd /home/user/Workspace/Humanoid/Locomotion/G1-Locomotion/legged_lab
conda activate env_isaaclab

NUM_ENVS=4 MAX_ITERATIONS=1 HEADLESS=True QUIET_TERMINAL=False \
RUN_NAME=smoke_g1_extreme_stand_smooth_torque_v4 \
bash scripts/extreme_stand_recovery/train_g1_extreme_stand_recovery_smooth_torque_v4.sh
```

可通过同名环境变量覆盖六个平滑系数：`JOINT_TORQUE_PENALTY`、`JOINT_TORQUE_RATE_PENALTY`、`JOINT_ACCELERATION_PENALTY`、`JOINT_JERK_PENALTY`、`ACTION_RATE_PENALTY`、`ACTION_SECOND_DIFFERENCE_PENALTY`。脚本拒绝非正值，避免误启动一个没有对应抗抖约束的 V4 实验。

2026-07-31 已实际完成 `4 env × 1 iteration` smoke。Reward Manager 正确注册27项奖励，其中新增/增强项显示为：

```text
dof_torques_l2                 -1e-5
dof_acc_l2                     -5e-7
action_rate_l2                 -0.05
joint_jerk_l2                  -5e-8
action_second_difference_l2    -0.10
joint_torque_rate_l2           -2e-7
```

训练成功完成96个仿真步、rollout、PPO反向传播以及主目录/专用目录双份 checkpoint 保存，二者 SHA256 相同：

```text
logs/rsl_rl/g1_extreme_stand_recovery/2026-07-31_16-20-32_smoke_g1_extreme_stand_smooth_torque_v4_20260731/model_0.pt
ExtremeStandRecovery Checkpoints/2026-07-31_16-20-32_smoke_g1_extreme_stand_smooth_torque_v4_20260731/model_0.pt
SHA256: cb6db485b88c9531ca905faf5420983d92d8e4c55acf5ad391a8bcde54f8dfd1
```

一次迭代不会结束完整20秒 episode，因此 TensorBoard 的 episode reward 项为0是预期行为；该 smoke 证明训练链路和两个有状态奖励无报错，不代表新策略已学会抗抖或可以部署。

### 5.5 Smooth-Settle V5：恢复后主动刹停

V5 是独立任务 `LeggedLab-Isaac-AMP-G1-ExtremeStandRecovery-SmoothSettleV5-v0`，不会改变 V4、Walk 或 ArmHack。它固定从已完成的 V4 最终模型续训：

```text
/home/user/Workspace/Humanoid/Locomotion/G1-Locomotion/legged_lab/ExtremeStandRecovery Checkpoints/2026-07-31_16-52-27_g1_extreme_stand_recovery_smooth_torque_v4_from_v3_model2999_full_20260731/model_2999.pt
SHA256: e0addb8ce23153498d4f805c75f4e3ba19568198f890ffc980160fea7c3b7fff
```

核心代码：

- 任务配置：`legged_lab/source/legged_lab/legged_lab/tasks/locomotion/amp/config/g1_extreme_stand_recovery/g1_extreme_stand_recovery_v5_env_cfg.py`；
- 奖励实现：同目录 `rewards.py`；
- 互斥扰动状态机：同目录 `disturbances.py`；
- 训练入口：`legged_lab/scripts/extreme_stand_recovery/train_g1_extreme_stand_recovery_smooth_settle_v5.sh`。

V5 新约束分为四组：

1. `action_l2` 直接压低策略输出；`target_q_default_error_l2`、`target_q_velocity_l2`、`target_q_acceleration_l2` 分别约束实际 PD 目标相对默认角度的偏差、速度和加速度，避免“实际关节接近默认，但目标角仍缓慢来回摆动”。
2. jerk、力矩和力矩变化率不再对29关节简单取均值，而是取归一化后最差4个关节的均值。jerk 使用腿/腰/臂分组尺度，力矩使用每个关节自身 effort limit；`soft_peak_joint_torque_topk_l2` 从限位的60%开始惩罚，因此未饱和但明显偏大的髋/膝力矩也有梯度。
3. `joint_velocity_l2`、`joint_acceleration_l2`、`mechanical_power_l2=(tau*qdot)^2` 约束速度、加速度和机械功率。`near_default_settle_penalty` 使用 `exp(-mean((q-default_q)^2)/0.01)` 门控：偏离较大时允许快速恢复，接近默认姿态后按 `5×关节速度 + 5×action + 3×归一化力矩跳变 + 3×Top-K jerk` 强制刹停。
4. 五个独立扰动时钟全部关闭，改为每个环境一个状态机。每次只选一个 body 和一个方向，持续 `0.1–0.3 s`，随后保证 `6–10 s` 不再施力；右推和后推概率为 `0.35/0.30`，高于左推和前推的 `0.20/0.15`。1500次续训的外力课程为 `0–299: 10 N`、`300–599: 20 N`、`600–999: 36 N`、`1000–1499: 45 N`。`post_disturbance_pose_recovery` 与 `post_disturbance_stillness` 只在受力后的安静窗口奖励恢复速度和静止保持。

完整训练命令：

```bash
cd /home/user/Workspace/Humanoid/Locomotion/G1-Locomotion
conda activate env_isaaclab

QUIET_TERMINAL=False \
RUN_NAME=g1_extreme_stand_recovery_smooth_settle_v5_from_v4_model2999 \
bash legged_lab/scripts/extreme_stand_recovery/train_g1_extreme_stand_recovery_smooth_settle_v5.sh
```

默认是 `4096 env × 1500 iterations`、学习率 `1e-5`、20秒 episode，每100次迭代保存一次 checkpoint。新终端只需激活 `env_isaaclab`；脚本会验证基础 checkpoint 的 SHA256，并以 policy-only 方式载入，输入输出仍为 `96 -> 29`。

2026-08-06 启动的正式1500轮训练位置：

```text
服务：g1-extreme-stand-smooth-settle-v5-1500-20260806.service
训练日志：legged_lab/logs/monitoring/g1_extreme_stand_smooth_settle_v5_from_v4_model2999_1500iter_20260806.log
运行目录：legged_lab/logs/rsl_rl/g1_extreme_stand_recovery/2026-08-06_20-23-31_g1_extreme_stand_smooth_settle_v5_from_v4_model2999_1500iter_20260806
checkpoint：legged_lab/ExtremeStandRecovery Checkpoints/2026-08-06_20-23-31_g1_extreme_stand_smooth_settle_v5_from_v4_model2999_1500iter_20260806
TensorBoard：http://127.0.0.1:6008/
```

最小 smoke：

```bash
cd /home/user/Workspace/Humanoid/Locomotion/G1-Locomotion
conda activate env_isaaclab

NUM_ENVS=32 MAX_ITERATIONS=2 HEADLESS=True QUIET_TERMINAL=False \
RUN_NAME=smoke_g1_extreme_stand_smooth_settle_v5 \
bash legged_lab/scripts/extreme_stand_recovery/train_g1_extreme_stand_recovery_smooth_settle_v5.sh
```

2026-08-06 已实际完成两次 `32 env × 2 iterations` smoke。第一次验证全部37项奖励、rollout、PPO反向传播和 checkpoint 保存；第二次把初始静置临时设为0，实际覆盖单扰动施力与清零路径，日志中 `disturbance_force_n=3.5417`、`disturbance_active_fraction=0.3542`，无 NaN、Inf、CUDA、Isaac 或状态缓存错误。随后又以最终命名的 `joint_acceleration_l2` 完成 `4 env × 1 iteration` 回归，Reward Manager 正确显示该项并完成训练。施力 smoke 最终产物为：

```text
logs/rsl_rl/g1_extreme_stand_recovery/2026-08-06_17-18-20_g1_extreme_stand_v5_force_smoke_20260806/model_1.pt
ExtremeStandRecovery Checkpoints/2026-08-06_17-18-20_g1_extreme_stand_v5_force_smoke_20260806/model_1.pt
SHA256: fc6e8f7096a3a1a97ae273d9b11462f1f09455e5d393ca87b11384c54ccefc78
```

smoke 模型不是可部署模型；必须完成1500次正式续训，再用同一批大推力、随机脚距和长期稳定场景与 V4 做同 seed 对比。建议重点对比 `model_300/600/1000/1200/1499.pt`，用于识别平滑约束是否造成恢复能力退化。

## 6. TensorBoard

在另一个新终端中：

```bash
cd /home/user/Workspace/Humanoid/Locomotion/G1-Locomotion/legged_lab
conda activate env_isaaclab
tensorboard --logdir logs/rsl_rl/g1_extreme_stand_recovery --port 6008 --bind_all
```

本机打开：

```text
http://127.0.0.1:6008/
```

Pose V3 重点观察 episode length、`termination_penalty`、`time_out/base_height/bad_orientation`、`default_joint_pose_exp`、`default_leg_joint_pose_exp`、`default_key_body_pose_gaussian`、`default_feet_distance_l2`、`default_feet_distance_gaussian`、`joint_jerk_l2`、躯干速度、roll/pitch、高度、脚滑和总 reward。`default_feet_distance_l2` 是已乘负权重的 episode reward，越接近 0 表示足间距偏差越小；两个窄高斯项越高表示越接近默认几何姿态；`joint_jerk_l2` 是负项，绝对值应随训练下降。仅看总 reward 不能证明抗扰恢复成功，也不能替代 40 秒以上的多初始姿态长期高频分析。

V4 还必须同时观察 `dof_torques_l2`、`joint_torque_rate_l2`、`dof_acc_l2`、`action_rate_l2` 和 `action_second_difference_l2`。这些都是已乘负权重的 episode reward，绝对值下降通常表示更平滑，但若同时出现 episode length 下降或 `bad_orientation/base_height` 上升，说明平滑惩罚过强、恢复动作被压制，不能仅因曲线更接近0就认为训练更好。

V5 重点观察 `target_q_default_error_l2`、`target_q_velocity_l2`、`target_q_acceleration_l2`、三项 `normalized_*_topk_l2`、`soft_peak_joint_torque_topk_l2`、`joint_velocity_l2`、`joint_acceleration_l2`、`mechanical_power_l2` 和 `near_default_settle_penalty`。同时检查 `ExtremeStand/disturbance_stage`、`disturbance_force_n`、`disturbance_active_fraction` 是否符合课程。平滑负项下降时，`post_disturbance_pose_recovery`、`post_disturbance_stillness`、episode length 和 time-out 率不应同步恶化；否则说明刹车权重压制了必要恢复动作。

## 7. Isaac Sim 测试与可视化

每个新终端都要重新定义 checkpoint 变量。本次最终模型可直接这样定义：

```bash
cd /home/user/Workspace/Humanoid/Locomotion/G1-Locomotion/legged_lab
conda activate env_isaaclab

EXTREME_STAND_CKPT="$PWD/ExtremeStandRecovery Checkpoints/2026-07-20_12-30-10_g1_extreme_stand_recovery_pose_v2_from_model4999_full_20260720/model_2999.pt"
test -f "$EXTREME_STAND_CKPT"
```

三种模式：

```bash
# 1. 默认姿态、无随机化、无外力：检查基本站立
CHECKPOINT="$EXTREME_STAND_CKPT" MODE=nominal \
bash scripts/extreme_stand_recovery/vis_g1_extreme_stand_recovery.sh

# 2. 随机初始关节/基座状态，但 episode 内不继续推：检查恢复能力
CHECKPOINT="$EXTREME_STAND_CKPT" MODE=recovery \
bash scripts/extreme_stand_recovery/vis_g1_extreme_stand_recovery.sh

# 3. 随机初始状态 + 完整动力学随机化 + 多部位外力：检查持续抗扰
CHECKPOINT="$EXTREME_STAND_CKPT" MODE=robust \
bash scripts/extreme_stand_recovery/vis_g1_extreme_stand_recovery.sh
```

无窗口的 500-step 自动回放：

```bash
CHECKPOINT="$EXTREME_STAND_CKPT" MODE=robust \
HEADLESS=True MAX_STEPS=500 \
bash scripts/extreme_stand_recovery/vis_g1_extreme_stand_recovery.sh
```

## 8. Pose V2 最终模型导出

项目根目录的专用导出器默认锁定上述 Pose V2 `model_2999.pt` 及其 SHA256。导出产物同时包含 TorchScript 和 ONNX；当前 Extreme Stand 的 MuJoCo 与真机入口统一使用复制到 `use/extreme_stand_recovery_pose_v2_model2999.onnx` 的 ONNX，TorchScript 保留用于兼容和数值对照。每个新终端执行：

```bash
cd /home/user/Workspace/Humanoid/Locomotion/G1-Locomotion
bash scripts/export_g1_extreme_stand_recovery.sh
```

脚本内部明确使用 `env_isaaclab` 的 Python，不要求当前终端预先 `conda activate`。输出目录为：

```text
legged_lab/ExtremeStandRecovery Checkpoints/2026-07-20_12-30-10_g1_extreme_stand_recovery_pose_v2_from_model4999_full_20260720/exported_extreme_stand_recovery/
├── policy.pt
├── policy.onnx
└── policy.deploy.json
```

已验证的导出结果：

| 产物 | SHA256 |
| --- | --- |
| `policy.pt` | `0091c9939f5a43f754dbb87f56648560d547cf8b9bfc7f8852d2ed44a0791d71` |
| `policy.onnx` | `0af2ffb24cd728352804b62669dc5264dd835329528311f9d50b90dbe2d0a0d1` |
| `policy.deploy.json` | `2bf0f21c511463b19bd8a1ef1f77122cc43cee41560bfb398e3b06ba00164fd7` |

ONNX 输入为 `[1, 96]`，输出为 `[1, 29]`；ONNX Runtime 与 TorchScript 在同一测试输入上的最大绝对误差为 `6.91413879e-06`，低于 `1e-05` 门槛。速度命令元数据固定为 `[0, 0, 0]`。

## 9. MuJoCo 全面 sim2sim 测试与可视化

专用入口复用 `scripts/sim2sim_g1_amp_mujoco.sh` 的模型、关节映射、PD 和指标链，但固定使用最终 checkpoint、`s3_g1_29dof`、零速度指令及完整 29 维 Actor 输出。初始随机关节会先按 MJCF 的实际关节范围裁剪，并预留 `0.02 rad` margin；裁剪次数写入报告。

一个交互可视化档位和七个固定测试档位为：

| `PROFILE` | 初始状态 | episode 内外力 | 用途 |
| --- | --- | --- | --- |
| `interactive` | 启动时为默认姿态；空格依次循环四向躯干大推力、随机29关节姿态、随机脚距和默认姿态；`K` 可单独生成随机脚距初值 | 空格的前四档固定对 `torso_link` 施加大推力；按 `F` 独立开关 `±35 N`、`±5 Nm` 随机脉冲 | 一次窗口内交互检查大推力振铃、姿态恢复、足距恢复与随机抗扰 |
| `nominal` | 默认姿态、零速度 | 无 | 基本站立 |
| `pose_recovery` | 只随机化29个关节姿态；root 与全部速度保持默认 | 无 | 隔离检查能否恢复默认全身姿态 |
| `feet_distance_recovery` | 只随机化双脚平面距离，保证相对默认值偏差 `5–12 cm`，其他初值为默认 | 无 | 隔离检查能否恢复资产默认双脚距离 |
| `recovery` | 训练范围内全身关节、root 姿态/速度噪声 | 无 | 只检查受扰初值恢复 |
| `robust` | 同 `recovery` | `±35 N`、`±5 Nm`，每 `2.5 s` 一次 | 训练同级持续抗扰 |
| `stress` | 比训练更大的噪声 | `±50 N`、`±8 Nm`，每 `2.0 s` 一次 | 超训练分布压力测试，不计入基础验收 |
| `large_push` | 默认姿态并先稳定 `5 s` | 在第 `5 s` 固定对 `torso_link` 施加一次水平大推力，默认 `120 N × 0.20 s`，可覆盖到 `360 N` | 隔离检查推力后 actor、PD 力矩和关节振动，自动生成诊断图 |

### 9.1 打开 MuJoCo GUI

脚本现在默认锁定 Smooth-Torque V4 最终 `model_2999.pt`，新终端无需再手工填写 checkpoint。推荐使用统一交互入口：

```bash
cd /home/user/Workspace/Humanoid/Locomotion/G1-Locomotion

PROFILE=interactive USE_GLFW=True REAL_TIME=True \
RENDER_FPS=60 REALTIME_STATUS_INTERVAL_S=5 \
FOLLOW_CAMERA=False \
INTERACTIVE_DATA_LOG=True \
SIMULATION_DURATION=300 SEED=20260722 \
bash scripts/sim2sim_g1_extreme_stand_recovery_mujoco.sh
```

只在 MuJoCo 侧实验“关节位置目标速度/加速度限幅”（不改训练和真机）时使用：

```bash
cd /home/user/Workspace/Humanoid/Locomotion/G1-Locomotion

UNITREE_PYTHON="$HOME/anaconda3/envs/gmr/bin/python" \
PROFILE=interactive USE_GLFW=True REAL_TIME=True \
RENDER_FPS=60 FOLLOW_CAMERA=False INTERACTIVE_DATA_LOG=True \
TARGET_LIMITER_ENABLE=True \
TARGET_LEG_VELOCITY_LIMIT_RAD_S=25.0 \
TARGET_WAIST_VELOCITY_LIMIT_RAD_S=10.0 \
TARGET_ARM_VELOCITY_LIMIT_RAD_S=15.0 \
TARGET_LEG_ACCELERATION_LIMIT_RAD_S2=600.0 \
TARGET_WAIST_ACCELERATION_LIMIT_RAD_S2=250.0 \
TARGET_ARM_ACCELERATION_LIMIT_RAD_S2=400.0 \
LARGE_PUSH_FORCE_N=36 LARGE_PUSH_DURATION_S=0.20 \
SIMULATION_DURATION=300 SEED=20260805 \
RESULTS_ROOT="$PWD/legged_lab/logs/monitoring/extreme_stand_v4_target_limiter" \
bash scripts/sim2sim_g1_extreme_stand_recovery_mujoco.sh
```

限幅器在 `50 Hz` 控制周期工作，并在姿态 reset 后从当前关节角、零目标速度重新初始化。该实验保留 Actor 的 29 维原始输出，只平滑 `default_joint_pos + action_scale × action` 形成的 PD 目标。开关默认关闭，便于与原始策略做 A/B 对照。当前 `25/10/15 rad/s` 与 `600/250/400 rad/s²` 是只裁极端尖峰的起点；更严格的 `6/4/4 rad/s` 与 `120/80/80 rad/s²` 已在 36 N 后推测试中导致摔倒，不能用于可视化结论。

实时可视化采用物理 `500 Hz`、策略 `50 Hz`、GUI 默认 `60 FPS` 的解耦频率。`REAL_TIME=True` 使用累计墙钟时限速并允许渲染超时后追赶，终端周期打印 `RTF`；`RTF≈1.000` 表示真实 1× 时间。若 `RTF` 长期低于 `0.95`，将 `RENDER_FPS=30` 降低图形负载。

交互入口默认 `FOLLOW_CAMERA=False`。启动时会设置一次初始视角，之后不会再覆盖 `viewer.cam`，因此可用 MuJoCo 鼠标操作自由旋转、平移和缩放。运行时按 `C` 可切换自由/跟随；跟随模式只更新观察中心，不覆盖鼠标调整的方位角、俯仰角和缩放距离。

窗口获得焦点后：

- 每按一次空格依次循环七个场景：`torso +X 前推`、`torso -X 后推`、`torso +Y 左推`、`torso -Y 右推`、随机29关节初始姿态、随机双脚间距初始姿态、默认初始姿态，然后重新循环。前四次不会重置策略或瞬移机器人，而是对当前状态直接施加默认 `120 N × 0.20 s` 的水平躯干推力；后三次会重置仿真并清零策略上一动作。可用 `LARGE_PUSH_FORCE_N` 和 `LARGE_PUSH_DURATION_S` 调整强度。
- `R` 随时立即把仿真恢复到默认站立姿态，并清零关节/根节点速度、当前外力和策略上一动作；机器人已经摔倒时也可手动复位。该操作是仿真 reset，不计作策略自主起身能力。
- `C` 在自由相机与跟随相机之间切换。两种模式都允许鼠标旋转和滚轮缩放；自由模式还不会自动移动观察中心。
- 每按一次 `K`：重置到默认状态，再对左右 hip-roll 施加反向偏置、对 ankle-roll 作对应补偿，从 MuJoCo 正向运动学可达集合中随机选择一个实际脚距偏差为 `5–12 cm` 的初值。该键不会叠加全身关节、root 或速度噪声，因此只测试脚距恢复；终端会打印默认脚距、随机初始脚距、偏差和 roll offset。
- 按一次 `F`：开启随机多部位脉冲外力，并立即施加第一下；之后每 `2.5 s` 施加一次、每次持续 `0.25 s`；再次按 `F` 立即关闭并清零外力。
- 三个按键互相独立。空格循环到姿态重置档或按 `K` 时会把策略上一动作清零，保证新 episode 的96维观测合同正确；空格循环到大推力档时不会清零策略状态，以便真实观察受推后的闭环恢复。它不会覆盖 Actor 输出。终端会打印每次切换状态，最终 `metrics.json` 也记录按键事件。

外力并不是只施加在脚上。当前 `F` 键随机候选 body 明确为：`pelvis`、`torso_link`、左右 `shoulder_pitch_link`、左右 `elbow_link`、左右 `hip_pitch_link`、左右 `knee_link`，不包含脚或踝。每个事件只从这十个 body 中随机选择一个，通过 MuJoCo 的 `data.xfrc_applied[body_id]` 写入世界坐标系三轴力和三轴力矩。

交互模式默认以 `50 Hz` 流式保存震荡诊断表。`interactive_diagnostics_all.csv` 是完整总表；每按一次空格会在 `space_trials/` 新建一个按顺序编号和场景命名的独立 CSV；`interactive_events.csv` 保存全部 `SPACE/R/K/F/C` 操作时间。逐帧列覆盖 29 维 actor 输出、未经限幅/实际送入 PD 的两套目标角、限幅器目标速度/加速度与每帧 clip 数量、实际关节角、速度、有限差分与 MuJoCo 加速度、jerk、PD/实际力矩、力矩限位、root 状态、29 个关节轴锚点的世界系 XYZ、左右脚世界系 XYZ/平面距离/三维距离/默认脚距误差、左右脚世界系 6D 地面反力、当前外部 6D wrench，以及当前/目标速度指令。每行立即 flush，便于窗口异常退出后保留已采数据。

这里要区分两类量：广义坐标 `qpos` 表示根节点世界位姿和各关节角，并不等于各关节的世界系笛卡尔位置。各关节世界坐标单独保存为 `joint_anchor_world_m/<joint_name>/{x,y,z}`。双脚 XY 平面距离已直接保存为 `feet/planar_distance_m`，它等于 `sqrt((x_right-x_left)^2 + (y_right-y_left)^2)`；三维距离为 `feet/distance_3d_m`。

为避免把两类箭头混淆，界面和报告现在按下面规则解释：

- **洋红色箭头**：本项目额外绘制的、当前真实写入 `xfrc_applied` 的随机外力；箭头起点就是被选中的 body，方向与注入力方向相同。力矩仍记录在报告和终端，不单独画旋转箭头。
- **脚底附近的 MuJoCo 原生箭头**：足底与地面的接触力，不是 `F` 键注入位置。此前没有为 `xfrc_applied` 画独立箭头，所以视觉上容易误判成“只在足部施力”。
- 每次外力开始时，终端打印 `[Extreme Stand wrench]`、时间、`body_name`、`force_world_n` 和 `torque_world_nm`；最终 `metrics.json` 的 `extreme_stand_recovery.wrench.events` 保存同样的逐事件记录，`body_event_counts` 汇总每个 body 被抽中的次数。

`interactive` 默认从“默认姿态、外力关闭”开始。若希望启动时就是随机姿态或已开启外力，可显式设置：

```bash
cd /home/user/Workspace/Humanoid/Locomotion/G1-Locomotion

PROFILE=interactive USE_GLFW=True REAL_TIME=True \
INTERACTIVE_POSE_START_RANDOM=True \
INTERACTIVE_WRENCH_START_ENABLED=True \
SIMULATION_DURATION=300 SEED=20260722 \
bash scripts/sim2sim_g1_extreme_stand_recovery_mujoco.sh
```

固定场景可使用 `nominal`、`pose_recovery`、`feet_distance_recovery`、`recovery`、`robust`、`stress` 或 `large_push`；固定场景用于可重复报告，不响应交互按键。`stress` 和 `large_push` 只用于仿真，不能照搬到真机。本机 MuJoCo/GLFW 偶尔会在窗口已经完成、JSON 已落盘后的 native viewer 关闭阶段返回 `139`；专用入口仅在 `USE_GLFW=True`、退出码确为 `139` 且完整 `metrics.json` 已存在时将其标记为 viewer-shutdown warning，仿真中途错误仍会失败。

从 `legged_lab` 目录也可以使用兼容入口，结果和根目录脚本相同：

```bash
cd /home/user/Workspace/Humanoid/Locomotion/G1-Locomotion/legged_lab
PROFILE=interactive USE_GLFW=True REAL_TIME=True SIMULATION_DURATION=300 \
bash scripts/extreme_stand_recovery/val_mujoco_g1_extreme_stand_recovery.sh
```

### 9.2 单项无窗口测试

```bash
cd /home/user/Workspace/Humanoid/Locomotion/G1-Locomotion

PROFILE=recovery USE_GLFW=False REAL_TIME=False \
SIMULATION_DURATION=30 SEED=20260721 \
bash scripts/sim2sim_g1_extreme_stand_recovery_mujoco.sh
```

### 9.3 七档多种子完整测试

```bash
cd /home/user/Workspace/Humanoid/Locomotion/G1-Locomotion

SUITE=True USE_GLFW=False REAL_TIME=False \
SUITE_DURATION=12 \
SUITE_SEEDS=20260719,20260720,20260721 \
SUITE_RESULTS_ROOT="$PWD/legged_lab/ExtremeStandRecovery Checkpoints/2026-07-24_10-33-45_g1_extreme_stand_recovery_anti_jitter_v3_resume1400_to2999_full_20260724/exported_extreme_stand_recovery/mujoco_tests/suite_manual" \
REQUIRE_PASS=True \
bash scripts/sim2sim_g1_extreme_stand_recovery_mujoco.sh
```

基础验收要求：`nominal`、`recovery` 全部健康，`robust` 至少 `2/3` 健康；`feet_distance_recovery` 必须实际生成至少 `5 cm` 脚距偏差，且至少 `2/3` 测试在最终窗口持续进入默认距离 `±2 cm`；这些场景的速度指令必须始终为零，且 `extreme_stand_recovery.action_override=false`。`stress` 只报告额外余量，不代表真机允许施加同等扰动。

每次运行保存 `metrics.json`、`torso_trace.csv`；完整套件另外生成 `summary.json` 和中文 `REPORT.md`。

### 9.4 Pose V2 `model_2999.pt` 代表性 smoke

报告路径：

```text
/home/user/Workspace/Humanoid/Locomotion/G1-Locomotion/legged_lab/ExtremeStandRecovery Checkpoints/2026-07-20_12-30-10_g1_extreme_stand_recovery_pose_v2_from_model4999_full_20260720/exported_extreme_stand_recovery/mujoco_tests/smoke_pose_v2_20260721/REPORT.md
```

2026-07-21 使用同一个 seed 对 `nominal`、`pose_recovery`、`recovery`、`robust` 各跑 8 秒。四次均未摔倒，速度指令始终为零，Actor 29 维输出未被覆盖；`robust` 实际施加了 3 次外力。该组测试只是一套快速回归，不替代三 seed 完整套件。

| 场景 | 健康 | 水平速度 MAE | yaw-rate MAE | 总分 | 外力次数 |
| --- | --- | ---: | ---: | ---: | ---: |
| nominal | 是 | `0.0068 m/s` | `0.0066 rad/s` | `95.72` | 0 |
| pose_recovery | 是 | `0.0495 m/s` | `0.1467 rad/s` | `87.33` | 0 |
| recovery | 是 | `0.0334 m/s` | `0.1678 rad/s` | `87.34` | 0 |
| robust | 是 | `0.0522 m/s` | `0.1884 rad/s` | `86.04` | 3 |

整组汇总显示 `Acceptance: FAIL`，原因不是摔倒，而是严格 `pose_recovery` 的单关节最大误差仍超过 `0.20 rad`。这一区分必须保留：稳定站立/存活已经通过，严格恢复全部默认关节姿态尚未通过。

### 9.5 随机全身姿态恢复专项测试

该测试与原 `recovery` 不同：只给29个关节加入训练范围内的随机位置偏差，关节速度、root 姿态、root 速度和外力全部为零，避免其他扰动掩盖“能否回到默认关节姿态”这一问题。

默认随机范围为腿 `±0.25 rad`、腰 `±0.35 rad`、双臂 `±0.60 rad`，初值按 MJCF 关节限位和 `0.02 rad` margin 裁剪。严格恢复判据为：最后1秒内全身关节 MAE 不超过 `0.12 rad`，同时任意单关节最大误差不超过 `0.20 rad`；只有两个条件都满足才记为恢复成功。

五个随机种子的无窗口测试：

```bash
cd /home/user/Workspace/Humanoid/Locomotion/G1-Locomotion

SEEDS=20260722,20260723,20260724,20260725,20260726 \
DURATION=15 USE_GLFW=False REQUIRE_PASS=False \
bash scripts/test_g1_extreme_stand_random_pose_recovery_mujoco.sh
```

若要把恢复失败作为 CI 非零退出门槛，设置 `REQUIRE_PASS=True`。GUI 可视化一个随机姿态：

```bash
cd /home/user/Workspace/Humanoid/Locomotion/G1-Locomotion

SEEDS=20260722 DURATION=30 USE_GLFW=True \
bash scripts/test_g1_extreme_stand_random_pose_recovery_mujoco.sh
```

Pose V2 `model_2999.pt` 的 5-seed、每次 15 秒专项报告：

```text
/home/user/Workspace/Humanoid/Locomotion/G1-Locomotion/legged_lab/ExtremeStandRecovery Checkpoints/2026-07-20_12-30-10_g1_extreme_stand_recovery_pose_v2_from_model4999_full_20260720/exported_extreme_stand_recovery/mujoco_tests/random_pose_recovery_pose_v2_model2999_20260721/REPORT.md
```

实测 5/5 均未摔倒，但严格默认姿态恢复仍为 0/5，因此专项验收未通过。初始关节 MAE 平均为 `0.2020 rad`，最后 1 秒降到 `0.0634 rad`，误差平均下降 `68.1%`；全身平均误差已经低于 `0.12 rad` 门槛，但单关节平均最大误差为 `0.2664 rad`，超过 `0.20 rad` 门槛。主要残余从旧模型的踝/膝转移到双侧 shoulder-pitch（平均约 `0.264/0.256 rad`），其次为双肘（约 `0.150/0.148 rad`）。

与旧 `model_4999.pt` 在同一组 seed 上的结果相比，最终 MAE 从 `0.1067 rad` 降至 `0.0634 rad`，平均误差下降比例从 `46.5%` 提升至 `68.1%`，说明新增腿部、笛卡尔和足间距奖励确实改善了默认姿态恢复；但严格全身姿态门槛仍未完全通过，不能把 5/5 存活表述成 5/5 恢复成功。

### 9.6 随机双脚间距恢复专项测试

该场景只改变左右 hip-roll 与 ankle-roll 的对称组合，不加入其他关节噪声、root 噪声、初始速度或外力。每次初始化先在实际 MJCF 关节限位内搜索，再随机选取一个使左右足 body 平面距离相对默认值偏差 `5–12 cm` 的可达构型；同时微调 root 高度，使较低一侧足 body 保持默认初始高度。它不会直接平移脚，也不会修改策略的 29 维输出。

GUI 中每按一次 `K` 都重新采样一次：

```bash
cd /home/user/Workspace/Humanoid/Locomotion/G1-Locomotion

PROFILE=interactive USE_GLFW=True REAL_TIME=True \
SIMULATION_DURATION=300 \
bash scripts/sim2sim_g1_extreme_stand_recovery_mujoco.sh
```

固定 seed、无窗口复现：

```bash
cd /home/user/Workspace/Humanoid/Locomotion/G1-Locomotion

PROFILE=feet_distance_recovery USE_GLFW=False REAL_TIME=False \
SIMULATION_DURATION=12 STEADY_START_S=5 SEED=20260727 \
bash scripts/sim2sim_g1_extreme_stand_recovery_mujoco.sh
```

三 seed 专项报告：

```bash
cd /home/user/Workspace/Humanoid/Locomotion/G1-Locomotion

SUITE=True USE_GLFW=False REAL_TIME=False \
SUITE_PROFILES=feet_distance_recovery \
SUITE_SEEDS=20260727,20260728,20260729 \
SUITE_DURATION=12 STEADY_START_S=5 \
REQUIRE_PASS=False \
SUITE_RESULTS_ROOT="$PWD/legged_lab/ExtremeStandRecovery Checkpoints/2026-07-24_10-33-45_g1_extreme_stand_recovery_anti_jitter_v3_resume1400_to2999_full_20260724/exported_extreme_stand_recovery/mujoco_tests/feet_distance_recovery_3seed_20260727" \
bash scripts/sim2sim_g1_extreme_stand_recovery_mujoco.sh
```

专项判据是：确实施加至少 `5 cm` 初始脚距偏差，随后在最终 `1 s` 窗口内所有样本都位于默认脚距 `±2 cm`，并且至少持续 `1 s` 进入该误差带。报告同时保存默认/初始脚距、初始偏差、最终 MAE/最大误差、首次恢复时间和逐帧 `motion_quality_trace.csv`。

2026-07-27 对 Anti-Jitter V3 `model_2999.pt` 的实测结果为：3/3 均保持站立，但只有 1/3 恢复默认脚距，因此专项验收失败。资产默认脚距为 `0.2370 m`：

| seed | 初始脚距 | 初始偏差 | 最终脚距误差 | 恢复 |
| ---: | ---: | ---: | ---: | --- |
| 20260727 | `0.2955 m` | `+5.85 cm` | 约 `+1.48 cm` | 是，首次持续进入误差带约 `0.742 s` |
| 20260728 | `0.1260 m` | `-11.10 cm` | 约 `+4.35 cm` | 否 |
| 20260729 | `0.1435 m` | `-9.35 cm` | 约 `+3.94 cm` | 否 |

完整报告：

```text
/home/user/Workspace/Humanoid/Locomotion/G1-Locomotion/legged_lab/ExtremeStandRecovery Checkpoints/2026-07-24_10-33-45_g1_extreme_stand_recovery_anti_jitter_v3_resume1400_to2999_full_20260724/exported_extreme_stand_recovery/mujoco_tests/feet_distance_recovery_3seed_20260727/REPORT.md
```

结论是当前模型能够从较宽脚距恢复到默认值附近，但对较窄脚距会越过默认值并最终稳定在约 `0.276–0.281 m` 的偏宽站姿；“不摔倒”不能等同于“恢复默认脚距”。这与训练末期 `default_key_body_pose_gaussian=0` 的现象一致，后续训练仍需改善默认几何姿态吸引域。

### 9.7 躯干大推力与持续 jerk 专项测试

`large_push` 用于隔离真机“受到较大推力后全身持续 jerk”的问题。测试先让默认站立稳定 `5 s`，再只对 `torso_link` 施加一次世界系水平力；不随机其他 body、不加力矩、不改策略输出。运行器同时记录29关节位置/速度/加速度/jerk、29维 actor action、关节目标、PD 力矩命令、实际执行器力矩和力矩限位。

可视化并在第5秒自动侧推：

```bash
cd /home/user/Workspace/Humanoid/Locomotion/G1-Locomotion

PROFILE=large_push USE_GLFW=True REAL_TIME=True \
SIMULATION_DURATION=30 STEADY_START_S=7.2 \
LARGE_PUSH_FORCE_N=360 LARGE_PUSH_DURATION_S=0.20 \
LARGE_PUSH_TIME_S=5 LARGE_PUSH_DIRECTION_INDEX=2 \
RESULTS_ROOT="$PWD/legged_lab/logs/monitoring/extreme_stand_large_push_visual" \
bash scripts/sim2sim_g1_extreme_stand_recovery_mujoco.sh
```

方向编号为：`0=前推(+X)`、`1=后推(-X)`、`2=左推(+Y)`、`3=右推(-Y)`；设为 `-1` 时按 seed 随机选一个水平轴向。默认固定场景是 `120 N × 0.20 s`，而上面的 `360 N × 0.20 s` 是用于复现问题的超分布仿真诊断，不能用于真机。

交互测试则使用：

```bash
cd /home/user/Workspace/Humanoid/Locomotion/G1-Locomotion

PROFILE=interactive USE_GLFW=True REAL_TIME=True \
SIMULATION_DURATION=300 \
LARGE_PUSH_FORCE_N=360 LARGE_PUSH_DURATION_S=0.20 \
RESULTS_ROOT="$PWD/legged_lab/logs/monitoring/extreme_stand_large_push_interactive" \
bash scripts/sim2sim_g1_extreme_stand_recovery_mujoco.sh
```

空格依次循环前推、后推、左推、右推、随机全身关节姿态、随机脚距和默认姿态。大推力事件显示为从 `torso_link` 出发的洋红色箭头；`K` 仍可随时单独随机脚距，`F` 仍独立开关小幅随机多 body 外力。

每次 `large_push` 或 `interactive` 运行额外保存：

- `motion_quality_trace.csv`：逐控制帧的关节、actor action、目标位置、PD 力矩和实际执行器力矩；
- `large_push_diagnostics.png`：推力区间、关节速度/jerk、actor action 变化率、PD 力矩/饱和率及高频最大的关节曲线；
- `metrics.json` 中的 `extreme_stand_recovery.large_push.post_push_diagnostics`：推力前2秒、恢复段以及最终5秒的8–25 Hz频带对比和诊断标签。

2026-07-31 对 Anti-Jitter V3 `model_2999.pt` 的实测：

- `120 N` 前推、`180 N` 前推和 `240 N` 侧推均未产生持续振动，推力后高频能量衰减；
- `360 N × 0.20 s` 侧推使最大 roll/pitch 达到约 `0.372/0.328 rad`，机器人没有摔倒，但出现明显高频恢复振铃；
- 恢复段相对稳定推力前，关节位置8–25 Hz能量放大 `32.89×`，actor action 高频放大 `123.24×`，action变化率放大 `683.77×`，PD力矩高频放大 `155.42×`；
- 推力后 PD 力矩 RMS/最大值约 `5.74/96.25 Nm`，但达到98%限位的比例只有约 `0.10%`，因此主因不是持续力矩饱和，而是受扰状态触发了策略 action 的高频闭环振荡；
- action变化率在推力结束约 `3.02 s` 后重新连续2秒低于稳定阈值；最终5秒的关节/actor长期高频标志均为否。因此 MuJoCo 复现的是“约3秒强烈振铃后收敛”，尚未复现真机所述无限持续抖动。剩余差异优先检查真机观测噪声、通信时延/丢帧、执行器动力学、摩擦和IMU坐标/滤波，而不是简单归因于力矩限位。

报告和曲线：

```text
/home/user/Workspace/Humanoid/Locomotion/G1-Locomotion/legged_lab/logs/monitoring/extreme_stand_large_push_diagnostic_20260731_force360_settled/REPORT.md
/home/user/Workspace/Humanoid/Locomotion/G1-Locomotion/legged_lab/logs/monitoring/extreme_stand_large_push_diagnostic_20260731_force360_settled/large_push/seed_20260734/large_push_diagnostics.png
```

## 10. 真机 ONNX 部署

真机入口和 MuJoCo 入口现已统一加载 `use/extreme_stand_recovery_pose_v2_model2999.onnx`；该文件来自同一个 Pose V2 `model_2999.pt` 的 ONNX 导出：

| 项目 | 路径/哈希 |
| --- | --- |
| 源 checkpoint | `ExtremeStandRecovery Checkpoints/2026-07-20_12-30-10_g1_extreme_stand_recovery_pose_v2_from_model4999_full_20260720/model_2999.pt` |
| checkpoint SHA256 | `ce7017ff810c5f24c533c1fac3b3fe8e539c712df8e64463076e557fb2df6264` |
| MuJoCo/真机统一 ONNX | `use/extreme_stand_recovery_pose_v2_model2999.onnx` |
| 统一 ONNX SHA256 | `0af2ffb24cd728352804b62669dc5264dd835329528311f9d50b90dbe2d0a0d1` |
| 兼容 TorchScript SHA256 | `0091c9939f5a43f754dbb87f56648560d547cf8b9bfc7f8852d2ed44a0791d71` |
| 共享部署元数据 SHA256 | `2bf0f21c511463b19bd8a1ef1f77122cc43cee41560bfb398e3b06ba00164fd7` |

MuJoCo runner 已支持 ONNX Runtime，因此两个入口现在加载同一个 ONNX 文件；真机仍不会加载 TorchScript。

空格和 `F` 是 MuJoCo GUI 测试键，不会下发给真机：真机的“初始姿态”由实体机器人当前状态决定，外力也必须来自安全、受控的物理测试，不能通过软件凭空切换。真机入口继续只接受零速度 Stand 策略，并保留现有确认门、短时运行和急停要求。

### 10.1 控制合同与 AMP 基线对齐

专用脚本会先校验 ONNX SHA、`96 -> 29` 接口、零速度命令、50 Hz 控制频率，并逐项比对导出元数据与 `unitree_sim2sim2real/deploy/deploy_real/configs/g1_amp.yaml` 中的：

- 29 个策略关节和 29 个电机的顺序映射；
- 默认关节角；
- 29 组 Kp/Kd；
- `action_scale=0.25`；
- `control_dt=0.02 s`。

校验后委托现有 `scripts/deploy_real_g1_amp_onnx.sh` 和 `deploy_real_g1_amp.py`，所以 LowCmd、PD、关节映射、动作尺度与 AMP 真机脚本是同一条链。该策略强制 `COMMAND_MODE=fixed`、`CMD_INIT=[0,0,0]`，不能由遥控器或 Nav2 改成行走命令。

该专用入口的启动语义是“**检查通过后自动开始策略推理**”：完成显式真机确认、文件/哈希/接口/依赖/网络检查，并收到第一帧有效 LowState 后，不等待空格键，也不先插值到默认姿态，直接用机器人当前观测构造 96 维输入；随后以 50 Hz 运行 ONNX，并将 29 维输出按 `q_target = default_angles + 0.25 * actor_action` 转成全身策略 PD 目标。因此它确实会尝试从启动时的当前姿态恢复到站立并保持，但恢复动作来自学习策略，而不是脚本内置的确定性“起身动作”。

“启动即推理”不应理解为双击后无条件发力：未设置 `CONFIRM_REAL_ROBOT=I_UNDERSTAND`、网卡错误、依赖缺失、模型合同不符或收不到 LowState 时，脚本都会在写入策略 LowCmd 前退出。策略也不承诺从倒地、机械干涉、关节越限或训练分布外的任意姿态恢复。

必须准确理解“限位对齐”的边界：现有 AMP 共享 runner 计算

```text
q_target = default_angles + 0.25 * actor_action
```

后直接写入 LowCmd，当前没有额外的软件 `q_target` 关节角裁剪。专用脚本为了严格对齐 AMP 基线，没有另造一套可能不一致的限位；实际安全仍依赖相同 Kp/Kd、action scale、Unitree 固件保护、急停和机械安全架。第一次真机测试必须吊架、限时、现场持急停，不能把 MuJoCo 的 `stress` 外力照搬到真机。

### 10.2 新终端 dry-run

先只做合同校验和打印最终命令，不连接机器人：

```bash
cd /home/user/Workspace/Humanoid/Locomotion/G1-Locomotion

DRY_RUN=True NET=enp11s0 \
UNITREE_PYTHON=/home/user/anaconda3/envs/gmr/bin/python \
bash scripts/deploy_real_g1_extreme_stand_recovery_onnx.sh
```

2026-07-21 已使用 Pose V2 的固定 ONNX/元数据哈希完成该 dry-run：`96 -> 29`、50 Hz、零速度命令、关节顺序、默认角、Kp/Kd 和 `action_scale=0.25` 全部通过。

### 10.3 真机执行

确认机器人在安全架上、网卡名正确、急停可用后：

```bash
cd /home/user/Workspace/Humanoid/Locomotion/G1-Locomotion

CONFIRM_REAL_ROBOT=I_UNDERSTAND \
NET=enp11s0 ROBOT_IP=192.168.123.161 \
UNITREE_PYTHON=/absolute/path/to/unitree-ready/bin/python \
RUN_DURATION=10 \
bash scripts/deploy_real_g1_extreme_stand_recovery_onnx.sh
```

这里的 `UNITREE_PYTHON` 必须替换为机器人侧真实存在、且在同一个环境中能够导入 `cyclonedds`、`unitree_sdk2py`、`onnxruntime`、`torch` 和 `yaml` 的 Python。当前开发机的 `gmr` 环境可完成 ONNX/配置 dry-run，但缺少 `cyclonedds`，不能用于实际 DDS 控制；脚本在非 dry-run 时会先做依赖检查，不会等到写 LowCmd 后才暴露环境问题。

先保留 `RUN_DURATION=10` 做短时验收。确认无异常后才可逐步延长；设置 `RUN_DURATION=0` 表示持续运行，不适合首次测试。部署沿用 AMP 基线的 direct handoff：不预先执行 zero-torque、damping、ReleaseMode 或默认姿态插值，因此启动前的实体姿态和现场安全条件尤其重要。

运行后的实际时序为：合同校验 → 真机确认门 → Python/DDS 依赖与网络检查 → 等待有效 LowState → **立即开始 ONNX 推理和策略 LowCmd**。退出或 `RUN_DURATION` 到期后，共享 runner 会发送阻尼命令。由于 direct handoff 没有独立的预备站姿阶段，机器人必须先处于策略训练覆盖的、机械上可恢复且有安全架保护的初始状态。

## 11. 已完成验证

旧 `model_4999.pt` 在 2026-07-19 已完成 TorchScript/ONNX 导出、真机部署 dry-run、4 档 × 3 seeds × 12 秒 MuJoCo 测试及 robust GUI 可视化。这些是历史基线，不代表 Pose V2 的新测试结果。

Pose V2 `model_2999.pt` 在 2026-07-21 已完成：

- checkpoint SHA256 锁定与 TorchScript/ONNX 重新导出；输入输出合同为 `96 -> 29`，ONNX 与 TorchScript 最大绝对差为 `6.91413879e-06`；
- `nominal`、`pose_recovery`、`recovery`、`robust` 四档各 1 seed × 8 秒无窗口回归，4/4 均未摔倒，`robust` 实际触发 3 次外力；
- 随机全身姿态恢复 5 seeds × 15 秒专项测试，5/5 均未摔倒，最终关节 MAE 平均为 `0.0634 rad`；
- 严格默认姿态恢复仍为 0/5，原因是双侧 shoulder-pitch 等单关节误差超过 `0.20 rad`，因此不能宣称严格恢复验收通过；
- GUI 可视化入口和新终端命令已切换到 Pose V2 默认 checkpoint；本轮实际自动验证使用的是 headless MuJoCo，没有声称已经人工观察 GUI 画面；
- 交互状态机 smoke 已在真实 G1 MuJoCo 模型上连续调用七次空格回调，验证前/后/左/右四向躯干大推力均不重置策略状态，随后依次完成随机29关节姿态、随机脚距和默认姿态重置；同时验证 `F` 开启后立即产生随机多 body 外力事件，关闭后 `xfrc_applied` 清零；
- 修改后的真实 MuJoCo 管线完成 `interactive` 1 秒和 `robust` 3 秒 headless smoke，二者均健康，后者实际产生 1 次外力事件；
- 真机脚本已改为同一 Pose V2 导出的 ONNX 和元数据，并通过完整 dry-run 合同校验；
- Shell 语法、Python 静态编译和 Extreme Stand 静态合同测试通过。

这些结果证明 Pose V2 的导出、MuJoCo 加载、零命令 29 维全身控制、初始姿态随机化和外力注入链路可运行，并显示默认姿态平均恢复误差相对旧模型明显改善；它们既不等于严格姿态恢复通过，也不构成无需安全架即可真机部署的保证。

Pose V3 奖励与训练代码在 2026-07-23 已完成：

- Python 静态编译、两个 Shell 启动器语法检查和 Extreme Stand 静态合同测试 `7/7` 通过；
- `8 env × 1 iteration` 完成真实 Isaac rollout、PPO 更新和 checkpoint 保存；
- `64 env × 45 iterations` 完成奖励量级 smoke，jerk 项、窄关键点高斯和窄足距高斯均产生有限数值，没有 NaN、Inf、CUDA、Isaac 或反向传播错误；
- Pose V2 启动器显式把三个 V3 新项置零，历史实验语义与新 V3 续训入口已分离；
- 2026-07-24 已从完整 `model_1400.pt` 恢复并完成至 iteration 2999；最终 checkpoint SHA256 为 `e2c694d2d7710315f41f1c6c75849ffb95b53d0fb29e612aa211e1525a7cb1e4`；
- 已生成 V3 ONNX（SHA256 `81bc3c1a1744e5549a8209f3e46a8b46863ff2fe68a38f3f50719a7f0f25784e`）并接入 MuJoCo；
- 2026-07-27 新增随机双脚间距专项场景与 `K` 交互键，真实 3-seed × 12 秒测试 3/3 未摔倒、1/3 严格恢复默认脚距，专项验收未通过；完整报告见 9.6；
- V3 MuJoCo 测试模型与现有 Pose V2 真机脚本尚未统一，不能把本节 V3 MuJoCo 结论直接当作真机部署模型结论。

Smooth-Torque V4 训练代码在 2026-07-31 已完成：

- 新增有 reset 安全历史缓存的 `action_second_difference_l2` 和 `joint_torque_rate_l2`，分别约束策略 action 高频曲率和 Isaac 实际执行力矩变化率；
- 强化原有力矩幅值、关节加速度、jerk 和一阶 action-rate 惩罚，同时完整保留 V3 默认姿态、笛卡尔关键点、足距、站立与摔倒奖励；
- 初始关节姿态和多部位随机外力范围扩大，但输入输出仍为 `96 -> 29`，没有修改 Walk 或 ArmHack；
- Shell 语法、Python 编译和 Extreme Stand 静态合同测试 `8/8` 通过；
- 真实 Isaac `4 env × 1 iteration` smoke 成功载入 V3 `model_2999.pt`，Reward Manager 显示27项，完成 rollout、反向传播和双份 checkpoint 保存，无 NaN、Inf、CUDA、Isaac 或历史缓存错误；
- smoke 仅证明代码可训练；必须完成正式续训并重新进行大推力长时 MuJoCo 高频对比后，才能判断是否缓解真机持续 jerk。
