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
| Isaac Sim 可视化 | `scripts/extreme_stand_recovery/vis_g1_extreme_stand_recovery.sh` |
| MuJoCo 测试 | `scripts/extreme_stand_recovery/val_mujoco_g1_extreme_stand_recovery.sh` |
| MuJoCo 扰动器 | `../unitree_sim2sim2real/deploy/deploy_mujoco/extreme_stand_recovery.py` |
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
| `default_joint_pose_exp` | `+2.5` | 29 个关节恢复到默认站立姿态 |
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

重点观察 episode length、`termination_penalty`、`time_out/base_height/bad_orientation`、`default_joint_pose_exp`、躯干速度、roll/pitch、高度、脚滑和总 reward。仅看总 reward 不能证明抗扰恢复成功。

## 7. Isaac Sim 测试与可视化

每个新终端都要重新定义 checkpoint 变量。下面只是格式示例，必须替换为本次完整训练的实际目录：

```bash
cd /home/user/Workspace/Humanoid/Locomotion/G1-Locomotion/legged_lab
conda activate env_isaaclab

EXTREME_STAND_CKPT="$PWD/logs/rsl_rl/g1_extreme_stand_recovery/<run>/model_<iter>.pt"
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

## 8. MuJoCo sim2sim 测试

MuJoCo 测试器会：

1. 从 checkpoint 导出 TorchScript `policy.pt` 和 ONNX `policy.onnx`；
2. 给全身关节、关节速度、基座 roll/pitch、基座线速度和角速度加入随机初值；
3. 每隔固定时间从骨盆、躯干、肩、肘、髋、膝中随机选一个位置，施加短时随机外力；
4. 原样使用 Actor 的 29 维输出，不做双臂或其他关节覆盖；
5. 保存健康状态、躯干指标、轨迹和扰动参数到 JSON。

GUI 可视化：

```bash
cd /home/user/Workspace/Humanoid/Locomotion/G1-Locomotion/legged_lab
conda activate env_isaaclab

EXTREME_STAND_CKPT="$PWD/logs/rsl_rl/g1_extreme_stand_recovery/<run>/model_<iter>.pt"
CHECKPOINT="$EXTREME_STAND_CKPT" USE_GLFW=True SIMULATION_DURATION=30 \
bash scripts/extreme_stand_recovery/val_mujoco_g1_extreme_stand_recovery.sh
```

Headless 批量测试：

```bash
CHECKPOINT="$EXTREME_STAND_CKPT" USE_GLFW=False SIMULATION_DURATION=30 \
SEED=20260719 FORCE_MAX_N=35 TORQUE_MAX_NM=5 \
bash scripts/extreme_stand_recovery/val_mujoco_g1_extreme_stand_recovery.sh
```

增大测试外力、使用另一随机种子：

```bash
CHECKPOINT="$EXTREME_STAND_CKPT" USE_GLFW=False SIMULATION_DURATION=60 \
SEED=20260720 FORCE_MAX_N=50 TORQUE_MAX_NM=8 \
WRENCH_INTERVAL_S=2.0 WRENCH_DURATION_S=0.25 \
bash scripts/extreme_stand_recovery/val_mujoco_g1_extreme_stand_recovery.sh
```

默认导出与报告位于 checkpoint 同级目录：

```text
exported_extreme_stand_recovery/policy.pt
exported_extreme_stand_recovery/policy.onnx
exported_extreme_stand_recovery/policy.deploy.json
exported_extreme_stand_recovery/mujoco_extreme_stand_recovery_metrics.json
exported_extreme_stand_recovery/mujoco_torso_trace.csv
```

报告中的 `extreme_stand_recovery.action_override` 必须为 `false`，`wrench.event_count` 应大于零。若运行时间小于首次外力间隔，则 `event_count=0` 只说明测试时间太短。

## 9. 已完成的 smoke 验证

2026-07-19 在本机 `env_isaaclab`、RTX 4090 `cuda:0` 上完成：

- 静态合同测试：5 项全部通过；
- IsaacLab 真实训练：32 个环境、1 次 PPO 迭代成功，生成 `model_0.pt`；
- 运行时接口：Policy observation 实测为 96，Action Manager 实测为 29；
- Isaac Sim headless 回放：新 Play task 成功创建并执行；
- MuJoCo headless：3.2 秒、随机初始全身状态、1 次随机身体外力事件成功执行；
- MuJoCo 报告确认 `action_override=false`，导出器确认 `obs_dim=96`、`action_dim=29`。

以上 smoke 只证明代码、环境、checkpoint 加载、训练一步、导出和测试链路没有运行错误；最终“极鲁棒”性能必须以完整训练和多随机种子验收结果为准。
