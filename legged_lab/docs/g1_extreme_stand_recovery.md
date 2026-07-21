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
| `default_key_body_pose_exp` | `+2.5`，`std=0.12 m` | 在根节点 yaw 坐标系下约束躯干、膝、踝、肘、腕的笛卡尔位置 |
| `default_feet_distance_l2` | `-8.0` | 双脚平面距离偏离资产默认距离时作对称二次惩罚 |
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

Pose V2 的新增约束针对旧模型“能站住，但会选择明显宽于默认值的支撑姿态”这一漏洞。旧的 29 关节奖励将所有误差先取平均，少数腿部关节出现较大偏差时仍可能获得较高分；因此当前配置同时使用全身和腿部广义坐标奖励，并增加笛卡尔约束。关键刚体为 `torso_link`、左右膝、左右踝、左右肘和左右腕，共 9 个。笛卡尔位置相对 root 表示并消除 root yaw，因此不会要求机器人固定在某个世界坐标或绝对朝向，但会约束由关节和 root roll/pitch 造成的身体几何偏差：

```text
r_cart = exp(-mean_i(||p_i - p_i_default||²) / 0.12²)
```

双脚距离奖励以启动时从资产默认姿态缓存的左右踝平面距离 `d_default` 为目标，不硬编码某个数值：

```text
r_feet_penalty = -8.0 × (||p_left,xy - p_right,xy|| - d_default)²
```

所以双脚过近和过远都会受到惩罚；这不是“越近越好”的奖励。默认参考在施加 reset 噪声前缓存，策略观测仍保持 96 维，不包含未来扰动或未来目标。

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

重点观察 episode length、`termination_penalty`、`time_out/base_height/bad_orientation`、`default_joint_pose_exp`、`default_leg_joint_pose_exp`、`default_key_body_pose_exp`、`default_feet_distance_l2`、躯干速度、roll/pitch、高度、脚滑和总 reward。`default_feet_distance_l2` 是已乘负权重的 episode reward，越接近 0 表示足间距偏差越小。仅看总 reward 不能证明抗扰恢复成功。

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

一个交互可视化档位和五个固定测试档位为：

| `PROFILE` | 初始状态 | episode 内外力 | 用途 |
| --- | --- | --- | --- |
| `interactive` | 启动时为默认姿态；按空格切换默认/新随机29关节姿态，root 与速度不随机 | 启动时关闭；按 `F` 开关 `±35 N`、`±5 Nm` 随机脉冲 | 一次窗口内交互检查姿态恢复与抗外力 |
| `nominal` | 默认姿态、零速度 | 无 | 基本站立 |
| `pose_recovery` | 只随机化29个关节姿态；root 与全部速度保持默认 | 无 | 隔离检查能否恢复默认全身姿态 |
| `recovery` | 训练范围内全身关节、root 姿态/速度噪声 | 无 | 只检查受扰初值恢复 |
| `robust` | 同 `recovery` | `±35 N`、`±5 Nm`，每 `2.5 s` 一次 | 训练同级持续抗扰 |
| `stress` | 比训练更大的噪声 | `±50 N`、`±8 Nm`，每 `2.0 s` 一次 | 超训练分布压力测试，不计入基础验收 |

### 9.1 打开 MuJoCo GUI

脚本现在默认锁定 Pose V2 `model_2999.pt`，新终端无需再手工填写 checkpoint。推荐使用统一交互入口：

```bash
cd /home/user/Workspace/Humanoid/Locomotion/G1-Locomotion

PROFILE=interactive USE_GLFW=True REAL_TIME=True \
SIMULATION_DURATION=300 SEED=20260722 \
bash scripts/sim2sim_g1_extreme_stand_recovery_mujoco.sh
```

窗口获得焦点后：

- 按一次空格：从默认初始姿态切换到一组新采样的随机29关节初始姿态，并立即重置仿真；root 姿态、关节/root 速度保持默认，因而该键只隔离测试姿态恢复。再次按空格回到默认初始姿态；以后重复交替，每次进入随机态都会重新采样。
- 按一次 `F`：开启随机多部位脉冲外力，并立即施加第一下；之后每 `2.5 s` 施加一次、每次持续 `0.25 s`；再次按 `F` 立即关闭并清零外力。
- 两个开关互相独立。空格重置时会把策略上一动作清零，保证新 episode 的 96 维观测合同正确；它不会覆盖 Actor 后续输出。终端会打印每次切换状态，最终 `metrics.json` 也记录按键事件。

外力并不是只施加在脚上。当前 `F` 键随机候选 body 明确为：`pelvis`、`torso_link`、左右 `shoulder_pitch_link`、左右 `elbow_link`、左右 `hip_pitch_link`、左右 `knee_link`，不包含脚或踝。每个事件只从这十个 body 中随机选择一个，通过 MuJoCo 的 `data.xfrc_applied[body_id]` 写入世界坐标系三轴力和三轴力矩。

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

固定场景仍可使用 `nominal`、`pose_recovery`、`recovery`、`robust` 或 `stress`；固定场景用于可重复报告，不响应这两个交互开关。`stress` 只用于仿真，不能照搬到真机。本机 MuJoCo/GLFW 偶尔会在窗口已经完成、JSON 已落盘后的 native viewer 关闭阶段返回 `139`；专用入口仅在 `USE_GLFW=True`、退出码确为 `139` 且完整 `metrics.json` 已存在时将其标记为 viewer-shutdown warning，仿真中途错误仍会失败。

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

### 9.3 五档多种子完整测试

```bash
cd /home/user/Workspace/Humanoid/Locomotion/G1-Locomotion

SUITE=True USE_GLFW=False REAL_TIME=False \
SUITE_DURATION=12 \
SUITE_SEEDS=20260719,20260720,20260721 \
SUITE_RESULTS_ROOT="$PWD/legged_lab/ExtremeStandRecovery Checkpoints/2026-07-20_12-30-10_g1_extreme_stand_recovery_pose_v2_from_model4999_full_20260720/exported_extreme_stand_recovery/mujoco_tests/suite_manual" \
REQUIRE_PASS=True \
bash scripts/sim2sim_g1_extreme_stand_recovery_mujoco.sh
```

基础验收要求：`nominal`、`recovery` 全部健康，`robust` 至少 `2/3` 健康；这些场景的速度指令必须始终为零，且 `extreme_stand_recovery.action_override=false`。`stress` 只报告额外余量，不代表真机允许施加同等扰动。

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
- 交互状态机单元 smoke 已实际调用空格与 `F` 回调，验证默认→随机→默认姿态重置、外力开启后立即产生事件、关闭后 `xfrc_applied` 清零；
- 修改后的真实 MuJoCo 管线完成 `interactive` 1 秒和 `robust` 3 秒 headless smoke，二者均健康，后者实际产生 1 次外力事件；
- 真机脚本已改为同一 Pose V2 导出的 ONNX 和元数据，并通过完整 dry-run 合同校验；
- Shell 语法、Python 静态编译和 Extreme Stand 静态合同测试通过。

这些结果证明 Pose V2 的导出、MuJoCo 加载、零命令 29 维全身控制、初始姿态随机化和外力注入链路可运行，并显示默认姿态平均恢复误差相对旧模型明显改善；它们既不等于严格姿态恢复通过，也不构成无需安全架即可真机部署的保证。
