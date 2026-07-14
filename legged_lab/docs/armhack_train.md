# ArmHack 训练实现、数据路径与验证指南

## 1. 当前状态

截至 2026-07-14，Stand 已完成数据整理、奖励修复、无跳变初始化、分阶段课程、确定性双臂测试集和逐关节报告。历史正式 `model_2999.pt` 是从 `model_7999.pt` 训练得到；当前 `scripts/train_g1_armhack_stand.sh` 已改为从 S3 G1 `model_9996.pt` 开始新的 policy-only 训练：

```text
LeggedLab-Isaac-AMP-G1-StandPerturb-v0
LeggedLab-Isaac-AMP-G1-WalkPerturbFinetune-v0
```

当前验证结果：

- 静态检查 `9 passed`；
- 当前 Stand 基线为 `BaselineModel9996/model_9996.pt`，大小 `16,202,421 bytes`，iteration `9996`，actor `96→29`、critic input `297`，SHA-256 为 `bc30bc5171d211fa414fbeab31452b92ad76ca7f6ad76a2417a6e7f7515a0fa6`；
- 随机静止姿态阶段：`4 env × 1 iteration` 正常完成，日志为 `csv_motion_scale=0`、`curriculum_stage=0`；随机起点时间均值 `198.6063 s`、标准差 `137.4946 s`，确认并行环境不是同一起点；
- 静止阶段长短测：`4 env × 42 iteration`、4032 个环境步正常完成，所有已结束 episode 均为 `time_out`，`base_height=0`、`bad_orientation=0`，保存 `model_41.pt`；
- 历史低速连续运动阶段：`2 env × 1 iteration` 正常完成，日志为 `csv_motion_scale=0.25`、`curriculum_stage=2`；该记录只对应旧课程；
- 当前全速连续运动阶段：`2 env × 1 iteration` 已从 `model_9996.pt` 正常完成，日志为 `csv_motion_scale=1.0000`、`curriculum_stage=2.0000`，并在 Stand 专用目录保存 `model_0.pt`；
- 当前入口已确认从 `model_9996.pt` 做 policy-only 恢复、加载同一冻结基线做 KL，并在 Stand 专用目录生成 `model_0.pt`；
- 本机历史 Stand 课程已完成 `4096 env × 3000 iteration`，其阶段 1/2 使用 `0→0.25x` 和 `0.25x`；最终模型为 `model_2999.pt`。当前新课程已改为 `0→1.0x→持续 1.0x`，历史模型不能代表 1.0x 新训练结果；
- 历史 `model_2999.pt` 已完成当前 103.96 s 确定性双臂测试集的 `1.0x` 整段回放：5198 step、0 次 termination/reset；该结果证明测试链和旧模型在所选样本上可运行，不替代 model9996 新课程的独立验收；
- 正式训练最终点为 mean episode length `1000/1000`、timeout `99.98%`、base-height termination `0`、bad-orientation termination `0.02%`，torso roll/pitch 误差 `0.0074/0.0130 rad`；
- 旧 Walk smoke 使用了来源错误的 Stand `model_7999`，只作为路径问题的诊断记录，不能再作为有效 Walk 起点测试；
- 正确 Walk 起点已锁定为 `checkpoint/model_9996/locomotion.onnx`；原始 `model_9996.pt` 的 8 个 actor 张量与 ONNX 逐元素完全一致；
- `pos2_down`、`pos1_back`、`pos3_front` 均从正确 model9996 完成 `2 env × 1 iteration`，pose index 为 1/0/2，初始化误差与两类摔倒终止均为 0；
- 从正确基线生成的 Walk `model_0.pt` 完成一次 `MODE=resume` full-state 恢复 smoke；
- 此前 Stand/旧 Walk smoke checkpoint 均从专用目录完成 `1 env × 5 steps` headless 回放；正确 model9996 Walk 的三姿态训练链与 full-resume 链已单独验证；
- 上述训练与回放均没有 traceback、`FileNotFoundError`、Isaac/Python 版本错误或 Nav2 数据缺失错误；
- 训练不再依赖 `/home/hecggdz/...` 等机器绝对路径。

这里的 smoke `model_0.pt` 只证明任务注册、数据加载、Isaac Sim、rollout、PPO 更新和保存执行链可运行，不代表一轮更新已经学会新任务。当前 Stand 使用 `BaselineModel9996/model_9996.pt`；既有 Stand 性能判断仍使用本机正式 `model_2999.pt`，并明确注明它来自历史 model7999 课程，不能假称为 model9996 续训结果。

## 2. Reference Data 目录

两个任务的姿态参考数据统一保存在：

```text
legged_lab/Reference Data/ArmHack/
├── README.md
├── StandPerturb/
│   ├── raw/
│   │   └── g1_full_body_motion_sdk_50hz.csv
│   ├── g1_arm_trajectory_named_50hz.csv
│   └── TestData/ArmOnly/
│       ├── poses/{representative,synthesized}/
│       ├── trajectories/{representative,synthesized}/
│       ├── sequences/
│       └── manifest.json
└── WalkPerturbFinetune/
    ├── g1_arm_pose_set.json
    └── nav2_cmd_vel_raw_success.csv
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

2026-07-14 对规范训练 CSV 做了只读数值审计：

```text
帧数：20,122
持续时间：404.917792 s
相邻时间间隔中位数：0.02010 s（约 49.75 Hz）
相邻帧最大单关节变化：0.029313 rad
全轨迹最大关节速度：1.4569 rad/s（左肘）
全轨迹最大关节加速度：25.5020 rad/s²（左肘）
```

轨迹内部大部分帧是连续的。历史实现从机器人默认手臂姿态直接切到随机目标：随机起点的“最大单关节差值”中位数、P95、最大值分别约为 `1.446 / 1.466 / 1.908 rad`，L2 差值中位数约 `2.477 rad`；即使固定从第 0 帧开始，最大单关节差值仍为 `1.440 rad`。当前实现已取消这次接入跳变：reset 时直接把仿真手臂关节状态初始化到所抽取的 CSV 姿态，并将手臂关节速度置零。

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

`reference_data.py` 会验证 JSON 单位为弧度、关节顺序正确、每侧恰好 7 个值、pose 名不重复、所有值有限，然后返回“名称 + 左右交错 14 维值”。专用脚本默认 `POSE_NAME=pos2_down`，也支持 `pos1_back`、`pos3_front` 和 `random`。每个 episode 内姿态固定不变。

### 2.3 Walk 的 Nav2 速度分布

动态任务使用：

```text
Reference Data/ArmHack/WalkPerturbFinetune/nav2_cmd_vel_raw_success.csv
```

本机搜索不到该文件后，从 HEC-5090 的下列位置下载：

```text
/home/hecggdz/workspace-zwd/legged_lab/nav2_loopback_actual/
actual_raw_success/all_cmd_vel_success.csv
```

数据契约与检查结果：

```text
行数：331,010
连续轨迹组：445
列数：17
vx：[-0.2, 0.6] m/s
vy：[-0.3, 0.3] m/s
wz：[-0.5187280178070068, 0.6] rad/s
augmentation：none
SHA-256：76a4516588b855351eb3eb8c2da26e291603876c1a4a1b9c7bacd77a53807b5a
```

它是实际 Nav2 成功轨迹的原始分布，不是离线均匀采样，也不是离线镜像文件。已在 HEC-5090 的整个 `workspace-zwd` 中搜索，远端只有这份 `all_cmd_vel_success.csv`，没有 `all_cmd_vel_augmented.csv`。当前 loader 会在内存中由每个 `none` 组生成 `(vx,-vy,-wz)` 的 `mirror_lr` 组，以恢复 Nav2 Stage-4 的左右分布；源 CSV 不改写。CSV 约 83 MiB，已加入 `.gitignore`；GitHub clone 后需要单独放回上述相对位置。

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
CSV 起点：每个环境 reset 时随机选择，并直接初始化 14 个手臂关节到该姿态
CSV 采样：连续时间相位 + 相邻帧线性插值
CSV 循环：关闭，结束后保持最后一帧
episode：20 s
AMP style reward：0.0
task/style lerp：1.0
RSI：关闭
初始根位置、yaw 和六维根速度：全为 0
外部 push：关闭
```

课程按全局训练步数自动切换；每个 PPO iteration 收集 24 个 control step：

| 阶段 | 默认 iteration | 手臂行为 |
|---|---:|---|
| 0：随机姿态静止 | `0..499` | reset 随机抽取任意 CSV 相位，整段 episode 保持该姿态 |
| 1：全速渐入 | `500..1499` | 轨迹速度从 `0` 连续线性增至原始 `1.0x` |
| 2：全速适应 | `1500..2999` | 持续按原始 `1.0x` 速度运动 |

所有阶段都从随机相位开始。reset 后直接把手臂关节状态写到该相位、关节速度写为 0，所以不存在从默认姿态到随机目标的首步跳变；进入运动阶段后，相位每个 control step 按当前倍率连续推进。非循环轨迹按最终 `1.0x` 和 20 s episode 计算最大随机起点，保证 episode 内不会因到达 CSV 末尾而提前停住。

行走、步态、摆腿和 directional 奖励被关闭。Stand 奖励全部显式重建，真实 Reward Manager 已确认 21 项有效：

| 奖励 | 权重 | 目的 |
|---|---:|---|
| `alive` | `+1.0` | 消除“早摔少累计代价”的错误激励 |
| `track_torso_lin_vel_xy_exp` | `+1.5` | 躯干水平速度保持为 0 |
| `track_torso_yaw_rate_exp` | `+0.75` | 躯干 yaw 角速度保持为 0 |
| `double_support` | `+0.25` | 鼓励双脚持续支撑 |
| `flat_orientation_l2` | `-1.0` | 根节点直立 |
| `torso_roll_pitch_l2` | `-3.0` | 躯干 roll/pitch 稳定 |
| `torso_ang_vel_xy_l2` | `-0.15` | 抑制躯干水平角速度 |
| `torso_vertical_velocity_l2` | `-0.30` | 抑制躯干上下运动 |
| `torso_height_band_l2` | `-0.60` | 躯干保持 `0.84 m` 高度带 |
| `torso_specific_force_xy_l2` | `-0.01` | 抑制水平加速度响应 |
| `root_xy_position_l2` | `-1.0` | 抑制相对环境原点的水平漂移 |
| `lin_vel_z_l2` / `ang_vel_xy_l2` | `-0.30` / `-0.10` | 根状态稳定 |
| 下肢 `torque` / `acc` / `action-rate` | `-2e-6` / `-1e-7` / `-0.005` | 只正则化可控的腿和腰 |
| 踝限位 / 髋偏差 / 腰偏差 | `-1.0` / `-0.10` / `-0.12` | 关节安全和自然站姿 |
| `feet_slide` | `-0.25` | 抑制脚滑 |
| `termination_penalty` | `-200.0` | 惩罚非 timeout 摔倒 |

policy 输入仍为原 96 维，没有添加轨迹相位、未来手臂目标、目标速度或 look-ahead。它只能通过当前 `joint_pos/joint_vel` 感知已经发生的手臂状态变化，满足“不能知道未来手臂怎么运动”的约束。

### 3.3 WalkPerturbFinetune 当前实现

Walk 使用独立 `G1WalkPerturbAmpEnv`，不修改共享的 Stand reset 路径。reset 时按名称选择 pose，并在首个 policy observation 前同步实际手臂 `q/dq`、关节 position target、ActionManager 当前/上一动作和 JointPositionAction 的 raw/processed buffer。日志项 `ArmHack/walk_pose_init_max_error_rad` 用于检查写入误差，`ArmHack/walk_pose_index_mean` 用于确认所选姿态。

速度命令使用 `Nav2RecordedVelocityCommandCfg`。采样单位不是互相独立的 `vx/vy/wz` 行，而是按 planner、controller、scenario、goal 分组后截取连续窗口：

```text
scenario family：complex_turn
数据窗口：4.0 s
数据采样周期：0.05 s
命令缩放：(0.85, 0.75, 0.75)
命令裁剪：vx [-0.20, 0.60]，vy [-0.30, 0.30]，wz [-0.60, 0.60]
平滑时间常数：0.30 s
最大线加速度：0.60 m/s²
最大 yaw 加速度：0.80 rad/s²
轨迹组均匀采样：开启
controller 权重：mppi 1.5，dwb 1.0
augmentation_filter：none,mirror_lr（由 raw none 在线生成镜像）
```

相对数据路径由 command loader 以 `legged_lab` 项目目录为锚点解析。源 CSV 只有 raw `none` 样本；loader 在内存中生成 `(vx,-vy,-wz)` 镜像组，不改写源文件。

Walk 现在直接继承 `G1AmpNav2FinetuneEnvCfg`，并显式锁定保存的 Nav2 Stage-4 command 与奖励。linear/yaw tracking 为 `1.8/1.5`，torso linear/yaw 为 `1.0/0.65`，flat orientation 为 `-1.0`，torso roll/pitch 为 `-0.04`，termination 为 `-200`；手臂 deviation/style 关闭，torque/acc/action-rate 和 discriminator 只看可控腿腰。原先只修改 `ranges`、对 recorded dataset 无效的两个 command curriculum 已删除。

## 4. 代码位置

| 作用 | 路径 |
|---|---|
| Reference Data 路径和 JSON 校验 | `source/legged_lab/legged_lab/tasks/locomotion/amp/config/g1_perturb/reference_data.py` |
| 29DoF action 覆盖、CSV 读取 | `source/legged_lab/legged_lab/envs/g1_perturb_env.py` |
| Walk 具名 pose、无跳变 reset 与 action-history 同步 | `source/legged_lab/legged_lab/envs/g1_walk_perturb_env.py` |
| Stand 配置 | `source/legged_lab/legged_lab/tasks/locomotion/amp/config/g1_perturb/g1_stand_perturb_env_cfg.py` |
| Walk 配置 | `source/legged_lab/legged_lab/tasks/locomotion/amp/config/g1_perturb/g1_walk_perturb_env_cfg.py` |
| Gym task 注册 | `source/legged_lab/legged_lab/tasks/locomotion/amp/config/g1_perturb/__init__.py` |
| Runner 配置 | `source/legged_lab/legged_lab/tasks/locomotion/amp/config/g1_perturb/agents/rsl_rl_ppo_cfg.py` |
| Nav2 录制命令窗口加载器 | `source/legged_lab/legged_lab/tasks/locomotion/amp/mdp/commands/nav2_recorded_velocity_command.py` |
| 全身 CSV 转 14 手臂 CSV | `scripts/tools/extract_armhack_stand_arm_csv.py` |
| ArmHack 数据完整性预检 | `scripts/tools/check_armhack_reference_data.py` |
| CSV 全身回放 | `scripts/tools/visualize_g1_csv_full_body_motion.py` |
| Stand 专用分阶段训练脚本 | `scripts/train_g1_armhack_stand.sh` |
| Walk 专用 S3 locomotion model9996 训练脚本 | `scripts/train_g1_armhack_walk.sh` |
| 通用训练脚本 | `scripts/train_g1_amp.sh` |
| checkpoint 可视化/回放 | `scripts/vis_isaacsim_g1_amp.sh` |
| Python 训练入口 | `scripts/rsl_rl/train.py` |
| policy-only 加载和专用 checkpoint 复制 | `../rsl_rl/rsl_rl/runners/amp_runner.py` |
| 无仿真静态检查 | `source/legged_lab/test/test_g1_perturb_static.py` |

执行链：

```text
train_g1_armhack_stand.sh
  -> 校验 model_9996 SHA-256/大小，计算课程 step，建立恢复用软链接
  -> train_g1_amp.sh
  -> scripts/rsl_rl/train.py
  -> Gym task registry
  -> Stand env cfg + runner cfg
  -> G1PerturbAmpEnv
  -> RslRlVecEnvWrapper
  -> AMPRunner / PPOAMP.learn()

train_g1_armhack_walk.sh
  -> 检查 Stand/重复 Walk 进程，校验 S3、locomotion.onnx、model9996、pose JSON 和 Nav2 CSV
  -> init: policy-only；resume: 完整恢复 Walk run
  -> train_g1_amp.sh -> Gym Walk task
  -> G1WalkPerturbFinetuneEnvCfg + G1WalkPerturbAmpEnv
  -> Nav2RecordedVelocityCommand + fixed named pose
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

本机 `pytest` 可执行文件位于 `~/.local/bin`，其自动加载的第三方插件可能与 Conda 环境中的 `packaging` 冲突；这不影响训练入口。执行仓库静态测试时应明确关闭外部插件自动加载：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/user/.local/bin/pytest -q \
  source/legged_lab/test/test_g1_perturb_static.py
```

## 6. 训练命令

### 6.1 Stand 正式分阶段训练

```bash
cd /home/user/Workspace/Humanoid/Locomotion/G1-Locomotion/legged_lab
bash scripts/train_g1_armhack_stand.sh
```

专用脚本默认参数：

```text
BASE_CHECKPOINT=ArmHack Checkpoints/StandPerturb/BaselineModel9996/model_9996.pt
NUM_ENVS=4096
MAX_ITERATIONS=3000
STATIC_ITERATIONS=500
RAMP_ITERATIONS=1000
FINAL_MOTION_SCALE=1.0
RANDOMIZATION_STRENGTH=0
STYLE_REWARD_SCALE=0.0
TASK_STYLE_LERP=1.0
BASELINE_KL_SCALE=0.003
ENTROPY_COEF=0.002
```

脚本会先验证 checkpoint 文件的固定 SHA-256 和 `16,202,421 bytes` 大小，再把基线以软链接暂存到 `logs/rsl_rl/g1_stand_perturb/_armhack_stand_baseline_model9996/`，供 IsaacLab/RSL-RL 的标准恢复逻辑解析。随后强制：

```text
RESUME=True
agent.load_policy_only=True
agent.reset_iteration_on_policy_only_load=True
RSI_ENABLE=False
BASELINE_KL_ENABLE=True
```

也就是说只加载 `model_9996` 的 actor/critic 权重，不继承旧 optimizer、discriminator 或 iteration 计数；新 run 从 iteration 0 开始执行完整课程。冻结的同一基线 policy 用于 KL 正则，防止下肢策略在新奖励初期快速漂移。policy 观测接口仍为 96 维，动作仍为 29 维，因此与当前 Stand 环境兼容。

覆盖课程参数示例：

```bash
RUN_NAME=armhack_stand_curriculum_v2 \
STATIC_ITERATIONS=800 \
RAMP_ITERATIONS=1200 \
MAX_ITERATIONS=4000 \
FINAL_MOTION_SCALE=1.0 \
bash scripts/train_g1_armhack_stand.sh
```

脚本末尾可继续附加 Hydra override，但不应添加任何未来轨迹观测。例如需要临时改 episode 时长可执行：

```bash
bash scripts/train_g1_armhack_stand.sh env.episode_length_s=20.0
```

### 6.2 Stand 两阶段最小测试

随机姿态静止阶段：

```bash
NUM_ENVS=2 MAX_ITERATIONS=1 \
STATIC_ITERATIONS=1 RAMP_ITERATIONS=1 \
RUN_NAME=armhack_stand_static_stage_smoke \
bash scripts/train_g1_armhack_stand.sh
```

直接强制进入原始 `1.0x` 运动阶段：

```bash
NUM_ENVS=2 MAX_ITERATIONS=1 \
STATIC_ITERATIONS=0 RAMP_ITERATIONS=0 \
FINAL_MOTION_SCALE=1.0 \
RUN_NAME=armhack_stand_full_speed_smoke \
bash scripts/train_g1_armhack_stand.sh
```

### 6.3 Walk 从 locomotion.onnx / model 9996 正式初始化

必须使用专用脚本；不要再直接用通用脚本启动 Walk：

```bash
cd /home/user/Workspace/Humanoid/Locomotion/G1-Locomotion/legged_lab
POSE_NAME=pos2_down \
RUN_NAME=armhack_walk_pos2_from_locomotion_model9996 \
bash scripts/train_g1_armhack_walk.sh
```

默认契约：`MODE=init`、S3 G1、4096 env、4000 iteration、Nav2 Stage-4 command、learning rate `3e-5`、style reward `0`、task/style lerp `1`、baseline KL `0.003`、domain randomization `0`、RSI 关闭。脚本会检查：

- `checkpoint/model_9996/locomotion.onnx` 的 SHA、96→29 接口；
- 原始 `model_9996.pt` 的 SHA、大小、iteration、actor/critic 形状；
- ONNX 与 `.pt` 中 8 个 actor 权重/偏置张量逐元素完全一致；
- Nav2 CSV 的 SHA 和 pose JSON 是否存在；
- 资产固定为 `s3_g1_29dof`；
- 当前没有 Stand 或另一个 Walk 训练进程；
- Hydra 参数不能覆盖 checkpoint 加载、baseline KL 或 robot spawn。

最小真实 smoke：

```bash
NUM_ENVS=2 MAX_ITERATIONS=1 \
POSE_NAME=pos2_down \
RUN_NAME=armhack_walk_locomotion_model9996_smoke \
QUIET_TERMINAL=False \
bash scripts/train_g1_armhack_walk.sh
```

### 6.4 完整恢复已有 Walk run

首次初始化只加载 policy；Walk 中断后必须用 `MODE=resume` 完整恢复 optimizer、discriminator、normalizer 和 iteration：

```bash
MODE=resume \
RESUME_RUN='<logs/rsl_rl/g1_walk_perturb 下的完整 run 目录名>' \
RESUME_CHECKPOINT='model_1200.pt' \
POSE_NAME=pos2_down \
RUN_NAME=armhack_walk_pos2_resume1200 \
bash scripts/train_g1_armhack_walk.sh
```

脚本会校验日志目录与 `ArmHack Checkpoints/WalkPerturbFinetune` 中两份 resume checkpoint 的 SHA 一致。`MODE=resume` 设置 `agent.load_policy_only=False`；不会错误地把每次续训都重置到 iteration 0。

## 7. checkpoint 保存位置

新目录结构为：

```text
ArmHack Checkpoints/
├── StandPerturb/
│   ├── BaselineModel9996/model_9996.pt
│   ├── <run_name>/Test Reports/StandArmOnly/*.md
│   └── <run_name>/model_*.pt
└── WalkPerturbFinetune/
    ├── BaselineLocomotionModel9996/model_9996.pt
    └── <run_name>/model_*.pt
```

runner 配置分别设置：

```python
checkpoint_output_dir = "ArmHack Checkpoints/StandPerturb"
checkpoint_output_dir = "ArmHack Checkpoints/WalkPerturbFinetune"
```

每次保存时，`AMPRunner.save()` 先按原逻辑写入：

```text
logs/rsl_rl/<experiment>/<run_name>/model_*.pt
```

再复制到对应的 ArmHack 专用目录。保留日志目录副本是为了不破坏 RSL-RL 原有的 `LOAD_RUN/CHECKPOINT` 恢复训练逻辑；专用目录便于集中查找、测试和交付两个策略。两边模型已经用 SHA-256 验证完全一致。

checkpoint 文件已加入 `.gitignore`，不会上传 GitHub。不要按文件修改时间自动选择“最新模型”：冒烟测试也会生成 `model_0.pt`，旧的 `latest_checkpoint` 写法可能把它误当成正式模型。两个任务的已验证起始模型分别保存在：

```text
StandPerturb/BaselineModel9996/model_9996.pt
SHA-256: bc30bc5171d211fa414fbeab31452b92ad76ca7f6ad76a2417a6e7f7515a0fa6
size: 16,202,421 bytes；iteration: 9996；actor: 96→29；critic input: 297

Walk actor: ../checkpoint/model_9996/locomotion.onnx
SHA-256: 05fc45f89d89eb136225754f6a2fcacf5324d9dfd428d08ed75cc52f89b09be6

Walk training checkpoint: WalkPerturbFinetune/BaselineLocomotionModel9996/model_9996.pt
SHA-256: bc30bc5171d211fa414fbeab31452b92ad76ca7f6ad76a2417a6e7f7515a0fa6
size: 16,202,421 bytes；iteration: 9996；actor: 96→29；critic input: 297
```

Walk `.pt` 来源由 `locomotion.deploy.json` 记录，并已从 HEC-5090 原始 S3 run 下载：

```text
/home/hecggdz/workspace-zwd/legged_lab/logs/rsl_rl/g1_amp/
2026-06-17_03-48-38_s3_g1_29dof_command_balanced_directional_strict_armprior_v3_resume8997_1000/model_9996.pt
```

正式课程训练直接使用第 6.1 节的专用脚本。测试或可视化时仍应显式填写完成正式训练且已经通过固定评估的路径：

```bash
STAND_CKPT='ArmHack Checkpoints/StandPerturb/<正式run>/model_<正式迭代>.pt'
WALK_CKPT='ArmHack Checkpoints/WalkPerturbFinetune/<正式run>/model_<正式迭代>.pt'

test -n "$STAND_CKPT" && test -f "$STAND_CKPT"
test -n "$WALK_CKPT" && test -f "$WALK_CKPT"
printf 'Stand: %s\nWalk:  %s\n' "$STAND_CKPT" "$WALK_CKPT"
```

最低检查要求：路径不得包含 `smoke`，不能仅因为文件名是 `model_0.pt` 就认为它有效；必须同时打开对应 run 的 `params/env.yaml`、`params/agent.yaml` 和 TensorBoard 曲线确认训练配置、训练轮数、平均 episode 长度与终止原因。

## 8. 测试代码与指令

### 8.1 外部数据预检

测试代码：

```text
scripts/tools/check_armhack_reference_data.py
```

它不启动 Isaac Sim，会检查：

- Stand 原始 CSV 和 14 关节训练 CSV 的 SHA-256；
- Stand 训练 CSV 的列名、20,122 行和数值有限性；
- Walk 三组 pose 的名称、单位、顺序和 14 个数值；
- Nav2 CSV 的 SHA-256、必需列、331,010 行、445 组、速度范围和 `augmentation=none`。

执行：

```bash
python scripts/tools/check_armhack_reference_data.py
```

若新机器缺少 Nav2 CSV，可在 `legged_lab` 目录执行：

```bash
mkdir -p 'Reference Data/ArmHack/WalkPerturbFinetune'
scp hecggdz@172.28.162.15:/home/hecggdz/workspace-zwd/legged_lab/nav2_loopback_actual/actual_raw_success/all_cmd_vel_success.csv \
  'Reference Data/ArmHack/WalkPerturbFinetune/nav2_cmd_vel_raw_success.csv'
python scripts/tools/check_armhack_reference_data.py
```

### 8.2 静态代码回归

```bash
bash -n scripts/train_g1_amp.sh
bash -n scripts/train_g1_armhack_stand.sh
bash -n scripts/train_g1_armhack_walk.sh
bash -n scripts/vis_isaacsim_g1_amp.sh
python -m compileall -q \
  source/legged_lab/legged_lab/envs/g1_walk_perturb_env.py \
  source/legged_lab/legged_lab/tasks/locomotion/amp/config/g1_perturb \
  source/legged_lab/legged_lab/tasks/locomotion/amp/mdp/commands/nav2_recorded_velocity_command.py \
  scripts/tools/check_armhack_reference_data.py \
  ../rsl_rl/rsl_rl/runners/amp_runner.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/user/.local/bin/pytest -q \
  source/legged_lab/test/test_g1_perturb_static.py
```

当前结果：`9 passed`。

### 8.3 最小真实训练

第 6.2 节和第 6.3 节的 smoke 命令就是两个任务的最小真实训练测试。Walk 测试必须等 Stand 进程退出；专用脚本会主动阻止并发。成功条件不是仅完成 import，而是同时满足：

```text
Isaac Sim 场景创建完成
Command Manager 类型正确
policy/critic/discriminator 网络创建完成
完成 Learning iteration 0/1
日志目录生成 model_0.pt
对应 ArmHack Checkpoints 子目录生成 model_0.pt
```

### 8.4 当前真实测试记录

Stand：

```text
静止阶段 run: logs/rsl_rl/g1_stand_perturb/2026-07-14_16-39-56_armhack_stand_random_phase_smoke_20260714
设置: 4 env × 1 iteration；STATIC_ITERATIONS=1；RAMP_ITERATIONS=1
恢复: model_7999.pt policy-only；baseline KL=0.003
运行日志: csv_motion_scale=0.0000；curriculum_stage=0.0000
随机相位: csv_start_time_mean_s=198.6063；csv_start_time_std_s=137.4946
结果: 正常完成 PPO iteration 0/1，Training time 1.55 s，并保存专用 model_0.pt

20 s episode/重复 reset run: logs/rsl_rl/g1_stand_perturb/2026-07-14_16-41-38_armhack_stand_static_full_episode_smoke_20260714
设置: 4 env × 42 iteration；共 4032 个环境步；env.episode_length_s=20.0
课程: 全程 csv_motion_scale=0.0000；curriculum_stage=0.0000
终止: Episode_Termination/time_out=1.0000；base_height=0.0000；bad_orientation=0.0000
结果: 21 项奖励出现实际非零统计，完成 42 次 PPO 更新，Training time 27.83 s，保存 model_41.pt
说明: IsaacLab 会错开首批环境的 episode counter，因此此短测窗口中的 mean episode length 不直接等于 1000；配置的单 episode 上限仍为 20 s/1000 control step。

低速阶段 run: logs/rsl_rl/g1_stand_perturb/2026-07-14_16-32-07_armhack_stand_slow_motion_smoke_20260714
设置: 2 env × 1 iteration；STATIC_ITERATIONS=0；RAMP_ITERATIONS=0；SLOW_MOTION_SCALE=0.25
恢复: model_7999.pt policy-only；baseline KL=0.003
运行日志: csv_motion_scale=0.2500；curriculum_stage=2.0000
结果: 正常完成 PPO iteration 0/1，Training time 1.52 s，并保存专用 model_0.pt

两个 run 的 Observation Manager 均打印 policy shape=(96,)，仅含当前状态和上一动作；Reward Manager 均打印 21 个有效奖励项。
```

当前 model9996 入口实测：

```text
run: logs/rsl_rl/g1_stand_perturb/2026-07-14_20-03-15_armhack_stand_model9996_smoke_20260714
checkpoint: ArmHack Checkpoints/StandPerturb/2026-07-14_20-03-15_armhack_stand_model9996_smoke_20260714/model_0.pt
设置: 2 env × 1 iteration；STATIC_ITERATIONS=1；RAMP_ITERATIONS=0
恢复: BaselineModel9996/model_9996.pt policy-only；baseline KL=0.003
运行日志: 明确打印 model_9996、actor 96→29、critic input 297、csv_motion_scale=0、curriculum_stage=0
结果: 正常完成 Learning iteration 0/1，共 48 environment steps；完成 PPO 更新并保存 Stand 专用 model_0.pt；无 Traceback、Hydra、Python、CUDA 或 Isaac 启动错误

全速 run: logs/rsl_rl/g1_stand_perturb/2026-07-14_20-28-18_armhack_stand_full_speed_1x_smoke_20260714
checkpoint: ArmHack Checkpoints/StandPerturb/2026-07-14_20-28-18_armhack_stand_full_speed_1x_smoke_20260714/model_0.pt
设置: 2 env × 1 iteration；STATIC_ITERATIONS=0；RAMP_ITERATIONS=0；FINAL_MOTION_SCALE=1.0
恢复: BaselineModel9996/model_9996.pt policy-only；baseline KL=0.003
运行日志: csv_motion_scale=1.0000；curriculum_stage=2.0000
结果: 正常完成 Learning iteration 0/1，共 48 environment steps；完成 PPO 更新并保存 Stand 专用 model_0.pt；无 Traceback、Hydra、Python、CUDA 或 Isaac 启动错误
```

Stand 本机正式课程训练：

```text
run: logs/rsl_rl/g1_stand_perturb/2026-07-14_16-49-13_armhack_stand_curriculum_from_model7999_full_20260714
checkpoint: ArmHack Checkpoints/StandPerturb/2026-07-14_16-49-13_armhack_stand_curriculum_from_model7999_full_20260714/model_2999.pt
设置: 4096 env × 3000 iteration；20 s/1000 control-step episode；S3 G1
恢复: model_7999.pt policy-only；baseline KL=0.003；style reward=0；task/style lerp=1
阶段 0: iteration 0/400 的 csv_motion_scale=0、curriculum_stage=0
阶段 1: iteration 600/1000/1400 的 csv_motion_scale=0.0251/0.1251/0.2251
阶段 2: iteration 1500..2999 的 csv_motion_scale=0.25、curriculum_stage=2
随机相位: 最终 csv_start_time_mean_s=198.8300、std_s=116.2282
最终: mean episode length=1000；timeout=0.9998；base_height=0；bad_orientation=0.0002
最终 torso: roll=0.0074 rad；pitch=0.0130 rad；ang_vel_xy=0.1069 rad/s；height error=0.0092 m
训练耗时: 5588.46 s；最终 checkpoint iteration=2999，已用 env_isaaclab 成功加载
checkpoint SHA-256: 03e0f06c86363f906bbd4ceeb4e51b3897b45de345f0d066b8244bbb354e93e8
```

现有测试覆盖必须按证据强度区分：

| 检查 | 已经证明 | 尚未证明 |
|---|---|---|
| `test_g1_perturb_static.py`，`9 passed` | 配置、注册、关节映射和推进函数存在 | Isaac 运行时手臂目标确实变化、实际关节跟踪误差 |
| 随机静态 smoke | 多环境起始相位不同，阶段 0 的速度倍率确为 0 | 固定 100 episode 的统计覆盖率；该随机模式不再作为 GUI 可视化方法 |
| 低速运动 smoke | 运行时进入阶段 2，`csv_motion_scale=0.25` | 该 smoke 只有 24 control step，只推进原 CSV `0.12 s`，不能作为明显运动的可视化验收 |
| 当前全速运动 smoke | model9996 入口可在阶段 2 以 `csv_motion_scale=1.0` 完成 rollout、PPO 更新和保存 | 一轮 48 environment steps 不代表全速策略已经收敛 |
| 3000-iteration 正式训练 | 三阶段真实执行；随机相位分布、episode 存活和躯干指标完整记录 | 独立固定评估中的手臂目标/实际 `q/dq` 变化量和逐窗口完成率 |
| 旧固定第 0 帧 GUI | checkpoint、任务和 GUI 加载正常 | 姿态切换或慢速运动；该配置实际上反复播放 CSV 开头保持段 |
| 当前 `1.0x` 确定性可视化数据集 | 全量遍历后固定选出 6 个代表姿态、4 段实测全速轨迹，并固定生成 3 个姿态和 3 条实测轨迹凸组合；CSV 完整性与重建可复现性通过；历史 model2999 整段 5198 step 为 0 次 termination | 新 model9996 课程完成后的同协议对照，以及约 405 s 完整源数据压力测试 |

因此，“课程确实执行过”可以确认；“所有姿态和运动都已完成独立可视化/量化验收”不能确认。第 9 节给出补齐这两类可视化的方法。

Walk（正确 locomotion model9996 起点）：

```text
权威 actor: checkpoint/model_9996/locomotion.onnx
训练 checkpoint: ArmHack Checkpoints/WalkPerturbFinetune/BaselineLocomotionModel9996/model_9996.pt
主 smoke run: logs/rsl_rl/g1_walk_perturb/2026-07-14_19-24-50_armhack_walk_locomotion_model9996_smoke_20260714
设置: 2 env × 1 iteration；S3 G1；pos2_down；RSI=False；randomization=0
恢复: model_9996.pt policy-only；其 actor 与 locomotion.onnx 逐张量一致；baseline KL=0.003
Command Manager: Nav2RecordedVelocityCommand
配置: complex_turn；4 s；scale=(0.85,0.75,0.75)；none+在线 mirror_lr
reset: walk_pose_index_mean=1.0；walk_pose_init_max_error_rad=0.0
终止: base_height=0；bad_orientation=0；正常完成 PPO、保存 model_0.pt

具名姿态 smoke:
pos1_back: 2026-07-14_19-35-39；pose_index=0；init_error=0；base_height=0；bad_orientation=0
pos3_front: 2026-07-14_19-35-55；pose_index=2；init_error=0；base_height=0；bad_orientation=0
两者均实际加载 model_9996、完成 PPO iteration 0/1 并保存 Walk 专用 checkpoint

完整恢复 smoke: logs/rsl_rl/g1_walk_perturb/2026-07-14_19-36-10_armhack_walk_model9996_resume_smoke_20260714
设置: MODE=resume；load_policy_only=false；reset_iteration=false
结果: 从已有 Walk model_0.pt 完整恢复并完成下一次训练/保存，无 traceback
```

旧错误基线曾用 32 env 对比 `RSI=True`，`bad_orientation=0.4688`，几乎等于 RSI 采样比例；关闭后为 0。这个对照只用于解释为何专用脚本默认 `RSI=False`，不作为 model9996 的性能证据，也不代表一轮训练已经达到稳定行走性能。

Walk（错误 Stand model7999 起点，已作废）：

```text
2026-07-14_18-32-34_armhack_walk_model7999_smoke32_no_rsi_20260714
2026-07-14_18-33-30_armhack_walk_pos1_back_smoke_20260714
2026-07-14_18-33-45_armhack_walk_pos3_front_smoke_20260714
2026-07-14_18-34-20_armhack_walk_resume_mode_smoke_20260714
```

这些 run 的环境链路曾正常运行，但起点是经过 Stand 训练的 model7999，不能用于本任务训练、比较或验收。

Walk（旧实现历史记录，已作废）：

```text
run: logs/rsl_rl/g1_amp/2026-07-14_15-39-55_armhack_walk_nav2_checkpoint_smoke
训练: 2 env × 1 iteration，Training time 1.7 s
Command Manager: Nav2RecordedVelocityCommand
checkpoint 恢复: resume=false；没有加载 Nav2 checkpoint
机器人资产: s3_g1_29dof（候选 model_6596 的资产是 original g1_29dof）
专用模型: ArmHack Checkpoints/WalkPerturbFinetune/2026-07-14_15-39-55_armhack_walk_nav2_checkpoint_smoke/model_0.pt
checkpoint SHA-256: a0a58f426b3f7e33729bb620f353c9ce8aa155a6da8aa3d413c8d087e469f35a
回放: 1 env × 5 steps，正常达到 max_steps 并退出
```

上述 `model_0.pt` 仅用于验证完整执行链。它既不是 Nav2 policy，也不能代表正式训练后的策略性能；run 名称中的 `nav2_checkpoint` 与保存配置不一致，后续报告应以 `params/agent.yaml` 为准。

## 9. 可视化两个 ArmHack checkpoint

### 9.1 为什么旧 Stand GUI 只显示一个固定姿态

2026-07-14 对规范 CSV 的数值复核结果如下：

```text
总时长: 404.917792 s
CSV 第 0..5 s 最大单关节变化: 0.011457 rad
CSV 第 0..20 s 最大单关节变化: 0.011457 rad
相对第 0 帧首次达到 0.02 rad: 25.311227 s
相对第 0 帧首次达到 0.10 rad: 25.613424 s
```

旧命令同时使用 `csv_randomize_start_on_reset=False`、20 s episode 和 `0.25x`。因此一次 episode 只读取原 CSV 的第 `0..5 s`，20 s 后又回到第 0 帧；该窗口本身就是保持段，视觉上必然只有一个固定双臂姿态。该命令只能证明 GUI 和 checkpoint 能加载，不能验证随机静态姿态或手臂运动。

以 5 s 为窗口统计整条 CSV，约 `61.0%` 的窗口最大单关节变化达到 `0.10 rad`，其余窗口可能仍是保持段。随机 phase 适合训练分布统计，但不适合人工可视化验收：每次看到的内容不同，也可能连续抽到保持段。现在的 GUI 测试全部改为离线选样后顺序播放，运行时不再随机抽取姿态或轨迹。

以下命令统一使用本机正式模型：

```bash
cd /home/user/Workspace/Humanoid/Locomotion/G1-Locomotion/legged_lab
STAND_CKPT='/home/user/Workspace/Humanoid/Locomotion/G1-Locomotion/legged_lab/ArmHack Checkpoints/StandPerturb/2026-07-14_16-49-13_armhack_stand_curriculum_from_model7999_full_20260714/model_2999.pt'
```

### 9.2 确定性可视化数据集如何构造

生成脚本会遍历完整的 404.92 s、14-DoF 双臂数据，不在 Isaac Sim 运行时抽样：

```bash
python scripts/tools/build_armhack_stand_visualization_suite.py
python scripts/tools/check_armhack_reference_data.py --stand-only
```

输出目录为 `Reference Data/ArmHack/StandPerturb/TestData/ArmOnly/`。`manifest.json` 显式记录 `data_scope=arm_only_14_dof` 和 `contains_full_body_state=false`；全部 21 个可回放 CSV 的表头都严格是 `time_s + 14 个双臂关节`，不含根节点、腰、髋、膝或踝。选样和生成规则如下：

- 代表姿态：按各关节的 P5–P95 范围归一化，再用从中位姿态开始的最远点覆盖法挑选 6 个姿态，避免只挑到相邻或重复姿态；
- 代表轨迹：用 5 s 滑动窗口遍历全数据，过滤低运动窗口，再按姿态、关节幅值、速度和端点变化的联合描述选出 4 段相互至少间隔 10 s 的高运动量窗口；
- 合成姿态：固定种子 `20260714`，每次只选两组实测代表手臂姿态做有界凸插值，一次性写出 3 个双臂 CSV；不再加入逐关节随机扰动，也不会生成全身姿态；
- 合成轨迹：固定种子选取两条等长的实测 `1.0x` 双臂轨迹，按同一时刻逐帧做凸组合，写出 3 条 5 s、50 Hz 合成轨迹；速度和加速度保持在两条父轨迹的凸包内，不生成腰腿数据；
- 运行时固定 `csv_randomize_start_on_reset=False`。所谓“随机合成”只发生在离线构建阶段，固定种子、父姿态、权重和文件 SHA-256 都记录在 `manifest.json`，因此每次回放一致。

6 个代表姿态来自原 CSV 的 `261.395829`、`404.897585`、`35.204045`、`133.214841`、`386.982487` 和 `105.240482 s`。4 段代表轨迹为：

| 编号 | 原始窗口 | 主要变化关节 | 最大关节跨度 | 可视化时长 |
|---|---:|---|---:|---:|
| 1 | 36–41 s | left elbow | 1.0577 rad | 5 s，`1.0x` |
| 2 | 102–107 s | left elbow | 0.7792 rad | 5 s，`1.0x` |
| 3 | 234–239 s | right elbow | 0.9130 rad | 5 s，`1.0x` |
| 4 | 385–390 s | right shoulder pitch | 1.1224 rad | 5 s，`1.0x` |

代表轨迹不再做时间拉伸：原始 5 s 窗口仍播放 5 s，`time_stretch=1.0`、`equivalent_source_speed=1.0`。合成轨迹也由两条实测 `1.0x` 窗口逐帧混合。播放端固定 `csv_motion_scale=1.0`，因此逐项轨迹测试与新课程最终阶段速度一致。

### 9.3 一条命令顺序看完全部测试

先进入 IsaacLab 环境，然后启动默认 `MODE=all`：

```bash
conda activate env_isaaclab
cd /home/user/Workspace/Humanoid/Locomotion/G1-Locomotion/legged_lab
bash scripts/vis_g1_armhack_stand_eval.sh
```

脚本默认使用历史正式 `model_2999.pt`、单环境、正面跟随相机和实时 GUI；训练完成后应通过 `CHECKPOINT=...` 指向新的 1.0x 模型。当前完整确定性序列总时长为 103.96 s，时间线固定为：

| 可视化时间 | 内容 |
|---:|---|
| 0.00–33.98 s | 6 个代表姿态；每个保持 4 s，相邻姿态用 2 s 平滑过渡 |
| 33.98–36.98 s | 平滑段间连接 |
| 36.98–52.96 s | 3 个固定种子合成姿态 |
| 52.96–55.96 s | 平滑段间连接 |
| 55.96–81.96 s | 4 条 `1.0x` 代表轨迹，轨迹之间 2 s 平滑连接 |
| 81.96–84.96 s | 平滑段间连接 |
| 84.96–103.96 s | 3 条固定种子合成 `1.0x` 轨迹 |

若模型中途触发终止，环境会从该 CSV 的第 0 帧重新开始，这是一个真实失败，不能把 reset 后的后续画面当作已覆盖剩余测试。此时应按下一节逐项启动，直接定位失败样本。

此前 208.96 s 的回放使用旧 `0.25x` 离线拉伸数据，只作为历史执行链记录。当前 103.96 s 数据集已在 manifest 中锁定 `trajectory_speed_scale=1.0`；新性能结论必须来自本节当前文件，而不能沿用旧报告。

### 9.4 分类别或逐项复测

四类顺序播放：

```bash
MODE=representative_poses bash scripts/vis_g1_armhack_stand_eval.sh
MODE=synthesized_poses bash scripts/vis_g1_armhack_stand_eval.sh
MODE=representative_trajectories bash scripts/vis_g1_armhack_stand_eval.sh
MODE=synthesized_trajectories bash scripts/vis_g1_armhack_stand_eval.sh
```

按编号复测单个姿态或单条轨迹：

```bash
MODE=representative_pose ITEM=1 bash scripts/vis_g1_armhack_stand_eval.sh       # ITEM=1..6
MODE=synthesized_pose ITEM=1 bash scripts/vis_g1_armhack_stand_eval.sh          # ITEM=1..3
MODE=representative_trajectory ITEM=1 bash scripts/vis_g1_armhack_stand_eval.sh # ITEM=1..4
MODE=synthesized_trajectory ITEM=1 bash scripts/vis_g1_armhack_stand_eval.sh    # ITEM=1..3
```

每个单姿态 CSV 都把同一组 14-DoF 手臂目标严格保持 20 s（1001 行，50 Hz），专门用于检查“给定姿态下手臂不动，躯干能否稳定”。单轨迹模式从对应轨迹起点直接开始，不需要等待原始 CSV 前 25 s 的保持段。

### 9.5 无窗口检查和录制

无窗口执行链 smoke：

```bash
MODE=representative_trajectory ITEM=1 \
HEADLESS=True REAL_TIME=False MAX_STEPS=20 \
bash scripts/vis_g1_armhack_stand_eval.sh
```

录制完整确定性序列：

```bash
HEADLESS=True REAL_TIME=False MAX_STEPS=5200 \
bash scripts/vis_g1_armhack_stand_eval.sh \
  --video --video_length 5200
```

20 step 只能证明 checkpoint、任务、确定性 CSV 和仿真链能加载；完整 103.96 s 序列需要 5,198 个 50 Hz control step。视频默认写到 checkpoint 所在 run 的 `videos/play/`，该目录已被 `.gitignore` 排除。

### 9.6 checkpoint 同目录测试报告

`vis_g1_armhack_stand_eval.sh` 每次退出时都会自动在被测模型目录下写报告：

```text
<checkpoint 所在目录>/Test Reports/StandArmOnly/
  <checkpoint_stem>__<MODE>[_itemN].md
```

例如默认完整序列对应：

```text
ArmHack Checkpoints/StandPerturb/2026-07-14_16-49-13_armhack_stand_curriculum_from_model7999_full_20260714/
Test Reports/StandArmOnly/model_2999__all.md
```

报告记录 checkpoint 与测试 CSV 的绝对路径/SHA-256、控制步数、测试时长、termination/reset 次数、躯干 Important Metrics，以及机器人全部 29 个实际关节的统计。`平均逐步波动` 定义为同一 episode 内相邻 50 Hz 控制帧的 `mean(|q[t]-q[t-1]|)`，单位为 `rad/step`；reset 前后的跳变不计入。报告把 14 个手臂关节标为“输入关节”，把 15 个腰腿关节标为“平衡策略关节”，避免误解为测试 CSV 在控制全身。

2026-07-14 对历史 `model_2999.pt` 完成了当前 `1.0x` 测试集的 `MODE=all`、5198-step headless 实测，报告成功写入上述路径，29 个关节均有统计。整段 103.96 s 回放出现 `0` 次 termination/reset；躯干 roll/pitch 误差均值为 `0.006214/0.034164 rad`，躯干水平角速度误差为 `0.062180 rad/s`，高度误差为 `0.007344 m`。这是对 6 个代表静态姿态、3 个合成静态姿态、4 条实测 `1.0x` 轨迹和 3 条实测混合 `1.0x` 轨迹的确定性测试结果；它不代表已覆盖约 405 s 的全部源数据，也不代表尚未完成的 model9996 新课程结果。

2026-07-14 已实际执行上述 `representative_trajectory ITEM=1` 的 20-step headless smoke：确定性 CSV 路径解析成功，`model_2999.pt` 以 policy-only 方式加载，Isaac Sim 完成 20 step 后按 `max_steps` 正常退出，无 Python、Hydra、Isaac 或 CUDA 异常。该结果只确认新入口可运行，不代表 20 s 轨迹已完整通过稳定性验收。

### 9.7 Walk GUI 可视化

```bash
TASK=LeggedLab-Isaac-AMP-G1-WalkPerturbFinetune-Play-v0 \
CHECKPOINT="$WALK_CKPT" \
NUM_ENVS=1 \
HEADLESS=False \
REAL_TIME=True \
RSI_ENABLE=False \
RANDOMIZATION_STRENGTH=0 \
FOLLOW_CAMERA=True \
CAMERA_VIEW=front \
bash scripts/vis_isaacsim_g1_amp.sh \
  env.upper_body_perturbation.pose_name=pos2_down
```

该命令继续从 Nav2 CSV 采样连续速度窗口。最后一行固定 `pos2_down`；可分别改为 `pos1_back`、`pos3_front`。逐姿态人工评估时不要用 `random`，以确保结果可复现。

### 9.8 TensorBoard

专用 checkpoint 目录只保存模型；事件、配置和训练曲线仍在原日志目录：

```bash
tensorboard --logdir logs/rsl_rl/g1_stand_perturb
tensorboard --logdir logs/rsl_rl/g1_walk_perturb
```

## 10. Stand 训练代码专项审计

### 10.1 已修复并完成运行验证

| 原问题 | 当前修复 | 真实运行证据 |
|---|---|---|
| walking 父配置把奖励设为 `None` | Stand 显式重新创建全部任务奖励 | Reward Manager 打印 21 个有效 term，包括 `alive` 和 `termination_penalty=-200` |
| 全负奖励可能鼓励提前摔倒 | 加入存活、零水平速度、零 yaw 角速度和双支撑正奖励 | 两个 smoke run 均成功解析并完成 PPO 更新 |
| 随机 CSV 起点首步跳变 | reset 后直接把 14 个手臂关节写到随机目标，速度为 0 | 静止阶段真实运行无接口错误，`csv_motion_scale=0` |
| CSV 零阶保持 | 改为连续相位并对相邻帧线性插值 | 最终阶段真实运行，`csv_motion_scale=1.0000`、`curriculum_stage=2.0000` |
| 初始根状态含随机位置/速度 | Stand 的 x/y/yaw 与六维根速度全部固定为 0 | 新 run 保存的解析配置与 Event Manager 正常 |
| 早期课程混入 push | Stand 显式关闭 `push_robot`，专用脚本默认随机化强度 0 | Event Manager 中不存在 interval push |
| 当前训练起点更新 | Stand 基线改为 `BaselineModel9996/model_9996.pt`；脚本锁定 SHA-256 和大小 | 真实 smoke 明确打印 `Loaded policy-only ... model_9996.pt`，并保存新的 `model_0.pt` |

其余已确认正确的执行链保持不变：绝对关节角先按 `(target-offset)/scale` 转回 raw action；CSV 使用具名关节映射；目标裁剪到 soft limit；手臂覆盖 14 维而腿/腰仍由 policy 控制；判别器和动作正则只使用 root 与 15 个腿/腰维度。

### 10.2 无未来信息约束

课程相位、下一帧目标和轨迹速度只存在于环境内部，用于生成外部手臂 action。真实运行的策略观测仍为：

```text
base_ang_vel(3) + projected_gravity(3) + velocity_command(3)
+ joint_pos(29) + joint_vel(29) + previous_action(29) = 96
```

没有新增 phase、target、target velocity 或 look-ahead。policy 可以看到当前手臂 `q/dq`，这是反馈平衡所必需的当前状态，不属于未来信息。当前训练恢复的 `model_9996` actor 也是 96 维输入、29 维输出，结构完全兼容。

### 10.3 仍需由正式长训练回答的问题

- 当前权重能否同时获得 20 s 高完成率和更低的 torso RMS，需看正式 TensorBoard 和固定评估，冒烟测试不能回答性能问题；
- 课程目前按全局 step 固定切换，不是按完成率自动晋级；如果阶段 0 在 500 iteration 时仍未收敛，应延长 `STATIC_ITERATIONS`；
- 当前课程最终速度已经提升为原轨迹 `1.0x`；必须单独观察阶段 1 后半段和阶段 2 的完成率，确认全速扰动没有引入新的摔倒模式；
- 仍需统计目标被 soft joint limit 裁剪的比例，并记录目标/实际 arm `q/dq/ddq` 以诊断执行器跟踪；
- 20 s 训练完成率不能替代约 405 s 全轨迹固定评估；
- Stand 仍继承 AMP walking 配置并加载运动数据，功能上无误但启动慢，后续可把纯静态任务拆成更轻的 runner/config。

“增加上半身加速度惩罚”仍不是合适的 policy 修复：手臂由脚本强制覆盖，policy 无法改变它。手臂目标速度、加速度与跟踪误差应作为诊断指标；policy 奖励只约束它能改变的躯干、根节点、足底和腿/腰响应。

## 11. 本机正式训练与历史日志复核

本机已经完成当前代码快照下的 3000-iteration 三阶段课程。最终点如下；它是 iteration 2999 的在线训练统计，不等于独立固定评估，也不是最后 100 iteration 的均值：

| 本机 run | 阶段 | 平均长度/1000 | timeout | base height | bad orientation | torso roll / pitch |
|---|---|---:|---:|---:|---:|---:|
| `2026-07-14_16-49-13_armhack_stand_curriculum_from_model7999_full_20260714` | `0.25x`，stage 2 | 1000.00 | 99.98% | 0% | 0.02% | 0.0074 / 0.0130 rad |

对应 `model_2999.pt` 已在日志目录和 Stand 专用目录各保存一份，两者 SHA-256 均为 `03e0f06c86363f906bbd4ceeb4e51b3897b45de345f0d066b8244bbb354e93e8`，并已用 `env_isaaclab` 成功反序列化。TensorBoard 事件文件完整写到 iteration 2999。

以下长期训练结果来自 HEC-5090 的历史 TensorBoard 事件文件，数值为最后 100 个 iteration 的均值；最大 episode 长度为 `1000` 个控制步，即 20 s。远端 run 使用当时保存的 `params/env.yaml`/`params/agent.yaml`，与当前课程代码不是完全同一个代码快照，因此只用于说明初始化模型来源和历史对照，不替代本机新模型的固定评估。

| 远端 run | 关键设置 | 平均长度/1000 | timeout | bad orientation | torso roll / pitch |
|---|---|---:|---:|---:|---:|
| `2026-07-09_22-58-27_stand_csv_perturb` | 早期实现，历史关节映射有问题 | 857.41 | 65.73% | 34.30% | 0.0410 / 0.1133 rad |
| `2026-07-11_17-58-24_stand_csv_perturb` | 从零、style=0、task lerp=1 | 15.65 | 0% | 约 100% | 0.1215 / 0.1011 rad |
| `2026-07-13_20-26-50_stand_csv_perturb_from_nav2_stage4_policy_kl` | Nav2 policy-only、KL=0.003、style=5、lerp=0.4 | 998.15 | 99.60% | 0.39% | 0.0317 / 0.0424 rad |
| `2026-07-14_01-08-16_stand_csv_perturb_from_nav2_stage4_policy_kl` | Nav2 policy-only、KL=0.003、style=0、lerp=1 | 999.95 | 99.98% | 0.0165% | 0.0189 / 0.0219 rad |

7 月 14 日正式模型的远端绝对路径是：

```text
/home/hecggdz/workspace-zwd/legged_lab/logs/rsl_rl/g1_amp/
2026-07-14_01-08-16_stand_csv_perturb_from_nav2_stage4_policy_kl/model_7999.pt
```

结论分三层：

1. 7 月 11 日实验确实是失败策略，最终几乎全部因 `bad_orientation` 终止；它同时包含旧数据映射、从零初始化和全负奖励等多个因素，不能把失败只归因于某一个权重。
2. 从 Nav2 policy 做 policy-only 初始化后，即使 7 月 14 日关闭 style reward，训练分布内也能达到接近完整 20 s 存活。这说明“不要从随机策略硬学静态任务”是高价值结论，AMP style reward 不是唯一解。
3. 上述 `model_7999.pt` 只解释历史 `model_2999.pt` 的来源。当前训练脚本已经固定校验并加载 `BaselineModel9996/model_9996.pt`；其一轮 smoke 只验证执行链。后续 model9996 新课程需形成独立 run/checkpoint，不能把历史 `model_2999.pt` 或 smoke `model_0.pt` 当作该课程的最终性能结果。

## 12. 已实现课程与正式训练判据

### 12.1 当前课程设计

当前默认训练总长 3000 iteration：

| 阶段 | iteration | 手臂扰动 | 目标 |
|---|---:|---|---|
| A：任意姿态静止 | `0..499` | 每次 reset 随机抽取 CSV 姿态并保持 | 先学会在轨迹任一点静态站稳 |
| B：全速渐入 | `500..1499` | 速度从 `0` 线性升到 `1.0x` | 避免课程边界突然增加扰动 |
| C：全速适应 | `1500..2999` | 连续原始 `1.0x` 轨迹 | 学会对完整速度手臂运动做纯反馈补偿 |

三段在一个 run 中完成，当前入口始终从同一个 `model_9996` policy-only 初始化，style reward 为 0，baseline KL 为 `0.003`。episode 固定 20 s，初始根状态严格静止，push 和 domain randomization 在第一轮课程中关闭。

### 12.2 是否进入下一实验的判断

课程是按训练 step 切换，而不是自动看成功率，所以不能盲目等脚本跑完。建议在 iteration 400–500 检查：

- 最近窗口的 episode length 是否接近 1000 step，timeout 是否接近 100%；
- `base_height` 和 `bad_orientation` 是否接近 0；
- torso roll/pitch、角速度、高度误差和根节点 XY 漂移是否持续下降；
- `ArmHack/csv_motion_scale` 在阶段 A 必须为 0；
- 固定 100 episode 的随机静止姿态评估是否通过。

若阶段 A 未收敛，应停止当前实验并增大 `STATIC_ITERATIONS`，而不是继续把手臂加速。阶段 C 已直接训练 `1.0x`；若其失败，应先比较阶段 B 中 `0.25x/0.5x/0.75x` 附近的指标定位临界速度，再调整 ramp 时长。质量、摩擦、执行器随机化和 interval push 最后逐步加入，不与手臂速度课程同时修改。

### 12.3 明确禁止的观测改动

本任务按用户要求采用纯反馈策略。后续也不要把以下量加入 policy observation：轨迹 phase、下一帧/未来手臂目标、未来目标速度、短时 look-ahead。允许使用当前实际手臂关节位置和速度，因为它们属于机器人当前本体状态。

数据侧仍可把 405 s 文件切成命名动作、重采样到严格 50 Hz，并限制目标速度、加速度和 jerk；这些是生成外部扰动的质量控制，不会向 policy 泄漏未来信息。

## 13. 固定评估协议

### 13.1 先确认 checkpoint 身份

每次评估报告必须记录：

```text
checkpoint 绝对路径和 SHA-256
训练 run 名、iteration、Git commit
params/env.yaml 与 params/agent.yaml
轨迹文件 SHA-256
trajectory id、起始 phase、幅值/速度倍率
seed、episode 时长、随机化开关
```

### 13.2 三类评估分开报告

- 训练分布统计：20 s、随机 phase、最终 `1.0x`，统计至少 100 个 episode 的完成率和 termination 分布；它用于总体统计，不作为人工 GUI 抽样；
- 确定性日常评估：使用第 9 节的固定可视化数据集，逐项覆盖 6+3 个姿态和 4+3 条轨迹，报告每个 item 是否完成；
- 完整源数据压力测试：固定 phase、固定零初速和固定物理参数，以 `1.0x` 播放约 405 s 原始 CSV；历史 `MAX_STEPS=10123` 只覆盖约 202.46 s，不能称作完整播放。

输出至少包括：survival/time-out 比例、`base_height`/`bad_orientation` 次数、torso roll/pitch 的 RMS/P95/max、torso 水平角速度 RMS、torso 高度误差 RMS、根节点 XY 漂移、脚滑、双脚接触率，以及手臂目标/实际跟踪误差。episode return 只作为辅助量。

### 13.3 严格长时回放模板

`STAND_CKPT` 必须手工指向正式模型。回放 reset 会直接把手臂初始化到第 0 帧，所以没有默认姿态到首帧的目标跳变。为了在约 410 s 内覆盖完整原始轨迹，下面显式关闭训练课程并用 `1.0x` 速度播放；这与当前课程最终阶段速度一致，是完整数据压力测试。约 404.92 s 原始轨迹需要约 20,246 个 50 Hz control step，模板保留少量尾部余量。

```bash
STAND_CKPT='/home/user/Workspace/Humanoid/Locomotion/G1-Locomotion/legged_lab/ArmHack Checkpoints/StandPerturb/2026-07-14_16-49-13_armhack_stand_curriculum_from_model7999_full_20260714/model_2999.pt'

TASK=LeggedLab-Isaac-AMP-G1-StandPerturb-Play-v0 \
CHECKPOINT="$STAND_CKPT" \
NUM_ENVS=1 \
HEADLESS=True \
MAX_STEPS=20500 \
REAL_TIME=False \
RSI_ENABLE=False \
RANDOMIZATION_STRENGTH=0 \
FOLLOW_CAMERA=True \
CAMERA_VIEW=front \
bash scripts/vis_isaacsim_g1_amp.sh \
  --video \
  --video_length 20500 \
  env.episode_length_s=410.0 \
  env.upper_body_perturbation.csv_randomize_start_on_reset=False \
  env.upper_body_perturbation.csv_curriculum_enabled=False \
  env.upper_body_perturbation.csv_curriculum_motion_scale=1.0 \
  env.upper_body_perturbation.csv_loop=False \
  env.upper_body_perturbation.csv_initialize_joint_state_on_reset=True
```

## 14. Stand 修改边界与当前结论

此前已修改 Stand 环境、奖励函数、Stand 配置、runner 配置、静态测试和两份文档，并新增 `scripts/train_g1_armhack_stand.sh`。当前初始化 `model_9996.pt` 保存在忽略 Git 的 `ArmHack Checkpoints/StandPerturb/BaselineModel9996/`；本轮数据构建、校验和回放报告只改变 Stand 评估数据与入口，没有增加未来信息观测。

结论：奖励继承、全负 return、随机起点首步跳变、随机根初速和早期 push 均已在代码层修复；当前 Stand 训练入口已从 model9996 完成真实 PPO smoke。历史 3000-iteration 课程及 `model_2999.pt` 仍作为既有结果保留。旧的固定第 0 帧 GUI 已被严格 14-DoF 双臂测试集替代：运行时不采样，代表样本和固定种子合成样本可顺序或逐项复现；每次回放自动在 checkpoint 同级保存 29 关节波动报告。构建与数据校验通过不等于策略性能通过，最终结论仍需结合逐项报告和第 13 节指标。

## 15. Walk 当前实现与历史审计对照

### 15.0 2026-07-14 已完成修复

本轮已经完成以下代码修改：

- 新增 `G1WalkPerturbAmpEnv`，Walk/Stand 使用不同 Gym entry point；
- reset 在首个 observation 前同步手臂 `q/dq`、position target 和 action history；
- Walk 直接继承 Nav2 配置，锁定 Stage-4 command/reward，删除无效 command curriculum；
- 原始 Nav2 CSV 在线生成 `mirror_lr`，保持源文件和 Git ignore 不变；
- 新增 `scripts/train_g1_armhack_walk.sh`，区分 `MODE=init` 和 `MODE=resume`；
- 固定 S3 G1、`locomotion.onnx`/`model_9996.pt` actor 一致性、低学习率、style=0 和 baseline KL；
- Walk 独占 `logs/rsl_rl/g1_walk_perturb`，检测到 Stand 训练会拒绝启动；
- 静态回归覆盖注册、继承关系、command、奖励、pose 初始化代码、脚本和 checkpoint，当前为 `9 passed`。

下面 15.1–15.10 保留的是“修改前审计快照”，用于解释为何修改；其中“当前实现”“下一轮”等措辞均指旧代码状态，不能覆盖本节及第 3、6 节记录的新实现。

### 15.1 当前执行链和已确认正确的部分

```text
train_g1_amp.sh
  -> scripts/rsl_rl/train.py
  -> LeggedLab-Isaac-AMP-G1-WalkPerturbFinetune-v0
  -> G1WalkPerturbFinetuneEnvCfg + Walk runner cfg
  -> G1PerturbAmpEnv.step()
  -> 29 维 policy action 中 14 维手臂 action 被 pose target 覆盖
  -> JointPositionAction -> Isaac Sim
```

以下部分已经确认正确：

- Gym Train/Play 注册和 Python 执行入口存在；
- `g1_arm_pose_set.json` 的左右各 7 维数据会正确转换成 G1 action 使用的左右交错 14 维顺序；
- 候选 Nav2 `model_6596.pt` 和当前 Walk policy 都是 actor `96 -> 29`、critic input `297`，网络形状兼容；
- 参考数据路径相对于 `legged_lab` 解析；
- policy 观察当前完整 29DoF 状态，不含未来信息；
- discriminator 关节项和 torque/acc/action-rate 正则已经按名字限制到腿和腰。

这些检查只能排除“文件不存在、关节重排错误、网络维度不匹配”等问题，不能证明当前训练定义正确。

### 15.2 P0：现有训练实际没有从 Nav2 开始

`train_g1_amp.sh` 默认 `RESUME=False`。runner 配置中的 `load_policy_only=True` 本身不会加载任何文件；只有 `RESUME=True` 后，`train.py` 才解析 checkpoint 并调用 `AMPRunner.load()`。

三个本机 Walk smoke 的保存配置均为 `resume: false`。最新一次是：

```text
logs/rsl_rl/g1_amp/
2026-07-14_15-39-55_armhack_walk_nav2_checkpoint_smoke/params/agent.yaml

resume: false
load_policy_only: true
```

因此其中 `model_0.pt` 是随机初始化网络完成一次 PPO update 后的产物，不是 Nav2 微调。正式入口必须把“确实加载 checkpoint”作为硬前置条件，训练日志同时出现实际路径和 `Loaded policy-only AMP checkpoint` 才算成功。

候选本机基线：

```text
路径：logs/rsl_rl/g1_amp/
      2026-06-01_04-23-10_finetune_nav2_stage4_complex_only/model_6596.pt
大小：16,202,421 bytes
SHA-256：56cdb6aa66cbab53e9fa5b2bf312b3ea38e14164ebbb1f1e8850d9d003649f82
iteration：6596
asset：original g1_29dof
```

该 run 最后一个 TensorBoard 点为：mean episode length `1000/1000`、timeout `0.9935`、base-height termination `0.00085`、bad-orientation termination `0.00572`。这些训练统计用于证明它是一个合理候选起点；开始修改前仍应在当前机器固定回放并保存对照结果。

### 15.3 P0：固定姿态 reset 首步冲击

当前 `_reset_idx()` 对 `source="pose_set"` 只调用 `_sample_pose_targets()`，没有把选中姿态写到实际机器人关节状态。第一次应用目标发生在下一次 `step()` 的 action overwrite 中。相对于 G1 默认手臂姿态：

| pose | 最大单关节初始差 | 14DoF 差值 L2 | 最大差对应关节 |
|---|---:|---:|---|
| `pos2_down` | `0.1469 rad` | `0.3193 rad` | wrist roll |
| `pos1_back` | `1.0300 rad` | `2.2417 rad` | wrist pitch |
| `pos3_front` | `1.4600 rad` | `2.5000 rad` | elbow |

`pos3_front` 的肘从默认 `0.97` 到 `-0.49 rad`；按 `Kp=40` 估算会请求约 `-58.4 Nm`，超过 25 Nm 上限。`pos1_back` 的 wrist pitch 会请求约 `-41.2 Nm`，而该执行器上限只有 5 Nm。实际仿真会限幅，但限幅本身就是很强的隐藏冲击。

正确顺序应是：先选定 pose，再在第一个 observation 前把 14 个手臂 `q` 写成目标、`dq` 写成 0，并同步 position target 和 composed `last_action`。完整 episode 内不再切换 pose。

### 15.4 当前 Walk 与候选 Nav2 基线的实际差异

不能用“都调用 `Nav2RecordedVelocityCommandCfg`”概括两者。以下数值来自两个 run 已保存的 `params/env.yaml` / `params/agent.yaml`：

| 配置 | Nav2 `model_6596` | 当前 Walk smoke |
|---|---|---|
| 机器人资产 | original `g1_29dof` | `s3_g1_29dof` |
| 父任务语义 | Nav2 fine-tune | Command-Balanced V3 + Nav2 command 替换 |
| AMP motion | `accad_g1used_50hz` | `command_balanced_directional_50hz` |
| command scenario | `complex_turn` | 全部 family |
| command window | 4 s | 2 s |
| command scale | `(0.85, 0.75, 0.75)` | `(0.70, 0.55, 0.55)` |
| augmentation | `none,mirror_lr` | `none` |
| MPPI 权重 | `1.5` | `1.15` |
| linear/yaw tracking | 普通 tracking，`1.8/1.5` | adaptive tracking，`1.25/1.25` |
| torso linear/yaw tracking | `1.0/0.65` | `0.45/0.30` |
| `flat_orientation_l2` | `-1.0` | `None` |
| torso roll/pitch | `-0.04` | `-1.8` |
| torso vertical velocity | `-0.01` | `-0.38` |
| `termination_penalty` | `-200` | `None` |
| V3 directional rewards | 无 | speed-floor/double-air/double-stance/leak |
| PPO learning rate | `3e-5` | `1e-4` |
| AMP grad penalty | `20` | `10` |
| task/style lerp | `0.36` | `0.40` |
| baseline KL | 开启，`0.006` | 关闭 |

因此当前 Walk 一次改变了 asset、initial policy、AMP demo、command 分布、tracking、gait shaping 和稳定性权重。任何失败都无法单独归因到固定手臂姿态。

只把父类改成当前源码中的 `G1AmpNav2FinetuneEnvCfg` 也不自动等于 `model_6596` 的真实训练配置，因为该 checkpoint 的 Stage-4 参数包含额外的启动 override。实现时应把 checkpoint 配套的已解析 YAML 当作基线真值，逐项建立受测试保护的 Walk 配置。

### 15.5 奖励、AMP 和动作优化问题

第一版 Walk 应保持 Nav2 任务奖励，只做两类必要变化：

1. 删除策略无法控制的手臂 deviation/style penalty，并把 torque/acc/action-rate 正则限制到 15 个可控腿腰关节；
2. discriminator 若屏蔽手臂，需要新建 lower-body discriminator，旧 discriminator 权重不能直接复用。

不应在第一版同时把 torso penalty 放大几十倍。已有 Nav2 policy 已经具备 gait，初期更稳妥的做法是 style reward 暂设为 0、task/style lerp 设为 1，使用较低学习率和冻结 Nav2 baseline KL 保持行为；单姿态完成后，再单变量验证 lower-body AMP 是否改善仿人性。

当前 actor 仍输出 29 维以兼容 checkpoint，但 14 维手臂 action 在物理 step 前被丢弃，PPO 却仍将它们放入 log-prob、entropy、symmetry 和 KL。这会增加无效梯度方差。P0 阶段可以先靠 KL/低熵/低学习率减轻；后续应评估只对 15 个可控 action 计算 loss 的 mask，或冻结 arm output head，同时保持部署接口为 29 维。

### 15.6 recorded-command curriculum 是无效的

当前 `lin_vel_cmd_levels` 和 `ang_vel_cmd_levels` 只修改 command cfg 的 `ranges`。`Nav2RecordedVelocityCommand._scaled_command()` 实际执行的是：

```text
dataset[row] * command_scale -> command_clip_min/max
```

它不读取动态变化后的 `ranges`，所以第 109–123 行配置的 curriculum 不会改变训练命令。应删除这两个 term，或实现真正调节 `command_scale`、scenario filter/weight 或 clip 的 recorded-command curriculum，并为每阶段记录实际采样分布。

### 15.7 分阶段修改与训练方案

按以下顺序实现，每一步只改变一个主要变量：

| 阶段 | 双臂条件 | command/随机化 | 目标与晋级条件 |
|---|---|---|---|
| A0：Nav2 对照 | 原策略控制双臂 | checkpoint 原分布；固定 seed；无新随机化 | 在当前机器复现候选基线，冻结评估窗口和指标 |
| A1：接管零差异 | 固定为原默认/基线臂位 | 与 A0 相同 | 验证 action overwrite、reset 和部署链本身不使 gait 退化 |
| B：最小姿态 | `pos2_down`，episode 内固定 | 与 A0 相同；低 lr + baseline KL | 每个固定窗口的 20 s 完成率和速度误差接近 A0 |
| C1：靠后姿态 | `pos1_back`，episode 内固定 | 同上 | 单姿态逐项验收，不用混合平均掩盖失败 |
| C2：靠前姿态 | `pos3_front`，episode 内固定 | 同上 | 重点检查肘/腕饱和、torso 和足底响应 |
| D：三姿态混合 | reset 时按名字均衡采样 | 仍保持同一 Nav2 契约 | 三种 pose 各自成功率均达标 |
| E：鲁棒性扩展 | 三姿态 | 扩展 scenario、加入 mirror，再逐步恢复质量/摩擦/push | 每次只新增一种随机化并与 D 成对比较 |

如果最终部署资产是 S3，而候选 checkpoint 是 original G1，应在 A0 前插入一个独立的资产迁移阶段：先在没有 ArmHack 手臂覆盖时让 S3 Nav2 达标，不能把资产迁移和固定手臂同时训练。

### 15.8 已实现的专用训练脚本契约

当前已新增 `scripts/train_g1_armhack_walk.sh`。首次初始化模式会：

- 强制检查 Nav2 checkpoint 存在、大小和 SHA-256；
- 强制声明并校验 `ROBOT_ASSET`，不得静默使用通用脚本默认值；
- 强制 `RESUME=True`、policy-only load、iteration 从 0 开始；
- 用同一 checkpoint 建立 frozen baseline KL；
- 打印 checkpoint 绝对路径、asset、pose name、command profile、学习率、style 和 KL；
- 缺少任一必要参数时退出，不开始随机训练；
- 提供独立 `MODE=resume`，完整恢复已有 Walk optimizer/discriminator/iteration。

脚本和 reset 修复均已完成；正式训练只使用第 6.3/6.4 节的新命令，历史通用脚本命令继续禁止使用。

### 15.9 固定评估矩阵和测试缺口

每个正式 checkpoint 都要在完全相同的 command window/seed 上评估以下矩阵：

```text
Nav2 baseline + 原双臂控制
Nav2 baseline + default pose overwrite
Nav2 baseline + pos1/pos2/pos3 overwrite（未微调）
Walk checkpoint + pos1
Walk checkpoint + pos2
Walk checkpoint + pos3
Walk checkpoint + 三姿态均衡混合
```

每一格至少输出 20 s completion、timeout/base-height/bad-orientation、`vx/vy/wz` MAE/RMSE、torso roll/pitch RMS/P95、高度误差、feet slide、double-air ratio、gait symmetry、arm target/actual error 和 actuator saturation ratio。逐姿态结果必须单列。

现有 `test_g1_perturb_static.py` 主要是源码字符串/数据契约断言。下一轮需补：

- pose 在第一个 policy observation 前已经写入实际 arm `q/dq`；
- 同一 episode 内 pose 不变，并能按名称固定；
- Walk 必需奖励不为 `None`，与 Nav2 基线的允许差异只有白名单项；
- 专用脚本在 checkpoint 缺失、哈希错误或 asset 不匹配时失败；
- 日志确认 checkpoint 实际加载，而不是只检查 `load_policy_only=True` 字符串；
- 逐姿态 headless 集成回放和固定窗口指标导出。

### 15.10 部署和 ONNX 边界

当前 ONNX exporter 只导出 actor `96 -> 29`，不会把 `G1PerturbAmpEnv` 中的固定手臂 action override 一起导出。部署端必须复制相同语义：按关节名覆盖 14 个手臂目标，并把组合后实际施加的 action 作为下一帧 `last_action`。若直接把 Walk actor 当普通 Nav2 actor 使用，或只在 Isaac 环境里覆盖手臂，仿真训练与部署输入历史会不一致。

以上是修复前的审计结论。当前代码已经按 15.0 实施；新的训练与测试状态以第 1、6.3 和 8.4 节为准。即使 smoke 完成，也只证明训练链和 checkpoint 加载正确，不能替代逐姿态 20 s 完成率与速度跟踪验收。
