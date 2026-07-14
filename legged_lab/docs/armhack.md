# ArmHack 实习任务原始记录

> 本文完整收录附件中的任务说明、训练记录、常用命令与技术说明。附件中无法导出的飞书内容和图片保留原占位标记。

## 2026-07-14 当前实现更新

附件以下内容作为历史任务与训练记录保留。当前可运行实现以本节和 `armhack_train.md` 为准。

### 统一参考数据

两个任务的数据已集中到仓库相对目录：

```text
legged_lab/Reference Data/ArmHack/
├── StandPerturb/
│   ├── raw/g1_full_body_motion_sdk_50hz.csv
│   └── g1_arm_trajectory_named_50hz.csv
└── WalkPerturbFinetune/
    └── g1_arm_pose_set.json
```

- Stand 原始数据来自 `/home/user/Workspace/whole_body_joints_20260708_143133.csv`。转换脚本按用户给定的 SDK `q0..q28` 顺序提取 `q15..q28`，生成只包含 `time_s + 14 个具名手臂关节` 的规范训练 CSV。下肢数据不会作为脚本目标施加。
- Walk 的三组双臂姿态保存在 JSON 中，单位为弧度，按左/右各 7 个关节记录；加载时校验单位、顺序、数量、名称唯一性和数值有限性，再转换成环境使用的左右交错 14 维顺序。
- 所有训练路径均相对于 `legged_lab` 项目目录解析，不再包含 `/home/hecggdz/...` 等机器绝对路径。

### 动态任务命令说明

当前动态任务不再依赖缺失的 Nav2 录制大 CSV，而是使用 `UniformVelocityCommandCfg` 生成与 Nav2 policy 接口一致的 `vx/vy/wz` 命令：

```text
vx: [-0.10, 0.30] m/s
vy: [-0.12, 0.12] m/s
wz: [-0.30, 0.30] rad/s
每 2.0 秒重采样，2% 环境为站立命令
```

因此它训练的是“可接收 Nav2 三维速度指令的行走抗手臂姿态扰动策略”，但训练分布不再声称来自真实 Nav2 recorded dataset。

### 当前环境和验证状态

本机统一使用：

```bash
source /home/user/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
cd /home/user/Workspace/Humanoid/Locomotion/G1-Locomotion/legged_lab
```

环境已验证为 Python `/home/user/anaconda3/envs/env_isaaclab/bin/python`、IsaacLab `0.54.2`、`rsl-rl-lib 3.2.0`、PyTorch `2.7.0+cu128`，CUDA 和 RTX 4090 可用。

最终代码已分别完成 Stand 和 Walk 的 `2 env × 1 iteration` 真实训练：两个任务均正常退出、生成 `model_0.pt`，日志中无 traceback、数据路径错误、Isaac/Python 版本错误或 Nav2 数据缺失错误。

## 附件历史原文

实习任务
1.任务目标
静止任务：目标是训练机器人在速度指令恒为 0 的静止站立状态下，面对双臂外部姿态扰动时保持稳定。
动态任务：目标是训练机器人在Nav2 速度指令跟踪/行走任务中，双臂被固定姿态扰动时仍然稳定跟踪速度命令。
2.实现流程与框架
2.1框架
legged_leb/source/legged_leb/legged_lab
     |--envs/
     |    |--g1_perturb_env.py   #包含两个类 UpperBodyPerturbationCfg类：csv路径等上身轨迹相关配置，统一扰动环境类
     |    |                       G1PerturbAmpEnv类：统一扰动环境类，拦截策略输出，只让下肢由策略控制，把上肢动作替换成轨迹扰动。
     |    |--__init__.py  #对应类暴露给外部配置
     |--tasks/locomotion/amp/config/
     |    |--g1_pertub/    #新增任务配置文件夹
     |    |      |--g1_stand_pertub_env_cfg.py  #新增静态任务对应配置，速度命令为零，关闭速度跟踪奖励，强化躯干姿态、身高、竖直速度等平衡奖励
     |    |      |--g1_walk_pertub_env_cfg.py  #新增动态任务配置类，接入3组手臂持物姿态。判别器观测只保留下肢和腰部。惩罚奖励只约束下肢，从Nav2 csv读取对应速度指令
     |    |      |--agent/rsl_rl_ppo_cfg.py    #新增两个任务对应rsl-rl的配置
     |--mdp/reward.py    #在原奖励基础上更改，增加action_rate_l2_selected，只给下肢惩罚约束
legged_lab/scripts/tools/visualize_g1_csv_full_body_motion.py   #测试csv数据，将机器人固定住，将csv关节角施加给机器人，并录制视频
rsl-rl/rsl-rl/runners/amp_runners.py   #新增load_policy_only支持，加载旧策略进行微调

更改部分训练和测试脚本代码，增加录制视频视角变化的接口。
2.2实现流程
policy的输入和输出
输入为原AMP训练96维输入：base_ang_vel(3) + projected_gravity(3) + velocity_command(3) + joint_pos_rel(29) + joint_vel_rel(29) + last_action(29)输出为29个关节目标位置。但双臂运动会被覆盖掉。
stand perturb 的双臂不是固定 pose，而是从 CSV 读取的时间序列姿态；动态任务在训练时每一个环境从三组中随机抽取一组上肢动作。
AMP判别器
两个任务均看，下肢关节状态和根状态（速度，角速度），屏蔽掉上肢
奖励
静态任务：关闭了速度跟踪、步态、摆腿、directional 等行走奖励；保留并加强稳定性项，例如 flat_orientation_l2=-2.0、torso_roll_pitch_l2=-4.0、torso_vertical_velocity_l2=-0.60、torso_height_band_l2=-1.20、lin_vel_z_l2=-1.20、ang_vel_xy_l2=-0.20、feet_slide=-0.20、termination_penalty=-250.0
动态任务：保留行走/速度跟踪主任务，使用 Nav2 command 跟踪；同时通过 common 配置让 torque、acc、action-rate 只约束下肢，去掉 joint_deviation_arms 和 arm_style_prior。walk 里还调了稳定性权重，例如 torso_roll_pitch_l2=-1.8、torso_vertical_velocity_l2=-0.38、torso_height_band_l2=-0.48、feet_slide=-0.20

3.训练记录
7月9日

暂时无法在飞书文档外展示此内容
第一次训练，初始的训练上半身不是这个，而是那个错误的未重新排序的关节角度。测试时换成更正后的角度，发现会摔倒。可见，目前的csv脚本貌似无法覆盖全部的扰动情况。换成其他上半身动作，那策略可能无法使用

[图片]
[图片]
[图片]
静态下肢强化学习，采用默认的的AMP风格奖励，启动命令如下：
CONDA_ENV_NAME=env_leglab \
TASK=LeggedLab-Isaac-AMP-G1-StandPerturb-v0 \
RUN_NAME=stand_csv_perturb \
NUM_ENVS=4096 \
MAX_ITERATIONS=8000 \
HEADLESS=True \
QUIET_TERMINAL=False \
RSI_ENABLE=False \
bash scripts/train_g1_amp.sh
虽然，在静态任务的环境配置中关掉了AMP风格奖励，但是在train_g1_amp.sh中会被默认参数覆盖掉
7月11日

暂时无法在飞书文档外展示此内容
问题：重头开始学失去AMP风格奖励，直接摔倒
暂时无法在飞书文档外展示此内容
这是错误的csv关节顺序，也是前两次训练时的关节顺序
[图片]
[图片]
[图片]
静态下肢强化学习，关掉AMP奖励，启动命令如下：
CONDA_ENV_NAME=env_leglab \
TASK=LeggedLab-Isaac-AMP-G1-StandPerturb-v0 \
RUN_NAME=stand_csv_perturb \
STYLE_REWARD_SCALE=0.0 \
TASK_STYLE_LERP=1.0 \
NUM_ENVS=4096 \
MAX_ITERATIONS=8000 \
HEADLESS=True \
QUIET_TERMINAL=False \
RSI_ENABLE=False \
bash scripts/train_g1_amp.sh
一共训练了5400轮，最后单轮存活步数都没有过20
7月13日
暂时无法在飞书文档外展示此内容
更新训练后的静态视频，上半身还是有一定扰动。后续可增大上半身加速度惩罚
暂时无法在飞书文档外展示此内容
更正后录制csv文件全身关节视频
之前的训练和测试时，csv关节的对应不对。
该项目中G1的关节共有两种配置顺序：SDK / GMR / motor 顺序：底层电机顺序；Lab / policy / action 顺序：这是 IsaacLab AMP policy 使用的顺序。
csv文件的顺序为前一种，而仿真为后一种，所以需要重新映射。之前的训练未考虑这一点。故上半身姿态错误。现已更正。
[图片]
[图片]
[图片]

总体延续7月9日的训练思路，但是前两次训练都没有续训原来的Nav2策略。这次在启动命令中显式指定。启动命令如下
CONDA_ENV_NAME=env_leglab \
TASK=LeggedLab-Isaac-AMP-G1-StandPerturb-v0 \
RUN_NAME=stand_csv_perturb_from_nav2_stage4_policy_kl \
RESUME=True \
LOAD_RUN=2026-06-01_04-23-10_finetune_nav2_stage4_complex_only \
CHECKPOINT=model_6596.pt \
BASELINE_KL_ENABLE=True \
BASELINE_KL_CHECKPOINT=logs/rsl_rl/g1_amp/2026-06-01_04-23-10_finetune_nav2_stage4_complex_only/model_6596.pt \
BASELINE_KL_SCALE=0.003 \
BASELINE_KL_MIN_STD=1e-4 \
STYLE_REWARD_SCALE=0.0 \
TASK_STYLE_LERP=1.0 \
NUM_ENVS=4096 \
MAX_ITERATIONS=4000 \
HEADLESS=True \
QUIET_TERMINAL=False \
RSI_ENABLE=False \
EXTRA_HYDRA_ARGS='agent.load_policy_only=True agent.reset_iteration_on_policy_only_load=True' \
bash scripts/train_g1_amp.sh
一共训练了4000轮，下一步可关闭风格奖励试一下（个人认为在续训的思路下应该影响不大）
4.常用命令
cd /home/hecggdz/workspace-zwd/legged_lab

ISAACLAB_PYTHON=/home/hecggdz/miniconda3/envs/env_leglab/bin/python \
ROBOT_ASSET=s3_g1_29dof \
TASK=LeggedLab-Isaac-AMP-G1-StandPerturb-Play-v0 \
CHECKPOINT="$CKPT" \
NUM_ENVS=1 \
HEADLESS=True \
MAX_STEPS=10123 \
REAL_TIME=False \
RSI_ENABLE=False \
RANDOMIZATION_STRENGTH=0 \
FOLLOW_CAMERA=True \
CAMERA_VIEW=front \
CAMERA_DISTANCE=5.5 \
CAMERA_HEIGHT=0.7 \
CAMERA_TARGET_HEIGHT=0.7 \
CAMERA_SMOOTHING=0.25 \
bash scripts/vis_isaacsim_g1_amp.sh \
  --video \
  --video_length 10123 \
  env.episode_length_s=260.0 \
  env.upper_body_perturbation.csv_randomize_start_on_reset=False \
  env.upper_body_perturbation.csv_loop=False \
  env.terminations.time_out=null
正面视角录制相关视频的shell语法，设置episode长度，防止中途切换reset切换视角
cd /home/hecggdz/workspace-zwd/legged_lab
/home/hecggdz/miniconda3/envs/env_leglab/bin/python scripts/tools/visualize_g1_csv_full_body_motion.py \
  --headless \
  --video \
  --csv_joint_order sdk \
  --camera_distance 5.5 \
  --camera_height 0.7 \
  --camera_target_height 0.7
静态任务下csv全身运动的视频录制
scp hecggdz@172.28.162.15:/home/hecggdz/workspace-zwd/legged_lab/logs/rsl_rl/g1_stand_perturb/2026-07-11_17-58-24_stand_csv_perturb/videos/play/rl-video-step-0.mp4 D:\科创\G1_AMP_Locomotion\训练video
训练相关命令
cd /home/hecggdz/workspace-zwd/legged_lab

CONDA_ENV_NAME=env_leglab \
TASK=LeggedLab-Isaac-AMP-G1-StandPerturb-v0 \
RUN_NAME=stand_csv_perturb_from_nav2_stage4_policy_kl \
RESUME=True \
LOAD_RUN=2026-06-01_04-23-10_finetune_nav2_stage4_complex_only \
CHECKPOINT=model_6596.pt \
BASELINE_KL_ENABLE=True \
BASELINE_KL_CHECKPOINT=logs/rsl_rl/g1_amp/2026-06-01_04-23-10_finetune_nav2_stage4_complex_only/model_6596.pt \
BASELINE_KL_SCALE=0.003 \
BASELINE_KL_MIN_STD=1e-4 \
STYLE_REWARD_SCALE=0.0 \
TASK_STYLE_LERP=1.0 \
NUM_ENVS=4096 \
MAX_ITERATIONS=8000 \
HEADLESS=True \
QUIET_TERMINAL=False \
RSI_ENABLE=False \
EXTRA_HYDRA_ARGS='agent.experiment_name=g1_amp agent.load_policy_only=True agent.reset_iteration_on_policy_only_load=True' \
bash scripts/train_g1_amp.sh
5.技术文档
G1 Perturb Policy Tasks 说明
当前有两个上半身扰动相关任务：
- LeggedLab-Isaac-AMP-G1-StandPerturb-v0
- LeggedLab-Isaac-AMP-G1-WalkPerturbFinetune-v0
它们的共同思路是：保持原 G1 29DoF policy 接口不变，但把双臂 action 从 policy 输出中接管出来，改由脚本姿态/轨迹驱动，让 policy 专注学习下肢和腰部如何在上半身扰动下保持稳定。
共同技术路线
两个任务都使用 G1PerturbAmpEnv。
policy 输入保持原 AMP G1 输入结构，约 96 维：
base_ang_vel(3)
+ projected_gravity(3)
+ velocity_command(3)
+ joint_pos_rel(29)
+ joint_vel_rel(29)
+ last_action(29)
policy 输出也保持原 29 维 joint position action：
G1_LOCOMOTION_JOINT_NAMES 对应的 29 个关节目标
但执行前 env 会做 action 替换：
policy 输出 29 维 action
-> 保留下肢/腰部 15 维
-> 覆盖双臂 14 维
-> 送入 JointPositionAction
这样做的好处是 checkpoint 维度兼容原 G1/Nav2 policy，不需要改 actor 网络结构，也不需要重新定义部署接口。
AMP 判别器方面，两个任务都会把 discriminator 的关节观测限制到下肢：
disc root 信息：保留
disc joint_pos/joint_vel：只看下肢
disc_demo ref_joint_pos/ref_joint_vel：只取下肢
所以更准确地说：AMP discriminator 不看双臂关节，但仍看 root 姿态、root 速度、root 角速度和下肢关节。
这样避免 CSV 或 pose-set 的双臂动作污染 gait/style 判别。

---
任务 1：StandPerturb
任务目标：
训练机器人在速度指令恒为 0 的静止站立场景下，面对双臂真实轨迹扰动时仍保持稳定站立。
策略能力目标：
- 原地站稳，不走路。
- 双臂按照录制 CSV 轨迹运动时，身体不倒、不大幅摇晃。
- 下肢和腰部主动补偿上半身运动带来的重心扰动。
- 保持躯干 upright、高度稳定、脚底不滑。
双臂运动来源：
- 来源是 whole_body_joints_20260708_143133.csv。
- CSV 包含 q0..q28 全身关节轨迹。
- 目前按 sdk / Unitree GMR 顺序读取，再映射到 IsaacLab Lab/action 顺序。
- env 只取其中 14 个上肢关节作为目标姿态。
- reset 时会随机 CSV 起点，提高扰动多样性。
速度指令：
lin_vel_x = 0
lin_vel_y = 0
ang_vel_z = 0
rel_standing_envs = 1.0
这个任务不是 Nav2 行走任务，而是纯站立抗扰动任务。
奖励设计：
关闭行走相关奖励：
- 关闭线速度跟踪。
- 关闭 yaw 速度跟踪。
- 关闭 torso velocity tracking。
- 关闭 feet air time。
- 关闭 swing clearance。
- 关闭 gait timing。
- 关闭 directional speed / double-air / backward gait 类奖励。
加强稳定性奖励/惩罚：
- flat_orientation_l2
- torso_roll_pitch_l2
- torso_ang_vel_xy_l2
- torso_vertical_velocity_l2
- torso_height_band_l2
- torso_specific_force_xy_l2
- lin_vel_z_l2
- ang_vel_xy_l2
- feet_slide
- joint_deviation_hip
- joint_deviation_waist
- termination_penalty
AMP 设置：
- stand 任务里 style_reward_scale = 0.0
- task_style_lerp = 1.0
所以 StandPerturb 实际主要靠任务奖励训练稳定性，AMP 判别器可不再提供 style reward。
技术定位一句话：StandPerturb 是“上半身按真实 CSV 轨迹运动时，下肢策略学习原地抗扰动站稳”的任务。

---
任务 2：WalkPerturbFinetune
任务目标：
在已有 G1/Nav2 行走策略基础上，训练机器人在双臂被占用或摆成固定扰动姿态时，仍能稳定跟踪 Nav2 速度指令。
策略能力目标：
- 继续跟踪 Nav2 cmd_vel。
- 能前进、侧移、转向等。
- 双臂不再由 policy 控制，而是被设置成若干预设姿态。
- 下肢/腰部需要适应上半身姿态变化，保持 gait 稳定。
- 减少双脚跳、脚滑、躯干晃动、高度不稳等问题。
双臂运动来源：
- 使用 pose_set，不是 CSV。
- 当前有 3 组双臂姿态。
- reset 时从 pose set 里采样一组。
- 该姿态作为上肢目标，覆盖 policy 输出中的双臂 action。
速度指令：
- 使用 Nav2RecordedVelocityCommandCfg
- 数据来自 Nav2 recorded cmd_vel 数据。
- rel_standing_envs = 0.02
- 大部分环境是运动指令，少部分站立指令。
奖励设计：
保留 Nav2 行走/速度跟踪目标，同时加强稳定性：
- 保留 track_lin_vel_xy_exp
- 保留 track_ang_vel_z_exp
- 保留 torso velocity tracking
- 保留 gait/feet 相关约束
- 加强 torso_roll_pitch_l2
- 加强 torso_vertical_velocity_l2
- 加强 torso_height_band_l2
- 保留/调整 feet_slide
- action rate、torque、acc 只约束下肢相关 action/joints
同时取消双臂风格约束：
- joint_deviation_arms = None
- arm_style_prior = None
因为双臂已经被脚本姿态接管，不能再要求 policy 自己生成 arm style。
AMP 设置：
- WalkPerturbFinetune 仍保留 AMP 训练结构。
- 判别器关节项只看下肢。
- 这样 AMP 仍可约束 gait/style，但不会因为双臂 pose-set 和参考 motion 不一致而惩罚策略。
Nav2 checkpoint 续训：
WalkPerturbFinetune 默认满足 Nav2 policy 续训路线：
experiment_name = "g1_amp"
load_policy_only = True
它适合从已有 Nav2/G1 AMP checkpoint 只加载 policy 权重，然后在双臂扰动条件下继续微调。
技术定位一句话：WalkPerturbFinetune 是“在 Nav2 行走策略基础上，加入双臂姿态扰动后的行走鲁棒性微调任务”。

---
两者区别
暂时无法在飞书文档外展示此内容
最终总结
StandPerturb 解决的是：
双臂按真实轨迹运动时，机器人能不能原地稳定站住。
WalkPerturbFinetune 解决的是：
已有 Nav2 行走策略在双臂被占用/固定姿态扰动时，能不能继续稳定跟踪速度指令。
