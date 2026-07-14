# ArmHack 任务说明

> 本文分为 Walk 动态任务、Stand 静态任务和附件历史原文三部分。当前实现以第一、二部分和 `armhack_train.md` 为准；历史记录中的旧参数仅用于追溯。

## 第一部分：Walk 动态任务

### 任务目标与控制边界

动态任务 `LeggedLab-Isaac-AMP-G1-WalkPerturbFinetune-v0` 的目标不是从随机策略重新学习行走，也不是另行设计一套 gait。目标是在一个已经验证可用的 Nav2 速度跟踪策略上，仅把双臂改成指定的固定姿态，继续追踪同一类 `vx/vy/wz` 指令，并尽量保留原策略的存活率、鲁棒性和仿人步态。

```text
策略观测：原 Nav2/AMP 的 96 维当前状态观测
策略输出：29 维 joint-position action
实际执行：保留腿和腰 15 维；用指定固定姿态覆盖双臂 14 维
手臂来源：Reference Data/ArmHack/WalkPerturbFinetune/g1_arm_pose_set.json
速度来源：Nav2RecordedVelocityCommand 连续窗口
姿态时序：每个 episode 开始前选定一次，完整 episode 内不切换
```

双臂当前实际关节位置和速度仍属于本体反馈，policy 可以观察；但双臂 action 已被环境接管，策略不能改变它们。第一阶段应对一个明确命名的姿态单独训练和验收，三姿态随机混训只能作为后续鲁棒性扩展，不能替代逐姿态测试。

### S3 locomotion ONNX 起点与训练 checkpoint

所谓“在 Nav2 基础上只改变双臂姿态”，必须同时锁定以下变量：

- 起始 checkpoint、SHA-256 和对应的网络接口；
- 明确指定的 S3 G1 资产；
- 相同的 Nav2 command 窗口、缩放、过滤和增广分布；
- 相同的 AMP motion、速度跟踪奖励和主要 gait 奖励；
- 相同 seed/command window 下，原 Nav2、固定臂但未微调、固定臂且已微调三组策略的成对结果。

本轮按最新要求使用 S3 G1，并把仓库中的部署 actor 作为唯一权威起点：

```text
ONNX: checkpoint/model_9996/locomotion.onnx
ONNX SHA-256: 05fc45f89d89eb136225754f6a2fcacf5324d9dfd428d08ed75cc52f89b09be6
ONNX interface: 96 -> 29

training checkpoint: ArmHack Checkpoints/WalkPerturbFinetune/
  BaselineLocomotionModel9996/model_9996.pt
checkpoint SHA-256: bc30bc5171d211fa414fbeab31452b92ad76ca7f6ad76a2417a6e7f7515a0fa6
checkpoint size: 16,202,421 bytes
checkpoint iteration: 9996
asset: scripts/train_g1_armhack_walk.sh 强制 s3_g1_29dof
actor: 96 -> 29
critic input: 297
```

`locomotion.onnx` 只包含 actor，不能直接恢复训练所需的 critic。其同目录 `locomotion.deploy.json` 记录了导出源，源文件是 S3 G1 run 的 `model_9996.pt`。训练脚本使用该原始 `.pt` 做 policy-only 初始化，同时逐元素检查 `.pt` 的 8 个 actor 权重/偏置张量与指定 ONNX 完全一致；因此训练起始 actor 就是该 ONNX，而不是此前误选的 Stand `model_7999.pt`。Walk 环境、速度命令和奖励继续使用 Nav2 Stage-4 语义。

### 功能要求与验收方式

- 双臂从 episode 的第一个 policy observation 前就处于指定姿态，之后保持不变，不能在首个仿真步突然跳转；
- 机器人能完成前进、侧移、转向、减速和停止，不能用跳跃、长期双脚腾空、明显压低躯干或大幅倾斜换取速度跟踪；
- 每个 `pos1_back`、`pos2_down`、`pos3_front` 必须单独统计，不只报告混合平均值；
- 至少报告 20 s 完成率、各终止原因、`vx/vy/wz` MAE/RMSE、躯干 roll/pitch、高度变化、脚滑、双脚腾空比例、步态对称性和相对基线 policy KL；
- 第一版验收应以“相对原 Nav2 基线的退化不超过预先设定比例”为主。固定评估跑完前，不凭一次视频或总 return 临时定义成功。

### 已实现的奖励结构

Walk 不再继承 Command-Balanced V3，也不再叠加 `-1.8` 的躯干强惩罚，而是直接继承 `G1AmpNav2FinetuneEnvCfg` 并锁定已保存的 Nav2 Stage-4 数值：

| 类别 | 奖励项 | 权重 |
|---|---|---:|
| 速度跟踪 | `track_lin_vel_xy_exp`，`std=0.30` | `+1.8` |
| 速度跟踪 | `track_ang_vel_z_exp`，`std=0.35` | `+1.5` |
| 躯干速度 | `track_torso_lin_vel_xy_exp` / `track_torso_yaw_rate_exp` | `+1.0` / `+0.65` |
| 姿态 | `flat_orientation_l2` / `torso_roll_pitch_l2` | `-1.0` / `-0.04` |
| 根状态 | `torso_vertical_velocity_l2` / `lin_vel_z_l2` / `ang_vel_xy_l2` | `-0.01` / `-0.20` / `-0.05` |
| 下肢正则 | `dof_torques_l2` / `dof_acc_l2` / 仅腿腰的 `action_rate_l2` | `-2e-6` / `-1e-7` / `-0.006` |
| 关节安全 | 踝限位 / 髋偏差 / 腰偏差 | `-1.0` / `-0.10` / `-0.10` |
| 步态 | `feet_air_time` / `feet_slide` | `+0.50` / `-0.10` |
| 摔倒 | `termination_penalty` | `-200.0` |

双臂目标由环境接管，因此 `joint_deviation_arms` 和 `arm_style_prior` 关闭；torque、acc、action-rate 和 AMP discriminator 只看可控的腿腰 15 维。首版训练把 AMP style reward 设为 0，以 task reward + `0.003` baseline KL 优先保留已有策略。

### 已修复问题与当前实现

| 原问题 | 已实现修复 |
|---|---|
| 起点误用了旧 Stand `model_7999` | 改为锁定 `checkpoint/model_9996/locomotion.onnx`；训练加载其原始 `model_9996.pt`，并在启动前校验 ONNX/`.pt` actor 逐张量一致 |
| reset 后第一步手臂跳变 | 新增 Walk 独立环境类；在首个 observation 前同步实际 `q/dq`、执行器目标、当前/上一 action，初始化误差写入日志 |
| 父配置与 Nav2 不一致 | Walk 改为直接继承 `G1AmpNav2FinetuneEnvCfg`，显式恢复 Stage-4 command、motion 和奖励 |
| JSON 名称丢失 | 保留 `pos1_back`、`pos2_down`、`pos3_front`；脚本默认单独训练 `pos2_down`，也支持 `random` |
| raw CSV 没有 `mirror_lr` | loader 可从 `none` 在线生成 `(vx,-vy,-wz)` 镜像组，不复制第二份 83 MiB CSV |
| recorded-command curriculum 无效 | 删除只改 `ranges`、不改变 dataset 的两个 no-op curriculum term |
| Walk/Stand 日志可能混放 | Walk 独占 `logs/rsl_rl/g1_walk_perturb` 和 `ArmHack Checkpoints/WalkPerturbFinetune`；脚本检测到 Stand 正在训练会拒绝启动 |

当前命令固定为 `complex_turn`、4 s 连续窗口、scale `(0.85,0.75,0.75)`、`none+mirror_lr`、MPPI/DWB 权重 `1.5/1.0`。训练脚本固定 S3 G1、学习率 `3e-5`、entropy `0.002`、baseline KL `0.003`，首版关闭 RSI 和 domain randomization。29 维 actor 输出为了兼容 checkpoint 仍然保留，其中 14 个手臂动作会被覆盖；这是已知的兼容性折中，不是环境控制漏洞。

后续课程顺序为 `pos2_down -> pos1_back -> pos3_front -> random`。普通 ONNX actor 不包含环境侧 14 维覆盖逻辑，部署端必须复现同样的固定臂覆盖和 `last_action` 维护。

## 第二部分：Stand 静态任务

### 任务目标与控制边界

静态任务 `LeggedLab-Isaac-AMP-G1-StandPerturb-v0` 的目标不是让策略复现手臂动作，而是让机器人在速度指令恒为零时，对脚本强制施加的双臂运动进行下肢和腰部补偿：双脚尽量保持原地双支撑，躯干姿态、角速度、高度和水平位置尽可能稳定，并在完整手臂轨迹期间不摔倒。

控制边界如下：

```text
策略观测：根状态、零速度命令、29 个关节状态和上一时刻 action
策略输出：29 维 joint-position action
实际执行：保留腿部和腰部 15 维；14 维手臂 action 被 CSV 目标覆盖
手臂来源：Reference Data/ArmHack/StandPerturb/g1_arm_trajectory_named_50hz.csv
速度命令：vx=0、vy=0、wz=0
```

因此，手臂轨迹是策略不可控的外部扰动。手臂目标跟踪误差、速度和加速度应作为数据质量/执行器跟踪指标记录，但不应直接作为下肢策略的优化奖励；否则会把策略无法改变的量加入 return。真正要优化的是这段外部运动引起的全身、尤其是躯干响应。

最终数据集应从当前长 CSV 中整理出若干个有明确名称的预定轨迹，每条轨迹都包含“默认或中性姿态 → 平滑进入 → 主要动作 → 平滑退出/保持”四段。训练按轨迹均衡采样，测试按轨迹逐一固定回放。当前实现仍把一条约 405 秒的连续 CSV 当作单个数据源并随机起点，尚未达到这个最终数据契约。

### 功能要求与验收指标

功能要求：

- 速度指令始终为零，不通过持续迈步或漂移来规避手臂扰动；
- 双臂按预定轨迹运动，腿部和腰部主动补偿，允许小幅关节调整；
- 双脚维持接触，脚底滑移和根节点水平漂移尽可能小；
- 躯干 roll/pitch、roll/pitch 角速度、竖直速度和高度变化尽可能小；
- 每条轨迹、不同初始相位及规定的质量/摩擦随机化条件下均能完成整段动作；
- 训练指标、固定种子评估和长时视频必须使用同一个明确 checkpoint，不能用冒烟测试的 `model_0.pt` 代替正式模型。

建议把以下数值作为第一版验收线，待固定基线评估后再调整，而不是只看 episode return：

| 指标 | 第一版建议目标 |
|---|---:|
| 每条预定轨迹完成率 | 100%，无 `base_height`/`bad_orientation` 终止 |
| 躯干 roll/pitch RMS | 各小于 3° |
| 躯干 roll/pitch P95 | 各小于 5° |
| 躯干水平角速度 RMS | 小于 0.30 rad/s |
| 躯干高度误差 RMS | 小于 0.02 m |
| 单条轨迹根节点水平漂移 | 小于 0.10 m |
| 单脚累计滑移 | 小于 0.02 m |

### 已实现的奖励结构

2026-07-14 已把 Stand 奖励改为显式构造，不再依赖 walking 父配置中的奖励对象是否为 `None`。真实 Isaac smoke run 的 Reward Manager 已确认以下 21 项全部生效：

| 类别 | 奖励项 | 权重 |
|---|---|---:|
| 存活 | `alive` | `+1.0` |
| 原地稳定 | `track_torso_lin_vel_xy_exp`，跟踪零水平速度，`std=0.20` | `+1.5` |
| 原地稳定 | `track_torso_yaw_rate_exp`，跟踪零 yaw 角速度，`std=0.20` | `+0.75` |
| 支撑 | `double_support` | `+0.25` |
| 姿态 | `flat_orientation_l2` | `-1.0` |
| 姿态 | `torso_roll_pitch_l2` | `-3.0` |
| 动态稳定 | `torso_ang_vel_xy_l2` | `-0.15` |
| 动态稳定 | `torso_vertical_velocity_l2` | `-0.30` |
| 高度 | `torso_height_band_l2`，目标 `0.84 m`，`std=0.04 m` | `-0.60` |
| 动态稳定 | `torso_specific_force_xy_l2` | `-0.01` |
| 原地约束 | `root_xy_position_l2` | `-1.0` |
| 根状态 | `lin_vel_z_l2` / `ang_vel_xy_l2` | `-0.30` / `-0.10` |
| 下肢正则 | `dof_torques_l2` / `dof_acc_l2` / `action_rate_l2` | `-2e-6` / `-1e-7` / `-0.005` |
| 安全与姿态 | 踝限位 / 髋偏差 / 腰偏差 | `-1.0` / `-0.10` / `-0.12` |
| 足底稳定 | `feet_slide` | `-0.25` |
| 摔倒 | 非 timeout 的 `termination_penalty` | `-200.0` |

手臂本身的目标误差、速度或加速度没有作为 policy 奖励，因为 14 维手臂 action 被脚本覆盖，policy 无法改变这些量。训练优化的是手臂扰动造成的躯干、根节点、双脚和可控腿/腰响应。

### 分阶段静态任务

静态任务现在在同一次训练中连续执行三段课程：

1. 前 500 个 PPO iteration：每个 episode 从 CSV 的随机相位抽取一组手臂姿态，双臂保持不动；
2. 接下来 1000 个 iteration：手臂轨迹速度从 `0` 连续线性升到原轨迹的 `0.25` 倍；
3. 剩余 iteration：维持 `0.25` 倍速度连续运动。

episode 仍为 20 s。reset 时先随机选择轨迹相位，再把仿真中的 14 个手臂关节位置直接写成该相位的目标，速度写成 0；因此第一阶段不存在“默认姿态突然跳到随机目标”的接入冲击。第二、三阶段使用连续时间相位和线性插值，避免零阶保持导致的目标台阶。

课程仅改变环境如何生成外部手臂扰动。policy 观测保持原来的 96 维，只包含当前机器人状态、零速度命令和上一动作；没有轨迹 phase、未来手臂目标、目标速度或 look-ahead，因此不会提前知道手臂接下来如何运动。

正式训练从已校验的 `model_7999.pt` 做 policy-only 初始化，并保留 `0.003` 的冻结基线 KL。专用入口是：

```bash
cd /home/user/Workspace/Humanoid/Locomotion/G1-Locomotion/legged_lab
bash scripts/train_g1_armhack_stand.sh
```

详细参数、checkpoint 身份和两阶段真实测试记录见 `armhack_train.md`。

## 两个任务的共享数据、环境与归档约定

附件以下内容作为历史任务与训练记录保留。当前可运行实现以本节和 `armhack_train.md` 为准。

### 统一参考数据

两个任务的数据已集中到仓库相对目录：

```text
legged_lab/Reference Data/ArmHack/
├── StandPerturb/
│   ├── raw/g1_full_body_motion_sdk_50hz.csv
│   └── g1_arm_trajectory_named_50hz.csv
└── WalkPerturbFinetune/
    ├── g1_arm_pose_set.json
    └── nav2_cmd_vel_raw_success.csv
```

- Stand 原始数据来自 `/home/user/Workspace/whole_body_joints_20260708_143133.csv`。转换脚本按用户给定的 SDK `q0..q28` 顺序提取 `q15..q28`，生成只包含 `time_s + 14 个具名手臂关节` 的规范训练 CSV。下肢数据不会作为脚本目标施加。
- Walk 的三组双臂姿态保存在 JSON 中，单位为弧度，按左/右各 7 个关节记录；加载时校验单位、顺序、数量、名称唯一性和数值有限性，再转换成环境使用的左右交错 14 维顺序。
- Walk 的速度分布来自 HEC-5090 上 331,010 行 Nav2 成功轨迹，保存为 `nav2_cmd_vel_raw_success.csv`；该 83 MiB 外部数据已忽略 Git，但训练路径保持仓库相对。
- 所有训练路径均相对于 `legged_lab` 项目目录解析，不再包含 `/home/hecggdz/...` 等机器绝对路径。

### 动态任务命令说明

当前动态任务使用 `Nav2RecordedVelocityCommandCfg`，按 planner/controller/scenario/goal 分组后，从 `complex_turn` 成功轨迹中截取连续 4 秒窗口：

```text
原始数据：331,010 行、445 个轨迹组
原始范围：vx [-0.2, 0.6]，vy [-0.3, 0.3]，wz [-0.518728, 0.6]
窗口：4.0 s；数据周期：0.05 s
缩放：(0.85, 0.75, 0.75)
平滑时间常数：0.30 s
最大线加速度：0.60 m/s²；最大 yaw 加速度：0.80 rad/s²
augmentation_filter：none,mirror_lr
镜像实现：从 raw none 在线生成 (vx,-vy,-wz)，不改写源 CSV
controller 权重：mppi 1.5，dwb 1.0
```

源文件仍只有 `augmentation=none`；`synthesize_mirror_lr=True` 在内存中生成左右镜像轨迹并赋予 `mirror_lr` 标签，源文件和相对路径不变。数据来源、SHA-256 和新机器复制方法见 `Reference Data/ArmHack/README.md`。

### checkpoint 目录

两个任务的模型按 run 独立归档：

```text
legged_lab/ArmHack Checkpoints/
├── StandPerturb/
│   ├── BaselineModel7999/model_7999.pt
│   └── <run_name>/model_*.pt
└── WalkPerturbFinetune/
    ├── BaselineLocomotionModel9996/model_9996.pt
    └── <run_name>/model_*.pt
```

runner 同时保留 `logs/rsl_rl/...` 中的原 checkpoint，保证旧的续训逻辑不变。模型文件已加入 `.gitignore`。训练、测试、最新 checkpoint 选择、GUI 可视化和视频录制命令统一记录在 `armhack_train.md`。

### 当前环境和验证状态

本机统一使用：

```bash
source /home/user/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
cd /home/user/Workspace/Humanoid/Locomotion/G1-Locomotion/legged_lab
```

环境已验证为 Python `/home/user/anaconda3/envs/env_isaaclab/bin/python`、IsaacLab `0.54.2`、`rsl-rl-lib 3.2.0`、PyTorch `2.7.0+cu128`，CUDA 和 RTX 4090 可用。

Stand 已完成从其独立 `model_7999.pt` policy-only 恢复、课程切换和长于完整 episode 的真实训练验证。Walk 起点已纠正为 S3 `locomotion.onnx` 对应的原始 `model_9996.pt`；`pos1_back`、`pos2_down`、`pos3_front` 均已从正确起点完成 `2 env × 1 iteration` 真训练 smoke，初始化误差与两类摔倒终止均为 0，full-state resume 也已通过。此前从 Stand `model_7999` 启动的 Walk smoke 全部只保留为错误基线诊断记录，不再作为有效 Walk 结果；这些 smoke 仍不能替代 20 s 正式性能验收。

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
