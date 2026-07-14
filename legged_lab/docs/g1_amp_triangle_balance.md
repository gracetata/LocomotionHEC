# G1 AMP 三角平衡总结：仿人步态、导航速度追踪、鲁棒安全性

本文梳理 `legged_lab` 中围绕 Unitree G1 AMP locomotion 的训练改进流程，目标是回答一个核心问题：

> 为了同时平衡仿人步态、导航速度追踪、鲁棒安全性，这套代码采用了哪些有效 trick，各自提升了什么效果？

结论先行：当前最有效的方向不是继续加大 velocity tracking reward，而是把任务定义改成双足可执行的导航命令分布，再用 AMP / KL / arm prior 约束动作仍留在人类步态流形内，最后用 posture、contact、domain randomization 和 termination 做安全兜底。

主要证据来自：

- `legged_lab/source/legged_lab/legged_lab/tasks/locomotion/amp/config/g1/g1_amp_env_cfg.py`
- `legged_lab/source/legged_lab/legged_lab/tasks/locomotion/amp/mdp/rewards.py`
- `legged_lab/source/legged_lab/legged_lab/tasks/locomotion/amp/mdp/commands/*.py`
- `legged_lab/source/legged_lab/legged_lab/tasks/locomotion/amp/amp_env_cfg.py`
- `legged_lab/logs/rsl_rl/g1_amp/*`
- `outputs/*.json`
- `task_improve.md`

## 1. 三角本质

### 1.1 三个目标不是天然一致

仿人步态要求：

- 离散支撑相和摆动相清晰；
- 躯干、手臂、腿部有自然相位关系；
- 足底接触不滑、不跳、不拖；
- 不能为了追命令把身体压低、前扑或高频抖动。

导航速度追踪要求：

- 按 `[vx, vy, wz]` 跟随 Nav2 / joystick 命令；
- 能前进、侧移、转向、低速绕障、停走切换；
- yaw 和 lateral tracking 不应只在固定前进场景成立。

鲁棒安全性要求：

- 不倒、不异常低身高、不大 roll/pitch；
- 扰动、质量、摩擦、执行器增益变化下仍可走；
- sim2sim / sim2real 中不依赖脆弱动作。

冲突点在于：精确速度追踪经常把策略推离自然步态，尤其在高前进速度同时叠加大侧向速度和大 yaw-rate 时。轮式底盘可以连续调速和原地旋转，双足机器人必须通过落脚、单支撑、质心转移来改变速度。对双足来说，速度调整天然有相位延迟，不能像轮足或差速底盘那样即时、平滑、精确地追每一帧命令。

### 1.2 当前 policy 信息也决定了上限

G1 AMP policy 观测固定为 96 维：

```text
base angular velocity 3
projected gravity 3
velocity command [vx, vy, wz] 3
joint position 29
joint velocity 29
last action 29
```

它没有未来轨迹、目标 heading 序列、command derivative、足步规划或 gait phase target。因此，如果训练时把 `vx/vy/wz` 做完全独立随机采样，策略只能把单帧速度命令强行映射到关节动作，容易出现两个坏结果：

- task reward 要求跟踪不自然命令；
- AMP discriminator 要求动作像 mocap；
- 两者形成零和博弈。

这也是后续从 independent joystick 转向 curvature / Nav2 recorded window 的根本动机。

## 2. 训练改进流程

### 2.1 Baseline：只训练近似前向速度

任务：`LeggedLab-Isaac-AMP-G1-v0`

配置要点：

- command: `lin_vel_x=(-0.2, 1.5)`, `lin_vel_y=0`, `ang_vel_z=0`
- AMP demo: ACCAD / G1 used motion
- RSI: reset 时部分环境从 reference motion 状态初始化
- policy/action 接口固定为 96/29

效果：

- 能学到稳定前向仿人步态；
- 但 yaw / lateral 能力几乎没有训练；
- 对 Nav2 全向命令分布是 OOD；
- MuJoCo fixed forward 有启动前倾、yaw 抖动问题。

代表 checkpoint：

```text
legged_lab/logs/rsl_rl/g1_amp/2026-05-20_23-50-16_baseline_3000_accad50hz_rsi_20260520/model_2999.pt
```

### 2.2 第一轮：直接打开三轴随机命令，被证明方向不对

早期 fine-tune 尝试把 `lin_vel_x / lin_vel_y / ang_vel_z` 范围独立打开。诊断结论是：这制造了很多双足不合理命令，例如高速前进同时大侧移和大 yaw-rate。策略为了追踪会压低身体、前倾、滑脚或破坏风格。

这一步的价值是排除了一个直觉上简单但实际有害的方向：单纯扩大 command range 不是解决导航跟踪的好办法。

### 2.3 Curvature / segmented-yaw：把高速 yaw 约束到更物理的分布

相关实现：

```text
legged_lab/.../mdp/commands/curvature_velocity_command.py
G1AmpSegmentedYawFinetuneEnvCfg
G1AmpCurvatureFinetuneEnvCfg
```

方法：

- 低速时允许更大 yaw 和侧移；
- 高速时 yaw-rate 收窄，避免高速大转弯；
- command 输出仍是 3 维 `[vx, vy, wz]`，baseline 可 resume；
- reset / ramp 降低启动瞬间的非物理追踪压力。

效果记录来自 `task_improve.md`：

| policy | 评估 | tracking | lin_xy_mae | yaw_mae | early torso drop | early pitch max | 结论 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| baseline | curvature random + ramp | 40.9 | 0.2051 | 1.3685 | 0.0470 | 0.2084 | 新分布下 yaw tracking 弱 |
| probe 200 iter | curvature random + ramp | 50.1 | 0.1420 | 1.0804 | 0.0625 | 0.3564 | tracking 提升，但姿态变差 |
| probe 200 iter | fixed ramp | 60.5 | 0.0881 | 0.7949 | 0.0469 | 0.2557 | 固定命令 tracking 改善 |

结论：

- 曲率命令分布是正确方向；
- 但只靠 task reward 提升 tracking 会牺牲 torso 姿态；
- 不能继续用强追踪 reward 硬拉。

### 2.4 Wide curvature 8000：证明“追得准”不等于“走得好”

代表输出：

```text
outputs/g1_amp_wide8000_mujoco_curvature_random.json
```

指标：

```text
tracking_score: 76.35
lin_vel_xy_mae: 0.0981
yaw_rate_mae: 0.1229
torso_score: 47.24
torso_pitch_error_rad: 0.4586
early torso_height_drop_m: 0.1300
min_root_height: 0.611
healthy: true
```

这个结果很关键：它说明策略可以把速度追得很好，但代价是明显压低身体、pitch 偏大、姿态风格变差。也就是说，追踪指标和仿人/安全不是同一个目标。

### 2.5 Nav2 recorded command：从真实成功导航窗口采样

相关实现：

```text
legged_lab/.../mdp/commands/nav2_recorded_velocity_command.py
G1AmpNav2FinetuneEnvCfg
scripts/finetune_g1_amp_nav2.sh 旧记录
```

动机来自真实 Nav2 数据统计：

```text
vx: [-0.20, 0.60] m/s
vy: [-0.30, 0.30] m/s
wz: [-0.519, 0.60] rad/s
median dt: about 0.05s
corr(speed, |wz|): about -0.51
fast && |wz| > 0.30: about 0.01%
```

方法：

- 从成功 Nav2 `/cmd_vel` CSV 按连续窗口采样；
- 按 planner/controller/scenario/goal/augmentation 分组；
- 使用 `none,mirror_lr`，保持左右对称但不过度增广；
- `smoothing_time_constant=0.30`;
- `max_linear_accel=0.60`;
- `max_yaw_accel=0.80`;
- reset command to zero；
- stage curriculum 从小 command scale 开始。

Stage 计划：

```text
Stage 1: scale=[0.70,0.55,0.55], KL=0.025, task_style_lerp=0.35
Stage 2: scale=[0.85,0.75,0.75], KL=0.012
Stage 3: scale=[1.00,1.00,1.00], KL=0.005
```

Stage-1 训练记录：

```text
run: legged_lab/logs/rsl_rl/g1_amp/2026-06-01_00-11-53_finetune_nav2_stage1_kl
Mean reward: 22.96
Mean AMP style reward: 0.98
Mean baseline_kl: 0.4225
track_lin_vel_xy_exp: 0.8923
track_ang_vel_z_exp: 0.3164
termination base_height: about 0.0018
termination bad_orientation: about 0.0241
torso_lin_vel_xy_cmd_error_m_per_s: about 0.2075
torso_yaw_rate_cmd_error_rad_per_s: about 0.8792
```

MuJoCo 复核：

| policy | eval | total | tracking | torso | lin_xy_mae | yaw_mae | min_root_h | max_roll_pitch | health |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Nav2 Stage-1 | fixed `(0.3,0,0)` | 84.2 | 70.5 | 78.3 | 0.039 | 0.666 | 0.773 | 0.173 | healthy |
| Nav2 Stage-1 | Nav2-ish curvature random | 78.9 | 58.8 | 73.1 | 0.141 | 0.457 | 0.757 | 0.339 | healthy |
| Nav2 replay seed53 | real window | 79.1 | 63.2 | 67.9 | 0.183 | 0.161 | 0.783 | 0.233 | healthy |

Stage-2 exported MuJoCo record:

```text
run: legged_lab/logs/rsl_rl/g1_amp/2026-06-01_02-23-30_finetune_nav2_stage2_realwindows_follow
total_score: 88.75
tracking_score: 82.70
torso_score: 79.21
lin_vel_xy_mae: 0.0286
yaw_rate_mae: 0.2501
early torso_height_drop_m: 0.0150
healthy: true
```

结论：

- Nav2 recorded window 明显优于人工宽随机命令；
- 它保留了启动姿态和安全高度；
- tracking 不再靠“压低身体”获得；
- yaw / lateral 仍需要进一步 curriculum，但方向正确。

### 2.6 CMU walk core/full/washed：扩充更自然的人类行走先验

相关任务：

```text
LeggedLab-Isaac-AMP-G1-CmuWalkCore-Adaptive-v0
LeggedLab-Isaac-AMP-G1-CmuWalkFull-Adaptive-v0
LeggedLab-Isaac-AMP-G1-CmuWalkWashed-Adaptive-v0
```

相关 run：

```text
2026-06-10_16-14-39_orig_g1_29dof_cmu_walk_core_adaptive_4000
2026-06-10_21-10-42_orig_g1_29dof_cmu_walk_full_adaptive_4000
2026-06-11_17-43-37_orig_g1_29dof_cmu_walk_washed_adaptive_style8_yawsigma07_yaww125_4000
```

方法：

- 用 CMU walk core/full/washed 数据替换或扩充 ACCAD demo；
- motion pickle 中启用 `target_dof_names` 和 `strict_dof_names`，避免关节顺序错位；
- 使用 adaptive tracking sigma；
- washed 版本提高 style weight，并加强 yaw tracking balance。

效果：

- 提供更完整的人类步态分布，减少只靠少量 ACCAD clip 的风格偏差；
- 为后续 strict upright、arm prior、command-balanced directional 数据打基础；
- 日志中 6 月 8 日的 orderfix/debug 也说明，关节顺序和资产对齐是 G1 AMP 能否稳定 resume 的前提。

### 2.7 Strict upright + arm prior：补 AMP 在姿态和摆臂细节上的盲区

相关任务：

```text
LeggedLab-Isaac-AMP-G1-CmuWalkWashed-Adaptive-Strict-v0
LeggedLab-Isaac-AMP-G1-CmuWalkWashed-Adaptive-Strict-ArmPrior-v0
```

strict reward 增加：

- torso roll/pitch penalty；
- torso vertical velocity；
- torso height band；
- foot swing clearance；
- gait timing symmetry；
- feet slide；
- light specific-force guard。

arm prior 增加：

- 从 root velocity + non-arm joint state 预测 demo arm pose；
- 奖励当前 arm joints 接近预测值；
- 防止为了下肢 tracking 把手臂冻结或乱摆。

摆臂诊断：

```text
legged_lab/logs/rsl_rl/g1_amp/2026-06-12_19-45-54.../arm_swing_analysis/arm_swing_report.json
```

可见结果：

- shoulder pitch 和 hip 的相位耦合接近 reference；
- 左右肩 pitch 负相关接近 reference；
- 但 wrist forward/back path 仍偏小；
- arm action 仍有高频成分，说明摆臂细节还没完全解决。

这说明 AMP 能守住整体风格，但手臂这种细粒度风格需要单独 prior 或诊断指标。

### 2.8 Command-balanced directional：解决前进以外的方向性步态

相关任务：

```text
LeggedLab-Isaac-AMP-G1-CommandBalancedDirectional-Strict-ArmPrior-v0
LeggedLab-Isaac-AMP-G1-CommandBalancedDirectional-Strict-ArmPrior-V2-v0
```

相关 command：

```text
ModeBalancedVelocityCommand
command_balanced_directional_50hz/task_sampling_config.json
```

方法：

- 不再用单一 uniform 分布；
- 先采样 mode，再采样该 mode 的 `[vx, vy, wz]`；
- 使用 command-balanced directional demo；
- V2 加入 backward/lateral anti-hop shaping：
  - `directional_speed_floor_l1`
  - `directional_double_air_l1`
  - `backward_single_stance`
  - `backward_double_stance_l1`

方向性评估结果：

| policy | mode | lin_mae | yaw_mae | single stance | double stance | double air | flags |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| strict armprior scratch | backward | 0.131 | 0.162 | 0.018 | 0.882 | 0.100 | stuck_or_hop, double_air, slide, low_clearance |
| command-balanced V1 | lateral_left | 0.096 | 0.172 | 0.314 | 0.684 | 0.002 | low_clearance |
| command-balanced V1 | turn_left | 0.052 | 0.149 | 0.551 | 0.449 | 0.000 | ok |
| command-balanced V2 | backward | 0.078 | 0.195 | 0.932 | 0.068 | 0.000 | low_clearance |
| command-balanced V2 | lateral_left | 0.068 | 0.201 | 0.744 | 0.256 | 0.001 | low_clearance |
| command-balanced V2 | turn_left | 0.060 | 0.174 | 0.593 | 0.407 | 0.000 | ok |

结论：

- V1 已经让侧移、慢走、转向稳定；
- 后退仍会退化成 double-stance 拖行；
- V2 的 anti-hop / single-stance shaping 明显把后退和侧移推回双足交替支撑；
- 这证明“仿人性”不能只靠 AMP discriminator，需要接触结构 reward。

## 3. 有效 trick 清单

### 3.1 AMP discriminator style reward

作用：

- 用 mocap/reference motion 约束动作分布；
- 避免手写所有“像人”的 reward；
- 在速度 tracking 外加入风格维度。

提升：

- baseline 可学到自然前向步态；
- fine-tune 时不会完全变成 reward hacking；
- 但 AMP 对 OOD task-heavy 状态不总是可靠，所以需要 KL / arm prior / posture guard。

### 3.2 Reference State Initialization

作用：

- reset 时部分环境从 reference motion 中间帧开始；
- 让策略覆盖不同 gait phase；
- 减少只从静止启动带来的局部最优。

提升：

- 训练稳定性更好；
- policy 更容易学到完整步态循环；
- 与 AMP demo 分布更一致。

### 3.3 固定 96/29 policy 接口

作用：

- G1 / S3 G1 asset 变化时仍保持 policy obs/action 不变；
- 只控制 locomotion 29 DoF；
- 新增 gripper/payload 不进入 action。

提升：

- baseline checkpoint 可 resume；
- sim2sim / sim2real 部署接口稳定；
- 避免资产扩展破坏训练维度。

### 3.4 物理相关 command distribution

作用：

- 避免三轴独立随机制造不合理命令；
- 低速允许大 yaw / lateral，高速限制 yaw；
- Nav2 真实窗口保留速度相关性和时序。

提升：

- Curvature probe 提升目标分布 tracking；
- Nav2 Stage-1/2 在保持健康姿态的同时提升真实导航窗口跟踪；
- 避免 wide curvature 那种 tracking 好但 torso 坏的伪进步。

### 3.5 Command ramp / smoothing / acceleration limit

作用：

- reset 后从零命令开始；
- 限制命令变化率；
- 降低启动瞬间为了追命令而前扑/下蹲。

提升：

- baseline fixed command 加 ramp 后 early pitch peak 从 `0.2296` 降到 `0.1320`；
- Nav2 Stage-2 early torso drop 约 `0.015m`，远低于 wide curvature 的 `0.130m`；
- 对真机部署更友好。

### 3.6 Adaptive tracking sigma

作用：

- 训练早期 tracking reward 宽容；
- 随 EMA error 单调收紧；
- 减少 early training 中 task reward 和 style reward 的硬冲突。

提升：

- CMU walk adaptive 系列能逐步提高 command tracking；
- 避免一开始就把策略拉出人类步态流形。

### 3.7 Upright-gated tracking reward

作用：

- velocity tracking reward 乘 upright scale；
- 姿态不直立时，即使速度碰巧接近也拿不到完整 tracking reward。

提升：

- 把速度追踪与不倒/不歪绑定；
- 防止策略通过摔倒、低身高、前扑刷 tracking。

### 3.8 轻量 posture guard

作用：

- torso roll/pitch；
- torso height band；
- vertical velocity；
- specific force；
- action smoothness。

提升：

- strict / Nav2 任务中改善 torso 稳定；
- 但权重保持轻量，避免把人类自然起伏训练没。

### 3.9 双足接触结构 reward

作用：

- single stance air-time；
- foot slide penalty；
- swing clearance；
- gait timing symmetry；
- backward/lateral anti-hop。

提升：

- 减少滑步、拖脚、双脚腾空；
- command-balanced V2 明显改善 backward/lateral 的支撑结构；
- 让 tracking 通过“走路”实现，而不是滑行或跳。

### 3.10 Domain randomization 和扰动

作用：

- friction / restitution；
- base mass；
- COM；
- link mass；
- actuator gains；
- armature；
- interval push。

提升：

- 提升 sim2real 和扰动鲁棒性；
- 让 policy 不依赖单一仿真动力学；
- 牺牲一点极限 tracking，换安全裕度。

### 3.11 Termination 和失败惩罚

作用：

- base height；
- bad orientation；
- large termination penalty；
- G1 中关闭 base_contact termination，避免正常资产接触误杀。

提升：

- 明确“不摔”优先级；
- 让训练统计中 `base_height` / `bad_orientation` 成为安全诊断指标。

### 3.12 Symmetry augmentation 和 mirror loss

作用：

- 对 obs/action 做左右镜像；
- 速度命令中 lateral / yaw 对应变号；
- 左右关节互换并翻转 roll/yaw。

提升：

- 减少左右偏置；
- 提高左右转、左右侧移泛化；
- 降低对单侧 motion data 的依赖。

### 3.13 Baseline-policy KL anchor

作用：

- fine-tune 时加载 frozen baseline policy；
- 对当前策略和 baseline 策略的动作分布计算 KL；
- 早期约束当前策略不要偏离原有仿人动作分布。

提升：

- Nav2 Stage-1 证明 KL 没阻止收敛；
- 减少 task-heavy fine-tune 丢风格；
- 适合分阶段逐渐释放 task following。

### 3.14 Arm style prior

作用：

- 用监督模型预测 demo arm pose；
- 奖励当前 arm joints 接近预测；
- 补 AMP 对摆臂细节约束不足的问题。

提升：

- 保留肩和髋的相位耦合；
- 暴露 wrist path 和 action 高频问题；
- 为后续真机视觉观感和自然性优化提供诊断指标。

## 4. 论文式叙事：Motivation, Difficulty, Method

### Motivation

服务机器人或人形平台最终要接入导航系统。Nav2 输出的自然接口是 `/cmd_vel`，但用户需要的不是一个像轮式底盘那样追速度的下肢控制器，而是一个能在导航中自然、安全、稳定行走的人形 locomotion policy。

因此目标是三重的：

1. 像人：步态相位、手臂摆动、足底接触和躯干姿态自然；
2. 听话：能跟随 Nav2 的前进、侧移、转向、停走命令；
3. 安全：不摔、不压低身体、不靠滑脚和高频动作完成任务。

### Difficulty

这个问题的难点不是“reward 不够多”，而是目标之间存在物理冲突。

第一，双足不是轮足。轮式或轮足平台可以连续调节速度，双足必须等待支撑脚、摆动脚和质心转移。速度变化有步态相位延迟，尤其 yaw 和 lateral 不能在任意高速状态下即时响应。

第二，policy 只看到当前 `[vx, vy, wz]`，看不到未来路径。如果命令分布不合理，策略无法知道应该提前减速、换支撑脚或转身，只能把单帧速度硬映射成动作。

第三，AMP 和 tracking reward 的偏好不同。AMP 偏好 mocap 中的人类动作，tracking reward 偏好数值误差最小。若命令落在 mocap 支撑之外，二者会互相拉扯。

第四，安全性不是 tracking 的自然副产物。wide curvature 8000 的结果说明，速度可以追得很好，但躯干明显前倾、身高下沉、风格变差。

### Method

因此方法不是“更强 tracking reward”，而是分层解耦：

1. Task distribution alignment  
   用 curvature / Nav2 recorded window 替代独立随机 joystick。让训练命令首先接近双足和导航实际会遇到的分布。

2. Smooth command interface  
   用 ramp、smoothing、acceleration limit 把高层命令变成双足可响应的速度目标。

3. Human motion manifold constraint  
   用 AMP discriminator、CMU/ACCAD demo、arm prior、baseline KL anchor 守住仿人风格。

4. Safety and contact guard  
   用 upright-gated reward、posture guard、contact shaping、domain randomization 和 termination 把安全性嵌入训练。

5. Directional curriculum  
   用 mode-balanced command 和 command-balanced directional demo 分别训练慢走、侧移、转向、后退，而不是指望一个 uniform 分布自然覆盖全部模式。

一句话概括：代码承认“无比精确平滑的速度追踪”和“仿人双足步态”本身存在冲突，因此把目标改为“在双足动力学和人类步态流形内尽量跟踪导航速度”。

## 5. 当前最佳实践

如果目标是 Nav2 式导航 locomotion，优先采用：

1. `Nav2RecordedVelocityCommand` 或等价真实导航窗口；
2. reset-to-zero command ramp；
3. command acceleration limit；
4. baseline KL anchor；
5. style reward scale / task_style_lerp 保守设置；
6. 轻量 torso guard；
7. contact shaping 而不是只加速度 reward；
8. 分阶段 command scale curriculum；
9. MuJoCo early-motion + tracking + torso + health 联合验收。

不建议：

1. 三轴 independent uniform 随机直接全开；
2. 单纯加大 `track_lin_vel_xy_exp` / `track_ang_vel_z_exp`；
3. 用强 torso penalty 锁死自然步态；
4. 只看 tracking score，不看 torso score 和 early height drop；
5. 只用 mocap synthetic corruption 的离线风格判别器做最终验收。

## 6. 仍然存在的问题

1. yaw / lateral tracking 还不够强  
   Nav2 Stage-1/2 已经安全，但复杂侧移和转向仍需 curriculum。

2. arm style 仍不完美  
   shoulder/hip phase 有较好耦合，但 wrist path 偏小，arm action 高频仍存在。

3. low clearance 仍常见  
   command-balanced V2 改善了 single stance，但后退/侧移仍有 low-clearance flags。

4. offline style metric 尚不可靠  
   只用 mocap 正样本和 synthetic corruption 训练的风格判别器能识别严重塌缩，但不能可靠排序 baseline 和正常 fine-tune policy。

5. ToTarget 与 velocity-follow 是相邻但不同任务  
   ToTarget 需要 stop latch、near-target stillness 和 success hold，不能简单等同于 velocity command tracking。

## 7. 推荐验收指标

不要只看单一 reward。建议同时报告：

```text
tracking_score
lin_vel_xy_mae
yaw_rate_mae
torso_score
torso_pitch_error_rad
torso_height_drop_m in first 1s
min_root_height
max_abs_roll/pitch
base_height termination
bad_orientation termination
single/double stance ratio
double air ratio
foot slide p95
foot clearance p05/p95
AMP style reward
baseline KL
arm phase correlation / wrist path amplitude
```

验收口径应该是：

```text
tracking improves,
healthy remains true,
early torso drop does not regress,
support pattern remains bipedal,
style does not collapse.
```

这才真正对应“仿人步态、导航速度追踪、鲁棒安全性”的三角平衡。

## 8. 2026-06-17 mask/MuJoCo 验证后的 V3 经验

### 8.1 重要发现

这次 `model_8997.pt` 的 IsaacLab directional eval 和 MuJoCo validation 给出了一组很有价值的 mask-test 经验：

1. V2 方向正确，但 contact reward 过硬  
   `directional_double_air_l1` 和速度 floor 成功修掉了后退双脚同时跳、卡住和滑步问题；但是 `backward_single_stance` 这种“单支撑越多越好”的 shaping 会把 backward/lateral 推成过度交替迈步。

2. tracking 好不代表接触节律像人  
   V2 在后退和侧移速度上明显更好，但 contact ratio 偏离数据集：backward 的 single stance 接近 `0.93`，lateral 约 `0.74-0.76`，而参考数据中 backward 约 `0.66`，lateral 约 `0.20-0.28`。所以后续 reward 应该约束 mode-specific support pattern，而不是只看速度误差。

3. MuJoCo 暴露的是 sim2sim 行为，不是训练 reward 的镜像  
   当前 MuJoCo full suite 中 `14/14` healthy，`humanoid_score_mean=88.2`，说明仿人性和安全性没有大幅下降；但 fixed-command tracking 仍显示 backward 欠速、lateral 有轻微 forward/yaw leakage，normal walk 有早期高度下沉。这些问题更适合用轻量泄漏惩罚、height/vertical guard 和接触模式 shaping 修，而不是继续盲目加 tracking 权重。

### 8.2 V3 任务设定

新增任务：

```text
LeggedLab-Isaac-AMP-G1-CommandBalancedDirectional-Strict-ArmPrior-V3-v0
LeggedLab-Isaac-AMP-G1-CommandBalancedDirectional-Strict-ArmPrior-V3-Play-v0
```

V3 保留 V2 的有效部分：

```text
command-balanced directional dataset
s3_g1_29dof
strict upright rewards
arm style prior
directional speed floor
directional double-air penalty
feet slide / clearance / gait timing symmetry
AMP style_reward_scale=8.0
TASK_STYLE_LERP=0.50
RSI + domain randomization
```

V3 修改点：

1. 关闭 `backward_single_stance` 和 `backward_double_stance_l1`  
   避免继续把 backward 推到接近连续单支撑。

2. 新增 `directional_double_stance_support`  
   对 lateral 给更强 double-stance support，对 backward 给较弱 support，让策略从 V2 的过度交替回到更接近数据集的支撑节律。

3. 新增 `directional_velocity_leak_l1`  
   对 lateral 约束 forward/yaw leakage，对 backward 约束 lateral/yaw leakage。目标是修 MuJoCo 中侧移 `vx≈+0.06`、`|wz|≈0.05` 的轻微泄漏。

4. 轻微加强高度和垂直速度 guard  
   `torso_height_band_l2` 和 `torso_vertical_velocity_l2` 略加强，用来压 normal/backward 的 early drop，但不做强 torso lock。

### 8.3 V3 finetune 计划

从 V2 final checkpoint 续跑：

```bash
ROBOT_ASSET=s3_g1_29dof \
TASK=LeggedLab-Isaac-AMP-G1-CommandBalancedDirectional-Strict-ArmPrior-V3-v0 \
RESUME=True \
LOAD_RUN=2026-06-16_17-34-12_s3_g1_29dof_command_balanced_directional_strict_armprior_v2_resume7498_1500 \
CHECKPOINT=model_8997.pt \
MAX_ITERATIONS=1000 \
RUN_NAME=s3_g1_29dof_command_balanced_directional_strict_armprior_v3_resume8997_1000 \
STYLE_REWARD_SCALE=8.0 \
TASK_STYLE_LERP=0.50 \
QUIET_TERMINAL=True \
bash scripts/train_g1_amp.sh
```

建议每 `200-300` iter 做一次固定命令评估：

```text
normal_walk
slow_walk
backward
lateral_left/right
turn_left/right
diagonal_left/right
random_curvature
random_omni
```

重点验收：

```text
healthy cases: 100%
humanoid_score_mean: 不低于 V2 的 88 左右
backward vx: 更接近 -0.40，同时 single stance 不再接近 0.93
lateral vx leakage: < 0.04 m/s
lateral yaw leakage: < 0.04 rad/s
lateral double stance: 明显高于 V2
early torso drop: normal/backward 不继续恶化
double air: 继续接近 0
```
