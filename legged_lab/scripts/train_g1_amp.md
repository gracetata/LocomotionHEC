# train_g1_amp.sh

## 入口

`scripts/train_g1_amp.sh` 是当前 G1 AMP 训练的便捷启动脚本。它会从仓库根目录启动：

```bash
bash scripts/train_g1_amp.sh
NUM_ENVS=128 MAX_ITERATIONS=2 bash scripts/train_g1_amp.sh
```

脚本默认使用 `.envrc` 对应的 conda 环境 `env_isaaclab`：

```bash
/home/hecggdz/miniconda3/envs/env_leglab/bin/python
```

也可以通过 `ISAACLAB_PYTHON`、`CONDA_ENV_NAME`、`CONDA_BASE` 覆盖。脚本会设置：

- `PYTHONPATH=${REPO_ROOT}/source/legged_lab`
- `LD_LIBRARY_PATH=${CONDA_ENV}/lib`
- `OMNI_LOG_DEFAULT_LEVEL=error`
- `OMNI_KIT_QUIET=1`

默认训练命令最终进入：

```bash
scripts/rsl_rl/train.py --task LeggedLab-Isaac-AMP-G1-v0
```

## 任务结构

Gym task 注册在：

`source/legged_lab/legged_lab/tasks/locomotion/amp/config/g1/__init__.py`

`LeggedLab-Isaac-AMP-G1-v0` 绑定：

- env cfg: `G1AmpEnvCfg`
- runner cfg: `G1RslRlOnPolicyRunnerAmpCfg`
- env class: `ManagerBasedAmpEnv`

环境配置在：

`source/legged_lab/legged_lab/tasks/locomotion/amp/config/g1/g1_amp_env_cfg.py`

它继承通用 `LocomotionAmpEnvCfg`，并覆盖 G1 专用内容：

- robot asset 使用 `UNITREE_S3_G1_29DOF_CFG`
- motion data 使用 `data/MotionData/g1_29dof/amp/accad_g1used_50hz`
- discriminator demo observation 使用 reference animation
- RSI reset 使用 `mdp.ref_state_init_subset`
- command 只训练前向速度：`lin_vel_x=(-0.2, 1.5)`，侧向和 yaw 默认 0
- curriculum 关闭
- base contact termination 关闭

## S3 资产

S3 G1 asset 定义在：

`source/legged_lab/legged_lab/assets/unitree.py`

当前训练使用：

```python
UNITREE_S3_G1_29DOF_CFG
```

它通过 IsaacLab 的 `MjcfFileCfg` 指向：

```text
source/legged_lab/legged_lab/data/Robots/Unitree/s3_g1_29dof/g1_29dof.xml
```

不要使用旧拷贝：

```text
source/legged_lab/legged_lab/data/Robots/Unitree/g1_29dof/s3_g1_29dof.xml
```

第一次启动时，IsaacLab 会把 MJCF 转成 USD，输出到：

```text
source/legged_lab/legged_lab/data/Robots/Unitree/s3_g1_29dof/usd/s3_g1_29dof.usd
```

当前 cfg 里显式启用了 `isaacsim.asset.importer.mjcf`，并设置 `force_usd_conversion=True`，因此每次启动都会用新 XML 重新生成 USD。资产稳定后可以把它改回 `False`，避免每次启动都转换。

## 动作和观测维度

虽然 S3 XML 多了夹爪和配重 link，policy 接口仍固定为原 G1 29 个 locomotion 关节：

```python
G1_LOCOMOTION_JOINT_NAMES = UNITREE_G1_29DOF_CFG.joint_sdk_names
```

这些关节被显式用于：

- `actions.joint_pos`
- policy joint position / velocity observations
- critic joint position / velocity observations
- discriminator joint position / velocity observations
- discriminator demonstration joint position / velocity observations
- RSI reference joint state 写回
- torque / acceleration reward penalties
- actuator gain / joint parameter randomization

因此策略仍保持原来的维度：

- policy obs: 96 = base angular velocity 3 + gravity 3 + command 3 + joint pos 29 + joint vel 29 + last action 29
- action: 29

新增的 S3 gripper joints 不进入 policy action/obs，也不由 reference motion 写入；它们保持资产默认零位。

## 奖励和 AMP

runner 配置在：

`source/legged_lab/legged_lab/tasks/locomotion/amp/config/g1/agents/rsl_rl_ppo_cfg.py`

核心设置：

- `class_name="AMPRunner"`
- `algorithm.class_name="PPOAMP"`
- `experiment_name="g1_amp"`
- PPO: 24 steps/env, 5 epochs, 4 minibatches, adaptive KL
- AMP discriminator: `[1024, 512]`, LSGAN, style reward scale 默认 5.0
- G1 symmetry augmentation 和 mirror loss 开启

脚本暴露的常用覆盖项包括：

- `NUM_ENVS`, `MAX_ITERATIONS`, `SEED`, `DEVICE`, `AGENT_DEVICE`
- `RSI_ENABLE`, `RSI_RATIO`, `POS_RSI`
- `RANDOMIZATION_STRENGTH`
- `TRACK_LIN_WEIGHT`, `TRACK_ANG_WEIGHT`
- `STYLE_REWARD_SCALE`, `TASK_STYLE_LERP`, `AMP_GRAD_PENALTY_SCALE`
- `BASELINE_KL_*`
- `EXTRA_HYDRA_ARGS`

## 当前启动风险

S3 XML 和原 G1 的同名 29 个电机关节属性一致，新增部分主要是 2F85 夹爪、front/back pack、go_pro 等 link。训练可以保持原 29 维接口启动，但需要注意：

- S3 MJCF 会在启动时转换 USD，所以本地必须能正常加载 Isaac Sim / MJCF importer。
- 新 S3 目录只包含部分原 G1 mesh；当前已在 `s3_g1_29dof/meshes` 中为缺失的原 G1 mesh 建相对 symlink 到 `g1_29dof/meshes`。
- 新增 link 会改变动力学和接触几何；即使 policy 维度不变，原 checkpoint 或超参表现也可能变化。



根据配置文件的分析，**速度命令（velocity command）在episode内是时变的**，具体机制如下：

## 命令的时变机制

### 1. **重采样间隔 (Resampling)**
在 amp_env_cfg.py 中，基础配置设置：
```python
resampling_time_range=(10.0, 10.0)
```
这意味着**命令每 10 秒重新采样一次**。每次重采样时，会在 `lin_vel_x=(-0.2, 1.5)` 范围内随机生成一个新的目标速度。

### 2. **命令平滑与加速度限制 (Smoothing)**
在重采样间隔之间，命令不是瞬间跳转，而是通过 CurvatureVelocityCommand._update_command() 逐步调整：

- **平滑时间常数**: `smoothing_time_constant = 0.30 s`（一阶低通滤波）
- **线性加速度限制**: `max_linear_accel = 0.80 m/s²`
- **角速度加速度限制**: `max_yaw_accel = 1.00 rad/s²`

### 3. **实际行为时序示例**
```
Time = 0s     → 新目标采样: 例如 v_cmd = 0.5 m/s
Time = 0-10s  → 通过平滑/加速度限制逐步调整到 0.5 m/s
Time = 10s    → 新目标采样: 例如 v_cmd = 1.2 m/s  
Time = 10-20s → 平滑过渡到 1.2 m/s
**一个 episode 是 20 秒**。

这在 amp_env_cfg.py 中定义：
```python
self.episode_length_s = 20.0
```

G1 AMP 环境继承了这个设置，没有覆盖。

### 补充信息
- **物理仿真步长**: `sim.dt = 0.005` (5 ms)
- **执行步数**: `decimation = 4` (每 4 个物理步执行一次策略)
- **每 episode 的策略步数**: 20 / (0.005 × 4) = **1000 步**

这与 SKILL.md 中提到的训练数据 `episode_length≈1000` 相符。
```

### 4. **当前G1 AMP配置的特点**
- **只训练前向速度**: `lin_vel_x=(-0.2, 1.5)` 范围内变化
- **侧向和yaw速度固定为 0**: `lin_vel_y=(0.0, 0.0)`, `ang_vel_z=(0.0, 0.0)`
- 这确保策略学习纯前向行走，不需要处理转向

**总结**: 命令是**时变且平滑的**，不是恒定值，也不是阶跃跳变，而是每 10 秒采样新目标后平滑过渡。


基于代码分析，**Policy Observation 的 96 维由以下 6 项构成**：

## Policy Observation (96 维)

| 项目 | 函数 | 维度 | 说明 |
|------|------|------|------|
| 1. **基础角速度** | `mdp.base_ang_vel` | **3** | 机器人躯干在世界坐标系下的角速度 (ωx, ωy, ωz) |
| 2. **投影重力** | `mdp.projected_gravity` | **3** | 重力向量投影到机器人身体坐标系（用于感知倾斜） |
| 3. **速度命令** | `mdp.generated_commands` | **3** | 当前采样的目标速度命令 (vx_cmd, vy_cmd, ωz_cmd) |
| 4. **关节相对位置** | `mdp.joint_pos_rel` | **29** | 29 个 locomotion 关节的相对位置（相对于默认值） |
| 5. **关节相对速度** | `mdp.joint_vel_rel` | **29** | 29 个 locomotion 关节的相对速度 |
| 6. **上一步动作** | `mdp.last_action` | **29** | 上一步应用的 29 维关节目标位置 |
| **总计** | | **96** | |

---

## 噪声配置
部分观测项添加了**训练时噪声**（提升鲁棒性）：
- `base_ang_vel`: ±0.35
- `projected_gravity`: ±0.05  
- `joint_pos_rel`: ±0.03
- `joint_vel_rel`: ±1.75
- `actions`: 无噪声

---

## 其他观测组（Critic & Discriminator）

**Critic Observation** (有特权信息)：
- 额外包含 `base_lin_vel` (3维)
- `joint_pos` 和 `joint_vel` 使用**绝对值**而非相对值
- `history_length = 3`（用于RNN或多步回顾）

**Discriminator Observation** (用于判别真动作)：
- 包含 `root_local_rot_tan_norm` (6维)：躯干本地坐标系的切向/法向向量
- `history_length = 10`：保持 10 步历史用于判别
- 不与命令关联（判别器只看动作流，不看指令）


## 当前 G1 AMP 任务中的随机化 / 扰动强度

### 1. 启动阶段的系统随机化（`startup`）
这些参数在环境一开始创建时随机化，作用于每个 env 的物理属性和机器人参数。

- `physics_material`
  - `static_friction_range = (0.3, 1.6)`
  - `dynamic_friction_range = (0.3, 1.2)`
  - `restitution_range = (0.0, 0.5)`
  - 含义：摩擦、弹性系数在这些范围内随机采样，用于改变接触摩擦/弹性。

- `add_base_mass`
  - `mass_distribution_params = (-3.0, 3.0)`
  - `operation = "add"`
  - 含义：在 `base_link` 上额外加/减质量，范围约 ±3kg。

- `randomize_rigid_body_com`
  - `com_range = {"x":(-0.03,0.03), "y":(-0.03,0.03), "z":(-0.03,0.03)}`
  - 作用对象：`torso_link` 和 `base_link`
  - 含义：质心位置在身体坐标系内随机偏移最多 3 cm。

- `scale_link_mass`
  - `mass_distribution_params = (0.8, 1.2)`
  - `operation = "scale"`
  - 作用对象：左右肢体链接 `left_.*_link`, `right_.*_link`
  - 含义：四肢链接质量成比例缩放到 80%-120%。

- `scale_actuator_gains`
  - `stiffness_distribution_params = (0.8, 1.2)`
  - `damping_distribution_params = (0.8, 1.2)`
  - `operation = "scale"`
  - 作用对象：全部 29 个 locomotion joint
  - 含义：执行器刚度和阻尼按 0.8~1.2 倍随机放缩。

- `scale_joint_parameters`
  - `friction_distribution_params = (1.0, 1.0)`
  - `armature_distribution_params = (0.8, 1.2)`
  - `operation = "scale"`
  - 作用对象：全部 locomotion joint
  - 含义：关节摩擦不变，惯性臂参数按 0.8~1.2 缩放。

### 2. 复位阶段的扰动 / 初始状态随机化（`reset`）
这些扰动会在每个 episode reset 时应用。

- `base_external_force_torque`
  - `force_range = (0.0, 0.0)`
  - `torque_range = (-0.0, 0.0)`
  - 含义：当前实际上没有外力/外扭矩扰动，等于禁用。

- `reset_base`
  - `pose_range`:
    - `x, y` 都是 `(-0.5, 0.5)` m
    - `yaw` 是 `(-3.14, 3.14)` rad
  - `velocity_range`:
    - `x, y, z` 都是 `(-0.2, 0.2)` m/s
    - `roll, pitch, yaw` 都是 `(-0.2, 0.2)` rad/s
  - 含义：机器人起始位置平移可达 0.5m，航向可随机全圈；线速度/角速度有 ±0.2 的随机偏置。

- `reset_robot_joints`
  - 基类默认是 `position_range=(0.8, 1.2)`，`velocity_range=(0.0, 0.0)`
  - 但 G1 专用配置覆盖为：
    - `position_range = (1.0, 1.0)`
    - `velocity_range = (0.0, 0.0)`
  - 含义：当前 G1 AMP 配置里，关节重置不再做随机缩放，关节位置保持默认、速度为零。

- `reset_from_ref`
  - 在 G1 config 中启用：
    - `mdp.ref_state_init_subset`
    - `rsi_ratio = 0.5`
    - `pos_rsi = False`
    - `height_offset = 0.05`
  - 含义：一半的关节/状态来自参考动画状态，用于 RSI 初始化；这是“参考状态初始化”而不是标准噪声。

### 3. 间隔扰动（`interval`）
- `push_robot`
  - `interval_range_s = (10.0, 20.0)`
  - `velocity_range = {"x":(-0.5,0.5), "y":(-0.5,0.5), "yaw":(-1.0,1.0)}`
  - 含义：每 10~20 秒随机一次，通过直接设置根速度实现“推一下”效果：
    - x/y 速度扰动 ±0.5 m/s
    - yaw 角速度扰动 ±1.0 rad/s

### 4. 观测层面的噪声
这些不是物理扰动，但也是训练时的扰动来源。

- `base_ang_vel`：±0.35
- `projected_gravity`：±0.05
- `joint_pos_rel`：±0.03
- `joint_vel_rel`：±1.75
- `actions`：无噪声

### 5. 需要特别说明的几个点

- `base_external_force_torque` 在当前 G1 配置中已经是“零扰动”，因此不真正施加外力/外扭矩。
- `reset_robot_joints` 在 G1 任务中也被取消随机缩放，说明训练里关节初始状态更严格。
- 大多数真正的随机化来自：
  - 物理材料
  - 质量 / COM / 链接质量
  - 执行器增益
  - 关节惯量臂
  - 复位时的根位置/速度
  - 间隔推力 `push_robot`


将数据按 speed_xy = sqrt(vx^2 + vy^2) 分段后：

0.00-0.10 m/s
mean(|wz|)=0.182
median(|wz|)=0.15
|wz|≤0.1 占比：29.2%
|wz|>0.3 占比：16.5%

0.10-0.20 m/s
mean(|wz|)=0.135
median(|wz|)=0.00
|wz|≤0.1 占比：60.8%
|wz|>0.3 占比：22.3%

0.20-0.30 m/s
mean(|wz|)=0.142
median(|wz|)=0.0537
|wz|≤0.1 占比：56.8%
|wz|>0.3 占比：20.4%

0.30-0.40 m/s
mean(|wz|)=0.0757
median(|wz|)=0.05
|wz|≤0.1 占比：80.6%
|wz|>0.3 占比：4.4%

0.40-0.50 m/s
mean(|wz|)=0.0503
median(|wz|)=0.0268
|wz|≤0.1 占比：81.4%
|wz|>0.3 占比：0.1%

0.50-0.62 m/s
mean(|wz|)=0.0006
median(|wz|)=0.0
|wz|≤0.1 占比：99.8%
|wz|>0.3 占比：0.0%

vx: (-0.4,1.0) vy: (-0.3,0.3)
将数据按 speed_xy = sqrt(vx^2 + vy^2) 分段后：
random时（当前数据中 speed_xy < 0.4 时：）呈现厚尾分布，平均值和中位数都较高，且有显著的高转向样本（|wz| > 0.3）。选择 wz（-0.5,0.5）范围的平均采样
walk时（当前数据中 speed_xy >= 0.4 时：）wz 均值：≈ 0，wz 标准差：≈ 0.08 rad/s，使用高斯分布采样，平均值 0，标准差 0.1 rad/s，限制在 (-0.3, 0.3) 范围内。


`wz` 的定义是：从 root/base quaternion 取 yaw，做 unwrap，然后对 yaw 求时间导数，再做 0.2s moving average。所以它是 **root/base 的平面航向角速度**，单位 rad/s。正号按 z-up 右手系，通常对应逆时针/左转。

1. 数据侧是 **root yaw frame**，不是完整 body frame。它只用 yaw 旋转世界速度，没有用 roll/pitch。这通常更好，因为行走时 roll/pitch 不应该污染水平速度命令。

2. 当前主奖励 `track_lin_vel_xy_exp` 用的是 `asset.data.root_lin_vel_b[:, :2]`，`track_ang_vel_z_exp` 用的是 `root_ang_vel_b[:, 2]`：  

任务维度本身 `[vx, vy, wz]` 合理，但要和数据支持匹配。当前 core 只有 51.46s，其中 walk 部分 829 帧，random 部分 1744 帧；而训练 sampler 会在整个 box 里采样。低速大转向、横移、倒退这些组合如果数据里少，AMP prior 可能会和速度 tracking reward 打架。建议训练时先用当前 segmented yaw 范围做 curriculum，后续再扩大或引入更多转向/横移 retarget 数据。**