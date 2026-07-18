# ArmHack Walk 训练对照与改进报告

> 审计日期：2026-07-18
>
> 范围：只分析 ArmHack Walk，不修改 Stand，也不在本报告阶段修改训练代码。
>
> 目标：在 S3 G1 双臂保持指定固定姿态、腕部末端负载变化时，继续稳定追踪 Nav2 `vx/vy/wz`，同时保留 model9996 已有的鲁棒、仿人、全向行走能力。

## 1. 结论摘要

当前 Walk 不是“完整复刻原 Nav2 训练并增加固定双臂”，而是把三部分组合在了一起：

1. actor 起点来自 S3 G1 的 Command-Balanced Directional V3 `model_9996.pt`；
2. command 和 task reward 使用旧 Nav2 Stage-4 的 `complex_turn` 语义；
3. 双臂由环境覆盖成固定姿态，并从 task regularizer 和 AMP discriminator 的关节项中排除。

目前最需要补齐的不是某一个奖励权重，而是以下四组训练内容：

- **重新启用排除双臂后的 AMP style reward**。下肢/腰部 discriminator mask 已经存在，缺的是让 style reward 真正进入 PPO return；
- **加入腕部末端负载随机化**。左右 `wrist_yaw_link` 应分别在 `[0,1] kg` 内独立采样附加质量，并重算惯量；
- **恢复物理 domain randomization**。当前只保留观测噪声与 reset 根状态随机化，原策略和旧 Nav2 使用的摩擦、质量、质心、执行器参数与 push 均被关闭；
- **补回全向 mode 和更宽速度范围的持续训练**。当前 `complex_turn` CSV 分布明显窄于 model9996 的八类 mode 分布，长时间只训练当前分布会产生能力遗忘。

用户报告的现有正式 Walk 已训练 3000 iteration，但本机 checkout 只有 2026-07-14 的 Walk smoke 及其解析配置，没有这次正式 run 的 checkpoint、TensorBoard 和 `params/`，所以本报告不对该模型的具体收敛曲线作未经验证的结论。源码脚本当前默认值其实是 `MAX_ITERATIONS=4000`；“3000 iteration”是本次实际实验设置，不是当前脚本默认值。

建议不要从 model9996 重新开始，也不要只把现有纯 task 配置机械地再跑数千轮。应从当前正式 Walk checkpoint 做 full-state resume，按“AMP → 末端负载 → command/mode 扩展 → 其他物理随机化”逐项加入，建议总训练预算先提高到约 10,000 iteration，并按固定评估结果决定是否继续到 12,000，而不是只看 iteration 数。

## 2. 三个容易混淆的对照对象

### 2.1 当前 Walk

任务为 `LeggedLab-Isaac-AMP-G1-WalkPerturbFinetune-v0`，入口为：

```text
scripts/train_g1_armhack_walk.sh
  -> G1WalkPerturbFinetuneEnvCfg
  -> Nav2RecordedVelocityCommand
  -> G1WalkPerturbAmpEnv
```

当前默认特征：S3 G1、model9996 policy-only 初始化、固定/随机三种双臂姿态、Nav2 `complex_turn` 连续窗口、纯 task reward、baseline KL、无 RSI、关闭标准物理 randomization。

### 2.2 当前 actor 的真正来源：model9996

`checkpoint/model_9996/locomotion.deploy.json` 记录的源模型是：

```text
2026-06-17_03-48-38_
s3_g1_29dof_command_balanced_directional_strict_armprior_v3_resume8997_1000/
model_9996.pt
```

所以 model9996 不是旧 Nav2 Stage-4 模型。它来自 S3 G1 的 Command-Balanced Directional V3 训练，使用八类均衡 mode、较宽速度范围、AMP、对称增强和完整物理 randomization。

### 2.3 旧 Nav2 Stage-4

旧 Nav2 Stage-4 的本机 run 是：

```text
logs/rsl_rl/g1_amp/
2026-06-01_04-23-10_finetune_nav2_stage4_complex_only/
```

它使用 original `g1_29dof`，不是当前 S3 G1；其 checkpoint 也不是当前 Walk 的起点。它的价值在于提供已验证的 Nav2 command、reward、AMP、KL 和 randomization 训练方法对照。

## 3. 当前 Walk、model9996 与旧 Nav2 的代码级比较

| 项目 | 当前 ArmHack Walk | model9996 源训练 | 旧 Nav2 Stage-4 |
|---|---|---|---|
| 机器人 | `s3_g1_29dof` | `s3_g1_29dof` | original `g1_29dof` |
| command | Nav2 录制窗口 | `ModeBalancedVelocityCommand` | Nav2 录制窗口 |
| command 采样 | 只选 `complex_turn`，4 s 连续窗口 | 8 个 mode，每 8 s 重采样 | 只选 `complex_turn`，4 s 连续窗口 |
| 配置范围 | `vx[-0.2,0.6]`、`vy[-0.3,0.3]`、`wz[-0.6,0.6]` | `vx[-0.65,1.05]`、`vy[-0.45,0.45]`、`wz[-1,1]` | 与当前 Walk 相同 |
| 当前数据实际覆盖 | 见第 6 节，窄于配置 clip | 由 8-mode 配置直接采样完整范围 | 与当前 Walk 使用同类 Stage-4 分布 |
| AMP motion | `accad_g1used_50hz` | `command_balanced_directional_50hz` | `accad_g1used_50hz` |
| discriminator 关节 | 腿+腰 15DoF；排除双臂 | 原 29DoF | 原 29DoF |
| AMP style scale | `0` | `8` | `5` |
| task/style lerp | `1.0`，即纯 task | `0.50` | `0.36` |
| PPO learning rate | `3e-5` | `1e-4` | `3e-5` |
| entropy | `0.002` | `0.01` | `0.01` |
| baseline KL | 开启，`0.003`，锚定 model9996 | 关闭 | 开启，`0.006` |
| symmetry/mirror loss | 开启，`0.1` | 开启，`0.1` | 开启，`0.1` |
| policy observation noise | 开启 | 开启 | 开启 |
| RSI | 关闭 | 开启，ratio `0.5` | 开启，ratio `0.5` |
| 标准物理 randomization | 关闭 | 开启 | 开启 |
| 双臂末端 `0–1 kg` | 缺少 | 缺少 | 缺少 |
| task reward | 锁定旧 Nav2 Stage-4；手臂项去掉、regularizer 只看腿腰 | Directional V3 gait shaping | Nav2 Stage-4 |

当前 Walk 已经保留的内容也应明确：

- 96 维 policy observation 和 29 维 actor 输出不变；
- 双臂仍在当前 `joint_pos/joint_vel/last_action` 中，策略能根据当前身体状态反馈补偿，但没有未来双臂目标；
- Nav2 Stage-4 的 18 个非零 task reward 已恢复；
- 双臂不参与 torque、acc、action-rate、arm deviation 和 arm style regularizer；
- mirror loss、baseline KL、观测噪声、根节点 reset 位姿/速度随机化仍然存在；
- 三个双臂姿态可按名字固定，也可在 reset 时均衡采样并在 episode 内保持不变。

因此不能把当前状态概括成“完全没有随机化”或“完全没有 AMP 结构”。准确说法是：**AMP 网络仍在训练，但 AMP reward 未进入 policy return；观测与 reset 仍有随机性，但物理参数和外力 randomization 被关闭。**

## 4. 旧 Nav2 使用了哪些当前 Walk 没有使用的训练方法

旧 Nav2 不是一次性从头跑一个 Stage-4，而是从已有 AMP locomotion checkpoint 连续/分支微调：

| 阶段 | 追加 iteration | command 变化 | AMP / KL |
|---|---:|---|---|
| Stage-1 | 1500 | 全 family，2 s 窗口，scale `(0.70,0.55,0.55)` | style `5`，lerp `0.35`，KL `0.025` |
| Stage-2 | 1500 | 全 family，4 s 窗口，scale `(0.85,0.75,0.75)` | style `5`，lerp `0.40`，KL `0.012` |
| Stage-3 | 1000 | 全 family，加强 lateral/complex，scale `(0.90,1.00,0.80)` | style `5`，lerp `0.38`，KL `0.008` |
| Stage-4 | 600 | 从 Stage-2 分支，只选 `complex_turn`，MPPI 权重 `1.5` | style `5`，lerp `0.36`，KL `0.006` |

当前 Walk 与这条链相比缺少：

1. 从易到难、从全分布到重点困难分布的 command 课程；
2. 训练全过程中的 AMP style reward；
3. 逐步放松的 baseline KL 课程；
4. friction、mass、CoM、actuator、armature 和 push randomization；
5. RSI 的参考状态初始化。

其中 RSI 不能直接照搬。旧 Walk smoke 曾出现 `RSI=True` 时约一半环境发生 bad-orientation，与 `rsi_ratio=0.5` 接近。固定双臂任务若重新引入 RSI，必须做“只初始化腿腰/根状态，随后再次同步固定双臂和 action history”的专用实现与测试；在此之前继续关闭 RSI 是合理选择，不应把它作为本轮 P0 修改。

## 5. AMP 奖励：缺的是什么，怎样排除双臂

### 5.1 当前状态

`g1_walk_perturb_env_cfg.py::_configure_perturbation_common()` 已经完成正确的关节 mask：

- policy discriminator 的 `joint_pos/joint_vel` 只取 15 个腿腰关节；
- demonstration discriminator 的 `ref_joint_pos/ref_joint_vel` 使用相同 15 个 joint id；
- root local rotation、root linear velocity 和 root angular velocity仍保留；
- 4 帧 discriminator history 保持不变。

所以“AMP 排除双臂”的数据边界已经有了，不需要把双臂设为零后送进原 29DoF discriminator，也不能把未来手臂轨迹或姿态编号加入 observation。

真正缺失的是 runner 和训练脚本把：

```text
style_reward_scale = 0.0
task_style_lerp = 1.0
```

锁死了。AMP 的组合公式为：

```text
total_reward = task_style_lerp * task_reward
             + (1 - task_style_lerp) * style_reward
```

因此当前 style reward 对策略更新的贡献严格为零。值得注意的是，PPOAMP 仍会更新 discriminator；如果现有 3000-iteration checkpoint 是 full checkpoint，则其 lower-body discriminator、normalizer 和 optimizer 已保存。后续应使用 `MODE=resume`，不能 policy-only 重启，否则会丢掉这部分状态。

### 5.2 建议做法

不建议从 `style=0, lerp=1` 一步跳到 model9996 的 `style=8, lerp=0.5`。固定双臂和新负载改变了动力学，应先做 AMP warm-in：

| AMP 阶段 | 建议初始值 | 目的 |
|---|---|---|
| AMP-A | `style=1`，`lerp=0.85` | 以 task tracking 为主，确认 lower-body style reward 数值稳定 |
| AMP-B | `style=3`，`lerp=0.70` | 增强仿人步态，观察速度误差、足滑和 termination |
| AMP-C | `style=5`，`lerp=0.55–0.65` | 接近旧 Nav2 强度；只有前两阶段通过后再使用 |

这些是首轮实验起点，不是无需验证的最终超参数。每阶段必须记录 `amp/style_reward`、discriminator score/loss、task reward、速度 MAE、足滑、episode length 和 termination，防止 style reward 提高但 command tracking 退化。

第一轮只恢复 reward，不同时更换 AMP motion 数据。当前先继续用旧 Nav2 的 `accad_g1used_50hz`，这样只改变一个变量；等 AMP-C 稳定后，再单独比较是否切换为 model9996 使用的 `command_balanced_directional_50hz` 下肢 demo。

## 6. command mode 与速度范围缺口

### 6.1 当前 Nav2 CSV 的真实覆盖

本报告直接统计了当前本机文件：

```text
Reference Data/ArmHack/WalkPerturbFinetune/nav2_cmd_vel_raw_success.csv
SHA-256: 76a4516588b855351eb3eb8c2da26e291603876c1a4a1b9c7bacd77a53807b5a
总行数: 331,010
```

当前硬过滤 `scenario_family=complex_turn` 后只剩 20,717 行、26 个原始轨迹组；在线生成左右镜像后是 41,434 行、52 组。也就是说，配置中为其他 family 写的 `scenario_family_weights` 在当前任务里不起作用，只有 controller 权重仍会改变所选组概率。

对当前 `scale=(0.85,0.75,0.75)` 和 clip 应用后的实际分布统计如下：

| 轴 | 最小值 | 1% | 中位数 | 99% | 最大值 |
|---|---:|---:|---:|---:|---:|
| `vx` m/s | -0.170 | -0.170 | 0.287 | 0.496 | 0.510 |
| `vy` m/s | -0.225 | -0.225 | 0.000 | 0.225 | 0.225 |
| `wz` rad/s | -0.450 | -0.360 | 0.000 | 0.360 | 0.450 |

按 model9996 mode 阈值做近似统计：`vx<-0.15` 约 14.0%，`|vy|>=0.15` 约 67.3%，`|wz|>=0.25` 只有约 4.7%，`vx>=0.55` 为 0。这里的 lateral 统计包含“前进+侧向”的复合命令，不能等同于纯侧步 mode。

这解释了为什么当前模型可能在 Nav2 CSV 上继续提高，却逐渐遗忘高速前进、强转向、纯后退和 mode 切换能力。把配置里的 clip 写得更宽不会自动生成数据中不存在的速度。

### 6.2 model9996 原有的 mode 能力

model9996 的源训练按以下八类 mode 均衡采样：

| mode | 权重 | 主要范围 |
|---|---:|---|
| `forward_slow` | 0.16 | `vx [0.20,0.55]` |
| `forward_normal` | 0.22 | `vx [0.55,1.05]` |
| `backward` | 0.18 | `vx [-0.65,-0.15]` |
| `lateral_left` | 0.11 | `vy [0.15,0.45]` |
| `lateral_right` | 0.11 | `vy [-0.45,-0.15]` |
| `turn_left` | 0.08 | `wz [0.25,1.0]` |
| `turn_right` | 0.08 | `wz [-1.0,-0.25]` |
| `stand` | 0.06 | `(0,0,0)` |

policy 并没有接收额外 mode id；mode 只是 command sampler 的分层标签，policy 仍只观察当前 `vx/vy/wz`。因此恢复 mode 训练不会给策略未来信息，也不改变 96 维部署接口。

### 6.3 应不应该补 mode

**应该补，但理由必须写准确。** 旧 Nav2 Stage-4 本身没有使用 `ModeBalancedVelocityCommand`；它使用的是录制 Nav2 连续窗口和多阶段 scenario/scale 课程。补 mode 是为了防止当前 Walk 遗忘 model9996 已有的全向宽范围能力，而不是“恢复 Stage-4 漏掉的一项配置”。

最终任务仍以 Nav2 为主，所以不建议直接把 Nav2 command 整体替换成均匀 mode。建议实现一个 Walk 专用混合 command sampler：

- 主要样本继续来自 Nav2 连续窗口，保留真实控制器时间相关性；
- 少量样本来自八类 mode，专门覆盖高速、后退、侧步和强转向；
- policy 只看到当前三维 command，不看到数据源、mode id 或未来窗口；
- 训练和评估分别记录 Nav2 source 与各 mode 的指标，不能只报总体均值。

建议 command 课程：

| 阶段 | Nav2 / mode 比例 | command 范围 |
|---|---|---|
| CMD-A | 100% / 0% | 当前 `complex_turn`，作为稳定对照 |
| CMD-B | 100% / 0% | 放开全部 Nav2 family，仍用当前 scale；增加场景多样性 |
| CMD-C | 80% / 20% | Nav2 scale 逐步到 `(1,1,1)`；mode 先限制在中低速 |
| CMD-D | 70% / 30% | mode 扩展到 model9996 完整范围，Nav2 仍占多数 |

比例和边界需要通过固定评估确定。不能只改变 `Nav2RecordedVelocityCommandCfg.ranges`，因为当前 loader 的实际命令来自 `dataset * command_scale -> clip`，`ranges` 不是独立采样范围。

## 7. 末端负载与全身物理 randomization

### 7.1 必须新增的末端负载

Walk 应复用 Stand 已验证的物理事件定义，在环境 startup 时对每个并行环境独立采样：

```text
body_names = [left_wrist_yaw_link, right_wrist_yaw_link]
operation = add
distribution = uniform
mass_distribution_params = [0.0, 1.0] kg
recompute_inertia = True
```

左右两侧分别采样，单个环境内负载在 episode 期间保持不变；4096 个并行环境共同覆盖 `[0,1] kg` 分布。策略不观察 payload 数值，只能通过当前姿态、速度和动力学反馈适应。

建议新增 `env.events.randomize_end_effector_payload`，不要把它混入已有 `scale_link_mass` 的正则表达式中。Play/评估任务应保留同一事件结构，但允许固定为 `[0,0]`、`[0.5,0.5]`、`[1,1] kg`，从而生成可复现实验。

### 7.2 当前被关闭的原有物理 randomization

`scripts/train_g1_armhack_walk.sh` 当前设置 `RANDOMIZATION_STRENGTH=0`，通用入口会删除：

- 地面/机身材料：静摩擦 `[0.3,1.6]`、动摩擦 `[0.3,1.2]`、恢复系数 `[0,0.5]`；
- torso 附加质量 `[-3,3] kg`；
- torso CoM 三轴各 `[-0.03,0.03] m`；
- 左右 link 质量 scale `[0.8,1.2]`；
- actuator stiffness/damping scale `[0.8,1.2]`；
- joint armature scale `[0.8,1.2]`；
- 每 `10–20 s` 的根速度 push：`x/y [-0.5,0.5] m/s`、yaw `[-1,1] rad/s`。

model9996 和旧 Nav2 Stage-4 的解析配置中上述项都开启。当前 Walk 仍保留 observation corruption、根节点 reset 位姿/速度随机化，因此不是完全确定性训练。

### 7.3 建议加入顺序

不能在同一个 continuation run 中同时首次打开 AMP、`0–1 kg` payload、全部物理 randomization 和完整 mode 范围，否则失败时无法定位原因。建议顺序：

1. payload-only：先 `[0,0.5] kg`，通过后再到 `[0,1] kg`；
2. friction + torso mass/CoM；
3. actuator gain + armature；
4. link mass scale；
5. interval push 最后加入。

`scale_link_mass` 会覆盖 wrist link，而末端事件还会额外加质量；实现时必须固定 startup event 顺序，并在日志中分别记录基础 link mass scale 和最终附加 payload，避免重复随机化含义不清。

## 8. 为什么 3000 iteration 仍可能未收敛

在 `4096 env × 24 step/iteration` 下，3000 iteration 已约有 2.95 亿条 environment transitions，所以它在数据量上并不“小”。当前仍未收敛更可能同时受以下因素影响：

- 从 mode-balanced model9996 切换到窄 `complex_turn` 分布，目标分布发生明显变化；
- actor 仍输出 29 维，但 14 维手臂 action 在物理 step 前被覆盖，PPO 的 log-prob、entropy、mirror loss 和 KL 仍包含这些无效 action 维度；
- style reward 为零，策略只在速度/稳定 task reward 下优化，仿人 gait 没有直接约束；
- 三种固定手臂姿态改变了质心和惯性，但当前没有 payload/物理随机化课程；
- 单一总体 return 可能掩盖某个 pose、某个速度象限或某种 termination 的失败；
- 当前没有基于固定 command window 的训练中独立 validation，无法区分“训练 reward 还在上升”和“泛化已经退化”。

因此，延长训练是必要条件之一，但不是单独解决方案。先修正训练目标和覆盖，再继续训练，比在当前纯 task、窄分布配置上直接跑到 10,000 更可靠。

## 9. 建议的 3000 → 10000 分阶段续训

以下把用户报告的 3000-iteration 正式模型记为 `WALK_3000`。实施前必须填写其真实 run 路径、checkpoint SHA 和保存的 `params/env.yaml`、`params/agent.yaml`，确认它确实由 model9996 启动。

| 阶段 | 目标累计 iteration | 主要变化 | 保持不变 | 晋级检查 |
|---|---:|---|---|---|
| W0 固定基线 | 3000 | 不训练；冻结 checkpoint 和评估集 | 当前配置 | 三姿态、固定 Nav2 window、0 kg 的成对基线 |
| W1 AMP warm-in | 4000 | lower-body AMP-A | command、payload、物理参数不变 | style/disc 数值稳定；速度 MAE和存活不退化 |
| W2 AMP target | 5000 | AMP-B，必要时再到 AMP-C | 仍无新 randomization | 步态、足滑改善；worst-pose 不退化 |
| W3 payload | 6500 | 先 `0–0.5 kg`，再 `0–1 kg` | 当前 command | 0/0.5/1 kg 固定测试均通过 |
| W4 command 扩展 | 8500 | 全 Nav2 family，再加入 20–30% mode | AMP、payload 保持 | 各 mode/速度 bin 均有覆盖，无能力遗忘 |
| W5 物理鲁棒性 | 10000 | 分批恢复其他物理 randomization，push 最后 | command 混合比例固定 | 扰动下存活、跟踪与 gait 达标 |

若 10,000 时固定 validation 仍持续改善，可以继续到 12,000；若连续三个评估点没有改善或 worst-case 退化，应调整课程/奖励，而不是继续堆 iteration。

所有阶段都应 full-state resume，保留 actor、critic、optimizer、AMP discriminator/normalizer、iteration 和当前 action noise。每次只改变表中的主要变量，并使用新 run name 保存，不能覆盖 `WALK_3000`。

## 10. 必须新增的验证矩阵

训练是否成功不能只看 GUI 或平均 return。至少需要以下固定矩阵：

| 维度 | 固定取值/分箱 |
|---|---|
| checkpoint | model9996、WALK_3000、每个新阶段末 checkpoint |
| arm pose | `pos1_back`、`pos2_down`、`pos3_front` |
| payload | 每侧 `0`、`0.5`、`1.0 kg`；另测左右不对称 `0/1` 与 `1/0 kg` |
| command source | 固定 Nav2 windows；8 个 mode |
| 速度 bin | stand、慢前进、正常前进、后退、左右侧步、左右转向、复合命令 |
| randomization | nominal；friction/mass/CoM/gain 边界；有/无 push |
| seed | 至少 5 个固定 seed，成对复用 |

每组至少输出：

- episode completion、timeout、base-height/bad-orientation termination；
- `vx/vy/wz` MAE、RMS、95% 分位和响应延迟；
- torso roll/pitch、角速度、竖直速度；
- feet slide、air time、双脚腾空比例；
- task reward、style reward、总 reward、discriminator score/loss；
- baseline KL；
- payload 与 mode 分箱后的 worst-case，而不仅是总体均值。

首轮验收建议使用相对标准：在相同固定 command/seed 下，新增 AMP、payload 或 randomization 后，相对上一阶段的速度跟踪和完成率不能显著退化，同时 gait/足滑指标必须改善或保持。绝对阈值应在 W0/WALK_3000 的真实评估结果出来后确定，不能先凭一次视频拍脑袋设定。

## 11. 下一步需要修改的文件

本报告没有改代码。实施时建议只改 Walk 专属路径：

| 文件 | 计划修改 |
|---|---|
| `source/.../g1_perturb/g1_walk_perturb_env_cfg.py` | 增加 wrist payload event；保留 discriminator 去臂；增加 Walk 专用 randomization 配置 |
| `source/.../g1_perturb/agents/rsl_rl_ppo_cfg.py` | 增加可审计的 Walk AMP 非零配置 |
| `scripts/train_g1_armhack_walk.sh` | 允许并锁定 AMP/负载/课程参数；提高训练预算；保持与 Stand 隔离 |
| `mdp/commands/` 下新的 Walk 混合 command | Nav2 连续窗口 + mode-balanced 分层采样；不增加 policy 输入 |
| Walk Play/评估入口 | 固定 payload、command source、window、mode 和 seed |
| `source/legged_lab/test/test_g1_perturb_static.py` | 更新当前强制 `style=0` 的旧断言；增加 payload、AMP mask 和 command 课程检查 |
| `docs/armhack.md`、`docs/armhack_train.md` | 代码实现并测试后再把计划状态改成“已实现” |

必须保持以下边界：

- 不修改 Stand 的任务类、runner、脚本和当前 checkpoint；
- policy 不观察 payload、mode id、未来 Nav2 window 或未来手臂姿态；
- 手臂固定目标仍由环境覆盖，AMP 和 task regularizer 均不惩罚不可控的手臂 action；
- 所有新训练从明确的正式 Walk checkpoint 恢复，不从 Stand checkpoint 开始；
- 每个配置变化都写入 resolved `params/env.yaml`、`params/agent.yaml` 并用静态测试保护。

## 12. 改进优先级

| 优先级 | 工作 | 原因 |
|---|---|---|
| P0 | 冻结 WALK_3000 并完成成对基线测试 | 没有基线就无法判断后续修改是否有效 |
| P0 | 开启排除双臂后的 AMP reward | 当前最直接的仿人 gait 缺口 |
| P0 | 加入左右腕末端独立 `U(0,1 kg)` 负载 | 任务明确要求，且 Stand 已有可复用实现 |
| P0 | 增加 Nav2+mode 混合 command 课程 | 防止丢失 model9996 全向和宽速度能力 |
| P1 | 分批恢复 friction/mass/CoM/gain/link randomization | 提高 sim2real 和负载鲁棒性 |
| P1 | 最后恢复 interval push | 强扰动过早加入会掩盖 AMP/command 问题 |
| P1 | 总预算提高到约 10,000，按 validation 早停 | 3000 尚未收敛，但延长必须和目标修正配套 |
| P2 | 研究 15 个可控 action 的 PPO loss mask | 减少 14 个被覆盖手臂 action 带来的无效梯度方差 |
| P2 | 设计固定双臂兼容的 lower-body RSI | 旧 RSI 直接开启有明确失败记录，不属于本轮首要修改 |
