# G1 极鲁棒站立策略简明使用说明

> 本文正文的 `/home/user/...` 命令用于当前本机。HEC-5090 是另一套独立服务器路径，只有明确需要服务器运行时才参考 `use/HEC5090三模型部署测试说明.md`。

本文同时区分两个模型边界：本机 MuJoCo 测试脚本当前默认使用 **Smooth-Torque V4 `model_2999.pt`**；真机脚本仍锁定历史 Pose V2 ONNX，尚未与 V4 统一。除显式说明外，所有命令均从项目根目录执行。导出与 MuJoCo 启动器使用本机虚拟环境的绝对 Python 路径，不要求当前终端手动执行 `conda activate`；仓库根目录的 `.envrc` 也会在 direnv 可用时自动激活 `env_isaaclab`。

项目根目录：

```text
/home/user/Workspace/Humanoid/Locomotion/G1-Locomotion
```

## 1. 模型路径

### 1.1 训练 checkpoint

```text
/home/user/Workspace/Humanoid/Locomotion/G1-Locomotion/legged_lab/ExtremeStandRecovery Checkpoints/2026-07-31_16-52-27_g1_extreme_stand_recovery_smooth_torque_v4_from_v3_model2999_full_20260731/model_2999.pt
```

这是当前本机 MuJoCo 默认测试的 Smooth-Torque V4 模型，checkpoint SHA256 为 `e0addb8ce23153498d4f805c75f4e3ba19568198f890ffc980160fea7c3b7fff`。当前真机入口仍锁定下面的 Pose V2 ONNX，二者尚未统一。

### 1.2 `use` 中真机使用的 Pose V2 ONNX

```text
/home/user/Workspace/Humanoid/Locomotion/G1-Locomotion/use/extreme_stand_recovery_pose_v2_model2999.onnx
```

该文件是当前真机部署脚本锁定的历史 Pose V2 模型，SHA256 为：

```text
0af2ffb24cd728352804b62669dc5264dd835329528311f9d50b90dbe2d0a0d1
```

### 1.3 当前 V4 的原始导出文件和部署元数据

```text
/home/user/Workspace/Humanoid/Locomotion/G1-Locomotion/legged_lab/ExtremeStandRecovery Checkpoints/2026-07-31_16-52-27_g1_extreme_stand_recovery_smooth_torque_v4_from_v3_model2999_full_20260731/exported_extreme_stand_recovery/policy.pt
/home/user/Workspace/Humanoid/Locomotion/G1-Locomotion/legged_lab/ExtremeStandRecovery Checkpoints/2026-07-31_16-52-27_g1_extreme_stand_recovery_smooth_torque_v4_from_v3_model2999_full_20260731/exported_extreme_stand_recovery/policy.onnx
/home/user/Workspace/Humanoid/Locomotion/G1-Locomotion/legged_lab/ExtremeStandRecovery Checkpoints/2026-07-31_16-52-27_g1_extreme_stand_recovery_smooth_torque_v4_from_v3_model2999_full_20260731/exported_extreme_stand_recovery/policy.deploy.json
```

模型接口为：`96` 维观测输入、`29` 维全身关节动作输出、`50 Hz` 控制频率、固定零速度指令。

## 2. 主要脚本路径

| 功能 | 路径 |
| --- | --- |
| 导出 TorchScript 和 ONNX | `scripts/export_g1_extreme_stand_recovery.sh` |
| MuJoCo 单项、交互和完整套件测试 | `scripts/sim2sim_g1_extreme_stand_recovery_mujoco.sh` |
| 随机初始关节姿态专项测试 | `scripts/test_g1_extreme_stand_random_pose_recovery_mujoco.sh` |
| MuJoCo 汇总报告 | `scripts/summarize_g1_extreme_stand_recovery_mujoco.py` |
| 大推力诊断曲线 | `scripts/plot_g1_extreme_stand_push_diagnostics.py` |
| 真机 ONNX 专用入口 | `scripts/deploy_real_g1_extreme_stand_recovery_onnx.sh` |
| MuJoCo 底层运行器 | `unitree_sim2sim2real/deploy/deploy_mujoco/deploy_mujoco_g1_amp.py` |
| 真机底层运行器 | `unitree_sim2sim2real/deploy/deploy_real/deploy_real_g1_amp.py` |
| 真机 PD、默认角和关节映射配置 | `unitree_sim2sim2real/deploy/deploy_real/configs/g1_amp.yaml` |

## 3. 模型导出与文件校验

重新导出当前 checkpoint：

```bash
cd /home/user/Workspace/Humanoid/Locomotion/G1-Locomotion
bash scripts/export_g1_extreme_stand_recovery.sh
```

导出脚本会自行使用本机 `/home/user/anaconda3/envs/env_isaaclab/bin/python`，不需要提前执行 `conda activate`。该命令导出当前 V4，仅供当前 MuJoCo 测试；不要直接覆盖 `use/extreme_stand_recovery_pose_v2_model2999.onnx`，因为真机脚本仍锁定 Pose V2 的哈希和元数据。

检查三个部署文件：

```bash
cd /home/user/Workspace/Humanoid/Locomotion/G1-Locomotion

EXPORT_DIR="$PWD/legged_lab/ExtremeStandRecovery Checkpoints/2026-07-31_16-52-27_g1_extreme_stand_recovery_smooth_torque_v4_from_v3_model2999_full_20260731/exported_extreme_stand_recovery"
USE_ONNX="$PWD/use/extreme_stand_recovery_pose_v2_model2999.onnx"
ls -lh "$EXPORT_DIR/policy.onnx" "$EXPORT_DIR/policy.deploy.json" "$USE_ONNX"
sha256sum "$EXPORT_DIR/policy.onnx" "$EXPORT_DIR/policy.deploy.json" "$USE_ONNX"
```

## 4. MuJoCo 交互可视化

### 4.1 推荐交互测试

```bash
cd /home/user/Workspace/Humanoid/Locomotion/G1-Locomotion

PROFILE=interactive USE_GLFW=True REAL_TIME=True \
RENDER_FPS=60 REALTIME_STATUS_INTERVAL_S=5 \
FOLLOW_CAMERA=False \
INTERACTIVE_DATA_LOG=True \
SIMULATION_DURATION=300 SEED=20260722 \
bash scripts/sim2sim_g1_extreme_stand_recovery_mujoco.sh
```

若要只验证“方案 1：MuJoCo 关节位置目标速度/加速度限幅”，使用下面的完整命令。该开关只作用于 MuJoCo 的 PD 位置目标，不修改 ONNX 输出、训练奖励或真机部署链路；默认不开启，必须显式写 `TARGET_LIMITER_ENABLE=True`：

```bash
cd /home/user/Workspace/Humanoid/Locomotion/G1-Locomotion

UNITREE_PYTHON="$HOME/anaconda3/envs/gmr/bin/python" \
PROFILE=interactive USE_GLFW=True REAL_TIME=True \
RENDER_FPS=60 REALTIME_STATUS_INTERVAL_S=5 FOLLOW_CAMERA=False \
TARGET_LIMITER_ENABLE=True \
TARGET_LEG_VELOCITY_LIMIT_RAD_S=25.0 \
TARGET_WAIST_VELOCITY_LIMIT_RAD_S=10.0 \
TARGET_ARM_VELOCITY_LIMIT_RAD_S=15.0 \
TARGET_LEG_ACCELERATION_LIMIT_RAD_S2=600.0 \
TARGET_WAIST_ACCELERATION_LIMIT_RAD_S2=250.0 \
TARGET_ARM_ACCELERATION_LIMIT_RAD_S2=400.0 \
INTERACTIVE_DATA_LOG=True \
LARGE_PUSH_FORCE_N=36 LARGE_PUSH_DURATION_S=0.20 \
SIMULATION_DURATION=300 SEED=20260805 \
RESULTS_ROOT="$PWD/legged_lab/logs/monitoring/extreme_stand_v4_target_limiter" \
bash scripts/sim2sim_g1_extreme_stand_recovery_mujoco.sh
```

这组限值是根据既有 V4 trial 标定的“只裁极端尖峰”起点：稳定段和普通恢复尽量保留原策略带宽。更严格的两组实验值都压制了恢复能力：其中 `1.2/0.8/1.5 rad/s`、`12/8/15 rad/s²` 在默认站立约 `1.85 s` 失稳；`6/4/4 rad/s`、`120/80/80 rad/s²` 虽能无外力站立，却在可由原策略承受的 `36 N × 0.20 s` 后推下摔倒，因此均不应使用。窗口内按空格依次测试四个方向；需要更强推力时只把 `LARGE_PUSH_FORCE_N=36` 改为 `120` 或 `360`。摔倒后按 `R` 复位。

这里 `REAL_TIME=True` 表示仿真按墙钟时间以 `1×` 速度运行；物理频率仍为 `500 Hz`、策略频率仍为 `50 Hz`，GUI 只按 `RENDER_FPS=60` 刷新，避免旧实现每个物理步都渲染造成慢放。终端每 5 秒打印一次 `RTF`：接近 `1.000` 即为实时。如果本机图形负载较高、`RTF` 长期小于 `0.95`，可把 `RENDER_FPS=60` 改为 `30`；不要用 `REAL_TIME=False`，后者会取消实时限速并尽可能快地跑仿真。

`FOLLOW_CAMERA=False` 是 Extreme Stand 交互模式的默认值：窗口打开时只设置一次初始观察角度，之后不再写入相机参数，可以在 MuJoCo 中用鼠标自由旋转、平移和缩放视角。运行时按 `C` 可在自由相机和跟随机器人之间切换；跟随模式现在只更新观察中心，不再覆盖鼠标设置的旋转角和缩放距离。

窗口获得焦点后：

- `空格键`：依次循环 `前推(+X)`、`后推(-X)`、`左推(+Y)`、`右推(-Y)`、随机29关节姿态、随机双脚间距和默认姿态。前四档直接对当前状态的 `torso_link` 施加大推力，不重置策略；后三档重置仿真。
- `R`：随时立即手动复位到默认站立姿态。即使机器人已经摔倒也可使用；复位会清零关节/根节点速度、当前外力和策略上一动作，然后继续策略推理。它是仿真状态复位，不表示策略自行完成起身。
- `K`：每次生成一个只改变双脚间距的随机初始姿态，实际脚距相对默认值偏差 `5–12 cm`，用于观察能否恢复默认距离。
- `F`：开启或关闭随机多部位外力。
- `C`：切换自由/跟随相机；两种模式都允许鼠标旋转和滚轮缩放。
- 洋红色箭头：项目实际写入 `xfrc_applied` 的外力，起点就是受力 body。
- 脚底附近的 MuJoCo 原生箭头：地面接触力，不是随机外力位置。

外力会在骨盆、躯干、左右肩、左右肘、左右髋和左右膝之间随机选择，不包含脚和踝。终端以及 `metrics.json` 会记录每次外力的 body、三轴力和三轴力矩。

空格键的大推力大小由 `LARGE_PUSH_FORCE_N` 任意设定，持续时间由 `LARGE_PUSH_DURATION_S` 设定。当前脚本未显式设置时默认为 `120 N × 0.20 s`。例如小力、中力和大力可分别写成：

```bash
# 小力：36 N，冲量 7.2 N·s
LARGE_PUSH_FORCE_N=36 LARGE_PUSH_DURATION_S=0.20

# 中力：120 N，冲量 24 N·s（脚本默认值）
LARGE_PUSH_FORCE_N=120 LARGE_PUSH_DURATION_S=0.20

# 大力：360 N，冲量 72 N·s
LARGE_PUSH_FORCE_N=360 LARGE_PUSH_DURATION_S=0.20
```

将其中一行放到交互测试命令的 `bash` 前即可。`F` 键控制的是另一套周期性随机外力；交互模式默认范围为三轴各自 `±35 N`、三轴力矩各自 `±5 N·m`，持续 `0.25 s`、每 `2.5 s` 采样一次。

### 4.2 交互震荡数据表

`PROFILE=interactive` 默认开启 `INTERACTIVE_DATA_LOG=True`，以策略控制频率 `50 Hz` 流式保存 CSV。每按一次空格都会结束上一段并新建一个独立 trial；`R/K/F/C` 不切分 trial，但会写入事件表，并在总表的 `last_operator_event` 列标记。

默认输出位于本次 `RESULTS_ROOT/interactive/seed_<SEED>/interactive_logs/<启动时间>/`，每次启动使用独立时间戳目录，不会覆盖上一次数据：

```text
interactive_logs/20260804_153000/
  interactive_diagnostics_all.csv       # 从启动到退出的完整总表
  interactive_events.csv                # SPACE/R/K/F/C 指令和时间索引
  space_trials/
    trial_001_large_torso_push_forward.csv
    trial_002_large_torso_push_backward.csv
    ...
```

总表与每个 trial 使用同一列结构，包含：

- trial 编号、场景、绝对时间、trial 相对时间、最近操作按键；
- 当前速度指令与目标速度指令；Extreme Stand 正常应始终为 `[0,0,0]`；
- root 世界系位置、四元数、线速度、角速度；
- 29 个策略关节轴锚点的世界系笛卡尔坐标，列名为 `joint_anchor_world_m/<joint_name>/{x,y,z}`；
- 左右脚 body 原点的世界系坐标、右脚减左脚的三轴差、XY 平面距离、三维距离及相对默认脚距的误差；
- 当前外力 body、来源、世界系三轴力与三轴力矩；
- 左右脚接触点数量、世界系三轴地面反力、关于各脚 body 原点的三轴合力矩；
- 29 关节的 actor 原始输出、未经限幅的目标角 `raw_target_qpos_rad/*`、实际送入 PD 的目标角 `target_qpos_rad/*`、目标限幅器速度/加速度、实际角度、速度、有限差分加速度、MuJoCo 加速度、jerk、PD 指令力矩、实际执行器力矩与力矩限位；另有每帧速度/加速度被限幅的关节数量。

注意：`qpos`/广义坐标是“根节点世界位姿 + 各关节角”，不是 29 个关节的世界系位置。后者是通过 MuJoCo 正向运动学得到的 `joint_anchor_world_m/...`。分析双脚间距离可直接使用 `feet/planar_distance_m`；其定义为 `sqrt((x_right-x_left)^2 + (y_right-y_left)^2)`，三维距离则使用 `feet/distance_3d_m`。

CSV 每个控制帧立即 flush，窗口异常关闭时已经记录的行仍会保留。若某次震荡出现在第 3 次空格之后，直接分析 `space_trials/trial_003_*.csv`，无需先从总表手动切片。若不需要记录，可显式设置 `INTERACTIVE_DATA_LOG=False`。

### 4.3 随机双脚间距固定测试

随机双脚间距的固定无窗口测试：

```bash
cd /home/user/Workspace/Humanoid/Locomotion/G1-Locomotion

PROFILE=feet_distance_recovery USE_GLFW=False REAL_TIME=False \
SIMULATION_DURATION=12 STEADY_START_S=5 SEED=20260727 \
bash scripts/sim2sim_g1_extreme_stand_recovery_mujoco.sh
```

报告中的 `foot_spacing_recovery.distance_recovered` 只有在初值确实偏差至少 `5 cm`，且最终 `1 s` 持续回到默认脚距 `±2 cm` 时才为 `true`。

### 4.4 启动时直接使用随机姿态并开启外力

```bash
cd /home/user/Workspace/Humanoid/Locomotion/G1-Locomotion

PROFILE=interactive USE_GLFW=True REAL_TIME=True \
INTERACTIVE_POSE_START_RANDOM=True \
INTERACTIVE_WRENCH_START_ENABLED=True \
SIMULATION_DURATION=300 SEED=20260722 \
bash scripts/sim2sim_g1_extreme_stand_recovery_mujoco.sh
```

### 4.5 固定 robust 场景可视化

固定场景不响应空格和 `F`，适合可重复对比：

```bash
cd /home/user/Workspace/Humanoid/Locomotion/G1-Locomotion

PROFILE=robust USE_GLFW=True REAL_TIME=True \
SIMULATION_DURATION=60 SEED=20260722 \
bash scripts/sim2sim_g1_extreme_stand_recovery_mujoco.sh
```

## 5. MuJoCo 无窗口测试

可用 `PROFILE`：

| PROFILE | 内容 |
| --- | --- |
| `nominal` | 默认姿态、无外力 |
| `pose_recovery` | 只随机29关节初始姿态 |
| `feet_distance_recovery` | 只随机双脚初始间距，检查是否恢复默认距离 |
| `recovery` | 随机关节、root 姿态和速度，无持续外力 |
| `robust` | 训练范围内初始扰动和随机外力 |
| `stress` | 超训练范围压力测试，只能用于仿真 |
| `large_push` | 默认姿态稳定5秒后，对 `torso_link` 施加一次固定水平大推力，并输出 actor/力矩/jerk 诊断 |

### 5.1 单个场景

```bash
cd /home/user/Workspace/Humanoid/Locomotion/G1-Locomotion

PROFILE=robust USE_GLFW=False REAL_TIME=False \
SIMULATION_DURATION=30 SEED=20260722 \
RESULTS_ROOT="$PWD/legged_lab/ExtremeStandRecovery Checkpoints/2026-07-31_16-52-27_g1_extreme_stand_recovery_smooth_torque_v4_from_v3_model2999_full_20260731/exported_extreme_stand_recovery/mujoco_tests/robust_manual" \
bash scripts/sim2sim_g1_extreme_stand_recovery_mujoco.sh
```

### 5.2 依次测试全部七个固定场景

```bash
cd /home/user/Workspace/Humanoid/Locomotion/G1-Locomotion

for PROFILE_NAME in nominal pose_recovery feet_distance_recovery recovery robust stress large_push; do
  PROFILE="$PROFILE_NAME" USE_GLFW=False REAL_TIME=False \
  SIMULATION_DURATION=30 SEED=20260722 \
  RESULTS_ROOT="$PWD/legged_lab/ExtremeStandRecovery Checkpoints/2026-07-31_16-52-27_g1_extreme_stand_recovery_smooth_torque_v4_from_v3_model2999_full_20260731/exported_extreme_stand_recovery/mujoco_tests/all_profiles_manual" \
  bash scripts/sim2sim_g1_extreme_stand_recovery_mujoco.sh || exit 1
done
```

### 5.3 七档、三种随机种子的完整测试

```bash
cd /home/user/Workspace/Humanoid/Locomotion/G1-Locomotion

SUITE=True USE_GLFW=False REAL_TIME=False \
SUITE_PROFILES=nominal,pose_recovery,feet_distance_recovery,recovery,robust,stress,large_push \
SUITE_DURATION=30 \
SUITE_SEEDS=20260719,20260720,20260721 \
SUITE_RESULTS_ROOT="$PWD/legged_lab/ExtremeStandRecovery Checkpoints/2026-07-31_16-52-27_g1_extreme_stand_recovery_smooth_torque_v4_from_v3_model2999_full_20260731/exported_extreme_stand_recovery/mujoco_tests/suite_manual" \
REQUIRE_PASS=False \
bash scripts/sim2sim_g1_extreme_stand_recovery_mujoco.sh
```

`REQUIRE_PASS=False` 表示即使某个严格恢复指标未通过，也会完成全部测试并生成报告。需要把验收失败变成非零退出码时，改成 `REQUIRE_PASS=True`。

结果目录会包含：

```text
metrics.json
torso_trace.csv
summary.json
REPORT.md
```

### 5.4 躯干大推力与持续 jerk 专项测试

可视化固定 `360 N × 0.20 s` 左侧推。模型先稳定5秒，洋红色箭头表示实际施加到 `torso_link` 的力：

```bash
cd /home/user/Workspace/Humanoid/Locomotion/G1-Locomotion

PROFILE=large_push USE_GLFW=True REAL_TIME=True \
SIMULATION_DURATION=30 STEADY_START_S=7.2 \
LARGE_PUSH_FORCE_N=360 LARGE_PUSH_DURATION_S=0.20 \
LARGE_PUSH_TIME_S=5 LARGE_PUSH_DIRECTION_INDEX=2 \
RESULTS_ROOT="$PWD/legged_lab/logs/monitoring/extreme_stand_large_push_visual" \
bash scripts/sim2sim_g1_extreme_stand_recovery_mujoco.sh
```

无窗口量化测试只需把 `USE_GLFW=False REAL_TIME=False`。每次运行生成：

```text
metrics.json
motion_quality_trace.csv
large_push_diagnostics.png
```

交互模式下也可设置 `LARGE_PUSH_FORCE_N=360`，然后反复按空格测试四个方向及三种重置姿态。方向编号：`0=前`、`1=后`、`2=左`、`3=右`、`-1=按seed随机`。

2026-07-31 实测 `360 N × 0.20 s` 侧推后未摔倒，但出现约3秒强烈恢复振铃。恢复段 actor action 高频放大 `123×`、action变化率放大 `684×`、PD力矩高频放大 `155×`；力矩饱和仅约 `0.10%`，因此主要是策略 action 驱动的高频闭环振铃，并非持续力矩饱和。最终5秒已恢复稳定，MuJoCo尚未复现真机无限持续抖动。

报告：

```text
/home/user/Workspace/Humanoid/Locomotion/G1-Locomotion/legged_lab/logs/monitoring/extreme_stand_large_push_diagnostic_20260731_force360_settled/REPORT.md
```

### 5.5 随机初始关节姿态专项测试

五个种子的无窗口测试：

```bash
cd /home/user/Workspace/Humanoid/Locomotion/G1-Locomotion

SEEDS=20260722,20260723,20260724,20260725,20260726 \
DURATION=15 USE_GLFW=False REQUIRE_PASS=False \
bash scripts/test_g1_extreme_stand_random_pose_recovery_mujoco.sh
```

只可视化一个随机姿态：

```bash
cd /home/user/Workspace/Humanoid/Locomotion/G1-Locomotion

SEEDS=20260722 DURATION=30 USE_GLFW=True \
bash scripts/test_g1_extreme_stand_random_pose_recovery_mujoco.sh
```

## 6. 真机运行

### 6.1 重要行为

专用脚本完成模型合同、依赖、网络、显式确认和 LowState 检查后，会立即从机器人当前状态开始 ONNX 推理，并以 `50 Hz` 发送29关节全身策略 PD 目标；不需要按空格键，也没有默认姿态预插值。

这不是传统的确定性起身动作。它只应从训练和仿真测试覆盖的、机械上可恢复的姿态启动。首次真机测试必须使用安全架、现场急停和短运行时间。

### 6.2 不连接机器人的 dry-run

下面命令可以直接复制执行，只校验模型、元数据和最终启动参数，不发送 LowCmd：

```bash
cd /home/user/Workspace/Humanoid/Locomotion/G1-Locomotion

DRY_RUN=True NET=lo PING_ROBOT=False \
ONNX_PATH="$PWD/use/extreme_stand_recovery_pose_v2_model2999.onnx" \
UNITREE_PYTHON=/home/user/anaconda3/envs/gmr/bin/python \
bash scripts/deploy_real_g1_extreme_stand_recovery_onnx.sh
```

### 6.3 检查真机网卡和 Python 环境

```bash
cd /home/user/Workspace/Humanoid/Locomotion/G1-Locomotion

ip -br link
```

真机 Python 必须在同一个环境里导入 `cyclonedds`、`unitree_sdk2py`、`onnxruntime`、`torch` 和 `yaml`。把下面第一行路径替换为真机实际环境后执行：

```bash
export UNITREE_PYTHON=/absolute/path/to/unitree-ready/bin/python

PYTHONPATH="$PWD/unitree_sim2sim2real:$PWD/unitree_sdk2_python${PYTHONPATH:+:$PYTHONPATH}" \
"$UNITREE_PYTHON" -c 'import cyclonedds, onnxruntime, torch, yaml; from unitree_sdk2py.core.channel import ChannelFactoryInitialize; print("真机 Python 依赖通过")'
```

当前开发机的 `gmr` 环境可以完成 dry-run，但缺少 `cyclonedds`，不能直接用于真实 DDS 控制。

### 6.4 真机首次短时运行

下面整段可以复制执行；运行后会提示输入实际网卡名和真机 Python 路径：

```bash
cd /home/user/Workspace/Humanoid/Locomotion/G1-Locomotion

read -r -p "请输入连接 G1 的网卡名，例如 eno1: " NET
read -r -p "请输入包含 cyclonedds、unitree_sdk2py、onnxruntime 的 Python 绝对路径: " UNITREE_PYTHON

CONFIRM_REAL_ROBOT=I_UNDERSTAND \
NET="$NET" ROBOT_IP=192.168.123.161 \
ONNX_PATH="$PWD/use/extreme_stand_recovery_pose_v2_model2999.onnx" \
UNITREE_PYTHON="$UNITREE_PYTHON" \
RUN_DURATION=10 PING_ROBOT=True \
bash scripts/deploy_real_g1_extreme_stand_recovery_onnx.sh
```

启动时序为：合同校验 → 确认门 → 依赖和网络检查 → 等待 LowState → 立即开始 ONNX 推理。`RUN_DURATION=10` 到期或按 `Ctrl+C` 退出后，共享真机 runner 会发送阻尼命令。

### 6.5 延长运行时间

只有在安全架上的10秒测试完全正常后，才逐步改为 `30` 或 `60` 秒。`RUN_DURATION=0` 表示持续运行，不建议首次测试使用。

```bash
cd /home/user/Workspace/Humanoid/Locomotion/G1-Locomotion

read -r -p "请输入连接 G1 的网卡名，例如 eno1: " NET
read -r -p "请输入真机 Python 绝对路径: " UNITREE_PYTHON

CONFIRM_REAL_ROBOT=I_UNDERSTAND \
NET="$NET" ROBOT_IP=192.168.123.161 \
ONNX_PATH="$PWD/use/extreme_stand_recovery_pose_v2_model2999.onnx" \
UNITREE_PYTHON="$UNITREE_PYTHON" \
RUN_DURATION=30 PING_ROBOT=True \
bash scripts/deploy_real_g1_extreme_stand_recovery_onnx.sh
```

## 7. 最低安全要求

- 真机必须使用安全架，现场人员必须能立即触发急停。
- 确认机器人当前姿态没有机械干涉、关节越限或倒地情况。
- 第一次只运行10秒，不要直接设置 `RUN_DURATION=0`。
- MuJoCo 的 `stress` 扰动不能照搬到真机。
- 真机脚本当前与 AMP 基线一致，没有额外的软件 `q_target` 关节角裁剪；安全还依赖相同的 Kp/Kd、动作尺度、固件保护和机械安全措施。

更详细的训练设计、奖励、测试结果和已知边界见：

```text
/home/user/Workspace/Humanoid/Locomotion/G1-Locomotion/legged_lab/docs/g1_extreme_stand_recovery.md
```
