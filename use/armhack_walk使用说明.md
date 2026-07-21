# ArmHack Walk 使用说明

> HEC-5090 使用 `/home/hecggdz/LocomotionHEC_use_20260721_1515` 和 `$HOME/miniconda3/envs/env_leglab/bin/python`。服务器一键测试与真机命令优先参见 `use/HEC5090三模型部署测试说明.md`；下文 `/home/user/...` 为原开发机示例。

本文命令以项目根目录 `/home/user/Workspace/Humanoid/Locomotion/G1-Locomotion` 为基准。真机运行必须使用吊架、现场急停和 Unitree 遥控器，并确保机器人周围无人。

## 文件位置

- 当前统一使用的 ONNX：`use/armhack_walk_model_10990.onnx`
- 对应训练 checkpoint：`checkpoint/walk/model_10990.pt`
- 真机入口：`scripts/deploy_real_g1_armhack_walk.sh`
- 真机 Python 控制器：`unitree_sim2sim2real/deploy/deploy_real/deploy_real_g1_armhack_walk.py`
- model_3999 MuJoCo 入口：`legged_lab/scripts/val_mujoco_g1_armhack_walk.sh`
- 最新 checkpoint 单场景 MuJoCo 入口：`legged_lab/scripts/val_mujoco_g1_armhack_walk_behavior.sh`
- 最新 checkpoint MuJoCo 测试矩阵：`legged_lab/scripts/test_mujoco_g1_armhack_walk_behavior.sh`
- 双臂姿态：`legged_lab/Reference Data/ArmHack/WalkPerturbFinetune/g1_arm_pose_set.json`
- 速度范围约束：`legged_lab/Reference Data/ArmHack/WalkPerturbFinetune/real_deployment_contract.json`

真机脚本默认直接读取 `/home/user/Workspace/Humanoid/Locomotion/G1-Locomotion/use/armhack_walk_model_10990.onnx`。脚本默认使用 `/home/user/anaconda3/envs/gmr/bin/python` 运行 MuJoCo/ONNX，导出模型时使用 `/home/user/anaconda3/envs/env_isaaclab/bin/python`。一般不需要手动激活环境；如果路径不同，用 `UNITREE_PYTHON=...` 或 `ISAAC_PYTHON=...` 覆盖。

注意：现有 MuJoCo sim2sim 运行器加载 TorchScript，不直接加载 ONNX，因此 MuJoCo 命令仍从同一个 `model_10990.pt` checkpoint 导出并使用 TorchScript；真机命令统一使用 `use/armhack_walk_model_10990.onnx`。

## MuJoCo 测试与可视化

先进入项目：

```bash
cd /home/user/Workspace/Humanoid/Locomotion/G1-Locomotion
```

对默认稳定模型 model_3999 做 20 秒无界面测试：

```bash
USE_GLFW=False REAL_TIME=False SIMULATION_DURATION=20 \
  bash legged_lab/scripts/val_mujoco_g1_armhack_walk.sh
```

可视化 model_3999；MuJoCo 窗口中按空格切换零速与固定速度：

```bash
USE_GLFW=True REAL_TIME=True SIMULATION_DURATION=60 \
  POSE_NAME=pos2_down FIXED_COMMAND='[0.35,0.0,0.0]' \
  bash legged_lab/scripts/val_mujoco_g1_armhack_walk.sh
```

对最新 `model_10990.pt` 做快速 smoke 测试。若导出文件不存在，脚本会先自动导出：

```bash
SUITE=smoke ENFORCE_THRESHOLDS=False \
  CHECKPOINT=/home/user/Workspace/Humanoid/Locomotion/G1-Locomotion/checkpoint/walk/model_10990.pt \
  bash legged_lab/scripts/test_mujoco_g1_armhack_walk_behavior.sh
```

针对零速、低速、小碎步、原地转向、侧移、斜向和前向步频运行核心测试：

```bash
SUITE=core ENFORCE_THRESHOLDS=False \
  CHECKPOINT=/home/user/Workspace/Humanoid/Locomotion/G1-Locomotion/checkpoint/walk/model_10990.pt \
  bash legged_lab/scripts/test_mujoco_g1_armhack_walk_behavior.sh
```

对三种双臂姿态运行完整测试矩阵：

```bash
SUITE=full ENFORCE_THRESHOLDS=False \
  CHECKPOINT=/home/user/Workspace/Humanoid/Locomotion/G1-Locomotion/checkpoint/walk/model_10990.pt \
  bash legged_lab/scripts/test_mujoco_g1_armhack_walk_behavior.sh
```

可视化最新模型的一个定时场景，例如“行走到零速”：

```bash
USE_GLFW=True REAL_TIME=True SCENARIO_NAME=walk_to_zero POSE_NAME=pos2_down \
  CHECKPOINT=/home/user/Workspace/Humanoid/Locomotion/G1-Locomotion/checkpoint/walk/model_10990.pt \
  bash legged_lab/scripts/val_mujoco_g1_armhack_walk_behavior.sh
```

可用场景包括 `zero_hold`、`walk_to_zero`、`micro_forward`、`micro_lateral`、`micro_diagonal`、`turn_in_place_left`、`turn_in_place_right`、`lateral_left`、`lateral_right`、`diagonal_front_left`、`diagonal_front_right`、`forward_cadence` 和 `smoke_walk_to_zero`。双臂姿态可选 `pos1_back`、`pos2_down`、`pos3_front`。

## 真机部署

先确认机器人网卡名；下列命令假设为 `enp11s0`：

```bash
ip -brief link
```

### 离线自检

自检不会初始化 DDS，也不会向机器人发送命令：

```bash
cd /home/user/Workspace/Humanoid/Locomotion/G1-Locomotion
MODEL=/home/user/Workspace/Humanoid/Locomotion/G1-Locomotion/use/armhack_walk_model_10990.onnx
DRY_RUN=True NET=enp11s0 COMMAND_MODE=fixed POLICY_PATH="$MODEL" \
  EXPECTED_POLICY_SHA256="$(sha256sum "$MODEL" | awk '{print $1}')" \
  bash scripts/deploy_real_g1_armhack_walk.sh
```

Joystick 配置也可离线检查：

```bash
cd /home/user/Workspace/Humanoid/Locomotion/G1-Locomotion
MODEL=/home/user/Workspace/Humanoid/Locomotion/G1-Locomotion/use/armhack_walk_model_10990.onnx
DRY_RUN=True NET=enp11s0 COMMAND_MODE=joystick JOYSTICK_DEVICE=/dev/input/js0 \
  POLICY_PATH="$MODEL" EXPECTED_POLICY_SHA256="$(sha256sum "$MODEL" | awk '{print $1}')" \
  bash scripts/deploy_real_g1_armhack_walk.sh
```

### 固定速度模式

```bash
cd /home/user/Workspace/Humanoid/Locomotion/G1-Locomotion
MODEL=/home/user/Workspace/Humanoid/Locomotion/G1-Locomotion/use/armhack_walk_model_10990.onnx
CONFIRM_REAL_ROBOT=I_UNDERSTAND NET=enp11s0 COMMAND_MODE=fixed \
  POLICY_PATH="$MODEL" EXPECTED_POLICY_SHA256="$(sha256sum "$MODEL" | awk '{print $1}')" \
  POSE_NAME=pos2_down FIXED_COMMAND='[0.35,0.0,0.0]' \
  bash scripts/deploy_real_g1_armhack_walk.sh
```

启动过程需要按两次回车：第一次允许双臂用 minimum-jerk 移到初始姿态，第二次启动零速度 policy。运行后按 `V` 在零速和 `FIXED_COMMAND` 间切换；按空格依次切换 `pos1_back → pos2_down → pos3_front`，每次切换默认用 2 秒 minimum-jerk 插值；按 `q`、`Ctrl-C` 或遥控器 `Select` 进入低层阻尼并退出。

### Joystick 模式

先确认 Joystick 设备存在且可读：

```bash
ls -l /dev/input/js*
```

默认映射为轴 `1/0/3` 控制前后、侧向和偏航，三个方向符号均为 `-1`，死区为 `0.05`。速度范围严格限制在训练 Nav2 CSV 包络内：前后 `[-0.2, 0.6] m/s`、侧向 `[-0.3, 0.3] m/s`、偏航 `[-0.5187280216217041, 0.6] rad/s`。

```bash
cd /home/user/Workspace/Humanoid/Locomotion/G1-Locomotion
MODEL=/home/user/Workspace/Humanoid/Locomotion/G1-Locomotion/use/armhack_walk_model_10990.onnx
CONFIRM_REAL_ROBOT=I_UNDERSTAND NET=enp11s0 COMMAND_MODE=joystick \
  POLICY_PATH="$MODEL" EXPECTED_POLICY_SHA256="$(sha256sum "$MODEL" | awk '{print $1}')" \
  JOYSTICK_DEVICE=/dev/input/js0 \
  JOYSTICK_AXIS_LIN_X=1 JOYSTICK_AXIS_LIN_Y=0 JOYSTICK_AXIS_YAW=3 \
  JOYSTICK_SIGN_LIN_X=-1 JOYSTICK_SIGN_LIN_Y=-1 JOYSTICK_SIGN_YAW=-1 \
  JOYSTICK_DEADZONE=0.05 \
  JOYSTICK_LIN_X_RANGE='[-0.2,0.6]' \
  JOYSTICK_LIN_Y_RANGE='[-0.3,0.3]' \
  JOYSTICK_YAW_RANGE='[-0.5187280216217041,0.6]' \
  POSE_NAME=pos2_down \
  bash scripts/deploy_real_g1_armhack_walk.sh
```

Joystick 模式默认不额外做速度指令斜坡，摇杆回中即给出零速度。按空格仍只负责切换双臂姿态；`V` 在此模式下无作用。若运动方向相反，只修改对应的 `JOYSTICK_SIGN_*` 为 `1` 或 `-1`，不要在机器人承重运行时试轴。

### use 目录中的 model_10990 ONNX

该模型已经保存在 `use/`，无需再次导出。固定速度运行：

```bash
cd /home/user/Workspace/Humanoid/Locomotion/G1-Locomotion
MODEL=/home/user/Workspace/Humanoid/Locomotion/G1-Locomotion/use/armhack_walk_model_10990.onnx
CONFIRM_REAL_ROBOT=I_UNDERSTAND NET=enp11s0 COMMAND_MODE=fixed \
  POLICY_PATH="$MODEL" EXPECTED_POLICY_SHA256="$(sha256sum "$MODEL" | awk '{print $1}')" \
  POSE_NAME=pos2_down FIXED_COMMAND='[0.35,0.0,0.0]' \
  bash scripts/deploy_real_g1_armhack_walk.sh
```

最新模型配合 Joystick：

```bash
cd /home/user/Workspace/Humanoid/Locomotion/G1-Locomotion
MODEL=/home/user/Workspace/Humanoid/Locomotion/G1-Locomotion/use/armhack_walk_model_10990.onnx
CONFIRM_REAL_ROBOT=I_UNDERSTAND NET=enp11s0 COMMAND_MODE=joystick \
  POLICY_PATH="$MODEL" EXPECTED_POLICY_SHA256="$(sha256sum "$MODEL" | awk '{print $1}')" \
  JOYSTICK_DEVICE=/dev/input/js0 POSE_NAME=pos2_down \
  bash scripts/deploy_real_g1_armhack_walk.sh
```

当前真机控制器已取消原先的 `0.05 rad` 关节限位内缩和 `4 rad/s` 逐帧目标变化限速。仍保留 ONNX 输出有限值检查、LowState 超时、躯干倾角、实测关节越界检查，以及按硬件原始关节范围进行的目标裁剪。
