# ArmHack Walk `model_3999` MuJoCo 与真机部署指南

## 1. 适用范围与模型身份

本文只适用于 S3 G1 29DoF 的 ArmHack Walk 起点模型 `model_3999.pt`。目标是在固定双臂姿态下追踪一个位于 Nav2 CSV 原始分布范围内的速度命令。本文模型是后续 P0/P1 训练的输入基线，不是九阶段训练完成后的最终模型。

部署包已保存在：

```text
legged_lab/deployment/armhack_walk/model_3999/
├── walk_model3999.onnx
├── walk_model3999.pt
├── walk_model3999.deploy.json
└── manifest.json
```

模型身份：

```text
source checkpoint iteration: 3999
source checkpoint SHA-256:
454c9bc0b5e38b2a9800c6faaa9e8ba6995f7d99bd3844155929a10a4fb8e2ff

ONNX SHA-256:
6d9b48cbbc0b35584f637f99f198a49a35e1eed16f303679cd75cb4fa03b0272

TorchScript SHA-256:
70e163e47e94fe5f3c96f890e95d1413db130df0458fc1cf5d2b0eb2aa6eb598

interface: obs float32 [1,96] -> actions float32 [1,29]
control frequency: 50 Hz
action scale: 0.25
```

ONNX 只包含 actor，不包含固定双臂覆盖、速度切换、PD 控制、关节限位和安全状态机。真机必须通过 `scripts/deploy_real_g1_armhack_walk.sh` 启动，不能直接把该 ONNX 交给通用 Nav2 launcher，否则双臂目标及 `last_action` 语义与训练不一致。

## 2. 固定双臂与 96 维输入契约

双臂姿态来自：

```text
legged_lab/Reference Data/ArmHack/WalkPerturbFinetune/g1_arm_pose_set.json
```

支持 `pos1_back`、`pos2_down`、`pos3_front`，默认使用 `pos2_down`。每次部署只选择一个姿态，策略运行过程中保持不变。14 个双臂目标按照左臂 7DoF、右臂 7DoF 写入 actor 输出中的下列位置：

```python
ARM_POLICY_INDICES = [
    11, 15, 19, 21, 23, 25, 27,
    12, 16, 20, 22, 24, 26, 28,
]
```

每个控制周期的正确执行顺序为：

```text
读取 IMU 与 29DoF q/dq
  -> obs[0:67] 写当前状态和当前速度命令
  -> obs[67:96] 写上一帧实际组合动作
  -> ONNX 输出 29 维网络动作
  -> 保留 15 维腰腿动作，覆盖 14 维双臂动作
  -> q_target = q_default + 0.25 * composed_action
  -> 安全限位和目标速度限制
  -> 发送 LowCmd
  -> 保存限位后实际执行的 composed_action 作为下一帧 last_action
```

policy 不观察未来速度或未来手臂目标。部署端也不能把额外信息拼接到 96 维输入。

## 3. Nav2 CSV 速度边界

部署契约保存在：

```text
legged_lab/Reference Data/ArmHack/WalkPerturbFinetune/real_deployment_contract.json
```

它记录了原始 Nav2 CSV 的 SHA-256、331,010 行数据的逐分量最小值/最大值和推荐固定速度。原始范围是：

```text
vx: [-0.2, 0.6] m/s
vy: [-0.3, 0.3] m/s
wz: [-0.5187280216, 0.6] rad/s
```

默认固定速度为 `[0.35, 0, 0]`。MuJoCo 和真机入口都会在启动前验证三个分量；任何分量超出上述范围都会拒绝运行。范围校验只是必要条件，不表示范围边缘的所有组合均已完成真机验收。第一次真机只应使用默认前进速度，并保持吊架承重。

真机 policy 启动时速度始终为 `[0,0,0]`。按一次空格后目标变为配置的固定速度，再按一次回到 `[0,0,0]`。切换使用默认 `0.5 m/s²` 线加速度和 `0.8 rad/s²` yaw 加速度限幅，不向 policy 施加速度阶跃。

## 4. MuJoCo 测试与可视化

### 4.1 使用的环境

首次从训练 checkpoint 重新导出时需要 IsaacLab 环境：

```bash
source /home/user/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
```

MuJoCo 运行使用当前已验证的 `gmr` 环境。默认脚本会显式调用：

```text
/home/user/anaconda3/envs/gmr/bin/python
```

部署目录中的 ONNX/TorchScript 已存在时，不需要训练 checkpoint 也不会重新导出；因此完整 clone 后仍可运行 MuJoCo 和真机入口。

### 4.2 20 秒 headless 测试

```bash
cd /home/user/Workspace/Humanoid/Locomotion/G1-Locomotion/legged_lab

USE_GLFW=False \
REAL_TIME=False \
SIMULATION_DURATION=20 \
POSE_NAME=pos2_down \
FIXED_COMMAND='[0.35,0.0,0.0]' \
START_ACTIVE=True \
bash scripts/val_mujoco_g1_armhack_walk.sh
```

脚本依次执行 checkpoint/部署文件检查、ONNX 与 TorchScript 五组输入数值一致性检查、姿态/速度契约检查和 S3 G1 MuJoCo rollout。默认指标写入本地忽略目录：

```text
legged_lab/deployment/armhack_walk/model_3999/Local Test Reports/
walk_model3999_pos2_down_metrics.json
```

2026-07-18 对起点模型的实际 20 秒结果为：

```text
pose: pos2_down
command: [0.35, 0, 0]
healthy: true
fallen: false
minimum root height: 0.775 m
maximum |roll|: 0.040 rad
maximum |pitch|: 0.097 rad
linear velocity tracking MAE: 0.0501 m/s
yaw-rate MAE: 0.0360 rad/s
torso roll/pitch error: 0.0118 / 0.0275 rad
```

这证明 sim2sim 执行链可以完整运行，不代表已经通过吊架真机验收。

### 4.3 GUI 可视化

```bash
cd /home/user/Workspace/Humanoid/Locomotion/G1-Locomotion/legged_lab

USE_GLFW=True \
REAL_TIME=True \
SIMULATION_DURATION=120 \
POSE_NAME=pos2_down \
FIXED_COMMAND='[0.35,0.0,0.0]' \
START_ACTIVE=False \
bash scripts/val_mujoco_g1_armhack_walk.sh
```

窗口启动时为零速度；焦点在 MuJoCo 窗口时按 `SPACE` 在固定速度和零速度间切换。可分别把 `POSE_NAME` 改为 `pos1_back` 或 `pos3_front`，但一次运行内姿态保持固定。

## 5. 新电脑从 clone 到离线自检

### 5.1 克隆并检查部署文件

```bash
cd "$HOME"
git clone https://github.com/gracetata/LocomotionHEC.git
cd LocomotionHEC
git checkout main
git pull --ff-only origin main
export REPO_ROOT="$PWD"

test -s legged_lab/deployment/armhack_walk/model_3999/walk_model3999.onnx
test -s legged_lab/deployment/armhack_walk/model_3999/manifest.json
test -s "legged_lab/Reference Data/ArmHack/WalkPerturbFinetune/g1_arm_pose_set.json"
test -s "legged_lab/Reference Data/ArmHack/WalkPerturbFinetune/real_deployment_contract.json"
test -s scripts/deploy_real_g1_armhack_walk.sh
test -s unitree_sim2sim2real/deploy/deploy_real/deploy_real_g1_armhack_walk.py
```

检查模型：

```bash
sha256sum legged_lab/deployment/armhack_walk/model_3999/walk_model3999.onnx
```

必须得到：

```text
6d9b48cbbc0b35584f637f99f198a49a35e1eed16f303679cd75cb4fa03b0272
```

### 5.2 安装真机 Python 环境

真机只需要 CPU，不需要安装 Isaac Sim、IsaacLab 或 CUDA。Ubuntu 22.04 可使用：

```bash
sudo apt update
sudo apt install -y build-essential cmake git iproute2 net-tools pkg-config wget

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda create -n armhack-real python=3.10 -y
conda activate armhack-real
python -m pip install --upgrade pip setuptools wheel
python -m pip install numpy==1.26.4 PyYAML==6.0.2 onnx==1.16.1 onnxruntime==1.18.1 opencv-python==4.10.0.84
python -m pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu
```

按 Unitree SDK 要求安装 CycloneDDS 0.10.2，并安装仓库内 SDK：

```bash
mkdir -p "$HOME/src"
git clone --depth 1 --branch releases/0.10.x https://github.com/eclipse-cyclonedds/cyclonedds.git "$HOME/src/cyclonedds"
cmake -S "$HOME/src/cyclonedds" -B "$HOME/src/cyclonedds/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$HOME/.local/cyclonedds-0.10.2" \
  -DBUILD_EXAMPLES=OFF
cmake --build "$HOME/src/cyclonedds/build" --parallel "$(nproc)"
cmake --install "$HOME/src/cyclonedds/build"

export CYCLONEDDS_HOME="$HOME/.local/cyclonedds-0.10.2"
export LD_LIBRARY_PATH="$CYCLONEDDS_HOME/lib:$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python -m pip install --no-cache-dir cyclonedds==0.10.2
python -m pip install -e "$REPO_ROOT/unitree_sdk2_python"
```

不要在 source 过 ROS 2 setup 的终端中运行，以免 DDS 动态库冲突。

### 5.3 离线自检

```bash
cd "$REPO_ROOT"
export UNITREE_PYTHON="$CONDA_PREFIX/bin/python"

UNITREE_PYTHON="$UNITREE_PYTHON" \
DRY_RUN=True NET=enp3s0 \
bash scripts/deploy_real_g1_armhack_walk.sh
```

必须看到 `[SELF-TEST PASS]`。该模式会检查 ONNX 96→29 接口、17 组输入有限输出、固定臂组合动作、速度范围和三个发布文件 SHA；不会初始化 DDS 或发送 LowCmd。

## 6. 吊架真机运行

### 6.1 网络

将 `NET` 替换为连接 G1 的实际有线网卡：

```bash
export NET=enp3s0
sudo ip link set dev "$NET" up
sudo ip addr replace 192.168.123.222/24 dev "$NET"
ping -c 3 192.168.123.161
```

若现场机器人使用其他 IP/网段，应使用现场配置。ping 不通时不要启动控制。

### 6.2 第一次 30 秒测试

机器人必须处于吊架保护，现场人员持有急停/遥控器，周围清空：

```bash
cd "$REPO_ROOT"

UNITREE_PYTHON="$CONDA_PREFIX/bin/python" \
CONFIRM_REAL_ROBOT=I_UNDERSTAND \
NET="$NET" \
POSE_NAME=pos2_down \
FIXED_COMMAND='[0.35,0.0,0.0]' \
RUN_DURATION=30 \
bash scripts/deploy_real_g1_armhack_walk.sh
```

人工顺序固定为：

1. 程序连接 LowState，保持当前 29DoF 姿态并调用 `ReleaseMode()`；
2. 出现 `[DEBUG MODE]` 后检查吊架、关节方向和急停，第一次按 `ENTER`；
3. 程序用 5 秒 minimum-jerk 只移动双臂到 `pos2_down`，腰腿保持当前目标；
4. 再次检查双脚着地和周围环境，第二次按 `ENTER`；
5. policy 以 `[0,0,0]` 速度启动；确认稳定后按一次 `SPACE`，平滑进入 `[0.35,0,0]`；
6. 再按一次 `SPACE` 平滑回零；`q`、`Ctrl-C` 或遥控器 `Select` 停止并进入低层阻尼。

退出后机器人可能失去主动站立，吊架必须持续承重。30 秒吊架测试不能证明无人值守安全；在完成逐姿态、逐速度和急停验收前不得去掉吊架。

## 7. 可配置项与禁止事项

常用参数：

```text
POSE_NAME=pos1_back|pos2_down|pos3_front
FIXED_COMMAND='[vx,vy,wz]'
STARTUP_MOVE_S=5.0
COMMAND_MAX_LINEAR_ACCEL=0.5
COMMAND_MAX_YAW_ACCEL=0.8
RUN_DURATION=30
```

禁止事项：

- 不要绕过 ONNX、姿态 JSON 或部署契约 SHA 检查；
- 不要把速度设置到部署契约范围外；
- 不要直接调用通用 `deploy_real_g1_amp_onnx.sh` 运行本模型；
- 不要把原始网络的 14 维手臂输出当成实际手臂目标；
- 不要把网络原始 action 写入 `last_action`，必须写安全过滤后的组合动作；
- 不要在无吊架、无急停或非交互终端中运行；
- 不要把本次 MuJoCo 通过描述成真机认证。
