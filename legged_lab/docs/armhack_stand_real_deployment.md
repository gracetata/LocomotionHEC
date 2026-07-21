# ArmHack Stand 最新鲁棒模型真机部署与交互式 MuJoCo 指南

> 2026-07-21 当前权威状态：默认模型已经从历史 `legged_lab/deployment/armhack_stand/stand.onnx` 切换到 `checkpoint/stand/stand_robust_model_2999.onnx`。当前状态机只使用一次 `ENTER`：启动阶段保持原生阻尼/待机且不初始化 DDS；`ENTER` 后进入低层调试控制并立即启用 actor，自动执行 `自然下垂→平直默认→向前伸直→收回平直默认`；初始化完成后才允许 `SPACE` 切换双臂姿态。下文若引用旧的两次 Enter/5 s 移动到 P0，均以本节和第 8.3 节的新顺序为准。

当前模型身份：

```text
checkpoint/stand/model_2999.pt
SHA-256 146aca1f547ce073756c942508e8ea43c8cea91b27eee3b8347dd4131c87bc5f

checkpoint/stand/stand_robust_model_2999.onnx
SHA-256 354bf4b35572cf6d91d44d448cde36b7bb748cffe40f1e220183bf21e5553fbf
```

> 2026-07-21 已新增 `LeggedLab-Isaac-AMP-G1-StandDownToDefault-v0`，专门从上述鲁棒 checkpoint 续训“自然下垂静止站立，并以不同速度同时抬到平直 P0”。该入口已通过 `8 env × 1 iteration` Isaac smoke，但正式续训尚未完成；smoke 的 `model_0.pt` 不能部署。因此本指南当前仍锁定上面的第三阶段正式 ONNX，待第四阶段完整训练、Isaac/MuJoCo 固定测试和重新导出全部通过后，才能更新本节模型 SHA。

## 快速开始：新电脑从 clone 到真机

本节给出一台新的 Ubuntu 电脑从零开始的完整命令。真机推理只使用 CPU，不需要安装 Isaac Sim、IsaacLab、CUDA 或训练环境。以下流程按 Ubuntu 22.04、x86-64、Conda 和有线网卡编写；G1 必须是与本项目一致的 S3 G1 29DoF HG 接口。

### A.1 克隆项目并确认部署文件完整

HTTPS 克隆：

```bash
cd "$HOME"
git clone https://github.com/gracetata/LocomotionHEC.git
cd LocomotionHEC
git checkout main
git pull --ff-only origin main
export REPO_ROOT="$PWD"
```

如果该仓库需要 SSH 权限，则把第一条 `git clone` 换成：

```bash
git clone git@github.com:gracetata/LocomotionHEC.git
```

确认当前工作副本包含本次部署所需的七个文件：

```bash
cd "$REPO_ROOT"

test -s checkpoint/stand/stand_robust_model_2999.onnx
test -s checkpoint/stand/stand_robust_model_2999.deploy.json
test -s "legged_lab/Reference Data/ArmHack/StandPerturb/RealDeployment/stand_arm_presets.json"
test -s "legged_lab/Reference Data/ArmHack/StandPerturb/TestData/ArmOnly/special/arms_down_flat_forward_return_flat_25p5s_50hz.csv"
test -s scripts/deploy_real_g1_armhack_stand.sh
test -s unitree_sim2sim2real/deploy/deploy_real/deploy_real_g1_armhack_stand.py
test -s legged_lab/scripts/val_mujoco_g1_armhack_stand.sh

chmod +x scripts/deploy_real_g1_armhack_stand.sh
chmod +x unitree_sim2sim2real/deploy/deploy_real/deploy_real_g1_armhack_stand.py
```

检查当前发布件的校验和：

```bash
cd "$REPO_ROOT"

sha256sum checkpoint/stand/stand_robust_model_2999.onnx
sha256sum checkpoint/stand/stand_robust_model_2999.deploy.json
sha256sum "legged_lab/Reference Data/ArmHack/StandPerturb/RealDeployment/stand_arm_presets.json"
sha256sum "legged_lab/Reference Data/ArmHack/StandPerturb/TestData/ArmOnly/special/arms_down_flat_forward_return_flat_25p5s_50hz.csv"
```

当前预期值为：

```text
354bf4b35572cf6d91d44d448cde36b7bb748cffe40f1e220183bf21e5553fbf  stand_robust_model_2999.onnx
49145392cdc1ece9bd05bebe45f79a405c871ea8d922e2181a13675228bf1ed8  stand_robust_model_2999.deploy.json
f7a79eda6b177ddc34f7483885c1bf83c42f30126616c418b6af88c5ff0f7414  stand_arm_presets.json
a02a0c08f181286995b1f72e4c1355739eae6c34aa06df0872e38131220895e4  arms_down_flat_forward_return_flat_25p5s_50hz.csv
```

任何一个 `test -s` 失败，都说明 GitHub 上的分支还没有包含完整部署包，不能继续真机测试。ONNX 哈希不一致时 launcher 也会拒绝启动，不要绕过该检查。

### A.1.1 本机最新第三阶段模型的离线自检

上面的 `legged_lab/deployment/armhack_stand/stand.onnx` 是仓库内的固定发布件。2026-07-20 本机还保存了 HEC-5090 第三阶段鲁棒训练的更新模型：

```text
checkpoint/stand/model_2999.pt
SHA-256: 146aca1f547ce073756c942508e8ea43c8cea91b27eee3b8347dd4131c87bc5f

checkpoint/stand/stand_robust_model_2999.onnx
SHA-256: 354bf4b35572cf6d91d44d448cde36b7bb748cffe40f1e220183bf21e5553fbf
```

2026-07-21 复核 HEC-5090 时，远端真机 Shell、真机 Python、预设 JSON 和历史 `stand.onnx` 与当时本机文件一致；远端 MuJoCo Stand 仍是固定 CSV 回放，并没有 Enter/SPACE。此后本机按新的交互需求继续更新，因此不能再用旧的远端 Shell 哈希覆盖本机。当前 launcher 已默认锁定更新 ONNX，无需手工覆盖路径：

```bash
cd /home/user/Workspace/Humanoid/Locomotion/G1-Locomotion

DRY_RUN=True NET=enp11s0 \
UNITREE_PYTHON=/home/user/anaconda3/envs/gmr/bin/python \
bash scripts/deploy_real_g1_armhack_stand.sh
```

本次命令已得到 `[SELF-TEST PASS]`，且明确打印“未初始化 DDS，也未向机器人发送任何命令”。这只是离线接口验证；`checkpoint/stand/` 可能不属于 Git 固定发布包，新电脑必须先确认文件已安全复制且 SHA 一致，不能把本机路径直接假设为 clone 后必然存在。

### A.1.2 MuJoCo 使用相同 Enter/Space 状态机

新终端运行 GUI：

```bash
source /home/user/anaconda3/etc/profile.d/conda.sh
conda activate gmr
cd /home/user/Workspace/Humanoid/Locomotion/G1-Locomotion/legged_lab

MODE=interactive bash scripts/val_mujoco_g1_armhack_stand.sh
```

MuJoCo 启动时不运行 actor，机器人保持自然下垂待机；`ENTER` 启动 actor 和 `AD→P0→F→P0` 自动初始化；终端出现 `INIT COMPLETE` 后 `SPACE` 才切换 P1/P2/P3/P0；`Q` 结束。无头回归命令为：

```bash
MODE=interactive USE_GLFW=False REAL_TIME=False \
INTERACTIVE_AUTO_ENTER_S=0.5 \
INTERACTIVE_AUTO_SPACE_INTERVAL_S=8.0 \
INTERACTIVE_AUTO_SPACE_MAX_SWITCHES=2 \
SIMULATION_DURATION=50 \
bash scripts/val_mujoco_g1_armhack_stand.sh
```

2026-07-21 当前最新模型实跑结果：初始化完成、两次 SPACE 均执行、CSV 完整播放、`healthy=True`、无摔倒；最低 root 高度 `0.775567 m`，最大绝对 roll/pitch `0.115899 / 0.086728 rad`。报告与阶段曲线位于 `checkpoint/stand/Test Reports/StandArmOnlyMuJoCo/model_2999_146aca1f547c__mujoco__interactive__payload_0kg.*`。

### A.2 安装系统工具与 Miniconda

安装编译 CycloneDDS 所需的系统包：

```bash
sudo apt update
sudo apt install -y build-essential cmake git iproute2 net-tools pkg-config wget
```

如果电脑还没有 Conda，安装 Miniconda：

```bash
mkdir -p "$HOME/miniconda3"
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O "$HOME/miniconda3/miniconda.sh"
bash "$HOME/miniconda3/miniconda.sh" -b -u -p "$HOME/miniconda3"
rm "$HOME/miniconda3/miniconda.sh"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda init bash
```

如果已经安装 Conda，只需执行：

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
```

### A.3 创建独立真机环境

创建只负责 Stand 真机推理的环境：

```bash
conda create -n armhack-real python=3.10 -y
conda activate armhack-real

python -m pip install --upgrade pip setuptools wheel
python -m pip install numpy==1.26.4 PyYAML==6.0.2 onnx==1.16.1 onnxruntime==1.18.1 opencv-python==4.10.0.84
python -m pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu
```

这里安装 CPU 版 PyTorch 是因为通用部署模块在加载时会导入 `torch`；实际 Stand actor 使用 ONNX Runtime CPU 推理。

### A.4 编译 CycloneDDS 0.10.x 并安装仓库内 Unitree SDK

不要在已经 `source /opt/ros/.../setup.bash` 的终端中安装或运行本部署入口，避免 ROS 2 的 DDS/动态库污染当前进程。建议新开一个干净终端，再执行：

```bash
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate armhack-real

mkdir -p "$HOME/src"
cd "$HOME/src"
git clone --depth 1 --branch releases/0.10.x https://github.com/eclipse-cyclonedds/cyclonedds.git

cmake -S "$HOME/src/cyclonedds" \
  -B "$HOME/src/cyclonedds/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$HOME/.local/cyclonedds-0.10.2" \
  -DBUILD_EXAMPLES=OFF
cmake --build "$HOME/src/cyclonedds/build" --parallel "$(nproc)"
cmake --install "$HOME/src/cyclonedds/build"

export CYCLONEDDS_HOME="$HOME/.local/cyclonedds-0.10.2"
export LD_LIBRARY_PATH="$CYCLONEDDS_HOME/lib:$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

python -m pip install --no-cache-dir cyclonedds==0.10.2

cd "$HOME/LocomotionHEC"
export REPO_ROOT="$PWD"
python -m pip install -e "$REPO_ROOT/unitree_sdk2_python"
```

`unitree_sdk2_python` 已经随本项目保存，不需要再次从 Unitree GitHub 克隆另一份。其[官方安装说明](https://github.com/unitreerobotics/unitree_sdk2_python)要求 Python ≥ 3.8、CycloneDDS 0.10.2、NumPy 和 OpenCV；若安装时出现 `Could not locate cyclonedds`，说明当前终端没有正确设置 `CYCLONEDDS_HOME`。

### A.5 验证环境，不连接机器人

```bash
cd "$HOME/LocomotionHEC"
export REPO_ROOT="$PWD"
export UNITREE_PYTHON="$CONDA_PREFIX/bin/python"

python - <<'PY'
import cyclonedds
import numpy
import onnx
import onnxruntime
import torch
import yaml
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.utils.crc import CRC

print("Unitree real environment imports: PASS")
print("python/torch/numpy/onnxruntime:", torch.__version__, numpy.__version__, onnxruntime.__version__)
PY
```

随后运行项目自己的离线自检。`NET` 在这一条命令中只是占位，不会初始化 DDS：

```bash
cd "$REPO_ROOT"

UNITREE_PYTHON="$UNITREE_PYTHON" \
DRY_RUN=True NET=enp3s0 \
bash scripts/deploy_real_g1_armhack_stand.sh
```

必须看到：

```text
[SELF-TEST PASS] 未初始化 Unitree DDS，未发送机器人命令。
actor : obs[1,96] -> actions[1,29]
poses : 4 arm-only presets
DRY_RUN 完成：未初始化 DDS，也未向机器人发送任何命令。
```

### A.6 每次新开真机终端都要执行

安装只做一次；以后每个新终端从下面这组命令开始。假设仓库 clone 在 `$HOME/LocomotionHEC`：

```bash
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate armhack-real

cd "$HOME/LocomotionHEC"
export REPO_ROOT="$PWD"
export UNITREE_PYTHON="$CONDA_PREFIX/bin/python"
export CYCLONEDDS_HOME="$HOME/.local/cyclonedds-0.10.2"
export LD_LIBRARY_PATH="$CYCLONEDDS_HOME/lib:$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

如果仓库或 Miniconda 安装在其他位置，只修改上面两条 `cd/source` 路径；后续脚本会从 `REPO_ROOT` 自动定位模型、配置、SDK 和双臂数据。

### A.7 配置 G1 有线网络

用网线直连 G1，先找出有线网卡名称：

```bash
ip -br link
```

以下假设网卡为 `enp3s0`。必须替换为新电脑的实际名称：

```bash
export NET=enp3s0
sudo ip link set dev "$NET" up
```

如果 G1 保持 Unitree 常用默认网段 `192.168.123.0/24`，可给电脑端设置 `192.168.123.222/24`：

```bash
sudo ip addr replace 192.168.123.222/24 dev "$NET"
ip -br addr show dev "$NET"
ping -c 3 192.168.123.161
```

若机器人实际 IP/网段已被现场修改，应使用现场配置，不能强行照抄上述地址。关闭该有线网卡上的 DHCP、VPN 和会抢占 `192.168.123.0/24` 的其他路由。`ping` 不通时不要启动控制；先解决网线、网卡、IP 和防火墙问题。

### A.8 第一次吊架真机测试命令

机器人必须有吊架承重，现场人员必须握住急停/遥控器，周围清空。第一次测试限制为 30 秒：

```bash
cd "$REPO_ROOT"

UNITREE_PYTHON="$UNITREE_PYTHON" \
CONFIRM_REAL_ROBOT=I_UNDERSTAND \
NET="$NET" RUN_DURATION=30 \
bash scripts/deploy_real_g1_armhack_stand.sh
```

终端中的人工顺序：

1. 启动后终端显示 `[DAMPING / STANDBY]`；此时程序尚未初始化 DDS、没有发送 LowCmd，机器人应由原生阻尼/待机保持直立且双臂自然下垂；
2. 检查吊架、双臂下垂姿态、关节方向和现场急停，只按一次 `Enter`；程序随后建立当前位置保持、检查下垂误差、调用 `ReleaseMode()`，进入低层调试控制并启动 50 Hz actor；
3. actor 持续推理腰腿，双臂自动用 25.5 s 执行 `自然下垂→平直默认→向前伸直→收回平直默认`；看到 `[INIT COMPLETE]` 前不要按空格，提前按会打印 `SPACE LOCKED` 并忽略；
4. 初始化完成后按一次空格进入 P1；再次按空格依次进入 P2、P3、P0，每段 7.5 s minimum-jerk；
5. 按 `q`、`Ctrl-C` 或遥控器 `Select` 停止。低层接管已经成功时，退出会发送低层阻尼，吊架必须持续承重。

30 秒吊架短测没有异常后，才可去掉 `RUN_DURATION=30` 做持续测试：

```bash
cd "$REPO_ROOT"

UNITREE_PYTHON="$UNITREE_PYTHON" \
CONFIRM_REAL_ROBOT=I_UNDERSTAND \
NET="$NET" \
bash scripts/deploy_real_g1_armhack_stand.sh
```

本节命令能保证软件安装、模型定位、依赖预检和启动顺序可复现；它不能替代真机吊架验收，也不代表该策略已经获得无人值守运行许可。

## 1. 适用范围与当前结论

本指南只适用于 S3 G1 29DoF 的 ArmHack Stand 策略，不适用于 Walk/Nav2。当前默认部署模型来自 HEC-5090 第三阶段鲁棒训练：

```text
checkpoint:
checkpoint/stand/model_2999.pt

checkpoint SHA-256:
146aca1f547ce073756c942508e8ea43c8cea91b27eee3b8347dd4131c87bc5f

ONNX:
checkpoint/stand/stand_robust_model_2999.onnx

ONNX SHA-256:
354bf4b35572cf6d91d44d448cde36b7bb748cffe40f1e220183bf21e5553fbf

deployment metadata:
checkpoint/stand/stand_robust_model_2999.deploy.json
```

`stand_robust_model_2999.onnx` 已通过 ONNX checker、ONNX Runtime 和 MuJoCo 交互状态机实跑。它仍然不是“已经通过真机安全认证”的控制器。第一次上机必须使用吊架、急停、限幅和状态超时保护，并先验证关节顺序。

策略契约是：

```text
控制频率：50 Hz，control_dt = 0.02 s
输入：     float32 [1, 96]，名字 obs
输出：     float32 [1, 29]，名字 actions
速度命令：始终 [0, 0, 0]
动作缩放：q_target = q_default + 0.25 * action
```

ONNX 只包含前馈 actor，不包含 IsaacLab 环境、双臂轨迹生成器、动作覆盖、PD 控制器、执行器限幅或安全状态机。Stand 专用入口 `../scripts/deploy_real_g1_armhack_stand.sh` 与 `../unitree_sim2sim2real/deploy/deploy_real/deploy_real_g1_armhack_stand.py` 已实现当前预设轨迹所需的组合动作和基础安全状态机；它仍须经过吊架分级验收，不能仅凭离线自检视为真机安全认证。

## 2. 新终端中检查模型

在真机电脑使用上面创建的 `armhack-real` 环境。普通 shell 可能没有加载 Conda，因此新终端中先完整执行：

```bash
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate armhack-real
cd "$HOME/LocomotionHEC"
export REPO_ROOT="$PWD"
cd legged_lab

sha256sum ../checkpoint/stand/stand_robust_model_2999.onnx
sha256sum ../checkpoint/stand/stand_robust_model_2999.deploy.json
```

检查 ONNX 接口：

```bash
python - <<'PY'
import onnx
import onnxruntime as ort

path = "../checkpoint/stand/stand_robust_model_2999.onnx"
model = onnx.load(path)
onnx.checker.check_model(model)
session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
print("opset:", [(item.domain or "ai.onnx", item.version) for item in model.opset_import])
print("inputs:", [(item.name, item.shape, item.type) for item in session.get_inputs()])
print("outputs:", [(item.name, item.shape, item.type) for item in session.get_outputs()])
PY
```

预期输出为 opset 11、`obs [1, 96] tensor(float)` 和 `actions [1, 29] tensor(float)`。模型 batch 维固定为 1；不能直接传入 `[96]`，也不能一次传多台机器的 batch。

真机电脑只需要 ONNX、元数据、预设 JSON 和初始化 CSV，不需要训练 checkpoint。下面的重新导出命令只应在仍保存当前 `checkpoint/stand/model_2999.pt` 和训练环境的电脑上执行，并且必须显式写出 Stand 的零速度命令：

```bash
python scripts/rsl_rl/export_amp_actor_to_onnx.py \
  --robot g1 \
  --checkpoint ../checkpoint/stand/model_2999.pt \
  --output ../checkpoint/stand/stand_robust_model_2999.onnx \
  --metadata ../checkpoint/stand/stand_robust_model_2999.deploy.json \
  --default-command 0 0 0
```

## 3. `stand_robust_model_2999.onnx` 的 96 维输入

输入必须是连续的 `numpy.float32` 数组，形状为 `[1, 96]`。部署时不加入训练噪声，也没有额外 observation normalizer。各段顺序如下：

| Python 切片 | 维数 | 内容 | 真机计算方法 |
|---|---:|---|---|
| `obs[0:3]` | 3 | pelvis 角速度 | IMU gyroscope，pelvis/body 坐标系，rad/s |
| `obs[3:6]` | 3 | 重力在 pelvis 坐标系的投影 | 由 pelvis 四元数计算，站直约为 `[0, 0, -1]` |
| `obs[6:9]` | 3 | `vx, vy, yaw_rate` 命令 | Stand 强制写 `[0, 0, 0]`，不能接遥控器或 Nav2 |
| `obs[9:38]` | 29 | 关节相对位置 | `q_policy - q_default`，rad |
| `obs[38:67]` | 29 | 关节速度 | `dq_policy`，rad/s |
| `obs[67:96]` | 29 | 上一控制周期实际执行的 raw action | 必须是“网络腿腰输出 + 外部双臂覆盖”后的 `a_exec_prev` |

当前所有 observation scale 都是 1.0。四元数约定为 `wxyz`。现有 Unitree real 配置使用 `imu_type: pelvis`；如果固件返回的是 torso IMU、`xyzw` 四元数或世界系角速度，必须先转换，不能直接送入 ONNX。

对 `q = [qw, qx, qy, qz]`，当前代码使用：

```python
projected_gravity = np.array([
    2.0 * (-qz * qx + qw * qy),
   -2.0 * ( qz * qy + qw * qx),
    1.0 - 2.0 * (qw * qw + qz * qz),
], dtype=np.float32)
```

policy 没有未来双臂姿态、未来轨迹 phase、末端负载质量或世界坐标位置输入。它只通过当前 `q/dq`、IMU 和上一帧组合动作观察双臂运动造成的实际扰动。因此部署程序不能把未来轨迹点追加到 96 维输入中。

## 4. 29 维 policy 顺序、默认姿态和 PD 参数

ONNX 输入中的 `q/dq/last_action` 和输出 `actions` 都使用下表的 policy 顺序。`motor index` 是 Unitree 29DoF low-state/low-command 顺序。双臂 14 维由外部抓取控制覆盖，腰腿 15 维保留 ONNX 输出。

| policy index | motor index | 关节 | `q_default` (rad) | Kp | Kd | 最终来源 |
|---:|---:|---|---:|---:|---:|---|
| 0 | 0 | `left_hip_pitch_joint` | -0.10 | 100 | 2 | ONNX |
| 1 | 6 | `right_hip_pitch_joint` | -0.10 | 100 | 2 | ONNX |
| 2 | 12 | `waist_yaw_joint` | 0.00 | 200 | 5 | ONNX |
| 3 | 1 | `left_hip_roll_joint` | 0.00 | 100 | 2 | ONNX |
| 4 | 7 | `right_hip_roll_joint` | 0.00 | 100 | 2 | ONNX |
| 5 | 13 | `waist_roll_joint` | 0.00 | 40 | 5 | ONNX |
| 6 | 2 | `left_hip_yaw_joint` | 0.00 | 100 | 2 | ONNX |
| 7 | 8 | `right_hip_yaw_joint` | 0.00 | 100 | 2 | ONNX |
| 8 | 14 | `waist_pitch_joint` | 0.00 | 40 | 5 | ONNX |
| 9 | 3 | `left_knee_joint` | 0.30 | 150 | 4 | ONNX |
| 10 | 9 | `right_knee_joint` | 0.30 | 150 | 4 | ONNX |
| 11 | 15 | `left_shoulder_pitch_joint` | 0.30 | 40 | 1 | 外部双臂目标 |
| 12 | 22 | `right_shoulder_pitch_joint` | 0.30 | 40 | 1 | 外部双臂目标 |
| 13 | 4 | `left_ankle_pitch_joint` | -0.20 | 40 | 2 | ONNX |
| 14 | 10 | `right_ankle_pitch_joint` | -0.20 | 40 | 2 | ONNX |
| 15 | 16 | `left_shoulder_roll_joint` | 0.25 | 40 | 1 | 外部双臂目标 |
| 16 | 23 | `right_shoulder_roll_joint` | -0.25 | 40 | 1 | 外部双臂目标 |
| 17 | 5 | `left_ankle_roll_joint` | 0.00 | 40 | 2 | ONNX |
| 18 | 11 | `right_ankle_roll_joint` | 0.00 | 40 | 2 | ONNX |
| 19 | 17 | `left_shoulder_yaw_joint` | 0.00 | 40 | 1 | 外部双臂目标 |
| 20 | 24 | `right_shoulder_yaw_joint` | 0.00 | 40 | 1 | 外部双臂目标 |
| 21 | 18 | `left_elbow_joint` | 0.97 | 40 | 1 | 外部双臂目标 |
| 22 | 25 | `right_elbow_joint` | 0.97 | 40 | 1 | 外部双臂目标 |
| 23 | 19 | `left_wrist_roll_joint` | 0.15 | 40 | 1 | 外部双臂目标 |
| 24 | 26 | `right_wrist_roll_joint` | -0.15 | 40 | 1 | 外部双臂目标 |
| 25 | 20 | `left_wrist_pitch_joint` | 0.00 | 40 | 1 | 外部双臂目标 |
| 26 | 27 | `right_wrist_pitch_joint` | 0.00 | 40 | 1 | 外部双臂目标 |
| 27 | 21 | `left_wrist_yaw_joint` | 0.00 | 40 | 1 | 外部双臂目标 |
| 28 | 28 | `right_wrist_yaw_joint` | 0.00 | 40 | 1 | 外部双臂目标 |

从 Unitree motor 顺序读取后，转成 policy 顺序：

```python
MOTOR_TO_POLICY_READ = np.array([
    0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10,
    16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28,
])
q_policy = q_motor[MOTOR_TO_POLICY_READ]
dq_policy = dq_motor[MOTOR_TO_POLICY_READ]
```

把 policy 目标转回 Unitree motor 顺序后发送：

```python
POLICY_TO_MOTOR_SEND = np.array([
    0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8,
    11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28,
])
q_target_motor = q_target_policy[POLICY_TO_MOTOR_SEND]
```

不要仅凭数组变量名猜方向；每次都应以关节名做启动自检。

## 5. 29 维输出是什么

`actions` 是 29 个无量纲 raw action，不是关节弧度，也不是力矩。actor 网络结构为 `96 → 512 → 256 → 128 → 29`，隐藏层使用 ELU，输出层无 `tanh`。因此输出理论上无界，部署程序必须检查 `NaN/Inf` 并在目标关节层实施机器安全限位。

对没有被双臂抓取控制覆盖的腰腿关节：

```python
q_target_policy = q_default + 0.25 * a_exec
tau = Kp * (q_target_policy - q_policy) - Kd * dq_policy
```

当前低层命令语义为 `qd_target = 0`、feed-forward torque `tau_ff = 0`，Kp/Kd 见上表。如果真机接口接收位置目标和增益，就发送 `q_target/Kp/Kd`；如果接口接收力矩，则由部署端按同一公式计算并再做执行器限幅。

## 6. 双臂抓取动作怎样与 ONNX 输出组合

### 6.1 抓取控制器提供的 14 维顺序

外部抓取/轨迹模块应输出“当前时刻”的 14 个双臂关节位置，单位为 rad，顺序固定为左臂 7 维后右臂 7 维：

```text
0  left_shoulder_pitch_joint
1  left_shoulder_roll_joint
2  left_shoulder_yaw_joint
3  left_elbow_joint
4  left_wrist_roll_joint
5  left_wrist_pitch_joint
6  left_wrist_yaw_joint
7  right_shoulder_pitch_joint
8  right_shoulder_roll_joint
9  right_shoulder_yaw_joint
10 right_elbow_joint
11 right_wrist_roll_joint
12 right_wrist_pitch_joint
13 right_wrist_yaw_joint
```

它们在 policy 输出中的位置是：

```python
ARM_POLICY_INDICES = np.array([
    11, 15, 19, 21, 23, 25, 27,
    12, 16, 20, 22, 24, 26, 28,
])
```

### 6.2 正确的组合不是增加维度，而是覆盖 14 个位置

不能把 29 维 ONNX 输出与 14 维双臂目标直接拼成 43 维。训练时的真实语义是：保留 ONNX 的 15 个腰腿 raw action，把 14 个网络手臂 raw action 替换为外部双臂位置目标对应的 raw action。

```python
import numpy as np

ACTION_SCALE = 0.25
ARM_POLICY_INDICES = np.array(
    [11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28],
    dtype=np.int64,
)

def compose_stand_action(
    network_action_29: np.ndarray,
    arm_position_target_14: np.ndarray,
    q_default_29: np.ndarray,
) -> np.ndarray:
    """返回实际执行的 29 维 raw action；输入位置单位为 rad。"""
    network_action_29 = np.asarray(network_action_29, dtype=np.float32)
    arm_position_target_14 = np.asarray(arm_position_target_14, dtype=np.float32)
    if network_action_29.shape != (29,):
        raise ValueError(f"network action shape must be (29,), got {network_action_29.shape}")
    if arm_position_target_14.shape != (14,):
        raise ValueError(f"arm target shape must be (14,), got {arm_position_target_14.shape}")
    if not np.all(np.isfinite(network_action_29)) or not np.all(np.isfinite(arm_position_target_14)):
        raise ValueError("Stand action or arm target contains NaN/Inf")

    executed_action = network_action_29.copy()
    executed_action[ARM_POLICY_INDICES] = (
        arm_position_target_14 - q_default_29[ARM_POLICY_INDICES]
    ) / ACTION_SCALE
    return executed_action
```

随后统一执行：

```python
q_target_policy = q_default_29 + 0.25 * executed_action
q_target_motor = q_target_policy[POLICY_TO_MOTOR_SEND]
last_action = executed_action.copy()
```

最后一行是关键：下一帧 `obs[67:96]` 必须写组合后的 `executed_action`，不能写原始 `network_action_29`。否则 ONNX 看到的 action history 与真正施加到机器人上的双臂动作不一致，破坏训练/部署契约。

### 6.3 每个 20 ms 控制周期的完整顺序

```python
# 第一次进入策略前：机器人已平滑到第一个双臂姿态；last_action 初始化为全 0。
last_action = np.zeros(29, dtype=np.float32)

while enabled:
    # 1. 读取同一时刻的 low-state，并按关节名转成 policy 顺序。
    q_policy, dq_policy, quat_wxyz, pelvis_omega = read_robot_state()

    # 2. 构造 observation。Stand command 永远为 0；此时没有未来双臂目标输入。
    obs = np.zeros(96, dtype=np.float32)
    obs[0:3] = pelvis_omega
    obs[3:6] = get_projected_gravity(quat_wxyz)
    obs[6:9] = 0.0
    obs[9:38] = q_policy - q_default_29
    obs[38:67] = dq_policy
    obs[67:96] = last_action

    # 3. ONNX 推理腰腿稳定动作。
    network_action = session.run(
        ["actions"], {"obs": np.ascontiguousarray(obs[None, :])}
    )[0][0]

    # 4. 抓取状态机只给当前时刻目标；不把未来轨迹点送进 policy。
    arm_target_14, finger_target = grasp_controller.sample_current_target()
    executed_action = compose_stand_action(network_action, arm_target_14, q_default_29)

    # 5. 关节限位、速度/加速度限位、状态超时、倾倒和有限值检查。
    q_target_policy = safety_filter(q_default_29 + 0.25 * executed_action)
    send_g1_low_command(q_target_policy[POLICY_TO_MOTOR_SEND])

    # 6. 手指/夹爪使用独立接口发送；见下一节。
    send_hand_command(finger_target)

    # 7. 保存真正执行的组合动作，供下一帧 observation 使用。
    last_action = executed_action.copy()
```

双臂目标必须连续。当前训练轨迹使用 2–6 s 的 minimum-jerk 插值并按原数据速度范围限速；真机抓取轨迹至少应满足同等平滑性，不能在相邻 20 ms 控制帧中跳变。策略启动前先用安全状态机把实际双臂平滑移动到首姿态，再启用 Stand；不要在启用策略的第一帧突然切换手臂。

## 7. 手指或夹爪动作不属于 29DoF ONNX

当前 29DoF 模型的手臂链只到左右 `wrist_yaw_joint`，没有手指、夹爪开合或灵巧手关节。若“抓取动作”同时包含双臂 14DoF 和手指/夹爪：

1. 从抓取控制器中拆出 `arm_position_target_14`，按第 6 节覆盖 ONNX 的 14 个手臂位置；
2. 手指/夹爪目标保持其设备自己的维数、单位、关节顺序和 topic，通过手部控制器单独发送；
3. 不把手指向量追加到 29 维 output，也不追加到 96 维 input；
4. 用同一个抓取状态机和时间戳同步“移臂、接近、闭合、保持、释放”阶段；
5. 闭合接触后的负载变化不需要额外输入，策略通过 IMU 和全身 `q/dq` 反馈进行补偿。

训练时左右腕末端分别在 startup 独立采样均匀分布 `0–1 kg` 的附加质量，因此该模型针对这一范围做过鲁棒化。超过每侧 1 kg、明显偏心负载、强冲击抓取或与环境持续接触都不属于已验证训练分布，必须另行仿真和分级上机测试。

## 8. Stand 专用真机入口

### 8.1 离线自检

从新终端在仓库根目录执行：

```bash
cd "$HOME/LocomotionHEC"
export REPO_ROOT="$PWD"
export UNITREE_PYTHON="$CONDA_PREFIX/bin/python"

UNITREE_PYTHON="$UNITREE_PYTHON" DRY_RUN=True NET=enp3s0 \
bash scripts/deploy_real_g1_armhack_stand.sh
```

该模式只读取 ONNX、YAML、预设 JSON 和来源 CSV，检查 `obs[1,96] → actions[1,29]`、有限输出、14 维关节顺序、来源数据一致性、硬件限位和 minimum-jerk 连续性；不会导入 Unitree SDK、初始化 DDS 或发送 LowCmd。默认应打印 `SELF-TEST PASS`。`NET` 在 dry-run 中只用于把最终命令完整打印出来。

### 8.2 真机 Python 环境

训练/IsaacLab 环境与真机 DDS 环境不是同一个职责。真机终端应使用本页 A.3–A.4 创建的 `armhack-real`：

```bash
conda activate armhack-real
export UNITREE_PYTHON="$CONDA_PREFIX/bin/python"
```

入口在任何 DDS 初始化之前检查 `cyclonedds`、`unitree_sdk2py.core.channel`、`onnxruntime`、`torch`、`numpy` 和 `PyYAML`。`onnx` 包用于第 2 节的模型结构检查。缺一项都会拒绝启动或无法完成完整自检。

### 8.3 真机命令与按键

仅在吊架、现场急停和遥控器均就绪后执行：

```bash
cd "$HOME/LocomotionHEC"
export REPO_ROOT="$PWD"
export UNITREE_PYTHON="$CONDA_PREFIX/bin/python"

UNITREE_PYTHON="$UNITREE_PYTHON" \
CONFIRM_REAL_ROBOT=I_UNDERSTAND \
NET=enp3s0 \
bash scripts/deploy_real_g1_armhack_stand.sh
```

状态顺序固定为：

1. `[DAMPING / STANDBY]`：不初始化 DDS、不发送 LowCmd；机器人必须已处于原生阻尼/待机、全身直立、双臂自然下垂；
2. 操作员只按一次 `ENTER`，程序才连接 `rt/lowstate`，用当前 29 个关节位置做安全预热，并检查躯干 roll/pitch 不超过 `0.20 rad`、腰腿相对 Stand 默认角的最大误差不超过 `0.50 rad`、双臂相对自然下垂 AD 的最大误差不超过 `0.45 rad`；
3. 检查通过后调用 `MotionSwitcher.ReleaseMode()`，进入低层调试控制并启动 50 Hz、29DoF actor；从这一阶段起每个控制帧都执行模型推理，速度命令固定 `[0,0,0]`；
4. actor 控制腰腿 15 维的同时，双臂按共享 CSV 自动执行 `AD→P0→F→P0`。三段各 7.5 s、三个端点各保持 1 s，总长 25.5 s；初始化完成前 `SPACE` 被锁定；
5. 出现 `[INIT COMPLETE]` 后，`SPACE` 才按 `P0→P1→P2→P3→P0` 切换；`q`、`Ctrl-C` 或遥控器 `Select` 停止 policy 并进入低层阻尼。

默认双臂循环为 `P0 对称基准 → P1 左臂变化 → P2 双臂非对称 → P3 左臂大范围变化 → P0`。预设保存在：

```text
Reference Data/ArmHack/StandPerturb/RealDeployment/stand_arm_presets.json
```

该 schema v2 文件只有按“左臂 7 维、右臂 7 维”排列的 14DoF 位置，并引用与 MuJoCo 共用的初始化 CSV。每次空格切换都从当前插值位置重新规划 7.5 s minimum-jerk 轨迹，所以连续快速按空格也不会在相邻控制帧产生位置跳变。腰腿 15 维从 Enter 后始终来自 actor；actor 本身仍输出 29 维，双臂 14 维随后按训练语义覆盖。默认每秒打印一次全部 29 个最终目标关节角，可用 `JOINT_PRINT_HZ=0` 关闭。

官方 README 的人工路径是在零力矩状态按遥控器 `L2+R2` 进入调试模式；本专用入口使用现有代码已经采用的 `ReleaseMode()` 软件路径，并在调用前持续保持当前位置。两种路径不要叠加操作；若机器人固件不允许 `ReleaseMode()` 或模式确认不为空，脚本会报错并停止，不应绕过检查。

常用安全参数可以在命令前覆盖：

```bash
TRANSITION_S=7.5 \
DAMPING_UPRIGHT_MAX_TILT_RAD=0.20 \
DAMPING_BODY_MAX_ERROR_RAD=0.50 DAMPING_ARM_MAX_ERROR_RAD=0.45 \
LOWSTATE_TIMEOUT_S=0.20 MAX_TILT_RAD=0.60 \
JOINT_PRINT_HZ=1.0 RUN_DURATION=30 \
UNITREE_PYTHON="$CONDA_PREFIX/bin/python" \
CONFIRM_REAL_ROBOT=I_UNDERSTAND NET=enp3s0 \
bash scripts/deploy_real_g1_armhack_stand.sh
```

2026-07-21 起，Stand 真机输出链与 HEC-5090 的 `scripts/deploy_real_g1_amp_onnx.sh` / `deploy_real_g1_amp.py` 对齐：最终目标按 `default_angles + action * action_scale` 直接发送，不再执行 `0.05 rad` 关节范围内缩裁剪，也不再执行 `4.0 rad/s` 的部署层逐帧目标变化限速。因此 `JOINT_LIMIT_MARGIN_RAD` 和 `MAX_TARGET_SPEED_RAD_S` 已从入口删除，设置它们不会生效。双臂的 7.5 s minimum-jerk 仍然保留；它是明确的动作轨迹，不是部署层输出过滤器。

首次上机不要放宽仍保留的状态阈值。脚本固定检查 policy SHA-256；若有意换模型，必须同时显式传入新的 `POLICY_PATH` 和 `EXPECTED_POLICY_SHA256`，避免模型路径改了但仍误以为是本文记录的 Stand `model_2999`。

### 8.4 吊架安全验收顺序

建议按以下顺序验收：

1. 离线检查 ONNX hash、输入输出形状、全零和随机有限输入无 `NaN/Inf`；
2. 在 MuJoCo 复用相同 50 Hz observation、关节映射和双臂覆盖，确认模型与报告一致；
3. 真机断电/卸力状态逐关节核验 `motor index ↔ joint name ↔ policy index`；
4. 吊架中只发送默认姿态，确认正负方向、零点、Kp/Kd 和限位；
5. 吊架中固定一组训练范围内双臂姿态，先不运动；
6. 吊架中执行低速 minimum-jerk 双臂轨迹，再逐步升至 `1.0x`；
7. 从 0 kg 开始逐级增加对称负载，最后才测试训练上限附近的每侧 1 kg；
8. 最后再接入实际手指/夹爪闭合，优先抓取柔软、轻量物体。

当前专用入口仍实现 low-state 超时、ONNX/状态有限值、实测硬件关节位置范围、IMU 四元数、roll/pitch、接管前姿态门槛、终端/遥控器停止和退出阻尼检查。它不再裁剪待发送的目标关节位置，也不再限制相邻控制帧的目标变化；同时没有世界系 base height 传感器输入或独立目标加速度限制器。这与 HEC-5090 通用 AMP ONNX 直通输出语义一致，但风险高于带输出过滤器的版本，首次验证必须使用吊架和现场急停。

## 9. 代码依据

```text
ONNX 导出器：
scripts/rsl_rl/export_amp_actor_to_onnx.py

96 维 observation 定义：
source/legged_lab/legged_lab/tasks/locomotion/amp/amp_env_cfg.py

训练时双臂覆盖和 action history：
source/legged_lab/legged_lab/envs/g1_perturb_env.py

Stand 任务与零速度命令：
source/legged_lab/legged_lab/tasks/locomotion/amp/config/g1_perturb/
g1_stand_perturb_env_cfg.py

随机姿态、轨迹和腕部负载：
source/legged_lab/legged_lab/tasks/locomotion/amp/config/g1_perturb/
g1_stand_randomized_payload_env_cfg.py

MuJoCo 中同语义的双臂覆盖实现：
../unitree_sim2sim2real/deploy/deploy_mujoco/armhack_stand.py

现有通用真机 observation 与 Unitree 映射参考：
../unitree_sim2sim2real/deploy/deploy_real/deploy_real_g1_amp.py
../unitree_sim2sim2real/deploy/deploy_real/configs/g1_amp.yaml

Stand 专用真机入口与双臂预设：
../scripts/deploy_real_g1_armhack_stand.sh
../unitree_sim2sim2real/deploy/deploy_real/deploy_real_g1_armhack_stand.py
Reference Data/ArmHack/StandPerturb/RealDeployment/stand_arm_presets.json
```
