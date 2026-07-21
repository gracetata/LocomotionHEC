# ArmHack Stand 简明使用说明

> HEC-5090 使用 `/home/hecggdz/LocomotionHEC_use_20260721_1515` 和 `$HOME/miniconda3/envs/env_leglab/bin/python`。服务器一键测试与真机命令优先参见 `use/HEC5090三模型部署测试说明.md`；下文 `/home/user/...` 为原开发机示例。

本文只说明当前 Stand 模型的 MuJoCo 测试和真机部署。项目根目录固定为：

```text
/home/user/Workspace/Humanoid/Locomotion/G1-Locomotion
```

## 1. 模型和脚本位置

- 真机 ONNX：`use/armhack_stand_model_2999.onnx`
- MuJoCo TorchScript：`use/armhack_stand_model_2999.torchscript.pt`
- 部署元数据：`use/armhack_stand_model_2999.deploy.json`
- 来源 checkpoint：`checkpoint/stand/model_2999.pt`
- MuJoCo 入口：`legged_lab/scripts/val_mujoco_g1_armhack_stand.sh`
- 真机入口：`scripts/deploy_real_g1_armhack_stand.sh`
- 真机 Python 控制器：`unitree_sim2sim2real/deploy/deploy_real/deploy_real_g1_armhack_stand.py`
- 双臂预设：`legged_lab/Reference Data/ArmHack/StandPerturb/RealDeployment/stand_arm_presets.json`

模型身份：

```text
来源 checkpoint SHA-256:
146aca1f547ce073756c942508e8ea43c8cea91b27eee3b8347dd4131c87bc5f

ONNX SHA-256:
354bf4b35572cf6d91d44d448cde36b7bb748cffe40f1e220183bf21e5553fbf
```

ONNX 输入为 `obs[1,96]`，输出为 `actions[1,29]`，控制频率为 50 Hz。真机控制时，策略负责全身平衡；双臂 14 个关节会在策略输出之后由预设的平滑双臂动作覆盖，因此双臂动作和腿、腰平衡动作会合成为最终 29 关节目标。

MuJoCo 运行器当前加载 TorchScript，真机运行器加载 ONNX。因此 `use/` 同时保存了由同一个 checkpoint 导出的两个等价文件；随机输入对比的最大输出绝对误差为 `5.22e-7`。

## 2. 文件校验

下面整段可直接复制：

```bash
cd /home/user/Workspace/Humanoid/Locomotion/G1-Locomotion

test -s use/armhack_stand_model_2999.onnx
test -s use/armhack_stand_model_2999.torchscript.pt
test -s use/armhack_stand_model_2999.deploy.json

sha256sum use/armhack_stand_model_2999.onnx
```

ONNX 的输出必须是：

```text
354bf4b35572cf6d91d44d448cde36b7bb748cffe40f1e220183bf21e5553fbf
```

## 3. MuJoCo 测试和可视化

MuJoCo 使用本机 `gmr` 环境。入口脚本默认会找到 `/home/user/anaconda3/envs/gmr/bin/python`，通常不需要先 `conda activate`。

### 3.1 推荐的交互式可视化

这条命令明确让 MuJoCo 加载 `use/` 中的 TorchScript，并在第 0 秒启用策略，避免在等待手动按 Enter 时因只有默认 PD 姿态而摔倒：

```bash
cd /home/user/Workspace/Humanoid/Locomotion/G1-Locomotion

CHECKPOINT="$PWD/checkpoint/stand/model_2999.pt" \
POLICY_PATH="$PWD/use/armhack_stand_model_2999.torchscript.pt" \
MODE=interactive FORCE_EXPORT=False \
USE_GLFW=True REAL_TIME=True \
INTERACTIVE_AUTO_ENTER_S=0.0 \
bash legged_lab/scripts/val_mujoco_g1_armhack_stand.sh
```

运行顺序是：自然下垂 `AD` → 平直默认 `P0` → 双臂向前伸直 `F` → 收回 `P0`。自动初始化完成并出现 `INIT COMPLETE` 后，按空格依次切换 `P1 → P2 → P3 → P0`；按 `Q` 退出。

如需亲自按 Enter 进入策略模式，把上面命令中的 `INTERACTIVE_AUTO_ENTER_S=0.0` 改成：

```bash
INTERACTIVE_AUTO_ENTER_S=-1
```

此时请在窗口启动后立即按 Enter。按 Enter 前 actor 没有运行，当前待机 PD 不能保证长时间站稳。

### 3.2 无界面自动回归

自动进入策略，完成初始化并自动切换两次双臂姿态：

```bash
cd /home/user/Workspace/Humanoid/Locomotion/G1-Locomotion

CHECKPOINT="$PWD/checkpoint/stand/model_2999.pt" \
POLICY_PATH="$PWD/use/armhack_stand_model_2999.torchscript.pt" \
MODE=interactive FORCE_EXPORT=False \
USE_GLFW=False REAL_TIME=False \
INTERACTIVE_AUTO_ENTER_S=0.0 \
INTERACTIVE_AUTO_SPACE_INTERVAL_S=8.0 \
INTERACTIVE_AUTO_SPACE_MAX_SWITCHES=2 \
SIMULATION_DURATION=50 \
bash legged_lab/scripts/val_mujoco_g1_armhack_stand.sh
```

测试结束后，终端会打印报告、JSON、CSV 和躯干 6D 曲线的保存路径。

### 3.3 全部固定姿态和轨迹

```bash
cd /home/user/Workspace/Humanoid/Locomotion/G1-Locomotion

CHECKPOINT="$PWD/checkpoint/stand/model_2999.pt" \
POLICY_PATH="$PWD/use/armhack_stand_model_2999.torchscript.pt" \
MODE=all FORCE_EXPORT=False USE_GLFW=False REAL_TIME=False \
bash legged_lab/scripts/val_mujoco_g1_armhack_stand.sh
```

### 3.4 自然下垂到双臂放平

```bash
cd /home/user/Workspace/Humanoid/Locomotion/G1-Locomotion

CHECKPOINT="$PWD/checkpoint/stand/model_2999.pt" \
POLICY_PATH="$PWD/use/armhack_stand_model_2999.torchscript.pt" \
MODE=down_to_horizontal FORCE_EXPORT=False \
USE_GLFW=True REAL_TIME=True \
bash legged_lab/scripts/val_mujoco_g1_armhack_stand.sh
```

### 3.5 默认姿态、前伸、收回、自然下垂

```bash
cd /home/user/Workspace/Humanoid/Locomotion/G1-Locomotion

CHECKPOINT="$PWD/checkpoint/stand/model_2999.pt" \
POLICY_PATH="$PWD/use/armhack_stand_model_2999.torchscript.pt" \
MODE=default_forward_return_down FORCE_EXPORT=False \
USE_GLFW=True REAL_TIME=True \
bash legged_lab/scripts/val_mujoco_g1_armhack_stand.sh
```

### 3.6 运行时随机生成双臂轨迹

随机轨迹只在训练双臂姿态范围内取点，再用 minimum-jerk 插值：

```bash
cd /home/user/Workspace/Humanoid/Locomotion/G1-Locomotion

CHECKPOINT="$PWD/checkpoint/stand/model_2999.pt" \
POLICY_PATH="$PWD/use/armhack_stand_model_2999.torchscript.pt" \
MODE=generated_random_trajectory FORCE_EXPORT=False \
RANDOM_SEED=20260721 RANDOM_WAYPOINTS=8 \
RANDOM_HOLD_S=0.5 RANDOM_TRANSITION_S=2.0 \
USE_GLFW=True REAL_TIME=True \
bash legged_lab/scripts/val_mujoco_g1_armhack_stand.sh
```

其他可用模式包括 `representative_poses`、`synthesized_poses`、`randomized_poses`、`representative_trajectories`、`synthesized_trajectories` 和 `randomized_trajectories`。单项模式为：

```text
MODE=representative_pose ITEM=1..6
MODE=synthesized_pose ITEM=1..3
MODE=randomized_pose ITEM=1..8
MODE=representative_trajectory ITEM=1..4
MODE=synthesized_trajectory ITEM=1..3
MODE=randomized_trajectory ITEM=1..6
```

在上述任一 MuJoCo 命令中增加 `PAYLOAD_KG=1.0`，可给左右腕末端各增加 1 kg 测试负载；允许范围为每侧 `0–3 kg`。

## 4. 真机运行

真机必须使用吊架、现场急停和 Unitree 遥控器，机器人周围不得站人。先确认连接 G1 的有线网卡：

```bash
ip -brief link
```

下文用 `eno1` 举例；如果实际网卡不同，只替换 `NET=eno1`。真机 Python 环境必须已安装 `cyclonedds`、`unitree_sdk2py`、`onnxruntime`、`torch`、`numpy` 和 `PyYAML`。当前本机 `gmr` 环境可做离线自检，但缺少真机所需的 CycloneDDS，不能直接连接机器人。

### 4.1 离线自检

这条命令不会初始化 DDS，也不会向机器人发送指令，已经在本机验证通过：

```bash
cd /home/user/Workspace/Humanoid/Locomotion/G1-Locomotion

MODEL="$PWD/use/armhack_stand_model_2999.onnx"
META="$PWD/use/armhack_stand_model_2999.deploy.json"

DRY_RUN=True NET=eno1 \
UNITREE_PYTHON=/home/user/anaconda3/envs/gmr/bin/python \
POLICY_PATH="$MODEL" POLICY_METADATA_PATH="$META" \
EXPECTED_POLICY_SHA256="$(sha256sum "$MODEL" | awk '{print $1}')" \
EXPECTED_CHECKPOINT_SHA256=146aca1f547ce073756c942508e8ea43c8cea91b27eee3b8347dd4131c87bc5f \
bash scripts/deploy_real_g1_armhack_stand.sh
```

必须看到 `[SELF-TEST PASS]` 和“未初始化 DDS，未发送机器人命令”。

### 4.2 真机正式启动

把 `UNITREE_PYTHON` 改成已经安装好 Unitree SDK 和 CycloneDDS 的环境路径，然后整段执行：

```bash
cd /home/user/Workspace/Humanoid/Locomotion/G1-Locomotion

MODEL="$PWD/use/armhack_stand_model_2999.onnx"
META="$PWD/use/armhack_stand_model_2999.deploy.json"
UNITREE_PYTHON="$HOME/miniconda3/envs/armhack-real/bin/python"

CONFIRM_REAL_ROBOT=I_UNDERSTAND NET=eno1 \
UNITREE_PYTHON="$UNITREE_PYTHON" \
POLICY_PATH="$MODEL" POLICY_METADATA_PATH="$META" \
EXPECTED_POLICY_SHA256="$(sha256sum "$MODEL" | awk '{print $1}')" \
EXPECTED_CHECKPOINT_SHA256=146aca1f547ce073756c942508e8ea43c8cea91b27eee3b8347dd4131c87bc5f \
bash scripts/deploy_real_g1_armhack_stand.sh
```

真机状态顺序：

1. 启动后为原生阻尼/待机，机器人直立、双手自然下垂。
2. 按一次 Enter，释放高层模式、进入低层调试控制并开始 actor 推理。
3. 自动完成 `自然下垂 → P0 → 前伸 F → 返回 P0`。
4. 出现 `INIT COMPLETE` 后，按空格切换 `P1 → P2 → P3 → P0`。
5. 按 `q`、`Ctrl-C` 或遥控器 `Select` 进入阻尼并退出。

当前部署层没有 `0.05 rad` 关节安全余量裁剪，也没有逐帧目标角速度限速；仍保留 LowState 超时、躯干倾角、实测关节越界、有限值检查，以及 MuJoCo/硬件自身的原始关节和执行器约束。

## 5. 自定义双臂动作，并加入空格切换

### 5.1 当前支持的方式

当前程序不接收 ROS、DDS、UDP 或终端文本形式的实时双臂目标。它会在启动时一次性读取一个预设 JSON；自动初始化完成后，每按一次空格，就按照 JSON 中的 `space_cycle_pose_ids` 切换到下一个双臂姿态。相邻姿态之间由程序生成 minimum-jerk 平滑轨迹，腿和腰仍由 Stand ONNX 策略实时控制。

因此，现有代码下“发送自定义双臂动作”的正确方式是：

1. 在预设 JSON 中增加一个或多个仅包含双臂 14 关节的姿态。
2. 把姿态 ID 加入 `space_cycle_pose_ids`。
3. 启动前用 self-test 校验。
4. 在 MuJoCo 中验证，再在吊架保护下启动真机；初始化完成后按空格执行。

预设只在程序启动时读取一次。运行过程中修改 JSON 不会热更新，修改后必须退出并重新启动。

### 5.2 建立独立的自定义预设

不要直接覆盖原始预设。把副本放在原文件旁边，这样其中的相对 CSV 路径仍然有效：

```bash
cd /home/user/Workspace/Humanoid/Locomotion/G1-Locomotion

PRESET_DIR="$PWD/legged_lab/Reference Data/ArmHack/StandPerturb/RealDeployment"
cp "$PRESET_DIR/stand_arm_presets.json" \
   "$PRESET_DIR/stand_arm_presets_custom.json"

nano "$PRESET_DIR/stand_arm_presets_custom.json"
```

每个自定义姿态的 `positions_rad` 必须严格包含 14 个弧度值，且只能是双臂关节，固定顺序如下：

```text
1  left_shoulder_pitch_joint
2  left_shoulder_roll_joint
3  left_shoulder_yaw_joint
4  left_elbow_joint
5  left_wrist_roll_joint
6  left_wrist_pitch_joint
7  left_wrist_yaw_joint
8  right_shoulder_pitch_joint
9  right_shoulder_roll_joint
10 right_shoulder_yaw_joint
11 right_elbow_joint
12 right_wrist_roll_joint
13 right_wrist_pitch_joint
14 right_wrist_yaw_joint
```

例如，下面是由当前 `P0` 和前伸姿态 `F` 在已训练范围内按 `75%/25%` 插值得到的保守示例。把这个对象加入 JSON 的 `poses` 数组；注意与前一个对象之间要有英文逗号：

```json
{
  "id": "CUSTOM_symmetric_slight_forward",
  "label_zh": "自定义双臂轻微前伸",
  "source": "",
  "positions_rad": [
    0.062333, 0.810283, -0.254395, -0.445570,
    0.810494, 0.444300, -0.192805,
    0.068628, -0.820056, 0.247786, -0.458184,
    -0.805984, 0.447296, 0.192047
  ]
}
```

然后把该 ID 加入 `space_cycle_pose_ids`。第一个元素必须保持为 `P0_symmetric_reference`，列表中不能有重复 ID：

```json
"space_cycle_pose_ids": [
  "P0_symmetric_reference",
  "CUSTOM_symmetric_slight_forward",
  "P1_left_arm_reach",
  "P2_bilateral_asymmetric",
  "P3_left_arm_extended"
]
```

如果想执行一个由多个阶段组成的自定义动作，可以增加 `CUSTOM_step_1`、`CUSTOM_step_2`、`CUSTOM_step_3` 等多个姿态，并按动作顺序写入 `space_cycle_pose_ids`；每按一次空格前进一个阶段。当前 SPACE 模式会在姿态之间自动插值，不需要自行生成逐帧轨迹 CSV。

### 5.3 离线校验自定义预设

下面命令不会初始化 DDS，也不会连接机器人：

```bash
cd /home/user/Workspace/Humanoid/Locomotion/G1-Locomotion

MODEL="$PWD/use/armhack_stand_model_2999.onnx"
CUSTOM_PRESET="$PWD/legged_lab/Reference Data/ArmHack/StandPerturb/RealDeployment/stand_arm_presets_custom.json"

PYTHONPATH="$PWD/unitree_sim2sim2real:$PWD/unitree_sdk2_python${PYTHONPATH:+:$PYTHONPATH}" \
/home/user/anaconda3/envs/gmr/bin/python \
  unitree_sim2sim2real/deploy/deploy_real/deploy_real_g1_armhack_stand.py \
  --self-test \
  --policy "$MODEL" \
  --presets "$CUSTOM_PRESET" \
  --config unitree_sim2sim2real/deploy/deploy_real/configs/g1_amp.yaml
```

必须看到 `[SELF-TEST PASS]`，并在 `poses` 和 `SPACE` 输出中看到自定义 ID。校验器会拒绝以下数据：不是 14 个数、包含 NaN/Inf、超过 G1 硬件关节范围、ID 重复、引用不存在的 ID，或把 `P0_symmetric_reference` 从 SPACE 列表首位移除。

### 5.4 在 MuJoCo 中通过空格测试

```bash
cd /home/user/Workspace/Humanoid/Locomotion/G1-Locomotion

CUSTOM_PRESET="$PWD/legged_lab/Reference Data/ArmHack/StandPerturb/RealDeployment/stand_arm_presets_custom.json"

CHECKPOINT="$PWD/checkpoint/stand/model_2999.pt" \
POLICY_PATH="$PWD/use/armhack_stand_model_2999.torchscript.pt" \
ARM_PRESET_PATH="$CUSTOM_PRESET" \
MODE=interactive FORCE_EXPORT=False \
USE_GLFW=True REAL_TIME=True \
INTERACTIVE_AUTO_ENTER_S=0.0 \
INTERACTIVE_TRANSITION_S=7.5 \
bash legged_lab/scripts/val_mujoco_g1_armhack_stand.sh
```

等待 `INIT COMPLETE` 后按空格。终端应打印 `SPACE -> CUSTOM_symmetric_slight_forward`，窗口中双臂平滑移动，躯干和腿由策略继续保持平衡。

### 5.5 真机离线自检自定义预设

```bash
cd /home/user/Workspace/Humanoid/Locomotion/G1-Locomotion

MODEL="$PWD/use/armhack_stand_model_2999.onnx"
META="$PWD/use/armhack_stand_model_2999.deploy.json"
CUSTOM_PRESET="$PWD/legged_lab/Reference Data/ArmHack/StandPerturb/RealDeployment/stand_arm_presets_custom.json"

DRY_RUN=True NET=eno1 \
UNITREE_PYTHON=/home/user/anaconda3/envs/gmr/bin/python \
POLICY_PATH="$MODEL" POLICY_METADATA_PATH="$META" \
PRESET_PATH="$CUSTOM_PRESET" TRANSITION_S=7.5 \
EXPECTED_POLICY_SHA256="$(sha256sum "$MODEL" | awk '{print $1}')" \
EXPECTED_CHECKPOINT_SHA256=146aca1f547ce073756c942508e8ea43c8cea91b27eee3b8347dd4131c87bc5f \
bash scripts/deploy_real_g1_armhack_stand.sh
```

### 5.6 真机运行自定义空格序列

先通过上一节的 self-test 和 MuJoCo 测试，再在吊架、急停和现场操作员保护下执行：

```bash
cd /home/user/Workspace/Humanoid/Locomotion/G1-Locomotion

MODEL="$PWD/use/armhack_stand_model_2999.onnx"
META="$PWD/use/armhack_stand_model_2999.deploy.json"
CUSTOM_PRESET="$PWD/legged_lab/Reference Data/ArmHack/StandPerturb/RealDeployment/stand_arm_presets_custom.json"
UNITREE_PYTHON="$HOME/miniconda3/envs/armhack-real/bin/python"

CONFIRM_REAL_ROBOT=I_UNDERSTAND NET=eno1 \
UNITREE_PYTHON="$UNITREE_PYTHON" \
POLICY_PATH="$MODEL" POLICY_METADATA_PATH="$META" \
PRESET_PATH="$CUSTOM_PRESET" TRANSITION_S=7.5 \
EXPECTED_POLICY_SHA256="$(sha256sum "$MODEL" | awk '{print $1}')" \
EXPECTED_CHECKPOINT_SHA256=146aca1f547ce073756c942508e8ea43c8cea91b27eee3b8347dd4131c87bc5f \
bash scripts/deploy_real_g1_armhack_stand.sh
```

按 Enter 完成自动初始化，看到 `INIT COMPLETE` 后再按空格。一次空格只切换到一个目标姿态；如果在上一段插值尚未完成时再次按空格，程序会从当前双臂位置连续地转向下一个目标，不会产生位置跳变，但可能永远没有到达中间姿态，因此正常操作应等待当前 7.5 秒动作完成。

`TRANSITION_S`（真机）和 `INTERACTIVE_TRANSITION_S`（MuJoCo）控制每次切换时间。代码要求至少 2 秒；训练主要覆盖 3–9 秒，建议保持在该范围内，首次真机测试建议使用默认 7.5 秒。

自定义姿态即使处于机械限位内，也不一定处于策略训练分布内。首次使用应从现有姿态之间的小比例插值开始，不要直接发送大幅、快速或左右严重失衡的动作。策略不会提前知道下一个双臂目标，只会根据当前观测和已经执行的动作实时补偿。

如果需要另一个程序在运行中连续发送任意 50 Hz 双臂轨迹，而不是通过空格切换预加载姿态，当前版本尚未提供外部消息接口；这需要新增带超时、范围校验和急停联锁的进程间通信接口，不能把未校验的关节数组直接接到真机控制循环。

## 6. 重新导出（一般不需要）

`use/` 中模型已经导出并验证，正常使用不需要执行本节。如来源 checkpoint 被明确替换，可重新生成 ONNX、TorchScript 和元数据：

```bash
cd /home/user/Workspace/Humanoid/Locomotion/G1-Locomotion

/home/user/anaconda3/envs/env_isaaclab/bin/python \
  legged_lab/scripts/rsl_rl/export_amp_actor_to_onnx.py \
  --robot g1 \
  --checkpoint checkpoint/stand/model_2999.pt \
  --output use/armhack_stand_model_2999.onnx \
  --jit-output use/armhack_stand_model_2999.torchscript.pt \
  --metadata use/armhack_stand_model_2999.deploy.json \
  --default-command 0 0 0
```

重新导出后必须重新执行 ONNX 校验、MuJoCo 测试和真机离线自检，并更新文档中的 SHA-256。
