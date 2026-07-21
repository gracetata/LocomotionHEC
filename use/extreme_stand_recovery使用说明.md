# G1 极鲁棒站立策略简明使用说明

> HEC-5090 使用 `/home/hecggdz/LocomotionHEC_use_20260721_1515` 和 `$HOME/miniconda3/envs/env_leglab/bin/python`。服务器一键测试与真机命令优先参见 `use/HEC5090三模型部署测试说明.md`；下文 `/home/user/...` 为原开发机示例。

本文只介绍当前最终使用的 **Extreme Stand Recovery Pose V2 `model_2999.pt`**，包括模型位置、MuJoCo 测试和真机 ONNX 部署。所有命令均从项目根目录执行。

项目根目录：

```text
/home/user/Workspace/Humanoid/Locomotion/G1-Locomotion
```

## 1. 模型路径

### 1.1 训练 checkpoint

```text
/home/user/Workspace/Humanoid/Locomotion/G1-Locomotion/legged_lab/ExtremeStandRecovery Checkpoints/2026-07-20_12-30-10_g1_extreme_stand_recovery_pose_v2_from_model4999_full_20260720/model_2999.pt
```

### 1.2 `use` 中统一使用的 ONNX

```text
/home/user/Workspace/Humanoid/Locomotion/G1-Locomotion/use/extreme_stand_recovery_pose_v2_model2999.onnx
```

该文件是当前 MuJoCo 测试和真机部署的统一模型，SHA256 为：

```text
0af2ffb24cd728352804b62669dc5264dd835329528311f9d50b90dbe2d0a0d1
```

### 1.3 原始导出文件和部署元数据

```text
/home/user/Workspace/Humanoid/Locomotion/G1-Locomotion/legged_lab/ExtremeStandRecovery Checkpoints/2026-07-20_12-30-10_g1_extreme_stand_recovery_pose_v2_from_model4999_full_20260720/exported_extreme_stand_recovery/policy.pt
/home/user/Workspace/Humanoid/Locomotion/G1-Locomotion/legged_lab/ExtremeStandRecovery Checkpoints/2026-07-20_12-30-10_g1_extreme_stand_recovery_pose_v2_from_model4999_full_20260720/exported_extreme_stand_recovery/policy.onnx
/home/user/Workspace/Humanoid/Locomotion/G1-Locomotion/legged_lab/ExtremeStandRecovery Checkpoints/2026-07-20_12-30-10_g1_extreme_stand_recovery_pose_v2_from_model4999_full_20260720/exported_extreme_stand_recovery/policy.deploy.json
```

模型接口为：`96` 维观测输入、`29` 维全身关节动作输出、`50 Hz` 控制频率、固定零速度指令。

## 2. 主要脚本路径

| 功能 | 路径 |
| --- | --- |
| 导出 TorchScript 和 ONNX | `scripts/export_g1_extreme_stand_recovery.sh` |
| MuJoCo 单项、交互和完整套件测试 | `scripts/sim2sim_g1_extreme_stand_recovery_mujoco.sh` |
| 随机初始关节姿态专项测试 | `scripts/test_g1_extreme_stand_random_pose_recovery_mujoco.sh` |
| MuJoCo 汇总报告 | `scripts/summarize_g1_extreme_stand_recovery_mujoco.py` |
| 真机 ONNX 专用入口 | `scripts/deploy_real_g1_extreme_stand_recovery_onnx.sh` |
| MuJoCo 底层运行器 | `unitree_sim2sim2real/deploy/deploy_mujoco/deploy_mujoco_g1_amp.py` |
| 真机底层运行器 | `unitree_sim2sim2real/deploy/deploy_real/deploy_real_g1_amp.py` |
| 真机 PD、默认角和关节映射配置 | `unitree_sim2sim2real/deploy/deploy_real/configs/g1_amp.yaml` |

## 3. 模型导出与文件校验

重新导出当前 checkpoint：

```bash
cd /home/user/Workspace/Humanoid/Locomotion/G1-Locomotion
bash scripts/export_g1_extreme_stand_recovery.sh

cp --preserve=mode,timestamps \
"$PWD/legged_lab/ExtremeStandRecovery Checkpoints/2026-07-20_12-30-10_g1_extreme_stand_recovery_pose_v2_from_model4999_full_20260720/exported_extreme_stand_recovery/policy.onnx" \
"$PWD/use/extreme_stand_recovery_pose_v2_model2999.onnx"
```

导出脚本会自行使用 `env_isaaclab`，不需要提前执行 `conda activate`。

检查三个部署文件：

```bash
cd /home/user/Workspace/Humanoid/Locomotion/G1-Locomotion

EXPORT_DIR="$PWD/legged_lab/ExtremeStandRecovery Checkpoints/2026-07-20_12-30-10_g1_extreme_stand_recovery_pose_v2_from_model4999_full_20260720/exported_extreme_stand_recovery"
USE_ONNX="$PWD/use/extreme_stand_recovery_pose_v2_model2999.onnx"
ls -lh "$USE_ONNX" "$EXPORT_DIR/policy.deploy.json"
sha256sum "$USE_ONNX" "$EXPORT_DIR/policy.deploy.json"
```

## 4. MuJoCo 交互可视化

### 4.1 推荐交互测试

```bash
cd /home/user/Workspace/Humanoid/Locomotion/G1-Locomotion

PROFILE=interactive USE_GLFW=True REAL_TIME=True \
POLICY_PATH="$PWD/use/extreme_stand_recovery_pose_v2_model2999.onnx" \
SIMULATION_DURATION=300 SEED=20260722 \
bash scripts/sim2sim_g1_extreme_stand_recovery_mujoco.sh
```

窗口获得焦点后：

- `空格键`：在默认初始姿态和新采样的随机29关节初始姿态之间切换，并重置仿真。
- `F`：开启或关闭随机多部位外力。
- 洋红色箭头：项目实际写入 `xfrc_applied` 的外力，起点就是受力 body。
- 脚底附近的 MuJoCo 原生箭头：地面接触力，不是随机外力位置。

外力会在骨盆、躯干、左右肩、左右肘、左右髋和左右膝之间随机选择，不包含脚和踝。终端以及 `metrics.json` 会记录每次外力的 body、三轴力和三轴力矩。

### 4.2 启动时直接使用随机姿态并开启外力

```bash
cd /home/user/Workspace/Humanoid/Locomotion/G1-Locomotion

PROFILE=interactive USE_GLFW=True REAL_TIME=True \
POLICY_PATH="$PWD/use/extreme_stand_recovery_pose_v2_model2999.onnx" \
INTERACTIVE_POSE_START_RANDOM=True \
INTERACTIVE_WRENCH_START_ENABLED=True \
SIMULATION_DURATION=300 SEED=20260722 \
bash scripts/sim2sim_g1_extreme_stand_recovery_mujoco.sh
```

### 4.3 固定 robust 场景可视化

固定场景不响应空格和 `F`，适合可重复对比：

```bash
cd /home/user/Workspace/Humanoid/Locomotion/G1-Locomotion

PROFILE=robust USE_GLFW=True REAL_TIME=True \
POLICY_PATH="$PWD/use/extreme_stand_recovery_pose_v2_model2999.onnx" \
SIMULATION_DURATION=60 SEED=20260722 \
bash scripts/sim2sim_g1_extreme_stand_recovery_mujoco.sh
```

## 5. MuJoCo 无窗口测试

可用 `PROFILE`：

| PROFILE | 内容 |
| --- | --- |
| `nominal` | 默认姿态、无外力 |
| `pose_recovery` | 只随机29关节初始姿态 |
| `recovery` | 随机关节、root 姿态和速度，无持续外力 |
| `robust` | 训练范围内初始扰动和随机外力 |
| `stress` | 超训练范围压力测试，只能用于仿真 |

### 5.1 单个场景

```bash
cd /home/user/Workspace/Humanoid/Locomotion/G1-Locomotion

PROFILE=robust USE_GLFW=False REAL_TIME=False \
POLICY_PATH="$PWD/use/extreme_stand_recovery_pose_v2_model2999.onnx" \
SIMULATION_DURATION=30 SEED=20260722 \
RESULTS_ROOT="$PWD/legged_lab/ExtremeStandRecovery Checkpoints/2026-07-20_12-30-10_g1_extreme_stand_recovery_pose_v2_from_model4999_full_20260720/exported_extreme_stand_recovery/mujoco_tests/robust_manual" \
bash scripts/sim2sim_g1_extreme_stand_recovery_mujoco.sh
```

### 5.2 依次测试全部五个固定场景

```bash
cd /home/user/Workspace/Humanoid/Locomotion/G1-Locomotion

for PROFILE_NAME in nominal pose_recovery recovery robust stress; do
  PROFILE="$PROFILE_NAME" USE_GLFW=False REAL_TIME=False \
  POLICY_PATH="$PWD/use/extreme_stand_recovery_pose_v2_model2999.onnx" \
  SIMULATION_DURATION=30 SEED=20260722 \
  RESULTS_ROOT="$PWD/legged_lab/ExtremeStandRecovery Checkpoints/2026-07-20_12-30-10_g1_extreme_stand_recovery_pose_v2_from_model4999_full_20260720/exported_extreme_stand_recovery/mujoco_tests/all_profiles_manual" \
  bash scripts/sim2sim_g1_extreme_stand_recovery_mujoco.sh || exit 1
done
```

### 5.3 五档、三种随机种子的完整测试

```bash
cd /home/user/Workspace/Humanoid/Locomotion/G1-Locomotion

SUITE=True USE_GLFW=False REAL_TIME=False \
POLICY_PATH="$PWD/use/extreme_stand_recovery_pose_v2_model2999.onnx" \
SUITE_PROFILES=nominal,pose_recovery,recovery,robust,stress \
SUITE_DURATION=30 \
SUITE_SEEDS=20260719,20260720,20260721 \
SUITE_RESULTS_ROOT="$PWD/legged_lab/ExtremeStandRecovery Checkpoints/2026-07-20_12-30-10_g1_extreme_stand_recovery_pose_v2_from_model4999_full_20260720/exported_extreme_stand_recovery/mujoco_tests/suite_manual" \
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

### 5.4 随机初始关节姿态专项测试

五个种子的无窗口测试：

```bash
cd /home/user/Workspace/Humanoid/Locomotion/G1-Locomotion

SEEDS=20260722,20260723,20260724,20260725,20260726 \
POLICY_PATH="$PWD/use/extreme_stand_recovery_pose_v2_model2999.onnx" \
DURATION=15 USE_GLFW=False REQUIRE_PASS=False \
bash scripts/test_g1_extreme_stand_random_pose_recovery_mujoco.sh
```

只可视化一个随机姿态：

```bash
cd /home/user/Workspace/Humanoid/Locomotion/G1-Locomotion

SEEDS=20260722 DURATION=30 USE_GLFW=True \
POLICY_PATH="$PWD/use/extreme_stand_recovery_pose_v2_model2999.onnx" \
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
