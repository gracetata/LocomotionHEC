# ArmHack 训练实现、数据路径与验证指南

## 1. 当前状态

截至 2026-07-18，Stand 已完成数据整理、奖励修复、无跳变初始化、两阶段正式训练、确定性双臂测试集、IsaacLab 报告和 MuJoCo sim2sim 验证。`scripts/train_g1_armhack_stand.sh` 已从 S3 G1 `model_9996.pt` 完成 `4096 env × 3000 iteration` 的第一阶段训练；`scripts/train_g1_armhack_stand_randomized_payload.sh` 又从第一阶段 `model_2999.pt` 完成 `4096 env × 3000 iteration` 的第二阶段训练，加入范围内随机姿态、minimum-jerk 插值轨迹和腕部末端附加质量。针对真机容易失稳的问题，第三阶段 `scripts/train_g1_armhack_stand_robust.sh` 已加入更大的摔倒惩罚、torso 外力和关节参数随机化，并完成最终配置 smoke；正式第三阶段尚未开始，所以当前默认测试模型仍是第二阶段的新 `model_2999.pt`。

```text
LeggedLab-Isaac-AMP-G1-StandPerturb-v0
LeggedLab-Isaac-AMP-G1-StandRandomizedPayload-v0
LeggedLab-Isaac-AMP-G1-StandRobust-v0
LeggedLab-Isaac-AMP-G1-WalkPerturbFinetune-v0
```

当前验证结果：

- 静态检查 `10 passed`；
- Stand 第一阶段基线为 `BaselineModel9996/model_9996.pt`，大小 `16,202,421 bytes`，iteration `9996`，actor `96→29`、critic input `297`，SHA-256 为 `bc30bc5171d211fa414fbeab31452b92ad76ca7f6ad76a2417a6e7f7515a0fa6`；当前第二阶段续训基线为正式 `model_2999.pt`；
- 随机静止姿态阶段：`4 env × 1 iteration` 正常完成，日志为 `csv_motion_scale=0`、`curriculum_stage=0`；随机起点时间均值 `198.6063 s`、标准差 `137.4946 s`，确认并行环境不是同一起点；
- 静止阶段长短测：`4 env × 42 iteration`、4032 个环境步正常完成，所有已结束 episode 均为 `time_out`，`base_height=0`、`bad_orientation=0`，保存 `model_41.pt`；
- 历史低速连续运动阶段：`2 env × 1 iteration` 正常完成，日志为 `csv_motion_scale=0.25`、`curriculum_stage=2`；该记录只对应旧课程；
- 当前全速连续运动阶段：`2 env × 1 iteration` 已从 `model_9996.pt` 正常完成，日志为 `csv_motion_scale=1.0000`、`curriculum_stage=2.0000`，并在 Stand 专用目录保存 `model_0.pt`；
- 当前入口已确认从 `model_9996.pt` 做 policy-only 恢复、加载同一冻结基线做 KL，并在 Stand 专用目录生成 `model_0.pt`；
- 第一阶段正式 run 为 `2026-07-14_20-34-20_armhack_stand_curriculum_1x_from_model9996_full_20260714`，最终 `model_2999.pt` 的 SHA-256 为 `2c87cc2cc3706c1024594d14d85a34e7bf468b54f6b66e49b6155ef72a2dbd16`；它是第二阶段训练起点，不再是默认测试模型；
- 当前最新 Stand 正式 run 为 `2026-07-15_14-12-54_armhack_stand_randomized_payload_from_model2999_full_20260715`，新 `model_2999.pt` 的 SHA-256 为 `877e929d516cffe9131cc235477ceef4b226ec69e41c0f1c23e48816cfa28821`，大小 `14,825,781 bytes`、内部 iteration `2999`；训练末次统计为 episode length `1000/1000`、timeout `99.95%`、base-height termination `0`、bad-orientation termination `0.05%`、torso roll/pitch 误差 `0.0168/0.0346 rad`；
- 新姿态库共 512 个 14-DoF 双臂姿态（64 个实测覆盖锚点 + 448 个 2–4 父姿态凸组合），所有关节均位于原训练数据范围内；
- 第二阶段入口除早期 `16 env × 1 iteration` smoke 外，已完成正式 `4096 env × 3000 iteration` 训练；最终 `random_motion_scale=1.0`、`curriculum_stage=2`、姿态库大小 `512`、平均过渡时长 `4.4112 s`；
- 第三阶段最终配置已完成 `8 env × 1 iteration` smoke：torso wrench 在短测试中按 `0.04–0.08 s` 实际重采样，actuator gain、joint parameter、腕端 payload 三类 startup event 均生效，Reward Manager 确认为 `termination_penalty=-500`，共 192 step、完成 PPO 更新并保存 `model_0.pt`；正式训练使用默认 `2–5 s` 外力间隔，细节见第 17 节；
- 本机历史 Stand 课程使用 `model_7999.pt` 起步、最终速度 `0.25x`；它只作为历史对照，不再作为当前 Stand 默认模型；
- schema v5 确定性测试集在旧样本上新增 8 个固定随机覆盖姿态和 6 条固定 minimum-jerk 轨迹；完整序列为 208.08 s / 10,404 step / 59 阶段。新正式模型已完成无负载全量测试，`termination/reset=0`；每侧 1 kg 全量测试结果见第 1.1.6 节；
- 同一模型已导出为 `96→29` TorchScript/ONNX，并在 S3 G1 MuJoCo XML 中完成 schema v5 全量 sim2sim：0 kg 和每侧 1 kg 均为 10,405 个记录样本、完整播放、`healthy=True` 且无摔倒；命令、实现边界和结果见第 1.1.9 节；
- 当前正式 Stand actor 另已固定导出为 `deployment/armhack_stand/stand.onnx`，并生成零速度部署元数据；96 维输入、29 维输出、Unitree motor 映射、双臂抓取动作覆盖和手指/夹爪独立控制边界见 `docs/armhack_stand_real_deployment.md`；
- 历史 `0.25x` 正式训练最终点为 mean episode length `1000/1000`、timeout `99.98%`、base-height termination `0`、bad-orientation termination `0.02%`，torso roll/pitch 误差 `0.0074/0.0130 rad`；
- 旧 Walk smoke 使用了来源错误的 Stand `model_7999`，只作为路径问题的诊断记录，不能再作为有效 Walk 起点测试；
- 正确 Walk 起点已锁定为 `checkpoint/model_9996/locomotion.onnx`；原始 `model_9996.pt` 的 8 个 actor 张量与 ONNX 逐元素完全一致；
- `pos2_down`、`pos1_back`、`pos3_front` 均从正确 model9996 完成 `2 env × 1 iteration`，pose index 为 1/0/2，初始化误差与两类摔倒终止均为 0；
- 从正确基线生成的 Walk `model_0.pt` 完成一次 `MODE=resume` full-state 恢复 smoke；
- 此前 Stand/旧 Walk smoke checkpoint 均从专用目录完成 `1 env × 5 steps` headless 回放；正确 model9996 Walk 的三姿态训练链与 full-resume 链已单独验证；
- 上述训练与回放均没有 traceback、`FileNotFoundError`、Isaac/Python 版本错误或 Nav2 数据缺失错误；
- 训练不再依赖 `/home/hecggdz/...` 等机器绝对路径。

这里的 smoke `model_0.pt` 只证明任务注册、数据加载、Isaac Sim、rollout、PPO 更新和保存执行链可运行，不代表一轮更新已经学会新任务。第三阶段正式模型生成前，测试和可视化仍应显式指向第二阶段最新正式 run 的 `model_2999.pt`。可视化脚本默认值已同步到该模型，但显式传入 `CHECKPOINT="$STAND_CKPT"` 更利于审计。

## 1.1 Stand 训练、测试与可视化快速入口

本节是当前 Stand 的权威操作入口。IsaacLab 训练/回放使用 `env_isaaclab`；第 1.1.9 节的 MuJoCo rollout 使用 `gmr`，但导出步骤仍由脚本显式调用 `env_isaaclab`。所有命令都在 `legged_lab` 目录执行；命令中的路径包含空格，必须保留双引号。

### 1.1.1 进入环境并指定当前正式模型

`STAND_CKPT` 不是系统内置变量，而是当前终端中临时保存 checkpoint 路径的 Shell 变量。**每次新开终端，它都会消失，因此都必须先完整执行下面这段初始化命令。** 后续测试、可视化和录制命令应在执行完本段的同一个终端中运行。

```bash
source /home/user/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
cd /home/user/Workspace/Humanoid/Locomotion/G1-Locomotion/legged_lab

export STAND_RUN='2026-07-15_14-12-54_armhack_stand_randomized_payload_from_model2999_full_20260715'
export STAND_CKPT="$PWD/ArmHack Checkpoints/StandPerturb/${STAND_RUN}/model_2999.pt"

printf '当前 Stand checkpoint：%s\n' "$STAND_CKPT"
test -f "$STAND_CKPT" || { echo "错误：checkpoint 不存在" >&2; }
sha256sum "$STAND_CKPT"
```

其中 `$STAND_CKPT` 表示“取出变量 `STAND_CKPT` 中保存的完整路径”；`export` 让随后启动的子脚本也能读取该变量。路径中包含 `ArmHack Checkpoints` 空格，所以后续必须写成 `"$STAND_CKPT"`，不能省略双引号。可以随时执行 `echo "$STAND_CKPT"` 检查当前终端是否已经定义。

预期 SHA-256：

```text
877e929d516cffe9131cc235477ceef4b226ec69e41c0f1c23e48816cfa28821
```

### 1.1.2 训练前数据与代码检查

```bash
python scripts/tools/check_armhack_reference_data.py --stand-only

bash -n scripts/train_g1_armhack_stand.sh
bash -n scripts/train_g1_armhack_stand_randomized_payload.sh
bash -n scripts/vis_g1_armhack_stand_eval.sh

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/user/.local/bin/pytest -q \
  source/legged_lab/test/test_g1_perturb_static.py
```

当前预期结果为数据检查 `[PASS]`、静态测试 `10 passed`。`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` 用于隔离 `~/.local` 中与 Conda 无关的 pytest 插件冲突；训练入口本身不依赖该设置。

### 1.1.3 先做两种训练冒烟测试

随机静态姿态阶段：

```bash
NUM_ENVS=2 MAX_ITERATIONS=1 \
STATIC_ITERATIONS=1 RAMP_ITERATIONS=1 \
RUN_NAME=armhack_stand_static_stage_smoke \
bash scripts/train_g1_armhack_stand.sh
```

直接进入最终 `1.0x` 轨迹阶段：

```bash
NUM_ENVS=2 MAX_ITERATIONS=1 \
STATIC_ITERATIONS=0 RAMP_ITERATIONS=0 \
FINAL_MOTION_SCALE=1.0 \
RUN_NAME=armhack_stand_full_speed_1x_smoke \
bash scripts/train_g1_armhack_stand.sh
```

成功标志是日志出现 `Learning iteration 0/1` 并保存 `model_0.pt`。第二条命令还必须打印 `ArmHack/csv_motion_scale=1.0000` 和 `ArmHack/curriculum_stage=2.0000`。smoke 模型不能作为正式性能模型。

### 1.1.4 启动第一阶段完整 Stand 训练

默认完整训练就是本轮已经验证的配置：4096 个环境、3000 iteration、20 s episode，从 `model_9996.pt` 做 policy-only 初始化。

```bash
RUN_NAME=armhack_stand_curriculum_1x_from_model9996_full \
NUM_ENVS=4096 \
MAX_ITERATIONS=3000 \
STATIC_ITERATIONS=500 \
RAMP_ITERATIONS=1000 \
FINAL_MOTION_SCALE=1.0 \
QUIET_TERMINAL=False \
bash scripts/train_g1_armhack_stand.sh
```

课程含义：

| iteration | 阶段 | 双臂输入 |
|---:|---|---|
| `0..499` | 随机姿态静止 | reset 随机选择轨迹内姿态，随后保持不动 |
| `500..1499` | 连续升速 | 从 `0` 线性升至原轨迹 `1.0x` |
| `1500..2999` | 全速适应 | 持续按原轨迹 `1.0x` 运动 |

训练日志、配置和 TensorBoard event 写入：

```text
logs/rsl_rl/g1_stand_perturb/<时间戳>_<RUN_NAME>/
```

checkpoint 同时保存到：

```text
ArmHack Checkpoints/StandPerturb/<时间戳>_<RUN_NAME>/model_*.pt
```

### 1.1.5 开启 TensorBoard

另开一个终端执行：

```bash
cd /home/user/Workspace/Humanoid/Locomotion/G1-Locomotion/legged_lab
/home/user/anaconda3/envs/proknee/bin/tensorboard \
  --logdir logs/rsl_rl/g1_stand_perturb \
  --host 127.0.0.1 \
  --port 6006 \
  --reload_interval 10
```

浏览器打开 `http://127.0.0.1:6006`。训练仍使用 `env_isaaclab`；这里使用 `proknee` 中已验证可用的 TensorBoard 2.19.0 只负责读取 event 文件，不参与 Isaac Sim 或策略训练。

### 1.1.6 确定性 headless 测试并生成报告

如测试数据未生成或构建脚本有改动，先重建并检查：

```bash
python scripts/tools/build_armhack_stand_randomized_training_data.py
python scripts/tools/build_armhack_stand_visualization_suite.py
python scripts/tools/check_armhack_reference_data.py --stand-only
```

然后用当前正式模型跑完整 208.08 s、10,404-step、`1.0x` schema v5 测试。以下是**新终端可以整段复制**的完整命令，不依赖之前终端中残留的变量：

```bash
source /home/user/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
cd /home/user/Workspace/Humanoid/Locomotion/G1-Locomotion/legged_lab

python scripts/tools/check_armhack_reference_data.py --stand-only

export STAND_CKPT="$PWD/ArmHack Checkpoints/StandPerturb/2026-07-15_14-12-54_armhack_stand_randomized_payload_from_model2999_full_20260715/model_2999.pt"
test -f "$STAND_CKPT" || { echo "错误：checkpoint 不存在" >&2; }

CHECKPOINT="$STAND_CKPT" \
MODE=all \
HEADLESS=True \
REAL_TIME=False \
PAYLOAD_KG=0 \
MAX_STEPS=10404 \
bash scripts/vis_g1_armhack_stand_eval.sh

# 负载验收：左右腕末端各固定附加 1 kg
CHECKPOINT="$STAND_CKPT" \
MODE=all \
HEADLESS=True \
REAL_TIME=False \
PAYLOAD_KG=1.0 \
MAX_STEPS=10404 \
bash scripts/vis_g1_armhack_stand_eval.sh
```

默认 `MODE=all` 按固定顺序覆盖 6 个训练可达代表姿态、3 个旧合成姿态、8 个新随机覆盖姿态、4 条实测 `1.0x` 轨迹、3 条实测轨迹凸组合和 6 条新 minimum-jerk 插值轨迹。原测试集中的 `404.897585 s` 双臂下垂姿态仍从这套默认验收序列中排除；它只会被第 1.1.7.1 节的 `MODE=down_to_horizontal` 边界测试显式复用。运行时不随机抽样。`PAYLOAD_KG` 是左右腕 yaw 末端 link 各自附加的固定测试质量，`0` 是无额外负载基准，`1.0` 是每侧 1 kg 负载测试。退出时自动生成：

```text
<checkpoint目录>/Test Reports/StandArmOnly/model_2999_877e929d516c__all__payload_0kg.md
<checkpoint目录>/Test Reports/StandArmOnly/model_2999_877e929d516c__all__payload_0kg__torso_world_6d.png
<checkpoint目录>/Test Reports/StandArmOnly/model_2999_877e929d516c__all__payload_1kg.md
<checkpoint目录>/Test Reports/StandArmOnly/model_2999_877e929d516c__all__payload_1kg__torso_world_6d.png
```

报告文件名自动包含 checkpoint SHA-256 前 12 位、测试项和每侧固定负载；例如新模型无负载整套测试为 `model_2999_877e929d516c__all__payload_0kg.md`，每侧 1 kg 为 `model_2999_877e929d516c__all__payload_1kg.md`。这样不会与第一阶段旧 `model_2999.pt` 或不同负载测试互相覆盖。报告包含固定测试负载、termination/reset 次数、躯干稳定指标、全部 29 个实际关节的平均逐步波动，以及 `torso_link` 在世界坐标系中的 6D 位移统计。PNG 上半部分绘制 `delta_x/y/z`，中部绘制 `delta_roll/pitch/yaw`，底部色带新增 `GP`（新随机姿态）和 `GT`（新插值轨迹）标记。重建测试集后必须重新运行本命令；旧 schema v4 报告不能作为当前 schema v5 结果。完整稳定通过的最低条件是 `termination/reset 事件数=0`。

新模型已在 schema v5 上分别完成每侧 0 kg 和每侧 1 kg 的完整 10,404-step 测试，两次均为 `termination/reset=0`。无负载 torso 水平位移 RMS 为 `0.01281 m`、pitch 位移 RMS 为 `0.05970 rad / 3.42°`；每侧 1 kg 时分别为 `0.02498 m` 和 `0.09972 rad / 5.71°`。因此“不摔倒”通过，但负载下 pitch 稳定性仍未达到文档建议的 3° RMS。总报告为 `Test Reports/StandArmOnly/model_2999_877e929d516c__schema_v5_test_summary.md`。原第一阶段 `model_2999.pt` 的 5198-step schema v4 结果只作历史基准，不能与本次 schema v5 数值直接对比。

### 1.1.7 GUI 可视化整套测试

以下同样是**新终端可以整段复制**的完整 GUI 命令：

```bash
source /home/user/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
cd /home/user/Workspace/Humanoid/Locomotion/G1-Locomotion/legged_lab

python scripts/tools/check_armhack_reference_data.py --stand-only

export STAND_CKPT="$PWD/ArmHack Checkpoints/StandPerturb/2026-07-15_14-12-54_armhack_stand_randomized_payload_from_model2999_full_20260715/model_2999.pt"
test -f "$STAND_CKPT" || { echo "错误：checkpoint 不存在" >&2; }

CHECKPOINT="$STAND_CKPT" \
MODE=all \
HEADLESS=False \
REAL_TIME=True \
FOLLOW_CAMERA=True \
CAMERA_VIEW=front \
bash scripts/vis_g1_armhack_stand_eval.sh
```

分类查看：

```bash
CHECKPOINT="$STAND_CKPT" MODE=representative_poses \
  bash scripts/vis_g1_armhack_stand_eval.sh
CHECKPOINT="$STAND_CKPT" MODE=synthesized_poses \
  bash scripts/vis_g1_armhack_stand_eval.sh
CHECKPOINT="$STAND_CKPT" MODE=randomized_poses PAYLOAD_KG=0 \
  bash scripts/vis_g1_armhack_stand_eval.sh
CHECKPOINT="$STAND_CKPT" MODE=representative_trajectories \
  bash scripts/vis_g1_armhack_stand_eval.sh
CHECKPOINT="$STAND_CKPT" MODE=synthesized_trajectories \
  bash scripts/vis_g1_armhack_stand_eval.sh
CHECKPOINT="$STAND_CKPT" MODE=randomized_trajectories PAYLOAD_KG=1.0 \
  bash scripts/vis_g1_armhack_stand_eval.sh

# 专项：双臂下垂保持 5 s，6 s 平滑抬到向前水平，再保持 9 s
CHECKPOINT="$STAND_CKPT" MODE=down_to_horizontal PAYLOAD_KG=0 \
  bash scripts/vis_g1_armhack_stand_eval.sh
```

逐项定位：

```bash
CHECKPOINT="$STAND_CKPT" MODE=representative_pose ITEM=1 \
  bash scripts/vis_g1_armhack_stand_eval.sh       # ITEM=1..6
CHECKPOINT="$STAND_CKPT" MODE=synthesized_pose ITEM=1 \
  bash scripts/vis_g1_armhack_stand_eval.sh       # ITEM=1..3
CHECKPOINT="$STAND_CKPT" MODE=randomized_pose ITEM=1 PAYLOAD_KG=1.0 \
  bash scripts/vis_g1_armhack_stand_eval.sh       # ITEM=1..8
CHECKPOINT="$STAND_CKPT" MODE=representative_trajectory ITEM=1 \
  bash scripts/vis_g1_armhack_stand_eval.sh       # ITEM=1..4
CHECKPOINT="$STAND_CKPT" MODE=synthesized_trajectory ITEM=1 \
  bash scripts/vis_g1_armhack_stand_eval.sh       # ITEM=1..3
CHECKPOINT="$STAND_CKPT" MODE=randomized_trajectory ITEM=1 PAYLOAD_KG=1.0 \
  bash scripts/vis_g1_armhack_stand_eval.sh       # ITEM=1..6
```

`CAMERA_VIEW` 可设为 `front`、`chase` 或 `side`。若整套回放发生 reset，应立即改用逐项命令定位失败姿态或轨迹；reset 后整套 CSV 会从开头重新播放，不能把后续画面视为已经覆盖剩余样本。

### 1.1.7.1 双臂下垂到向前放平专项测试

这是独立于默认 schema v5 `MODE=all` 的 20 s 边界压力测试，使用固定文件：

```text
Reference Data/ArmHack/StandPerturb/TestData/ArmOnly/
special/arms_down_to_forward_horizontal_20s_50hz.csv
```

测试时间线固定为：

| 时间 | 阶段标记 | 内容 |
|---:|---|---|
| 0–5 s | `AD` | 双臂下垂静止保持 |
| 5–11 s | `D→H` | 五次 minimum-jerk 平滑抬臂 |
| 11–20 s | `AH` | 双臂向前水平静止保持 |

起点是原始双臂 CSV 的 `404.897585 s` 姿态。它超出静态随机姿态课程可采样的 `0–384.667792 s`，因此继续从代表姿态、合成姿态和默认 `all` 验收序列中排除，只在本专项测试中显式使用。终点不是凭空逐关节随机生成，而是把实测左臂 `72.238928 s` 与实测右臂 `323.462679 s` 的完整 7-DoF 姿态组合成一个 14-DoF 双臂姿态；MuJoCo 正向运动学复核得到左右腕相对同侧肩的竖直偏差为 `-0.00119 m`、`+0.00396 m`，即两腕约与肩同高。过渡时长 6 s 也大于按训练姿态库安全速度估算的最低 `5.93848 s`。CSV 不含腰、腿或根节点目标。

新终端用 IsaacLab GUI 可视化时，完整复制：

```bash
source /home/user/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
cd /home/user/Workspace/Humanoid/Locomotion/G1-Locomotion/legged_lab

export STAND_CKPT="$PWD/ArmHack Checkpoints/StandPerturb/2026-07-15_14-12-54_armhack_stand_randomized_payload_from_model2999_full_20260715/model_2999.pt"
test -f "$STAND_CKPT" || { echo "错误：checkpoint 不存在" >&2; exit 1; }

CHECKPOINT="$STAND_CKPT" MODE=down_to_horizontal PAYLOAD_KG=0 \
HEADLESS=False REAL_TIME=True FOLLOW_CAMERA=True CAMERA_VIEW=front \
bash scripts/vis_g1_armhack_stand_eval.sh
```

无窗口复测并生成 Markdown 报告和 torso 世界系 6D 曲线：

```bash
CHECKPOINT="$STAND_CKPT" MODE=down_to_horizontal PAYLOAD_KG=0 \
HEADLESS=True REAL_TIME=False MAX_STEPS=1000 \
bash scripts/vis_g1_armhack_stand_eval.sh
```

录制完整 20 s 视频：

```bash
CHECKPOINT="$STAND_CKPT" MODE=down_to_horizontal PAYLOAD_KG=0 \
HEADLESS=True REAL_TIME=False MAX_STEPS=1000 \
bash scripts/vis_g1_armhack_stand_eval.sh \
  --video --video_length 1000
```

MuJoCo 无头测试和 GUI 可视化使用同一个模式名：

```bash
# 新终端先进入 MuJoCo 环境；脚本会用 env_isaaclab 导出 checkpoint
source /home/user/anaconda3/etc/profile.d/conda.sh
conda activate gmr
cd /home/user/Workspace/Humanoid/Locomotion/G1-Locomotion/legged_lab
export STAND_CKPT="$PWD/ArmHack Checkpoints/StandPerturb/2026-07-15_14-12-54_armhack_stand_randomized_payload_from_model2999_full_20260715/model_2999.pt"

# 无窗口报告
CHECKPOINT="$STAND_CKPT" MODE=down_to_horizontal PAYLOAD_KG=0 \
USE_GLFW=False REAL_TIME=False \
bash scripts/val_mujoco_g1_armhack_stand.sh

# 实时 GUI
CHECKPOINT="$STAND_CKPT" MODE=down_to_horizontal PAYLOAD_KG=0 \
USE_GLFW=True REAL_TIME=True \
bash scripts/val_mujoco_g1_armhack_stand.sh
```

2026-07-17 对 SHA 前缀 `877e929d516c` 的第二阶段新 `model_2999.pt` 已实际完成两种模拟器的无负载测试：

| 模拟器 | 完整执行 | 终止/摔倒 | 最低 root 高度 | torso 水平位移 RMS/最大值 | torso pitch 位移 RMS/最大值 |
|---|---|---|---:|---:|---:|
| IsaacLab | 20 s / 1000 step | `termination/reset=0` | 报告未单列 | 0.02135 / 0.03903 m | 0.10102 / 0.15291 rad |
| MuJoCo | 20 s / 1001 sample | `healthy=True`，无摔倒 | 0.78596 m | 0.02657 / 0.04724 m | 0.07880 / 0.10860 rad |

IsaacLab 报告和曲线位于：

```text
<checkpoint目录>/Test Reports/StandArmOnly/
  model_2999_877e929d516c__down_to_horizontal__payload_0kg.md
  model_2999_877e929d516c__down_to_horizontal__payload_0kg__torso_world_6d.png
```

MuJoCo 报告、JSON、逐帧 CSV 和曲线位于：

```text
<checkpoint目录>/Test Reports/StandArmOnlyMuJoCo/
  model_2999_877e929d516c__mujoco__down_to_horizontal__payload_0kg.md
  model_2999_877e929d516c__mujoco__down_to_horizontal__payload_0kg.json
  model_2999_877e929d516c__mujoco__down_to_horizontal__payload_0kg__trace.csv
  model_2999_877e929d516c__mujoco__down_to_horizontal__payload_0kg__torso_world_6d.png
```

两种模拟器都没有摔倒，说明最低生存稳定性通过；但 IsaacLab pitch 位移 RMS 为 `0.10102 rad ≈ 5.79°`、最大值 `0.15291 rad ≈ 8.76°`，明显超过第 1 节建议的 3° RMS。曲线也显示前倾主要持续到 `AH` 放平保持段，所以当前结果应表述为“能够站住，但向前放平后的躯干姿态精度不足”，不能只按 termination 数宣称高精度稳定。

### 1.1.8 无窗口 smoke 与视频录制

本节命令复用当前终端中的 `STAND_CKPT`。如果是新终端，先完整执行第 1.1.1 节的初始化块，再执行下面的命令。

只检查 checkpoint、Isaac Sim 和回放入口能否加载：

```bash
CHECKPOINT="$STAND_CKPT" \
MODE=representative_trajectory ITEM=1 \
HEADLESS=True REAL_TIME=False MAX_STEPS=20 \
bash scripts/vis_g1_armhack_stand_eval.sh
```

录制完整确定性测试：

```bash
CHECKPOINT="$STAND_CKPT" \
MODE=all \
PAYLOAD_KG=0 \
HEADLESS=True REAL_TIME=False MAX_STEPS=10404 \
bash scripts/vis_g1_armhack_stand_eval.sh \
  --video --video_length 10404
```

视频写入 checkpoint 所在目录的 `videos/play/`；测试报告写入同一目录下的 `Test Reports/StandArmOnly/`。20-step smoke 只证明入口可运行，不代表策略通过完整轨迹测试。

### 1.1.9 MuJoCo sim2sim 测试与可视化

Stand 不能直接把导出的 actor 交给通用 locomotion MuJoCo 命令。训练环境会在 actor 推理后覆盖 14 个双臂 action，并把覆盖后的 29 维 raw action 作为下一帧 `last_action`；若省略这一步，policy 观察和实际执行都会与训练不一致。专用入口 `scripts/val_mujoco_g1_armhack_stand.sh` 在现有 `scripts/sim2sim_g1_amp_mujoco.sh` 基础上增加以下适配：

- 使用 `s3_g1_29dof` XML、`0.002 s` physics step、10 倍 decimation 和 50 Hz policy；
- 速度命令固定为 `[0, 0, 0]`，只让 policy 控制腰腿 15 维；
- 初始化时直接把双臂写到测试 CSV 第一帧，之后按时间插值并覆盖双臂 14 维，不产生默认姿态接入跳变；
- 把覆盖后的完整 29 维 action 写回 observation 的 `last_action`；
- `PAYLOAD_KG` 分别加到左右 `wrist_yaw_link`，同时按最终/原始质量比例缩放该 link 惯量；
- 自动生成 checkpoint 短 SHA 区分的 Markdown、JSON、逐帧 CSV 和 torso 世界系 6D 曲线。

本机已补齐并验证 `/home/user/anaconda3/envs/gmr`：MuJoCo `3.8.0`、Torch `2.11.0+cu130`、PyYAML `6.0.3`。新开终端做完整 headless 测试时可整段执行：

```bash
source /home/user/anaconda3/etc/profile.d/conda.sh
conda activate gmr
cd /home/user/Workspace/Humanoid/Locomotion/G1-Locomotion/legged_lab

export STAND_RUN='2026-07-15_14-12-54_armhack_stand_randomized_payload_from_model2999_full_20260715'
export STAND_CKPT="$PWD/ArmHack Checkpoints/StandPerturb/${STAND_RUN}/model_2999.pt"
test -f "$STAND_CKPT" || { echo "错误：checkpoint 不存在" >&2; exit 1; }
sha256sum "$STAND_CKPT"

# 无额外负载，完整 208.08 s / schema v5
CHECKPOINT="$STAND_CKPT" MODE=all PAYLOAD_KG=0 \
USE_GLFW=False REAL_TIME=False \
bash scripts/val_mujoco_g1_armhack_stand.sh

# 左右腕末端各附加 1 kg，完整 208.08 s / schema v5
CHECKPOINT="$STAND_CKPT" MODE=all PAYLOAD_KG=1.0 \
USE_GLFW=False REAL_TIME=False \
bash scripts/val_mujoco_g1_armhack_stand.sh
```

第一次运行会自动用 `env_isaaclab` 从 checkpoint 导出 TorchScript 和 ONNX；以后检测到三个导出文件齐全会直接复用。需要强制重导出时加 `FORCE_EXPORT=True`。导出文件位于：

```text
<checkpoint目录>/MuJoCo Export/StandArmOnly/
  policy.pt
  policy.onnx
  policy.deploy.json
```

MuJoCo GUI 中顺序看完整数据集：

```bash
CHECKPOINT="$STAND_CKPT" MODE=all PAYLOAD_KG=0 \
USE_GLFW=True REAL_TIME=True \
bash scripts/val_mujoco_g1_armhack_stand.sh
```

完整 GUI 需要约 208 秒。更适合快速检查的是逐项 5 秒 `1.0x` 轨迹：

```bash
CHECKPOINT="$STAND_CKPT" MODE=randomized_trajectory ITEM=1 PAYLOAD_KG=1.0 \
USE_GLFW=True REAL_TIME=True \
bash scripts/val_mujoco_g1_armhack_stand.sh
```

`MODE` 与 IsaacLab 确定性入口保持一致，支持 `all`、六类集合，`representative_pose`/`synthesized_pose`/`randomized_pose`、三类 `*_trajectory` 的单项模式，以及无需 `ITEM` 的 `down_to_horizontal` 专项模式；单项模式必须同时指定合法 `ITEM`。若在 GUI 中途手工关闭窗口，rollout 会提前结束，并在报告中得到 `complete_csv_playback=False`，不能当作完整通过。

报告保存到：

```text
<checkpoint目录>/Test Reports/StandArmOnlyMuJoCo/
  model_2999_877e929d516c__mujoco__all__payload_0kg.md
  model_2999_877e929d516c__mujoco__all__payload_0kg.json
  model_2999_877e929d516c__mujoco__all__payload_0kg__trace.csv
  model_2999_877e929d516c__mujoco__all__payload_0kg__torso_world_6d.png
```

MuJoCo 为了把 CSV 最后一帧 `t=208.08 s` 也执行一次，运行到 `208.082 s`，因此记录 10,405 个控制样本；IsaacLab 原命令的 10,404 step 是停止边界定义不同，不是丢帧。2026-07-16 的真实全量结果如下：

| 每侧负载 | 完整播放 | MuJoCo health | 摔倒 | 最低 root 高度 | max abs roll/pitch | torso 水平位移 RMS/最大值 | torso RPY 位移 RMS/最大值 | 双臂跟踪 MAE/RMS/最大值 |
|---:|---|---|---|---:|---:|---:|---:|---:|
| 0 kg | 是，10,405 样本 | `True` | 无 | 0.78786 m | 0.01353 / 0.06515 rad | 0.01067 / 0.02523 m | 0.06132 / 0.10162 rad | 0.05655 / 0.07024 / 0.21352 rad |
| 1 kg | 是，10,405 样本 | `True` | 无 | 0.78791 m | 0.02895 / 0.13721 rad | 0.03511 / 0.04849 m | 0.12540 / 0.14589 rad | 0.07345 / 0.09001 / 0.25349 rad |

两组都满足“完整播放且 health 全程有效”的 sim2sim 最低通过条件。1 kg 明显增大水平位移、姿态位移和手臂跟踪误差，因此不能把无负载结果外推成带载性能。MuJoCo 与 IsaacLab 使用不同接触、执行器和刚体模型，两组数值不能混成一份统计，也不能替代真实机器人测试。

### 1.1.10 导出 ONNX 与真机部署接口

当前第二阶段正式模型已导出为：

```text
deployment/armhack_stand/stand.onnx
SHA-256: 0801f6463211503b69a231855f7488180713eef8b9c1705d6dce818d7605b8ce

deployment/armhack_stand/stand.deploy.json
```

新终端重新导出命令：

```bash
source /home/user/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
cd /home/user/Workspace/Humanoid/Locomotion/G1-Locomotion/legged_lab

python scripts/rsl_rl/export_amp_actor_to_onnx.py \
  --robot g1 \
  --checkpoint "ArmHack Checkpoints/StandPerturb/2026-07-15_14-12-54_armhack_stand_randomized_payload_from_model2999_full_20260715/model_2999.pt" \
  --output deployment/armhack_stand/stand.onnx \
  --metadata deployment/armhack_stand/stand.deploy.json \
  --default-command 0 0 0
```

`--default-command 0 0 0` 不能省略：通用 G1 exporter 的默认 metadata 面向行走，Stand 输入的 `obs[6:9]` 必须固定为零。完整真机接口、96 维输入切片、29 维关节顺序、外部 14 维双臂目标覆盖、组合后 `last_action` 回写、Unitree motor 顺序和手指/夹爪独立控制见：

```text
docs/armhack_stand_real_deployment.md
```

当前已新增 Stand 专用真机入口 `../scripts/deploy_real_g1_armhack_stand.sh` 和控制器 `../unitree_sim2sim2real/deploy/deploy_real/deploy_real_g1_armhack_stand.py`。它复用通用 96 维 observation/29 维 actor，但在每帧把 14 维双臂位置覆盖成预设 minimum-jerk 轨迹，并把经过关节限位和目标速度限幅后的实际组合 action 回写到下一帧 `obs[67:96]`；Stand 速度命令被强制为 `[0,0,0]`。

新终端先做完全离线自检；该命令不会初始化 DDS，也不会连接机器人：

```bash
cd /home/user/Workspace/Humanoid/Locomotion/G1-Locomotion

DRY_RUN=True NET=enp11s0 \
bash scripts/deploy_real_g1_armhack_stand.sh
```

真机终端必须显式指定已经安装 `unitree_sdk2py==1.0.1`、`cyclonedds==0.10.2`、`onnxruntime`、`torch`、`numpy` 和 `PyYAML` 的 Python；本机现有 `env_isaaclab`/`gmr` 只够导出和离线自检，当前未安装 Unitree DDS 依赖，不能把它们直接当真机环境。真机命令为：

```bash
cd /home/user/Workspace/Humanoid/Locomotion/G1-Locomotion

UNITREE_PYTHON=/path/to/unitree-rl/bin/python \
CONFIRM_REAL_ROBOT=I_UNDERSTAND \
NET=enp11s0 \
bash scripts/deploy_real_g1_armhack_stand.sh
```

程序先用当前关节位置发送 LowCmd 保持，再调用 `MotionSwitcher.ReleaseMode()` 并确认高层 motion mode 已清空，即进入本入口所需的低层调试控制状态。第一次 `ENTER` 后，用 5 s minimum-jerk 移动到 P0；第二次 `ENTER` 才启动 50 Hz policy。运行期间 `SPACE` 按 `P0→P1→P2→P3→P0` 循环切换双臂姿态，每次用 4 s minimum-jerk 生成轨迹；`q`、`Ctrl-C` 或遥控器 `Select` 退出并发送阻尼。预设数据位于 `Reference Data/ArmHack/StandPerturb/RealDeployment/stand_arm_presets.json`，只包含双臂 14DoF，不包含腰腿。完整安全顺序、关节表和参数说明见 `docs/armhack_stand_real_deployment.md`。此入口目前完成的是离线模型/数据/状态机检查，不代表已完成真机吊架验收。

## 1.2 Stand 范围内随机姿态、插值轨迹和末端负载续训

本节是当前第二阶段 Stand 训练的权威入口。它不从 `model_9996.pt` 重新开始，而是从已经完成原 CSV `1.0x` 课程的以下正式模型继续：

```text
ArmHack Checkpoints/StandPerturb/
2026-07-14_20-34-20_armhack_stand_curriculum_1x_from_model9996_full_20260714/model_2999.pt
SHA-256: 2c87cc2cc3706c1024594d14d85a34e7bf468b54f6b66e49b6155ef72a2dbd16
size: 14,825,781 bytes
```

### 1.2.1 原数据范围

范围分析使用原 405 s CSV 中训练真正可达的前 20,109 帧，末端时刻为 `404.656061 s`；这与 `csv_end_margin_s=0.25` 一致。下表是每个关节的观测最小/最大值以及用于轨迹限速的原数据速度 P99：

| 关节 | min (rad) | max (rad) | 速度 P99 (rad/s) |
|---|---:|---:|---:|
| `left_shoulder_pitch_joint` | -0.832233 | 0.280335 | 0.387553 |
| `left_shoulder_roll_joint` | 0.437796 | 1.064786 | 0.184981 |
| `left_shoulder_yaw_joint` | -0.665951 | 0.124121 | 0.211684 |
| `left_elbow_joint` | -0.938305 | 0.476205 | 0.498593 |
| `left_wrist_roll_joint` | 0.248385 | 0.867490 | 0.157251 |
| `left_wrist_pitch_joint` | -0.252580 | 0.747228 | 0.361817 |
| `left_wrist_yaw_joint` | -0.969488 | 0.047757 | 0.477206 |
| `right_shoulder_pitch_joint` | -0.946107 | 0.280790 | 0.422175 |
| `right_shoulder_roll_joint` | -0.998130 | -0.430222 | 0.216559 |
| `right_shoulder_yaw_joint` | -0.124564 | 0.691046 | 0.213857 |
| `right_elbow_joint` | -0.594933 | 0.491641 | 0.502645 |
| `right_wrist_roll_joint` | -0.867514 | -0.249536 | 0.183295 |
| `right_wrist_pitch_joint` | -0.252568 | 0.612190 | 0.368062 |
| `right_wrist_yaw_joint` | -0.048093 | 1.034754 | 0.480617 |

对 P99 低于 `0.20 rad/s` 的关节，插值限速下限取 `0.20 rad/s`，避免个别原轨迹几乎静止时把所有新过渡无限拉长。

### 1.2.2 姿态与轨迹怎样生成

训练姿态库位于：

```text
Reference Data/ArmHack/StandPerturb/RandomizedTraining/
random_arm_pose_bank_seed20260715.json
```

共有 512 个完整 14-DoF 双臂姿态：64 个用最远点覆盖选出的实测锚点，448 个由 2–4 个实测姿态的 Dirichlet 权重凸组合得到。不对 14 个关节各自做独立均匀采样，因此能保留原数据中左右臂以及同一手臂各关节之间的配合关系。凸组合也保证所有新姿态不超出上表的分量范围。

每个 episode reset 时为每个并行环境随机选起点和终点。起点会直接写入 14 个实际手臂关节，所以首帧不会从默认姿态跳变。动态阶段使用五次 minimum-jerk 曲线
`10u^3-15u^4+6u^5` 从起点插值到终点。名义时长为 2–6 s；环境还会用五次曲线峰值速度 `1.875*|delta_q|/duration` 检查每个关节，必要时自动延长时长。到达终点后再采样新终点，所以 20 s episode 内可以出现多段连续平滑双臂运动。

重建和校验命令：

```bash
python scripts/tools/build_armhack_stand_randomized_training_data.py
python scripts/tools/build_armhack_stand_visualization_suite.py
python scripts/tools/check_armhack_reference_data.py --stand-only
```

### 1.2.3 末端负载 domain randomization

新任务在 Isaac Lab startup event 中调用 `randomize_rigid_body_mass`，对 `left_wrist_yaw_link` 和 `right_wrist_yaw_link` 分别附加均匀分布 `0–1.0 kg` 质量，`operation=add`，并重新计算惯量。这两个 link 是当前 29-DoF 可控手臂链的末端，因此这里的含义是“腕末端等效携带负载”。负载值、轨迹终点、phase 和未来目标都没有加入 policy observation；policy 仍只能根据当前机器人状态反馈补偿。

### 1.2.4 完整续训命令

在新终端中整段执行：

```bash
source /home/user/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
cd /home/user/Workspace/Humanoid/Locomotion/G1-Locomotion/legged_lab

python scripts/tools/check_armhack_reference_data.py --stand-only
bash scripts/train_g1_armhack_stand_randomized_payload.sh
```

默认配置为 4096 环境、3000 iteration、20 s episode、每侧 `0–1.0 kg` 附加质量；从上述正式 `model_2999.pt` policy-only 初始化，保留 `0.003` 冻结基线 KL。课程为：

| iteration | 阶段 | 双臂输入 |
|---:|---|---|
| `0..499` | 新随机姿态静止 | reset 选姿态，整个 episode 保持 |
| `500..1499` | 逐步加速 | minimum-jerk 轨迹从 `0` 线性升至 `1.0x` |
| `1500..2999` | 全速适应 | 持续 `1.0x` 姿态插值轨迹 |

如需显式写出参数：

```bash
NUM_ENVS=4096 \
MAX_ITERATIONS=3000 \
STATIC_ITERATIONS=500 \
RAMP_ITERATIONS=1000 \
FINAL_MOTION_SCALE=1.0 \
TRANSITION_MIN_S=2.0 \
TRANSITION_MAX_S=6.0 \
PAYLOAD_MAX_KG=1.0 \
RUN_NAME=armhack_stand_randomized_poses_payload_from_model2999 \
QUIET_TERMINAL=False \
bash scripts/train_g1_armhack_stand_randomized_payload.sh
```

### 1.2.5 训练 smoke 和已验证结果

最短真实 Isaac/PPO smoke：

```bash
NUM_ENVS=16 \
MAX_ITERATIONS=1 \
STATIC_ITERATIONS=0 \
RAMP_ITERATIONS=0 \
FINAL_MOTION_SCALE=1.0 \
PAYLOAD_MAX_KG=1.0 \
RUN_NAME=armhack_stand_randomized_payload_from_model2999_smoke \
HEADLESS=True \
QUIET_TERMINAL=False \
bash scripts/train_g1_armhack_stand_randomized_payload.sh
```

2026-07-15 已以同配置的 `..._smoke_v2_20260715` 真实执行成功：16 环境共收集 384 步，打印 `Learning iteration 0/1`，加载 512 姿态，`random_motion_scale=1.0`、`curriculum_stage=2`、平均插值时长 `4.1725 s`，完成 PPO 更新并保存：

```text
logs/monitoring/armhack_stand_randomized_payload_from_model2999_smoke_v2_20260715.log
ArmHack Checkpoints/StandPerturb/
2026-07-15_13-22-56_armhack_stand_randomized_payload_from_model2999_smoke_v2_20260715/model_0.pt
```

这只证明新数据、负载事件、rollout、PPO 和保存链无报错，不代表 `model_0.pt` 已学会新分布。随后已用默认正式配置完成 `4096 env × 3000 iteration`：

```text
run: 2026-07-15_14-12-54_armhack_stand_randomized_payload_from_model2999_full_20260715
checkpoint: ArmHack Checkpoints/StandPerturb/<run>/model_2999.pt
SHA-256: 877e929d516cffe9131cc235477ceef4b226ec69e41c0f1c23e48816cfa28821
```

最终 iteration 2999 的在线统计为 mean episode length `1000/1000`、time-out `99.95%`、base-height termination `0`、bad-orientation termination `0.05%`，课程已处于 `random_motion_scale=1.0`、`curriculum_stage=2`，512 姿态库的平均插值过渡时长为 `4.4112 s`。

### 1.2.6 新姿态/轨迹测试与可视化

测试不在运行时随机抽样，而是从 512 姿态库固定选出 8 个覆盖姿态，并固定 6 条 5 s minimum-jerk 轨迹，保证每次回放可复现。新终端先定义被测模型：

```bash
source /home/user/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
cd /home/user/Workspace/Humanoid/Locomotion/G1-Locomotion/legged_lab
export STAND_CKPT="$PWD/ArmHack Checkpoints/StandPerturb/2026-07-15_14-12-54_armhack_stand_randomized_payload_from_model2999_full_20260715/model_2999.pt"
test -f "$STAND_CKPT" || { echo "checkpoint 不存在: $STAND_CKPT"; exit 1; }
```

顺序看 8 个姿态，无额外负载：

```bash
CHECKPOINT="$STAND_CKPT" MODE=randomized_poses PAYLOAD_KG=0 \
  bash scripts/vis_g1_armhack_stand_eval.sh
```

顺序看 6 条 `1.0x` 插值轨迹，每侧腕部末端固定附加 1 kg：

```bash
CHECKPOINT="$STAND_CKPT" MODE=randomized_trajectories PAYLOAD_KG=1.0 \
  bash scripts/vis_g1_armhack_stand_eval.sh
```

逐项定位：

```bash
CHECKPOINT="$STAND_CKPT" MODE=randomized_pose ITEM=1 PAYLOAD_KG=1.0 \
  bash scripts/vis_g1_armhack_stand_eval.sh       # ITEM=1..8

CHECKPOINT="$STAND_CKPT" MODE=randomized_trajectory ITEM=1 PAYLOAD_KG=1.0 \
  bash scripts/vis_g1_armhack_stand_eval.sh       # ITEM=1..6
```

无窗口执行链 smoke：

```bash
CHECKPOINT="$STAND_CKPT" MODE=randomized_pose ITEM=1 PAYLOAD_KG=1.0 \
HEADLESS=True REAL_TIME=False MAX_STEPS=20 \
bash scripts/vis_g1_armhack_stand_eval.sh
```

逐项命令生成的文件也遵循新命名规则，例如 `model_2999_877e929d516c__randomized_pose_item1__payload_1kg.md` 及同名 `__torso_world_6d.png`。报告明确记录 checkpoint 完整 SHA 和左右腕末端固定负载；当前 `STAND_CKPT` 已指向第二阶段新正式模型。

## 2. Reference Data 目录

两个任务的姿态参考数据统一保存在：

```text
legged_lab/Reference Data/ArmHack/
├── README.md
├── StandPerturb/
│   ├── raw/
│   │   └── g1_full_body_motion_sdk_50hz.csv
│   ├── g1_arm_trajectory_named_50hz.csv
│   ├── RandomizedTraining/
│   │   └── random_arm_pose_bank_seed20260715.json
│   └── TestData/ArmOnly/
│       ├── poses/{representative,synthesized,randomized}/
│       ├── trajectories/{representative,synthesized,randomized}/
│       ├── sequences/
│       ├── special/arms_down_to_forward_horizontal_20s_50hz.csv
│       └── manifest.json
└── WalkPerturbFinetune/
    ├── g1_arm_pose_set.json
    └── nav2_cmd_vel_raw_success.csv
```

代码中的相对根路径是：

```python
Path("Reference Data") / "ArmHack"
```

运行时以 `legged_lab` 项目目录为锚点解析，因此从不同用户名或不同仓库父目录运行都不会依赖固定主目录。

### 2.1 Stand 原始数据和训练数据

原始文件来自：

```text
/home/user/Workspace/whole_body_joints_20260708_143133.csv
```

仓库保存副本：

```text
Reference Data/ArmHack/StandPerturb/raw/g1_full_body_motion_sdk_50hz.csv
```

原始文件共 20,122 帧，SHA-256：

```text
b43256da27b11a593fc244ab2dd7fb899490a575d7749ed858ac342e3a208c50
```

原始 `q0..q28` 顺序是：

```text
q0..q14  : 左右腿和腰部
q15..q21 : 左臂 shoulder pitch/roll/yaw、elbow、wrist roll/pitch/yaw
q22..q28 : 右臂 shoulder pitch/roll/yaw、elbow、wrist roll/pitch/yaw
```

源 CSV 有 1,346 行自然语言字段包含未引用逗号。为避免把容错逻辑留在训练关键路径中，转换脚本：

```text
scripts/tools/extract_armhack_stand_arm_csv.py
```

会读取每行最后 29 个 SDK q 值，只提取 `q15..q28`，生成严格的具名训练 CSV：

```text
Reference Data/ArmHack/StandPerturb/g1_arm_trajectory_named_50hz.csv
```

该文件仍为 20,122 帧，只含 15 列：`time_s + 14 个具名手臂关节`，SHA-256：

```text
afe3819937ecfa19fae835b8cc77038378ec40a821acd0fdf2feef0054583601
```

重新生成命令：

```bash
cd /home/user/Workspace/Humanoid/Locomotion/G1-Locomotion
conda run -n env_isaaclab \
  python legged_lab/scripts/tools/extract_armhack_stand_arm_csv.py
```

Stand 训练只读取规范的 14 关节文件，不会把 CSV 下肢姿态覆盖到机器人上。

2026-07-14 对规范训练 CSV 做了只读数值审计：

```text
帧数：20,122
持续时间：404.917792 s
相邻时间间隔中位数：0.02010 s（约 49.75 Hz）
相邻帧最大单关节变化：0.029313 rad
全轨迹最大关节速度：1.4569 rad/s（左肘）
全轨迹最大关节加速度：25.5020 rad/s²（左肘）
```

轨迹内部大部分帧是连续的。历史实现从机器人默认手臂姿态直接切到随机目标：随机起点的“最大单关节差值”中位数、P95、最大值分别约为 `1.446 / 1.466 / 1.908 rad`，L2 差值中位数约 `2.477 rad`；即使固定从第 0 帧开始，最大单关节差值仍为 `1.440 rad`。当前实现已取消这次接入跳变：reset 时直接把仿真手臂关节状态初始化到所抽取的 CSV 姿态，并将手臂关节速度置零。

### 2.2 Walk 三组双臂姿态

文件：

```text
Reference Data/ArmHack/WalkPerturbFinetune/g1_arm_pose_set.json
```

每只手臂的 7 关节顺序为：

```text
shoulder_pitch, shoulder_roll, shoulder_yaw, elbow,
wrist_roll, wrist_pitch, wrist_yaw
```

数据包含：

```text
pos1_back
left  = [0.91, 0.52, 0.11, 0.01, -0.12, -1.03, 0.01]
right = [0.91, -0.52, -0.11, 0.01, 0.12, -1.03, -0.01]

pos2_down
left  = [0.2504, 0.2650, -0.0919, 0.8356, 0.0031, 0.0104, -0.0102]
right = [0.2504, -0.2650, 0.0919, 0.8356, -0.0031, 0.0104, 0.0102]

pos3_front
left  = [0.27, 0.79, -0.22, -0.49, 0.85, 0.40, 0.05]
right = [0.27, -0.79, 0.22, -0.49, -0.85, 0.40, -0.05]
```

`reference_data.py` 会验证 JSON 单位为弧度、关节顺序正确、每侧恰好 7 个值、pose 名不重复、所有值有限，然后返回“名称 + 左右交错 14 维值”。专用脚本默认 `POSE_NAME=pos2_down`，也支持 `pos1_back`、`pos3_front` 和 `random`。每个 episode 内姿态固定不变。

### 2.3 Walk 的 Nav2 速度分布

动态任务使用：

```text
Reference Data/ArmHack/WalkPerturbFinetune/nav2_cmd_vel_raw_success.csv
```

本机搜索不到该文件后，从 HEC-5090 的下列位置下载：

```text
/home/hecggdz/workspace-zwd/legged_lab/nav2_loopback_actual/
actual_raw_success/all_cmd_vel_success.csv
```

数据契约与检查结果：

```text
行数：331,010
连续轨迹组：445
列数：17
vx：[-0.2, 0.6] m/s
vy：[-0.3, 0.3] m/s
wz：[-0.5187280178070068, 0.6] rad/s
augmentation：none
SHA-256：76a4516588b855351eb3eb8c2da26e291603876c1a4a1b9c7bacd77a53807b5a
```

它是实际 Nav2 成功轨迹的原始分布，不是离线均匀采样，也不是离线镜像文件。已在 HEC-5090 的整个 `workspace-zwd` 中搜索，远端只有这份 `all_cmd_vel_success.csv`，没有 `all_cmd_vel_augmented.csv`。当前 loader 会在内存中由每个 `none` 组生成 `(vx,-vy,-wz)` 的 `mirror_lr` 组，以恢复 Nav2 Stage-4 的左右分布；源 CSV 不改写。CSV 约 83 MiB，已加入 `.gitignore`；GitHub clone 后需要单独放回上述相对位置。

## 3. 两个任务的实现

### 3.1 共同 action 路径

policy 输入仍是原 G1 AMP 的 96 维：

```text
base_ang_vel 3 + projected_gravity 3 + velocity_command 3
+ joint_pos 29 + joint_vel 29 + previous_action 29 = 96
```

actor 输出保持 29 维：

```text
policy action 29
  -> 保留 15 个腿部和腰部 action
  -> 用 CSV 或 JSON pose 覆盖 14 个手臂 action
  -> JointPositionAction
```

因此网络和部署接口维度不变，已有 G1/Nav2 policy checkpoint 可以继续作为初始化策略。

### 3.2 StandPerturb

配置：

```text
速度命令：vx=0、vy=0、wz=0
手臂来源：g1_arm_trajectory_named_50hz.csv
CSV 起点：每个环境 reset 时随机选择，并直接初始化 14 个手臂关节到该姿态
CSV 采样：连续时间相位 + 相邻帧线性插值
CSV 循环：关闭，结束后保持最后一帧
episode：20 s
AMP style reward：0.0
task/style lerp：1.0
RSI：关闭
初始根位置、yaw 和六维根速度：全为 0
外部 push：关闭
```

课程按全局训练步数自动切换；每个 PPO iteration 收集 24 个 control step：

| 阶段 | 默认 iteration | 手臂行为 |
|---|---:|---|
| 0：随机姿态静止 | `0..499` | reset 随机抽取任意 CSV 相位，整段 episode 保持该姿态 |
| 1：全速渐入 | `500..1499` | 轨迹速度从 `0` 连续线性增至原始 `1.0x` |
| 2：全速适应 | `1500..2999` | 持续按原始 `1.0x` 速度运动 |

所有阶段都从随机相位开始。reset 后直接把手臂关节状态写到该相位、关节速度写为 0，所以不存在从默认姿态到随机目标的首步跳变；进入运动阶段后，相位每个 control step 按当前倍率连续推进。非循环轨迹按最终 `1.0x` 和 20 s episode 计算最大随机起点，保证 episode 内不会因到达 CSV 末尾而提前停住。

行走、步态、摆腿和 directional 奖励被关闭。Stand 奖励全部显式重建，真实 Reward Manager 已确认 21 项有效：

| 奖励 | 权重 | 目的 |
|---|---:|---|
| `alive` | `+1.0` | 消除“早摔少累计代价”的错误激励 |
| `track_torso_lin_vel_xy_exp` | `+1.5` | 躯干水平速度保持为 0 |
| `track_torso_yaw_rate_exp` | `+0.75` | 躯干 yaw 角速度保持为 0 |
| `double_support` | `+0.25` | 鼓励双脚持续支撑 |
| `flat_orientation_l2` | `-1.0` | 根节点直立 |
| `torso_roll_pitch_l2` | `-3.0` | 躯干 roll/pitch 稳定 |
| `torso_ang_vel_xy_l2` | `-0.15` | 抑制躯干水平角速度 |
| `torso_vertical_velocity_l2` | `-0.30` | 抑制躯干上下运动 |
| `torso_height_band_l2` | `-0.60` | 躯干保持 `0.84 m` 高度带 |
| `torso_specific_force_xy_l2` | `-0.01` | 抑制水平加速度响应 |
| `root_xy_position_l2` | `-1.0` | 抑制相对环境原点的水平漂移 |
| `lin_vel_z_l2` / `ang_vel_xy_l2` | `-0.30` / `-0.10` | 根状态稳定 |
| 下肢 `torque` / `acc` / `action-rate` | `-2e-6` / `-1e-7` / `-0.005` | 只正则化可控的腿和腰 |
| 踝限位 / 髋偏差 / 腰偏差 | `-1.0` / `-0.10` / `-0.12` | 关节安全和自然站姿 |
| `feet_slide` | `-0.25` | 抑制脚滑 |
| `termination_penalty` | `-200.0` | 惩罚非 timeout 摔倒 |

policy 输入仍为原 96 维，没有添加轨迹相位、未来手臂目标、目标速度或 look-ahead。它只能通过当前 `joint_pos/joint_vel` 感知已经发生的手臂状态变化，满足“不能知道未来手臂怎么运动”的约束。

#### 3.2.1 StandRandomizedPayload 续训实现

新配置 `g1_stand_randomized_payload_env_cfg.py` 继承上述 Stand 的零速度命令、21 项奖励和 termination，仅替换外部手臂扰动分布并加入末端负载：

```text
task: LeggedLab-Isaac-AMP-G1-StandRandomizedPayload-v0
arm source: random_pose_trajectory
pose bank: RandomizedTraining/random_arm_pose_bank_seed20260715.json
reset: 实际双臂 q 直接写成随机起点，dq=0
trajectory: 两个完整 14-DoF 姿态间五次 minimum-jerk 插值
nominal duration: 2–6 s，速度不安全时自动延长
payload: left/right_wrist_yaw_link 各自附加 U(0,1.0) kg，重算惯量
policy observation: 不包含姿态库 id、终点、phase、duration、未来目标或 payload
```

环境记录 `ArmHack/random_motion_scale`、`ArmHack/random_pose_bank_size`、`ArmHack/random_transition_duration_mean_s` 和 `ArmHack/curriculum_stage`，用于确认课程真正进入了对应阶段。Play 配置保留同一负载 event，可视化入口把质量区间改成 `[PAYLOAD_KG, PAYLOAD_KG]`，从而得到可复现的固定负载测试。

### 3.3 WalkPerturbFinetune 当前实现

Walk 使用独立 `G1WalkPerturbAmpEnv`，不修改共享的 Stand reset 路径。reset 时按名称选择 pose，并在首个 policy observation 前同步实际手臂 `q/dq`、关节 position target、ActionManager 当前/上一动作和 JointPositionAction 的 raw/processed buffer。日志项 `ArmHack/walk_pose_init_max_error_rad` 用于检查写入误差，`ArmHack/walk_pose_index_mean` 用于确认所选姿态。

速度命令使用 `Nav2RecordedVelocityCommandCfg`。采样单位不是互相独立的 `vx/vy/wz` 行，而是按 planner、controller、scenario、goal 分组后截取连续窗口：

```text
scenario family：complex_turn
数据窗口：4.0 s
数据采样周期：0.05 s
命令缩放：(0.85, 0.75, 0.75)
命令裁剪：vx [-0.20, 0.60]，vy [-0.30, 0.30]，wz [-0.60, 0.60]
平滑时间常数：0.30 s
最大线加速度：0.60 m/s²
最大 yaw 加速度：0.80 rad/s²
轨迹组均匀采样：开启
controller 权重：mppi 1.5，dwb 1.0
augmentation_filter：none,mirror_lr（由 raw none 在线生成镜像）
```

相对数据路径由 command loader 以 `legged_lab` 项目目录为锚点解析。源 CSV 只有 raw `none` 样本；loader 在内存中生成 `(vx,-vy,-wz)` 镜像组，不改写源文件。

Walk 现在直接继承 `G1AmpNav2FinetuneEnvCfg`，并显式锁定保存的 Nav2 Stage-4 command 与奖励。原先只修改 `ranges`、对 recorded dataset 无效的两个 command curriculum 已删除。

#### 3.3.1 当前 Walk 训练奖励

当前源码 `g1_walk_perturb_env_cfg.py` 与正确 model9996 smoke run 保存的
`params/env.yaml` 一致，共有 18 个非零环境奖励项：

| 类别 | Reward Manager 名称 | 权重/参数 | 实际作用和范围 |
|---|---|---:|---|
| 根速度跟踪 | `track_lin_vel_xy_exp` | `+1.8`，`std=0.30` | 跟踪 Nav2 的 `vx/vy` |
| 根速度跟踪 | `track_ang_vel_z_exp` | `+1.5`，`std=0.35` | 跟踪 Nav2 的 yaw 角速度 |
| 躯干速度跟踪 | `track_torso_lin_vel_xy_exp` | `+1.0`，`std=0.30` | 让 `torso_link` 的水平速度跟随命令 |
| 躯干速度跟踪 | `track_torso_yaw_rate_exp` | `+0.65`，`std=0.35` | 让 `torso_link` 的 yaw 角速度跟随命令 |
| 直立姿态 | `flat_orientation_l2` | `-1.0` | 惩罚根节点相对重力方向的倾斜 |
| 躯干姿态 | `torso_roll_pitch_l2` | `-0.04` | 抑制躯干 roll/pitch，但保持 Nav2 原始弱约束，不额外放大 |
| 躯干竖直运动 | `torso_vertical_velocity_l2` | `-0.01` | 抑制 `torso_link` 上下抖动 |
| 根竖直运动 | `lin_vel_z_l2` | `-0.20` | 抑制根节点竖直速度 |
| 根水平转动 | `ang_vel_xy_l2` | `-0.05` | 抑制根节点 roll/pitch 角速度 |
| 控制正则 | `dof_torques_l2` | `-2e-6` | 只统计可控的 15 个腿和腰关节 |
| 控制正则 | `dof_acc_l2` | `-1e-7` | 只统计可控的 15 个腿和腰关节 |
| 控制平滑 | `action_rate_l2` | `-0.006` | 只统计腿腰对应的 15 个 action index |
| 关节安全 | `dof_pos_limits` | `-1.0` | 惩罚左右踝 pitch/roll 接近限位 |
| 自然姿态 | `joint_deviation_hip` | `-0.10` | 惩罚左右髋 yaw/roll 偏离默认姿态 |
| 自然姿态 | `joint_deviation_waist` | `-0.10` | 惩罚三个腰关节偏离默认姿态 |
| 步态 | `feet_air_time` | `+0.50`，阈值 `0.40 s` | 鼓励移动命令下合理的单脚腾空时间 |
| 足底稳定 | `feet_slide` | `-0.10` | 惩罚接触地面时的脚底滑移 |
| 摔倒 | `termination_penalty` | `-200.0` | 惩罚 `base_height`、`bad_orientation` 等非 timeout 终止 |

以下项在当前 Walk 中不参与 return：

- `alive`、`double_support`、`root_xy_position_l2` 为 `None`；
- `torso_height_band_l2`、`joint_deviation_arms`、`arm_style_prior` 为 `None`；
- `feet_swing_clearance_band_l2`、`gait_timing_symmetry_l1` 和全部 directional/backward shaping 为 `None`；
- `torso_ang_vel_xy_l2`、`torso_lateral_vel_cmd_l2`、`torso_specific_force_xy_l2` 的权重为 `0`。

训练脚本还固定 `style_reward_scale=0`、`task_style_lerp=1.0`，因此 PPO return 完全来自上表的
task reward，AMP discriminator 不提供 style reward。`baseline KL=0.003`、mirror loss `0.1` 和
entropy coefficient `0.002` 属于优化 loss 的正则项，不是环境奖励。双臂 14 维 action 被环境覆盖，
所以没有加入策略无法控制的手臂目标误差奖励；torque、acc、action-rate 与 discriminator 都只看
可控的 15 个腿腰维度。

#### 3.3.2 当前 Walk 怎样测试和可视化

当前验证分为四层，不能把其中任意一层当作全部性能验收：

1. 数据与静态检查：校验三组姿态 JSON、Nav2 CSV、任务注册、奖励关键权重和脚本语法；
2. 训练 smoke：用 `2 env × 1 iteration` 确认 Isaac Sim、model9996 加载、PPO 更新和 checkpoint 保存链；
3. checkpoint headless 回放：对同一 checkpoint 分别固定 `pos1_back`、`pos2_down`、`pos3_front`，每个姿态用 `1 env × 1000 control step` 跑 20 s Nav2 连续速度窗口；
4. 可视化：同一 `Play` task 可以打开 Isaac Sim GUI 实时观察，或 headless 录制 1000-step 视频；训练曲线另用 TensorBoard 查看。

具体可复制命令见第 8.4 节和第 9.7 节。当前通用 `Play` 会输出速度跟踪 MAE、综合 score 和
Important Metrics，但尚未为 Nav2 Walk 生成逐姿态独立 Markdown 报告，也不会在通用摘要中单独
打印 Walk termination 分类。因此当前测试仍需人工观察 reset；严格验收下一步应增加固定
seed/固定 command window 的 Walk 专用报告，并在相同分布下成对比较原 model9996 与训练后模型。

## 4. 代码位置

| 作用 | 路径 |
|---|---|
| Reference Data 路径和 JSON 校验 | `source/legged_lab/legged_lab/tasks/locomotion/amp/config/g1_perturb/reference_data.py` |
| 29DoF action 覆盖、CSV/随机姿态库读取、minimum-jerk 轨迹状态机 | `source/legged_lab/legged_lab/envs/g1_perturb_env.py` |
| Walk 具名 pose、无跳变 reset 与 action-history 同步 | `source/legged_lab/legged_lab/envs/g1_walk_perturb_env.py` |
| Stand 配置 | `source/legged_lab/legged_lab/tasks/locomotion/amp/config/g1_perturb/g1_stand_perturb_env_cfg.py` |
| Stand 随机姿态+末端负载配置 | `source/legged_lab/legged_lab/tasks/locomotion/amp/config/g1_perturb/g1_stand_randomized_payload_env_cfg.py` |
| Walk 配置 | `source/legged_lab/legged_lab/tasks/locomotion/amp/config/g1_perturb/g1_walk_perturb_env_cfg.py` |
| Gym task 注册 | `source/legged_lab/legged_lab/tasks/locomotion/amp/config/g1_perturb/__init__.py` |
| Runner 配置 | `source/legged_lab/legged_lab/tasks/locomotion/amp/config/g1_perturb/agents/rsl_rl_ppo_cfg.py` |
| Nav2 录制命令窗口加载器 | `source/legged_lab/legged_lab/tasks/locomotion/amp/mdp/commands/nav2_recorded_velocity_command.py` |
| 全身 CSV 转 14 手臂 CSV | `scripts/tools/extract_armhack_stand_arm_csv.py` |
| Stand 512 姿态训练库构建 | `scripts/tools/build_armhack_stand_randomized_training_data.py` |
| Stand schema v5 确定性姿态/轨迹构建 | `scripts/tools/build_armhack_stand_visualization_suite.py` |
| ArmHack 数据完整性预检 | `scripts/tools/check_armhack_reference_data.py` |
| CSV 全身回放 | `scripts/tools/visualize_g1_csv_full_body_motion.py` |
| Stand 专用分阶段训练脚本 | `scripts/train_g1_armhack_stand.sh` |
| Stand 从 model2999 随机姿态+负载续训 | `scripts/train_g1_armhack_stand_randomized_payload.sh` |
| Stand 固定姿态/轨迹/负载回放 | `scripts/vis_g1_armhack_stand_eval.sh` |
| Stand MuJoCo 导出、测试与 GUI 入口 | `scripts/val_mujoco_g1_armhack_stand.sh` |
| Stand MuJoCo 双臂覆盖、负载和报告适配器 | `../unitree_sim2sim2real/deploy/deploy_mujoco/armhack_stand.py` |
| G1 MuJoCo 通用 observation/PD/rollout runner | `../unitree_sim2sim2real/deploy/deploy_mujoco/deploy_mujoco_g1_amp.py` |
| G1 MuJoCo 通用 Shell 入口 | `../scripts/sim2sim_g1_amp_mujoco.sh` |
| Walk 专用 S3 locomotion model9996 训练脚本 | `scripts/train_g1_armhack_walk.sh` |
| 通用训练脚本 | `scripts/train_g1_amp.sh` |
| checkpoint 可视化/回放 | `scripts/vis_isaacsim_g1_amp.sh` |
| Python 训练入口 | `scripts/rsl_rl/train.py` |
| policy-only 加载和专用 checkpoint 复制 | `../rsl_rl/rsl_rl/runners/amp_runner.py` |
| 无仿真静态检查 | `source/legged_lab/test/test_g1_perturb_static.py` |

执行链：

```text
train_g1_armhack_stand.sh
  -> 校验 model_9996 SHA-256/大小，计算课程 step，建立恢复用软链接
  -> train_g1_amp.sh
  -> scripts/rsl_rl/train.py
  -> Gym task registry
  -> Stand env cfg + runner cfg
  -> G1PerturbAmpEnv
  -> RslRlVecEnvWrapper
  -> AMPRunner / PPOAMP.learn()

train_g1_armhack_stand_randomized_payload.sh
  -> 校验第一阶段正式 model_2999 SHA-256/大小和 512 姿态库
  -> train_g1_amp.sh -> Gym StandRandomizedPayload task
  -> G1StandRandomizedPayloadEnvCfg + G1PerturbAmpEnv
  -> 姿态库采样 + minimum-jerk 插值 + wrist-yaw link 附加质量
  -> AMPRunner / PPOAMP.learn()

train_g1_armhack_walk.sh
  -> 检查 Stand/重复 Walk 进程，校验 S3、locomotion.onnx、model9996、pose JSON 和 Nav2 CSV
  -> init: policy-only；resume: 完整恢复 Walk run
  -> train_g1_amp.sh -> Gym Walk task
  -> G1WalkPerturbFinetuneEnvCfg + G1WalkPerturbAmpEnv
  -> Nav2RecordedVelocityCommand + fixed named pose
  -> AMPRunner / PPOAMP.learn()
```

## 5. 必须使用的本机环境

进入环境：

```bash
source /home/user/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
cd /home/user/Workspace/Humanoid/Locomotion/G1-Locomotion/legged_lab
```

已验证版本：

```text
Python:     /home/user/anaconda3/envs/env_isaaclab/bin/python
IsaacLab:  0.54.2
rsl-rl-lib: 3.2.0
PyTorch:   2.7.0+cu128
GPU:       NVIDIA GeForce RTX 4090
CUDA:      available
```

不要使用旧的 `isaaclab` 环境：其中 `rsl-rl-lib=2.3.1`，低于训练入口要求的 `3.0.1`。

`train_g1_amp.sh` 的默认环境已修正为 `env_isaaclab`，默认 Conda 根目录为 `${HOME}/anaconda3`。已激活该环境后，无需再手写 `ISAACLAB_PYTHON`。

本机 `pytest` 可执行文件位于 `~/.local/bin`，其自动加载的第三方插件可能与 Conda 环境中的 `packaging` 冲突；这不影响训练入口。执行仓库静态测试时应明确关闭外部插件自动加载：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/user/.local/bin/pytest -q \
  source/legged_lab/test/test_g1_perturb_static.py
```

## 6. 训练命令

### 6.1 Stand 正式分阶段训练

```bash
cd /home/user/Workspace/Humanoid/Locomotion/G1-Locomotion/legged_lab
bash scripts/train_g1_armhack_stand.sh
```

专用脚本默认参数：

```text
BASE_CHECKPOINT=ArmHack Checkpoints/StandPerturb/BaselineModel9996/model_9996.pt
NUM_ENVS=4096
MAX_ITERATIONS=3000
STATIC_ITERATIONS=500
RAMP_ITERATIONS=1000
FINAL_MOTION_SCALE=1.0
RANDOMIZATION_STRENGTH=0
STYLE_REWARD_SCALE=0.0
TASK_STYLE_LERP=1.0
BASELINE_KL_SCALE=0.003
ENTROPY_COEF=0.002
```

脚本会先验证 checkpoint 文件的固定 SHA-256 和 `16,202,421 bytes` 大小，再把基线以软链接暂存到 `logs/rsl_rl/g1_stand_perturb/_armhack_stand_baseline_model9996/`，供 IsaacLab/RSL-RL 的标准恢复逻辑解析。随后强制：

```text
RESUME=True
agent.load_policy_only=True
agent.reset_iteration_on_policy_only_load=True
RSI_ENABLE=False
BASELINE_KL_ENABLE=True
```

也就是说只加载 `model_9996` 的 actor/critic 权重，不继承旧 optimizer、discriminator 或 iteration 计数；新 run 从 iteration 0 开始执行完整课程。冻结的同一基线 policy 用于 KL 正则，防止下肢策略在新奖励初期快速漂移。policy 观测接口仍为 96 维，动作仍为 29 维，因此与当前 Stand 环境兼容。

覆盖课程参数示例：

```bash
RUN_NAME=armhack_stand_curriculum_v2 \
STATIC_ITERATIONS=800 \
RAMP_ITERATIONS=1200 \
MAX_ITERATIONS=4000 \
FINAL_MOTION_SCALE=1.0 \
bash scripts/train_g1_armhack_stand.sh
```

脚本末尾可继续附加 Hydra override，但不应添加任何未来轨迹观测。例如需要临时改 episode 时长可执行：

```bash
bash scripts/train_g1_armhack_stand.sh env.episode_length_s=20.0
```

#### 6.1.1 当前从 model2999 的续训

当前实际应启动的第二阶段命令是：

```bash
source /home/user/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
cd /home/user/Workspace/Humanoid/Locomotion/G1-Locomotion/legged_lab
bash scripts/train_g1_armhack_stand_randomized_payload.sh
```

它默认从正式 `model_2999.pt` 续训 3000 iteration，使用 512 姿态库、2–6 s 限速 minimum-jerk 轨迹和每侧 `0–1 kg` 腕末端负载。全部参数、smoke 和验证记录见第 1.2 节。

### 6.2 Stand 两阶段最小测试

随机姿态静止阶段：

```bash
NUM_ENVS=2 MAX_ITERATIONS=1 \
STATIC_ITERATIONS=1 RAMP_ITERATIONS=1 \
RUN_NAME=armhack_stand_static_stage_smoke \
bash scripts/train_g1_armhack_stand.sh
```

直接强制进入原始 `1.0x` 运动阶段：

```bash
NUM_ENVS=2 MAX_ITERATIONS=1 \
STATIC_ITERATIONS=0 RAMP_ITERATIONS=0 \
FINAL_MOTION_SCALE=1.0 \
RUN_NAME=armhack_stand_full_speed_smoke \
bash scripts/train_g1_armhack_stand.sh
```

### 6.3 Walk 正式训练：从 locomotion.onnx 对应的 model 9996 初始化

Walk 必须使用专用脚本 `scripts/train_g1_armhack_walk.sh`，不要直接调用通用的
`train_g1_amp.sh`。训练前先进入已经验证的环境和目录：

```bash
source /home/user/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
cd /home/user/Workspace/Humanoid/Locomotion/G1-Locomotion/legged_lab
```

正式训练一个固定的 `pos2_down` 双臂姿态：

```bash
POSE_NAME=pos2_down \
RUN_NAME=armhack_walk_pos2_from_locomotion_model9996 \
bash scripts/train_g1_armhack_walk.sh
```

`POSE_NAME` 支持以下值：

| 值 | 含义 | 适用场景 |
|---|---|---|
| `pos1_back` | 双臂靠后 | 单一固定姿态策略 |
| `pos2_down` | 双臂下垂 | 默认单一固定姿态策略 |
| `pos3_front` | 双臂靠前 | 单一固定姿态策略 |
| `random` | 每个环境在 reset 时从三种姿态中随机选择一种，episode 内保持不变 | 训练一个覆盖三种姿态的统一策略 |

例如训练覆盖三种固定姿态的统一策略：

```bash
POSE_NAME=random \
RUN_NAME=armhack_walk_three_fixed_poses_from_locomotion_model9996 \
bash scripts/train_g1_armhack_walk.sh
```

默认契约：`MODE=init`、S3 G1、4096 env、4000 iteration、Nav2 Stage-4 command、learning rate `3e-5`、style reward `0`、task/style lerp `1`、baseline KL `0.003`、domain randomization `0`、RSI 关闭。脚本会检查：

- `checkpoint/model_9996/locomotion.onnx` 的 SHA、96→29 接口；
- 原始 `model_9996.pt` 的 SHA、大小、iteration、actor/critic 形状；
- ONNX 与 `.pt` 中 8 个 actor 权重/偏置张量逐元素完全一致；
- Nav2 CSV 的 SHA 和 pose JSON 是否存在；
- 资产固定为 `s3_g1_29dof`；
- 当前没有 Stand 或另一个 Walk 训练进程；
- Hydra 参数不能覆盖 checkpoint 加载、baseline KL 或 robot spawn。

这里的 `locomotion.onnx` 只用于确认部署 actor 的来源和 96→29 接口；实际训练从与它
actor 权重逐元素一致的 `BaselineLocomotionModel9996/model_9996.pt` 做 policy-only 初始化。
不要使用历史 `model_7999.pt`，也不要使用任何 Stand 训练后的 checkpoint 作为 Walk 起点。

训练结果同时写到两个位置：

```text
logs/rsl_rl/g1_walk_perturb/<时间戳>_<RUN_NAME>/
ArmHack Checkpoints/WalkPerturbFinetune/<时间戳>_<RUN_NAME>/
```

第一个目录包含 TensorBoard、`params/env.yaml`、`params/agent.yaml` 和 checkpoint；第二个目录
保存同名 checkpoint 副本，供测试、可视化和归档使用。

最小真实 smoke：

```bash
NUM_ENVS=2 MAX_ITERATIONS=1 \
POSE_NAME=pos2_down \
RUN_NAME=armhack_walk_locomotion_model9996_smoke \
QUIET_TERMINAL=False \
bash scripts/train_g1_armhack_walk.sh
```

### 6.4 Walk 断点续训

首次初始化只加载 policy。Walk 中断后必须用 `MODE=resume` 完整恢复 optimizer、
discriminator、normalizer 和 iteration。先填写已有 run 目录名和模型文件名，并确认日志目录与
专用 checkpoint 目录中的两份文件都存在：

```bash
cd /home/user/Workspace/Humanoid/Locomotion/G1-Locomotion/legged_lab

RESUME_RUN='<已有 run 的完整目录名，不含路径>'
RESUME_CHECKPOINT='model_1200.pt'

test -f "logs/rsl_rl/g1_walk_perturb/${RESUME_RUN}/${RESUME_CHECKPOINT}"
test -f "ArmHack Checkpoints/WalkPerturbFinetune/${RESUME_RUN}/${RESUME_CHECKPOINT}"

MODE=resume \
RESUME_RUN="${RESUME_RUN}" \
RESUME_CHECKPOINT="${RESUME_CHECKPOINT}" \
POSE_NAME=pos2_down \
MAX_ITERATIONS=2000 \
RUN_NAME=armhack_walk_pos2_resume1200 \
bash scripts/train_g1_armhack_walk.sh
```

`MAX_ITERATIONS=2000` 表示从 checkpoint 当前 iteration 起再训练 2000 iteration，而不是训练到
绝对编号 2000。`POSE_NAME` 应与原 run 保持一致；如果有意改变姿态分布，应使用新的 `RUN_NAME`
并将其视为新的实验。脚本会校验日志目录与 `ArmHack Checkpoints/WalkPerturbFinetune` 中两份
resume checkpoint 的 SHA 一致。`MODE=resume` 设置 `agent.load_policy_only=False`，不会把续训重置到
iteration 0。

## 7. checkpoint 保存位置

新目录结构为：

```text
ArmHack Checkpoints/
├── StandPerturb/
│   ├── BaselineModel9996/model_9996.pt
│   ├── <run_name>/Test Reports/StandArmOnly/*.md
│   ├── <run_name>/Test Reports/StandArmOnly/*__torso_world_6d.png
│   └── <run_name>/model_*.pt
└── WalkPerturbFinetune/
    ├── BaselineLocomotionModel9996/model_9996.pt
    └── <run_name>/model_*.pt
```

runner 配置分别设置：

```python
checkpoint_output_dir = "ArmHack Checkpoints/StandPerturb"
checkpoint_output_dir = "ArmHack Checkpoints/WalkPerturbFinetune"
```

每次保存时，`AMPRunner.save()` 先按原逻辑写入：

```text
logs/rsl_rl/<experiment>/<run_name>/model_*.pt
```

再复制到对应的 ArmHack 专用目录。保留日志目录副本是为了不破坏 RSL-RL 原有的 `LOAD_RUN/CHECKPOINT` 恢复训练逻辑；专用目录便于集中查找、测试和交付两个策略。两边模型已经用 SHA-256 验证完全一致。

checkpoint 文件已加入 `.gitignore`，不会上传 GitHub。不要按文件修改时间自动选择“最新模型”：冒烟测试也会生成 `model_0.pt`，旧的 `latest_checkpoint` 写法可能把它误当成正式模型。两个任务的已验证起始模型分别保存在：

```text
StandPerturb/BaselineModel9996/model_9996.pt
SHA-256: bc30bc5171d211fa414fbeab31452b92ad76ca7f6ad76a2417a6e7f7515a0fa6
size: 16,202,421 bytes；iteration: 9996；actor: 96→29；critic input: 297

Walk actor: ../checkpoint/model_9996/locomotion.onnx
SHA-256: 05fc45f89d89eb136225754f6a2fcacf5324d9dfd428d08ed75cc52f89b09be6

Walk training checkpoint: WalkPerturbFinetune/BaselineLocomotionModel9996/model_9996.pt
SHA-256: bc30bc5171d211fa414fbeab31452b92ad76ca7f6ad76a2417a6e7f7515a0fa6
size: 16,202,421 bytes；iteration: 9996；actor: 96→29；critic input: 297
```

Walk `.pt` 来源由 `locomotion.deploy.json` 记录，并已从 HEC-5090 原始 S3 run 下载：

```text
/home/hecggdz/workspace-zwd/legged_lab/logs/rsl_rl/g1_amp/
2026-06-17_03-48-38_s3_g1_29dof_command_balanced_directional_strict_armprior_v3_resume8997_1000/model_9996.pt
```

正式课程训练直接使用第 6.1 节的专用脚本。测试或可视化时仍应显式填写完成正式训练且已经通过固定评估的路径：

```bash
STAND_CKPT='ArmHack Checkpoints/StandPerturb/<正式run>/model_<正式迭代>.pt'
WALK_CKPT='ArmHack Checkpoints/WalkPerturbFinetune/<正式run>/model_<正式迭代>.pt'

test -n "$STAND_CKPT" && test -f "$STAND_CKPT"
test -n "$WALK_CKPT" && test -f "$WALK_CKPT"
printf 'Stand: %s\nWalk:  %s\n' "$STAND_CKPT" "$WALK_CKPT"
```

最低检查要求：路径不得包含 `smoke`，不能仅因为文件名是 `model_0.pt` 就认为它有效；必须同时打开对应 run 的 `params/env.yaml`、`params/agent.yaml` 和 TensorBoard 曲线确认训练配置、训练轮数、平均 episode 长度与终止原因。

## 8. 测试代码与指令

### 8.1 外部数据预检

测试代码：

```text
scripts/tools/check_armhack_reference_data.py
```

它不启动 Isaac Sim，会检查：

- Stand 原始 CSV 和 14 关节训练 CSV 的 SHA-256；
- Stand 训练 CSV 的列名、20,122 行和数值有限性；
- Walk 三组 pose 的名称、单位、顺序和 14 个数值；
- Nav2 CSV 的 SHA-256、必需列、331,010 行、445 组、速度范围和 `augmentation=none`。

执行：

```bash
python scripts/tools/check_armhack_reference_data.py
```

若新机器缺少 Nav2 CSV，可在 `legged_lab` 目录执行：

```bash
mkdir -p 'Reference Data/ArmHack/WalkPerturbFinetune'
scp hecggdz@172.28.162.15:/home/hecggdz/workspace-zwd/legged_lab/nav2_loopback_actual/actual_raw_success/all_cmd_vel_success.csv \
  'Reference Data/ArmHack/WalkPerturbFinetune/nav2_cmd_vel_raw_success.csv'
python scripts/tools/check_armhack_reference_data.py
```

### 8.2 静态代码回归

```bash
bash -n scripts/train_g1_amp.sh
bash -n scripts/train_g1_armhack_stand.sh
bash -n scripts/train_g1_armhack_walk.sh
bash -n scripts/vis_isaacsim_g1_amp.sh
python -m compileall -q \
  source/legged_lab/legged_lab/envs/g1_walk_perturb_env.py \
  source/legged_lab/legged_lab/tasks/locomotion/amp/config/g1_perturb \
  source/legged_lab/legged_lab/tasks/locomotion/amp/mdp/commands/nav2_recorded_velocity_command.py \
  scripts/tools/check_armhack_reference_data.py \
  ../rsl_rl/rsl_rl/runners/amp_runner.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/user/.local/bin/pytest -q \
  source/legged_lab/test/test_g1_perturb_static.py
```

当前结果：`10 passed`。

### 8.3 最小真实训练

第 6.2 节和第 6.3 节的 smoke 命令就是两个任务的最小真实训练测试。Walk 测试必须等 Stand 进程退出；专用脚本会主动阻止并发。成功条件不是仅完成 import，而是同时满足：

```text
Isaac Sim 场景创建完成
Command Manager 类型正确
policy/critic/discriminator 网络创建完成
完成 Learning iteration 0/1
日志目录生成 model_0.pt
对应 ArmHack Checkpoints 子目录生成 model_0.pt
```

### 8.4 Walk checkpoint 的 headless 测试

这里的“测试”是加载已经训练出的 Walk checkpoint，在 `Play` 任务中继续使用 Nav2 CSV 的连续
速度窗口，并固定一种双臂姿态进行推理。先指定需要测试的正式 run 和 checkpoint：

```bash
source /home/user/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
cd /home/user/Workspace/Humanoid/Locomotion/G1-Locomotion/legged_lab

WALK_RUN='<ArmHack Checkpoints/WalkPerturbFinetune 下的正式 run 目录名>'
WALK_MODEL='model_<迭代号>.pt'
WALK_CKPT="$PWD/ArmHack Checkpoints/WalkPerturbFinetune/${WALK_RUN}/${WALK_MODEL}"
test -f "$WALK_CKPT" || { echo "checkpoint 不存在: $WALK_CKPT"; exit 1; }
```

对 `pos2_down` 做 1000 个 control step（20 s 仿真时间）的无窗口测试：

```bash
TASK=LeggedLab-Isaac-AMP-G1-WalkPerturbFinetune-Play-v0 \
CHECKPOINT="$WALK_CKPT" \
ROBOT_ASSET=s3_g1_29dof \
NUM_ENVS=1 \
HEADLESS=True \
REAL_TIME=False \
MAX_STEPS=1000 \
SKIP_EXPORT=True \
RSI_ENABLE=False \
RANDOMIZATION_STRENGTH=0 \
bash scripts/vis_isaacsim_g1_amp.sh \
  --seed 42 \
  env.upper_body_perturbation.pose_name=pos2_down
```

在同一个 shell 中逐一测试三种固定姿态：

```bash
for WALK_POSE in pos1_back pos2_down pos3_front; do
  echo "===== testing ${WALK_POSE} ====="
  TASK=LeggedLab-Isaac-AMP-G1-WalkPerturbFinetune-Play-v0 \
  CHECKPOINT="$WALK_CKPT" \
  ROBOT_ASSET=s3_g1_29dof \
  NUM_ENVS=1 \
  HEADLESS=True \
  REAL_TIME=False \
  MAX_STEPS=1000 \
  SKIP_EXPORT=True \
  RSI_ENABLE=False \
  RANDOMIZATION_STRENGTH=0 \
  bash scripts/vis_isaacsim_g1_amp.sh \
    --seed 42 \
    "env.upper_body_perturbation.pose_name=${WALK_POSE}"
done
```

三个进程都显式使用 `--seed 42`，用于尽量保持初始随机状态和 Nav2 窗口采样一致；正式验收还应
更换多个 seed 重复测试，不能只报告 seed 42。

脚本退出前会打印：

```text
[METRIC] IsaacSim play task tracking
[METRIC] IsaacSim play score
[METRIC] IsaacSim play Important Metrics
```

至少检查 `lin_vel_xy_mae`、`yaw_rate_mae`、reward score 以及输出的躯干 roll/pitch、高度误差，
并人工观察是否发生摔倒或 reset。当前通用 `Play` 汇总不会为 Nav2 Walk 单独打印
`done_rate`/termination 分类，因此不能仅凭末尾没有 termination 文本就断言零 reset；严格自动验收还需要
增加 Walk 专用测试报告。固定对比时分别使用三个具名 pose，不要使用 `random`。1000 step 无异常只能
说明该姿态下完成了一次 20 s 测试，不能单独证明所有 Nav2 速度片段和随机种子均已通过。

这里应使用 IsaacLab `Play` 入口。现有通用 MuJoCo 验证脚本没有复现环境内部的固定双臂 action
覆盖，因此不能作为 ArmHack Walk 的等价验收命令。

### 8.5 当前真实测试记录

Stand：

```text
静止阶段 run: logs/rsl_rl/g1_stand_perturb/2026-07-14_16-39-56_armhack_stand_random_phase_smoke_20260714
设置: 4 env × 1 iteration；STATIC_ITERATIONS=1；RAMP_ITERATIONS=1
恢复: model_7999.pt policy-only；baseline KL=0.003
运行日志: csv_motion_scale=0.0000；curriculum_stage=0.0000
随机相位: csv_start_time_mean_s=198.6063；csv_start_time_std_s=137.4946
结果: 正常完成 PPO iteration 0/1，Training time 1.55 s，并保存专用 model_0.pt

20 s episode/重复 reset run: logs/rsl_rl/g1_stand_perturb/2026-07-14_16-41-38_armhack_stand_static_full_episode_smoke_20260714
设置: 4 env × 42 iteration；共 4032 个环境步；env.episode_length_s=20.0
课程: 全程 csv_motion_scale=0.0000；curriculum_stage=0.0000
终止: Episode_Termination/time_out=1.0000；base_height=0.0000；bad_orientation=0.0000
结果: 21 项奖励出现实际非零统计，完成 42 次 PPO 更新，Training time 27.83 s，保存 model_41.pt
说明: IsaacLab 会错开首批环境的 episode counter，因此此短测窗口中的 mean episode length 不直接等于 1000；配置的单 episode 上限仍为 20 s/1000 control step。

低速阶段 run: logs/rsl_rl/g1_stand_perturb/2026-07-14_16-32-07_armhack_stand_slow_motion_smoke_20260714
设置: 2 env × 1 iteration；STATIC_ITERATIONS=0；RAMP_ITERATIONS=0；SLOW_MOTION_SCALE=0.25
恢复: model_7999.pt policy-only；baseline KL=0.003
运行日志: csv_motion_scale=0.2500；curriculum_stage=2.0000
结果: 正常完成 PPO iteration 0/1，Training time 1.52 s，并保存专用 model_0.pt

两个 run 的 Observation Manager 均打印 policy shape=(96,)，仅含当前状态和上一动作；Reward Manager 均打印 21 个有效奖励项。
```

当前 model9996 入口实测：

```text
run: logs/rsl_rl/g1_stand_perturb/2026-07-14_20-03-15_armhack_stand_model9996_smoke_20260714
checkpoint: ArmHack Checkpoints/StandPerturb/2026-07-14_20-03-15_armhack_stand_model9996_smoke_20260714/model_0.pt
设置: 2 env × 1 iteration；STATIC_ITERATIONS=1；RAMP_ITERATIONS=0
恢复: BaselineModel9996/model_9996.pt policy-only；baseline KL=0.003
运行日志: 明确打印 model_9996、actor 96→29、critic input 297、csv_motion_scale=0、curriculum_stage=0
结果: 正常完成 Learning iteration 0/1，共 48 environment steps；完成 PPO 更新并保存 Stand 专用 model_0.pt；无 Traceback、Hydra、Python、CUDA 或 Isaac 启动错误

全速 run: logs/rsl_rl/g1_stand_perturb/2026-07-14_20-28-18_armhack_stand_full_speed_1x_smoke_20260714
checkpoint: ArmHack Checkpoints/StandPerturb/2026-07-14_20-28-18_armhack_stand_full_speed_1x_smoke_20260714/model_0.pt
设置: 2 env × 1 iteration；STATIC_ITERATIONS=0；RAMP_ITERATIONS=0；FINAL_MOTION_SCALE=1.0
恢复: BaselineModel9996/model_9996.pt policy-only；baseline KL=0.003
运行日志: csv_motion_scale=1.0000；curriculum_stage=2.0000
结果: 正常完成 Learning iteration 0/1，共 48 environment steps；完成 PPO 更新并保存 Stand 专用 model_0.pt；无 Traceback、Hydra、Python、CUDA 或 Isaac 启动错误
```

Stand 当前 model9996→`1.0x` 正式课程训练：

```text
run: logs/rsl_rl/g1_stand_perturb/2026-07-14_20-34-20_armhack_stand_curriculum_1x_from_model9996_full_20260714
checkpoint: ArmHack Checkpoints/StandPerturb/2026-07-14_20-34-20_armhack_stand_curriculum_1x_from_model9996_full_20260714/model_2999.pt
设置: 4096 env × 3000 iteration；20 s/1000 control-step episode；S3 G1
恢复: BaselineModel9996/model_9996.pt policy-only；baseline KL=0.003；style reward=0；task/style lerp=1
阶段 0: iteration 0..499，csv_motion_scale=0、curriculum_stage=0
阶段 1: iteration 500..1499，csv_motion_scale 从 0 连续升至 1.0、curriculum_stage=1
阶段 2: iteration 1500..2999，csv_motion_scale=1.0、curriculum_stage=2
最终: mean episode length=1000；timeout=1.0；base_height=0；bad_orientation=0
最终 torso: roll=0.01113 rad；pitch=0.02257 rad；ang_vel_xy=0.1632 rad/s；height error=0.0094 m
训练耗时: 5548.62 s；最终 checkpoint iteration=2999，已用 env_isaaclab 成功加载
checkpoint SHA-256: 2c87cc2cc3706c1024594d14d85a34e7bf468b54f6b66e49b6155ef72a2dbd16
TensorBoard: 63 个 scalar tag，关键指标均写到 step 2999
```

Stand 历史 model7999→`0.25x` 正式课程训练：

```text
run: logs/rsl_rl/g1_stand_perturb/2026-07-14_16-49-13_armhack_stand_curriculum_from_model7999_full_20260714
checkpoint: ArmHack Checkpoints/StandPerturb/2026-07-14_16-49-13_armhack_stand_curriculum_from_model7999_full_20260714/model_2999.pt
设置: 4096 env × 3000 iteration；20 s/1000 control-step episode；S3 G1
恢复: model_7999.pt policy-only；baseline KL=0.003；style reward=0；task/style lerp=1
阶段 0: iteration 0/400 的 csv_motion_scale=0、curriculum_stage=0
阶段 1: iteration 600/1000/1400 的 csv_motion_scale=0.0251/0.1251/0.2251
阶段 2: iteration 1500..2999 的 csv_motion_scale=0.25、curriculum_stage=2
随机相位: 最终 csv_start_time_mean_s=198.8300、std_s=116.2282
最终: mean episode length=1000；timeout=0.9998；base_height=0；bad_orientation=0.0002
最终 torso: roll=0.0074 rad；pitch=0.0130 rad；ang_vel_xy=0.1069 rad/s；height error=0.0092 m
训练耗时: 5588.46 s；最终 checkpoint iteration=2999，已用 env_isaaclab 成功加载
checkpoint SHA-256: 03e0f06c86363f906bbd4ceeb4e51b3897b45de345f0d066b8244bbb354e93e8
```

现有测试覆盖必须按证据强度区分：

| 检查 | 已经证明 | 尚未证明 |
|---|---|---|
| `test_g1_perturb_static.py`，`10 passed` | 旧/新配置、任务注册、姿态库、关节映射、负载事件和回放入口的静态契约 | 独立长时评估的策略性能 |
| 随机静态 smoke | 多环境起始相位不同，阶段 0 的速度倍率确为 0 | 固定 100 episode 的统计覆盖率；该随机模式不再作为 GUI 可视化方法 |
| 低速运动 smoke | 运行时进入阶段 2，`csv_motion_scale=0.25` | 该 smoke 只有 24 control step，只推进原 CSV `0.12 s`，不能作为明显运动的可视化验收 |
| 当前全速运动 smoke | model9996 入口可在阶段 2 以 `csv_motion_scale=1.0` 完成 rollout、PPO 更新和保存 | 一轮 48 environment steps 不代表全速策略已经收敛 |
| 3000-iteration 正式训练 | 三阶段真实执行；随机相位分布、episode 存活和躯干指标完整记录 | 独立固定评估中的手臂目标/实际 `q/dq` 变化量和逐窗口完成率 |
| 旧固定第 0 帧 GUI | checkpoint、任务和 GUI 加载正常 | 姿态切换或慢速运动；该配置实际上反复播放 CSV 开头保持段 |
| 当前 schema v5 `1.0x` 确定性数据集 | 512 姿态库、默认 37 个可回放 CSV、1 个下垂到放平专项 CSV、59 阶段默认时间线均校验通过；新 checkpoint 的 0/1 kg 两次 10,404-step 全量回放均无 termination，专项 20 s 无负载回放也无 termination，逐关节和 6D 报告已保存 | 多 seed、多 episode 的训练分布统计，以及与旧模型在完全相同 schema v5 条件下的性能对照 |

因此，可以确认第二阶段课程真实执行、新模型完整覆盖固定姿态/轨迹且在 0/1 kg 两个条件下未摔倒；不能据此确认所有随机姿态、随机负载和 seed 的统计泛化。第 9 节给出可复现的逐项与整套可视化方法。

Walk（正确 locomotion model9996 起点）：

```text
权威 actor: checkpoint/model_9996/locomotion.onnx
训练 checkpoint: ArmHack Checkpoints/WalkPerturbFinetune/BaselineLocomotionModel9996/model_9996.pt
主 smoke run: logs/rsl_rl/g1_walk_perturb/2026-07-14_19-24-50_armhack_walk_locomotion_model9996_smoke_20260714
设置: 2 env × 1 iteration；S3 G1；pos2_down；RSI=False；randomization=0
恢复: model_9996.pt policy-only；其 actor 与 locomotion.onnx 逐张量一致；baseline KL=0.003
Command Manager: Nav2RecordedVelocityCommand
配置: complex_turn；4 s；scale=(0.85,0.75,0.75)；none+在线 mirror_lr
reset: walk_pose_index_mean=1.0；walk_pose_init_max_error_rad=0.0
终止: base_height=0；bad_orientation=0；正常完成 PPO、保存 model_0.pt

具名姿态 smoke:
pos1_back: 2026-07-14_19-35-39；pose_index=0；init_error=0；base_height=0；bad_orientation=0
pos3_front: 2026-07-14_19-35-55；pose_index=2；init_error=0；base_height=0；bad_orientation=0
两者均实际加载 model_9996、完成 PPO iteration 0/1 并保存 Walk 专用 checkpoint

完整恢复 smoke: logs/rsl_rl/g1_walk_perturb/2026-07-14_19-36-10_armhack_walk_model9996_resume_smoke_20260714
设置: MODE=resume；load_policy_only=false；reset_iteration=false
结果: 从已有 Walk model_0.pt 完整恢复并完成下一次训练/保存，无 traceback
```

旧错误基线曾用 32 env 对比 `RSI=True`，`bad_orientation=0.4688`，几乎等于 RSI 采样比例；关闭后为 0。这个对照只用于解释为何专用脚本默认 `RSI=False`，不作为 model9996 的性能证据，也不代表一轮训练已经达到稳定行走性能。

Walk（错误 Stand model7999 起点，已作废）：

```text
2026-07-14_18-32-34_armhack_walk_model7999_smoke32_no_rsi_20260714
2026-07-14_18-33-30_armhack_walk_pos1_back_smoke_20260714
2026-07-14_18-33-45_armhack_walk_pos3_front_smoke_20260714
2026-07-14_18-34-20_armhack_walk_resume_mode_smoke_20260714
```

这些 run 的环境链路曾正常运行，但起点是经过 Stand 训练的 model7999，不能用于本任务训练、比较或验收。

Walk（旧实现历史记录，已作废）：

```text
run: logs/rsl_rl/g1_amp/2026-07-14_15-39-55_armhack_walk_nav2_checkpoint_smoke
训练: 2 env × 1 iteration，Training time 1.7 s
Command Manager: Nav2RecordedVelocityCommand
checkpoint 恢复: resume=false；没有加载 Nav2 checkpoint
机器人资产: s3_g1_29dof（候选 model_6596 的资产是 original g1_29dof）
专用模型: ArmHack Checkpoints/WalkPerturbFinetune/2026-07-14_15-39-55_armhack_walk_nav2_checkpoint_smoke/model_0.pt
checkpoint SHA-256: a0a58f426b3f7e33729bb620f353c9ce8aa155a6da8aa3d413c8d087e469f35a
回放: 1 env × 5 steps，正常达到 max_steps 并退出
```

上述 `model_0.pt` 仅用于验证完整执行链。它既不是 Nav2 policy，也不能代表正式训练后的策略性能；run 名称中的 `nav2_checkpoint` 与保存配置不一致，后续报告应以 `params/agent.yaml` 为准。

## 9. 可视化两个 ArmHack checkpoint

### 9.1 为什么旧 Stand GUI 只显示一个固定姿态

2026-07-14 对规范 CSV 的数值复核结果如下：

```text
总时长: 404.917792 s
CSV 第 0..5 s 最大单关节变化: 0.011457 rad
CSV 第 0..20 s 最大单关节变化: 0.011457 rad
相对第 0 帧首次达到 0.02 rad: 25.311227 s
相对第 0 帧首次达到 0.10 rad: 25.613424 s
```

旧命令同时使用 `csv_randomize_start_on_reset=False`、20 s episode 和 `0.25x`。因此一次 episode 只读取原 CSV 的第 `0..5 s`，20 s 后又回到第 0 帧；该窗口本身就是保持段，视觉上必然只有一个固定双臂姿态。该命令只能证明 GUI 和 checkpoint 能加载，不能验证随机静态姿态或手臂运动。

以 5 s 为窗口统计整条 CSV，约 `61.0%` 的窗口最大单关节变化达到 `0.10 rad`，其余窗口可能仍是保持段。随机 phase 适合训练分布统计，但不适合人工可视化验收：每次看到的内容不同，也可能连续抽到保持段。现在的 GUI 测试全部改为离线选样后顺序播放，运行时不再随机抽取姿态或轨迹。

以下命令统一使用当前从 model9996 训练得到的 `1.0x` 正式模型：

```bash
cd /home/user/Workspace/Humanoid/Locomotion/G1-Locomotion/legged_lab
STAND_CKPT='/home/user/Workspace/Humanoid/Locomotion/G1-Locomotion/legged_lab/ArmHack Checkpoints/StandPerturb/2026-07-15_14-12-54_armhack_stand_randomized_payload_from_model2999_full_20260715/model_2999.pt'
```

### 9.2 确定性可视化数据集如何构造

生成脚本会读取完整的 404.92 s、14-DoF 双臂数据，但姿态和轨迹只从训练实际可达的时间范围选样，不在 Isaac Sim 运行时抽样：

```bash
python scripts/tools/build_armhack_stand_randomized_training_data.py
python scripts/tools/build_armhack_stand_visualization_suite.py
python scripts/tools/check_armhack_reference_data.py --stand-only
```

训练姿态库输出到 `Reference Data/ArmHack/StandPerturb/RandomizedTraining/`，测试集输出到 `Reference Data/ArmHack/StandPerturb/TestData/ArmOnly/`。`manifest.json` 当前为 schema v5，显式记录 `data_scope=arm_only_14_dof`、`contains_full_body_state=false`、训练采样边界、姿态库 SHA-256、`files.all.detailed_timeline` 和 `special_tests.down_to_horizontal`；默认时间线共 59 段，供 `MODE=all` 的 6D 测试图阶段色带使用。默认 37 个 CSV 加上 1 个专项 CSV，共 38 个可回放 CSV，表头都严格是 `time_s + 14 个双臂关节`，不含根节点、腰、髋、膝或踝。选样和生成规则如下：

- 训练使用 20 s episode、`csv_loop=False` 和 `csv_end_margin_s=0.25`。静态课程的随机保持姿态起点只能落在 `0–384.667792 s`；运动课程最多推进到 `404.667792 s`；
- 代表姿态：只在 `0–384.667792 s` 静态起点可达区间内，按各关节的 P5–P95 范围归一化，再用从中位姿态开始的最远点覆盖法挑选 6 个姿态。原 `404.897585 s` 双臂下垂姿态被显式排除，不出现在代表姿态、合成姿态或默认完整序列中；
- 代表轨迹：用 5 s 滑动窗口遍历运动课程可达区间，过滤低运动窗口，再按姿态、关节幅值、速度和端点变化的联合描述选出 4 段相互至少间隔 10 s 的高运动量窗口；
- 合成姿态：固定种子 `20260714`，每次只选两组实测代表手臂姿态做有界凸插值，一次性写出 3 个双臂 CSV；不再加入逐关节随机扰动，也不会生成全身姿态；
- 合成轨迹：固定种子选取两条等长的实测 `1.0x` 双臂轨迹，按同一时刻逐帧做凸组合，写出 3 条 5 s、50 Hz 合成轨迹；速度和加速度保持在两条父轨迹的凸包内，不生成腰腿数据；
- 新随机覆盖姿态：从 512 姿态库固定选出 8 个样本，用于补充旧 9 个姿态的覆盖；
- 新插值轨迹：固定选择姿态库端点，生成 6 条 5 s、50 Hz 五次 minimum-jerk 轨迹，关节速度不超过姿态库中记录的安全限制；
- 下垂到放平专项：仅 `special/arms_down_to_forward_horizontal_20s_50hz.csv` 复用 `404.897585 s` 下垂姿态，并在 5 s 保持后用 6 s minimum-jerk 抬到实测左右臂组合的向前水平姿态，再保持 9 s；该起点超出静态随机课程范围，所以专项结果与默认分布内验收分开报告；
- 运行时固定 `csv_randomize_start_on_reset=False`。所谓“随机合成”只发生在离线构建阶段，固定种子、父姿态、权重和文件 SHA-256 都记录在 `manifest.json`，因此每次回放一致。

当前 6 个代表姿态来自原 CSV 的 `261.395829`、`106.447880`、`133.214841`、`240.075848`、`285.331942` 和 `135.831008 s`。它们都在训练静态阶段可采样的 `0–384.667792 s` 范围内。4 段代表轨迹为：

| 编号 | 原始窗口 | 主要变化关节 | 最大关节跨度 | 可视化时长 |
|---|---:|---|---:|---:|
| 1 | 36–41 s | left elbow | 1.0577 rad | 5 s，`1.0x` |
| 2 | 102–107 s | left elbow | 0.7792 rad | 5 s，`1.0x` |
| 3 | 234–239 s | right elbow | 0.9130 rad | 5 s，`1.0x` |
| 4 | 385–390 s | right shoulder pitch | 1.1224 rad | 5 s，`1.0x` |

代表轨迹不再做时间拉伸：原始 5 s 窗口仍播放 5 s，`time_stretch=1.0`、`equivalent_source_speed=1.0`。合成轨迹也由两条实测 `1.0x` 窗口逐帧混合。播放端固定 `csv_motion_scale=1.0`，因此逐项轨迹测试与新课程最终阶段速度一致。

### 9.3 一条命令顺序看完全部测试

先进入 IsaacLab 环境，然后启动默认 `MODE=all`：

```bash
source /home/user/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
cd /home/user/Workspace/Humanoid/Locomotion/G1-Locomotion/legged_lab

python scripts/tools/check_armhack_reference_data.py --stand-only

export STAND_CKPT="$PWD/ArmHack Checkpoints/StandPerturb/2026-07-15_14-12-54_armhack_stand_randomized_payload_from_model2999_full_20260715/model_2999.pt"
test -f "$STAND_CKPT" || { echo "错误：checkpoint 不存在" >&2; }

CHECKPOINT="$STAND_CKPT" MODE=all PAYLOAD_KG=0 \
  bash scripts/vis_g1_armhack_stand_eval.sh
```

建议显式传入 `CHECKPOINT="$STAND_CKPT"`，以便在新终端中明确被测模型；回放使用单环境、正面跟随相机和实时 GUI。当前完整确定性序列总时长为 208.08 s，时间线固定为：

| 可视化时间 | 内容 |
|---:|---|
| 0.00–33.98 s | 6 个代表姿态；每个保持 4 s，相邻姿态用 2 s 平滑过渡 |
| 33.98–36.98 s | 平滑段间连接 |
| 36.98–52.96 s | 3 个固定种子合成姿态 |
| 52.96–55.96 s | 平滑段间连接 |
| 55.96–114.08 s | 8 个新随机覆盖姿态；每个保持 4 s，之间为限速平滑过渡 |
| 114.08–117.08 s | 平滑段间连接 |
| 117.08–143.08 s | 4 条 `1.0x` 代表轨迹，轨迹之间 2 s 平滑连接 |
| 143.08–146.08 s | 平滑段间连接 |
| 146.08–165.08 s | 3 条固定种子合成 `1.0x` 轨迹 |
| 165.08–168.08 s | 平滑段间连接 |
| 168.08–208.08 s | 6 条新 minimum-jerk `1.0x` 轨迹 |

若模型中途触发终止，环境会从该 CSV 的第 0 帧重新开始，这是一个真实失败，不能把 reset 后的后续画面当作已覆盖剩余测试。此时应按下一节逐项启动，直接定位失败样本。

测试报告的 6D 曲线使用同一份详细时间线：`RP` 表示代表姿态，`SP` 表示旧合成姿态，`GP` 表示新随机覆盖姿态，`RT` 表示代表轨迹，`ST` 表示旧合成轨迹，`GT` 表示新 minimum-jerk 轨迹，`T`/`B` 分别表示姿态过渡和段间连接。

此前 208.96 s 回放使用旧 `0.25x` 离线拉伸数据；schema v3 的 103.96 s 数据集包含现已删除的下垂姿态；schema v4 的 103.96 s 数据集还没有新随机姿态/轨迹。三者都只是历史记录。当前 schema v5 锁定 `trajectory_speed_scale=1.0` 并限制在训练可达范围内；新性能结论必须重新运行当前文件。

### 9.4 分类别或逐项复测

六类顺序播放：

```bash
CHECKPOINT="$STAND_CKPT" MODE=representative_poses bash scripts/vis_g1_armhack_stand_eval.sh
CHECKPOINT="$STAND_CKPT" MODE=synthesized_poses bash scripts/vis_g1_armhack_stand_eval.sh
CHECKPOINT="$STAND_CKPT" MODE=randomized_poses PAYLOAD_KG=0 bash scripts/vis_g1_armhack_stand_eval.sh
CHECKPOINT="$STAND_CKPT" MODE=representative_trajectories bash scripts/vis_g1_armhack_stand_eval.sh
CHECKPOINT="$STAND_CKPT" MODE=synthesized_trajectories bash scripts/vis_g1_armhack_stand_eval.sh
CHECKPOINT="$STAND_CKPT" MODE=randomized_trajectories PAYLOAD_KG=1.0 bash scripts/vis_g1_armhack_stand_eval.sh
CHECKPOINT="$STAND_CKPT" MODE=down_to_horizontal PAYLOAD_KG=0 bash scripts/vis_g1_armhack_stand_eval.sh
```

按编号复测单个姿态或单条轨迹：

```bash
CHECKPOINT="$STAND_CKPT" MODE=representative_pose ITEM=1 bash scripts/vis_g1_armhack_stand_eval.sh       # ITEM=1..6
CHECKPOINT="$STAND_CKPT" MODE=synthesized_pose ITEM=1 bash scripts/vis_g1_armhack_stand_eval.sh          # ITEM=1..3
CHECKPOINT="$STAND_CKPT" MODE=randomized_pose ITEM=1 PAYLOAD_KG=1.0 bash scripts/vis_g1_armhack_stand_eval.sh # ITEM=1..8
CHECKPOINT="$STAND_CKPT" MODE=representative_trajectory ITEM=1 bash scripts/vis_g1_armhack_stand_eval.sh # ITEM=1..4
CHECKPOINT="$STAND_CKPT" MODE=synthesized_trajectory ITEM=1 bash scripts/vis_g1_armhack_stand_eval.sh    # ITEM=1..3
CHECKPOINT="$STAND_CKPT" MODE=randomized_trajectory ITEM=1 PAYLOAD_KG=1.0 bash scripts/vis_g1_armhack_stand_eval.sh # ITEM=1..6
```

每个单姿态 CSV 都把同一组 14-DoF 手臂目标严格保持 20 s（1001 行，50 Hz），专门用于检查“给定姿态下手臂不动，躯干能否稳定”。单轨迹模式从对应轨迹起点直接开始，不需要等待原始 CSV 前 25 s 的保持段。

`down_to_horizontal` 不需要 `ITEM`：它固定播放 5 s 下垂保持、6 s 平滑抬臂和 9 s 向前水平保持，并在 6D 图底部依次标注 `AD`、`D→H`、`AH`。完整数据来源、IsaacLab/MuJoCo 命令和当前模型实测结果见第 1.1.7.1 节。

### 9.5 无窗口检查和录制

无窗口执行链 smoke：

```bash
CHECKPOINT="$STAND_CKPT" MODE=representative_trajectory ITEM=1 \
HEADLESS=True REAL_TIME=False MAX_STEPS=20 \
bash scripts/vis_g1_armhack_stand_eval.sh
```

录制完整确定性序列：

```bash
CHECKPOINT="$STAND_CKPT" MODE=all PAYLOAD_KG=0 \
HEADLESS=True REAL_TIME=False MAX_STEPS=10404 \
bash scripts/vis_g1_armhack_stand_eval.sh \
  --video --video_length 10404
```

20 step 只能证明 checkpoint、任务、确定性 CSV 和仿真链能加载；完整 208.08 s 序列需要 10,404 个 50 Hz control step。视频默认写到 checkpoint 所在目录的 `videos/play/`，该目录已被 `.gitignore` 排除。

### 9.6 checkpoint 同目录测试报告

`vis_g1_armhack_stand_eval.sh` 每次退出时都会自动在被测模型目录下写报告：

```text
<checkpoint 所在目录>/Test Reports/StandArmOnly/
  <checkpoint_stem>_<SHA前12位>__<MODE>[_itemN]__payload_<每侧kg>kg.md
  <checkpoint_stem>_<SHA前12位>__<MODE>[_itemN]__payload_<每侧kg>kg__torso_world_6d.png
```

例如默认完整序列对应：

```text
ArmHack Checkpoints/StandPerturb/2026-07-15_14-12-54_armhack_stand_randomized_payload_from_model2999_full_20260715/
Test Reports/StandArmOnly/model_2999_877e929d516c__all__payload_0kg.md
Test Reports/StandArmOnly/model_2999_877e929d516c__all__payload_0kg__torso_world_6d.png
```

报告记录 checkpoint、测试 CSV 和 manifest 的绝对路径/SHA-256、左右腕末端各自的固定附加质量、控制步数、测试时长、termination/reset 次数、躯干 Important Metrics，以及机器人全部 29 个实际关节的统计。`平均逐步波动` 定义为同一 episode 内相邻 50 Hz 控制帧的 `mean(|q[t]-q[t-1]|)`，单位为 `rad/step`；reset 前后的跳变不计入。报告把 14 个手臂关节标为“输入关节”，把 15 个腰腿关节标为“平衡策略关节”。

torso 6D 统计使用 `torso_link` 的世界系位姿。每个 episode 以其开始时刻为参考：平移为 `p_w(t)-p_w(0)`，旋转为世界系 XYZ Euler 角之差并 wrap 到 `[-pi, pi]`。六个分量分别保存有符号均值、绝对值均值、标准差、RMS、最大绝对值、最小值、最大值和极差；另保存水平平移范数、3D 平移范数和 RPY 位移范数的均值、标准差、RMS 和最大值。发生 reset 时，参考位姿会在 reset 后重建，因此 reset 瞬间的坐标跳变不污染统计，但 `termination/reset` 仍会明确计数并判为失败。

同名 PNG 保存逐控制帧的世界系 6D 位移曲线。确定性测试默认只有一个环境，因此曲线就是该机器人实际轨迹；多环境运行时曲线是各环境逐帧均值。背景色与底部色带来自 manifest 的详细时间线，报告还附有完整阶段起止时间表，可用来定位某个峰值发生在哪个姿态、轨迹或过渡阶段。启动脚本会自动向 `play.py` 传入 manifest，不需要用户额外添加参数。

历史 schema v3/v4 的 5198-step 报告与当前 schema v5 文件 SHA-256 不一致，只能作为旧测试集基准。2026-07-15 已对 SHA 前缀 `877e929d516c` 的新模型真实完成每侧 0 kg 与每侧 1 kg 两次 10,404-step schema v5 headless 全量评估，均为 `termination/reset=0`。详细报告、6D PNG 和总报告均位于新 checkpoint 的 `Test Reports/StandArmOnly/`。

2026-07-14 已实际执行上述 `representative_trajectory ITEM=1` 的 20-step headless smoke：确定性 CSV 路径解析成功，`model_2999.pt` 以 policy-only 方式加载，Isaac Sim 完成 20 step 后按 `max_steps` 正常退出，无 Python、Hydra、Isaac 或 CUDA 异常。该结果只确认新入口可运行，不代表 20 s 轨迹已完整通过稳定性验收。

### 9.7 Walk GUI 和视频可视化

先在 `env_isaaclab` 环境的 `legged_lab` 目录中指定正式 checkpoint；不要把 smoke 的
`model_0.pt` 当成训练结果：

```bash
source /home/user/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
cd /home/user/Workspace/Humanoid/Locomotion/G1-Locomotion/legged_lab

WALK_RUN='<ArmHack Checkpoints/WalkPerturbFinetune 下的正式 run 目录名>'
WALK_MODEL='model_<迭代号>.pt'
WALK_CKPT="$PWD/ArmHack Checkpoints/WalkPerturbFinetune/${WALK_RUN}/${WALK_MODEL}"
test -f "$WALK_CKPT" || { echo "checkpoint 不存在: $WALK_CKPT"; exit 1; }
```

打开 Isaac Sim GUI，实时查看 `pos2_down` 姿态下的 Nav2 速度跟踪：

```bash
TASK=LeggedLab-Isaac-AMP-G1-WalkPerturbFinetune-Play-v0 \
CHECKPOINT="$WALK_CKPT" \
ROBOT_ASSET=s3_g1_29dof \
NUM_ENVS=1 \
HEADLESS=False \
REAL_TIME=True \
SKIP_EXPORT=True \
RSI_ENABLE=False \
RANDOMIZATION_STRENGTH=0 \
FOLLOW_CAMERA=True \
CAMERA_VIEW=front \
bash scripts/vis_isaacsim_g1_amp.sh \
  --seed 42 \
  env.upper_body_perturbation.pose_name=pos2_down
```

不设置 `MAX_STEPS` 时窗口会持续运行，关闭 Isaac Sim 即可退出。将最后一行依次改为
`pos1_back`、`pos2_down`、`pos3_front` 可以固定查看三种姿态；逐姿态人工比较时不要使用
`random`。该入口仍从 `Reference Data/ArmHack/WalkPerturbFinetune/` 中的 Nav2 CSV 采样连续速度
窗口，并不是恒定速度演示。

无窗口录制 `pos2_down` 的 20 s 视频：

```bash
TASK=LeggedLab-Isaac-AMP-G1-WalkPerturbFinetune-Play-v0 \
CHECKPOINT="$WALK_CKPT" \
ROBOT_ASSET=s3_g1_29dof \
NUM_ENVS=1 \
HEADLESS=True \
REAL_TIME=False \
MAX_STEPS=1000 \
SKIP_EXPORT=True \
RSI_ENABLE=False \
RANDOMIZATION_STRENGTH=0 \
FOLLOW_CAMERA=True \
CAMERA_VIEW=front \
bash scripts/vis_isaacsim_g1_amp.sh \
  --video \
  --video_length 1000 \
  --seed 42 \
  env.upper_body_perturbation.pose_name=pos2_down
```

视频写入 checkpoint 所在目录的 `videos/play/`。如果 `WALK_CKPT` 指向
`ArmHack Checkpoints/WalkPerturbFinetune/<run>/model_*.pt`，对应输出就是：

```text
ArmHack Checkpoints/WalkPerturbFinetune/<run>/videos/play/
```

`CAMERA_VIEW` 可改为 `front`、`chase` 或 `side`；`FOLLOW_CAMERA=True` 会让相机跟随机器人。
测试和视频命令中的 `1000 step` 对应 50 Hz 控制频率下的 20 s 仿真时间。

### 9.8 TensorBoard

专用 checkpoint 目录只保存模型；事件、配置和训练曲线仍在原日志目录：

```bash
cd /home/user/Workspace/Humanoid/Locomotion/G1-Locomotion/legged_lab

# 查看全部 Stand run
/home/user/anaconda3/envs/proknee/bin/tensorboard \
  --logdir logs/rsl_rl/g1_stand_perturb \
  --host 127.0.0.1 --port 6006 --reload_interval 10

# 只查看当前第二阶段 1.0x 正式 Stand run
STAND_RUN='2026-07-15_14-12-54_armhack_stand_randomized_payload_from_model2999_full_20260715'
/home/user/anaconda3/envs/proknee/bin/tensorboard \
  --logdir "logs/rsl_rl/g1_stand_perturb/${STAND_RUN}" \
  --host 127.0.0.1 --port 6006 --reload_interval 10

# 查看全部 Walk run
/home/user/anaconda3/envs/proknee/bin/tensorboard \
  --logdir logs/rsl_rl/g1_walk_perturb \
  --host 127.0.0.1 --port 6006 --reload_interval 10

# 只查看一个 Walk run
WALK_RUN='<logs/rsl_rl/g1_walk_perturb 下的正式 run 目录名>'
/home/user/anaconda3/envs/proknee/bin/tensorboard \
  --logdir "logs/rsl_rl/g1_walk_perturb/${WALK_RUN}" \
  --host 127.0.0.1 --port 6006 --reload_interval 10
```

浏览器打开 `http://127.0.0.1:6006`。TensorBoard 用于观察训练曲线，Isaac Sim GUI 和录制视频用于观察实际姿态与运动；两者不能互相替代。训练使用 `env_isaaclab`，TensorBoard 进程只读取 event 文件，可以使用本机已验证的 `proknee` 环境启动。

## 10. Stand 训练代码专项审计

### 10.1 已修复并完成运行验证

| 原问题 | 当前修复 | 真实运行证据 |
|---|---|---|
| walking 父配置把奖励设为 `None` | Stand 显式重新创建全部任务奖励 | Reward Manager 打印 21 个有效 term，包括 `alive` 和 `termination_penalty=-200` |
| 全负奖励可能鼓励提前摔倒 | 加入存活、零水平速度、零 yaw 角速度和双支撑正奖励 | 两个 smoke run 均成功解析并完成 PPO 更新 |
| 随机 CSV 起点首步跳变 | reset 后直接把 14 个手臂关节写到随机目标，速度为 0 | 静止阶段真实运行无接口错误，`csv_motion_scale=0` |
| CSV 零阶保持 | 改为连续相位并对相邻帧线性插值 | 最终阶段真实运行，`csv_motion_scale=1.0000`、`curriculum_stage=2.0000` |
| 初始根状态含随机位置/速度 | Stand 的 x/y/yaw 与六维根速度全部固定为 0 | 新 run 保存的解析配置与 Event Manager 正常 |
| 早期课程混入 push | Stand 显式关闭 `push_robot`，专用脚本默认随机化强度 0 | Event Manager 中不存在 interval push |
| 当前训练起点更新 | Stand 基线改为 `BaselineModel9996/model_9996.pt`；脚本锁定 SHA-256 和大小 | 真实 smoke 明确打印 `Loaded policy-only ... model_9996.pt`，并保存新的 `model_0.pt` |

其余已确认正确的执行链保持不变：绝对关节角先按 `(target-offset)/scale` 转回 raw action；CSV 使用具名关节映射；目标裁剪到 soft limit；手臂覆盖 14 维而腿/腰仍由 policy 控制；判别器和动作正则只使用 root 与 15 个腿/腰维度。

### 10.2 无未来信息约束

课程相位、下一帧目标和轨迹速度只存在于环境内部，用于生成外部手臂 action。真实运行的策略观测仍为：

```text
base_ang_vel(3) + projected_gravity(3) + velocity_command(3)
+ joint_pos(29) + joint_vel(29) + previous_action(29) = 96
```

没有新增 phase、target、target velocity 或 look-ahead。policy 可以看到当前手臂 `q/dq`，这是反馈平衡所必需的当前状态，不属于未来信息。当前训练恢复的 `model_9996` actor 也是 96 维输入、29 维输出，结构完全兼容。

### 10.3 正式长训练完成后仍需回答的问题

- 第一阶段 3000-iteration 训练在最终 `1.0x` 阶段达到在线 episode length `1000/1000`、timeout `100%` 和零摔倒终止；第二阶段最终点为 timeout `99.95%`、bad-orientation `0.05%`；这些都是训练分布在线统计；
- 第二阶段新 checkpoint 已按第 1.2.6/9 节完成 0 kg 和每侧 1 kg 的确定性整套测试，两者均无 termination；仍缺少多 seed、多 episode 统计，以及与旧模型使用同一 schema v5 的公平对照；
- 课程按全局 step 固定切换，不是按完成率自动晋级；下一次训练仍应在 iteration 400–500 检查阶段 0 是否收敛；
- 仍需统计目标被 soft joint limit 裁剪的比例，并记录目标/实际 arm `q/dq/ddq` 以诊断执行器跟踪；
- 20 s 训练完成率不能替代约 405 s 全轨迹固定评估；
- Stand 仍继承 AMP walking 配置并加载运动数据，功能上无误但启动慢，后续可把纯静态任务拆成更轻的 runner/config。

“增加上半身加速度惩罚”仍不是合适的 policy 修复：手臂由脚本强制覆盖，policy 无法改变它。手臂目标速度、加速度与跟踪误差应作为诊断指标；policy 奖励只约束它能改变的躯干、根节点、足底和腿/腰响应。

## 11. 本机正式训练与历史日志复核

本机已经完成当前代码快照下的两轮 3000-iteration 三阶段课程。最终点如下；它们是 iteration 2999 的在线训练统计，不等于独立固定评估，也不是最后 100 iteration 的均值：

| 本机 run | 阶段 | 平均长度/1000 | timeout | base height | bad orientation | torso roll / pitch |
|---|---|---:|---:|---:|---:|---:|
| `2026-07-15_14-12-54_armhack_stand_randomized_payload_from_model2999_full_20260715` | 随机姿态/插值轨迹/负载，`1.0x`，stage 2 | 1000.00 | 99.95% | 0% | 0.05% | 0.0168 / 0.0346 rad |
| `2026-07-14_20-34-20_armhack_stand_curriculum_1x_from_model9996_full_20260714` | `1.0x`，stage 2 | 1000.00 | 100% | 0% | 0% | 0.0111 / 0.0226 rad |
| `2026-07-14_16-49-13_armhack_stand_curriculum_from_model7999_full_20260714` | 历史 `0.25x`，stage 2 | 1000.00 | 99.98% | 0% | 0.02% | 0.0074 / 0.0130 rad |

最新第二阶段 `1.0x` run 的 `model_2999.pt` SHA-256 为 `877e929d516cffe9131cc235477ceef4b226ec69e41c0f1c23e48816cfa28821`，已用 `env_isaaclab` 完成两次全量回放。第一阶段模型 SHA-256 为 `2c87cc2cc3706c1024594d14d85a34e7bf468b54f6b66e49b6155ef72a2dbd16`；历史 `0.25x` 模型 SHA-256 为 `03e0f06c86363f906bbd4ceeb4e51b3897b45de345f0d066b8244bbb354e93e8`。必须用 run 路径或 SHA 区分三个同名 checkpoint。

以下长期训练结果来自 HEC-5090 的历史 TensorBoard 事件文件，数值为最后 100 个 iteration 的均值；最大 episode 长度为 `1000` 个控制步，即 20 s。远端 run 使用当时保存的 `params/env.yaml`/`params/agent.yaml`，与当前课程代码不是完全同一个代码快照，因此只用于说明初始化模型来源和历史对照，不替代本机新模型的固定评估。

| 远端 run | 关键设置 | 平均长度/1000 | timeout | bad orientation | torso roll / pitch |
|---|---|---:|---:|---:|---:|
| `2026-07-09_22-58-27_stand_csv_perturb` | 早期实现，历史关节映射有问题 | 857.41 | 65.73% | 34.30% | 0.0410 / 0.1133 rad |
| `2026-07-11_17-58-24_stand_csv_perturb` | 从零、style=0、task lerp=1 | 15.65 | 0% | 约 100% | 0.1215 / 0.1011 rad |
| `2026-07-13_20-26-50_stand_csv_perturb_from_nav2_stage4_policy_kl` | Nav2 policy-only、KL=0.003、style=5、lerp=0.4 | 998.15 | 99.60% | 0.39% | 0.0317 / 0.0424 rad |
| `2026-07-14_01-08-16_stand_csv_perturb_from_nav2_stage4_policy_kl` | Nav2 policy-only、KL=0.003、style=0、lerp=1 | 999.95 | 99.98% | 0.0165% | 0.0189 / 0.0219 rad |

7 月 14 日正式模型的远端绝对路径是：

```text
/home/hecggdz/workspace-zwd/legged_lab/logs/rsl_rl/g1_amp/
2026-07-14_01-08-16_stand_csv_perturb_from_nav2_stage4_policy_kl/model_7999.pt
```

结论分三层：

1. 7 月 11 日实验确实是失败策略，最终几乎全部因 `bad_orientation` 终止；它同时包含旧数据映射、从零初始化和全负奖励等多个因素，不能把失败只归因于某一个权重。
2. 从 Nav2 policy 做 policy-only 初始化后，即使 7 月 14 日关闭 style reward，训练分布内也能达到接近完整 20 s 存活。这说明“不要从随机策略硬学静态任务”是高价值结论，AMP style reward 不是唯一解。
3. 上述 `model_7999.pt` 只解释历史 `0.25x model_2999.pt` 的来源。第一阶段脚本加载 `BaselineModel9996/model_9996.pt`；第二阶段 `train_g1_armhack_stand_randomized_payload.sh` 则锁定并校验第一阶段正式 `model_2999.pt`。测试时必须用 run 路径区分同名 checkpoint，也不能把 smoke `model_0.pt` 当作正式模型。

## 12. 已实现课程与正式训练判据

### 12.1 当前课程设计

当前默认训练总长 3000 iteration：

| 阶段 | iteration | 手臂扰动 | 目标 |
|---|---:|---|---|
| A：任意姿态静止 | `0..499` | 每次 reset 从 512 姿态库抽取起点并保持 | 先在更广但不超出数据范围的姿态下站稳 |
| B：全速渐入 | `500..1499` | minimum-jerk 轨迹速度从 `0` 线性升到 `1.0x` | 避免课程边界突然增加扰动 |
| C：全速适应 | `1500..2999` | 持续 `1.0x` 姿态插值轨迹 | 学会对多组新起点/终点手臂运动做纯反馈补偿 |

三段在一个 run 中完成，当前入口从第一阶段正式 `model_2999.pt` policy-only 初始化，style reward 为 0，baseline KL 为 `0.003`。episode 固定 20 s，初始根状态严格静止，一般 push/广义 randomization strength 仍为 0，但专用的左右腕末端附加质量随机化独立开启，默认每侧 `U(0,1.0) kg`。

### 12.2 是否进入下一实验的判断

课程是按训练 step 切换，而不是自动看成功率，所以不能盲目等脚本跑完。建议在 iteration 400–500 检查：

- 最近窗口的 episode length 是否接近 1000 step，timeout 是否接近 100%；
- `base_height` 和 `bad_orientation` 是否接近 0；
- torso roll/pitch、角速度、高度误差和根节点 XY 漂移是否持续下降；
- `ArmHack/random_motion_scale` 在阶段 A 必须为 0，阶段 C 必须为 1.0；
- 固定 100 episode 的随机静止姿态评估是否通过。

若阶段 A 未收敛，应停止当前实验并增大 `STATIC_ITERATIONS`，而不是继续把手臂加速。阶段 C 已直接训练 `1.0x`；若其失败，应先比较阶段 B 中 `0.25x/0.5x/0.75x` 附近的指标，再分别用 `PAYLOAD_KG=0` 和 `1.0` 的确定性测试区分“轨迹速度问题”和“负载泛化问题”。其他摩擦、执行器随机化和 interval push 仍应最后逐步加入。

### 12.3 明确禁止的观测改动

本任务按用户要求采用纯反馈策略。后续也不要把以下量加入 policy observation：轨迹 phase、下一帧/未来手臂目标、未来目标速度、短时 look-ahead。允许使用当前实际手臂关节位置和速度，因为它们属于机器人当前本体状态。

数据侧仍可把 405 s 文件切成命名动作、重采样到严格 50 Hz，并限制目标速度、加速度和 jerk；这些是生成外部扰动的质量控制，不会向 policy 泄漏未来信息。

## 13. 固定评估协议

### 13.1 先确认 checkpoint 身份

每次评估报告必须记录：

```text
checkpoint 绝对路径和 SHA-256
训练 run 名、iteration、Git commit
params/env.yaml 与 params/agent.yaml
轨迹文件 SHA-256
trajectory id、起始 phase、幅值/速度倍率
seed、episode 时长、随机化开关、左右腕末端附加质量
```

### 13.2 三类评估分开报告

- 训练分布统计：20 s、随机姿态库起点/终点、最终 `1.0x`、每侧 `U(0,1 kg)` 负载，统计至少 100 个 episode 的完成率和 termination 分布；
- 确定性日常评估：使用第 9 节的 schema v5 固定数据集，逐项覆盖 6+3+8 个姿态和 4+3+6 条轨迹，并分别用 `PAYLOAD_KG=0` 和规定负载报告结果；
- 完整源数据压力测试：固定 phase、固定零初速和固定物理参数，以 `1.0x` 播放约 405 s 原始 CSV；历史 `MAX_STEPS=10123` 只覆盖约 202.46 s，不能称作完整播放。

输出至少包括：survival/time-out 比例、`base_height`/`bad_orientation` 次数、torso roll/pitch 的 RMS/P95/max、torso 水平角速度 RMS、torso 高度误差 RMS、根节点 XY 漂移、脚滑、双脚接触率，以及手臂目标/实际跟踪误差。episode return 只作为辅助量。

### 13.3 严格长时回放模板

`STAND_CKPT` 必须手工指向正式模型。回放 reset 会直接把手臂初始化到第 0 帧，所以没有默认姿态到首帧的目标跳变。为了在约 410 s 内覆盖完整原始轨迹，下面显式关闭训练课程并用 `1.0x` 速度播放；这与当前课程最终阶段速度一致，是完整数据压力测试。约 404.92 s 原始轨迹需要约 20,246 个 50 Hz control step，模板保留少量尾部余量。

```bash
STAND_CKPT='/home/user/Workspace/Humanoid/Locomotion/G1-Locomotion/legged_lab/ArmHack Checkpoints/StandPerturb/2026-07-15_14-12-54_armhack_stand_randomized_payload_from_model2999_full_20260715/model_2999.pt'

TASK=LeggedLab-Isaac-AMP-G1-StandPerturb-Play-v0 \
CHECKPOINT="$STAND_CKPT" \
NUM_ENVS=1 \
HEADLESS=True \
MAX_STEPS=20500 \
REAL_TIME=False \
RSI_ENABLE=False \
RANDOMIZATION_STRENGTH=0 \
FOLLOW_CAMERA=True \
CAMERA_VIEW=front \
bash scripts/vis_isaacsim_g1_amp.sh \
  --video \
  --video_length 20500 \
  env.episode_length_s=410.0 \
  env.upper_body_perturbation.csv_randomize_start_on_reset=False \
  env.upper_body_perturbation.csv_curriculum_enabled=False \
  env.upper_body_perturbation.csv_curriculum_motion_scale=1.0 \
  env.upper_body_perturbation.csv_loop=False \
  env.upper_body_perturbation.csv_initialize_joint_state_on_reset=True
```

## 14. Stand 修改边界与当前结论

当前已新增 512 姿态库构建器、环境内 minimum-jerk 轨迹状态机、`G1StandRandomizedPayloadEnvCfg`、末端附加质量事件、`train_g1_armhack_stand_randomized_payload.sh` 以及 schema v5 确定性测试。续训从已校验的第一阶段正式 `model_2999.pt` 开始；姿态库、负载和未来轨迹信息均没有增加到 policy observation。

结论：第一阶段 Stand 已完成 model9996→model2999、3000 iteration、最终 `1.0x` 的正式训练；第二阶段也已从第一阶段 model2999 完成随机姿态、插值轨迹和每侧 `0–1 kg` 负载的 3000-iteration 正式训练。新 checkpoint SHA 前缀为 `877e929d516c`。schema v5 回放在运行时不抽样，IsaacLab 与 MuJoCo 报告分别保存在 `StandArmOnly/` 和 `StandArmOnlyMuJoCo/`，均带模型 SHA、测试项和负载标签。新模型在两种模拟器中的 0/1 kg 完整确定性测试均未摔倒；MuJoCo 0/1 kg 也均完整执行 10,405 个记录样本并保持 `healthy=True`。不过 IsaacLab 每侧 1 kg 时 pitch 位移 RMS 为 `5.71°`，MuJoCo 每侧 1 kg 时 RPY 位移范数 RMS 为 `0.12540 rad`，带载姿态稳定性仍是下一步应改进的主要指标。

## 15. Walk 当前实现与历史审计对照

### 15.0 2026-07-14 已完成修复

本轮已经完成以下代码修改：

- 新增 `G1WalkPerturbAmpEnv`，Walk/Stand 使用不同 Gym entry point；
- reset 在首个 observation 前同步手臂 `q/dq`、position target 和 action history；
- Walk 直接继承 Nav2 配置，锁定 Stage-4 command/reward，删除无效 command curriculum；
- 原始 Nav2 CSV 在线生成 `mirror_lr`，保持源文件和 Git ignore 不变；
- 新增 `scripts/train_g1_armhack_walk.sh`，区分 `MODE=init` 和 `MODE=resume`；
- 固定 S3 G1、`locomotion.onnx`/`model_9996.pt` actor 一致性、低学习率、style=0 和 baseline KL；
- Walk 独占 `logs/rsl_rl/g1_walk_perturb`，检测到 Stand 训练会拒绝启动；
- 静态回归覆盖新旧任务注册、继承关系、command、奖励、pose 初始化、随机姿态/轨迹、负载事件、脚本和 checkpoint，当前为 `10 passed`。

下面 15.1–15.10 保留的是“修改前审计快照”，用于解释为何修改；其中“当前实现”“下一轮”等措辞均指旧代码状态，不能覆盖本节及第 3、6 节记录的新实现。

### 15.1 当前执行链和已确认正确的部分

```text
train_g1_amp.sh
  -> scripts/rsl_rl/train.py
  -> LeggedLab-Isaac-AMP-G1-WalkPerturbFinetune-v0
  -> G1WalkPerturbFinetuneEnvCfg + Walk runner cfg
  -> G1PerturbAmpEnv.step()
  -> 29 维 policy action 中 14 维手臂 action 被 pose target 覆盖
  -> JointPositionAction -> Isaac Sim
```

以下部分已经确认正确：

- Gym Train/Play 注册和 Python 执行入口存在；
- `g1_arm_pose_set.json` 的左右各 7 维数据会正确转换成 G1 action 使用的左右交错 14 维顺序；
- 候选 Nav2 `model_6596.pt` 和当前 Walk policy 都是 actor `96 -> 29`、critic input `297`，网络形状兼容；
- 参考数据路径相对于 `legged_lab` 解析；
- policy 观察当前完整 29DoF 状态，不含未来信息；
- discriminator 关节项和 torque/acc/action-rate 正则已经按名字限制到腿和腰。

这些检查只能排除“文件不存在、关节重排错误、网络维度不匹配”等问题，不能证明当前训练定义正确。

### 15.2 P0：现有训练实际没有从 Nav2 开始

`train_g1_amp.sh` 默认 `RESUME=False`。runner 配置中的 `load_policy_only=True` 本身不会加载任何文件；只有 `RESUME=True` 后，`train.py` 才解析 checkpoint 并调用 `AMPRunner.load()`。

三个本机 Walk smoke 的保存配置均为 `resume: false`。最新一次是：

```text
logs/rsl_rl/g1_amp/
2026-07-14_15-39-55_armhack_walk_nav2_checkpoint_smoke/params/agent.yaml

resume: false
load_policy_only: true
```

因此其中 `model_0.pt` 是随机初始化网络完成一次 PPO update 后的产物，不是 Nav2 微调。正式入口必须把“确实加载 checkpoint”作为硬前置条件，训练日志同时出现实际路径和 `Loaded policy-only AMP checkpoint` 才算成功。

候选本机基线：

```text
路径：logs/rsl_rl/g1_amp/
      2026-06-01_04-23-10_finetune_nav2_stage4_complex_only/model_6596.pt
大小：16,202,421 bytes
SHA-256：56cdb6aa66cbab53e9fa5b2bf312b3ea38e14164ebbb1f1e8850d9d003649f82
iteration：6596
asset：original g1_29dof
```

该 run 最后一个 TensorBoard 点为：mean episode length `1000/1000`、timeout `0.9935`、base-height termination `0.00085`、bad-orientation termination `0.00572`。这些训练统计用于证明它是一个合理候选起点；开始修改前仍应在当前机器固定回放并保存对照结果。

### 15.3 P0：固定姿态 reset 首步冲击

当前 `_reset_idx()` 对 `source="pose_set"` 只调用 `_sample_pose_targets()`，没有把选中姿态写到实际机器人关节状态。第一次应用目标发生在下一次 `step()` 的 action overwrite 中。相对于 G1 默认手臂姿态：

| pose | 最大单关节初始差 | 14DoF 差值 L2 | 最大差对应关节 |
|---|---:|---:|---|
| `pos2_down` | `0.1469 rad` | `0.3193 rad` | wrist roll |
| `pos1_back` | `1.0300 rad` | `2.2417 rad` | wrist pitch |
| `pos3_front` | `1.4600 rad` | `2.5000 rad` | elbow |

`pos3_front` 的肘从默认 `0.97` 到 `-0.49 rad`；按 `Kp=40` 估算会请求约 `-58.4 Nm`，超过 25 Nm 上限。`pos1_back` 的 wrist pitch 会请求约 `-41.2 Nm`，而该执行器上限只有 5 Nm。实际仿真会限幅，但限幅本身就是很强的隐藏冲击。

正确顺序应是：先选定 pose，再在第一个 observation 前把 14 个手臂 `q` 写成目标、`dq` 写成 0，并同步 position target 和 composed `last_action`。完整 episode 内不再切换 pose。

### 15.4 当前 Walk 与候选 Nav2 基线的实际差异

不能用“都调用 `Nav2RecordedVelocityCommandCfg`”概括两者。以下数值来自两个 run 已保存的 `params/env.yaml` / `params/agent.yaml`：

| 配置 | Nav2 `model_6596` | 当前 Walk smoke |
|---|---|---|
| 机器人资产 | original `g1_29dof` | `s3_g1_29dof` |
| 父任务语义 | Nav2 fine-tune | Command-Balanced V3 + Nav2 command 替换 |
| AMP motion | `accad_g1used_50hz` | `command_balanced_directional_50hz` |
| command scenario | `complex_turn` | 全部 family |
| command window | 4 s | 2 s |
| command scale | `(0.85, 0.75, 0.75)` | `(0.70, 0.55, 0.55)` |
| augmentation | `none,mirror_lr` | `none` |
| MPPI 权重 | `1.5` | `1.15` |
| linear/yaw tracking | 普通 tracking，`1.8/1.5` | adaptive tracking，`1.25/1.25` |
| torso linear/yaw tracking | `1.0/0.65` | `0.45/0.30` |
| `flat_orientation_l2` | `-1.0` | `None` |
| torso roll/pitch | `-0.04` | `-1.8` |
| torso vertical velocity | `-0.01` | `-0.38` |
| `termination_penalty` | `-200` | `None` |
| V3 directional rewards | 无 | speed-floor/double-air/double-stance/leak |
| PPO learning rate | `3e-5` | `1e-4` |
| AMP grad penalty | `20` | `10` |
| task/style lerp | `0.36` | `0.40` |
| baseline KL | 开启，`0.006` | 关闭 |

因此当前 Walk 一次改变了 asset、initial policy、AMP demo、command 分布、tracking、gait shaping 和稳定性权重。任何失败都无法单独归因到固定手臂姿态。

只把父类改成当前源码中的 `G1AmpNav2FinetuneEnvCfg` 也不自动等于 `model_6596` 的真实训练配置，因为该 checkpoint 的 Stage-4 参数包含额外的启动 override。实现时应把 checkpoint 配套的已解析 YAML 当作基线真值，逐项建立受测试保护的 Walk 配置。

### 15.5 奖励、AMP 和动作优化问题

第一版 Walk 应保持 Nav2 任务奖励，只做两类必要变化：

1. 删除策略无法控制的手臂 deviation/style penalty，并把 torque/acc/action-rate 正则限制到 15 个可控腿腰关节；
2. discriminator 若屏蔽手臂，需要新建 lower-body discriminator，旧 discriminator 权重不能直接复用。

不应在第一版同时把 torso penalty 放大几十倍。已有 Nav2 policy 已经具备 gait，初期更稳妥的做法是 style reward 暂设为 0、task/style lerp 设为 1，使用较低学习率和冻结 Nav2 baseline KL 保持行为；单姿态完成后，再单变量验证 lower-body AMP 是否改善仿人性。

当前 actor 仍输出 29 维以兼容 checkpoint，但 14 维手臂 action 在物理 step 前被丢弃，PPO 却仍将它们放入 log-prob、entropy、symmetry 和 KL。这会增加无效梯度方差。P0 阶段可以先靠 KL/低熵/低学习率减轻；后续应评估只对 15 个可控 action 计算 loss 的 mask，或冻结 arm output head，同时保持部署接口为 29 维。

### 15.6 recorded-command curriculum 是无效的

当前 `lin_vel_cmd_levels` 和 `ang_vel_cmd_levels` 只修改 command cfg 的 `ranges`。`Nav2RecordedVelocityCommand._scaled_command()` 实际执行的是：

```text
dataset[row] * command_scale -> command_clip_min/max
```

它不读取动态变化后的 `ranges`，所以第 109–123 行配置的 curriculum 不会改变训练命令。应删除这两个 term，或实现真正调节 `command_scale`、scenario filter/weight 或 clip 的 recorded-command curriculum，并为每阶段记录实际采样分布。

### 15.7 分阶段修改与训练方案

按以下顺序实现，每一步只改变一个主要变量：

| 阶段 | 双臂条件 | command/随机化 | 目标与晋级条件 |
|---|---|---|---|
| A0：Nav2 对照 | 原策略控制双臂 | checkpoint 原分布；固定 seed；无新随机化 | 在当前机器复现候选基线，冻结评估窗口和指标 |
| A1：接管零差异 | 固定为原默认/基线臂位 | 与 A0 相同 | 验证 action overwrite、reset 和部署链本身不使 gait 退化 |
| B：最小姿态 | `pos2_down`，episode 内固定 | 与 A0 相同；低 lr + baseline KL | 每个固定窗口的 20 s 完成率和速度误差接近 A0 |
| C1：靠后姿态 | `pos1_back`，episode 内固定 | 同上 | 单姿态逐项验收，不用混合平均掩盖失败 |
| C2：靠前姿态 | `pos3_front`，episode 内固定 | 同上 | 重点检查肘/腕饱和、torso 和足底响应 |
| D：三姿态混合 | reset 时按名字均衡采样 | 仍保持同一 Nav2 契约 | 三种 pose 各自成功率均达标 |
| E：鲁棒性扩展 | 三姿态 | 扩展 scenario、加入 mirror，再逐步恢复质量/摩擦/push | 每次只新增一种随机化并与 D 成对比较 |

如果最终部署资产是 S3，而候选 checkpoint 是 original G1，应在 A0 前插入一个独立的资产迁移阶段：先在没有 ArmHack 手臂覆盖时让 S3 Nav2 达标，不能把资产迁移和固定手臂同时训练。

### 15.8 已实现的专用训练脚本契约

当前已新增 `scripts/train_g1_armhack_walk.sh`。首次初始化模式会：

- 强制检查 Nav2 checkpoint 存在、大小和 SHA-256；
- 强制声明并校验 `ROBOT_ASSET`，不得静默使用通用脚本默认值；
- 强制 `RESUME=True`、policy-only load、iteration 从 0 开始；
- 用同一 checkpoint 建立 frozen baseline KL；
- 打印 checkpoint 绝对路径、asset、pose name、command profile、学习率、style 和 KL；
- 缺少任一必要参数时退出，不开始随机训练；
- 提供独立 `MODE=resume`，完整恢复已有 Walk optimizer/discriminator/iteration。

脚本和 reset 修复均已完成；正式训练只使用第 6.3/6.4 节的新命令，历史通用脚本命令继续禁止使用。

### 15.9 固定评估矩阵和测试缺口

每个正式 checkpoint 都要在完全相同的 command window/seed 上评估以下矩阵：

```text
Nav2 baseline + 原双臂控制
Nav2 baseline + default pose overwrite
Nav2 baseline + pos1/pos2/pos3 overwrite（未微调）
Walk checkpoint + pos1
Walk checkpoint + pos2
Walk checkpoint + pos3
Walk checkpoint + 三姿态均衡混合
```

每一格至少输出 20 s completion、timeout/base-height/bad-orientation、`vx/vy/wz` MAE/RMSE、torso roll/pitch RMS/P95、高度误差、feet slide、double-air ratio、gait symmetry、arm target/actual error 和 actuator saturation ratio。逐姿态结果必须单列。

现有 `test_g1_perturb_static.py` 主要是源码字符串/数据契约断言。下一轮需补：

- pose 在第一个 policy observation 前已经写入实际 arm `q/dq`；
- 同一 episode 内 pose 不变，并能按名称固定；
- Walk 必需奖励不为 `None`，与 Nav2 基线的允许差异只有白名单项；
- 专用脚本在 checkpoint 缺失、哈希错误或 asset 不匹配时失败；
- 日志确认 checkpoint 实际加载，而不是只检查 `load_policy_only=True` 字符串；
- 逐姿态 headless 集成回放和固定窗口指标导出。

### 15.10 部署和 ONNX 边界

当前 ONNX exporter 只导出 actor `96 -> 29`，不会把 `G1PerturbAmpEnv` 中的固定手臂 action override 一起导出。部署端必须复制相同语义：按关节名覆盖 14 个手臂目标，并把组合后实际施加的 action 作为下一帧 `last_action`。若直接把 Walk actor 当普通 Nav2 actor 使用，或只在 Isaac 环境里覆盖手臂，仿真训练与部署输入历史会不一致。

以上是修复前的审计结论。当前代码已经按 15.0 实施；新的训练与测试状态以第 1、6.3、8.4
和 8.5 节为准。即使 smoke 完成，也只证明训练链和 checkpoint 加载正确，不能替代逐姿态
20 s 完成率与速度跟踪验收。

## 16. Walk P0/P1 Robust 训练、测试与可视化（2026-07-18）

本节是当前 Walk P0/P1 实现的权威入口。第 15 节保留旧实现的审计过程；从现在起，正式续训使用
`LeggedLab-Isaac-AMP-G1-WalkRobustFinetune-v0`，不再把旧的
`LeggedLab-Isaac-AMP-G1-WalkPerturbFinetune-v0` 当作 P0/P1 训练任务。

本轮只新增或修改 Walk 专用代码、Walk runner 注册、Walk 脚本和本节文字。Stand 的环境类、训练脚本、
测试脚本、`armhack.md` 中的 Stand 任务说明以及现有 Stand checkpoint 均不参与该执行链。

### 16.1 已实现的 P0/P1 项目

| 优先级 | 已实现内容 | 当前代码位置与约束 |
|---|---|---|
| P0 | 恢复 AMP 奖励 | `G1WalkRobustFinetuneRslRlOnPolicyRunnerAmpCfg` 默认 `style=1.0`、`task/style lerp=0.85`；训练脚本按阶段升到 `style=5.0` |
| P0 | AMP 排除双臂 | 继续复用已测试的 15DoF 腿腰 discriminator joint mask；demo joint `pos/vel` 同样只取腿腰，双臂不会进入 discriminator |
| P0 | 修复 AMP demo 关节顺序 | ACCAD pickle 的非交错 DoF 顺序按名字严格重排成 `G1_LOCOMOTION_JOINT_NAMES` 后再选 15DoF；缺名直接失败，不再用错误 raw index |
| P0 | 双腕末端负载 | 左、右 `wrist_yaw_link` 使用两个独立 startup event，各自采样 `U(0, max_payload)`；`operation=add` 且 `recompute_inertia=True` |
| P0 | Nav2 + mode 混合命令 | 新增 `HybridNav2ModeVelocityCommand`；每个 4 s 窗口选择 Nav2 连续数据或一个命名 mode，策略仍只看到当前 `[vx, vy, wz]` |
| P1 | 分批恢复物理 randomization | `domain_base` 只加摩擦和 torso mass/CoM；`domain_actuator` 再加 gain/armature；`domain_link` 最后加 link-mass scale；三阶段均关闭 push |
| P1 | push 最后加入 | 只有 `robust` 阶段启用 interval push，避免强扰动掩盖 AMP、payload 或 command 问题 |
| P1 | 延长到约 10000 iteration | 以用户已有 WALK_3000 为起点，再续训 `1000+1000+500+1000+2000+500+500+250+250=7000` iteration |

策略输入维度仍为 96，输出仍为 29。没有新增 payload 数值、command source、mode id、未来 Nav2
窗口或未来双臂信息。双臂 14 维 action 仍由 `G1WalkPerturbAmpEnv` 在物理 step 前替换为本 episode
固定姿态；AMP 和腿腰正则不对这些不可控的双臂 action 计分。

### 16.2 新 Walk 执行链

```text
scripts/train_g1_armhack_walk.sh
  -> LeggedLab-Isaac-AMP-G1-WalkRobustFinetune-v0
  -> G1WalkRobustFinetuneEnvCfg
       -> G1WalkPerturbFinetuneEnvCfg（原 Nav2 reward + 腿腰 mask）
       -> HybridNav2ModeVelocityCommand
       -> 左/右 wrist_yaw_link 独立附加质量事件
  -> G1WalkPerturbAmpEnv（固定双臂覆盖）
  -> G1WalkRobustFinetuneRslRlOnPolicyRunnerAmpCfg
  -> lower-body/waist AMP + task reward + frozen model9996 baseline KL
```

主要文件如下：

- `source/legged_lab/legged_lab/tasks/locomotion/amp/config/g1_perturb/g1_walk_robust_env_cfg.py`：
  Walk-only 环境、混合命令、左右腕 payload 和确定性 Play 配置；
- `source/legged_lab/legged_lab/tasks/locomotion/amp/mdp/commands/hybrid_nav2_mode_velocity_command.py`：
  Nav2/mode 窗口选择、命令平滑、加速度限制和 source/mode 覆盖率 metrics；
- `source/legged_lab/legged_lab/tasks/locomotion/amp/config/g1_perturb/agents/rsl_rl_ppo_cfg.py`：
  Walk Robust AMP runner；
- `../rsl_rl/rsl_rl/runners/amp_runner.py`：仅在 Walk 第一阶段明确请求时，保留 policy/PPO 状态但重置旧 AMP discriminator、normalizer 和 optimizer；默认值为关闭，Stand 不触发；
- `scripts/train_g1_armhack_walk.sh`：锁定 P0/P1 课程和 checkpoint 身份；
- `scripts/vis_g1_armhack_walk.sh`：单条件 GUI/headless/video 回放；
- `scripts/test_g1_armhack_walk.sh`：固定 smoke/core/full 测试矩阵和 Markdown 汇总；
- `source/legged_lab/test/test_g1_armhack_walk_robust_static.py`：Walk/Stand 隔离、配置、数据和脚本回归。

### 16.3 混合 command 的准确含义

八个 mode 名称以实际 `task_sampling_config.json` 为准：

```text
stand
forward_slow
forward_normal
backward
lateral_left
lateral_right
turn_left
turn_right
```

每次重采样先建立合法的 Nav2 连续窗口，再按 `mode_probability` 决定该环境是否改用 mode target。
Nav2 和 mode 都经过相同的一阶平滑与线速度/yaw 加速度限制；reset 后从 0 平滑进入目标，不产生 command
阶跃。命令 term 记录 `source_nav2_ratio`、`source_mode_ratio` 和八个 `mode_*_ratio`，用于确认实际采样
覆盖，避免只改配置却没有改变真实训练分布。

训练阶段的 source 比例如下：

| 阶段 | Nav2 family | Nav2 / mode | Nav2 scale | mode scale |
|---|---|---:|---|---|
| `amp_warmup`、`amp_target`、`payload_half`、`payload_full` | `complex_turn` | 100% / 0% | `[0.85,0.75,0.75]` | 不采样 |
| `command` | 全 family | 80% / 20% | `[1,1,1]` | `[0.75,0.75,0.75]` |
| `domain_base`、`domain_actuator`、`domain_link`、`robust` | 全 family | 70% / 30% | `[1,1,1]` | `[1,1,1]` |

### 16.4 必须使用的环境

```bash
cd /home/user/Workspace/Humanoid/Locomotion/G1-Locomotion/legged_lab
source /home/user/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
which python
python -c 'import isaaclab, torch; print(torch.cuda.is_available())'
```

预期 Python 为 `/home/user/anaconda3/envs/env_isaaclab/bin/python`。不要用系统 Python，也不要在
Stand 正式训练进程仍运行时启动 Walk；专用脚本检测到任一 ArmHack Stand 训练进程会直接退出。

### 16.5 九阶段正式续训命令

正式训练应从用户已有 WALK_3000 的真实 `model_2999.pt` 做 full-state resume。必须先确认日志 checkpoint
与导出 checkpoint 都存在且 SHA-256 相同：

```bash
cd /home/user/Workspace/Humanoid/Locomotion/G1-Locomotion/legged_lab

WALK_3000_RUN='<实际的3000-iteration-Walk-run目录名>'
sha256sum \
  "logs/rsl_rl/g1_walk_perturb/$WALK_3000_RUN/model_2999.pt" \
  "ArmHack Checkpoints/WalkPerturbFinetune/$WALK_3000_RUN/model_2999.pt"
```

若两行不同，不能续训。脚本能够把旧 `g1_walk_perturb` 日志只读链接到新的
`g1_walk_robust` load namespace；不会修改旧 run。

旧 WALK_3000 在 `style=0` 时仍会更新 discriminator，而当时 ACCAD pickle 的 DoF 被 raw index 消费；
该顺序与 policy 的交错 G1 顺序不同。新的 `amp_warmup` 会完整恢复 actor、critic、PPO optimizer 和
iteration，但只在这一次把 AMP discriminator、其 observation normalizer 和 discriminator optimizer
重新初始化。后续八个阶段都完整恢复已经修正后的 AMP 状态，不再次重置。

第一阶段命令：

```bash
MODE=resume \
PHASE=amp_warmup \
RESUME_RUN="$WALK_3000_RUN" \
RESUME_CHECKPOINT=model_2999.pt \
POSE_NAME=random \
RUN_NAME=armhack_walk_p0_amp_warmup_from_walk3000 \
QUIET_TERMINAL=False \
bash scripts/train_g1_armhack_walk.sh
```

之后每一阶段都把 `PREV_RUN` 和 `PREV_CKPT` 替换成上一阶段真实输出；不要猜 checkpoint 编号：

```bash
PREV_RUN='<上一阶段实际run目录名>'
PREV_CKPT=$(find "logs/rsl_rl/g1_walk_robust/$PREV_RUN" \
  -maxdepth 1 -type f -name 'model_*.pt' -printf '%f\n' | sort -V | tail -1)
echo "$PREV_RUN / $PREV_CKPT"
```

按顺序执行：

```bash
# +1000：提高 lower-body AMP 占比
MODE=resume PHASE=amp_target RESUME_RUN="$PREV_RUN" RESUME_CHECKPOINT="$PREV_CKPT" \
POSE_NAME=random RUN_NAME=armhack_walk_p0_amp_target QUIET_TERMINAL=False \
bash scripts/train_g1_armhack_walk.sh

# +500：左右腕分别采样 U(0, 0.5 kg)
MODE=resume PHASE=payload_half RESUME_RUN="$PREV_RUN" RESUME_CHECKPOINT="$PREV_CKPT" \
POSE_NAME=random RUN_NAME=armhack_walk_p0_payload_half QUIET_TERMINAL=False \
bash scripts/train_g1_armhack_walk.sh

# +1000：左右腕分别采样 U(0, 1.0 kg)
MODE=resume PHASE=payload_full RESUME_RUN="$PREV_RUN" RESUME_CHECKPOINT="$PREV_CKPT" \
POSE_NAME=random RUN_NAME=armhack_walk_p0_payload_full QUIET_TERMINAL=False \
bash scripts/train_g1_armhack_walk.sh

# +2000：全 Nav2 family + 20% 中低速 mode
MODE=resume PHASE=command RESUME_RUN="$PREV_RUN" RESUME_CHECKPOINT="$PREV_CKPT" \
POSE_NAME=random RUN_NAME=armhack_walk_p0_command_expand QUIET_TERMINAL=False \
bash scripts/train_g1_armhack_walk.sh

# +500：只恢复 friction、torso mass 和 torso CoM；actuator/link/push 仍关闭
MODE=resume PHASE=domain_base RESUME_RUN="$PREV_RUN" RESUME_CHECKPOINT="$PREV_CKPT" \
POSE_NAME=random RUN_NAME=armhack_walk_p1_domain_base QUIET_TERMINAL=False \
bash scripts/train_g1_armhack_walk.sh

# +500：再恢复 actuator stiffness/damping 和 joint armature；link/push 仍关闭
MODE=resume PHASE=domain_actuator RESUME_RUN="$PREV_RUN" RESUME_CHECKPOINT="$PREV_CKPT" \
POSE_NAME=random RUN_NAME=armhack_walk_p1_domain_actuator QUIET_TERMINAL=False \
bash scripts/train_g1_armhack_walk.sh

# +250：再恢复左右 link-mass scale；只剩 push 关闭
MODE=resume PHASE=domain_link RESUME_RUN="$PREV_RUN" RESUME_CHECKPOINT="$PREV_CKPT" \
POSE_NAME=random RUN_NAME=armhack_walk_p1_domain_link QUIET_TERMINAL=False \
bash scripts/train_g1_armhack_walk.sh

# +250：最后启用 interval push
MODE=resume PHASE=robust RESUME_RUN="$PREV_RUN" RESUME_CHECKPOINT="$PREV_CKPT" \
POSE_NAME=random RUN_NAME=armhack_walk_p1_robust_push QUIET_TERMINAL=False \
bash scripts/train_g1_armhack_walk.sh
```

每条命令结束后都要重新设置 `PREV_RUN/PREV_CKPT`。`MAX_ITERATIONS` 表示本阶段追加的 iteration；只有
诊断时可以显式缩短。除 `amp_warmup` 外，`MODE=init` 会被拒绝，防止把后期强随机化误用于从
model9996 重新初始化。正式训练不得设置 `ALLOW_PHASE_INIT=True`。

如果用户报告的 WALK_3000 尚未同步到本机，只允许先验证从 model9996 初始化的代码链：

```bash
MODE=init PHASE=amp_warmup POSE_NAME=pos2_down \
RUN_NAME=armhack_walk_amp_warmup_from_model9996 \
QUIET_TERMINAL=False \
bash scripts/train_g1_armhack_walk.sh
```

该命令是新的训练起点，不等价于续训已有 WALK_3000。

### 16.6 checkpoint 与日志位置

```text
训练日志和完整状态：
logs/rsl_rl/g1_walk_robust/<timestamp>_<RUN_NAME>/

独立导出副本：
ArmHack Checkpoints/WalkPerturbFinetune/<timestamp>_<RUN_NAME>/model_*.pt
```

`MODE=resume` 会同时检查两个位置的 checkpoint 是否存在且 SHA-256 一致。通常它恢复 actor、critic、
PPO optimizer、AMP discriminator/normalizer 和 iteration；唯一例外是 `amp_warmup` 的一次性 AMP
重置，上节已经说明原因。不能用 policy-only load 伪装成阶段续训。

### 16.7 单条件测试和 GUI 可视化

先设置待测 checkpoint：

```bash
WALK_CKPT='/home/user/Workspace/Humanoid/Locomotion/G1-Locomotion/legged_lab/ArmHack Checkpoints/WalkPerturbFinetune/<run>/model_<iteration>.pt'
```

20 s Nav2 headless 测试，固定 `pos2_down`、左右各 1 kg：

```bash
CHECKPOINT="$WALK_CKPT" \
POSE_NAME=pos2_down \
LEFT_PAYLOAD_KG=1.0 RIGHT_PAYLOAD_KG=1.0 \
COMMAND_SOURCE=nav2 \
HEADLESS=True REAL_TIME=False FOLLOW_CAMERA=False MAX_STEPS=1000 \
bash scripts/vis_g1_armhack_walk.sh
```

固定侧步 mode 的 GUI 可视化：

```bash
CHECKPOINT="$WALK_CKPT" \
POSE_NAME=pos1_back \
LEFT_PAYLOAD_KG=0.0 RIGHT_PAYLOAD_KG=1.0 \
COMMAND_SOURCE=mode MODE_NAME=lateral_left \
HEADLESS=False REAL_TIME=True FOLLOW_CAMERA=True MAX_STEPS=1000 \
bash scripts/vis_g1_armhack_walk.sh
```

按训练后期的 70% Nav2 / 30% mode 分布可视化：

```bash
CHECKPOINT="$WALK_CKPT" \
POSE_NAME=random \
LEFT_PAYLOAD_KG=0.5 RIGHT_PAYLOAD_KG=0.5 \
COMMAND_SOURCE=hybrid MODE_PROBABILITY=0.30 \
HEADLESS=False REAL_TIME=True FOLLOW_CAMERA=True MAX_STEPS=1000 \
bash scripts/vis_g1_armhack_walk.sh
```

录制同一条件的视频：

```bash
CHECKPOINT="$WALK_CKPT" \
POSE_NAME=pos3_front \
LEFT_PAYLOAD_KG=1.0 RIGHT_PAYLOAD_KG=1.0 \
COMMAND_SOURCE=mode MODE_NAME=turn_right \
HEADLESS=True REAL_TIME=False FOLLOW_CAMERA=True \
MAX_STEPS=1000 VIDEO=True VIDEO_LENGTH=1000 \
bash scripts/vis_g1_armhack_walk.sh
```

### 16.8 固定测试矩阵

最短真实 smoke 运行 20 个 control step，并要求进程正常退出、到达 `max_steps` 且输出 tracking metrics：

```bash
CHECKPOINT="$WALK_CKPT" SUITE=smoke MAX_STEPS=20 \
bash scripts/test_g1_armhack_walk.sh
```

正式阶段验收应使用 20 s 窗口：

```bash
# 24 项：3 pose × 2 对称 payload × (Nav2 + 3 个代表 mode)
CHECKPOINT="$WALK_CKPT" SUITE=core MAX_STEPS=1000 \
bash scripts/test_g1_armhack_walk.sh

# 135 项：3 pose × 5 payload（含 0/1、1/0 不对称）× (Nav2 + 8 mode)
CHECKPOINT="$WALK_CKPT" SUITE=full MAX_STEPS=1000 \
bash scripts/test_g1_armhack_walk.sh
```

每个条件的原始日志和总表写入：

```text
ArmHack Checkpoints/WalkPerturbFinetune/Test Reports/
  <checkpoint名>_<SHA前12位>/<时间>_<suite>/
    summary.md
    <pose>__payload_L*_R*kg__<command>.log
```

测试脚本把以下任一情况判为失败：子进程非 0、Python traceback、Hydra job error、segmentation
fault、未到达指定 `max_steps`，或没有输出 `[METRIC] IsaacSim play task tracking:`。这些检查证明任务能
构造、checkpoint 能加载、环境能 step 并生成指标；20-step smoke 不能证明训练已经收敛。是否达到
ArmHack Walk 目标，仍必须比较 20 s completion、`vx/vy/wz` 跟踪、torso 稳定、足滑、AMP/style、
不同 pose/payload/mode 的 worst case。

### 16.9 本轮真实 smoke 记录

2026-07-18 在 `env_isaaclab`、RTX 4090 上完成以下真实验证：

1. `robust` 最强配置用 4 env 跑 1 iteration 成功。事件表同时出现基础六类物理 DR、左右两个
   wrist payload event 和 interval push；policy/critic/discriminator shape 分别为 `96`、`297`、
   `4×42`，其中 live/demo joint 项均为 `4×15`。AMP discriminator loss、style 路径和 baseline-KL
   均实际参加 update；诊断 checkpoint 为：

   ```text
   ArmHack Checkpoints/WalkPerturbFinetune/
   2026-07-18_14-01-18_armhack_walk_p0p1_robust_doforderfix_smoke_20260718/model_0.pt
   SHA-256: 73d6148ff0eff2aeaa326c34049781474a06a29f3318dbeaff07f7e643ef126d
   ```

2. 对该 checkpoint 执行 `MODE=resume PHASE=amp_warmup` 的 1-iteration 测试成功。日志明确打印：

   ```text
   Reset AMP discriminator, normalizer, and optimizer after full-state policy resume.
   ```

   证明一次性 AMP reset 分支实际执行，而不是只存在于配置字符串中。

3. `domain_base`、`domain_actuator`、`domain_link` 各用 2 env 跑 1 iteration，均正常保存 checkpoint。
   三份 Event Manager 表分别确认：

   ```text
   domain_base     : material + torso mass/CoM + wrist payload
   domain_actuator : 上述 + actuator gains + joint armature
   domain_link     : 上述 + link mass scale
   robust          : 上述 + interval push
   ```

4. 专用测试脚本的两个 20-step 条件均为 PASS：

   ```text
   Test Reports/model_0_73d6148ff0ef/20260718_140221_smoke/
     pos2_down + left/right 0/0 kg + Nav2

   Test Reports/model_0_73d6148ff0ef/20260718_140241_smoke/
     pos3_front + left/right 0/1 kg + lateral_left mode
   ```

   两项都正常到达 `max_steps=20` 并输出 task tracking 与 torso Important Metrics。

5. `pos1_back + 左右各 0.5 kg + 70/30 hybrid` 的 5-step 渲染与视频编码成功，视频为：

   ```text
   ArmHack Checkpoints/WalkPerturbFinetune/
   2026-07-18_14-01-18_armhack_walk_p0p1_robust_doforderfix_smoke_20260718/
   videos/play/rl-video-step-0.mp4
   ```

6. Bash 语法、Python compile、静态 Walk/Stand 边界回归和原有 perturbation 回归合计
   `15 passed`，`git diff --check` 通过。真实运行没有 Python traceback、Hydra override、shape、
   checkpoint load/save 或 CUDA 执行错误。Isaac Sim 4.5 仍打印其既有的 USD/MaterialX/IOMMU 等
   非阻断 warning；这些 warning 没有导致测试失败，也不是本轮 Walk 配置错误。

上述 `model_0.pt` 只用于证明代码链，不是训练结果。正式 WALK_3000→约 WALK_10000 仍应严格按
第 16.5 节执行，并用 `core/full` 的 20 s 矩阵判断收敛与鲁棒性。

## 17. Stand 真机鲁棒性第三阶段训练（2026-07-18）

本节只描述 Stand。新增代码没有修改 Walk 环境类、Walk command、Walk 训练/测试/可视化脚本或
Walk checkpoint。共享任务注册文件只增加一个 Stand task id，原有 Walk 注册项保持不变。

### 17.1 为什么需要第三阶段

第二阶段 Stand 已经学过 512 个范围内随机双臂姿态、`1.0x` minimum-jerk 姿态插值和左右腕末端
各自 `U(0,1 kg)` 的负载，但其专用训练脚本设置了 `RANDOMIZATION_STRENGTH=0`。因此当时实际关闭了
actuator gain、joint parameter 和其他通用物理随机化，`push_robot` 也在 Stand 配置中显式关闭；
非 timeout 摔倒惩罚仍为 `-200`。这可以解释为什么仿真固定协议下能够站立，但真机上的关节增益、
摩擦、惯量误差和未建模外力容易把策略推离训练分布。

第三阶段不是重写原任务，也不是从头训练。它从下面这个已经核验的第二阶段模型做 policy-only 初始化：

```text
ArmHack Checkpoints/StandPerturb/
2026-07-15_14-12-54_armhack_stand_randomized_payload_from_model2999_full_20260715/
model_2999.pt

SHA-256: 877e929d516cffe9131cc235477ceef4b226ec69e41c0f1c23e48816cfa28821
size: 14825781 bytes
```

### 17.2 新任务和代码边界

执行链为：

```text
scripts/train_g1_armhack_stand_robust.sh
  -> LeggedLab-Isaac-AMP-G1-StandRobust-v0
  -> G1StandRobustEnvCfg
       -> G1StandRandomizedPayloadEnvCfg
       -> G1StandPerturbEnvCfg
  -> G1PerturbAmpEnv
  -> G1StandPerturbRslRlOnPolicyRunnerAmpCfg
```

本轮新增文件：

- `source/legged_lab/legged_lab/tasks/locomotion/amp/config/g1_perturb/g1_stand_robust_env_cfg.py`：
  Stand-only reward、外力和关节 domain randomization；
- `scripts/train_g1_armhack_stand_robust.sh`：锁定输入 checkpoint、参数范围和正式训练默认值。

任务注册只在 `config/g1_perturb/__init__.py` 中新增
`LeggedLab-Isaac-AMP-G1-StandRobust-v0`。runner 继续复用 Stand 的
`g1_stand_perturb` 日志目录和 `ArmHack Checkpoints/StandPerturb` 导出目录，不会写入 Walk 目录。

### 17.3 真实训练分布

| 项目 | 默认分布 | 采样时机与解释 |
|---|---|---|
| 双臂姿态 | 512 个 14-DoF 姿态库 | 每个环境 reset 时独立采样起点/终点 |
| 双臂轨迹 | `1.0x`，过渡 `2–6 s` | 从第 0 iteration 开始全速 minimum-jerk；不再重复静态/低速课程 |
| 左腕负载 | `U(0,1 kg)` | startup 独立采样，质量加到 `left_wrist_yaw_link` 并重算惯量 |
| 右腕负载 | `U(0,1 kg)` | 与左侧独立采样 |
| torso 外力 | x/y/z 每轴独立 `U(-20,20) N` | 每环境独立 interval event，默认每 `2–5 s` 重采样 |
| torso 外力矩 | x/y/z 每轴独立 `U(-3,3) Nm` | 与外力同时写入 `torso_link` |
| actuator stiffness | 标称值 × `U(0.90,1.10)` | startup，每环境、每关节独立 |
| actuator damping | 标称值 × `U(0.90,1.10)` | startup，每环境、每关节独立 |
| joint friction | 标称值 × `U(0.80,1.20)` | startup，每环境、每关节独立 |
| joint armature | 标称值 × `U(0.90,1.10)` | startup，每环境、每关节独立 |
| 非 timeout 摔倒惩罚 | `-500` | 原值 `-200`；只在 `base_height`/`bad_orientation` 等非 timeout 终止帧触发 |

`random_torso_external_wrench` 不是速度 teleport：它调用
`mdp.apply_external_force_torque` 写入物理引擎。reset 模式下的零 wrench 事件会先清掉上一个 episode
的外力，之后 interval event 在双臂持续运动时重新采样。S3 G1 asset 已设置 8 次 position iteration、
4 次 velocity iteration；新 Stand task 额外设置
`sim.physx.enable_external_forces_every_iteration=True`，保证持续 wrench 在每次 solver iteration 施加。

为了让第三阶段的效果能够归因，以下宽泛随机化仍显式关闭：地面 material、base mass、base/torso CoM、
全身 link-mass scale 和 `push_by_setting_velocity`。腕端 payload 是唯一额外 link mass。后续如果需要加入
地面摩擦或机身 CoM，应另设阶段并逐项验收，不能在当前命令中无记录地全部打开。

策略接口没有变化：actor 输入仍为 96 维、输出仍为 29 维；双臂 14 维实际 action 继续由环境覆盖。
未来姿态、轨迹 phase、payload、外力和关节随机参数都没有进入 policy observation。

### 17.4 正式训练命令

每个新终端完整执行：

```bash
source /home/user/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
cd /home/user/Workspace/Humanoid/Locomotion/G1-Locomotion/legged_lab

which python
python scripts/tools/check_armhack_reference_data.py --stand-only

QUIET_TERMINAL=False \
RUN_NAME=armhack_stand_robust_wrench_joint_dr_from_model2999_full_20260718 \
bash scripts/train_g1_armhack_stand_robust.sh
```

默认是 `4096 env × 3000 iteration`，学习率 `5e-5`、entropy `0.002`、desired KL `0.01`，并用输入
`model_2999.pt` 建立 `0.001` 权重的 frozen baseline KL，降低强外力初期发生策略灾难性漂移的风险。
训练过程直接显示在终端；若需要写日志，可以显式设置：

```bash
QUIET_TERMINAL=True \
TRAIN_LOG_FILE="$PWD/logs/monitoring/armhack_stand_robust_full_20260718.log" \
RUN_NAME=armhack_stand_robust_wrench_joint_dr_from_model2999_full_20260718 \
bash scripts/train_g1_armhack_stand_robust.sh
```

TensorBoard：

```bash
source /home/user/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
cd /home/user/Workspace/Humanoid/Locomotion/G1-Locomotion/legged_lab
tensorboard --logdir logs/rsl_rl/g1_stand_perturb --port 6006 --bind_all
```

正式 checkpoint 输出到：

```text
logs/rsl_rl/g1_stand_perturb/<timestamp>_<RUN_NAME>/model_*.pt
ArmHack Checkpoints/StandPerturb/<timestamp>_<RUN_NAME>/model_*.pt
```

脚本允许通过同名环境变量收窄或扩大参数，但带有硬边界检查。真机问题没有定量标定前，不建议直接把
外力或关节参数范围扩大到默认值两倍；应先完成默认 3000 iteration，并用相同 deterministic Stand
测试集对比第三阶段前后的 termination、torso 6D 和足滑。

### 17.5 Smoke 命令与真实结果

用于开发检查的短命令为：

```bash
NUM_ENVS=8 \
MAX_ITERATIONS=1 \
RUN_NAME=armhack_stand_robust_wrench_joint_dr_smoke_v2_20260718 \
QUIET_TERMINAL=True \
TRAIN_LOG_FILE="$PWD/logs/monitoring/armhack_stand_robust_wrench_joint_dr_smoke_v2_20260718.log" \
FORCE_INTERVAL_MIN_S=0.04 \
FORCE_INTERVAL_MAX_S=0.08 \
bash scripts/train_g1_armhack_stand_robust.sh
```

这里只把外力 interval 缩短为 `0.04–0.08 s`，目的是让 24 control-step smoke 中 interval event
真正多次触发；正式训练必须恢复脚本默认的 `2–5 s`。2026-07-18 的最终 smoke 结果：

- 正确加载 SHA 前缀 `877e929d516c` 的第二阶段 `model_2999.pt`；
- Event Manager 的 startup 项为 `scale_actuator_gains`、`scale_joint_parameters`、
  `randomize_end_effector_payload`；interval 项为 `random_torso_external_wrench`；
- 保存的 `params/env.yaml` 明确为 `enable_external_forces_every_iteration: true`；
- Reward Manager 显示 `termination_penalty=-500.0`；
- policy/critic/action shape 保持 `96/297/29`；
- `ArmHack/random_motion_scale=1.0`、姿态库 `512`、课程阶段 `2`；
- 8 个环境共收集 192 step，完成 `Learning iteration 0/1`、PPO update、TensorBoard event 和 checkpoint 保存；
- 进程 exit code 为 0，没有 Python traceback、Hydra override、CUDA、Isaac 或 checkpoint 错误。

证据位置：

```text
logs/monitoring/armhack_stand_robust_wrench_joint_dr_smoke_v2_20260718.log

logs/rsl_rl/g1_stand_perturb/
2026-07-18_14-37-16_armhack_stand_robust_wrench_joint_dr_smoke_v2_20260718/
  params/env.yaml
  params/agent.yaml
  events.out.tfevents.*
  model_0.pt

ArmHack Checkpoints/StandPerturb/
2026-07-18_14-37-16_armhack_stand_robust_wrench_joint_dr_smoke_v2_20260718/model_0.pt
SHA-256: 64c3d4ed8ef9471f0190cfe94874985fa4041309a3434cf749e0f010da675a51
```

日志目录和独立 checkpoint 副本的 `model_0.pt` SHA-256 完全一致。该 smoke 只证明最终代码链能够运行；
因为 24 step 内没有完整 20 s episode，episode reward/termination 汇总为 0 不能证明“不摔倒”，
`model_0.pt` 也不能替代正式训练结果。

### 17.6 正式训练后的测试

第三阶段 checkpoint 的网络接口与第二阶段相同，因此正式训练结束后继续使用既有确定性 Stand 入口，
不需要也不应该修改 Walk 测试代码。新终端先指定实际最终模型：

```bash
source /home/user/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
cd /home/user/Workspace/Humanoid/Locomotion/G1-Locomotion/legged_lab

ROBUST_RUN='<第三阶段实际run目录名>'
ROBUST_CKPT="$PWD/ArmHack Checkpoints/StandPerturb/$ROBUST_RUN/model_2999.pt"
test -f "$ROBUST_CKPT" || { echo "checkpoint 不存在: $ROBUST_CKPT"; exit 1; }

# 默认 schema v5 全量无负载测试
CHECKPOINT="$ROBUST_CKPT" MODE=all PAYLOAD_KG=0 \
HEADLESS=True REAL_TIME=False MAX_STEPS=10404 \
bash scripts/vis_g1_armhack_stand_eval.sh

# 下垂到向前放平专项测试
CHECKPOINT="$ROBUST_CKPT" MODE=down_to_horizontal PAYLOAD_KG=0 \
HEADLESS=True REAL_TIME=False MAX_STEPS=1000 \
bash scripts/vis_g1_armhack_stand_eval.sh
```

至少还要补跑 `MODE=all PAYLOAD_KG=1.0`。最终判断必须把新模型与 SHA 前缀 `877e929d516c` 的输入模型
放在同一 schema、同一 payload 和同一 step 数下比较；重点看 termination/reset、torso 世界系 6D、
pitch RMS/最大值、水平漂移和逐关节波动。第三阶段正式训练尚未在本节启动，因此目前不能宣称真机稳定性
已经改善。
