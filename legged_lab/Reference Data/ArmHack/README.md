# ArmHack Reference Data

本目录集中保存两个 ArmHack 任务直接使用的外部姿态数据。训练代码通过仓库相对路径定位这些文件，不依赖用户名、主目录或当前工作目录。

## 目录

```text
ArmHack/
├── StandPerturb/
│   ├── raw/
│   │   └── g1_full_body_motion_sdk_50hz.csv
│   ├── g1_arm_trajectory_named_50hz.csv
│   └── TestData/ArmOnly/
│       ├── manifest.json
│       ├── poses/
│       │   ├── arm_pose_catalog.csv
│       │   ├── representative/*_hold20s_50hz.csv
│       │   └── synthesized/*_hold20s_50hz.csv
│       ├── trajectories/{representative,synthesized}/
│       └── sequences/*_arm_only_*_50hz.csv
└── WalkPerturbFinetune/
    ├── g1_arm_pose_set.json
    └── nav2_cmd_vel_raw_success.csv
```

### StandPerturb

`raw/g1_full_body_motion_sdk_50hz.csv` 来自：

```text
/home/user/Workspace/whole_body_joints_20260708_143133.csv
```

原始文件包含 `q0..q28` 全身关节数据，顺序为 Unitree SDK/GMR/motor 顺序。源文件部分自然语言列包含未引用逗号，因此不直接用于训练。

`g1_arm_trajectory_named_50hz.csv` 是由 `scripts/tools/extract_armhack_stand_arm_csv.py` 生成的规范训练文件，只包含 `time_s` 和 14 个具名手臂关节。训练环境不再读取或执行 CSV 中的下肢目标，也不再依赖数字 q 列的隐式顺序。

`TestData/ArmOnly/` 是从完整规范轨迹离线构造的确定性 Stand 测试集，不是新的训练分布。所有回放 CSV 都严格只有 `time_s + 14 个双臂关节`；不含根节点、腰、髋、膝或踝。`scripts/tools/build_armhack_stand_visualization_suite.py` 会遍历完整轨迹并固定选出 6 个代表姿态、4 段代表运动窗口；随后以种子 `20260714` 对两组实测双臂姿态做凸插值，生成 3 个双臂姿态，并对两段等长实测 `1.0x` 双臂轨迹做逐帧凸组合，生成 3 条合成 `1.0x` 轨迹。合成过程不会扰动或生成任何腰腿关节。生成结果、父数据与权重、源时刻/窗口、速度倍率和每个 CSV 的 SHA-256 均记录在 `manifest.json`。

可视化运行时不再随机抽取数据，统一由以下入口顺序或逐项播放：

```bash
bash scripts/vis_g1_armhack_stand_eval.sh
MODE=representative_trajectory ITEM=1 bash scripts/vis_g1_armhack_stand_eval.sh
```

重新生成和校验：

```bash
python scripts/tools/build_armhack_stand_visualization_suite.py
python scripts/tools/check_armhack_reference_data.py --stand-only
```

固定种子重建已经验证能逐文件复现全部 22 个生成 CSV 的 SHA-256，其中 1 个是姿态目录表、21 个是可回放 CSV。9 个单姿态文件均保持 20 s；4 条代表轨迹和 3 条合成轨迹均为 5 s、50 Hz、原始 `1.0x` 时间尺度。播放端固定使用 `csv_motion_scale=1.0`，不会再次缩放。

SHA-256：

```text
raw/g1_full_body_motion_sdk_50hz.csv:
b43256da27b11a593fc244ab2dd7fb899490a575d7749ed858ac342e3a208c50

g1_arm_trajectory_named_50hz.csv:
afe3819937ecfa19fae835b8cc77038378ec40a821acd0fdf2feef0054583601
```

### WalkPerturbFinetune

`g1_arm_pose_set.json` 保存三组双臂姿态：`pos1_back`、`pos2_down`、`pos3_front`。训练脚本默认固定一个具名姿态完成整个 episode，也可用 `POSE_NAME=random` 在 reset 时抽样。reset 会在首个 observation 前同步手臂 `q/dq`、执行器目标和 action history，避免第一步从默认姿态跳变。

`nav2_cmd_vel_raw_success.csv` 是动态任务的外部速度分布，来自 HEC-5090：

```text
/home/hecggdz/workspace-zwd/legged_lab/nav2_loopback_actual/
actual_raw_success/all_cmd_vel_success.csv
```

本机保存位置：

```text
Reference Data/ArmHack/WalkPerturbFinetune/nav2_cmd_vel_raw_success.csv
```

数据包含 331,010 行 Nav2 成功轨迹命令、445 个按 planner/controller/scenario/goal 分组的连续窗口。核心范围为：

```text
vx: [-0.2, 0.6] m/s
vy: [-0.3, 0.3] m/s
wz: [-0.5187280178070068, 0.6] rad/s
augmentation: none
```

SHA-256：

```text
76a4516588b855351eb3eb8c2da26e291603876c1a4a1b9c7bacd77a53807b5a
```

该 83 MiB CSV 已加入 `.gitignore`，不会推送到 GitHub。新机器需按上述来源单独复制到同一相对位置，然后运行：

```bash
python scripts/tools/check_armhack_reference_data.py
```

动态训练使用 `Nav2RecordedVelocityCommandCfg` 按轨迹组截取 `complex_turn` 连续 4 秒窗口，并保留 Nav2 Stage-4 的缩放、裁剪、平滑和限加速度逻辑。当前文件是原始成功分布，仅包含 `augmentation=none`；loader 在内存中从每个 raw 组生成 `(vx,-vy,-wz)` 的 `mirror_lr` 组，不改写源 CSV，也不需要第二份大文件。已在 HEC-5090 的整个 `workspace-zwd` 中确认没有 `all_cmd_vel_augmented.csv`。
