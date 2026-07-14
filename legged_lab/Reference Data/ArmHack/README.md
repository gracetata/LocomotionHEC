# ArmHack Reference Data

本目录集中保存两个 ArmHack 任务直接使用的外部姿态数据。训练代码通过仓库相对路径定位这些文件，不依赖用户名、主目录或当前工作目录。

## 目录

```text
ArmHack/
├── StandPerturb/
│   ├── raw/
│   │   └── g1_full_body_motion_sdk_50hz.csv
│   └── g1_arm_trajectory_named_50hz.csv
└── WalkPerturbFinetune/
    └── g1_arm_pose_set.json
```

### StandPerturb

`raw/g1_full_body_motion_sdk_50hz.csv` 来自：

```text
/home/user/Workspace/whole_body_joints_20260708_143133.csv
```

原始文件包含 `q0..q28` 全身关节数据，顺序为 Unitree SDK/GMR/motor 顺序。源文件部分自然语言列包含未引用逗号，因此不直接用于训练。

`g1_arm_trajectory_named_50hz.csv` 是由 `scripts/tools/extract_armhack_stand_arm_csv.py` 生成的规范训练文件，只包含 `time_s` 和 14 个具名手臂关节。训练环境不再读取或执行 CSV 中的下肢目标，也不再依赖数字 q 列的隐式顺序。

SHA-256：

```text
raw/g1_full_body_motion_sdk_50hz.csv:
b43256da27b11a593fc244ab2dd7fb899490a575d7749ed858ac342e3a208c50

g1_arm_trajectory_named_50hz.csv:
afe3819937ecfa19fae835b8cc77038378ec40a821acd0fdf2feef0054583601
```

### WalkPerturbFinetune

`g1_arm_pose_set.json` 保存三组双臂姿态：`pos1_back`、`pos2_down`、`pos3_front`。每个环境 reset 时随机选择一组，覆盖策略输出中的 14 个手臂 action。

动态任务的速度命令由 `UniformVelocityCommandCfg` 在配置范围内生成，不再依赖外部 Nav2 大 CSV。策略输入仍保留标准 `vx/vy/wz` 速度命令接口，可用于后续 Nav2 command 跟踪微调。
