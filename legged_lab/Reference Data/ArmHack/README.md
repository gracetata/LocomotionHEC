# ArmHack Reference Data

本目录集中保存两个 ArmHack 任务直接使用的外部姿态数据。训练代码通过仓库相对路径定位这些文件，不依赖用户名、主目录或当前工作目录。

## 目录

```text
ArmHack/
├── StandPerturb/
│   ├── raw/
│   │   └── g1_full_body_motion_sdk_50hz.csv
│   ├── g1_arm_trajectory_named_50hz.csv
│   ├── RandomizedTraining/
│   │   └── random_arm_pose_bank_seed20260715.json
│   └── TestData/ArmOnly/
│       ├── manifest.json
│       ├── poses/
│       │   ├── arm_pose_catalog.csv
│       │   ├── representative/*_hold20s_50hz.csv
│       │   ├── synthesized/*_hold20s_50hz.csv
│       │   └── randomized/*_hold20s_50hz.csv
│       ├── trajectories/{representative,synthesized,randomized}/
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

`RandomizedTraining/random_arm_pose_bank_seed20260715.json` 是 Stand 第二阶段续训的姿态库。构建脚本 `scripts/tools/build_armhack_stand_randomized_training_data.py` 先在训练可达的 20,109 帧中选出 64 个最远点覆盖锚点，再用 2–4 个完整 14-DoF 实测姿态的 Dirichlet 凸组合生成 448 个新姿态，合计 512 个。这种做法保留左右臂和同一手臂各关节之间的相关性，且每个关节都严格位于原训练数据的分量范围内。训练时在两个姿态间做五次 minimum-jerk 插值；名义时长随机取 2–6 s，若峰值关节速度超过原数据 P99 速度上限，环境会自动延长过渡时间。姿态库只含双臂，不含根节点或腰腿数据。

`TestData/ArmOnly/` 是确定性 Stand 测试集。所有回放 CSV 都严格只有 `time_s + 14 个双臂关节`，不含根节点、腰、髋、膝或踝。schema v5 保留原有 6 个代表姿态、3 个旧合成姿态、4 条实测轨迹和 3 条实测轨迹凸组合，又从新姿态库固定选出 8 个覆盖姿态和 6 条 minimum-jerk 插值轨迹。完整序列为 208.08 s / 10,404 个 50 Hz 控制步，共 59 个保持、轨迹或过渡阶段。原 `404.897585 s` 双臂下垂姿态仍显式排除。姿态库来源、固定样本、轨迹端点、每个 CSV 的 SHA-256 和详细时间线都记录在 `manifest.json`，并用于 torso 世界系 6D 测试图的阶段标注。

可视化运行时不再随机抽取数据，统一由以下入口顺序或逐项播放：

```bash
bash scripts/vis_g1_armhack_stand_eval.sh
MODE=representative_trajectory ITEM=1 bash scripts/vis_g1_armhack_stand_eval.sh
MODE=randomized_poses PAYLOAD_KG=0 bash scripts/vis_g1_armhack_stand_eval.sh
MODE=randomized_trajectories PAYLOAD_KG=1.0 bash scripts/vis_g1_armhack_stand_eval.sh
```

重新生成和校验：

```bash
python scripts/tools/build_armhack_stand_randomized_training_data.py
python scripts/tools/build_armhack_stand_visualization_suite.py
python scripts/tools/check_armhack_reference_data.py --stand-only
```

固定种子重建已验证能复现 512 个训练姿态和全部测试 CSV。当前姿态目录含 17 个单姿态（6 代表 + 3 旧合成 + 8 新随机覆盖）；轨迹目录含 13 条 5 s 轨迹（4 代表 + 3 旧合成 + 6 新姿态插值）。播放端固定使用 `csv_motion_scale=1.0`，不会再次缩放。

SHA-256：

```text
raw/g1_full_body_motion_sdk_50hz.csv:
b43256da27b11a593fc244ab2dd7fb899490a575d7749ed858ac342e3a208c50

g1_arm_trajectory_named_50hz.csv:
afe3819937ecfa19fae835b8cc77038378ec40a821acd0fdf2feef0054583601

RandomizedTraining/random_arm_pose_bank_seed20260715.json:
1ff56b9a59abaa01d5aadaa2f60685517b3f5db1dc3c2f06ad6acfd5ff3246a1
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
