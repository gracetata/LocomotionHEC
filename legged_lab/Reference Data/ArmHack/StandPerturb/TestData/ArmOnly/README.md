# ArmHack Stand 双臂专用测试数据

本目录只保存 ArmHack Stand 测试用的双臂目标，不包含根节点、腰部、腿部或任何全身姿态列。

所有可播放 CSV 的列严格固定为：

```text
time_s + 14 个 left/right shoulder、elbow、wrist 关节
```

目录含义：

```text
ArmOnly/
├── manifest.json
├── poses/
│   ├── arm_pose_catalog.csv
│   ├── representative/    # 从原始双臂 CSV 选出的姿态，每个固定保持 20 s
│   └── synthesized/       # 两个实测双臂姿态之间的固定种子凸插值，保持 20 s
├── trajectories/
│   ├── representative/    # 原始双臂高运动量窗口，离线降速到 0.25x
│   └── synthesized/       # 仅双臂姿态锚点之间的 minimum-jerk 轨迹
└── sequences/             # 用于 GUI 顺序检查或一次性统计报告
```

合成姿态不再叠加任意逐关节噪声，也不再混合三个以上姿态；每个姿态只在两个实测双臂姿态之间插值。合成轨迹只连接这种 14-DoF 双臂姿态锚点。机器人测试时腰腿仍由策略主动调节以维持平衡，但这些腰腿动作来自策略，不来自本目录的数据。

重新生成和只校验 Stand：

```bash
python scripts/tools/build_armhack_stand_visualization_suite.py
python scripts/tools/check_armhack_reference_data.py --stand-only
```

`manifest.json` 记录数据范围、源 CSV 哈希、固定种子、父姿态与插值权重以及每个生成 CSV 的 SHA-256。
