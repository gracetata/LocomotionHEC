# Nav2 Loopback G1 Task Distribution

这个文件夹保存了基于 Nav2 loopback sim 生成的 Unitree G1 `vx / vy / wz` 任务分布数据、增广数据、统计报告、轨迹图片和 RViz2 复现实跑脚本。

所有数据来自真实 Nav2 链路：

- `map_server`
- `nav2_loopback_sim`
- `/scan`，包括 loopback simulator 内注入的慢速移动障碍物 scan
- global planner
- local controller
- velocity smoother
- `/cmd_vel`
- `/odom`

不是离线手写曲线。

## 当前规模

当前快照：

- goal 总数：`503`
- 成功 goal：`445`
- 原始成功 `cmd_vel`：`331010` 行
- 增广 `cmd_vel`：`1324040` 行
- planner/controller 组合：`navfn_mppi`、`smac2d_mppi`、`navfn_dwb`、`smac2d_dwb`
- 新增 5 个不同地图尺寸随机障碍场景：10m、14m、18m、22m、26m。
- 新增 4 个专门急转场景：原地旋转、多 90 度转角、U-turn lanes、sawtooth 连续急转。
- 新增 12 个完全不同 seed 的 sample500 随机场景：每图 4 个不同 goal，四种组合共 192 条 goal，`192/192` 成功。

## 目录结构

```text
actual_raw_success/
  all_cmd_vel_success.csv          # 原始成功轨迹速度分布

actual_augmented/
  all_cmd_vel_augmented.csv        # 对称性增广后的速度分布

analysis/
  actual_manifest_all.csv          # 每个 goal 成功/失败记录
  actual_success_summary.json      # 按 combo/controller/planner/scenario 的统计
  actual_summary.json              # 全量 run 汇总
  complex_scene_summary.json       # 复杂/大地图/hard-turn 场景统计

docs/
  nav2_actual_replay_report.md     # 总报告
  complex_scene_review_zh.md       # 中文复盘和建议
  final_review_zh.md               # 从需求到分布、增广、问题和优化建议的最终复盘
  sample500_review_zh.md           # 500 条随机补量复盘
  dataset_integrity_audit_zh.md    # 数据真实性和可靠性审计

runs/<combo>/<scenario>/
  cmd_vel_samples.csv              # 单次场景原始速度和 odom
  planned_paths.csv                # Nav2 规划路径
  manifest.csv                     # 单场景 goal 结果

visualizations/
  summary/                         # 分布图和成功矩阵
  trajectories/                    # 每个 combo/scenario 的轨迹图

scripts/
  run_one_nav2_loopback.sh          # 运行一个 Nav2 loopback 场景
  run_rviz_demo.sh                 # 一键打开 RViz2 demo
  test_distribution_package.sh      # 数据包自检脚本
  add_random_obstacle_scenarios.py  # 生成不同尺寸随机障碍地图和 goal
  add_sharp_turn_scenarios.py       # 生成急转/旋转地图和 goal
  add_sample500_random_scenarios.py # 生成补到 500 条用的随机地图和 goal
  verify_dataset_integrity.py       # 反推原始 run、校验 raw/augmented/标签/相关性
```

## 最重要的数据文件

原始成功分布：

```text
actual_raw_success/all_cmd_vel_success.csv
```

增广分布：

```text
actual_augmented/all_cmd_vel_augmented.csv
```

每行核心字段：

```text
combo, planner, controller, scenario, goal_id,
scenario_family, goal_source, augmentation, topic, t,
vx, vy, wz, speed_xy,
odom_x, odom_y, odom_yaw
```

`topic=cmd_vel` 是 velocity smoother 后最终下发的速度。训练优先使用这个。

## 直接取矩阵

取原始 `N x 3` command matrix：

```python
import pandas as pd

root = "/mnt/d/userfiles/733438/My Documents/New project 2/nav2_task_distribution_g1/nav2_loopback_actual"

raw = pd.read_csv(root + "/actual_raw_success/all_cmd_vel_success.csv")
raw_cmd = raw[["vx", "vy", "wz"]].to_numpy()
print(raw_cmd.shape)
```

取增广 `N x 3` command matrix：

```python
import pandas as pd

root = "/mnt/d/userfiles/733438/My Documents/New project 2/nav2_task_distribution_g1/nav2_loopback_actual"

aug = pd.read_csv(root + "/actual_augmented/all_cmd_vel_augmented.csv")
aug_cmd = aug[["vx", "vy", "wz"]].to_numpy()
print(aug_cmd.shape)
```

取带 odom 的矩阵：

```python
raw_state_cmd = raw[["vx", "vy", "wz", "odom_x", "odom_y", "odom_yaw"]].dropna().to_numpy()
```

建议训练时按 `combo / scenario / goal_id` 分组后切连续窗口，不要把 `vx / vy / wz` 独立随机采样。

## 来源标签

现在 CSV 里有显式来源标签：

```text
combo             # navfn_mppi / smac2d_mppi / navfn_dwb / smac2d_dwb
planner           # navfn / smac2d
controller        # mppi / dwb
scenario_family   # baseline / complex_turn / moving_obstacle / large_success / hard_turn / random_obstacle / sharp_turn / sample500_random
goal_source       # goal 生成脚本来源，例如 sharp_turn_seed、random_obstacle_seed
augmentation      # none / mirror_lr / mirror_fb / rotate_180
```

例如只取新增急转原始样本：

```python
sharp = raw[(raw["scenario_family"] == "sharp_turn") & (raw["augmentation"] == "none")]
sharp_cmd = sharp[["vx", "vy", "wz"]].to_numpy()
```

只取 sample500 随机补量数据：

```python
sample500 = raw[(raw["scenario_family"] == "sample500_random") & (raw["augmentation"] == "none")]
sample500_cmd = sample500[["vx", "vy", "wz"]].to_numpy()
```

## 随机障碍地图

本轮增加的随机地图已经生成并跑入数据集：

```text
random_10m_weave_clutter
random_14m_slalom_clutter
random_18m_dense_islands
random_22m_mixed_turns
random_26m_long_mixed
```

这些地图用随机矩形障碍岛增加环境复杂度，同时预留多段 weave/slalom 走廊，目标是保持高成功率并扩展不同地图尺度下的长轨迹、急转和侧向速度分布。四种组合在这 5 个场景上都是 `2/2` 成功。

重新生成地图/goal：

```bash
./scripts/add_random_obstacle_scenarios.py
```

重新跑某个随机场景：

```bash
./scripts/run_one_nav2_loopback.sh navfn_mppi random_18m_dense_islands 2 false 240
```

## 急转样本

本轮增加的 sharp-turn 场景：

```text
sharp_rotation_bank_12m
sharp_l_turn_quadrants_14m
sharp_u_turn_lanes_18m
sharp_sawtooth_20m
```

这批场景新增 `39797` 行原始成功样本，sharp goal 成功率为 `58/64`。失败集中在 DWB 的 L-turn/U-turn 局部控制，失败轨迹不会进入训练 CSV。

重新生成地图/goal：

```bash
./scripts/add_sharp_turn_scenarios.py
```

重新跑急转场景：

```bash
./scripts/run_one_nav2_loopback.sh navfn_mppi sharp_rotation_bank_12m 6 false 180
./scripts/run_one_nav2_loopback.sh navfn_dwb sharp_u_turn_lanes_18m 4 false 240
```

## sample500 随机补量

本轮新增 12 个完全不同 seed 的随机地图，尺寸覆盖 10m、11.5m、12.5m、14m、15.5m、17m：

```text
sample500_rand_00_100
sample500_rand_01_115
sample500_rand_02_125
sample500_rand_03_140
sample500_rand_04_155
sample500_rand_05_170
sample500_rand_06_100
sample500_rand_07_115
sample500_rand_08_125
sample500_rand_09_140
sample500_rand_10_155
sample500_rand_11_170
```

每个场景 4 个不同 goal，四种 planner/controller 组合共 `192` 条 goal，结果 `192/192` 成功。新增 sample500 原始成功样本 `133806` 行。

重新生成地图/goal：

```bash
./scripts/add_sample500_random_scenarios.py
```

重新跑其中一个：

```bash
./scripts/run_one_nav2_loopback.sh navfn_mppi sample500_rand_05_170 4 false 180
```

## 自检

在当前 WSL 里运行：

```bash
cd "/mnt/d/userfiles/733438/My Documents/New project 2/nav2_task_distribution_g1/nav2_loopback_actual"
./scripts/test_distribution_package.sh
```

这个脚本会检查：

- 关键 CSV/JSON/报告/图片是否存在
- CSV 字段是否完整
- 原始和增广样本数量
- 成功/失败 goal 数量
- `vx / vy / wz / speed_xy` 范围
- `corr(speed_xy, abs(wz))`
- 常用场景地图和 goal 文件是否存在

如果想顺便实跑一个很短的 Nav2 smoke：

```bash
RUN_NAV2_SMOKE=1 ./scripts/test_distribution_package.sh
```

## RViz2 实时查看

在 WSL 任意目录直接运行：

```bash
"/mnt/d/userfiles/733438/My Documents/New project 2/nav2_task_distribution_g1/nav2_loopback_actual/scripts/run_rviz_demo.sh" navfn_mppi sample500_rand_05_170 1 180
```

参数：

```text
run_rviz_demo.sh <combo> <scenario> <max_goals> <timeout>
```

例子：

```bash
# 急转双发夹
"/mnt/d/userfiles/733438/My Documents/New project 2/nav2_task_distribution_g1/nav2_loopback_actual/scripts/run_rviz_demo.sh" navfn_mppi hard_double_hairpin_18m 1 180

# 近原地旋转
"/mnt/d/userfiles/733438/My Documents/New project 2/nav2_task_distribution_g1/nav2_loopback_actual/scripts/run_rviz_demo.sh" navfn_dwb hard_rotation_station 1 120

# 移动障碍物双发夹
"/mnt/d/userfiles/733438/My Documents/New project 2/nav2_task_distribution_g1/nav2_loopback_actual/scripts/run_rviz_demo.sh" smac2d_mppi hard_moving_double_hairpin 1 180

# pinwheel 回环
"/mnt/d/userfiles/733438/My Documents/New project 2/nav2_task_distribution_g1/nav2_loopback_actual/scripts/run_rviz_demo.sh" smac2d_mppi hard_pinwheel_loop 1 180

# 18m 随机障碍岛
"/mnt/d/userfiles/733438/My Documents/New project 2/nav2_task_distribution_g1/nav2_loopback_actual/scripts/run_rviz_demo.sh" navfn_mppi random_18m_dense_islands 1 220

# 26m 长距离随机障碍
"/mnt/d/userfiles/733438/My Documents/New project 2/nav2_task_distribution_g1/nav2_loopback_actual/scripts/run_rviz_demo.sh" smac2d_dwb random_26m_long_mixed 1 260

# 专门急转/旋转
"/mnt/d/userfiles/733438/My Documents/New project 2/nav2_task_distribution_g1/nav2_loopback_actual/scripts/run_rviz_demo.sh" navfn_mppi sharp_rotation_bank_12m 2 120

# U-turn lanes
"/mnt/d/userfiles/733438/My Documents/New project 2/nav2_task_distribution_g1/nav2_loopback_actual/scripts/run_rviz_demo.sh" navfn_dwb sharp_u_turn_lanes_18m 1 240

# sample500 随机地图
"/mnt/d/userfiles/733438/My Documents/New project 2/nav2_task_distribution_g1/nav2_loopback_actual/scripts/run_rviz_demo.sh" navfn_mppi sample500_rand_05_170 1 180
```

如果你是在 Windows PowerShell 里启动 WSL，用：

```powershell
wsl -- zsh -lc '"/mnt/d/userfiles/733438/My Documents/New project 2/nav2_task_distribution_g1/nav2_loopback_actual/scripts/run_rviz_demo.sh" navfn_mppi sample500_rand_05_170 1 180'
```

RViz 中重点看：

```text
/map
/scan
/plan
/odom
/tf
/cmd_vel
/cmd_vel_nav
```

如果 RViz2 没弹出来，先检查 WSL GUI：

```bash
echo "$DISPLAY"
rviz2
```

## 结论摘要

当前成功轨迹里 `speed_xy` 和 `abs(wz)` 全部呈负相关：

```text
navfn_mppi   ≈ -0.60
smac2d_mppi  ≈ -0.56
navfn_dwb    ≈ -0.52
smac2d_dwb   ≈ -0.55
```

含义：

- 高速平移时 yaw rate 通常很小。
- 急转或旋转时 Nav2 会主动降速。
- `vx / vy / wz` 不应该独立 uniform 采样。
- Isaac Sim gait tracking 训练应优先使用成功轨迹的连续窗口。

建议优先筛这些高价值成功场景：

```text
hard_rotation_station
hard_double_hairpin_18m
hard_moving_double_hairpin
hard_pinwheel_loop
complex_spiral_corridor
complex_lateral_zigzag
large_lateral_weave_fixed_yaw
random_18m_dense_islands
random_22m_mixed_turns
random_26m_long_mixed
sharp_rotation_bank_12m
sharp_u_turn_lanes_18m
sharp_sawtooth_20m
sample500_rand_05_170
sample500_rand_10_155
sample500_rand_11_170
```

失败场景不要混进正样本训练，可以作为边界评估和鲁棒性测试。
