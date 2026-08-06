# G1 Nav2 Behavior Finetune

这个任务是原始全身 G1 velocity-tracking/Nav2 策略的行为微调，不是双臂劫持任务。
Gym 入口为 `LeggedLab-Isaac-AMP-G1-Nav2BehaviorFinetune-v0`，环境类保持
`ManagerBasedAmpEnv`，actor 直接输出全部 29 个关节动作，接口保持
`policy=96`、`critic=297`、`action=29`。

## 受保护基线

唯一输入基线是：

```text
checkpoint/walk/model_10990.pt
size:   14826139 bytes
sha256: 1af3b722e1d07f8d7a40e32265cf67e46cfd2c74c50f6556cb369d2ea1e22c00
```

训练入口 `scripts/train_g1_amp_nav2_behavior.sh` 会在启动前和退出时验证这两个值，
并拒绝把输出或训练日志写入 `checkpoint/walk`。基线 discriminator 是每帧 42 维，
而本任务使用全身每帧 70 维，因此仅导入 `actor.*`；critic、动作噪声、PPO optimizer、
AMP discriminator、normalizer 和 discriminator optimizer 都从新状态开始。

新实验名是 `g1_amp_nav2_behavior`，独立 checkpoint 副本保存在
`legged_lab/Nav2BehaviorFinetune/<run>/`。iteration 从 0 开始。

## 训练

```bash
cd /home/user/Workspace/Humanoid/Locomotion/G1-Locomotion
NUM_ENVS=4096 MAX_ITERATIONS=3000 \
RUN_NAME=nav2_behavior_from_model10990 \
bash legged_lab/scripts/train_g1_amp_nav2_behavior.sh
```

命令分布由 40% 真实 Nav2 连续窗口和 60% 显式模式组成。显式模式覆盖严格静止、
`0.01–0.15 m/s` 微速前后/侧向/斜向、`0.05–0.25 rad/s` 微速原地转向、
普通原地转向、侧移、斜移和正常前向。训练保留 0.30 秒平滑与
`0.60/0.80` 的线速度/角速度加速度上限，只有显式 stand 立即给出精确零。

新增奖励包括无非零 deadband 的相对速度响应、纯 yaw 躯干侧倾惩罚、
command-conditioned touchdown cadence，以及基于实际足底矩形四角投影和 SAT
间距的严厉足底接近/交叉惩罚。

## Smoke

IsaacLab 最小 smoke：

```bash
NUM_ENVS=8 MAX_ITERATIONS=1 RUN_NAME=smoke_nav2_behavior \
bash legged_lab/scripts/train_g1_amp_nav2_behavior.sh
```

预期只生成 `model_0.pt`。对这个新 checkpoint 运行通用 G1 MuJoCo headless smoke：

```bash
CHECKPOINT=/absolute/path/to/Nav2BehaviorFinetune/<run>/model_0.pt \
bash legged_lab/scripts/test_g1_amp_nav2_behavior_mujoco.sh
```

该脚本离线导出 ONNX/TorchScript，验证 `96→29` 推理，再以固定
`[0.06, 0.0, 0.0]` 速度命令调用通用 `sim2sim_g1_amp_mujoco.sh`，要求指标均有限且
健康状态为 true。一次 PPO 更新只验证执行链，不代表低速、原地转向和交叉步行为已收敛。
