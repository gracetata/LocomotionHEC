# G1 Locomotion ONNX Deployment Interface

本目录用于存放可直接真机部署的 ONNX 策略包：

```bash
checkpoint/<model_name>/locomotion.onnx
```

## 1. 导出 ONNX

```bash
CHECKPOINT=legged_lab/logs/rsl_rl/g1_amp/2026-06-17_03-48-38_s3_g1_29dof_command_balanced_directional_strict_armprior_v3_resume8997_1000/model_9996.pt \
  bash scripts/export_g1_amp_locomotion_onnx.sh
```

输出：

```bash
checkpoint/model_9996/locomotion.onnx
checkpoint/model_9996/locomotion.deploy.json
```

## 2. 真机部署入口

部署脚本只读取上一步的 ONNX 文件，不读取 checkpoint：

```bash
CONFIRM_REAL_ROBOT=I_UNDERSTAND NET=enp11s0 \
  ONNX_PATH=checkpoint/model_9996/locomotion.onnx \
  COMMAND_MODE=fixed CMD_INIT='[0.0, 0.0, 0.0]' \
  bash scripts/deploy_real_g1_amp_onnx.sh
```

启动流程固定为：

1. 释放高层运动模式后进入零力矩低层命令。
2. 终端按 `SPACE` 后进入阻尼模式。
3. 确认机器人初始姿态良好且双脚触地。
4. 再按 `SPACE` 后进入 ONNX 策略循环。
5. `CTRL-C` 会发送阻尼命令并退出。

## 3. 速度命令模式

固定速度：

```bash
COMMAND_MODE=fixed CMD_INIT='[0.3, 0.0, 0.0]'
```

Linux 摇杆，映射与 `scripts/sim2sim_g1_amp_mujoco.sh` 对齐：

```bash
COMMAND_MODE=joystick JOYSTICK_DEVICE=/dev/input/js0
```

运行时终端会打印策略实际收到的速度：

```text
[COMMAND joystick] vx= 0.420 m/s, vy=-0.030 m/s, yaw= 0.120 rad/s
```

导航 mock 链路，以 50Hz 伪装导航发送 `vx=0.7m/s`：

```bash
COMMAND_MODE=nav_mock NAV_MOCK_CMD='[0.7, 0.0, 0.0]'
```

## 4. 未来导航 UDP 接口

导航模块用 UDP 以 50Hz 发送当前帧速度命令。默认接收端：

```text
host: 机器人本机 IP
port: 15050
frame: 机器人 base frame
unit: vx/vy 为 m/s，yaw 为 rad/s 的 yaw rate
rate: 50Hz
timeout: 0.25s，无新包则命令归零
```

推荐 JSON payload：

```json
{"vx": 0.7, "vy": 0.0, "yaw": 0.0}
```

也兼容空格或逗号分隔：

```text
0.7 0.0 0.0
0.7,0.0,0.0
```

部署时使用真实导航输入：

```bash
CONFIRM_REAL_ROBOT=I_UNDERSTAND NET=enp11s0 \
  ONNX_PATH=checkpoint/model_9996/locomotion.onnx \
  COMMAND_MODE=nav_udp NAV_UDP_PORT=15050 \
  bash scripts/deploy_real_g1_amp_onnx.sh
```

安全限制默认裁剪到：

```text
vx  [-0.8, 0.8] m/s
vy  [-0.5, 0.5] m/s
yaw [-1.57, 1.57] rad/s
```
