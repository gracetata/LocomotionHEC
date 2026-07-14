# ArmHack Checkpoints

两个 ArmHack 任务的 checkpoint 按任务和训练 run 分开保存：

```text
ArmHack Checkpoints/
├── StandPerturb/
│   ├── BaselineModel7999/model_7999.pt
│   └── <run_name>/model_*.pt
└── WalkPerturbFinetune/
    ├── BaselineLocomotionModel9996/model_9996.pt
    └── <run_name>/model_*.pt
```

Stand 的固定起点仍为：

```text
StandPerturb/BaselineModel7999/model_7999.pt
757ea67854c1608b5d187079c4b9c2452db7ad17a56d0c7ef5ac749c043fe343
```

Walk 的权威 actor 和训练载体分别为：

```text
../checkpoint/model_9996/locomotion.onnx
SHA-256: 05fc45f89d89eb136225754f6a2fcacf5324d9dfd428d08ed75cc52f89b09be6

WalkPerturbFinetune/BaselineLocomotionModel9996/model_9996.pt
SHA-256: bc30bc5171d211fa414fbeab31452b92ad76ca7f6ad76a2417a6e7f7515a0fa6
```

ONNX 只包含 actor；`model_9996.pt` 是其元数据记录的原始 S3 G1 checkpoint，提供训练所需的 critic。`scripts/train_g1_armhack_walk.sh` 会校验两者哈希、接口、iteration，并逐元素确认 8 个 actor 张量完全一致。Stand 和 Walk 不共用基线文件。

`<run_name>` 与 `logs/rsl_rl/<experiment>/<run_name>` 中的目录名一致。训练时，RSL-RL 仍在原日志目录生成 checkpoint，并同时把同名文件复制到这里，避免破坏原有的续训和实验日志查找逻辑。

模型文件可能很大，`.pt`、`.pth` 和 `.ckpt` 已由仓库根目录的 `.gitignore` 排除；本说明和两个任务空目录会进入 Git。
