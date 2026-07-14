# ArmHack Checkpoints

两个 ArmHack 任务的 checkpoint 按任务和训练 run 分开保存：

```text
ArmHack Checkpoints/
├── StandPerturb/
│   ├── BaselineModel9996/model_9996.pt
│   └── <run_name>/model_*.pt
└── WalkPerturbFinetune/
    ├── BaselineLocomotionModel9996/model_9996.pt
    └── <run_name>/model_*.pt
```

Stand 当前训练入口的固定起点为：

```text
StandPerturb/BaselineModel9996/model_9996.pt
SHA-256: bc30bc5171d211fa414fbeab31452b92ad76ca7f6ad76a2417a6e7f7515a0fa6
size: 16,202,421 bytes；iteration: 9996；actor: 96→29；critic input: 297
```

Walk 的权威 actor 和训练载体分别为：

```text
../checkpoint/model_9996/locomotion.onnx
SHA-256: 05fc45f89d89eb136225754f6a2fcacf5324d9dfd428d08ed75cc52f89b09be6

WalkPerturbFinetune/BaselineLocomotionModel9996/model_9996.pt
SHA-256: bc30bc5171d211fa414fbeab31452b92ad76ca7f6ad76a2417a6e7f7515a0fa6
```

ONNX 只包含 actor；`model_9996.pt` 是其元数据记录的原始 S3 G1 checkpoint，提供训练所需的 critic。Stand 脚本会校验固定 `.pt` 的哈希和大小后做 policy-only 初始化，并把同一策略作为冻结 KL 基线。各任务仍使用各自目录中的基线副本，避免路径交叉。

`<run_name>` 与 `logs/rsl_rl/<experiment>/<run_name>` 中的目录名一致。训练时，RSL-RL 仍在原日志目录生成 checkpoint，并同时把同名文件复制到这里，避免破坏原有的续训和实验日志查找逻辑。Stand 回放报告写入被测模型同级的 `Test Reports/StandArmOnly/`，保证报告与 checkpoint 身份绑定。

模型文件可能很大，`.pt`、`.pth` 和 `.ckpt` 已由仓库根目录的 `.gitignore` 排除；本说明和两个任务空目录会进入 Git。
