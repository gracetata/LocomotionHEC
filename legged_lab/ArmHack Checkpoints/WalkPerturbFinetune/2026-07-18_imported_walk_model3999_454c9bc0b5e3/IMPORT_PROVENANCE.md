# Walk checkpoint 导入记录

- 原始文件：`/home/user/Downloads/model_3999.pt`
- 导入时间：2026-07-18（Asia/Shanghai）
- SHA-256：`454c9bc0b5e38b2a9800c6faaa9e8ba6995f7d99bd3844155929a10a4fb8e2ff`
- 文件大小：`14,825,781 bytes`
- checkpoint iteration：`3999`
- actor：`96 -> 29`
- critic：`297 -> 1`
- 完整状态：包含 policy optimizer、AMP discriminator、AMP normalizer 和 discriminator optimizer
- 日志加载副本：`logs/rsl_rl/g1_walk_robust/2026-07-18_imported_walk_model3999_454c9bc0b5e3/model_3999.pt`
- 独立保存副本：`ArmHack Checkpoints/WalkPerturbFinetune/2026-07-18_imported_walk_model3999_454c9bc0b5e3/model_3999.pt`

两份导入副本及原文件的 SHA-256 已核验一致。该模型只用于 ArmHack Walk Robust 九阶段续训，不用于 Stand。
