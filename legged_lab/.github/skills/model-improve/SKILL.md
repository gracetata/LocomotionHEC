---
name: model-improve
description: 记录算法迭代、超参数敏感度洞见、收敛特性与隐蔽 Bug 的避坑图鉴。
---

# 调试历程与模型进化 (Model Improve)

## 1. 错误图鉴 (Bug Ledger)
 - 错误现象：直接运行 `python scripts/rsl_rl/play.py ... --checkpoint .../model_600.pt` 会报 `ModuleNotFoundError: No module named 'legged_lab'`；同一 checkpoint 用 headless play 可正常加载并跑完 `--max_steps=10`。
 - 根本原因：当前仓库没有把 `source/legged_lab` 安装进默认 Python 环境，且 VS Code 自动选择的 `.conda/bin/python` 不是 IsaacLab 训练环境；T1 AMP 播放/训练必须使用 `/home/hecggdz/miniconda3/envs/env_leglab/bin/python` 并显式设置 `PYTHONPATH="$PWD/source/legged_lab${PYTHONPATH:+:$PYTHONPATH}"`。
 - 法则级解决方案：以后所有 `scripts/rsl_rl/*.py` 的 T1/G1 IsaacLab 命令都优先套用 env_isaaclab Python + `PYTHONPATH` 前缀；若只出现 `legged_lab` import 失败，先修环境前缀，不要怀疑 checkpoint 或任务注册。
 - 错误现象：T1 CmuWalkCore XML-zero/no-airtime 长训在 `model_600.pt` 附近任务奖励和 episode length 已稳定，但 AMP 判别器仍呈现 `disc_score≈-1`、`disc_demo_score≈+1`、style reward 接近死区。
 - 根本原因：已验证 CmuWalkCore 判别器没有直接输入 world X/Y 绝对位置，但每帧输入包含 root local rot tan-norm、base linear velocity、base angular velocity、joint pos、joint vel，且当前 reset 仍是 `reset_base`/`reset_robot_joints` 的默认随机初始化而不是 reference state initialization；`ref_state_init_root()` 还存在缺失 `motion_dataset` 参数的潜在 bug，启用 RSI 前必须先修。
 - 法则级解决方案：分析 T1 AMP 风格失效时按“root 速度捷径 + RSI 缺失 + discriminator 学习率/更新频率”优先排查；`base_contact` 在 T1 中已关闭，`base_height`/`bad_orientation` 后期触发很少，通常不是 600 附近 style 死区的主因。
 - 错误现象：T1 AMP 在 `grad_penalty_scale=20..500` sweep 中判别器长期饱和，`disc_score≈-1`、`disc_demo_score≈+1`、style reward 接近 0；gp500 可视化出现长单脚支撑/高抬腿，手臂 style 也没有被自然摆臂牵引。
 - 根本原因：当前 T1 task 命令被限定为 `lin_y=0, ang_z=0` 的前向行走，但 AMP motion weights 同时包含大量 turn、side-step、backwards、run/change-direction demo；判别器输入包含 root linear/angular velocity，因此能用与 task 冲突的根速度/转向信息轻易区分 demo 与 policy，不需要学习手臂细节。另一个风险点是 `PPOAMP.update()` 只用 policy discriminator observations 更新 AMP normalizer，demo observations 未参与 normalizer 统计，会进一步强化两类分布的可分性。
 - 法则级解决方案：style 不起作用时不要只继续加大 GP。先检查 demo motion distribution 与 command space 是否一致；若继续做 x-only walk，应筛选/重权 forward-walk demo 或从 discriminator 中移除/条件化 root velocity；同时把 AMP normalizer 改为 policy+demo 联合更新，再考虑调高 style scale、加入 key-body/手部特征或使用分组/加权 discriminator。
 - 错误现象：T1 retarget walk 数据在现有 motion manager 中无法直接训练，表现为 `joblib.load` 反序列化失败、缺少 `key_body_pos`/`loop_mode`、AMP motion 权重格式不匹配，以及 DoF 维度与 T1 机器人不一致。
 - 根本原因：当前 AMP/DeepMimic 管线默认假设 motion pickle 已经是 Isaac 风格标准格式，而 T1 `cmu_walk_50hz` 数据属于外部 retarget 输出，实际字段是 `local_body_pos`、`link_body_list`、`root_rot(xyzw)`、`dof_pos(27)`，且没有手工平均速度。
 - 法则级解决方案：以后接入外部 retarget 动作时，先在 motion loader 做统一标准化，再让任务配置复用该标准格式。不要把四元数重排、DoF padding、body 名选择、clip 平均速度估计散落在具体 task config 中。
 - 错误现象：T1 AMP 长训若出现 `base_contact`/`base_height` 终止异常、`feet_air_time` 长期为 0 或脚底接触时序异常，不应立刻判断为 AMP/PPOAMP 结构错误。
 - 根本原因：用户已观察到当前 T1 reference 数据集中脚底距地面的解算略有问题，且 T1 asset 初始化离地高度也可能与 URDF/collision 脚底高度存在偏差；这些会直接影响接触传感器、非法接触终止和脚步奖励。
 - 法则级解决方案：未来遇到 T1 接触/高度相关 bug 时，优先检查 `T1_29DOF_CFG.init_state.pos.z`、reference root height、`left_foot_link/right_foot_link` 到地距离、contact sensor body ids 和 `T1_WALK_TERMINATION_BODY_NAMES`，不要先改 `AMPRunner`、`PPOAMP` 或 discriminator loss。
 - 错误现象：T1 AMP 环境和网络均创建成功后，`AMPRunner -> PPOAMP` 初始化报 `unexpected keyword argument 'optimizer'`，继续过滤后还会遇到 `share_cnn_encoders`。
 - 根本原因：IsaacLab 2.3 的 `RslRlPpoAlgorithmCfg` 新增了当前 forked `PPOAMP` 不支持的 PPO 配置字段；如果 AMP cfg 直接继承该类，`agent_cfg.to_dict()` 会把这些字段传入算法构造函数。G1 README task 在当前资产命名下会更早卡在 `base_link`/contact body 正则，因此不能证明算法参数层没有问题。
 - 法则级解决方案：禁止通过修改 `rsl_rl` runner 或 `PPOAMP` 本体来绕过该问题。应在 `legged_lab.rsl_rl.RslRlPpoAmpAlgorithmCfg` 这个配置桥接层对齐 forked `PPOAMP.__init__` 签名，使 G1/T1 agent 继续共用同一 AMP runner/algorithm 结构。
 - 错误现象：T1 CmuWalkCore XML-zero 版本中 demo 的自然手臂 shoulder-roll 分布约为 `Left=-1.25 / Right=+1.28`，但当前 URDF 上肢限位仍沿用旧 baked-origin 语义，RSI 或物理策略会把 demo 的自然手臂状态 clamp 到错误范围。
 - 根本原因：把 URDF joint origin rpy 回退到 GMR XML zero 后，手臂 joint limit 也必须同步到 GMR XML `range`；只改 origin 而不改 limit，会让 XML-zero 坐标下的自然垂臂 demo 落到物理限位外，表现为 shoulder-roll/elbow/wrist 的 policy-demo gap 异常大。
 - 法则级解决方案：T1 XML-zero 资产中上肢限位必须与 `/home/user/Code/T1-Locomotion/GMR/assets/booster_t1_29dof/t1_mocap.xml` 保持一致；不要通过 retarget offset 修 demo 角度，优先修 asset limit/default init，并用测试锁住 `T1_LAB_BAKED_ZERO_OFFSETS == {}`。
 - 错误现象：启用 AMP RSI reset 后，首次 env reset 可能依次报 `active_terms` 不是 callable、`MotionDataTerm.sample_times()` 不接受 `truncate_time` 参数。
 - 根本原因：旧 `ref_state_init_root/dof` 路径长期未实际执行，API 已漂移；当前 `motion_data_manager.active_terms` 是 list 属性，`sample_times` 使用 `truncate_time_end`/`truncate_time_start`。
 - 法则级解决方案：RSI 必须通过单个函数一次采样同一 motion/time，然后同时写 root pose/velocity 和 dof pos/vel；`reset_from_ref` 事件顺序要排在 `reset_base`、`reset_robot_joints` 后面。已验证 `ref_state_init_subset` + `DEFAULT_RSI_RATIO=0.5` 可完成 1 iteration headless smoke。
 - 错误现象：T1 CmuWalkCore 使用自然垂臂默认 init、RSI=0.5、demo-static normalizer、`disc_learning_rate=1e-5` 长训到 `model_4999.pt` 后，任务奖励和 episode length 已收敛（`track_lin_vel_xy_exp≈0.94`、`track_ang_vel_z_exp≈0.86`、`episode_length≈1000`），但 `disc_score≈-0.999`、`disc_demo_score≈0.999`、style reward 仍接近 0。
 - 根本原因：该组合能稳定 locomotion，但不能阻止 AMP 判别器完全分离 policy/demo；最终 checkpoint 的 feature gap 仍集中在 XML-zero 上肢自然姿态相关维度，top gaps 为 `Right_Shoulder_Roll`、`Left_Shoulder_Roll`、`Right/Left_Elbow_Yaw`、`Wrist_Pitch/Yaw`，而 shoulder pitch 统计和左右反相关系与 demo 基本一致。
 - 法则级解决方案：下一轮不要继续只调 GP、RSI ratio 或判别器 LR。应优先把 AMP 目标改成“更难走捷径/更强手臂约束”的结构：例如移除或条件化 root velocity discriminator 输入、对 upper-body joint pos/vel 单独建 discriminator 或加权、加入手臂 imitation/key-body reward、降低/隔离 task reward 对手臂的自由度，并用 `analyze_t1_amp_discriminator_features.py` 先看 top gaps 是否从 shoulder-roll/elbow/wrist 消失。
 - 错误现象：同一 run 的 `model_400.pt` 可视化有自然摆臂，但后续 checkpoint 逐渐退化为 shoulder-roll 接近 0、elbow-yaw 绝对值变大的抬臂模式。
 - 根本原因：`analyze_t1_amp_checkpoint_evolution.py` 已量化：自然 shoulder-roll 误差从 `model_400≈0.10` 到 `model_1200≈0.59`、`model_2200≈1.03`、`model_4999≈0.95`；TensorBoard 中 `Episode_Termination/time_out` 在 iter200 已约 0.94，但 model400/model1000 仍较自然，因此 timeout 不是充分解释，style 崩塌更贴近 reward/AMP 目标结构在后期把手臂自由度推离 demo。
 - 法则级解决方案：分析“中期自然、后期漂移”时优先跑 `scripts/tools/analyze_t1_amp_checkpoint_evolution.py`，不要只看最终 checkpoint。对照训练可用 `LeggedLab-Isaac-AMP-T1-CmuWalkCore-CommandOnly-DemoNorm-v0`，该任务只保留 `track_lin_vel_xy_exp` 和 `track_ang_vel_z_exp` 两个环境 reward，AMP style reward 仍由 PPOAMP 注入。
 - 错误现象：CommandOnly 对照长训到 5000 后，任务追踪仍收敛（末尾 `track_lin_vel_xy_exp≈0.92`、`track_ang_vel_z_exp≈0.78`、`episode_length≈1000`），但判别器再次饱和（`disc_score≈-0.9997`、`disc_demo_score≈0.9999`），style 接近死区。
 - 根本原因：CommandOnly 并没有立刻导致手臂漂移；checkpoint 演进显示 `model_200..4600` 的 natural shoulder-roll error 约 `0.08..0.14`，到 `model_4800≈0.385`、`model_4999≈0.761` 才剧烈崩塌。因此“关闭非命令 reward”能显著延后自然摆臂崩塌，但不能单独阻止后期 policy/discriminator 共同找到非 demo 手臂模式。
 - 法则级解决方案：后续对照不要只看最终模型；CommandOnly 的可视化优先看 `model_400/model_1200/model_2400/model_4600/model_4800/model_4999`。若要防止最后阶段崩塌，需要加入显式手臂约束（如 `arm_style_prior`、upper-body imitation reward 或独立 upper-body discriminator），而不是继续单纯删环境 reward。
 - 错误现象：ArmPrior 长训仍无法让主 AMP discriminator 完全混淆（末尾 `disc_score≈-0.985`、`disc_demo_score≈0.986`），且 yaw tracking 比 CommandOnly 略弱（末尾 `track_ang_vel_z_exp≈0.62` vs CommandOnly `≈0.78`）。
 - 根本原因：监督式 `arm_style_prior` 是显式手臂姿态约束，不是从根本上修复 AMP 判别器输入捷径；它能把手臂约束住，但会和纯任务追踪存在权衡。
 - 法则级解决方案：若目标是“不再抬肘/肩平齐”，`CommandOnlyArmPrior` 是当前最有效对照：`model_4999` 的 natural shoulder-roll error 约 `0.087`，对比 CommandOnly `0.761`、原 natural-init run `0.947`；若目标是进一步提高 style/disc 分数，则下一步应改 discriminator 目标结构（如 upper-body discriminator 或 root-velocity 条件化），而不是继续只加 ArmPrior 权重。
## 2. 炼丹秘籍 (Hyperparameter Insights)
 - T1 walk 的第一版 AMP 配置限定为纯前向步行命令：`lin_vel_x > 0`，`lin_vel_y = 0`，`ang_vel_z = 0`。这样先把学习目标收敛到“稳定步行”，再考虑转向或侧移，更符合当前数据集和用户目标。
## 3. 性能演进记录
 - 已验证 `source/legged_lab/legged_lab/data/MotionData/cmu_walk_50hz` 中 109 个 pickle 全部满足当前兼容前提：`fps = 50`、`dof_pos.shape[1] = 27`、所选 8 个关键 body 全可从 `link_body_list` 映射、四元数单位范数正常、最短 clip 帧数为 110。
 - 已验证 T1 AMP walk 在 `num_envs=8`、`max_iterations=1`、`LEGGED_LAB_T1_AMP_MAX_MOTIONS=8` 下完成完整 rollout 和 PPOAMP update：policy 观测 96 维、critic 观测 297 维、动作 29 维、discriminator 输入 183 维。
 - 已验证 T1 AMP walk 可按现有成功 AMP 规模使用 `num_envs=8192` 启动训练；RTX 4090 上进入 iteration 0-2 时显存约 13.5GB/24.6GB、吞吐约 6.1e4-9.0e4 steps/s。`scripts/train_t1_amp_walk.sh` 默认应保持 `NUM_ENVS=8192`，并参考 README 长训命令使用 `MAX_ITERATIONS=50000`。
 - 已验证 G1 AMP 在当前分支存在 IsaacLab API 漂移导致的启动链路断裂：`g1_amp_env_cfg` 仍引用已从 AMP 基类移除的观测/事件字段（`key_body_pos_b`、`ref_key_body_pos_b`、`reset_from_ref`），并且 `randomize_rigid_body_com` 默认 body 含 `base_link`（G1 资产不存在）。这些会在训练前直接触发 `AttributeError` 或 scene entity resolve 报错。
 - 已验证 G1 AMP 的 `PPOAMP` 构造参数必须由桥接层严格裁剪：`RslRlPpoAmpAlgorithmCfg` 若继承 IsaacLab 新版 `RslRlPpoAlgorithmCfg`，会向 `PPOAMP` 透传不支持字段（如 `optimizer`）或与 runner 显式参数冲突（如 `multi_gpu_cfg`），导致算法初始化失败。桥接层应仅保留 `PPOAMP.__init__` 支持字段。
 - 已验证 G1 AMP 对称增强函数需与当前 policy 观测布局一致。当前 policy 维度为 96（`3+3+3+29+29+29`），若沿用旧的 `history/key_body` 索引映射会在训练迭代阶段触发 CUDA index/device assert。应按实时观测布局做 left-right 变换，旧布局仅作 fallback。
 - 已验证 T1 CmuWalkCore DemoNorm 新训练路径可用：`reset_from_ref` 在 reset 模式中顺序为 `reset_base -> reset_robot_joints -> reset_from_ref`，Reward Manager 中无 `feet_air_time`，AMP discriminator demo normalizer 日志为 `Initialized frozen AMP discriminator Demo normalizer from 4096 demo frames.`，`num_envs=64/max_iterations=1` smoke 成功完成 PPOAMP update。
 - 已验证 CommandOnly 对照任务 `LeggedLab-Isaac-AMP-T1-CmuWalkCore-CommandOnly-DemoNorm-v0` 可启动长训：RewardManager 只有两个 active terms（`track_lin_vel_xy_exp`、`track_ang_vel_z_exp`），8192 env 进入 iteration 后早期 AMP style reward 明显高于旧任务，但 `bad_orientation` 和 yaw tracking error 也较高，说明去掉杂项 reward 会放大 style 话语权但削弱稳定性约束。
 - 已训练轻量 T1 手臂风格先验 `logs/arm_style_prior/t1_cmu_walk_core_arm_prior.pt`：输入维度 32（root lin/ang vel + 13 个非手臂关节 pos/vel），输出维度 14（手臂关节 pos），MLP hidden `[64, 64]`，CMU walk core demo 上 normalized validation MSE≈0.194。可选任务为 `LeggedLab-Isaac-AMP-T1-CmuWalkCore-CommandOnlyArmPrior-DemoNorm-v0`，对应 reward 函数为 `mdp.arm_style_prior_exp`。
 - 已验证 `LeggedLab-Isaac-AMP-T1-CmuWalkCore-CommandOnlyArmPrior-DemoNorm-v0` 的 64-env smoke：RewardManager 正确包含 `track_lin_vel_xy_exp`、`track_ang_vel_z_exp`、`arm_style_prior` 三项，能加载 `logs/arm_style_prior/t1_cmu_walk_core_arm_prior.pt` 并完成 1 iteration PPOAMP update。
 - 已完成 `CommandOnlyArmPrior` 8192-env/5000-iteration 长训，run 目录为 `logs/rsl_rl/t1_amp/2026-05-07_01-44-24_cmu_walk_core_command_only_arm_prior_rsi05_disc1e5_demo_static_norm_iter5000`。`analyze_t1_amp_checkpoint_evolution.py` 显示 `model_200..4999` 的 natural shoulder-roll error 始终约 `0.07..0.10`，没有复现 CommandOnly 在 `4800->4999` 的手臂崩塌；`model_4999.pt` 已通过 headless play `--max_steps 200 --skip_export`。
