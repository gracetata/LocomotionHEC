# G1 Nav2 two-goal curriculum

This refinement starts from the protected `model_12995.pt` actor and frozen AMP
state. It targets only two new capabilities: collision-free pure lateral motion
and zero-linear-velocity in-place yaw. The 96/297/29 policy interface and the
280-D full-body AMP discriminator are unchanged.

## Training contract

- Sampling is 40% pure lateral, 40% exact-zero-linear pure yaw, and 20% recorded
  Nav2 retention commands.
- Baseline-policy KL uses scale `0.005` on the two specialization command
  families and `0.08` on retention samples. Only retention KL is subject to the
  `0.15` hard limit.
- The first actor hidden layer and AMP discriminator/normalizer stay frozen.
  Iterations 0-7 update only the fresh critic; remaining actor layers and action
  noise become trainable at iteration 8. Actor updates use `7.5e-6`, two PPO
  epochs, and clip `0.12` so the retention KL guard is approached gradually.
- AMP style reward remains at 15% on retention samples, but is disabled on the
  two specialization modes because the frozen forward-walking discriminator
  has no lateral or in-place-turn demonstrations.
- Checkpoints are saved every 10 iterations. A stage is limited to 60 iterations
  and must be evaluated before continuation.

## Reward contract

The authoritative lateral and yaw rewards are calculated from actual root
displacement and wrapped root-heading change between safe alternating
touchdowns. A second dense term uses finite differences of the same root pose to
provide a signal before the first touchdown: static behavior gets zero and a
periodic sway receives positive then negative progress, so it cannot substitute
for net displacement or heading change. Productive progress is quality-gated by
undesired forward/yaw leakage or planar drift. Standing still receives a bounded
response-shortfall penalty, and the stage-1 leak/drift penalties are themselves
bounded so exploratory motion cannot be dominated by an unbounded quadratic.
Before the policy can produce a touchdown, a bounded swing-cycle term rewards
only safe single-foot swings that alternate feet. Holding one foot up stops
earning credit after 0.35 s, and repeating the same foot earns nothing. Stage 1
disables the cadence ceiling; stage 2 restores it after locomotion exists.

The swept oriented sole geometry uses a 40 mm soft margin and a separate 25 mm
hard barrier. Productive touchdown rewards require clearance of at least 25 mm;
overlap receives the strongest penalty.

## Curriculum

The protected policy has a measured gait-onset threshold: lateral response is
about 0.16 m/s at `[0.20, 0.25, 0]` but only 0.05 m/s at `[0.10, 0.25, 0]`;
yaw response is about 0.40 rad/s at `[0.15, 0, 0.35]` but 0.11 rad/s at
`[0.10, 0, 0.35]`. A carrier-only PPO run did not move this threshold and
consumed most of the retention-KL budget, so it is diagnostic only and is not
the source of the strict-goal run.

The strict stage starts again from protected `model_12995.pt`. For each strict
lateral or strict pure-yaw rollout state, the frozen baseline actor is also
evaluated with the measured carrier command (`vx=0.20` for lateral and
`vx=0.15` for yaw). The student target is 60% of the action delta from the
baseline strict-command action toward that carrier action. This bounded
counterfactual teacher applies only to the 80% specialization samples; the 20%
Nav2 retention samples remain governed only by their baseline KL and frozen AMP
style anchor. Task rewards and drift/clearance guards still determine whether
the transferred gait becomes true lateral motion or true root-yaw motion.

Carrier stage uses `vx=0.15-0.20` for lateral and `vx=0.10-0.15` for yaw:

```bash
STAGE=carrier MAX_ITERATIONS=20 RUN_NAME=nav2_two_goal_carrier \
  bash legged_lab/scripts/train_g1_amp_nav2_two_goal.sh
```

Bridge stage lowers those ranges to `0.05-0.10` and `0.03-0.08`. It must load
the accepted carrier checkpoint with its exact size and hash; this selects a
full-state continuation automatically:

```bash
STAGE=bridge SOURCE_CHECKPOINT=/absolute/path/model_19.pt \
SOURCE_SIZE=$(stat -c '%s' /absolute/path/model_19.pt) \
SOURCE_SHA256=$(sha256sum /absolute/path/model_19.pt | awk '{print $1}') \
MAX_ITERATIONS=20 RUN_NAME=nav2_two_goal_bridge \
  bash legged_lab/scripts/train_g1_amp_nav2_two_goal.sh
```

Stage 1 is the authoritative restart. It uses `|vy|=0.25-0.45 m/s` and
`|wz|=0.35-0.60 rad/s`, with reduced
leak/drift/cadence penalties so the policy can initiate motion:

```bash
STAGE=1 COMMAND_BRIDGE_SCALE=0.20 MAX_ITERATIONS=20 \
RUN_NAME=nav2_two_goal_stage1_bridge_teacher \
  bash legged_lab/scripts/train_g1_amp_nav2_two_goal.sh
```

Evaluate deterministic MuJoCo at iteration 10 and 20. If neither strict
response improves, stop rather than adding iterations; change the bridge
strength or reward balance and restart from `model_12995.pt`.

Stage 2 uses `|vy|=0.10-0.40 m/s` and `|wz|=0.20-0.50 rad/s` and restores the
quality guards. It must start from an accepted stage-1 full-state checkpoint:

```bash
STAGE=2 SOURCE_CHECKPOINT=/absolute/path/model_59.pt \
SOURCE_SIZE=$(stat -c '%s' /absolute/path/model_59.pt) \
SOURCE_SHA256=$(sha256sum /absolute/path/model_59.pt | awk '{print $1}') \
MAX_ITERATIONS=60 RUN_NAME=nav2_two_goal_stage2 \
  bash legged_lab/scripts/train_g1_amp_nav2_two_goal.sh
```

Acceptance requires no hard-clearance violation or falls, signed lateral speed
of at least 0.18 m/s for `vy=0.25`, signed yaw rate of at least 0.25 rad/s for
`wz=0.35`, pure-yaw planar drift below 0.08 m/s, safe alternating touchdowns,
retention loss below 15%, and `bad_orientation < 0.02`.
