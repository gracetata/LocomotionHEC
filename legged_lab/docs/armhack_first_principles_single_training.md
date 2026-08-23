# ArmHack first-principles single-policy training

This training line contains exactly two independently deployed feed-forward
policies: one Stand actor and one Walk actor. Producer rollouts are used only
as reset states while training the other actor. There is no inference-time
router, gated expert, ensemble, hierarchy, or composite policy.

## Locked sources

- Stand: `2026-08-14_18-32-52_armhack_stand_low_torque_robust_explicit_3pose_2000_from_stage2_20260814/model_1999.pt`
  - SHA-256: `9ab48719840c98f1332693a56f58ed069463c0670737e339b90411985484a729`
- Walk: `WalkAnkleSpacingFinetune/base/2026-08-14_16-56-58_ankle30_base_full_20260814/model_199.pt`
  - SHA-256: `9d4583a535ea67086f429b20793a4f75dd00afbacab7c2aee5bc868be5a6e355`
- Nav2 command data:
  - SHA-256: `76a4516588b855351eb3eb8c2da26e291603876c1a4a1b9c7bacd77a53807b5a`

Both actors have a 96-D policy observation, hidden sizes 512/256/128, and a
29-D joint-position action. Formal continuation runs are rejected when fewer
than 2000 iterations are requested; short runs require `SMOKE=True`.

## Training cycle

1. Capture zero-command Walk states and train Stand for at least 2000 iterations.
2. Capture the trained Stand states and train Walk for at least 2000 iterations.
3. Capture the trained Walk zero-command takeover states and continue Stand for
   at least 2000 iterations.
4. Repeat producer refresh and 2000-iteration continuation if any MuJoCo
   acceptance item regresses.

The state library stores root pose/velocity, all joint positions/velocities,
the previous policy action, torso state, foot contacts, and foot contact force.
The consumer restores the generalized state and previous action; contact force
is reconstructed by physics and retained in the library as sampling provenance.

## Launcher

```bash
MODEL=stand NUM_ENVS=8192 MAX_ITERATIONS=2000 \
HANDOFF_STATE_LIBRARY=/absolute/walk_states.pt HANDOFF_RESET_PROBABILITY=0.25 \
bash legged_lab/scripts/train_g1_armhack_first_principles_single.sh

MODEL=walk NUM_ENVS=8192 MAX_ITERATIONS=2000 \
HANDOFF_STATE_LIBRARY=/absolute/stand_states.pt HANDOFF_RESET_PROBABILITY=0.25 \
bash legged_lab/scripts/train_g1_armhack_first_principles_single.sh
```

Acceptance requires smoke, long MuJoCo episodes, arm motion, external wrench,
low-speed and pure-yaw tracking, and repeated Stand/Walk handoffs. Training
success alone is never an acceptance result.
