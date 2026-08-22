# ArmHack First-Principles Training Contract

This document is the normative training and acceptance contract for the unified
`ArmHack Stand` and gated `ArmHack Walk` policies. Older experiment-specific
reward combinations are not acceptance criteria.

## Physical targets

- Both Stand and every Walk expert target a 0.30 m three-dimensional distance
  between the left and right ankle-link origins.
- Stand uses one contact-selected step per foot. The less-loaded foot moves
  first; phase completion is latched and cannot restart within the episode.
- Both final foot positions are reset-relative `S2 +/- 0.15 m` targets and both
  foot yaws match the reset pelvis yaw.
- Stand preserves reset-relative torso planar position and yaw.
- Walk pure-yaw has zero commanded translation and must not purchase turning or
  ankle width with planar drift or backward lean.

## Observation contract

The actor remains 96-dimensional. Script-controlled wrist action-history slots
carry additive reset-relative `dx`, `dy`, and `dyaw`; zero error retains the old
observation exactly. Two other wrist slots carry the contact-selected active
foot and lifted state. Scripted arm control owns these joints downstream.

## Reward groups

### Stand phase 0/1

- ordered lift/target/landing progress;
- one lift event and one completion event per foot;
- support-foot drift and wrong-order penalties;
- swing clearance band and gentle active-foot velocity;
- torso SE(2), ankle torque, and action continuity.

### Stand phase 2

- exact final foot XY and yaw targets;
- 0.30 m ankle distance;
- continuous double support and balanced vertical contact force;
- foot linear/angular velocity, planted-foot displacement, feet slide;
- lower-body joint velocity and action-rate;
- torso reset-relative SE(2) and low ankle torque.

### Walk all modes

- 0.30 m ankle-distance kernel is active for base, lateral, and yaw experts;
- useful-command velocity/yaw tracking in the +/-0.4 range;
- exact-zero stability, double support, and zero joint/foot motion;
- swing clearance without ground scraping;
- pure-yaw planar drift, yaw-rate error, and torso posture;
- action continuity from Stand-produced states.

## Curriculum

1. `smoke`: 128 environments, one iteration, no disturbances.
2. `nominal`: learn task geometry and command tracking without DR.
3. `randomized`: progressively enable dynamics, payloads, wrist/torso wrenches,
   pushes, arm poses, and arm motion.
4. `handoff`: mix 35% real producer states from the opposite policy. Commands
   start at exact zero; movement begins only after a stable zero window.

Each stage starts from a SHA-locked checkpoint and is accepted only if all
previous-stage tests remain non-degraded.

## Bidirectional producer datasets

- Walk-to-Stand records use `from=primary,to=secondary`.
- Stand-to-Walk records use `from=secondary,to=primary`.
- Each record contains root pose/velocity, all joint positions/velocities,
  previous action, command, arm pose through joint state, and the simulator tag.
- The consumer restores physics state and action history atomically.

## Runtime handoff

- Manual Walk-to-Stand first forces a zero target. The policy changes only
  after command norm, root linear speed, yaw-rate, and double-support gates are
  continuously satisfied for the configured hold window.
- Stand-to-Walk follows the same zero gate. Walk receives zero first and only
  then ramps to a user command.
- A pending handoff cannot be retriggered by repeated `M` presses.

## Strict MuJoCo acceptance

- no fall in nominal and 40 N repeated-handoff tests;
- exactly two final Stand steps and zero sustained extra air events;
- final 1 s ankle distance within 0.30 +/- 0.03 m for Stand and every moving
  Walk segment;
- Stand completion SE(2): XY <= 0.05 m and yaw <= 0.10 rad;
- planted-foot XY displacement <= 0.01 m;
- post-completion lower-joint peak-to-peak <= 0.08 rad;
- post-completion lower-joint velocity RMS <= 0.10 rad/s;
- post-completion lower-action delta RMS <= 0.015;
- pure-yaw planar drift <= 0.05 m/s with correct yaw-rate response;
- at least five policy switches in the repeated-handoff scenario.

Run `scripts/test_g1_armhack_first_principles_mujoco.sh` only after every smoke
stage succeeds. A checkpoint is never promoted on Isaac metrics alone.
