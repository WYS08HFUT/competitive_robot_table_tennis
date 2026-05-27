# Single Paddle Receive RL Training Blueprint

This file is now the active training blueprint for the current structure.
It keeps the work already completed and reorganizes the remaining stages around
the RL policy, not around more planner refactoring.

## Current Structure

The environment currently works like this:

- `BallPredictor` predicts reachable future ball states.
- `ImpactPlanner` chooses a nominal hit target.
- `PaddlePathPlanner` produces a nominal paddle command.
- In `planner_residual` mode, the policy adds a residual action every control
  step from serve start onward.

So the policy already acts throughout the episode.
What still makes the planner dominant is:

- planner-tracking reward,
- planner-heavy observations,
- full nominal command assistance.

## What Is Already Done

### Planner stack

- [x] `BallPredictor` implemented
- [x] `ImpactPlanner` implemented
- [x] `PaddlePathPlanner` implemented
- [x] `planner_only` / `planner_residual` control modes implemented

### Planner-only validation

- [x] headless planner-only batch evaluator added
- [x] planner-only pytest coverage added
- [x] planner-only contact gate defined
- [x] planner-only contact gate checked in the `mujoco` conda env

Artifacts:

- `tests/planner_only_eval.py`
- `tests/test_planner_only_validation.py`

Current result:

- planner-only is good enough for stable contact on the easy bucket,
- planner-only is not yet good at cross-net return,
- that is acceptable under the current task split.

## Current Task Split

Current intended division of labor:

- Planner:
  - get the paddle to a sensible hittable target,
  - make contact likely,
  - provide a useful prior.
- Policy:
  - tune the hit,
  - turn contact into cross-net return,
  - later improve landing quality,
  - eventually learn with less planner help.

This means the next work is primarily:

1. reward redesign,
2. training diagnostics,
3. staged fade-out of planner dependence.

## Core Decision Locked In

Planner-tracking reward should be removed first.

Reason:

- the policy already acts from serve start,
- but current planner-tracking reward still pays it to obey the nominal plan up
  to contact,
- that conflicts with the goal that the policy should own the hit tuning.

So the fade-out order will be:

1. remove planner-tracking reward first,
2. later reduce planner observation dependence,
3. only after that reduce control assistance.

## Reward Audit For The Current Structure

### Keep and strengthen for current Stage 1

- `legal_hit`
  - purpose: preserve interception and contact
  - recommendation: keep medium-strong
  - recommended weight: `2.5`

- `cross_net`
  - purpose: main Stage 1 neural objective
  - recommendation: keep strongest
  - recommended weight: `5.0`

### Keep but weaken a lot

- `send_forward`
  - purpose: weak post-contact shaping
  - recommendation: keep weak
  - recommended weight: `0.30`

- `lift`
  - purpose: weak anti-net shaping
  - recommendation: keep weak
  - recommended weight: `0.10`

- `action_rate_penalty`
  - purpose: mild smoothing only
  - recommendation: keep tiny
  - recommended weight: `0.001`

- `action_mag_penalty`
  - purpose: tiny regularization only
  - recommendation: keep tiny
  - recommended weight: `0.0005`

### Keep but secondary for Stage 1

- `landing`
  - purpose: later return-quality objective
  - recommendation: keep smaller than `cross_net`
  - recommended weight: `2.0`

### Disable first

- `survive`
  - recommendation: disable
  - recommended weight: `0.0`

- `tracking_pos`
  - recommendation: disable first
  - reason: planner-tracking reward is the first dependence to remove
  - recommended weight: `0.0`

- `tracking_rot`
  - recommendation: disable first
  - recommended weight: `0.0`

- `tracking_vel`
  - recommendation: disable
  - recommended weight: `0.0`

- `landing_shape`
  - recommendation: disable for early RL training
  - recommended weight: `0.0`

- `qvel_penalty`
  - recommendation: disable first
  - recommended weight: `0.0`

### Keep failure penalties

- `floor_penalty`
  - keep
  - recommended weight: `-2.0`

- `double_bounce_penalty`
  - keep
  - recommended weight: `-2.0`

- `multi_hit_penalty`
  - keep
  - recommended weight: `-2.0`

- `wrong_landing_penalty`
  - keep, but softer in early Stage 1
  - recommended weight: `-1.0`

- `net_fail_penalty`
  - keep and make meaningful
  - recommended weight: `-3.0`

- `out_of_bounds_penalty`
  - keep
  - recommended weight: `-2.0`

## Immediate TODO

### A. Reward redesign for Stage 1 RL

- [ ] In `mdp/rewards.py`, disable:
  - `survive`
  - `tracking_pos`
  - `tracking_rot`
  - `tracking_vel`
  - `landing_shape`
  - `qvel_penalty`
- [ ] Reweight:
  - `legal_hit -> 2.5`
  - `send_forward -> 0.30`
  - `lift -> 0.10`
  - `cross_net -> 5.0`
  - `landing -> 2.0`
  - `action_rate_penalty -> 0.001`
  - `action_mag_penalty -> 0.0005`
  - `floor_penalty -> -2.0`
  - `double_bounce_penalty -> -2.0`
  - `multi_hit_penalty -> -2.0`
  - `wrong_landing_penalty -> -1.0`
  - `net_fail_penalty -> -3.0`
  - `out_of_bounds_penalty -> -2.0`
- [ ] Keep `send_forward` and `lift` only as weak shaping, not main objectives.

### B. Add RL-focused diagnostics

- [ ] Add to runtime state / `info`:
  - cross-net success flag
  - first-hit ball speed after contact
  - first-hit outgoing direction quality
  - landing error when available
  - terminal reason breakdown
- [ ] Log these in `agents/mjlab/train.py`:
  - legal-hit rate
  - cross-net rate
  - opponent-landing rate
  - net-fail rate
  - wrong-landing rate
  - out-of-bounds rate

### C. Start Stage 1 PPO on easy bucket

- [ ] Train only on `easy`
- [ ] Judge progress using event rates, not total return alone
- [ ] Do not touch planner logic first unless contact quality degrades

## Reorganized Training Stages

### Stage 0: Planner Contact Baseline

Goal:

- verify nominal planner can create stable contact.

Current status:

- done.

Gate:

- planner-only contact gate passes on easy bucket.

### Stage 1: Policy Learns Cross-Net Return

Goal:

- keep planner as contact prior,
- remove planner-tracking reward,
- train policy to convert contact into cross-net return.

Reward emphasis:

- strong `legal_hit`
- strongest `cross_net`
- weak `send_forward`
- weak `lift`
- smaller `landing`
- no planner-tracking reward

Planner dependence:

- control assistance: full
- planner observations: full
- planner reward: removed

Recommended promotion criteria:

- legal-hit rate stays healthy
- cross-net rate becomes clearly nontrivial
- net-fail becomes less dominant

### Stage 2: Policy Improves Landing After Cross-Net Exists

Goal:

- once cross-net behavior exists, improve opponent-side landing quality.

Reward changes:

- keep `cross_net` strong
- increase `landing`
- optionally reintroduce a very weak landing-shape-like term only if it helps
  and does not create model bias

Planner dependence:

- control assistance: still full
- planner observations: still present
- planner reward: still absent

Recommended promotion criteria:

- stable cross-net rate
- opponent-landing rate becomes nontrivial
- wrong-landing begins to drop

### Stage 3: Reduce Planner Observation Dependence

Goal:

- make the policy rely less on planner hints and more on raw game state.

Changes:

- apply planner observation dropout or masking on some episodes
- keep planner residual control assistance unchanged at first

Recommendation:

- do not do this until Stage 2 is stable

### Stage 4: Reduce Control Assistance

Goal:

- let the policy increasingly own the whole stroke from serve start.

Changes:

- reduce nominal command dominance gradually
- optionally shrink residual-vs-nominal coupling over curriculum

Recommendation:

- this should be the last fade-out step, not the first

### Stage 5: Planner Optional / Distillation Stage

Goal:

- support a future version where the policy can play with minimal planner help.

Possible methods:

- planner dropout on full episodes
- teacher-student transfer
- mixed planner-assisted and planner-light curriculum

This is a future goal, not an immediate task.

## File-Level Priorities

### First priority

- `competitive_robot_table_tennis_rl/tasks/manager_based/single_paddle_receive/env_cfg.py`
- `competitive_robot_table_tennis_rl/tasks/manager_based/single_paddle_receive/mdp/rewards.py`

Reason:

- reward weights and enabled terms should be aligned first.

### Second priority

- `competitive_robot_table_tennis_rl/tasks/manager_based/single_paddle_receive/env.py`
- `competitive_robot_table_tennis_rl/tasks/manager_based/single_paddle_receive/mdp/events.py`
- `competitive_robot_table_tennis_rl/tasks/manager_based/single_paddle_receive/agents/mjlab/train.py`

Reason:

- we need RL-facing diagnostics before judging training.

### Lower priority for now

- `competitive_robot_table_tennis_rl/tasks/manager_based/single_paddle_receive/planning/paddle_path_planner.py`
- `competitive_robot_table_tennis_rl/tasks/manager_based/single_paddle_receive/planning/impact_planner.py`

Reason:

- only revisit these if RL runs show contact instability, planner thrash, or a
  regression in legal-hit rate.

## Practical Answer

Current focus should be:

- not more planner ambition,
- not more planner reward,
- but better RL reward and better RL diagnostics.

The next safe big change is:

- remove planner-tracking reward first,
- then train the policy to own cross-net return while planner still provides
  contact assistance.
