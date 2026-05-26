Below is the refactor blueprint I would use for your current single-paddle receive task.

The core change is:

```text
Current design:
  PPO policy directly learns paddle motion from reward shaping.

Refactored design:
  trajectory predictor → impact planner → path planner → residual RL policy → paddle actuator target.
```

The goal is to stop using reward as the main “teacher” for where/how to hit. The planner should provide the nominal stroke. RL should only learn residual corrections that improve robustness and contact quality.

Your current task already has a learned 6-DoF direct-paddle policy with delta-pose action, observation containing ball state/paddle state/predicted intercept/predicted landing/rule flags, and event rewards for legal hit, forward speed, lift, cross-net, and landing.  The refactor should keep the useful parts, but move the gross decision-making out of PPO.

## 1. Target architecture

Use this module layout:

```text
single_paddle_receive/
  env.py
  env_cfg.py
  control.py

  mdp/
    observations.py
    rewards.py
    terminations.py
    events.py
    transforms.py

  planning/
    ball_predictor.py
    impact_planner.py
    paddle_path_planner.py
    planner_types.py

  policy/
    residual_policy.py   # optional later, if separated from generic PPO policy

  assets/
    single_paddle_receive_dataset_frame.xml
```

Main data flow per control step:

```text
ball state, paddle state
        ↓
BallPredictor
        ↓
predicted trajectory
        ↓
ImpactPlanner
        ↓
planned hit time / hit pose / outgoing velocity / target landing
        ↓
PaddlePathPlanner
        ↓
nominal paddle command at current time
        ↓
RL residual action
        ↓
final paddle command = nominal command + residual
        ↓
MuJoCo actuator target
```

## 2. Phase 1: make the planner-only baseline work

Before training PPO, create a no-learning baseline. This is the most important refactor step.

Add config:

```python
@dataclass
class PlannerCfg:
    enabled: bool = True
    mode: str = "analytic"  # analytic first, neural later
    target_landing_x_range: tuple[float, float] = (-0.35, 0.35)
    target_landing_y_range: tuple[float, float] = (0.45, 1.05)
    hit_time_min: float = 0.12
    hit_time_max: float = 0.75
    hit_time_samples: int = 64
    min_hit_z: float = 0.25
    max_hit_z: float = 1.30
    receive_side_y: float = -0.90
    follow_through_time: float = 0.15
```

Add `planning/planner_types.py`:

```python
from dataclasses import dataclass
import numpy as np

@dataclass
class BallTrajectoryPoint:
    t: float
    pos: np.ndarray
    vel: np.ndarray

@dataclass
class HitPlan:
    valid: bool
    hit_time: float
    hit_pos: np.ndarray
    hit_euler: np.ndarray
    hit_vel: np.ndarray
    outgoing_vel_des: np.ndarray
    target_landing_xy: np.ndarray
    cost: float
```

Add `planning/ball_predictor.py`:

```python
class BallPredictor:
    def __init__(self, gravity=-9.81):
        self.g = np.array([0.0, 0.0, gravity], dtype=np.float32)

    def predict(self, pos, vel, times):
        points = []
        for t in times:
            p = pos + vel * t + 0.5 * self.g * (t * t)
            v = vel + self.g * t
            points.append((float(t), p, v))
        return points
```

Start with gravity-only. Do not add a neural predictor yet. Later, add drag/spin correction.

## 3. Phase 2: implement analytic impact planner

Add `planning/impact_planner.py`.

The planner should search feasible hit times and choose a reachable paddle pose.

Pseudo-code:

```python
class ImpactPlanner:
    def __init__(self, cfg, table_cfg, workspace_cfg):
        self.cfg = cfg
        self.table = table_cfg
        self.workspace = workspace_cfg

    def plan(self, ball_pos, ball_vel, paddle_pos, paddle_euler):
        target_xy = self.sample_or_set_target_landing()
        times = np.linspace(
            self.cfg.hit_time_min,
            self.cfg.hit_time_max,
            self.cfg.hit_time_samples,
        )

        best = None
        for t in times:
            p_hit, v_in = predict_ball_state(ball_pos, ball_vel, t)

            if not self.inside_workspace(p_hit):
                continue
            if p_hit[2] < self.cfg.min_hit_z or p_hit[2] > self.cfg.max_hit_z:
                continue

            v_out_des = self.solve_outgoing_velocity(
                start=p_hit,
                target_xy=target_xy,
                target_z=self.table.surface_z,
                flight_time=0.45,
            )

            hit_euler = self.estimate_paddle_orientation(v_in, v_out_des)
            hit_vel = self.estimate_paddle_velocity(v_in, v_out_des)

            cost = self.plan_cost(
                paddle_pos=paddle_pos,
                paddle_euler=paddle_euler,
                hit_pos=p_hit,
                hit_euler=hit_euler,
                hit_time=t,
            )

            if best is None or cost < best.cost:
                best = HitPlan(
                    valid=True,
                    hit_time=t,
                    hit_pos=p_hit,
                    hit_euler=hit_euler,
                    hit_vel=hit_vel,
                    outgoing_vel_des=v_out_des,
                    target_landing_xy=target_xy,
                    cost=cost,
                )

        if best is None:
            return HitPlan(valid=False, ...)
        return best
```

For outgoing velocity, use a ballistic solve. If the opponent target is `(x_target, y_target, z_table)`, and the hit point is `p_hit`, choose a flight time `T` and solve:

```python
v_out = (target_pos - p_hit - 0.5 * g * T**2) / T
```

At first, set `T` in `0.35–0.60 s` and reject impossible speeds.

For paddle orientation, approximate the paddle normal from incoming/outgoing velocity:

```python
n = normalize(v_out_des - v_in)
```

Then convert normal to yaw/pitch/roll. Keep this approximate. The residual policy can correct it.

## 4. Phase 3: add paddle path planner

Add `planning/paddle_path_planner.py`.

The path planner should produce a smooth nominal paddle command. Use minimum-jerk interpolation.

```python
def smoothstep5(tau):
    tau = np.clip(tau, 0.0, 1.0)
    return 10*tau**3 - 15*tau**4 + 6*tau**5
```

Command generation:

```python
class PaddlePathPlanner:
    def reset(self, current_pose, hit_plan, now):
        self.start_pose = current_pose
        self.hit_plan = hit_plan
        self.start_time = now

    def command(self, now):
        if not self.hit_plan.valid:
            return self.safe_ready_pose()

        tau = (now - self.start_time) / max(self.hit_plan.hit_time, 1e-3)
        s = smoothstep5(tau)

        pos_cmd = (1 - s) * self.start_pose.pos + s * self.hit_plan.hit_pos
        euler_cmd = interpolate_euler(self.start_pose.euler, self.hit_plan.hit_euler, s)

        return PaddleCommand(pos=pos_cmd, euler=euler_cmd)
```

After contact, follow through:

```text
continue 0.10–0.20 s along desired outgoing direction,
then reset to ready pose.
```

## 5. Phase 4: change action semantics

Current action is full delta-pose control. Refactor it to residual control.

Old:

```python
cmd_pose = previous_cmd_pose + action_scaled_delta
```

New:

```python
nominal_cmd = path_planner.command(t)
residual = scale_action(action)
cmd_pose = nominal_cmd + residual
cmd_pose = clip_to_workspace(cmd_pose)
```

Recommended residual limits:

```python
residual_limits = {
    "x": 0.08,
    "y": 0.08,
    "z": 0.08,
    "roll": 0.25,
    "pitch": 0.25,
    "yaw": 0.25,
}
```

This keeps PPO from becoming the planner.

Also add a mode switch:

```python
control_mode: Literal[
    "direct_delta",        # current behavior
    "planner_only",        # no RL residual
    "planner_residual",    # recommended
]
```

This lets you compare cleanly.

## 6. Phase 5: observation redesign

Current observation already includes ball state, paddle state, predicted intercept/landing, rule flags, and previous action.  Add planner outputs explicitly.

Recommended observation:

```python
obs = [
    # ball
    ball_pos,
    ball_vel,
    ball_ang_vel,

    # paddle actual
    paddle_pos,
    paddle_euler,
    paddle_lin_vel,
    paddle_ang_vel,

    # planner
    planned_hit_pos - paddle_pos,
    planned_hit_euler,
    planned_hit_time_remaining,
    planned_cmd_pos - paddle_pos,
    planned_cmd_euler,
    planned_outgoing_vel,
    target_landing_xy,

    # phase / rule state
    has_hit,
    own_side_bounce_count,
    paddle_hit_count,
    crossed_net,
    planner_valid,

    # previous action
    prev_action,
]
```

Normalize everything.

Important normalizations:

```text
position: divide by table length or workspace radius
velocity: divide by 5.0 m/s
angular velocity: divide by 30 rad/s
time_to_hit: divide by 1.0 s
angles: divide by pi
```

## 7. Phase 6: reward redesign

The new reward should evaluate the outcome. It should not teach the full movement.

### Pre-hit rewards: small tracking only

```python
r_pre = 0.0

if not has_hit and planner_valid:
    r_pre += 0.10 * exp(-norm(paddle_pos - planned_cmd_pos) / 0.08)
    r_pre += 0.05 * exp(-orientation_error(paddle_rot, planned_cmd_rot) / 0.25)
    r_pre += 0.05 * exp(-norm(paddle_vel - planned_cmd_vel) / 1.0)
```

Remove or reduce:

```text
survive: 0 or +0.001
intercept: replace with planner tracking
align: replace with planner orientation tracking
```

### Contact reward

```python
if first_legal_hit:
    r += 3.0

    v_out = ball_vel_after_hit

    # direction to target
    desired_dir = normalize(target_landing_3d - ball_pos_after_hit)
    actual_dir = normalize(v_out)
    r += 2.0 * clip(dot(desired_dir, actual_dir), 0.0, 1.0)

    # forward speed
    r += 1.0 * clip((v_out[1] - 0.8) / 2.5, 0.0, 1.0)

    # lift
    r += 0.5 * clip((v_out[2] - 0.2) / 1.2, 0.0, 1.0)
```

### Net reward

```python
if crossed_net_above_height:
    r += 4.0

if net_fail:
    r -= 5.0
    done = True
```

### Landing reward

```python
if legal_opponent_landing:
    r += 15.0
    err = norm(actual_landing_xy - target_landing_xy)
    r += 3.0 * exp(-err / 0.25)
    success = True
    done = True
```

### Failure penalties

Use stronger penalties than now:

```python
double_bounce: -6.0
multi_hit: -4.0
net_fail: -5.0
wrong_landing: -6.0
floor: -6.0
out_of_bounds: -6.0
planner_invalid_timeout: -2.0
```

### Regularization

Keep it weak early:

```python
r -= 0.001 * sum(action**2)
r -= 0.002 * sum((action - prev_action)**2)
r -= 0.0001 * sum(paddle_qvel**2)
```

Increase later only if the policy becomes too violent.

## 8. Phase 7: curriculum

Use this sequence.

### Stage 0: planner-only sanity

No RL. Run analytic predictor + impact planner + path planner.

Required before PPO:

```text
legal_hit_rate > 50%
cross_net_rate > 10–20%
bounce model stable
no repeated energy gain
```

If planner-only cannot hit, do not train PPO yet.

### Stage 1: synthetic easy balls

Use simple ball states, not full `serves.json`.

```text
speed: 0.8–1.8 m/s
spin: zero
lateral spread: small
height: reachable
```

Reward: first hit + direction + cross-net.

### Stage 2: filtered serves.json

Filter by reachability and speed.

```text
speed < 3.0 m/s
hit candidate exists in [0.15, 0.75] s
predicted hit z in [0.35, 1.20]
```

### Stage 3: harder serves

Increase speed, lateral spread, spin/noise.

### Stage 4: legal receive

Enable bounce-before-hit requirement.

```python
require_bounce_before_hit = True
```

Do this only after the policy can reliably return.

### Stage 5: OpenArm transfer

Use the virtual paddle planner output as:

```text
desired end-effector/paddle trajectory
```

Then train OpenArm to track it with IK/RL residual.

## 9. Event logging required after refactor

Add these logs. Without them, debugging will still be vague.

```text
planner/valid_rate
planner/hit_time_mean
planner/hit_time_std
planner/hit_pos_error_at_contact
planner/nominal_success_rate

episode_events/legal_hit_rate
episode_events/cross_net_rate
episode_events/opponent_landing_rate
episode_events/net_fail_rate
episode_events/wrong_landing_rate
episode_events/floor_rate
episode_events/out_of_bounds_rate
episode_events/double_bounce_rate
episode_events/multi_hit_rate

contact/ball_speed_before_hit
contact/ball_speed_after_hit
contact/outgoing_direction_score
contact/landing_error
contact/paddle_speed_at_hit
```

These should be logged per episode and rolling mean.

## 10. Refactor order by file

### `env_cfg.py`

Add:

```python
control_mode = "planner_residual"
sim_dt = 0.001
control_decimation = 10

planner_cfg = PlannerCfg(...)
reward_cfg = RewardCfg(...)
curriculum_cfg = CurriculumCfg(...)
```

Move reward weights into named groups:

```python
RewardCfg(
    tracking_pos=0.10,
    tracking_rot=0.05,
    tracking_vel=0.05,
    legal_hit=3.0,
    outgoing_direction=2.0,
    outgoing_forward=1.0,
    outgoing_lift=0.5,
    cross_net=4.0,
    landing=15.0,
    landing_accuracy=3.0,
    ...
)
```

### `env.py`

Add planner state:

```python
self.ball_predictor
self.impact_planner
self.path_planner
self.current_hit_plan
self.current_nominal_cmd
self.target_landing_xy
```

At reset:

```python
sample ball state
sample target landing
compute initial hit plan
reset path planner
```

At each control step:

```python
update ball prediction
if not has_hit:
    update or keep hit plan
nominal_cmd = path_planner.command(time)
final_cmd = control.apply_residual(nominal_cmd, action)
step simulation
detect events
compute reward
```

### `control.py`

Change from direct delta to modes:

```python
def action_to_command(action, mode, nominal_cmd, prev_cmd):
    if mode == "direct_delta":
        return prev_cmd + scale_delta(action)
    if mode == "planner_only":
        return nominal_cmd
    if mode == "planner_residual":
        return nominal_cmd + scale_residual(action)
```

### `observations.py`

Add planner fields.

Do not just add everything raw; add normalized relative quantities:

```python
planned_hit_pos_rel = planned_hit_pos - paddle_pos
planned_cmd_pos_rel = planned_cmd_pos - paddle_pos
target_landing_xy
time_to_hit
planner_valid
```

### `rewards.py`

Remove old intercept/align as primary terms.

Replace with:

```python
tracking_reward()
contact_outcome_reward()
post_hit_prediction_reward()
regularization_reward()
```

Keep event rewards in `env.py` or move event reward computation to `events.py`.

### `events.py`

Create this if you do not already have it.

Responsibilities:

```text
detect first paddle contact
detect table bounce side
detect net crossing
detect net fail
detect floor/out-of-bounds
classify terminal reason
```

This avoids mixing rule logic with PPO/reward code.

## 11. Acceptance criteria

Do not proceed to the next stage unless the current stage passes.

### Physics/contact acceptance

```text
Drop test: bounce height decays.
Scripted paddle hit: same initial state gives same outcome.
No random energy injection.
```

### Planner-only acceptance

```text
legal_hit_rate > 50%
cross_net_rate > 10–20%
out_of_bounds not dominant from planner itself
```

### RL residual Stage 1 acceptance

```text
legal_hit_rate > 70%
cross_net_rate > 40%
success_rate > 15%
```

### RL residual Stage 2 acceptance

```text
success_rate > 30% on filtered serves.json
wrong_landing and floor are no longer dominant
```

## 12. Minimal implementation sequence

Do not implement neural planner first. Use this order:

```text
1. Fix MJCF/contact timestep to stable 0.001.
2. Add BallPredictor.
3. Add ImpactPlanner.
4. Add PaddlePathPlanner.
5. Add planner_only mode.
6. Verify planner-only behavior.
7. Add planner_residual action mode.
8. Add planner fields to observation.
9. Replace reward with planner-tracking + ball-outcome reward.
10. Train PPO residual on easy synthetic serves.
11. Move to filtered serves.json.
12. Only then consider a neural impact planner.
```

## 13. Final recommendation

Do not refactor toward a “bigger neural policy” yet. Refactor toward a **structured controller with residual learning**.

The final task should be:

```text
Analytic modules:
  predict ball trajectory
  choose impact target
  generate paddle path

RL module:
  correct impact pose/timing/contact angle

Reward:
  score ball outcome, not paddle-chasing behavior
```

This gives you much lower debug cost because every failure becomes localized:

```text
misses ball       → planner/path/control issue
hits but net fail → outgoing velocity/orientation issue
crosses but out   → landing planner/contact issue
unstable result   → contact physics issue
poor residual     → RL/reward issue
```

That is the refactor path I would use.
