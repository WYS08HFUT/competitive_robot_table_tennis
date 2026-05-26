For `serves.json`, design the paddle task as a **single-return receiving task**:

```text
sample serve ball state
→ ball flies toward your side
→ paddle must intercept after/near legal timing
→ ball must cross net
→ ball must land on opponent table half
→ episode ends after success/failure
```

Do **not** reward just “touching the ball.” Reward the **post-hit trajectory**.

A good reward structure is:

```python
reward = (
    r_survive
  + r_intercept_precontact
  + r_legal_hit
  + r_send_forward
  + r_cross_net
  + r_land_opponent_side
  + r_target_landing
  + r_control_smooth
  + r_rule_penalty
)
```

The most important terms are below.

### 1. Serve-state filtering before RL

Before reward design, filter `serves.json`. Many sampled serves may be unreachable for your virtual paddle workspace.

```python
def valid_serve_state(s):
    pos = np.array([s["pos_x"], s["pos_y"], s["pos_z"]])
    vel = np.array([s["vel_x"], s["vel_y"], s["vel_z"]])

    # assuming paddle is on x < 0 side and ball should come toward paddle
    if vel[0] > -0.2:
        return False

    speed = np.linalg.norm(vel)
    if speed < 0.5 or speed > curriculum.max_speed:
        return False

    # predict whether ball will enter paddle reachable zone
    p_hit = predict_ball_at_x_or_height(pos, vel)
    if not inside_paddle_workspace(p_hit):
        return False

    return True
```

For early training, use only easy serves:

```text
speed: 0.8–2.5 m/s
incoming height near paddle: 0.85–1.25 m
lateral y: inside [-0.45, 0.45]
low spin or ignore spin first
```

### 2. Pre-contact intercept shaping

This helps the paddle move toward the expected intercept point before contact.

Predict an intercept point, for example where the ball reaches the paddle-side hitting plane:

```python
p_intercept = predict_intercept(ball_pos, ball_vel, paddle_workspace)
```

Reward paddle center closeness:

```python
dist = np.linalg.norm(paddle_pos - p_intercept)
r_intercept_precontact = w_intercept * np.exp(-dist / 0.12)
```

Use a moderate weight:

```python
w_intercept = 0.2 ~ 0.6
```

This term should be active only before paddle-ball contact. After contact, turn it off.

Also add orientation shaping: paddle normal should face the incoming ball direction.

```python
incoming_dir = -normalize(ball_vel)
paddle_normal = get_paddle_normal(data)

normal_align = np.dot(paddle_normal, incoming_dir)
r_paddle_align = w_align * np.clip(normal_align, 0.0, 1.0)
```

Use small weight:

```python
w_align = 0.05 ~ 0.2
```

Do not over-weight this, because the best paddle angle depends on desired outgoing trajectory, not only incoming direction.

### 3. Legal contact reward

Reward first valid paddle-ball contact:

```python
if paddle_ball_contact and not has_hit:
    r_legal_hit = +1.0
    has_hit = True
```

But this is not enough. Contact-only reward is dangerous because the policy can learn to block, trap, or slap the ball down.

Add conditions:

```python
valid_hit = (
    paddle_ball_contact
    and ball_is_incoming
    and own_side_bounce_count <= 1
    and not net_contact_before_hit
)
```

Reward only valid hit:

```python
r_legal_hit = 1.0 if valid_hit else -1.0
```

### 4. Post-hit outgoing velocity reward

After paddle contact, reward the ball moving toward the opponent side.

Assume:

```text
paddle side: x < 0
opponent side: x > 0
net plane: x = 0
```

Then after hit:

```python
v = ball_vel_after_hit

r_send_forward = w_forward * clip((v[0] - v_min) / v_scale, 0.0, 1.0)
```

Example:

```python
v_min = 0.8
v_scale = 3.0
w_forward = 0.5
```

Also require enough upward/arc component to clear the net:

```python
r_lift = w_lift * clip((v[2] - 0.2) / 1.0, 0.0, 1.0)
```

Use this only immediately after contact, not throughout the whole episode.

### 5. Net-crossing reward

Once the ball crosses the net plane after paddle hit:

```python
if has_hit and previous_ball_x < 0 and ball_x >= 0 and ball_z > net_height + ball_radius:
    r_cross_net = +1.5
    crossed_net = True
```

If it crosses below net height or contacts the net:

```python
if has_hit and net_contact:
    r_net_fail = -2.0
    done = True
```

### 6. Opponent-side landing reward

This is the main sparse success term.

```python
if crossed_net and ball_table_contact:
    if ball_x > 0 and abs(ball_y) < table_half_y:
        r_land_opponent_side = +5.0
        success = True
        done = True
    else:
        r_wrong_landing = -2.0
        done = True
```

For a standard table:

```python
table_half_x = 1.37
table_half_y = 0.7625
net_x = 0.0
net_height = 0.1525
```

If your table is rotated or translated, compute these in table-local coordinates, not raw world coordinates.

### 7. Landing target shaping

Before sparse landing reward becomes learnable, use a predicted landing point after hit.

After paddle contact, simulate/estimate where the ball will intersect table height:

```python
landing_xy = predict_landing_xy(ball_pos, ball_vel, table_z)
target_xy = np.array([0.65, 0.0])  # opponent side center-ish
landing_error = np.linalg.norm(landing_xy - target_xy)
```

Reward:

```python
r_target_landing = w_land_shape * np.exp(-landing_error / 0.35)
```

Use:

```python
w_land_shape = 0.5 ~ 1.5
```

But only after contact. Before contact, this term is meaningless.

### 8. Serve rule penalties

For receiving a serve, the environment should enforce a simplified legal-return rule.

Terminate with penalty if:

```python
# ball falls to floor
if ball_z < 0.05:
    reward -= 3.0
    done = True

# ball bounces twice on your side before hit
if own_side_bounce_count > 1 and not has_hit:
    reward -= 3.0
    done = True

# paddle hits ball before it reaches legal incoming phase, if you require post-bounce return
if require_bounce_before_hit and paddle_ball_contact and own_side_bounce_count == 0:
    reward -= 1.0
    done = True

# paddle hits ball multiple times
if paddle_hit_count > 1:
    reward -= 2.0
    done = True

# ball goes outside table region after hit
if abs(ball_x) > table_half_x + 0.7 or abs(ball_y) > table_half_y + 0.7:
    reward -= 2.0
    done = True
```

For early training, I would **not require bounce-before-hit**. Let the paddle learn interception first. Later, enforce legal receive timing.

### 9. Smoothness and actuator penalties

Since your paddle is a virtual 6DOF body, it can exploit high acceleration unless penalized.

Use:

```python
r_action_rate = -w_action_rate * np.sum((action_t - action_tm1) ** 2)
r_action_mag  = -w_action_mag  * np.sum(action_t ** 2)
r_paddle_vel  = -w_vel         * np.sum(paddle_qvel ** 2)
```

Recommended small weights:

```python
w_action_rate = 0.005 ~ 0.03
w_action_mag  = 0.0005 ~ 0.005
w_vel         = 0.0005 ~ 0.005
```

Do not over-penalize motion early. The paddle must move fast.

### 10. Full reward example

```python
def compute_reward(state):
    r = 0.0

    # small time reward before failure
    r += 0.005

    if not state.has_hit:
        # move to predicted intercept
        dist = np.linalg.norm(state.paddle_pos - state.predicted_intercept)
        r += 0.4 * np.exp(-dist / 0.12)

        # orient paddle toward incoming ball
        incoming_dir = -normalize(state.ball_vel)
        align = np.dot(state.paddle_normal, incoming_dir)
        r += 0.1 * np.clip(align, 0.0, 1.0)

    if state.new_paddle_ball_contact:
        if state.own_side_bounce_count <= 1:
            r += 1.0
            state.has_hit = True

            v = state.ball_vel
            r += 0.5 * np.clip((v[0] - 0.8) / 3.0, 0.0, 1.0)
            r += 0.2 * np.clip((v[2] - 0.2) / 1.0, 0.0, 1.0)
        else:
            r -= 2.0
            state.done = True

    if state.just_crossed_net_after_hit:
        if state.ball_pos[2] > state.net_height + state.ball_radius:
            r += 1.5
        else:
            r -= 2.0
            state.done = True

    if state.has_hit and state.predicted_landing_valid:
        err = np.linalg.norm(state.predicted_landing_xy - state.target_landing_xy)
        r += 0.8 * np.exp(-err / 0.35)

    if state.new_table_contact_after_crossing:
        if state.ball_on_opponent_side and state.ball_inside_table_y:
            r += 5.0
            state.success = True
            state.done = True
        else:
            r -= 2.0
            state.done = True

    # illegal events
    if state.ball_floor_contact:
        r -= 3.0
        state.done = True

    if state.paddle_hit_count > 1:
        r -= 2.0
        state.done = True

    if state.own_side_bounce_count > 1 and not state.has_hit:
        r -= 3.0
        state.done = True

    # regularization
    r -= 0.01 * np.sum((state.action - state.prev_action) ** 2)
    r -= 0.001 * np.sum(state.paddle_qvel ** 2)

    return r
```

### 11. Recommended curriculum for `serves.json`

Use `serves.json` in stages:

```text
Stage 0: no spin, slow synthetic serves, large paddle workspace
Stage 1: easy filtered serves.json, contact + cross-net reward
Stage 2: require opponent-table landing
Stage 3: increase speed/y spread/spin
Stage 4: enforce bounce-before-hit if you want strict receive rule
Stage 5: reduce dense shaping, keep mostly legal-return sparse reward
```

A practical reward-weight table:

| Term                   |    Early weight |   Later weight |
| ---------------------- | --------------: | -------------: |
| intercept distance     |         0.4–0.8 |        0.1–0.3 |
| paddle normal align    |         0.1–0.2 |        0.0–0.1 |
| legal first hit        |             1.0 |            0.5 |
| send forward           |             0.5 |            0.2 |
| cross net              |             1.5 |            1.0 |
| opponent-side landing  |             5.0 |       8.0–10.0 |
| target landing shaping |             0.8 |            0.2 |
| action rate penalty    | -0.005 to -0.02 | -0.02 to -0.05 |
| illegal event penalty  |        -2 to -5 |      -5 to -10 |

### 12. The main trap

Avoid this reward:

```python
reward += distance_to_ball
reward += paddle_ball_contact
reward += episode_alive
```

That often produces bad behavior: the paddle chases the ball, touches it weakly, blocks it, or keeps the episode alive without making a valid return.

The correct training signal is:

```text
intercept → legal hit → cross net → land on opponent side
```

For `serves.json`, the final success condition should be:

```python
success = (
    has_paddle_hit
    and crossed_net_after_hit
    and first_table_contact_after_hit_is_opponent_side
)
```

That should be the core of your reward and termination logic.
