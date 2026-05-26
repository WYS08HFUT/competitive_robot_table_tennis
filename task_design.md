**Control Pattern**

The paddle is not using a scripted tracking controller. It is a learned 6-DoF direct-paddle policy: `x, y, z, roll, pitch, yaw`. The policy outputs an action in `[-1, 1]`, and each step that action is treated as a *delta pose* added onto the previous commanded paddle pose, then clipped to workspace/joint limits in [control.py](/home/ubuntu/Desktop/OY_openarm/competitive_robot_table_tennis/competitive_robot_table_tennis_rl/tasks/manager_based/single_paddle_receive/control.py:24) and [env_cfg.py](/home/ubuntu/Desktop/OY_openarm/competitive_robot_table_tennis/competitive_robot_table_tennis_rl/tasks/manager_based/single_paddle_receive/env_cfg.py:50).

Concretely, one control step can move about:
- `x`: `0.05 m`
- `y`: `0.08 m`
- `z`: `0.05 m`
- `roll/pitch/yaw`: `0.18 rad`

The environment runs MuJoCo at `0.002 s`, but the policy acts every `10` physics steps, so the effective control period is `0.02 s` and a `2.0 s` episode is about `100` control steps in [env_cfg.py](/home/ubuntu/Desktop/OY_openarm/competitive_robot_table_tennis/competitive_robot_table_tennis_rl/tasks/manager_based/single_paddle_receive/env_cfg.py:29). The MuJoCo model itself is a 3-slide + 3-hinge paddle with position actuators, so the action sets actuator targets rather than teleporting the racket in [single_paddle_receive_dataset_frame.xml](/home/ubuntu/Desktop/OY_openarm/competitive_robot_table_tennis/competitive_robot_table_tennis_rl/tasks/manager_based/single_paddle_receive/assets/single_paddle_receive_dataset_frame.xml:90) and [single_paddle_receive_dataset_frame.xml](/home/ubuntu/Desktop/OY_openarm/competitive_robot_table_tennis/competitive_robot_table_tennis_rl/tasks/manager_based/single_paddle_receive/assets/single_paddle_receive_dataset_frame.xml:167).

The observation is what makes the paddle “see” the incoming ball. It includes:
- ball position, velocity, and angular velocity
- paddle pose and velocity
- relative ball position to paddle
- predicted intercept point relative to paddle
- predicted landing point
- rule flags like `has_hit`, bounce count, hit count, crossed-net flag
- previous action

That is assembled in [observations.py](/home/ubuntu/Desktop/OY_openarm/competitive_robot_table_tennis/competitive_robot_table_tennis_rl/tasks/manager_based/single_paddle_receive/mdp/observations.py:10). The prediction helpers estimate where the ball will cross the paddle’s current `y` plane and where it will land ballistically in [transforms.py](/home/ubuntu/Desktop/OY_openarm/competitive_robot_table_tennis/competitive_robot_table_tennis_rl/tasks/manager_based/single_paddle_receive/mdp/transforms.py:61) and are refreshed every physics step in [env.py](/home/ubuntu/Desktop/OY_openarm/competitive_robot_table_tennis/competitive_robot_table_tennis_rl/tasks/manager_based/single_paddle_receive/env.py:316).

**Termination And Rewards**

Episodes end immediately on:
- success: ball lands on opponent side inside table bounds after the paddle’s hit
- any failure reason already set by event logic
- ball drops below `z < -0.70`
- ball goes too far out in `x` or `y`

That logic is in [terminations.py](/home/ubuntu/Desktop/OY_openarm/competitive_robot_table_tennis/competitive_robot_table_tennis_rl/tasks/manager_based/single_paddle_receive/mdp/terminations.py:11). There is also a time limit truncation at the episode horizon in [env.py](/home/ubuntu/Desktop/OY_openarm/competitive_robot_table_tennis/competitive_robot_table_tennis_rl/tasks/manager_based/single_paddle_receive/env.py:220).

The reward has two parts.

Dense shaping every control step, from [rewards.py](/home/ubuntu/Desktop/OY_openarm/competitive_robot_table_tennis/competitive_robot_table_tennis_rl/tasks/manager_based/single_paddle_receive/mdp/rewards.py:12):
- `survive = +0.005`
- `intercept`: bigger when paddle center is close to predicted intercept point before contact
- `align`: bigger when paddle face normal points against the incoming ball direction before contact
- `landing_shape`: after hit, bigger when predicted landing is near target landing `(x=0, y=0.70)`
- penalties for changing action too abruptly, using large actions, and moving paddle joints too fast

Sparse event rewards and penalties, from [env.py](/home/ubuntu/Desktop/OY_openarm/competitive_robot_table_tennis/competitive_robot_table_tennis_rl/tasks/manager_based/single_paddle_receive/env.py:337) and reward scales in [env_cfg.py](/home/ubuntu/Desktop/OY_openarm/competitive_robot_table_tennis/competitive_robot_table_tennis_rl/tasks/manager_based/single_paddle_receive/env_cfg.py:89):
- first legal hit: `+1.0`
- forward outgoing speed after hit: up to `+0.5`
- upward lift after hit: up to `+0.2`
- crosses net cleanly: `+1.5`
- successful opponent-side table landing: `+5.0`

Penalties:
- floor: `-3.0`
- double bounce on own side before hit: `-3.0`
- multiple paddle hits: `-2.0`
- wrong landing side / off table: `-2.0`
- net failure: `-2.0`
- out of bounds: `-2.0`

One important rule detail: `require_bounce_before_hit=False` right now in [env_cfg.py](/home/ubuntu/Desktop/OY_openarm/competitive_robot_table_tennis/competitive_robot_table_tennis_rl/tasks/manager_based/single_paddle_receive/env_cfg.py:117), so the agent is allowed to contact the ball before an own-side bounce. This makes the task closer to “return the incoming ball however you can” than “must wait for a legal receive bounce.”

**How This Leads The Paddle To Play The Incoming Ball**

The behavior pressure is pretty coherent:

1. Before contact, the agent is rewarded for getting the paddle near the predicted intercept and orienting the face against the incoming ball.
2. At contact, it gets a large bonus only for the first hit, and extra reward if that hit sends the ball forward and slightly upward.
3. After contact, it gets more reward if the new trajectory is predicted to land near a desired target on the far side.
4. The biggest reward comes only if the ball actually crosses the net and lands legally on the opponent side.
5. Bad strategies end quickly and get penalized: whiffing until the ball falls, double-bounce, netting the ball, hitting twice, or sending it out.

So the learned policy should naturally discover a sequence like:
- move toward the incoming ball’s predicted meeting point
- rotate the paddle to face the ball
- make a first contact that flips the ball back toward `+y`
- add enough lift to clear the net
- shape the contact so the post-hit landing is near the target region

A subtle caveat: the shaping is still fairly local. The intercept target is based on the ball crossing the paddle’s *current* `y` plane, not a globally optimized swing plan, and the policy only has delta-pose control. So this setup encourages reactive receiving and redirection, not an explicit stroke planner.

If you want, I can turn this into a compact flow diagram from `obs -> PPO action -> paddle target -> contact events -> rewards/termination`.