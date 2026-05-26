#@title Heplers and resources


# System identified Table Tennis Table and Ball contacts.
pp_ball_with_table_xml = """
<mujoco model="pp_ball_with_table">
  <compiler angle="radian" autolimits="true"/>
  <option density="1.225" viscosity="1.8e-5" wind="0 0 0" integrator="implicitfast" timestep="0.001" cone="elliptic"/>

  <visual>
    <headlight diffuse=".5 .5 .5" specular="1 1 1"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1=".5 .5 .5" rgb2="0 0 0" width="10" height="10"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="1 1 1" rgb2="1 1 1" markrgb="0 0 0" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.3"/>
  </asset>

  <worldbody>
    <light pos="0 0 3" diffuse="1 1 1" specular="1 1 1"/>

    <body name="ball" pos="0.0 -0.5 0.652">
      <inertial pos="0 0 0" mass="2.7e-3" diaginertia="0.00000072 0.00000072 0.00000072"/>
      <freejoint/>
      <geom name="geom_ball" size=".02" shellinertia="false" fluidcoef="0.235 0.25 0.0 1.0 1.0" group="1" rgba="0.98 0.70 0.015 1" solref="-100000 -0" fluidshape="ellipsoid"/>
    </body>

    <body name="table" pos="0 0 -0.76" quat="0.7071068 0 0 0.7071068">
      <geom name="geom_table" size="1.37 0.7625 0.02" pos="0 0 0.74" type="box" rgba="0.27 0.51 0.14 1"/>
      <geom size="0.005 0.915 0.07625" pos="0 0 0.83625" type="box" rgba="1 1 1 0.5"/>
      <geom size="1.371 0.7635 0.02" pos="0 0 0.739" type="box" contype="0" conaffinity="0" group="1" density="0" rgba="0 0 0 1"/>
    </body>

  </worldbody>

  <contact>
      <pair geom1="geom_ball" geom2="geom_table" solref="-1000000 -17" friction="0.1 0.1 0.005 0.0001 0.0001" solimp="0.98 0.99 0.001 0.5 2" solreffriction="-0.0 -200.0"/>
  </contact>

</mujoco>
"""


@dataclasses.dataclass
class BallState:
  """Initial ball state."""
  id: int
  pos_x: float
  pos_y: float
  pos_z: float
  vel_x: float
  vel_y: float
  vel_z: float
  w_vel_x: float
  w_vel_y: float
  w_vel_z: float

  @property
  def position(self) -> np.ndarray:
    return np.array([self.pos_x, self.pos_y, self.pos_z])

  @property
  def linear_velocity(self) -> np.ndarray:
    return np.array([self.vel_x, self.vel_y, self.vel_z])

  @property
  def angular_velocity(self) -> np.ndarray:
    return np.array([self.w_vel_x, self.w_vel_y, self.w_vel_z])


def rollout_and_render(
            ball_state: BallState,
            model: mj.MjModel,
            data: mj.MjData,
            duration: float = 1.0,
            framerate: int = 30):

  scene_option = mj.MjvOption()
  scene_option.flags[mj.mjtVisFlag.mjVIS_CONTACTPOINT] = True

  frames = []
  with mj.Renderer(model, 480, 640) as renderer:
    mj.mj_resetData(model, data)
    data.qpos[:3] = ball_state.position[:]
    data.qvel[:3] = ball_state.linear_velocity[:]
    data.qvel[3:] = ball_state.angular_velocity[:]
    mj.mj_forward(model, data)
    print("qpos:", data.qpos)
    print("qvel:", data.qvel)
    while data.time < duration:
      mj.mj_step(model, data)
      if len(frames) < data.time * framerate:
        renderer.update_scene(data, scene_option=scene_option)
        pixels = renderer.render()
        frames.append(pixels)

  media.show_video(frames, fps=framerate)