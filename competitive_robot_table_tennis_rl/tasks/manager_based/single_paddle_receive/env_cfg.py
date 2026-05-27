"""Configuration and constants for the single-paddle serve-receive task."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


TABLE_HALF_WIDTH_M = 0.7625
TABLE_HALF_LENGTH_M = 1.37
TABLE_TOP_Z_M = 0.0
NET_HEIGHT_M = 0.1525
NET_HALF_WIDTH_M = 0.915
BALL_RADIUS_M = 0.02
PADDLE_CONTACT_HALF_THICKNESS_M = 0.01
DEFAULT_GRAVITY_M_S2 = 9.81


@dataclass(frozen=True)
class TaskIds:
    base: str = "TableTennis-SinglePaddleServeReceive-v0"
    easy: str = "TableTennis-SinglePaddleServeReceiveEasy-v0"
    curriculum: str = "TableTennis-SinglePaddleServeReceiveCurriculum-v0"


TASK_IDS = TaskIds()


@dataclass(frozen=True)
class SceneCfg:
    mjcf_path: str = (
        "competitive_robot_table_tennis_rl/tasks/manager_based/"
        "single_paddle_receive/assets/single_paddle_receive_dataset_frame.xml"
    )
    timestep_s: float = 0.002
    frame_skip: int = 10
    episode_length_s: float = 2.0
    ball_body_name: str = "ball"
    ball_geom_name: str = "geom_ball"
    table_geom_name: str = "geom_table"
    net_geom_name: str = "geom_net"
    floor_geom_name: str = "floor"
    paddle_body_name: str = "paddle"
    paddle_geom_name: str = "paddle_collision"
    paddle_center_site_name: str = "paddle_center_site"
    paddle_face_site_name: str = "paddle_face_site"
    ball_site_name: str = "ball_site"


@dataclass(frozen=True)
class ControlCfg:
    control_backend: str = "direct_paddle"
    control_mode: str = "planner_residual"
    action_scale_m: tuple[float, float, float] = (0.05, 0.08, 0.05)
    action_scale_rad: tuple[float, float, float] = (0.18, 0.18, 0.18)
    residual_scale_m: tuple[float, float, float] = (0.08, 0.08, 0.08)
    residual_scale_rad: tuple[float, float, float] = (0.25, 0.25, 0.25)
    x_range_m: tuple[float, float] = (-0.60, 0.60)
    y_range_m: tuple[float, float] = (-1.25, -0.15)
    z_range_m: tuple[float, float] = (0.1, 0.60)
    roll_range_rad: tuple[float, float] = (-3.141592653589793, 3.141592653589793)
    pitch_range_rad: tuple[float, float] = (-3.141592653589793, 3.141592653589793)
    yaw_range_rad: tuple[float, float] = (-3.141592653589793, 3.141592653589793)
    home_qpos: tuple[float, float, float, float, float, float] = (
        0.0,
        -0.90,
        0.28,
        0.0,
        0.0,
        0.0,
    )


@dataclass(frozen=True)
class PlannerCfg:
    enabled: bool = True
    mode: str = "analytic"
    target_landing_x_range: tuple[float, float] = (-0.35, 0.35)
    target_landing_y_range: tuple[float, float] = (0.45, 1.05)
    hit_time_min_s: float = 0.12
    hit_time_max_s: float = 0.75
    hit_time_samples: int = 64
    min_hit_z_m: float = 0.25
    max_hit_z_m: float = 0.60
    desired_flight_time_s: float = 0.45
    max_outgoing_speed_m_s: float = 8.0
    follow_through_time_s: float = 0.15
    follow_through_distance_m: float = 0.08
    apf_integration_dt_s: float = 0.01
    apf_stage_standoff_m: float = 0.10
    apf_stage_release_time_s: float = 0.12
    apf_linear_gain: float = 10.0
    apf_angular_gain: float = 12.0
    apf_linear_max_speed_m_s: float = 3.0
    apf_angular_max_speed_rad_s: float = 8.0
    apf_boundary_margin_m: float = 0.05
    apf_boundary_repulsion_gain: float = 0.0015


@dataclass(frozen=True)
class DatasetCfg:
    serves_path: str = "serves.json"
    easy_speed_range_m_s: tuple[float, float] = (3.0, 7.0)
    easy_spin_max_rad_s: float = 35.0
    easy_height_range_m: tuple[float, float] = (0.07, 0.40)
    base_spin_max_rad_s: float = 60.0
    max_intercept_horizon_s: float = 1.5
    table_bounce_restitution_z: float = 0.88
    table_bounce_damping_xy: float = 0.96


@dataclass(frozen=True)
class CurriculumCfg:
    success_window: int = 20
    easy_to_base_success_rate: float = 0.35
    base_to_full_success_rate: float = 0.60


@dataclass(frozen=True)
class RewardCfg:
    survive: float = 0.005
    tracking_pos: float = 0.10
    tracking_pos_sigma_m: float = 0.08
    tracking_rot: float = 0.05
    tracking_rot_sigma_rad: float = 0.25
    tracking_vel: float = 0.05
    tracking_vel_sigma_m_s: float = 1.0
    legal_hit: float = 2.0
    send_forward: float = 2.0
    send_forward_min_speed_m_s: float = 0.80
    send_forward_scale_m_s: float = 3.0
    lift: float = 0.20
    lift_min_m_s: float = 0.20
    lift_scale_m_s: float = 1.0
    cross_net: float = 3.0
    landing: float = 10.0
    landing_shape: float = 0.80
    landing_sigma_m: float = 0.35
    action_rate_penalty: float = 0.004
    action_mag_penalty: float = 0.001
    qvel_penalty: float = 0.0003
    floor_penalty: float = -3.0
    double_bounce_penalty: float = -3.0
    multi_hit_penalty: float = -2.0
    wrong_landing_penalty: float = -2.0
    net_fail_penalty: float = -2.0
    out_of_bounds_penalty: float = -2.0


@dataclass(frozen=True)
class TerminationCfg:
    max_abs_x_m: float = TABLE_HALF_WIDTH_M + 0.75
    max_abs_y_m: float = TABLE_HALF_LENGTH_M + 0.90
    min_ball_z_m: float = -0.70
    require_bounce_before_hit: bool = False


@dataclass(frozen=True)
class TaskCfg:
    task_id: str
    task_mode: str = "single_receive"
    scene: SceneCfg = SceneCfg()
    control: ControlCfg = ControlCfg()
    planner: PlannerCfg = PlannerCfg()
    dataset: DatasetCfg = DatasetCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    reward: RewardCfg = RewardCfg()
    termination: TerminationCfg = TerminationCfg()

    @property
    def max_steps(self) -> int:
        return int(self.scene.episode_length_s / (self.scene.timestep_s * self.scene.frame_skip))

    @property
    def observation_dim(self) -> int:
        return 50

    @property
    def action_dim(self) -> int:
        return 6

    @property
    def mjcf_path(self) -> Path:
        return Path(self.scene.mjcf_path)


def make_task_cfg(difficulty: str = "base") -> TaskCfg:
    task_id = {
        "easy": TASK_IDS.easy,
        "base": TASK_IDS.base,
        "curriculum": TASK_IDS.curriculum,
    }[difficulty]
    return TaskCfg(task_id=task_id)
