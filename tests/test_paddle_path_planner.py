from dataclasses import replace

import mujoco
import numpy as np

from competitive_robot_table_tennis_rl.tasks.manager_based.single_paddle_receive.env_cfg import (
    BALL_RADIUS_M,
    DEFAULT_GRAVITY_M_S2,
    PADDLE_CONTACT_HALF_THICKNESS_M,
    make_task_cfg,
)
from competitive_robot_table_tennis_rl.tasks.manager_based.single_paddle_receive.mdp.transforms import (
    paddle_normal_from_euler,
)
from competitive_robot_table_tennis_rl.tasks.manager_based.single_paddle_receive.control import (
    ControlBackend,
)
from competitive_robot_table_tennis_rl.tasks.manager_based.single_paddle_receive.planning import (
    BallPredictor,
    HitPlan,
    ImpactPlanner,
    PaddlePathPlanner,
    should_reset_hit_plan,
)
from competitive_robot_table_tennis_rl.tasks.manager_based.single_paddle_receive.utils.serve_dataset import (
    load_serves,
)
from competitive_robot_table_tennis_rl.tasks.manager_based.single_paddle_receive.utils.paths import (
    resolve_repo_path,
)


def _simulate_paddle_pose_at_time(
    *,
    cfg,
    hit_plan: HitPlan,
) -> tuple[np.ndarray, np.ndarray]:
    xml_path = resolve_repo_path(cfg.scene.mjcf_path)
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    control_timestep_s = float(model.opt.timestep) * cfg.scene.frame_skip
    paddle_joint_names = (
        "paddle_x",
        "paddle_y",
        "paddle_z",
        "paddle_roll",
        "paddle_pitch",
        "paddle_yaw",
    )
    actuator_names = (
        "paddle_x_pos",
        "paddle_y_pos",
        "paddle_z_pos",
        "paddle_roll_pos",
        "paddle_pitch_pos",
        "paddle_yaw_pos",
    )
    paddle_joint_ids = [
        int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name))
        for name in paddle_joint_names
    ]
    actuator_ids = [
        int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name))
        for name in actuator_names
    ]
    paddle_qpos_adr = np.array([int(model.jnt_qposadr[joint_id]) for joint_id in paddle_joint_ids], dtype=np.int32)
    paddle_center_site_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, cfg.scene.paddle_center_site_name))
    paddle_root_body_id = int(model.jnt_bodyid[paddle_joint_ids[0]])
    control_backend = ControlBackend(
        control_cfg=cfg.control,
        actuator_ids=tuple(actuator_ids),
        linear_offset_m=np.asarray(model.body_pos[paddle_root_body_id], dtype=np.float64).copy(),
    )

    home_qpos = np.array(cfg.control.home_qpos, dtype=np.float64)
    model_home_qpos = control_backend.task_to_model_qpos(home_qpos)
    data.qpos[paddle_qpos_adr] = model_home_qpos
    data.ctrl[actuator_ids] = model_home_qpos
    mujoco.mj_forward(model, data)

    planner = PaddlePathPlanner(control_cfg=cfg.control, planner_cfg=cfg.planner)
    planner.reset(
        current_qpos=home_qpos,
        hit_plan=hit_plan,
        now_s=0.0,
    )

    next_control_update_s = 0.0
    while data.time + 1e-9 < float(hit_plan.hit_time_s):
        if data.time + 1e-9 >= next_control_update_s:
            cmd = planner.command(next_control_update_s)
            data.ctrl[actuator_ids] = control_backend.task_to_model_qpos(cmd.qpos)
            next_control_update_s += control_timestep_s
        mujoco.mj_step(model, data)

    actual_qpos = control_backend.model_to_task_qpos(data.qpos[paddle_qpos_adr])
    actual_site_xyz = data.site_xpos[paddle_center_site_id].copy()
    return actual_qpos, actual_site_xyz


def _print_contact_trace(
    *,
    cfg,
    hit_plan: HitPlan,
) -> None:
    actual_qpos, actual_site_xyz = _simulate_paddle_pose_at_time(cfg=cfg, hit_plan=hit_plan)
    print(
        {
            "predicted_contact_time_s": round(float(hit_plan.hit_time_s), 6),
            "predicted_contact_xyz": np.round(hit_plan.ball_hit_pos, 6).tolist(),
            "planned_paddle_contact_pose": np.round(hit_plan.hit_pos, 6).tolist(),
            "planned_paddle_euler": np.round(hit_plan.hit_euler, 6).tolist(),
            "mujoco_actual_paddle_qpos_at_contact_time": np.round(actual_qpos, 6).tolist(),
            "mujoco_actual_paddle_center_xyz_at_contact_time": np.round(actual_site_xyz, 6).tolist(),
        }
    )


def test_impact_planner_uses_contact_pose_offset() -> None:
    cfg = make_task_cfg("easy")
    serve = load_serves(cfg.dataset.serves_path)[0]
    planner = ImpactPlanner(
        cfg.planner,
        cfg.control,
        ball_predictor=BallPredictor(
            DEFAULT_GRAVITY_M_S2,
            table_bounce_restitution_z=cfg.dataset.table_bounce_restitution_z,
            table_bounce_damping_xy=cfg.dataset.table_bounce_damping_xy,
        ),
    )

    hit_plan = planner.plan(
        ball_pos=serve.position,
        ball_vel=serve.linear_velocity,
        paddle_qpos=np.array(cfg.control.home_qpos, dtype=np.float64),
        target_landing_xy=np.array([0.0, 0.80], dtype=np.float64),
        bounce_has_occurred=False,
    )

    assert hit_plan.valid
    _print_contact_trace(cfg=cfg, hit_plan=hit_plan)
    expected_offset = paddle_normal_from_euler(hit_plan.hit_euler) * (
        BALL_RADIUS_M + PADDLE_CONTACT_HALF_THICKNESS_M
    )
    np.testing.assert_allclose(hit_plan.ball_hit_pos - hit_plan.hit_pos, expected_offset, atol=1e-6)


def test_bounce_aware_impact_planner_finds_receive_plan_before_bounce() -> None:
    cfg = make_task_cfg("easy")
    serve = load_serves(cfg.dataset.serves_path)[0]
    planner = ImpactPlanner(
        cfg.planner,
        cfg.control,
        ball_predictor=BallPredictor(
            DEFAULT_GRAVITY_M_S2,
            table_bounce_restitution_z=cfg.dataset.table_bounce_restitution_z,
            table_bounce_damping_xy=cfg.dataset.table_bounce_damping_xy,
        ),
    )

    hit_plan = planner.plan(
        ball_pos=serve.position,
        ball_vel=serve.linear_velocity,
        paddle_qpos=np.array(cfg.control.home_qpos, dtype=np.float64),
        target_landing_xy=np.array([0.0, 0.8], dtype=np.float64),
        bounce_has_occurred=False,
    )

    assert hit_plan.valid
    assert hit_plan.ball_hit_pos[1] < 0.0


def test_ball_predictor_rejects_bounce_outside_table_bounds() -> None:
    predictor = BallPredictor(DEFAULT_GRAVITY_M_S2)

    point = predictor.predict_point(
        np.array([0.95, 0.0, 0.35], dtype=np.float64),
        np.array([0.0, 0.0, -1.0], dtype=np.float64),
        0.30,
        allow_table_bounce=True,
        bounce_has_occurred=False,
    )

    ballistic_z = 0.35 + (-1.0 * 0.30) - 0.5 * DEFAULT_GRAVITY_M_S2 * (0.30**2)
    assert point.pos[2] < 0.0
    np.testing.assert_allclose(point.pos, np.array([0.95, 0.0, ballistic_z], dtype=np.float64), atol=1e-8)


def test_impact_planner_rejects_hits_that_exceed_paddle_time_to_reach() -> None:
    cfg = make_task_cfg("easy")
    slow_cfg = replace(
        cfg.planner,
        hit_time_min_s=0.12,
        hit_time_max_s=0.12,
        hit_time_samples=1,
        apf_linear_max_speed_m_s=0.20,
        apf_angular_max_speed_rad_s=1.0,
    )
    planner = ImpactPlanner(
        slow_cfg,
        cfg.control,
        ball_predictor=BallPredictor(DEFAULT_GRAVITY_M_S2),
    )

    hit_plan = planner.plan(
        ball_pos=np.array([0.0, -0.20, 0.30], dtype=np.float64),
        ball_vel=np.array([0.0, -1.5, 0.2], dtype=np.float64),
        paddle_qpos=np.array(cfg.control.home_qpos, dtype=np.float64),
        target_landing_xy=np.array([0.0, 0.70], dtype=np.float64),
        bounce_has_occurred=True,
    )

    assert not hit_plan.valid


def test_apf_path_stages_behind_contact_before_final_release() -> None:
    cfg = make_task_cfg("easy")
    planner = PaddlePathPlanner(control_cfg=cfg.control, planner_cfg=cfg.planner)
    hit_plan = HitPlan(
        valid=True,
        hit_time_s=0.40,
        ball_hit_pos=np.array([0.0, -0.44, 0.33], dtype=np.float64),
        hit_pos=np.array([0.0, -0.47, 0.33], dtype=np.float64),
        hit_euler=np.zeros(3, dtype=np.float64),
        incoming_ball_vel=np.array([0.0, -4.5, 0.1], dtype=np.float64),
        hit_vel=np.zeros(3, dtype=np.float64),
        outgoing_vel_des=np.array([0.0, 3.5, 0.8], dtype=np.float64),
        target_landing_xy=np.array([0.0, 0.8], dtype=np.float64),
        cost=0.0,
    )
    planner.reset(
        current_qpos=np.array(cfg.control.home_qpos, dtype=np.float64),
        hit_plan=hit_plan,
        now_s=0.0,
    )

    pre_release_cmd = planner.command(0.30)
    hit_cmd = planner.command(hit_plan.hit_time_s)

    assert pre_release_cmd.qpos[1] < hit_plan.hit_pos[1] - 0.015
    np.testing.assert_allclose(hit_cmd.qpos[:3], hit_plan.hit_pos, atol=1e-8)
    np.testing.assert_allclose(hit_cmd.qpos[3:], hit_plan.hit_euler, atol=1e-8)


def test_active_plan_is_not_replaced_by_invalid_replan_flicker() -> None:
    cfg = make_task_cfg("easy")
    planner = PaddlePathPlanner(control_cfg=cfg.control, planner_cfg=cfg.planner)
    current_hit_plan = HitPlan(
        valid=True,
        hit_time_s=0.12,
        ball_hit_pos=np.array([0.0, -0.70, 0.30], dtype=np.float64),
        hit_pos=np.array([0.0, -0.73, 0.30], dtype=np.float64),
        hit_euler=np.zeros(3, dtype=np.float64),
        incoming_ball_vel=np.array([0.0, -3.0, 0.2], dtype=np.float64),
        hit_vel=np.zeros(3, dtype=np.float64),
        outgoing_vel_des=np.array([0.0, 3.2, 0.7], dtype=np.float64),
        target_landing_xy=np.array([0.0, 0.8], dtype=np.float64),
        cost=0.0,
    )
    planner.reset(
        current_qpos=np.array(cfg.control.home_qpos, dtype=np.float64),
        hit_plan=current_hit_plan,
        now_s=0.46,
    )

    invalid_replan = HitPlan.invalid(current_hit_plan.target_landing_xy)

    assert planner.is_plan_active(0.48, current_hit_plan)
    assert not should_reset_hit_plan(
        current_hit_plan=current_hit_plan,
        new_hit_plan=invalid_replan,
        force=False,
        has_hit=False,
        now_s=0.48,
        path_planner=planner,
    )
    assert should_reset_hit_plan(
        current_hit_plan=current_hit_plan,
        new_hit_plan=invalid_replan,
        force=False,
        has_hit=False,
        now_s=0.90,
        path_planner=planner,
    )
