"""Gymnasium environment for single-paddle serve receive."""

from __future__ import annotations

from collections import deque
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np

from .control import ControlBackend
from .env_cfg import (
    BALL_RADIUS_M,
    DEFAULT_GRAVITY_M_S2,
    NET_HEIGHT_M,
    TABLE_HALF_WIDTH_M,
    TABLE_TOP_Z_M,
    TaskCfg,
    make_task_cfg,
)
from .mdp.commands import resolve_bucket_name, sample_reset_state, update_curriculum_level
from .mdp.events import ContactState, RuntimeState, apply_serve_sample, reset_runtime_state
from .mdp.observations import build_observation
from .mdp.rewards import dense_reward, penalty, reward_for_cross_net, reward_for_legal_hit, reward_for_success
from .mdp.terminations import terminal_from_state
from .mdp.transforms import (
    compute_intercept_point_at_y,
    normalize,
    paddle_normal_from_xmat,
    point_in_table_bounds,
    point_on_opponent_side,
    point_on_own_side,
    predict_ballistic_landing,
)
from .planning import BallPredictor, HitPlan, ImpactPlanner, PaddlePathPlanner, should_reset_hit_plan
from .utils.paths import resolve_repo_path
from .utils.serve_dataset import ServeSample, filter_serves, load_serves


class SinglePaddleServeReceiveEnv(gym.Env[np.ndarray, np.ndarray]):
    """Direct-paddle serve-receive environment with dataset-driven resets."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(
        self,
        difficulty: str = "base",
        render_mode: str | None = None,
        cfg: TaskCfg | None = None,
    ) -> None:
        super().__init__()
        self.difficulty = difficulty
        self.cfg = cfg or make_task_cfg(difficulty)
        self.render_mode = render_mode
        self.rng = np.random.default_rng()
        self._renderer: mujoco.Renderer | None = None

        mjcf_path = resolve_repo_path(self.cfg.scene.mjcf_path)
        self.model = mujoco.MjModel.from_xml_path(str(mjcf_path))
        self.data = mujoco.MjData(self.model)
        self.physics_timestep_s = float(self.model.opt.timestep)
        self.control_timestep_s = self.physics_timestep_s * self.cfg.scene.frame_skip
        self.max_steps = int(self.cfg.scene.episode_length_s / self.control_timestep_s)
        self.metadata["render_fps"] = max(1, int(round(1.0 / self.control_timestep_s)))

        self.ball_body_id = self._name_to_id(mujoco.mjtObj.mjOBJ_BODY, self.cfg.scene.ball_body_name)
        self.ball_geom_id = self._name_to_id(mujoco.mjtObj.mjOBJ_GEOM, self.cfg.scene.ball_geom_name)
        self.table_geom_id = self._name_to_id(mujoco.mjtObj.mjOBJ_GEOM, self.cfg.scene.table_geom_name)
        self.net_geom_id = self._name_to_id(mujoco.mjtObj.mjOBJ_GEOM, self.cfg.scene.net_geom_name)
        self.floor_geom_id = self._name_to_id(mujoco.mjtObj.mjOBJ_GEOM, self.cfg.scene.floor_geom_name)
        self.paddle_body_id = self._name_to_id(mujoco.mjtObj.mjOBJ_BODY, self.cfg.scene.paddle_body_name)
        self.paddle_geom_id = self._name_to_id(mujoco.mjtObj.mjOBJ_GEOM, self.cfg.scene.paddle_geom_name)
        self.paddle_center_site_id = self._name_to_id(mujoco.mjtObj.mjOBJ_SITE, self.cfg.scene.paddle_center_site_name)
        self.ball_site_id = self._name_to_id(mujoco.mjtObj.mjOBJ_SITE, self.cfg.scene.ball_site_name)

        self.ball_joint_id = self._name_to_id(mujoco.mjtObj.mjOBJ_JOINT, "ball_freejoint")
        self.ball_qpos_adr = int(self.model.jnt_qposadr[self.ball_joint_id])
        self.ball_qvel_adr = int(self.model.jnt_dofadr[self.ball_joint_id])

        self.paddle_joint_names = (
            "paddle_x",
            "paddle_y",
            "paddle_z",
            "paddle_roll",
            "paddle_pitch",
            "paddle_yaw",
        )
        self.paddle_joint_ids = tuple(self._name_to_id(mujoco.mjtObj.mjOBJ_JOINT, name) for name in self.paddle_joint_names)
        self.paddle_joint_qpos_adr = np.array([int(self.model.jnt_qposadr[joint_id]) for joint_id in self.paddle_joint_ids], dtype=np.int32)
        self.paddle_joint_qvel_adr = np.array([int(self.model.jnt_dofadr[joint_id]) for joint_id in self.paddle_joint_ids], dtype=np.int32)

        actuator_names = (
            "paddle_x_pos",
            "paddle_y_pos",
            "paddle_z_pos",
            "paddle_roll_pos",
            "paddle_pitch_pos",
            "paddle_yaw_pos",
        )
        actuator_ids = tuple(self._name_to_id(mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in actuator_names)
        paddle_root_body_id = int(self.model.jnt_bodyid[self.paddle_joint_ids[0]])
        paddle_linear_offset_m = np.asarray(self.model.body_pos[paddle_root_body_id], dtype=np.float64).copy()
        self.control_backend = ControlBackend(
            control_cfg=self.cfg.control,
            actuator_ids=actuator_ids,
            linear_offset_m=paddle_linear_offset_m,
        )

        self.serves = load_serves(self.cfg.dataset.serves_path)
        self.serve_by_id = {sample.id: sample for sample in self.serves}
        self.serve_buckets = {
            "easy": filter_serves(
                self.serves,
                difficulty="easy",
                dataset_cfg=self.cfg.dataset,
                workspace_x=self.cfg.control.x_range_m,
                workspace_y=self.cfg.control.y_range_m,
                workspace_z=self.cfg.control.z_range_m,
            ),
            "base": filter_serves(
                self.serves,
                difficulty="base",
                dataset_cfg=self.cfg.dataset,
                workspace_x=self.cfg.control.x_range_m,
                workspace_y=self.cfg.control.y_range_m,
                workspace_z=self.cfg.control.z_range_m,
            ),
            "full": filter_serves(
                self.serves,
                difficulty="full",
                dataset_cfg=self.cfg.dataset,
                workspace_x=self.cfg.control.x_range_m,
                workspace_y=self.cfg.control.y_range_m,
                workspace_z=self.cfg.control.z_range_m,
            ),
        }
        self.curriculum_level = 0
        self.curriculum_history: deque[bool] = deque(maxlen=self.cfg.curriculum.success_window)

        self.state: RuntimeState = reset_runtime_state()
        self.prev_contacts = ContactState()
        self.prev_ball_pos = np.zeros(3, dtype=np.float64)
        self.prev_action = np.zeros(self.cfg.action_dim, dtype=np.float64)
        self.commanded_paddle_qpos = np.array(self.cfg.control.home_qpos, dtype=np.float64)
        self.current_nominal_cmd_qpos = np.array(self.cfg.control.home_qpos, dtype=np.float64)
        self.current_nominal_cmd_qvel = np.zeros(self.cfg.action_dim, dtype=np.float64)
        self.current_serve: ServeSample | None = None
        self.target_landing_xy = np.zeros(2, dtype=np.float64)
        self.current_hit_plan = HitPlan.invalid(self.target_landing_xy)
        self.episode_time_s = 0.0
        self.step_count = 0
        self.last_info: dict[str, Any] = {}
        self.ball_predictor = BallPredictor(
            DEFAULT_GRAVITY_M_S2,
            table_bounce_restitution_z=self.cfg.dataset.table_bounce_restitution_z,
            table_bounce_damping_xy=self.cfg.dataset.table_bounce_damping_xy,
        )
        self.impact_planner = ImpactPlanner(
            self.cfg.planner,
            self.cfg.control,
            ball_predictor=self.ball_predictor,
        )
        self.path_planner = PaddlePathPlanner(
            control_cfg=self.cfg.control,
            planner_cfg=self.cfg.planner,
        )

        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.cfg.action_dim,), dtype=np.float32)
        high = np.full(self.cfg.observation_dim, np.inf, dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-high, high=high, dtype=np.float32)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        options = options or {}

        mujoco.mj_resetData(self.model, self.data)
        self.state = reset_runtime_state()
        self.prev_contacts = ContactState()
        self.prev_action[:] = 0.0
        self.step_count = 0
        self.episode_time_s = 0.0

        override_paddle_qpos = options.get("paddle_qpos")
        if override_paddle_qpos is None:
            self.commanded_paddle_qpos = np.array(self.cfg.control.home_qpos, dtype=np.float64)
        else:
            self.commanded_paddle_qpos = np.asarray(override_paddle_qpos, dtype=np.float64).reshape(self.cfg.action_dim)
        self.current_nominal_cmd_qpos = self.commanded_paddle_qpos.copy()
        self.current_nominal_cmd_qvel = np.zeros(self.cfg.action_dim, dtype=np.float64)
        self._set_paddle_pose(self.commanded_paddle_qpos)
        self.target_landing_xy = self._sample_target_landing_xy(options)
        self.current_hit_plan = HitPlan.invalid(self.target_landing_xy)
        self.path_planner.reset(
            current_qpos=self.commanded_paddle_qpos,
            hit_plan=self.current_hit_plan,
            now_s=self.episode_time_s,
        )

        requested_serve_id = options.get("serve_id")
        if requested_serve_id is None:
            bucket_name = resolve_bucket_name(self.difficulty, self.curriculum_level)
            self.current_serve = sample_reset_state(self.rng, self.serve_by_id, self.serve_buckets[bucket_name])
        else:
            requested_serve_id = int(requested_serve_id)
            if requested_serve_id not in self.serve_by_id:
                raise ValueError(f"Unknown serve_id: {requested_serve_id}")
            self.current_serve = self.serve_by_id[requested_serve_id]
        apply_serve_sample(
            self.model,
            self.data,
            self.ball_qpos_adr,
            self.ball_qvel_adr,
            self.current_serve,
        )
        mujoco.mj_forward(self.model, self.data)
        self._update_predictions()
        self._refresh_planner(reset_path=True)
        self.prev_ball_pos = self.ball_pos.copy()
        self.prev_contacts = self._scan_contacts()
        obs = self._build_observation()
        info = self._build_info(episode_success=False)
        self.last_info = info
        return obs, info

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float64).reshape(self.cfg.action_dim)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._refresh_planner(reset_path=False)
        self.commanded_paddle_qpos = self.control_backend.apply(
            action,
            self.data,
            prev_cmd_qpos=self.commanded_paddle_qpos.copy(),
            nominal_cmd_qpos=self.current_nominal_cmd_qpos,
        )

        reward_terms: dict[str, float] = {}
        terminated = False
        truncated = False

        for _ in range(self.cfg.scene.frame_skip):
            mujoco.mj_step(self.model, self.data)
            self._update_predictions()
            reward_terms = self._merge_rewards(reward_terms, self._handle_events())
            terminated, reason = terminal_from_state(self.state, self.ball_pos, self.cfg.termination)
            if terminated and not self.state.failure_reason and reason != "success":
                self.state.failure_reason = reason
                reward_terms = self._merge_rewards(reward_terms, self._failure_penalty(reason))
            self.prev_ball_pos = self.ball_pos.copy()
            self.prev_contacts = self._scan_contacts()
            if terminated:
                break

        self.step_count += 1
        self.episode_time_s += self.control_timestep_s
        if not terminated:
            reward_terms = self._merge_rewards(reward_terms, self._dense_reward(action))
            self._refresh_planner(reset_path=False)
        truncated = self.step_count >= self.max_steps

        reward = float(sum(reward_terms.values()))
        obs = self._build_observation()
        info = self._build_info(episode_success=self.state.success)
        info["reward_terms"] = reward_terms
        if terminated or truncated:
            self.curriculum_history.append(self.state.success)
            self.curriculum_level = update_curriculum_level(
                self.curriculum_history,
                self.curriculum_level,
                self.cfg.curriculum,
            )
            info["terminal_reason"] = "success" if self.state.success else self.state.failure_reason or "timeout"
        self.prev_action = action.copy()
        self.last_info = info
        return obs, reward, terminated, truncated, info

    def render(self) -> np.ndarray:
        """Render the current scene to an RGB array."""

        if self._renderer is None:
            try:
                self._renderer = mujoco.Renderer(self.model, 480, 640)
            except Exception as exc:  # pragma: no cover - depends on local GL backend
                raise RuntimeError(
                    "MuJoCo rendering requires a working OpenGL context. "
                    "Run the viewer in a desktop session or configure a headless GL backend."
                ) from exc
        self._renderer.update_scene(self.data)
        return self._renderer.render()

    def close(self) -> None:
        """Close rendering resources."""

        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    @property
    def ball_pos(self) -> np.ndarray:
        return self.data.qpos[self.ball_qpos_adr : self.ball_qpos_adr + 3].copy()

    @property
    def ball_vel(self) -> np.ndarray:
        return self.data.qvel[self.ball_qvel_adr : self.ball_qvel_adr + 3].copy()

    @property
    def ball_angvel(self) -> np.ndarray:
        return self.data.qvel[self.ball_qvel_adr + 3 : self.ball_qvel_adr + 6].copy()

    @property
    def paddle_qpos(self) -> np.ndarray:
        return self.control_backend.model_to_task_qpos(self.data.qpos[self.paddle_joint_qpos_adr])

    @property
    def paddle_qvel(self) -> np.ndarray:
        return self.data.qvel[self.paddle_joint_qvel_adr].copy()

    @property
    def paddle_pos(self) -> np.ndarray:
        return self.data.site_xpos[self.paddle_center_site_id].copy()

    @property
    def paddle_normal(self) -> np.ndarray:
        return paddle_normal_from_xmat(self.data.site_xmat[self.paddle_center_site_id])

    def _name_to_id(self, obj_type: mujoco.mjtObj, name: str) -> int:
        obj_id = mujoco.mj_name2id(self.model, obj_type, name)
        if obj_id < 0:
            raise ValueError(f"Missing MuJoCo object: {name}")
        return int(obj_id)

    def _set_paddle_pose(self, qpos: np.ndarray) -> None:
        model_qpos = self.control_backend.task_to_model_qpos(qpos)
        self.data.qpos[self.paddle_joint_qpos_adr] = model_qpos
        for actuator_id, target in zip(self.control_backend.actuator_ids, model_qpos):
            self.data.ctrl[actuator_id] = target

    def _sample_target_landing_xy(self, options: dict[str, Any]) -> np.ndarray:
        target = options.get("target_landing_xy")
        if target is not None:
            return np.asarray(target, dtype=np.float64).reshape(2)
        return np.array(
            [
                self.rng.uniform(*self.cfg.planner.target_landing_x_range),
                self.rng.uniform(*self.cfg.planner.target_landing_y_range),
            ],
            dtype=np.float64,
        )

    def _scan_contacts(self) -> ContactState:
        flags = ContactState()
        for idx in range(self.data.ncon):
            contact = self.data.contact[idx]
            pair = {int(contact.geom1), int(contact.geom2)}
            if {self.ball_geom_id, self.table_geom_id} == pair:
                flags.ball_table = True
            elif {self.ball_geom_id, self.paddle_geom_id} == pair:
                flags.ball_paddle = True
            elif {self.ball_geom_id, self.net_geom_id} == pair:
                flags.ball_net = True
            elif {self.ball_geom_id, self.floor_geom_id} == pair:
                flags.ball_floor = True
        return flags

    def _update_predictions(self) -> None:
        intercept = compute_intercept_point_at_y(
            self.ball_pos,
            self.ball_vel,
            y_target=self.paddle_pos[1],
            gravity=DEFAULT_GRAVITY_M_S2,
            max_time_s=self.cfg.dataset.max_intercept_horizon_s,
        )
        self.state.predicted_intercept = (
            np.array(intercept, dtype=np.float64) if intercept is not None else None
        )
        landing = predict_ballistic_landing(
            self.ball_pos,
            self.ball_vel,
            z_target=TABLE_TOP_Z_M + BALL_RADIUS_M,
            gravity=DEFAULT_GRAVITY_M_S2,
        )
        self.state.predicted_landing = (
            np.array(landing, dtype=np.float64) if landing is not None else None
        )

    def _refresh_planner(self, *, reset_path: bool) -> None:
        if not self.cfg.planner.enabled:
            self.current_hit_plan = HitPlan.invalid(self.target_landing_xy)
            cmd_qpos = np.array(self.cfg.control.home_qpos, dtype=np.float64)
            cmd_qvel = np.zeros(self.cfg.action_dim, dtype=np.float64)
            self.current_nominal_cmd_qpos = cmd_qpos
            self.current_nominal_cmd_qvel = cmd_qvel
            self._update_runtime_planner_state(self.current_hit_plan, cmd_qpos, cmd_qvel)
            return

        if not self.state.has_hit:
            new_plan = self.impact_planner.plan(
                ball_pos=self.ball_pos,
                ball_vel=self.ball_vel,
                paddle_qpos=self.paddle_qpos,
                target_landing_xy=self.target_landing_xy,
                bounce_has_occurred=self.state.own_side_bounce_count > 0,
            )
            if self._should_reset_hit_plan(new_plan, force=reset_path, now_s=self.episode_time_s):
                self.current_hit_plan = new_plan
                self.path_planner.reset(
                    current_qpos=self.paddle_qpos,
                    hit_plan=self.current_hit_plan,
                    now_s=self.episode_time_s,
                )

        command = self.path_planner.command(self.episode_time_s)
        self.current_nominal_cmd_qpos = command.qpos.copy()
        self.current_nominal_cmd_qvel = command.qvel.copy()
        self._update_runtime_planner_state(self.current_hit_plan, command.qpos, command.qvel)

    def _should_reset_hit_plan(self, new_plan: HitPlan, *, force: bool, now_s: float) -> bool:
        return should_reset_hit_plan(
            current_hit_plan=self.current_hit_plan,
            new_hit_plan=new_plan,
            force=force,
            has_hit=self.state.has_hit,
            now_s=now_s,
            path_planner=self.path_planner,
        )

    def _update_runtime_planner_state(
        self,
        hit_plan: HitPlan,
        cmd_qpos: np.ndarray,
        cmd_qvel: np.ndarray,
    ) -> None:
        self.state.planner_valid = hit_plan.valid
        self.state.planned_hit_pos = hit_plan.hit_pos.copy() if hit_plan.valid else None
        self.state.planned_hit_euler = hit_plan.hit_euler.copy() if hit_plan.valid else None
        self.state.planned_hit_time_s = (
            max(0.0, self.path_planner.start_time_s + hit_plan.hit_time_s - self.episode_time_s)
            if hit_plan.valid
            else 0.0
        )
        self.state.planned_cmd_qpos = np.asarray(cmd_qpos, dtype=np.float64).copy()
        self.state.planned_cmd_qvel = np.asarray(cmd_qvel, dtype=np.float64).copy()
        self.state.planned_outgoing_vel = hit_plan.outgoing_vel_des.copy() if hit_plan.valid else None
        self.state.target_landing_xy = self.target_landing_xy.copy()

    @staticmethod
    def _wrapped_euler_delta(delta: np.ndarray) -> np.ndarray:
        return (np.asarray(delta, dtype=np.float64) + np.pi) % (2.0 * np.pi) - np.pi

    def _handle_events(self) -> dict[str, float]:
        rewards: dict[str, float] = {}
        contacts = self._scan_contacts()
        new_table_contact = contacts.ball_table and not self.prev_contacts.ball_table
        new_paddle_contact = contacts.ball_paddle and not self.prev_contacts.ball_paddle
        new_net_contact = contacts.ball_net and not self.prev_contacts.ball_net

        if new_table_contact and not self.state.has_hit and point_on_own_side(self.ball_pos[1]):
            self.state.own_side_bounce_count += 1
            if self.state.own_side_bounce_count > 1:
                self.state.failure_reason = "double_bounce"
                rewards = self._merge_rewards(
                    rewards,
                    penalty("double_bounce_penalty", self.cfg.reward.double_bounce_penalty),
                )

        if new_paddle_contact:
            self.state.paddle_hit_count += 1
            if self.state.paddle_hit_count > 1:
                self.state.failure_reason = "multi_hit"
                rewards = self._merge_rewards(
                    rewards,
                    penalty("multi_hit_penalty", self.cfg.reward.multi_hit_penalty),
                )
            elif self.cfg.termination.require_bounce_before_hit and self.state.own_side_bounce_count == 0:
                self.state.failure_reason = "illegal_early_hit"
                rewards = self._merge_rewards(
                    rewards,
                    penalty("wrong_landing_penalty", self.cfg.reward.wrong_landing_penalty),
                )
            else:
                self.state.has_hit = True
                self.state.legal_hit = True
                self.state.first_hit_ball_speed_m_s = float(np.linalg.norm(self.ball_vel))
                if self.state.planned_outgoing_vel is not None:
                    desired_dir = normalize(self.state.planned_outgoing_vel)
                    actual_dir = normalize(self.ball_vel)
                    self.state.outgoing_direction_score = float(np.clip(np.dot(desired_dir, actual_dir), 0.0, 1.0))
                rewards = self._merge_rewards(rewards, reward_for_legal_hit(self.cfg.reward, self.ball_vel))

        if self.state.has_hit and not self.state.crossed_net_after_hit:
            crossed_plane = self.prev_ball_pos[1] < 0.0 <= self.ball_pos[1]
            if crossed_plane:
                if self.ball_pos[2] > NET_HEIGHT_M + BALL_RADIUS_M and not contacts.ball_net:
                    self.state.crossed_net_after_hit = True
                    rewards = self._merge_rewards(rewards, reward_for_cross_net(self.cfg.reward))
                else:
                    self.state.failure_reason = "net_fail"
                    rewards = self._merge_rewards(
                        rewards,
                        penalty("net_fail_penalty", self.cfg.reward.net_fail_penalty),
                    )

        if self.state.has_hit and new_net_contact:
            self.state.net_contact_after_hit = True
            self.state.failure_reason = "net_fail"
            rewards = self._merge_rewards(
                rewards,
                penalty("net_fail_penalty", self.cfg.reward.net_fail_penalty),
            )

        if self.state.has_hit and new_table_contact:
            self.state.landing_error_m = float(np.linalg.norm(self.ball_pos[:2] - self.target_landing_xy[:2]))
            if point_on_opponent_side(self.ball_pos[1]) and point_in_table_bounds(float(self.ball_pos[0]), float(self.ball_pos[1])):
                self.state.success = True
                rewards = self._merge_rewards(rewards, reward_for_success(self.cfg.reward))
            else:
                self.state.failure_reason = "wrong_landing"
                rewards = self._merge_rewards(
                    rewards,
                    penalty("wrong_landing_penalty", self.cfg.reward.wrong_landing_penalty),
                )

        if contacts.ball_floor and not self.state.success and not self.state.failure_reason:
            self.state.failure_reason = "ball_floor"
            rewards = self._merge_rewards(
                rewards,
                penalty("floor_penalty", self.cfg.reward.floor_penalty),
            )

        return rewards

    def _dense_reward(self, action: np.ndarray) -> dict[str, float]:
        return dense_reward(
            reward_cfg=self.cfg.reward,
            runtime_state=self.state,
            ball_vel=self.ball_vel,
            paddle_qpos=self.paddle_qpos,
            paddle_qvel=self.paddle_qvel,
            predicted_landing=self.state.predicted_landing,
            target_landing_xy=self.target_landing_xy,
            action=action,
            prev_action=self.prev_action,
        )

    def _failure_penalty(self, reason: str) -> dict[str, float]:
        if reason == "ball_floor":
            return penalty("floor_penalty", self.cfg.reward.floor_penalty)
        if reason == "ball_out_of_bounds":
            return penalty("out_of_bounds_penalty", self.cfg.reward.out_of_bounds_penalty)
        if reason == "double_bounce":
            return penalty("double_bounce_penalty", self.cfg.reward.double_bounce_penalty)
        if reason == "multi_hit":
            return penalty("multi_hit_penalty", self.cfg.reward.multi_hit_penalty)
        if reason == "net_fail":
            return penalty("net_fail_penalty", self.cfg.reward.net_fail_penalty)
        return penalty("wrong_landing_penalty", self.cfg.reward.wrong_landing_penalty)

    def _build_observation(self) -> np.ndarray:
        return build_observation(
            ball_pos=self.ball_pos,
            ball_vel=self.ball_vel,
            paddle_qpos=self.paddle_qpos,
            paddle_qvel=self.paddle_qvel,
            prev_action=self.prev_action,
            runtime_state=self.state,
        )

    def _build_info(self, episode_success: bool) -> dict[str, Any]:
        terminal_reason = "success" if episode_success else self.state.failure_reason
        return {
            "task_id": self.cfg.task_id,
            "task_mode": self.cfg.task_mode,
            "control_backend": self.cfg.control.control_backend,
            "control_mode": self.cfg.control.control_mode,
            "difficulty": self.difficulty,
            "curriculum_level": self.curriculum_level,
            "serve_id": self.current_serve.id if self.current_serve is not None else None,
            "episode_success": episode_success,
            "planner_valid": self.state.planner_valid,
            "legal_hit": self.state.legal_hit,
            "cross_net": self.state.crossed_net_after_hit,
            "opponent_landing": self.state.success,
            "first_hit_ball_speed_m_s": self.state.first_hit_ball_speed_m_s,
            "outgoing_direction_score": self.state.outgoing_direction_score,
            "landing_error_m": self.state.landing_error_m,
            "event_double_bounce": terminal_reason == "double_bounce",
            "event_multi_hit": terminal_reason == "multi_hit",
            "event_net_fail": terminal_reason == "net_fail",
            "event_wrong_landing": terminal_reason == "wrong_landing",
            "event_floor": terminal_reason == "ball_floor",
            "event_out_of_bounds": terminal_reason == "ball_out_of_bounds",
            "target_landing_xy": self.target_landing_xy.copy(),
            "physics_timestep_s": self.physics_timestep_s,
            "control_timestep_s": self.control_timestep_s,
            "max_steps": self.max_steps,
        }

    @staticmethod
    def _merge_rewards(base: dict[str, float], delta: dict[str, float]) -> dict[str, float]:
        merged = dict(base)
        for key, value in delta.items():
            merged[key] = merged.get(key, 0.0) + value
        return merged
