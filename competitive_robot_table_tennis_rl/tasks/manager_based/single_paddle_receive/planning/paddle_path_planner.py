"""Artificial-potential-field paddle path planning for the receive task."""

from __future__ import annotations

import numpy as np

from ..env_cfg import ControlCfg, PlannerCfg
from ..mdp.transforms import normalize, paddle_normal_from_euler
from .planner_types import HitPlan, PaddleCommand


def should_reset_hit_plan(
    *,
    current_hit_plan: HitPlan,
    new_hit_plan: HitPlan,
    force: bool,
    has_hit: bool,
    now_s: float,
    path_planner: "PaddlePathPlanner",
) -> bool:
    """Decide whether replanning should replace the current nominal hit plan."""

    if force:
        return True
    if has_hit:
        return False
    if current_hit_plan.valid and not new_hit_plan.valid:
        return not path_planner.is_plan_active(now_s, current_hit_plan)
    if current_hit_plan.valid != new_hit_plan.valid:
        return True
    if not new_hit_plan.valid:
        return False

    hit_pos_delta = float(np.linalg.norm(new_hit_plan.hit_pos - current_hit_plan.hit_pos))
    hit_euler_delta = float(
        np.linalg.norm(PaddlePathPlanner.wrapped_euler_delta(new_hit_plan.hit_euler - current_hit_plan.hit_euler))
    )
    hit_time_delta = abs(float(new_hit_plan.hit_time_s - current_hit_plan.hit_time_s))
    return hit_pos_delta > 0.05 or hit_euler_delta > 0.20 or hit_time_delta > 0.10


def _smoothstep5(tau: float) -> float:
    """Fifth-order smoothstep interpolation."""

    tau = float(np.clip(tau, 0.0, 1.0))
    return 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5


def _smoothstep5_derivative(tau: float) -> float:
    """Time derivative of the fifth-order smoothstep basis."""

    tau = float(np.clip(tau, 0.0, 1.0))
    return 30.0 * tau**2 - 60.0 * tau**3 + 30.0 * tau**4


class PaddlePathPlanner:
    """Generate a nominal paddle command with APF-guided pre-hit motion."""

    def __init__(
        self,
        *,
        control_cfg: ControlCfg,
        planner_cfg: PlannerCfg,
    ) -> None:
        self.control_cfg = control_cfg
        self.planner_cfg = planner_cfg
        self.home_qpos = np.asarray(control_cfg.home_qpos, dtype=np.float64).copy()
        self.follow_through_time_s = float(planner_cfg.follow_through_time_s)
        self.follow_through_distance_m = float(planner_cfg.follow_through_distance_m)
        self.integration_dt_s = float(planner_cfg.apf_integration_dt_s)
        self.stage_standoff_m = float(planner_cfg.apf_stage_standoff_m)
        self.stage_release_time_s = float(planner_cfg.apf_stage_release_time_s)
        self.linear_gain = float(planner_cfg.apf_linear_gain)
        self.angular_gain = float(planner_cfg.apf_angular_gain)
        self.linear_max_speed = float(planner_cfg.apf_linear_max_speed_m_s)
        self.angular_max_speed = float(planner_cfg.apf_angular_max_speed_rad_s)
        self.boundary_margin_m = float(planner_cfg.apf_boundary_margin_m)
        self.boundary_repulsion_gain = float(planner_cfg.apf_boundary_repulsion_gain)
        self.lower_limits = np.array(
            [
                control_cfg.x_range_m[0],
                control_cfg.y_range_m[0],
                control_cfg.z_range_m[0],
                control_cfg.roll_range_rad[0],
                control_cfg.pitch_range_rad[0],
                control_cfg.yaw_range_rad[0],
            ],
            dtype=np.float64,
        )
        self.upper_limits = np.array(
            [
                control_cfg.x_range_m[1],
                control_cfg.y_range_m[1],
                control_cfg.z_range_m[1],
                control_cfg.roll_range_rad[1],
                control_cfg.pitch_range_rad[1],
                control_cfg.yaw_range_rad[1],
            ],
            dtype=np.float64,
        )
        self.start_qpos = self.home_qpos.copy()
        self.start_time_s = 0.0
        self.hit_plan = HitPlan.invalid()
        self.last_qpos = self.home_qpos.copy()
        self.last_qvel = np.zeros_like(self.home_qpos)
        self.last_time_s = 0.0

    def reset(
        self,
        *,
        current_qpos: np.ndarray,
        hit_plan: HitPlan,
        now_s: float,
    ) -> None:
        """Reset the nominal motion from the current pose to a new hit plan."""

        self.start_qpos = np.asarray(current_qpos, dtype=np.float64).copy()
        self.start_time_s = float(now_s)
        self.hit_plan = hit_plan
        self.last_qpos = self.start_qpos.copy()
        self.last_qvel = np.zeros_like(self.start_qpos)
        self.last_time_s = float(now_s)

    def command(self, now_s: float) -> PaddleCommand:
        """Return the nominal command at the requested time."""

        now_s = float(now_s)
        if not self.hit_plan.valid:
            home_cmd = PaddleCommand(qpos=self.home_qpos.copy(), qvel=np.zeros_like(self.home_qpos))
            self.last_qpos = home_cmd.qpos.copy()
            self.last_qvel = home_cmd.qvel.copy()
            self.last_time_s = now_s
            return home_cmd

        hit_qpos = np.concatenate([self.hit_plan.hit_pos, self.hit_plan.hit_euler]).astype(np.float64, copy=False)
        follow_qpos = hit_qpos.copy()
        outgoing_dir = self.hit_plan.outgoing_vel_des
        norm = float(np.linalg.norm(outgoing_dir))
        if norm > 1e-8:
            follow_qpos[:3] += outgoing_dir / norm * self.follow_through_distance_m
        stage_qpos = self._stage_qpos(hit_qpos)

        prehit_end_s = self.start_time_s + max(self.hit_plan.hit_time_s, 1e-3)
        follow_end_s = prehit_end_s + self.follow_through_time_s
        return_end_s = follow_end_s + self.follow_through_time_s
        if now_s <= self.start_time_s:
            cmd = PaddleCommand(qpos=self.start_qpos.copy(), qvel=np.zeros_like(self.start_qpos))
        elif now_s <= prehit_end_s:
            cmd = self._advance_apf(
                now_s=now_s,
                prehit_end_s=prehit_end_s,
                hit_qpos=hit_qpos,
                stage_qpos=stage_qpos,
            )
        elif now_s <= follow_end_s:
            cmd = self._interpolate(
                start=hit_qpos,
                end=follow_qpos,
                now_s=now_s,
                start_s=prehit_end_s,
                end_s=follow_end_s,
            )
        elif now_s <= return_end_s:
            cmd = self._interpolate(
                start=follow_qpos,
                end=self.home_qpos,
                now_s=now_s,
                start_s=follow_end_s,
                end_s=return_end_s,
            )
        else:
            cmd = PaddleCommand(qpos=self.home_qpos.copy(), qvel=np.zeros_like(self.home_qpos))

        self.last_qpos = cmd.qpos.copy()
        self.last_qvel = cmd.qvel.copy()
        self.last_time_s = now_s
        return cmd

    def plan_end_time_s(self, hit_plan: HitPlan | None = None) -> float:
        """Return the nominal completion time of the current or provided plan."""

        plan = self.hit_plan if hit_plan is None else hit_plan
        if not plan.valid:
            return self.start_time_s
        return self.start_time_s + float(plan.hit_time_s) + 2.0 * self.follow_through_time_s

    def is_plan_active(self, now_s: float, hit_plan: HitPlan | None = None) -> bool:
        """Return whether the nominal motion for a hit plan should still be tracked."""

        plan = self.hit_plan if hit_plan is None else hit_plan
        if not plan.valid:
            return False
        return float(now_s) <= self.plan_end_time_s(plan) + 1e-6

    def _stage_qpos(self, hit_qpos: np.ndarray) -> np.ndarray:
        stage_qpos = hit_qpos.copy()
        hit_normal = paddle_normal_from_euler(hit_qpos[3:])
        stage_qpos[:3] -= hit_normal * self.stage_standoff_m
        return self._clip_qpos(stage_qpos)

    def _advance_apf(
        self,
        *,
        now_s: float,
        prehit_end_s: float,
        hit_qpos: np.ndarray,
        stage_qpos: np.ndarray,
    ) -> PaddleCommand:
        if now_s <= self.last_time_s:
            return PaddleCommand(qpos=self.last_qpos.copy(), qvel=self.last_qvel.copy())

        qpos = self.last_qpos.copy()
        qvel = self.last_qvel.copy()
        sim_time_s = self.last_time_s
        while sim_time_s < now_s - 1e-9:
            dt = min(self.integration_dt_s, now_s - sim_time_s)
            remaining_s = max(prehit_end_s - sim_time_s, dt)
            target_qpos = self._scheduled_prehit_target(
                query_time_s=sim_time_s + dt,
                prehit_end_s=prehit_end_s,
                hit_qpos=hit_qpos,
                stage_qpos=stage_qpos,
            )
            qvel = self._apf_velocity(
                qpos=qpos,
                target_qpos=target_qpos,
                hit_qpos=hit_qpos,
                remaining_s=remaining_s,
            )
            qpos = self._clip_qpos(qpos + qvel * dt)
            sim_time_s += dt

        if now_s >= prehit_end_s - 1e-6:
            qvel = self._apf_velocity(
                qpos=hit_qpos,
                target_qpos=hit_qpos,
                hit_qpos=hit_qpos,
                remaining_s=max(self.integration_dt_s, 1e-3),
            )
            qpos = hit_qpos.copy()

        return PaddleCommand(qpos=qpos, qvel=qvel)

    def _scheduled_prehit_target(
        self,
        *,
        query_time_s: float,
        prehit_end_s: float,
        hit_qpos: np.ndarray,
        stage_qpos: np.ndarray,
    ) -> np.ndarray:
        release_s = min(self.stage_release_time_s, max(0.5 * (prehit_end_s - self.start_time_s), 1e-3))
        stage_end_s = max(self.start_time_s, prehit_end_s - release_s)
        if stage_end_s <= self.start_time_s + 1e-6:
            return hit_qpos.copy()
        if query_time_s <= stage_end_s:
            return self._interpolate(
                start=self.start_qpos,
                end=stage_qpos,
                now_s=query_time_s,
                start_s=self.start_time_s,
                end_s=stage_end_s,
            ).qpos
        return self._interpolate(
            start=stage_qpos,
            end=hit_qpos,
            now_s=query_time_s,
            start_s=stage_end_s,
            end_s=prehit_end_s,
        ).qpos

    def _apf_velocity(
        self,
        *,
        qpos: np.ndarray,
        target_qpos: np.ndarray,
        hit_qpos: np.ndarray,
        remaining_s: float,
    ) -> np.ndarray:
        pos_error = target_qpos[:3] - qpos[:3]
        terminal_error = hit_qpos[:3] - qpos[:3]
        attractive_linear = self.linear_gain * pos_error
        terminal_linear = terminal_error / max(remaining_s, self.integration_dt_s)
        repulsive_linear = self._boundary_repulsion(qpos[:3])
        linear = attractive_linear + 0.35 * terminal_linear + repulsive_linear
        linear = self._clip_vector(linear, self.linear_max_speed)

        rot_error = self.wrapped_euler_delta(target_qpos[3:] - qpos[3:])
        terminal_rot_error = self.wrapped_euler_delta(hit_qpos[3:] - qpos[3:])
        angular = self.angular_gain * rot_error + 0.35 * terminal_rot_error / max(remaining_s, self.integration_dt_s)
        angular = self._clip_vector(angular, self.angular_max_speed)
        return np.concatenate([linear, angular]).astype(np.float64, copy=False)

    def _boundary_repulsion(self, pos: np.ndarray) -> np.ndarray:
        repulsion = np.zeros(3, dtype=np.float64)
        for axis in range(3):
            lower = float(self.lower_limits[axis])
            upper = float(self.upper_limits[axis])
            dist_lower = float(pos[axis] - lower)
            dist_upper = float(upper - pos[axis])
            if dist_lower < self.boundary_margin_m:
                repulsion[axis] += self._repulsion_strength(dist_lower)
            if dist_upper < self.boundary_margin_m:
                repulsion[axis] -= self._repulsion_strength(dist_upper)
        return repulsion

    def _repulsion_strength(self, distance: float) -> float:
        distance = max(float(distance), 1e-6)
        margin = max(self.boundary_margin_m, 1e-6)
        return self.boundary_repulsion_gain * (1.0 / distance - 1.0 / margin) / (distance * distance)

    def _clip_qpos(self, qpos: np.ndarray) -> np.ndarray:
        clipped = np.asarray(qpos, dtype=np.float64).copy()
        clipped[:3] = np.clip(clipped[:3], self.lower_limits[:3], self.upper_limits[:3])
        clipped[3:] = self.wrapped_euler_delta(clipped[3:])
        clipped[3:] = np.clip(clipped[3:], self.lower_limits[3:], self.upper_limits[3:])
        return clipped

    @staticmethod
    def _clip_vector(vector: np.ndarray, max_norm: float) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        if norm <= max_norm or norm < 1e-8:
            return np.asarray(vector, dtype=np.float64)
        return normalize(vector) * max_norm

    @staticmethod
    def wrapped_euler_delta(delta: np.ndarray) -> np.ndarray:
        return (np.asarray(delta, dtype=np.float64) + np.pi) % (2.0 * np.pi) - np.pi

    def _interpolate(
        self,
        *,
        start: np.ndarray,
        end: np.ndarray,
        now_s: float,
        start_s: float,
        end_s: float,
    ) -> PaddleCommand:
        duration_s = max(end_s - start_s, 1e-6)
        tau = (now_s - start_s) / duration_s
        s = _smoothstep5(tau)
        ds_dt = _smoothstep5_derivative(tau) / duration_s
        qpos = (1.0 - s) * start + s * end
        qvel = (end - start) * ds_dt
        return PaddleCommand(qpos=qpos.astype(np.float64, copy=False), qvel=qvel.astype(np.float64, copy=False))
