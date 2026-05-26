"""Nominal paddle path planning for the single-paddle receive task."""

from __future__ import annotations

import numpy as np

from .planner_types import HitPlan, PaddleCommand


def _smoothstep5(tau: float) -> float:
    """Fifth-order smoothstep interpolation."""

    tau = float(np.clip(tau, 0.0, 1.0))
    return 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5


def _smoothstep5_derivative(tau: float) -> float:
    """Time derivative of the fifth-order smoothstep basis."""

    tau = float(np.clip(tau, 0.0, 1.0))
    return 30.0 * tau**2 - 60.0 * tau**3 + 30.0 * tau**4


class PaddlePathPlanner:
    """Generate a smooth nominal command from the current paddle pose to a hit plan."""

    def __init__(
        self,
        *,
        home_qpos: np.ndarray,
        follow_through_time_s: float,
        follow_through_distance_m: float,
    ) -> None:
        self.home_qpos = np.asarray(home_qpos, dtype=np.float64).copy()
        self.follow_through_time_s = float(follow_through_time_s)
        self.follow_through_distance_m = float(follow_through_distance_m)
        self.start_qpos = self.home_qpos.copy()
        self.start_time_s = 0.0
        self.hit_plan = HitPlan.invalid()

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

    def command(self, now_s: float) -> PaddleCommand:
        """Return the nominal command at the requested time."""

        if not self.hit_plan.valid:
            return PaddleCommand(qpos=self.home_qpos.copy(), qvel=np.zeros_like(self.home_qpos))

        hit_qpos = np.concatenate([self.hit_plan.hit_pos, self.hit_plan.hit_euler]).astype(np.float64, copy=False)
        follow_qpos = hit_qpos.copy()
        outgoing_dir = self.hit_plan.outgoing_vel_des
        norm = float(np.linalg.norm(outgoing_dir))
        if norm > 1e-8:
            follow_qpos[:3] += outgoing_dir / norm * self.follow_through_distance_m

        prehit_end_s = self.start_time_s + max(self.hit_plan.hit_time_s, 1e-3)
        follow_end_s = prehit_end_s + self.follow_through_time_s
        return_end_s = follow_end_s + self.follow_through_time_s
        now_s = float(now_s)

        if now_s <= prehit_end_s:
            return self._interpolate(
                start=self.start_qpos,
                end=hit_qpos,
                now_s=now_s,
                start_s=self.start_time_s,
                end_s=prehit_end_s,
            )
        if now_s <= follow_end_s:
            return self._interpolate(
                start=hit_qpos,
                end=follow_qpos,
                now_s=now_s,
                start_s=prehit_end_s,
                end_s=follow_end_s,
            )
        if now_s <= return_end_s:
            return self._interpolate(
                start=follow_qpos,
                end=self.home_qpos,
                now_s=now_s,
                start_s=follow_end_s,
                end_s=return_end_s,
            )
        return PaddleCommand(qpos=self.home_qpos.copy(), qvel=np.zeros_like(self.home_qpos))

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
