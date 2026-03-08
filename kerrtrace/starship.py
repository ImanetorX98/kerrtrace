from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch

from .config import RenderConfig
from .geometry import event_horizon_radius, inverse_metric_components, inverse_metric_derivatives, metric_components
from .raytracer import PointEmitter


@dataclass(frozen=True)
class StarshipThrustCommand:
    acceleration: float = 0.0
    direction_mode: str = "azimuthal_prograde"
    direction_vector: tuple[float, float, float] = (0.0, 0.0, 1.0)


@dataclass(frozen=True)
class StarshipThrustSegment:
    start_time: float
    end_time: float
    command: StarshipThrustCommand


class Starship:
    """
    Single controllable ship orbiting the black hole.

    The ship follows timelike dynamics in (t, r, theta, phi, p_t, p_r, p_theta, p_phi)
    and supports programmable thrust in user-selected local directions.
    """

    def __init__(
        self,
        config: RenderConfig,
        radius: float = 11.0,
        theta_deg: float = 62.0,
        phi_deg: float = 20.0,
        v_phi: float = 0.46,
        v_theta: float = 0.04,
        v_r: float = 0.0,
    ) -> None:
        self.config = config.validated()
        self.device = self.config.resolve_device()
        self.dtype = self.config.resolve_dtype()
        self.horizon = float(
            event_horizon_radius(
                self.config.spin,
                self.config.metric_model,
                self.config.charge,
                self.config.cosmological_constant,
            )
        )
        self._time = 0.0
        self._alive = True
        self._manual_command = StarshipThrustCommand()
        self._program: list[StarshipThrustSegment] = []
        self._program_idx = 0
        self._mode_direction_vectors = {
            "radial_out": torch.tensor([1.0, 0.0, 0.0], dtype=self.dtype, device=self.device),
            "radial": torch.tensor([1.0, 0.0, 0.0], dtype=self.dtype, device=self.device),
            "radial_in": torch.tensor([-1.0, 0.0, 0.0], dtype=self.dtype, device=self.device),
            "azimuthal_prograde": torch.tensor([0.0, 0.0, 1.0], dtype=self.dtype, device=self.device),
            "azimuthal": torch.tensor([0.0, 0.0, 1.0], dtype=self.dtype, device=self.device),
            "phi_plus": torch.tensor([0.0, 0.0, 1.0], dtype=self.dtype, device=self.device),
            "azimuthal_retrograde": torch.tensor([0.0, 0.0, -1.0], dtype=self.dtype, device=self.device),
            "phi_minus": torch.tensor([0.0, 0.0, -1.0], dtype=self.dtype, device=self.device),
            "polar_north": torch.tensor([0.0, -1.0, 0.0], dtype=self.dtype, device=self.device),
            "theta_minus": torch.tensor([0.0, -1.0, 0.0], dtype=self.dtype, device=self.device),
            "polar_south": torch.tensor([0.0, 1.0, 0.0], dtype=self.dtype, device=self.device),
            "polar": torch.tensor([0.0, 1.0, 0.0], dtype=self.dtype, device=self.device),
            "theta_plus": torch.tensor([0.0, 1.0, 0.0], dtype=self.dtype, device=self.device),
        }
        self._default_direction = torch.tensor([0.0, 0.0, 1.0], dtype=self.dtype, device=self.device)
        self._state = self._initial_state(
            radius=radius,
            theta_deg=theta_deg,
            phi_deg=phi_deg,
            v_phi=v_phi,
            v_theta=v_theta,
            v_r=v_r,
        )

    @property
    def proper_time(self) -> float:
        return self._time

    @property
    def alive(self) -> bool:
        return self._alive

    def set_acceleration(
        self,
        acceleration: float,
        direction_mode: str = "custom",
        direction_vector: tuple[float, float, float] = (0.0, 0.0, 1.0),
    ) -> None:
        self._manual_command = StarshipThrustCommand(
            acceleration=max(0.0, float(acceleration)),
            direction_mode=str(direction_mode),
            direction_vector=tuple(float(v) for v in direction_vector),
        )

    def set_acceleration_program(self, segments: Sequence[StarshipThrustSegment]) -> None:
        ordered = sorted(list(segments), key=lambda s: (s.start_time, s.end_time))
        for seg in ordered:
            if seg.end_time <= seg.start_time:
                raise ValueError("Each thrust segment must satisfy end_time > start_time")
        self._program = ordered
        self._program_idx = 0

    def clear_acceleration_program(self) -> None:
        self._program = []
        self._program_idx = 0

    def state_dict(self) -> dict[str, float | bool]:
        st = self._state[0]
        return {
            "alive": self._alive,
            "proper_time": float(self._time),
            "t": float(st[0].item()),
            "r": float(st[1].item()),
            "theta": float(st[2].item()),
            "phi": float(st[3].item()),
            "p_t": float(st[4].item()),
            "p_r": float(st[5].item()),
            "p_theta": float(st[6].item()),
            "p_phi": float(st[7].item()),
        }

    def step(self, dt: float, substeps: int = 1) -> None:
        if dt <= 0.0:
            raise ValueError("dt must be > 0")
        if substeps < 1:
            raise ValueError("substeps must be >= 1")
        h = float(dt) / float(substeps)

        for _ in range(substeps):
            if not self._alive:
                break
            cmd = self._active_command(self._time)
            self._state = self._rk4_step(self._state, h, cmd)
            self._state[:, 1] = torch.clamp(self._state[:, 1], min=1.0e-3)
            self._state[:, 2] = torch.clamp(self._state[:, 2], min=1.0e-4, max=math.pi - 1.0e-4)
            self._state[:, 3] = torch.remainder(self._state[:, 3], 2.0 * math.pi)
            self._time += h

            r_now = float(self._state[0, 1].item())
            if r_now <= 1.005 * self.horizon or r_now >= float(self.config.escape_radius):
                self._alive = False

    def to_point_emitter(
        self,
        radius: float = 0.08,
        intensity: float = 0.95,
        color_rgb: tuple[float, float, float] = (0.35, 0.85, 1.0),
    ) -> PointEmitter | None:
        if not self._alive:
            return None
        state = self._state
        r = state[:, 1]
        theta = state[:, 2]
        phi = state[:, 3]
        p_t = state[:, 4]
        p_r = state[:, 5]
        p_th = state[:, 6]
        p_phi = state[:, 7]

        inv = self._inv_metric(r, theta)
        u_t = (inv.gtt * p_t + inv.gtphi * p_phi)[0]
        u_r = (inv.grr * p_r)[0]
        u_th = (inv.gthth * p_th)[0]
        u_phi = (inv.gtphi * p_t + inv.gphiphi * p_phi)[0]

        return PointEmitter(
            r=float(r[0].item()),
            theta=float(theta[0].item()),
            phi=float(phi[0].item()),
            u_t=float(u_t.item()),
            u_r=float(u_r.item()),
            u_theta=float(u_th.item()),
            u_phi=float(u_phi.item()),
            radius=float(radius),
            intensity=float(intensity),
            color_rgb=color_rgb,
        )

    def to_composite_emitters(
        self,
        model_scale: float = 1.0,
        intensity: float = 1.1,
        color_rgb: tuple[float, float, float] = (0.84, 0.88, 0.95),
    ) -> list[PointEmitter]:
        """
        Build a simple spaceship silhouette as a union of small emitters.
        The layout is expressed in local (r, theta, phi) offsets around the ship state.
        """
        base = self.to_point_emitter(
            radius=0.10 * model_scale,
            intensity=intensity,
            color_rgb=color_rgb,
        )
        if base is None:
            return []

        r0 = max(1.0e-4, float(base.r))
        th0 = float(base.theta)
        ph0 = float(base.phi)
        sin_th = max(1.0e-4, abs(math.sin(th0)))

        # Local conversion: length scale -> angular offsets.
        dphi = (0.32 * model_scale) / (r0 * sin_th)
        dtheta = (0.18 * model_scale) / r0
        dr = 0.14 * model_scale

        # (dr, dtheta, dphi, radius_scale) - dense overlap to avoid "bubble" look.
        components: list[tuple[float, float, float, float]] = []
        # Fuselage spine.
        for i in range(-6, 7):
            s = i / 6.0
            rad = max(0.56, 1.22 - 0.55 * abs(s))
            components.append((0.05 * dr, 0.0, 1.45 * s * dphi, rad))
        # Upper/lower wings.
        for i in range(-5, 6):
            s = i / 5.0
            lat = 1.20 * s * dphi
            wing_h = (0.70 + 0.30 * (1.0 - abs(s))) * dtheta
            wing_rad = max(0.34, 0.54 - 0.12 * abs(s))
            components.append((-0.12 * dr, +wing_h, lat, wing_rad))
            components.append((-0.12 * dr, -wing_h, lat, wing_rad))
        # Nose and engine pods.
        components.extend(
            [
                (0.16 * dr, 0.00, 0.0, 0.92),
                (-0.34 * dr, 0.00, +1.58 * dphi, 0.62),
                (-0.34 * dr, 0.00, -1.58 * dphi, 0.62),
            ]
        )

        out: list[PointEmitter] = []
        base_radius = max(0.02, float(base.radius))
        for dr_i, dth_i, dph_i, rs in components:
            th = min(math.pi - 2.0e-3, max(2.0e-3, th0 + dth_i))
            ph = (ph0 + dph_i) % (2.0 * math.pi)
            out.append(
                PointEmitter(
                    r=max(1.0e-4, r0 + dr_i),
                    theta=th,
                    phi=ph,
                    u_t=base.u_t,
                    u_r=base.u_r,
                    u_theta=base.u_theta,
                    u_phi=base.u_phi,
                    radius=base_radius * rs,
                    intensity=float(base.intensity),
                    color_rgb=base.color_rgb,
                )
            )
        return out

    def _active_command(self, t: float) -> StarshipThrustCommand:
        if self._program_idx >= len(self._program):
            return self._manual_command

        while self._program_idx < len(self._program) and t >= self._program[self._program_idx].end_time:
            self._program_idx += 1

        if self._program_idx < len(self._program):
            seg = self._program[self._program_idx]
            if seg.start_time <= t < seg.end_time:
                return seg.command
        return self._manual_command

    def _direction_local(self, command: StarshipThrustCommand) -> torch.Tensor:
        mode = command.direction_mode.strip().lower()
        if mode == "custom":
            d = torch.as_tensor(command.direction_vector, dtype=self.dtype, device=self.device)
        elif mode in self._mode_direction_vectors:
            d = self._mode_direction_vectors[mode]
        else:
            raise ValueError(f"Unknown thrust direction_mode: {command.direction_mode}")

        nrm = torch.linalg.norm(d)
        if float(nrm.item()) < 1.0e-12:
            return self._default_direction
        return d / nrm

    def _metric(self, r: torch.Tensor, theta: torch.Tensor):
        return metric_components(
            r,
            theta,
            self.config.spin,
            self.config.metric_model,
            self.config.charge,
            self.config.cosmological_constant,
        )

    def _inv_metric(self, r: torch.Tensor, theta: torch.Tensor):
        return inverse_metric_components(
            r,
            theta,
            self.config.spin,
            self.config.metric_model,
            self.config.charge,
            self.config.cosmological_constant,
        )

    def _inv_metric_derivs(self, r: torch.Tensor, theta: torch.Tensor):
        return inverse_metric_derivatives(
            r,
            theta,
            self.config.spin,
            self.config.metric_model,
            self.config.charge,
            self.config.cosmological_constant,
        )

    def _initial_state(
        self,
        radius: float,
        theta_deg: float,
        phi_deg: float,
        v_phi: float,
        v_theta: float,
        v_r: float = 0.0,
    ) -> torch.Tensor:
        r0 = torch.tensor([max(radius, 1.15 * self.horizon + 0.2)], dtype=self.dtype, device=self.device)
        theta0 = torch.tensor([math.radians(theta_deg)], dtype=self.dtype, device=self.device)
        theta0 = torch.clamp(theta0, min=2.0e-3, max=math.pi - 2.0e-3)
        phi0 = torch.tensor([math.radians(phi_deg)], dtype=self.dtype, device=self.device)
        t0 = torch.zeros((1,), dtype=self.dtype, device=self.device)

        metric = self._metric(r0, theta0)
        omega = -metric.g_tphi / torch.clamp(metric.g_phiphi, min=1.0e-9)
        lapse = torch.sqrt(
            torch.clamp(
                -(metric.g_tt - (metric.g_tphi * metric.g_tphi) / torch.clamp(metric.g_phiphi, min=1.0e-9)),
                min=1.0e-12,
            )
        )

        e_r_up = 1.0 / torch.sqrt(torch.clamp(metric.g_rr, min=1.0e-12))
        e_th_up = 1.0 / torch.sqrt(torch.clamp(metric.g_thth, min=1.0e-12))
        e_phi_up = 1.0 / torch.sqrt(torch.clamp(metric.g_phiphi, min=1.0e-12))

        v_r_t = torch.tensor([v_r], dtype=self.dtype, device=self.device)
        v_th_t = torch.tensor([v_theta], dtype=self.dtype, device=self.device)
        v_phi_t = torch.tensor([v_phi], dtype=self.dtype, device=self.device)
        v2 = torch.clamp(v_r_t * v_r_t + v_th_t * v_th_t + v_phi_t * v_phi_t, max=0.92)
        gamma = torch.rsqrt(torch.clamp(1.0 - v2, min=1.0e-6))

        u_t_up = gamma / lapse
        u_r_up = gamma * v_r_t * e_r_up
        u_th_up = gamma * v_th_t * e_th_up
        u_phi_up = gamma * (omega / lapse + v_phi_t * e_phi_up)

        p_t = metric.g_tt * u_t_up + metric.g_tphi * u_phi_up
        p_r = metric.g_rr * u_r_up
        p_th = metric.g_thth * u_th_up
        p_phi = metric.g_tphi * u_t_up + metric.g_phiphi * u_phi_up

        return torch.stack([t0, r0, theta0, phi0, p_t, p_r, p_th, p_phi], dim=-1).reshape(1, 8)

    def _rhs(self, state: torch.Tensor, command: StarshipThrustCommand) -> torch.Tensor:
        r = torch.clamp(state[:, 1], min=1.0e-3)
        theta = torch.clamp(state[:, 2], min=1.0e-4, max=math.pi - 1.0e-4)
        p_t = state[:, 4]
        p_r = state[:, 5]
        p_th = state[:, 6]
        p_phi = state[:, 7]

        inv = self._inv_metric(r, theta)
        d_r, d_th = self._inv_metric_derivs(r, theta)

        dt = inv.gtt * p_t + inv.gtphi * p_phi
        dr = inv.grr * p_r
        dtheta = inv.gthth * p_th
        dphi = inv.gtphi * p_t + inv.gphiphi * p_phi

        p_t2 = p_t * p_t
        p_r2 = p_r * p_r
        p_th2 = p_th * p_th
        p_phi2 = p_phi * p_phi

        contract_r = (
            d_r.gtt * p_t2
            + 2.0 * d_r.gtphi * p_t * p_phi
            + d_r.grr * p_r2
            + d_r.gthth * p_th2
            + d_r.gphiphi * p_phi2
        )
        contract_th = (
            d_th.gtt * p_t2
            + 2.0 * d_th.gtphi * p_t * p_phi
            + d_th.grr * p_r2
            + d_th.gthth * p_th2
            + d_th.gphiphi * p_phi2
        )

        dp_t = torch.zeros_like(p_t)
        dp_r = -0.5 * contract_r
        dp_th = -0.5 * contract_th
        dp_phi = torch.zeros_like(p_phi)

        accel = max(0.0, float(command.acceleration))
        if accel > 0.0:
            d_local = self._direction_local(command).view(1, 3)
            metric = self._metric(r, theta)
            scale_r = torch.sqrt(torch.clamp(metric.g_rr, min=1.0e-12))
            scale_th = torch.sqrt(torch.clamp(metric.g_thth, min=1.0e-12))
            scale_phi = torch.sqrt(torch.clamp(metric.g_phiphi, min=1.0e-12))
            accel_t = torch.as_tensor(accel, dtype=self.dtype, device=self.device)
            dp_r = dp_r + accel_t * d_local[:, 0] * scale_r
            dp_th = dp_th + accel_t * d_local[:, 1] * scale_th
            dp_phi = dp_phi + accel_t * d_local[:, 2] * scale_phi

        return torch.stack([dt, dr, dtheta, dphi, dp_t, dp_r, dp_th, dp_phi], dim=-1)

    def _rk4_step(self, state: torch.Tensor, h: float, command: StarshipThrustCommand) -> torch.Tensor:
        k1 = self._rhs(state, command)
        k2 = self._rhs(state + 0.5 * h * k1, command)
        k3 = self._rhs(state + 0.5 * h * k2, command)
        k4 = self._rhs(state + h * k3, command)
        nxt = state + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        nxt[:, 1] = torch.clamp(nxt[:, 1], min=1.0e-3)
        nxt[:, 2] = torch.clamp(nxt[:, 2], min=1.0e-4, max=math.pi - 1.0e-4)
        nxt[:, 3] = torch.remainder(nxt[:, 3], 2.0 * math.pi)
        return nxt
