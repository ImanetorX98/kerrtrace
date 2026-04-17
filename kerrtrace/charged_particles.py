from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
import math
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

from .animation import VIDEO_SUFFIXES, _encode_video_ffmpeg
from .config import RenderConfig
from .geometry import event_horizon_radius, inverse_metric_components, inverse_metric_derivatives
from .raytracer import KerrRayTracer, PointEmitter


@dataclass
class ChargedParticleAnimationStats:
    frames: int
    fps: int
    elapsed_seconds: float
    output_path: Path
    particles: int
    survivors: int


class ChargedParticleOrbiter:
    def __init__(
        self,
        config: RenderConfig,
        particle_count: int = 240,
        particle_charge: float = 0.4,
        particle_speed: float = 0.42,
        particle_radius_min: float = 8.0,
        particle_radius_max: float = 22.0,
        seed: int = 42,
    ) -> None:
        self.config = config.validated()
        self.device = self.config.resolve_device()
        self.dtype = self.config.resolve_dtype()
        self.particle_count = int(particle_count)
        self.particle_charge = float(abs(particle_charge))
        self.particle_speed = float(particle_speed)
        self.particle_radius_min = float(particle_radius_min)
        self.particle_radius_max = float(particle_radius_max)
        self.seed = int(seed)

        self.spin = float(self.config.spin)
        self.charge_bh = float(self.config.charge)
        self.lambda_cosmo = float(self.config.cosmological_constant)
        self.horizon = float(
            event_horizon_radius(
                self.config.spin,
                self.config.metric_model,
                self.config.charge,
                self.config.cosmological_constant,
            )
        )

        self._state: torch.Tensor | None = None
        self._specific_charge: torch.Tensor | None = None
        self._active: torch.Tensor | None = None
        self._rng = torch.Generator(device=self.device)
        self._rng.manual_seed(self.seed)

    def _electromagnetic_potential(self, r: torch.Tensor, theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        a = torch.as_tensor(self.spin, dtype=self.dtype, device=self.device)
        q = torch.as_tensor(self.charge_bh, dtype=self.dtype, device=self.device)
        lmb = torch.as_tensor(self.lambda_cosmo, dtype=self.dtype, device=self.device)

        cos_th = torch.cos(theta)
        sin_th = torch.sin(theta)
        sin2 = sin_th * sin_th
        sigma = torch.clamp(r * r + a * a * cos_th * cos_th, min=1.0e-9)
        xi = 1.0 + (lmb * a * a / 3.0)

        a_t = -q * r / sigma
        a_phi = q * a * r * sin2 / (sigma * xi)
        return a_t, a_phi

    def _electromagnetic_potential_derivatives(
        self, r: torch.Tensor, theta: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        a = torch.as_tensor(self.spin, dtype=self.dtype, device=self.device)
        q = torch.as_tensor(self.charge_bh, dtype=self.dtype, device=self.device)
        lmb = torch.as_tensor(self.lambda_cosmo, dtype=self.dtype, device=self.device)

        cos_th = torch.cos(theta)
        sin_th = torch.sin(theta)
        sin2 = sin_th * sin_th
        sigma = torch.clamp(r * r + a * a * cos_th * cos_th, min=1.0e-9)
        sigma2 = torch.clamp(sigma * sigma, min=1.0e-12)
        xi = 1.0 + (lmb * a * a / 3.0)

        d_sigma_r = 2.0 * r
        d_sigma_th = -2.0 * a * a * cos_th * sin_th

        d_r_a_t = -q * (sigma - r * d_sigma_r) / sigma2
        d_th_a_t = q * r * d_sigma_th / sigma2

        pref = q * a / torch.clamp(xi, min=1.0e-9)
        d_r_core = sin2 * (sigma - r * d_sigma_r) / sigma2
        d_th_core = r * (2.0 * sin_th * cos_th / sigma - sin2 * d_sigma_th / sigma2)

        d_r_a_phi = pref * d_r_core
        d_th_a_phi = pref * d_th_core
        return d_r_a_t, d_th_a_t, d_r_a_phi, d_th_a_phi

    def _initial_state(self) -> None:
        if self.particle_count < 1:
            raise ValueError("particle_count must be >= 1")

        n = self.particle_count
        pi = math.pi
        two_pi = 2.0 * pi

        r_min = max(self.particle_radius_min, 1.2 * self.horizon + 0.5, 4.5)
        r_max = max(self.particle_radius_max, r_min + 0.2)

        r0 = r_min + (r_max - r_min) * torch.rand((n,), generator=self._rng, device=self.device, dtype=self.dtype)
        theta0 = torch.full((n,), pi * 0.5, dtype=self.dtype, device=self.device)
        theta0 = theta0 + 0.035 * torch.randn((n,), generator=self._rng, device=self.device, dtype=self.dtype)
        theta0 = torch.clamp(theta0, min=2.0e-3, max=pi - 2.0e-3)
        phi0 = two_pi * torch.rand((n,), generator=self._rng, device=self.device, dtype=self.dtype)
        t0 = torch.zeros((n,), dtype=self.dtype, device=self.device)

        metric = self._metric(r0, theta0)
        a_t0, a_phi0 = self._electromagnetic_potential(r0, theta0)

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

        v_r = 0.010 * torch.randn((n,), generator=self._rng, device=self.device, dtype=self.dtype)
        v_th = 0.009 * torch.randn((n,), generator=self._rng, device=self.device, dtype=self.dtype)
        v_phi_base = self.particle_speed + 0.04 * torch.randn((n,), generator=self._rng, device=self.device, dtype=self.dtype)
        v_phi = torch.clamp(v_phi_base, min=0.22, max=0.68)

        v2 = torch.clamp(v_r * v_r + v_th * v_th + v_phi * v_phi, max=0.92)
        gamma = torch.rsqrt(torch.clamp(1.0 - v2, min=1.0e-6))

        u_t_up = gamma / lapse
        u_r_up = gamma * v_r * e_r_up
        u_th_up = gamma * v_th * e_th_up
        u_phi_up = gamma * (omega / lapse + v_phi * e_phi_up)

        pi_t = metric.g_tt * u_t_up + metric.g_tphi * u_phi_up
        pi_r = metric.g_rr * u_r_up
        pi_th = metric.g_thth * u_th_up
        pi_phi = metric.g_tphi * u_t_up + metric.g_phiphi * u_phi_up

        sign = torch.where(
            torch.rand((n,), generator=self._rng, device=self.device, dtype=self.dtype) > 0.5,
            torch.ones((n,), dtype=self.dtype, device=self.device),
            -torch.ones((n,), dtype=self.dtype, device=self.device),
        )
        qspec = sign * self.particle_charge * (0.6 + 0.8 * torch.rand((n,), generator=self._rng, device=self.device, dtype=self.dtype))

        p_t = pi_t + qspec * a_t0
        p_r = pi_r
        p_th = pi_th
        p_phi = pi_phi + qspec * a_phi0

        self._state = torch.stack([t0, r0, theta0, phi0, p_t, p_r, p_th, p_phi], dim=-1)
        self._specific_charge = qspec
        self._active = torch.ones((n,), dtype=torch.bool, device=self.device)

    def _metric(self, r: torch.Tensor, theta: torch.Tensor):
        from .geometry import metric_components

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

    def _rhs(self, state: torch.Tensor, qspec: torch.Tensor) -> torch.Tensor:
        r = torch.clamp(state[:, 1], min=1.0e-3)
        theta = torch.clamp(state[:, 2], min=1.0e-4, max=math.pi - 1.0e-4)

        p_t = state[:, 4]
        p_r = state[:, 5]
        p_th = state[:, 6]
        p_phi = state[:, 7]

        inv = self._inv_metric(r, theta)
        d_r, d_th = self._inv_metric_derivs(r, theta)

        a_t, a_phi = self._electromagnetic_potential(r, theta)
        d_r_a_t, d_th_a_t, d_r_a_phi, d_th_a_phi = self._electromagnetic_potential_derivatives(r, theta)

        pi_t = p_t - qspec * a_t
        pi_r = p_r
        pi_th = p_th
        pi_phi = p_phi - qspec * a_phi

        dt = inv.gtt * pi_t + inv.gtphi * pi_phi
        dr = inv.grr * pi_r
        dtheta = inv.gthth * pi_th
        dphi = inv.gtphi * pi_t + inv.gphiphi * pi_phi

        pi_t2 = pi_t * pi_t
        pi_r2 = pi_r * pi_r
        pi_th2 = pi_th * pi_th
        pi_phi2 = pi_phi * pi_phi

        contract_r = (
            d_r.gtt * pi_t2
            + 2.0 * d_r.gtphi * pi_t * pi_phi
            + d_r.grr * pi_r2
            + d_r.gthth * pi_th2
            + d_r.gphiphi * pi_phi2
        )
        contract_th = (
            d_th.gtt * pi_t2
            + 2.0 * d_th.gtphi * pi_t * pi_phi
            + d_th.grr * pi_r2
            + d_th.gthth * pi_th2
            + d_th.gphiphi * pi_phi2
        )

        vt = inv.gtt * pi_t + inv.gtphi * pi_phi
        vphi = inv.gtphi * pi_t + inv.gphiphi * pi_phi
        em_r = qspec * (vt * d_r_a_t + vphi * d_r_a_phi)
        em_th = qspec * (vt * d_th_a_t + vphi * d_th_a_phi)

        dp_t = torch.zeros_like(p_t)
        dp_r = -0.5 * contract_r + em_r
        dp_th = -0.5 * contract_th + em_th
        dp_phi = torch.zeros_like(p_phi)

        return torch.stack([dt, dr, dtheta, dphi, dp_t, dp_r, dp_th, dp_phi], dim=-1)

    def _rk4_step(self, state: torch.Tensor, qspec: torch.Tensor, h: float) -> torch.Tensor:
        k1 = self._rhs(state, qspec)
        k2 = self._rhs(state + 0.5 * h * k1, qspec)
        k3 = self._rhs(state + 0.5 * h * k2, qspec)
        k4 = self._rhs(state + h * k3, qspec)
        nxt = state + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        nxt[:, 1] = torch.clamp(nxt[:, 1], min=1.0e-3)
        nxt[:, 2] = torch.clamp(nxt[:, 2], min=1.0e-4, max=math.pi - 1.0e-4)
        nxt[:, 3] = torch.remainder(nxt[:, 3], 2.0 * math.pi)
        return nxt

    def _positions_cartesian(self, state: torch.Tensor) -> torch.Tensor:
        r = state[:, 1]
        theta = state[:, 2]
        phi = state[:, 3]
        a = torch.as_tensor(self.spin, dtype=self.dtype, device=self.device)

        rho = torch.sqrt(torch.clamp(r * r + a * a, min=1.0e-9)) * torch.sin(theta)
        x = rho * torch.cos(phi)
        y = rho * torch.sin(phi)
        z = r * torch.cos(theta)
        return torch.stack([x, y, z], dim=-1)

    def _camera_basis(self, azimuth: float, elevation: float, radius: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        ce = math.cos(elevation)
        se = math.sin(elevation)
        ca = math.cos(azimuth)
        sa = math.sin(azimuth)
        cam = np.array([radius * ce * ca, radius * ce * sa, radius * se], dtype=np.float64)
        target = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        forward = target - cam
        forward /= max(np.linalg.norm(forward), 1.0e-12)
        up_world = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if abs(float(np.dot(forward, up_world))) > 0.97:
            up_world = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        right = np.cross(forward, up_world)
        right /= max(np.linalg.norm(right), 1.0e-12)
        up = np.cross(right, forward)
        up /= max(np.linalg.norm(up), 1.0e-12)
        return cam, right, up, forward

    def _camera_basis_from_config(self, cfg: RenderConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        theta = math.radians(cfg.observer_inclination_deg)
        phi = math.radians(cfg.observer_azimuth_deg)
        r = float(cfg.observer_radius)

        sin_th = math.sin(theta)
        cos_th = math.cos(theta)
        sin_ph = math.sin(phi)
        cos_ph = math.cos(phi)

        cam = np.array([r * sin_th * cos_ph, r * sin_th * sin_ph, r * cos_th], dtype=np.float64)
        forward = -cam / max(np.linalg.norm(cam), 1.0e-12)

        up_world = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if abs(float(np.dot(forward, up_world))) > 0.97:
            up_world = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        right = np.cross(up_world, forward)
        right /= max(np.linalg.norm(right), 1.0e-12)
        up = np.cross(forward, right)
        up /= max(np.linalg.norm(up), 1.0e-12)

        roll = math.radians(cfg.observer_roll_deg)
        c = math.cos(roll)
        s = math.sin(roll)
        right_roll = c * right + s * up
        up_roll = -s * right + c * up
        return cam, right_roll, up_roll, forward

    def _initial_single_particle_state(
        self,
        radius: float,
        theta_deg: float,
        phi_deg: float,
        specific_charge: float,
        v_phi: float,
        v_theta: float,
        v_r: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r0 = torch.tensor([max(radius, 1.15 * self.horizon + 0.2)], dtype=self.dtype, device=self.device)
        theta0 = torch.tensor([math.radians(theta_deg)], dtype=self.dtype, device=self.device)
        theta0 = torch.clamp(theta0, min=2.0e-3, max=math.pi - 2.0e-3)
        phi0 = torch.tensor([math.radians(phi_deg)], dtype=self.dtype, device=self.device)
        t0 = torch.zeros((1,), dtype=self.dtype, device=self.device)

        metric = self._metric(r0, theta0)
        a_t0, a_phi0 = self._electromagnetic_potential(r0, theta0)

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

        pi_t = metric.g_tt * u_t_up + metric.g_tphi * u_phi_up
        pi_r = metric.g_rr * u_r_up
        pi_th = metric.g_thth * u_th_up
        pi_phi = metric.g_tphi * u_t_up + metric.g_phiphi * u_phi_up

        qspec = torch.tensor([specific_charge], dtype=self.dtype, device=self.device)
        p_t = pi_t + qspec * a_t0
        p_r = pi_r
        p_th = pi_th
        p_phi = pi_phi + qspec * a_phi0

        state = torch.stack([t0, r0, theta0, phi0, p_t, p_r, p_th, p_phi], dim=-1).reshape(1, 8)
        return state, qspec

    def _project_points(
        self,
        pts: np.ndarray,
        cam: np.ndarray,
        right: np.ndarray,
        up: np.ndarray,
        forward: np.ndarray,
        width: int,
        height: int,
        fov_deg: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        rel = pts - cam[None, :]
        x_cam = rel @ right
        y_cam = rel @ up
        z_cam = rel @ forward
        z_safe = np.maximum(z_cam, 1.0e-6)
        f = 0.5 * float(width) / math.tan(math.radians(0.5 * fov_deg))
        u = f * (x_cam / z_safe) + 0.5 * float(width)
        v = 0.5 * float(height) - f * (y_cam / z_safe)
        visible = z_cam > 1.0e-3
        return np.stack([u, v], axis=-1), visible

    def _background_image(self, width: int, height: int) -> Image.Image:
        arr = np.zeros((height, width, 3), dtype=np.uint8)
        for y in range(height):
            t = y / max(height - 1, 1)
            arr[y, :, 0] = int(4 + 8 * t)
            arr[y, :, 1] = int(8 + 14 * t)
            arr[y, :, 2] = int(16 + 34 * t)

        rng = np.random.default_rng(self.seed)
        star_n = max(400, (width * height) // 1100)
        ys = rng.integers(0, height, size=star_n)
        xs = rng.integers(0, width, size=star_n)
        mag = rng.random(star_n)
        for i in range(star_n):
            c = int(130 + 120 * (mag[i] ** 0.2))
            arr[ys[i], xs[i], :] = np.array([c, c, min(255, c + 20)], dtype=np.uint8)

        return Image.fromarray(arr, mode="RGB")

    def render_animation(
        self,
        output_path: str | Path,
        frames: int = 72,
        fps: int = 24,
        dt: float = 0.03,
        substeps: int = 6,
        camera_radius: float = 55.0,
        fov_deg: float = 40.0,
    ) -> ChargedParticleAnimationStats:
        if frames <= 0:
            raise ValueError("frames must be > 0")
        if fps <= 0:
            raise ValueError("fps must be > 0")
        if substeps <= 0:
            raise ValueError("substeps must be > 0")

        if self._state is None:
            self._initial_state()
        if self._state is None or self._specific_charge is None or self._active is None:
            raise RuntimeError("Particle state initialization failed")

        cfg = replace(self.config, width=max(256, self.config.width), height=max(144, self.config.height))
        w = cfg.width
        h = cfg.height

        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)

        t0 = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="kerrtrace_particles_", dir=str(target.parent)) as tmp:
            frame_dir = Path(tmp)
            base_bg = self._background_image(w, h)
            state = self._state.clone()
            qspec = self._specific_charge.clone()
            active = self._active.clone()
            history: list[np.ndarray] = []
            max_hist = 18

            for frame_idx in range(frames):
                for _ in range(substeps):
                    idx = torch.nonzero(active, as_tuple=False).squeeze(-1)
                    if not bool(idx.numel()):
                        break
                    cur = state[idx]
                    nxt = self._rk4_step(cur, qspec[idx], dt)
                    state[idx] = nxt

                    r = nxt[:, 1]
                    dead = (r <= 1.005 * self.horizon) | (r >= self.config.escape_radius)
                    if bool(dead.any()):
                        active[idx[dead]] = False

                pts = self._positions_cartesian(state).detach().cpu().numpy()
                active_np = active.detach().cpu().numpy()
                charge_np = qspec.detach().cpu().numpy()
                history.append(pts.copy())
                if len(history) > max_hist:
                    history.pop(0)

                phase = 0.0 if frames == 1 else frame_idx / (frames - 1)
                az = math.radians(35.0 + 120.0 * phase)
                el = math.radians(38.0 - 76.0 * phase)
                cam, right, up, forward = self._camera_basis(az, el, camera_radius)

                img = base_bg.copy().convert("RGBA")
                draw = ImageDraw.Draw(img, "RGBA")

                origin_2d, origin_vis = self._project_points(
                    np.array([[0.0, 0.0, 0.0]], dtype=np.float64),
                    cam,
                    right,
                    up,
                    forward,
                    w,
                    h,
                    fov_deg,
                )
                if bool(origin_vis[0]):
                    dist = max(float(np.dot(-cam, forward)), 1.0e-6)
                    bh_r = 1.8 * self.horizon
                    pix_r = max(2.0, 0.5 * w * bh_r / (dist * math.tan(math.radians(0.5 * fov_deg))))
                    cx, cy = float(origin_2d[0, 0]), float(origin_2d[0, 1])
                    draw.ellipse((cx - 1.6 * pix_r, cy - 1.6 * pix_r, cx + 1.6 * pix_r, cy + 1.6 * pix_r), fill=(20, 20, 24, 90))
                    draw.ellipse((cx - pix_r, cy - pix_r, cx + pix_r, cy + pix_r), fill=(0, 0, 0, 255))
                    draw.ellipse((cx - 1.15 * pix_r, cy - 1.15 * pix_r, cx + 1.15 * pix_r, cy + 1.15 * pix_r), outline=(210, 210, 235, 80), width=1)

                for hidx, pts_hist in enumerate(history):
                    age = hidx + 1
                    alpha = int(16 + 160 * age / len(history))
                    proj, vis = self._project_points(pts_hist, cam, right, up, forward, w, h, fov_deg)
                    for i in range(proj.shape[0]):
                        if (not vis[i]) or (not active_np[i]):
                            continue
                        x2 = float(proj[i, 0])
                        y2 = float(proj[i, 1])
                        if x2 < -3.0 or x2 > (w + 3.0) or y2 < -3.0 or y2 > (h + 3.0):
                            continue
                        if charge_np[i] >= 0.0:
                            col = (255, 92, 92, alpha)
                        else:
                            col = (92, 170, 255, alpha)
                        draw.ellipse((x2 - 1.2, y2 - 1.2, x2 + 1.2, y2 + 1.2), fill=col)

                frame_path = frame_dir / f"frame_{frame_idx:05d}.png"
                img.convert("RGB").save(frame_path)
                print(f"Particle frame {frame_idx + 1}/{frames}: {frame_path}")

            suffix = target.suffix.lower()
            if suffix in VIDEO_SUFFIXES:
                _encode_video_ffmpeg(frame_dir, target, fps, cfg)
            else:
                raise ValueError("Unsupported output extension for particle animation. Use .mp4, .mov, or .mkv")

        self._state = state
        self._active = active
        survivors = int(active.sum().item())
        dt_wall = time.perf_counter() - t0
        return ChargedParticleAnimationStats(
            frames=frames,
            fps=fps,
            elapsed_seconds=dt_wall,
            output_path=target,
            particles=self.particle_count,
            survivors=survivors,
        )

    def render_single_particle_over_raytrace(
        self,
        output_path: str | Path,
        frames: int = 36,
        fps: int = 12,
        dt: float = 0.03,
        substeps: int = 6,
        theta_deg: float = 62.0,
        phi_deg: float = 20.0,
        radius: float = 11.0,
        specific_charge: float = -0.45,
        v_phi: float = 0.46,
        v_theta: float = 0.09,
        v_r: float = 0.0,
        trail_length: int = 40,
    ) -> ChargedParticleAnimationStats:
        if frames <= 0:
            raise ValueError("frames must be > 0")
        if fps <= 0:
            raise ValueError("fps must be > 0")
        if substeps <= 0:
            raise ValueError("substeps must be > 0")
        if trail_length < 1:
            raise ValueError("trail_length must be > 0")

        cfg = self.config.validated()
        tracer = KerrRayTracer(cfg)
        cam, right, up, forward = self._camera_basis_from_config(cfg)

        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)

        state, qspec = self._initial_single_particle_state(
            radius=radius,
            theta_deg=theta_deg,
            phi_deg=phi_deg,
            specific_charge=specific_charge,
            v_phi=v_phi,
            v_theta=v_theta,
            v_r=v_r,
        )
        active = torch.ones((1,), dtype=torch.bool, device=self.device)

        t0 = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="kerrtrace_particle_rt_", dir=str(target.parent)) as tmp:
            frame_dir = Path(tmp)
            trail_xyz: list[np.ndarray] = []
            for frame_idx in range(frames):
                for _ in range(substeps):
                    if not bool(active.any()):
                        break
                    nxt = self._rk4_step(state, qspec, dt)
                    state = nxt
                    r = state[:, 1]
                    dead = (r <= 1.005 * self.horizon) | (r >= cfg.escape_radius)
                    if bool(dead.any()):
                        active[dead] = False

                emitter: PointEmitter | None = None
                if bool(active[0].item()):
                    r = state[:, 1]
                    theta = state[:, 2]
                    phi = state[:, 3]
                    p_t = state[:, 4]
                    p_r = state[:, 5]
                    p_th = state[:, 6]
                    p_phi = state[:, 7]

                    inv = self._inv_metric(r, theta)
                    a_t, a_phi = self._electromagnetic_potential(r, theta)
                    pi_t = p_t - qspec * a_t
                    pi_r = p_r
                    pi_th = p_th
                    pi_phi = p_phi - qspec * a_phi

                    u_t = (inv.gtt * pi_t + inv.gtphi * pi_phi)[0]
                    u_r = (inv.grr * pi_r)[0]
                    u_th = (inv.gthth * pi_th)[0]
                    u_phi = (inv.gtphi * pi_t + inv.gphiphi * pi_phi)[0]

                    if qspec[0].item() >= 0.0:
                        color = (1.00, 0.38, 0.34)
                    else:
                        color = (0.34, 0.68, 1.00)

                    emitter = PointEmitter(
                        r=float(r[0].item()),
                        theta=float(theta[0].item()),
                        phi=float(phi[0].item()),
                        u_t=float(u_t.item()),
                        u_r=float(u_r.item()),
                        u_theta=float(u_th.item()),
                        u_phi=float(u_phi.item()),
                        radius=0.08,
                        intensity=0.75,
                        color_rgb=color,
                    )
                    pos = self._positions_cartesian(state).detach().cpu().numpy()[0]
                    trail_xyz.append(pos.copy())
                    if len(trail_xyz) > trail_length:
                        trail_xyz.pop(0)
                elif trail_xyz:
                    # Keep the recent trail visible for a few frames after capture.
                    trail_xyz.pop(0)

                # The particle light is ray-traced with the same null geodesics as disk/background.
                img = tracer.render(emitter=emitter).image.convert("RGBA")
                if trail_xyz:
                    pts = np.asarray(trail_xyz, dtype=np.float64)
                    proj, vis = self._project_points(pts, cam, right, up, forward, cfg.width, cfg.height, cfg.fov_deg)
                    if qspec[0].item() >= 0.0:
                        trail_col = (255, 90, 90)
                    else:
                        trail_col = (90, 165, 255)
                    draw = ImageDraw.Draw(img, "RGBA")
                    ntrail = len(trail_xyz)
                    for i in range(1, ntrail):
                        if (not bool(vis[i - 1])) or (not bool(vis[i])):
                            continue
                        x0, y0 = float(proj[i - 1, 0]), float(proj[i - 1, 1])
                        x1, y1 = float(proj[i, 0]), float(proj[i, 1])
                        age = i / max(ntrail - 1, 1)
                        alpha = int(24 + 200 * age)
                        width_px = 1 if age < 0.7 else 2
                        draw.line((x0, y0, x1, y1), fill=(trail_col[0], trail_col[1], trail_col[2], alpha), width=width_px)
                    xh, yh = float(proj[-1, 0]), float(proj[-1, 1])
                    draw.ellipse((xh - 1.8, yh - 1.8, xh + 1.8, yh + 1.8), fill=(trail_col[0], trail_col[1], trail_col[2], 235))
                img = img.convert("RGB")
                frame_path = frame_dir / f"frame_{frame_idx:05d}.png"
                img.save(frame_path)
                print(f"Raytraced particle frame {frame_idx + 1}/{frames}: {frame_path}")

            suffix = target.suffix.lower()
            if suffix in VIDEO_SUFFIXES:
                _encode_video_ffmpeg(frame_dir, target, fps, cfg)
            else:
                raise ValueError("Unsupported output extension for raytraced particle animation. Use .mp4, .mov, or .mkv")

        elapsed = time.perf_counter() - t0
        survivors = int(active.sum().item())
        return ChargedParticleAnimationStats(
            frames=frames,
            fps=fps,
            elapsed_seconds=elapsed,
            output_path=target,
            particles=1,
            survivors=survivors,
        )
