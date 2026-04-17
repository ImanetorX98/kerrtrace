from __future__ import annotations

import math
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw
import torch

from .animation import VIDEO_SUFFIXES, _encode_video_ffmpeg
from .charged_particles import ChargedParticleAnimationStats
from .config import RenderConfig


@dataclass
class ParticleFrame:
    """Named standard format for physical particle state from a simulation step."""

    state_bl: torch.Tensor   # (N, 8): t, r, θ, φ, p_t, p_r, p_θ, p_φ
    charges: torch.Tensor    # (N,) specific charge q/m
    active: torch.Tensor     # (N,) bool — particles not yet captured
    frame_index: int
    spin: float              # black hole spin parameter a

    def positions_cartesian(self) -> np.ndarray:
        """Convert BL (r, θ, φ) → Cartesian XYZ for visualisation."""
        r = self.state_bl[:, 1].cpu().numpy()
        theta = self.state_bl[:, 2].cpu().numpy()
        phi = self.state_bl[:, 3].cpu().numpy()
        a = self.spin
        rho2 = r ** 2 + a ** 2
        x = np.sqrt(rho2) * np.sin(theta) * np.cos(phi)
        y = np.sqrt(rho2) * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        return np.stack([x, y, z], axis=-1)


class ParticleRenderer:
    """Perspective-projection PIL renderer for multi-particle animations."""

    def __init__(self, config: RenderConfig, seed: int = 42) -> None:
        self.config = config
        self.seed = seed

    # ── camera helpers ────────────────────────────────────────────────────────

    def _camera_basis(
        self, azimuth: float, elevation: float, radius: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        ce = math.cos(elevation)
        se = math.sin(elevation)
        ca = math.cos(azimuth)
        sa = math.sin(azimuth)
        cam = np.array([radius * ce * ca, radius * ce * sa, radius * se], dtype=np.float64)
        forward = -cam / max(np.linalg.norm(cam), 1.0e-12)
        up_world = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if abs(float(np.dot(forward, up_world))) > 0.97:
            up_world = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        right = np.cross(forward, up_world)
        right /= max(np.linalg.norm(right), 1.0e-12)
        up = np.cross(right, forward)
        up /= max(np.linalg.norm(up), 1.0e-12)
        return cam, right, up, forward

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

    # ── public drawing API ────────────────────────────────────────────────────

    def draw_frame(
        self,
        frame: ParticleFrame,
        history: list[ParticleFrame],
        az: float,
        el: float,
        camera_radius: float,
        fov_deg: float,
        background: Image.Image,
        horizon: float,
    ) -> Image.Image:
        w, h = background.size
        cam, right, up, forward = self._camera_basis(az, el, camera_radius)
        active_np = frame.active.detach().cpu().numpy()
        charge_np = frame.charges.detach().cpu().numpy()
        max_hist = len(history)

        img = background.copy().convert("RGBA")
        draw = ImageDraw.Draw(img, "RGBA")

        # Black hole disk
        origin_2d, origin_vis = self._project_points(
            np.array([[0.0, 0.0, 0.0]], dtype=np.float64),
            cam, right, up, forward, w, h, fov_deg,
        )
        if bool(origin_vis[0]):
            dist = max(float(np.dot(-cam, forward)), 1.0e-6)
            bh_r = 1.8 * horizon
            pix_r = max(2.0, 0.5 * w * bh_r / (dist * math.tan(math.radians(0.5 * fov_deg))))
            cx, cy = float(origin_2d[0, 0]), float(origin_2d[0, 1])
            draw.ellipse((cx - 1.6 * pix_r, cy - 1.6 * pix_r, cx + 1.6 * pix_r, cy + 1.6 * pix_r), fill=(20, 20, 24, 90))
            draw.ellipse((cx - pix_r, cy - pix_r, cx + pix_r, cy + pix_r), fill=(0, 0, 0, 255))
            draw.ellipse((cx - 1.15 * pix_r, cy - 1.15 * pix_r, cx + 1.15 * pix_r, cy + 1.15 * pix_r), outline=(210, 210, 235, 80), width=1)

        for hidx, hist_frame in enumerate(history):
            age = hidx + 1
            alpha = int(16 + 160 * age / max(max_hist, 1))
            pts_hist = hist_frame.positions_cartesian()
            hist_active = hist_frame.active.detach().cpu().numpy()
            proj, vis = self._project_points(pts_hist, cam, right, up, forward, w, h, fov_deg)
            for i in range(proj.shape[0]):
                if (not vis[i]) or (not hist_active[i]):
                    continue
                x2 = float(proj[i, 0])
                y2 = float(proj[i, 1])
                if x2 < -3.0 or x2 > (w + 3.0) or y2 < -3.0 or y2 > (h + 3.0):
                    continue
                col = (255, 92, 92, alpha) if charge_np[i] >= 0.0 else (92, 170, 255, alpha)
                draw.ellipse((x2 - 1.2, y2 - 1.2, x2 + 1.2, y2 + 1.2), fill=col)

        return img.convert("RGB")

    def render_animation(
        self,
        frames: Iterable[ParticleFrame],
        output_path: str | Path,
        fps: int,
        camera_radius: float,
        fov_deg: float,
        horizon: float,
    ) -> ChargedParticleAnimationStats:
        from dataclasses import replace as dc_replace

        cfg = dc_replace(self.config, width=max(256, self.config.width), height=max(144, self.config.height))
        w = cfg.width
        h = cfg.height
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)

        t_start = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="kerrtrace_particles_", dir=str(target.parent)) as tmp:
            frame_dir = Path(tmp)
            base_bg = self._background_image(w, h)
            history: list[ParticleFrame] = []
            max_hist = 18
            last_frame: ParticleFrame | None = None

            for frame in frames:
                last_frame = frame
                history.append(frame)
                if len(history) > max_hist:
                    history.pop(0)

                frame_count = frame.frame_index
                total_frames = None  # not known in advance from iterator
                phase = 0.0 if frame_count == 0 else frame_count / max(frame_count, 1)
                az = math.radians(35.0 + 120.0 * phase)
                el = math.radians(38.0 - 76.0 * phase)

                img = self.draw_frame(frame, history, az, el, camera_radius, fov_deg, base_bg, horizon)
                frame_path = frame_dir / f"frame_{frame_count:05d}.png"
                img.save(frame_path)
                print(f"Particle frame {frame_count + 1}: {frame_path}")

            suffix = target.suffix.lower()
            if suffix in VIDEO_SUFFIXES:
                _encode_video_ffmpeg(frame_dir, target, fps, cfg)
            else:
                raise ValueError("Unsupported output extension for particle animation. Use .mp4, .mov, or .mkv")

        n_frames = (last_frame.frame_index + 1) if last_frame is not None else 0
        n_particles = int(last_frame.active.shape[0]) if last_frame is not None else 0
        survivors = int(last_frame.active.sum().item()) if last_frame is not None else 0
        dt_wall = time.perf_counter() - t_start
        return ChargedParticleAnimationStats(
            frames=n_frames,
            fps=fps,
            elapsed_seconds=dt_wall,
            output_path=target,
            particles=n_particles,
            survivors=survivors,
        )


class RaytracedParticleRenderer:
    """Renders a single particle orbit via null geodesics + PIL trail overlay."""

    def __init__(self, tracer: object) -> None:  # tracer: KerrRayTracer
        self.tracer = tracer

    def _camera_basis_from_config(
        self, cfg: RenderConfig
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

    def draw_frame(
        self,
        frame: ParticleFrame,
        trail: list[np.ndarray],
        emitter_payload: object,
    ) -> Image.Image:
        from .raytracer import PointEmitter

        cfg = self.tracer.config
        img = self.tracer.render(emitter=emitter_payload).image.convert("RGBA")

        if trail:
            cam, right, up, forward = self._camera_basis_from_config(cfg)
            pts = np.asarray(trail, dtype=np.float64)
            proj, vis = self._project_points(pts, cam, right, up, forward, cfg.width, cfg.height, cfg.fov_deg)
            charge = float(frame.charges[0].item())
            trail_col = (255, 90, 90) if charge >= 0.0 else (90, 165, 255)
            draw = ImageDraw.Draw(img, "RGBA")
            ntrail = len(trail)
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

        return img.convert("RGB")

    def render_animation(
        self,
        frames: Iterable[ParticleFrame],
        output_path: str | Path,
        fps: int,
        trail_length: int,
        inv_metric_fn: object,
        em_potential_fn: object,
        horizon: float,
        escape_radius: float,
    ) -> ChargedParticleAnimationStats:
        from .raytracer import PointEmitter

        cfg = self.tracer.config
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)

        t_start = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="kerrtrace_particle_rt_", dir=str(target.parent)) as tmp:
            frame_dir = Path(tmp)
            trail_xyz: list[np.ndarray] = []
            last_frame: ParticleFrame | None = None
            last_active: torch.Tensor | None = None

            for frame in frames:
                last_frame = frame
                last_active = frame.active
                frame_idx = frame.frame_index

                emitter_payload: PointEmitter | None = None
                if bool(frame.active[0].item()):
                    state = frame.state_bl
                    qspec = frame.charges
                    r = state[:, 1]
                    theta = state[:, 2]
                    phi = state[:, 3]
                    p_t = state[:, 4]
                    p_r = state[:, 5]
                    p_th = state[:, 6]
                    p_phi = state[:, 7]

                    inv = inv_metric_fn(r, theta)
                    a_t, a_phi = em_potential_fn(r, theta)
                    pi_t = p_t - qspec * a_t
                    pi_r = p_r
                    pi_th = p_th
                    pi_phi = p_phi - qspec * a_phi

                    u_t = (inv.gtt * pi_t + inv.gtphi * pi_phi)[0]
                    u_r = (inv.grr * pi_r)[0]
                    u_th = (inv.gthth * pi_th)[0]
                    u_phi = (inv.gtphi * pi_t + inv.gphiphi * pi_phi)[0]

                    charge = float(qspec[0].item())
                    color = (1.00, 0.38, 0.34) if charge >= 0.0 else (0.34, 0.68, 1.00)
                    emitter_payload = PointEmitter(
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
                    pos = frame.positions_cartesian()[0]
                    trail_xyz.append(pos.copy())
                    if len(trail_xyz) > trail_length:
                        trail_xyz.pop(0)
                elif trail_xyz:
                    trail_xyz.pop(0)

                img = self.draw_frame(frame, trail_xyz, emitter_payload)
                frame_path = frame_dir / f"frame_{frame_idx:05d}.png"
                img.save(frame_path)
                print(f"Raytraced particle frame {frame_idx + 1}: {frame_path}")

            suffix = target.suffix.lower()
            if suffix in VIDEO_SUFFIXES:
                _encode_video_ffmpeg(frame_dir, target, fps, cfg)
            else:
                raise ValueError("Unsupported output extension for raytraced particle animation. Use .mp4, .mov, or .mkv")

        n_frames = (last_frame.frame_index + 1) if last_frame is not None else 0
        survivors = int(last_active.sum().item()) if last_active is not None else 0
        dt_wall = time.perf_counter() - t_start
        return ChargedParticleAnimationStats(
            frames=n_frames,
            fps=fps,
            elapsed_seconds=dt_wall,
            output_path=target,
            particles=1,
            survivors=survivors,
        )
