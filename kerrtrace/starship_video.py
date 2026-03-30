#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import shutil
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import torch

from .config import RenderConfig
from .raytracer import KerrRayTracer
from .starship import Starship, StarshipThrustCommand, StarshipThrustSegment

VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv"}


def _aces_fitted_tonemap(x: np.ndarray) -> np.ndarray:
    a = 2.51
    b = 0.03
    c = 2.43
    d = 0.59
    e = 0.14
    return np.clip((x * (a * x + b)) / np.maximum(x * (c * x + d) + e, 1.0e-6), 0.0, 1.0)


def _load_obj_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4:
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                continue
            if line.startswith("f "):
                parts = line.split()[1:]
                idx: list[int] = []
                for p in parts:
                    tok = p.split("/")[0]
                    if not tok:
                        continue
                    vi = int(tok)
                    if vi < 0:
                        vi = len(vertices) + vi
                    else:
                        vi = vi - 1
                    idx.append(vi)
                if len(idx) < 3:
                    continue
                for i in range(1, len(idx) - 1):
                    faces.append([idx[0], idx[i], idx[i + 1]])

    if not vertices or not faces:
        raise ValueError(f"Invalid or empty OBJ: {path}")
    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int32)


def _normalize_vertices(vertices: np.ndarray) -> np.ndarray:
    vmin = vertices.min(axis=0)
    vmax = vertices.max(axis=0)
    center = 0.5 * (vmin + vmax)
    v = vertices - center
    max_extent = float(np.max(np.linalg.norm(v, axis=1)))
    if max_extent <= 1.0e-9:
        return v
    return v / max_extent


def _rotation_matrix(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    yaw = math.radians(float(yaw_deg))
    pitch = math.radians(float(pitch_deg))
    roll = math.radians(float(roll_deg))

    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)

    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float32)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]], dtype=np.float32)
    rz = np.array([[cr, -sr, 0.0], [sr, cr, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return (rz @ rx @ ry).astype(np.float32)


def _quat_rotate(tracer: KerrRayTracer, q: torch.Tensor, vec: np.ndarray) -> np.ndarray:
    t = torch.as_tensor(vec, dtype=tracer.dtype, device=tracer.device).view(1, 1, 3)
    out = tracer._quat_rotate_batch(q, t).reshape(3).detach().cpu().numpy()
    return out.astype(np.float32)


def _camera_basis(tracer: KerrRayTracer) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    theta0, phi0, q_cam, _, _, _ = tracer._camera_axes()
    obs_xyz = (
        tracer._bl_to_cartesian_kerr_schild(
            torch.as_tensor([tracer.config.observer_radius], dtype=tracer.dtype, device=tracer.device),
            torch.as_tensor([theta0], dtype=tracer.dtype, device=tracer.device),
            torch.as_tensor([phi0], dtype=tracer.dtype, device=tracer.device),
        )
        .reshape(3)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    right = _quat_rotate(tracer, q_cam, np.array([1.0, 0.0, 0.0], dtype=np.float32))
    up = _quat_rotate(tracer, q_cam, np.array([0.0, 1.0, 0.0], dtype=np.float32))
    forward = _quat_rotate(tracer, q_cam, np.array([0.0, 0.0, 1.0], dtype=np.float32))
    return obs_xyz, right, up, forward


def _project_vertices(
    world_vertices: np.ndarray,
    eye: np.ndarray,
    right: np.ndarray,
    up: np.ndarray,
    forward: np.ndarray,
    width: int,
    height: int,
    fov_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    rel = world_vertices - eye.reshape(1, 3)
    x_cam = rel @ right
    y_cam = rel @ up
    z_cam = rel @ forward

    aspect = float(width) / float(height)
    tan_half = math.tan(math.radians(0.5 * fov_deg))
    u = x_cam / np.maximum(z_cam * aspect * tan_half, 1.0e-9)
    v = y_cam / np.maximum(z_cam * tan_half, 1.0e-9)

    px = (u + 1.0) * 0.5 * float(width)
    py = (1.0 - v) * 0.5 * float(height)
    screen = np.stack([px, py], axis=-1).astype(np.float32)
    return screen, z_cam.astype(np.float32)


def _render_mesh_overlay(
    base: Image.Image,
    world_vertices: np.ndarray,
    faces: np.ndarray,
    eye: np.ndarray,
    right: np.ndarray,
    up: np.ndarray,
    forward: np.ndarray,
    fov_deg: float,
    cinematic_strength: float,
    ship_opacity: float,
) -> Image.Image:
    width, height = base.size
    screen, z_cam = _project_vertices(world_vertices, eye, right, up, forward, width, height, fov_deg)

    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")

    z_face = np.mean(z_cam[faces], axis=1)
    order = np.argsort(z_face)[::-1]

    s = max(0.0, min(2.0, float(cinematic_strength)))
    opacity = max(0.0, min(1.0, float(ship_opacity)))
    key = np.array([-0.58, 0.22, 0.78], dtype=np.float32)
    key = key / (np.linalg.norm(key) + 1.0e-9)
    fill = np.array([0.46, -0.30, 0.70], dtype=np.float32)
    fill = fill / (np.linalg.norm(fill) + 1.0e-9)
    base_col = np.array([170.0, 184.0, 214.0], dtype=np.float32)
    highlight_col = np.array([255.0, 229.0, 191.0], dtype=np.float32)

    for fi in order:
        tri = faces[fi]
        ztri = z_cam[tri]
        if float(np.min(ztri)) <= 1.0e-4:
            continue

        pts = screen[tri]
        if np.all((pts[:, 0] < -8) | (pts[:, 0] > width + 8) | (pts[:, 1] < -8) | (pts[:, 1] > height + 8)):
            continue

        w0, w1, w2 = world_vertices[tri[0]], world_vertices[tri[1]], world_vertices[tri[2]]
        normal = np.cross(w1 - w0, w2 - w0)
        nrm = np.linalg.norm(normal)
        if nrm <= 1.0e-9:
            continue
        normal = normal / nrm
        center = (w0 + w1 + w2) / 3.0
        view_dir = eye - center
        view_nrm = np.linalg.norm(view_dir)
        if view_nrm > 1.0e-9:
            view_dir = view_dir / view_nrm
        else:
            view_dir = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        if float(np.dot(normal, view_dir)) <= 0.0:
            continue

        lambert_key = max(0.0, float(np.dot(normal, key)))
        lambert_fill = max(0.0, float(np.dot(normal, fill)))
        ndotv = max(0.0, float(np.dot(normal, view_dir)))
        rim = pow(max(0.0, 1.0 - ndotv), 2.6)
        half_vec = key + view_dir
        half_norm = float(np.linalg.norm(half_vec))
        if half_norm > 1.0e-9:
            half_vec = half_vec / half_norm
        spec = pow(max(0.0, float(np.dot(normal, half_vec))), 30.0)

        panel = 0.5 + 0.5 * math.sin(11.0 * float(center[0]) + 6.0 * float(center[1]) + 2.0 * float(center[2]))
        intensity = (
            0.12
            + (0.64 + 0.10 * s) * lambert_key
            + (0.24 + 0.06 * s) * lambert_fill
            + (0.42 + 0.10 * s) * rim
            + (0.30 + 0.10 * s) * spec
        )
        intensity = max(0.0, min(1.65, intensity))
        metal = base_col * (0.84 + 0.16 * panel) * intensity
        spec_tint = highlight_col * (0.12 + 0.28 * spec + 0.12 * rim)
        col_f = np.clip(metal + spec_tint, 0.0, 255.0)
        col = col_f.astype(np.uint8)
        alpha_dyn = int(max(170, min(255, 206 + 44 * ndotv + 15 * rim)))
        alpha = int(max(0, min(255, (1.0 - opacity) * alpha_dyn + opacity * 255.0)))
        polygon = [(float(pts[0, 0]), float(pts[0, 1])), (float(pts[1, 0]), float(pts[1, 1])), (float(pts[2, 0]), float(pts[2, 1]))]
        draw.polygon(polygon, fill=(int(col[0]), int(col[1]), int(col[2]), alpha))
        if s > 0.0:
            edge_alpha = int(min(255, max(70, (45 + 35 * s) * (0.7 + 0.3 * opacity))))
            edge_col = (int(min(255, col[0] + 16)), int(min(255, col[1] + 18)), int(min(255, col[2] + 22)), edge_alpha)
            draw.line([polygon[0], polygon[1]], fill=edge_col, width=1)
            draw.line([polygon[1], polygon[2]], fill=edge_col, width=1)
            draw.line([polygon[2], polygon[0]], fill=edge_col, width=1)

    ov_np = np.asarray(overlay, dtype=np.uint8)
    alpha_mask = ov_np[:, :, 3] > 8
    if bool(np.any(alpha_mask)):
        ys, xs = np.where(alpha_mask)
        xmin, xmax = int(xs.min()), int(xs.max())
        ymin, ymax = int(ys.min()), int(ys.max())
        cx = int(0.5 * (xmin + xmax))
        cy = int(0.5 * (ymin + ymax))
        span_x = max(8, xmax - xmin)
        span_y = max(6, ymax - ymin)
        glow = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        gdraw = ImageDraw.Draw(glow, "RGBA")
        r1 = max(3, int(0.09 * span_x))
        r2 = max(4, int(0.13 * span_x))
        dy = int(0.04 * span_y)
        p_left = (cx - int(0.11 * span_x), cy + dy)
        p_right = (cx + int(0.11 * span_x), cy + dy)
        for px, py in (p_left, p_right):
            gdraw.ellipse((px - r2, py - r2, px + r2, py + r2), fill=(95, 168, 255, 72))
            gdraw.ellipse((px - r1, py - r1, px + r1, py + r1), fill=(165, 222, 255, 135))
        glow = glow.filter(ImageFilter.GaussianBlur(radius=max(2.0, 2.8 * s)))
        overlay = Image.alpha_composite(overlay, glow)

    out = base.convert("RGBA")
    out.alpha_composite(overlay)
    return out.convert("RGB")


def _postprocess_cinematic(image: Image.Image, strength: float) -> Image.Image:
    s = max(0.0, min(2.0, float(strength)))
    if s <= 1.0e-6:
        return image

    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    lum = 0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2]

    hi = np.clip((lum - 0.54) / 0.46, 0.0, 1.0)
    bloom_src = np.clip(arr * (0.25 + 0.75 * hi[:, :, None]), 0.0, 1.0)
    bloom_img = Image.fromarray(np.clip(bloom_src * 255.0, 0.0, 255.0).astype(np.uint8), mode="RGB")
    bloom_img = bloom_img.filter(ImageFilter.GaussianBlur(radius=3.2 + 2.2 * s))
    bloom = np.asarray(bloom_img, dtype=np.float32) / 255.0
    h, w, _ = arr.shape
    flare_h = max(2, h // 22)
    flare_img = Image.fromarray(np.clip(bloom_src * 255.0, 0.0, 255.0).astype(np.uint8), mode="RGB")
    flare_img = flare_img.resize((w, flare_h), resample=Image.BILINEAR).resize((w, h), resample=Image.BILINEAR)
    flare_img = flare_img.filter(ImageFilter.GaussianBlur(radius=4.6 + 1.8 * s))
    flare = np.asarray(flare_img, dtype=np.float32) / 255.0
    arr = np.clip(arr + (0.12 + 0.10 * s) * bloom + (0.08 + 0.10 * s) * flare, 0.0, 1.0)

    arr = np.clip((arr - 0.5) * (1.0 + 0.24 * s) + 0.5, 0.0, 1.0)
    lum2 = 0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2]
    shadows = np.clip(1.0 - lum2 * 1.5, 0.0, 1.0)[:, :, None]
    highlights = np.clip((lum2 - 0.42) / 0.58, 0.0, 1.0)[:, :, None]
    cool = np.array([0.88, 0.96, 1.08], dtype=np.float32).reshape(1, 1, 3)
    warm = np.array([1.12, 1.04, 0.90], dtype=np.float32).reshape(1, 1, 3)
    arr = arr * (1.0 + 0.18 * s * (cool - 1.0) * shadows + 0.24 * s * (warm - 1.0) * highlights)
    sat = 1.0 + 0.10 * s
    gray = (0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2])[:, :, None]
    arr = np.clip(gray + (arr - gray) * sat, 0.0, 1.0)
    arr = _aces_fitted_tonemap(arr * (1.0 + 0.05 * s))

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx = (xx - 0.5 * w) / max(1.0, 0.5 * w)
    ny = (yy - 0.5 * h) / max(1.0, 0.5 * h)
    rr = np.sqrt(nx * nx + ny * ny)
    vign = np.clip(1.0 - 0.34 * s * (rr ** 1.55), 0.62, 1.0)
    arr *= vign[:, :, None]

    seed = int(1000.0 * s + 137)
    rng = np.random.default_rng(seed)
    grain = rng.normal(loc=0.0, scale=1.0, size=(h, w, 1)).astype(np.float32)
    arr = np.clip(arr + (0.010 + 0.014 * s) * grain, 0.0, 1.0)

    arr = np.clip(arr, 0.0, 1.0)
    return Image.fromarray(np.clip(arr * 255.0, 0.0, 255.0).astype(np.uint8), mode="RGB")


def _parse_direction_vector(raw: object) -> tuple[float, float, float]:
    if isinstance(raw, (list, tuple)) and len(raw) == 3:
        return (float(raw[0]), float(raw[1]), float(raw[2]))
    if isinstance(raw, str):
        parts = [p.strip() for p in raw.split(",")]
        if len(parts) == 3:
            return (float(parts[0]), float(parts[1]), float(parts[2]))
    return (0.0, 0.0, 1.0)


@dataclass
class ShipVisual:
    name: str
    starship: Starship
    vertices: np.ndarray
    faces: np.ndarray
    size: float
    yaw_deg: float
    pitch_deg: float
    roll_deg: float
    opacity: float
    cinematic_strength: float


def _load_ship_records(args: argparse.Namespace) -> tuple[list[dict[str, object]], Path]:
    if args.ship_config_json is None:
        rec = {
            "name": "ship0",
            "obj": str(args.obj),
            "radius": float(args.ship_radius),
            "theta_deg": float(args.ship_theta_deg),
            "phi_deg": float(args.ship_phi_deg),
            "size": float(args.ship_size),
            "yaw_deg": float(args.ship_yaw_deg),
            "pitch_deg": float(args.ship_pitch_deg),
            "roll_deg": float(args.ship_roll_deg),
            "opacity": float(args.ship_opacity),
            "cinematic_strength": float(args.cinematic_strength),
            "v_phi": float(args.ship_v_phi),
            "v_theta": float(args.ship_v_theta),
            "v_r": float(args.ship_v_r),
            "acceleration": float(args.ship_acceleration),
            "direction_mode": str(args.ship_direction_mode),
            "direction_vector": _parse_direction_vector(args.ship_direction_vector),
            "thrust_program": [],
        }
        return [rec], Path.cwd()

    cfg_path = Path(args.ship_config_json).expanduser().resolve()
    payload = json.loads(cfg_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        ships = payload.get("ships", [])
    elif isinstance(payload, list):
        ships = payload
    else:
        raise ValueError("ship_config_json must be an object with 'ships' or a list")

    if not isinstance(ships, list) or not ships:
        raise ValueError("ship_config_json contains no ships")
    out: list[dict[str, object]] = []
    for idx, item in enumerate(ships):
        if not isinstance(item, dict):
            raise ValueError(f"ship[{idx}] must be an object")
        out.append(item)
    return out, cfg_path.parent


def _resolve_obj_path(raw: object, base_dir: Path, fallback: Path) -> Path:
    raw_str = str(raw).strip() if raw is not None else ""
    candidate = Path(raw_str) if raw_str else fallback
    if not candidate.is_absolute():
        p1 = (base_dir / candidate).resolve()
        if p1.exists():
            return p1
        p2 = (Path.cwd() / candidate).resolve()
        if p2.exists():
            return p2
        return p1
    return candidate.resolve()


def _ship_from_record(
    rec: dict[str, object],
    cfg: RenderConfig,
    tracer: KerrRayTracer,
    base_dir: Path,
    fallback_obj: Path,
) -> ShipVisual:
    name = str(rec.get("name", "ship"))
    obj_path = _resolve_obj_path(rec.get("obj", rec.get("obj_path")), base_dir=base_dir, fallback=fallback_obj)
    vertices, faces = _load_obj_mesh(obj_path)
    vertices = _normalize_vertices(vertices)

    ship = Starship(
        cfg,
        radius=float(rec.get("radius", 20.0)),
        theta_deg=float(rec.get("theta_deg", 87.0)),
        phi_deg=float(rec.get("phi_deg", 0.0)),
        v_phi=float(rec.get("v_phi", 0.46)),
        v_theta=float(rec.get("v_theta", 0.04)),
        v_r=float(rec.get("v_r", 0.0)),
    )
    ship.set_acceleration(
        acceleration=float(rec.get("acceleration", 0.0)),
        direction_mode=str(rec.get("direction_mode", "azimuthal_prograde")),
        direction_vector=_parse_direction_vector(rec.get("direction_vector", [0.0, 0.0, 1.0])),
    )

    segments_raw = rec.get("thrust_program", [])
    segments: list[StarshipThrustSegment] = []
    if isinstance(segments_raw, list):
        for seg in segments_raw:
            if not isinstance(seg, dict):
                continue
            cmd = StarshipThrustCommand(
                acceleration=max(0.0, float(seg.get("acceleration", 0.0))),
                direction_mode=str(seg.get("direction_mode", "azimuthal_prograde")),
                direction_vector=_parse_direction_vector(seg.get("direction_vector", [0.0, 0.0, 1.0])),
            )
            segments.append(
                StarshipThrustSegment(
                    start_time=float(seg.get("start_time", 0.0)),
                    end_time=float(seg.get("end_time", 0.0)),
                    command=cmd,
                )
            )
    if segments:
        ship.set_acceleration_program(segments)

    return ShipVisual(
        name=name,
        starship=ship,
        vertices=vertices,
        faces=faces,
        size=float(rec.get("size", 1.9)),
        yaw_deg=float(rec.get("yaw_deg", 34.0)),
        pitch_deg=float(rec.get("pitch_deg", -12.0)),
        roll_deg=float(rec.get("roll_deg", 16.0)),
        opacity=float(rec.get("opacity", 0.95)),
        cinematic_strength=float(rec.get("cinematic_strength", 1.45)),
    )


def _ship_world_vertices(
    tracer: KerrRayTracer,
    sv: ShipVisual,
    right: np.ndarray,
    up: np.ndarray,
    forward: np.ndarray,
) -> np.ndarray:
    st = sv.starship.state_dict()
    ship_pos = (
        tracer._bl_to_cartesian_kerr_schild(
            torch.as_tensor([float(st["r"])], dtype=tracer.dtype, device=tracer.device),
            torch.as_tensor([float(st["theta"])], dtype=tracer.dtype, device=tracer.device),
            torch.as_tensor([float(st["phi"])], dtype=tracer.dtype, device=tracer.device),
        )
        .reshape(3)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )
    basis = np.stack([right, up, -forward], axis=1)
    rot_local = _rotation_matrix(sv.yaw_deg, sv.pitch_deg, sv.roll_deg)
    vertices_local = (sv.vertices @ rot_local.T).astype(np.float32)
    return ship_pos.reshape(1, 3) + (vertices_local * sv.size) @ basis.T


def _encode_video_ffmpeg(frames_dir: Path, fps: int, output_path: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found in PATH")
    cmd = [
        ffmpeg,
        "-y",
        "-framerate",
        str(int(fps)),
        "-i",
        str(frames_dir / "frame_%05d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "18",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)


def _encode_gif(frames_dir: Path, fps: int, output_path: Path) -> None:
    frame_files = sorted(frames_dir.glob("frame_*.png"))
    if not frame_files:
        raise RuntimeError("No frames found for GIF encoding")
    images = [Image.open(p).convert("RGB") for p in frame_files]
    duration_ms = int(round(1000.0 / max(1, int(fps))))
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
    )
    for img in images:
        img.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Render KNdS frame/video with polygonal OBJ starship overlays.")
    parser.add_argument("--obj", type=Path, default=Path("assets/models/quaternius_ultimate/omen/Omen.obj"))
    parser.add_argument("--ship-config-json", type=Path, default=None, help="JSON file with one or more ship specs")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=288)
    parser.add_argument("--frames", type=int, default=1)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--ship-substeps", type=int, default=3)
    parser.add_argument("--keep-frames", action="store_true")
    parser.add_argument("--observer-radius", type=float, default=30.0)
    parser.add_argument("--observer-theta-deg", type=float, default=87.0)
    parser.add_argument("--observer-phi-deg", type=float, default=0.0)
    parser.add_argument("--ship-radius", type=float, default=20.0)
    parser.add_argument("--ship-theta-deg", type=float, default=87.0)
    parser.add_argument("--ship-phi-deg", type=float, default=0.0)
    parser.add_argument("--ship-size", type=float, default=1.9)
    parser.add_argument("--ship-yaw-deg", type=float, default=34.0)
    parser.add_argument("--ship-pitch-deg", type=float, default=-12.0)
    parser.add_argument("--ship-roll-deg", type=float, default=16.0)
    parser.add_argument("--ship-opacity", type=float, default=0.95)
    parser.add_argument("--ship-v-phi", type=float, default=0.46)
    parser.add_argument("--ship-v-theta", type=float, default=0.04)
    parser.add_argument("--ship-v-r", type=float, default=0.0)
    parser.add_argument("--ship-acceleration", type=float, default=0.0)
    parser.add_argument("--ship-direction-mode", type=str, default="azimuthal_prograde")
    parser.add_argument("--ship-direction-vector", type=str, default="0,0,1")
    parser.add_argument("--cinematic-strength", type=float, default=1.45)
    parser.add_argument("--disk-inner-radius", type=float, default=None)
    parser.add_argument("--disk-outer-radius", type=float, default=10.0)
    parser.add_argument("--disk-emission-gain", type=float, default=1.0)
    parser.add_argument("--disk-beaming-strength", type=float, default=0.45)
    parser.add_argument("--disk-self-occlusion-strength", type=float, default=0.35)
    parser.add_argument("--step-size", type=float, default=0.3)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"])
    args = parser.parse_args()

    cfg = RenderConfig(
        width=int(args.width),
        height=int(args.height),
        coordinate_system="generalized_doran",
        metric_model="kerr_newman_de_sitter",
        spin=0.94,
        charge=0.2,
        cosmological_constant=1.0e-7,
        observer_radius=float(args.observer_radius),
        observer_inclination_deg=float(args.observer_theta_deg),
        observer_azimuth_deg=float(args.observer_phi_deg),
        observer_roll_deg=0.0,
        disk_inner_radius=args.disk_inner_radius,
        disk_outer_radius=float(args.disk_outer_radius),
        disk_emission_gain=float(args.disk_emission_gain),
        disk_beaming_strength=float(args.disk_beaming_strength),
        disk_self_occlusion_strength=float(args.disk_self_occlusion_strength),
        step_size=float(args.step_size),
        max_steps=int(args.max_steps),
        shadow_absorb_radius_factor=1.0,
        background_mode="hdri",
        background_projection="equirectangular",
        hdri_path="assets/backgrounds/downloads_imported/sfondo2.jpg",
        hdri_exposure=1.0,
        device=str(args.device),
    )
    tracer = KerrRayTracer(cfg)
    base = tracer.render().image
    eye, right, up, forward = _camera_basis(tracer)

    records, ship_cfg_base = _load_ship_records(args)
    ships: list[ShipVisual] = []
    for rec in records:
        ships.append(
            _ship_from_record(
                rec=rec,
                cfg=cfg,
                tracer=tracer,
                base_dir=ship_cfg_base,
                fallback_obj=Path(args.obj),
            )
        )
    if not ships:
        raise ValueError("No ships configured")

    frames = max(1, int(args.frames))
    fps = max(1, int(args.fps))
    ship_substeps = max(1, int(args.ship_substeps))
    dt = 1.0 / float(fps)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    frame_dir = output.parent / f"{output.stem}_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    max_cine = max(float(sv.cinematic_strength) for sv in ships)

    for fi in range(frames):
        if fi > 0:
            for sv in ships:
                if sv.starship.alive:
                    sv.starship.step(dt=dt, substeps=ship_substeps)

        out = base.copy()
        for sv in ships:
            if not sv.starship.alive:
                continue
            world_vertices = _ship_world_vertices(tracer, sv, right=right, up=up, forward=forward)
            out = _render_mesh_overlay(
                base=out,
                world_vertices=world_vertices,
                faces=sv.faces,
                eye=eye,
                right=right,
                up=up,
                forward=forward,
                fov_deg=float(cfg.fov_deg),
                cinematic_strength=float(sv.cinematic_strength),
                ship_opacity=float(sv.opacity),
            )
        out = _postprocess_cinematic(out, strength=max_cine)
        frame_path = frame_dir / f"frame_{fi:05d}.png"
        out.save(frame_path)
        print(f"Frame {fi+1}/{frames}: {frame_path}")

    if frames == 1 and output.suffix.lower() not in {".gif", *VIDEO_SUFFIXES}:
        src = frame_dir / "frame_00000.png"
        output.write_bytes(src.read_bytes())
        print(f"SAVED={output}")
        if not args.keep_frames:
            shutil.rmtree(frame_dir, ignore_errors=True)
    elif output.suffix.lower() == ".gif":
        _encode_gif(frames_dir=frame_dir, fps=fps, output_path=output)
        print(f"SAVED={output}")
        if not args.keep_frames:
            shutil.rmtree(frame_dir, ignore_errors=True)
    elif output.suffix.lower() in VIDEO_SUFFIXES:
        _encode_video_ffmpeg(frame_dir=frame_dir, fps=fps, output_path=output)
        print(f"SAVED={output}")
        if not args.keep_frames:
            shutil.rmtree(frame_dir, ignore_errors=True)
    else:
        # For non-video outputs with multiple frames, keep frame sequence.
        preview = frame_dir / "frame_00000.png"
        output.write_bytes(preview.read_bytes())
        print(f"SAVED_PREVIEW={output}")
        print(f"SAVED_FRAMES_DIR={frame_dir}")

    print(f"DEVICE={tracer.device}")
    print(f"SHIPS={len(ships)}")
    print(f"TRIANGLES_TOTAL={sum(int(sv.faces.shape[0]) for sv in ships)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
