#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import torch

from kerrtrace.config import RenderConfig
from kerrtrace.raytracer import KerrRayTracer


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
        raise ValueError(f"OBJ non valido o vuoto: {path}")
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

    # Yaw (Y), pitch (X), roll (Z): R = Rz * Rx * Ry
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
) -> tuple[Image.Image, np.ndarray]:
    width, height = base.size
    screen, z_cam = _project_vertices(world_vertices, eye, right, up, forward, width, height, fov_deg)

    # Software z-buffer for solid opaque rendering.
    ship_rgb = np.zeros((height, width, 3), dtype=np.uint8)
    ship_alpha = np.zeros((height, width), dtype=np.uint8)
    zbuf = np.full((height, width), np.inf, dtype=np.float32)
    z_face = np.mean(z_cam[faces], axis=1)
    order = np.argsort(z_face)  # z-buffer handles visibility
    visible_faces: list[tuple[int, int, int]] = []

    s = max(0.0, min(2.0, float(cinematic_strength)))
    opacity = max(0.0, min(1.0, float(ship_opacity)))
    key = np.array([-0.58, 0.22, 0.78], dtype=np.float32)
    key = key / (np.linalg.norm(key) + 1.0e-9)
    fill = np.array([0.46, -0.30, 0.70], dtype=np.float32)
    fill = fill / (np.linalg.norm(fill) + 1.0e-9)
    base_col_blue = np.array([56.0, 108.0, 228.0], dtype=np.float32)
    base_col_red = np.array([228.0, 56.0, 82.0], dtype=np.float32)
    highlight_col = np.array([245.0, 248.0, 255.0], dtype=np.float32)

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

        # Keep full culling only when some translucency is requested.
        facing = float(np.dot(normal, view_dir))
        if facing <= 0.0:
            if opacity >= 0.98:
                # Double-sided fallback for fully opaque ship to avoid hollow/translucent appearance.
                normal = -normal
            else:
                continue

        lambert_key = max(0.0, float(np.dot(normal, key)))
        lambert_fill = max(0.0, float(np.dot(normal, fill)))
        ndotv = max(0.0, float(np.dot(normal, view_dir)))
        rim = pow(max(0.0, 1.0 - ndotv), 2.6)
        # Blinn-Phong style highlight.
        half_vec = key + view_dir
        half_norm = float(np.linalg.norm(half_vec))
        if half_norm > 1.0e-9:
            half_vec = half_vec / half_norm
        spec = pow(max(0.0, float(np.dot(normal, half_vec))), 30.0)

        # Futuristic red/blue hull palette with panel variation.
        panel = 0.5 + 0.5 * math.sin(11.0 * float(center[0]) + 6.0 * float(center[1]) + 2.0 * float(center[2]))
        tone = 0.5 + 0.5 * math.sin(8.5 * float(center[0]) - 4.2 * float(center[1]) + 6.8 * float(center[2]))
        stripe = 0.5 + 0.5 * math.sin(18.0 * float(center[2]) + 5.0 * float(center[0]))
        blend_rb = max(0.0, min(1.0, 0.70 * tone + 0.30 * stripe))
        base_col = base_col_blue * (1.0 - blend_rb) + base_col_red * blend_rb

        intensity = (
            0.12
            + (0.64 + 0.10 * s) * lambert_key
            + (0.24 + 0.06 * s) * lambert_fill
            + (0.42 + 0.10 * s) * rim
            + (0.30 + 0.10 * s) * spec
        )
        intensity = max(0.0, min(1.75, intensity))
        metal = base_col * (0.84 + 0.16 * panel) * intensity
        spec_tint = highlight_col * (0.10 + 0.30 * spec + 0.10 * rim)
        col_f = np.clip(metal + spec_tint, 0.0, 255.0)
        col = col_f.astype(np.uint8)
        if opacity >= 0.98:
            alpha = 255
        else:
            alpha_dyn = int(max(170, min(255, 206 + 44 * ndotv + 15 * rim)))
            alpha = int(max(0, min(255, (1.0 - opacity) * alpha_dyn + opacity * 255.0)))
        x0, y0 = float(pts[0, 0]), float(pts[0, 1])
        x1, y1 = float(pts[1, 0]), float(pts[1, 1])
        x2, y2 = float(pts[2, 0]), float(pts[2, 1])
        xmin = max(0, int(math.floor(min(x0, x1, x2))))
        xmax = min(width - 1, int(math.ceil(max(x0, x1, x2))))
        ymin = max(0, int(math.floor(min(y0, y1, y2))))
        ymax = min(height - 1, int(math.ceil(max(y0, y1, y2))))
        if xmin > xmax or ymin > ymax:
            continue

        den = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(den) <= 1.0e-9:
            continue

        yy, xx = np.mgrid[ymin : ymax + 1, xmin : xmax + 1]
        px = xx.astype(np.float32) + 0.5
        py = yy.astype(np.float32) + 0.5
        w0 = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) / den
        w1 = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) / den
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -1.0e-4) & (w1 >= -1.0e-4) & (w2 >= -1.0e-4)
        if not bool(np.any(inside)):
            continue

        zpix = w0 * float(ztri[0]) + w1 * float(ztri[1]) + w2 * float(ztri[2])
        sub_z = zbuf[ymin : ymax + 1, xmin : xmax + 1]
        closer = inside & (zpix > 1.0e-4) & (zpix < sub_z)
        if not bool(np.any(closer)):
            continue
        sub_z[closer] = zpix[closer]
        zbuf[ymin : ymax + 1, xmin : xmax + 1] = sub_z
        sub_rgb = ship_rgb[ymin : ymax + 1, xmin : xmax + 1]
        sub_a = ship_alpha[ymin : ymax + 1, xmin : xmax + 1]
        sub_rgb[closer] = col
        sub_a[closer] = alpha
        ship_rgb[ymin : ymax + 1, xmin : xmax + 1] = sub_rgb
        ship_alpha[ymin : ymax + 1, xmin : xmax + 1] = sub_a
        visible_faces.append((int(tri[0]), int(tri[1]), int(tri[2])))

    overlay_np = np.zeros((height, width, 4), dtype=np.uint8)
    overlay_np[:, :, :3] = ship_rgb
    overlay_np[:, :, 3] = ship_alpha
    overlay = Image.fromarray(overlay_np, mode="RGBA")
    alpha_mask = ship_alpha > 8

    # Neon strips on selected visible edges.
    if visible_faces and bool(np.any(alpha_mask)):
        neon_core = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        ndraw = ImageDraw.Draw(neon_core, "RGBA")
        edge_set: set[tuple[int, int]] = set()
        for a, b, c in visible_faces:
            e0 = (a, b) if a < b else (b, a)
            e1 = (b, c) if b < c else (c, b)
            e2 = (c, a) if c < a else (a, c)
            edge_set.add(e0)
            edge_set.add(e1)
            edge_set.add(e2)

        for i, j in edge_set:
            hcode = (i * 73856093) ^ (j * 19349663)
            if hcode % 9 not in (0, 1, 2):
                continue
            p0 = screen[i]
            p1 = screen[j]
            z0 = float(z_cam[i])
            z1 = float(z_cam[j])
            if z0 <= 1.0e-4 or z1 <= 1.0e-4:
                continue
            mx = int(round(0.5 * (p0[0] + p1[0])))
            my = int(round(0.5 * (p0[1] + p1[1])))
            if mx < 0 or mx >= width or my < 0 or my >= height:
                continue
            if 0.5 * (z0 + z1) > float(zbuf[my, mx]) + 0.03:
                continue
            if (hcode & 1) == 0:
                neon_col = (74, 156, 255, 225)
                glow_col = (66, 133, 255, 120)
            else:
                neon_col = (255, 66, 96, 225)
                glow_col = (255, 70, 118, 120)
            width_px = 1 if (hcode % 3) else 2
            ndraw.line([(float(p0[0]), float(p0[1])), (float(p1[0]), float(p1[1]))], fill=neon_col, width=width_px)
            ndraw.line([(float(p0[0]), float(p0[1])), (float(p1[0]), float(p1[1]))], fill=glow_col, width=width_px + 1)

        mask_img = Image.fromarray((alpha_mask.astype(np.uint8) * 255), mode="L")
        core_np = np.array(neon_core, dtype=np.uint8, copy=True)
        mask_core = np.asarray(mask_img.filter(ImageFilter.MaxFilter(size=3)), dtype=np.uint8) > 0
        core_np[~mask_core, 3] = 0
        neon_blur = Image.fromarray(core_np, mode="RGBA").filter(ImageFilter.GaussianBlur(radius=max(1.6, 2.1 * s)))
        blur_np = np.array(neon_blur, dtype=np.uint8, copy=True)
        mask_glow = np.asarray(mask_img.filter(ImageFilter.MaxFilter(size=13)), dtype=np.uint8) > 0
        blur_np[~mask_glow, 3] = 0
        overlay = Image.alpha_composite(overlay, Image.fromarray(blur_np, mode="RGBA"))
        overlay = Image.alpha_composite(overlay, Image.fromarray(core_np, mode="RGBA"))

    # Add subtle engine glow near ship bounds for cinematic feel.
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
        glow_np = np.array(glow, dtype=np.uint8, copy=True)
        hull_block = np.asarray(
            Image.fromarray((alpha_mask.astype(np.uint8) * 255), mode="L").filter(ImageFilter.MaxFilter(size=11)),
            dtype=np.uint8,
        ) > 0
        glow_np[hull_block, 3] = 0
        glow = Image.fromarray(glow_np, mode="RGBA")
        overlay = Image.alpha_composite(overlay, glow)

    out = base.convert("RGBA")
    out.alpha_composite(overlay)
    return out.convert("RGB"), alpha_mask


def _postprocess_cinematic(image: Image.Image, strength: float) -> Image.Image:
    s = max(0.0, min(2.0, float(strength)))
    if s <= 1.0e-6:
        return image

    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    lum = 0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2]

    # Bloom + anamorphic flare from brightest highlights.
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

    # Contrast + split toning (cool shadows, warm highlights).
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

    # Filmic tonemap for smoother highlight rolloff.
    arr = _aces_fitted_tonemap(arr * (1.0 + 0.05 * s))

    # Vignette.
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx = (xx - 0.5 * w) / max(1.0, 0.5 * w)
    ny = (yy - 0.5 * h) / max(1.0, 0.5 * h)
    rr = np.sqrt(nx * nx + ny * ny)
    vign = np.clip(1.0 - 0.34 * s * (rr ** 1.55), 0.62, 1.0)
    arr *= vign[:, :, None]

    # Soft film grain.
    seed = int(1000.0 * s + 137)
    rng = np.random.default_rng(seed)
    grain = rng.normal(loc=0.0, scale=1.0, size=(h, w, 1)).astype(np.float32)
    grain_amp = 0.010 + 0.014 * s
    arr = np.clip(arr + grain_amp * grain, 0.0, 1.0)

    # Mild cinematic letterbox for a stronger look.
    target_aspect = 2.39
    current_aspect = float(w) / float(h)
    if current_aspect < target_aspect:
        active_h = int(round(float(w) / target_aspect))
        pad = max(0, (h - active_h) // 2)
        if pad > 0:
            arr[:pad, :, :] *= 0.02
            arr[h - pad :, :, :] *= 0.02

    arr = np.clip(arr, 0.0, 1.0)
    return Image.fromarray(np.clip(arr * 255.0, 0.0, 255.0).astype(np.uint8), mode="RGB")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a KNdS frame with polygonal OBJ spaceship overlay.")
    parser.add_argument("--obj", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=288)
    parser.add_argument("--observer-radius", type=float, default=30.0)
    parser.add_argument("--observer-theta-deg", type=float, default=87.0)
    parser.add_argument("--observer-phi-deg", type=float, default=0.0)
    parser.add_argument("--ship-radius", type=float, default=20.0)
    parser.add_argument("--ship-theta-deg", type=float, default=87.0)
    parser.add_argument("--ship-phi-deg", type=float, default=0.0)
    parser.add_argument("--ship-size", type=float, default=1.9, help="Characteristic ship size in M units")
    parser.add_argument("--ship-yaw-deg", type=float, default=34.0)
    parser.add_argument("--ship-pitch-deg", type=float, default=-12.0)
    parser.add_argument("--ship-roll-deg", type=float, default=16.0)
    parser.add_argument("--ship-opacity", type=float, default=1.0, help="0=trasparente, 1=opaco")
    parser.add_argument("--cinematic-strength", type=float, default=1.45)
    parser.add_argument("--disk-inner-radius", type=float, default=None, help="None => r_in automatico (ISCO)")
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

    vertices, faces = _load_obj_mesh(args.obj)
    vertices = _normalize_vertices(vertices)

    eye, right, up, forward = _camera_basis(tracer)
    ship_pos = (
        tracer._bl_to_cartesian_kerr_schild(
            torch.as_tensor([float(args.ship_radius)], dtype=tracer.dtype, device=tracer.device),
            torch.as_tensor([math.radians(float(args.ship_theta_deg))], dtype=tracer.dtype, device=tracer.device),
            torch.as_tensor([math.radians(float(args.ship_phi_deg))], dtype=tracer.dtype, device=tracer.device),
        )
        .reshape(3)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    # Local-to-world basis: x->right, y->up, z->toward camera.
    basis = np.stack([right, up, -forward], axis=1)
    rot_local = _rotation_matrix(
        yaw_deg=float(args.ship_yaw_deg),
        pitch_deg=float(args.ship_pitch_deg),
        roll_deg=float(args.ship_roll_deg),
    )
    vertices_local = (vertices @ rot_local.T).astype(np.float32)
    world_vertices = ship_pos.reshape(1, 3) + (vertices_local * float(args.ship_size)) @ basis.T

    out, ship_mask = _render_mesh_overlay(
        base=base,
        world_vertices=world_vertices,
        faces=faces,
        eye=eye,
        right=right,
        up=up,
        forward=forward,
        fov_deg=float(cfg.fov_deg),
        cinematic_strength=float(args.cinematic_strength),
        ship_opacity=float(args.ship_opacity),
    )
    graded = _postprocess_cinematic(out, strength=float(args.cinematic_strength))
    # Keep ship body visually solid after global bloom/flare.
    if bool(np.any(ship_mask)):
        raw_np = np.asarray(out, dtype=np.uint8)
        grd_np = np.array(graded, dtype=np.uint8, copy=True)
        grd_np[ship_mask] = raw_np[ship_mask]
        out = Image.fromarray(grd_np, mode="RGB")
    else:
        out = graded
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.save(args.output)
    print(f"SAVED={args.output.resolve()}")
    print(f"DEVICE={tracer.device}")
    print(f"TRIANGLES={faces.shape[0]}")


if __name__ == "__main__":
    main()
