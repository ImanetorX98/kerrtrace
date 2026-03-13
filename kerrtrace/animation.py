from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
import math
import numpy as np
from pathlib import Path
import queue
import shutil
import subprocess
import tempfile
import threading
import time
from typing import Callable

from PIL import Image
import torch

from .config import RenderConfig
from .raytracer import KerrRayTracer


VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv"}


@dataclass
class AnimationStats:
    frames: int
    fps: int
    elapsed_seconds: float
    output_path: Path


def _format_eta(seconds: float | None) -> str:
    if seconds is None or (not math.isfinite(seconds)):
        return "--:--:--"
    secs = max(0, int(round(seconds)))
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _missing_frame_indices(frames_dir: Path, frames: int) -> list[int]:
    return [idx for idx in range(frames) if not (frames_dir / f"frame_{idx:05d}.png").exists()]


def _clear_torch_cache() -> None:
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass


def _angular_delta_deg(a: float, b: float) -> float:
    d = b - a
    while d > 180.0:
        d -= 360.0
    while d < -180.0:
        d += 360.0
    return d


def _estimate_motion_shift_pixels(prev_cfg: RenderConfig, cur_cfg: RenderConfig) -> tuple[float, float]:
    f = 0.5 * float(cur_cfg.width) / math.tan(math.radians(0.5 * float(cur_cfg.fov_deg)))
    daz = math.radians(_angular_delta_deg(prev_cfg.observer_azimuth_deg, cur_cfg.observer_azimuth_deg))
    din = math.radians(cur_cfg.observer_inclination_deg - prev_cfg.observer_inclination_deg)
    scale = float(cur_cfg.motion_vector_scale)
    return -f * daz * scale, f * din * scale


def _shift_frame_integer(src: np.ndarray, shift_x: float, shift_y: float) -> np.ndarray:
    dx = int(round(shift_x))
    dy = int(round(shift_y))
    h, w = src.shape[0], src.shape[1]
    out = np.zeros_like(src)

    if abs(dx) >= w or abs(dy) >= h:
        return out

    if dx >= 0:
        sx0, sx1 = 0, w - dx
        dx0, dx1 = dx, w
    else:
        sx0, sx1 = -dx, w
        dx0, dx1 = 0, w + dx

    if dy >= 0:
        sy0, sy1 = 0, h - dy
        dy0, dy1 = dy, h
    else:
        sy0, sy1 = -dy, h
        dy0, dy1 = 0, h + dy

    out[dy0:dy1, dx0:dx1, :] = src[sy0:sy1, sx0:sx1, :]
    return out


def _local_min_max_rgb(src: np.ndarray, radius: int) -> tuple[np.ndarray, np.ndarray]:
    r = max(0, int(radius))
    if r <= 0:
        return src.copy(), src.copy()
    h, w = src.shape[0], src.shape[1]
    padded = np.pad(src, ((r, r), (r, r), (0, 0)), mode="edge")
    local_min = np.full_like(src, 255.0, dtype=np.float32)
    local_max = np.zeros_like(src, dtype=np.float32)
    for dy in range(-r, r + 1):
        ys = dy + r
        row_slice = padded[ys : ys + h, :, :]
        for dx in range(-r, r + 1):
            xs = dx + r
            view = row_slice[:, xs : xs + w, :]
            local_min = np.minimum(local_min, view)
            local_max = np.maximum(local_max, view)
    return local_min, local_max


def _apply_temporal_denoise(
    *,
    current_u8: np.ndarray,
    prev_temporal: np.ndarray,
    prev_cfg: RenderConfig,
    cur_cfg: RenderConfig,
    cfg: RenderConfig,
) -> np.ndarray:
    cur = current_u8.astype(np.float32, copy=False)
    shift_x, shift_y = _estimate_motion_shift_pixels(prev_cfg, cur_cfg)
    reproj = _shift_frame_integer(prev_temporal.astype(np.float32, copy=False), shift_x, shift_y)
    clamp_v = float(cfg.temporal_clamp)
    reproj = cur + np.clip(reproj - cur, -clamp_v, clamp_v)

    alpha = float(cfg.temporal_blend)
    mode = str(cfg.temporal_denoise_mode)
    if mode == "robust":
        radius = int(cfg.temporal_denoise_radius)
        local_min, local_max = _local_min_max_rgb(cur, radius=radius)
        clip = float(cfg.temporal_denoise_clip)
        reproj = np.minimum(np.maximum(reproj, local_min - clip), local_max + clip)

        delta = np.abs(reproj - cur)
        luma_delta = (
            0.2126 * delta[:, :, 0]
            + 0.7152 * delta[:, :, 1]
            + 0.0722 * delta[:, :, 2]
        )
        sigma = max(1.0e-6, float(cfg.temporal_denoise_sigma))
        confidence = np.exp(-np.square(luma_delta / sigma))
        alpha_map = alpha * confidence
        out = (1.0 - alpha_map[..., None]) * cur + alpha_map[..., None] * reproj
        return np.clip(out, 0.0, 255.0)

    out = (1.0 - alpha) * cur + alpha * reproj
    return np.clip(out, 0.0, 255.0)


def _build_generalized_doran_radius_schedule(
    base_config: RenderConfig,
    frames: int,
    observer_radius_start: float,
    observer_radius_end: float,
    inclination_probe_deg: float,
    samples: int,
    proper_time: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build r(lambda) for generalized Doran sweeps by inverting a PG-like relation.

    The local infall speed proxy follows the same camera construction used by
    `_camera_rays_generalized_doran`:
      v_ff(r) = sqrt(1 - alpha(r)^2)
    with alpha obtained from the KS-family metric lapse at the probe worldline.
    If `proper_time=True`, lambda approximates the observer proper time using
      d tau = dt / gamma(v_ff), gamma = 1/sqrt(1-v_ff^2).
    """
    if frames <= 0:
        raise ValueError("frames must be > 0")
    if samples < 64:
        raise ValueError("generalized_doran_time_samples must be >= 64")
    if frames == 1:
        return np.asarray([float(observer_radius_start)], dtype=np.float64), np.asarray([0.0], dtype=np.float64)

    probe_cfg = replace(
        base_config,
        width=max(64, min(256, int(base_config.width))),
        height=max(64, min(256, int(base_config.height))),
        show_progress_bar=False,
    ).validated()
    probe = KerrRayTracer(probe_cfg)

    theta = math.radians(float(inclination_probe_deg))
    axis_eps = math.radians(0.35)
    theta = min(math.pi - axis_eps, max(axis_eps, theta))
    phi = math.radians(float(base_config.observer_azimuth_deg))

    r_grid = np.linspace(float(observer_radius_start), float(observer_radius_end), int(samples), dtype=np.float64)
    with torch.no_grad():
        r_t = torch.as_tensor(r_grid, dtype=probe.dtype, device=probe.device)
        th_t = torch.full_like(r_t, theta)
        ph_t = torch.full_like(r_t, phi)
        xyz = probe._bl_to_cartesian_kerr_schild(r_t, th_t, ph_t)
        _, g_inv, _ = probe._kerr_schild_metric_and_inverse(xyz)
        alpha = torch.rsqrt(torch.clamp(-g_inv[:, 0, 0], min=1.0e-12))
        v_ff = torch.sqrt(torch.clamp(1.0 - alpha * alpha, min=1.0e-8, max=0.999 * 0.999))
        v = v_ff.detach().to(device="cpu").to(dtype=torch.float64).numpy()

    dr = np.abs(np.diff(r_grid))
    v_mid = 0.5 * (v[:-1] + v[1:])
    dt = dr / np.clip(v_mid, 1.0e-8, None)
    if proper_time:
        gamma_mid = 1.0 / np.sqrt(np.clip(1.0 - v_mid * v_mid, 1.0e-8, None))
        dparam = dt / np.clip(gamma_mid, 1.0e-8, None)
    else:
        dparam = dt

    param_grid = np.concatenate(([0.0], np.cumsum(dparam)))
    param_end = float(param_grid[-1])
    if not np.isfinite(param_end) or param_end <= 0.0:
        param_uniform = np.linspace(0.0, 1.0, frames, dtype=np.float64)
        r_uniform = np.linspace(float(observer_radius_start), float(observer_radius_end), frames, dtype=np.float64)
        return r_uniform, param_uniform

    param_uniform = np.linspace(0.0, param_end, frames, dtype=np.float64)
    r_uniform = np.interp(param_uniform, param_grid, r_grid)
    return r_uniform, param_uniform



def _render_frames(
    base_config: RenderConfig,
    frames_dir: Path,
    frames: int,
    fps: int,
    azimuth_orbits: float,
    inclination_wobble_deg: float,
    inclination_start_deg: float | None,
    inclination_end_deg: float | None,
    observer_radius_start: float | None,
    observer_radius_end: float | None,
    inclination_sweep_ease: bool,
    taa_samples: int,
    shutter_fraction: float,
    spatial_jitter: bool,
    generalized_doran_fixed_time: bool = False,
    generalized_doran_fixed_proper_time: bool = False,
    generalized_doran_time_samples: int = 2048,
    generalized_doran_radius_log: str | Path | None = None,
    resume_frames: bool = False,
    workers: int = 1,
    frame_sink: Callable[[int, np.ndarray], None] | None = None,
    save_frames: bool = True,
    adaptive_frame_steps: bool = True,
    adaptive_frame_steps_min_scale: float = 0.60,
) -> None:
    two_pi = 2.0 * math.pi
    if save_frames:
        frames_dir.mkdir(parents=True, exist_ok=True)

    if frames <= 0:
        raise ValueError("frames must be >= 1")
    if fps <= 0:
        raise ValueError("fps must be >= 1")
    if taa_samples <= 0:
        raise ValueError("taa_samples must be >= 1")
    shutter_fraction = max(0.0, min(1.0, shutter_fraction))
    jitter_strength = 0.28
    radius_schedule: np.ndarray | None = None
    adaptive_frame_steps_min_scale = min(1.0, max(0.25, float(adaptive_frame_steps_min_scale)))

    if generalized_doran_fixed_time and generalized_doran_fixed_proper_time:
        raise ValueError("Use only one of generalized_doran_fixed_time or generalized_doran_fixed_proper_time")

    if generalized_doran_fixed_time or generalized_doran_fixed_proper_time:
        if base_config.coordinate_system != "generalized_doran":
            raise ValueError(
                "generalized_doran_fixed_time/generalized_doran_fixed_proper_time are only valid "
                "with --coordinate-system generalized_doran"
            )
        if observer_radius_start is None or observer_radius_end is None:
            raise ValueError(
                "generalized_doran_fixed_time/generalized_doran_fixed_proper_time require "
                "observer_radius_start and observer_radius_end"
            )
        inclination_probe = (
            float(inclination_start_deg)
            if inclination_start_deg is not None
            else float(base_config.observer_inclination_deg)
        )
        radius_schedule, time_schedule = _build_generalized_doran_radius_schedule(
            base_config=base_config,
            frames=frames,
            observer_radius_start=float(observer_radius_start),
            observer_radius_end=float(observer_radius_end),
            inclination_probe_deg=inclination_probe,
            samples=int(generalized_doran_time_samples),
            proper_time=bool(generalized_doran_fixed_proper_time),
        )
        time_label = "tau_end" if generalized_doran_fixed_proper_time else "t_end"
        mode_label = "fixed-proper-time" if generalized_doran_fixed_proper_time else "fixed-time"
        print(
            "Generalized-Doran {mode} sampling enabled: {tlab}={tend:.4f}, r_start={rs:.4f}, r_end={re:.4f}, frames={n}".format(
                mode=mode_label,
                tlab=time_label,
                tend=float(time_schedule[-1]),
                rs=float(radius_schedule[0]),
                re=float(radius_schedule[-1]),
                n=frames,
            )
        )
        if generalized_doran_radius_log is not None:
            log_path = Path(generalized_doran_radius_log)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("w", encoding="utf-8") as fh:
                fh.write("frame,time,radius\n")
                for idx, (tt, rr) in enumerate(zip(time_schedule, radius_schedule)):
                    fh.write(f"{idx},{float(tt):.10f},{float(rr):.10f}\n")
            print(f"Generalized-Doran radius schedule written: {log_path}")

    def _radical_inverse(i: int, base: int) -> float:
        inv = 1.0 / float(base)
        denom = inv
        x = 0.0
        n = i
        while n > 0:
            x += (n % base) * denom
            n //= base
            denom *= inv
        return x

    def _radius_at_phase(phase_value: float) -> float:
        if radius_schedule is None:
            if observer_radius_start is not None and observer_radius_end is not None:
                return observer_radius_start + (observer_radius_end - observer_radius_start) * phase_value
            return float(base_config.observer_radius)

        if frames <= 1:
            return float(radius_schedule[0])
        pos = phase_value * (frames - 1)
        lo = int(math.floor(pos))
        hi = min(frames - 1, lo + 1)
        frac = pos - lo
        return float((1.0 - frac) * radius_schedule[lo] + frac * radius_schedule[hi])

    def config_at_phase(phase_value: float) -> RenderConfig:
        azimuth = base_config.observer_azimuth_deg + 360.0 * azimuth_orbits * phase_value
        wobble = inclination_wobble_deg * math.sin(two_pi * phase_value)
        duration_s = float(frames) / float(fps)
        disk_phase = float(base_config.disk_layer_global_phase) + two_pi * float(base_config.disk_layer_phase_rate_hz) * (phase_value * duration_s)

        if inclination_start_deg is not None and inclination_end_deg is not None:
            sweep = phase_value
            if inclination_sweep_ease:
                sweep = sweep * sweep * (3.0 - 2.0 * sweep)
            inclination = inclination_start_deg + (inclination_end_deg - inclination_start_deg) * sweep
        else:
            inclination = base_config.observer_inclination_deg
        inclination = inclination + wobble

        radius = _radius_at_phase(phase_value)

        return replace(
            base_config,
            observer_azimuth_deg=azimuth,
            observer_inclination_deg=inclination,
            observer_radius=radius,
            disk_layer_global_phase=disk_phase,
        ).validated()

    def _frame_steps_for_config(frame_cfg: RenderConfig) -> int:
        if (not adaptive_frame_steps) or base_config.max_steps <= 64:
            return int(base_config.max_steps)
        # Edge-on views and smaller observer radii are generally more expensive.
        inc = math.radians(float(frame_cfg.observer_inclination_deg))
        inc_factor = abs(math.sin(inc))
        radius = max(1.0, float(frame_cfg.observer_radius))
        radius_factor = min(1.0, 24.0 / radius)
        complexity = 0.82 * inc_factor + 0.18 * radius_factor
        scale = adaptive_frame_steps_min_scale + (1.0 - adaptive_frame_steps_min_scale) * complexity
        steps = int(round(float(base_config.max_steps) * scale))
        return max(64, min(int(base_config.max_steps), steps))

    def _frame_priority(index: int) -> float:
        phase = 0.0 if frames == 1 else index / (frames - 1)
        cfg_i = config_at_phase(phase)
        inc = math.radians(float(cfg_i.observer_inclination_deg))
        inc_factor = abs(math.sin(inc))
        radius = max(1.0, float(cfg_i.observer_radius))
        radius_factor = min(1.0, 24.0 / radius)
        return 0.80 * inc_factor + 0.20 * radius_factor

    tracer_tls = threading.local()
    sequential_tracer: KerrRayTracer | None = None

    def _get_thread_tracer(seed_cfg: RenderConfig) -> KerrRayTracer:
        tracer = getattr(tracer_tls, "tracer", None)
        if tracer is None:
            tracer = KerrRayTracer(seed_cfg)
            tracer_tls.tracer = tracer
        return tracer

    def _render_single(index: int, tracer: KerrRayTracer | None = None) -> tuple[int, RenderConfig, np.ndarray]:
        phase = 0.0 if frames == 1 else index / (frames - 1)
        nominal_cfg = config_at_phase(phase)
        accum: np.ndarray | None = None
        work_tracer = tracer if tracer is not None else _get_thread_tracer(nominal_cfg)

        for sample_idx in range(taa_samples):
            if taa_samples == 1:
                sample_phase = phase
            else:
                sample_t = (sample_idx + 0.5) / taa_samples
                dt = (sample_t - 0.5) * shutter_fraction / max(frames - 1, 1)
                sample_phase = min(1.0, max(0.0, phase + dt))

            frame_config = config_at_phase(sample_phase)
            frame_steps = _frame_steps_for_config(frame_config)
            work_tracer.set_observer(
                observer_radius=frame_config.observer_radius,
                observer_inclination_deg=frame_config.observer_inclination_deg,
                observer_azimuth_deg=frame_config.observer_azimuth_deg,
                observer_roll_deg=frame_config.observer_roll_deg,
                max_steps=frame_steps,
                disk_layer_global_phase=frame_config.disk_layer_global_phase,
            )

            x_jitter = 0.0
            y_jitter = 0.0
            if spatial_jitter and taa_samples > 1:
                j = index * taa_samples + sample_idx + 1
                x_jitter = (2.0 * _radical_inverse(j, 2) - 1.0) * jitter_strength
                y_jitter = (2.0 * _radical_inverse(j, 3) - 1.0) * jitter_strength

            result = work_tracer.render(x_pixel_offset=x_jitter, y_pixel_offset=y_jitter)
            rgb = np.asarray(result.image, dtype=np.float32)
            del result
            if accum is None:
                accum = rgb
            else:
                accum += rgb

        if accum is None:
            raise RuntimeError("Frame accumulation failed")
        avg = np.clip(accum / float(taa_samples), 0.0, 255.0).astype(np.uint8)
        return index, nominal_cfg, avg

    frame_retry_attempts = 2
    slow_frame_warn_seconds = 180.0

    def _render_single_with_retry(index: int) -> tuple[int, RenderConfig, np.ndarray, float, int]:
        last_exc: Exception | None = None
        for attempt in range(1, frame_retry_attempts + 1):
            t_frame = time.perf_counter()
            try:
                idx, nominal_cfg, avg = _render_single(index)
                dt = time.perf_counter() - t_frame
                if dt >= slow_frame_warn_seconds:
                    print(
                        f"Frame {idx + 1}/{frames}: slow frame ({dt:.2f}s). "
                        "If interrupted, rerun with --resume-frames."
                    )
                return idx, nominal_cfg, avg, dt, attempt
            except Exception as exc:
                last_exc = exc
                print(f"Frame {index + 1}/{frames}: attempt {attempt}/{frame_retry_attempts} failed: {exc}")
                _clear_torch_cache()
                if attempt < frame_retry_attempts:
                    time.sleep(min(6.0, 1.5 * attempt))
        if last_exc is None:
            raise RuntimeError(f"Frame {index + 1}/{frames} failed for unknown reasons")
        raise RuntimeError(f"Frame {index + 1}/{frames} failed after {frame_retry_attempts} attempts: {last_exc}") from last_exc

    pending_indices = list(range(frames))
    if resume_frames:
        pending_indices = [idx for idx in pending_indices if not (frames_dir / f"frame_{idx:05d}.png").exists()]
        if not pending_indices:
            print(f"All {frames} frames already present in {frames_dir}; skipping render.")
            return
        pending_indices.sort(key=_frame_priority, reverse=True)

    can_parallel = (
        workers > 1
        and (not base_config.temporal_reprojection)
        and frame_sink is None
        and save_frames
    )
    if can_parallel:
        parallel_t0 = time.perf_counter()
        done = 0
        total = len(pending_indices)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_render_single_with_retry, index) for index in pending_indices]
            for future in as_completed(futures):
                index, _, avg, frame_dt, attempts = future.result()
                if save_frames:
                    frame_image = Image.fromarray(avg, mode="RGB")
                    frame_path = frames_dir / f"frame_{index:05d}.png"
                    frame_image.save(frame_path)
                else:
                    frame_path = Path(f"(stream:{index:05d})")
                done += 1
                eta_seconds: float | None = None
                if 0 < done < total:
                    elapsed = max(1.0e-6, time.perf_counter() - parallel_t0)
                    eta_seconds = elapsed * float(total - done) / float(done)
                retry_txt = "" if attempts == 1 else f" | retries={attempts - 1}"
                print(
                    f"Frame {index + 1}/{frames}: {frame_path} | frame {frame_dt:.2f}s"
                    f"{retry_txt} | ETA ~{_format_eta(eta_seconds)}"
                )
                if frame_sink is not None:
                    frame_sink(index, avg)
        return

    prev_temporal: np.ndarray | None = None
    prev_temporal_cfg: RenderConfig | None = None
    sequential_tracer = KerrRayTracer(config_at_phase(0.0))
    sequential_t0 = time.perf_counter()
    rendered_done = 0
    total_to_render = len(pending_indices)
    for index in range(frames):
        frame_path = frames_dir / f"frame_{index:05d}.png"
        if resume_frames and frame_path.exists():
            print(f"Frame {index + 1}/{frames}: {frame_path} (existing, skipped)")
            continue
        t_frame = time.perf_counter()
        try:
            idx, nominal_cfg, avg_u8 = _render_single(index, tracer=sequential_tracer)
            frame_dt = time.perf_counter() - t_frame
            attempts = 1
            index = idx
            if frame_dt >= slow_frame_warn_seconds:
                print(
                    f"Frame {index + 1}/{frames}: slow frame ({frame_dt:.2f}s). "
                    "If interrupted, rerun with --resume-frames."
                )
        except Exception as exc:
            print(f"Frame {index + 1}/{frames}: sequential reusable tracer failed, fallback to retry path: {exc}")
            index, nominal_cfg, avg_u8, frame_dt, attempts = _render_single_with_retry(index)
        avg = avg_u8.astype(np.float32)
        if base_config.temporal_reprojection and prev_temporal is not None and prev_temporal_cfg is not None:
            avg = _apply_temporal_denoise(
                current_u8=avg_u8,
                prev_temporal=prev_temporal,
                prev_cfg=prev_temporal_cfg,
                cur_cfg=nominal_cfg,
                cfg=base_config,
            )

        prev_temporal = avg.copy()
        prev_temporal_cfg = nominal_cfg
        out_u8 = np.clip(avg, 0.0, 255.0).astype(np.uint8)
        if save_frames:
            frame_image = Image.fromarray(out_u8, mode="RGB")
            frame_image.save(frame_path)
            frame_label = str(frame_path)
        else:
            frame_label = f"(stream:{index:05d})"
        if frame_sink is not None:
            frame_sink(index, out_u8)
        rendered_done += 1
        eta_seconds: float | None = None
        if 0 < rendered_done < total_to_render:
            elapsed = max(1.0e-6, time.perf_counter() - sequential_t0)
            eta_seconds = elapsed * float(total_to_render - rendered_done) / float(rendered_done)
        retry_txt = "" if attempts == 1 else f" | retries={attempts - 1}"
        print(
            f"Frame {index + 1}/{frames}: {frame_label} | frame {frame_dt:.2f}s"
            f"{retry_txt} | ETA ~{_format_eta(eta_seconds)}"
        )



def _encode_gif(frames_dir: Path, output_path: Path, fps: int) -> None:
    frame_paths = sorted(frames_dir.glob("frame_*.png"))
    if not frame_paths:
        raise RuntimeError("No frames found for GIF encoding")

    images: list[Image.Image] = []
    try:
        for path in frame_paths:
            images.append(Image.open(path).convert("RGB"))

        duration_ms = max(1, int(round(1000.0 / fps)))
        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            duration=duration_ms,
            loop=0,
            optimize=False,
            disposal=2,
        )
    finally:
        for image in images:
            image.close()



def _encode_video_ffmpeg(frames_dir: Path, output_path: Path, fps: int, cfg: RenderConfig) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found in PATH. Install ffmpeg or render GIF instead.")

    input_pattern = str(frames_dir / "frame_%05d.png")
    cmd = [ffmpeg, "-y", "-framerate", str(fps), "-i", input_pattern]
    if cfg.video_codec == "h265_10bit":
        cmd += [
            "-c:v",
            "libx265",
            "-pix_fmt",
            "yuv420p10le",
            "-tag:v",
            "hvc1",
            "-x265-params",
            "profile=main10",
            "-crf",
            str(cfg.video_crf),
        ]
    else:
        cmd += [
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            str(cfg.video_crf),
        ]
    cmd.append(str(output_path))
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        stderr = proc.stderr.strip()
        raise RuntimeError(f"ffmpeg failed with code {proc.returncode}: {stderr}")


def _open_video_ffmpeg_stream(
    output_path: Path,
    fps: int,
    cfg: RenderConfig,
    width: int,
    height: int,
) -> subprocess.Popen[bytes]:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found in PATH. Install ffmpeg or disable stream encoding.")
    cmd: list[str] = [
        ffmpeg,
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s:v",
        f"{width}x{height}",
        "-framerate",
        str(fps),
        "-i",
        "-",
    ]
    if cfg.video_codec == "h265_10bit":
        cmd += [
            "-c:v",
            "libx265",
            "-pix_fmt",
            "yuv420p10le",
            "-tag:v",
            "hvc1",
            "-x265-params",
            "profile=main10",
            "-crf",
            str(cfg.video_crf),
        ]
    else:
        cmd += [
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            str(cfg.video_crf),
        ]
    cmd.append(str(output_path))
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if proc.stdin is None:
        raise RuntimeError("Failed to open ffmpeg stdin for stream encoding")
    return proc


def _close_video_ffmpeg_stream(proc: subprocess.Popen[bytes]) -> None:
    if proc.stdin is not None:
        try:
            proc.stdin.close()
        except Exception:
            pass
    stderr_bytes = b""
    if proc.stderr is not None:
        stderr_bytes = proc.stderr.read() or b""
    ret = proc.wait()
    if ret != 0:
        stderr = stderr_bytes.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg stream encoding failed with code {ret}: {stderr}")



def render_animation(
    base_config: RenderConfig,
    output_path: str | Path,
    frames: int = 120,
    fps: int = 30,
    azimuth_orbits: float = 1.0,
    inclination_wobble_deg: float = 0.0,
    inclination_start_deg: float | None = None,
    inclination_end_deg: float | None = None,
    observer_radius_start: float | None = None,
    observer_radius_end: float | None = None,
    inclination_sweep_ease: bool = True,
    taa_samples: int = 1,
    shutter_fraction: float = 0.85,
    spatial_jitter: bool = False,
    generalized_doran_fixed_time: bool = False,
    generalized_doran_fixed_proper_time: bool = False,
    generalized_doran_time_samples: int = 2048,
    generalized_doran_radius_log: str | Path | None = None,
    frames_dir: str | Path | None = None,
    keep_frames: bool = False,
    resume_frames: bool = False,
    workers: int = 1,
    render_frames: bool = True,
    encode_output: bool = True,
    adaptive_frame_steps: bool = True,
    adaptive_frame_steps_min_scale: float = 0.60,
    stream_encode: bool = True,
    stream_encode_async: bool = True,
    stream_encode_queue_size: int = 4,
) -> AnimationStats:
    cfg = base_config.validated()
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    if fps <= 0:
        raise ValueError("fps must be > 0")
    if frames <= 0:
        raise ValueError("frames must be > 0")
    if (inclination_start_deg is None) ^ (inclination_end_deg is None):
        raise ValueError("Provide both inclination_start_deg and inclination_end_deg, or neither")
    if (observer_radius_start is None) ^ (observer_radius_end is None):
        raise ValueError("Provide both observer_radius_start and observer_radius_end, or neither")
    if generalized_doran_fixed_time and generalized_doran_fixed_proper_time:
        raise ValueError("Use only one of generalized_doran_fixed_time or generalized_doran_fixed_proper_time")
    if generalized_doran_fixed_time or generalized_doran_fixed_proper_time:
        if cfg.coordinate_system != "generalized_doran":
            raise ValueError(
                "generalized_doran_fixed_time/generalized_doran_fixed_proper_time require "
                "coordinate_system='generalized_doran'"
            )
        if observer_radius_start is None or observer_radius_end is None:
            raise ValueError(
                "generalized_doran_fixed_time/generalized_doran_fixed_proper_time require "
                "observer_radius_start and observer_radius_end"
            )
        if generalized_doran_time_samples < 64:
            raise ValueError("generalized_doran_time_samples must be >= 64")
    if workers < 1:
        raise ValueError("workers must be >= 1")
    if (not render_frames) and frames_dir is None:
        raise ValueError("render_frames=False requires frames_dir")
    if adaptive_frame_steps_min_scale <= 0.0 or adaptive_frame_steps_min_scale > 1.0:
        raise ValueError("adaptive_frame_steps_min_scale must be in (0, 1]")
    if stream_encode_queue_size < 1 or stream_encode_queue_size > 64:
        raise ValueError("stream_encode_queue_size must be in [1, 64]")

    t0 = time.perf_counter()

    suffix = target.suffix.lower()
    allow_stream = (
        stream_encode
        and render_frames
        and encode_output
        and (suffix in VIDEO_SUFFIXES)
        and (frames_dir is None)
        and (not keep_frames)
        and (not resume_frames)
    )
    save_frames = not allow_stream

    if save_frames:
        if frames_dir is not None:
            frame_dir = Path(frames_dir)
            frame_dir.mkdir(parents=True, exist_ok=True)
            temp_context = None
        elif keep_frames:
            frame_dir = target.parent / f"{target.stem}_frames"
            frame_dir.mkdir(parents=True, exist_ok=True)
            temp_context = None
        else:
            temp_context = tempfile.TemporaryDirectory(prefix="kerrtrace_frames_", dir=str(target.parent))
            frame_dir = Path(temp_context.name)
    else:
        frame_dir = target.parent / f"{target.stem}_stream_frames_unused"
        temp_context = None

    if save_frames and render_frames and (not resume_frames):
        for old_frame in frame_dir.glob("frame_*.png"):
            old_frame.unlink()

    stream_proc: subprocess.Popen[bytes] | None = None
    stream_q: queue.Queue[np.ndarray | None] | None = None
    stream_writer: threading.Thread | None = None
    stream_errors: list[Exception] = []

    def _start_async_stream_writer(proc: subprocess.Popen[bytes], qsize: int) -> tuple[queue.Queue[np.ndarray | None], threading.Thread]:
        if proc.stdin is None:
            raise RuntimeError("ffmpeg stream process has no stdin pipe")
        q: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=max(1, int(qsize)))

        def _writer_loop() -> None:
            try:
                while True:
                    item = q.get()
                    try:
                        if item is None:
                            break
                        proc.stdin.write(item.astype(np.uint8, copy=False).tobytes())
                    finally:
                        q.task_done()
            except Exception as exc:
                stream_errors.append(exc)

        thread = threading.Thread(target=_writer_loop, name="kerrtrace-ffmpeg-writer", daemon=True)
        thread.start()
        return q, thread

    try:
        resolved = cfg.resolve_device()
        eff_workers = workers if resolved.type == "cpu" else 1
        if allow_stream:
            stream_proc = _open_video_ffmpeg_stream(
                output_path=target,
                fps=fps,
                cfg=cfg,
                width=int(cfg.width),
                height=int(cfg.height),
            )
            if stream_encode_async:
                stream_q, stream_writer = _start_async_stream_writer(
                    proc=stream_proc,
                    qsize=stream_encode_queue_size,
                )

                def _stream_sink(_: int, frame_u8: np.ndarray) -> None:
                    if stream_errors:
                        raise RuntimeError(f"Async ffmpeg writer failed: {stream_errors[-1]}")
                    assert stream_q is not None
                    stream_q.put(frame_u8.astype(np.uint8, copy=False))
            else:
                def _stream_sink(_: int, frame_u8: np.ndarray) -> None:
                    assert stream_proc is not None and stream_proc.stdin is not None
                    stream_proc.stdin.write(frame_u8.astype(np.uint8, copy=False).tobytes())

            frame_sink = _stream_sink
        else:
            frame_sink = None

        if render_frames:
            _render_frames(
                base_config=cfg,
                frames_dir=frame_dir,
                frames=frames,
                fps=fps,
                azimuth_orbits=azimuth_orbits,
                inclination_wobble_deg=inclination_wobble_deg,
                inclination_start_deg=inclination_start_deg,
                inclination_end_deg=inclination_end_deg,
                observer_radius_start=observer_radius_start,
                observer_radius_end=observer_radius_end,
                inclination_sweep_ease=inclination_sweep_ease,
                taa_samples=taa_samples,
                shutter_fraction=shutter_fraction,
                spatial_jitter=spatial_jitter,
                generalized_doran_fixed_time=generalized_doran_fixed_time,
                generalized_doran_fixed_proper_time=generalized_doran_fixed_proper_time,
                generalized_doran_time_samples=generalized_doran_time_samples,
                generalized_doran_radius_log=generalized_doran_radius_log,
                resume_frames=resume_frames,
                workers=eff_workers,
                frame_sink=frame_sink,
                save_frames=save_frames,
                adaptive_frame_steps=adaptive_frame_steps,
                adaptive_frame_steps_min_scale=adaptive_frame_steps_min_scale,
            )

        if allow_stream and stream_proc is not None:
            if stream_q is not None:
                stream_q.put(None)
                stream_q.join()
                if stream_writer is not None:
                    stream_writer.join(timeout=2.0)
                if stream_errors:
                    raise RuntimeError(f"Async ffmpeg writer failed: {stream_errors[-1]}")
            _close_video_ffmpeg_stream(stream_proc)
            stream_proc = None

        if encode_output and (not allow_stream):
            missing = _missing_frame_indices(frame_dir, frames)
            if missing:
                preview = ", ".join(f"{idx:05d}" for idx in missing[:10])
                if len(missing) > 10:
                    preview += ", ..."
                raise RuntimeError(
                    f"Cannot encode animation: {len(missing)} frame(s) missing in {frame_dir} "
                    f"(first missing: {preview}). Rerun with --resume-frames."
                )
            if suffix == ".gif":
                _encode_gif(frame_dir, target, fps)
            elif suffix in VIDEO_SUFFIXES:
                _encode_video_ffmpeg(frame_dir, target, fps, cfg)
            else:
                raise ValueError("Unsupported animation output extension. Use .mp4, .mov, .mkv, or .gif")
    finally:
        if stream_q is not None:
            try:
                stream_q.put_nowait(None)
            except Exception:
                pass
            if stream_writer is not None:
                stream_writer.join(timeout=1.0)
        if stream_proc is not None:
            try:
                _close_video_ffmpeg_stream(stream_proc)
            except Exception:
                pass
        if temp_context is not None:
            temp_context.cleanup()

    dt = time.perf_counter() - t0
    output_ref = target if encode_output else frame_dir
    return AnimationStats(frames=frames, fps=fps, elapsed_seconds=dt, output_path=output_ref)
