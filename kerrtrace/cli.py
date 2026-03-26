from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import os
import platform
from pathlib import Path
import re
import time

import numpy as np
import torch

from .config import RenderConfig
from .animation import render_animation
from .charged_particles import ChargedParticleOrbiter
from .raytracer import KerrRayTracer

VIDEO_OUTPUT_SUFFIXES = {".mp4", ".mov", ".mkv", ".gif"}


def _is_video_like_output(path_value: str | None) -> bool:
    if not path_value:
        return False
    return Path(path_value).suffix.lower() in VIDEO_OUTPUT_SUFFIXES


def _resolve_progressive_output(path_value: str | None) -> str | None:
    """
    Expand progressive placeholders in output filenames.

    Supported forms:
    - "...{progressivo}..."
    - "...progressivo_..."
    """
    if not path_value:
        return path_value

    p = Path(path_value)
    name = p.name
    token = "{progressivo}"

    if token in name:
        prefix, suffix = name.split(token, 1)
    elif "progressivo_" in name:
        cut = name.index("progressivo_") + len("progressivo_")
        prefix = name[:cut]
        suffix = name[cut:]
        m_digits = re.match(r"^\d+", suffix)
        if m_digits:
            suffix = suffix[m_digits.end():]
    else:
        return path_value

    parent = p.parent if str(p.parent) else Path(".")
    max_idx = 0
    pattern = re.compile(r"^" + re.escape(prefix) + r"(\d+)" + re.escape(suffix) + r"$")

    if parent.exists():
        for child in parent.iterdir():
            m = pattern.match(child.name)
            if m:
                try:
                    max_idx = max(max_idx, int(m.group(1)))
                except ValueError:
                    continue

    next_idx = max_idx + 1
    resolved = p.with_name(f"{prefix}{next_idx:04d}{suffix}")
    return str(resolved)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kerr black hole GPU ray tracer")
    parser.add_argument("--config", type=str, help="Path to JSON config")

    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--fov-deg", type=float)
    parser.add_argument("--coordinate-system", choices=["boyer_lindquist", "kerr_schild", "generalized_doran"])
    parser.add_argument(
        "--metric-model",
        choices=[
            "schwarzschild",
            "kerr",
            "reissner_nordstrom",
            "kerr_newman",
            "schwarzschild_de_sitter",
            "kerr_de_sitter",
            "reissner_nordstrom_de_sitter",
            "kerr_newman_de_sitter",
            "morris_thorne",
        ],
    )
    parser.add_argument("--spin", type=float)
    parser.add_argument("--charge", type=float)
    parser.add_argument("--cosmological-constant", type=float)
    parser.add_argument("--wormhole-throat-radius", type=float)
    parser.add_argument("--wormhole-length-scale", type=float)
    parser.add_argument("--enable-wormhole-throat-crossing", action="store_true")
    parser.add_argument("--disable-wormhole-throat-crossing", action="store_true")
    parser.add_argument("--observer-radius", type=float)
    parser.add_argument("--observer-inclination-deg", type=float)
    parser.add_argument("--observer-azimuth-deg", type=float)
    parser.add_argument("--observer-roll-deg", type=float)
    parser.add_argument("--disk-inner-radius", type=float)
    parser.add_argument("--disk-outer-radius", type=float)
    parser.add_argument("--disable-accretion-disk", action="store_true")
    parser.add_argument("--enable-accretion-disk", action="store_true")
    parser.add_argument("--disk-model", choices=["legacy", "physical_nt"])
    parser.add_argument("--disk-radial-profile", choices=["nt_proxy", "nt_page_thorne"])
    parser.add_argument("--emissivity-index", type=float)
    parser.add_argument("--inner-edge-boost", type=float)
    parser.add_argument("--outer-edge-boost", type=float)
    parser.add_argument("--star-density", type=float)
    parser.add_argument("--star-brightness", type=float)
    parser.add_argument("--star-seed", type=int)
    parser.add_argument("--background-mode", choices=["procedural", "hdri", "darkspace"])
    parser.add_argument("--background-projection", choices=["cubemap", "equirectangular", "darkspace"])
    parser.add_argument("--cubemap-face-size", type=int)
    parser.add_argument("--hdri-path", type=str)
    parser.add_argument("--hdri-exposure", type=float)
    parser.add_argument("--hdri-rotation-deg", type=float)
    parser.add_argument("--wormhole-remote-hdri-path", type=str)
    parser.add_argument("--wormhole-remote-hdri-exposure", type=float)
    parser.add_argument("--wormhole-remote-hdri-rotation-deg", type=float)
    parser.add_argument("--enable-wormhole-remote-cubemap-coherent", action="store_true")
    parser.add_argument("--disable-wormhole-remote-cubemap-coherent", action="store_true")
    parser.add_argument("--wormhole-background-blend-width", type=float)
    parser.add_argument("--enable-wormhole-background-continuous-blend", action="store_true")
    parser.add_argument("--disable-wormhole-background-continuous-blend", action="store_true")
    parser.add_argument("--enable-wormhole-mt-force-reference-trace", action="store_true")
    parser.add_argument("--disable-wormhole-mt-force-reference-trace", action="store_true")
    parser.add_argument("--enable-wormhole-mt-unwrap-phi", action="store_true")
    parser.add_argument("--disable-wormhole-mt-unwrap-phi", action="store_true")
    parser.add_argument("--enable-wormhole-mt-shortest-arc-phi", action="store_true")
    parser.add_argument("--disable-wormhole-mt-shortest-arc-phi", action="store_true")
    parser.add_argument("--enable-wormhole-mt-sky-from-xyz", action="store_true")
    parser.add_argument("--disable-wormhole-mt-sky-from-xyz", action="store_true")
    parser.add_argument("--background-meridian-offset-deg", type=float)
    parser.add_argument("--disable-star-background", action="store_true")
    parser.add_argument("--disable-meridian-supersample", action="store_true")
    parser.add_argument("--enable-meridian-destripe", action="store_true")
    parser.add_argument("--disable-meridian-destripe", action="store_true")
    parser.add_argument("--disable-physical-disk-model", action="store_true")
    parser.add_argument("--disk-temperature-inner", type=float)
    parser.add_argument("--disk-color-correction", type=float)
    parser.add_argument("--disk-plasma-warmth", type=float)
    parser.add_argument("--disk-palette", choices=["default", "interstellar_warm"])
    parser.add_argument("--enable-disk-layered-palette", action="store_true")
    parser.add_argument("--disable-disk-layered-palette", action="store_true")
    parser.add_argument("--disk-layer-count", type=int)
    parser.add_argument("--disk-layer-mix", type=float)
    parser.add_argument("--disk-layer-pattern-count", type=float)
    parser.add_argument("--disk-layer-pattern-contrast", type=float)
    parser.add_argument("--disk-layer-time-scale", type=float)
    parser.add_argument("--disk-layer-global-phase", type=float)
    parser.add_argument("--disk-layer-phase-rate-hz", type=float)
    parser.add_argument("--disk-layer-accident-strength", type=float)
    parser.add_argument("--disk-layer-accident-count", type=float)
    parser.add_argument("--disk-layer-accident-sharpness", type=float)
    parser.add_argument("--enable-disk-differential-rotation", action="store_true")
    parser.add_argument("--disable-disk-differential-rotation", action="store_true")
    parser.add_argument("--disk-diffrot-model", choices=["keplerian_lut", "keplerian_metric"])
    parser.add_argument("--disk-diffrot-visual-mode", choices=["layer_phase", "annular_tiles", "hybrid"])
    parser.add_argument("--disk-diffrot-strength", type=float)
    parser.add_argument("--disk-diffrot-seed", type=int)
    parser.add_argument("--disk-diffrot-iteration", choices=["v1_basic", "v2_visibility", "v3_robust"])
    parser.add_argument("--enable-adaptive-disk-stratification", action="store_true")
    parser.add_argument("--disable-adaptive-disk-stratification", action="store_true")
    parser.add_argument("--disk-adaptive-layers-min", type=int)
    parser.add_argument("--disk-adaptive-layers-max", type=int)
    parser.add_argument("--disk-adaptive-complexity-mix", type=float)
    parser.add_argument("--disk-emission-gain", type=float)
    parser.add_argument("--disk-structure-mode", choices=["continuous", "concentric_annuli"])
    parser.add_argument("--disk-annuli-count", type=int)
    parser.add_argument("--disk-annuli-blend", type=float)
    parser.add_argument("--thick-disk", action="store_true")
    parser.add_argument("--thin-disk", action="store_true")
    parser.add_argument("--disk-thickness-ratio", type=float)
    parser.add_argument("--disk-thickness-power", type=float)
    parser.add_argument("--disk-vertical-softness", type=float)
    parser.add_argument("--enable-disk-volume-emission", action="store_true")
    parser.add_argument("--disable-disk-volume-emission", action="store_true")
    parser.add_argument("--disk-volume-samples", type=int)
    parser.add_argument("--disk-volume-density-scale", type=float)
    parser.add_argument("--disk-volume-temperature-drop", type=float)
    parser.add_argument("--disk-volume-strength", type=float)
    parser.add_argument("--vertical-transition-mode", choices=["snap", "continuous"])
    parser.add_argument("--disable-black-hole-shadow", action="store_true")
    parser.add_argument("--shadow-absorb-radius-factor", type=float)
    parser.add_argument("--disable-adaptive-integrator", action="store_true")
    parser.add_argument("--adaptive-rtol", type=float)
    parser.add_argument("--adaptive-atol", type=float)
    parser.add_argument("--adaptive-step-min", type=float)
    parser.add_argument("--adaptive-step-max", type=float)
    parser.add_argument("--disable-adaptive-event-aware", action="store_true")
    parser.add_argument("--enable-adaptive-event-aware", action="store_true")
    parser.add_argument("--disable-adaptive-fallback-rk4", action="store_true")
    parser.add_argument("--adaptive-fallback-substeps", type=int)
    parser.add_argument("--kerr-schild-mode", choices=["off", "fsal_only", "analytic"])
    parser.add_argument(
        "--disable-kerr-schild-improvements",
        action="store_true",
        help="Disable optional Kerr-Schild analytic RHS + FSAL optimizations",
    )
    parser.add_argument(
        "--enable-kerr-schild-null-diagnostic",
        action="store_true",
        help="Print periodic null-norm diagnostics in Kerr-Schild tracing",
    )
    parser.add_argument("--kerr-schild-null-interval", type=int)
    parser.add_argument("--kerr-schild-null-tol", type=float)
    parser.add_argument("--compile-rhs", action="store_true")
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--mps-optimized-kernel", action="store_true", help="Enable fast MPS-oriented tracing path")
    parser.add_argument("--disable-mps-auto-chunking", action="store_true")
    parser.add_argument("--enable-mps-auto-chunking", action="store_true")
    parser.add_argument("--disable-mps-emitter-fastpath", action="store_true")
    parser.add_argument("--enable-temporal-reprojection", action="store_true")
    parser.add_argument("--disable-temporal-reprojection", action="store_true")
    parser.add_argument("--temporal-denoise-mode", choices=["basic", "robust"])
    parser.add_argument("--temporal-blend", type=float)
    parser.add_argument("--temporal-clamp", type=float)
    parser.add_argument("--motion-vector-scale", type=float)
    parser.add_argument("--temporal-denoise-radius", type=int)
    parser.add_argument("--temporal-denoise-sigma", type=float)
    parser.add_argument("--temporal-denoise-clip", type=float)
    parser.add_argument("--disk-beaming-strength", type=float)
    parser.add_argument("--disk-self-occlusion-strength", type=float)
    parser.add_argument("--disable-multi-hit-disk", action="store_true")
    parser.add_argument("--enable-multi-hit-disk", action="store_true")
    parser.add_argument("--max-disk-crossings", type=int)
    parser.add_argument("--lensing-order-strength", type=float)
    parser.add_argument("--lensing-order-gamma", type=float)
    parser.add_argument("--enable-emitter-polarization", action="store_true")
    parser.add_argument("--magnetic-field-strength", type=float)
    parser.add_argument("--faraday-rotation-strength", type=float)
    parser.add_argument("--tone-mapper", choices=["reinhard", "aces", "filmic"])
    parser.add_argument("--tone-exposure", type=float)
    parser.add_argument("--tone-white-point", type=float)
    parser.add_argument("--tone-highlight-rolloff", type=float)
    parser.add_argument("--postprocess-pipeline", choices=["off", "gargantua"])
    parser.add_argument("--gargantua-look-strength", type=float)
    parser.add_argument(
        "--enable-gargantua-look",
        action="store_true",
        help="Enable Gargantua-inspired visual preset (thin disk + stronger beaming + filmic/postprocess)",
    )
    parser.add_argument(
        "--disable-gargantua-look",
        action="store_true",
        help="Disable Gargantua-inspired visual preset",
    )
    parser.add_argument("--video-codec", choices=["h264", "h265_10bit"])
    parser.add_argument("--video-crf", type=int)
    parser.add_argument("--render-tile-rows", type=int, help="Render in row tiles (0 disables tiling)")
    parser.add_argument(
        "--disable-cuda-graph-finalize",
        action="store_true",
        help="Disable CUDA graph capture for the final image postprocess path",
    )
    parser.add_argument(
        "--enable-cuda-graph-finalize",
        action="store_true",
        help="Enable CUDA graph capture for the final image postprocess path",
    )
    parser.add_argument("--disable-camera-fastpath", action="store_true", help="Use legacy camera-ray initialization path")
    parser.add_argument("--enable-camera-fastpath", action="store_true", help="Use optimized camera-ray initialization path")
    parser.add_argument(
        "--enable-atlas-cartesian-variant",
        action="store_true",
        help=(
            "Enable atlas/cartesian camera variant in Kerr-Schild-family coordinates "
            "(preserves pole azimuth continuity; useful for axis sweeps)"
        ),
    )
    parser.add_argument(
        "--disable-atlas-cartesian-variant",
        action="store_true",
        help="Disable atlas/cartesian camera variant and use legacy BL-angle regularization",
    )
    parser.add_argument(
        "--enable-roi-supersampling",
        action="store_true",
        help="Enable selective ROI supersampling around high-gradient/ring regions",
    )
    parser.add_argument(
        "--disable-roi-supersampling",
        action="store_true",
        help="Disable selective ROI supersampling",
    )
    parser.add_argument(
        "--roi-supersample-threshold",
        type=float,
        help="Gradient quantile used to detect supersampling ROI [0.50, 0.999]",
    )
    parser.add_argument(
        "--roi-supersample-jitter",
        type=float,
        help="Subpixel jitter amplitude for ROI supersampling (0, 1]",
    )
    parser.add_argument(
        "--roi-supersample-samples",
        type=int,
        help="Number of extra ROI supersample passes [1, 8]",
    )
    parser.add_argument("--disable-persistent-cache", action="store_true", help="Disable persistent disk cache for LUT/cubemap")
    parser.add_argument("--persistent-cache-dir", type=str, help="Persistent disk cache directory (default: out/cache)")
    parser.add_argument(
        "--enable-adaptive-spatial-sampling",
        action="store_true",
        help="Enable adaptive per-row spatial step scheduling",
    )
    parser.add_argument(
        "--disable-adaptive-spatial-sampling",
        action="store_true",
        help="Disable adaptive per-row spatial step scheduling",
    )
    parser.add_argument(
        "--adaptive-spatial-preview-steps",
        type=int,
        help="Low-cost preview max_steps used to estimate row complexity",
    )
    parser.add_argument(
        "--adaptive-spatial-min-scale",
        type=float,
        help="Minimum per-row max_steps scale used by adaptive spatial sampling (0,1]",
    )
    parser.add_argument(
        "--adaptive-spatial-quantile",
        type=float,
        help="Gradient quantile used to normalize adaptive spatial complexity [0.5, 0.995]",
    )
    parser.add_argument("--disable-progress-bar", action="store_true", help="Disable terminal row progress bar")
    parser.add_argument("--enable-progress-bar", action="store_true", help="Enable terminal row progress bar")
    parser.add_argument(
        "--progress-backend",
        choices=["manual", "tqdm", "auto"],
        help="Progress renderer backend (manual custom bar or tqdm)",
    )
    parser.add_argument("--animation-workers", type=int, help="Parallel frame workers for CPU animation")
    parser.add_argument("--enable-quality-lock", action="store_true", help="Compare optimized KS mode vs baseline before final render")
    parser.add_argument("--disable-quality-lock-fallback", action="store_true", help="Do not fallback to baseline if quality lock fails")
    parser.add_argument("--quality-lock-psnr-min", type=float)
    parser.add_argument("--quality-lock-ssim-min", type=float)
    parser.add_argument("--quality-lock-sample-width", type=int)
    parser.add_argument("--quality-lock-sample-height", type=int)
    parser.add_argument("--step-size", type=float)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--escape-radius", type=float)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--require-gpu", action="store_true", help="Fail if resolved device is CPU")
    parser.add_argument("--dtype", choices=["float32", "float64"])
    parser.add_argument("--output", type=str)
    parser.add_argument("--animate", action="store_true", help="Render animation and encode to video/GIF")
    parser.add_argument("--animate-charged-particles", action="store_true", help="Render charged massive particle orbits around KNdS")
    parser.add_argument(
        "--animate-raytraced-particle",
        action="store_true",
        help="Render a single off-equatorial charged particle over a raytraced (lightlike) accretion-disk frame",
    )
    parser.add_argument("--frames", type=int, default=120, help="Animation frame count")
    parser.add_argument("--fps", type=int, default=30, help="Animation frames per second")
    parser.add_argument("--azimuth-orbits", type=float, default=1.0, help="Observer azimuth rotations across animation")
    parser.add_argument("--inclination-wobble-deg", type=float, default=0.0, help="Sinusoidal camera inclination wobble")
    parser.add_argument("--inclination-start-deg", type=float, help="Animation start camera inclination")
    parser.add_argument("--inclination-end-deg", type=float, help="Animation end camera inclination")
    parser.add_argument("--observer-radius-start", type=float, help="Animation start observer radius")
    parser.add_argument("--observer-radius-end", type=float, help="Animation end observer radius")
    parser.add_argument(
        "--generalized-doran-fixed-time",
        action="store_true",
        help=(
            "For generalized_doran sweeps, sample frames at uniform coordinate-time steps by numerically inverting "
            "the PG-like relation t(r) instead of using linear radius interpolation"
        ),
    )
    parser.add_argument(
        "--generalized-doran-fixed-proper-time",
        action="store_true",
        help=(
            "For generalized_doran sweeps, sample frames at uniform observer proper-time proxy steps "
            "by numerically inverting tau(r) instead of using linear radius interpolation"
        ),
    )
    parser.add_argument(
        "--generalized-doran-time-samples",
        type=int,
        default=2048,
        help="Integration samples used to build and invert t(r) for generalized_doran fixed-time sweeps",
    )
    parser.add_argument(
        "--generalized-doran-radius-log",
        type=str,
        help="Optional CSV output path for per-frame generalized_doran fixed-time schedule (frame,time,radius)",
    )
    parser.add_argument("--linear-inclination-sweep", action="store_true", help="Use linear (non-eased) inclination sweep")
    parser.add_argument("--taa-samples", type=int, default=1, help="Temporal AA samples per output frame")
    parser.add_argument("--shutter-fraction", type=float, default=0.85, help="Shutter duration fraction for motion blur [0..1]")
    parser.add_argument("--spatial-jitter", action="store_true", help="Enable spatial jitter for anti-aliasing")
    parser.add_argument(
        "--disable-adaptive-frame-steps",
        action="store_true",
        help="Disable adaptive per-frame max_steps scaling in animation",
    )
    parser.add_argument(
        "--adaptive-frame-steps-min-scale",
        type=float,
        default=0.60,
        help="Minimum max_steps scale used by adaptive frame steps (0,1]",
    )
    parser.add_argument(
        "--disable-stream-encode",
        action="store_true",
        help="Disable ffmpeg stream encoding and always encode from PNG frames",
    )
    parser.add_argument(
        "--enable-stream-encode",
        action="store_true",
        help="Enable ffmpeg stream encoding when compatible with requested output mode",
    )
    parser.add_argument(
        "--disable-stream-encode-async",
        action="store_true",
        help="Disable async ffmpeg writer thread and write frames synchronously",
    )
    parser.add_argument(
        "--enable-stream-encode-async",
        action="store_true",
        help="Enable async ffmpeg writer thread during stream encoding",
    )
    parser.add_argument(
        "--stream-encode-queue-size",
        type=int,
        help="Buffered frame queue length for async stream encoding [1, 64]",
    )
    parser.add_argument("--frames-dir", type=str, help="Directory for frame PNG sequence")
    parser.add_argument("--keep-frames", action="store_true", help="Keep frame PNG files after encoding")
    parser.add_argument("--resume-frames", action="store_true", help="Reuse existing frame_XXXXX.png files and render only missing ones")
    parser.add_argument("--render-frames-only", action="store_true", help="Render PNG frame sequence and skip encoding")
    parser.add_argument("--encode-frames-only", action="store_true", help="Encode existing PNG frame sequence without rendering")
    parser.add_argument("--particle-count", type=int, default=240, help="Number of massive particles in charged-particle animation")
    parser.add_argument("--particle-specific-charge", type=float, default=0.4, help="Absolute specific charge |q/m| for particles")
    parser.add_argument("--particle-speed", type=float, default=0.42, help="Initial azimuthal speed in local ZAMO frame")
    parser.add_argument("--particle-radius-min", type=float, default=8.0, help="Minimum initial radius for particles")
    parser.add_argument("--particle-radius-max", type=float, default=22.0, help="Maximum initial radius for particles")
    parser.add_argument("--particle-substeps", type=int, default=6, help="Integrator substeps per frame for particles")
    parser.add_argument("--particle-dt", type=float, default=0.03, help="Affine-step size for particle integrator")
    parser.add_argument("--particle-seed", type=int, default=42, help="Random seed for particle initialization")
    parser.add_argument("--particle-camera-radius", type=float, default=55.0, help="Camera radius for particle animation")
    parser.add_argument("--particle-fov-deg", type=float, default=40.0, help="FOV for particle animation")
    parser.add_argument("--single-particle-theta-deg", type=float, default=62.0, help="Initial polar angle of the single particle")
    parser.add_argument("--single-particle-phi-deg", type=float, default=20.0, help="Initial azimuth of the single particle")
    parser.add_argument("--single-particle-radius", type=float, default=11.0, help="Initial radius of the single particle")
    parser.add_argument("--single-particle-specific-charge", type=float, default=-0.45, help="Specific charge q/m of the single particle")
    parser.add_argument("--single-particle-vphi", type=float, default=0.46, help="Initial local azimuthal speed of the single particle")
    parser.add_argument("--single-particle-vtheta", type=float, default=0.09, help="Initial local polar speed of the single particle")
    parser.add_argument("--single-particle-vr", type=float, default=0.0, help="Initial local radial speed of the single particle")
    parser.add_argument("--single-particle-trail-length", type=int, default=40, help="Trail length in frames for the single particle")

    parser.add_argument("--dump-config", type=str, help="Write resolved config to JSON and exit")
    parser.add_argument("--diagnose-device", action="store_true", help="Print GPU backend diagnostics and exit")
    return parser.parse_args()


def _merge_cli_config(base: RenderConfig, args: argparse.Namespace) -> RenderConfig:
    mapping = {
        "width": args.width,
        "height": args.height,
        "fov_deg": args.fov_deg,
        "coordinate_system": args.coordinate_system,
        "metric_model": args.metric_model,
        "spin": args.spin,
        "charge": args.charge,
        "cosmological_constant": args.cosmological_constant,
        "wormhole_throat_radius": args.wormhole_throat_radius,
        "wormhole_length_scale": args.wormhole_length_scale,
        "wormhole_allow_throat_crossing": None,
        "observer_radius": args.observer_radius,
        "observer_inclination_deg": args.observer_inclination_deg,
        "observer_azimuth_deg": args.observer_azimuth_deg,
        "observer_roll_deg": args.observer_roll_deg,
        "disk_inner_radius": args.disk_inner_radius,
        "disk_outer_radius": args.disk_outer_radius,
        "enable_accretion_disk": None,
        "disk_model": args.disk_model,
        "disk_radial_profile": args.disk_radial_profile,
        "emissivity_index": args.emissivity_index,
        "inner_edge_boost": args.inner_edge_boost,
        "outer_edge_boost": args.outer_edge_boost,
        "star_density": args.star_density,
        "star_brightness": args.star_brightness,
        "star_seed": args.star_seed,
        "background_mode": args.background_mode,
        "background_projection": args.background_projection,
        "cubemap_face_size": args.cubemap_face_size,
        "hdri_path": args.hdri_path,
        "hdri_exposure": args.hdri_exposure,
        "hdri_rotation_deg": args.hdri_rotation_deg,
        "wormhole_remote_hdri_path": args.wormhole_remote_hdri_path,
        "wormhole_remote_hdri_exposure": args.wormhole_remote_hdri_exposure,
        "wormhole_remote_hdri_rotation_deg": args.wormhole_remote_hdri_rotation_deg,
        "wormhole_remote_cubemap_coherent": None,
        "wormhole_background_blend_width": args.wormhole_background_blend_width,
        "wormhole_mt_force_reference_trace": None,
        "wormhole_mt_unwrap_phi": None,
        "wormhole_mt_shortest_arc_phi_interp": None,
        "wormhole_mt_sky_sample_from_xyz": None,
        "background_meridian_offset_deg": args.background_meridian_offset_deg,
        "disk_temperature_inner": args.disk_temperature_inner,
        "disk_color_correction": args.disk_color_correction,
        "disk_plasma_warmth": args.disk_plasma_warmth,
        "disk_palette": args.disk_palette,
        "disk_layer_count": args.disk_layer_count,
        "disk_layer_mix": args.disk_layer_mix,
        "disk_layer_pattern_count": args.disk_layer_pattern_count,
        "disk_layer_pattern_contrast": args.disk_layer_pattern_contrast,
        "disk_layer_time_scale": args.disk_layer_time_scale,
        "disk_layer_global_phase": args.disk_layer_global_phase,
        "disk_layer_phase_rate_hz": args.disk_layer_phase_rate_hz,
        "disk_layer_accident_strength": args.disk_layer_accident_strength,
        "disk_layer_accident_count": args.disk_layer_accident_count,
        "disk_layer_accident_sharpness": args.disk_layer_accident_sharpness,
        "disk_diffrot_model": args.disk_diffrot_model,
        "disk_diffrot_visual_mode": args.disk_diffrot_visual_mode,
        "disk_diffrot_strength": args.disk_diffrot_strength,
        "disk_diffrot_seed": args.disk_diffrot_seed,
        "disk_diffrot_iteration": args.disk_diffrot_iteration,
        "disk_adaptive_layers_min": args.disk_adaptive_layers_min,
        "disk_adaptive_layers_max": args.disk_adaptive_layers_max,
        "disk_adaptive_complexity_mix": args.disk_adaptive_complexity_mix,
        "disk_emission_gain": args.disk_emission_gain,
        "disk_structure_mode": args.disk_structure_mode,
        "disk_annuli_count": args.disk_annuli_count,
        "disk_annuli_blend": args.disk_annuli_blend,
        "disk_thickness_ratio": args.disk_thickness_ratio,
        "disk_thickness_power": args.disk_thickness_power,
        "disk_vertical_softness": args.disk_vertical_softness,
        "disk_volume_samples": args.disk_volume_samples,
        "disk_volume_density_scale": args.disk_volume_density_scale,
        "disk_volume_temperature_drop": args.disk_volume_temperature_drop,
        "disk_volume_strength": args.disk_volume_strength,
        "vertical_transition_mode": args.vertical_transition_mode,
        "shadow_absorb_radius_factor": args.shadow_absorb_radius_factor,
        "adaptive_rtol": args.adaptive_rtol,
        "adaptive_atol": args.adaptive_atol,
        "adaptive_step_min": args.adaptive_step_min,
        "adaptive_step_max": args.adaptive_step_max,
        "adaptive_fallback_substeps": args.adaptive_fallback_substeps,
        "kerr_schild_mode": args.kerr_schild_mode,
        "kerr_schild_null_norm_interval": args.kerr_schild_null_interval,
        "kerr_schild_null_norm_tol": args.kerr_schild_null_tol,
        "temporal_denoise_mode": args.temporal_denoise_mode,
        "temporal_blend": args.temporal_blend,
        "temporal_clamp": args.temporal_clamp,
        "motion_vector_scale": args.motion_vector_scale,
        "temporal_denoise_radius": args.temporal_denoise_radius,
        "temporal_denoise_sigma": args.temporal_denoise_sigma,
        "temporal_denoise_clip": args.temporal_denoise_clip,
        "disk_beaming_strength": args.disk_beaming_strength,
        "disk_self_occlusion_strength": args.disk_self_occlusion_strength,
        "max_disk_crossings": args.max_disk_crossings,
        "lensing_order_strength": args.lensing_order_strength,
        "lensing_order_gamma": args.lensing_order_gamma,
        "magnetic_field_strength": args.magnetic_field_strength,
        "faraday_rotation_strength": args.faraday_rotation_strength,
        "tone_mapper": args.tone_mapper,
        "tone_exposure": args.tone_exposure,
        "tone_white_point": args.tone_white_point,
        "tone_highlight_rolloff": args.tone_highlight_rolloff,
        "postprocess_pipeline": args.postprocess_pipeline,
        "gargantua_look_strength": args.gargantua_look_strength,
        "gargantua_look_preset": None,
        "video_codec": args.video_codec,
        "video_crf": args.video_crf,
        "render_tile_rows": args.render_tile_rows,
        "roi_supersample_threshold": args.roi_supersample_threshold,
        "roi_supersample_jitter": args.roi_supersample_jitter,
        "roi_supersample_samples": args.roi_supersample_samples,
        "persistent_cache_dir": args.persistent_cache_dir,
        "adaptive_spatial_preview_steps": args.adaptive_spatial_preview_steps,
        "adaptive_spatial_min_scale": args.adaptive_spatial_min_scale,
        "adaptive_spatial_quantile": args.adaptive_spatial_quantile,
        "progress_backend": args.progress_backend,
        "animation_workers": args.animation_workers,
        "stream_encode_queue_size": args.stream_encode_queue_size,
        "quality_lock_psnr_min": args.quality_lock_psnr_min,
        "quality_lock_ssim_min": args.quality_lock_ssim_min,
        "quality_lock_sample_width": args.quality_lock_sample_width,
        "quality_lock_sample_height": args.quality_lock_sample_height,
        "step_size": args.step_size,
        "max_steps": args.max_steps,
        "escape_radius": args.escape_radius,
        "device": args.device,
        "dtype": args.dtype,
        "output": args.output,
    }
    updates = {key: value for key, value in mapping.items() if value is not None}
    # Keep disk_inner_radius tied to ISCO by default. Only an explicit CLI value
    # should lock it to a fixed radius.
    if args.disk_inner_radius is None:
        updates["disk_inner_radius"] = None
    if args.disable_star_background:
        updates["enable_star_background"] = False
    if args.enable_wormhole_throat_crossing:
        updates["wormhole_allow_throat_crossing"] = True
    if args.disable_wormhole_throat_crossing:
        updates["wormhole_allow_throat_crossing"] = False
    if args.enable_wormhole_remote_cubemap_coherent:
        updates["wormhole_remote_cubemap_coherent"] = True
    if args.disable_wormhole_remote_cubemap_coherent:
        updates["wormhole_remote_cubemap_coherent"] = False
    if args.enable_wormhole_background_continuous_blend:
        updates["wormhole_background_continuous_blend"] = True
    if args.disable_wormhole_background_continuous_blend:
        updates["wormhole_background_continuous_blend"] = False
    if args.enable_wormhole_mt_force_reference_trace:
        updates["wormhole_mt_force_reference_trace"] = True
    if args.disable_wormhole_mt_force_reference_trace:
        updates["wormhole_mt_force_reference_trace"] = False
    if args.enable_wormhole_mt_unwrap_phi:
        updates["wormhole_mt_unwrap_phi"] = True
    if args.disable_wormhole_mt_unwrap_phi:
        updates["wormhole_mt_unwrap_phi"] = False
    if args.enable_wormhole_mt_shortest_arc_phi:
        updates["wormhole_mt_shortest_arc_phi_interp"] = True
    if args.disable_wormhole_mt_shortest_arc_phi:
        updates["wormhole_mt_shortest_arc_phi_interp"] = False
    if args.enable_wormhole_mt_sky_from_xyz:
        updates["wormhole_mt_sky_sample_from_xyz"] = True
    if args.disable_wormhole_mt_sky_from_xyz:
        updates["wormhole_mt_sky_sample_from_xyz"] = False
    if args.disable_accretion_disk:
        updates["enable_accretion_disk"] = False
    if args.enable_accretion_disk:
        updates["enable_accretion_disk"] = True
    if args.disable_meridian_supersample:
        updates["meridian_supersample"] = False
    if args.enable_meridian_destripe:
        updates["destripe_meridian"] = True
    if args.disable_meridian_destripe:
        updates["destripe_meridian"] = False
    if args.disable_physical_disk_model:
        updates["physical_disk_model"] = False
        updates["disk_model"] = "legacy"
    if args.disk_model is not None:
        updates["physical_disk_model"] = args.disk_model == "physical_nt"
    if args.thick_disk:
        updates["thick_disk"] = True
    if args.thin_disk:
        updates["thick_disk"] = False
    if args.enable_disk_layered_palette:
        updates["disk_layered_palette"] = True
    if args.disable_disk_layered_palette:
        updates["disk_layered_palette"] = False
    if args.enable_disk_differential_rotation:
        updates["enable_disk_differential_rotation"] = True
    if args.disable_disk_differential_rotation:
        updates["enable_disk_differential_rotation"] = False
    if args.enable_adaptive_disk_stratification:
        updates["disk_adaptive_stratification"] = True
    if args.disable_adaptive_disk_stratification:
        updates["disk_adaptive_stratification"] = False
    if args.enable_disk_volume_emission:
        updates["disk_volume_emission"] = True
    if args.disable_disk_volume_emission:
        updates["disk_volume_emission"] = False
    if args.disable_black_hole_shadow:
        updates["enforce_black_hole_shadow"] = False
    if args.disable_adaptive_integrator:
        updates["adaptive_integrator"] = False
    if args.disable_adaptive_event_aware:
        updates["adaptive_event_aware"] = False
    if args.enable_adaptive_event_aware:
        updates["adaptive_event_aware"] = True
    if args.disable_adaptive_fallback_rk4:
        updates["adaptive_fallback_rk4"] = False
    if args.disable_kerr_schild_improvements:
        updates["kerr_schild_improvements"] = False
        updates["kerr_schild_mode"] = "off"
    if args.enable_kerr_schild_null_diagnostic:
        updates["kerr_schild_null_norm_diagnostic"] = True
    if args.enable_quality_lock:
        updates["quality_lock"] = True
    if args.disable_quality_lock_fallback:
        updates["quality_lock_fallback_to_baseline"] = False
    if args.compile_rhs:
        updates["compile_rhs"] = True
    if args.mixed_precision:
        updates["mixed_precision"] = True
    if args.mps_optimized_kernel:
        updates["mps_optimized_kernel"] = True
    if args.disable_mps_auto_chunking:
        updates["mps_auto_chunking"] = False
    if args.enable_mps_auto_chunking:
        updates["mps_auto_chunking"] = True
    if args.disable_mps_emitter_fastpath:
        updates["allow_mps_emitter_fastpath"] = False
    if args.disable_camera_fastpath:
        updates["camera_fastpath"] = False
    if args.enable_camera_fastpath:
        updates["camera_fastpath"] = True
    if args.enable_atlas_cartesian_variant:
        updates["atlas_cartesian_variant"] = True
    if args.disable_atlas_cartesian_variant:
        updates["atlas_cartesian_variant"] = False
    if args.disable_cuda_graph_finalize:
        updates["cuda_graph_finalize"] = False
    if args.enable_cuda_graph_finalize:
        updates["cuda_graph_finalize"] = True
    if args.enable_roi_supersampling:
        updates["roi_supersampling"] = True
    if args.disable_roi_supersampling:
        updates["roi_supersampling"] = False
    if args.disable_persistent_cache:
        updates["persistent_cache_enabled"] = False
    if args.enable_adaptive_spatial_sampling:
        updates["adaptive_spatial_sampling"] = True
    if args.disable_adaptive_spatial_sampling:
        updates["adaptive_spatial_sampling"] = False
    if args.enable_temporal_reprojection:
        updates["temporal_reprojection"] = True
    if args.disable_temporal_reprojection:
        updates["temporal_reprojection"] = False
    if args.enable_emitter_polarization:
        updates["enable_emitter_polarization"] = True
    if args.enable_gargantua_look:
        updates["gargantua_look_preset"] = True
    if args.disable_gargantua_look:
        updates["gargantua_look_preset"] = False
    if args.disable_multi_hit_disk:
        updates["multi_hit_disk"] = False
    if args.enable_multi_hit_disk:
        updates["multi_hit_disk"] = True
    if args.disable_progress_bar:
        updates["show_progress_bar"] = False
    if args.enable_progress_bar:
        updates["show_progress_bar"] = True
    if args.disable_stream_encode_async:
        updates["stream_encode_async"] = False
    if args.enable_stream_encode_async:
        updates["stream_encode_async"] = True
    return replace(base, **updates)


def _print_device_diagnostics(config: RenderConfig) -> None:
    sandbox_mode = os.environ.get("CODEX_SANDBOX")
    print("Device diagnostics")
    print(f"Platform: {platform.platform()}")
    print(f"Machine: {platform.machine()}")
    print(f"Python: {platform.python_version()}")
    print(f"Torch: {torch.__version__}")
    print(f"CODEX_SANDBOX: {sandbox_mode or 'not set'}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"MPS built: {torch.backends.mps.is_built()}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"MPS optimized kernel requested: {config.mps_optimized_kernel}")
    print(f"Metric model: {config.metric_model}")
    if torch.backends.mps.is_available():
        try:
            x = torch.ones((8, 8), device="mps")
            y = (x @ x).sum().item()
            print(f"MPS probe op: ok (sum={y:.1f})")
        except Exception as exc:
            print(f"MPS probe op: failed ({exc})")
    else:
        print("MPS probe op: skipped (backend unavailable)")
    try:
        print(f"Resolved device: {config.resolve_device()}")
    except Exception as exc:
        print(f"Resolved device error: {exc}")

    if torch.backends.mps.is_built() and not torch.backends.mps.is_available():
        print("MPS is built but unavailable. On macOS, check:")
        print("1) native arm64 terminal (not Rosetta)")
        print("2) current Python/venv is arm64")
        print("3) latest stable PyTorch in this venv")
        print("4) macOS up to date")
        if sandbox_mode:
            print("5) run outside sandbox (CODEX_SANDBOX is active in this session)")


def _compute_psnr_ssim(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    aa = np.clip(a.astype(np.float32) / 255.0, 0.0, 1.0)
    bb = np.clip(b.astype(np.float32) / 255.0, 0.0, 1.0)
    diff = aa - bb
    mse = float(np.mean(diff * diff))
    if mse <= 1.0e-12:
        psnr = 120.0
    else:
        psnr = 10.0 * float(np.log10(1.0 / mse))

    mu_a = np.mean(aa, axis=(0, 1), keepdims=True)
    mu_b = np.mean(bb, axis=(0, 1), keepdims=True)
    sigma_a = np.mean((aa - mu_a) ** 2, axis=(0, 1), keepdims=True)
    sigma_b = np.mean((bb - mu_b) ** 2, axis=(0, 1), keepdims=True)
    sigma_ab = np.mean((aa - mu_a) * (bb - mu_b), axis=(0, 1), keepdims=True)
    c1 = (0.01 ** 2)
    c2 = (0.03 ** 2)
    num = (2.0 * mu_a * mu_b + c1) * (2.0 * sigma_ab + c2)
    den = (mu_a * mu_a + mu_b * mu_b + c1) * (sigma_a + sigma_b + c2)
    ssim = float(np.mean(num / np.clip(den, 1.0e-8, None)))
    return psnr, ssim


def _apply_quality_lock_if_needed(config: RenderConfig) -> RenderConfig:
    if not config.quality_lock:
        return config
    if config.coordinate_system not in {"kerr_schild", "generalized_doran"}:
        return config
    if config.metric_model == "kerr_newman_de_sitter":
        print(
            "Quality-lock skipped for kerr_newman_de_sitter: baseline mode "
            "'kerr_schild_mode=off' is disallowed (GKS-only)."
        )
        return config
    if config.kerr_schild_mode == "off":
        return config

    sample_cfg = replace(
        config,
        width=config.quality_lock_sample_width,
        height=config.quality_lock_sample_height,
        meridian_supersample=False,
    ).validated()
    baseline_cfg = replace(
        sample_cfg,
        kerr_schild_mode="off",
        kerr_schild_improvements=False,
    ).validated()

    cand = KerrRayTracer(sample_cfg).render()
    base = KerrRayTracer(baseline_cfg).render()
    cand_img = np.asarray(cand.image.convert("RGB"), dtype=np.uint8)
    base_img = np.asarray(base.image.convert("RGB"), dtype=np.uint8)
    psnr, ssim = _compute_psnr_ssim(cand_img, base_img)

    passed = (psnr >= config.quality_lock_psnr_min) and (ssim >= config.quality_lock_ssim_min)
    print(
        "Quality-lock: mode={mode} vs baseline | PSNR={psnr:.2f} dB (min {pmin:.2f}) | SSIM={ssim:.5f} (min {smin:.5f})".format(
            mode=config.kerr_schild_mode,
            psnr=psnr,
            pmin=config.quality_lock_psnr_min,
            ssim=ssim,
            smin=config.quality_lock_ssim_min,
        )
    )
    if passed:
        return config
    if not config.quality_lock_fallback_to_baseline:
        return config
    print("Quality-lock fallback: switching final render to kerr_schild_mode=off")
    return replace(config, kerr_schild_mode="off", kerr_schild_improvements=False).validated()


def main() -> int:
    args = _parse_args()

    config = RenderConfig()
    if args.config:
        config = RenderConfig.from_json(args.config)

    config = _merge_cli_config(config, args).validated()
    if config.quality_lock:
        config = _apply_quality_lock_if_needed(config)

    resolved_output = _resolve_progressive_output(config.output)
    if resolved_output and resolved_output != config.output:
        config = replace(config, output=resolved_output)
        print(f"Auto output path: {config.output}")

    auto_video_mode = (
        (not args.animate)
        and (not args.animate_charged_particles)
        and (not args.animate_raytraced_particle)
        and _is_video_like_output(config.output)
    )
    if auto_video_mode:
        args.animate = True
        print("Auto mode: video output detected, enabling animation render.")

    if args.diagnose_device:
        _print_device_diagnostics(config)
        return 0

    if args.require_gpu:
        resolved = config.resolve_device()
        if resolved.type == "cpu":
            raise RuntimeError("GPU is required but the resolved device is CPU. Use --device mps/cuda and check --diagnose-device.")

    if args.dump_config:
        target = Path(args.dump_config)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
        print(f"Written config: {target}")
        return 0

    if args.animate:
        if not config.output:
            raise ValueError("output is required for animation (use .mp4, .mov, .mkv, or .gif)")
        if args.encode_frames_only and not args.frames_dir:
            raise ValueError("--encode-frames-only requires --frames-dir")
        if args.encode_frames_only and args.render_frames_only:
            raise ValueError("Cannot combine --encode-frames-only with --render-frames-only")
        stream_encode = True
        if args.disable_stream_encode:
            stream_encode = False
        if args.enable_stream_encode:
            stream_encode = True
        stats = render_animation(
            base_config=config,
            output_path=config.output,
            frames=args.frames,
            fps=args.fps,
            azimuth_orbits=args.azimuth_orbits,
            inclination_wobble_deg=args.inclination_wobble_deg,
            inclination_start_deg=args.inclination_start_deg,
            inclination_end_deg=args.inclination_end_deg,
            observer_radius_start=args.observer_radius_start,
            observer_radius_end=args.observer_radius_end,
            inclination_sweep_ease=not args.linear_inclination_sweep,
            taa_samples=args.taa_samples,
            shutter_fraction=args.shutter_fraction,
            spatial_jitter=args.spatial_jitter,
            generalized_doran_fixed_time=args.generalized_doran_fixed_time,
            generalized_doran_fixed_proper_time=args.generalized_doran_fixed_proper_time,
            generalized_doran_time_samples=args.generalized_doran_time_samples,
            generalized_doran_radius_log=args.generalized_doran_radius_log,
            frames_dir=args.frames_dir,
            keep_frames=args.keep_frames,
            resume_frames=args.resume_frames,
            workers=config.animation_workers,
            render_frames=not args.encode_frames_only,
            encode_output=not args.render_frames_only,
            adaptive_frame_steps=not args.disable_adaptive_frame_steps,
            adaptive_frame_steps_min_scale=args.adaptive_frame_steps_min_scale,
            stream_encode=stream_encode,
            stream_encode_async=config.stream_encode_async,
            stream_encode_queue_size=config.stream_encode_queue_size,
        )
        print(f"Saved animation: {stats.output_path}")
        print(f"Frames: {stats.frames} | FPS: {stats.fps} | Time: {stats.elapsed_seconds:.2f}s")
        return 0

    if args.animate_charged_particles:
        if not config.output:
            raise ValueError("output is required for charged-particle animation (use .mp4, .mov, or .mkv)")
        orbiter = ChargedParticleOrbiter(
            config=config,
            particle_count=args.particle_count,
            particle_charge=args.particle_specific_charge,
            particle_speed=args.particle_speed,
            particle_radius_min=args.particle_radius_min,
            particle_radius_max=args.particle_radius_max,
            seed=args.particle_seed,
        )
        stats = orbiter.render_animation(
            output_path=config.output,
            frames=args.frames,
            fps=args.fps,
            dt=args.particle_dt,
            substeps=args.particle_substeps,
            camera_radius=args.particle_camera_radius,
            fov_deg=args.particle_fov_deg,
        )
        print(f"Saved charged-particle animation: {stats.output_path}")
        print(
            "Frames: {frames} | FPS: {fps} | Particles: {particles} | Survivors: {surv} | Time: {sec:.2f}s".format(
                frames=stats.frames,
                fps=stats.fps,
                particles=stats.particles,
                surv=stats.survivors,
                sec=stats.elapsed_seconds,
            )
        )
        return 0

    if args.animate_raytraced_particle:
        if not config.output:
            raise ValueError("output is required for raytraced-particle animation (use .mp4, .mov, or .mkv)")
        orbiter = ChargedParticleOrbiter(
            config=config,
            particle_count=1,
            particle_charge=abs(args.single_particle_specific_charge),
            particle_speed=args.single_particle_vphi,
            particle_radius_min=args.single_particle_radius,
            particle_radius_max=args.single_particle_radius + 0.1,
            seed=args.particle_seed,
        )
        stats = orbiter.render_single_particle_over_raytrace(
            output_path=config.output,
            frames=args.frames,
            fps=args.fps,
            dt=args.particle_dt,
            substeps=args.particle_substeps,
            theta_deg=args.single_particle_theta_deg,
            phi_deg=args.single_particle_phi_deg,
            radius=args.single_particle_radius,
            specific_charge=args.single_particle_specific_charge,
            v_phi=args.single_particle_vphi,
            v_theta=args.single_particle_vtheta,
            v_r=args.single_particle_vr,
            trail_length=args.single_particle_trail_length,
        )
        print(f"Saved raytraced single-particle animation: {stats.output_path}")
        print(
            "Frames: {frames} | FPS: {fps} | Particle: {particles} | Survivors: {surv} | Time: {sec:.2f}s".format(
                frames=stats.frames,
                fps=stats.fps,
                particles=stats.particles,
                surv=stats.survivors,
                sec=stats.elapsed_seconds,
            )
        )
        return 0

    tracer = KerrRayTracer(config)
    t0 = time.perf_counter()
    stats = tracer.render_to_file(config.output)
    dt = time.perf_counter() - t0

    print(f"Saved image: {config.output}")
    print(f"Device: {tracer.device}")
    if tracer.use_mps_optimized_kernel:
        print("Kernel mode: mps_optimized")
    else:
        print("Kernel mode: standard")
    print(
        "Rays: {total} | Disk: {disk} | Horizon: {horizon} | Escaped: {escaped} | Steps: {steps} | Time: {sec:.2f}s".format(
            total=stats.total_rays,
            disk=stats.disk_hits,
            horizon=stats.horizon_hits,
            escaped=stats.escaped,
            steps=stats.steps_used,
            sec=dt,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
