from __future__ import annotations

from dataclasses import dataclass, fields, replace
from pathlib import Path
import json
import math
import os

import torch
from .geometry import (
    METRIC_MODELS,
    canonical_metric_model,
    effective_metric_parameters,
    event_horizon_radius,
    horizon_radii,
    isco_radius_general,
)


def isco_radius(spin: float) -> float:
    """Prograde ISCO radius in units of M for |a| <= 1."""
    a = max(-0.999, min(0.999, spin))
    z1 = 1.0 + (1.0 - a * a) ** (1.0 / 3.0) * ((1.0 + a) ** (1.0 / 3.0) + (1.0 - a) ** (1.0 / 3.0))
    z2 = math.sqrt(3.0 * a * a + z1 * z1)
    sign = 1.0 if a >= 0.0 else -1.0
    return 3.0 + z2 - sign * math.sqrt((3.0 - z1) * (3.0 + z1 + 2.0 * z2))


@dataclass(frozen=True)
class RenderConfig:
    width: int = 960
    height: int = 540
    fov_deg: float = 38.0
    coordinate_system: str = "boyer_lindquist"
    metric_model: str = "kerr"
    spin: float = 0.92
    charge: float = 0.0
    cosmological_constant: float = 0.0
    observer_radius: float = 50.0
    observer_inclination_deg: float = 70.0
    observer_azimuth_deg: float = 0.0
    observer_roll_deg: float = 0.0
    disk_inner_radius: float | None = None
    disk_outer_radius: float = 28.0
    emissivity_index: float = 2.3
    inner_edge_boost: float = 2.2
    outer_edge_boost: float = 0.35
    enable_star_background: bool = True
    star_density: float = 0.0018
    star_brightness: float = 2.4
    star_seed: int = 7
    background_mode: str = "procedural"
    background_projection: str = "cubemap"
    cubemap_face_size: int = 768
    hdri_path: str | None = None
    hdri_exposure: float = 1.0
    hdri_rotation_deg: float = 0.0
    background_meridian_offset_deg: float = 137.5
    meridian_supersample: bool = True
    destripe_meridian: bool = False
    physical_disk_model: bool = True
    disk_model: str = "physical_nt"
    disk_radial_profile: str = "nt_proxy"
    disk_temperature_inner: float = 18000.0
    disk_color_correction: float = 1.7
    disk_plasma_warmth: float = 0.38
    disk_palette: str = "default"
    disk_layered_palette: bool = False
    disk_layer_count: int = 12
    disk_layer_mix: float = 0.55
    disk_layer_pattern_count: float = 14.0
    disk_layer_pattern_contrast: float = 0.45
    disk_layer_time_scale: float = 1.0
    disk_layer_global_phase: float = 0.0
    disk_layer_phase_rate_hz: float = 0.35
    disk_layer_accident_strength: float = 0.42
    disk_layer_accident_count: float = 3.8
    disk_layer_accident_sharpness: float = 7.0
    disk_adaptive_stratification: bool = False
    disk_adaptive_layers_min: int = 8
    disk_adaptive_layers_max: int = 48
    disk_adaptive_complexity_mix: float = 0.65
    disk_emission_gain: float = 1.0
    disk_structure_mode: str = "continuous"
    disk_annuli_count: int = 48
    disk_annuli_blend: float = 1.0
    thick_disk: bool = False
    disk_thickness_ratio: float = 0.12
    disk_thickness_power: float = 0.0
    disk_vertical_softness: float = 0.30
    disk_volume_emission: bool = False
    disk_volume_samples: int = 5
    disk_volume_density_scale: float = 1.0
    disk_volume_temperature_drop: float = 0.28
    disk_volume_strength: float = 0.85
    vertical_transition_mode: str = "continuous"
    enforce_black_hole_shadow: bool = True
    shadow_absorb_radius_factor: float = 1.35
    adaptive_integrator: bool = True
    adaptive_event_aware: bool = True
    adaptive_rtol: float = 2.0e-4
    adaptive_atol: float = 1.0e-6
    adaptive_step_min: float = 0.03
    adaptive_step_max: float = 0.90
    adaptive_fallback_rk4: bool = True
    adaptive_fallback_substeps: int = 2
    kerr_schild_mode: str = "fsal_only"
    kerr_schild_improvements: bool = True
    kerr_schild_null_norm_diagnostic: bool = False
    kerr_schild_null_norm_interval: int = 50
    kerr_schild_null_norm_tol: float = 1.0e-3
    compile_rhs: bool = False
    mixed_precision: bool = False
    mps_optimized_kernel: bool = False
    mps_auto_chunking: bool = True
    allow_mps_emitter_fastpath: bool = True
    temporal_reprojection: bool = False
    temporal_denoise_mode: str = "basic"
    temporal_blend: float = 0.18
    temporal_clamp: float = 24.0
    motion_vector_scale: float = 1.0
    temporal_denoise_radius: int = 1
    temporal_denoise_sigma: float = 18.0
    temporal_denoise_clip: float = 9.0
    disk_beaming_strength: float = 0.45
    disk_self_occlusion_strength: float = 0.35
    multi_hit_disk: bool = True
    max_disk_crossings: int = 6
    lensing_order_strength: float = 0.24
    lensing_order_gamma: float = 0.75
    enable_emitter_polarization: bool = False
    magnetic_field_strength: float = 0.0
    faraday_rotation_strength: float = 0.6
    tone_mapper: str = "reinhard"
    tone_exposure: float = 1.0
    tone_white_point: float = 11.2
    tone_highlight_rolloff: float = 0.18
    postprocess_pipeline: str = "off"
    gargantua_look_strength: float = 0.0
    gargantua_look_preset: bool = False
    video_codec: str = "h264"
    video_crf: int = 18
    render_tile_rows: int = 0
    camera_fastpath: bool = True
    cuda_graph_finalize: bool = True
    adaptive_spatial_sampling: bool = False
    adaptive_spatial_preview_steps: int = 96
    adaptive_spatial_min_scale: float = 0.65
    adaptive_spatial_quantile: float = 0.78
    roi_supersampling: bool = False
    roi_supersample_threshold: float = 0.92
    roi_supersample_jitter: float = 0.35
    roi_supersample_samples: int = 2
    persistent_cache_enabled: bool = True
    persistent_cache_dir: str = "out/cache"
    show_progress_bar: bool = True
    progress_backend: str = "manual"
    animation_workers: int = 1
    stream_encode_async: bool = True
    stream_encode_queue_size: int = 4
    quality_lock: bool = False
    quality_lock_psnr_min: float = 45.0
    quality_lock_ssim_min: float = 0.985
    quality_lock_sample_width: int = 256
    quality_lock_sample_height: int = 144
    quality_lock_fallback_to_baseline: bool = True
    step_size: float = 0.22
    max_steps: int = 1300
    escape_radius: float = 180.0
    device: str = "auto"
    dtype: str = "float32"
    output: str = "render.png"

    @classmethod
    def from_json(cls, path: str | Path) -> "RenderConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        valid = {field.name for field in fields(cls)}
        unknown = [key for key in data if key not in valid]
        if unknown:
            raise ValueError(f"Unknown config keys: {', '.join(unknown)}")
        return cls(**data).with_defaults()

    def with_defaults(self) -> "RenderConfig":
        model = canonical_metric_model(self.metric_model)
        cfg = replace(self, metric_model=model)

        if cfg.disk_inner_radius is None:
            try:
                h = event_horizon_radius(cfg.spin, model, cfg.charge, cfg.cosmological_constant)
            except Exception:
                h = 2.0
            try:
                if model == "kerr":
                    rin = isco_radius(cfg.spin)
                else:
                    rin = isco_radius_general(
                        spin=cfg.spin,
                        metric_model=model,
                        charge=cfg.charge,
                        cosmological_constant=cfg.cosmological_constant,
                        prograde=True,
                    )
            except Exception:
                rin = max(6.0, 1.25 * h)
            cfg = replace(cfg, disk_inner_radius=rin)
        if (not cfg.kerr_schild_improvements) and cfg.kerr_schild_mode != "off":
            cfg = replace(cfg, kerr_schild_mode="off")
        # Backward compatibility with older configs/CLI that toggle physical_disk_model.
        if (not cfg.physical_disk_model) and cfg.disk_model == "physical_nt":
            cfg = replace(cfg, disk_model="legacy")
        if cfg.gargantua_look_preset:
            # Preset inspired by DNGR descriptions (James et al. 2015 / CERN Courier):
            # thin bright disk, stronger Doppler/intensity asymmetry, cinematic flare/rolloff.
            defaults = RenderConfig()
            updates: dict[str, object] = {}

            def _set_if_default(name: str, value: object) -> None:
                if getattr(cfg, name) == getattr(defaults, name):
                    updates[name] = value

            _set_if_default("physical_disk_model", True)
            _set_if_default("disk_model", "physical_nt")
            _set_if_default("disk_radial_profile", "nt_page_thorne")
            _set_if_default("thick_disk", False)
            _set_if_default("disk_plasma_warmth", 0.42)
            _set_if_default("disk_palette", "interstellar_warm")
            _set_if_default("disk_layered_palette", True)
            _set_if_default("disk_layer_count", 10)
            _set_if_default("disk_layer_mix", 0.45)
            _set_if_default("disk_layer_pattern_count", 10.0)
            _set_if_default("disk_layer_pattern_contrast", 0.34)
            _set_if_default("disk_emission_gain", 1.35)
            _set_if_default("disk_beaming_strength", 0.78)
            _set_if_default("disk_self_occlusion_strength", 0.22)
            _set_if_default("multi_hit_disk", True)
            _set_if_default("max_disk_crossings", 8)
            _set_if_default("lensing_order_strength", 0.34)
            _set_if_default("lensing_order_gamma", 0.72)
            _set_if_default("tone_mapper", "filmic")
            _set_if_default("tone_exposure", 1.12)
            _set_if_default("tone_white_point", 9.5)
            _set_if_default("tone_highlight_rolloff", 0.30)
            _set_if_default("postprocess_pipeline", "gargantua")
            _set_if_default("gargantua_look_strength", 1.0)
            _set_if_default("meridian_supersample", True)
            _set_if_default("enforce_black_hole_shadow", True)
            if updates:
                cfg = replace(cfg, **updates)
        return cfg

    def validated(self) -> "RenderConfig":
        cfg = self.with_defaults()
        if cfg.width < 64 or cfg.height < 64:
            raise ValueError("Resolution too low. Use at least 64x64.")
        if cfg.width > 5000 or cfg.height > 5000:
            raise ValueError("Resolution too high for this reference implementation.")
        if cfg.metric_model not in METRIC_MODELS:
            raise ValueError(f"metric_model must be one of: {', '.join(sorted(METRIC_MODELS))}")
        if cfg.coordinate_system not in {"boyer_lindquist", "kerr_schild", "generalized_doran"}:
            raise ValueError("coordinate_system must be 'boyer_lindquist', 'kerr_schild', or 'generalized_doran'")
        ks_family = cfg.coordinate_system in {"kerr_schild", "generalized_doran"}
        if ks_family:
            # Generalized KS path: allow all supported metric families, including de Sitter variants.
            if cfg.kerr_schild_mode not in {"off", "fsal_only", "analytic"}:
                raise ValueError("kerr_schild_mode must be one of: off, fsal_only, analytic")
            if cfg.metric_model == "kerr_newman_de_sitter":
                if cfg.kerr_schild_mode == "off":
                    raise ValueError(
                        "For metric_model='kerr_newman_de_sitter' in Kerr-Schild-family coordinates, "
                        "normal Kerr-Schild mode 'off' is not allowed. Use generalized KS modes "
                        "('fsal_only' or 'analytic')."
                    )
                if not cfg.kerr_schild_improvements:
                    raise ValueError(
                        "For metric_model='kerr_newman_de_sitter' in Kerr-Schild-family coordinates, "
                        "kerr_schild_improvements cannot be disabled (GKS-only path)."
                    )
        elif cfg.kerr_schild_mode != "off":
            # Keep BL path semantics unchanged.
            cfg = replace(cfg, kerr_schild_mode="off")
        if not (-1.0e6 <= cfg.observer_inclination_deg <= 1.0e6):
            raise ValueError("observer_inclination_deg is outside a sane range")
        if not (-1.0e6 <= cfg.observer_azimuth_deg <= 1.0e6):
            raise ValueError("observer_azimuth_deg is outside a sane range")
        if not (-360.0 <= cfg.observer_roll_deg <= 360.0):
            raise ValueError("observer_roll_deg must be in [-360, 360]")
        if abs(cfg.charge) > 2.0:
            raise ValueError("charge magnitude too large for this implementation (|charge| <= 2)")
        if abs(cfg.cosmological_constant) > 0.2:
            raise ValueError("cosmological_constant magnitude too large for this implementation (|Lambda| <= 0.2)")

        a_eff, _, _ = effective_metric_parameters(cfg.metric_model, cfg.spin, cfg.charge, cfg.cosmological_constant)
        if abs(a_eff) >= 1.0:
            raise ValueError("effective spin must satisfy |a| < 1 for rotating metrics")

        try:
            horizon = event_horizon_radius(cfg.spin, cfg.metric_model, cfg.charge, cfg.cosmological_constant)
        except Exception as exc:
            raise ValueError(f"Invalid metric parameters: no regular event horizon ({exc})") from exc

        _, _, lmb_eff = effective_metric_parameters(cfg.metric_model, cfg.spin, cfg.charge, cfg.cosmological_constant)
        if lmb_eff > 0.0:
            roots = horizon_radii(cfg.spin, cfg.metric_model, cfg.charge, cfg.cosmological_constant)
            if len(roots) >= 2:
                cosmological_horizon = roots[-1]
                if cfg.observer_radius >= 0.98 * cosmological_horizon:
                    raise ValueError("observer_radius must stay inside the cosmological horizon for de Sitter metrics")
                if cfg.escape_radius >= 0.995 * cosmological_horizon:
                    raise ValueError("escape_radius must be below the cosmological horizon for de Sitter metrics")

        if cfg.coordinate_system == "generalized_doran":
            if cfg.observer_radius <= 1.0e-3:
                raise ValueError("observer_radius must be > 0 for generalized_doran coordinates")
        else:
            if cfg.observer_radius <= max(4.0, 1.02 * horizon):
                raise ValueError("observer_radius must be safely outside the event horizon")
        if cfg.disk_inner_radius is None:
            raise ValueError("disk_inner_radius default resolution failed")
        if cfg.disk_inner_radius <= max(1.0, 1.001 * horizon):
            raise ValueError("disk_inner_radius must be outside the event horizon")
        if cfg.disk_outer_radius <= cfg.disk_inner_radius:
            raise ValueError("disk_outer_radius must be greater than disk_inner_radius")
        if cfg.inner_edge_boost < 0.0:
            raise ValueError("inner_edge_boost must be >= 0")
        if cfg.outer_edge_boost < 0.0:
            raise ValueError("outer_edge_boost must be >= 0")
        if cfg.background_mode not in {"procedural", "hdri"}:
            raise ValueError("background_mode must be 'procedural' or 'hdri'")
        if cfg.background_projection not in {"cubemap", "equirectangular"}:
            raise ValueError("background_projection must be 'cubemap' or 'equirectangular'")
        if cfg.cubemap_face_size < 64 or cfg.cubemap_face_size > 4096:
            raise ValueError("cubemap_face_size must be in [64, 4096]")
        if cfg.background_mode == "hdri" and not cfg.hdri_path:
            raise ValueError("hdri_path is required when background_mode='hdri'")
        if cfg.hdri_exposure <= 0.0:
            raise ValueError("hdri_exposure must be positive")
        if not (-1.0e6 <= cfg.background_meridian_offset_deg <= 1.0e6):
            raise ValueError("background_meridian_offset_deg is outside a sane range")
        if not (0.0 <= cfg.star_density <= 0.05):
            raise ValueError("star_density must be in [0, 0.05]")
        if cfg.star_brightness < 0.0:
            raise ValueError("star_brightness must be >= 0")
        if cfg.star_seed < 0:
            raise ValueError("star_seed must be >= 0")
        if cfg.disk_temperature_inner <= 1000.0:
            raise ValueError("disk_temperature_inner must be > 1000 K")
        if cfg.disk_model not in {"legacy", "physical_nt"}:
            raise ValueError("disk_model must be 'legacy' or 'physical_nt'")
        if cfg.disk_radial_profile not in {"nt_proxy", "nt_page_thorne"}:
            raise ValueError("disk_radial_profile must be 'nt_proxy' or 'nt_page_thorne'")
        if cfg.disk_structure_mode not in {"continuous", "concentric_annuli"}:
            raise ValueError("disk_structure_mode must be 'continuous' or 'concentric_annuli'")
        if cfg.disk_annuli_count < 4 or cfg.disk_annuli_count > 4096:
            raise ValueError("disk_annuli_count must be in [4, 4096]")
        if cfg.disk_annuli_blend < 0.0 or cfg.disk_annuli_blend > 1.0:
            raise ValueError("disk_annuli_blend must be in [0, 1]")
        if cfg.disk_emission_gain < 0.1 or cfg.disk_emission_gain > 1000.0:
            raise ValueError("disk_emission_gain must be in [0.1, 1000.0]")
        if cfg.disk_color_correction < 1.0 or cfg.disk_color_correction > 4.0:
            raise ValueError("disk_color_correction must be in [1.0, 4.0]")
        if cfg.disk_plasma_warmth < 0.0 or cfg.disk_plasma_warmth > 1.0:
            raise ValueError("disk_plasma_warmth must be in [0, 1]")
        if cfg.disk_palette not in {"default", "interstellar_warm"}:
            raise ValueError("disk_palette must be 'default' or 'interstellar_warm'")
        if cfg.disk_layer_count < 2 or cfg.disk_layer_count > 512:
            raise ValueError("disk_layer_count must be in [2, 512]")
        if cfg.disk_layer_mix < 0.0 or cfg.disk_layer_mix > 1.0:
            raise ValueError("disk_layer_mix must be in [0, 1]")
        if cfg.disk_layer_pattern_count <= 0.0 or cfg.disk_layer_pattern_count > 1024.0:
            raise ValueError("disk_layer_pattern_count must be in (0, 1024]")
        if cfg.disk_layer_pattern_contrast < 0.0 or cfg.disk_layer_pattern_contrast > 1.0:
            raise ValueError("disk_layer_pattern_contrast must be in [0, 1]")
        if cfg.disk_layer_time_scale <= 0.0 or cfg.disk_layer_time_scale > 64.0:
            raise ValueError("disk_layer_time_scale must be in (0, 64]")
        if cfg.disk_layer_global_phase < -1.0e6 or cfg.disk_layer_global_phase > 1.0e6:
            raise ValueError("disk_layer_global_phase must be in [-1e6, 1e6]")
        if cfg.disk_layer_phase_rate_hz < 0.0 or cfg.disk_layer_phase_rate_hz > 64.0:
            raise ValueError("disk_layer_phase_rate_hz must be in [0, 64]")
        if cfg.disk_layer_accident_strength < 0.0 or cfg.disk_layer_accident_strength > 4.0:
            raise ValueError("disk_layer_accident_strength must be in [0, 4]")
        if cfg.disk_layer_accident_count < 0.0 or cfg.disk_layer_accident_count > 128.0:
            raise ValueError("disk_layer_accident_count must be in [0, 128]")
        if cfg.disk_layer_accident_sharpness < 1.0 or cfg.disk_layer_accident_sharpness > 32.0:
            raise ValueError("disk_layer_accident_sharpness must be in [1, 32]")
        if cfg.disk_adaptive_layers_min < 2 or cfg.disk_adaptive_layers_min > 1024:
            raise ValueError("disk_adaptive_layers_min must be in [2, 1024]")
        if cfg.disk_adaptive_layers_max < 2 or cfg.disk_adaptive_layers_max > 2048:
            raise ValueError("disk_adaptive_layers_max must be in [2, 2048]")
        if cfg.disk_adaptive_layers_max < cfg.disk_adaptive_layers_min:
            raise ValueError("disk_adaptive_layers_max must be >= disk_adaptive_layers_min")
        if cfg.disk_adaptive_complexity_mix < 0.0 or cfg.disk_adaptive_complexity_mix > 1.0:
            raise ValueError("disk_adaptive_complexity_mix must be in [0, 1]")
        if cfg.disk_thickness_ratio < 0.0 or cfg.disk_thickness_ratio > 1.2:
            raise ValueError("disk_thickness_ratio must be in [0, 1.2]")
        if cfg.disk_thickness_power < -2.0 or cfg.disk_thickness_power > 2.0:
            raise ValueError("disk_thickness_power must be in [-2, 2]")
        if cfg.disk_vertical_softness < 0.01 or cfg.disk_vertical_softness > 2.0:
            raise ValueError("disk_vertical_softness must be in [0.01, 2.0]")
        if cfg.disk_volume_samples < 1 or cfg.disk_volume_samples > 64:
            raise ValueError("disk_volume_samples must be in [1, 64]")
        if cfg.disk_volume_density_scale < 0.0 or cfg.disk_volume_density_scale > 1000.0:
            raise ValueError("disk_volume_density_scale must be in [0, 1000]")
        if cfg.disk_volume_temperature_drop < 0.0 or cfg.disk_volume_temperature_drop > 1.0:
            raise ValueError("disk_volume_temperature_drop must be in [0, 1]")
        if cfg.disk_volume_strength < 0.0 or cfg.disk_volume_strength > 10.0:
            raise ValueError("disk_volume_strength must be in [0, 10]")
        if cfg.vertical_transition_mode not in {"snap", "continuous"}:
            raise ValueError("vertical_transition_mode must be 'snap' or 'continuous'")
        if cfg.shadow_absorb_radius_factor < 1.0:
            raise ValueError("shadow_absorb_radius_factor must be >= 1.0")
        if cfg.step_size <= 0.0:
            raise ValueError("step_size must be positive")
        if cfg.adaptive_rtol <= 0.0:
            raise ValueError("adaptive_rtol must be positive")
        if cfg.adaptive_atol <= 0.0:
            raise ValueError("adaptive_atol must be positive")
        if cfg.adaptive_step_min <= 0.0:
            raise ValueError("adaptive_step_min must be positive")
        if cfg.adaptive_step_max <= cfg.adaptive_step_min:
            raise ValueError("adaptive_step_max must be greater than adaptive_step_min")
        if cfg.adaptive_fallback_substeps < 1 or cfg.adaptive_fallback_substeps > 16:
            raise ValueError("adaptive_fallback_substeps must be in [1, 16]")
        if cfg.kerr_schild_null_norm_interval < 1 or cfg.kerr_schild_null_norm_interval > 100000:
            raise ValueError("kerr_schild_null_norm_interval must be in [1, 100000]")
        if cfg.kerr_schild_null_norm_tol <= 0.0:
            raise ValueError("kerr_schild_null_norm_tol must be positive")
        if cfg.render_tile_rows < 0:
            raise ValueError("render_tile_rows must be >= 0")
        if cfg.roi_supersample_threshold < 0.50 or cfg.roi_supersample_threshold > 0.999:
            raise ValueError("roi_supersample_threshold must be in [0.50, 0.999]")
        if cfg.roi_supersample_jitter <= 0.0 or cfg.roi_supersample_jitter > 1.0:
            raise ValueError("roi_supersample_jitter must be in (0, 1]")
        if cfg.roi_supersample_samples < 1 or cfg.roi_supersample_samples > 8:
            raise ValueError("roi_supersample_samples must be in [1, 8]")
        if cfg.adaptive_spatial_preview_steps < 16 or cfg.adaptive_spatial_preview_steps > 10000:
            raise ValueError("adaptive_spatial_preview_steps must be in [16, 10000]")
        if cfg.adaptive_spatial_min_scale <= 0.0 or cfg.adaptive_spatial_min_scale > 1.0:
            raise ValueError("adaptive_spatial_min_scale must be in (0, 1]")
        if cfg.adaptive_spatial_quantile < 0.50 or cfg.adaptive_spatial_quantile > 0.995:
            raise ValueError("adaptive_spatial_quantile must be in [0.50, 0.995]")
        if not cfg.persistent_cache_dir:
            raise ValueError("persistent_cache_dir cannot be empty")
        if cfg.progress_backend not in {"manual", "tqdm", "auto"}:
            raise ValueError("progress_backend must be 'manual', 'tqdm', or 'auto'")
        if cfg.animation_workers < 1 or cfg.animation_workers > 64:
            raise ValueError("animation_workers must be in [1, 64]")
        if cfg.stream_encode_queue_size < 1 or cfg.stream_encode_queue_size > 64:
            raise ValueError("stream_encode_queue_size must be in [1, 64]")
        if cfg.quality_lock_psnr_min <= 0.0:
            raise ValueError("quality_lock_psnr_min must be > 0")
        if not (0.0 < cfg.quality_lock_ssim_min <= 1.0):
            raise ValueError("quality_lock_ssim_min must be in (0, 1]")
        if cfg.quality_lock:
            if cfg.quality_lock_sample_width < 64 or cfg.quality_lock_sample_width > cfg.width:
                raise ValueError("quality_lock_sample_width must be in [64, width] when quality_lock is enabled")
            if cfg.quality_lock_sample_height < 64 or cfg.quality_lock_sample_height > cfg.height:
                raise ValueError("quality_lock_sample_height must be in [64, height] when quality_lock is enabled")
        if not (0.0 <= cfg.temporal_blend <= 1.0):
            raise ValueError("temporal_blend must be in [0, 1]")
        if cfg.temporal_clamp <= 0.0:
            raise ValueError("temporal_clamp must be > 0")
        if cfg.motion_vector_scale < 0.0:
            raise ValueError("motion_vector_scale must be >= 0")
        if cfg.temporal_denoise_mode not in {"basic", "robust"}:
            raise ValueError("temporal_denoise_mode must be 'basic' or 'robust'")
        if cfg.temporal_denoise_radius < 1 or cfg.temporal_denoise_radius > 4:
            raise ValueError("temporal_denoise_radius must be in [1, 4]")
        if cfg.temporal_denoise_sigma <= 0.0:
            raise ValueError("temporal_denoise_sigma must be > 0")
        if cfg.temporal_denoise_clip <= 0.0:
            raise ValueError("temporal_denoise_clip must be > 0")
        if cfg.disk_beaming_strength < 0.0:
            raise ValueError("disk_beaming_strength must be >= 0")
        if not (0.0 <= cfg.disk_self_occlusion_strength <= 1.0):
            raise ValueError("disk_self_occlusion_strength must be in [0, 1]")
        if cfg.max_disk_crossings < 1 or cfg.max_disk_crossings > 16:
            raise ValueError("max_disk_crossings must be in [1, 16]")
        if cfg.lensing_order_strength < 0.0 or cfg.lensing_order_strength > 4.0:
            raise ValueError("lensing_order_strength must be in [0, 4]")
        if cfg.lensing_order_gamma < 0.0 or cfg.lensing_order_gamma > 3.0:
            raise ValueError("lensing_order_gamma must be in [0, 3]")
        if cfg.faraday_rotation_strength < 0.0:
            raise ValueError("faraday_rotation_strength must be >= 0")
        if cfg.tone_mapper not in {"reinhard", "aces", "filmic"}:
            raise ValueError("tone_mapper must be 'reinhard', 'aces', or 'filmic'")
        if cfg.tone_exposure <= 0.0 or cfg.tone_exposure > 64.0:
            raise ValueError("tone_exposure must be in (0, 64]")
        if cfg.tone_white_point <= 0.0 or cfg.tone_white_point > 128.0:
            raise ValueError("tone_white_point must be in (0, 128]")
        if cfg.tone_highlight_rolloff < 0.0 or cfg.tone_highlight_rolloff > 4.0:
            raise ValueError("tone_highlight_rolloff must be in [0, 4]")
        if cfg.postprocess_pipeline not in {"off", "gargantua"}:
            raise ValueError("postprocess_pipeline must be 'off' or 'gargantua'")
        if cfg.gargantua_look_strength < 0.0 or cfg.gargantua_look_strength > 2.0:
            raise ValueError("gargantua_look_strength must be in [0, 2]")
        if cfg.video_codec not in {"h264", "h265_10bit"}:
            raise ValueError("video_codec must be 'h264' or 'h265_10bit'")
        if cfg.video_crf < 0 or cfg.video_crf > 51:
            raise ValueError("video_crf must be in [0, 51]")
        if cfg.max_steps < 16:
            raise ValueError("max_steps must be >= 16")
        if cfg.escape_radius <= cfg.observer_radius:
            raise ValueError("escape_radius must be greater than observer_radius")
        return cfg

    def resolve_device(self) -> torch.device:
        if self.device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        if self.device == "mps" and not torch.backends.mps.is_available():
            details = []
            if torch.backends.mps.is_built():
                details.append("PyTorch has MPS support but runtime reports it unavailable")
            if os.environ.get("CODEX_SANDBOX"):
                details.append("CODEX_SANDBOX is active (sandboxed runs can block MPS)")
            suffix = f" ({'; '.join(details)})" if details else ""
            raise RuntimeError(
                "MPS requested but not available"
                f"{suffix}. Run from a native arm64 terminal outside sandbox and retry."
            )
        if self.device not in {"cpu", "cuda", "mps"}:
            raise ValueError("device must be one of: auto, cpu, cuda, mps")
        return torch.device(self.device)

    def resolve_dtype(self) -> torch.dtype:
        lookup = {
            "float32": torch.float32,
            "float64": torch.float64,
        }
        if self.dtype not in lookup:
            raise ValueError("dtype must be float32 or float64")
        return lookup[self.dtype]
