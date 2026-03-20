from __future__ import annotations

import math
from dataclasses import replace

from kerrtrace.config import RenderConfig
from kerrtrace.raytracer import KerrRayTracer


def _base_cfg() -> RenderConfig:
    return RenderConfig(
        width=96,
        height=64,
        coordinate_system="kerr_schild",
        metric_model="kerr",
        spin=0.9,
        observer_radius=40.0,
        observer_inclination_deg=0.0,
        observer_azimuth_deg=123.0,
        disk_outer_radius=10.0,
        device="cpu",
        dtype="float32",
        show_progress_bar=False,
        max_steps=64,
        step_size=0.2,
    ).validated()


def test_atlas_cartesian_variant_preserves_axis_azimuth_in_ks() -> None:
    cfg = _base_cfg()

    tracer_legacy = KerrRayTracer(replace(cfg, atlas_cartesian_variant=False).validated())
    theta_legacy, phi_legacy = tracer_legacy._observer_angles_regularized()
    assert theta_legacy > 0.0
    assert abs(phi_legacy) < 1.0e-12

    tracer_atlas = KerrRayTracer(replace(cfg, atlas_cartesian_variant=True).validated())
    theta_atlas, phi_atlas = tracer_atlas._observer_angles_regularized()
    assert abs(theta_atlas) < 1.0e-12
    assert abs(phi_atlas - math.radians(123.0)) < 1.0e-12


def test_atlas_safe_wormhole_bl_preserves_phi_with_theta_clamp() -> None:
    cfg = RenderConfig(
        width=96,
        height=64,
        coordinate_system="boyer_lindquist",
        metric_model="morris_thorne",
        observer_radius=8.0,
        observer_inclination_deg=0.0,
        observer_azimuth_deg=137.0,
        device="cpu",
        dtype="float32",
        show_progress_bar=False,
        max_steps=64,
        step_size=0.2,
    ).validated()

    tracer_legacy = KerrRayTracer(replace(cfg, atlas_cartesian_variant=False).validated())
    theta_legacy, phi_legacy = tracer_legacy._observer_angles_regularized()
    assert theta_legacy > 0.0
    assert abs(phi_legacy) < 1.0e-12

    tracer_atlas = KerrRayTracer(replace(cfg, atlas_cartesian_variant=True).validated())
    theta_atlas, phi_atlas = tracer_atlas._observer_angles_regularized()
    assert theta_atlas > 0.0
    assert theta_atlas < math.pi
    assert abs(phi_atlas - math.radians(137.0)) < 1.0e-12
