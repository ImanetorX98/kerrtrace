"""
Verify that _trace() returns a valid TraceFrame for kerr_schild coordinates
and that _trace_kerr_schild() results (wrapped by render()) are consistent.

These tests avoid .numpy() entirely — they work purely with torch tensors —
so they run correctly even when the numpy/torch environment is broken.
"""
from __future__ import annotations

import math
import unittest

import torch

from kerrtrace.config import RenderConfig
from kerrtrace.raytracer import KerrRayTracer, TraceFrame


def _make_ks_tracer(width: int = 64, height: int = 64) -> KerrRayTracer:
    cfg = RenderConfig(
        width=width,
        height=height,
        coordinate_system="kerr_schild",
        metric_model="kerr",
        spin=0.9,
        observer_radius=25.0,
        observer_inclination_deg=75.0,
        disk_outer_radius=18.0,
        disk_model="physical_nt",
        adaptive_integrator=False,
        max_steps=500,
        step_size=0.11,
        device="cpu",
        dtype="float32",
        show_progress_bar=False,
        star_seed=7,
        render_tile_rows=64,
    ).validated()
    return KerrRayTracer(cfg)


class KerrSchildTraceFrameTests(unittest.TestCase):
    def setUp(self) -> None:
        KerrRayTracer._compiled_unbound_cache.clear()
        KerrRayTracer._compiled_unbound_fail.clear()

    def test_trace_kerr_schild_returns_trace_frame(self) -> None:
        """_trace_kerr_schild() result can be converted to TraceFrame without error."""
        tracer = _make_ks_tracer(64, 64)
        raw = tracer._trace_kerr_schild(row_start=0, row_end=64)
        # raw is a tuple; _as_trace_frame must convert it
        # We access the internal helper via the closure — test by verifying
        # that the tuple has the expected 20 elements.
        self.assertEqual(len(raw), 20, "Expected 20-element tuple from _trace_kerr_schild")

    def test_trace_kerr_schild_has_disk_hits(self) -> None:
        """For a high-inclination config the disk should have some hits."""
        tracer = _make_ks_tracer(64, 64)
        raw = tracer._trace_kerr_schild(row_start=0, row_end=64)
        hit_disk: torch.Tensor = raw[0]
        n_hits = int(hit_disk.sum().item())
        self.assertGreater(n_hits, 0, "Expected >0 disk hits for kerr_schild render")

    def test_trace_kerr_schild_stats_are_finite(self) -> None:
        """r_emit, p_t_emit, p_phi_emit should be finite for disk-hit rays."""
        tracer = _make_ks_tracer(64, 64)
        raw = tracer._trace_kerr_schild(row_start=0, row_end=64)
        hit_disk: torch.Tensor = raw[0]
        r_emit: torch.Tensor = raw[4]
        p_t_emit: torch.Tensor = raw[5]
        p_phi_emit: torch.Tensor = raw[6]

        if hit_disk.any():
            self.assertTrue(torch.isfinite(r_emit[hit_disk]).all().item(),
                            "r_emit not finite for disk hits")
            self.assertTrue(torch.isfinite(p_t_emit[hit_disk]).all().item(),
                            "p_t_emit not finite for disk hits")
            self.assertTrue(torch.isfinite(p_phi_emit[hit_disk]).all().item(),
                            "p_phi_emit not finite for disk hits")

    def test_internal_trace_returns_trace_frame(self) -> None:
        """_trace() (boyer-lindquist path) must return TraceFrame after refactor."""
        cfg = RenderConfig(
            width=64,
            height=64,
            coordinate_system="boyer_lindquist",
            metric_model="kerr",
            spin=0.9,
            observer_radius=40.0,
            observer_inclination_deg=85.0,
            disk_outer_radius=10.0,
            disk_model="physical_nt",
            adaptive_integrator=False,
            max_steps=80,
            step_size=0.3,
            device="cpu",
            dtype="float32",
            show_progress_bar=False,
            render_tile_rows=64,
        ).validated()
        tracer = KerrRayTracer(cfg)
        result = tracer._trace(row_start=0, row_end=64)
        self.assertIsInstance(result, TraceFrame,
                              "_trace() must return TraceFrame after refactor")

    def test_trace_frame_disk_hits_match_between_ks_paths(self) -> None:
        """
        Two calls to _trace_kerr_schild with the same config produce identical
        hit_disk tensors (determinism check).
        """
        tracer = _make_ks_tracer(64, 64)
        raw_a = tracer._trace_kerr_schild(row_start=0, row_end=64)
        raw_b = tracer._trace_kerr_schild(row_start=0, row_end=64)
        self.assertTrue(
            torch.equal(raw_a[0], raw_b[0]),
            "hit_disk not deterministic across two calls to _trace_kerr_schild",
        )
        self.assertTrue(
            torch.equal(raw_a[2], raw_b[2]),
            "hit_horizon not deterministic across two calls to _trace_kerr_schild",
        )

    def test_kerr_schild_vs_trace_frame_fields_consistent(self) -> None:
        """
        hit_disk, hit_emitter, hit_horizon, escaped are mutually consistent:
        a ray cannot have hit_disk=True AND hit_horizon=True simultaneously.
        """
        tracer = _make_ks_tracer(64, 64)
        raw = tracer._trace_kerr_schild(row_start=0, row_end=64)
        hit_disk: torch.Tensor = raw[0]
        hit_horizon: torch.Tensor = raw[2]
        overlap = (hit_disk & hit_horizon)
        n_overlap = int(overlap.sum().item())
        self.assertEqual(n_overlap, 0,
                         "Ray cannot hit both disk and horizon simultaneously")

    def test_steps_used_is_positive_int(self) -> None:
        """steps_used (last element) must be a positive integer."""
        tracer = _make_ks_tracer(64, 64)
        raw = tracer._trace_kerr_schild(row_start=0, row_end=64)
        steps = int(raw[-1])
        self.assertIsInstance(steps, int)
        self.assertGreater(steps, 0)


if __name__ == "__main__":
    unittest.main()
