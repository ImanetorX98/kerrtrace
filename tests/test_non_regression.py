from __future__ import annotations

import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from kerrtrace.animation import render_animation
from kerrtrace.config import RenderConfig
from kerrtrace.raytracer import KerrRayTracer


class NonRegressionTests(unittest.TestCase):
    def setUp(self) -> None:
        KerrRayTracer._compiled_unbound_cache.clear()
        KerrRayTracer._compiled_unbound_fail.clear()

    def test_compiled_unbound_cache_reuse(self) -> None:
        calls: list[tuple[str, str]] = []

        def fake_compile(fn, mode: str = "reduce-overhead"):
            calls.append((fn.__name__, mode))
            return fn

        def dummy_rhs(self, state):  # pragma: no cover - simple placeholder
            return state

        with patch("kerrtrace.raytracer.torch.compile", side_effect=fake_compile):
            compiled_a = KerrRayTracer._get_compiled_unbound(("dummy", "cpu"), dummy_rhs)
            compiled_b = KerrRayTracer._get_compiled_unbound(("dummy", "cpu"), dummy_rhs)

        self.assertIsNotNone(compiled_a)
        self.assertIs(compiled_a, compiled_b)
        self.assertEqual(len(calls), 1)

    def test_adaptive_spatial_sampling_similarity(self) -> None:
        base_cfg = RenderConfig(
            width=128,
            height=72,
            coordinate_system="boyer_lindquist",
            metric_model="kerr",
            spin=0.9,
            observer_radius=40.0,
            observer_inclination_deg=85.0,
            disk_outer_radius=10.0,
            disk_model="physical_nt",
            disk_radial_profile="nt_page_thorne",
            adaptive_integrator=False,
            max_steps=140,
            step_size=0.24,
            device="cpu",
            dtype="float32",
            show_progress_bar=False,
            star_seed=11,
            render_tile_rows=36,
            adaptive_spatial_sampling=False,
        ).validated()
        adaptive_cfg = RenderConfig(
            **{
                **base_cfg.__dict__,
                "adaptive_spatial_sampling": True,
                "adaptive_spatial_preview_steps": 72,
                "adaptive_spatial_min_scale": 0.68,
                "adaptive_spatial_quantile": 0.78,
            }
        ).validated()

        base_img = np.asarray(KerrRayTracer(base_cfg).render().image.convert("RGB"), dtype=np.float32)
        adaptive_img = np.asarray(KerrRayTracer(adaptive_cfg).render().image.convert("RGB"), dtype=np.float32)

        mae = float(np.mean(np.abs(base_img - adaptive_img)))
        self.assertLess(mae, 18.0)

    def test_animation_multithread_smoke(self) -> None:
        cfg = RenderConfig(
            width=96,
            height=64,
            coordinate_system="boyer_lindquist",
            metric_model="kerr",
            spin=0.9,
            observer_radius=35.0,
            observer_inclination_deg=75.0,
            disk_outer_radius=10.0,
            disk_model="physical_nt",
            disk_radial_profile="nt_page_thorne",
            adaptive_integrator=False,
            max_steps=96,
            step_size=0.24,
            device="cpu",
            dtype="float32",
            show_progress_bar=False,
            render_tile_rows=32,
        ).validated()

        with tempfile.TemporaryDirectory(prefix="kerrtrace_nonreg_") as tmp:
            tmp_path = Path(tmp)
            out_video = tmp_path / "smoke.mp4"
            frames_dir = tmp_path / "frames"
            stats = render_animation(
                base_config=cfg,
                output_path=out_video,
                frames=3,
                fps=3,
                workers=2,
                frames_dir=frames_dir,
                keep_frames=True,
                render_frames=True,
                encode_output=False,
                stream_encode=False,
            )
            self.assertEqual(stats.frames, 3)
            produced = sorted(frames_dir.glob("frame_*.png"))
            self.assertEqual(len(produced), 3)


if __name__ == "__main__":
    unittest.main()
