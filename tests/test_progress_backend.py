from __future__ import annotations

import unittest
from unittest.mock import patch

from kerrtrace.config import RenderConfig
from kerrtrace.raytracer import KerrRayTracer


class ProgressBackendTests(unittest.TestCase):
    def _base_config(self, **updates) -> RenderConfig:
        cfg = RenderConfig(
            width=64,
            height=64,
            coordinate_system="boyer_lindquist",
            metric_model="kerr",
            spin=0.9,
            observer_radius=30.0,
            observer_inclination_deg=70.0,
            disk_outer_radius=8.0,
            adaptive_integrator=False,
            max_steps=64,
            step_size=0.3,
            device="cpu",
            dtype="float32",
            show_progress_bar=False,
        )
        if updates:
            cfg = RenderConfig(**{**cfg.__dict__, **updates})
        return cfg.validated()

    def test_progress_backend_validation_rejects_invalid(self) -> None:
        with self.assertRaises(ValueError):
            self._base_config(progress_backend="invalid")

    def test_progress_backend_auto_prefers_tqdm_on_tty(self) -> None:
        tracer = KerrRayTracer(self._base_config(progress_backend="auto"))
        with patch("kerrtrace.raytracer.tqdm_auto", object()), patch(
            "kerrtrace.raytracer.sys.stdout.isatty", return_value=True
        ):
            self.assertEqual(tracer._resolve_progress_backend(), "tqdm")

    def test_progress_backend_tqdm_falls_back_when_missing(self) -> None:
        tracer = KerrRayTracer(self._base_config(progress_backend="tqdm"))
        with patch("kerrtrace.raytracer.tqdm_auto", None), patch(
            "kerrtrace.raytracer.sys.stdout.isatty", return_value=False
        ):
            self.assertEqual(tracer._resolve_progress_backend(), "manual")


if __name__ == "__main__":
    unittest.main()
