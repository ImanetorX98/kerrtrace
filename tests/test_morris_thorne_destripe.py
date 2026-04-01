from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch

from kerrtrace.raytracer import KerrRayTracer


def _edge_peak_ratio(rgb: torch.Tensor) -> float:
    lum = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
    edge = torch.mean(torch.abs(lum - torch.roll(lum, shifts=1, dims=1)), dim=0)
    peak = float(torch.max(edge).item())
    med = float(torch.median(edge).item())
    return peak / (med + 1.0e-9)


class MorrisThorneDestripeTests(unittest.TestCase):
    def _dummy_tracer(
        self,
        *,
        is_wormhole: bool,
        destripe_meridian: bool,
        beam_samples: int = 1,
        beam_jitter: float = 0.4,
        beam_threshold: float = 6.0,
        beam_band_halfwidth: int = 2,
    ) -> KerrRayTracer:
        tracer = object.__new__(KerrRayTracer)
        tracer.is_wormhole = is_wormhole
        tracer.config = SimpleNamespace(
            destripe_meridian=destripe_meridian,
            wormhole_mt_beam_samples=beam_samples,
            wormhole_mt_beam_jitter=beam_jitter,
            wormhole_mt_beam_threshold=beam_threshold,
            wormhole_mt_beam_band_halfwidth=beam_band_halfwidth,
        )
        return tracer

    def test_policy_non_wormhole_default_disabled(self) -> None:
        tracer = self._dummy_tracer(is_wormhole=False, destripe_meridian=False)
        self.assertEqual(tracer._meridian_destripe_passes(), 0)

    def test_policy_non_wormhole_manual_enabled(self) -> None:
        tracer = self._dummy_tracer(is_wormhole=False, destripe_meridian=True)
        self.assertEqual(tracer._meridian_destripe_passes(), 1)

    def test_policy_wormhole_auto_enabled(self) -> None:
        tracer = self._dummy_tracer(is_wormhole=True, destripe_meridian=False)
        self.assertEqual(tracer._meridian_destripe_passes(), 2)

    def test_destripe_reduces_strong_vertical_seam(self) -> None:
        tracer = self._dummy_tracer(is_wormhole=True, destripe_meridian=False)
        rgb = torch.zeros((24, 24, 3), dtype=torch.float32)
        rgb[:, 0, :] = 1.0
        before = _edge_peak_ratio(rgb)
        out = tracer._destripe_meridian(rgb, force=True)
        after = _edge_peak_ratio(out)
        self.assertLess(after, before)

    def test_beam_offsets_are_symmetric(self) -> None:
        tracer = self._dummy_tracer(is_wormhole=True, destripe_meridian=False, beam_samples=2, beam_jitter=0.4)
        offsets = tracer._wormhole_mt_beam_offsets()
        self.assertEqual(offsets, [0.4, -0.4, 0.2, -0.2])

    def test_beam_mask_targets_seam_band(self) -> None:
        tracer = self._dummy_tracer(
            is_wormhole=True,
            destripe_meridian=False,
            beam_samples=1,
            beam_jitter=0.4,
            beam_threshold=1.5,
            beam_band_halfwidth=1,
        )
        rgb = torch.zeros((16, 16, 3), dtype=torch.float32)
        rgb[:, 0, :] = 1.0
        rgb[:, 1, :] = 0.0
        mask = tracer._build_wormhole_mt_beam_mask(rgb)
        self.assertTrue(bool(mask.any()))
        self.assertTrue(bool(mask[:, 0].any() or mask[:, 1].any()))


if __name__ == "__main__":
    unittest.main()
