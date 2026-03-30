from __future__ import annotations

import unittest
from pathlib import Path

from kerrtrace.starship_cli import STARSHIP_VIDEO_MODULE, build_starship_command


class StarshipCliCommandTests(unittest.TestCase):
    def test_build_starship_command_required_arguments(self) -> None:
        cmd = build_starship_command(
            python_exec="python3",
            ship_cfg_path=Path("/tmp/ships.json"),
            output_path="/tmp/out.mp4",
            width=320,
            height=180,
            observer_radius=30.0,
            observer_theta_deg=80.0,
            observer_phi_deg=10.0,
            frames=1,
            fps=10,
            ship_substeps=2,
            disk_outer_radius=10.0,
            disk_emission_gain=1.2,
            step_size=0.25,
            max_steps=120,
            device="cpu",
        )
        self.assertEqual(cmd[0:3], ["python3", "-m", STARSHIP_VIDEO_MODULE])
        self.assertIn("--ship-config-json", cmd)
        self.assertIn("/tmp/ships.json", cmd)
        self.assertIn("--output", cmd)
        self.assertIn("/tmp/out.mp4", cmd)
        self.assertIn("--device", cmd)
        self.assertIn("cpu", cmd)
        self.assertNotIn("--keep-frames", cmd)
        self.assertNotIn("--disk-inner-radius", cmd)

    def test_build_starship_command_optional_arguments(self) -> None:
        cmd = build_starship_command(
            python_exec="python3",
            ship_cfg_path=Path("/tmp/ships.json"),
            output_path="/tmp/out.gif",
            width=640,
            height=360,
            observer_radius=25.0,
            observer_theta_deg=75.0,
            observer_phi_deg=0.0,
            frames=0,  # should be clamped to >=1
            fps=0,  # should be clamped to >=1
            ship_substeps=0,  # should be clamped to >=1
            disk_outer_radius=12.0,
            disk_emission_gain=2.0,
            step_size=0.2,
            max_steps=0,  # should be clamped to >=1
            device="auto",
            keep_frames=True,
            disk_inner_radius=4.5,
        )
        self.assertIn("--keep-frames", cmd)
        self.assertIn("--disk-inner-radius", cmd)
        idx = cmd.index("--disk-inner-radius")
        self.assertEqual(cmd[idx + 1], "4.5")
        self.assertIn("--frames", cmd)
        self.assertEqual(cmd[cmd.index("--frames") + 1], "1")
        self.assertEqual(cmd[cmd.index("--fps") + 1], "1")
        self.assertEqual(cmd[cmd.index("--ship-substeps") + 1], "1")
        self.assertEqual(cmd[cmd.index("--max-steps") + 1], "1")


if __name__ == "__main__":
    unittest.main()
