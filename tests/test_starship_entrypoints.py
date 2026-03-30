from __future__ import annotations

import subprocess
import sys
import unittest


class StarshipEntrypointsTests(unittest.TestCase):
    def test_starship_module_help(self) -> None:
        proc = subprocess.run(
            [sys.executable, "-m", "kerrtrace.starship_video", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(proc.returncode, 0)
        self.assertIn("Render KNdS frame/video", proc.stdout)

    def test_legacy_script_wrapper_help(self) -> None:
        proc = subprocess.run(
            [sys.executable, "-m", "scripts.render_obj_starship_video", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(proc.returncode, 0)
        self.assertIn("Render KNdS frame/video", proc.stdout)


if __name__ == "__main__":
    unittest.main()
