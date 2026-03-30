from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

from kerrtrace.starship_video import _parse_direction_vector, _resolve_obj_path


class StarshipVideoModuleTests(unittest.TestCase):
    def test_parse_direction_vector_from_csv(self) -> None:
        v = _parse_direction_vector("1.0, 2.5, -3")
        self.assertEqual(v, (1.0, 2.5, -3.0))

    def test_parse_direction_vector_invalid_falls_back(self) -> None:
        v = _parse_direction_vector("bad_input")
        self.assertEqual(v, (0.0, 0.0, 1.0))

    def test_resolve_obj_path_prefers_base_dir_existing_file(self) -> None:
        with tempfile.TemporaryDirectory(prefix="kerrtrace_starship_test_") as tmp:
            base = Path(tmp)
            obj = base / "ship.obj"
            obj.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")
            resolved = _resolve_obj_path("ship.obj", base_dir=base, fallback=base / "fallback.obj")
            self.assertEqual(resolved, obj.resolve())


if __name__ == "__main__":
    unittest.main()
