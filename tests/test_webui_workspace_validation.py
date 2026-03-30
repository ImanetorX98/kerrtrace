from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from kerrtrace.webui_runtime import validate_workspace_path


class WebUiWorkspaceValidationTests(unittest.TestCase):
    def test_workspace_path_is_resolved(self) -> None:
        resolved = validate_workspace_path(".")
        self.assertIsInstance(resolved, Path)
        self.assertTrue(resolved.is_absolute())

    def test_rejects_system_paths(self) -> None:
        candidate = "/etc" if os.name != "nt" else "C:/Windows/System32"
        with self.assertRaises(ValueError):
            validate_workspace_path(candidate)

    def test_rejects_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            file_path = Path(tmp) / "not_a_dir.txt"
            file_path.write_text("x", encoding="utf-8")
            with self.assertRaises(ValueError):
                validate_workspace_path(str(file_path))

    def test_rejects_root_path(self) -> None:
        root = Path.cwd().anchor or ("/" if os.name != "nt" else "C:\\")
        with self.assertRaises(ValueError):
            validate_workspace_path(root)


if __name__ == "__main__":
    unittest.main()
