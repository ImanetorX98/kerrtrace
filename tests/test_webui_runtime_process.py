from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from kerrtrace.webui_runtime import launch_background_process


class WebUiRuntimeProcessTests(unittest.TestCase):
    def test_launch_background_process_writes_log_and_meta(self) -> None:
        with tempfile.TemporaryDirectory(prefix="kerrtrace_bg_") as tmp:
            workspace = Path(tmp)
            log_path = workspace / "run.log"
            cfg_path = workspace / "cfg.json"
            cfg_path.write_text("{}", encoding="utf-8")

            cmd = [sys.executable, "-c", "print('bg-ok')"]
            proc, meta = launch_background_process(
                cmd=cmd,
                workspace_path=workspace,
                log_path=log_path,
                cfg_path=cfg_path,
                output_hint="out.png",
                stamp="20260330_101010",
                job_id="job_1",
            )

            rc = proc.wait(timeout=10)
            self.assertEqual(rc, 0)
            self.assertTrue(log_path.exists())
            log_text = log_path.read_text(encoding="utf-8")
            self.assertIn("bg-ok", log_text)
            self.assertEqual(meta["job_id"], "job_1")
            self.assertEqual(meta["cfg_path"], str(cfg_path))
            self.assertEqual(meta["workspace"], str(workspace))
            self.assertEqual(meta["output_hint"], "out.png")
            self.assertIn(sys.executable, meta["cmd"])


if __name__ == "__main__":
    unittest.main()
