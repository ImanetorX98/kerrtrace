from __future__ import annotations

import argparse
import os
from pathlib import Path
import socket
import subprocess
import threading
import time
import webbrowser
import sys


def _resolve_workspace(path: str | None) -> Path:
    if path is None or not str(path).strip():
        return Path.cwd().resolve()
    return Path(path).expanduser().resolve()


def _resolve_webui_script(workspace: Path) -> Path:
    candidates = [
        workspace / "kerrtrace" / "webui.py",
        Path(__file__).resolve().with_name("webui.py"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    joined = "\n - ".join(str(c) for c in candidates)
    raise FileNotFoundError(f"Cannot locate webui.py. Checked:\n - {joined}")


def _is_port_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def _pick_port(host: str, preferred: int, scan: int = 200) -> int:
    if _is_port_free(host, preferred):
        return preferred
    for port in range(preferred + 1, preferred + 1 + scan):
        if _is_port_free(host, port):
            return port
    raise RuntimeError(f"No free port found in range [{preferred}, {preferred + scan}]")


def _open_browser_later(url: str, delay_s: float) -> None:
    time.sleep(max(delay_s, 0.0))
    try:
        webbrowser.open(url)
    except Exception:
        # Non-fatal: Streamlit still starts and user can open URL manually.
        pass


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch KerrTrace WebUI as a desktop-style app",
    )
    parser.add_argument("--workspace", default=".", help="Project workspace path")
    parser.add_argument("--host", default="127.0.0.1", help="HTTP bind host")
    parser.add_argument("--port", type=int, default=8501, help="HTTP bind port")
    parser.add_argument(
        "--auto-port",
        action="store_true",
        help="Auto-pick a free port when the preferred one is busy",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without browser auto-open",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Disable browser auto-open",
    )
    parser.add_argument(
        "--browser-delay",
        type=float,
        default=1.2,
        help="Delay before opening browser (seconds)",
    )
    parser.add_argument(
        "--no-chdir-workspace",
        action="store_true",
        help="Do not change cwd to workspace before launching",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    workspace = _resolve_workspace(args.workspace)
    webui_script = _resolve_webui_script(workspace)

    host = str(args.host)
    port = int(args.port)
    if args.auto_port:
        port = _pick_port(host, port)

    if not args.no_chdir_workspace:
        os.chdir(workspace)

    if (not args.headless) and (not args.no_browser):
        url = f"http://{host}:{port}"
        thread = threading.Thread(
            target=_open_browser_later,
            args=(url, float(args.browser_delay)),
            name="kerrtrace-open-browser",
            daemon=True,
        )
        thread.start()

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(webui_script),
        "--server.address",
        host,
        "--server.port",
        str(port),
        "--browser.gatherUsageStats",
        "false",
        "--server.headless",
        "true" if bool(args.headless) else "false",
    ]
    print(f"[kerrtrace-ui] workspace={workspace}")
    print(f"[kerrtrace-ui] url=http://{host}:{port}")
    print(f"[kerrtrace-ui] launching: {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(workspace), check=False)
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
