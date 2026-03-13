#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a desktop executable for KerrTrace WebUI using PyInstaller",
    )
    parser.add_argument("--name", default="KerrTraceUI", help="Executable/App name")
    parser.add_argument("--onefile", action="store_true", help="Build as onefile binary")
    parser.add_argument(
        "--console",
        action="store_true",
        help="Keep console window visible (default: windowed)",
    )
    parser.add_argument(
        "--no-torch-collect",
        action="store_true",
        help="Do not force torch collect-all (smaller build config, but may miss runtime deps)",
    )
    parser.add_argument("--distpath", default="dist", help="PyInstaller dist path")
    parser.add_argument("--workpath", default="build", help="PyInstaller work path")
    parser.add_argument("--specpath", default=".", help="PyInstaller spec output path")
    parser.add_argument("--clean", action="store_true", help="Clean PyInstaller cache before build")
    return parser.parse_args()


def _ensure_pyinstaller() -> None:
    try:
        import PyInstaller  # noqa: F401
    except Exception as exc:
        raise SystemExit(
            "PyInstaller is not installed. Install it with:\n"
            "  .venv/bin/pip install pyinstaller\n"
            f"Original error: {exc}"
        )


def main() -> int:
    args = _parse_args()
    _ensure_pyinstaller()

    root = Path(__file__).resolve().parents[1]
    entry = root / "kerrtrace" / "desktop_launcher.py"
    assets = root / "assets"

    data_sep = ";" if sys.platform.startswith("win") else ":"

    cmd: list[str] = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--name",
        str(args.name),
        "--distpath",
        str(Path(args.distpath)),
        "--workpath",
        str(Path(args.workpath)),
        "--specpath",
        str(Path(args.specpath)),
        "--collect-data",
        "kerrtrace",
        "--collect-submodules",
        "kerrtrace",
        "--collect-data",
        "streamlit",
        "--collect-submodules",
        "streamlit",
        "--hidden-import",
        "streamlit.web.bootstrap",
    ]

    if not args.no_torch_collect:
        cmd += [
            "--collect-data",
            "torch",
            "--collect-submodules",
            "torch",
        ]

    if assets.exists():
        cmd += ["--add-data", f"{assets}{data_sep}assets"]

    if args.clean:
        cmd.append("--clean")
    if args.onefile:
        cmd.append("--onefile")
    else:
        cmd.append("--onedir")

    if not args.console:
        cmd.append("--windowed")

    cmd.append(str(entry))

    print("Building KerrTrace desktop executable with command:\n")
    print(" ".join(cmd))
    print("")

    proc = subprocess.run(cmd, cwd=str(root), check=False)
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
