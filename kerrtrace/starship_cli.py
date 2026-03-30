from __future__ import annotations

from pathlib import Path

STARSHIP_VIDEO_MODULE = "kerrtrace.starship_video"


def build_starship_command(
    *,
    python_exec: str,
    ship_cfg_path: Path,
    output_path: str | Path,
    width: int,
    height: int,
    observer_radius: float,
    observer_theta_deg: float,
    observer_phi_deg: float,
    frames: int,
    fps: int,
    ship_substeps: int,
    disk_outer_radius: float,
    disk_emission_gain: float,
    step_size: float,
    max_steps: int,
    device: str,
    keep_frames: bool = False,
    disk_inner_radius: float | None = None,
) -> list[str]:
    cmd = [
        str(python_exec),
        "-m",
        STARSHIP_VIDEO_MODULE,
        "--ship-config-json",
        str(ship_cfg_path),
        "--output",
        str(output_path),
        "--width",
        str(int(width)),
        "--height",
        str(int(height)),
        "--observer-radius",
        str(float(observer_radius)),
        "--observer-theta-deg",
        str(float(observer_theta_deg)),
        "--observer-phi-deg",
        str(float(observer_phi_deg)),
        "--frames",
        str(max(1, int(frames))),
        "--fps",
        str(max(1, int(fps))),
        "--ship-substeps",
        str(max(1, int(ship_substeps))),
        "--disk-outer-radius",
        str(float(disk_outer_radius)),
        "--disk-emission-gain",
        str(float(disk_emission_gain)),
        "--step-size",
        str(float(step_size)),
        "--max-steps",
        str(max(1, int(max_steps))),
        "--device",
        str(device),
    ]
    if keep_frames:
        cmd.append("--keep-frames")
    if disk_inner_radius is not None:
        cmd += ["--disk-inner-radius", str(float(disk_inner_radius))]
    return cmd
