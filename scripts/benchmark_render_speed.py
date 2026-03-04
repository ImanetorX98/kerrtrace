from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime
import json
from pathlib import Path
import statistics
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kerrtrace.config import RenderConfig
from kerrtrace.raytracer import KerrRayTracer


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark KerrTrace render speed and write a timing report.")
    p.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    p.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    p.add_argument("--width", type=int, default=854)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--scenario", choices=["bl", "gks", "both"], default="both")
    p.add_argument("--require-gpu", action="store_true")
    p.add_argument("--output-dir", type=str, default="out/benchmarks")
    return p.parse_args()


def _clear_torch_cache() -> None:
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
    except Exception:
        pass


def _scenario_configs(base: RenderConfig, which: str) -> list[tuple[str, RenderConfig]]:
    out: list[tuple[str, RenderConfig]] = []
    if which in {"bl", "both"}:
        out.append(
            (
                "BL_Kerr",
                replace(
                    base,
                    coordinate_system="boyer_lindquist",
                    metric_model="kerr",
                    spin=0.95,
                    charge=0.0,
                    cosmological_constant=0.0,
                    observer_radius=40.0,
                    observer_inclination_deg=85.0,
                    disk_model="physical_nt",
                    disk_radial_profile="nt_page_thorne",
                    disk_outer_radius=10.0,
                ),
            )
        )
    if which in {"gks", "both"}:
        out.append(
            (
                "GKS_KNdS",
                replace(
                    base,
                    coordinate_system="generalized_doran",
                    metric_model="kerr_newman_de_sitter",
                    kerr_schild_mode="fsal_only",
                    kerr_schild_improvements=True,
                    spin=0.9,
                    charge=0.2,
                    cosmological_constant=1.0e-7,
                    observer_radius=40.0,
                    observer_inclination_deg=85.0,
                    disk_model="physical_nt",
                    disk_radial_profile="nt_page_thorne",
                    disk_outer_radius=10.0,
                ),
            )
        )
    return out


def _profile_config(cfg: RenderConfig, profile: str) -> RenderConfig:
    if profile == "baseline_legacy":
        return replace(
            cfg,
            camera_fastpath=False,
            compile_rhs=False,
            mixed_precision=False,
            mps_optimized_kernel=False,
            render_tile_rows=0,
            show_progress_bar=False,
        )
    if profile == "optimized":
        tile_rows = max(64, min(int(cfg.height), int(cfg.height // 4)))
        return replace(
            cfg,
            camera_fastpath=True,
            compile_rhs=True,
            mixed_precision=True,
            mps_optimized_kernel=True,
            adaptive_spatial_sampling=True,
            adaptive_spatial_preview_steps=max(64, min(int(cfg.max_steps // 2), 120)),
            adaptive_spatial_min_scale=0.65,
            adaptive_spatial_quantile=0.78,
            render_tile_rows=tile_rows,
            show_progress_bar=False,
        )
    raise ValueError(f"Unknown profile: {profile}")


def _run_profile(cfg: RenderConfig, warmup: int, repeats: int) -> dict[str, object]:
    timed_runs: list[float] = []
    resolved_device = "unknown"
    total_runs = max(0, warmup) + max(1, repeats)
    for idx in range(total_runs):
        _clear_torch_cache()
        t0 = time.perf_counter()
        tracer = KerrRayTracer(cfg)
        resolved_device = str(tracer.device)
        _ = tracer.render()
        dt = time.perf_counter() - t0
        if idx >= warmup:
            timed_runs.append(float(dt))
            print(f"  run {idx - warmup + 1}/{repeats}: {dt:.3f}s [{resolved_device}]")
        else:
            print(f"  warmup {idx + 1}/{warmup}: {dt:.3f}s [{resolved_device}]")
    avg = float(statistics.mean(timed_runs)) if timed_runs else 0.0
    std = float(statistics.stdev(timed_runs)) if len(timed_runs) > 1 else 0.0
    return {
        "device": resolved_device,
        "runs_sec": timed_runs,
        "mean_sec": avg,
        "std_sec": std,
    }


def _format_markdown_report(
    started_at: str,
    args: argparse.Namespace,
    base_cfg: RenderConfig,
    results: list[dict[str, object]],
) -> str:
    lines: list[str] = []
    lines.append("# KerrTrace Speed Benchmark Report")
    lines.append("")
    lines.append(f"- Timestamp: `{started_at}`")
    lines.append(f"- Resolution: `{args.width}x{args.height}`")
    lines.append(f"- Device request: `{args.device}`")
    lines.append(f"- Dtype: `{args.dtype}`")
    lines.append(f"- Warmup: `{args.warmup}`")
    lines.append(f"- Repeats: `{args.repeats}`")
    lines.append("")
    lines.append("## Base Config")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(asdict(base_cfg), indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("| Scenario | Baseline mean (s) | Optimized mean (s) | Speedup | Device |")
    lines.append("|---|---:|---:|---:|---|")
    for row in results:
        lines.append(
            "| {scenario} | {b:.3f} | {o:.3f} | {s:.2f}x | {dev} |".format(
                scenario=row["scenario"],
                b=float(row["baseline_mean_sec"]),
                o=float(row["optimized_mean_sec"]),
                s=float(row["speedup"]),
                dev=str(row["device"]),
            )
        )
    lines.append("")
    lines.append("## Detail")
    lines.append("")
    for row in results:
        lines.append(f"### {row['scenario']}")
        lines.append("")
        lines.append(f"- Baseline runs (s): `{row['baseline_runs_sec']}`")
        lines.append(f"- Optimized runs (s): `{row['optimized_runs_sec']}`")
        lines.append(f"- Baseline std (s): `{float(row['baseline_std_sec']):.4f}`")
        lines.append(f"- Optimized std (s): `{float(row['optimized_std_sec']):.4f}`")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = _parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_cfg = RenderConfig(
        width=int(args.width),
        height=int(args.height),
        device=str(args.device),
        dtype=str(args.dtype),
        max_steps=int(args.max_steps),
        adaptive_integrator=True,
        disk_model="physical_nt",
        disk_radial_profile="nt_page_thorne",
        show_progress_bar=False,
    ).validated()

    if args.require_gpu:
        resolved = str(base_cfg.resolve_device())
        if resolved == "cpu":
            raise RuntimeError("require-gpu active: resolved device is CPU")

    scenarios = _scenario_configs(base_cfg, args.scenario)
    if not scenarios:
        raise RuntimeError("No scenarios selected")

    results: list[dict[str, object]] = []
    for scenario_name, scenario_cfg in scenarios:
        print(f"\n[scenario] {scenario_name}")
        baseline_cfg = _profile_config(scenario_cfg, "baseline_legacy").validated()
        optimized_cfg = _profile_config(scenario_cfg, "optimized").validated()

        print("[profile] baseline_legacy")
        baseline = _run_profile(baseline_cfg, warmup=int(args.warmup), repeats=int(args.repeats))

        print("[profile] optimized")
        optimized = _run_profile(optimized_cfg, warmup=int(args.warmup), repeats=int(args.repeats))

        b_mean = float(baseline["mean_sec"])
        o_mean = float(optimized["mean_sec"])
        speedup = (b_mean / o_mean) if o_mean > 1.0e-12 else 0.0
        row = {
            "scenario": scenario_name,
            "device": str(optimized["device"]),
            "baseline_runs_sec": baseline["runs_sec"],
            "baseline_mean_sec": b_mean,
            "baseline_std_sec": float(baseline["std_sec"]),
            "optimized_runs_sec": optimized["runs_sec"],
            "optimized_mean_sec": o_mean,
            "optimized_std_sec": float(optimized["std_sec"]),
            "speedup": speedup,
        }
        results.append(row)
        print(
            "=> {name}: baseline={b:.3f}s optimized={o:.3f}s speedup={s:.2f}x [{dev}]".format(
                name=scenario_name,
                b=b_mean,
                o=o_mean,
                s=speedup,
                dev=row["device"],
            )
        )

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"benchmark_render_speed_{stamp}.json"
    md_path = out_dir / f"benchmark_render_speed_{stamp}.md"
    report_payload = {
        "timestamp": stamp,
        "args": vars(args),
        "base_config": asdict(base_cfg),
        "results": results,
    }
    json_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
    md_path.write_text(_format_markdown_report(stamp, args, base_cfg, results), encoding="utf-8")

    print(f"\nSaved JSON report: {json_path}")
    print(f"Saved Markdown report: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
