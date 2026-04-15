#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class ImageMeta:
    path: Path
    width: int
    height: int
    sha1: str

    @property
    def area(self) -> int:
        return int(self.width) * int(self.height)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build long social videos from high-quality unique images in a folder. "
            "Outputs YouTube 16:9 and Instagram 9:16 MP4 files."
        )
    )
    parser.add_argument("--input-dir", type=str, default="out", help="Root folder with images (recursive).")
    parser.add_argument("--output-dir", type=str, default="out/social_exports", help="Output folder for videos.")
    parser.add_argument("--per-image-sec", type=float, default=5.0, help="Duration per image in seconds.")
    parser.add_argument("--min-short-edge", type=int, default=720, help="Minimum short edge in pixels.")
    parser.add_argument(
        "--min-area",
        type=int,
        default=1280 * 720,
        help="Minimum image area in pixels (width*height).",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Optional cap on selected images (0 = no cap, longest possible video).",
    )
    parser.add_argument("--fps", type=int, default=30, help="Output frame rate.")
    parser.add_argument("--crf", type=int, default=17, help="x264 quality factor (lower = better quality).")
    parser.add_argument("--preset", type=str, default="slow", help="x264 preset (slow/medium/fast...).")
    parser.add_argument(
        "--basename",
        type=str,
        default="kerrtrace_social",
        help="Base name for output files (timestamp is appended).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Analyze and print plan without encoding.")
    return parser.parse_args()


def _require_binary(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"Required binary not found in PATH: {name}")


def _file_sha1(path: Path) -> str:
    hasher = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _probe_image_dimensions(path: Path) -> tuple[int, int] | None:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "csv=s=x:p=0",
        str(path),
    ]
    try:
        out = subprocess.check_output(cmd, text=True).strip()
        width_s, height_s = out.split("x", maxsplit=1)
        return int(width_s), int(height_s)
    except Exception:
        return None


def _collect_unique_hq_images(
    input_dir: Path,
    min_short_edge: int,
    min_area: int,
    max_images: int,
) -> list[ImageMeta]:
    seen_hashes: set[str] = set()
    selected: list[ImageMeta] = []

    for path in sorted(input_dir.rglob("*")):
        if (not path.is_file()) or (path.suffix.lower() not in IMAGE_EXTENSIONS):
            continue

        dims = _probe_image_dimensions(path)
        if dims is None:
            continue
        width, height = dims
        if min(width, height) < min_short_edge:
            continue
        if (width * height) < min_area:
            continue

        digest = _file_sha1(path)
        if digest in seen_hashes:
            continue
        seen_hashes.add(digest)

        selected.append(ImageMeta(path=path, width=width, height=height, sha1=digest))
        if max_images > 0 and len(selected) >= max_images:
            break

    # Keep temporal flow for slideshow while preserving quality filter and dedupe.
    selected.sort(key=lambda item: str(item.path))
    return selected


def _concat_escape(path: Path) -> str:
    return str(path).replace("'", "'\\''")


def _build_concat_manifest(images: list[ImageMeta], per_image_sec: float, output_path: Path) -> None:
    if not images:
        raise ValueError("No images available to build concat manifest.")
    lines: list[str] = []
    for image in images:
        lines.append(f"file '{_concat_escape(image.path.resolve())}'")
        lines.append(f"duration {per_image_sec:.6f}")
    lines.append(f"file '{_concat_escape(images[-1].path.resolve())}'")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _render_video(
    concat_manifest: Path,
    output_path: Path,
    width: int,
    height: int,
    fps: int,
    preset: str,
    crf: int,
) -> None:
    vf = (
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,"
        "setsar=1,format=yuv420p"
    )
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_manifest),
        "-vf",
        vf,
        "-r",
        str(fps),
        "-c:v",
        "libx264",
        "-preset",
        str(preset),
        "-crf",
        str(crf),
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)


def main() -> int:
    args = _parse_args()
    _require_binary("ffprobe")
    _require_binary("ffmpeg")

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        raise RuntimeError(f"Input directory does not exist: {input_dir}")

    images = _collect_unique_hq_images(
        input_dir=input_dir,
        min_short_edge=int(args.min_short_edge),
        min_area=int(args.min_area),
        max_images=int(args.max_images),
    )
    if not images:
        raise RuntimeError(
            "No valid images found after filtering. "
            "Try lowering --min-short-edge or --min-area."
        )

    total_seconds = len(images) * float(args.per_image_sec)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    youtube_out = output_dir / f"{args.basename}_{stamp}_youtube_1080p.mp4"
    instagram_out = output_dir / f"{args.basename}_{stamp}_instagram_1080x1920.mp4"

    print(f"Input dir: {input_dir}")
    print(f"Selected images (unique + HQ): {len(images)}")
    print(f"Estimated duration: {total_seconds/60.0:.1f} minutes")
    print(f"YouTube output: {youtube_out}")
    print(f"Instagram output: {instagram_out}")

    if args.dry_run:
        return 0

    with tempfile.TemporaryDirectory(prefix="kerrtrace_social_") as temp_dir:
        manifest = Path(temp_dir) / "concat.txt"
        _build_concat_manifest(images=images, per_image_sec=float(args.per_image_sec), output_path=manifest)

        print("Rendering YouTube 16:9 video...")
        _render_video(
            concat_manifest=manifest,
            output_path=youtube_out,
            width=1920,
            height=1080,
            fps=int(args.fps),
            preset=str(args.preset),
            crf=int(args.crf),
        )

        print("Rendering Instagram 9:16 video...")
        _render_video(
            concat_manifest=manifest,
            output_path=instagram_out,
            width=1080,
            height=1920,
            fps=int(args.fps),
            preset=str(args.preset),
            crf=int(args.crf),
        )

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
