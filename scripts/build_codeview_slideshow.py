#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import math
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw
from pygments import highlight
from pygments.formatters import ImageFormatter
from pygments.lexers import TextLexer, get_lexer_for_filename


DEFAULT_FILES = [
    "kerrtrace/raytracer.py",
    "kerrtrace/geometry.py",
    "kerrtrace/animation.py",
    "kerrtrace/cli.py",
    "kerrtrace/webui.py",
    "kerrtrace/starship_video.py",
    "tests/test_non_regression.py",
    "README.md",
]


@dataclass(frozen=True)
class CodeChunk:
    path: Path
    start_line: int
    end_line: int
    text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a coding-style slideshow video from KerrTrace source files. "
            "Each slide shows syntax-highlighted code with file and line range."
        )
    )
    parser.add_argument(
        "--files",
        nargs="*",
        default=DEFAULT_FILES,
        help="Files to include (relative to repository root).",
    )
    parser.add_argument("--output-dir", default="out/social_exports", help="Output folder.")
    parser.add_argument("--basename", default="kerrtrace_codeview", help="Output base filename.")
    parser.add_argument("--resolution", default="1920x1080", help="Video resolution (for example 1920x1080).")
    parser.add_argument("--fps", type=int, default=30, help="Output frame rate.")
    parser.add_argument("--per-slide-sec", type=float, default=2.6, help="Seconds shown for each slide.")
    parser.add_argument("--lines-per-slide", type=int, default=28, help="Source lines shown in each slide.")
    parser.add_argument(
        "--max-chunks-per-file",
        type=int,
        default=8,
        help="Maximum sampled chunks per file (large files are sampled uniformly).",
    )
    parser.add_argument("--preset", default="slow", help="x264 preset.")
    parser.add_argument("--crf", type=int, default=17, help="x264 CRF quality.")
    parser.add_argument("--title", default="KerrTrace | Programming Session", help="Overlay title.")
    parser.add_argument("--style", default="monokai", help="Pygments color style.")
    parser.add_argument("--font-name", default="Menlo", help="Code font name used by Pygments.")
    parser.add_argument("--font-size", type=int, default=25, help="Code font size.")
    parser.add_argument(
        "--motion-style",
        choices=["static", "vertical_scroll"],
        default="static",
        help="Slide motion style. vertical_scroll pans code from top to bottom.",
    )
    parser.add_argument(
        "--bg-music",
        default="",
        help="Optional background music file (mp3/wav). The track is looped and trimmed to video length.",
    )
    parser.add_argument("--music-volume", type=float, default=0.20, help="Background music volume.")
    parser.add_argument("--dry-run", action="store_true", help="Print plan and stop before encoding.")
    return parser.parse_args()


def require_binary(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"Required binary missing from PATH: {name}")


def parse_resolution(value: str) -> tuple[int, int]:
    try:
        w_s, h_s = value.lower().split("x", maxsplit=1)
        w = int(w_s)
        h = int(h_s)
    except Exception as exc:  # noqa: BLE001
        raise argparse.ArgumentTypeError(f"Invalid resolution format: {value}") from exc
    if w <= 0 or h <= 0:
        raise argparse.ArgumentTypeError(f"Resolution must be positive: {value}")
    return w, h


def sample_indices(total_count: int, target_count: int) -> list[int]:
    if total_count <= 0:
        return []
    if target_count <= 0:
        return []
    if total_count <= target_count:
        return list(range(total_count))
    if target_count == 1:
        return [total_count // 2]
    indices: set[int] = set()
    for i in range(target_count):
        pos = round(i * (total_count - 1) / (target_count - 1))
        indices.add(int(pos))
    return sorted(indices)


def chunk_file(path: Path, lines_per_slide: int, max_chunks_per_file: int) -> list[CodeChunk]:
    raw_text = path.read_text(encoding="utf-8", errors="replace")
    lines = raw_text.splitlines()
    if not lines:
        return []

    total_chunks = math.ceil(len(lines) / lines_per_slide)
    indices = sample_indices(total_chunks, max_chunks_per_file)
    chunks: list[CodeChunk] = []
    for idx in indices:
        start = idx * lines_per_slide
        end = min(start + lines_per_slide, len(lines))
        chunk_text = "\n".join(lines[start:end]).rstrip() + "\n"
        chunks.append(
            CodeChunk(
                path=path,
                start_line=start + 1,
                end_line=end,
                text=chunk_text,
            )
        )
    return chunks


def pick_lexer(path: Path):
    try:
        return get_lexer_for_filename(path.name)
    except Exception:  # noqa: BLE001
        return TextLexer()


def render_code_image(chunk: CodeChunk, style: str, font_name: str, font_size: int) -> Image.Image:
    formatter = ImageFormatter(
        style=style,
        font_name=font_name,
        font_size=font_size,
        line_numbers=True,
        image_format="PNG",
        image_pad=24,
        line_pad=3,
        line_number_pad=12,
        line_number_bg="#10141f",
        line_number_fg="#8b949e",
        line_number_separator=True,
        background_color="#0d1117",
    )
    highlighted = highlight(chunk.text, pick_lexer(chunk.path), formatter)
    return Image.open(io.BytesIO(highlighted)).convert("RGB")


def fit_inside(src_size: tuple[int, int], dst_size: tuple[int, int]) -> tuple[int, int]:
    src_w, src_h = src_size
    dst_w, dst_h = dst_size
    if src_w <= 0 or src_h <= 0:
        return 1, 1
    scale = min(dst_w / src_w, dst_h / src_h)
    return max(1, int(src_w * scale)), max(1, int(src_h * scale))


def draw_slide(
    chunk: CodeChunk,
    code_img: Image.Image,
    slide_index: int,
    total_slides: int,
    title: str,
    size: tuple[int, int],
    out_path: Path,
    motion_style: str,
    motion_t: float,
    global_progress: float,
) -> None:
    width, height = size
    canvas = Image.new("RGB", (width, height), "#070b14")
    draw = ImageDraw.Draw(canvas)

    top_h = 132
    draw.rectangle([0, 0, width, top_h], fill="#0e1625")
    draw.text((36, 24), title, fill="#e6edf3")
    subtitle = f"{chunk.path.as_posix()}  |  lines {chunk.start_line}-{chunk.end_line}"
    draw.text((36, 78), subtitle, fill="#8b949e")

    margin_x = 52
    margin_bottom = 64
    box = (margin_x, top_h + 20, width - margin_x, height - margin_bottom)
    draw.rounded_rectangle(box, radius=22, fill="#0d1117", outline="#1f2937", width=3)

    content_w = (box[2] - box[0]) - 36
    content_h = (box[3] - box[1]) - 36
    if motion_style == "vertical_scroll":
        # Keep full-width readability and pan vertically through code lines.
        target_w = max(1, content_w)
        target_h = max(1, int(code_img.height * (target_w / max(1, code_img.width))))
        code_resized = code_img.resize((target_w, target_h), Image.Resampling.LANCZOS)

        if target_h <= content_h:
            start_y = 0
        else:
            scroll_max = target_h - content_h
            t = max(0.0, min(1.0, float(motion_t)))
            smooth_t = t * t * (3.0 - 2.0 * t)
            start_y = int(scroll_max * smooth_t)

        cropped = code_resized.crop((0, start_y, target_w, min(target_h, start_y + content_h)))
        if cropped.height < content_h:
            padded = Image.new("RGB", (target_w, content_h), "#0d1117")
            pad_y = (content_h - cropped.height) // 2
            padded.paste(cropped, (0, pad_y))
            cropped = padded

        paste_x = box[0] + 18
        paste_y = box[1] + 18
        canvas.paste(cropped, (paste_x, paste_y))
    else:
        resized_w, resized_h = fit_inside(code_img.size, (content_w, content_h))
        code_resized = code_img.resize((resized_w, resized_h), Image.Resampling.LANCZOS)
        paste_x = box[0] + ((box[2] - box[0] - resized_w) // 2)
        paste_y = box[1] + ((box[3] - box[1] - resized_h) // 2)
        canvas.paste(code_resized, (paste_x, paste_y))

    progress_left = 36
    progress_right = width - 36
    progress_y0 = height - 26
    progress_y1 = height - 14
    draw.rectangle((progress_left, progress_y0, progress_right, progress_y1), fill="#1f2937")
    progress = max(0.0, min(1.0, float(global_progress)))
    fill_x = progress_left + int((progress_right - progress_left) * progress)
    draw.rectangle((progress_left, progress_y0, fill_x, progress_y1), fill="#58a6ff")

    canvas.save(out_path, format="PNG")


def render_video(
    frame_pattern: Path,
    output_path: Path,
    fps: int,
    preset: str,
    crf: int,
    bg_music: Path | None = None,
    music_volume: float = 0.20,
) -> None:
    cmd = ["ffmpeg", "-y", "-framerate", str(fps), "-i", str(frame_pattern)]

    if bg_music is not None:
        cmd.extend(
            [
                "-stream_loop",
                "-1",
                "-i",
                str(bg_music),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-filter:a",
                f"volume={music_volume:.3f}",
                "-c:v",
                "libx264",
                "-preset",
                str(preset),
                "-crf",
                str(crf),
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-shortest",
                "-movflags",
                "+faststart",
                str(output_path),
            ]
        )
    else:
        cmd.extend(
            [
                "-c:v",
                "libx264",
                "-preset",
                str(preset),
                "-crf",
                str(crf),
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(output_path),
            ]
        )

    subprocess.run(cmd, check=True)


def main() -> int:
    args = parse_args()
    require_binary("ffmpeg")

    width, height = parse_resolution(args.resolution)
    repo_root = Path.cwd().resolve()
    output_dir = (repo_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    files: list[Path] = []
    for rel in args.files:
        path = (repo_root / rel).resolve()
        if path.exists() and path.is_file():
            files.append(path)
        else:
            print(f"Skipping missing file: {rel}")

    if not files:
        raise RuntimeError("No valid files were found to build the code slideshow.")

    all_chunks: list[CodeChunk] = []
    for file_path in files:
        chunks = chunk_file(
            path=file_path,
            lines_per_slide=int(args.lines_per_slide),
            max_chunks_per_file=int(args.max_chunks_per_file),
        )
        all_chunks.extend(chunks)

    if not all_chunks:
        raise RuntimeError("No code chunks were generated.")

    total_seconds = len(all_chunks) * float(args.per_slide_sec)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_video = output_dir / f"{args.basename}_{stamp}_{width}x{height}.mp4"
    bg_music: Path | None = None
    if args.bg_music:
        music_path = Path(args.bg_music).resolve()
        if not music_path.exists():
            raise RuntimeError(f"Background music file does not exist: {music_path}")
        bg_music = music_path

    print(f"Files included: {len(files)}")
    print(f"Slides generated: {len(all_chunks)}")
    print(f"Estimated duration: {total_seconds / 60.0:.1f} minutes")
    print(f"Output video: {output_video}")
    print(f"Motion style: {args.motion_style}")
    if bg_music is not None:
        print(f"Background music: {bg_music}")

    if args.dry_run:
        return 0

    with tempfile.TemporaryDirectory(prefix="kerrtrace_code_slides_") as temp_dir:
        temp_path = Path(temp_dir)
        frames_per_slide = max(1, int(round(float(args.per_slide_sec) * int(args.fps))))
        total_frames = len(all_chunks) * frames_per_slide
        frame_index = 0
        for idx, chunk in enumerate(all_chunks):
            code_img = render_code_image(
                chunk=chunk,
                style=str(args.style),
                font_name=str(args.font_name),
                font_size=int(args.font_size),
            )
            for local_frame in range(frames_per_slide):
                if frames_per_slide > 1:
                    motion_t = local_frame / (frames_per_slide - 1)
                else:
                    motion_t = 0.0
                if total_frames > 1:
                    global_progress = frame_index / (total_frames - 1)
                else:
                    global_progress = 1.0
                frame_path = temp_path / f"frame_{frame_index:06d}.png"
                draw_slide(
                    chunk=chunk,
                    code_img=code_img,
                    slide_index=idx,
                    total_slides=len(all_chunks),
                    title=str(args.title),
                    size=(width, height),
                    out_path=frame_path,
                    motion_style=str(args.motion_style),
                    motion_t=motion_t,
                    global_progress=global_progress,
                )
                frame_index += 1

        frame_pattern = temp_path / "frame_%06d.png"
        render_video(
            frame_pattern=frame_pattern,
            output_path=output_video,
            fps=int(args.fps),
            preset=str(args.preset),
            crf=int(args.crf),
            bg_music=bg_music,
            music_volume=float(args.music_volume),
        )

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
