from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime
import json
import os
from pathlib import Path
import re
import select
import subprocess
import sys
from typing import Any

import streamlit as st

if os.name == "posix":
    import pty
else:
    pty = None

try:
    from .config import RenderConfig
except ImportError:
    from kerrtrace.config import RenderConfig


QUALITY_PRESETS: dict[str, tuple[int, int]] = {
    "144p (256x144)": (256, 144),
    "288p (512x288)": (512, 288),
    "480p (854x480)": (854, 480),
    "720p (1280x720)": (1280, 720),
    "1080p (1920x1080)": (1920, 1080),
    "2K (2560x1440)": (2560, 1440),
    "4K (3840x2160)": (3840, 2160),
}

CHOICE_FIELDS: dict[str, list[str]] = {
    "coordinate_system": ["boyer_lindquist", "kerr_schild", "generalized_doran"],
    "metric_model": [
        "schwarzschild",
        "kerr",
        "reissner_nordstrom",
        "kerr_newman",
        "schwarzschild_de_sitter",
        "kerr_de_sitter",
        "reissner_nordstrom_de_sitter",
        "kerr_newman_de_sitter",
    ],
    "disk_model": ["physical_nt", "legacy"],
    "disk_radial_profile": ["nt_proxy", "nt_page_thorne"],
    "background_mode": ["procedural", "hdri"],
    "background_projection": ["cubemap", "equirectangular"],
    "kerr_schild_mode": ["off", "fsal_only", "analytic"],
    "device": ["auto", "cpu", "cuda", "mps"],
    "dtype": ["float32", "float64"],
    "progress_backend": ["manual", "tqdm", "auto"],
    "video_codec": ["h264", "h265_10bit"],
    "tone_mapper": ["reinhard", "aces"],
    "postprocess_pipeline": ["off", "gargantua"],
}

AUTHOR_SIGNATURE = "Iman Rosignoli"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".gif"}
MEDIA_SUFFIXES = IMAGE_SUFFIXES | VIDEO_SUFFIXES

LANGUAGE_OPTIONS: dict[str, str] = {
    "it": "Italiano",
    "en": "English",
    "es": "Español",
    "ro": "Română",
    "ru": "Русский",
    "zh": "中文(简体)",
    "fr": "Français",
    "de": "Deutsch",
    "pt": "Português",
}

I18N: dict[str, dict[str, str]] = {
    "en": {
        "page_title": "KerrTrace WebUI",
        "title": "KerrTrace WebUI",
        "author_label": "Author",
        "intro_caption": "This interface creates a RenderConfig JSON and runs `python -m kerrtrace --config ...`.",
        "last_output": "Last output preview",
        "manual_open": "Open file manually",
        "manual_path": "File path (absolute or relative to workspace)",
        "open_file": "Open file",
        "use_last_output": "Use last output",
        "clear": "Clear",
        "empty_path": "Enter a file path.",
        "missing_file": "File not found:",
        "unsupported_file": "Unsupported file extension for preview:",
        "resolved_file": "Resolved file:",
        "run_header": "Run",
        "language_label": "Language",
        "workspace_label": "Workspace",
        "require_gpu": "Require GPU (--require-gpu)",
        "upload_json": "Upload JSON config",
        "json_not_object": "Uploaded JSON is not an object.",
        "json_invalid": "Invalid JSON:",
        "mode_header": "Mode",
        "mode_label": "Simulation type",
        "mode_single_frame": "Single Frame",
        "mode_video": "Video",
        "quality_header": "Quality / Resolution",
        "quality_preset": "Quality preset",
        "resolution_set": "Resolution set to",
        "run_live": "Run simulation (live)",
        "run_bg": "Run in background",
        "bg_job": "Background job",
        "job_running": "Running",
        "job_completed": "Job completed",
        "job_failed": "Job failed",
        "refresh_monitor": "Refresh monitor",
        "stop_job": "Stop job",
        "clear_job": "Clear job state",
        "output_not_found_auto": "Output not found automatically. Check the path in the log.",
        "cmd_launched": "Command launched:",
        "cfg_used": "Config JSON used:",
        "sim_completed": "Simulation completed. Log:",
        "sim_failed": "Simulation failed",
        "output_not_found_expected": "Output not found automatically. Expected:",
    },
    "es": {
        "author_label": "Autor",
        "intro_caption": "Esta interfaz genera un JSON RenderConfig y ejecuta `python -m kerrtrace --config ...`.",
        "last_output": "Vista previa de la última salida",
        "manual_open": "Abrir archivo manualmente",
        "manual_path": "Ruta del archivo (absoluta o relativa al workspace)",
        "open_file": "Abrir archivo",
        "use_last_output": "Usar última salida",
        "clear": "Limpiar",
        "empty_path": "Introduce una ruta de archivo.",
        "missing_file": "Archivo no encontrado:",
        "unsupported_file": "Extensión no soportada para vista previa:",
        "resolved_file": "Archivo resuelto:",
        "run_header": "Ejecución",
        "language_label": "Idioma",
        "workspace_label": "Workspace",
        "require_gpu": "Requerir GPU (--require-gpu)",
        "upload_json": "Subir configuración JSON",
        "json_not_object": "El JSON cargado no es un objeto.",
        "json_invalid": "JSON inválido:",
        "mode_header": "Modo",
        "mode_label": "Tipo de simulación",
        "mode_single_frame": "Fotograma único",
        "mode_video": "Video",
        "quality_header": "Calidad / Resolución",
        "quality_preset": "Preset de calidad",
        "resolution_set": "Resolución establecida en",
        "run_live": "Lanzar simulación (en vivo)",
        "run_bg": "Lanzar en segundo plano",
    },
    "ro": {
        "author_label": "Autor",
        "manual_open": "Deschide fișier manual",
        "manual_path": "Cale fișier (absolută sau relativă la workspace)",
        "open_file": "Deschide fișier",
        "use_last_output": "Folosește ultimul output",
        "clear": "Șterge",
        "run_header": "Rulare",
        "language_label": "Limbă",
        "mode_header": "Mod",
        "mode_label": "Tip simulare",
        "mode_single_frame": "Cadru unic",
        "mode_video": "Video",
        "run_live": "Rulează simularea (live)",
        "run_bg": "Rulează în fundal",
    },
    "ru": {
        "author_label": "Автор",
        "manual_open": "Открыть файл вручную",
        "manual_path": "Путь к файлу (абсолютный или относительно workspace)",
        "open_file": "Открыть файл",
        "use_last_output": "Использовать последний вывод",
        "clear": "Очистить",
        "run_header": "Запуск",
        "language_label": "Язык",
        "mode_header": "Режим",
        "mode_label": "Тип симуляции",
        "mode_single_frame": "Один кадр",
        "mode_video": "Видео",
        "run_live": "Запустить симуляцию (live)",
        "run_bg": "Запустить в фоне",
    },
    "zh": {
        "author_label": "作者",
        "manual_open": "手动打开文件",
        "manual_path": "文件路径（绝对路径或相对 workspace）",
        "open_file": "打开文件",
        "use_last_output": "使用最近输出",
        "clear": "清除",
        "run_header": "运行",
        "language_label": "语言",
        "mode_header": "模式",
        "mode_label": "模拟类型",
        "mode_single_frame": "单帧",
        "mode_video": "视频",
        "run_live": "实时运行模拟",
        "run_bg": "后台运行",
    },
    "fr": {
        "author_label": "Auteur",
        "language_label": "Langue",
        "manual_open": "Ouvrir un fichier manuellement",
    },
    "de": {
        "author_label": "Autor",
        "language_label": "Sprache",
        "manual_open": "Datei manuell öffnen",
    },
    "pt": {
        "author_label": "Autor",
        "language_label": "Idioma",
        "manual_open": "Abrir ficheiro manualmente",
    },
}


def tr(lang: str, key: str, default: str) -> str:
    return I18N.get(lang, {}).get(key, default)


def _default_python() -> str:
    project_python = Path(".venv/bin/python")
    if project_python.exists():
        return str(project_python)
    return sys.executable


def _safe_choice(options: list[str], value: str) -> str:
    if value in options:
        return value
    return options[0]


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _run_command_live(
    cmd: list[str],
    cwd: Path,
    log_placeholder: Any,
    progress_widget: Any | None = None,
) -> tuple[int, str]:
    chunks: list[str] = []
    max_log_chars = 120_000
    progress_re = re.compile(r"(\d+)\s*/\s*(\d+)")
    pending = ""
    last_progress_key: tuple[int, int] | None = None

    def _push(text: str) -> None:
        chunks.append(text)
        joined = "".join(chunks)
        if len(joined) > max_log_chars:
            joined = joined[-max_log_chars:]
            chunks[:] = [joined]
        log_placeholder.code(joined, language="bash")

    def _process_line(line: str, from_carriage: bool) -> None:
        nonlocal last_progress_key
        if not line:
            return
        stripped = line.strip()
        if stripped.startswith("Render rows"):
            matches = progress_re.findall(stripped)
            if matches:
                done_s, total_s = matches[-1]
                key = (int(done_s), int(total_s))
                # Ignore ticker refreshes when progress did not advance.
                if key == last_progress_key:
                    return
                last_progress_key = key
                if progress_widget is not None and key[1] > 0:
                    ratio = max(0.0, min(1.0, float(key[0]) / float(key[1])))
                    progress_widget.progress(
                        ratio,
                        text=f"Render rows: {key[0]}/{key[1]} ({ratio * 100.0:.1f}%)",
                    )
            _push(stripped + "\n")
            return
        _push(line + "\n")

    def _process_chunk_text(text: str) -> None:
        nonlocal pending
        pending += text
        start = 0
        for idx, ch in enumerate(pending):
            if ch == "\r" or ch == "\n":
                line = pending[start:idx]
                _process_line(line, from_carriage=(ch == "\r"))
                start = idx + 1
        pending = pending[start:]

    if os.name == "posix":
        master_fd, slave_fd = pty.openpty()
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=slave_fd,
            stderr=slave_fd,
            text=False,
            close_fds=True,
        )
        os.close(slave_fd)
        try:
            while True:
                ready, _, _ = select.select([master_fd], [], [], 0.20)
                if ready:
                    data = os.read(master_fd, 4096)
                    if data:
                        txt = data.decode("utf-8", errors="replace")
                        _process_chunk_text(txt)
                if proc.poll() is not None:
                    if not ready:
                        break
            rc = int(proc.wait())
        finally:
            os.close(master_fd)
    else:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            _process_chunk_text(line)
        rc = int(proc.wait())

    if pending:
        _process_line(pending, from_carriage=False)

    if progress_widget is not None and last_progress_key is not None and last_progress_key[1] > 0:
        done, total = last_progress_key
        ratio = max(0.0, min(1.0, float(done) / float(total)))
        progress_widget.progress(
            ratio,
            text=f"Render rows: {done}/{total} ({ratio * 100.0:.1f}%)",
        )

    return rc, "".join(chunks)


def _parse_patch(patch_text: str) -> dict[str, Any]:
    if not patch_text.strip():
        return {}
    patch = json.loads(patch_text)
    if not isinstance(patch, dict):
        raise ValueError("Override JSON must be an object")
    return patch


def _show_output_media(out_file: Path) -> None:
    suffix = out_file.suffix.lower()
    st.subheader("Risultato")
    if suffix in IMAGE_SUFFIXES:
        try:
            st.image(out_file.read_bytes(), caption=str(out_file))
        except Exception:
            st.image(str(out_file), caption=str(out_file))
        return
    if suffix in VIDEO_SUFFIXES:
        mime = "video/mp4"
        if suffix == ".gif":
            mime = "image/gif"
        elif suffix == ".mov":
            mime = "video/quicktime"
        elif suffix == ".mkv":
            mime = "video/x-matroska"
        try:
            data = out_file.read_bytes()
            st.video(data, format=mime)
        except Exception:
            st.video(str(out_file))
        st.caption(str(out_file))
        return
    st.caption(str(out_file))


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def _extract_output_path_from_log(log_text: str, workspace_path: Path) -> Path | None:
    if not log_text:
        return None
    patterns = (
        re.compile(r"Saved image:\s*(.+)$"),
        re.compile(r"Saved animation:\s*(.+)$"),
        re.compile(r"Saved charged-particle animation:\s*(.+)$"),
        re.compile(r"Saved raytraced single-particle animation:\s*(.+)$"),
        re.compile(r"Auto output path:\s*(.+)$"),
    )
    lines = [_strip_ansi(ln).strip() for ln in log_text.splitlines()]
    for line in reversed(lines):
        for pat in patterns:
            m = pat.search(line)
            if m is None:
                continue
            raw = m.group(1).strip().strip("'").strip('"')
            p = Path(raw).expanduser()
            if not p.is_absolute():
                p = workspace_path / p
            return p.resolve()
    return None


def _tail_text_file(path: Path, max_chars: int = 120_000) -> str:
    if not path.exists():
        return ""
    try:
        max_bytes = max(4096, max_chars * 4)
        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - max_bytes), os.SEEK_SET)
            data = f.read()
        txt = data.decode("utf-8", errors="replace")
        if len(txt) > max_chars:
            txt = txt[-max_chars:]
        return txt
    except Exception:
        try:
            txt = path.read_text(encoding="utf-8", errors="replace")
            if len(txt) > max_chars:
                txt = txt[-max_chars:]
            return txt
        except Exception:
            return ""


def _latest_media_in_out(workspace_path: Path, suffix: str | None = None) -> Path | None:
    out_dir = (workspace_path / "out").resolve()
    if not out_dir.exists():
        return None
    if suffix is not None and suffix.lower() not in MEDIA_SUFFIXES:
        suffix = None
    try:
        if suffix is not None:
            candidates = sorted(
                out_dir.glob(f"**/*{suffix.lower()}"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if candidates:
                return candidates[0].resolve()
        media_candidates = sorted(
            (p for p in out_dir.glob("**/*") if p.is_file() and p.suffix.lower() in MEDIA_SUFFIXES),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if media_candidates:
            return media_candidates[0].resolve()
    except Exception:
        return None
    return None


def _resolve_output_file(
    log_text: str,
    workspace_path: Path,
    out_hint: Path,
) -> Path | None:
    out_from_log = _extract_output_path_from_log(log_text, workspace_path)
    candidates: list[Path] = []
    if out_from_log is not None:
        candidates.append(out_from_log)
    if out_hint:
        out_hint_abs = out_hint if out_hint.is_absolute() else (workspace_path / out_hint)
        candidates.append(out_hint_abs.resolve())
    for cand in candidates:
        if cand.exists():
            return cand.resolve()
    preferred_suffix = out_hint.suffix.lower() if out_hint.suffix else None
    return _latest_media_in_out(workspace_path=workspace_path, suffix=preferred_suffix)


def _resolve_manual_media_path(raw_path: str, workspace_path: Path) -> tuple[Path | None, str]:
    raw = str(raw_path or "").strip()
    if not raw:
        return None, "empty"
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = workspace_path / path
    try:
        path = path.resolve()
    except Exception:
        path = path.absolute()
    if not path.exists():
        return None, "missing"
    if path.suffix.lower() not in MEDIA_SUFFIXES:
        return None, "unsupported"
    return path, ""


def main() -> None:
    if "ui_lang" not in st.session_state:
        st.session_state["ui_lang"] = "it"
    lang = str(st.session_state.get("ui_lang", "it"))
    if lang not in LANGUAGE_OPTIONS:
        lang = "it"
        st.session_state["ui_lang"] = "it"

    st.set_page_config(page_title=tr(lang, "page_title", "KerrTrace WebUI"), layout="wide")
    st.title(tr(lang, "title", "KerrTrace WebUI"))
    st.markdown(f"**{tr(lang, 'author_label', 'Autore')}:** `{AUTHOR_SIGNATURE}`")
    st.caption(
        tr(
            lang,
            "intro_caption",
            "Questa interfaccia genera un JSON RenderConfig e lancia `python -m kerrtrace --config ...`.",
        )
    )
    if "last_output_path" not in st.session_state:
        st.session_state["last_output_path"] = ""
    if "async_proc" not in st.session_state:
        st.session_state["async_proc"] = None
    if "async_meta" not in st.session_state:
        st.session_state["async_meta"] = {}
    last_output_raw = str(st.session_state.get("last_output_path") or "").strip()
    if last_output_raw:
        last_output = Path(last_output_raw)
        if last_output.exists():
            with st.expander(tr(lang, "last_output", "Anteprima ultimo output"), expanded=True):
                _show_output_media(last_output)

    async_proc = st.session_state.get("async_proc")
    async_meta = st.session_state.get("async_meta") or {}
    if async_proc is not None and isinstance(async_meta, dict) and async_meta:
        log_path = Path(str(async_meta.get("log_path", "")))
        workspace_async = Path(str(async_meta.get("workspace", str(Path.cwd()))))
        out_hint = Path(str(async_meta.get("output_hint", "out/webui_frame.png")))
        started_at = str(async_meta.get("started_at", ""))
        cfg_async = str(async_meta.get("cfg_path", ""))
        rc = async_proc.poll()
        running = rc is None
        with st.expander(tr(lang, "bg_job", "Job in background"), expanded=True):
            if running:
                st.info(
                    f"{tr(lang, 'job_running', 'In esecuzione')} (PID {getattr(async_proc, 'pid', 'n/a')}) "
                    f"- started: {started_at}"
                )
            else:
                if int(rc) == 0:
                    st.success(f"{tr(lang, 'job_completed', 'Job completato')} (exit={rc})")
                else:
                    st.error(f"{tr(lang, 'job_failed', 'Job terminato con errore')} (exit={rc})")
            if cfg_async:
                st.caption(f"Config: {cfg_async}")
            if log_path:
                st.caption(f"Log: {log_path}")
            tail_txt = _tail_text_file(log_path)
            if tail_txt:
                st.code(tail_txt, language="bash")

            c_job_1, c_job_2 = st.columns(2)
            with c_job_1:
                if running:
                    if st.button(tr(lang, "refresh_monitor", "Aggiorna monitor")):
                        st.rerun()
            with c_job_2:
                if running and st.button(tr(lang, "stop_job", "Interrompi job")):
                    try:
                        async_proc.terminate()
                    except Exception:
                        pass
                    st.rerun()

            if not running:
                out_file = _resolve_output_file(
                    log_text=tail_txt,
                    workspace_path=workspace_async,
                    out_hint=out_hint,
                )
                if out_file is not None and out_file.exists():
                    st.session_state["last_output_path"] = str(out_file)
                    _show_output_media(out_file)
                else:
                    st.warning(tr(lang, "output_not_found_auto", "Output non trovato automaticamente. Controlla il path nel log."))
                if st.button(tr(lang, "clear_job", "Pulisci stato job")):
                    st.session_state["async_proc"] = None
                    st.session_state["async_meta"] = {}
                    st.rerun()

    default_cfg = asdict(RenderConfig())
    supports_adaptive_spatial = "adaptive_spatial_sampling" in default_cfg
    loaded_cfg: dict[str, Any] = {}

    with st.sidebar:
        st.header(tr(lang, "run_header", "Run"))
        st.caption(f"{tr(lang, 'author_label', 'Autore')}: {AUTHOR_SIGNATURE}")
        lang_options = list(LANGUAGE_OPTIONS.keys())
        lang_idx = lang_options.index(lang) if lang in lang_options else 0
        lang = st.selectbox(
            tr(lang, "language_label", "Lingua / Language"),
            options=lang_options,
            index=lang_idx,
            format_func=lambda code: LANGUAGE_OPTIONS.get(code, code),
        )
        st.session_state["ui_lang"] = lang
        python_exec = st.text_input("Python executable", value=_default_python())
        workspace = st.text_input(tr(lang, "workspace_label", "Workspace"), value=str(Path.cwd()))
        require_gpu = st.checkbox(tr(lang, "require_gpu", "Richiedi GPU (--require-gpu)"), value=True)
        uploaded = st.file_uploader(tr(lang, "upload_json", "Carica config JSON"), type=["json"])
        if uploaded is not None:
            try:
                loaded_cfg = json.loads(uploaded.read().decode("utf-8"))
                if not isinstance(loaded_cfg, dict):
                    st.error(tr(lang, "json_not_object", "Il JSON caricato non è un oggetto."))
                    loaded_cfg = {}
            except Exception as exc:
                st.error(f"{tr(lang, 'json_invalid', 'JSON non valido')}: {exc}")
                loaded_cfg = {}

    cfg_seed = dict(default_cfg)
    cfg_seed.update(loaded_cfg)
    cfg_seed.setdefault("adaptive_spatial_sampling", bool(default_cfg.get("adaptive_spatial_sampling", False)))
    cfg_seed.setdefault("adaptive_spatial_preview_steps", int(default_cfg.get("adaptive_spatial_preview_steps", 96)))
    cfg_seed.setdefault("adaptive_spatial_min_scale", float(default_cfg.get("adaptive_spatial_min_scale", 0.65)))
    cfg_seed.setdefault("adaptive_spatial_quantile", float(default_cfg.get("adaptive_spatial_quantile", 0.78)))

    workspace_path_preview = Path(workspace).expanduser().resolve()
    with st.expander(tr(lang, "manual_open", "Apri file manuale"), expanded=False):
        manual_default = str(
            st.session_state.get(
                "manual_open_path_input",
                st.session_state.get("manual_open_path", ""),
            )
        )
        manual_input = st.text_input(
            tr(lang, "manual_path", "Percorso file (assoluto o relativo al workspace)"),
            value=manual_default,
            key="manual_open_path_input",
        )
        c_open_1, c_open_2, c_open_3 = st.columns(3)
        with c_open_1:
            open_manual = st.button(tr(lang, "open_file", "Apri file"))
        with c_open_2:
            use_last = st.button(tr(lang, "use_last_output", "Usa ultimo output"))
        with c_open_3:
            clear_manual = st.button(tr(lang, "clear", "Pulisci"))
        if use_last:
            st.session_state["manual_open_path_input"] = str(st.session_state.get("last_output_path") or "")
            st.rerun()
        if clear_manual:
            st.session_state["manual_open_path_input"] = ""
            st.rerun()
        if open_manual:
            resolved, reason = _resolve_manual_media_path(manual_input, workspace_path_preview)
            if resolved is not None:
                st.session_state["manual_open_path"] = str(resolved)
                st.session_state["manual_open_path_input"] = str(resolved)
                st.session_state["last_output_path"] = str(resolved)
                st.caption(f"{tr(lang, 'resolved_file', 'File risolto')}: `{resolved}`")
                _show_output_media(resolved)
            elif reason == "empty":
                st.warning(tr(lang, "empty_path", "Inserisci un percorso file."))
            elif reason == "missing":
                st.error(f"{tr(lang, 'missing_file', 'File non trovato')}: `{manual_input}`")
            elif reason == "unsupported":
                st.error(f"{tr(lang, 'unsupported_file', 'Estensione non supportata per preview')}: `{manual_input}`")

    st.subheader(tr(lang, "mode_header", "Modalità"))
    mode = st.radio(
        tr(lang, "mode_label", "Tipo simulazione"),
        ["single_frame", "video"],
        format_func=lambda v: tr(lang, f"mode_{v}", "Single Frame" if v == "single_frame" else "Video"),
        horizontal=True,
    )

    st.subheader(tr(lang, "quality_header", "Qualità / Risoluzione"))
    reverse_quality = {v: k for k, v in QUALITY_PRESETS.items()}
    current_wh = (int(cfg_seed["width"]), int(cfg_seed["height"]))
    preset_labels = ["Custom"] + list(QUALITY_PRESETS.keys())
    default_preset = reverse_quality.get(current_wh, "Custom")
    preset = st.selectbox(tr(lang, "quality_preset", "Preset qualità"), options=preset_labels, index=preset_labels.index(default_preset))
    if preset == "Custom":
        width = st.number_input("Width", min_value=64, max_value=5000, value=int(cfg_seed["width"]), step=1)
        height = st.number_input("Height", min_value=64, max_value=5000, value=int(cfg_seed["height"]), step=1)
    else:
        width, height = QUALITY_PRESETS[preset]
        st.info(f"{tr(lang, 'resolution_set', 'Risoluzione impostata a')} {width}x{height}")

    c1, c2, c3 = st.columns(3)
    with c1:
        output_default = "out/webui_frame.png" if mode == "single_frame" else "out/webui_video.mp4"
        output_seed = loaded_cfg.get("output", output_default)
        output_path = st.text_input("Output file", value=str(output_seed))
        fov_deg = st.number_input("FOV (deg)", value=float(cfg_seed["fov_deg"]), step=0.1, format="%.3f")
        coordinate_system = st.selectbox(
            "Coordinate system",
            options=CHOICE_FIELDS["coordinate_system"],
            index=CHOICE_FIELDS["coordinate_system"].index(
                _safe_choice(CHOICE_FIELDS["coordinate_system"], str(cfg_seed["coordinate_system"]))
            ),
        )
        metric_model = st.selectbox(
            "Metric model",
            options=CHOICE_FIELDS["metric_model"],
            index=CHOICE_FIELDS["metric_model"].index(
                _safe_choice(CHOICE_FIELDS["metric_model"], str(cfg_seed["metric_model"]))
            ),
        )
    with c2:
        spin_default = max(-1.0, min(1.0, float(cfg_seed["spin"])))
        charge_default = max(-1.0, min(1.0, float(cfg_seed["charge"])))
        theta_default = _clamp(float(cfg_seed["observer_inclination_deg"]), 0.0, 180.0)
        spin = st.number_input(
            "Spin a",
            min_value=-1.0,
            max_value=1.0,
            value=spin_default,
            step=0.01,
            format="%.6f",
        )
        charge = st.number_input(
            "Charge Q",
            min_value=-1.0,
            max_value=1.0,
            value=charge_default,
            step=0.01,
            format="%.6f",
        )
        cosmological_constant = st.number_input(
            "Lambda",
            value=float(cfg_seed["cosmological_constant"]),
            step=0.000001,
            format="%.9f",
        )
        observer_radius = st.number_input("Observer radius", value=float(cfg_seed["observer_radius"]), step=0.5)
        observer_inclination_deg = st.number_input(
            "Observer inclination (deg)",
            min_value=0.0,
            max_value=180.0,
            value=theta_default,
            step=0.5,
        )
    with c3:
        phi_default = _clamp(float(cfg_seed["observer_azimuth_deg"]), 0.0, 360.0)
        observer_azimuth_deg = st.number_input(
            "Observer azimuth (deg)",
            min_value=0.0,
            max_value=360.0,
            value=phi_default,
            step=0.5,
        )
        roll_default = _clamp(float(cfg_seed["observer_roll_deg"]), 0.0, 360.0)
        observer_roll_deg = st.number_input(
            "Observer roll (deg)",
            min_value=0.0,
            max_value=360.0,
            value=roll_default,
            step=0.5,
        )
        disk_model = st.selectbox(
            "Disk model",
            options=CHOICE_FIELDS["disk_model"],
            index=CHOICE_FIELDS["disk_model"].index(_safe_choice(CHOICE_FIELDS["disk_model"], str(cfg_seed["disk_model"]))),
        )
        disk_radial_profile = st.selectbox(
            "Disk radial profile",
            options=CHOICE_FIELDS["disk_radial_profile"],
            index=CHOICE_FIELDS["disk_radial_profile"].index(
                _safe_choice(CHOICE_FIELDS["disk_radial_profile"], str(cfg_seed["disk_radial_profile"]))
            ),
        )
        disk_outer_radius = st.number_input("Disk outer radius", value=float(cfg_seed["disk_outer_radius"]), step=0.5)

    st.subheader("Disco e rendering")
    perf_profile = st.selectbox(
        "Performance profile",
        options=[
            "Manual",
            "GPU Balanced (Recommended)",
            "Fast Preview",
            "High Fidelity",
        ],
        index=1,
        help=(
            "Manual: non forza nulla. GPU Balanced: abilita compile+mixed precision e tiling ottimizzato. "
            "Fast Preview: riduce costo per preview rapida. High Fidelity: privilegia stabilità numerica."
        ),
    )
    d1, d2, d3, d4 = st.columns(4)
    with d1:
        use_default_rin = st.checkbox("r_in = default (ISCO)", value=cfg_seed["disk_inner_radius"] is None)
        disk_inner_radius = None
        if not use_default_rin:
            disk_inner_radius = st.number_input(
                "Disk inner radius",
                value=float(cfg_seed["disk_inner_radius"] or 6.0),
                step=0.1,
            )
        disk_emission_gain = st.number_input("Disk emission gain", value=float(cfg_seed["disk_emission_gain"]), step=0.5)
    with d2:
        max_steps = st.number_input("Max steps", min_value=16, value=int(cfg_seed["max_steps"]), step=10)
        step_size = st.number_input("Step size", min_value=0.001, value=float(cfg_seed["step_size"]), step=0.01)
        adaptive_integrator = st.checkbox("Adaptive integrator", value=bool(cfg_seed["adaptive_integrator"]))
    with d3:
        device = st.selectbox(
            "Device",
            options=CHOICE_FIELDS["device"],
            index=CHOICE_FIELDS["device"].index(_safe_choice(CHOICE_FIELDS["device"], str(cfg_seed["device"]))),
        )
        dtype = st.selectbox(
            "Dtype",
            options=CHOICE_FIELDS["dtype"],
            index=CHOICE_FIELDS["dtype"].index(_safe_choice(CHOICE_FIELDS["dtype"], str(cfg_seed["dtype"]))),
        )
        show_progress_bar = st.checkbox("Show progress bar", value=bool(cfg_seed["show_progress_bar"]))
        progress_backend = st.selectbox(
            "Progress backend",
            options=CHOICE_FIELDS["progress_backend"],
            index=CHOICE_FIELDS["progress_backend"].index(
                _safe_choice(CHOICE_FIELDS["progress_backend"], str(cfg_seed.get("progress_backend", "manual")))
            ),
            help="manual: barra custom; tqdm: barra tqdm; auto: usa tqdm quando disponibile",
        )
    with d4:
        mps_optimized_kernel = st.checkbox("MPS optimized kernel", value=bool(cfg_seed["mps_optimized_kernel"]))
        compile_rhs = st.checkbox("Compile RHS", value=bool(cfg_seed["compile_rhs"]))
        mixed_precision = st.checkbox("Mixed precision", value=bool(cfg_seed["mixed_precision"]))
        camera_fastpath = st.checkbox("Camera fastpath", value=bool(cfg_seed["camera_fastpath"]))
        adaptive_spatial_sampling = bool(cfg_seed.get("adaptive_spatial_sampling", False))
        adaptive_spatial_preview_steps = int(cfg_seed.get("adaptive_spatial_preview_steps", 96))
        adaptive_spatial_min_scale = float(cfg_seed.get("adaptive_spatial_min_scale", 0.65))
        adaptive_spatial_quantile = float(cfg_seed.get("adaptive_spatial_quantile", 0.78))
        if supports_adaptive_spatial:
            adaptive_spatial_sampling = st.checkbox(
                "Adaptive spatial sampling",
                value=adaptive_spatial_sampling,
            )
            adaptive_spatial_preview_steps = st.number_input(
                "Adaptive preview steps",
                min_value=16,
                max_value=10000,
                value=adaptive_spatial_preview_steps,
                step=8,
            )
            adaptive_spatial_min_scale = st.number_input(
                "Adaptive min scale",
                min_value=0.10,
                max_value=1.00,
                value=adaptive_spatial_min_scale,
                step=0.05,
            )
            adaptive_spatial_quantile = st.number_input(
                "Adaptive quantile",
                min_value=0.50,
                max_value=0.995,
                value=adaptive_spatial_quantile,
                step=0.01,
                format="%.3f",
            )
        render_tile_rows = st.number_input(
            "Render tile rows (0=auto)",
            min_value=0,
            max_value=2048,
            value=int(cfg_seed["render_tile_rows"]),
            step=8,
        )
        postprocess_pipeline = st.selectbox(
            "Postprocess pipeline",
            options=CHOICE_FIELDS["postprocess_pipeline"],
            index=CHOICE_FIELDS["postprocess_pipeline"].index(
                _safe_choice(CHOICE_FIELDS["postprocess_pipeline"], str(cfg_seed["postprocess_pipeline"]))
            ),
        )
        gargantua_look_strength = st.slider(
            "Gargantua look strength",
            min_value=0.0,
            max_value=2.0,
            value=float(cfg_seed["gargantua_look_strength"]),
            step=0.05,
        )

    if perf_profile == "GPU Balanced (Recommended)":
        compile_rhs = True
        mixed_precision = True
        camera_fastpath = True
        if supports_adaptive_spatial:
            adaptive_spatial_sampling = True
        if device in {"mps", "auto"}:
            mps_optimized_kernel = True
        if int(render_tile_rows) <= 0 and int(height) >= 256:
            render_tile_rows = max(64, min(256, int(height // 4)))
    elif perf_profile == "Fast Preview":
        compile_rhs = True
        mixed_precision = True
        camera_fastpath = True
        if supports_adaptive_spatial:
            adaptive_spatial_sampling = True
        if device in {"mps", "auto"}:
            mps_optimized_kernel = True
        adaptive_integrator = False
        max_steps = min(int(max_steps), 360)
        step_size = max(float(step_size), 0.24)
        if supports_adaptive_spatial:
            adaptive_spatial_min_scale = min(float(adaptive_spatial_min_scale), 0.58)
            adaptive_spatial_preview_steps = min(int(adaptive_spatial_preview_steps), 96)
        if int(render_tile_rows) <= 0:
            render_tile_rows = max(48, min(192, int(height // 3)))
    elif perf_profile == "High Fidelity":
        adaptive_integrator = True
        mixed_precision = False
        if supports_adaptive_spatial:
            adaptive_spatial_sampling = False

    st.subheader("Sfondo")
    b1, b2, b3 = st.columns(3)
    with b1:
        background_mode = st.selectbox(
            "Background mode",
            options=CHOICE_FIELDS["background_mode"],
            index=CHOICE_FIELDS["background_mode"].index(
                _safe_choice(CHOICE_FIELDS["background_mode"], str(cfg_seed["background_mode"]))
            ),
        )
        background_projection = st.selectbox(
            "Background projection",
            options=CHOICE_FIELDS["background_projection"],
            index=CHOICE_FIELDS["background_projection"].index(
                _safe_choice(CHOICE_FIELDS["background_projection"], str(cfg_seed["background_projection"]))
            ),
        )
    with b2:
        enable_star_background = st.checkbox("Enable star background", value=bool(cfg_seed["enable_star_background"]))
        star_density = st.number_input("Star density", min_value=0.0, value=float(cfg_seed["star_density"]), step=0.0001, format="%.6f")
        star_brightness = st.number_input("Star brightness", min_value=0.0, value=float(cfg_seed["star_brightness"]), step=0.1)
    with b3:
        hdri_path = st.text_input("HDRI path", value=str(cfg_seed.get("hdri_path") or ""))
        hdri_exposure = st.number_input("HDRI exposure", min_value=0.01, value=float(cfg_seed["hdri_exposure"]), step=0.1)
        hdri_rotation_default = _clamp(float(cfg_seed["hdri_rotation_deg"]), 0.0, 360.0)
        hdri_rotation_deg = st.number_input(
            "HDRI rotation (deg)",
            min_value=0.0,
            max_value=360.0,
            value=hdri_rotation_default,
            step=1.0,
        )

    video_params: dict[str, Any] = {}
    if mode == "video":
        st.subheader("Parametri video")
        v1, v2, v3, v4 = st.columns(4)
        with v1:
            video_params["frames"] = st.number_input("Frames", min_value=1, value=100, step=1)
            video_params["fps"] = st.number_input("FPS", min_value=1, value=10, step=1)
            video_params["azimuth_orbits"] = st.number_input("Azimuth orbits", value=1.0, step=0.1)
        with v2:
            video_params["inclination_wobble_deg"] = st.number_input("Inclination wobble", value=0.0, step=0.5)
            use_incl_sweep = st.checkbox("Inclination sweep", value=True)
            video_params["inclination_start_deg"] = (
                st.number_input(
                    "Inclination start",
                    min_value=0.0,
                    max_value=180.0,
                    value=0.0,
                    step=1.0,
                )
                if use_incl_sweep
                else None
            )
            video_params["inclination_end_deg"] = (
                st.number_input(
                    "Inclination end",
                    min_value=0.0,
                    max_value=180.0,
                    value=180.0,
                    step=1.0,
                )
                if use_incl_sweep
                else None
            )
        with v3:
            use_radius_sweep = st.checkbox("Radius sweep", value=False)
            video_params["observer_radius_start"] = st.number_input("Radius start", value=float(observer_radius), step=0.5) if use_radius_sweep else None
            video_params["observer_radius_end"] = st.number_input("Radius end", value=float(observer_radius), step=0.5) if use_radius_sweep else None
            video_params["taa_samples"] = st.number_input("TAA samples", min_value=1, value=1, step=1)
        with v4:
            video_params["shutter_fraction"] = st.number_input("Shutter fraction", min_value=0.0, max_value=1.0, value=0.85, step=0.05)
            video_params["spatial_jitter"] = st.checkbox("Spatial jitter", value=False)
            video_params["stream_encode"] = st.checkbox("Stream encode", value=True)
            video_params["adaptive_frame_steps"] = st.checkbox("Adaptive frame steps", value=True)
            video_params["adaptive_frame_steps_min_scale"] = st.number_input(
                "Adaptive min scale",
                min_value=0.1,
                max_value=1.0,
                value=0.60,
                step=0.05,
            )

    with st.expander("Override JSON avanzato (facoltativo)"):
        st.write("Inserisci solo i campi da sovrascrivere, ad esempio: `{\"disk_beaming_strength\": 0.6}`")
        patch_text = st.text_area("JSON patch", value="{}", height=220)

    config_dict = dict(default_cfg)
    config_dict.update(
        {
            "width": int(width),
            "height": int(height),
            "fov_deg": float(fov_deg),
            "coordinate_system": coordinate_system,
            "metric_model": metric_model,
            "spin": float(spin),
            "charge": float(charge),
            "cosmological_constant": float(cosmological_constant),
            "observer_radius": float(observer_radius),
            "observer_inclination_deg": float(observer_inclination_deg),
            "observer_azimuth_deg": float(observer_azimuth_deg),
            "observer_roll_deg": float(observer_roll_deg),
            "disk_model": disk_model,
            "disk_radial_profile": disk_radial_profile,
            "disk_outer_radius": float(disk_outer_radius),
            "disk_inner_radius": disk_inner_radius,
            "disk_emission_gain": float(disk_emission_gain),
            "max_steps": int(max_steps),
            "step_size": float(step_size),
            "adaptive_integrator": bool(adaptive_integrator),
            "device": device,
            "dtype": dtype,
            "show_progress_bar": bool(show_progress_bar),
            "progress_backend": progress_backend,
            "mps_optimized_kernel": bool(mps_optimized_kernel),
            "compile_rhs": bool(compile_rhs),
            "mixed_precision": bool(mixed_precision),
            "camera_fastpath": bool(camera_fastpath),
            "render_tile_rows": int(render_tile_rows),
            "postprocess_pipeline": postprocess_pipeline,
            "gargantua_look_strength": float(gargantua_look_strength),
            "background_mode": background_mode,
            "background_projection": background_projection,
            "enable_star_background": bool(enable_star_background),
            "star_density": float(star_density),
            "star_brightness": float(star_brightness),
            "hdri_path": hdri_path or None,
            "hdri_exposure": float(hdri_exposure),
            "hdri_rotation_deg": float(hdri_rotation_deg),
            "output": output_path,
        }
    )
    if supports_adaptive_spatial:
        config_dict.update(
            {
                "adaptive_spatial_sampling": bool(adaptive_spatial_sampling),
                "adaptive_spatial_preview_steps": int(adaptive_spatial_preview_steps),
                "adaptive_spatial_min_scale": float(adaptive_spatial_min_scale),
                "adaptive_spatial_quantile": float(adaptive_spatial_quantile),
            }
        )

    run_col, preview_col = st.columns([1, 2])
    with run_col:
        run_now_live = st.button(tr(lang, "run_live", "Lancia simulazione (live)"), type="primary")
        run_now_bg = st.button(tr(lang, "run_bg", "Lancia in background"))
    with preview_col:
        st.code(json.dumps(config_dict, indent=2), language="json")

    if not run_now_live and not run_now_bg:
        return

    try:
        patch = _parse_patch(patch_text)
        config_dict.update(patch)
        cfg_obj = replace(RenderConfig(), **config_dict).validated()
    except Exception as exc:
        st.error(f"Configurazione non valida: {exc}")
        return

    workspace_path = Path(workspace).expanduser().resolve()
    run_dir = workspace_path / "out" / "webui_runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg_path = run_dir / f"config_{stamp}.json"
    cfg_path.write_text(json.dumps(asdict(cfg_obj), indent=2), encoding="utf-8")

    cmd = [python_exec, "-m", "kerrtrace", "--config", str(cfg_path), "--output", str(cfg_obj.output)]
    if require_gpu:
        cmd.append("--require-gpu")
    if mode == "video":
        cmd += [
            "--animate",
            "--frames",
            str(int(video_params["frames"])),
            "--fps",
            str(int(video_params["fps"])),
            "--azimuth-orbits",
            str(float(video_params["azimuth_orbits"])),
            "--inclination-wobble-deg",
            str(float(video_params["inclination_wobble_deg"])),
            "--taa-samples",
            str(int(video_params["taa_samples"])),
            "--shutter-fraction",
            str(float(video_params["shutter_fraction"])),
            "--adaptive-frame-steps-min-scale",
            str(float(video_params["adaptive_frame_steps_min_scale"])),
        ]
        if video_params["inclination_start_deg"] is not None and video_params["inclination_end_deg"] is not None:
            cmd += [
                "--inclination-start-deg",
                str(float(video_params["inclination_start_deg"])),
                "--inclination-end-deg",
                str(float(video_params["inclination_end_deg"])),
            ]
        if video_params["observer_radius_start"] is not None and video_params["observer_radius_end"] is not None:
            cmd += [
                "--observer-radius-start",
                str(float(video_params["observer_radius_start"])),
                "--observer-radius-end",
                str(float(video_params["observer_radius_end"])),
            ]
        if bool(video_params["spatial_jitter"]):
            cmd.append("--spatial-jitter")
        if bool(video_params["stream_encode"]):
            cmd.append("--enable-stream-encode")
        else:
            cmd.append("--disable-stream-encode")
        if not bool(video_params["adaptive_frame_steps"]):
            cmd.append("--disable-adaptive-frame-steps")

    st.info(tr(lang, "cmd_launched", "Comando lanciato:"))
    st.code(" ".join(cmd), language="bash")
    st.info(f"{tr(lang, 'cfg_used', 'Config JSON usata')}: {cfg_path}")

    if run_now_bg:
        existing_proc = st.session_state.get("async_proc")
        if existing_proc is not None and existing_proc.poll() is None:
            st.error("C'è già un job in background in esecuzione. Fermalo o attendi la fine.")
            return
        log_path = run_dir / f"run_{stamp}.log"
        log_file = log_path.open("w", encoding="utf-8")
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(workspace_path),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        finally:
            log_file.close()
        st.session_state["async_proc"] = proc
        st.session_state["async_meta"] = {
            "started_at": stamp,
            "cfg_path": str(cfg_path),
            "log_path": str(log_path),
            "workspace": str(workspace_path),
            "output_hint": str(cfg_obj.output),
            "cmd": " ".join(cmd),
        }
        st.success(
            f"Job avviato in background (PID {proc.pid}). "
            f"Apri il pannello '{tr(lang, 'bg_job', 'Job in background')}' per monitorarlo."
        )
        return

    progress_widget = st.progress(0.0, text="Render rows: 0/0 (0.0%)")
    log_placeholder = st.empty()
    rc, log_text = _run_command_live(cmd, workspace_path, log_placeholder, progress_widget)
    log_path = run_dir / f"run_{stamp}.log"
    log_path.write_text(log_text, encoding="utf-8")

    if rc == 0:
        progress_widget.progress(1.0, text="Render rows: completed (100.0%)")
        st.success(f"{tr(lang, 'sim_completed', 'Simulazione completata. Log')}: {log_path}")
    else:
        progress_widget.empty()
        st.error(f"{tr(lang, 'sim_failed', 'Simulazione fallita')} (exit={rc}). Log: {log_path}")
        return

    out_file = _resolve_output_file(
        log_text=log_text,
        workspace_path=workspace_path,
        out_hint=Path(cfg_obj.output),
    )
    if out_file is None or (not out_file.exists()):
        st.warning(f"{tr(lang, 'output_not_found_expected', 'Output non trovato automaticamente. Atteso')}: {Path(cfg_obj.output)}")
        return

    st.session_state["last_output_path"] = str(out_file)
    _show_output_media(out_file)


if __name__ == "__main__":
    main()
