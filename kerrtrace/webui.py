from __future__ import annotations

from collections import deque
from dataclasses import asdict, replace
from datetime import datetime
import json
import logging
import math
import os
from pathlib import Path
import re
import select
import subprocess
import sys
import time
from typing import Any

from PIL import Image, ImageStat
import streamlit as st

if os.name == "posix":
    import fcntl
    import pty
else:
    fcntl = None
    pty = None

try:
    from .config import RenderConfig
    from .geometry import event_horizon_radius, horizon_radii
    from .webui_runtime import (
        launch_background_process as _launch_background_process,
        validate_workspace_path as _validate_workspace_path,
    )
except ImportError:
    from kerrtrace.config import RenderConfig
    from kerrtrace.geometry import event_horizon_radius, horizon_radii
    from kerrtrace.webui_runtime import (
        launch_background_process as _launch_background_process,
        validate_workspace_path as _validate_workspace_path,
    )

logger = logging.getLogger(__name__)


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
        "morris_thorne",
    ],
    "disk_model": ["physical_nt", "legacy"],
    "disk_radial_profile": ["nt_proxy", "nt_page_thorne"],
    "disk_palette": ["default", "interstellar_warm"],
    "disk_diffrot_model": ["keplerian_lut", "keplerian_metric"],
    "disk_diffrot_visual_mode": ["layer_phase", "annular_tiles", "hybrid"],
    "disk_diffrot_iteration": ["v1_basic", "v2_visibility", "v3_robust"],
    "background_mode": ["procedural", "hdri", "darkspace"],
    "background_projection": ["cubemap", "equirectangular", "darkspace"],
    "kerr_schild_mode": ["off", "fsal_only", "analytic"],
    "device": ["auto", "cpu", "cuda", "mps"],
    "dtype": ["float32", "float64"],
    "progress_backend": ["manual", "tqdm", "auto"],
    "temporal_denoise_mode": ["basic", "robust"],
    "video_codec": ["h264", "h265_10bit"],
    "tone_mapper": ["reinhard", "aces", "filmic"],
    "postprocess_pipeline": ["off", "gargantua"],
}

PARAM_SUPPORTED_METRICS: dict[str, set[str]] = {
    "spin": {
        "kerr",
        "kerr_newman",
        "kerr_de_sitter",
        "kerr_newman_de_sitter",
    },
    "charge": {
        "reissner_nordstrom",
        "kerr_newman",
        "reissner_nordstrom_de_sitter",
        "kerr_newman_de_sitter",
    },
    "lambda": {
        "schwarzschild_de_sitter",
        "kerr_de_sitter",
        "reissner_nordstrom_de_sitter",
        "kerr_newman_de_sitter",
    },
}

AUTHOR_SIGNATURE = "Iman Rosignoli"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".gif"}
MEDIA_SUFFIXES = IMAGE_SUFFIXES | VIDEO_SUFFIXES
PROGRESSIVE_STATE_FILENAME = ".webui_progressive_state.json"
PROGRESSIVE_INDEX_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)(?:^|[_-])progressiv[oa]?[_-]?(\d{1,9})(?:$|[_-])"),
    re.compile(r"(?i)(?:^|[_-])p(\d{1,9})(?:$|[_-])"),
)
PRESET_CRITICAL_FIELDS: list[str] = [
    "coordinate_system",
    "metric_model",
    "spin",
    "charge",
    "cosmological_constant",
    "observer_radius",
    "observer_inclination_deg",
    "disk_inner_radius",
    "disk_outer_radius",
    "enable_disk_differential_rotation",
    "disk_diffrot_model",
    "disk_diffrot_visual_mode",
    "disk_diffrot_strength",
    "disk_diffrot_seed",
    "disk_diffrot_iteration",
    "step_size",
    "max_steps",
    "device",
    "dtype",
]

# Default baseline shown/used by WebUI when no preset/JSON is loaded.
WEBUI_BASE_DEFAULTS: dict[str, Any] = {
    "width": 1280,
    "height": 720,
    "fov_deg": 38.0,
    "metric_model": "kerr",
    "coordinate_system": "kerr_schild",
    "spin": 0.85,
    "charge": 0.0,
    "cosmological_constant": 0.0,
    "observer_radius": 30.0,
    "observer_inclination_deg": 80.0,
    "observer_azimuth_deg": 0.0,
    "observer_roll_deg": 0.0,
    "disk_model": "physical_nt",
    "disk_radial_profile": "nt_page_thorne",
    "disk_inner_radius": None,
    "disk_outer_radius": 12.0,
    "disk_emission_gain": 30.0,
    "disk_palette": "default",
    "disk_layered_palette": True,
    "disk_layer_count": 30,
    "disk_layer_mix": 0.55,
    "disk_layer_accident_strength": 0.42,
    "disk_layer_accident_count": 3.8,
    "disk_layer_accident_sharpness": 7.0,
    "disk_layer_global_phase": 0.0,
    "disk_layer_phase_rate_hz": 0.35,
    "max_steps": 500,
    "step_size": 0.11,
    "adaptive_integrator": True,
    "temporal_reprojection": False,
    "temporal_blend": 0.18,
    "temporal_clamp": 24.0,
    "device": "auto",
    "dtype": "float32",
    "show_progress_bar": True,
    "progress_backend": "manual",
    "mps_optimized_kernel": True,
    "mps_auto_chunking": True,
    "compile_rhs": False,
    "mixed_precision": False,
    "camera_fastpath": True,
    "atlas_cartesian_variant": False,
}

LANGUAGE_OPTIONS: dict[str, str] = {
    "it": "Italiano",
    "en": "English",
    "es": "Español",
    "fr": "Français",
    "de": "Deutsch",
    "ro": "Română",
    "zh": "中文(简体)",
}

I18N: dict[str, dict[str, str]] = {
    "it": {
        "page_title": "KerrTrace WebUI",
        "title": "KerrTrace WebUI",
        "author_label": "Autore",
        "intro_caption": "Questa interfaccia genera un JSON RenderConfig e lancia `python -m kerrtrace --config ...`.",
        "result": "Risultato",
        "last_output": "Anteprima ultimo output",
        "manual_open": "Apri file manuale",
        "manual_path": "Percorso file (assoluto o relativo al workspace)",
        "open_file": "Apri file",
        "use_last_output": "Usa ultimo output",
        "clear": "Pulisci",
        "empty_path": "Inserisci un percorso file.",
        "missing_file": "File non trovato:",
        "unsupported_file": "Estensione non supportata per preview:",
        "unsupported_upload": "Estensione file caricato non supportata per preview.",
        "upload_read_error": "Impossibile leggere il file caricato.",
        "resolved_file": "File risolto:",
        "browse_local_file": "Sfoglia file dal computer",
        "local_file_selected": "File locale selezionato",
        "local_picker_hint": "Questa opzione apre il selettore file del browser per scegliere un file locale.",
        "run_header": "Run",
        "language_label": "Lingua",
        "workspace_label": "Workspace",
        "require_gpu": "Richiedi GPU (--require-gpu)",
        "upload_json": "Carica config JSON",
        "json_not_object": "Il JSON caricato non è un oggetto.",
        "json_invalid": "JSON non valido:",
        "mode_header": "Modalità",
        "mode_label": "Tipo simulazione",
        "mode_single_frame": "Frame singolo",
        "mode_video": "Video",
        "mode_starship_frame": "Frame astronave",
        "mode_starship_video": "Video astronave",
        "quality_header": "Qualità / Risoluzione",
        "quality_preset": "Preset qualità",
        "resolution_set": "Risoluzione impostata a",
        "run_live": "Lancia simulazione (live)",
        "run_bg": "Lancia in background",
        "bg_job": "Job in background",
        "job_running": "In esecuzione",
        "job_completed": "Job completato",
        "job_failed": "Job terminato con errore",
        "refresh_monitor": "Aggiorna monitor",
        "stop_job": "Interrompi job",
        "clear_job": "Pulisci stato job",
        "output_not_found_auto": "Output non trovato automaticamente. Controlla il path nel log.",
        "cmd_launched": "Comando lanciato:",
        "cfg_used": "Config JSON usata:",
        "log_label": "Log",
        "sim_completed": "Simulazione completata. Log:",
        "sim_failed": "Simulazione fallita",
        "output_not_found_expected": "Output non trovato automaticamente. Atteso:",
        "theme_label": "Tema UI",
        "theme_auto": "Auto (browser/sistema)",
        "theme_dark": "Scuro",
        "theme_light": "Chiaro",
        "queue_job_background": "Queue job (background)",
        "queued_job_started": "Avviato job in coda",
        "in_queue_label": "In coda:",
        "no_jobs_in_queue": "Nessun job in coda.",
        "recent_history": "Storico recente:",
        "history_empty": "Storico vuoto.",
        "clear_queue": "Svuota coda",
        "clear_history": "Pulisci storico",
        "preset_manager": "Preset manager",
        "preset_active": "Preset attivo",
        "load_preset": "Carica preset",
        "preset_none": "(nessuno)",
        "lock_critical_fields_on_load": "Blocca campi critici al caricamento",
        "load_selected_preset": "Carica preset selezionato",
        "preset_loaded": "Preset caricato",
        "preset_load_error": "Errore caricamento preset",
        "unlock_preset_critical_fields": "Sblocca campi critici preset",
        "preset_name": "Nome preset",
        "preset_tags_comma": "Tag preset (separati da virgola)",
        "store_critical_lock_fields": "Salva lock campi critici",
        "save_preset": "Salva preset",
        "preset_name_invalid": "Nome preset vuoto o non valido.",
        "preset_saved": "Preset salvato",
        "preset_save_error": "Errore salvataggio preset",
        "morris_thorne_areolar_only": "Per Morris-Thorne è disponibile solo la coordinata in variabile areolare.",
        "unused_metric_params": "Parametri non usati dalla metrica selezionata",
        "layered_disk_options": "Opzioni disco stratificato",
        "differential_rotation_options": "Opzioni rotazione differenziale",
        "volume_emission_options": "Opzioni volume emission",
        "kernel_camera_options": "Kernel e camera",
        "morris_thorne_seam_fixes": "Fix seam Morris-Thorne",
        "roi_quality_cache_options": "ROI, qualità e cache",
        "encoding_adaptive_postprocess": "Encoding, adattivo e postprocess",
        "wormhole_remote_background": "Sfondo remoto wormhole",
        "live_frame_preview": "Anteprima frame live",
        "keep_frames_resume_preview": "Mantieni frame (resume/preview)",
        "resume_from_existing_frames": "Riprendi da frame esistenti",
        "frames_directory": "Cartella frame",
        "run_ab_compare_quick": "Esegui confronto A/B (frame rapido)",
        "preflight_auto": "Preflight fisico automatico",
        "dryrun_gate_before_video": "Dry-run 256p gate prima del video",
        "autotune_quick_benchmark": "Autotune device/tiling (benchmark rapido)",
        "ab_compare_settings": "Impostazioni confronto A/B",
        "ab_width": "Larghezza A/B",
        "ab_max_steps": "Max step A/B",
        "ab_patch_a_json": "Patch JSON A",
        "ab_patch_b_json": "Patch JSON B",
        "json_patch_short": "JSON",
        "config_json": "Config JSON",
        "preset_lock_active_critical": "Preset lock attivo sui campi critici",
        "preflight_prefix": "Preflight",
        "ab_compare_only_single_video": "Confronto A/B rapido supportato solo per single_frame/video (kerrtrace).",
        "ab_compare_result": "Risultato confronto A/B",
        "ab_patch_json_invalid": "Patch JSON A/B non valida",
        "ab_config_invalid": "Configurazione A/B non valida",
        "ab_faster_case": "Caso più veloce",
        "ab_speedup": "speedup",
        "ab_case": "Caso",
        "failed": "fallito",
        "autotune_benchmark": "Benchmark autotune",
        "autotune_benchmark_caption": "Benchmark rapido su frame ridotto per selezionare device e tiling più efficienti.",
        "autotune_applied": "Autotune applicato",
        "autotune_not_better": "Autotune non ha trovato un profilo migliore; mantengo la configurazione corrente.",
        "progressive_output_allocated": "Output progressivo allocato",
        "script_not_found": "Script non trovato",
        "obj_model_required_starship": "OBJ model path è obbligatorio in modalità Starship.",
        "obj_not_found": "OBJ non trovato",
        "gpu_required_not_cpu": "Richiesta GPU attiva: imposta device su auto/mps/cuda (non cpu).",
        "ship_program_json_invalid": "Ship thrust program JSON non valido",
        "multi_ship_json_invalid": "Multi-ship config JSON non valido",
        "dryrun_gate_title": "Dry-run gate 256p",
        "dryrun_gate_caption": "Eseguo 1 frame rapido prima del video per evitare render lunghi non validi.",
        "dryrun_preparing": "Dry-run: preparazione",
        "dryrun_failed": "Dry-run fallito",
        "dryrun_completed": "Dry-run completato",
        "dryrun_preview": "Anteprima dry-run",
        "dryrun_metrics": "Metriche dry-run",
        "dryrun_gate_reason": "Motivo dry-run gate",
        "video_blocked_quality_gate": "Video bloccato dal gate di qualità. Correggi parametri e rilancia.",
        "dryrun_passed_starting_video": "Dry-run superato: avvio del rendering video.",
        "phase_2_2_start_video": "Fase 2/2: avvio render video. Durante warmup kernel i primi frame possono impiegare un po'.",
        "job_queued": "Job accodato",
        "position": "posizione",
        "open_queue_panel_to_monitor": "Apri il pannello queue per monitorare.",
        "job_started_background": "Job avviato in background",
        "open_panel_to_monitor": "Apri il pannello",
        "to_monitor": "per monitorarlo",
        "auto_refresh_monitor": "Auto-refresh monitor",
        "auto_refresh_interval_sec": "Intervallo auto-refresh (s)",
        "auto_refresh_active_every": "Auto-refresh attivo: aggiornamento ogni {seconds}s.",
        "render_rows_initial": "Render rows: 0/0 (0.0%)",
        "render_rows_completed": "Render rows: completed (100.0%)",
        "pending_video_ready": "Dry-run superato. Render video pronto.",
        "pending_video_confirm_hint": "Conferma per avviare il render completo con questo JSON.",
        "pending_video_confirm": "Conferma render video",
        "pending_video_cancel": "Annulla render pendente",
        "pending_video_missing": "Configurazione video pendente non valida o incompleta.",
        "pending_video_mode_label": "Modalità lancio",
        "pending_video_live": "live",
        "pending_video_bg": "background",
        "darkspace_active_msg": "Darkspace attivo: i controlli sfondo avanzati sono disabilitati.",
    },
    "en": {
        "page_title": "KerrTrace WebUI",
        "title": "KerrTrace WebUI",
        "author_label": "Author",
        "intro_caption": "This interface creates a RenderConfig JSON and runs `python -m kerrtrace --config ...`.",
        "result": "Result",
        "last_output": "Last output preview",
        "manual_open": "Open file manually",
        "manual_path": "File path (absolute or relative to workspace)",
        "open_file": "Open file",
        "use_last_output": "Use last output",
        "clear": "Clear",
        "empty_path": "Enter a file path.",
        "missing_file": "File not found:",
        "unsupported_file": "Unsupported file extension for preview:",
        "unsupported_upload": "Unsupported uploaded file extension for preview.",
        "upload_read_error": "Cannot read uploaded file.",
        "resolved_file": "Resolved file:",
        "browse_local_file": "Browse file from your computer",
        "local_file_selected": "Selected local file",
        "local_picker_hint": "This opens the browser file picker so you can choose a local media file.",
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
        "mode_starship_frame": "Starship Frame",
        "mode_starship_video": "Starship Video",
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
        "log_label": "Log",
        "sim_completed": "Simulation completed. Log:",
        "sim_failed": "Simulation failed",
        "output_not_found_expected": "Output not found automatically. Expected:",
        "theme_label": "UI Theme",
        "theme_auto": "Auto (browser/system)",
        "theme_dark": "Dark",
        "theme_light": "Light",
        "queue_job_background": "Queue job (background)",
        "queued_job_started": "Queued job started",
        "in_queue_label": "Queued:",
        "no_jobs_in_queue": "No queued jobs.",
        "recent_history": "Recent history:",
        "history_empty": "History is empty.",
        "clear_queue": "Clear queue",
        "clear_history": "Clear history",
        "preset_manager": "Preset manager",
        "preset_active": "Active preset",
        "load_preset": "Load preset",
        "preset_none": "(none)",
        "lock_critical_fields_on_load": "Lock critical fields on load",
        "load_selected_preset": "Load selected preset",
        "preset_loaded": "Preset loaded",
        "preset_load_error": "Preset load error",
        "unlock_preset_critical_fields": "Unlock preset critical fields",
        "preset_name": "Preset name",
        "preset_tags_comma": "Preset tags (comma separated)",
        "store_critical_lock_fields": "Store critical lock fields",
        "save_preset": "Save preset",
        "preset_name_invalid": "Preset name is empty or invalid.",
        "preset_saved": "Preset saved",
        "preset_save_error": "Preset save error",
        "morris_thorne_areolar_only": "For Morris-Thorne only the areolar-variable coordinate is available.",
        "unused_metric_params": "Parameters not used by selected metric",
        "layered_disk_options": "Layered disk options",
        "differential_rotation_options": "Differential rotation options",
        "volume_emission_options": "Volume emission options",
        "kernel_camera_options": "Kernel & camera",
        "morris_thorne_seam_fixes": "Morris-Thorne seam fixes",
        "roi_quality_cache_options": "ROI, quality & cache",
        "encoding_adaptive_postprocess": "Encoding, adaptive & postprocess",
        "wormhole_remote_background": "Wormhole remote background",
        "live_frame_preview": "Live frame preview",
        "keep_frames_resume_preview": "Keep frames (resume/preview)",
        "resume_from_existing_frames": "Resume from existing frames",
        "frames_directory": "Frames directory",
        "run_ab_compare_quick": "Run A/B compare (quick frame)",
        "preflight_auto": "Automatic physical preflight",
        "dryrun_gate_before_video": "Dry-run 256p gate before video",
        "autotune_quick_benchmark": "Autotune device/tiling (quick benchmark)",
        "ab_compare_settings": "A/B compare settings",
        "ab_width": "A/B width",
        "ab_max_steps": "A/B max steps",
        "ab_patch_a_json": "A patch JSON",
        "ab_patch_b_json": "B patch JSON",
        "json_patch_short": "JSON",
        "config_json": "Config JSON",
        "preset_lock_active_critical": "Preset lock active on critical fields",
        "preflight_prefix": "Preflight",
        "ab_compare_only_single_video": "Quick A/B compare is supported only for single_frame/video (kerrtrace).",
        "ab_compare_result": "A/B compare result",
        "ab_patch_json_invalid": "Invalid A/B patch JSON",
        "ab_config_invalid": "Invalid A/B configuration",
        "ab_faster_case": "Faster case",
        "ab_speedup": "speedup",
        "ab_case": "Case",
        "failed": "failed",
        "autotune_benchmark": "Autotune benchmark",
        "autotune_benchmark_caption": "Quick reduced-frame benchmark to choose the best device and tiling.",
        "autotune_applied": "Autotune applied",
        "autotune_not_better": "Autotune did not find a better profile; keeping current settings.",
        "progressive_output_allocated": "Progressive output allocated",
        "script_not_found": "Script not found",
        "obj_model_required_starship": "OBJ model path is required in Starship mode.",
        "obj_not_found": "OBJ not found",
        "gpu_required_not_cpu": "GPU required: set device to auto/mps/cuda (not cpu).",
        "ship_program_json_invalid": "Invalid ship thrust program JSON",
        "multi_ship_json_invalid": "Invalid multi-ship config JSON",
        "dryrun_gate_title": "Dry-run gate 256p",
        "dryrun_gate_caption": "Run one quick frame before video to avoid long invalid renders.",
        "dryrun_preparing": "Dry-run: preparing",
        "dryrun_failed": "Dry-run failed",
        "dryrun_completed": "Dry-run: completed",
        "dryrun_preview": "Dry-run preview",
        "dryrun_metrics": "Dry-run metrics",
        "dryrun_gate_reason": "Dry-run gate reason",
        "video_blocked_quality_gate": "Video blocked by quality gate. Fix parameters and rerun.",
        "dryrun_passed_starting_video": "Dry-run passed: starting video render.",
        "phase_2_2_start_video": "Phase 2/2: starting video render. During kernel warmup first frames may take longer.",
        "job_queued": "Job queued",
        "position": "position",
        "open_queue_panel_to_monitor": "Open the queue panel to monitor it.",
        "job_started_background": "Background job started",
        "open_panel_to_monitor": "Open panel",
        "to_monitor": "to monitor it",
        "auto_refresh_monitor": "Auto-refresh monitor",
        "auto_refresh_interval_sec": "Auto-refresh interval (s)",
        "auto_refresh_active_every": "Auto-refresh active: updating every {seconds}s.",
        "render_rows_initial": "Render rows: 0/0 (0.0%)",
        "render_rows_completed": "Render rows: completed (100.0%)",
        "pending_video_ready": "Dry-run passed. Video render is ready.",
        "pending_video_confirm_hint": "Confirm to start the full render with this JSON.",
        "pending_video_confirm": "Confirm video render",
        "pending_video_cancel": "Cancel pending render",
        "pending_video_missing": "Pending video configuration is invalid or incomplete.",
        "pending_video_mode_label": "Launch mode",
        "pending_video_live": "live",
        "pending_video_bg": "background",
        "darkspace_active_msg": "Darkspace active: advanced background controls are disabled.",
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
        "theme_label": "Tema UI",
        "theme_auto": "Auto (navegador/sistema)",
        "theme_dark": "Oscuro",
        "theme_light": "Claro",
    },
    "ro": {
        "page_title": "KerrTrace WebUI",
        "title": "KerrTrace WebUI",
        "author_label": "Autor",
        "intro_caption": "Această interfață generează un JSON RenderConfig și rulează `python -m kerrtrace --config ...`.",
        "result": "Rezultat",
        "last_output": "Previzualizare ultimului output",
        "manual_open": "Deschide fișier manual",
        "manual_path": "Cale fișier (absolută sau relativă la workspace)",
        "open_file": "Deschide fișier",
        "use_last_output": "Folosește ultimul output",
        "clear": "Curăță",
        "empty_path": "Introdu o cale către fișier.",
        "missing_file": "Fișier negăsit:",
        "unsupported_file": "Extensie de fișier nesuportată pentru previzualizare:",
        "unsupported_upload": "Extensie nesuportată pentru fișierul încărcat.",
        "upload_read_error": "Nu se poate citi fișierul încărcat.",
        "resolved_file": "Fișier rezolvat:",
        "browse_local_file": "Alege fișier de pe calculator",
        "local_file_selected": "Fișier local selectat",
        "local_picker_hint": "Aceasta deschide selectorul de fișiere din browser pentru a alege un fișier media local.",
        "run_header": "Rulare",
        "language_label": "Limbă",
        "workspace_label": "Workspace",
        "require_gpu": "Necesită GPU (--require-gpu)",
        "upload_json": "Încarcă configurație JSON",
        "json_not_object": "JSON-ul încărcat nu este un obiect.",
        "json_invalid": "JSON invalid:",
        "mode_header": "Mod",
        "mode_label": "Tip simulare",
        "mode_single_frame": "Cadru unic",
        "mode_video": "Video",
        "mode_starship_frame": "Cadru navă spațială",
        "mode_starship_video": "Video navă spațială",
        "quality_header": "Calitate / Rezoluție",
        "quality_preset": "Preset calitate",
        "resolution_set": "Rezoluție setată la",
        "run_live": "Rulează simularea (live)",
        "run_bg": "Rulează în fundal",
        "bg_job": "Job în fundal",
        "job_running": "În execuție",
        "job_completed": "Job finalizat",
        "job_failed": "Job eșuat",
        "refresh_monitor": "Actualizează monitorizarea",
        "stop_job": "Oprește jobul",
        "clear_job": "Șterge starea jobului",
        "output_not_found_auto": "Output-ul nu a fost găsit automat. Verifică calea din log.",
        "cmd_launched": "Comandă lansată:",
        "cfg_used": "Config JSON folosit:",
        "log_label": "Log",
        "sim_completed": "Simulare finalizată. Log:",
        "sim_failed": "Simulare eșuată",
        "output_not_found_expected": "Output-ul nu a fost găsit automat. Așteptat:",
        "theme_label": "Temă UI",
        "theme_auto": "Auto (browser/sistem)",
        "theme_dark": "Întunecat",
        "theme_light": "Luminos",
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
        "page_title": "KerrTrace WebUI",
        "title": "KerrTrace WebUI",
        "author_label": "作者",
        "intro_caption": "此界面会生成 RenderConfig JSON，并运行 `python -m kerrtrace --config ...`。",
        "result": "结果",
        "last_output": "上次输出预览",
        "manual_open": "手动打开文件",
        "manual_path": "文件路径（绝对路径或相对 workspace）",
        "open_file": "打开文件",
        "use_last_output": "使用上次输出",
        "clear": "清除",
        "empty_path": "请输入文件路径。",
        "missing_file": "未找到文件：",
        "unsupported_file": "该扩展名不支持预览：",
        "unsupported_upload": "上传文件扩展名不支持预览。",
        "upload_read_error": "无法读取上传文件。",
        "resolved_file": "解析后的文件：",
        "browse_local_file": "从电脑选择文件",
        "local_file_selected": "已选择本地文件",
        "local_picker_hint": "这会打开浏览器文件选择器，以便选择本地媒体文件。",
        "run_header": "运行",
        "language_label": "语言",
        "workspace_label": "工作目录",
        "require_gpu": "强制使用 GPU (--require-gpu)",
        "upload_json": "上传 JSON 配置",
        "json_not_object": "上传的 JSON 不是对象。",
        "json_invalid": "无效 JSON：",
        "mode_header": "模式",
        "mode_label": "模拟类型",
        "mode_single_frame": "单帧",
        "mode_video": "视频",
        "mode_starship_frame": "飞船单帧",
        "mode_starship_video": "飞船视频",
        "quality_header": "质量 / 分辨率",
        "quality_preset": "质量预设",
        "resolution_set": "分辨率已设置为",
        "run_live": "启动模拟（实时）",
        "run_bg": "后台运行",
        "bg_job": "后台任务",
        "job_running": "运行中",
        "job_completed": "任务已完成",
        "job_failed": "任务失败",
        "refresh_monitor": "刷新监控",
        "stop_job": "停止任务",
        "clear_job": "清除任务状态",
        "output_not_found_auto": "无法自动找到输出。请检查日志中的路径。",
        "cmd_launched": "已启动命令：",
        "cfg_used": "使用的配置 JSON：",
        "log_label": "日志",
        "sim_completed": "模拟完成。日志：",
        "sim_failed": "模拟失败",
        "output_not_found_expected": "无法自动找到输出。期望路径：",
        "theme_label": "界面主题",
        "theme_auto": "自动（浏览器/系统）",
        "theme_dark": "深色",
        "theme_light": "浅色",
    },
    "fr": {
        "author_label": "Auteur",
        "language_label": "Langue",
        "manual_open": "Ouvrir un fichier manuellement",
        "theme_label": "Thème UI",
        "theme_auto": "Auto (navigateur/système)",
        "theme_dark": "Sombre",
        "theme_light": "Clair",
    },
    "de": {
        "page_title": "KerrTrace WebUI",
        "title": "KerrTrace WebUI",
        "author_label": "Autor",
        "intro_caption": "Diese Oberfläche erzeugt eine RenderConfig-JSON und führt `python -m kerrtrace --config ...` aus.",
        "result": "Ergebnis",
        "last_output": "Vorschau der letzten Ausgabe",
        "language_label": "Sprache",
        "manual_open": "Datei manuell öffnen",
        "manual_path": "Dateipfad (absolut oder relativ zum Workspace)",
        "open_file": "Datei öffnen",
        "use_last_output": "Letzte Ausgabe verwenden",
        "clear": "Leeren",
        "empty_path": "Bitte einen Dateipfad eingeben.",
        "missing_file": "Datei nicht gefunden:",
        "unsupported_file": "Nicht unterstützte Dateiendung für Vorschau:",
        "unsupported_upload": "Nicht unterstützte Dateiendung der hochgeladenen Datei.",
        "upload_read_error": "Hochgeladene Datei kann nicht gelesen werden.",
        "resolved_file": "Aufgelöste Datei:",
        "browse_local_file": "Datei vom Computer auswählen",
        "local_file_selected": "Ausgewählte lokale Datei",
        "local_picker_hint": "Öffnet den Datei-Dialog des Browsers, um eine lokale Mediendatei auszuwählen.",
        "run_header": "Ausführung",
        "workspace_label": "Workspace",
        "require_gpu": "GPU erzwingen (--require-gpu)",
        "upload_json": "JSON-Konfiguration hochladen",
        "json_not_object": "Die hochgeladene JSON ist kein Objekt.",
        "json_invalid": "Ungültige JSON:",
        "mode_header": "Modus",
        "mode_label": "Simulationstyp",
        "mode_single_frame": "Einzelbild",
        "mode_video": "Video",
        "mode_starship_frame": "Raumschiff-Frame",
        "mode_starship_video": "Raumschiff-Video",
        "quality_header": "Qualität / Auflösung",
        "quality_preset": "Qualitäts-Preset",
        "resolution_set": "Auflösung gesetzt auf",
        "run_live": "Simulation starten (live)",
        "run_bg": "Im Hintergrund starten",
        "bg_job": "Hintergrund-Job",
        "job_running": "Läuft",
        "job_completed": "Job abgeschlossen",
        "job_failed": "Job fehlgeschlagen",
        "refresh_monitor": "Monitoring aktualisieren",
        "stop_job": "Job stoppen",
        "clear_job": "Job-Status löschen",
        "output_not_found_auto": "Ausgabe konnte nicht automatisch gefunden werden. Prüfe den Pfad im Log.",
        "cmd_launched": "Gestarteter Befehl:",
        "cfg_used": "Verwendete Config-JSON:",
        "log_label": "Log",
        "sim_completed": "Simulation abgeschlossen. Log:",
        "sim_failed": "Simulation fehlgeschlagen",
        "output_not_found_expected": "Ausgabe konnte nicht automatisch gefunden werden. Erwartet:",
        "theme_label": "UI-Thema",
        "theme_auto": "Auto (Browser/System)",
        "theme_dark": "Dunkel",
        "theme_light": "Hell",
    },
    "pt": {
        "author_label": "Autor",
        "language_label": "Idioma",
        "manual_open": "Abrir ficheiro manualmente",
    },
}

# Ensure every language dictionary contains all UI keys.
# Missing items fall back to English so widgets never show hardcoded literals.
for _lang_code, _lang_dict in I18N.items():
    if _lang_code == "en":
        continue
    for _k, _v in I18N["en"].items():
        _lang_dict.setdefault(_k, _v)

FIELD_I18N: dict[str, dict[str, str]] = {
    "it": {
        "Python executable": "Eseguibile Python",
        "Width": "Larghezza",
        "Height": "Altezza",
        "Output file": "File di output",
        "Compose output path from folder + filename": "Componi output da cartella + nome file",
        "Output directory": "Cartella output",
        "Output file name": "Nome file output",
        "FOV (deg)": "FOV (gradi)",
        "Coordinate system": "Sistema di coordinate",
        "Metric model": "Modello metrico",
        "Spin a": "Spin a",
        "Charge Q": "Carica Q",
        "Lambda": "Lambda",
        "Observer radius": "Raggio osservatore",
        "Observer inclination (deg)": "Inclinazione osservatore (gradi)",
        "Observer azimuth (deg)": "Azimut osservatore (gradi)",
        "Observer roll (deg)": "Roll osservatore (gradi)",
        "Disk model": "Modello del disco",
        "Disk radial profile": "Profilo radiale del disco",
        "Disk outer radius": "Raggio esterno del disco",
        "Disk & Rendering": "Disco e rendering",
        "Performance profile": "Profilo performance",
        "Manual": "Manuale",
        "GPU Balanced (Recommended)": "GPU bilanciato (Consigliato)",
        "Fast Preview": "Anteprima veloce",
        "High Fidelity": "Alta fedelta'",
        "r_in = default (ISCO)": "r_in = default (ISCO)",
        "Disk inner radius": "Raggio interno del disco",
        "Disk emission gain": "Guadagno emissione disco",
        "Disk palette": "Palette disco",
        "Enable layered disk": "Abilita disco stratificato",
        "Layer count": "Numero strati",
        "Layer mix": "Mix strati",
        "Differential rotation": "Rotazione differenziale",
        "Enable differential disk rotation": "Abilita rotazione differenziale del disco",
        "Differential rotation model": "Modello rotazione differenziale",
        "Differential rotation visual mode": "Modalita' visiva rotazione differenziale",
        "Differential rotation strength": "Intensita' rotazione differenziale",
        "Differential rotation seed": "Seed rotazione differenziale",
        "Differential rotation iteration": "Iterazione rotazione differenziale",
        "Max steps": "Max step",
        "Step size": "Dimensione step",
        "Adaptive integrator": "Integratore adattivo",
        "Device": "Dispositivo",
        "Dtype": "Dtype",
        "Show progress bar": "Mostra barra di avanzamento",
        "Progress backend": "Backend progress bar",
        "manual: custom bar; tqdm: tqdm bar; auto: use tqdm when available": "manual: barra custom; tqdm: barra tqdm; auto: usa tqdm se disponibile",
        "MPS optimized kernel": "Kernel MPS ottimizzato",
        "Compile RHS": "Compila RHS",
        "Mixed precision": "Precisione mista",
        "Camera fastpath": "Camera fastpath",
        "Atlas/cartesian camera variant": "Variante camera atlas/cartesiana",
        "Adaptive spatial sampling": "Campionamento spaziale adattivo",
        "Adaptive preview steps": "Step preview adattiva",
        "Adaptive min scale": "Scala minima adattiva",
        "Adaptive quantile": "Quantile adattivo",
        "Render tile rows (0=auto)": "Righe tile render (0=auto)",
        "Postprocess pipeline": "Pipeline postprocess",
        "Gargantua look strength": "Intensita' look Gargantua",
        "Background": "Sfondo",
        "Background mode": "Modalita' sfondo",
        "Background projection": "Proiezione sfondo",
        "Enable star background": "Abilita sfondo stellare",
        "Star density": "Densita' stelle",
        "Star brightness": "Luminosita' stelle",
        "HDRI path": "Percorso HDRI",
        "HDRI exposure": "Esposizione HDRI",
        "HDRI rotation (deg)": "Rotazione HDRI (gradi)",
        "Video parameters": "Parametri video",
        "Starship overlay (OBJ)": "Overlay astronave (OBJ)",
        "OBJ model path": "Percorso modello OBJ",
        "Ship radius (M)": "Raggio nave (M)",
        "Ship theta (deg)": "Theta nave (gradi)",
        "Ship phi (deg)": "Phi nave (gradi)",
        "Ship size (M)": "Dimensione nave (M)",
        "Ship yaw (deg)": "Yaw nave (gradi)",
        "Ship pitch (deg)": "Pitch nave (gradi)",
        "Ship roll (deg)": "Roll nave (gradi)",
        "Ship opacity": "Opacità nave",
        "Cinematic strength": "Intensità cinematica",
        "Ship v_phi": "Velocità nave v_phi",
        "Ship v_theta": "Velocità nave v_theta",
        "Ship v_r": "Velocità nave v_r",
        "Ship acceleration": "Accelerazione nave",
        "Ship direction mode": "Modalità direzione spinta",
        "Ship dir x": "Direzione nave x",
        "Ship dir y": "Direzione nave y",
        "Ship dir z": "Direzione nave z",
        "Ship thrust program JSON": "Programma spinta nave JSON",
        "Multi-ship config JSON": "Config multi-astronave JSON",
        "Starship frames": "Frame astronave",
        "Starship FPS": "FPS astronave",
        "Ship integration substeps": "Sotto-step integrazione nave",
        "Keep starship frames": "Mantieni frame astronave",
        "Frames": "Frame",
        "FPS": "FPS",
        "Azimuth orbits": "Orbite azimutali",
        "Inclination wobble": "Wobble inclinazione",
        "Inclination sweep": "Sweep inclinazione",
        "Inclination start": "Inclinazione iniziale",
        "Inclination end": "Inclinazione finale",
        "Radius sweep": "Sweep raggio",
        "Radius start": "Raggio iniziale",
        "Radius end": "Raggio finale",
        "TAA samples": "Campioni TAA",
        "Shutter fraction": "Frazione otturatore",
        "Spatial jitter": "Jitter spaziale",
        "Stream encode": "Encoding in stream",
        "Adaptive frame steps": "Frame step adattivi",
        "Advanced JSON override (optional)": "Override JSON avanzato (facoltativo)",
        "Enter only fields to override, for example: `{\"disk_beaming_strength\": 0.6}`": "Inserisci solo i campi da sovrascrivere, ad esempio: `{\"disk_beaming_strength\": 0.6}`",
        "JSON patch": "Patch JSON",
        "Invalid configuration": "Configurazione non valida",
    },
    "de": {
        "Python executable": "Python-Executable",
        "Width": "Breite",
        "Height": "Höhe",
        "Output file": "Ausgabedatei",
        "FOV (deg)": "FOV (Grad)",
        "Coordinate system": "Koordinatensystem",
        "Metric model": "Metrikmodell",
        "Spin a": "Spin a",
        "Charge Q": "Ladung Q",
        "Lambda": "Lambda",
        "Observer radius": "Beobachterradius",
        "Observer inclination (deg)": "Beobachterneigung (Grad)",
        "Observer azimuth (deg)": "Beobachterazimut (Grad)",
        "Observer roll (deg)": "Beobachter-Rollwinkel (Grad)",
        "Disk model": "Scheibenmodell",
        "Disk radial profile": "Radialprofil der Scheibe",
        "Disk outer radius": "Außenradius der Scheibe",
        "Disk & Rendering": "Scheibe und Rendering",
        "Performance profile": "Leistungsprofil",
        "Manual": "Manuell",
        "GPU Balanced (Recommended)": "GPU ausgewogen (Empfohlen)",
        "Fast Preview": "Schnelle Vorschau",
        "High Fidelity": "Hohe Genauigkeit",
        "r_in = default (ISCO)": "r_in = Standard (ISCO)",
        "Disk inner radius": "Innenradius der Scheibe",
        "Disk emission gain": "Scheiben-Emissionsverstärkung",
        "Disk palette": "Scheibenpalette",
        "Enable layered disk": "Geschichtete Scheibe aktivieren",
        "Layer count": "Anzahl Schichten",
        "Layer mix": "Schichtmischung",
        "Max steps": "Maximale Schritte",
        "Step size": "Schrittweite",
        "Adaptive integrator": "Adaptiver Integrator",
        "Device": "Gerät",
        "Dtype": "Datentyp",
        "Show progress bar": "Fortschrittsbalken anzeigen",
        "Progress backend": "Fortschritts-Backend",
        "manual: custom bar; tqdm: tqdm bar; auto: use tqdm when available": "manual: eigene Leiste; tqdm: tqdm-Leiste; auto: verwende tqdm wenn verfügbar",
        "MPS optimized kernel": "MPS-optimierter Kernel",
        "Compile RHS": "RHS kompilieren",
        "Mixed precision": "Gemischte Präzision",
        "Camera fastpath": "Kamera-Fastpath",
        "Atlas/cartesian camera variant": "Atlas-/kartesische Kameravariante",
        "Adaptive spatial sampling": "Adaptives räumliches Sampling",
        "Adaptive preview steps": "Adaptive Vorschau-Schritte",
        "Adaptive min scale": "Adaptive Mindest-Skalierung",
        "Adaptive quantile": "Adaptives Quantil",
        "Render tile rows (0=auto)": "Render-Kachelzeilen (0=auto)",
        "Postprocess pipeline": "Postprozess-Pipeline",
        "Gargantua look strength": "Gargantua-Look-Stärke",
        "Background": "Hintergrund",
        "Background mode": "Hintergrundmodus",
        "Background projection": "Hintergrundprojektion",
        "Enable star background": "Sternhintergrund aktivieren",
        "Star density": "Sterndichte",
        "Star brightness": "Sternhelligkeit",
        "HDRI path": "HDRI-Pfad",
        "HDRI exposure": "HDRI-Belichtung",
        "HDRI rotation (deg)": "HDRI-Rotation (Grad)",
        "Video parameters": "Video-Parameter",
        "Starship overlay (OBJ)": "Raumschiff-Overlay (OBJ)",
        "OBJ model path": "OBJ-Modellpfad",
        "Ship radius (M)": "Schiffsradius (M)",
        "Ship theta (deg)": "Schiff-Theta (Grad)",
        "Ship phi (deg)": "Schiff-Phi (Grad)",
        "Ship size (M)": "Schiffsgröße (M)",
        "Ship yaw (deg)": "Schiff-Yaw (Grad)",
        "Ship pitch (deg)": "Schiff-Pitch (Grad)",
        "Ship roll (deg)": "Schiff-Roll (Grad)",
        "Ship opacity": "Schiffs-Opazität",
        "Cinematic strength": "Cinematic-Stärke",
        "Ship v_phi": "Schiffsgeschwindigkeit v_phi",
        "Ship v_theta": "Schiffsgeschwindigkeit v_theta",
        "Ship v_r": "Schiffsgeschwindigkeit v_r",
        "Ship acceleration": "Schiffsbeschleunigung",
        "Ship direction mode": "Schubrichtungsmodus",
        "Ship dir x": "Schiffsrichtung x",
        "Ship dir y": "Schiffsrichtung y",
        "Ship dir z": "Schiffsrichtung z",
        "Ship thrust program JSON": "Schubprogramm JSON",
        "Multi-ship config JSON": "Multi-Schiff-Konfiguration JSON",
        "Starship frames": "Raumschiff-Frames",
        "Starship FPS": "Raumschiff-FPS",
        "Ship integration substeps": "Schiffsintegrations-Substeps",
        "Keep starship frames": "Raumschiff-Frames behalten",
        "Frames": "Frames",
        "FPS": "FPS",
        "Azimuth orbits": "Azimut-Orbits",
        "Inclination wobble": "Neigungs-Wobble",
        "Inclination sweep": "Neigungs-Sweep",
        "Inclination start": "Neigungsstart",
        "Inclination end": "Neigungsende",
        "Radius sweep": "Radius-Sweep",
        "Radius start": "Radius-Start",
        "Radius end": "Radius-Ende",
        "TAA samples": "TAA-Samples",
        "Shutter fraction": "Shutter-Anteil",
        "Spatial jitter": "Räumliches Jitter",
        "Stream encode": "Stream-Encoding",
        "Adaptive frame steps": "Adaptive Frame-Schritte",
        "Advanced JSON override (optional)": "Erweiterte JSON-Überschreibung (optional)",
        "Enter only fields to override, for example: `{\"disk_beaming_strength\": 0.6}`": "Nur Felder zum Überschreiben eingeben, z. B.: `{\"disk_beaming_strength\": 0.6}`",
        "JSON patch": "JSON-Patch",
        "Invalid configuration": "Ungültige Konfiguration",
    },
    "es": {
        "Python executable": "Ejecutable de Python",
        "Output file": "Archivo de salida",
        "Coordinate system": "Sistema de coordenadas",
        "Metric model": "Modelo métrico",
        "Observer radius": "Radio del observador",
        "Observer inclination (deg)": "Inclinación del observador (grados)",
        "Observer azimuth (deg)": "Acimut del observador (grados)",
        "Observer roll (deg)": "Roll del observador (grados)",
        "Disk model": "Modelo del disco",
        "Disk radial profile": "Perfil radial del disco",
        "Disk outer radius": "Radio exterior del disco",
        "Disk & Rendering": "Disco y renderizado",
        "Performance profile": "Perfil de rendimiento",
        "Manual": "Manual",
        "GPU Balanced (Recommended)": "GPU equilibrada (Recomendado)",
        "Fast Preview": "Vista previa rápida",
        "High Fidelity": "Alta fidelidad",
        "Disk inner radius": "Radio interior del disco",
        "Disk emission gain": "Ganancia de emisión del disco",
        "Disk palette": "Paleta del disco",
        "Enable layered disk": "Activar disco por capas",
        "Layer count": "Número de capas",
        "Layer mix": "Mezcla de capas",
        "Max steps": "Pasos máximos",
        "Step size": "Tamaño de paso",
        "Adaptive integrator": "Integrador adaptativo",
        "Show progress bar": "Mostrar barra de progreso",
        "Background": "Fondo",
        "Background mode": "Modo de fondo",
        "Background projection": "Proyección de fondo",
        "Enable star background": "Activar fondo estelar",
        "Star density": "Densidad estelar",
        "Star brightness": "Brillo estelar",
        "HDRI path": "Ruta HDRI",
        "HDRI exposure": "Exposición HDRI",
        "HDRI rotation (deg)": "Rotación HDRI (grados)",
        "Video parameters": "Parámetros de video",
        "Advanced JSON override (optional)": "Override JSON avanzado (opcional)",
        "JSON patch": "Patch JSON",
        "Invalid configuration": "Configuración no válida",
    },
    "fr": {
        "Python executable": "Exécutable Python",
        "Output file": "Fichier de sortie",
        "Coordinate system": "Système de coordonnées",
        "Metric model": "Modèle métrique",
        "Observer radius": "Rayon observateur",
        "Observer inclination (deg)": "Inclinaison observateur (degrés)",
        "Observer azimuth (deg)": "Azimut observateur (degrés)",
        "Observer roll (deg)": "Roulis observateur (degrés)",
        "Disk model": "Modèle de disque",
        "Disk radial profile": "Profil radial du disque",
        "Disk outer radius": "Rayon externe du disque",
        "Disk & Rendering": "Disque et rendu",
        "Performance profile": "Profil de performance",
        "Manual": "Manuel",
        "GPU Balanced (Recommended)": "GPU équilibré (recommandé)",
        "Fast Preview": "Aperçu rapide",
        "High Fidelity": "Haute fidélité",
        "Disk palette": "Palette du disque",
        "Enable layered disk": "Activer disque stratifié",
        "Layer count": "Nombre de couches",
        "Layer mix": "Mélange des couches",
        "Background": "Arrière-plan",
        "Video parameters": "Paramètres vidéo",
        "Advanced JSON override (optional)": "Surcharge JSON avancée (optionnel)",
        "JSON patch": "Patch JSON",
        "Invalid configuration": "Configuration invalide",
    },
    "pt": {
        "Python executable": "Executável Python",
        "Output file": "Ficheiro de saída",
        "Coordinate system": "Sistema de coordenadas",
        "Metric model": "Modelo métrico",
        "Observer radius": "Raio do observador",
        "Observer inclination (deg)": "Inclinação do observador (graus)",
        "Observer azimuth (deg)": "Azimute do observador (graus)",
        "Observer roll (deg)": "Rolagem do observador (graus)",
        "Disk model": "Modelo do disco",
        "Disk radial profile": "Perfil radial do disco",
        "Disk outer radius": "Raio externo do disco",
        "Disk & Rendering": "Disco e renderização",
        "Performance profile": "Perfil de desempenho",
        "Manual": "Manual",
        "GPU Balanced (Recommended)": "GPU equilibrada (Recomendado)",
        "Fast Preview": "Pré-visualização rápida",
        "High Fidelity": "Alta fidelidade",
        "Disk palette": "Paleta do disco",
        "Enable layered disk": "Ativar disco em camadas",
        "Layer count": "Número de camadas",
        "Layer mix": "Mistura de camadas",
        "Background": "Fundo",
        "Video parameters": "Parâmetros de vídeo",
        "Advanced JSON override (optional)": "Substituição JSON avançada (opcional)",
        "JSON patch": "Patch JSON",
        "Invalid configuration": "Configuração inválida",
    },
    "ro": {
        "Python executable": "Executabil Python",
        "Width": "Lățime",
        "Height": "Înălțime",
        "Output file": "Fișier de ieșire",
        "FOV (deg)": "FOV (grade)",
        "Coordinate system": "Sistem de coordonate",
        "Metric model": "Model metric",
        "Spin a": "Spin a",
        "Charge Q": "Sarcină Q",
        "Lambda": "Lambda",
        "Observer radius": "Raza observatorului",
        "Observer inclination (deg)": "Înclinarea observatorului (grade)",
        "Observer azimuth (deg)": "Azimutul observatorului (grade)",
        "Observer roll (deg)": "Ruliu observator (grade)",
        "Disk model": "Model disc",
        "Disk radial profile": "Profil radial al discului",
        "Disk outer radius": "Raza externă a discului",
        "Disk & Rendering": "Disc și randare",
        "Performance profile": "Profil de performanță",
        "Manual": "Manual",
        "GPU Balanced (Recommended)": "GPU echilibrat (Recomandat)",
        "Fast Preview": "Previzualizare rapidă",
        "High Fidelity": "Fidelitate înaltă",
        "r_in = default (ISCO)": "r_in = implicit (ISCO)",
        "Disk inner radius": "Raza internă a discului",
        "Disk emission gain": "Amplificare emisie disc",
        "Disk palette": "Paleta discului",
        "Enable layered disk": "Activează disc stratificat",
        "Layer count": "Număr straturi",
        "Layer mix": "Mix straturi",
        "Max steps": "Pași maximi",
        "Step size": "Mărime pas",
        "Adaptive integrator": "Integrator adaptiv",
        "Device": "Dispozitiv",
        "Dtype": "Tip date",
        "Show progress bar": "Afișează bara de progres",
        "Progress backend": "Backend progres",
        "manual: custom bar; tqdm: tqdm bar; auto: use tqdm when available": "manual: bară custom; tqdm: bară tqdm; auto: folosește tqdm când este disponibil",
        "MPS optimized kernel": "Kernel MPS optimizat",
        "Compile RHS": "Compilează RHS",
        "Mixed precision": "Precizie mixtă",
        "Camera fastpath": "Cale rapidă cameră",
        "Atlas/cartesian camera variant": "Variantă cameră atlas/carteziană",
        "Adaptive spatial sampling": "Eșantionare spațială adaptivă",
        "Adaptive preview steps": "Pași preview adaptivi",
        "Adaptive min scale": "Scală minimă adaptivă",
        "Adaptive quantile": "Cuantilă adaptivă",
        "Render tile rows (0=auto)": "Rânduri tile randare (0=auto)",
        "Postprocess pipeline": "Pipeline postprocesare",
        "Gargantua look strength": "Intensitate look Gargantua",
        "Background": "Fundal",
        "Background mode": "Mod fundal",
        "Background projection": "Proiecție fundal",
        "Enable star background": "Activează fundal stelar",
        "Star density": "Densitate stele",
        "Star brightness": "Luminozitate stele",
        "HDRI path": "Cale HDRI",
        "HDRI exposure": "Expunere HDRI",
        "HDRI rotation (deg)": "Rotație HDRI (grade)",
        "Video parameters": "Parametri video",
        "Starship overlay (OBJ)": "Overlay navă spațială (OBJ)",
        "OBJ model path": "Cale model OBJ",
        "Ship radius (M)": "Raza navei (M)",
        "Ship theta (deg)": "Theta navă (grade)",
        "Ship phi (deg)": "Phi navă (grade)",
        "Ship size (M)": "Dimensiune navă (M)",
        "Ship yaw (deg)": "Yaw navă (grade)",
        "Ship pitch (deg)": "Pitch navă (grade)",
        "Ship roll (deg)": "Roll navă (grade)",
        "Ship opacity": "Opacitate navă",
        "Cinematic strength": "Intensitate cinematică",
        "Ship v_phi": "Viteză navă v_phi",
        "Ship v_theta": "Viteză navă v_theta",
        "Ship v_r": "Viteză navă v_r",
        "Ship acceleration": "Accelerație navă",
        "Ship direction mode": "Mod direcție propulsie",
        "Ship dir x": "Direcție navă x",
        "Ship dir y": "Direcție navă y",
        "Ship dir z": "Direcție navă z",
        "Ship thrust program JSON": "Program propulsie navă JSON",
        "Multi-ship config JSON": "Configurație multi-navă JSON",
        "Starship frames": "Cadre navă spațială",
        "Starship FPS": "FPS navă spațială",
        "Ship integration substeps": "Subpași integrare navă",
        "Keep starship frames": "Păstrează cadrele navei",
        "Frames": "Cadre",
        "FPS": "FPS",
        "Azimuth orbits": "Orbite azimutale",
        "Inclination wobble": "Oscilație înclinare",
        "Inclination sweep": "Sweep înclinare",
        "Inclination start": "Înclinare inițială",
        "Inclination end": "Înclinare finală",
        "Radius sweep": "Sweep rază",
        "Radius start": "Rază inițială",
        "Radius end": "Rază finală",
        "TAA samples": "Eșantioane TAA",
        "Shutter fraction": "Fracție obturator",
        "Spatial jitter": "Jitter spațial",
        "Stream encode": "Encodare stream",
        "Adaptive frame steps": "Pași de cadru adaptivi",
        "Advanced JSON override (optional)": "Suprascriere JSON avansată (opțional)",
        "Enter only fields to override, for example: `{\"disk_beaming_strength\": 0.6}`": "Introdu doar câmpurile de suprascris, de exemplu: `{\"disk_beaming_strength\": 0.6}`",
        "JSON patch": "Patch JSON",
        "Invalid configuration": "Configurație invalidă",
    },
    "zh": {
        "Python executable": "Python 可执行文件",
        "Width": "宽度",
        "Height": "高度",
        "Output file": "输出文件",
        "FOV (deg)": "FOV（度）",
        "Coordinate system": "坐标系",
        "Metric model": "度规模型",
        "Spin a": "自旋 a",
        "Charge Q": "电荷 Q",
        "Lambda": "Lambda",
        "Observer radius": "观察者半径",
        "Observer inclination (deg)": "观察者倾角（度）",
        "Observer azimuth (deg)": "观察者方位角（度）",
        "Observer roll (deg)": "观察者滚转角（度）",
        "Disk model": "吸积盘模型",
        "Disk radial profile": "吸积盘径向剖面",
        "Disk outer radius": "吸积盘外半径",
        "Disk & Rendering": "吸积盘与渲染",
        "Performance profile": "性能配置",
        "Manual": "手动",
        "GPU Balanced (Recommended)": "GPU 平衡（推荐）",
        "Fast Preview": "快速预览",
        "High Fidelity": "高保真",
        "r_in = default (ISCO)": "r_in = 默认（ISCO）",
        "Disk inner radius": "吸积盘内半径",
        "Disk emission gain": "吸积盘发射增益",
        "Disk palette": "吸积盘调色板",
        "Enable layered disk": "启用分层吸积盘",
        "Layer count": "层数",
        "Layer mix": "层混合比例",
        "Max steps": "最大步数",
        "Step size": "步长",
        "Adaptive integrator": "自适应积分器",
        "Device": "设备",
        "Dtype": "数据类型",
        "Show progress bar": "显示进度条",
        "Progress backend": "进度后端",
        "manual: custom bar; tqdm: tqdm bar; auto: use tqdm when available": "manual: 自定义进度条；tqdm: tqdm 进度条；auto: 可用时使用 tqdm",
        "MPS optimized kernel": "MPS 优化内核",
        "Compile RHS": "编译 RHS",
        "Mixed precision": "混合精度",
        "Camera fastpath": "相机快速路径",
        "Atlas/cartesian camera variant": "Atlas/笛卡尔相机变体",
        "Adaptive spatial sampling": "自适应空间采样",
        "Adaptive preview steps": "自适应预览步数",
        "Adaptive min scale": "自适应最小缩放",
        "Adaptive quantile": "自适应分位数",
        "Render tile rows (0=auto)": "渲染分块行数（0=自动）",
        "Postprocess pipeline": "后处理流程",
        "Gargantua look strength": "Gargantua 风格强度",
        "Background": "背景",
        "Background mode": "背景模式",
        "Background projection": "背景投影",
        "Enable star background": "启用星空背景",
        "Star density": "恒星密度",
        "Star brightness": "恒星亮度",
        "HDRI path": "HDRI 路径",
        "HDRI exposure": "HDRI 曝光",
        "HDRI rotation (deg)": "HDRI 旋转（度）",
        "Video parameters": "视频参数",
        "Starship overlay (OBJ)": "飞船叠加层 (OBJ)",
        "OBJ model path": "OBJ 模型路径",
        "Ship radius (M)": "飞船半径 (M)",
        "Ship theta (deg)": "飞船 theta（度）",
        "Ship phi (deg)": "飞船 phi（度）",
        "Ship size (M)": "飞船尺寸 (M)",
        "Ship yaw (deg)": "飞船偏航（度）",
        "Ship pitch (deg)": "飞船俯仰（度）",
        "Ship roll (deg)": "飞船滚转（度）",
        "Ship opacity": "飞船不透明度",
        "Cinematic strength": "电影感强度",
        "Ship v_phi": "飞船速度 v_phi",
        "Ship v_theta": "飞船速度 v_theta",
        "Ship v_r": "飞船速度 v_r",
        "Ship acceleration": "飞船加速度",
        "Ship direction mode": "飞船推力方向模式",
        "Ship dir x": "飞船方向 x",
        "Ship dir y": "飞船方向 y",
        "Ship dir z": "飞船方向 z",
        "Ship thrust program JSON": "飞船推力程序 JSON",
        "Multi-ship config JSON": "多飞船配置 JSON",
        "Starship frames": "飞船帧数",
        "Starship FPS": "飞船 FPS",
        "Ship integration substeps": "飞船积分子步",
        "Keep starship frames": "保留飞船帧",
        "Frames": "帧数",
        "FPS": "帧率 FPS",
        "Azimuth orbits": "方位角轨道数",
        "Inclination wobble": "倾角摆动",
        "Inclination sweep": "倾角扫掠",
        "Inclination start": "起始倾角",
        "Inclination end": "结束倾角",
        "Radius sweep": "半径扫掠",
        "Radius start": "起始半径",
        "Radius end": "结束半径",
        "TAA samples": "TAA 采样数",
        "Shutter fraction": "快门比例",
        "Spatial jitter": "空间抖动",
        "Stream encode": "流式编码",
        "Adaptive frame steps": "自适应帧步数",
        "Advanced JSON override (optional)": "高级 JSON 覆盖（可选）",
        "Enter only fields to override, for example: `{\"disk_beaming_strength\": 0.6}`": "仅输入要覆盖的字段，例如：`{\"disk_beaming_strength\": 0.6}`",
        "JSON patch": "JSON 补丁",
        "Invalid configuration": "配置无效",
    },
}


def tr(lang: str, key: str, default: str) -> str:
    lang_dict = I18N.get(lang, {})
    if key in lang_dict:
        return lang_dict[key]
    en_dict = I18N.get("en", {})
    if key in en_dict:
        return en_dict[key]
    return default


def tfield(lang: str, text: str) -> str:
    return FIELD_I18N.get(lang, {}).get(text, text)


def _default_python() -> str:
    project_python = Path(".venv/bin/python")
    if project_python.exists():
        return str(project_python)
    return sys.executable


def _default_starship_obj(workspace_path: Path) -> str:
    candidates = [
        workspace_path / "assets/models/quaternius_ultimate/omen/Omen.obj",
        Path("assets/models/quaternius_ultimate/omen/Omen.obj"),
    ]
    for cand in candidates:
        try:
            if cand.exists():
                return str(cand.resolve())
        except Exception:
            continue
    return ""


def _safe_choice(options: list[str], value: str) -> str:
    if value in options:
        return value
    return options[0]


def _coordinate_options_for_metric(metric_model: str) -> list[str]:
    if metric_model == "morris_thorne":
        return ["boyer_lindquist"]
    return list(CHOICE_FIELDS["coordinate_system"])


def _coordinate_label_for_metric(coordinate_system: str, metric_model: str) -> str:
    if metric_model == "morris_thorne" and coordinate_system == "boyer_lindquist":
        return "variabile_areolare"
    return coordinate_system


def _metric_supports_parameters(metric_model: str) -> tuple[bool, bool, bool]:
    """
    Return support flags for (spin, charge, cosmological_constant).
    """
    model = str(metric_model)
    supports_spin = model in PARAM_SUPPORTED_METRICS["spin"]
    supports_charge = model in PARAM_SUPPORTED_METRICS["charge"]
    supports_lambda = model in PARAM_SUPPORTED_METRICS["lambda"]
    return supports_spin, supports_charge, supports_lambda


def _metric_param_tooltip(metric_model: str, param_key: str, enabled: bool, lang: str) -> str:
    supported = sorted(PARAM_SUPPORTED_METRICS.get(param_key, set()))
    supported_txt = ", ".join(supported) if supported else "n/a"
    model = str(metric_model)
    if lang == "it":
        if enabled:
            return (
                f"Parametro attivo per la metrica `{model}`. "
                f"Metriche supportate: {supported_txt}."
            )
        return (
            f"Parametro disattivato: la metrica `{model}` non lo utilizza. "
            f"Metriche che lo supportano: {supported_txt}."
        )
    if enabled:
        return f"Enabled for `{model}`. Supported metrics: {supported_txt}."
    return f"Disabled: `{model}` does not use this parameter. Supported metrics: {supported_txt}."


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _run_command_live(
    cmd: list[str],
    cwd: Path,
    log_placeholder: Any,
    progress_widget: Any | None = None,
    frame_preview_placeholder: Any | None = None,
) -> tuple[int, str]:
    chunks: list[str] = []
    max_log_chars = 120_000
    progress_re = re.compile(r"(\d+)\s*/\s*(\d+)")
    frame_line_re = re.compile(r"Frame\s+(\d+)\s*/\s*(\d+):\s*([^|]+)")
    frame_dt_re = re.compile(r"\|\s*frame\s*([0-9]*\.?[0-9]+)s")
    pending = ""
    last_progress_key: tuple[int, int] | None = None
    last_frame_key: tuple[int, int] | None = None
    seen_frame_progress = False
    last_row_t: float | None = None
    last_frame_t: float | None = None
    row_sec_per_unit: deque[float] = deque(maxlen=12)
    frame_sec_per_unit: deque[float] = deque(maxlen=12)
    launch_t = time.perf_counter()
    last_heartbeat_t = launch_t
    saw_any_output = False

    def _push(text: str) -> None:
        chunks.append(text)
        joined = "".join(chunks)
        if len(joined) > max_log_chars:
            joined = joined[-max_log_chars:]
            chunks[:] = [joined]
        log_placeholder.code(joined, language="bash")

    def _process_line(line: str, from_carriage: bool) -> None:
        nonlocal last_progress_key, last_frame_key, seen_frame_progress, last_row_t, last_frame_t, saw_any_output
        if not line:
            return
        saw_any_output = True
        stripped = line.strip()
        frame_match = frame_line_re.search(stripped)
        if frame_match:
            seen_frame_progress = True
            done_f = int(frame_match.group(1))
            total_f = int(frame_match.group(2))
            key_f = (done_f, total_f)
            now = time.perf_counter()
            if last_frame_key is not None and key_f[1] == last_frame_key[1] and key_f[0] > last_frame_key[0]:
                if last_frame_t is not None:
                    dt = max(1.0e-6, now - last_frame_t)
                    frame_sec_per_unit.append(dt / float(key_f[0] - last_frame_key[0]))
            explicit_dt = None
            dt_match = frame_dt_re.search(stripped)
            if dt_match is not None:
                try:
                    explicit_dt = float(dt_match.group(1))
                except Exception:
                    explicit_dt = None
            if explicit_dt is not None and explicit_dt > 0.0:
                frame_sec_per_unit.append(explicit_dt)
            eta_f = None
            if frame_sec_per_unit and total_f > done_f:
                avg = float(sum(frame_sec_per_unit)) / float(len(frame_sec_per_unit))
                eta_f = avg * float(total_f - done_f)
            if progress_widget is not None and total_f > 0 and key_f != last_frame_key:
                ratio = max(0.0, min(1.0, float(done_f) / float(total_f)))
                elapsed_s = max(0.0, time.perf_counter() - launch_t)
                progress_widget.progress(
                    ratio,
                    text=(
                        f"Frames: {done_f}/{total_f} ({ratio * 100.0:.1f}%) "
                        f"ETA ~{_format_eta_short(eta_f)} | Elapsed { _format_eta_short(elapsed_s) }"
                    ),
                )
            if key_f != last_frame_key and frame_preview_placeholder is not None:
                raw_path = frame_match.group(3).strip().strip("'").strip('"')
                if raw_path and (not raw_path.startswith("(")):
                    frame_path = Path(raw_path).expanduser()
                    if not frame_path.is_absolute():
                        frame_path = cwd / frame_path
                    try:
                        frame_path = frame_path.resolve()
                    except Exception:
                        frame_path = frame_path.absolute()
                    if frame_path.exists():
                        try:
                            frame_preview_placeholder.image(
                                str(frame_path),
                                caption=f"Live frame {done_f}/{total_f}",
                            )
                        except Exception:
                            pass
            last_frame_key = key_f
            last_frame_t = now
            _push(stripped + "\n")
            return

        if stripped.startswith("Render rows"):
            matches = progress_re.findall(stripped)
            if matches:
                done_s, total_s = matches[-1]
                key = (int(done_s), int(total_s))
                # Ignore ticker refreshes when progress did not advance.
                if key == last_progress_key:
                    return
                now = time.perf_counter()
                if last_progress_key is not None and key[1] == last_progress_key[1] and key[0] > last_progress_key[0]:
                    if last_row_t is not None:
                        dt = max(1.0e-6, now - last_row_t)
                        row_sec_per_unit.append(dt / float(key[0] - last_progress_key[0]))
                last_progress_key = key
                last_row_t = now
                eta_s = None
                if row_sec_per_unit and key[1] > key[0]:
                    avg = float(sum(row_sec_per_unit)) / float(len(row_sec_per_unit))
                    eta_s = avg * float(key[1] - key[0])
                if (not seen_frame_progress) and progress_widget is not None and key[1] > 0:
                    ratio = max(0.0, min(1.0, float(key[0]) / float(key[1])))
                    elapsed_s = max(0.0, time.perf_counter() - launch_t)
                    progress_widget.progress(
                        ratio,
                        text=(
                            f"Render rows: {key[0]}/{key[1]} ({ratio * 100.0:.1f}%) "
                            f"ETA ~{_format_eta_short(eta_s)} | Elapsed { _format_eta_short(elapsed_s) }"
                        ),
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
                now = time.perf_counter()
                # Keep UI responsive during warmup/compilation phases before first frame/row log.
                if (
                    progress_widget is not None
                    and (not seen_frame_progress)
                    and (last_progress_key is None)
                    and proc.poll() is None
                    and (now - last_heartbeat_t) >= 1.5
                ):
                    elapsed = int(max(0.0, now - launch_t))
                    msg = (
                        f"Render avviato: attendo i primi log... {elapsed}s"
                        if (not saw_any_output)
                        else f"Render in preparazione... {elapsed}s"
                    )
                    progress_widget.progress(0.0, text=msg)
                    last_heartbeat_t = now
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

    if progress_widget is not None:
        if seen_frame_progress and last_frame_key is not None and last_frame_key[1] > 0:
            done, total = last_frame_key
            ratio = max(0.0, min(1.0, float(done) / float(total)))
            elapsed_s = max(0.0, time.perf_counter() - launch_t)
            progress_widget.progress(
                ratio,
                text=f"Frames: {done}/{total} ({ratio * 100.0:.1f}%) | Elapsed { _format_eta_short(elapsed_s) }",
            )
        elif last_progress_key is not None and last_progress_key[1] > 0:
            done, total = last_progress_key
            ratio = max(0.0, min(1.0, float(done) / float(total)))
            elapsed_s = max(0.0, time.perf_counter() - launch_t)
            progress_widget.progress(
                ratio,
                text=f"Render rows: {done}/{total} ({ratio * 100.0:.1f}%) | Elapsed { _format_eta_short(elapsed_s) }",
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
    lang = str(st.session_state.get("ui_lang", "it"))
    suffix = out_file.suffix.lower()
    st.subheader(tr(lang, "result", "Result"))
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


def _show_uploaded_media(uploaded_file: Any) -> tuple[bool, str]:
    lang = str(st.session_state.get("ui_lang", "it"))
    name = str(getattr(uploaded_file, "name", "uploaded"))
    suffix = Path(name).suffix.lower()
    if suffix not in MEDIA_SUFFIXES:
        return False, "unsupported"
    try:
        data = uploaded_file.getvalue()
    except Exception:
        return False, "read_error"

    st.subheader(tr(lang, "result", "Result"))
    if suffix in IMAGE_SUFFIXES:
        st.image(data, caption=name)
        return True, ""
    if suffix in VIDEO_SUFFIXES:
        mime = "video/mp4"
        if suffix == ".gif":
            mime = "image/gif"
        elif suffix == ".mov":
            mime = "video/quicktime"
        elif suffix == ".mkv":
            mime = "video/x-matroska"
        st.video(data, format=mime)
        st.caption(name)
        return True, ""
    return False, "unsupported"


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
        re.compile(r"SAVED=\s*(.+)$"),
        re.compile(r"SAVED_PREVIEW=\s*(.+)$"),
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


def _extract_latest_frame_path_from_log(log_text: str, workspace_path: Path) -> Path | None:
    if not log_text:
        return None
    frame_re = re.compile(r"Frame\s+\d+\s*/\s*\d+:\s*([^|]+)")
    lines = [_strip_ansi(ln).strip() for ln in log_text.splitlines()]
    for line in reversed(lines):
        m = frame_re.search(line)
        if m is None:
            continue
        raw = m.group(1).strip().strip("'").strip('"')
        if (not raw) or raw.startswith("("):
            continue
        p = Path(raw).expanduser()
        if not p.is_absolute():
            p = workspace_path / p
        try:
            p = p.resolve()
        except Exception:
            p = p.absolute()
        if p.exists():
            return p
    return None


def _estimate_eta_from_log(log_text: str) -> str | None:
    if not log_text:
        return None
    lines = [_strip_ansi(ln).strip() for ln in log_text.splitlines() if ln.strip()]
    frame_re = re.compile(r"Frame\s+(\d+)\s*/\s*(\d+):")
    frame_dt_re = re.compile(r"\|\s*frame\s*([0-9]*\.?[0-9]+)s")
    row_re = re.compile(r"Render rows:\s*(\d+)\s*/\s*(\d+)")
    row_dt_re = re.compile(r"([0-9]*\.?[0-9]+)s/row")

    frame_done_total: tuple[int, int] | None = None
    frame_dts: list[float] = []
    for line in reversed(lines):
        fm = frame_re.search(line)
        if fm is not None and frame_done_total is None:
            frame_done_total = (int(fm.group(1)), int(fm.group(2)))
        fdt = frame_dt_re.search(line)
        if fdt is not None:
            try:
                val = float(fdt.group(1))
                if val > 0.0:
                    frame_dts.append(val)
            except Exception:
                pass
        if frame_done_total is not None and len(frame_dts) >= 8:
            break
    if frame_done_total is not None:
        done, total = frame_done_total
        if total > done and frame_dts:
            avg = float(sum(frame_dts)) / float(len(frame_dts))
            return f"Frames {done}/{total} ETA ~{_format_eta_short(avg * float(total - done))}"

    row_done_total: tuple[int, int] | None = None
    row_dts: list[float] = []
    for line in reversed(lines):
        rm = row_re.search(line)
        if rm is not None and row_done_total is None:
            row_done_total = (int(rm.group(1)), int(rm.group(2)))
        rdt = row_dt_re.search(line)
        if rdt is not None:
            try:
                val = float(rdt.group(1))
                if val > 0.0:
                    row_dts.append(val)
            except Exception:
                pass
        if row_done_total is not None and len(row_dts) >= 8:
            break
    if row_done_total is not None:
        done, total = row_done_total
        if total > done and row_dts:
            avg = float(sum(row_dts)) / float(len(row_dts))
            return f"Rows {done}/{total} ETA ~{_format_eta_short(avg * float(total - done))}"
    return None


def _extract_progress_from_log(log_text: str) -> tuple[float, str] | None:
    if not log_text:
        return None
    lines = [_strip_ansi(ln).strip() for ln in log_text.splitlines() if ln.strip()]
    frame_re = re.compile(r"Frame\s+(\d+)\s*/\s*(\d+):")
    row_re = re.compile(r"Render rows:\s*(\d+)\s*/\s*(\d+)")

    for line in reversed(lines):
        fm = frame_re.search(line)
        if fm is not None:
            done = max(0, int(fm.group(1)))
            total = max(0, int(fm.group(2)))
            if total > 0:
                done = min(done, total)
                ratio = float(done) / float(total)
                return ratio, f"Frames {done}/{total} ({ratio * 100.0:.1f}%)"

    for line in reversed(lines):
        rm = row_re.search(line)
        if rm is not None:
            done = max(0, int(rm.group(1)))
            total = max(0, int(rm.group(2)))
            if total > 0:
                done = min(done, total)
                ratio = float(done) / float(total)
                return ratio, f"Render rows {done}/{total} ({ratio * 100.0:.1f}%)"
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


def _extract_progressive_index_from_stem(stem: str) -> int:
    max_idx = 0
    for pat in PROGRESSIVE_INDEX_PATTERNS:
        for m in pat.finditer(stem):
            try:
                max_idx = max(max_idx, int(m.group(1)))
            except Exception:
                continue
    return max_idx


def _strip_progressive_tokens_from_stem(stem: str) -> str:
    clean = str(stem).replace("{progressivo}", "").replace("{PROGRESSIVO}", "")
    clean = re.sub(r"(?i)(?:[_-]?progressiv[oa]?[_-]?\d*)$", "", clean).rstrip("_- ")
    clean = re.sub(r"(?i)(?:[_-]?p\d+)$", "", clean).rstrip("_- ")
    return clean or "render"


def _scan_max_progressive_index_in_out(out_dir: Path) -> int:
    if not out_dir.exists():
        return 0
    max_idx = 0
    for child in out_dir.glob("**/*"):
        if not child.is_file():
            continue
        if child.name == PROGRESSIVE_STATE_FILENAME:
            continue
        idx = _extract_progressive_index_from_stem(child.stem)
        if idx > max_idx:
            max_idx = idx
    return max_idx


def _read_progressive_state_unlocked(fh: Any) -> dict[str, Any]:
    try:
        fh.seek(0)
        raw = fh.read()
        if not raw:
            return {}
        payload = json.loads(raw)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _is_path_inside_dir(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except Exception:
        return False


def _to_workspace_relative_or_abs(path: Path, workspace_path: Path) -> str:
    try:
        return str(path.resolve().relative_to(workspace_path.resolve()))
    except Exception:
        return str(path.resolve())


def _reserve_progressive_output_path(
    *,
    workspace_path: Path,
    requested_output: str,
) -> tuple[str, int | None]:
    requested_raw = str(requested_output or "").strip()
    if not requested_raw:
        return requested_output, None

    req_path = Path(requested_raw).expanduser()
    req_abs = (workspace_path / req_path).resolve() if not req_path.is_absolute() else req_path.resolve()
    out_dir = (workspace_path / "out").resolve()
    if not _is_path_inside_dir(req_abs, out_dir):
        return requested_output, None

    out_dir.mkdir(parents=True, exist_ok=True)
    req_abs.parent.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / PROGRESSIVE_STATE_FILENAME
    state_path.touch(exist_ok=True)

    try:
        with state_path.open("r+", encoding="utf-8") as fh:
            if fcntl is not None:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            state = _read_progressive_state_unlocked(fh)
            try:
                last_allocated = int(state.get("last_allocated", 0))
            except Exception:
                last_allocated = 0
            max_on_disk = _scan_max_progressive_index_in_out(out_dir)
            next_idx = max(last_allocated, max_on_disk) + 1

            clean_stem = _strip_progressive_tokens_from_stem(req_abs.stem)
            next_name = f"{clean_stem}_p{next_idx:06d}{req_abs.suffix}"
            next_abs = req_abs.with_name(next_name)

            state.update(
                {
                    "last_allocated": int(next_idx),
                    "last_allocated_output": str(next_abs),
                    "updated_at": datetime.now().isoformat(timespec="seconds"),
                }
            )
            fh.seek(0)
            fh.truncate()
            fh.write(json.dumps(state, indent=2))
            fh.flush()
            if fcntl is not None:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    except Exception:
        return requested_output, None

    return _to_workspace_relative_or_abs(next_abs, workspace_path), int(next_idx)


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


def _benchmark_single_config(
    *,
    python_exec: str,
    workspace_path: Path,
    cfg_path: Path,
    output_path: Path,
    require_gpu: bool,
    timeout_s: float = 240.0,
) -> tuple[bool, float, str]:
    cmd = [
        str(python_exec),
        "-m",
        "kerrtrace",
        "--config",
        str(cfg_path),
        "--output",
        str(output_path),
    ]
    if require_gpu:
        cmd.append("--require-gpu")
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(workspace_path),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        elapsed = max(0.0, float(time.perf_counter() - t0))
        ok = proc.returncode == 0 and output_path.exists()
        log = (proc.stdout or "") + "\n" + (proc.stderr or "")
    except subprocess.TimeoutExpired as exc:
        elapsed = max(0.0, float(time.perf_counter() - t0))
        ok = False
        log = f"autotune timeout after {timeout_s:.1f}s: {exc}"
    if len(log) > 6000:
        log = log[-6000:]
    return ok, elapsed, log.strip()


def _autotune_quick_device_and_tiling(
    *,
    python_exec: str,
    workspace_path: Path,
    run_dir: Path,
    cfg_obj: RenderConfig,
    require_gpu: bool,
) -> tuple[RenderConfig, list[dict[str, Any]]]:
    bench_width = int(min(320, max(144, cfg_obj.width)))
    bench_height = int(max(96, round(float(cfg_obj.height) * (float(bench_width) / max(1.0, float(cfg_obj.width))))))
    bench_steps = int(max(96, min(int(cfg_obj.max_steps), 240)))

    device_candidates = [str(cfg_obj.device)]
    if str(cfg_obj.device) == "auto":
        device_candidates = ["mps", "cuda", "cpu"]
    if require_gpu:
        device_candidates = [d for d in device_candidates if d in {"mps", "cuda"}]
        if not device_candidates:
            return cfg_obj, [{"candidate": "none", "ok": False, "reason": "require_gpu=True and no GPU candidates"}]

    report: list[dict[str, Any]] = []
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    best_device: str | None = None
    best_device_t = float("inf")
    for dev in device_candidates:
        out_path = run_dir / f"autotune_device_{dev}_{stamp}.png"
        cfg_bench = replace(
            cfg_obj,
            width=bench_width,
            height=bench_height,
            max_steps=bench_steps,
            render_tile_rows=0,
            show_progress_bar=False,
            device=dev,
            output=str(out_path),
        )
        cfg_path = run_dir / f"autotune_device_{dev}_{stamp}.json"
        cfg_path.write_text(json.dumps(asdict(cfg_bench), indent=2), encoding="utf-8")
        ok, elapsed, log = _benchmark_single_config(
            python_exec=python_exec,
            workspace_path=workspace_path,
            cfg_path=cfg_path,
            output_path=out_path,
            require_gpu=require_gpu,
        )
        report.append({"stage": "device", "candidate": dev, "ok": ok, "elapsed_s": elapsed, "log_tail": log})
        if ok and elapsed < best_device_t:
            best_device_t = elapsed
            best_device = dev

    if best_device is None:
        return cfg_obj, report

    tile_candidates = [0]
    if bench_height >= 256:
        tile_candidates += [64, 128]
    elif bench_height >= 144:
        tile_candidates += [48, 96]

    best_tile = int(cfg_obj.render_tile_rows) if int(cfg_obj.render_tile_rows) > 0 else 0
    best_tile_t = float("inf")
    for tile in tile_candidates:
        out_path = run_dir / f"autotune_tile_{best_device}_{tile}_{stamp}.png"
        cfg_bench = replace(
            cfg_obj,
            width=bench_width,
            height=bench_height,
            max_steps=bench_steps,
            render_tile_rows=int(tile),
            show_progress_bar=False,
            device=best_device,
            output=str(out_path),
        )
        cfg_path = run_dir / f"autotune_tile_{best_device}_{tile}_{stamp}.json"
        cfg_path.write_text(json.dumps(asdict(cfg_bench), indent=2), encoding="utf-8")
        ok, elapsed, log = _benchmark_single_config(
            python_exec=python_exec,
            workspace_path=workspace_path,
            cfg_path=cfg_path,
            output_path=out_path,
            require_gpu=require_gpu,
        )
        report.append(
            {
                "stage": "tiling",
                "candidate": f"{best_device}/tile={tile}",
                "ok": ok,
                "elapsed_s": elapsed,
                "log_tail": log,
            }
        )
        if ok and elapsed < best_tile_t:
            best_tile_t = elapsed
            best_tile = int(tile)

    tuned = replace(cfg_obj, device=best_device, render_tile_rows=int(best_tile))
    if best_device == "mps":
        tuned = replace(tuned, mps_optimized_kernel=True)
    return tuned, report


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


def _format_eta_short(seconds: float | None) -> str:
    if seconds is None or (not math.isfinite(seconds)) or seconds < 0.0:
        return "--:--"
    total = int(round(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _preset_dir(workspace_path: Path) -> Path:
    return (workspace_path / "out" / "webui_presets").resolve()


def _list_presets(workspace_path: Path) -> list[Path]:
    root = _preset_dir(workspace_path)
    if not root.exists():
        return []
    return sorted(
        [p for p in root.glob("*.json") if p.is_file()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def _load_preset(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Preset file must contain a JSON object")
    if "config" in payload and isinstance(payload["config"], dict):
        cfg = dict(payload["config"])
        meta = dict(payload.get("meta") or {})
        return cfg, meta
    # backward-compatible: plain config object
    return dict(payload), {}


def _save_preset(
    *,
    path: Path,
    config_payload: dict[str, Any],
    tags: list[str],
    critical_fields: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "tags": tags,
            "critical_fields": critical_fields,
        },
        "config": config_payload,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _hist_ratio(hist: list[int], lo: int, hi: int) -> float:
    total = float(sum(hist))
    if total <= 0.0:
        return 0.0
    lo_i = max(0, int(lo))
    hi_i = min(255, int(hi))
    if hi_i < lo_i:
        return 0.0
    return float(sum(hist[lo_i : hi_i + 1])) / total


def _analyze_dryrun_image(image_path: Path) -> tuple[bool, list[str], dict[str, float]]:
    reasons: list[str] = []
    metrics: dict[str, float] = {}
    try:
        with Image.open(image_path) as img:
            gray = img.convert("L")
            if gray.width > 768:
                new_h = max(64, int(round(gray.height * (768.0 / float(gray.width)))))
                resampling = getattr(getattr(Image, "Resampling", Image), "BILINEAR")
                gray = gray.resize((768, new_h), resample=resampling)
            hist = gray.histogram()
            stat = ImageStat.Stat(gray)
            mean_l = float(stat.mean[0])
            std_l = float(stat.stddev[0])
            dark_ratio = _hist_ratio(hist, 0, 8)
            bright_ratio = _hist_ratio(hist, 220, 255)

            cw = gray.width
            ch = gray.height
            cx0 = int(cw * 0.35)
            cx1 = int(cw * 0.65)
            cy0 = int(ch * 0.35)
            cy1 = int(ch * 0.65)
            center = gray.crop((cx0, cy0, cx1, cy1))
            center_hist = center.histogram()
            center_dark_ratio = _hist_ratio(center_hist, 0, 16)

            band = max(2, int(round(min(cw, ch) * 0.05)))
            top = gray.crop((0, 0, cw, band))
            bottom = gray.crop((0, ch - band, cw, ch))
            left = gray.crop((0, 0, band, ch))
            right = gray.crop((cw - band, 0, cw, ch))
            border_bright_ratio = (
                _hist_ratio(top.histogram(), 210, 255)
                + _hist_ratio(bottom.histogram(), 210, 255)
                + _hist_ratio(left.histogram(), 210, 255)
                + _hist_ratio(right.histogram(), 210, 255)
            ) / 4.0

            metrics = {
                "mean_luma": mean_l,
                "std_luma": std_l,
                "dark_ratio": dark_ratio,
                "bright_ratio": bright_ratio,
                "center_dark_ratio": center_dark_ratio,
                "border_bright_ratio": border_bright_ratio,
            }

            # Strong failure signatures only: keep this conservative to avoid false negatives.
            if mean_l < 2.0 and std_l < 1.2:
                reasons.append("Frame quasi nero/uniforme (luma media e contrasto troppo bassi).")
            if dark_ratio > 0.985 and bright_ratio < 0.001:
                reasons.append("Frame quasi completamente nero (assenza segnale utile).")
            if center_dark_ratio < 0.06:
                reasons.append("Centro poco oscuro: possibile assenza della shadow del buco nero.")
            if border_bright_ratio > 0.30 and bright_ratio > 0.08:
                reasons.append("Bordo molto luminoso: possibile disco tagliato o inquadratura fuori scala.")
    except Exception as exc:
        reasons.append(f"Impossibile analizzare il dry-run: {exc}")
    return len(reasons) == 0, reasons, metrics


def _preflight_physical_checks(
    cfg_obj: RenderConfig,
    mode: str,
    video_params: dict[str, Any],
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    model = str(cfg_obj.metric_model)
    if model == "morris_thorne":
        if str(cfg_obj.coordinate_system) != "boyer_lindquist":
            errors.append("Morris-Thorne supporta solo coordinate in variabile areolare.")
        if bool(cfg_obj.enable_accretion_disk):
            warnings.append("Nel wormhole il disco di accrescimento è disattivato automaticamente.")
        return errors, warnings
    try:
        horizon = float(
            event_horizon_radius(
                cfg_obj.spin,
                cfg_obj.metric_model,
                cfg_obj.charge,
                cfg_obj.cosmological_constant,
            )
        )
    except Exception as exc:
        return [f"Impossibile determinare l'orizzonte degli eventi: {exc}"], warnings

    rin = float(cfg_obj.disk_inner_radius or 0.0)
    rout = float(cfg_obj.disk_outer_radius)
    robs = float(cfg_obj.observer_radius)
    coord = str(cfg_obj.coordinate_system)

    if robs < rout * 0.55:
        warnings.append(
            "Observer radius molto vicino al disco esterno: alto rischio di inquadratura parziale o deformazioni estreme."
        )
    if robs > 120.0 and mode == "video":
        warnings.append(
            "Observer radius molto grande: il buco nero potrebbe risultare troppo piccolo nel video."
        )
    if rin < max(1.03 * horizon, horizon + 0.05):
        warnings.append(
            "r_in è molto vicino all'orizzonte: il disco interno può diventare numericamente instabile."
        )

    if model.endswith("_de_sitter"):
        try:
            roots = horizon_radii(cfg_obj.spin, model, cfg_obj.charge, cfg_obj.cosmological_constant)
            if len(roots) >= 2:
                cosmological_horizon = float(roots[-1])
                if robs > 0.94 * cosmological_horizon:
                    warnings.append(
                        "Observer radius vicino all'orizzonte cosmologico: possibile distorsione non fisica del background."
                    )
        except Exception:
            pass

    if mode == "video":
        r_start = video_params.get("observer_radius_start")
        r_end = video_params.get("observer_radius_end")
        for label, val in (("observer_radius_start", r_start), ("observer_radius_end", r_end)):
            if val is None:
                continue
            rv = float(val)
            if coord != "generalized_doran" and rv <= 1.01 * horizon:
                errors.append(
                    f"{label}={rv:.4g} è dentro/attaccato all'orizzonte ma il sistema di coordinate non è generalized_doran."
                )
            elif coord == "generalized_doran" and rv <= 1.01 * horizon:
                warnings.append(
                    f"{label}={rv:.4g} attraversa l'orizzonte (ok in generalized_doran, ma fisicamente molto estremo)."
                )
        i_start = video_params.get("inclination_start_deg")
        i_end = video_params.get("inclination_end_deg")
        if (i_start is not None) and (i_end is not None):
            if math.isclose(float(i_start), float(i_end), abs_tol=1.0e-6):
                warnings.append("Inclination sweep nullo: il video non cambierà piano di vista.")
    return errors, warnings


def main() -> None:
    if "ui_lang" not in st.session_state:
        st.session_state["ui_lang"] = "it"
    lang = str(st.session_state.get("ui_lang", "it"))
    if lang not in LANGUAGE_OPTIONS:
        lang = "it"
        st.session_state["ui_lang"] = "it"
    if "ui_theme_mode" not in st.session_state:
        st.session_state["ui_theme_mode"] = "auto"
    theme_mode = str(st.session_state.get("ui_theme_mode", "auto")).lower().strip()
    if theme_mode not in {"auto", "dark", "light"}:
        theme_mode = "auto"
        st.session_state["ui_theme_mode"] = "auto"

    st.set_page_config(page_title=tr(lang, "page_title", "KerrTrace WebUI"), layout="wide")
    dark_vars = """
  --kt-bg: #0a0a0c;
  --kt-surface: #141418;
  --kt-surface-2: #1e1f25;
  --kt-border: #343741;
  --kt-text: #f2f3f7;
  --kt-muted: #a4a8b3;
  --kt-accent: #c27b2b;
  --kt-disabled-bg: #20242c;
  --kt-disabled-tx: #7f8694;
  --kt-input-bg: #24262e;
  --kt-input-disabled-bg: #181b22;
  --kt-input-disabled-tx: #7b8392;
  --kt-tooltip-bg: #1a1d24;
  --kt-tooltip-text: #f2f3f7;
"""
    light_vars = """
  --kt-bg: #eee8dc;
  --kt-surface: #ffffff;
  --kt-surface-2: #e4d8c2;
  --kt-border: #c7b391;
  --kt-text: #171411;
  --kt-muted: #585046;
  --kt-accent: #2f6df6;
  --kt-disabled-bg: #d6c6aa;
  --kt-disabled-tx: #5c5140;
  --kt-input-bg: #fbf7ef;
  --kt-input-disabled-bg: #cbb89a;
  --kt-input-disabled-tx: #54493b;
  --kt-tooltip-bg: #fffaf1;
  --kt-tooltip-text: #171411;
"""
    if theme_mode == "dark":
        root_block = ":root {\n" + dark_vars + "\n}"
    elif theme_mode == "light":
        root_block = ":root {\n" + light_vars + "\n}"
    else:
        root_block = (
            ":root {\n"
            + dark_vars
            + "\n}\n@media (prefers-color-scheme: light) {\n  :root {\n"
            + light_vars
            + "\n  }\n}"
        )

    ui_css = (
        "<style>\n"
        + root_block
        + """
html, body, .stApp, [data-testid="stAppViewContainer"] {
  background: var(--kt-bg) !important;
  color: var(--kt-text) !important;
}
[data-testid="stMainBlockContainer"] {
  background: transparent !important;
}
[data-testid="stHeader"] {
  background: transparent !important;
  border-bottom: 1px solid var(--kt-border) !important;
}
[data-testid="stSidebar"] {
  background: var(--kt-surface) !important;
  border-right: 1px solid var(--kt-border) !important;
}
section.main > div.block-container {
  padding-top: 0.9rem !important;
  padding-bottom: 1.4rem !important;
  max-width: 1700px !important;
}
h1, h2, h3 {
  letter-spacing: -0.015em;
}
h1, h2, h3, h4, h5, h6, p, li, label, span, div, small {
  color: var(--kt-text);
}
[data-testid="stWidgetLabel"],
[data-testid="stWidgetLabel"] *,
[data-testid="stSelectbox"] label,
[data-testid="stNumberInput"] label,
[data-testid="stTextInput"] label,
[data-testid="stTextArea"] label,
[data-testid="stSlider"] label,
[data-testid="stCheckbox"] label,
[data-testid="stRadio"] label,
div[role="radiogroup"] label,
div[role="radiogroup"] label *,
[data-testid="stFileUploader"] label {
  color: var(--kt-text) !important;
  opacity: 1 !important;
}
[data-testid="stMarkdownContainer"] * {
  color: var(--kt-text);
}

div[data-testid="stExpander"] {
  border: 1px solid var(--kt-border) !important;
  border-radius: 12px !important;
  background: var(--kt-surface) !important;
  overflow: hidden !important;
}
div[data-testid="stExpander"] details {
  border-radius: 12px !important;
  overflow: hidden !important;
}
div[data-testid="stExpander"] details > summary {
  color: var(--kt-text) !important;
  background: var(--kt-surface-2) !important;
  border-bottom: none !important;
  border-radius: 10px !important;
  margin: 0 !important;
  box-sizing: border-box !important;
}
div[data-testid="stExpander"] details[open] > summary {
  background: var(--kt-surface-2) !important;
  border-bottom: 1px solid var(--kt-border) !important;
  border-radius: 10px 10px 0 0 !important;
}
div[data-testid="stExpander"] details[open] > div {
  border-radius: 0 0 10px 10px !important;
  overflow: hidden !important;
}

div[data-baseweb="select"] > div,
div[data-testid="stTextInput"] input,
div[data-testid="stNumberInput"] input,
div[data-testid="stTextArea"] textarea,
div[data-testid="stFileUploader"] section {
  background: var(--kt-input-bg) !important;
  color: var(--kt-text) !important;
  border: 1px solid var(--kt-border) !important;
  border-radius: 10px !important;
}
div[data-baseweb="select"] * {
  color: var(--kt-text) !important;
}
div[data-baseweb="menu"],
ul[role="listbox"],
div[role="listbox"] {
  background: var(--kt-surface) !important;
  color: var(--kt-text) !important;
  border: 1px solid var(--kt-border) !important;
}
[data-baseweb="menu"] {
  background: transparent !important;
}
li[role="option"],
div[role="option"] {
  background: var(--kt-surface) !important;
  color: var(--kt-text) !important;
}
li[role="option"][aria-selected="true"],
div[role="option"][aria-selected="true"] {
  background: var(--kt-surface-2) !important;
}
li[role="option"]:hover,
div[role="option"]:hover {
  background: var(--kt-surface-2) !important;
}
li[role="option"] *,
div[role="option"] * {
  color: var(--kt-text) !important;
}
[data-baseweb="tooltip"],
div[role="tooltip"] {
  background: var(--kt-tooltip-bg) !important;
  color: var(--kt-tooltip-text) !important;
  border: 1px solid var(--kt-border) !important;
  max-width: min(560px, 82vw) !important;
  white-space: normal !important;
  overflow-wrap: anywhere !important;
  line-height: 1.35 !important;
  padding: 0.45rem 0.6rem !important;
  border-radius: 8px !important;
}
[data-baseweb="tooltip"] *,
div[role="tooltip"] * {
  color: var(--kt-tooltip-text) !important;
}
[data-testid="stTooltipIcon"] button,
[data-testid="stTooltipIcon"] [role="button"] {
  background: var(--kt-surface-2) !important;
  border: 1px solid var(--kt-border) !important;
  color: var(--kt-text) !important;
}
[data-testid="stTooltipIcon"] svg {
  fill: var(--kt-text) !important;
  stroke: var(--kt-text) !important;
}
div[data-testid="stTextInput"] input:focus,
div[data-testid="stNumberInput"] input:focus,
div[data-testid="stTextArea"] textarea:focus {
  border-color: var(--kt-accent) !important;
  box-shadow: 0 0 0 1px var(--kt-accent) !important;
}

div[data-testid="stNumberInput"] input:disabled,
div[data-testid="stTextInput"] input:disabled,
div[data-testid="stTextArea"] textarea:disabled {
  background-color: var(--kt-input-disabled-bg) !important;
  color: var(--kt-input-disabled-tx) !important;
  -webkit-text-fill-color: var(--kt-input-disabled-tx) !important;
  border: 1px solid var(--kt-border) !important;
  opacity: 1 !important;
}

div.stButton > button {
  background: var(--kt-surface-2) !important;
  color: var(--kt-text) !important;
  border-radius: 10px !important;
  border: 1px solid var(--kt-border) !important;
}
button[kind="secondary"],
button[kind="tertiary"],
button[aria-haspopup="dialog"],
[data-testid="stDownloadButton"] > button {
  background: var(--kt-surface-2) !important;
  color: var(--kt-text) !important;
  border: 1px solid var(--kt-border) !important;
}
button[aria-haspopup="dialog"] *,
[data-testid="stDownloadButton"] > button * {
  color: var(--kt-text) !important;
  fill: var(--kt-text) !important;
}
div.stButton > button[kind="primary"] {
  background: var(--kt-accent) !important;
  border-color: var(--kt-accent) !important;
  color: #ffffff !important;
}
div.stButton > button:hover {
  filter: brightness(1.04);
}

small, .stCaption {
  color: var(--kt-muted) !important;
}
pre, code {
  border-radius: 10px !important;
}
[data-testid="stCodeBlock"] pre, [data-testid="stCodeBlock"] code {
  color: inherit !important;
}
[data-testid="stCodeBlock"] pre {
  background: var(--kt-surface-2) !important;
  border: 1px solid var(--kt-border) !important;
}
code {
  background: var(--kt-surface-2) !important;
  color: var(--kt-text) !important;
}
[data-testid="stFileUploader"] section {
  background: var(--kt-input-bg) !important;
  border: 1px solid var(--kt-border) !important;
}
[data-testid="stFileUploader"] button {
  background: var(--kt-surface) !important;
  color: var(--kt-text) !important;
  border: 1px solid var(--kt-border) !important;
}
[data-testid="stCheckbox"] [role="checkbox"] {
  background: var(--kt-surface) !important;
  border: 1px solid var(--kt-border) !important;
}
[data-baseweb="radio"] > div {
  color: var(--kt-text) !important;
}
div[data-testid="stNumberInput"] button,
div[data-testid="stNumberInput"] [role="button"] {
  background: var(--kt-input-bg) !important;
  color: var(--kt-text) !important;
  border-left: 1px solid var(--kt-border) !important;
}
div[data-testid="stNumberInput"] button svg {
  fill: var(--kt-text) !important;
  stroke: var(--kt-text) !important;
}
</style>
"""
    )
    st.markdown(ui_css, unsafe_allow_html=True)
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
    if "async_queue" not in st.session_state:
        st.session_state["async_queue"] = []
    if "job_history" not in st.session_state:
        st.session_state["job_history"] = []
    if "job_counter" not in st.session_state:
        st.session_state["job_counter"] = 0
    if "bg_launch_notice" not in st.session_state:
        st.session_state["bg_launch_notice"] = ""
    if "preset_loaded_cfg" not in st.session_state:
        st.session_state["preset_loaded_cfg"] = {}
    if "preset_lock_active" not in st.session_state:
        st.session_state["preset_lock_active"] = False
    if "preset_locked_values" not in st.session_state:
        st.session_state["preset_locked_values"] = {}
    if "preset_locked_fields" not in st.session_state:
        st.session_state["preset_locked_fields"] = []
    if "preset_loaded_name" not in st.session_state:
        st.session_state["preset_loaded_name"] = ""
    if "pending_video_render" not in st.session_state:
        st.session_state["pending_video_render"] = {}
    if "pending_video_skip_clear_once" not in st.session_state:
        st.session_state["pending_video_skip_clear_once"] = False
    if "bg_auto_refresh" not in st.session_state:
        st.session_state["bg_auto_refresh"] = True
    if "bg_auto_refresh_interval_s" not in st.session_state:
        st.session_state["bg_auto_refresh_interval_s"] = 3
    launch_notice = str(st.session_state.get("bg_launch_notice") or "").strip()
    if launch_notice:
        st.success(launch_notice)
        st.session_state["bg_launch_notice"] = ""
    last_output_raw = str(st.session_state.get("last_output_path") or "").strip()
    if last_output_raw:
        last_output = Path(last_output_raw)
        if last_output.exists():
            with st.expander(tr(lang, "last_output", "Anteprima ultimo output"), expanded=True):
                _show_output_media(last_output)

    async_proc = st.session_state.get("async_proc")
    async_meta = st.session_state.get("async_meta") or {}
    queue_entries = list(st.session_state.get("async_queue") or [])
    history_entries = list(st.session_state.get("job_history") or [])

    # Finalize completed active job and auto-start next queued job.
    if async_proc is not None and isinstance(async_meta, dict) and async_meta:
        rc_now = async_proc.poll()
        if rc_now is not None and not bool(async_meta.get("history_recorded", False)):
            finished_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            history_entries.append(
                {
                    "job_id": str(async_meta.get("job_id", "job?")),
                    "status": "done" if int(rc_now) == 0 else "failed",
                    "exit": int(rc_now),
                    "started_at": str(async_meta.get("started_at", "")),
                    "finished_at": finished_ts,
                    "output_hint": str(async_meta.get("output_hint", "")),
                    "log_path": str(async_meta.get("log_path", "")),
                }
            )
            st.session_state["job_history"] = history_entries[-100:]
            async_meta["history_recorded"] = True
            st.session_state["async_meta"] = async_meta
            queue_entries = list(st.session_state.get("async_queue") or [])
            if queue_entries:
                next_job = queue_entries.pop(0)
                st.session_state["async_queue"] = queue_entries
                proc_next, meta_next = _launch_background_process(
                    cmd=list(next_job["cmd"]),
                    workspace_path=Path(str(next_job["workspace"])),
                    log_path=Path(str(next_job["log_path"])),
                    cfg_path=Path(str(next_job["cfg_path"])),
                    output_hint=str(next_job["output_hint"]),
                    stamp=str(next_job["stamp"]),
                    job_id=str(next_job["job_id"]),
                )
                st.session_state["async_proc"] = proc_next
                st.session_state["async_meta"] = meta_next
                async_proc = proc_next
                async_meta = meta_next
                st.info(f"{tr(lang, 'queued_job_started', 'Avviato job in coda')}: `{meta_next['job_id']}`")

    queue_entries = list(st.session_state.get("async_queue") or [])
    history_entries = list(st.session_state.get("job_history") or [])
    with st.expander(tr(lang, "queue_job_background", "Queue job (background)"), expanded=bool(queue_entries or history_entries)):
        st.caption(
            f"Queued: {len(queue_entries)} | Running: "
            f"{1 if (async_proc is not None and async_proc.poll() is None) else 0} | "
            f"History: {len(history_entries)}"
        )
        q_col_1, q_col_2 = st.columns(2)
        with q_col_1:
            if queue_entries:
                st.write(tr(lang, "in_queue_label", "In coda:"))
                st.table(
                    [
                        {
                            "job_id": str(item.get("job_id", "")),
                            "queued_at": str(item.get("stamp", "")),
                            "output": str(item.get("output_hint", "")),
                        }
                        for item in queue_entries[:10]
                    ]
                )
            else:
                st.caption(tr(lang, "no_jobs_in_queue", "Nessun job in coda."))
        with q_col_2:
            if history_entries:
                st.write(tr(lang, "recent_history", "Storico recente:"))
                st.table(
                    [
                        {
                            "job_id": str(item.get("job_id", "")),
                            "status": str(item.get("status", "")),
                            "exit": str(item.get("exit", "")),
                            "output": str(item.get("output_hint", "")),
                        }
                        for item in history_entries[-10:]
                    ]
                )
            else:
                st.caption(tr(lang, "history_empty", "Storico vuoto."))
        q_btn_1, q_btn_2 = st.columns(2)
        with q_btn_1:
            if st.button(tr(lang, "clear_queue", "Svuota coda"), key="queue_clear_pending"):
                st.session_state["async_queue"] = []
                st.rerun()
        with q_btn_2:
            if st.button(tr(lang, "clear_history", "Pulisci storico"), key="queue_clear_history"):
                st.session_state["job_history"] = []
                st.rerun()

    if async_proc is not None and isinstance(async_meta, dict) and async_meta:
        auto_refresh_enabled = bool(st.session_state.get("bg_auto_refresh", True))
        auto_refresh_interval_s = int(max(1, min(30, int(st.session_state.get("bg_auto_refresh_interval_s", 3)))))
        run_every: str | None = None
        try:
            if async_proc.poll() is None and auto_refresh_enabled:
                run_every = f"{auto_refresh_interval_s}s"
        except Exception:
            run_every = None

        @st.fragment(run_every=run_every)
        def _render_background_job_fragment() -> None:
            async_proc_local = st.session_state.get("async_proc")
            async_meta_local = st.session_state.get("async_meta") or {}
            if async_proc_local is None or (not isinstance(async_meta_local, dict)) or (not async_meta_local):
                return

            log_path = Path(str(async_meta_local.get("log_path", "")))
            workspace_async = Path(str(async_meta_local.get("workspace", str(Path.cwd()))))
            out_hint = Path(str(async_meta_local.get("output_hint", "out/webui_frame.png")))
            started_at = str(async_meta_local.get("started_at", ""))
            elapsed_bg_s: float | None = None
            try:
                if started_at:
                    started_dt = datetime.strptime(started_at, "%Y%m%d_%H%M%S")
                    elapsed_bg_s = max(0.0, (datetime.now() - started_dt).total_seconds())
            except Exception:
                elapsed_bg_s = None
            elapsed_bg_txt = _format_eta_short(elapsed_bg_s) if elapsed_bg_s is not None else "--:--"
            cfg_async = str(async_meta_local.get("cfg_path", ""))
            rc = async_proc_local.poll()
            running = rc is None

            with st.expander(tr(lang, "bg_job", "Job in background"), expanded=True):
                r1, r2 = st.columns(2)
                with r1:
                    st.checkbox(
                        tr(lang, "auto_refresh_monitor", "Auto-refresh monitor"),
                        key="bg_auto_refresh",
                    )
                with r2:
                    st.number_input(
                        tr(lang, "auto_refresh_interval_sec", "Intervallo auto-refresh (s)"),
                        min_value=1,
                        max_value=30,
                        step=1,
                        key="bg_auto_refresh_interval_s",
                    )

                if running and bool(st.session_state.get("bg_auto_refresh", True)):
                    refresh_s = int(max(1, min(30, int(st.session_state.get("bg_auto_refresh_interval_s", 3)))))
                    st.caption(
                        tr(
                            lang,
                            "auto_refresh_active_every",
                            "Auto-refresh attivo: aggiornamento ogni {seconds}s.",
                        ).format(seconds=refresh_s)
                    )

                if running:
                    st.info(
                        f"{tr(lang, 'job_running', 'In esecuzione')} (PID {getattr(async_proc_local, 'pid', 'n/a')}) "
                        f"- started: {started_at} - elapsed: {elapsed_bg_txt}"
                    )
                else:
                    if int(rc) == 0:
                        st.success(f"{tr(lang, 'job_completed', 'Job completato')} (exit={rc})")
                    else:
                        st.error(f"{tr(lang, 'job_failed', 'Job terminato con errore')} (exit={rc})")
                if cfg_async:
                    st.caption(f"{tr(lang, 'cfg_used', 'Config JSON usata:')} {cfg_async}")
                if log_path:
                    st.caption(f"{tr(lang, 'log_label', 'Log')}: {log_path}")
                tail_txt = _tail_text_file(log_path)
                if tail_txt:
                    progress_hint = _extract_progress_from_log(tail_txt)
                    if progress_hint is not None:
                        ratio, progress_label = progress_hint
                        st.progress(float(ratio), text=progress_label)
                    elif running:
                        st.progress(0.0, text=tr(lang, "render_rows_initial", "Render rows: 0/0 (0.0%)"))
                    st.code(tail_txt, language="bash")
                    eta_hint = _estimate_eta_from_log(tail_txt)
                    if eta_hint:
                        st.caption(eta_hint)
                    live_frame = _extract_latest_frame_path_from_log(tail_txt, workspace_async)
                    if live_frame is not None and live_frame.exists():
                        st.image(
                            str(live_frame),
                            caption=f"{tr(lang, 'live_frame_preview', 'Live frame preview')}: {live_frame.name}",
                        )

                c_job_1, c_job_2 = st.columns(2)
                with c_job_1:
                    if running:
                        if st.button(tr(lang, "refresh_monitor", "Aggiorna monitor"), key="refresh_monitor_running"):
                            st.rerun()
                with c_job_2:
                    if running and st.button(tr(lang, "stop_job", "Interrompi job"), key="stop_job_running"):
                        try:
                            async_proc_local.terminate()
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
                    if st.button(tr(lang, "clear_job", "Pulisci stato job"), key="clear_job_done"):
                        st.session_state["async_proc"] = None
                        st.session_state["async_meta"] = {}
                        st.rerun()

        _render_background_job_fragment()

    default_cfg = asdict(RenderConfig())
    supports_adaptive_spatial = "adaptive_spatial_sampling" in default_cfg
    loaded_cfg: dict[str, Any] = {}
    save_preset_requested = False
    preset_save_name = ""
    preset_save_tags_raw = ""
    preset_save_lock_fields = True

    with st.sidebar:
        st.header(tr(lang, "run_header", "Run"))
        st.caption(f"{tr(lang, 'author_label', 'Autore')}: {AUTHOR_SIGNATURE}")
        lang_options = [code for code, _ in sorted(LANGUAGE_OPTIONS.items(), key=lambda kv: kv[1].casefold())]
        st.selectbox(
            tr(lang, "language_label", "Lingua / Language"),
            options=lang_options,
            key="ui_lang",
            format_func=lambda code: LANGUAGE_OPTIONS.get(code, code),
        )
        theme_options = ["auto", "dark", "light"]
        theme_labels = {
            "auto": tr(lang, "theme_auto", "Auto (browser/system)"),
            "dark": tr(lang, "theme_dark", "Dark"),
            "light": tr(lang, "theme_light", "Light"),
        }
        st.selectbox(
            tr(lang, "theme_label", "Tema UI"),
            options=theme_options,
            key="ui_theme_mode",
            format_func=lambda code: theme_labels.get(code, code),
        )
        lang = str(st.session_state.get("ui_lang", "it"))
        python_exec = st.text_input(tfield(lang, "Python executable"), value=_default_python())
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

        try:
            workspace_sidebar = _validate_workspace_path(workspace)
        except ValueError as exc:
            st.error(str(exc))
            return
        preset_files = _list_presets(workspace_sidebar)
        preset_options = [""] + [p.stem for p in preset_files]
        with st.expander(tr(lang, "preset_manager", "Preset manager"), expanded=False):
            if st.session_state.get("preset_loaded_name"):
                st.caption(
                    f"{tr(lang, 'preset_active', 'Preset attivo')}: `{st.session_state.get('preset_loaded_name')}` | "
                    f"lock={bool(st.session_state.get('preset_lock_active', False))}"
                )
            preset_to_load = st.selectbox(
                tr(lang, "load_preset", "Load preset"),
                options=preset_options,
                format_func=lambda name: tr(lang, "preset_none", "(none)") if not name else name,
                key="preset_to_load_select",
            )
            lock_on_load = st.checkbox(
                tr(lang, "lock_critical_fields_on_load", "Lock critical fields on load"),
                value=bool(st.session_state.get("preset_lock_active", False)),
                key="preset_lock_on_load",
            )
            if st.button(tr(lang, "load_selected_preset", "Load selected preset"), disabled=(not preset_to_load), key="preset_load_btn"):
                try:
                    preset_path = next(p for p in preset_files if p.stem == preset_to_load)
                    cfg_payload, meta_payload = _load_preset(preset_path)
                    st.session_state["preset_loaded_cfg"] = cfg_payload
                    st.session_state["preset_loaded_name"] = preset_to_load
                    if lock_on_load:
                        locked_fields = list(meta_payload.get("critical_fields") or PRESET_CRITICAL_FIELDS)
                        st.session_state["preset_lock_active"] = True
                        st.session_state["preset_locked_fields"] = locked_fields
                        st.session_state["preset_locked_values"] = {
                            k: cfg_payload.get(k)
                            for k in locked_fields
                            if k in cfg_payload
                        }
                    else:
                        st.session_state["preset_lock_active"] = False
                        st.session_state["preset_locked_fields"] = []
                        st.session_state["preset_locked_values"] = {}
                    st.success(f"{tr(lang, 'preset_loaded', 'Preset caricato')}: {preset_to_load}")
                    st.rerun()
                except Exception as exc:
                    st.error(f"{tr(lang, 'preset_load_error', 'Errore load preset')}: {exc}")
            if st.button(tr(lang, "unlock_preset_critical_fields", "Unlock preset critical fields"), key="preset_unlock_btn"):
                st.session_state["preset_lock_active"] = False
                st.session_state["preset_locked_fields"] = []
                st.session_state["preset_locked_values"] = {}
                st.rerun()
            st.divider()
            preset_save_name = st.text_input(tr(lang, "preset_name", "Preset name"), value="")
            preset_save_tags_raw = st.text_input(tr(lang, "preset_tags_comma", "Preset tags (comma separated)"), value="")
            preset_save_lock_fields = st.checkbox(tr(lang, "store_critical_lock_fields", "Store critical lock fields"), value=True)
            save_preset_requested = st.button(tr(lang, "save_preset", "Save preset"), key="preset_save_btn")

    preset_cfg = st.session_state.get("preset_loaded_cfg") or {}
    if isinstance(preset_cfg, dict) and preset_cfg:
        # Uploaded JSON overrides loaded preset values when keys overlap.
        loaded_cfg = {**preset_cfg, **loaded_cfg}

    cfg_seed = dict(default_cfg)
    cfg_seed.update(loaded_cfg)
    for _k, _v in WEBUI_BASE_DEFAULTS.items():
        if _k not in loaded_cfg:
            cfg_seed[_k] = _v
    cfg_seed.setdefault("adaptive_spatial_sampling", bool(default_cfg.get("adaptive_spatial_sampling", False)))
    cfg_seed.setdefault("adaptive_spatial_preview_steps", int(default_cfg.get("adaptive_spatial_preview_steps", 96)))
    cfg_seed.setdefault("adaptive_spatial_min_scale", float(default_cfg.get("adaptive_spatial_min_scale", 0.65)))
    cfg_seed.setdefault("adaptive_spatial_quantile", float(default_cfg.get("adaptive_spatial_quantile", 0.78)))
    cfg_seed.setdefault("disk_layer_accident_strength", float(default_cfg.get("disk_layer_accident_strength", 0.42)))
    cfg_seed.setdefault("disk_layer_accident_count", float(default_cfg.get("disk_layer_accident_count", 3.8)))
    cfg_seed.setdefault("disk_layer_accident_sharpness", float(default_cfg.get("disk_layer_accident_sharpness", 7.0)))
    cfg_seed.setdefault("disk_layer_global_phase", float(default_cfg.get("disk_layer_global_phase", 0.0)))
    cfg_seed.setdefault("disk_layer_phase_rate_hz", float(default_cfg.get("disk_layer_phase_rate_hz", 0.35)))
    cfg_seed.setdefault(
        "enable_disk_differential_rotation",
        bool(default_cfg.get("enable_disk_differential_rotation", False)),
    )
    cfg_seed.setdefault("disk_diffrot_model", str(default_cfg.get("disk_diffrot_model", "keplerian_lut")))
    cfg_seed.setdefault(
        "disk_diffrot_visual_mode",
        str(default_cfg.get("disk_diffrot_visual_mode", "layer_phase")),
    )
    cfg_seed.setdefault("disk_diffrot_strength", float(default_cfg.get("disk_diffrot_strength", 1.0)))
    cfg_seed.setdefault("disk_diffrot_seed", int(default_cfg.get("disk_diffrot_seed", 7)))
    cfg_seed.setdefault("disk_diffrot_iteration", str(default_cfg.get("disk_diffrot_iteration", "v1_basic")))
    cfg_seed.setdefault("roi_supersampling", bool(default_cfg.get("roi_supersampling", False)))
    cfg_seed.setdefault("roi_supersample_threshold", float(default_cfg.get("roi_supersample_threshold", 0.92)))
    cfg_seed.setdefault("roi_supersample_jitter", float(default_cfg.get("roi_supersample_jitter", 0.35)))
    cfg_seed.setdefault("roi_supersample_samples", int(default_cfg.get("roi_supersample_samples", 2)))
    cfg_seed.setdefault("persistent_cache_enabled", bool(default_cfg.get("persistent_cache_enabled", True)))
    cfg_seed.setdefault("persistent_cache_dir", str(default_cfg.get("persistent_cache_dir", "out/cache")))
    cfg_seed.setdefault("quality_lock", bool(default_cfg.get("quality_lock", False)))
    cfg_seed.setdefault("quality_lock_psnr_min", float(default_cfg.get("quality_lock_psnr_min", 45.0)))
    cfg_seed.setdefault("quality_lock_ssim_min", float(default_cfg.get("quality_lock_ssim_min", 0.985)))
    cfg_seed.setdefault("quality_lock_sample_width", int(default_cfg.get("quality_lock_sample_width", 256)))
    cfg_seed.setdefault("quality_lock_sample_height", int(default_cfg.get("quality_lock_sample_height", 144)))
    cfg_seed.setdefault(
        "quality_lock_fallback_to_baseline",
        bool(default_cfg.get("quality_lock_fallback_to_baseline", True)),
    )
    cfg_seed.setdefault("animation_workers", int(default_cfg.get("animation_workers", 1)))
    cfg_seed.setdefault("stream_encode_async", bool(default_cfg.get("stream_encode_async", True)))
    cfg_seed.setdefault("stream_encode_queue_size", int(default_cfg.get("stream_encode_queue_size", 4)))
    cfg_seed.setdefault("mps_auto_chunking", bool(default_cfg.get("mps_auto_chunking", True)))

    workspace_path_preview = workspace_sidebar
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
        st.divider()
        st.caption(
            tr(
                lang,
                "local_picker_hint",
                "Questa opzione apre il selettore file del browser per scegliere un file locale.",
            )
        )
        uploaded_media = st.file_uploader(
            tr(lang, "browse_local_file", "Sfoglia file dal computer"),
            type=["png", "jpg", "jpeg", "webp", "mp4", "mov", "mkv", "gif"],
            key="manual_open_local_picker",
        )
        if uploaded_media is not None:
            st.caption(f"{tr(lang, 'local_file_selected', 'File locale selezionato')}: `{uploaded_media.name}`")
            ok, reason = _show_uploaded_media(uploaded_media)
            if ok:
                st.session_state["manual_open_path"] = str(uploaded_media.name)
            elif reason == "unsupported":
                st.error(tr(lang, "unsupported_upload", "Estensione file caricato non supportata per preview."))
            elif reason == "read_error":
                st.error(tr(lang, "upload_read_error", "Impossibile leggere il file caricato."))

    st.subheader(tr(lang, "mode_header", "Modalità"))
    mode = st.radio(
        tr(lang, "mode_label", "Tipo simulazione"),
        ["single_frame", "video", "starship_frame", "starship_video"],
        format_func=lambda v: tr(
            lang,
            f"mode_{v}",
            "Single Frame"
            if v == "single_frame"
            else ("Video" if v == "video" else ("Starship Frame" if v == "starship_frame" else "Starship Video")),
        ),
        horizontal=True,
    )

    st.subheader(tr(lang, "quality_header", "Qualità / Risoluzione"))
    reverse_quality = {v: k for k, v in QUALITY_PRESETS.items()}
    current_wh = (int(cfg_seed["width"]), int(cfg_seed["height"]))
    preset_labels = ["Custom"] + list(QUALITY_PRESETS.keys())
    default_preset = reverse_quality.get(current_wh, "Custom")
    q_preset, q_w, q_h = st.columns([2.4, 1.0, 1.0])
    with q_preset:
        preset = st.selectbox(
            tr(lang, "quality_preset", "Preset qualità"),
            options=preset_labels,
            index=preset_labels.index(default_preset),
        )
    if preset == "Custom":
        with q_w:
            width = st.number_input(
                tfield(lang, "Width"),
                min_value=64,
                max_value=5000,
                value=int(cfg_seed["width"]),
                step=1,
                key="quality_width_custom",
            )
        with q_h:
            height = st.number_input(
                tfield(lang, "Height"),
                min_value=64,
                max_value=5000,
                value=int(cfg_seed["height"]),
                step=1,
                key="quality_height_custom",
            )
    else:
        width, height = QUALITY_PRESETS[preset]
        with q_w:
            st.number_input(
                tfield(lang, "Width"),
                min_value=64,
                max_value=5000,
                value=int(width),
                step=1,
                disabled=True,
                key="quality_width_preset",
            )
        with q_h:
            st.number_input(
                tfield(lang, "Height"),
                min_value=64,
                max_value=5000,
                value=int(height),
                step=1,
                disabled=True,
                key="quality_height_preset",
            )

    if mode == "video":
        output_default = "out/webui_video.mp4"
    elif mode == "starship_video":
        output_default = "out/webui_starship_video.mp4"
    elif mode == "starship_frame":
        output_default = "out/webui_starship.png"
    else:
        output_default = "out/webui_frame.png"
    output_seed_raw = str(loaded_cfg.get("output", output_default))
    output_seed_path = Path(output_seed_raw)

    row1_col1, row1_col2, row1_col3 = st.columns(3)
    with row1_col1:
        use_output_parts = st.checkbox(
            tfield(lang, "Compose output path from folder + filename"),
            value=False,
        )
    with row1_col2:
        fov_deg = st.number_input(tfield(lang, "FOV (deg)"), value=float(cfg_seed["fov_deg"]), step=0.1, format="%.3f")
    with row1_col3:
        metric_model = st.selectbox(
            tfield(lang, "Metric model"),
            options=CHOICE_FIELDS["metric_model"],
            index=CHOICE_FIELDS["metric_model"].index(
                _safe_choice(CHOICE_FIELDS["metric_model"], str(cfg_seed["metric_model"]))
            ),
        )

    coord_options = _coordinate_options_for_metric(metric_model)
    coord_seed = _safe_choice(coord_options, str(cfg_seed["coordinate_system"]))
    supports_spin, supports_charge, supports_lambda = _metric_supports_parameters(metric_model)
    spin_default = max(-1.0, min(1.0, float(cfg_seed["spin"])))
    charge_default = max(-1.0, min(1.0, float(cfg_seed["charge"])))
    theta_default = _clamp(float(cfg_seed["observer_inclination_deg"]), 0.0, 180.0)
    phi_default = _clamp(float(cfg_seed["observer_azimuth_deg"]), 0.0, 360.0)
    roll_default = _clamp(float(cfg_seed["observer_roll_deg"]), 0.0, 360.0)

    row2_col1, row2_col2, row2_col3 = st.columns(3)
    with row2_col1:
        if use_output_parts:
            output_dir_default = str(output_seed_path.parent) if str(output_seed_path.parent).strip() else "out"
            output_dir = st.text_input(tfield(lang, "Output directory"), value=output_dir_default)
        else:
            output_path = st.text_input(tfield(lang, "Output file"), value=output_seed_raw)
            output_dir = str(Path(output_path).expanduser().parent)
    with row2_col2:
        observer_radius = st.number_input(tfield(lang, "Observer radius"), value=float(cfg_seed["observer_radius"]), step=0.5)
    with row2_col3:
        coordinate_system = st.selectbox(
            tfield(lang, "Coordinate system"),
            options=coord_options,
            index=coord_options.index(coord_seed),
            format_func=lambda c: _coordinate_label_for_metric(c, metric_model),
            disabled=(not bool(metric_model)),
        )

    row3_col1, row3_col2, row3_col3 = st.columns(3)
    with row3_col1:
        if use_output_parts:
            output_name_default = output_seed_path.name.strip() or Path(output_default).name
            output_name = st.text_input(tfield(lang, "Output file name"), value=output_name_default)
            output_path = str(Path(output_dir).expanduser() / output_name)
            st.caption(f"{tfield(lang, 'Output file')}: `{output_path}`")
        else:
            st.text_input(
                tfield(lang, "Output file name"),
                value=Path(output_seed_raw).name,
                disabled=True,
                key="output_file_name_readonly_aligned",
            )
    with row3_col2:
        observer_inclination_deg = st.number_input(
            tfield(lang, "Observer inclination (deg)"),
            min_value=0.0,
            max_value=180.0,
            value=theta_default,
            step=0.5,
        )
    with row3_col3:
        disk_model = st.selectbox(
            tfield(lang, "Disk model"),
            options=CHOICE_FIELDS["disk_model"],
            index=CHOICE_FIELDS["disk_model"].index(_safe_choice(CHOICE_FIELDS["disk_model"], str(cfg_seed["disk_model"]))),
        )

    row4_col1, row4_col2, row4_col3 = st.columns(3)
    with row4_col1:
        observer_azimuth_deg = st.number_input(
            tfield(lang, "Observer azimuth (deg)"),
            min_value=0.0,
            max_value=360.0,
            value=phi_default,
            step=0.5,
        )
    with row4_col2:
        observer_roll_deg = st.number_input(
            tfield(lang, "Observer roll (deg)"),
            min_value=0.0,
            max_value=360.0,
            value=roll_default,
            step=0.5,
        )
    with row4_col3:
        disk_radial_profile = st.selectbox(
            tfield(lang, "Disk radial profile"),
            options=CHOICE_FIELDS["disk_radial_profile"],
            index=CHOICE_FIELDS["disk_radial_profile"].index(
                _safe_choice(CHOICE_FIELDS["disk_radial_profile"], str(cfg_seed["disk_radial_profile"]))
            ),
        )

    row5_col1, row5_col2, row5_col3 = st.columns(3)
    with row5_col1:
        spin = st.number_input(
            tfield(lang, "Spin a"),
            min_value=-1.0,
            max_value=1.0,
            value=(spin_default if supports_spin else 0.0),
            step=0.01,
            format="%.6f",
            disabled=(not supports_spin),
            help=_metric_param_tooltip(metric_model, "spin", supports_spin, lang),
        )
    with row5_col2:
        charge = st.number_input(
            tfield(lang, "Charge Q"),
            min_value=-1.0,
            max_value=1.0,
            value=(charge_default if supports_charge else 0.0),
            step=0.01,
            format="%.6f",
            disabled=(not supports_charge),
            help=_metric_param_tooltip(metric_model, "charge", supports_charge, lang),
        )
    with row5_col3:
        cosmological_constant = st.number_input(
            tfield(lang, "Lambda"),
            value=(float(cfg_seed["cosmological_constant"]) if supports_lambda else 0.0),
            step=0.000001,
            format="%.9f",
            disabled=(not supports_lambda),
            help=_metric_param_tooltip(metric_model, "lambda", supports_lambda, lang),
        )

    row6_col1, row6_col2, row6_col3 = st.columns(3)
    with row6_col3:
        disk_outer_radius = st.number_input(tfield(lang, "Disk outer radius"), value=float(cfg_seed["disk_outer_radius"]), step=0.5)

    if metric_model == "morris_thorne":
        st.caption(tr(lang, "morris_thorne_areolar_only", "Per Morris-Thorne è disponibile solo la coordinata in variabile areolare."))

    disabled_labels: list[str] = []
    if not supports_spin:
        disabled_labels.append("spin")
    if not supports_charge:
        disabled_labels.append("charge")
    if not supports_lambda:
        disabled_labels.append("lambda")
    if disabled_labels:
        st.caption(f"{tr(lang, 'unused_metric_params', 'Parametri non usati dalla metrica selezionata')}: {', '.join(disabled_labels)}.")

    st.subheader(tfield(lang, "Disk & Rendering"))
    perf_option_labels = {
        "manual": tfield(lang, "Manual"),
        "gpu_balanced": tfield(lang, "GPU Balanced (Recommended)"),
        "fast_preview": tfield(lang, "Fast Preview"),
        "high_fidelity": tfield(lang, "High Fidelity"),
    }
    perf_profile = st.selectbox(
        tfield(lang, "Performance profile"),
        options=list(perf_option_labels.keys()),
        index=0,
        format_func=lambda k: perf_option_labels.get(k, k),
        help=(
            "Manual: leaves settings unchanged. "
            "GPU Balanced: enables compile+mixed precision and optimized tiling. "
            "Fast Preview: reduces cost for quick previews. "
            "High Fidelity: prioritizes numeric stability."
        ),
    )
    d1, d2, d3, d4 = st.columns(4)
    with d1:
        use_default_rin = st.checkbox(tfield(lang, "r_in = default (ISCO)"), value=cfg_seed["disk_inner_radius"] is None)
        disk_inner_radius = None
        if not use_default_rin:
            disk_inner_radius = st.number_input(
                tfield(lang, "Disk inner radius"),
                value=float(cfg_seed["disk_inner_radius"] or 6.0),
                step=0.1,
            )
        disk_emission_gain = st.number_input(tfield(lang, "Disk emission gain"), value=float(cfg_seed["disk_emission_gain"]), step=0.5)
        disk_palette = st.selectbox(
            tfield(lang, "Disk palette"),
            options=CHOICE_FIELDS["disk_palette"],
            index=CHOICE_FIELDS["disk_palette"].index(
                _safe_choice(CHOICE_FIELDS["disk_palette"], str(cfg_seed.get("disk_palette", "default")))
            ),
        )
        disk_layered_palette = st.checkbox(
            tfield(lang, "Enable layered disk"),
            value=bool(cfg_seed.get("disk_layered_palette", False)),
        )
        with st.expander(tr(lang, "layered_disk_options", "Opzioni disco stratificato"), expanded=bool(disk_layered_palette)):
            disk_layer_count = st.number_input(
                tfield(lang, "Layer count"),
                min_value=2,
                max_value=512,
                value=int(cfg_seed.get("disk_layer_count", 12)),
                step=1,
                disabled=(not disk_layered_palette),
            )
            disk_layer_mix = st.slider(
                tfield(lang, "Layer mix"),
                min_value=0.0,
                max_value=1.0,
                value=float(cfg_seed.get("disk_layer_mix", 0.55)),
                step=0.01,
                disabled=(not disk_layered_palette),
            )
            disk_layer_accident_strength = st.number_input(
                tfield(lang, "Layer accident strength"),
                min_value=0.0,
                max_value=4.0,
                value=float(cfg_seed.get("disk_layer_accident_strength", 0.42)),
                step=0.05,
                disabled=(not disk_layered_palette),
            )
            disk_layer_accident_count = st.number_input(
                tfield(lang, "Layer accident count"),
                min_value=0.0,
                max_value=128.0,
                value=float(cfg_seed.get("disk_layer_accident_count", 3.8)),
                step=0.1,
                disabled=(not disk_layered_palette),
            )
            disk_layer_accident_sharpness = st.number_input(
                tfield(lang, "Layer accident sharpness"),
                min_value=1.0,
                max_value=32.0,
                value=float(cfg_seed.get("disk_layer_accident_sharpness", 7.0)),
                step=0.5,
                disabled=(not disk_layered_palette),
            )
            disk_layer_global_phase = st.number_input(
                tfield(lang, "Layer global phase"),
                value=float(cfg_seed.get("disk_layer_global_phase", 0.0)),
                step=0.1,
                disabled=(not disk_layered_palette),
            )
            disk_layer_phase_rate_hz = st.number_input(
                tfield(lang, "Layer phase rate (Hz)"),
                min_value=0.0,
                max_value=64.0,
                value=float(cfg_seed.get("disk_layer_phase_rate_hz", 0.35)),
                step=0.05,
                disabled=(not disk_layered_palette),
            )
        st.markdown(f"**{tfield(lang, 'Differential rotation')}**")
        disk_diffrot_enabled = st.checkbox(
            tfield(lang, "Enable differential disk rotation"),
            value=bool(cfg_seed.get("enable_disk_differential_rotation", False)),
        )
        diffrot_model_labels = {
            "keplerian_lut": "Keplerian LUT",
            "keplerian_metric": "Keplerian metric",
        }
        diffrot_visual_labels = {
            "layer_phase": "Layer phase",
            "annular_tiles": "Annular tiles",
            "hybrid": "Hybrid",
        }
        diffrot_iteration_labels = {
            "v1_basic": "v1 basic",
            "v2_visibility": "v2 visibility",
            "v3_robust": "v3 robust",
        }
        with st.expander(tr(lang, "differential_rotation_options", "Opzioni rotazione differenziale"), expanded=bool(disk_diffrot_enabled)):
            disk_diffrot_model = st.selectbox(
                tfield(lang, "Differential rotation model"),
                options=CHOICE_FIELDS["disk_diffrot_model"],
                index=CHOICE_FIELDS["disk_diffrot_model"].index(
                    _safe_choice(CHOICE_FIELDS["disk_diffrot_model"], str(cfg_seed.get("disk_diffrot_model", "keplerian_lut")))
                ),
                format_func=lambda key: diffrot_model_labels.get(key, key),
                disabled=(not disk_diffrot_enabled),
            )
            disk_diffrot_visual_mode = st.selectbox(
                tfield(lang, "Differential rotation visual mode"),
                options=CHOICE_FIELDS["disk_diffrot_visual_mode"],
                index=CHOICE_FIELDS["disk_diffrot_visual_mode"].index(
                    _safe_choice(
                        CHOICE_FIELDS["disk_diffrot_visual_mode"],
                        str(cfg_seed.get("disk_diffrot_visual_mode", "layer_phase")),
                    )
                ),
                format_func=lambda key: diffrot_visual_labels.get(key, key),
                disabled=(not disk_diffrot_enabled),
            )
            disk_diffrot_iteration = st.selectbox(
                tfield(lang, "Differential rotation iteration"),
                options=CHOICE_FIELDS["disk_diffrot_iteration"],
                index=CHOICE_FIELDS["disk_diffrot_iteration"].index(
                    _safe_choice(CHOICE_FIELDS["disk_diffrot_iteration"], str(cfg_seed.get("disk_diffrot_iteration", "v1_basic")))
                ),
                format_func=lambda key: diffrot_iteration_labels.get(key, key),
                disabled=(not disk_diffrot_enabled),
            )
            disk_diffrot_strength = st.slider(
                tfield(lang, "Differential rotation strength"),
                min_value=0.0,
                max_value=3.0,
                value=float(cfg_seed.get("disk_diffrot_strength", 1.0)),
                step=0.01,
                disabled=(not disk_diffrot_enabled),
            )
            disk_diffrot_seed = st.number_input(
                tfield(lang, "Differential rotation seed"),
                min_value=0,
                value=int(cfg_seed.get("disk_diffrot_seed", 7)),
                step=1,
                disabled=(not disk_diffrot_enabled),
            )
            disk_adaptive_stratification = st.checkbox(
                tfield(lang, "Adaptive disk stratification"),
                value=bool(cfg_seed.get("disk_adaptive_stratification", False)),
                disabled=(not disk_layered_palette),
            )
            disk_adaptive_layers_min = st.number_input(
                tfield(lang, "Adaptive layers min"),
                min_value=2,
                max_value=1024,
                value=int(cfg_seed.get("disk_adaptive_layers_min", 8)),
                step=1,
                disabled=(not (disk_layered_palette and disk_adaptive_stratification)),
            )
            disk_adaptive_layers_max = st.number_input(
                tfield(lang, "Adaptive layers max"),
                min_value=2,
                max_value=2048,
                value=int(cfg_seed.get("disk_adaptive_layers_max", 48)),
                step=1,
                disabled=(not (disk_layered_palette and disk_adaptive_stratification)),
            )
            disk_adaptive_complexity_mix = st.slider(
                tfield(lang, "Adaptive complexity mix"),
                min_value=0.0,
                max_value=1.0,
                value=float(cfg_seed.get("disk_adaptive_complexity_mix", 0.65)),
                step=0.01,
                disabled=(not (disk_layered_palette and disk_adaptive_stratification)),
            )
        disk_volume_emission = st.checkbox(
            tfield(lang, "Continuous disk volume emission"),
            value=bool(cfg_seed.get("disk_volume_emission", False)),
        )
        with st.expander(tr(lang, "volume_emission_options", "Opzioni volume emission"), expanded=bool(disk_volume_emission)):
            disk_volume_samples = st.number_input(
                tfield(lang, "Disk volume samples"),
                min_value=1,
                max_value=64,
                value=int(cfg_seed.get("disk_volume_samples", 5)),
                step=1,
                disabled=(not disk_volume_emission),
            )
            disk_volume_density_scale = st.number_input(
                tfield(lang, "Disk volume density scale"),
                min_value=0.0,
                max_value=1000.0,
                value=float(cfg_seed.get("disk_volume_density_scale", 1.0)),
                step=0.05,
                disabled=(not disk_volume_emission),
            )
            disk_volume_temperature_drop = st.slider(
                tfield(lang, "Disk volume temperature drop"),
                min_value=0.0,
                max_value=1.0,
                value=float(cfg_seed.get("disk_volume_temperature_drop", 0.28)),
                step=0.01,
                disabled=(not disk_volume_emission),
            )
            disk_volume_strength = st.number_input(
                tfield(lang, "Disk volume strength"),
                min_value=0.0,
                max_value=10.0,
                value=float(cfg_seed.get("disk_volume_strength", 0.85)),
                step=0.05,
                disabled=(not disk_volume_emission),
            )
    with d2:
        max_steps = st.number_input(tfield(lang, "Max steps"), min_value=16, value=int(cfg_seed["max_steps"]), step=10)
        step_size = st.number_input(tfield(lang, "Step size"), min_value=0.001, value=float(cfg_seed["step_size"]), step=0.01)
        adaptive_integrator = st.checkbox(tfield(lang, "Adaptive integrator"), value=bool(cfg_seed["adaptive_integrator"]))
        temporal_reprojection = st.checkbox(
            tfield(lang, "Temporal reprojection"),
            value=bool(cfg_seed.get("temporal_reprojection", False)),
        )
        temporal_blend = st.slider(
            tfield(lang, "Temporal blend"),
            min_value=0.0,
            max_value=1.0,
            value=float(cfg_seed.get("temporal_blend", 0.18)),
            step=0.01,
            disabled=(not temporal_reprojection),
        )
        temporal_clamp = st.number_input(
            tfield(lang, "Temporal clamp"),
            min_value=0.1,
            value=float(cfg_seed.get("temporal_clamp", 24.0)),
            step=0.5,
            disabled=(not temporal_reprojection),
        )
    with d3:
        device = st.selectbox(
            tfield(lang, "Device"),
            options=CHOICE_FIELDS["device"],
            index=CHOICE_FIELDS["device"].index(_safe_choice(CHOICE_FIELDS["device"], str(cfg_seed["device"]))),
        )
        dtype = st.selectbox(
            tfield(lang, "Dtype"),
            options=CHOICE_FIELDS["dtype"],
            index=CHOICE_FIELDS["dtype"].index(_safe_choice(CHOICE_FIELDS["dtype"], str(cfg_seed["dtype"]))),
        )
        show_progress_bar = st.checkbox(tfield(lang, "Show progress bar"), value=bool(cfg_seed["show_progress_bar"]))
        progress_backend = st.selectbox(
            tfield(lang, "Progress backend"),
            options=CHOICE_FIELDS["progress_backend"],
            index=CHOICE_FIELDS["progress_backend"].index(
                _safe_choice(CHOICE_FIELDS["progress_backend"], str(cfg_seed.get("progress_backend", "manual")))
            ),
            help=tfield(lang, "manual: custom bar; tqdm: tqdm bar; auto: use tqdm when available"),
        )
    with d4:
        with st.expander(tr(lang, "kernel_camera_options", "Kernel & camera"), expanded=False):
            mps_optimized_kernel = st.checkbox(tfield(lang, "MPS optimized kernel"), value=bool(cfg_seed["mps_optimized_kernel"]))
            mps_auto_chunking = st.checkbox(tfield(lang, "MPS auto chunking"), value=bool(cfg_seed.get("mps_auto_chunking", True)))
            compile_rhs = st.checkbox(tfield(lang, "Compile RHS"), value=bool(cfg_seed["compile_rhs"]))
            mixed_precision = st.checkbox(tfield(lang, "Mixed precision"), value=bool(cfg_seed["mixed_precision"]))
            camera_fastpath = st.checkbox(tfield(lang, "Camera fastpath"), value=bool(cfg_seed["camera_fastpath"]))
            atlas_cartesian_variant = st.checkbox(
                tfield(lang, "Atlas/cartesian camera variant"),
                value=bool(cfg_seed.get("atlas_cartesian_variant", False)),
                help="Per coordinate Kerr-Schild/Generalized-Doran: preserva la continuita' azimutale vicino ai poli.",
            )

        mt_metric_active = metric_model == "morris_thorne"
        mt_any_enabled = any(
            bool(cfg_seed.get(k, False))
            for k in (
                "wormhole_mt_force_reference_trace",
                "wormhole_mt_unwrap_phi",
                "wormhole_mt_shortest_arc_phi_interp",
                "wormhole_mt_sky_sample_from_xyz",
            )
        )
        with st.expander(tr(lang, "morris_thorne_seam_fixes", "Morris-Thorne seam fixes"), expanded=bool(mt_metric_active and mt_any_enabled)):
            wormhole_mt_force_reference_trace = st.checkbox(
                "Morris-Thorne: force robust tracer",
                value=bool(cfg_seed.get("wormhole_mt_force_reference_trace", False)),
                disabled=(not mt_metric_active),
                help="Disattiva il fast path MPS e usa il tracer robusto (_trace) per ridurre seam/artefatti.",
            )
            wormhole_mt_unwrap_phi = st.checkbox(
                "Morris-Thorne: phi unwrapped",
                value=bool(cfg_seed.get("wormhole_mt_unwrap_phi", False)),
                disabled=(not mt_metric_active),
                help="Mantiene phi non wrapped durante l'integrazione geodetica.",
            )
            wormhole_mt_shortest_arc_phi_interp = st.checkbox(
                "Morris-Thorne: shortest-arc phi interpolation",
                value=bool(cfg_seed.get("wormhole_mt_shortest_arc_phi_interp", False)),
                disabled=(not mt_metric_active),
                help="Interpolazione angolare robusta vicino al branch cut.",
            )
            wormhole_mt_sky_sample_from_xyz = st.checkbox(
                "Morris-Thorne: sky sample from XYZ",
                value=bool(cfg_seed.get("wormhole_mt_sky_sample_from_xyz", False)),
                disabled=(not mt_metric_active),
                help="Ricava gli angoli sky da direzioni cartesiane finali (meno sensibile alle seam di phi).",
            )

        roi_expanded = bool(cfg_seed.get("roi_supersampling", False) or cfg_seed.get("quality_lock", False))
        with st.expander(tr(lang, "roi_quality_cache_options", "ROI, quality & cache"), expanded=roi_expanded):
            roi_supersampling = st.checkbox(
                tfield(lang, "ROI supersampling"),
                value=bool(cfg_seed.get("roi_supersampling", False)),
            )
            roi_supersample_threshold = st.number_input(
                tfield(lang, "ROI threshold"),
                min_value=0.50,
                max_value=0.999,
                value=float(cfg_seed.get("roi_supersample_threshold", 0.92)),
                step=0.01,
                format="%.3f",
                disabled=(not roi_supersampling),
            )
            roi_supersample_jitter = st.number_input(
                tfield(lang, "ROI jitter"),
                min_value=0.01,
                max_value=1.00,
                value=float(cfg_seed.get("roi_supersample_jitter", 0.35)),
                step=0.01,
                format="%.2f",
                disabled=(not roi_supersampling),
            )
            roi_supersample_samples = st.number_input(
                tfield(lang, "ROI samples"),
                min_value=1,
                max_value=8,
                value=int(cfg_seed.get("roi_supersample_samples", 2)),
                step=1,
                disabled=(not roi_supersampling),
            )
            persistent_cache_enabled = st.checkbox(
                tfield(lang, "Persistent cache"),
                value=bool(cfg_seed.get("persistent_cache_enabled", True)),
            )
            persistent_cache_dir = st.text_input(
                tfield(lang, "Persistent cache dir"),
                value=str(cfg_seed.get("persistent_cache_dir", "out/cache")),
                disabled=(not persistent_cache_enabled),
            )
            quality_lock = st.checkbox(
                tfield(lang, "Quality lock"),
                value=bool(cfg_seed.get("quality_lock", False)),
            )
            quality_lock_psnr_min = st.number_input(
                tfield(lang, "Quality lock PSNR min"),
                min_value=1.0,
                max_value=120.0,
                value=float(cfg_seed.get("quality_lock_psnr_min", 45.0)),
                step=0.5,
                disabled=(not quality_lock),
            )
            quality_lock_ssim_min = st.number_input(
                tfield(lang, "Quality lock SSIM min"),
                min_value=0.10,
                max_value=1.0,
                value=float(cfg_seed.get("quality_lock_ssim_min", 0.985)),
                step=0.001,
                format="%.3f",
                disabled=(not quality_lock),
            )
            quality_lock_sample_width = st.number_input(
                tfield(lang, "Quality lock sample width"),
                min_value=64,
                max_value=int(width),
                value=min(int(width), int(cfg_seed.get("quality_lock_sample_width", 256))),
                step=8,
                disabled=(not quality_lock),
            )
            quality_lock_sample_height = st.number_input(
                tfield(lang, "Quality lock sample height"),
                min_value=64,
                max_value=int(height),
                value=min(int(height), int(cfg_seed.get("quality_lock_sample_height", 144))),
                step=8,
                disabled=(not quality_lock),
            )
            quality_lock_fallback_to_baseline = st.checkbox(
                tfield(lang, "Quality lock fallback"),
                value=bool(cfg_seed.get("quality_lock_fallback_to_baseline", True)),
                disabled=(not quality_lock),
            )

        with st.expander(tr(lang, "encoding_adaptive_postprocess", "Encoding, adaptive & postprocess"), expanded=False):
            animation_workers = st.number_input(
                tfield(lang, "Animation workers"),
                min_value=1,
                max_value=32,
                value=int(cfg_seed.get("animation_workers", 1)),
                step=1,
            )
            stream_encode_async = st.checkbox(
                tfield(lang, "Stream encode async"),
                value=bool(cfg_seed.get("stream_encode_async", True)),
            )
            stream_encode_queue_size = st.number_input(
                tfield(lang, "Stream encode queue"),
                min_value=1,
                max_value=64,
                value=int(cfg_seed.get("stream_encode_queue_size", 4)),
                step=1,
                disabled=(not stream_encode_async),
            )
            adaptive_spatial_sampling = bool(cfg_seed.get("adaptive_spatial_sampling", False))
            adaptive_spatial_preview_steps = int(cfg_seed.get("adaptive_spatial_preview_steps", 96))
            adaptive_spatial_min_scale = float(cfg_seed.get("adaptive_spatial_min_scale", 0.65))
            adaptive_spatial_quantile = float(cfg_seed.get("adaptive_spatial_quantile", 0.78))
            if supports_adaptive_spatial:
                adaptive_spatial_sampling = st.checkbox(
                    tfield(lang, "Adaptive spatial sampling"),
                    value=adaptive_spatial_sampling,
                )
                adaptive_spatial_preview_steps = st.number_input(
                    tfield(lang, "Adaptive preview steps"),
                    min_value=16,
                    max_value=10000,
                    value=adaptive_spatial_preview_steps,
                    step=8,
                )
                adaptive_spatial_min_scale = st.number_input(
                    tfield(lang, "Adaptive min scale"),
                    min_value=0.10,
                    max_value=1.00,
                    value=adaptive_spatial_min_scale,
                    step=0.05,
                )
                adaptive_spatial_quantile = st.number_input(
                    tfield(lang, "Adaptive quantile"),
                    min_value=0.50,
                    max_value=0.995,
                    value=adaptive_spatial_quantile,
                    step=0.01,
                    format="%.3f",
                )
            render_tile_rows = st.number_input(
                tfield(lang, "Render tile rows (0=auto)"),
                min_value=0,
                max_value=2048,
                value=int(cfg_seed["render_tile_rows"]),
                step=8,
            )
            postprocess_pipeline = st.selectbox(
                tfield(lang, "Postprocess pipeline"),
                options=CHOICE_FIELDS["postprocess_pipeline"],
                index=CHOICE_FIELDS["postprocess_pipeline"].index(
                    _safe_choice(CHOICE_FIELDS["postprocess_pipeline"], str(cfg_seed["postprocess_pipeline"]))
                ),
            )
            gargantua_look_strength = st.slider(
                tfield(lang, "Gargantua look strength"),
                min_value=0.0,
                max_value=2.0,
                value=float(cfg_seed["gargantua_look_strength"]),
                step=0.05,
            )
            temporal_denoise_mode = st.selectbox(
                tfield(lang, "Temporal denoise mode"),
                options=CHOICE_FIELDS["temporal_denoise_mode"],
                index=CHOICE_FIELDS["temporal_denoise_mode"].index(
                    _safe_choice(CHOICE_FIELDS["temporal_denoise_mode"], str(cfg_seed.get("temporal_denoise_mode", "basic")))
                ),
                disabled=(not temporal_reprojection),
            )
            temporal_denoise_radius = st.number_input(
                tfield(lang, "Temporal denoise radius"),
                min_value=1,
                max_value=4,
                value=int(cfg_seed.get("temporal_denoise_radius", 1)),
                step=1,
                disabled=(not temporal_reprojection),
            )
            temporal_denoise_sigma = st.number_input(
                tfield(lang, "Temporal denoise sigma"),
                min_value=0.1,
                value=float(cfg_seed.get("temporal_denoise_sigma", 18.0)),
                step=0.5,
                disabled=(not temporal_reprojection),
            )
            temporal_denoise_clip = st.number_input(
                tfield(lang, "Temporal denoise clip"),
                min_value=0.1,
                value=float(cfg_seed.get("temporal_denoise_clip", 9.0)),
                step=0.5,
                disabled=(not temporal_reprojection),
            )
            motion_vector_scale = st.number_input(
                tfield(lang, "Motion vector scale"),
                min_value=0.0,
                value=float(cfg_seed.get("motion_vector_scale", 1.0)),
                step=0.05,
                disabled=(not temporal_reprojection),
            )

    if perf_profile == "gpu_balanced":
        compile_rhs = True
        mixed_precision = True
        camera_fastpath = True
        mps_auto_chunking = True
        if supports_adaptive_spatial:
            adaptive_spatial_sampling = True
        if device in {"mps", "auto"}:
            mps_optimized_kernel = True
        if int(render_tile_rows) <= 0 and int(height) >= 256:
            render_tile_rows = max(64, min(256, int(height // 4)))
    elif perf_profile == "fast_preview":
        compile_rhs = True
        mixed_precision = True
        camera_fastpath = True
        mps_auto_chunking = True
        if supports_adaptive_spatial:
            adaptive_spatial_sampling = True
        if device in {"mps", "auto"}:
            mps_optimized_kernel = True
        adaptive_integrator = False
        max_steps = min(int(max_steps), 360)
        step_size = max(float(step_size), 0.24)
        quality_lock = False
        if supports_adaptive_spatial:
            adaptive_spatial_min_scale = min(float(adaptive_spatial_min_scale), 0.58)
            adaptive_spatial_preview_steps = min(int(adaptive_spatial_preview_steps), 96)
        if int(render_tile_rows) <= 0:
            render_tile_rows = max(48, min(192, int(height // 3)))
    elif perf_profile == "high_fidelity":
        adaptive_integrator = True
        mixed_precision = False
        if supports_adaptive_spatial:
            adaptive_spatial_sampling = False

    st.subheader(tfield(lang, "Background"))
    b1, b2, b3 = st.columns(3)
    with b1:
        background_mode = st.selectbox(
            tfield(lang, "Background mode"),
            options=CHOICE_FIELDS["background_mode"],
            index=CHOICE_FIELDS["background_mode"].index(
                _safe_choice(CHOICE_FIELDS["background_mode"], str(cfg_seed["background_mode"]))
            ),
        )
        mode_is_darkspace = background_mode == "darkspace"
        projection_seed = "darkspace" if mode_is_darkspace else str(cfg_seed.get("background_projection", "cubemap"))
        if projection_seed == "darkspace" and not mode_is_darkspace:
            projection_seed = "cubemap"
        background_projection = st.selectbox(
            tfield(lang, "Background projection"),
            options=CHOICE_FIELDS["background_projection"],
            index=CHOICE_FIELDS["background_projection"].index(
                _safe_choice(CHOICE_FIELDS["background_projection"], projection_seed)
            ),
            disabled=mode_is_darkspace,
        )
    if background_mode == "darkspace":
        background_projection = "darkspace"
    is_darkspace = background_mode == "darkspace"
    is_hdri = background_mode == "hdri"
    wormhole_remote_hdri_path = str(cfg_seed.get("wormhole_remote_hdri_path") or "")
    wormhole_remote_hdri_exposure = float(cfg_seed.get("wormhole_remote_hdri_exposure", 1.0))
    wormhole_remote_hdri_rotation_deg = float(cfg_seed.get("wormhole_remote_hdri_rotation_deg", 0.0))
    wormhole_remote_cubemap_coherent = bool(cfg_seed.get("wormhole_remote_cubemap_coherent", False))
    wormhole_background_continuous_blend = bool(cfg_seed.get("wormhole_background_continuous_blend", False))
    wormhole_background_blend_width = float(cfg_seed.get("wormhole_background_blend_width", 0.10))
    with b2:
        enable_star_background = st.checkbox(
            tfield(lang, "Enable star background"),
            value=bool(cfg_seed["enable_star_background"]),
            disabled=is_darkspace,
            help="Con darkspace lo sfondo locale e` forzato a nero.",
        )
        star_density = st.number_input(
            tfield(lang, "Star density"),
            min_value=0.0,
            value=float(cfg_seed["star_density"]),
            step=0.0001,
            format="%.6f",
            disabled=is_darkspace,
        )
        star_brightness = st.number_input(
            tfield(lang, "Star brightness"),
            min_value=0.0,
            value=float(cfg_seed["star_brightness"]),
            step=0.1,
            disabled=is_darkspace,
        )
    with b3:
        hdri_path = st.text_input(
            tfield(lang, "HDRI path"),
            value=str(cfg_seed.get("hdri_path") or ""),
            disabled=(not is_hdri) or is_darkspace,
        )
        hdri_exposure = st.number_input(
            tfield(lang, "HDRI exposure"),
            min_value=0.01,
            value=float(cfg_seed["hdri_exposure"]),
            step=0.1,
            disabled=(not is_hdri) or is_darkspace,
        )
        hdri_rotation_default = _clamp(float(cfg_seed["hdri_rotation_deg"]), 0.0, 360.0)
        hdri_rotation_deg = st.number_input(
            tfield(lang, "HDRI rotation (deg)"),
            min_value=0.0,
            max_value=360.0,
            value=hdri_rotation_default,
            step=1.0,
            disabled=(not is_hdri) or is_darkspace,
        )
        if metric_model == "morris_thorne":
            with st.expander(tr(lang, "wormhole_remote_background", "Wormhole remote background"), expanded=True):
                wormhole_remote_hdri_path = st.text_input(
                    tfield(lang, "Wormhole remote HDRI path"),
                    value=wormhole_remote_hdri_path,
                    disabled=is_darkspace,
                )
                wormhole_remote_hdri_exposure = st.number_input(
                    tfield(lang, "Wormhole remote HDRI exposure"),
                    min_value=0.01,
                    value=float(wormhole_remote_hdri_exposure),
                    step=0.1,
                    disabled=is_darkspace,
                )
                wormhole_remote_hdri_rotation_default = _clamp(float(wormhole_remote_hdri_rotation_deg), 0.0, 360.0)
                wormhole_remote_hdri_rotation_deg = st.number_input(
                    tfield(lang, "Wormhole remote HDRI rotation (deg)"),
                    min_value=0.0,
                    max_value=360.0,
                    value=wormhole_remote_hdri_rotation_default,
                    step=1.0,
                    disabled=is_darkspace,
                )
                wormhole_remote_cubemap_coherent = st.checkbox(
                    tfield(lang, "Wormhole remote cubemap coherent"),
                    value=bool(wormhole_remote_cubemap_coherent),
                    help="Usa campionamento cubemap coerente anche sul lato remoto del wormhole (fix seam forte).",
                    disabled=is_darkspace,
                )
                wormhole_background_continuous_blend = st.checkbox(
                    tfield(lang, "Wormhole continuous background blend"),
                    value=bool(wormhole_background_continuous_blend),
                    help="Sfuma gradualmente lo switch locale/remoto vicino alla gola (r=0).",
                    disabled=is_darkspace,
                )
                wormhole_background_blend_width = st.number_input(
                    tfield(lang, "Wormhole background blend width"),
                    min_value=0.0001,
                    max_value=100.0,
                    value=float(wormhole_background_blend_width),
                    step=0.01,
                    format="%.4f",
                    disabled=(not wormhole_background_continuous_blend) or is_darkspace,
                )
    if is_darkspace:
        st.info(tr(lang, "darkspace_active_msg", "Darkspace attivo: i controlli sfondo avanzati sono disabilitati."))
        enable_star_background = False
        star_density = 0.0
        star_brightness = 0.0

    video_params: dict[str, Any] = {}
    if mode == "video":
        st.subheader(tfield(lang, "Video parameters"))
        v1, v2, v3, v4 = st.columns(4)
        with v1:
            video_params["frames"] = st.number_input(tfield(lang, "Frames"), min_value=1, value=100, step=1)
            video_params["fps"] = st.number_input(tfield(lang, "FPS"), min_value=1, value=10, step=1)
            video_params["azimuth_orbits"] = st.number_input(tfield(lang, "Azimuth orbits"), value=1.0, step=0.1)
        with v2:
            video_params["inclination_wobble_deg"] = st.number_input(tfield(lang, "Inclination wobble"), value=0.0, step=0.5)
            use_incl_sweep = st.checkbox(tfield(lang, "Inclination sweep"), value=True)
            video_params["inclination_start_deg"] = (
                st.number_input(
                    tfield(lang, "Inclination start"),
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
                    tfield(lang, "Inclination end"),
                    min_value=0.0,
                    max_value=180.0,
                    value=180.0,
                    step=1.0,
                )
                if use_incl_sweep
                else None
            )
        with v3:
            use_radius_sweep = st.checkbox(tfield(lang, "Radius sweep"), value=False)
            video_params["observer_radius_start"] = st.number_input(tfield(lang, "Radius start"), value=float(observer_radius), step=0.5) if use_radius_sweep else None
            video_params["observer_radius_end"] = st.number_input(tfield(lang, "Radius end"), value=float(observer_radius), step=0.5) if use_radius_sweep else None
            video_params["taa_samples"] = st.number_input(tfield(lang, "TAA samples"), min_value=1, value=1, step=1)
        with v4:
            video_params["shutter_fraction"] = st.number_input(tfield(lang, "Shutter fraction"), min_value=0.0, max_value=1.0, value=0.85, step=0.05)
            video_params["spatial_jitter"] = st.checkbox(tfield(lang, "Spatial jitter"), value=False)
            video_params["stream_encode"] = st.checkbox(tfield(lang, "Stream encode"), value=True)
            video_params["adaptive_frame_steps"] = st.checkbox(tfield(lang, "Adaptive frame steps"), value=True)
            video_params["live_frame_preview"] = st.checkbox(tr(lang, "live_frame_preview", "Live frame preview"), value=True)
            video_params["keep_frames"] = st.checkbox(tr(lang, "keep_frames_resume_preview", "Keep frames (resume/preview)"), value=True)
            video_params["resume_frames"] = st.checkbox(tr(lang, "resume_from_existing_frames", "Resume from existing frames"), value=False)
            video_params["adaptive_frame_steps_min_scale"] = st.number_input(
                tfield(lang, "Adaptive min scale"),
                min_value=0.1,
                max_value=1.0,
                value=0.60,
                step=0.05,
            )
        default_frames_dir = str(
            loaded_cfg.get("frames_dir")
            or (workspace_path_preview / "out" / "webui_runs" / f"frames_{Path(str(output_path)).stem}")
        )
        video_params["frames_dir"] = st.text_input(tr(lang, "frames_directory", "Frames directory"), value=default_frames_dir)

    starship_params: dict[str, Any] = {}
    starship_video_params: dict[str, Any] = {}
    if mode in {"starship_frame", "starship_video"}:
        st.subheader(tfield(lang, "Starship overlay (OBJ)"))
        default_obj_path = str(
            loaded_cfg.get("starship_obj_path")
            or _default_starship_obj(workspace_path_preview)
        )
        s1, s2, s3 = st.columns(3)
        with s1:
            starship_params["obj_path"] = st.text_input(
                tfield(lang, "OBJ model path"),
                value=default_obj_path,
            )
            starship_params["ship_radius"] = st.number_input(
                tfield(lang, "Ship radius (M)"),
                value=float(loaded_cfg.get("ship_radius", 20.0)),
                step=0.5,
            )
            starship_params["ship_theta_deg"] = st.number_input(
                tfield(lang, "Ship theta (deg)"),
                min_value=0.0,
                max_value=180.0,
                value=float(loaded_cfg.get("ship_theta_deg", 87.0)),
                step=0.5,
            )
            starship_params["ship_phi_deg"] = st.number_input(
                tfield(lang, "Ship phi (deg)"),
                min_value=0.0,
                max_value=360.0,
                value=float(loaded_cfg.get("ship_phi_deg", 0.0)),
                step=0.5,
            )
        with s2:
            starship_params["ship_size"] = st.number_input(
                tfield(lang, "Ship size (M)"),
                min_value=0.05,
                value=float(loaded_cfg.get("ship_size", 1.9)),
                step=0.1,
            )
            starship_params["ship_yaw_deg"] = st.number_input(
                tfield(lang, "Ship yaw (deg)"),
                value=float(loaded_cfg.get("ship_yaw_deg", 34.0)),
                step=1.0,
            )
            starship_params["ship_pitch_deg"] = st.number_input(
                tfield(lang, "Ship pitch (deg)"),
                value=float(loaded_cfg.get("ship_pitch_deg", -12.0)),
                step=1.0,
            )
            starship_params["ship_roll_deg"] = st.number_input(
                tfield(lang, "Ship roll (deg)"),
                value=float(loaded_cfg.get("ship_roll_deg", 16.0)),
                step=1.0,
            )
        with s3:
            starship_params["ship_opacity"] = st.slider(
                tfield(lang, "Ship opacity"),
                min_value=0.0,
                max_value=1.0,
                value=float(loaded_cfg.get("ship_opacity", 0.95)),
                step=0.01,
            )
            starship_params["cinematic_strength"] = st.slider(
                tfield(lang, "Cinematic strength"),
                min_value=0.0,
                max_value=2.0,
                value=float(loaded_cfg.get("cinematic_strength", 1.45)),
                step=0.05,
            )

        m1, m2, m3 = st.columns(3)
        with m1:
            starship_params["ship_v_phi"] = st.number_input(
                tfield(lang, "Ship v_phi"),
                value=float(loaded_cfg.get("ship_v_phi", 0.46)),
                step=0.01,
            )
            starship_params["ship_v_theta"] = st.number_input(
                tfield(lang, "Ship v_theta"),
                value=float(loaded_cfg.get("ship_v_theta", 0.04)),
                step=0.01,
            )
            starship_params["ship_v_r"] = st.number_input(
                tfield(lang, "Ship v_r"),
                value=float(loaded_cfg.get("ship_v_r", 0.0)),
                step=0.01,
            )
        with m2:
            starship_params["ship_acceleration"] = st.number_input(
                tfield(lang, "Ship acceleration"),
                min_value=0.0,
                value=float(loaded_cfg.get("ship_acceleration", 0.0)),
                step=0.01,
            )
            direction_modes = [
                "azimuthal_prograde",
                "azimuthal_retrograde",
                "radial_out",
                "radial_in",
                "polar_north",
                "polar_south",
                "custom",
            ]
            default_mode = str(loaded_cfg.get("ship_direction_mode", "azimuthal_prograde"))
            if default_mode not in direction_modes:
                default_mode = "azimuthal_prograde"
            starship_params["ship_direction_mode"] = st.selectbox(
                tfield(lang, "Ship direction mode"),
                options=direction_modes,
                index=direction_modes.index(default_mode),
            )
        with m3:
            starship_params["ship_dir_x"] = st.number_input(
                tfield(lang, "Ship dir x"),
                value=float(loaded_cfg.get("ship_dir_x", 0.0)),
                step=0.1,
            )
            starship_params["ship_dir_y"] = st.number_input(
                tfield(lang, "Ship dir y"),
                value=float(loaded_cfg.get("ship_dir_y", 0.0)),
                step=0.1,
            )
            starship_params["ship_dir_z"] = st.number_input(
                tfield(lang, "Ship dir z"),
                value=float(loaded_cfg.get("ship_dir_z", 1.0)),
                step=0.1,
            )

        default_program = str(loaded_cfg.get("ship_program_json", "[]"))
        starship_params["ship_program_json"] = st.text_area(
            tfield(lang, "Ship thrust program JSON"),
            value=default_program,
            height=120,
        )
        default_multi_ship = str(loaded_cfg.get("multi_ship_config_json", ""))
        starship_params["multi_ship_config_json"] = st.text_area(
            tfield(lang, "Multi-ship config JSON"),
            value=default_multi_ship,
            height=140,
        )

        if mode == "starship_video":
            st.subheader(tfield(lang, "Video parameters"))
            sv1, sv2 = st.columns(2)
            with sv1:
                starship_video_params["frames"] = st.number_input(
                    tfield(lang, "Starship frames"),
                    min_value=1,
                    value=int(loaded_cfg.get("starship_frames", 100)),
                    step=1,
                )
                starship_video_params["fps"] = st.number_input(
                    tfield(lang, "Starship FPS"),
                    min_value=1,
                    value=int(loaded_cfg.get("starship_fps", 10)),
                    step=1,
                )
            with sv2:
                starship_video_params["ship_substeps"] = st.number_input(
                    tfield(lang, "Ship integration substeps"),
                    min_value=1,
                    value=int(loaded_cfg.get("ship_substeps", 3)),
                    step=1,
                )
                starship_video_params["keep_frames"] = st.checkbox(
                    tfield(lang, "Keep starship frames"),
                    value=bool(loaded_cfg.get("keep_starship_frames", False)),
                )

    patch_caption = tfield(lang, "Advanced JSON override (optional)")
    patch_help = tfield(lang, "Enter only fields to override, for example: `{\"disk_beaming_strength\": 0.6}`")
    patch_default = "{}"
    if hasattr(st, "popover"):
        # Compact icon-based access to keep the main layout clean.
        with st.popover(f"🧩 {tr(lang, 'json_patch_short', 'JSON')}", use_container_width=False):
            st.caption(patch_caption)
            st.write(patch_help)
            patch_text = st.text_area(tfield(lang, "JSON patch"), value=patch_default, height=220)
    else:
        with st.expander(f"🧩 {patch_caption}", expanded=False):
            st.write(patch_help)
            patch_text = st.text_area(tfield(lang, "JSON patch"), value=patch_default, height=220)

    config_dict = dict(default_cfg)
    config_dict.update(
        {
            "width": int(width),
            "height": int(height),
            "fov_deg": float(fov_deg),
            "coordinate_system": coordinate_system,
            "metric_model": metric_model,
            "spin": (float(spin) if supports_spin else 0.0),
            "charge": (float(charge) if supports_charge else 0.0),
            "cosmological_constant": (float(cosmological_constant) if supports_lambda else 0.0),
            "observer_radius": float(observer_radius),
            "observer_inclination_deg": float(observer_inclination_deg),
            "observer_azimuth_deg": float(observer_azimuth_deg),
            "observer_roll_deg": float(observer_roll_deg),
            "disk_model": disk_model,
            "disk_radial_profile": disk_radial_profile,
            "disk_outer_radius": float(disk_outer_radius),
            "disk_inner_radius": disk_inner_radius,
            "disk_emission_gain": float(disk_emission_gain),
            "disk_palette": disk_palette,
            "disk_layered_palette": bool(disk_layered_palette),
            "disk_layer_count": int(disk_layer_count),
            "disk_layer_mix": float(disk_layer_mix),
            "disk_layer_accident_strength": float(disk_layer_accident_strength),
            "disk_layer_accident_count": float(disk_layer_accident_count),
            "disk_layer_accident_sharpness": float(disk_layer_accident_sharpness),
            "disk_layer_global_phase": float(disk_layer_global_phase),
            "disk_layer_phase_rate_hz": float(disk_layer_phase_rate_hz),
            "enable_disk_differential_rotation": bool(disk_diffrot_enabled),
            "disk_diffrot_model": disk_diffrot_model,
            "disk_diffrot_visual_mode": disk_diffrot_visual_mode,
            "disk_diffrot_strength": float(disk_diffrot_strength),
            "disk_diffrot_seed": int(disk_diffrot_seed),
            "disk_diffrot_iteration": disk_diffrot_iteration,
            "disk_adaptive_stratification": bool(disk_adaptive_stratification),
            "disk_adaptive_layers_min": int(disk_adaptive_layers_min),
            "disk_adaptive_layers_max": int(disk_adaptive_layers_max),
            "disk_adaptive_complexity_mix": float(disk_adaptive_complexity_mix),
            "disk_volume_emission": bool(disk_volume_emission),
            "disk_volume_samples": int(disk_volume_samples),
            "disk_volume_density_scale": float(disk_volume_density_scale),
            "disk_volume_temperature_drop": float(disk_volume_temperature_drop),
            "disk_volume_strength": float(disk_volume_strength),
            "max_steps": int(max_steps),
            "step_size": float(step_size),
            "adaptive_integrator": bool(adaptive_integrator),
            "temporal_reprojection": bool(temporal_reprojection),
            "temporal_denoise_mode": temporal_denoise_mode,
            "temporal_blend": float(temporal_blend),
            "temporal_clamp": float(temporal_clamp),
            "motion_vector_scale": float(motion_vector_scale),
            "temporal_denoise_radius": int(temporal_denoise_radius),
            "temporal_denoise_sigma": float(temporal_denoise_sigma),
            "temporal_denoise_clip": float(temporal_denoise_clip),
            "device": device,
            "dtype": dtype,
            "show_progress_bar": bool(show_progress_bar),
            "progress_backend": progress_backend,
            "mps_optimized_kernel": bool(mps_optimized_kernel),
            "mps_auto_chunking": bool(mps_auto_chunking),
            "compile_rhs": bool(compile_rhs),
            "mixed_precision": bool(mixed_precision),
            "camera_fastpath": bool(camera_fastpath),
            "atlas_cartesian_variant": bool(atlas_cartesian_variant),
            "roi_supersampling": bool(roi_supersampling),
            "roi_supersample_threshold": float(roi_supersample_threshold),
            "roi_supersample_jitter": float(roi_supersample_jitter),
            "roi_supersample_samples": int(roi_supersample_samples),
            "persistent_cache_enabled": bool(persistent_cache_enabled),
            "persistent_cache_dir": str(persistent_cache_dir).strip() or "out/cache",
            "quality_lock": bool(quality_lock),
            "quality_lock_psnr_min": float(quality_lock_psnr_min),
            "quality_lock_ssim_min": float(quality_lock_ssim_min),
            "quality_lock_sample_width": int(quality_lock_sample_width),
            "quality_lock_sample_height": int(quality_lock_sample_height),
            "quality_lock_fallback_to_baseline": bool(quality_lock_fallback_to_baseline),
            "animation_workers": int(animation_workers),
            "stream_encode_async": bool(stream_encode_async),
            "stream_encode_queue_size": int(stream_encode_queue_size),
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
            "wormhole_remote_hdri_path": wormhole_remote_hdri_path or None,
            "wormhole_remote_hdri_exposure": float(wormhole_remote_hdri_exposure),
            "wormhole_remote_hdri_rotation_deg": float(wormhole_remote_hdri_rotation_deg),
            "wormhole_remote_cubemap_coherent": bool(wormhole_remote_cubemap_coherent),
            "wormhole_background_continuous_blend": bool(wormhole_background_continuous_blend),
            "wormhole_background_blend_width": float(wormhole_background_blend_width),
            "wormhole_mt_force_reference_trace": bool(wormhole_mt_force_reference_trace),
            "wormhole_mt_unwrap_phi": bool(wormhole_mt_unwrap_phi),
            "wormhole_mt_shortest_arc_phi_interp": bool(wormhole_mt_shortest_arc_phi_interp),
            "wormhole_mt_sky_sample_from_xyz": bool(wormhole_mt_sky_sample_from_xyz),
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

    run_col, preview_col = st.columns([1, 1])
    with run_col:
        run_now_live = st.button(tr(lang, "run_live", "Lancia simulazione (live)"), type="primary")
        run_now_bg = st.button(tr(lang, "run_bg", "Lancia in background"))
        run_ab_compare = st.button(tr(lang, "run_ab_compare_quick", "Run A/B compare (quick frame)"))
        preflight_enabled = st.checkbox(tr(lang, "preflight_auto", "Preflight fisico automatico"), value=True)
        dryrun_enabled = st.checkbox(
            tr(lang, "dryrun_gate_before_video", "Dry-run 256p gate prima del video"),
            value=True,
            disabled=(mode != "video"),
        )
        autotune_enabled = st.checkbox(
            tr(lang, "autotune_quick_benchmark", "Autotune device/tiling (quick benchmark)"),
            value=False,
            disabled=(mode in {"starship_frame", "starship_video"}),
        )
        with st.expander(tr(lang, "ab_compare_settings", "A/B compare settings"), expanded=False):
            ab_compare_width = st.number_input(tr(lang, "ab_width", "A/B width"), min_value=128, max_value=1920, value=512, step=32)
            ab_compare_max_steps = st.number_input(tr(lang, "ab_max_steps", "A/B max_steps"), min_value=64, max_value=4000, value=320, step=16)
            ab_patch_a_text = st.text_area(tr(lang, "ab_patch_a_json", "A patch JSON"), value="{}", height=100)
            ab_patch_b_text = st.text_area(tr(lang, "ab_patch_b_json", "B patch JSON"), value='{"disk_emission_gain": 3.0}', height=100)
    if bool(st.session_state.get("pending_video_skip_clear_once", False)):
        st.session_state["pending_video_skip_clear_once"] = False
    elif run_now_live or run_now_bg or run_ab_compare:
        st.session_state["pending_video_render"] = {}

    pending_video_payload = st.session_state.get("pending_video_render")
    confirm_pending_video = False
    cancel_pending_video = False
    if mode == "video" and isinstance(pending_video_payload, dict) and pending_video_payload:
        with st.expander(tr(lang, "pending_video_ready", "Dry-run superato. Render video pronto."), expanded=True):
            st.success(tr(lang, "pending_video_ready", "Dry-run superato. Render video pronto."))
            st.caption(tr(lang, "pending_video_confirm_hint", "Conferma per avviare il render completo con questo JSON."))
            launch_mode_txt = str(pending_video_payload.get("launch_mode", "live")).strip().lower()
            launch_mode_label = (
                tr(lang, "pending_video_bg", "background")
                if launch_mode_txt == "bg"
                else tr(lang, "pending_video_live", "live")
            )
            st.caption(f"{tr(lang, 'pending_video_mode_label', 'Modalità lancio')}: {launch_mode_label}")
            cmd_preview = list(pending_video_payload.get("cmd") or [])
            if cmd_preview:
                st.code(" ".join(str(x) for x in cmd_preview), language="bash")
            c_confirm, c_cancel = st.columns(2)
            with c_confirm:
                confirm_pending_video = st.button(
                    tr(lang, "pending_video_confirm", "Conferma render video"),
                    key="pending_video_confirm_btn",
                    type="primary",
                )
            with c_cancel:
                cancel_pending_video = st.button(
                    tr(lang, "pending_video_cancel", "Annulla render pendente"),
                    key="pending_video_cancel_btn",
                )
    if cancel_pending_video:
        st.session_state["pending_video_render"] = {}
        st.rerun()
    with preview_col:
        preview_json = json.dumps(config_dict, indent=2)
        if hasattr(st, "popover"):
            with st.popover(f"🧾 {tr(lang, 'config_json', 'Config JSON')}", use_container_width=False):
                st.code(preview_json, language="json")
        else:
            with st.expander(f"🧾 {tr(lang, 'config_json', 'Config JSON')}", expanded=False):
                st.code(preview_json, language="json")

    # Keep live execution feedback near run controls so it is always visible.
    live_progress_slot = st.empty()
    live_log_slot = st.empty()
    live_frame_slot = st.empty()

    if save_preset_requested:
        save_name = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(preset_save_name or "").strip()).strip("._-")
        if not save_name:
            st.error(tr(lang, "preset_name_invalid", "Preset name vuoto o non valido."))
        else:
            workspace_path_for_preset = Path(workspace).expanduser().resolve()
            preset_path = _preset_dir(workspace_path_for_preset) / f"{save_name}.json"
            tags = [x.strip() for x in str(preset_save_tags_raw or "").split(",") if x.strip()]
            critical_fields = list(PRESET_CRITICAL_FIELDS if preset_save_lock_fields else [])
            try:
                _save_preset(
                    path=preset_path,
                    config_payload=dict(config_dict),
                    tags=tags,
                    critical_fields=critical_fields,
                )
                st.success(f"{tr(lang, 'preset_saved', 'Preset salvato')}: {preset_path}")
            except Exception as exc:
                st.error(f"{tr(lang, 'preset_save_error', 'Errore salvataggio preset')}: {exc}")

    if not run_now_live and not run_now_bg and (not run_ab_compare) and (not confirm_pending_video):
        return

    if confirm_pending_video:
        pending = dict(st.session_state.get("pending_video_render") or {})
        st.session_state["pending_video_render"] = {}
        cmd = [str(x) for x in list(pending.get("cmd") or [])]
        cfg_path_raw = str(pending.get("cfg_path") or "").strip()
        workspace_raw = str(pending.get("workspace_path") or "").strip()
        output_raw = str(pending.get("output_path") or pending.get("output_hint") or "").strip()
        launch_mode = str(pending.get("launch_mode") or "live").strip().lower()
        if (not cmd) or (not cfg_path_raw) or (not workspace_raw):
            st.error(tr(lang, "pending_video_missing", "Configurazione video pendente non valida o incompleta."))
            return

        workspace_path = Path(workspace_raw).expanduser().resolve()
        cfg_path = Path(cfg_path_raw).expanduser()
        if not cfg_path.is_absolute():
            cfg_path = (workspace_path / cfg_path).resolve()
        output_hint = Path(output_raw) if output_raw else Path("out/webui_video.mp4")
        run_dir = workspace_path / "out" / "webui_runs"
        run_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        st.info(tr(lang, "cmd_launched", "Comando lanciato:"))
        st.code(" ".join(cmd), language="bash")
        st.info(f"{tr(lang, 'cfg_used', 'Config JSON usata')}: {cfg_path}")

        if launch_mode == "bg":
            counter = int(st.session_state.get("job_counter", 0)) + 1
            st.session_state["job_counter"] = counter
            job_id = f"job_{counter:04d}"
            log_path = run_dir / f"run_{stamp}_{job_id}.log"
            queue_entry = {
                "job_id": job_id,
                "stamp": stamp,
                "cmd": list(cmd),
                "workspace": str(workspace_path),
                "log_path": str(log_path),
                "cfg_path": str(cfg_path),
                "output_hint": str(output_hint),
            }
            existing_proc = st.session_state.get("async_proc")
            if existing_proc is not None and existing_proc.poll() is None:
                queued = list(st.session_state.get("async_queue") or [])
                queued.append(queue_entry)
                st.session_state["async_queue"] = queued
                st.session_state["bg_launch_notice"] = (
                    f"{tr(lang, 'job_queued', 'Job accodato')}: `{job_id}` "
                    f"({tr(lang, 'position', 'posizione')} {len(queued)}). "
                    f"{tr(lang, 'open_queue_panel_to_monitor', 'Apri il pannello queue per monitorare.')}"
                )
                st.rerun()
            proc, meta = _launch_background_process(
                cmd=list(cmd),
                workspace_path=workspace_path,
                log_path=log_path,
                cfg_path=cfg_path,
                output_hint=str(output_hint),
                stamp=stamp,
                job_id=job_id,
            )
            st.session_state["async_proc"] = proc
            st.session_state["async_meta"] = meta
            st.session_state["bg_launch_notice"] = (
                f"{tr(lang, 'job_started_background', 'Job avviato in background')} "
                f"(PID {proc.pid}, id {job_id}). "
                f"{tr(lang, 'open_panel_to_monitor', 'Apri il pannello')} "
                f"'{tr(lang, 'bg_job', 'Job in background')}' "
                f"{tr(lang, 'to_monitor', 'per monitorarlo')}."
            )
            st.rerun()

        progress_widget = live_progress_slot.progress(0.0, text=tr(lang, "render_rows_initial", "Render rows: 0/0 (0.0%)"))
        log_placeholder = live_log_slot
        frame_preview_placeholder = live_frame_slot
        rc, log_text = _run_command_live(
            cmd,
            workspace_path,
            log_placeholder,
            progress_widget,
            frame_preview_placeholder=frame_preview_placeholder,
        )
        log_path = run_dir / f"run_{stamp}_confirm.log"
        log_path.write_text(log_text, encoding="utf-8")
        if rc == 0:
            progress_widget.progress(1.0, text=tr(lang, "render_rows_completed", "Render rows: completed (100.0%)"))
            st.success(f"{tr(lang, 'sim_completed', 'Simulazione completata. Log')}: {log_path}")
        else:
            progress_widget.empty()
            st.error(f"{tr(lang, 'sim_failed', 'Simulazione fallita')} (exit={rc}). Log: {log_path}")
            return

        out_file = _resolve_output_file(
            log_text=log_text,
            workspace_path=workspace_path,
            out_hint=output_hint,
        )
        if out_file is None or (not out_file.exists()):
            st.warning(f"{tr(lang, 'output_not_found_expected', 'Output non trovato automaticamente. Atteso')}: {output_hint}")
            return
        st.session_state["last_output_path"] = str(out_file)
        _show_output_media(out_file)
        return

    try:
        patch = _parse_patch(patch_text)
        config_dict.update(patch)
        if bool(st.session_state.get("preset_lock_active", False)):
            locked_values = dict(st.session_state.get("preset_locked_values") or {})
            for key, value in locked_values.items():
                if key in config_dict:
                    config_dict[key] = value
        cfg_obj = replace(RenderConfig(), **config_dict).validated()
    except Exception as exc:
        st.error(f"{tfield(lang, 'Invalid configuration')}: {exc}")
        return

    if bool(st.session_state.get("preset_lock_active", False)):
        locked_fields = list(st.session_state.get("preset_locked_fields") or [])
        if locked_fields:
            st.caption(
                tr(lang, "preset_lock_active_critical", "Preset lock active on critical fields")
                + ": "
                + ", ".join(str(x) for x in locked_fields)
            )

    if preflight_enabled:
        pre_errors, pre_warnings = _preflight_physical_checks(
            cfg_obj=cfg_obj,
            mode=mode,
            video_params=video_params,
        )
        for wmsg in pre_warnings:
            st.warning(f"{tr(lang, 'preflight_prefix', 'Preflight')}: {wmsg}")
        if pre_errors:
            for emsg in pre_errors:
                st.error(f"{tr(lang, 'preflight_prefix', 'Preflight')}: {emsg}")
            st.stop()

    workspace_path = workspace_path_preview
    run_dir = workspace_path / "out" / "webui_runs"
    run_dir.mkdir(parents=True, exist_ok=True)

    if run_ab_compare:
        if mode in {"starship_frame", "starship_video"}:
            st.error(tr(lang, "ab_compare_only_single_video", "A/B compare rapido supportato solo per single_frame/video (kerrtrace)."))
            st.stop()
        with st.expander(tr(lang, "ab_compare_result", "A/B compare result"), expanded=True):
            try:
                patch_a = _parse_patch(str(ab_patch_a_text))
                patch_b = _parse_patch(str(ab_patch_b_text))
            except Exception as exc:
                st.error(f"{tr(lang, 'ab_patch_json_invalid', 'A/B patch JSON non valido')}: {exc}")
                st.stop()

            stamp_ab = datetime.now().strftime("%Y%m%d_%H%M%S")
            cases = [("A", patch_a), ("B", patch_b)]
            results: list[dict[str, Any]] = []
            lock_active = bool(st.session_state.get("preset_lock_active", False))
            locked_values = dict(st.session_state.get("preset_locked_values") or {})

            for label, patch_case in cases:
                case_dict = dict(config_dict)
                case_dict.update(patch_case)
                if lock_active:
                    for key, value in locked_values.items():
                        if key in case_dict:
                            case_dict[key] = value
                try:
                    case_cfg = replace(RenderConfig(), **case_dict).validated()
                except Exception as exc:
                    st.error(f"{tr(lang, 'ab_config_invalid', 'Configurazione A/B non valida')} ({label}): {exc}")
                    st.stop()
                ab_w = int(ab_compare_width)
                ab_h = max(96, int(round(float(case_cfg.height) * (float(ab_w) / max(1.0, float(case_cfg.width))))))
                case_out = run_dir / f"ab_{label}_{stamp_ab}.png"
                case_cfg = replace(
                    case_cfg,
                    width=ab_w,
                    height=ab_h,
                    max_steps=int(ab_compare_max_steps),
                    show_progress_bar=False,
                    output=str(case_out),
                )
                case_cfg_path = run_dir / f"ab_{label}_{stamp_ab}.json"
                case_cfg_path.write_text(json.dumps(asdict(case_cfg), indent=2), encoding="utf-8")
                ok, elapsed_s, log_tail = _benchmark_single_config(
                    python_exec=str(python_exec),
                    workspace_path=workspace_path,
                    cfg_path=case_cfg_path,
                    output_path=case_out,
                    require_gpu=bool(require_gpu),
                    timeout_s=600.0,
                )
                results.append(
                    {
                        "label": label,
                        "ok": ok,
                        "elapsed_s": elapsed_s,
                        "output": case_out,
                        "cfg_path": case_cfg_path,
                        "log_tail": log_tail,
                    }
                )

            st.table(
                [
                    {
                        "case": row["label"],
                        "ok": bool(row["ok"]),
                        "elapsed_s": round(float(row["elapsed_s"]), 3),
                        "output": str(row["output"]),
                    }
                    for row in results
                ]
            )

            if len(results) == 2 and results[0]["ok"] and results[1]["ok"]:
                t_a = float(results[0]["elapsed_s"])
                t_b = float(results[1]["elapsed_s"])
                if t_a > 1.0e-9 and t_b > 1.0e-9:
                    faster = "A" if t_a < t_b else "B"
                    speedup = (max(t_a, t_b) / min(t_a, t_b)) if min(t_a, t_b) > 0.0 else float("inf")
                    st.info(
                        f"{tr(lang, 'ab_faster_case', 'Faster case')}: {faster} | "
                        f"{tr(lang, 'ab_speedup', 'speedup')}: {speedup:.3f}x "
                        f"(A={t_a:.3f}s, B={t_b:.3f}s)"
                    )

            c_a, c_b = st.columns(2)
            for col, row in zip([c_a, c_b], results):
                with col:
                    st.caption(f"{tr(lang, 'ab_case', 'Case')} {row['label']}")
                    if bool(row["ok"]) and Path(str(row["output"])).exists():
                        st.image(str(row["output"]), caption=str(row["output"]))
                    else:
                        st.error(f"{tr(lang, 'ab_case', 'Case')} {row['label']} {tr(lang, 'failed', 'failed')}")
                        if row.get("log_tail"):
                            st.code(str(row["log_tail"]), language="bash")
            st.stop()

    if autotune_enabled and mode in {"single_frame", "video"}:
        with st.expander(tr(lang, "autotune_benchmark", "Autotune benchmark"), expanded=True):
            st.caption(tr(lang, "autotune_benchmark_caption", "Benchmark rapido su frame ridotto per selezionare device e tiling più efficienti."))
            tuned_cfg, tune_report = _autotune_quick_device_and_tiling(
                python_exec=str(python_exec),
                workspace_path=workspace_path,
                run_dir=run_dir,
                cfg_obj=cfg_obj,
                require_gpu=bool(require_gpu),
            )
            if tune_report:
                st.table(
                    [
                        {
                            "stage": str(row.get("stage", "")),
                            "candidate": str(row.get("candidate", "")),
                            "ok": bool(row.get("ok", False)),
                            "elapsed_s": round(float(row.get("elapsed_s", 0.0)), 3),
                        }
                        for row in tune_report
                    ]
                )
            if tuned_cfg != cfg_obj:
                st.success(
                    f"{tr(lang, 'autotune_applied', 'Autotune applicato')}: "
                    f"device={tuned_cfg.device}, render_tile_rows={tuned_cfg.render_tile_rows}, "
                    f"mps_optimized_kernel={tuned_cfg.mps_optimized_kernel}"
                )
                cfg_obj = tuned_cfg
            else:
                st.warning(tr(lang, "autotune_not_better", "Autotune non ha trovato un profilo migliore; mantengo la configurazione corrente."))

    allocated_output, allocated_idx = _reserve_progressive_output_path(
        workspace_path=workspace_path,
        requested_output=str(cfg_obj.output),
    )
    if allocated_idx is not None:
        cfg_obj = replace(cfg_obj, output=str(allocated_output))
        st.caption(f"{tr(lang, 'progressive_output_allocated', 'Output progressivo allocato')}: `{cfg_obj.output}` (#{allocated_idx:06d})")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg_path = run_dir / f"config_{stamp}.json"
    cfg_path.write_text(json.dumps(asdict(cfg_obj), indent=2), encoding="utf-8")

    if mode in {"starship_frame", "starship_video"}:
        obj_raw = str(starship_params.get("obj_path") or "").strip()
        if not obj_raw:
            st.error(tr(lang, "obj_model_required_starship", "OBJ model path è obbligatorio in modalità Starship."))
            return
        obj_path = Path(obj_raw).expanduser()
        if not obj_path.is_absolute():
            obj_path = workspace_path / obj_path
        obj_path = obj_path.resolve()
        if not obj_path.exists():
            st.error(f"{tr(lang, 'obj_not_found', 'OBJ non trovato')}: {obj_path}")
            return

        if require_gpu and device == "cpu":
            st.error(tr(lang, "gpu_required_not_cpu", "Richiesta GPU attiva: imposta device su auto/mps/cuda (non cpu)."))
            return

        program_raw = str(starship_params.get("ship_program_json") or "").strip()
        if not program_raw:
            program_raw = "[]"
        try:
            thrust_program_payload = json.loads(program_raw)
            if not isinstance(thrust_program_payload, list):
                raise ValueError("ship thrust program must be a JSON list")
        except Exception as exc:
            st.error(f"{tr(lang, 'ship_program_json_invalid', 'Ship thrust program JSON non valido')}: {exc}")
            return

        multi_ship_raw = str(starship_params.get("multi_ship_config_json") or "").strip()
        if multi_ship_raw:
            try:
                multi_payload = json.loads(multi_ship_raw)
                if isinstance(multi_payload, list):
                    ship_cfg_payload: dict[str, Any] = {"ships": multi_payload}
                elif isinstance(multi_payload, dict):
                    ship_cfg_payload = multi_payload
                else:
                    raise ValueError("multi-ship JSON must be an object or list")
            except Exception as exc:
                st.error(f"{tr(lang, 'multi_ship_json_invalid', 'Multi-ship config JSON non valido')}: {exc}")
                return
        else:
            ship_cfg_payload = {
                "ships": [
                    {
                        "name": "ship0",
                        "obj": str(obj_path),
                        "radius": float(starship_params["ship_radius"]),
                        "theta_deg": float(starship_params["ship_theta_deg"]),
                        "phi_deg": float(starship_params["ship_phi_deg"]),
                        "size": float(starship_params["ship_size"]),
                        "yaw_deg": float(starship_params["ship_yaw_deg"]),
                        "pitch_deg": float(starship_params["ship_pitch_deg"]),
                        "roll_deg": float(starship_params["ship_roll_deg"]),
                        "opacity": float(starship_params["ship_opacity"]),
                        "cinematic_strength": float(starship_params["cinematic_strength"]),
                        "v_phi": float(starship_params["ship_v_phi"]),
                        "v_theta": float(starship_params["ship_v_theta"]),
                        "v_r": float(starship_params["ship_v_r"]),
                        "acceleration": float(starship_params["ship_acceleration"]),
                        "direction_mode": str(starship_params["ship_direction_mode"]),
                        "direction_vector": [
                            float(starship_params["ship_dir_x"]),
                            float(starship_params["ship_dir_y"]),
                            float(starship_params["ship_dir_z"]),
                        ],
                        "thrust_program": thrust_program_payload,
                    }
                ]
            }

        ship_cfg_path = run_dir / f"starships_{stamp}.json"
        ship_cfg_path.write_text(json.dumps(ship_cfg_payload, indent=2), encoding="utf-8")

        starship_frames = 1
        starship_fps = 10
        ship_substeps = 3
        keep_starship_frames = False
        if mode == "starship_video":
            starship_frames = int(starship_video_params.get("frames", 100))
            starship_fps = int(starship_video_params.get("fps", 10))
            ship_substeps = int(starship_video_params.get("ship_substeps", 3))
            keep_starship_frames = bool(starship_video_params.get("keep_frames", False))

        cmd = [
            python_exec,
            "-m",
            "kerrtrace.starship_video",
            "--ship-config-json",
            str(ship_cfg_path),
            "--output",
            str(cfg_obj.output),
            "--width",
            str(int(width)),
            "--height",
            str(int(height)),
            "--observer-radius",
            str(float(observer_radius)),
            "--observer-theta-deg",
            str(float(observer_inclination_deg)),
            "--observer-phi-deg",
            str(float(observer_azimuth_deg)),
            "--frames",
            str(int(starship_frames)),
            "--fps",
            str(int(starship_fps)),
            "--ship-substeps",
            str(int(ship_substeps)),
            "--disk-outer-radius",
            str(float(disk_outer_radius)),
            "--disk-emission-gain",
            str(float(disk_emission_gain)),
            "--step-size",
            str(float(step_size)),
            "--max-steps",
            str(int(max_steps)),
            "--device",
            str(device),
        ]
        if keep_starship_frames:
            cmd.append("--keep-frames")
        if disk_inner_radius is not None:
            cmd += ["--disk-inner-radius", str(float(disk_inner_radius))]
    else:
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
            frames_dir_raw = str(video_params.get("frames_dir") or "").strip()
            use_live_preview = bool(video_params.get("live_frame_preview", False))
            keep_frames = bool(video_params.get("keep_frames", False))
            if use_live_preview:
                keep_frames = True
            if frames_dir_raw:
                cmd += ["--frames-dir", frames_dir_raw]
            if keep_frames:
                cmd.append("--keep-frames")
            if bool(video_params.get("resume_frames", False)):
                cmd.append("--resume-frames")

    if mode == "video" and dryrun_enabled:
        dry_width = 256
        dry_height = max(96, int(round(float(cfg_obj.height) * (float(dry_width) / max(1.0, float(cfg_obj.width))))))
        dry_output = run_dir / f"dryrun_preview_{stamp}.png"
        dry_cfg_path = run_dir / f"dryrun_config_{stamp}.json"
        dry_max_steps = max(120, min(int(cfg_obj.max_steps), 320))
        dry_cfg = replace(
            cfg_obj,
            width=int(dry_width),
            height=int(dry_height),
            max_steps=int(dry_max_steps),
            output=str(dry_output),
            render_tile_rows=0,
        )
        if video_params.get("observer_radius_start") is not None:
            dry_cfg = replace(dry_cfg, observer_radius=float(video_params["observer_radius_start"]))
        if video_params.get("inclination_start_deg") is not None:
            dry_cfg = replace(dry_cfg, observer_inclination_deg=float(video_params["inclination_start_deg"]))
        if video_params.get("inclination_end_deg") is not None:
            # Evaluate also a middle point when sweep is enabled to reduce false positives.
            i_start = float(video_params.get("inclination_start_deg") or dry_cfg.observer_inclination_deg)
            i_end = float(video_params["inclination_end_deg"])
            dry_cfg = replace(dry_cfg, observer_inclination_deg=0.5 * (i_start + i_end))

        dry_cfg_path.write_text(json.dumps(asdict(dry_cfg), indent=2), encoding="utf-8")
        dry_cmd = [
            str(python_exec),
            "-m",
            "kerrtrace",
            "--config",
            str(dry_cfg_path),
            "--output",
            str(dry_output),
        ]
        if require_gpu:
            dry_cmd.append("--require-gpu")

        with st.expander(tr(lang, "dryrun_gate_title", "Dry-run gate 256p"), expanded=True):
            st.caption(tr(lang, "dryrun_gate_caption", "Eseguo 1 frame rapido prima del video per evitare render lunghi non validi."))
            st.code(" ".join(dry_cmd), language="bash")
            dry_progress = st.progress(0.0, text=tr(lang, "dryrun_preparing", "Dry-run: preparing"))
            dry_log_placeholder = st.empty()
            dry_rc, dry_log = _run_command_live(
                dry_cmd,
                workspace_path,
                dry_log_placeholder,
                dry_progress,
            )
            dry_log_path = run_dir / f"dryrun_{stamp}.log"
            dry_log_path.write_text(dry_log, encoding="utf-8")
            if dry_rc != 0:
                dry_progress.empty()
                st.error(f"{tr(lang, 'dryrun_failed', 'Dry-run fallito')} (exit={dry_rc}). {tr(lang, 'log_label', 'Log')}: {dry_log_path}")
                st.stop()
            dry_progress.progress(1.0, text=tr(lang, "dryrun_completed", "Dry-run: completed"))
            if dry_output.exists():
                st.image(str(dry_output), caption=f"{tr(lang, 'dryrun_preview', 'Dry-run preview')}: {dry_output}")
            ok_dryrun, dry_reasons, dry_metrics = _analyze_dryrun_image(dry_output)
            st.caption(
                tr(lang, "dryrun_metrics", "Dry-run metrics")
                + ": "
                + ", ".join(
                    f"{k}={v:.4f}" for k, v in sorted(dry_metrics.items())
                )
            )
            if not ok_dryrun:
                for reason in dry_reasons:
                    st.error(f"{tr(lang, 'dryrun_gate_reason', 'Dry-run gate')}: {reason}")
                st.error(tr(lang, "video_blocked_quality_gate", "Video bloccato dal gate di qualità. Correggi parametri e rilancia."))
                st.stop()
            st.session_state["pending_video_render"] = {
                "cmd": list(cmd),
                "cfg_path": str(cfg_path),
                "workspace_path": str(workspace_path),
                "output_path": str(cfg_obj.output),
                "launch_mode": ("bg" if run_now_bg else "live"),
            }
            st.session_state["pending_video_skip_clear_once"] = True
            st.rerun()

    st.info(tr(lang, "cmd_launched", "Comando lanciato:"))
    st.code(" ".join(cmd), language="bash")
    st.info(f"{tr(lang, 'cfg_used', 'Config JSON usata')}: {cfg_path}")

    if run_now_bg:
        counter = int(st.session_state.get("job_counter", 0)) + 1
        st.session_state["job_counter"] = counter
        job_id = f"job_{counter:04d}"
        log_path = run_dir / f"run_{stamp}_{job_id}.log"
        queue_entry = {
            "job_id": job_id,
            "stamp": stamp,
            "cmd": list(cmd),
            "workspace": str(workspace_path),
            "log_path": str(log_path),
            "cfg_path": str(cfg_path),
            "output_hint": str(cfg_obj.output),
        }

        existing_proc = st.session_state.get("async_proc")
        if existing_proc is not None and existing_proc.poll() is None:
            queued = list(st.session_state.get("async_queue") or [])
            queued.append(queue_entry)
            st.session_state["async_queue"] = queued
            st.session_state["bg_launch_notice"] = (
                f"{tr(lang, 'job_queued', 'Job accodato')}: `{job_id}` "
                f"({tr(lang, 'position', 'posizione')} {len(queued)}). "
                f"{tr(lang, 'open_queue_panel_to_monitor', 'Apri il pannello queue per monitorare.')}"
            )
            st.rerun()

        proc, meta = _launch_background_process(
            cmd=list(cmd),
            workspace_path=workspace_path,
            log_path=log_path,
            cfg_path=cfg_path,
            output_hint=str(cfg_obj.output),
            stamp=stamp,
            job_id=job_id,
        )
        st.session_state["async_proc"] = proc
        st.session_state["async_meta"] = meta
        st.session_state["bg_launch_notice"] = (
            f"{tr(lang, 'job_started_background', 'Job avviato in background')} "
            f"(PID {proc.pid}, id {job_id}). "
            f"{tr(lang, 'open_panel_to_monitor', 'Apri il pannello')} "
            f"'{tr(lang, 'bg_job', 'Job in background')}' "
            f"{tr(lang, 'to_monitor', 'per monitorarlo')}."
        )
        st.rerun()

    progress_widget = live_progress_slot.progress(0.0, text=tr(lang, "render_rows_initial", "Render rows: 0/0 (0.0%)"))
    log_placeholder = live_log_slot
    frame_preview_placeholder = None
    if mode == "video" and bool(video_params.get("live_frame_preview", False)):
        frame_preview_placeholder = live_frame_slot
    rc, log_text = _run_command_live(
        cmd,
        workspace_path,
        log_placeholder,
        progress_widget,
        frame_preview_placeholder=frame_preview_placeholder,
    )
    log_path = run_dir / f"run_{stamp}.log"
    log_path.write_text(log_text, encoding="utf-8")

    if rc == 0:
        progress_widget.progress(1.0, text=tr(lang, "render_rows_completed", "Render rows: completed (100.0%)"))
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
