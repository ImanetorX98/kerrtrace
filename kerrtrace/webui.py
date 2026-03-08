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
    },
    "fr": {
        "author_label": "Auteur",
        "language_label": "Langue",
        "manual_open": "Ouvrir un fichier manuellement",
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
    },
    "pt": {
        "author_label": "Autor",
        "language_label": "Idioma",
        "manual_open": "Abrir ficheiro manualmente",
    },
}

FIELD_I18N: dict[str, dict[str, str]] = {
    "it": {
        "Python executable": "Eseguibile Python",
        "Width": "Larghezza",
        "Height": "Altezza",
        "Output file": "File di output",
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
                st.caption(f"{tr(lang, 'cfg_used', 'Config JSON usata:')} {cfg_async}")
            if log_path:
                st.caption(f"{tr(lang, 'log_label', 'Log')}: {log_path}")
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
    preset = st.selectbox(tr(lang, "quality_preset", "Preset qualità"), options=preset_labels, index=preset_labels.index(default_preset))
    if preset == "Custom":
        width = st.number_input(tfield(lang, "Width"), min_value=64, max_value=5000, value=int(cfg_seed["width"]), step=1)
        height = st.number_input(tfield(lang, "Height"), min_value=64, max_value=5000, value=int(cfg_seed["height"]), step=1)
    else:
        width, height = QUALITY_PRESETS[preset]
        st.info(f"{tr(lang, 'resolution_set', 'Risoluzione impostata a')} {width}x{height}")

    c1, c2, c3 = st.columns(3)
    with c1:
        if mode == "video":
            output_default = "out/webui_video.mp4"
        elif mode == "starship_video":
            output_default = "out/webui_starship_video.mp4"
        elif mode == "starship_frame":
            output_default = "out/webui_starship.png"
        else:
            output_default = "out/webui_frame.png"
        output_seed = loaded_cfg.get("output", output_default)
        output_path = st.text_input(tfield(lang, "Output file"), value=str(output_seed))
        fov_deg = st.number_input(tfield(lang, "FOV (deg)"), value=float(cfg_seed["fov_deg"]), step=0.1, format="%.3f")
        coordinate_system = st.selectbox(
            tfield(lang, "Coordinate system"),
            options=CHOICE_FIELDS["coordinate_system"],
            index=CHOICE_FIELDS["coordinate_system"].index(
                _safe_choice(CHOICE_FIELDS["coordinate_system"], str(cfg_seed["coordinate_system"]))
            ),
        )
        metric_model = st.selectbox(
            tfield(lang, "Metric model"),
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
            tfield(lang, "Spin a"),
            min_value=-1.0,
            max_value=1.0,
            value=spin_default,
            step=0.01,
            format="%.6f",
        )
        charge = st.number_input(
            tfield(lang, "Charge Q"),
            min_value=-1.0,
            max_value=1.0,
            value=charge_default,
            step=0.01,
            format="%.6f",
        )
        cosmological_constant = st.number_input(
            tfield(lang, "Lambda"),
            value=float(cfg_seed["cosmological_constant"]),
            step=0.000001,
            format="%.9f",
        )
        observer_radius = st.number_input(tfield(lang, "Observer radius"), value=float(cfg_seed["observer_radius"]), step=0.5)
        observer_inclination_deg = st.number_input(
            tfield(lang, "Observer inclination (deg)"),
            min_value=0.0,
            max_value=180.0,
            value=theta_default,
            step=0.5,
        )
    with c3:
        phi_default = _clamp(float(cfg_seed["observer_azimuth_deg"]), 0.0, 360.0)
        observer_azimuth_deg = st.number_input(
            tfield(lang, "Observer azimuth (deg)"),
            min_value=0.0,
            max_value=360.0,
            value=phi_default,
            step=0.5,
        )
        roll_default = _clamp(float(cfg_seed["observer_roll_deg"]), 0.0, 360.0)
        observer_roll_deg = st.number_input(
            tfield(lang, "Observer roll (deg)"),
            min_value=0.0,
            max_value=360.0,
            value=roll_default,
            step=0.5,
        )
        disk_model = st.selectbox(
            tfield(lang, "Disk model"),
            options=CHOICE_FIELDS["disk_model"],
            index=CHOICE_FIELDS["disk_model"].index(_safe_choice(CHOICE_FIELDS["disk_model"], str(cfg_seed["disk_model"]))),
        )
        disk_radial_profile = st.selectbox(
            tfield(lang, "Disk radial profile"),
            options=CHOICE_FIELDS["disk_radial_profile"],
            index=CHOICE_FIELDS["disk_radial_profile"].index(
                _safe_choice(CHOICE_FIELDS["disk_radial_profile"], str(cfg_seed["disk_radial_profile"]))
            ),
        )
        disk_outer_radius = st.number_input(tfield(lang, "Disk outer radius"), value=float(cfg_seed["disk_outer_radius"]), step=0.5)

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
        index=1,
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
    with d2:
        max_steps = st.number_input(tfield(lang, "Max steps"), min_value=16, value=int(cfg_seed["max_steps"]), step=10)
        step_size = st.number_input(tfield(lang, "Step size"), min_value=0.001, value=float(cfg_seed["step_size"]), step=0.01)
        adaptive_integrator = st.checkbox(tfield(lang, "Adaptive integrator"), value=bool(cfg_seed["adaptive_integrator"]))
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
        mps_optimized_kernel = st.checkbox(tfield(lang, "MPS optimized kernel"), value=bool(cfg_seed["mps_optimized_kernel"]))
        compile_rhs = st.checkbox(tfield(lang, "Compile RHS"), value=bool(cfg_seed["compile_rhs"]))
        mixed_precision = st.checkbox(tfield(lang, "Mixed precision"), value=bool(cfg_seed["mixed_precision"]))
        camera_fastpath = st.checkbox(tfield(lang, "Camera fastpath"), value=bool(cfg_seed["camera_fastpath"]))
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

    if perf_profile == "gpu_balanced":
        compile_rhs = True
        mixed_precision = True
        camera_fastpath = True
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
        background_projection = st.selectbox(
            tfield(lang, "Background projection"),
            options=CHOICE_FIELDS["background_projection"],
            index=CHOICE_FIELDS["background_projection"].index(
                _safe_choice(CHOICE_FIELDS["background_projection"], str(cfg_seed["background_projection"]))
            ),
        )
    with b2:
        enable_star_background = st.checkbox(tfield(lang, "Enable star background"), value=bool(cfg_seed["enable_star_background"]))
        star_density = st.number_input(tfield(lang, "Star density"), min_value=0.0, value=float(cfg_seed["star_density"]), step=0.0001, format="%.6f")
        star_brightness = st.number_input(tfield(lang, "Star brightness"), min_value=0.0, value=float(cfg_seed["star_brightness"]), step=0.1)
    with b3:
        hdri_path = st.text_input(tfield(lang, "HDRI path"), value=str(cfg_seed.get("hdri_path") or ""))
        hdri_exposure = st.number_input(tfield(lang, "HDRI exposure"), min_value=0.01, value=float(cfg_seed["hdri_exposure"]), step=0.1)
        hdri_rotation_default = _clamp(float(cfg_seed["hdri_rotation_deg"]), 0.0, 360.0)
        hdri_rotation_deg = st.number_input(
            tfield(lang, "HDRI rotation (deg)"),
            min_value=0.0,
            max_value=360.0,
            value=hdri_rotation_default,
            step=1.0,
        )

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
            video_params["adaptive_frame_steps_min_scale"] = st.number_input(
                tfield(lang, "Adaptive min scale"),
                min_value=0.1,
                max_value=1.0,
                value=0.60,
                step=0.05,
            )

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

    with st.expander(tfield(lang, "Advanced JSON override (optional)")):
        st.write(tfield(lang, "Enter only fields to override, for example: `{\"disk_beaming_strength\": 0.6}`"))
        patch_text = st.text_area(tfield(lang, "JSON patch"), value="{}", height=220)

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
        st.error(f"{tfield(lang, 'Invalid configuration')}: {exc}")
        return

    workspace_path = Path(workspace).expanduser().resolve()
    run_dir = workspace_path / "out" / "webui_runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg_path = run_dir / f"config_{stamp}.json"
    cfg_path.write_text(json.dumps(asdict(cfg_obj), indent=2), encoding="utf-8")

    if mode in {"starship_frame", "starship_video"}:
        script_path = workspace_path / "scripts" / "render_obj_starship_video.py"
        if not script_path.exists():
            st.error(f"Script non trovato: {script_path}")
            return
        obj_raw = str(starship_params.get("obj_path") or "").strip()
        if not obj_raw:
            st.error("OBJ model path è obbligatorio in modalità Starship.")
            return
        obj_path = Path(obj_raw).expanduser()
        if not obj_path.is_absolute():
            obj_path = workspace_path / obj_path
        obj_path = obj_path.resolve()
        if not obj_path.exists():
            st.error(f"OBJ non trovato: {obj_path}")
            return

        if require_gpu and device == "cpu":
            st.error("Richiesta GPU attiva: imposta device su auto/mps/cuda (non cpu).")
            return

        program_raw = str(starship_params.get("ship_program_json") or "").strip()
        if not program_raw:
            program_raw = "[]"
        try:
            thrust_program_payload = json.loads(program_raw)
            if not isinstance(thrust_program_payload, list):
                raise ValueError("ship thrust program must be a JSON list")
        except Exception as exc:
            st.error(f"Ship thrust program JSON non valido: {exc}")
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
                st.error(f"Multi-ship config JSON non valido: {exc}")
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
            "scripts.render_obj_starship_video",
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
