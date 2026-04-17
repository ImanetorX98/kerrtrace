from __future__ import annotations

import time


class RenderProgressWindow:
    """Tk progress window updated directly from the render loop (main thread)."""

    def __init__(self, title: str = "KerrTrace Render Progress") -> None:
        self._available = False
        self._init_error: str | None = None
        self._last_emit = 0.0
        self._min_emit_interval = 0.12
        self._tk = None
        self._root = None
        self._status_var = None
        self._completed_var = None
        self._current_var = None
        self._overall_var = None
        self._frame_bar = None
        self._overall_bar = None

        try:
            import tkinter as tk
            from tkinter import ttk

            root = tk.Tk()
            root.title(title)
            root.geometry("520x210")
            root.minsize(420, 180)

            frame = ttk.Frame(root, padding=12)
            frame.pack(fill="both", expand=True)

            self._status_var = tk.StringVar(value="Preparazione render...")
            self._completed_var = tk.StringVar(value="Frame completati: 0/0")
            self._current_var = tk.StringVar(value="Frame corrente: --")
            self._overall_var = tk.StringVar(value="Progresso totale: 0.0%")

            ttk.Label(frame, textvariable=self._status_var, font=("Helvetica", 12, "bold")).pack(anchor="w", pady=(0, 6))
            ttk.Label(frame, textvariable=self._completed_var).pack(anchor="w")
            ttk.Label(frame, textvariable=self._current_var).pack(anchor="w", pady=(2, 0))
            ttk.Label(frame, textvariable=self._overall_var).pack(anchor="w", pady=(0, 6))

            self._frame_bar = ttk.Progressbar(frame, orient="horizontal", mode="determinate", maximum=100.0)
            self._frame_bar.pack(fill="x", pady=(2, 6))
            self._overall_bar = ttk.Progressbar(frame, orient="horizontal", mode="determinate", maximum=100.0)
            self._overall_bar.pack(fill="x")

            def _on_close() -> None:
                try:
                    root.iconify()
                except Exception:
                    root.withdraw()

            root.protocol("WM_DELETE_WINDOW", _on_close)

            self._tk = tk
            self._root = root
            self._available = True
            self._pump_ui()
        except Exception as exc:
            self._init_error = f"{type(exc).__name__}: {exc}"
            self._available = False

    @property
    def available(self) -> bool:
        return self._available

    @property
    def init_error(self) -> str | None:
        return self._init_error

    def update(
        self,
        *,
        current_frame: int,
        completed_frames: int,
        total_frames: int,
        frame_units_done: int,
        frame_units_total: int,
    ) -> None:
        if not self._available or self._root is None:
            return

        now = time.perf_counter()
        frame_units_total = max(1, int(frame_units_total))
        frame_units_done = max(0, min(frame_units_total, int(frame_units_done)))
        is_boundary = frame_units_done in {0, frame_units_total} or int(completed_frames) >= int(total_frames)
        if (not is_boundary) and ((now - self._last_emit) < self._min_emit_interval):
            return
        self._last_emit = now

        total_frames_i = max(1, int(total_frames))
        completed_frames_i = max(0, int(completed_frames))
        current_frame_i = max(0, int(current_frame))

        frame_ratio = float(frame_units_done) / float(frame_units_total)
        overall_ratio = (float(completed_frames_i) + frame_ratio) / float(total_frames_i)
        overall_ratio = max(0.0, min(1.0, overall_ratio))

        if self._frame_bar is not None:
            self._frame_bar["value"] = 100.0 * frame_ratio
        if self._overall_bar is not None:
            self._overall_bar["value"] = 100.0 * overall_ratio

        if self._completed_var is not None:
            self._completed_var.set(f"Frame completati: {completed_frames_i}/{total_frames_i}")
        if self._current_var is not None:
            if current_frame_i > 0:
                self._current_var.set(f"Frame corrente: {current_frame_i}/{total_frames_i} ({100.0 * frame_ratio:5.1f}%)")
            else:
                self._current_var.set("Frame corrente: --")
        if self._overall_var is not None:
            self._overall_var.set(f"Progresso totale: {100.0 * overall_ratio:5.1f}%")
        if self._status_var is not None:
            if completed_frames_i >= total_frames_i:
                self._status_var.set("Render completato")
            elif current_frame_i > 0:
                self._status_var.set("Rendering in corso...")
            else:
                self._status_var.set("Preparazione render...")

        self._pump_ui()

    def close(self) -> None:
        if not self._available or self._root is None:
            return
        try:
            self._root.destroy()
        except Exception:
            pass
        self._available = False

    def _pump_ui(self) -> None:
        if self._root is None or self._tk is None:
            return
        try:
            self._root.update_idletasks()
            self._root.update()
        except self._tk.TclError:
            self._available = False
