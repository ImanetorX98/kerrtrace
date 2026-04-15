from __future__ import annotations

from dataclasses import dataclass, replace
from contextlib import nullcontext
import hashlib
import logging
import math
import numpy as np
from pathlib import Path
import sys
import tempfile
import threading
import time
from typing import Callable, Sequence

logger = logging.getLogger(__name__)

from PIL import Image
import torch
import torch.nn.functional as F
try:
    import imageio.v3 as iio
except Exception:
    iio = None
try:
    import click
except Exception:
    click = None
try:
    from tqdm.auto import tqdm as tqdm_auto
except Exception:
    tqdm_auto = None

from .config import RenderConfig
from .cache_utils import DEFAULT_CACHE_MAX_SIZE, LRUDict
from .geometry import (
    THETA_EPS,
    effective_metric_parameters,
    event_horizon_radius,
    inverse_metric_components,
    inverse_metric_derivatives,
    metric_components,
)


def _write_radiance_hdr(path: Path, rgb: np.ndarray) -> None:
    """
    Minimal Radiance RGBE writer fallback.

    Expects float array in HxWx3 with non-negative values.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Radiance HDR writer expects HxWx3 array")
    h, w, _ = rgb.shape
    arr = np.clip(rgb.astype(np.float32, copy=False), 0.0, None)
    maxc = np.max(arr, axis=2)
    rgbe = np.zeros((h, w, 4), dtype=np.uint8)
    nonzero = maxc > 1.0e-32
    if np.any(nonzero):
        m, e = np.frexp(maxc[nonzero])
        scale = (m * 256.0) / np.maximum(maxc[nonzero], 1.0e-32)
        rgb_scaled = arr[nonzero] * scale[:, None]
        rgbe_nonzero = np.clip(rgb_scaled, 0.0, 255.0).astype(np.uint8)
        rgbe[nonzero, 0:3] = rgbe_nonzero
        rgbe[nonzero, 3] = np.clip(e + 128, 0, 255).astype(np.uint8)

    header = (
        "#?RADIANCE\n"
        "FORMAT=32-bit_rle_rgbe\n"
        "\n"
        f"-Y {h} +X {w}\n"
    ).encode("ascii")
    with path.open("wb") as fh:
        fh.write(header)
        fh.write(rgbe.tobytes())


@dataclass
class RenderStats:
    total_rays: int
    disk_hits: int
    horizon_hits: int
    escaped: int
    steps_used: int


@dataclass
class RenderOutput:
    image: Image.Image
    stats: RenderStats


@dataclass(frozen=True)
class PointEmitter:
    r: float
    theta: float
    phi: float
    u_t: float
    u_r: float
    u_theta: float
    u_phi: float
    radius: float = 0.45
    intensity: float = 4.0
    color_rgb: tuple[float, float, float] = (0.55, 0.78, 1.0)


EmitterInput = PointEmitter | Sequence[PointEmitter] | None


# Maximum number of entries per class-level cache to prevent unbounded memory growth.
_CACHE_MAX_SIZE = DEFAULT_CACHE_MAX_SIZE
_LRUDict = LRUDict


class KerrRayTracer:
    _cubemap_cache: _LRUDict = _LRUDict()
    _wormhole_remote_cubemap_cache: _LRUDict = _LRUDict()
    _hdri_cache: _LRUDict = _LRUDict()
    _nt_page_thorne_cache: _LRUDict = _LRUDict()
    _keplerian_omega_cache: _LRUDict = _LRUDict()
    _disk_flux_reference_cache: _LRUDict = _LRUDict()
    _camera_1d_cache: _LRUDict = _LRUDict()
    _compiled_unbound_cache: _LRUDict = _LRUDict()
    _compiled_unbound_fail: set[tuple[object, ...]] = set()
    _compiled_unbound_lock = threading.Lock()

    @classmethod
    def _get_compiled_unbound(
        cls,
        key: tuple[object, ...],
        fn: Callable[..., torch.Tensor],
    ) -> Callable[..., torch.Tensor] | None:
        if not hasattr(torch, "compile"):
            return None
        with cls._compiled_unbound_lock:
            cached = cls._compiled_unbound_cache.get(key)
            if cached is not None:
                return cached
            if key in cls._compiled_unbound_fail:
                return None
        try:
            compiled = torch.compile(fn, mode="reduce-overhead")
        except (RuntimeError, torch.fx.proxy.TraceError) as exc:
            logger.debug("torch.compile failed for key %s: %s", key, exc)
            with cls._compiled_unbound_lock:
                cls._compiled_unbound_fail.add(key)
            return None
        except Exception as exc:
            logger.warning("Unexpected error during torch.compile for key %s: %s", key, exc)
            with cls._compiled_unbound_lock:
                cls._compiled_unbound_fail.add(key)
            return None
        with cls._compiled_unbound_lock:
            cls._compiled_unbound_cache[key] = compiled
        return compiled

    def __init__(self, config: RenderConfig):
        self.config = config.validated()
        self.device = self.config.resolve_device()
        self.dtype = self.config.resolve_dtype()
        self.metric_spin, self.metric_charge, self.metric_lambda = effective_metric_parameters(
            self.config.metric_model,
            self.config.spin,
            self.config.charge,
            self.config.cosmological_constant,
        )
        self.is_wormhole = self.config.metric_model in {"morris_thorne", "dneg_wormhole"}
        self.wormhole_throat_radius = float(max(1.0e-6, self.config.wormhole_throat_radius))
        self.wormhole_length_scale = float(max(0.0, self.config.wormhole_length_scale))
        self.wormhole_lensing_scale = float(max(1.0e-6, self.config.wormhole_lensing_scale))
        self.enable_disk = bool(self.config.enable_accretion_disk and (not self.is_wormhole))
        self.use_mps_optimized_kernel = bool(
            self.config.mps_optimized_kernel
            and self.device.type == "mps"
            and self.config.coordinate_system == "boyer_lindquist"
        )
        self.is_ks_family = self.config.coordinate_system in {"kerr_schild", "generalized_doran"}
        self.kerr_schild_mode = "off"
        if self.is_ks_family:
            mode = self.config.kerr_schild_mode
            if not self.config.kerr_schild_improvements:
                mode = "off"
            self.kerr_schild_mode = mode
        # The analytic KS RHS is derived for the flat-background (Lambda=0) case.
        self.ks_use_analytic_rhs = (self.kerr_schild_mode == "analytic") and (abs(float(self.metric_lambda)) <= 1.0e-14)
        self.ks_use_fsal = self.kerr_schild_mode in {"fsal_only", "analytic"}
        self._rhs_kerr_schild_fn = self._rhs_kerr_schild_analytic if self.ks_use_analytic_rhs else self._rhs_kerr_schild_numeric
        self._camera_axes_cache_key: tuple[float, float, float] | None = None
        self._camera_axes_cache_value: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        self._ks_observer_cache_key: tuple[float, float, float] | None = None
        self._ks_observer_cache_value: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        self.pi = torch.tensor(math.pi, dtype=self.dtype, device=self.device)
        self.equatorial = torch.tensor(math.pi * 0.5, dtype=self.dtype, device=self.device)
        self.has_horizon = not self.is_wormhole
        if self.has_horizon:
            self.horizon = torch.tensor(
                event_horizon_radius(
                    self.config.spin,
                    self.config.metric_model,
                    self.config.charge,
                    self.config.cosmological_constant,
                ),
                dtype=self.dtype,
                device=self.device,
            )
        else:
            self.horizon = torch.tensor(0.0, dtype=self.dtype, device=self.device)
        self._persistent_cache_root: Path | None = None
        if self.config.persistent_cache_enabled:
            try:
                root = Path(self.config.persistent_cache_dir).expanduser().resolve()
                root.mkdir(parents=True, exist_ok=True)
                self._persistent_cache_root = root
            except OSError as exc:
                logger.warning("Could not create persistent cache directory '%s': %s", root, exc)
                self._persistent_cache_root = None
        self._cuda_finalize_graph_ready = False
        self._cuda_finalize_graph_failed = False
        self._cuda_finalize_graph: torch.cuda.CUDAGraph | None = None
        self._cuda_finalize_static_in: torch.Tensor | None = None
        self._cuda_finalize_static_out: torch.Tensor | None = None
        self._disk_flux_lut_main: tuple[torch.Tensor, torch.Tensor] | None = None
        self._disk_omega_lut_main: tuple[torch.Tensor, torch.Tensor] | None = None
        self._last_rgb_float: torch.Tensor | None = None
        self._rhs_fn = self._rhs
        self._rk4_bl_step_fn: Callable[[torch.Tensor, float], torch.Tensor] = self._rk4_step
        self._rk4_ks_step_fn: Callable[[torch.Tensor, float], torch.Tensor] = self._rk4_step_kerr_schild
        if (self.config.compile_rhs or self.use_mps_optimized_kernel) and self.config.coordinate_system == "boyer_lindquist":
            compiled_unbound = self._get_compiled_unbound(
                (
                    "rhs_bl_unbound",
                    "reduce-overhead",
                    self.device.type,
                    str(self.dtype),
                ),
                KerrRayTracer._rhs,
            )
            if compiled_unbound is not None:
                self._rhs_fn = lambda state, _fn=compiled_unbound, _self=self: _fn(_self, state)
            compiled_rk4_bl_unbound = self._get_compiled_unbound(
                (
                    "rk4_bl_unbound",
                    "reduce-overhead",
                    self.device.type,
                    str(self.dtype),
                ),
                KerrRayTracer._rk4_step,
            )
            if compiled_rk4_bl_unbound is not None:
                self._rk4_bl_step_fn = (
                    lambda state, h, _fn=compiled_rk4_bl_unbound, _self=self: _fn(_self, state, h)
                )
        if self.config.compile_rhs and self.is_ks_family and (not self.ks_use_analytic_rhs):
            compiled_ks_unbound = self._get_compiled_unbound(
                (
                    "rhs_ks_numeric_unbound",
                    "reduce-overhead",
                    self.device.type,
                    str(self.dtype),
                    self.config.metric_model,
                ),
                KerrRayTracer._rhs_kerr_schild_numeric,
            )
            if compiled_ks_unbound is not None:
                self._rhs_kerr_schild_fn = lambda state, _fn=compiled_ks_unbound, _self=self: _fn(_self, state)
            compiled_rk4_ks_unbound = self._get_compiled_unbound(
                (
                    "rk4_ks_unbound",
                    "reduce-overhead",
                    self.device.type,
                    str(self.dtype),
                    self.config.metric_model,
                ),
                KerrRayTracer._rk4_step_kerr_schild,
            )
            if compiled_rk4_ks_unbound is not None:
                self._rk4_ks_step_fn = (
                    lambda state, h, _fn=compiled_rk4_ks_unbound, _self=self: _fn(_self, state, h)
                )

        self._hdri_tex: torch.Tensor | None = None
        self._wormhole_remote_hdri_tex: torch.Tensor | None = None
        self._wormhole_remote_hdri_file: Path | None = None
        if self.config.background_mode == "hdri" and self.config.hdri_path:
            hdri_file = Path(self.config.hdri_path).expanduser().resolve()
            if not hdri_file.exists():
                raise FileNotFoundError(f"HDRI file not found: {hdri_file}")
            st = hdri_file.stat()
            hdri_key = (
                str(hdri_file),
                int(st.st_mtime_ns),
                int(st.st_size),
                str(self.device),
                str(self.dtype),
            )
            cached_tex = KerrRayTracer._hdri_cache.get(hdri_key)
            if cached_tex is None:
                with Image.open(hdri_file) as hdri_img:
                    hdri_np = np.asarray(hdri_img.convert("RGB"), dtype=np.float32) / 255.0
                cached_tex = torch.from_numpy(hdri_np).to(device=self.device, dtype=self.dtype)
                KerrRayTracer._hdri_cache[hdri_key] = cached_tex
                if len(KerrRayTracer._hdri_cache) > 8:
                    KerrRayTracer._hdri_cache.pop(next(iter(KerrRayTracer._hdri_cache)))
            self._hdri_tex = cached_tex
        if self.is_wormhole and self.config.wormhole_remote_hdri_path:
            remote_hdri_file = Path(self.config.wormhole_remote_hdri_path).expanduser().resolve()
            if not remote_hdri_file.exists():
                raise FileNotFoundError(f"Wormhole remote HDRI file not found: {remote_hdri_file}")
            self._wormhole_remote_hdri_file = remote_hdri_file
            st = remote_hdri_file.stat()
            remote_hdri_key = (
                str(remote_hdri_file),
                int(st.st_mtime_ns),
                int(st.st_size),
                str(self.device),
                str(self.dtype),
            )
            cached_remote = KerrRayTracer._hdri_cache.get(remote_hdri_key)
            if cached_remote is None:
                with Image.open(remote_hdri_file) as hdri_img:
                    hdri_np = np.asarray(hdri_img.convert("RGB"), dtype=np.float32) / 255.0
                cached_remote = torch.from_numpy(hdri_np).to(device=self.device, dtype=self.dtype)
                KerrRayTracer._hdri_cache[remote_hdri_key] = cached_remote
                if len(KerrRayTracer._hdri_cache) > 8:
                    KerrRayTracer._hdri_cache.pop(next(iter(KerrRayTracer._hdri_cache)))
            self._wormhole_remote_hdri_tex = cached_remote

        self._cubemap_faces: tuple[torch.Tensor, ...] | None = None
        self._wormhole_remote_cubemap_faces: tuple[torch.Tensor, ...] | None = None
        if self.config.enable_star_background and self.config.background_projection == "cubemap":
            self._cubemap_faces = self._get_or_build_cubemap()
            if self.is_wormhole and self.config.wormhole_remote_cubemap_coherent:
                self._wormhole_remote_cubemap_faces = self._get_or_build_wormhole_remote_cubemap()
        need_omega_lut = self.enable_disk and (
            self.config.disk_model == "physical_nt"
            or (
                self.config.enable_disk_differential_rotation
                and self.config.disk_diffrot_model == "keplerian_lut"
            )
        )
        if self.enable_disk and self.config.disk_model == "physical_nt":
            try:
                rin0 = float(self.config.disk_inner_radius)
                rout0 = float(self.config.disk_outer_radius)
                if self.config.disk_radial_profile == "nt_page_thorne" and self.config.metric_model == "kerr":
                    self._disk_flux_lut_main = self._build_nt_page_thorne_lut(rin=rin0, rmax=rout0)
                else:
                    self._disk_flux_lut_main = self._build_nt_general_lut(rin=rin0, rmax=rout0)
            except Exception:
                self._disk_flux_lut_main = None
            try:
                rin0 = float(self.config.disk_inner_radius)
                rout0 = float(self.config.disk_outer_radius)
                self._disk_omega_lut_main = self._build_keplerian_omega_lut(rin=rin0, rmax=rout0)
            except Exception:
                self._disk_omega_lut_main = None
        elif need_omega_lut:
            try:
                rin0 = float(self.config.disk_inner_radius)
                rout0 = float(self.config.disk_outer_radius)
                self._disk_omega_lut_main = self._build_keplerian_omega_lut(rin=rin0, rmax=rout0)
            except Exception:
                self._disk_omega_lut_main = None
        self._disk_flux_reference = self._compute_disk_flux_reference() if self.enable_disk else 1.0

    def set_observer(
        self,
        observer_radius: float | None = None,
        observer_inclination_deg: float | None = None,
        observer_azimuth_deg: float | None = None,
        observer_roll_deg: float | None = None,
        max_steps: int | None = None,
        disk_layer_global_phase: float | None = None,
    ) -> None:
        """
        Lightweight per-frame camera update used by animation/TAA loops.
        """
        updates: dict[str, float | int] = {}
        if observer_radius is not None:
            updates["observer_radius"] = float(observer_radius)
        if observer_inclination_deg is not None:
            updates["observer_inclination_deg"] = float(observer_inclination_deg)
        if observer_azimuth_deg is not None:
            updates["observer_azimuth_deg"] = float(observer_azimuth_deg)
        if observer_roll_deg is not None:
            updates["observer_roll_deg"] = float(observer_roll_deg)
        if max_steps is not None:
            updates["max_steps"] = int(max_steps)
        if disk_layer_global_phase is not None:
            updates["disk_layer_global_phase"] = float(disk_layer_global_phase)
        if not updates:
            return

        new_cfg = replace(self.config, **updates).validated()
        if new_cfg.resolve_device() != self.device:
            raise RuntimeError("set_observer cannot change device on an existing tracer")
        if new_cfg.resolve_dtype() != self.dtype:
            raise RuntimeError("set_observer cannot change dtype on an existing tracer")
        self.config = new_cfg

    def _normalize_emitters(self, emitter: EmitterInput) -> list[PointEmitter]:
        if emitter is None:
            return []
        if isinstance(emitter, PointEmitter):
            return [emitter]
        out: list[PointEmitter] = []
        for item in emitter:
            if not isinstance(item, PointEmitter):
                raise TypeError("emitter sequence must contain only PointEmitter instances")
            out.append(item)
        return out

    def _persistent_cache_path(self, namespace: str, key: tuple[object, ...]) -> Path | None:
        root = self._persistent_cache_root
        if root is None:
            return None
        try:
            digest = hashlib.sha1(repr((namespace, key)).encode("utf-8")).hexdigest()[:24]
            ns_dir = root / namespace
            ns_dir.mkdir(parents=True, exist_ok=True)
            return ns_dir / f"{digest}.npz"
        except Exception:
            return None

    def _persistent_load_lut_pair(self, namespace: str, key: tuple[object, ...]) -> tuple[torch.Tensor, torch.Tensor] | None:
        path = self._persistent_cache_path(namespace, key)
        if path is None or (not path.exists()):
            return None
        try:
            with np.load(path) as data:
                rr = data["rr"]
                flux = data["flux"]
            rr_t = torch.from_numpy(rr).to(device=self.device, dtype=self.dtype)
            flux_t = torch.from_numpy(flux).to(device=self.device, dtype=self.dtype)
            if rr_t.ndim != 1 or flux_t.ndim != 1 or int(rr_t.shape[0]) != int(flux_t.shape[0]) or int(rr_t.shape[0]) < 2:
                return None
            return rr_t.contiguous(), flux_t.contiguous()
        except Exception:
            return None

    def _persistent_save_lut_pair(
        self,
        namespace: str,
        key: tuple[object, ...],
        rr: torch.Tensor,
        flux: torch.Tensor,
    ) -> None:
        path = self._persistent_cache_path(namespace, key)
        if path is None:
            return
        try:
            rr_np = rr.detach().to(device="cpu", dtype=torch.float32).numpy()
            flux_np = flux.detach().to(device="cpu", dtype=torch.float32).numpy()
            np.savez_compressed(path, rr=rr_np, flux=flux_np)
        except Exception:
            return

    def _persistent_load_cubemap(self, key: tuple[object, ...]) -> tuple[torch.Tensor, ...] | None:
        path = self._persistent_cache_path("cubemap", key)
        if path is None or (not path.exists()):
            return None
        try:
            with np.load(path) as data:
                faces: list[torch.Tensor] = []
                for idx in range(6):
                    arr = data[f"face_{idx}"]
                    faces.append(torch.from_numpy(arr).to(device=self.device, dtype=self.dtype).contiguous())
            if len(faces) != 6:
                return None
            return tuple(faces)
        except Exception:
            return None

    def _persistent_save_cubemap(self, key: tuple[object, ...], faces: tuple[torch.Tensor, ...]) -> None:
        path = self._persistent_cache_path("cubemap", key)
        if path is None:
            return
        try:
            payload = {
                f"face_{idx}": face.detach().to(device="cpu", dtype=torch.float16).numpy()
                for idx, face in enumerate(faces)
            }
            np.savez_compressed(path, **payload)
        except Exception:
            return

    def _compute_disk_flux_reference(self) -> float:
        cfg = self.config
        if cfg.disk_model != "physical_nt":
            return 1.0
        rin = float(cfg.disk_inner_radius)
        rout = float(cfg.disk_outer_radius)
        key = (
            cfg.disk_radial_profile,
            cfg.metric_model,
            round(float(cfg.spin), 8),
            round(float(cfg.charge), 8),
            round(float(cfg.cosmological_constant), 12),
            round(rin, 6),
            round(rout, 6),
            str(self.device),
            str(self.dtype),
        )
        cached = KerrRayTracer._disk_flux_reference_cache.get(key)
        if cached is not None:
            return cached
        try:
            rr = torch.linspace(
                max(1.0e-4, rin * 1.0005),
                max(rout, rin * 1.0005 + 1.0e-3),
                2048,
                dtype=self.dtype,
                device=self.device,
            )
            rin_t = torch.as_tensor(rin, dtype=self.dtype, device=self.device)
            if cfg.disk_radial_profile == "nt_page_thorne":
                flux = self._novikov_thorne_flux_profile_page_thorne(rr, rin_t)
            else:
                flux = self._novikov_thorne_flux_profile(rr, rin_t)
            flux = torch.nan_to_num(flux, nan=0.0, posinf=0.0, neginf=0.0)
            positive = flux[flux > 0.0]
            if positive.numel() > 0:
                ref = float(positive.mean().item())
            else:
                ref = float(flux.mean().item())
            if (not math.isfinite(ref)) or ref <= 0.0:
                ref = 1.0
        except Exception:
            ref = 1.0
        KerrRayTracer._disk_flux_reference_cache[key] = ref
        if len(KerrRayTracer._disk_flux_reference_cache) > 32:
            KerrRayTracer._disk_flux_reference_cache.pop(next(iter(KerrRayTracer._disk_flux_reference_cache)))
        return ref

    def _quat_rotate_batch(self, q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        q_xyz = q[1:].view(1, 1, 3)
        q_w = q[0]
        t = 2.0 * torch.cross(q_xyz.expand_as(v), v, dim=-1)
        return v + q_w * t + torch.cross(q_xyz.expand_as(v), t, dim=-1)

    def _quat_from_matrix(self, mat: np.ndarray) -> torch.Tensor:
        m00, m01, m02 = float(mat[0, 0]), float(mat[0, 1]), float(mat[0, 2])
        m10, m11, m12 = float(mat[1, 0]), float(mat[1, 1]), float(mat[1, 2])
        m20, m21, m22 = float(mat[2, 0]), float(mat[2, 1]), float(mat[2, 2])

        tr = m00 + m11 + m22
        if tr > 0.0:
            s = math.sqrt(tr + 1.0) * 2.0
            w = 0.25 * s
            x = (m21 - m12) / s
            y = (m02 - m20) / s
            z = (m10 - m01) / s
        elif (m00 > m11) and (m00 > m22):
            s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
            w = (m21 - m12) / s
            x = 0.25 * s
            y = (m01 + m10) / s
            z = (m02 + m20) / s
        elif m11 > m22:
            s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
            w = (m02 - m20) / s
            x = (m01 + m10) / s
            y = 0.25 * s
            z = (m12 + m21) / s
        else:
            s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
            w = (m10 - m01) / s
            x = (m02 + m20) / s
            y = (m12 + m21) / s
            z = 0.25 * s
        q = torch.tensor([w, x, y, z], dtype=self.dtype, device=self.device)
        return q / torch.clamp(torch.linalg.norm(q), min=1.0e-12)

    def _camera_orientation_quaternion(self, theta_rad: float, phi_rad: float, roll_deg: float) -> torch.Tensor:
        sin_th = math.sin(theta_rad)
        cos_th = math.cos(theta_rad)
        sin_ph = math.sin(phi_rad)
        cos_ph = math.cos(phi_rad)

        e_r = np.array([sin_th * cos_ph, sin_th * sin_ph, cos_th], dtype=np.float64)
        forward = -e_r

        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if abs(float(np.dot(forward, world_up))) > 0.98:
            world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)

        right = np.cross(world_up, forward)
        right = right / max(np.linalg.norm(right), 1.0e-12)
        up = np.cross(forward, right)
        up = up / max(np.linalg.norm(up), 1.0e-12)

        roll = math.radians(roll_deg)
        c_roll = math.cos(roll)
        s_roll = math.sin(roll)
        right_roll = c_roll * right + s_roll * up
        up_roll = -s_roll * right + c_roll * up

        rot = np.stack([right_roll, up_roll, forward], axis=1)
        return self._quat_from_matrix(rot)

    def _observer_angles_regularized(self) -> tuple[float, float]:
        theta0_raw = math.radians(self.config.observer_inclination_deg)
        phi0_raw = math.radians(self.config.observer_azimuth_deg)

        two_pi = 2.0 * math.pi
        theta0 = math.fmod(theta0_raw, two_pi)
        if theta0 < 0.0:
            theta0 += two_pi

        phi0 = phi0_raw
        if theta0 > math.pi:
            theta0 = two_pi - theta0
            phi0 += math.pi
        phi0 = math.fmod(phi0, two_pi)
        if phi0 < 0.0:
            phi0 += two_pi

        # Atlas/cartesian variant: preserve azimuth continuity on the axis.
        if bool(self.config.atlas_cartesian_variant) and self.config.coordinate_system in {"kerr_schild", "generalized_doran"}:
            return theta0, phi0

        # Atlas-safe variant for wormhole in Boyer-Lindquist chart:
        # keep BL regularization on theta (avoid pole singularity),
        # but preserve normalized azimuth instead of forcing phi=0 at poles.
        if (
            bool(self.config.atlas_cartesian_variant)
            and self.config.coordinate_system == "boyer_lindquist"
            and self.config.metric_model == "morris_thorne"
        ):
            axis_eps = max(8.0 * THETA_EPS, math.radians(0.35))
            theta_safe = min(max(theta0, axis_eps), math.pi - axis_eps)
            return theta_safe, phi0

        # Legacy path: Boyer-Lindquist angular coordinates are singular on the rotation axis.
        axis_eps = max(8.0 * THETA_EPS, math.radians(0.35))
        if theta0 <= axis_eps:
            return axis_eps, 0.0
        if theta0 >= (math.pi - axis_eps):
            return math.pi - axis_eps, 0.0
        return theta0, phi0

    def _camera_grid_1d(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg = self.config
        key = (
            int(cfg.width),
            int(cfg.height),
            round(float(cfg.fov_deg), 8),
            str(self.device),
            str(self.dtype),
        )
        cached = KerrRayTracer._camera_1d_cache.get(key)
        if cached is not None:
            return cached

        xs = (torch.arange(cfg.width, dtype=self.dtype, device=self.device) + 0.5) * (2.0 / float(cfg.width)) - 1.0
        ys = 1.0 - ((torch.arange(cfg.height, dtype=self.dtype, device=self.device) + 0.5) * (2.0 / float(cfg.height)))
        aspect = torch.as_tensor(cfg.width / cfg.height, dtype=self.dtype, device=self.device)
        tan_half_fov = torch.tan(torch.as_tensor(math.radians(cfg.fov_deg * 0.5), dtype=self.dtype, device=self.device))
        out = (xs, ys, aspect, tan_half_fov)
        KerrRayTracer._camera_1d_cache[key] = out
        if len(KerrRayTracer._camera_1d_cache) > 16:
            KerrRayTracer._camera_1d_cache.pop(next(iter(KerrRayTracer._camera_1d_cache)))
        return out

    def _camera_axes(self) -> tuple[float, float, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        theta0_val, phi0_val = self._observer_angles_regularized()
        key = (round(theta0_val, 12), round(phi0_val, 12), round(float(self.config.observer_roll_deg), 10))
        cached = self._camera_axes_cache_value
        if (self._camera_axes_cache_key == key) and (cached is not None):
            q_cam, e_r, e_theta, e_phi = cached
            return theta0_val, phi0_val, q_cam, e_r, e_theta, e_phi

        q_cam = self._camera_orientation_quaternion(theta0_val, phi0_val, self.config.observer_roll_deg)
        sin_th = torch.as_tensor(math.sin(theta0_val), dtype=self.dtype, device=self.device)
        cos_th = torch.as_tensor(math.cos(theta0_val), dtype=self.dtype, device=self.device)
        sin_ph = torch.as_tensor(math.sin(phi0_val), dtype=self.dtype, device=self.device)
        cos_ph = torch.as_tensor(math.cos(phi0_val), dtype=self.dtype, device=self.device)

        e_r = torch.stack([sin_th * cos_ph, sin_th * sin_ph, cos_th])
        e_theta = torch.stack([cos_th * cos_ph, cos_th * sin_ph, -sin_th])
        e_phi = torch.stack([-sin_ph, cos_ph, torch.zeros((), dtype=self.dtype, device=self.device)])
        self._camera_axes_cache_key = key
        self._camera_axes_cache_value = (q_cam, e_r, e_theta, e_phi)
        return theta0_val, phi0_val, q_cam, e_r, e_theta, e_phi

    def _camera_world_dirs(
        self,
        x_pixel_offset: float = 0.0,
        y_pixel_offset: float = 0.0,
        row_start: int = 0,
        row_end: int | None = None,
    ) -> tuple[torch.Tensor, int, int, float, float, torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg = self.config
        full_h = int(cfg.height)
        w = int(cfg.width)
        if row_end is None:
            row_end = full_h
        row_start = max(0, min(full_h, int(row_start)))
        row_end = max(row_start, min(full_h, int(row_end)))

        xs_base, ys_base, aspect, tan_half_fov = self._camera_grid_1d()
        xs = xs_base
        ys = ys_base[row_start:row_end]
        if x_pixel_offset != 0.0 and w > 0:
            xs = xs + (2.0 * x_pixel_offset / float(w))
        if y_pixel_offset != 0.0 and full_h > 0:
            ys = ys - (2.0 * y_pixel_offset / float(full_h))
        xx, yy = torch.meshgrid(xs, ys, indexing="xy")

        px = xx * aspect * tan_half_fov
        py = yy * tan_half_fov
        d_cam = torch.stack([px, py, torch.ones_like(px)], dim=-1)
        d_cam = d_cam / torch.clamp(torch.linalg.norm(d_cam, dim=-1, keepdim=True), min=1.0e-12)

        theta0_val, phi0_val, q_cam, e_r, e_theta, e_phi = self._camera_axes()
        d_world = self._quat_rotate_batch(q_cam, d_cam)
        return d_world, row_start, row_end, theta0_val, phi0_val, e_r, e_theta, e_phi

    def _ks_observer_metric(self, theta0: float, phi0: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        key = (round(float(self.config.observer_radius), 8), round(theta0, 12), round(phi0, 12))
        cached = self._ks_observer_cache_value
        if (self._ks_observer_cache_key == key) and (cached is not None):
            return cached

        r0 = torch.as_tensor([self.config.observer_radius], dtype=self.dtype, device=self.device)
        th0 = torch.as_tensor([theta0], dtype=self.dtype, device=self.device)
        ph0 = torch.as_tensor([phi0], dtype=self.dtype, device=self.device)
        obs_xyz = self._bl_to_cartesian_kerr_schild(r0, th0, ph0).reshape(1, 3)
        g_cov, g_inv, _ = self._kerr_schild_metric_and_inverse(obs_xyz)
        g_cov0 = g_cov[0]
        g_inv0 = g_inv[0]
        alpha = torch.rsqrt(torch.clamp(-g_inv0[0, 0], min=1.0e-12))
        beta = (alpha * alpha) * g_inv0[0, 1:4]
        out = (obs_xyz, g_cov0, g_inv0, alpha, beta)
        self._ks_observer_cache_key = key
        self._ks_observer_cache_value = out
        return out

    def _camera_rays_legacy(
        self,
        x_pixel_offset: float = 0.0,
        y_pixel_offset: float = 0.0,
        row_start: int = 0,
        row_end: int | None = None,
    ) -> torch.Tensor:
        cfg = self.config
        full_h, w = cfg.height, cfg.width
        if row_end is None:
            row_end = full_h
        row_start = max(0, min(full_h, int(row_start)))
        row_end = max(row_start, min(full_h, int(row_end)))
        h = row_end - row_start

        xs = (torch.arange(w, dtype=self.dtype, device=self.device) + 0.5) * (2.0 / float(w)) - 1.0
        ys = 1.0 - ((torch.arange(row_start, row_end, dtype=self.dtype, device=self.device) + 0.5) * (2.0 / float(full_h)))
        if x_pixel_offset != 0.0 and w > 0:
            xs = xs + (2.0 * x_pixel_offset / float(w))
        if y_pixel_offset != 0.0 and full_h > 0:
            ys = ys - (2.0 * y_pixel_offset / float(full_h))
        xx, yy = torch.meshgrid(xs, ys, indexing="xy")

        aspect = torch.as_tensor(w / full_h, dtype=self.dtype, device=self.device)
        tan_half_fov = torch.tan(torch.as_tensor(math.radians(cfg.fov_deg * 0.5), dtype=self.dtype, device=self.device))

        px = xx * aspect * tan_half_fov
        py = yy * tan_half_fov
        d_cam = torch.stack([px, py, torch.ones_like(px)], dim=-1)
        d_cam = d_cam / torch.clamp(torch.linalg.norm(d_cam, dim=-1, keepdim=True), min=1.0e-12)

        theta0_val, phi0_val = self._observer_angles_regularized()
        q_cam = self._camera_orientation_quaternion(theta0_val, phi0_val, cfg.observer_roll_deg)
        d_world = self._quat_rotate_batch(q_cam, d_cam)

        sin_th = torch.as_tensor(math.sin(theta0_val), dtype=self.dtype, device=self.device)
        cos_th = torch.as_tensor(math.cos(theta0_val), dtype=self.dtype, device=self.device)
        sin_ph = torch.as_tensor(math.sin(phi0_val), dtype=self.dtype, device=self.device)
        cos_ph = torch.as_tensor(math.cos(phi0_val), dtype=self.dtype, device=self.device)
        e_r = torch.stack([sin_th * cos_ph, sin_th * sin_ph, cos_th])
        e_theta = torch.stack([cos_th * cos_ph, cos_th * sin_ph, -sin_th])
        e_phi = torch.stack([-sin_ph, cos_ph, torch.zeros((), dtype=self.dtype, device=self.device)])

        n_r = d_world[..., 0] * e_r[0] + d_world[..., 1] * e_r[1] + d_world[..., 2] * e_r[2]
        n_theta = d_world[..., 0] * e_theta[0] + d_world[..., 1] * e_theta[1] + d_world[..., 2] * e_theta[2]
        n_phi = d_world[..., 0] * e_phi[0] + d_world[..., 1] * e_phi[1] + d_world[..., 2] * e_phi[2]

        r0 = torch.full((h, w), cfg.observer_radius, dtype=self.dtype, device=self.device)
        theta0 = torch.full((h, w), theta0_val, dtype=self.dtype, device=self.device)
        phi0 = torch.full((h, w), phi0_val, dtype=self.dtype, device=self.device)
        t0 = torch.zeros((h, w), dtype=self.dtype, device=self.device)
        metric = metric_components(
            r0,
            theta0,
            cfg.spin,
            cfg.metric_model,
            cfg.charge,
            cfg.cosmological_constant,
            cfg.wormhole_throat_radius,
            cfg.wormhole_length_scale,
            cfg.wormhole_lensing_scale,
        )

        omega = -metric.g_tphi / torch.clamp(metric.g_phiphi, min=1.0e-9)
        lapse = torch.sqrt(torch.clamp(-(metric.g_tt - (metric.g_tphi * metric.g_tphi) / torch.clamp(metric.g_phiphi, min=1.0e-9)), min=1.0e-12))

        n_t_up = 1.0 / lapse
        n_phi_up = omega / lapse
        e_r_up = 1.0 / torch.sqrt(torch.clamp(metric.g_rr, min=1.0e-12))
        e_theta_up = 1.0 / torch.sqrt(torch.clamp(metric.g_thth, min=1.0e-12))
        e_phi_up = 1.0 / torch.sqrt(torch.clamp(metric.g_phiphi, min=1.0e-12))

        p_t_up = n_t_up
        p_r_up = n_r * e_r_up
        p_theta_up = n_theta * e_theta_up
        p_phi_up = n_phi_up + n_phi * e_phi_up

        p_t = metric.g_tt * p_t_up + metric.g_tphi * p_phi_up
        p_r = metric.g_rr * p_r_up
        p_theta = metric.g_thth * p_theta_up
        p_phi = metric.g_tphi * p_t_up + metric.g_phiphi * p_phi_up

        state = torch.stack(
            [
                t0.reshape(-1),
                r0.reshape(-1),
                theta0.reshape(-1),
                phi0.reshape(-1),
                p_t.reshape(-1),
                p_r.reshape(-1),
                p_theta.reshape(-1),
                p_phi.reshape(-1),
            ],
            dim=-1,
        )
        return state

    def _camera_rays(
        self,
        x_pixel_offset: float = 0.0,
        y_pixel_offset: float = 0.0,
        row_start: int = 0,
        row_end: int | None = None,
    ) -> torch.Tensor:
        if not bool(self.config.camera_fastpath):
            return self._camera_rays_legacy(
                x_pixel_offset=x_pixel_offset,
                y_pixel_offset=y_pixel_offset,
                row_start=row_start,
                row_end=row_end,
            )
        cfg = self.config
        w = int(cfg.width)
        d_world, row_start, row_end, theta0_val, phi0_val, e_r, e_theta, e_phi = self._camera_world_dirs(
            x_pixel_offset=x_pixel_offset,
            y_pixel_offset=y_pixel_offset,
            row_start=row_start,
            row_end=row_end,
        )
        h = row_end - row_start

        n_r = d_world[..., 0] * e_r[0] + d_world[..., 1] * e_r[1] + d_world[..., 2] * e_r[2]
        n_theta = d_world[..., 0] * e_theta[0] + d_world[..., 1] * e_theta[1] + d_world[..., 2] * e_theta[2]
        n_phi = d_world[..., 0] * e_phi[0] + d_world[..., 1] * e_phi[1] + d_world[..., 2] * e_phi[2]

        r0_t = torch.as_tensor([cfg.observer_radius], dtype=self.dtype, device=self.device)
        theta0_t = torch.as_tensor([theta0_val], dtype=self.dtype, device=self.device)
        metric = metric_components(
            r0_t,
            theta0_t,
            cfg.spin,
            cfg.metric_model,
            cfg.charge,
            cfg.cosmological_constant,
            cfg.wormhole_throat_radius,
            cfg.wormhole_length_scale,
            cfg.wormhole_lensing_scale,
        )
        g_tt = metric.g_tt[0]
        g_tphi = metric.g_tphi[0]
        g_rr = metric.g_rr[0]
        g_thth = metric.g_thth[0]
        g_phiphi = metric.g_phiphi[0]

        # ZAMO tetrad: physically local orthonormal frame for camera rays.
        omega = -g_tphi / torch.clamp(g_phiphi, min=1.0e-9)
        lapse = torch.sqrt(torch.clamp(-(g_tt - (g_tphi * g_tphi) / torch.clamp(g_phiphi, min=1.0e-9)), min=1.0e-12))

        n_t_up = 1.0 / lapse
        n_phi_up = omega / lapse

        e_r_up = 1.0 / torch.sqrt(torch.clamp(g_rr, min=1.0e-12))
        e_theta_up = 1.0 / torch.sqrt(torch.clamp(g_thth, min=1.0e-12))
        e_phi_up = 1.0 / torch.sqrt(torch.clamp(g_phiphi, min=1.0e-12))

        p_t_up = n_t_up
        p_r_up = n_r * e_r_up
        p_theta_up = n_theta * e_theta_up
        p_phi_up = n_phi_up + n_phi * e_phi_up

        p_t = g_tt * p_t_up + g_tphi * p_phi_up
        p_r = g_rr * p_r_up
        p_theta = g_thth * p_theta_up
        p_phi = g_tphi * p_t_up + g_phiphi * p_phi_up

        n_pix = h * w
        state = torch.empty((n_pix, 8), dtype=self.dtype, device=self.device)
        state[:, 0] = 0.0
        state[:, 1] = cfg.observer_radius
        state[:, 2] = theta0_val
        state[:, 3] = phi0_val
        state[:, 4] = p_t.reshape(-1)
        state[:, 5] = p_r.reshape(-1)
        state[:, 6] = p_theta.reshape(-1)
        state[:, 7] = p_phi.reshape(-1)
        return state

    def _ks_radius_from_xyz(self, xyz: torch.Tensor) -> torch.Tensor:
        x = xyz[:, 0]
        y = xyz[:, 1]
        z = xyz[:, 2]
        a2 = torch.as_tensor(self.metric_spin * self.metric_spin, dtype=self.dtype, device=self.device)
        rho2 = x * x + y * y + z * z
        disc = torch.sqrt(torch.clamp((rho2 - a2) * (rho2 - a2) + 4.0 * a2 * z * z, min=1.0e-12))
        r2 = 0.5 * (rho2 - a2 + disc)
        return torch.sqrt(torch.clamp(r2, min=1.0e-12))

    def _bl_to_cartesian_kerr_schild(self, r: torch.Tensor, theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        """Boyer-Lindquist-like (r,theta,phi) -> Cartesian generalized Kerr-Schild coordinates."""
        a = torch.as_tensor(self.metric_spin, dtype=self.dtype, device=self.device)
        sin_th = torch.sin(theta)
        cos_th = torch.cos(theta)
        cos_ph = torch.cos(phi)
        sin_ph = torch.sin(phi)

        x = sin_th * (r * cos_ph - a * sin_ph)
        y = sin_th * (r * sin_ph + a * cos_ph)
        z = r * cos_th
        return torch.stack([x, y, z], dim=-1)

    def _cartesian_to_bl(self, xyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Cartesian generalized Kerr-Schild -> (r,theta,phi) oblate coordinates."""
        r = self._ks_radius_from_xyz(xyz)
        z = xyz[:, 2]
        theta = torch.acos(torch.clamp(z / torch.clamp(r, min=1.0e-8), min=-1.0, max=1.0))
        theta = torch.clamp(theta, min=THETA_EPS, max=math.pi - THETA_EPS)
        a = torch.as_tensor(self.metric_spin, dtype=self.dtype, device=self.device)
        x = xyz[:, 0]
        y = xyz[:, 1]
        num = r * y - a * x
        den = r * x + a * y
        phi = torch.remainder(torch.atan2(num, den), 2.0 * self.pi)
        return r, theta, phi

    def _cartesian_covector_to_bl(
        self,
        xyz: torch.Tensor,
        p_xyz: torch.Tensor,
        r: torch.Tensor | None = None,
        theta: torch.Tensor | None = None,
        phi: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if r is None or theta is None or phi is None:
            r, theta, phi = self._cartesian_to_bl(xyz)

        a = torch.as_tensor(self.metric_spin, dtype=self.dtype, device=self.device)
        sin_th = torch.sin(theta)
        cos_th = torch.cos(theta)
        sin_ph = torch.sin(phi)
        cos_ph = torch.cos(phi)

        dx_dr = sin_th * cos_ph
        dy_dr = sin_th * sin_ph
        dz_dr = cos_th

        dx_dth = cos_th * (r * cos_ph - a * sin_ph)
        dy_dth = cos_th * (r * sin_ph + a * cos_ph)
        dz_dth = -r * sin_th

        x = xyz[:, 0]
        y = xyz[:, 1]
        p_x = p_xyz[:, 0]
        p_y = p_xyz[:, 1]
        p_z = p_xyz[:, 2]

        p_r = p_x * dx_dr + p_y * dy_dr + p_z * dz_dr
        p_theta = p_x * dx_dth + p_y * dy_dth + p_z * dz_dth
        p_phi = -p_x * y + p_y * x
        return p_r, p_theta, p_phi

    def _kerr_schild_background_metric_and_inverse(
        self,
        xyz: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Background metric for generalized Kerr-Schild tracing.

        - Lambda = 0: Minkowski background.
        - Lambda != 0: static de Sitter/anti-de Sitter-like background in Cartesian form.
        """
        n = xyz.shape[0]
        g_cov = torch.zeros((n, 4, 4), dtype=self.dtype, device=self.device)
        g_inv = torch.zeros((n, 4, 4), dtype=self.dtype, device=self.device)

        g_cov[:, 0, 0] = -1.0
        g_inv[:, 0, 0] = -1.0
        g_cov[:, 1, 1] = 1.0
        g_cov[:, 2, 2] = 1.0
        g_cov[:, 3, 3] = 1.0
        g_inv[:, 1, 1] = 1.0
        g_inv[:, 2, 2] = 1.0
        g_inv[:, 3, 3] = 1.0

        lmb = float(self.metric_lambda)
        if abs(lmb) <= 1.0e-14:
            return g_cov, g_inv

        rho2 = torch.sum(xyz * xyz, dim=-1)
        rho = torch.sqrt(torch.clamp(rho2, min=1.0e-12))
        n_hat = xyz / torch.clamp(rho.unsqueeze(-1), min=1.0e-8)

        f = 1.0 - (lmb / 3.0) * rho2
        f_safe = torch.where(f >= 0.0, torch.clamp(f, min=1.0e-6), torch.clamp(f, max=-1.0e-6))
        inv_f = 1.0 / f_safe

        eye3 = torch.eye(3, dtype=self.dtype, device=self.device).unsqueeze(0)
        nn = n_hat.unsqueeze(-1) * n_hat.unsqueeze(-2)
        alpha = inv_f - 1.0
        spatial_cov = eye3 + alpha.view(-1, 1, 1) * nn

        # (I + alpha nn^T)^-1 = I - alpha/(1+alpha) nn^T
        denom = torch.where((1.0 + alpha) >= 0.0, torch.clamp(1.0 + alpha, min=1.0e-8), torch.clamp(1.0 + alpha, max=-1.0e-8))
        ratio = alpha / denom
        spatial_inv = eye3 - ratio.view(-1, 1, 1) * nn

        g_cov[:, 0, 0] = -f_safe
        g_inv[:, 0, 0] = -inv_f
        g_cov[:, 1:4, 1:4] = spatial_cov
        g_inv[:, 1:4, 1:4] = spatial_inv
        return g_cov, g_inv

    def _kerr_schild_kn_fields(
        self,
        xyz: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generalized Kerr-Schild fields in Cartesian form:
        - r: oblate radial coordinate
        - H: scalar profile in g = g_background + 2 H l l
        - l_cov, l_contra: principal null one-form/vector
        - A_cov: electromagnetic 4-potential proxy (zero for uncharged metrics)
        """
        x = xyz[:, 0]
        y = xyz[:, 1]
        z = xyz[:, 2]
        r = self._ks_radius_from_xyz(xyz)
        r_safe = torch.clamp(r, min=1.0e-8)

        a = torch.as_tensor(self.metric_spin, dtype=self.dtype, device=self.device)
        q = torch.as_tensor(self.metric_charge, dtype=self.dtype, device=self.device)
        a2 = a * a
        q2 = q * q

        r2 = r_safe * r_safe
        r3 = r2 * r_safe
        r4 = r2 * r2
        den = torch.clamp(r4 + a2 * z * z, min=1.0e-10)
        # M = 1 in code units.
        H = (r2 * (r_safe - 0.5 * q2)) / den

        den_l = torch.clamp(r2 + a2, min=1.0e-10)
        l_t = torch.ones_like(r_safe)
        l_x = (r_safe * x + a * y) / den_l
        l_y = (r_safe * y - a * x) / den_l
        l_z = z / r_safe
        l_cov = torch.stack([l_t, l_x, l_y, l_z], dim=-1)

        l_contra = l_cov.clone()
        l_contra[:, 0] = -l_contra[:, 0]

        if abs(float(self.metric_charge)) > 1.0e-14:
            # A_mu = -(Q r / Sigma) l_mu, with Sigma = (r^4 + a^2 z^2) / r^2.
            amp = -(q * r3) / den
            a_cov = amp.unsqueeze(-1) * l_cov
        else:
            a_cov = torch.zeros_like(l_cov)

        return r_safe, H, l_cov, l_contra, a_cov

    def _kerr_schild_metric_and_inverse(self, xyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r, H, l_cov, l_contra, _ = self._kerr_schild_kn_fields(xyz)
        if abs(float(self.metric_lambda)) <= 1.0e-14:
            eta = torch.zeros((4, 4), dtype=self.dtype, device=self.device)
            eta[0, 0] = -1.0
            eta[1, 1] = 1.0
            eta[2, 2] = 1.0
            eta[3, 3] = 1.0

            g_cov = eta.unsqueeze(0) + 2.0 * H.view(-1, 1, 1) * (l_cov.unsqueeze(-1) * l_cov.unsqueeze(-2))
            g_inv = eta.unsqueeze(0) - 2.0 * H.view(-1, 1, 1) * (l_contra.unsqueeze(-1) * l_contra.unsqueeze(-2))
            return g_cov, g_inv, r

        g_bg_cov, g_bg_inv = self._kerr_schild_background_metric_and_inverse(xyz)
        ll_cov = l_cov.unsqueeze(-1) * l_cov.unsqueeze(-2)
        g_cov = g_bg_cov + 2.0 * H.view(-1, 1, 1) * ll_cov

        # Sherman-Morrison on the background metric (stable for generalized KS updates).
        l_up_bg = torch.einsum("nij,nj->ni", g_bg_inv, l_cov)
        scalar = torch.sum(l_cov * l_up_bg, dim=-1)
        denom = 1.0 + 2.0 * H * scalar
        denom = torch.where(denom >= 0.0, torch.clamp(denom, min=1.0e-8), torch.clamp(denom, max=-1.0e-8))
        corr = (2.0 * H / denom).view(-1, 1, 1)
        g_inv = g_bg_inv - corr * (l_up_bg.unsqueeze(-1) * l_up_bg.unsqueeze(-2))

        finite = torch.isfinite(g_inv).all(dim=(1, 2))
        if not bool(finite.all()):
            # Robust regularized inverse fallback for rare ill-conditioned updates.
            g_inv = g_inv.clone()
            bad_idx = torch.nonzero(~finite, as_tuple=False).squeeze(-1)
            if bad_idx.numel() > 0:
                mats = g_cov[bad_idx]
                mats = torch.where(torch.isfinite(mats), mats, torch.zeros_like(mats))
                inv_bad = torch.zeros_like(mats)
                unresolved = torch.ones((mats.shape[0],), dtype=torch.bool, device=self.device)
                eye = torch.eye(4, dtype=self.dtype, device=self.device).unsqueeze(0)
                for eps in (1.0e-10, 1.0e-8, 1.0e-6, 1.0e-4, 1.0e-3):
                    if not bool(unresolved.any()):
                        break
                    local_idx = torch.nonzero(unresolved, as_tuple=False).squeeze(-1)
                    mats_try = mats[local_idx] + eps * eye.expand(local_idx.numel(), -1, -1)
                    try:
                        inv_try = torch.linalg.inv(mats_try)
                    except RuntimeError:
                        continue
                    ok = torch.isfinite(inv_try).all(dim=(1, 2))
                    if bool(ok.any()):
                        ok_local = local_idx[ok]
                        inv_bad[ok_local] = inv_try[ok]
                        unresolved[ok_local] = False

                if bool(unresolved.any()):
                    # Final safe fallback: use the background inverse.
                    local_idx = torch.nonzero(unresolved, as_tuple=False).squeeze(-1)
                    inv_bad[local_idx] = g_bg_inv[bad_idx[local_idx]]

                g_inv[bad_idx] = inv_bad

        return g_cov, g_inv, r

    def _metric_dot(self, g_cov: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        ga = torch.matmul(a, g_cov.T)
        return torch.sum(ga * b, dim=-1)

    def _normalize_timelike(self, vec: torch.Tensor, g_cov: torch.Tensor) -> torch.Tensor:
        norm2 = self._metric_dot(g_cov, vec, vec)
        scale = torch.rsqrt(torch.clamp(-norm2, min=1.0e-12))
        if scale.ndim < vec.ndim:
            scale = scale.unsqueeze(-1)
        return vec * scale

    def _camera_rays_kerr_schild_legacy(
        self,
        x_pixel_offset: float = 0.0,
        y_pixel_offset: float = 0.0,
        row_start: int = 0,
        row_end: int | None = None,
    ) -> torch.Tensor:
        cfg = self.config
        full_h, w = cfg.height, cfg.width
        if row_end is None:
            row_end = full_h
        row_start = max(0, min(full_h, int(row_start)))
        row_end = max(row_start, min(full_h, int(row_end)))

        xs = (torch.arange(w, dtype=self.dtype, device=self.device) + 0.5) * (2.0 / float(w)) - 1.0
        ys = 1.0 - ((torch.arange(row_start, row_end, dtype=self.dtype, device=self.device) + 0.5) * (2.0 / float(full_h)))
        if x_pixel_offset != 0.0 and w > 0:
            xs = xs + (2.0 * x_pixel_offset / float(w))
        if y_pixel_offset != 0.0 and full_h > 0:
            ys = ys - (2.0 * y_pixel_offset / float(full_h))
        xx, yy = torch.meshgrid(xs, ys, indexing="xy")

        aspect = torch.as_tensor(w / full_h, dtype=self.dtype, device=self.device)
        tan_half_fov = torch.tan(torch.as_tensor(math.radians(cfg.fov_deg * 0.5), dtype=self.dtype, device=self.device))
        px = xx * aspect * tan_half_fov
        py = yy * tan_half_fov
        d_cam = torch.stack([px, py, torch.ones_like(px)], dim=-1)
        d_cam = d_cam / torch.clamp(torch.linalg.norm(d_cam, dim=-1, keepdim=True), min=1.0e-12)

        theta0, phi0 = self._observer_angles_regularized()
        q_cam = self._camera_orientation_quaternion(theta0, phi0, cfg.observer_roll_deg)
        d_world = self._quat_rotate_batch(q_cam, d_cam).reshape(-1, 3)

        r0 = torch.as_tensor([cfg.observer_radius], dtype=self.dtype, device=self.device)
        th0 = torch.as_tensor([theta0], dtype=self.dtype, device=self.device)
        ph0 = torch.as_tensor([phi0], dtype=self.dtype, device=self.device)
        obs_xyz = self._bl_to_cartesian_kerr_schild(r0, th0, ph0).reshape(1, 3)

        g_cov, g_inv, _ = self._kerr_schild_metric_and_inverse(obs_xyz)
        g_cov0 = g_cov[0]
        g_inv0 = g_inv[0]

        alpha = torch.rsqrt(torch.clamp(-g_inv0[0, 0], min=1.0e-12))
        beta = (alpha * alpha) * g_inv0[0, 1:4]
        n_t = 1.0 / alpha
        n_space = -beta / alpha

        gamma = g_cov0[1:4, 1:4]
        d_gamma = torch.matmul(d_world, gamma)
        norm2 = torch.sum(d_gamma * d_world, dim=-1, keepdim=True)
        v_space = d_world / torch.sqrt(torch.clamp(norm2, min=1.0e-12))

        p_contra = torch.zeros((d_world.shape[0], 4), dtype=self.dtype, device=self.device)
        p_contra[:, 0] = n_t
        p_contra[:, 1:4] = v_space + n_space.view(1, 3)
        p_cov = torch.matmul(p_contra, g_cov0.T)

        state = torch.zeros((d_world.shape[0], 8), dtype=self.dtype, device=self.device)
        state[:, 1:4] = obs_xyz.expand(d_world.shape[0], 3)
        state[:, 4:8] = p_cov
        return state

    def _camera_rays_generalized_doran_legacy(
        self,
        x_pixel_offset: float = 0.0,
        y_pixel_offset: float = 0.0,
        row_start: int = 0,
        row_end: int | None = None,
    ) -> torch.Tensor:
        cfg = self.config
        full_h, w = cfg.height, cfg.width
        if row_end is None:
            row_end = full_h
        row_start = max(0, min(full_h, int(row_start)))
        row_end = max(row_start, min(full_h, int(row_end)))

        xs = (torch.arange(w, dtype=self.dtype, device=self.device) + 0.5) * (2.0 / float(w)) - 1.0
        ys = 1.0 - ((torch.arange(row_start, row_end, dtype=self.dtype, device=self.device) + 0.5) * (2.0 / float(full_h)))
        if x_pixel_offset != 0.0 and w > 0:
            xs = xs + (2.0 * x_pixel_offset / float(w))
        if y_pixel_offset != 0.0 and full_h > 0:
            ys = ys - (2.0 * y_pixel_offset / float(full_h))
        xx, yy = torch.meshgrid(xs, ys, indexing="xy")

        aspect = torch.as_tensor(w / full_h, dtype=self.dtype, device=self.device)
        tan_half_fov = torch.tan(torch.as_tensor(math.radians(cfg.fov_deg * 0.5), dtype=self.dtype, device=self.device))
        px = xx * aspect * tan_half_fov
        py = yy * tan_half_fov
        d_cam = torch.stack([px, py, torch.ones_like(px)], dim=-1)
        d_cam = d_cam / torch.clamp(torch.linalg.norm(d_cam, dim=-1, keepdim=True), min=1.0e-12)

        theta0, phi0 = self._observer_angles_regularized()
        q_cam = self._camera_orientation_quaternion(theta0, phi0, cfg.observer_roll_deg)
        d_world = self._quat_rotate_batch(q_cam, d_cam).reshape(-1, 3)

        r0 = torch.as_tensor([cfg.observer_radius], dtype=self.dtype, device=self.device)
        th0 = torch.as_tensor([theta0], dtype=self.dtype, device=self.device)
        ph0 = torch.as_tensor([phi0], dtype=self.dtype, device=self.device)
        obs_xyz = self._bl_to_cartesian_kerr_schild(r0, th0, ph0).reshape(1, 3)

        g_cov, g_inv, _ = self._kerr_schild_metric_and_inverse(obs_xyz)
        g_cov0 = g_cov[0]
        g_inv0 = g_inv[0]

        alpha = torch.rsqrt(torch.clamp(-g_inv0[0, 0], min=1.0e-12))
        beta = (alpha * alpha) * g_inv0[0, 1:4]

        n_euler = torch.zeros((4,), dtype=self.dtype, device=self.device)
        n_euler[0] = 1.0 / alpha
        n_euler[1:4] = -beta / alpha
        n_euler = self._normalize_timelike(n_euler, g_cov0)

        radial_dir = -obs_xyz.reshape(3)
        radial_norm = torch.linalg.norm(radial_dir)
        fallback = torch.tensor([0.0, 0.0, -1.0], dtype=self.dtype, device=self.device)
        radial_unit = torch.where(
            radial_norm > 1.0e-8,
            radial_dir / torch.clamp(radial_norm, min=1.0e-8),
            fallback,
        )

        q_rad = torch.zeros((4,), dtype=self.dtype, device=self.device)
        q_rad[1:4] = radial_unit
        n_cov = torch.matmul(g_cov0, n_euler)
        n_dot_qrad = torch.sum(n_cov * q_rad)
        s_rad = q_rad + n_dot_qrad * n_euler
        s_norm2 = torch.clamp(self._metric_dot(g_cov0, s_rad, s_rad), min=1.0e-12)
        s_rad = s_rad / torch.sqrt(s_norm2)

        v_ff = torch.sqrt(torch.clamp(1.0 - alpha * alpha, min=0.0, max=0.999 * 0.999))
        gamma_ff = torch.rsqrt(torch.clamp(1.0 - v_ff * v_ff, min=1.0e-8))
        u_obs = gamma_ff * (n_euler + v_ff * s_rad)
        u_obs = self._normalize_timelike(u_obs, g_cov0)
        u_cov = torch.matmul(g_cov0, u_obs)

        q_rays = torch.zeros((d_world.shape[0], 4), dtype=self.dtype, device=self.device)
        q_rays[:, 1:4] = d_world
        u_dot_q = torch.sum(q_rays * u_cov.view(1, 4), dim=-1)
        s_rays = q_rays + u_dot_q.unsqueeze(-1) * u_obs.view(1, 4)
        s_rays_norm2 = self._metric_dot(g_cov0, s_rays, s_rays)
        s_rays = s_rays / torch.sqrt(torch.clamp(s_rays_norm2, min=1.0e-12)).unsqueeze(-1)

        k_contra = u_obs.view(1, 4) + s_rays
        p_cov = torch.matmul(k_contra, g_cov0.T)

        state = torch.zeros((d_world.shape[0], 8), dtype=self.dtype, device=self.device)
        state[:, 1:4] = obs_xyz.expand(d_world.shape[0], 3)
        state[:, 4:8] = p_cov
        return state

    def _camera_rays_kerr_schild(
        self,
        x_pixel_offset: float = 0.0,
        y_pixel_offset: float = 0.0,
        row_start: int = 0,
        row_end: int | None = None,
    ) -> torch.Tensor:
        if not bool(self.config.camera_fastpath):
            return self._camera_rays_kerr_schild_legacy(
                x_pixel_offset=x_pixel_offset,
                y_pixel_offset=y_pixel_offset,
                row_start=row_start,
                row_end=row_end,
            )
        d_world, _, _, theta0, phi0, _, _, _ = self._camera_world_dirs(
            x_pixel_offset=x_pixel_offset,
            y_pixel_offset=y_pixel_offset,
            row_start=row_start,
            row_end=row_end,
        )
        d_world = d_world.reshape(-1, 3)
        obs_xyz, g_cov0, g_inv0, alpha, beta = self._ks_observer_metric(theta0, phi0)
        n_t = 1.0 / alpha
        n_space = -beta / alpha

        gamma = g_cov0[1:4, 1:4]
        d_gamma = torch.matmul(d_world, gamma)
        norm2 = torch.sum(d_gamma * d_world, dim=-1, keepdim=True)
        v_space = d_world / torch.sqrt(torch.clamp(norm2, min=1.0e-12))

        p_contra = torch.zeros((d_world.shape[0], 4), dtype=self.dtype, device=self.device)
        p_contra[:, 0] = n_t
        p_contra[:, 1:4] = v_space + n_space.view(1, 3)
        p_cov = torch.matmul(p_contra, g_cov0.T)

        state = torch.zeros((d_world.shape[0], 8), dtype=self.dtype, device=self.device)
        state[:, 1:4] = obs_xyz.expand(d_world.shape[0], 3)
        state[:, 4:8] = p_cov
        return state

    def _camera_rays_generalized_doran(
        self,
        x_pixel_offset: float = 0.0,
        y_pixel_offset: float = 0.0,
        row_start: int = 0,
        row_end: int | None = None,
    ) -> torch.Tensor:
        """
        PG-like (Doran-inspired) observer frame on top of generalized KS coordinates.

        Rays are initialized from a radially infalling observer tetrad so that
        horizon crossing is regular in KNdS and smoothly reduces to Schwarzschild
        PG-like behavior when a=Q=Lambda=0.
        """
        if not bool(self.config.camera_fastpath):
            return self._camera_rays_generalized_doran_legacy(
                x_pixel_offset=x_pixel_offset,
                y_pixel_offset=y_pixel_offset,
                row_start=row_start,
                row_end=row_end,
            )
        d_world, _, _, theta0, phi0, _, _, _ = self._camera_world_dirs(
            x_pixel_offset=x_pixel_offset,
            y_pixel_offset=y_pixel_offset,
            row_start=row_start,
            row_end=row_end,
        )
        d_world = d_world.reshape(-1, 3)
        obs_xyz, g_cov0, g_inv0, alpha, beta = self._ks_observer_metric(theta0, phi0)

        n_euler = torch.zeros((4,), dtype=self.dtype, device=self.device)
        n_euler[0] = 1.0 / alpha
        n_euler[1:4] = -beta / alpha
        n_euler = self._normalize_timelike(n_euler, g_cov0)

        radial_dir = -obs_xyz.reshape(3)
        radial_norm = torch.linalg.norm(radial_dir)
        fallback = torch.tensor([0.0, 0.0, -1.0], dtype=self.dtype, device=self.device)
        radial_unit = torch.where(
            radial_norm > 1.0e-8,
            radial_dir / torch.clamp(radial_norm, min=1.0e-8),
            fallback,
        )

        q_rad = torch.zeros((4,), dtype=self.dtype, device=self.device)
        q_rad[1:4] = radial_unit
        n_cov = torch.matmul(g_cov0, n_euler)
        n_dot_qrad = torch.sum(n_cov * q_rad)
        s_rad = q_rad + n_dot_qrad * n_euler
        s_norm2 = torch.clamp(self._metric_dot(g_cov0, s_rad, s_rad), min=1.0e-12)
        s_rad = s_rad / torch.sqrt(s_norm2)

        # PG-like infall speed proxy from lapse; gives v=sqrt(2M/r) in Schwarzschild.
        v_ff = torch.sqrt(torch.clamp(1.0 - alpha * alpha, min=0.0, max=0.999 * 0.999))
        gamma_ff = torch.rsqrt(torch.clamp(1.0 - v_ff * v_ff, min=1.0e-8))
        u_obs = gamma_ff * (n_euler + v_ff * s_rad)
        u_obs = self._normalize_timelike(u_obs, g_cov0)
        u_cov = torch.matmul(g_cov0, u_obs)

        q_rays = torch.zeros((d_world.shape[0], 4), dtype=self.dtype, device=self.device)
        q_rays[:, 1:4] = d_world
        u_dot_q = torch.sum(q_rays * u_cov.view(1, 4), dim=-1)
        s_rays = q_rays + u_dot_q.unsqueeze(-1) * u_obs.view(1, 4)
        s_rays_norm2 = self._metric_dot(g_cov0, s_rays, s_rays)
        s_rays = s_rays / torch.sqrt(torch.clamp(s_rays_norm2, min=1.0e-12)).unsqueeze(-1)

        k_contra = u_obs.view(1, 4) + s_rays
        p_cov = torch.matmul(k_contra, g_cov0.T)

        state = torch.zeros((d_world.shape[0], 8), dtype=self.dtype, device=self.device)
        state[:, 1:4] = obs_xyz.expand(d_world.shape[0], 3)
        state[:, 4:8] = p_cov
        return state

    def _rhs_kerr_schild_numeric(self, state: torch.Tensor) -> torch.Tensor:
        xyz = state[:, 1:4]
        p_cov = state[:, 4:8]

        _, g_inv, _ = self._kerr_schild_metric_and_inverse(xyz)
        p_contra = torch.einsum("nij,nj->ni", g_inv, p_cov)

        eps = torch.as_tensor(2.5e-3, dtype=self.dtype, device=self.device)
        dginv = []
        for axis in range(3):
            offset = torch.zeros_like(xyz)
            offset[:, axis] = eps
            _, g_inv_p, _ = self._kerr_schild_metric_and_inverse(xyz + offset)
            _, g_inv_m, _ = self._kerr_schild_metric_and_inverse(xyz - offset)
            dginv.append((g_inv_p - g_inv_m) / (2.0 * eps))

        dp_x = -0.5 * torch.einsum("nij,ni,nj->n", dginv[0], p_cov, p_cov)
        dp_y = -0.5 * torch.einsum("nij,ni,nj->n", dginv[1], p_cov, p_cov)
        dp_z = -0.5 * torch.einsum("nij,ni,nj->n", dginv[2], p_cov, p_cov)
        dp_t = torch.zeros_like(dp_x)

        deriv = torch.stack(
            [
                p_contra[:, 0],
                p_contra[:, 1],
                p_contra[:, 2],
                p_contra[:, 3],
                dp_t,
                dp_x,
                dp_y,
                dp_z,
            ],
            dim=-1,
        )
        return deriv

    def _rhs_kerr_schild_analytic(self, state: torch.Tensor) -> torch.Tensor:
        xyz = state[:, 1:4]
        x = xyz[:, 0]
        y = xyz[:, 1]
        z = xyz[:, 2]

        p_cov = state[:, 4:8]
        p_t = p_cov[:, 0]
        p_x = p_cov[:, 1]
        p_y = p_cov[:, 2]
        p_z = p_cov[:, 3]

        r = self._ks_radius_from_xyz(xyz)
        r_safe = torch.clamp(r, min=1.0e-8)
        r2 = r_safe * r_safe
        r3 = r2 * r_safe
        r4 = r2 * r2

        a = torch.as_tensor(self.metric_spin, dtype=self.dtype, device=self.device)
        a2 = a * a
        q2 = torch.as_tensor(self.metric_charge * self.metric_charge, dtype=self.dtype, device=self.device)

        den = torch.clamp(r4 + a2 * z * z, min=1.0e-10)
        den2 = den * den

        dr_dx = (r3 * x) / den
        dr_dy = (r3 * y) / den
        dr_dz = (a2 * r_safe * z) / den

        num = r2 * (r_safe - 0.5 * q2)
        num_prime = 3.0 * r2 - q2 * r_safe
        H = num / den

        dH_dr = (num_prime * den - 4.0 * r3 * num) / den2
        dH_dx = dH_dr * dr_dx
        dH_dy = dH_dr * dr_dy
        dH_dz = dH_dr * dr_dz - (2.0 * a2 * z * num) / den2

        den_l = torch.clamp(r2 + a2, min=1.0e-10)
        den_l2 = den_l * den_l

        num_lx = r_safe * x + a * y
        num_ly = r_safe * y - a * x
        l_x = num_lx / den_l
        l_y = num_ly / den_l
        l_z = z / r_safe

        l_dot_p = -p_t + l_x * p_x + l_y * p_y + l_z * p_z
        l_dot_p2 = l_dot_p * l_dot_p

        p_contra_t = -p_t + 2.0 * H * l_dot_p
        p_contra_x = p_x - 2.0 * H * l_dot_p * l_x
        p_contra_y = p_y - 2.0 * H * l_dot_p * l_y
        p_contra_z = p_z - 2.0 * H * l_dot_p * l_z

        dden_l_dx = 2.0 * r_safe * dr_dx
        dden_l_dy = 2.0 * r_safe * dr_dy
        dden_l_dz = 2.0 * r_safe * dr_dz

        dnum_lx_dx = r_safe + x * dr_dx
        dnum_lx_dy = x * dr_dy + a
        dnum_lx_dz = x * dr_dz

        dnum_ly_dx = y * dr_dx - a
        dnum_ly_dy = r_safe + y * dr_dy
        dnum_ly_dz = y * dr_dz

        dlx_dx = (dnum_lx_dx * den_l - num_lx * dden_l_dx) / den_l2
        dlx_dy = (dnum_lx_dy * den_l - num_lx * dden_l_dy) / den_l2
        dlx_dz = (dnum_lx_dz * den_l - num_lx * dden_l_dz) / den_l2

        dly_dx = (dnum_ly_dx * den_l - num_ly * dden_l_dx) / den_l2
        dly_dy = (dnum_ly_dy * den_l - num_ly * dden_l_dy) / den_l2
        dly_dz = (dnum_ly_dz * den_l - num_ly * dden_l_dz) / den_l2

        r2_inv = 1.0 / torch.clamp(r2, min=1.0e-12)
        dlz_dx = -(z * dr_dx) * r2_inv
        dlz_dy = -(z * dr_dy) * r2_inv
        dlz_dz = (1.0 / r_safe) - (z * dr_dz) * r2_inv

        dl_dot_p_dx = dlx_dx * p_x + dly_dx * p_y + dlz_dx * p_z
        dl_dot_p_dy = dlx_dy * p_x + dly_dy * p_y + dlz_dy * p_z
        dl_dot_p_dz = dlx_dz * p_x + dly_dz * p_y + dlz_dz * p_z

        common = 2.0 * H * l_dot_p
        dp_x = dH_dx * l_dot_p2 + common * dl_dot_p_dx
        dp_y = dH_dy * l_dot_p2 + common * dl_dot_p_dy
        dp_z = dH_dz * l_dot_p2 + common * dl_dot_p_dz
        dp_t = torch.zeros_like(dp_x)

        deriv = torch.stack(
            [
                p_contra_t,
                p_contra_x,
                p_contra_y,
                p_contra_z,
                dp_t,
                dp_x,
                dp_y,
                dp_z,
            ],
            dim=-1,
        )
        return deriv

    def _rhs_kerr_schild(self, state: torch.Tensor) -> torch.Tensor:
        return self._rhs_kerr_schild_fn(state)

    def _null_norm_kerr_schild(self, state: torch.Tensor) -> torch.Tensor:
        xyz = state[:, 1:4]
        p_cov = state[:, 4:8]
        _, g_inv, _ = self._kerr_schild_metric_and_inverse(xyz)
        p_contra = torch.einsum("nij,nj->ni", g_inv, p_cov)
        return torch.sum(p_cov * p_contra, dim=-1)

    def _rk4_step_kerr_schild(self, state: torch.Tensor, h: float) -> torch.Tensor:
        k1 = self._rhs_kerr_schild(state)
        k2 = self._rhs_kerr_schild(state + 0.5 * h * k1)
        k3 = self._rhs_kerr_schild(state + 0.5 * h * k2)
        k4 = self._rhs_kerr_schild(state + h * k3)
        return state + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def _rk4_step_kerr_schild_with_k1(self, state: torch.Tensor, h: float) -> tuple[torch.Tensor, torch.Tensor]:
        k1 = self._rhs_kerr_schild(state)
        k2 = self._rhs_kerr_schild(state + 0.5 * h * k1)
        k3 = self._rhs_kerr_schild(state + 0.5 * h * k2)
        k4 = self._rhs_kerr_schild(state + h * k3)
        nxt = state + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return nxt, k1

    def _rk45_adaptive_step_kerr_schild(
        self,
        state: torch.Tensor,
        h: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg = self.config
        h_col = h.unsqueeze(-1)

        k1 = self._rhs_kerr_schild(state)
        k2 = self._rhs_kerr_schild(state + h_col * (1.0 / 5.0) * k1)
        k3 = self._rhs_kerr_schild(state + h_col * ((3.0 / 40.0) * k1 + (9.0 / 40.0) * k2))
        k4 = self._rhs_kerr_schild(state + h_col * ((44.0 / 45.0) * k1 + (-56.0 / 15.0) * k2 + (32.0 / 9.0) * k3))
        k5 = self._rhs_kerr_schild(
            state
            + h_col
            * (
                (19372.0 / 6561.0) * k1
                + (-25360.0 / 2187.0) * k2
                + (64448.0 / 6561.0) * k3
                + (-212.0 / 729.0) * k4
            )
        )
        k6 = self._rhs_kerr_schild(
            state
            + h_col
            * (
                (9017.0 / 3168.0) * k1
                + (-355.0 / 33.0) * k2
                + (46732.0 / 5247.0) * k3
                + (49.0 / 176.0) * k4
                + (-5103.0 / 18656.0) * k5
            )
        )

        y5 = state + h_col * (
            (35.0 / 384.0) * k1
            + (500.0 / 1113.0) * k3
            + (125.0 / 192.0) * k4
            + (-2187.0 / 6784.0) * k5
            + (11.0 / 84.0) * k6
        )
        k7 = self._rhs_kerr_schild(y5)

        y4 = state + h_col * (
            (5179.0 / 57600.0) * k1
            + (7571.0 / 16695.0) * k3
            + (393.0 / 640.0) * k4
            + (-92097.0 / 339200.0) * k5
            + (187.0 / 2100.0) * k6
            + (1.0 / 40.0) * k7
        )

        err = y5 - y4
        scale = cfg.adaptive_atol + cfg.adaptive_rtol * torch.maximum(torch.abs(state), torch.abs(y5))
        scale = torch.clamp(scale, min=1.0e-12)
        err_ratio = torch.sqrt(torch.mean(torch.square(err / scale), dim=-1))

        finite = torch.isfinite(y5).all(dim=-1) & torch.isfinite(err_ratio)
        err_ratio = torch.where(finite, err_ratio, torch.full_like(err_ratio, float("inf")))

        h_min = torch.as_tensor(cfg.adaptive_step_min, dtype=self.dtype, device=self.device)
        h_max = torch.as_tensor(cfg.adaptive_step_max, dtype=self.dtype, device=self.device)
        at_min = h <= (1.0001 * h_min)

        accept = ((err_ratio <= 1.0) | at_min) & finite
        fatal = (~finite) & at_min

        err_safe = torch.clamp(err_ratio, min=1.0e-9)
        fac_accept = torch.clamp(0.9 * torch.pow(err_safe, -0.2), min=0.30, max=2.50)
        fac_reject = torch.clamp(0.9 * torch.pow(err_safe, -0.25), min=0.10, max=0.50)
        h_new = torch.where(accept, h * fac_accept, h * fac_reject)
        h_new = torch.clamp(h_new, min=h_min, max=h_max)

        return y5, accept, h_new, fatal

    def _rk45_adaptive_step_kerr_schild_fsal(
        self,
        state: torch.Tensor,
        h: torch.Tensor,
        k1: torch.Tensor | None = None,
        k1_valid: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg = self.config
        h_col = h.unsqueeze(-1)

        if k1 is None:
            k1_eff = self._rhs_kerr_schild(state)
        else:
            k1_eff = k1
            if k1_valid is not None:
                if k1_valid.dtype != torch.bool:
                    k1_valid = k1_valid.to(dtype=torch.bool)
                missing = ~k1_valid
                if bool(missing.any()):
                    missing_pos = torch.nonzero(missing, as_tuple=False).squeeze(-1)
                    k1_eff = k1_eff.clone()
                    k1_eff[missing_pos] = self._rhs_kerr_schild(state[missing_pos])

        k2 = self._rhs_kerr_schild(state + h_col * (1.0 / 5.0) * k1_eff)
        k3 = self._rhs_kerr_schild(state + h_col * ((3.0 / 40.0) * k1_eff + (9.0 / 40.0) * k2))
        k4 = self._rhs_kerr_schild(state + h_col * ((44.0 / 45.0) * k1_eff + (-56.0 / 15.0) * k2 + (32.0 / 9.0) * k3))
        k5 = self._rhs_kerr_schild(
            state
            + h_col
            * (
                (19372.0 / 6561.0) * k1_eff
                + (-25360.0 / 2187.0) * k2
                + (64448.0 / 6561.0) * k3
                + (-212.0 / 729.0) * k4
            )
        )
        k6 = self._rhs_kerr_schild(
            state
            + h_col
            * (
                (9017.0 / 3168.0) * k1_eff
                + (-355.0 / 33.0) * k2
                + (46732.0 / 5247.0) * k3
                + (49.0 / 176.0) * k4
                + (-5103.0 / 18656.0) * k5
            )
        )

        y5 = state + h_col * (
            (35.0 / 384.0) * k1_eff
            + (500.0 / 1113.0) * k3
            + (125.0 / 192.0) * k4
            + (-2187.0 / 6784.0) * k5
            + (11.0 / 84.0) * k6
        )
        k7 = self._rhs_kerr_schild(y5)

        y4 = state + h_col * (
            (5179.0 / 57600.0) * k1_eff
            + (7571.0 / 16695.0) * k3
            + (393.0 / 640.0) * k4
            + (-92097.0 / 339200.0) * k5
            + (187.0 / 2100.0) * k6
            + (1.0 / 40.0) * k7
        )

        err = y5 - y4
        scale = cfg.adaptive_atol + cfg.adaptive_rtol * torch.maximum(torch.abs(state), torch.abs(y5))
        scale = torch.clamp(scale, min=1.0e-12)
        err_ratio = torch.sqrt(torch.mean(torch.square(err / scale), dim=-1))

        finite = torch.isfinite(y5).all(dim=-1) & torch.isfinite(err_ratio)
        err_ratio = torch.where(finite, err_ratio, torch.full_like(err_ratio, float("inf")))

        h_min = torch.as_tensor(cfg.adaptive_step_min, dtype=self.dtype, device=self.device)
        h_max = torch.as_tensor(cfg.adaptive_step_max, dtype=self.dtype, device=self.device)
        at_min = h <= (1.0001 * h_min)

        accept = ((err_ratio <= 1.0) | at_min) & finite
        fatal = (~finite) & at_min

        err_safe = torch.clamp(err_ratio, min=1.0e-9)
        fac_accept = torch.clamp(0.9 * torch.pow(err_safe, -0.2), min=0.30, max=2.50)
        fac_reject = torch.clamp(0.9 * torch.pow(err_safe, -0.25), min=0.10, max=0.50)
        h_new = torch.where(accept, h * fac_accept, h * fac_reject)
        h_new = torch.clamp(h_new, min=h_min, max=h_max)

        return y5, accept, h_new, fatal, k7

    def _bl_to_cartesian(self, r: torch.Tensor, theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        a = torch.as_tensor(self.metric_spin, dtype=self.dtype, device=self.device)
        rho = torch.sqrt(torch.clamp(r * r + a * a, min=1.0e-9)) * torch.sin(theta)
        x = rho * torch.cos(phi)
        y = rho * torch.sin(phi)
        z = r * torch.cos(theta)
        return torch.stack([x, y, z], dim=-1)

    def _rhs(self, state: torch.Tensor) -> torch.Tensor:
        if self.is_wormhole:
            r = state[:, 1]
        else:
            r = torch.clamp(state[:, 1], min=1.0e-3)
        theta = torch.clamp(state[:, 2], min=THETA_EPS, max=math.pi - THETA_EPS)

        p_t = state[:, 4]
        p_r = state[:, 5]
        p_theta = state[:, 6]
        p_phi = state[:, 7]

        inv = inverse_metric_components(
            r,
            theta,
            self.config.spin,
            self.config.metric_model,
            self.config.charge,
            self.config.cosmological_constant,
            self.config.wormhole_throat_radius,
            self.config.wormhole_length_scale,
            self.config.wormhole_lensing_scale,
        )
        d_r, d_theta = inverse_metric_derivatives(
            r,
            theta,
            self.config.spin,
            self.config.metric_model,
            self.config.charge,
            self.config.cosmological_constant,
            self.config.wormhole_throat_radius,
            self.config.wormhole_length_scale,
            self.config.wormhole_lensing_scale,
        )

        dt = inv.gtt * p_t + inv.gtphi * p_phi
        dr = inv.grr * p_r
        dtheta = inv.gthth * p_theta
        dphi = inv.gtphi * p_t + inv.gphiphi * p_phi

        p_t2 = p_t * p_t
        p_r2 = p_r * p_r
        p_theta2 = p_theta * p_theta
        p_phi2 = p_phi * p_phi

        contract_r = (
            d_r.gtt * p_t2
            + 2.0 * d_r.gtphi * p_t * p_phi
            + d_r.grr * p_r2
            + d_r.gthth * p_theta2
            + d_r.gphiphi * p_phi2
        )
        contract_theta = (
            d_theta.gtt * p_t2
            + 2.0 * d_theta.gtphi * p_t * p_phi
            + d_theta.grr * p_r2
            + d_theta.gthth * p_theta2
            + d_theta.gphiphi * p_phi2
        )

        dp_t = torch.zeros_like(p_t)
        dp_r = -0.5 * contract_r
        dp_theta = -0.5 * contract_theta
        dp_phi = torch.zeros_like(p_phi)

        deriv = torch.stack([dt, dr, dtheta, dphi, dp_t, dp_r, dp_theta, dp_phi], dim=-1)
        return deriv

    def _regularize_angular_state(self, state: torch.Tensor) -> torch.Tensor:
        two_pi = 2.0 * math.pi
        theta_raw = torch.remainder(state[:, 2], two_pi)
        crossed_pole = theta_raw > math.pi

        theta_fold = torch.where(crossed_pole, two_pi - theta_raw, theta_raw)
        phi_shift = torch.where(crossed_pole, torch.full_like(theta_raw, math.pi), torch.zeros_like(theta_raw))

        state[:, 2] = torch.clamp(theta_fold, min=THETA_EPS, max=math.pi - THETA_EPS)
        if self.is_wormhole and bool(self.config.wormhole_mt_unwrap_phi):
            # Morris-Thorne seam fix: keep phi unwrapped during integration.
            state[:, 3] = state[:, 3] + phi_shift
        else:
            state[:, 3] = torch.remainder(state[:, 3] + phi_shift, two_pi)
        state[:, 6] = torch.where(crossed_pole, -state[:, 6], state[:, 6])
        return state

    def _mt_shortest_arc_enabled(self) -> bool:
        return bool(self.is_wormhole and self.config.wormhole_mt_shortest_arc_phi_interp)

    def _interp_phi_shortest_arc(
        self,
        phi0: torch.Tensor,
        phi1: torch.Tensor,
        alpha: torch.Tensor,
    ) -> torch.Tensor:
        two_pi = torch.as_tensor(2.0 * math.pi, dtype=self.dtype, device=self.device)
        pi = torch.as_tensor(math.pi, dtype=self.dtype, device=self.device)
        dphi = torch.remainder((phi1 - phi0) + pi, two_pi) - pi
        return phi0 + alpha * dphi

    def _mt_impact_parameter_side(self, state: torch.Tensor) -> torch.Tensor:
        """Return (b - b₀) for each ray in the Morris-Thorne wormhole case.

        b = L_total / E is the ray's impact parameter, where L_total is the
        conserved total angular momentum and E = -p_t the conserved energy.
        b₀ = wormhole_throat_radius is the critical impact parameter: rays with
        b > b₀ do not cross the throat (local side, positive result) and rays
        with b < b₀ cross it (remote side, negative result).

        Because MT is spherically symmetric, L_total² = p_θ² + p_φ²/sin²θ is a
        constant of motion and can be evaluated from the final integration state.
        Using (b − b₀) as the local/remote indicator allows the tanh blend in
        _star_background to anti-alias both sides of the critical curve
        symmetrically, unlike r_min which saturates at ±escape_radius for all
        non-grazing rays and gives a binary result.
        """
        p_t = state[:, 4]
        p_theta = state[:, 6]
        p_phi = state[:, 7]
        sin2_theta = torch.clamp(torch.sin(state[:, 2]) ** 2, min=1.0e-9)
        L_total_sq = torch.clamp(p_theta * p_theta + p_phi * p_phi / sin2_theta, min=0.0)
        L_total = torch.sqrt(L_total_sq)
        E = torch.clamp(torch.abs(p_t), min=1.0e-9)
        b = L_total / E
        b0 = torch.as_tensor(
            float(self.config.wormhole_throat_radius), dtype=self.dtype, device=self.device
        )
        return b - b0

    def _sky_angles_from_state(
        self,
        state: torch.Tensor,
        r_min: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.is_wormhole and bool(self.config.wormhole_mt_sky_sample_from_xyz):
            sky_xyz = self._bl_to_cartesian(state[:, 1], state[:, 2], state[:, 3])
            sky_norm = torch.clamp(torch.linalg.norm(sky_xyz, dim=-1), min=1.0e-8)
            sky_theta = torch.acos(torch.clamp(sky_xyz[:, 2] / sky_norm, min=-1.0, max=1.0))
            sky_theta = torch.clamp(sky_theta, min=THETA_EPS, max=math.pi - THETA_EPS)
            sky_phi = torch.remainder(torch.atan2(sky_xyz[:, 1], sky_xyz[:, 0]), 2.0 * self.pi)
            sky_side = self._mt_impact_parameter_side(state) if self.is_wormhole else state[:, 1]
            return sky_theta, sky_phi, sky_side

        sky_theta = torch.clamp(state[:, 2], min=THETA_EPS, max=math.pi - THETA_EPS)
        sky_phi = torch.remainder(state[:, 3], 2.0 * self.pi)
        # For Morris-Thorne: use (b − b₀) so the tanh blend in _star_background
        # gives symmetric anti-aliasing on both sides of the critical curve.
        sky_side = self._mt_impact_parameter_side(state) if self.is_wormhole else state[:, 1]
        return sky_theta, sky_phi, sky_side

    def _apply_radial_domain(self, state: torch.Tensor) -> torch.Tensor:
        # Kerr/Schwarzschild families use r > 0.
        # Morris-Thorne uses signed proper radial coordinate and must allow r < 0.
        if self.is_wormhole:
            return state
        state[:, 1] = torch.clamp(state[:, 1], min=1.0e-3)
        return state

    def _rk4_step(self, state: torch.Tensor, h: float) -> torch.Tensor:
        k1 = self._rhs_fn(state)
        k2 = self._rhs_fn(state + 0.5 * h * k1)
        k3 = self._rhs_fn(state + 0.5 * h * k2)
        k4 = self._rhs_fn(state + h * k3)
        nxt = state + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        nxt = self._apply_radial_domain(nxt)
        return self._regularize_angular_state(nxt)

    def _rk4_step_with_k1(self, state: torch.Tensor, h: float) -> tuple[torch.Tensor, torch.Tensor]:
        k1 = self._rhs_fn(state)
        k2 = self._rhs_fn(state + 0.5 * h * k1)
        k3 = self._rhs_fn(state + 0.5 * h * k2)
        k4 = self._rhs_fn(state + h * k3)
        nxt = state + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        nxt = self._apply_radial_domain(nxt)
        return self._regularize_angular_state(nxt), k1

    def _event_aware_step_factor_bl(
        self,
        r: torch.Tensor,
        theta: torch.Tensor,
        horizon_cut: torch.Tensor,
    ) -> torch.Tensor:
        cfg = self.config
        horizon_band = torch.as_tensor(
            max(0.45, 2.0 * float(cfg.adaptive_step_max)),
            dtype=self.dtype,
            device=self.device,
        )
        dist_h = torch.abs(r - horizon_cut)
        scale_h = 0.45 + 0.55 * torch.clamp(dist_h / horizon_band, min=0.0, max=1.0)

        if cfg.thick_disk and cfg.disk_thickness_ratio > 0.0:
            z = r * torch.cos(theta)
            h_disk, _ = self._disk_half_thickness_and_slope(r)
            dist_d = torch.abs(torch.abs(z) - h_disk)
            band_d = torch.clamp(0.40 * h_disk + 0.05 * r, min=0.04)
        else:
            dist_d = torch.abs(r * torch.cos(theta))
            band_d = torch.clamp(0.08 * r + 0.06, min=0.05)
        scale_d = 0.48 + 0.52 * torch.clamp(dist_d / torch.clamp(band_d, min=1.0e-6), min=0.0, max=1.0)
        return torch.clamp(torch.minimum(scale_h, scale_d), min=0.45, max=1.0)

    def _event_aware_step_factor_ks(
        self,
        xyz: torch.Tensor,
        horizon_cut: torch.Tensor,
    ) -> torch.Tensor:
        cfg = self.config
        r = self._ks_radius_from_xyz(xyz)
        horizon_band = torch.as_tensor(
            max(0.45, 2.0 * float(cfg.adaptive_step_max)),
            dtype=self.dtype,
            device=self.device,
        )
        dist_h = torch.abs(r - horizon_cut)
        scale_h = 0.45 + 0.55 * torch.clamp(dist_h / horizon_band, min=0.0, max=1.0)

        z = torch.abs(xyz[:, 2])
        if cfg.thick_disk and cfg.disk_thickness_ratio > 0.0:
            h_disk, _ = self._disk_half_thickness_and_slope(r)
            dist_d = torch.abs(z - h_disk)
            band_d = torch.clamp(0.40 * h_disk + 0.05 * r, min=0.04)
        else:
            dist_d = z
            band_d = torch.clamp(0.08 * r + 0.06, min=0.05)
        scale_d = 0.48 + 0.52 * torch.clamp(dist_d / torch.clamp(band_d, min=1.0e-6), min=0.0, max=1.0)
        return torch.clamp(torch.minimum(scale_h, scale_d), min=0.45, max=1.0)

    def _rk45_adaptive_step(
        self,
        state: torch.Tensor,
        h: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Dormand-Prince 5(4) step with per-ray local error control."""
        cfg = self.config
        h_col = h.unsqueeze(-1)

        k1 = self._rhs_fn(state)
        k2 = self._rhs_fn(state + h_col * (1.0 / 5.0) * k1)
        k3 = self._rhs_fn(state + h_col * ((3.0 / 40.0) * k1 + (9.0 / 40.0) * k2))
        k4 = self._rhs_fn(state + h_col * ((44.0 / 45.0) * k1 + (-56.0 / 15.0) * k2 + (32.0 / 9.0) * k3))
        k5 = self._rhs_fn(
            state
            + h_col
            * (
                (19372.0 / 6561.0) * k1
                + (-25360.0 / 2187.0) * k2
                + (64448.0 / 6561.0) * k3
                + (-212.0 / 729.0) * k4
            )
        )
        k6 = self._rhs_fn(
            state
            + h_col
            * (
                (9017.0 / 3168.0) * k1
                + (-355.0 / 33.0) * k2
                + (46732.0 / 5247.0) * k3
                + (49.0 / 176.0) * k4
                + (-5103.0 / 18656.0) * k5
            )
        )

        y5 = state + h_col * (
            (35.0 / 384.0) * k1
            + (500.0 / 1113.0) * k3
            + (125.0 / 192.0) * k4
            + (-2187.0 / 6784.0) * k5
            + (11.0 / 84.0) * k6
        )
        k7 = self._rhs_fn(y5)

        y4 = state + h_col * (
            (5179.0 / 57600.0) * k1
            + (7571.0 / 16695.0) * k3
            + (393.0 / 640.0) * k4
            + (-92097.0 / 339200.0) * k5
            + (187.0 / 2100.0) * k6
            + (1.0 / 40.0) * k7
        )

        err = y5 - y4
        scale = cfg.adaptive_atol + cfg.adaptive_rtol * torch.maximum(torch.abs(state), torch.abs(y5))
        scale = torch.clamp(scale, min=1.0e-12)
        err_ratio = torch.sqrt(torch.mean(torch.square(err / scale), dim=-1))

        y5 = self._apply_radial_domain(y5)
        y5 = self._regularize_angular_state(y5)

        finite = torch.isfinite(y5).all(dim=-1) & torch.isfinite(err_ratio)
        err_ratio = torch.where(finite, err_ratio, torch.full_like(err_ratio, float("inf")))

        h_min = torch.as_tensor(cfg.adaptive_step_min, dtype=self.dtype, device=self.device)
        h_max = torch.as_tensor(cfg.adaptive_step_max, dtype=self.dtype, device=self.device)
        at_min = h <= (1.0001 * h_min)

        accept = ((err_ratio <= 1.0) | at_min) & finite
        fatal = (~finite) & at_min

        err_safe = torch.clamp(err_ratio, min=1.0e-9)
        fac_accept = torch.clamp(0.9 * torch.pow(err_safe, -0.2), min=0.30, max=2.50)
        fac_reject = torch.clamp(0.9 * torch.pow(err_safe, -0.25), min=0.10, max=0.50)
        h_new = torch.where(accept, h * fac_accept, h * fac_reject)
        h_new = torch.clamp(h_new, min=h_min, max=h_max)

        return y5, accept, h_new, fatal

    def _rk45_adaptive_step_fsal(
        self,
        state: torch.Tensor,
        h: torch.Tensor,
        k1: torch.Tensor | None = None,
        k1_valid: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Dormand-Prince 5(4) with optional FSAL k1 reuse for BL tracing."""
        cfg = self.config
        h_col = h.unsqueeze(-1)

        if k1 is None:
            k1_eff = self._rhs_fn(state)
        else:
            k1_eff = k1
            if k1_valid is not None:
                if k1_valid.dtype != torch.bool:
                    k1_valid = k1_valid.to(dtype=torch.bool)
                missing = ~k1_valid
                if bool(missing.any()):
                    missing_pos = torch.nonzero(missing, as_tuple=False).squeeze(-1)
                    k1_eff = k1_eff.clone()
                    k1_eff[missing_pos] = self._rhs_fn(state[missing_pos])

        k2 = self._rhs_fn(state + h_col * (1.0 / 5.0) * k1_eff)
        k3 = self._rhs_fn(state + h_col * ((3.0 / 40.0) * k1_eff + (9.0 / 40.0) * k2))
        k4 = self._rhs_fn(state + h_col * ((44.0 / 45.0) * k1_eff + (-56.0 / 15.0) * k2 + (32.0 / 9.0) * k3))
        k5 = self._rhs_fn(
            state
            + h_col
            * (
                (19372.0 / 6561.0) * k1_eff
                + (-25360.0 / 2187.0) * k2
                + (64448.0 / 6561.0) * k3
                + (-212.0 / 729.0) * k4
            )
        )
        k6 = self._rhs_fn(
            state
            + h_col
            * (
                (9017.0 / 3168.0) * k1_eff
                + (-355.0 / 33.0) * k2
                + (46732.0 / 5247.0) * k3
                + (49.0 / 176.0) * k4
                + (-5103.0 / 18656.0) * k5
            )
        )

        y5 = state + h_col * (
            (35.0 / 384.0) * k1_eff
            + (500.0 / 1113.0) * k3
            + (125.0 / 192.0) * k4
            + (-2187.0 / 6784.0) * k5
            + (11.0 / 84.0) * k6
        )
        k7 = self._rhs_fn(y5)

        y4 = state + h_col * (
            (5179.0 / 57600.0) * k1_eff
            + (7571.0 / 16695.0) * k3
            + (393.0 / 640.0) * k4
            + (-92097.0 / 339200.0) * k5
            + (187.0 / 2100.0) * k6
            + (1.0 / 40.0) * k7
        )

        err = y5 - y4
        scale = cfg.adaptive_atol + cfg.adaptive_rtol * torch.maximum(torch.abs(state), torch.abs(y5))
        scale = torch.clamp(scale, min=1.0e-12)
        err_ratio = torch.sqrt(torch.mean(torch.square(err / scale), dim=-1))

        y5 = self._apply_radial_domain(y5)
        y5 = self._regularize_angular_state(y5)

        finite = torch.isfinite(y5).all(dim=-1) & torch.isfinite(err_ratio)
        err_ratio = torch.where(finite, err_ratio, torch.full_like(err_ratio, float("inf")))

        h_min = torch.as_tensor(cfg.adaptive_step_min, dtype=self.dtype, device=self.device)
        h_max = torch.as_tensor(cfg.adaptive_step_max, dtype=self.dtype, device=self.device)
        at_min = h <= (1.0001 * h_min)

        accept = ((err_ratio <= 1.0) | at_min) & finite
        fatal = (~finite) & at_min

        err_safe = torch.clamp(err_ratio, min=1.0e-9)
        fac_accept = torch.clamp(0.9 * torch.pow(err_safe, -0.2), min=0.30, max=2.50)
        fac_reject = torch.clamp(0.9 * torch.pow(err_safe, -0.25), min=0.10, max=0.50)
        h_new = torch.where(accept, h * fac_accept, h * fac_reject)
        h_new = torch.clamp(h_new, min=h_min, max=h_max)

        return y5, accept, h_new, fatal, k7

    def _hermite_interp(
        self,
        y0: torch.Tensor,
        y1: torch.Tensor,
        dy0: torch.Tensor,
        dy1: torch.Tensor,
        h: torch.Tensor,
        alpha: torch.Tensor,
    ) -> torch.Tensor:
        a2 = alpha * alpha
        a3 = a2 * alpha
        h00 = 2.0 * a3 - 3.0 * a2 + 1.0
        h10 = a3 - 2.0 * a2 + alpha
        h01 = -2.0 * a3 + 3.0 * a2
        h11 = a3 - a2
        return h00 * y0 + h10 * (h * dy0) + h01 * y1 + h11 * (h * dy1)

    def _refine_event_alpha(
        self,
        y0: torch.Tensor,
        y1: torch.Tensor,
        dy0: torch.Tensor,
        dy1: torch.Tensor,
        h: torch.Tensor,
        target: torch.Tensor,
        iterations: int = 7,
    ) -> torch.Tensor:
        f0 = y0 - target
        f1 = y1 - target
        sign_change = ((f0 <= 0.0) & (f1 >= 0.0)) | ((f0 >= 0.0) & (f1 <= 0.0))
        alpha = torch.abs(f0) / (torch.abs(f0) + torch.abs(f1) + 1.0e-12)
        alpha = torch.clamp(alpha, min=0.0, max=1.0)

        if bool(sign_change.any()):
            lo = torch.zeros_like(alpha)
            hi = torch.ones_like(alpha)
            for _ in range(iterations):
                mid = 0.5 * (lo + hi)
                y_mid = self._hermite_interp(y0, y1, dy0, dy1, h, mid)
                f_mid = y_mid - target
                left_has_root = ((f0 <= 0.0) & (f_mid >= 0.0)) | ((f0 >= 0.0) & (f_mid <= 0.0))
                hi = torch.where(sign_change & left_has_root, mid, hi)
                lo = torch.where(sign_change & (~left_has_root), mid, lo)
            alpha_refined = 0.5 * (lo + hi)
            alpha = torch.where(sign_change, alpha_refined, alpha)
        return alpha

    def _fract(self, x: torch.Tensor) -> torch.Tensor:
        return x - torch.floor(x)

    def _lerp(self, a: torch.Tensor, b: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return a + (b - a) * t

    def _hash2(self, x: torch.Tensor, y: torch.Tensor, offset: float, period_x: float | None = None, period_y: float | None = None) -> torch.Tensor:
        if period_x is not None and period_x > 0.0:
            x = torch.remainder(x, period_x)
        if period_y is not None and period_y > 0.0:
            y = torch.remainder(y, period_y)
        value = torch.sin(x * 127.1 + y * 311.7 + offset + float(self.config.star_seed) * 17.0) * 43758.5453
        return self._fract(value)

    def _value_noise(self, u: torch.Tensor, v: torch.Tensor, scale: float, offset: float) -> torch.Tensor:
        x = u * scale
        y = v * scale
        x0 = torch.floor(x)
        y0 = torch.floor(y)
        x1 = x0 + 1.0
        y1 = y0 + 1.0

        fx = x - x0
        fy = y - y0
        sx = fx * fx * (3.0 - 2.0 * fx)
        sy = fy * fy * (3.0 - 2.0 * fy)

        period_x = max(1.0, float(round(scale)))
        n00 = self._hash2(x0, y0, offset + 0.1, period_x=period_x)
        n10 = self._hash2(x1, y0, offset + 0.2, period_x=period_x)
        n01 = self._hash2(x0, y1, offset + 0.3, period_x=period_x)
        n11 = self._hash2(x1, y1, offset + 0.4, period_x=period_x)

        nx0 = self._lerp(n00, n10, sx)
        nx1 = self._lerp(n01, n11, sx)
        return self._lerp(nx0, nx1, sy)

    def _fbm(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        base_scale: float,
        octaves: int,
        lacunarity: float,
        gain: float,
        offset: float,
    ) -> torch.Tensor:
        value = torch.zeros_like(u)
        amplitude = 1.0
        frequency = base_scale
        norm = 0.0

        for octave in range(octaves):
            value = value + amplitude * self._value_noise(u, v, frequency, offset + 19.3 * octave)
            norm += amplitude
            amplitude *= gain
            frequency *= lacunarity

        return value / max(norm, 1.0e-6)

    def _star_layer(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        grid_u: float,
        grid_v: float,
        density: float,
        brightness: float,
        offset: float,
    ) -> torch.Tensor:
        ux = u * grid_u
        vy = v * grid_v
        cell_u = torch.floor(ux)
        cell_v = torch.floor(vy)
        local_u = ux - cell_u
        local_v = vy - cell_v

        density_clamped = max(0.0, min(0.25, density))
        period_x = max(1.0, float(round(grid_u)))
        presence = self._hash2(cell_u, cell_v, offset + 0.6, period_x=period_x) < density_clamped
        star_u = self._hash2(cell_u, cell_v, offset + 4.1, period_x=period_x)
        star_v = self._hash2(cell_u, cell_v, offset + 8.6, period_x=period_x)
        dx = local_u - star_u
        dy = local_v - star_v
        dist2 = dx * dx + dy * dy

        sharp = 140.0 + 300.0 * self._hash2(cell_u, cell_v, offset + 12.7, period_x=period_x)
        core = torch.exp(-sharp * dist2)
        halo = torch.exp(-(18.0 + 40.0 * self._hash2(cell_u, cell_v, offset + 18.2, period_x=period_x)) * torch.sqrt(dist2 + 1.0e-9))
        spikes = torch.exp(-32.0 * torch.abs(dx * dy))

        mag = torch.pow(self._hash2(cell_u, cell_v, offset + 24.9, period_x=period_x), 6.0)
        power = presence.to(self.dtype) * (core + 0.25 * halo + 0.20 * spikes) * (0.12 + brightness * (0.25 + 4.2 * mag))

        temp = self._hash2(cell_u, cell_v, offset + 33.4, period_x=period_x)
        color = torch.stack(
            [
                torch.clamp(0.62 + 0.48 * temp, min=0.0, max=1.3),
                torch.clamp(0.68 + 0.24 * (1.0 - torch.abs(temp - 0.45)), min=0.0, max=1.2),
                torch.clamp(1.10 - 0.35 * temp, min=0.0, max=1.3),
            ],
            dim=-1,
        )

        return color * power.unsqueeze(-1)

    def _plasma_palette(self, t: torch.Tensor) -> torch.Tensor:
        t = torch.clamp(t, min=0.0, max=1.0)
        c0 = torch.tensor([0.08, 0.00, 0.01], dtype=self.dtype, device=self.device)
        c1 = torch.tensor([0.45, 0.03, 0.01], dtype=self.dtype, device=self.device)
        c2 = torch.tensor([0.92, 0.28, 0.02], dtype=self.dtype, device=self.device)
        c3 = torch.tensor([1.00, 0.74, 0.08], dtype=self.dtype, device=self.device)
        c4 = torch.tensor([1.00, 0.97, 0.82], dtype=self.dtype, device=self.device)

        color = torch.zeros((t.shape[0], 3), dtype=self.dtype, device=self.device)

        m0 = t < 0.25
        s0 = torch.clamp(t / 0.25, min=0.0, max=1.0).unsqueeze(-1)
        c01 = c0 + (c1 - c0) * s0
        color = torch.where(m0.unsqueeze(-1), c01, color)

        m1 = (t >= 0.25) & (t < 0.55)
        s1 = torch.clamp((t - 0.25) / 0.30, min=0.0, max=1.0).unsqueeze(-1)
        c12 = c1 + (c2 - c1) * s1
        color = torch.where(m1.unsqueeze(-1), c12, color)

        m2 = (t >= 0.55) & (t < 0.82)
        s2 = torch.clamp((t - 0.55) / 0.27, min=0.0, max=1.0).unsqueeze(-1)
        c23 = c2 + (c3 - c2) * s2
        color = torch.where(m2.unsqueeze(-1), c23, color)

        m3 = t >= 0.82
        s3 = torch.clamp((t - 0.82) / 0.18, min=0.0, max=1.0).unsqueeze(-1)
        c34 = c3 + (c4 - c3) * s3
        color = torch.where(m3.unsqueeze(-1), c34, color)
        return color

    def _interstellar_warm_palette(self, t: torch.Tensor) -> torch.Tensor:
        t = torch.clamp(t, min=0.0, max=1.0)
        c0 = torch.tensor([1.00, 0.95, 0.82], dtype=self.dtype, device=self.device)
        c1 = torch.tensor([1.00, 0.82, 0.42], dtype=self.dtype, device=self.device)
        c2 = torch.tensor([0.97, 0.56, 0.20], dtype=self.dtype, device=self.device)
        c3 = torch.tensor([0.78, 0.28, 0.10], dtype=self.dtype, device=self.device)
        c4 = torch.tensor([0.42, 0.08, 0.03], dtype=self.dtype, device=self.device)

        color = torch.zeros((t.shape[0], 3), dtype=self.dtype, device=self.device)

        m0 = t < 0.20
        s0 = torch.clamp(t / 0.20, min=0.0, max=1.0).unsqueeze(-1)
        c01 = c0 + (c1 - c0) * s0
        color = torch.where(m0.unsqueeze(-1), c01, color)

        m1 = (t >= 0.20) & (t < 0.46)
        s1 = torch.clamp((t - 0.20) / 0.26, min=0.0, max=1.0).unsqueeze(-1)
        c12 = c1 + (c2 - c1) * s1
        color = torch.where(m1.unsqueeze(-1), c12, color)

        m2 = (t >= 0.46) & (t < 0.74)
        s2 = torch.clamp((t - 0.46) / 0.28, min=0.0, max=1.0).unsqueeze(-1)
        c23 = c2 + (c3 - c2) * s2
        color = torch.where(m2.unsqueeze(-1), c23, color)

        m3 = t >= 0.74
        s3 = torch.clamp((t - 0.74) / 0.26, min=0.0, max=1.0).unsqueeze(-1)
        c34 = c3 + (c4 - c3) * s3
        color = torch.where(m3.unsqueeze(-1), c34, color)
        return torch.clamp(color, min=0.0, max=1.2)

    def _disk_layered_palette_color(
        self,
        x: torch.Tensor,
        phi_disk: torch.Tensor,
        t_disk: torch.Tensor,
        layer_complexity: torch.Tensor | None = None,
        omega_layer: torch.Tensor | None = None,
        layer_key: torch.Tensor | None = None,
    ) -> torch.Tensor:
        cfg = self.config
        n_layers = max(2, int(cfg.disk_layer_count))
        if cfg.disk_adaptive_stratification:
            lmin = max(2, int(cfg.disk_adaptive_layers_min))
            lmax = max(lmin, int(cfg.disk_adaptive_layers_max))
            if layer_complexity is None:
                layer_complexity = torch.pow(torch.clamp(1.0 - x, min=0.0, max=1.0), 0.70)
            layer_complexity = torch.clamp(layer_complexity, min=0.0, max=1.0)
            mix = torch.as_tensor(cfg.disk_adaptive_complexity_mix, dtype=self.dtype, device=self.device)
            radial_ref = torch.pow(torch.clamp(1.0 - x, min=0.0, max=1.0), 0.70)
            c_eff = torch.clamp(mix * layer_complexity + (1.0 - mix) * radial_ref, min=0.0, max=1.0)
            n_layers_t = torch.clamp(
                torch.as_tensor(float(lmin), dtype=self.dtype, device=self.device)
                + torch.as_tensor(float(lmax - lmin), dtype=self.dtype, device=self.device) * c_eff,
                min=float(lmin),
                max=float(lmax),
            )
        else:
            n_layers_t = torch.full_like(x, float(n_layers))
        rin = torch.as_tensor(cfg.disk_inner_radius, dtype=self.dtype, device=self.device)
        span = torch.as_tensor(
            max(1.0e-6, cfg.disk_outer_radius - cfg.disk_inner_radius),
            dtype=self.dtype,
            device=self.device,
        )

        layer_pos = torch.clamp(x * n_layers_t, min=0.0)
        layer_pos = torch.minimum(layer_pos, n_layers_t - 1.0e-6)
        layer_idx = torch.floor(layer_pos)
        layer_idx_next = torch.minimum(layer_idx + 1.0, n_layers_t - 1.0)
        layer_frac = torch.clamp(layer_pos - layer_idx, min=0.0, max=1.0)

        u0 = (layer_idx + 0.5) / n_layers_t
        u1 = (layer_idx_next + 0.5) / n_layers_t
        c0 = self._interstellar_warm_palette(u0)
        c1 = self._interstellar_warm_palette(u1)
        base_color = c0 + (c1 - c0) * layer_frac.unsqueeze(-1)

        layer_r = rin + u0 * span
        if omega_layer is None:
            omega_layer = (
                self._disk_diffrot_omega(layer_r)
                if cfg.enable_disk_differential_rotation
                else self._disk_keplerian_omega(layer_r)
            )
        if layer_key is None:
            layer_key = layer_idx
        flow_phi = (
            phi_disk
            - omega_layer
            * t_disk
            * torch.as_tensor(cfg.disk_layer_time_scale, dtype=self.dtype, device=self.device)
            - torch.as_tensor(cfg.disk_layer_global_phase, dtype=self.dtype, device=self.device)
        )
        if cfg.enable_disk_differential_rotation:
            flow_phi, tile_gain = self._disk_diffrot_phase_terms(flow_phi, layer_key, u0)
        else:
            tile_gain = torch.ones_like(flow_phi)
        phase = (
            torch.as_tensor(cfg.disk_layer_pattern_count, dtype=self.dtype, device=self.device) * flow_phi
            + (2.0 * math.pi) * layer_idx / n_layers_t
        )
        wave = 0.5 + 0.5 * torch.sin(phase)
        wave2 = 0.5 + 0.5 * torch.sin(0.73 * phase + 8.0 * u0)
        contrast = torch.as_tensor(cfg.disk_layer_pattern_contrast, dtype=self.dtype, device=self.device)
        band = (1.0 - contrast) + contrast * (0.28 + 0.72 * wave * (0.82 + 0.18 * wave2))
        band = band * tile_gain

        # Add coherent rotating inhomogeneities per layer as azimuthal
        # rectangular segments, so disk rotation is readable in animation.
        accident_strength = torch.as_tensor(cfg.disk_layer_accident_strength, dtype=self.dtype, device=self.device)
        if float(cfg.disk_layer_accident_strength) > 0.0:
            accident_count = torch.as_tensor(cfg.disk_layer_accident_count, dtype=self.dtype, device=self.device)
            accident_sharp = torch.as_tensor(cfg.disk_layer_accident_sharpness, dtype=self.dtype, device=self.device)
            inner_weight = torch.pow(torch.clamp(1.0 - u0, min=0.0, max=1.0), 0.35)
            seg_count = torch.clamp(2.0 + 4.0 * accident_count, min=2.0, max=512.0)
            phi_unit = torch.remainder(flow_phi / (2.0 * math.pi), 1.0)
            seg_pos = phi_unit * seg_count
            seg_idx = torch.floor(seg_pos)
            seg_frac = seg_pos - seg_idx

            edge = torch.clamp(0.10 / torch.clamp(accident_sharp, min=1.0), min=0.004, max=0.08)
            duty = torch.clamp(0.78 - 0.055 * torch.clamp(accident_sharp - 1.0, min=0.0), min=0.14, max=0.88)
            rise = torch.clamp(seg_frac / torch.clamp(edge, min=1.0e-4), min=0.0, max=1.0)
            fall = torch.clamp((duty - seg_frac) / torch.clamp(edge, min=1.0e-4), min=0.0, max=1.0)
            in_rect = torch.where(seg_frac < duty, torch.minimum(rise, fall), torch.zeros_like(seg_frac))

            seg_hash_a = torch.frac(
                torch.sin((layer_idx + 1.0) * 127.1 + (seg_idx + 1.0) * 311.7 + 17.0 * u0) * 43758.5453
            )
            seg_hash_b = torch.frac(
                torch.sin((layer_idx + 1.0) * 269.5 + (seg_idx + 1.0) * 183.3 + 11.0 * u0) * 24634.6345
            )
            accident_profile = torch.clamp(in_rect * (0.35 + 0.65 * seg_hash_a), min=0.0, max=1.0)
            accident_boost = torch.clamp(
                1.0 + accident_strength * inner_weight * (1.60 * accident_profile - 0.30),
                min=0.45,
                max=3.2,
            )
            band = band * accident_boost
            tint_cool = torch.tensor([0.88, 0.97, 1.20], dtype=self.dtype, device=self.device).view(1, 3)
            tint_warm = torch.tensor([1.22, 1.08, 0.86], dtype=self.dtype, device=self.device).view(1, 3)
            accident_tint = tint_cool + (tint_warm - tint_cool) * seg_hash_b.unsqueeze(-1)
        else:
            accident_profile = torch.zeros_like(band)
            inner_weight = torch.ones_like(band)
            accident_tint = torch.ones((band.shape[0], 3), dtype=self.dtype, device=self.device)

        odd = torch.remainder(layer_idx, 2.0)
        tint_shift = (odd * 2.0 - 1.0) * 0.03
        tint = torch.stack(
            [
                torch.clamp(1.0 + 0.5 * tint_shift, min=0.85, max=1.15),
                torch.clamp(1.0 - 0.2 * tint_shift, min=0.85, max=1.15),
                torch.clamp(1.0 - 0.8 * tint_shift, min=0.80, max=1.20),
            ],
            dim=-1,
        )

        layer_color = base_color * tint * band.unsqueeze(-1)
        accident_mix = (0.28 * accident_strength * inner_weight * accident_profile).unsqueeze(-1)
        layer_color = layer_color * (1.0 + accident_mix * (accident_tint - 1.0))
        inner_hot = torch.exp(-torch.pow(torch.clamp(u0 / 0.18, min=0.0), 2.0)).unsqueeze(-1)
        hot_tint = torch.tensor([1.00, 0.92, 0.70], dtype=self.dtype, device=self.device).view(1, 3)
        layer_color = layer_color + 0.20 * inner_hot * hot_tint
        return torch.clamp(layer_color, min=0.0, max=1.5)

    def _hsv_to_rgb(
        self,
        h: torch.Tensor,
        s: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """Vectorized HSV → RGB. All inputs shape (n,) in [0, 1]. Returns (n, 3)."""
        h6 = torch.clamp(h, min=0.0, max=1.0 - 1.0e-7) * 6.0
        i = torch.floor(h6).long() % 6
        f = h6 - torch.floor(h6)
        p = v * (1.0 - s)
        q = v * (1.0 - f * s)
        tv = v * (1.0 - (1.0 - f) * s)
        r = torch.where(i == 0, v,  torch.where(i == 1, q,  torch.where(i == 2, p,  torch.where(i == 3, p,  torch.where(i == 4, tv, v)))))
        g = torch.where(i == 0, tv, torch.where(i == 1, v,  torch.where(i == 2, v,  torch.where(i == 3, q,  torch.where(i == 4, p,  p)))))
        b = torch.where(i == 0, p,  torch.where(i == 1, p,  torch.where(i == 2, tv, torch.where(i == 3, v,  torch.where(i == 4, v,  q)))))
        return torch.stack([r, g, b], dim=-1)

    def _disk_segmented_color(
        self,
        x: torch.Tensor,
        phi_disk: torch.Tensor,
        t_disk: torch.Tensor,
        omega: torch.Tensor,
    ) -> torch.Tensor:
        """
        Color each disk ray according to which (ring, sector) cell it belongs to
        in the Keplerian co-rotating frame, blending neighbouring cells with a
        Gaussian kernel so boundaries fade smoothly.

        Ring index increases outward; hue rotates continuously around azimuth so
        the result looks like a color-wheel split into concentric annuli.
        """
        cfg = self.config
        n_rings = max(1, int(cfg.disk_segmented_rings))
        n_sectors = max(2, int(cfg.disk_segmented_sectors))
        sigma = max(0.05, float(cfg.disk_segmented_sigma))
        hue_offset = float(cfg.disk_segmented_hue_offset)
        pi2 = 2.0 * math.pi

        # Phase in the Keplerian co-rotating frame, wrapped to [0, 2π).
        # Subtracting disk_layer_global_phase allows the animation system to
        # advance the apparent rotation via --disk-layer-phase-rate-hz.
        global_phase = torch.as_tensor(cfg.disk_layer_global_phase, dtype=self.dtype, device=self.device)
        phase = torch.remainder(phi_disk - omega * t_disk - global_phase, pi2)

        # Floating-point coordinates in (ring, sector) cell space
        r_float = torch.clamp(x, min=0.0, max=1.0 - 1.0e-7) * float(n_rings)
        s_float = phase / pi2 * float(n_sectors)

        n_rings_f = float(n_rings)
        n_sectors_f = float(n_sectors)
        K = max(1, int(math.ceil(2.5 * sigma)))

        color_acc = torch.zeros((x.shape[0], 3), dtype=self.dtype, device=self.device)
        weight_acc = torch.zeros(x.shape[0], dtype=self.dtype, device=self.device)

        for dr in range(-K, K + 1):
            for ds in range(-K, K + 1):
                # Centre of this neighbour cell
                ring_j = torch.floor(r_float) + float(dr)
                sec_j = torch.floor(s_float) + float(ds)
                ring_center = ring_j + 0.5
                sec_center = sec_j + 0.5

                # Radial distance
                d_ring = r_float - ring_center

                # Azimuthal distance (shortest arc, accounting for wrap)
                d_sec_raw = s_float - sec_center
                d_sec = d_sec_raw - torch.round(d_sec_raw / n_sectors_f) * n_sectors_f

                w = torch.exp(-0.5 * (d_ring * d_ring + d_sec * d_sec) / (sigma * sigma))

                # Out-of-range rings contribute nothing; sectors wrap
                w = torch.where((ring_j >= 0.0) & (ring_j < n_rings_f), w, torch.zeros_like(w))

                ring_idx = torch.clamp(ring_j, min=0.0, max=n_rings_f - 1.0)
                sec_idx = torch.remainder(sec_j, n_sectors_f)

                # Normalised ring coordinate [0 = inner, 1 = outer]
                ring_norm = ring_idx / max(1.0, n_rings_f - 1.0)

                palette_mode = cfg.disk_segmented_palette_mode
                if palette_mode == "accretion_warm":
                    # Build deterministic per-cell jitter in warm tones.
                    # Using both ring+sector avoids long coherent bands where a
                    # single azimuthal sector dominates the full disk.
                    cell_id = ring_idx * n_sectors_f + sec_idx
                    rnd_a = torch.sin(cell_id * 12.9898 + 78.233) * 43758.5453
                    rnd_b = torch.sin(cell_id * 19.9137 + 11.135) * 24634.6345
                    rnd_c = torch.sin(cell_id * 7.1231 + 93.531) * 35412.2381
                    rnd_a = rnd_a - torch.floor(rnd_a)
                    rnd_b = rnd_b - torch.floor(rnd_b)
                    rnd_c = rnd_c - torch.floor(rnd_c)

                    # Warm hue span only (red -> orange -> yellow), but sampled
                    # per-cell instead of sweeping linearly with azimuth.
                    hue = torch.remainder(hue_offset + 0.13 * rnd_a, 1.0)

                    # Keep a broad thermal trend with radius while adding
                    # per-cell variance so colors stay patchy and irregular.
                    sat_base = 0.86 + 0.12 * ring_norm
                    val_base = 0.84 - 0.34 * ring_norm
                    saturation = torch.clamp(sat_base + 0.16 * (rnd_b - 0.5), min=0.45, max=1.0)
                    value = torch.clamp(val_base + 0.30 * (rnd_c - 0.5), min=0.18, max=1.0)
                else:
                    # rainbow: full hue wheel, pastel outward
                    hue = torch.remainder(sec_idx / n_sectors_f + hue_offset, 1.0)
                    saturation = torch.clamp(0.82 - 0.34 * ring_norm, min=0.1, max=1.0)
                    value = torch.clamp(0.76 + 0.20 * ring_norm, min=0.1, max=1.0)

                cell_rgb = self._hsv_to_rgb(hue, saturation, value)
                color_acc = color_acc + w.unsqueeze(-1) * cell_rgb
                weight_acc = weight_acc + w

        return torch.clamp(
            color_acc / torch.clamp(weight_acc, min=1.0e-8).unsqueeze(-1),
            min=0.0,
            max=1.5,
        )

    def _disk_diffrot_iteration_params(self) -> tuple[float, float, float]:
        """
        Return iteration tuning knobs as:
        - phase offset scale
        - visual tile mix
        - tile smoothing / sharpness
        """
        iteration = self.config.disk_diffrot_iteration
        if iteration == "v1_basic":
            return 0.0, 0.0, 1.0
        if iteration == "v2_visibility":
            return 0.58, 0.34, 1.2
        return 0.36, 0.22, 0.72

    def _disk_diffrot_phase_offset(self, layer_key: torch.Tensor, radial_key: torch.Tensor) -> torch.Tensor:
        seed = torch.as_tensor(float(self.config.disk_diffrot_seed), dtype=self.dtype, device=self.device)
        raw = torch.sin(
            (layer_key + 1.0) * 12.9898
            + (radial_key + 1.0) * 78.233
            + seed * 37.719
            + 0.5
        ) * 43758.5453
        return (self._fract(raw) - 0.5) * (2.0 * math.pi)

    def _disk_diffrot_phase_terms(
        self,
        phase: torch.Tensor,
        layer_key: torch.Tensor,
        radial_key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.config
        if (not cfg.enable_disk_differential_rotation) or cfg.disk_diffrot_strength <= 0.0:
            return phase, torch.ones_like(phase)

        phase_offset_scale, tile_mix, tile_sharpness = self._disk_diffrot_iteration_params()
        if phase_offset_scale > 0.0:
            phase = torch.remainder(
                phase + phase_offset_scale * self._disk_diffrot_phase_offset(layer_key, radial_key),
                2.0 * math.pi,
            )

        if cfg.disk_diffrot_visual_mode == "layer_phase" or tile_mix <= 0.0:
            return phase, torch.ones_like(phase)

        strength = torch.as_tensor(cfg.disk_diffrot_strength, dtype=self.dtype, device=self.device)
        tile_drive = phase * (1.0 + 0.15 * strength)
        tile_drive = tile_drive + (layer_key + 1.0) * 0.37 + radial_key * 3.1
        tile_wave = 0.5 + 0.5 * torch.sin(tile_drive)

        if cfg.disk_diffrot_visual_mode == "hybrid":
            fine_wave = 0.5 + 0.5 * torch.sin(1.73 * tile_drive + 0.5 * phase)
            tile_wave = 0.62 * tile_wave + 0.38 * fine_wave

        if cfg.disk_diffrot_iteration == "v3_robust":
            sharp = torch.as_tensor(tile_sharpness, dtype=self.dtype, device=self.device)
            tile_wave = 0.5 + 0.5 * torch.tanh((tile_wave - 0.5) * (1.0 + sharp))

        tile_gain = torch.clamp((1.0 - tile_mix) + tile_mix * (0.82 + 0.36 * tile_wave), min=0.55, max=1.45)
        return phase, tile_gain

    def _disk_diffrot_omega(self, r: torch.Tensor, metric: object | None = None) -> torch.Tensor:
        cfg = self.config
        if cfg.disk_diffrot_model == "keplerian_metric":
            omega = self._metric_keplerian_omega(r, metric=metric)
        else:
            omega = self._disk_keplerian_omega(r)
        if cfg.enable_disk_differential_rotation:
            omega = omega * torch.as_tensor(cfg.disk_diffrot_strength, dtype=self.dtype, device=self.device)
        return omega

    def _blackbody_rgb(self, temperature_kelvin: torch.Tensor) -> torch.Tensor:
        t = torch.clamp(temperature_kelvin, min=1000.0, max=40000.0) / 100.0

        ln_t = torch.log(torch.clamp(t, min=1.0))
        red_low = torch.ones_like(t)
        red_high = 1.292936186062745 * torch.pow(torch.clamp(t - 60.0, min=1.0e-4), -0.1332047592)
        red = torch.where(t <= 66.0, red_low, red_high)

        green_low = 0.3900815787690196 * ln_t - 0.6318414437886275
        green_high = 1.129890860895294 * torch.pow(torch.clamp(t - 60.0, min=1.0e-4), -0.0755148492)
        green = torch.where(t <= 66.0, green_low, green_high)

        blue_low = torch.zeros_like(t)
        blue_mid = 0.5432067891101961 * torch.log(torch.clamp(t - 10.0, min=1.0e-4)) - 1.19625408914
        blue = torch.where(t >= 66.0, torch.ones_like(t), torch.where(t <= 19.0, blue_low, blue_mid))

        rgb = torch.stack([red, green, blue], dim=-1)
        return torch.clamp(rgb, min=0.0, max=1.0)

    def _novikov_thorne_flux_profile_proxy(self, r: torch.Tensor, rin: torch.Tensor) -> torch.Tensor:
        """
        Fast Novikov-Thorne-inspired proxy.

        This fallback is kept for robustness when the full metric-aware
        thin-disk integral becomes numerically ill-conditioned.
        """
        r_safe = torch.clamp(r, min=rin * 1.0005)
        a = torch.as_tensor(self.metric_spin, dtype=self.dtype, device=self.device)
        r2 = r_safe * r_safe
        r3 = r2 * r_safe
        r32 = torch.clamp(r_safe * torch.sqrt(torch.clamp(r_safe, min=1.0e-8)), min=1.0e-8)

        a2 = a * a
        A = 1.0 - 2.0 / torch.clamp(r_safe, min=1.0e-8) + a2 / torch.clamp(r2, min=1.0e-8)
        B = 1.0 - 3.0 / torch.clamp(r_safe, min=1.0e-8) + 2.0 * a / r32
        C = 1.0 - 4.0 * a / r32 + 3.0 * a2 / torch.clamp(r2, min=1.0e-8)

        zero_torque = torch.clamp(1.0 - torch.sqrt(torch.clamp(rin / r_safe, max=1.0)), min=0.0)
        relativistic = torch.sqrt(torch.clamp(C / torch.clamp(A, min=1.0e-6), min=0.02, max=50.0))
        flux = zero_torque * relativistic / torch.clamp(r3 * torch.clamp(B, min=0.02), min=1.0e-8)
        return torch.clamp(flux, min=0.0)

    def _finite_diff_1d(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        dy = torch.zeros_like(y)
        if y.numel() < 2:
            return dy
        dx = torch.clamp(x[1:] - x[:-1], min=1.0e-9)
        dy[1:-1] = (y[2:] - y[:-2]) / torch.clamp(x[2:] - x[:-2], min=1.0e-9)
        dy[0] = (y[1] - y[0]) / dx[0]
        dy[-1] = (y[-1] - y[-2]) / dx[-1]
        return torch.nan_to_num(dy, nan=0.0, posinf=0.0, neginf=0.0)

    def _sample_curve_lut(self, r: torch.Tensor, rr_lut: torch.Tensor, value_lut: torch.Tensor) -> torch.Tensor:
        idx_hi = torch.searchsorted(rr_lut, r, right=False)
        idx_hi = torch.clamp(idx_hi, min=1, max=rr_lut.shape[0] - 1)
        idx_lo = idx_hi - 1
        r_lo = rr_lut[idx_lo]
        r_hi = rr_lut[idx_hi]
        f_lo = value_lut[idx_lo]
        f_hi = value_lut[idx_hi]
        t = (r - r_lo) / torch.clamp(r_hi - r_lo, min=1.0e-8)
        return f_lo + (f_hi - f_lo) * t

    def _sample_flux_lut(self, r: torch.Tensor, rr_lut: torch.Tensor, flux_lut: torch.Tensor) -> torch.Tensor:
        return torch.clamp(self._sample_curve_lut(r, rr_lut, flux_lut), min=0.0)

    def _metric_keplerian_omega(
        self,
        rr: torch.Tensor,
        metric: object | None = None,
    ) -> torch.Tensor:
        if metric is None:
            theta = torch.full_like(rr, math.pi * 0.5)
            metric = metric_components(
                rr,
                theta,
                self.config.spin,
                self.config.metric_model,
                self.config.charge,
                self.config.cosmological_constant,
                self.config.wormhole_throat_radius,
                self.config.wormhole_length_scale,
                self.config.wormhole_lensing_scale,
            )
        gtt = metric.g_tt
        gtphi = metric.g_tphi
        gphiphi = metric.g_phiphi

        dgtt = self._finite_diff_1d(gtt, rr)
        dgtphi = self._finite_diff_1d(gtphi, rr)
        dgphiphi = self._finite_diff_1d(gphiphi, rr)

        disc = torch.clamp(dgtphi * dgtphi - dgtt * dgphiphi, min=0.0)
        sqrt_disc = torch.sqrt(disc)
        den_om = torch.where(
            dgphiphi >= 0.0,
            torch.clamp(dgphiphi, min=1.0e-9),
            torch.clamp(dgphiphi, max=-1.0e-9),
        )
        omega_p = (-dgtphi + sqrt_disc) / den_om
        omega_m = (-dgtphi - sqrt_disc) / den_om

        # Prograde branch selection with timelike-condition filtering.
        guess = torch.sqrt(1.0 / torch.clamp(rr * rr * rr, min=1.0e-9))
        denom_p = -(gtt + 2.0 * gtphi * omega_p + gphiphi * omega_p * omega_p)
        denom_m = -(gtt + 2.0 * gtphi * omega_m + gphiphi * omega_m * omega_m)
        valid_p = denom_p > 1.0e-10
        valid_m = denom_m > 1.0e-10
        err_p = torch.abs(omega_p - guess)
        err_m = torch.abs(omega_m - guess)
        prefer_p = (err_p <= err_m) | (~valid_m)
        omega = torch.where(prefer_p, omega_p, omega_m)
        omega = torch.where(valid_p | valid_m, omega, guess)

        den_fb = torch.pow(torch.clamp(rr, min=1.0e-8), 1.5) + torch.as_tensor(
            self.metric_spin, dtype=self.dtype, device=self.device
        )
        den_fb = torch.where(den_fb >= 0.0, torch.clamp(den_fb, min=1.0e-8), torch.clamp(den_fb, max=-1.0e-8))
        fallback = 1.0 / den_fb
        omega = torch.where(torch.isfinite(omega), omega, fallback)
        return torch.nan_to_num(omega, nan=0.0, posinf=0.0, neginf=0.0)

    def _build_keplerian_omega_lut(self, rin: float, rmax: float) -> tuple[torch.Tensor, torch.Tensor]:
        rin_safe = max(1.0e-4, float(rin) * 1.0005)
        rmax_safe = max(rin_safe + 1.0e-3, float(rmax))
        span = rmax_safe - rin_safe
        samples = max(512, min(4096, int(512 + 64.0 * span)))
        key = (
            "omega_keplerian",
            str(self.device),
            str(self.dtype),
            self.config.metric_model,
            round(float(self.config.spin), 8),
            round(float(self.config.charge), 8),
            round(float(self.config.cosmological_constant), 12),
            round(rin_safe, 6),
            round(rmax_safe, 3),
            int(samples),
        )
        cached = KerrRayTracer._keplerian_omega_cache.get(key)
        if cached is not None:
            return cached

        disk_key = (
            self.config.metric_model,
            round(float(self.config.spin), 8),
            round(float(self.config.charge), 8),
            round(float(self.config.cosmological_constant), 12),
            round(rin_safe, 6),
            round(rmax_safe, 3),
            int(samples),
        )
        persisted = self._persistent_load_lut_pair("omega_keplerian", disk_key)
        if persisted is not None:
            KerrRayTracer._keplerian_omega_cache[key] = persisted
            return persisted

        rr = torch.linspace(rin_safe, rmax_safe, samples, dtype=self.dtype, device=self.device)
        omega = self._metric_keplerian_omega(rr)
        omega = torch.nan_to_num(omega, nan=0.0, posinf=0.0, neginf=0.0)
        if not bool((torch.abs(omega) > 0.0).any()):
            den = torch.pow(torch.clamp(rr, min=1.0e-8), 1.5) + torch.as_tensor(
                self.metric_spin, dtype=self.dtype, device=self.device
            )
            den = torch.where(den >= 0.0, torch.clamp(den, min=1.0e-8), torch.clamp(den, max=-1.0e-8))
            omega = 1.0 / den

        KerrRayTracer._keplerian_omega_cache[key] = (rr, omega)
        self._persistent_save_lut_pair("omega_keplerian", disk_key, rr, omega)
        if len(KerrRayTracer._keplerian_omega_cache) > 24:
            KerrRayTracer._keplerian_omega_cache.pop(next(iter(KerrRayTracer._keplerian_omega_cache)))
        return rr, omega

    def _disk_keplerian_omega(self, r: torch.Tensor) -> torch.Tensor:
        if self._disk_omega_lut_main is not None:
            rr_lut, omega_lut = self._disk_omega_lut_main
            r_sample = torch.clamp(r, min=rr_lut[0], max=rr_lut[-1])
            omega = self._sample_curve_lut(r_sample, rr_lut, omega_lut)
            return torch.nan_to_num(omega, nan=0.0, posinf=0.0, neginf=0.0)
        den = torch.pow(torch.clamp(r, min=1.0e-8), 1.5) + torch.as_tensor(self.metric_spin, dtype=self.dtype, device=self.device)
        den = torch.where(den >= 0.0, torch.clamp(den, min=1.0e-8), torch.clamp(den, max=-1.0e-8))
        return 1.0 / den

    def _build_nt_general_lut(self, rin: float, rmax: float) -> tuple[torch.Tensor, torch.Tensor]:
        rin_safe = max(1.0e-4, float(rin) * 1.0005)
        rmax_safe = max(rin_safe + 1.0e-3, float(rmax))
        span = rmax_safe - rin_safe
        samples = max(512, min(4096, int(512 + 64.0 * span)))
        key = (
            "nt_general",
            str(self.device),
            str(self.dtype),
            self.config.metric_model,
            round(float(self.config.spin), 8),
            round(float(self.config.charge), 8),
            round(float(self.config.cosmological_constant), 12),
            round(rin_safe, 6),
            round(rmax_safe, 3),
            int(samples),
        )
        cached = KerrRayTracer._nt_page_thorne_cache.get(key)
        if cached is not None:
            return cached

        disk_key = (
            self.config.metric_model,
            round(float(self.config.spin), 8),
            round(float(self.config.charge), 8),
            round(float(self.config.cosmological_constant), 12),
            round(rin_safe, 6),
            round(rmax_safe, 3),
            int(samples),
        )
        persisted = self._persistent_load_lut_pair("nt_general", disk_key)
        if persisted is not None:
            KerrRayTracer._nt_page_thorne_cache[key] = persisted
            return persisted

        rr = torch.linspace(rin_safe, rmax_safe, samples, dtype=self.dtype, device=self.device)
        theta = torch.full_like(rr, math.pi * 0.5)
        metric = metric_components(
            rr,
            theta,
            self.config.spin,
            self.config.metric_model,
            self.config.charge,
            self.config.cosmological_constant,
            self.config.wormhole_throat_radius,
            self.config.wormhole_length_scale,
        )
        gtt = metric.g_tt
        gtphi = metric.g_tphi
        gphiphi = metric.g_phiphi

        omega = self._metric_keplerian_omega(rr, metric=metric)

        denom_ut = -(gtt + 2.0 * gtphi * omega + gphiphi * omega * omega)
        ut = torch.rsqrt(torch.clamp(denom_ut, min=1.0e-10))
        energy = -(gtt + gtphi * omega) * ut
        ang_mom = (gtphi + gphiphi * omega) * ut

        d_omega_dr = self._finite_diff_1d(omega, rr)
        dL_dr = self._finite_diff_1d(ang_mom, rr)
        e_minus_ol = energy - omega * ang_mom

        dr = rr[1:] - rr[:-1]
        integrand = e_minus_ol * dL_dr
        trapezoids = 0.5 * (integrand[1:] + integrand[:-1]) * dr
        integral = torch.zeros_like(rr)
        integral[1:] = torch.cumsum(trapezoids, dim=0)

        geom = torch.clamp(rr * rr, min=1.0e-8)
        flux = -d_omega_dr * integral / torch.clamp(torch.square(e_minus_ol) * geom, min=1.0e-8)
        flux = torch.nan_to_num(flux, nan=0.0, posinf=0.0, neginf=0.0)
        flux = torch.clamp(flux, min=0.0)
        flux[0] = 0.0

        if not bool((flux > 0.0).any()):
            flux = self._novikov_thorne_flux_profile_proxy(rr, torch.as_tensor(rin_safe, dtype=self.dtype, device=self.device))

        KerrRayTracer._nt_page_thorne_cache[key] = (rr, flux)
        self._persistent_save_lut_pair("nt_general", disk_key, rr, flux)
        if len(KerrRayTracer._nt_page_thorne_cache) > 24:
            KerrRayTracer._nt_page_thorne_cache.pop(next(iter(KerrRayTracer._nt_page_thorne_cache)))
        return rr, flux

    def _novikov_thorne_flux_profile(self, r: torch.Tensor, rin: torch.Tensor) -> torch.Tensor:
        """
        Metric-aware Novikov-Thorne thin-disk flux profile.

        Computes the relativistic Page-Thorne-style integral using local
        circular geodesic quantities derived from the selected metric
        (`a`, `Q`, `Lambda`) and falls back to a stable proxy if needed.
        """
        r_safe = torch.clamp(r, min=rin * 1.0005)
        rin_val = float(torch.clamp(rin, min=1.0e-4).item())
        rmax_val = float(torch.max(r_safe).item())
        try:
            rr_lut, flux_lut = self._build_nt_general_lut(rin=rin_val, rmax=rmax_val)
            flux = self._sample_flux_lut(r_safe, rr_lut, flux_lut)
            return torch.clamp(flux, min=0.0)
        except Exception:
            return self._novikov_thorne_flux_profile_proxy(r_safe, rin)

    def _kerr_circular_orbit_quantities(self, r: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        r_safe = torch.clamp(r, min=1.0e-6)
        a = torch.as_tensor(self.metric_spin, dtype=self.dtype, device=self.device)
        sqrt_r = torch.sqrt(r_safe)
        r32 = torch.clamp(r_safe * sqrt_r, min=1.0e-8)
        r34 = torch.clamp(torch.pow(r_safe, 0.75), min=1.0e-8)
        denom = torch.clamp(r32 - 3.0 * sqrt_r + 2.0 * a, min=1.0e-8)
        sqrt_denom = torch.sqrt(denom)

        omega = 1.0 / torch.clamp(r32 + a, min=1.0e-8)
        energy = (r32 - 2.0 * sqrt_r + a) / (r34 * sqrt_denom)
        angular_momentum = (r_safe * r_safe - 2.0 * a * sqrt_r + a * a) / (r34 * sqrt_denom)
        d_omega_dr = -1.5 * sqrt_r / torch.clamp(torch.square(r32 + a), min=1.0e-8)
        return omega, energy, angular_momentum, d_omega_dr

    def _build_nt_page_thorne_lut(self, rin: float, rmax: float) -> tuple[torch.Tensor, torch.Tensor]:
        rin_safe = max(1.0e-4, float(rin) * 1.0005)
        rmax_safe = max(rin_safe + 1.0e-3, float(rmax))
        span = rmax_safe - rin_safe
        samples = max(512, min(4096, int(512 + 64.0 * span)))
        a_key = round(float(self.metric_spin), 8)
        key = (
            str(self.device),
            str(self.dtype),
            a_key,
            round(rin_safe, 6),
            round(rmax_safe, 6),
            int(samples),
        )
        cached = KerrRayTracer._nt_page_thorne_cache.get(key)
        if cached is not None:
            return cached

        disk_key = (
            round(float(self.metric_spin), 8),
            round(rin_safe, 6),
            round(rmax_safe, 6),
            int(samples),
        )
        persisted = self._persistent_load_lut_pair("nt_page_thorne", disk_key)
        if persisted is not None:
            KerrRayTracer._nt_page_thorne_cache[key] = persisted
            return persisted

        rr = torch.linspace(rin_safe, rmax_safe, samples, dtype=self.dtype, device=self.device)
        omega, energy, ang_mom, d_omega_dr = self._kerr_circular_orbit_quantities(rr)

        dL_dr = torch.zeros_like(ang_mom)
        dr = rr[1:] - rr[:-1]
        dL_dr[1:-1] = (ang_mom[2:] - ang_mom[:-2]) / torch.clamp(rr[2:] - rr[:-2], min=1.0e-8)
        dL_dr[0] = (ang_mom[1] - ang_mom[0]) / torch.clamp(dr[0], min=1.0e-8)
        dL_dr[-1] = (ang_mom[-1] - ang_mom[-2]) / torch.clamp(dr[-1], min=1.0e-8)

        e_minus_ol = energy - omega * ang_mom
        integrand = e_minus_ol * dL_dr
        trapezoids = 0.5 * (integrand[1:] + integrand[:-1]) * dr
        integral = torch.zeros_like(rr)
        integral[1:] = torch.cumsum(trapezoids, dim=0)

        flux = -d_omega_dr * integral / torch.clamp(torch.square(e_minus_ol), min=1.0e-8)
        flux = torch.clamp(flux, min=0.0)
        flux[0] = 0.0

        KerrRayTracer._nt_page_thorne_cache[key] = (rr, flux)
        self._persistent_save_lut_pair("nt_page_thorne", disk_key, rr, flux)
        if len(KerrRayTracer._nt_page_thorne_cache) > 24:
            KerrRayTracer._nt_page_thorne_cache.pop(next(iter(KerrRayTracer._nt_page_thorne_cache)))
        return rr, flux

    def _novikov_thorne_flux_profile_page_thorne(self, r: torch.Tensor, rin: torch.Tensor) -> torch.Tensor:
        if self.config.metric_model != "kerr":
            return self._novikov_thorne_flux_profile(r, rin)

        r_safe = torch.clamp(r, min=rin * 1.0005)
        rin_val = float(torch.clamp(rin, min=1.0e-4).item())
        rmax_val = float(torch.max(r_safe).item())
        rr_lut, flux_lut = self._build_nt_page_thorne_lut(rin=rin_val, rmax=rmax_val)

        return self._sample_flux_lut(r_safe, rr_lut, flux_lut)

    def _disk_half_thickness_and_slope(self, r: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.config
        if (not cfg.thick_disk) or cfg.disk_thickness_ratio <= 0.0:
            z = torch.zeros_like(r)
            return z, z
        rin = torch.as_tensor(cfg.disk_inner_radius, dtype=self.dtype, device=self.device)
        rr = torch.clamp(r, min=1.0e-8)
        rin_safe = torch.clamp(rin, min=1.0e-8)
        ratio = torch.as_tensor(cfg.disk_thickness_ratio, dtype=self.dtype, device=self.device)
        growth = torch.pow(torch.clamp(rr / rin_safe, min=1.0e-6), cfg.disk_thickness_power)
        half_h_smooth = torch.clamp(ratio * rr * growth, min=0.0)
        slope_smooth = half_h_smooth * (1.0 + cfg.disk_thickness_power) / rr

        if cfg.disk_structure_mode != "concentric_annuli":
            return half_h_smooth, slope_smooth

        rout = torch.as_tensor(cfg.disk_outer_radius, dtype=self.dtype, device=self.device)
        span = torch.clamp(rout - rin, min=1.0e-6)
        n_ann = max(4, int(cfg.disk_annuli_count))
        n_ann_t = torch.as_tensor(float(n_ann), dtype=self.dtype, device=self.device)
        dr = span / n_ann_t

        u = torch.clamp((rr - rin) / span, min=0.0, max=1.0 - 1.0e-7)
        ring_idx = torch.floor(u * n_ann_t)
        r_center = rin + (ring_idx + 0.5) * dr
        r_center = torch.clamp(r_center, min=rin_safe, max=rout)

        growth_center = torch.pow(torch.clamp(r_center / rin_safe, min=1.0e-6), cfg.disk_thickness_power)
        half_h_ring = torch.clamp(ratio * r_center * growth_center, min=0.0)

        # Cylindrical annuli with gently larger scale-height near the inner edge
        # and slightly thinner outer annuli (avoid excessive puffing at Rin).
        proximity_inner = torch.clamp((rout - r_center) / span, min=0.0, max=1.0)
        height_scale = 0.80 + 0.35 * proximity_inner
        half_h_ring = half_h_ring * height_scale

        blend = torch.clamp(
            torch.as_tensor(cfg.disk_annuli_blend, dtype=self.dtype, device=self.device),
            min=0.0,
            max=1.0,
        )
        half_h = (1.0 - blend) * half_h_smooth + blend * half_h_ring
        slope = (1.0 - blend) * slope_smooth
        return half_h, slope

    def _disk_half_thickness(self, r: torch.Tensor) -> torch.Tensor:
        return self._disk_half_thickness_and_slope(r)[0]

    def _disk_volume_emission_factor(self, r: torch.Tensor, p_theta_abs: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        if (not cfg.disk_volume_emission) or cfg.disk_volume_strength <= 0.0:
            return torch.ones_like(r)

        h = self._disk_half_thickness(r)
        h_fallback = 0.015 * torch.clamp(r, min=1.0)
        h_eff = torch.where(h > 1.0e-6, h, h_fallback)

        mu_los = p_theta_abs / torch.clamp(p_theta_abs + 0.10, min=1.0e-6)
        path_len = (2.0 * h_eff) / torch.clamp(mu_los, min=0.04)
        tau = (
            torch.as_tensor(cfg.disk_volume_density_scale, dtype=self.dtype, device=self.device)
            * path_len
            / torch.clamp(r, min=1.0e-6)
        )
        trans = 1.0 - torch.exp(-torch.clamp(tau, min=0.0, max=40.0))

        ns = max(1, int(cfg.disk_volume_samples))
        if ns == 1:
            temp_factor = torch.ones_like(r)
        else:
            zeta = torch.linspace(-1.0, 1.0, ns, dtype=self.dtype, device=self.device).view(1, ns)
            sigma = torch.as_tensor(0.48, dtype=self.dtype, device=self.device)
            rho = torch.exp(-0.5 * torch.square(zeta / sigma))
            temp_drop = torch.as_tensor(cfg.disk_volume_temperature_drop, dtype=self.dtype, device=self.device)
            temp_mod = 1.0 - temp_drop * torch.abs(zeta)
            angle_mod = 0.85 + 0.15 * mu_los.unsqueeze(-1)
            kernel = rho * temp_mod * angle_mod
            denom = torch.clamp(torch.sum(rho, dim=-1, keepdim=True), min=1.0e-8)
            temp_factor = torch.sum(kernel, dim=-1) / denom.squeeze(-1)

        strength = torch.as_tensor(cfg.disk_volume_strength, dtype=self.dtype, device=self.device)
        factor = 1.0 + strength * trans * torch.clamp(temp_factor, min=0.0)
        return torch.clamp(factor, min=0.0, max=8.0)

    def _sample_equirectangular(self, texture: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        tex_h = texture.shape[0]
        tex_w = texture.shape[1]
        x = torch.clamp(u, min=0.0, max=1.0) * (tex_w - 1)
        y = torch.clamp(v, min=0.0, max=1.0) * (tex_h - 1)

        x0 = torch.floor(x).to(torch.long)
        y0 = torch.floor(y).to(torch.long)
        x1 = torch.remainder(x0 + 1, tex_w)
        y1 = torch.clamp(y0 + 1, max=tex_h - 1)

        tx = (x - x0.to(self.dtype)).unsqueeze(-1)
        ty = (y - y0.to(self.dtype)).unsqueeze(-1)

        c00 = texture[y0, x0]
        c10 = texture[y0, x1]
        c01 = texture[y1, x0]
        c11 = texture[y1, x1]

        c0 = c00 * (1.0 - tx) + c10 * tx
        c1 = c01 * (1.0 - tx) + c11 * tx
        return c0 * (1.0 - ty) + c1 * ty

    def _background_cache_key(self) -> tuple[object, ...]:
        cfg = self.config
        if cfg.background_mode == "hdri" and cfg.hdri_path:
            p = Path(cfg.hdri_path).expanduser().resolve()
            st = p.stat()
            src = ("hdri", str(p), int(st.st_mtime_ns), int(st.st_size))
        elif cfg.background_mode == "darkspace":
            src = ("darkspace",)
        else:
            src = (
                "procedural",
                int(cfg.star_seed),
                float(cfg.star_density),
                float(cfg.star_brightness),
                float(cfg.background_meridian_offset_deg),
            )
        return (
            src,
            int(cfg.cubemap_face_size),
            float(cfg.hdri_exposure),
            float(cfg.hdri_rotation_deg),
            self.device.type,
            str(self.dtype),
        )

    def _cube_face_dirs(self, face_idx: int, size: int) -> torch.Tensor:
        coords = (torch.arange(size, dtype=self.dtype, device=self.device) + 0.5) * (2.0 / float(size)) - 1.0
        s, t = torch.meshgrid(coords, -coords, indexing="xy")
        ones = torch.ones_like(s)

        if face_idx == 0:  # +X
            dirs = torch.stack([ones, t, -s], dim=-1)
        elif face_idx == 1:  # -X
            dirs = torch.stack([-ones, t, s], dim=-1)
        elif face_idx == 2:  # +Y
            dirs = torch.stack([s, ones, -t], dim=-1)
        elif face_idx == 3:  # -Y
            dirs = torch.stack([s, -ones, t], dim=-1)
        elif face_idx == 4:  # +Z
            dirs = torch.stack([s, t, ones], dim=-1)
        else:  # face_idx == 5, -Z
            dirs = torch.stack([-s, t, -ones], dim=-1)

        return dirs / torch.clamp(torch.linalg.norm(dirs, dim=-1, keepdim=True), min=1.0e-12)

    def _spherical_to_dirs(self, theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        sin_th = torch.sin(theta)
        return torch.stack(
            [
                sin_th * torch.cos(phi),
                sin_th * torch.sin(phi),
                torch.cos(theta),
            ],
            dim=-1,
        )

    def _sample_procedural_equirect(self, theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        two_pi = 2.0 * self.pi
        meridian_rot = torch.as_tensor(math.radians(cfg.background_meridian_offset_deg), dtype=self.dtype, device=self.device)

        # Keep seam away from symmetric lensing axes for equirectangular fallback.
        u = torch.remainder(phi + meridian_rot, two_pi) / two_pi
        v = torch.clamp(theta / self.pi, min=0.0, max=1.0)

        top = torch.tensor([0.0008, 0.0010, 0.0020], dtype=self.dtype, device=self.device)
        bottom = torch.tensor([0.0022, 0.0026, 0.0048], dtype=self.dtype, device=self.device)
        sky = top + (bottom - top) * torch.pow(v.unsqueeze(-1), 0.60)

        large_noise = self._fbm(u + 0.17, v + 0.11, base_scale=6.0, octaves=4, lacunarity=2.15, gain=0.52, offset=3.7)
        medium_noise = self._fbm(u * 1.3 + 0.5, v * 1.2 + 0.9, base_scale=14.0, octaves=4, lacunarity=2.2, gain=0.50, offset=15.1)

        band_center = 0.5 + 0.10 * torch.sin(two_pi * (u + 0.08 * large_noise)) + 0.04 * torch.sin(two_pi * (3.2 * u + 0.3))
        band_dist = torch.abs(v - band_center)
        band_width = 0.050 + 0.030 * (1.0 - medium_noise)
        milky = torch.exp(-torch.pow(band_dist / torch.clamp(band_width, min=1.0e-4), 2.0))

        filament = self._fbm(u * 3.0 + 1.2, v * 3.2 + 2.7, base_scale=34.0, octaves=3, lacunarity=2.35, gain=0.55, offset=29.4)
        dust_mask = torch.clamp(1.1 * medium_noise - 0.8 * filament, min=0.0, max=1.0)
        milky_color = torch.tensor([0.90, 0.84, 0.74], dtype=self.dtype, device=self.device)
        sky = sky + (milky * (1.0 - 0.55 * dust_mask) * 0.20).unsqueeze(-1) * milky_color

        core_u = torch.abs(u - 0.62)
        core_u = torch.minimum(core_u, 1.0 - core_u)
        core_v = v - (band_center + 0.01)
        core = torch.exp(-torch.pow(core_u / 0.035, 2.0) - torch.pow(core_v / 0.020, 2.0))
        core_noise = self._fbm(u + 2.8, v + 4.1, base_scale=42.0, octaves=3, lacunarity=2.2, gain=0.5, offset=40.6)
        core_color = torch.tensor([1.20, 0.92, 0.56], dtype=self.dtype, device=self.device)
        sky = sky + (core * (0.45 + 0.55 * core_noise) * 0.55).unsqueeze(-1) * core_color

        nebula_a_u = torch.abs(u - 0.18)
        nebula_a_u = torch.minimum(nebula_a_u, 1.0 - nebula_a_u)
        nebula_a = torch.exp(-torch.pow(nebula_a_u / 0.13, 2.0) - torch.pow((v - 0.28) / 0.09, 2.0))
        nebula_a_noise = self._fbm(u * 2.0 + 0.8, v * 2.1 + 2.3, base_scale=22.0, octaves=4, lacunarity=2.3, gain=0.53, offset=54.2)
        nebula_a_col = torch.tensor([0.24, 0.46, 0.95], dtype=self.dtype, device=self.device)
        sky = sky + (nebula_a * torch.pow(nebula_a_noise, 1.6) * 0.12).unsqueeze(-1) * nebula_a_col

        nebula_b_u = torch.abs(u - 0.79)
        nebula_b_u = torch.minimum(nebula_b_u, 1.0 - nebula_b_u)
        nebula_b = torch.exp(-torch.pow(nebula_b_u / 0.10, 2.0) - torch.pow((v - 0.64) / 0.08, 2.0))
        nebula_b_noise = self._fbm(u * 2.4 + 1.9, v * 1.9 + 0.6, base_scale=26.0, octaves=4, lacunarity=2.2, gain=0.52, offset=69.0)
        nebula_b_col = torch.tensor([0.96, 0.30, 0.52], dtype=self.dtype, device=self.device)
        sky = sky + (nebula_b * torch.pow(nebula_b_noise, 1.7) * 0.10).unsqueeze(-1) * nebula_b_col

        haze = self._fbm(u * 1.6 + 3.0, v * 1.4 + 5.0, base_scale=10.0, octaves=3, lacunarity=2.0, gain=0.55, offset=82.4)
        sky = sky + (0.003 * haze).unsqueeze(-1) * torch.tensor([0.45, 0.57, 0.80], dtype=self.dtype, device=self.device)

        density = self.config.star_density
        brightness = self.config.star_brightness
        fine_stars = self._star_layer(u, v, grid_u=2600.0, grid_v=1300.0, density=2.8 * density, brightness=1.5 * brightness, offset=101.0)
        giant_stars = self._star_layer(u, v, grid_u=680.0, grid_v=340.0, density=0.35 * density, brightness=3.8 * brightness, offset=141.0)
        return sky + fine_stars + giant_stars

    def _sample_sky_equirect(self, theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        return self._sample_sky_equirect_custom(
            theta,
            phi,
            texture=self._hdri_tex,
            exposure=float(self.config.hdri_exposure),
            rotation_deg=float(self.config.hdri_rotation_deg),
        )

    def _sample_sky_equirect_custom(
        self,
        theta: torch.Tensor,
        phi: torch.Tensor,
        *,
        texture: torch.Tensor | None,
        exposure: float,
        rotation_deg: float,
    ) -> torch.Tensor:
        cfg = self.config
        if texture is not None:
            two_pi = 2.0 * self.pi
            meridian_rot = torch.as_tensor(math.radians(cfg.background_meridian_offset_deg), dtype=self.dtype, device=self.device)
            rot = torch.as_tensor(math.radians(rotation_deg), dtype=self.dtype, device=self.device) + meridian_rot
            u = torch.remainder(phi + rot, two_pi) / two_pi
            v = torch.clamp(theta / self.pi, min=0.0, max=1.0)
            return self._sample_equirectangular(texture, u, v) * float(exposure)
        if cfg.background_mode == "darkspace":
            return torch.zeros((theta.shape[0], 3), dtype=self.dtype, device=self.device)
        if cfg.background_mode == "hdri":
            return torch.zeros((theta.shape[0], 3), dtype=self.dtype, device=self.device)
        return self._sample_procedural_equirect(theta, phi)

    def _build_cubemap_faces(self, face_size: int) -> tuple[torch.Tensor, ...]:
        faces: list[torch.Tensor] = []
        for face_idx in range(6):
            dirs = self._cube_face_dirs(face_idx, face_size).reshape(-1, 3)
            theta = torch.acos(torch.clamp(dirs[:, 2], min=-1.0, max=1.0))
            phi = torch.atan2(dirs[:, 1], dirs[:, 0])
            face = self._sample_sky_equirect(theta, phi).reshape(face_size, face_size, 3).contiguous()
            faces.append(face)
        return tuple(faces)

    def _build_cubemap_faces_custom(
        self,
        face_size: int,
        *,
        texture: torch.Tensor | None,
        exposure: float,
        rotation_deg: float,
    ) -> tuple[torch.Tensor, ...]:
        faces: list[torch.Tensor] = []
        for face_idx in range(6):
            dirs = self._cube_face_dirs(face_idx, face_size).reshape(-1, 3)
            theta = torch.acos(torch.clamp(dirs[:, 2], min=-1.0, max=1.0))
            phi = torch.atan2(dirs[:, 1], dirs[:, 0])
            face = self._sample_sky_equirect_custom(
                theta,
                phi,
                texture=texture,
                exposure=exposure,
                rotation_deg=rotation_deg,
            ).reshape(face_size, face_size, 3).contiguous()
            faces.append(face)
        return tuple(faces)

    def _get_or_build_cubemap(self) -> tuple[torch.Tensor, ...]:
        key = self._background_cache_key()
        cached = KerrRayTracer._cubemap_cache.get(key)
        if cached is not None:
            return cached
        disk_key = key[:-2]
        persisted = self._persistent_load_cubemap(disk_key)
        if persisted is not None:
            KerrRayTracer._cubemap_cache[key] = persisted
            return persisted
        built = self._build_cubemap_faces(self.config.cubemap_face_size)
        KerrRayTracer._cubemap_cache[key] = built
        self._persistent_save_cubemap(disk_key, built)
        if len(KerrRayTracer._cubemap_cache) > 8:
            KerrRayTracer._cubemap_cache.pop(next(iter(KerrRayTracer._cubemap_cache)))
        return built

    def _get_or_build_wormhole_remote_cubemap(self) -> tuple[torch.Tensor, ...] | None:
        if self._wormhole_remote_hdri_tex is None:
            return None
        if self._wormhole_remote_hdri_file is None:
            return None
        try:
            st = self._wormhole_remote_hdri_file.stat()
            key = (
                "wormhole_remote_hdri",
                str(self._wormhole_remote_hdri_file),
                int(st.st_mtime_ns),
                int(st.st_size),
                int(self.config.cubemap_face_size),
                float(self.config.wormhole_remote_hdri_exposure),
                float(self.config.wormhole_remote_hdri_rotation_deg),
                float(self.config.background_meridian_offset_deg),
                self.device.type,
                str(self.dtype),
            )
        except Exception:
            key = (
                "wormhole_remote_hdri",
                int(self.config.cubemap_face_size),
                float(self.config.wormhole_remote_hdri_exposure),
                float(self.config.wormhole_remote_hdri_rotation_deg),
                float(self.config.background_meridian_offset_deg),
                self.device.type,
                str(self.dtype),
            )
        cached = KerrRayTracer._wormhole_remote_cubemap_cache.get(key)
        if cached is not None:
            return cached
        built = self._build_cubemap_faces_custom(
            int(self.config.cubemap_face_size),
            texture=self._wormhole_remote_hdri_tex,
            exposure=float(self.config.wormhole_remote_hdri_exposure),
            rotation_deg=float(self.config.wormhole_remote_hdri_rotation_deg),
        )
        KerrRayTracer._wormhole_remote_cubemap_cache[key] = built
        if len(KerrRayTracer._wormhole_remote_cubemap_cache) > 8:
            KerrRayTracer._wormhole_remote_cubemap_cache.pop(next(iter(KerrRayTracer._wormhole_remote_cubemap_cache)))
        return built

    def _sample_cubemap_faces_from(
        self,
        faces: tuple[torch.Tensor, ...],
        face: torch.Tensor,
        s: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        if faces is None:
            raise RuntimeError("Cubemap sampler requested without cubemap data")

        u = torch.clamp(0.5 * (s + 1.0), min=0.0, max=1.0)
        v = torch.clamp(0.5 * (1.0 - t), min=0.0, max=1.0)
        out = torch.zeros((face.shape[0], 3), dtype=self.dtype, device=self.device)
        for face_idx, tex in enumerate(faces):
            mask = face == face_idx
            if not bool(mask.any()):
                continue
            uu = u[mask]
            vv = v[mask]
            side = tex.shape[0]
            x_tex = uu * (side - 1)
            y_tex = vv * (side - 1)
            x0 = torch.floor(x_tex).to(torch.long)
            y0 = torch.floor(y_tex).to(torch.long)
            x1 = torch.clamp(x0 + 1, max=side - 1)
            y1 = torch.clamp(y0 + 1, max=side - 1)
            tx = (x_tex - x0.to(self.dtype)).unsqueeze(-1)
            ty = (y_tex - y0.to(self.dtype)).unsqueeze(-1)

            c00 = tex[y0, x0]
            c10 = tex[y0, x1]
            c01 = tex[y1, x0]
            c11 = tex[y1, x1]
            c0 = c00 * (1.0 - tx) + c10 * tx
            c1 = c01 * (1.0 - tx) + c11 * tx
            out[mask] = c0 * (1.0 - ty) + c1 * ty
        return out

    def _sample_cubemap(self, dirs: torch.Tensor, faces: tuple[torch.Tensor, ...] | None = None) -> torch.Tensor:
        faces_eff = self._cubemap_faces if faces is None else faces
        if faces_eff is None:
            theta = torch.acos(torch.clamp(dirs[:, 2], min=-1.0, max=1.0))
            phi = torch.atan2(dirs[:, 1], dirs[:, 0])
            return self._sample_sky_equirect(theta, phi)

        d = dirs / torch.clamp(torch.linalg.norm(dirs, dim=-1, keepdim=True), min=1.0e-12)
        x = d[:, 0]
        y = d[:, 1]
        z = d[:, 2]
        ax = torch.abs(x)
        ay = torch.abs(y)
        az = torch.abs(z)

        safe_ax = torch.clamp(ax, min=1.0e-12)
        safe_ay = torch.clamp(ay, min=1.0e-12)
        safe_az = torch.clamp(az, min=1.0e-12)

        base_face = torch.zeros_like(x, dtype=torch.long)
        face_x = torch.where(x >= 0.0, base_face, torch.ones_like(base_face))
        sx = torch.where(x >= 0.0, -z / safe_ax, z / safe_ax)
        tx = y / safe_ax
        col_x = self._sample_cubemap_faces_from(faces_eff, face_x, sx, tx)

        face_y = torch.where(y >= 0.0, torch.full_like(face_x, 2), torch.full_like(face_x, 3))
        sy = x / safe_ay
        ty = torch.where(y >= 0.0, -z / safe_ay, z / safe_ay)
        col_y = self._sample_cubemap_faces_from(faces_eff, face_y, sy, ty)

        face_z = torch.where(z >= 0.0, torch.full_like(face_x, 4), torch.full_like(face_x, 5))
        sz = torch.where(z >= 0.0, x / safe_az, -x / safe_az)
        tz = y / safe_az
        col_z = self._sample_cubemap_faces_from(faces_eff, face_z, sz, tz)

        # Seamless cubemap blend near face boundaries.
        sharpness = 16.0
        valid_x = ((torch.abs(sx) <= 1.0) & (torch.abs(tx) <= 1.0)).to(self.dtype).unsqueeze(-1)
        valid_y = ((torch.abs(sy) <= 1.0) & (torch.abs(ty) <= 1.0)).to(self.dtype).unsqueeze(-1)
        valid_z = ((torch.abs(sz) <= 1.0) & (torch.abs(tz) <= 1.0)).to(self.dtype).unsqueeze(-1)

        wx = torch.pow(torch.clamp(ax, min=1.0e-6), sharpness).unsqueeze(-1) * valid_x
        wy = torch.pow(torch.clamp(ay, min=1.0e-6), sharpness).unsqueeze(-1) * valid_y
        wz = torch.pow(torch.clamp(az, min=1.0e-6), sharpness).unsqueeze(-1) * valid_z
        wsum = torch.clamp(wx + wy + wz, min=1.0e-9)
        blended = (col_x * wx + col_y * wy + col_z * wz) / wsum

        x_major = (ax >= ay) & (ax >= az)
        y_major = (~x_major) & (ay >= az)
        z_major = ~(x_major | y_major)

        face_major = torch.zeros_like(face_x)
        s_major = torch.zeros_like(x)
        t_major = torch.zeros_like(x)

        px = x_major & (x >= 0.0)
        nx = x_major & (x < 0.0)
        py = y_major & (y >= 0.0)
        ny = y_major & (y < 0.0)
        pz = z_major & (z >= 0.0)
        nz = z_major & (z < 0.0)

        face_major = torch.where(px, torch.full_like(face_major, 0), face_major)
        face_major = torch.where(nx, torch.full_like(face_major, 1), face_major)
        face_major = torch.where(py, torch.full_like(face_major, 2), face_major)
        face_major = torch.where(ny, torch.full_like(face_major, 3), face_major)
        face_major = torch.where(pz, torch.full_like(face_major, 4), face_major)
        face_major = torch.where(nz, torch.full_like(face_major, 5), face_major)

        s_major = torch.where(px, -z / safe_ax, s_major)
        t_major = torch.where(px, y / safe_ax, t_major)
        s_major = torch.where(nx, z / safe_ax, s_major)
        t_major = torch.where(nx, y / safe_ax, t_major)
        s_major = torch.where(py, x / safe_ay, s_major)
        t_major = torch.where(py, -z / safe_ay, t_major)
        s_major = torch.where(ny, x / safe_ay, s_major)
        t_major = torch.where(ny, z / safe_ay, t_major)
        s_major = torch.where(pz, x / safe_az, s_major)
        t_major = torch.where(pz, y / safe_az, t_major)
        s_major = torch.where(nz, -x / safe_az, s_major)
        t_major = torch.where(nz, y / safe_az, t_major)

        fallback = self._sample_cubemap_faces_from(faces_eff, face_major, s_major, t_major)
        use_blend = (wx + wy + wz) > 1.0e-8
        return torch.where(use_blend, blended, fallback)

    def _star_background(
        self,
        theta: torch.Tensor,
        phi: torch.Tensor,
        side_sign: torch.Tensor | None = None,
    ) -> torch.Tensor:
        dirs = None
        if self.config.background_projection == "cubemap":
            dirs = self._spherical_to_dirs(theta, phi)
            local = self._sample_cubemap(dirs)
        else:
            local = self._sample_sky_equirect(theta, phi)

        if (not self.is_wormhole) or side_sign is None:
            return local

        if (
            self.config.background_projection == "cubemap"
            and bool(self.config.wormhole_remote_cubemap_coherent)
            and (self._wormhole_remote_cubemap_faces is not None)
            and (dirs is not None)
        ):
            remote = self._sample_cubemap(dirs, faces=self._wormhole_remote_cubemap_faces)
        else:
            remote = self._sample_sky_equirect_custom(
                theta,
                phi,
                texture=self._wormhole_remote_hdri_tex,
                exposure=float(self.config.wormhole_remote_hdri_exposure),
                rotation_deg=float(self.config.wormhole_remote_hdri_rotation_deg),
            )
        # Always apply a smooth blend for Morris-Thorne wormholes.
        # side_sign here is r_min (minimum proper-distance coordinate reached during
        # integration), so tanh(r_min / blend_w) transitions smoothly through 0 at
        # the throat, anti-aliasing the critical-curve seam that appeared as a hard
        # vertical bar when the final r value (±escape_radius) was used instead.
        # wormhole_background_continuous_blend now controls blend width only.
        blend_w = torch.as_tensor(
            max(1.0e-6, float(self.config.wormhole_background_blend_width)),
            dtype=self.dtype,
            device=self.device,
        )
        remote_w = 0.5 * (1.0 - torch.tanh(side_sign / blend_w))
        remote_w = torch.clamp(remote_w, min=0.0, max=1.0)
        if remote_w.ndim == 1:
            remote_w = remote_w.unsqueeze(-1)
        return local * (1.0 - remote_w) + remote * remote_w

    def _trace(
        self,
        x_pixel_offset: float = 0.0,
        y_pixel_offset: float = 0.0,
        emitter: EmitterInput = None,
        row_start: int = 0,
        row_end: int | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        int,
    ]:
        cfg = self.config
        state = self._camera_rays(
            x_pixel_offset=x_pixel_offset,
            y_pixel_offset=y_pixel_offset,
            row_start=row_start,
            row_end=row_end,
        )
        n = state.shape[0]

        active = torch.ones(n, dtype=torch.bool, device=self.device)
        hit_disk = torch.zeros(n, dtype=torch.bool, device=self.device)
        hit_emitter = torch.zeros(n, dtype=torch.bool, device=self.device)
        hit_horizon = torch.zeros(n, dtype=torch.bool, device=self.device)
        escaped = torch.zeros(n, dtype=torch.bool, device=self.device)
        r_min = state[:, 1].clone()

        r_emit = torch.zeros(n, dtype=self.dtype, device=self.device)
        p_t_emit = torch.zeros(n, dtype=self.dtype, device=self.device)
        p_phi_emit = torch.zeros(n, dtype=self.dtype, device=self.device)
        p_theta_emit = torch.zeros(n, dtype=self.dtype, device=self.device)
        disk_vertical_weight = torch.zeros(n, dtype=self.dtype, device=self.device)
        disk_hit_order = torch.zeros(n, dtype=self.dtype, device=self.device)
        disk_cross_count = torch.zeros(n, dtype=torch.int16, device=self.device)
        t_emit = torch.zeros(n, dtype=self.dtype, device=self.device)
        phi_emit = torch.zeros(n, dtype=self.dtype, device=self.device)
        p_t_particle = torch.zeros(n, dtype=self.dtype, device=self.device)
        p_r_particle = torch.zeros(n, dtype=self.dtype, device=self.device)
        p_theta_particle = torch.zeros(n, dtype=self.dtype, device=self.device)
        p_phi_particle = torch.zeros(n, dtype=self.dtype, device=self.device)

        emitters = self._normalize_emitters(emitter)
        emitter_pos: torch.Tensor | None = None
        emitter_radius2: torch.Tensor | None = None
        if emitters:
            em_r = torch.as_tensor([em.r for em in emitters], dtype=self.dtype, device=self.device)
            em_theta = torch.as_tensor([em.theta for em in emitters], dtype=self.dtype, device=self.device)
            em_phi = torch.as_tensor([em.phi for em in emitters], dtype=self.dtype, device=self.device)
            emitter_pos = self._bl_to_cartesian_kerr_schild(em_r, em_theta, em_phi)
            emitter_radius2 = torch.as_tensor(
                [em.radius * em.radius for em in emitters],
                dtype=self.dtype,
                device=self.device,
            )

        has_horizon = self.has_horizon
        horizon_cut = (
            self.horizon * 1.0005
            if has_horizon
            else torch.as_tensor(-1.0e30, dtype=self.dtype, device=self.device)
        )
        ray_step = torch.full((n,), float(cfg.step_size), dtype=self.dtype, device=self.device)
        if cfg.adaptive_integrator:
            ray_step = torch.clamp(ray_step, min=cfg.adaptive_step_min, max=cfg.adaptive_step_max)
        use_bl_fsal = bool(cfg.adaptive_integrator)
        bl_k1_cache = (
            torch.zeros((n, state.shape[1]), dtype=self.dtype, device=self.device)
            if use_bl_fsal
            else None
        )
        bl_k1_valid = (
            torch.zeros((n,), dtype=torch.bool, device=self.device)
            if use_bl_fsal
            else None
        )

        steps_used = 0
        max_attempts = cfg.max_steps * (2 if cfg.adaptive_integrator else 1)

        for step in range(max_attempts):
            if not bool(active.any()):
                break

            steps_used = step + 1
            idx = torch.nonzero(active, as_tuple=False).squeeze(-1)
            prev_state = state[idx]
            local_step = ray_step[idx]
            step_used = local_step

            if cfg.adaptive_integrator:
                if cfg.adaptive_event_aware:
                    event_fac = self._event_aware_step_factor_bl(
                        r=prev_state[:, 1],
                        theta=prev_state[:, 2],
                        horizon_cut=horizon_cut,
                    )
                    local_step = torch.clamp(
                        local_step * event_fac,
                        min=cfg.adaptive_step_min,
                        max=cfg.adaptive_step_max,
                    )
                recovered_local = torch.zeros_like(local_step, dtype=torch.bool)
                if use_bl_fsal and bl_k1_cache is not None and bl_k1_valid is not None:
                    next_all, accepted_local, next_step, fatal_local, k7 = self._rk45_adaptive_step_fsal(
                        prev_state,
                        local_step,
                        k1=bl_k1_cache[idx],
                        k1_valid=bl_k1_valid[idx],
                    )
                else:
                    next_all, accepted_local, next_step, fatal_local = self._rk45_adaptive_step(prev_state, local_step)
                    k7 = None
                ray_step[idx] = next_step

                if bool(fatal_local.any()) and cfg.adaptive_fallback_rk4:
                    fatal_pos = torch.nonzero(fatal_local, as_tuple=False).squeeze(-1)
                    fatal_prev = prev_state[fatal_pos]
                    substeps = int(cfg.adaptive_fallback_substeps)
                    h_fb = float(cfg.adaptive_step_min) / max(1, substeps)
                    fb_state = fatal_prev
                    fb_ok = torch.ones((fatal_prev.shape[0],), dtype=torch.bool, device=self.device)
                    for _ in range(substeps):
                        fb_next = self._rk4_bl_step_fn(fb_state, h_fb)
                        finite_fb = torch.isfinite(fb_next).all(dim=-1)
                        fb_ok = fb_ok & finite_fb
                        fb_state = torch.where(finite_fb.unsqueeze(-1), fb_next, fb_state)
                    recovered_local[fatal_pos] = fb_ok
                    if bool(recovered_local.any()):
                        next_all = torch.where(recovered_local.unsqueeze(-1), fb_state.new_zeros(next_all.shape), next_all)
                        next_all[fatal_pos[fb_ok]] = fb_state[fb_ok]
                        accepted_local = accepted_local | recovered_local
                        ray_step[idx[recovered_local]] = torch.as_tensor(cfg.adaptive_step_min, dtype=self.dtype, device=self.device)
                        if use_bl_fsal and bl_k1_valid is not None:
                            bl_k1_valid[idx[recovered_local]] = False
                    fatal_local = fatal_local & (~recovered_local)

                if bool(fatal_local.any()):
                    fatal_idx = idx[fatal_local]
                    hit_horizon[fatal_idx] = True
                    escaped[fatal_idx] = False
                    active[fatal_idx] = False
                    if use_bl_fsal and bl_k1_valid is not None:
                        bl_k1_valid[fatal_idx] = False

                if use_bl_fsal and bl_k1_cache is not None and bl_k1_valid is not None and k7 is not None:
                    accepted_non_recovered = accepted_local & (~recovered_local)
                    if bool(accepted_non_recovered.any()):
                        accepted_idx = idx[accepted_non_recovered]
                        bl_k1_cache[accepted_idx] = k7[accepted_non_recovered]
                        bl_k1_valid[accepted_idx] = True

                if not bool(accepted_local.any()):
                    continue

                idx = idx[accepted_local]
                prev_state = prev_state[accepted_local]
                next_state = next_all[accepted_local]
                step_used = local_step[accepted_local]
                deriv_prev = None
            else:
                next_state, deriv_prev = self._rk4_step_with_k1(prev_state, float(cfg.step_size))
                finite = torch.isfinite(next_state).all(dim=-1)
                if bool((~finite).any()):
                    bad_idx = idx[~finite]
                    hit_horizon[bad_idx] = True
                    escaped[bad_idx] = False
                    active[bad_idx] = False

                if not bool(finite.any()):
                    continue

                idx = idx[finite]
                prev_state = prev_state[finite]
                next_state = next_state[finite]
                deriv_prev = deriv_prev[finite]
                step_used = torch.full((next_state.shape[0],), float(cfg.step_size), dtype=self.dtype, device=self.device)

            state[idx] = next_state

            prev_r = prev_state[:, 1]
            next_r = next_state[:, 1]
            prev_theta = prev_state[:, 2]
            next_theta = next_state[:, 2]
            r_min[idx] = torch.minimum(r_min[idx], torch.minimum(prev_r, next_r))

            if deriv_prev is None:
                deriv_prev = self._rhs_fn(prev_state)
            deriv_next = self._rhs_fn(next_state)

            touch_eps = torch.as_tensor(2.0e-4, dtype=self.dtype, device=self.device)
            if cfg.thick_disk and cfg.disk_thickness_ratio > 0.0:
                prev_z = prev_r * torch.cos(prev_theta)
                next_z = next_r * torch.cos(next_theta)
                prev_h, prev_dh = self._disk_half_thickness_and_slope(prev_r)
                next_h, next_dh = self._disk_half_thickness_and_slope(next_r)
                prev_q = torch.abs(prev_z) - prev_h
                next_q = torch.abs(next_z) - next_h
                cross_disk = (prev_q * next_q <= 0.0) | (prev_q <= touch_eps) | (next_q <= touch_eps)

                dr_prev = deriv_prev[:, 1]
                dr_next = deriv_next[:, 1]
                dth_prev = deriv_prev[:, 2]
                dth_next = deriv_next[:, 2]
                dz_prev = dr_prev * torch.cos(prev_theta) - prev_r * torch.sin(prev_theta) * dth_prev
                dz_next = dr_next * torch.cos(next_theta) - next_r * torch.sin(next_theta) * dth_next
                sign_eps_prev = 0.15 * prev_h + 1.0e-4
                sign_eps_next = 0.15 * next_h + 1.0e-4
                sign_prev = prev_z / torch.sqrt(prev_z * prev_z + sign_eps_prev * sign_eps_prev)
                sign_next = next_z / torch.sqrt(next_z * next_z + sign_eps_next * sign_eps_next)
                dq_prev = sign_prev * dz_prev - prev_dh * dr_prev
                dq_next = sign_next * dz_next - next_dh * dr_next

                alpha_disk = self._refine_event_alpha(
                    prev_q,
                    next_q,
                    dq_prev,
                    dq_next,
                    step_used,
                    torch.zeros_like(prev_q),
                )
                alpha_disk = torch.where(prev_q <= 0.0, torch.zeros_like(alpha_disk), alpha_disk)
                alpha_disk = torch.clamp(alpha_disk, min=0.0, max=1.0)

                h_mid = 0.5 * (prev_h + next_h)
                soft_band = torch.clamp(
                    torch.as_tensor(cfg.disk_vertical_softness, dtype=self.dtype, device=self.device) * h_mid,
                    min=1.0e-4,
                )
                min_q = torch.minimum(prev_q, next_q)
                near_surface = (~cross_disk) & (min_q > 0.0) & (min_q <= soft_band)
                if cfg.vertical_transition_mode == "snap":
                    alpha_near = torch.where(prev_q <= next_q, torch.zeros_like(alpha_disk), torch.ones_like(alpha_disk))
                else:
                    alpha_near = torch.clamp(
                        prev_q / torch.clamp(prev_q + next_q, min=1.0e-8),
                        min=0.0,
                        max=1.0,
                    )
                alpha_disk = torch.where(near_surface, alpha_near, alpha_disk)
                vertical_local = torch.where(
                    cross_disk,
                    torch.ones_like(alpha_disk),
                    torch.where(
                        near_surface,
                        torch.exp(-torch.pow(min_q / torch.clamp(soft_band, min=1.0e-6), 2.0)),
                        torch.zeros_like(alpha_disk),
                    ),
                )
                disk_candidate = cross_disk | near_surface
            else:
                prev_equ = prev_theta - self.equatorial
                next_equ = next_theta - self.equatorial
                cross_disk = (prev_equ * next_equ <= 0.0) | (torch.abs(prev_equ) < touch_eps) | (torch.abs(next_equ) < touch_eps)
                alpha_disk = self._refine_event_alpha(
                    prev_theta,
                    next_theta,
                    deriv_prev[:, 2],
                    deriv_next[:, 2],
                    step_used,
                    self.equatorial,
                )
                disk_candidate = cross_disk
                vertical_local = torch.where(cross_disk, torch.ones_like(alpha_disk), torch.zeros_like(alpha_disk))
            r_cross = self._hermite_interp(prev_r, next_r, deriv_prev[:, 1], deriv_next[:, 1], step_used, alpha_disk)

            valid_disk = (
                disk_candidate
                & (r_cross >= cfg.disk_inner_radius)
                & (r_cross <= cfg.disk_outer_radius)
                & (r_cross > horizon_cut)
            )
            if not self.enable_disk:
                valid_disk = torch.zeros_like(valid_disk)
            order_local = disk_cross_count[idx] + valid_disk.to(dtype=disk_cross_count.dtype)
            if cfg.multi_hit_disk:
                order_local = torch.clamp(order_local, min=0, max=int(cfg.max_disk_crossings))
            disk_cross_count[idx] = order_local

            alpha_particle = torch.zeros_like(prev_r)
            particle_local = torch.zeros_like(valid_disk)
            if emitter_pos is not None and emitter_radius2 is not None:
                prev_xyz = self._bl_to_cartesian(prev_r, prev_theta, prev_state[:, 3])
                next_xyz = self._bl_to_cartesian(next_r, next_theta, next_state[:, 3])
                seg = next_xyz - prev_xyz
                seg2 = torch.clamp(torch.sum(seg * seg, dim=-1), min=1.0e-9)
                alpha_best = torch.full_like(prev_r, 2.0)
                for em_idx in range(int(emitter_pos.shape[0])):
                    em_pos = emitter_pos[em_idx]
                    em_r2 = emitter_radius2[em_idx]
                    to_emitter = em_pos.view(1, 3) - prev_xyz
                    alpha_seg = torch.clamp(torch.sum(to_emitter * seg, dim=-1) / seg2, min=0.0, max=1.0)
                    # Refine hit test on the interpolated geodesic point (not the straight segment chord)
                    # to reduce false-positive hits in strongly bent trajectories.
                    r_part = self._hermite_interp(
                        prev_r,
                        next_r,
                        deriv_prev[:, 1],
                        deriv_next[:, 1],
                        step_used,
                        alpha_seg,
                    )
                    theta_part = self._hermite_interp(
                        prev_theta,
                        next_theta,
                        deriv_prev[:, 2],
                        deriv_next[:, 2],
                        step_used,
                        alpha_seg,
                    )
                    phi_part = self._hermite_interp(
                        prev_state[:, 3],
                        next_state[:, 3],
                        deriv_prev[:, 3],
                        deriv_next[:, 3],
                        step_used,
                        alpha_seg,
                    )
                    geodesic_xyz = self._bl_to_cartesian(r_part, theta_part, phi_part)
                    delta = geodesic_xyz - em_pos.view(1, 3)
                    dist2 = torch.sum(delta * delta, dim=-1)
                    local_hit = (dist2 <= em_r2) & torch.isfinite(dist2) & (r_part > horizon_cut)
                    better = local_hit & (alpha_seg < alpha_best)
                    alpha_best = torch.where(better, alpha_seg, alpha_best)
                    particle_local = particle_local | local_hit
                alpha_particle = torch.where(particle_local, alpha_best, alpha_particle)

            if has_horizon:
                horizon_cross = ((prev_r > horizon_cut) & (next_r <= horizon_cut)) | (next_r <= horizon_cut)
                horizon_local = horizon_cross & (~valid_disk)
            else:
                horizon_local = torch.zeros_like(valid_disk)

            escape_r = torch.as_tensor(cfg.escape_radius, dtype=self.dtype, device=self.device)
            if self.is_wormhole:
                prev_abs = torch.abs(prev_r)
                next_abs = torch.abs(next_r)
                escape_cross = ((prev_abs < escape_r) & (next_abs >= escape_r)) | (next_abs >= escape_r)
            else:
                escape_cross = ((prev_r < escape_r) & (next_r >= escape_r)) | (next_r >= escape_r)
            escape_local = escape_cross & (~valid_disk) & (~horizon_local)

            alpha_disk_eff = torch.where(valid_disk, alpha_disk, torch.full_like(prev_r, 2.0))

            if has_horizon:
                alpha_h = self._refine_event_alpha(
                    prev_r,
                    next_r,
                    deriv_prev[:, 1],
                    deriv_next[:, 1],
                    step_used,
                    horizon_cut,
                )
                alpha_h_eff = torch.where(horizon_local, alpha_h, torch.full_like(prev_r, 2.0))
            else:
                alpha_h_eff = torch.full_like(prev_r, 2.0)

            alpha_e = self._refine_event_alpha(
                prev_r,
                next_r,
                deriv_prev[:, 1],
                deriv_next[:, 1],
                step_used,
                escape_r,
            )
            alpha_e_eff = torch.where(escape_local, alpha_e, torch.full_like(prev_r, 2.0))

            alpha_p_eff = torch.where(particle_local, alpha_particle, torch.full_like(prev_r, 2.0))

            event_alpha = torch.full_like(prev_r, 2.0)
            event_code = torch.zeros_like(prev_r, dtype=torch.int64)

            select_particle = alpha_p_eff < event_alpha
            event_alpha = torch.where(select_particle, alpha_p_eff, event_alpha)
            event_code = torch.where(select_particle, torch.ones_like(event_code), event_code)

            select_disk = alpha_disk_eff < event_alpha
            event_alpha = torch.where(select_disk, alpha_disk_eff, event_alpha)
            event_code = torch.where(select_disk, torch.full_like(event_code, 2), event_code)

            select_horizon = alpha_h_eff < event_alpha
            event_alpha = torch.where(select_horizon, alpha_h_eff, event_alpha)
            event_code = torch.where(select_horizon, torch.full_like(event_code, 3), event_code)

            select_escape = alpha_e_eff < event_alpha
            event_alpha = torch.where(select_escape, alpha_e_eff, event_alpha)
            event_code = torch.where(select_escape, torch.full_like(event_code, 4), event_code)

            any_event = event_code > 0
            if not bool(any_event.any()):
                continue

            p_t_cross = self._hermite_interp(
                prev_state[:, 4],
                next_state[:, 4],
                deriv_prev[:, 4],
                deriv_next[:, 4],
                step_used,
                alpha_disk,
            )
            p_phi_cross = self._hermite_interp(
                prev_state[:, 7],
                next_state[:, 7],
                deriv_prev[:, 7],
                deriv_next[:, 7],
                step_used,
                alpha_disk,
            )
            p_theta_cross = self._hermite_interp(
                prev_state[:, 6],
                next_state[:, 6],
                deriv_prev[:, 6],
                deriv_next[:, 6],
                step_used,
                alpha_disk,
            )
            t_cross = self._hermite_interp(
                prev_state[:, 0],
                next_state[:, 0],
                deriv_prev[:, 0],
                deriv_next[:, 0],
                step_used,
                alpha_disk,
            )
            if self._mt_shortest_arc_enabled():
                phi_cross = self._interp_phi_shortest_arc(
                    prev_state[:, 3],
                    next_state[:, 3],
                    alpha_disk,
                )
            else:
                phi_cross = self._hermite_interp(
                    prev_state[:, 3],
                    next_state[:, 3],
                    deriv_prev[:, 3],
                    deriv_next[:, 3],
                    step_used,
                    alpha_disk,
                )

            p_t_part = self._hermite_interp(
                prev_state[:, 4],
                next_state[:, 4],
                deriv_prev[:, 4],
                deriv_next[:, 4],
                step_used,
                alpha_particle,
            )
            p_r_part = self._hermite_interp(
                prev_state[:, 5],
                next_state[:, 5],
                deriv_prev[:, 5],
                deriv_next[:, 5],
                step_used,
                alpha_particle,
            )
            p_theta_part = self._hermite_interp(
                prev_state[:, 6],
                next_state[:, 6],
                deriv_prev[:, 6],
                deriv_next[:, 6],
                step_used,
                alpha_particle,
            )
            p_phi_part = self._hermite_interp(
                prev_state[:, 7],
                next_state[:, 7],
                deriv_prev[:, 7],
                deriv_next[:, 7],
                step_used,
                alpha_particle,
            )

            sel_particle = event_code == 1
            if bool(sel_particle.any()):
                hit_idx = idx[sel_particle]
                hit_emitter[hit_idx] = True
                p_t_particle[hit_idx] = p_t_part[sel_particle]
                p_r_particle[hit_idx] = p_r_part[sel_particle]
                p_theta_particle[hit_idx] = p_theta_part[sel_particle]
                p_phi_particle[hit_idx] = p_phi_part[sel_particle]

            sel_disk = event_code == 2
            if bool(sel_disk.any()):
                hit_idx = idx[sel_disk]
                hit_disk[hit_idx] = True
                r_emit[hit_idx] = r_cross[sel_disk]
                p_t_emit[hit_idx] = p_t_cross[sel_disk]
                p_phi_emit[hit_idx] = p_phi_cross[sel_disk]
                p_theta_emit[hit_idx] = p_theta_cross[sel_disk]
                disk_vertical_weight[hit_idx] = vertical_local[sel_disk]
                t_emit[hit_idx] = t_cross[sel_disk]
                phi_emit[hit_idx] = phi_cross[sel_disk]

            sel_horizon = event_code == 3
            if bool(sel_horizon.any()):
                hit_horizon[idx[sel_horizon]] = True

            sel_escape = event_code == 4
            if bool(sel_escape.any()):
                escaped[idx[sel_escape]] = True

            active[idx[any_event]] = False

        if bool(active.any()):
            escaped[active] = True

        if cfg.enforce_black_hole_shadow and self.has_horizon:
            absorb_cut = horizon_cut * cfg.shadow_absorb_radius_factor
            captured = (r_min <= absorb_cut) & (~hit_disk) & (~hit_emitter)
            if bool(captured.any()):
                hit_horizon[captured] = True
                escaped[captured] = False

        sky_theta, sky_phi, sky_side = self._sky_angles_from_state(state, r_min=r_min)
        return (
            hit_disk,
            hit_emitter,
            hit_horizon,
            escaped,
            r_emit,
            p_t_emit,
            p_phi_emit,
            p_theta_emit,
            disk_vertical_weight,
            disk_hit_order,
            t_emit,
            phi_emit,
            p_t_particle,
            p_r_particle,
            p_theta_particle,
            p_phi_particle,
            sky_theta,
            sky_phi,
            sky_side,
            steps_used,
        )

    def _trace_kerr_schild(
        self,
        x_pixel_offset: float = 0.0,
        y_pixel_offset: float = 0.0,
        emitter: EmitterInput = None,
        row_start: int = 0,
        row_end: int | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        int,
    ]:
        cfg = self.config
        if cfg.coordinate_system == "generalized_doran":
            state = self._camera_rays_generalized_doran(
                x_pixel_offset=x_pixel_offset,
                y_pixel_offset=y_pixel_offset,
                row_start=row_start,
                row_end=row_end,
            )
        else:
            state = self._camera_rays_kerr_schild(
                x_pixel_offset=x_pixel_offset,
                y_pixel_offset=y_pixel_offset,
                row_start=row_start,
                row_end=row_end,
            )
        n = state.shape[0]

        active = torch.ones(n, dtype=torch.bool, device=self.device)
        hit_disk = torch.zeros(n, dtype=torch.bool, device=self.device)
        hit_emitter = torch.zeros(n, dtype=torch.bool, device=self.device)
        hit_horizon = torch.zeros(n, dtype=torch.bool, device=self.device)
        escaped = torch.zeros(n, dtype=torch.bool, device=self.device)
        r_min = self._ks_radius_from_xyz(state[:, 1:4]).clone()

        r_emit = torch.zeros(n, dtype=self.dtype, device=self.device)
        p_t_emit = torch.zeros(n, dtype=self.dtype, device=self.device)
        p_phi_emit = torch.zeros(n, dtype=self.dtype, device=self.device)
        p_theta_emit = torch.zeros(n, dtype=self.dtype, device=self.device)
        disk_vertical_weight = torch.zeros(n, dtype=self.dtype, device=self.device)
        disk_hit_order = torch.zeros(n, dtype=self.dtype, device=self.device)
        disk_cross_count = torch.zeros(n, dtype=torch.int16, device=self.device)
        t_emit = torch.zeros(n, dtype=self.dtype, device=self.device)
        phi_emit = torch.zeros(n, dtype=self.dtype, device=self.device)
        p_t_particle = torch.zeros(n, dtype=self.dtype, device=self.device)
        p_r_particle = torch.zeros(n, dtype=self.dtype, device=self.device)
        p_theta_particle = torch.zeros(n, dtype=self.dtype, device=self.device)
        p_phi_particle = torch.zeros(n, dtype=self.dtype, device=self.device)

        emitters = self._normalize_emitters(emitter)
        emitter_pos: torch.Tensor | None = None
        emitter_radius2: torch.Tensor | None = None
        if emitters:
            em_r = torch.as_tensor([em.r for em in emitters], dtype=self.dtype, device=self.device)
            em_theta = torch.as_tensor([em.theta for em in emitters], dtype=self.dtype, device=self.device)
            em_phi = torch.as_tensor([em.phi for em in emitters], dtype=self.dtype, device=self.device)
            emitter_pos = self._bl_to_cartesian_kerr_schild(em_r, em_theta, em_phi)
            emitter_radius2 = torch.as_tensor(
                [em.radius * em.radius for em in emitters],
                dtype=self.dtype,
                device=self.device,
            )

        has_horizon = self.has_horizon
        horizon_cut = (
            self.horizon * 1.0005
            if has_horizon
            else torch.as_tensor(-1.0e30, dtype=self.dtype, device=self.device)
        )
        ray_step = torch.full((n,), float(cfg.step_size), dtype=self.dtype, device=self.device)
        if cfg.adaptive_integrator:
            ray_step = torch.clamp(ray_step, min=cfg.adaptive_step_min, max=cfg.adaptive_step_max)
        use_ks_fsal = bool(cfg.adaptive_integrator and self.ks_use_fsal)
        ks_k1_cache = (
            torch.zeros((n, state.shape[1]), dtype=self.dtype, device=self.device)
            if use_ks_fsal
            else None
        )
        ks_k1_valid = (
            torch.zeros((n,), dtype=torch.bool, device=self.device)
            if use_ks_fsal
            else None
        )
        escape_r = torch.as_tensor(cfg.escape_radius, dtype=self.dtype, device=self.device)
        touch_eps = torch.as_tensor(2.0e-4, dtype=self.dtype, device=self.device)

        steps_used = 0
        max_attempts = cfg.max_steps * (2 if cfg.adaptive_integrator else 1)
        for step in range(max_attempts):
            if not bool(active.any()):
                break
            steps_used = step + 1

            idx = torch.nonzero(active, as_tuple=False).squeeze(-1)
            prev_state = state[idx]
            local_step = ray_step[idx]
            step_used = local_step

            if cfg.adaptive_integrator:
                if cfg.adaptive_event_aware:
                    event_fac = self._event_aware_step_factor_ks(
                        xyz=prev_state[:, 1:4],
                        horizon_cut=horizon_cut,
                    )
                    local_step = torch.clamp(
                        local_step * event_fac,
                        min=cfg.adaptive_step_min,
                        max=cfg.adaptive_step_max,
                    )
                recovered_local = torch.zeros_like(local_step, dtype=torch.bool)
                if use_ks_fsal and ks_k1_cache is not None and ks_k1_valid is not None:
                    next_all, accepted_local, next_step, fatal_local, k7 = self._rk45_adaptive_step_kerr_schild_fsal(
                        prev_state,
                        local_step,
                        k1=ks_k1_cache[idx],
                        k1_valid=ks_k1_valid[idx],
                    )
                else:
                    next_all, accepted_local, next_step, fatal_local = self._rk45_adaptive_step_kerr_schild(prev_state, local_step)
                    k7 = None
                ray_step[idx] = next_step

                if bool(fatal_local.any()) and cfg.adaptive_fallback_rk4:
                    fatal_pos = torch.nonzero(fatal_local, as_tuple=False).squeeze(-1)
                    fatal_prev = prev_state[fatal_pos]
                    substeps = int(cfg.adaptive_fallback_substeps)
                    h_fb = float(cfg.adaptive_step_min) / max(1, substeps)
                    fb_state = fatal_prev
                    fb_ok = torch.ones((fatal_prev.shape[0],), dtype=torch.bool, device=self.device)
                    for _ in range(substeps):
                        fb_next = self._rk4_ks_step_fn(fb_state, h_fb)
                        finite_fb = torch.isfinite(fb_next).all(dim=-1)
                        fb_ok = fb_ok & finite_fb
                        fb_state = torch.where(finite_fb.unsqueeze(-1), fb_next, fb_state)
                    recovered_local[fatal_pos] = fb_ok
                    if bool(recovered_local.any()):
                        next_all[fatal_pos[fb_ok]] = fb_state[fb_ok]
                        accepted_local = accepted_local | recovered_local
                        ray_step[idx[recovered_local]] = torch.as_tensor(cfg.adaptive_step_min, dtype=self.dtype, device=self.device)
                        if use_ks_fsal and ks_k1_valid is not None:
                            ks_k1_valid[idx[recovered_local]] = False
                    fatal_local = fatal_local & (~recovered_local)

                if bool(fatal_local.any()):
                    fatal_idx = idx[fatal_local]
                    hit_horizon[fatal_idx] = True
                    escaped[fatal_idx] = False
                    active[fatal_idx] = False
                    if use_ks_fsal and ks_k1_valid is not None:
                        ks_k1_valid[fatal_idx] = False

                if use_ks_fsal and ks_k1_cache is not None and ks_k1_valid is not None and k7 is not None:
                    accepted_non_recovered = accepted_local & (~recovered_local)
                    if bool(accepted_non_recovered.any()):
                        accepted_idx = idx[accepted_non_recovered]
                        ks_k1_cache[accepted_idx] = k7[accepted_non_recovered]
                        ks_k1_valid[accepted_idx] = True

                if not bool(accepted_local.any()):
                    continue

                idx = idx[accepted_local]
                prev_state = prev_state[accepted_local]
                next_state = next_all[accepted_local]
                step_used = local_step[accepted_local]
                deriv_prev = None
            else:
                next_state, deriv_prev = self._rk4_step_kerr_schild_with_k1(prev_state, float(cfg.step_size))
                finite = torch.isfinite(next_state).all(dim=-1)
                if bool((~finite).any()):
                    bad_idx = idx[~finite]
                    hit_horizon[bad_idx] = True
                    escaped[bad_idx] = False
                    active[bad_idx] = False
                if not bool(finite.any()):
                    continue
                idx = idx[finite]
                prev_state = prev_state[finite]
                next_state = next_state[finite]
                deriv_prev = deriv_prev[finite]
                step_used = torch.full((next_state.shape[0],), float(cfg.step_size), dtype=self.dtype, device=self.device)

            state[idx] = next_state
            if cfg.kerr_schild_null_norm_diagnostic and (step % cfg.kerr_schild_null_norm_interval == 0) and bool(idx.numel() > 0):
                null_norm = self._null_norm_kerr_schild(state[idx])
                max_abs = torch.max(torch.abs(null_norm))
                if torch.isfinite(max_abs):
                    max_abs_value = float(max_abs.item())
                    if max_abs_value > cfg.kerr_schild_null_norm_tol:
                        print(
                            "[kerr-schild] null-norm drift "
                            f"(step={step}, max_abs={max_abs_value:.3e}, tol={cfg.kerr_schild_null_norm_tol:.3e})"
                        )

            prev_xyz = prev_state[:, 1:4]
            next_xyz = next_state[:, 1:4]
            prev_r = self._ks_radius_from_xyz(prev_xyz)
            next_r = self._ks_radius_from_xyz(next_xyz)
            r_min[idx] = torch.minimum(r_min[idx], torch.minimum(prev_r, next_r))

            prev_z = prev_xyz[:, 2]
            next_z = next_xyz[:, 2]
            if cfg.thick_disk and cfg.disk_thickness_ratio > 0.0:
                if deriv_prev is None:
                    deriv_prev = self._rhs_kerr_schild_fn(prev_state)
                deriv_next = self._rhs_kerr_schild_fn(next_state)
                prev_dx, prev_dy, prev_dz = deriv_prev[:, 1], deriv_prev[:, 2], deriv_prev[:, 3]
                next_dx, next_dy, next_dz = deriv_next[:, 1], deriv_next[:, 2], deriv_next[:, 3]
                prev_dr = (
                    prev_xyz[:, 0] * prev_dx + prev_xyz[:, 1] * prev_dy + prev_xyz[:, 2] * prev_dz
                ) / torch.clamp(prev_r, min=1.0e-8)
                next_dr = (
                    next_xyz[:, 0] * next_dx + next_xyz[:, 1] * next_dy + next_xyz[:, 2] * next_dz
                ) / torch.clamp(next_r, min=1.0e-8)

                prev_h, prev_dh = self._disk_half_thickness_and_slope(prev_r)
                next_h, next_dh = self._disk_half_thickness_and_slope(next_r)
                prev_q = torch.abs(prev_z) - prev_h
                next_q = torch.abs(next_z) - next_h
                cross_disk = (prev_q * next_q <= 0.0) | (prev_q <= touch_eps) | (next_q <= touch_eps)

                sign_eps_prev = 0.15 * prev_h + 1.0e-4
                sign_eps_next = 0.15 * next_h + 1.0e-4
                sign_prev = prev_z / torch.sqrt(prev_z * prev_z + sign_eps_prev * sign_eps_prev)
                sign_next = next_z / torch.sqrt(next_z * next_z + sign_eps_next * sign_eps_next)
                dq_prev = sign_prev * prev_dz - prev_dh * prev_dr
                dq_next = sign_next * next_dz - next_dh * next_dr

                alpha_disk = self._refine_event_alpha(
                    prev_q,
                    next_q,
                    dq_prev,
                    dq_next,
                    step_used,
                    torch.zeros_like(prev_q),
                )
                alpha_disk = torch.where(prev_q <= 0.0, torch.zeros_like(alpha_disk), alpha_disk)
                alpha_disk = torch.clamp(alpha_disk, min=0.0, max=1.0)

                h_mid = 0.5 * (prev_h + next_h)
                soft_band = torch.clamp(
                    torch.as_tensor(cfg.disk_vertical_softness, dtype=self.dtype, device=self.device) * h_mid,
                    min=1.0e-4,
                )
                min_q = torch.minimum(prev_q, next_q)
                near_surface = (~cross_disk) & (min_q > 0.0) & (min_q <= soft_band)
                if cfg.vertical_transition_mode == "snap":
                    alpha_near = torch.where(prev_q <= next_q, torch.zeros_like(alpha_disk), torch.ones_like(alpha_disk))
                else:
                    alpha_near = torch.clamp(
                        prev_q / torch.clamp(prev_q + next_q, min=1.0e-8),
                        min=0.0,
                        max=1.0,
                    )
                alpha_disk = torch.where(near_surface, alpha_near, alpha_disk)
                vertical_local = torch.where(
                    cross_disk,
                    torch.ones_like(alpha_disk),
                    torch.where(
                        near_surface,
                        torch.exp(-torch.pow(min_q / torch.clamp(soft_band, min=1.0e-6), 2.0)),
                        torch.zeros_like(alpha_disk),
                    ),
                )
                disk_candidate = cross_disk | near_surface
            else:
                cross_disk = (prev_z * next_z <= 0.0) | (torch.abs(prev_z) < touch_eps) | (torch.abs(next_z) < touch_eps)
                alpha_disk = torch.abs(prev_z) / (torch.abs(prev_z) + torch.abs(next_z) + 1.0e-12)
                alpha_disk = torch.clamp(alpha_disk, min=0.0, max=1.0)
                disk_candidate = cross_disk
                vertical_local = torch.where(cross_disk, torch.ones_like(alpha_disk), torch.zeros_like(alpha_disk))
            xyz_disk = prev_xyz + alpha_disk.unsqueeze(-1) * (next_xyz - prev_xyz)
            r_cross, _, phi_cross = self._cartesian_to_bl(xyz_disk)

            valid_disk = (
                disk_candidate
                & (r_cross >= cfg.disk_inner_radius)
                & (r_cross <= cfg.disk_outer_radius)
                & (r_cross > horizon_cut)
            )
            if not self.enable_disk:
                valid_disk = torch.zeros_like(valid_disk)
            order_local = disk_cross_count[idx] + valid_disk.to(dtype=disk_cross_count.dtype)
            if cfg.multi_hit_disk:
                order_local = torch.clamp(order_local, min=0, max=int(cfg.max_disk_crossings))
            disk_cross_count[idx] = order_local

            alpha_particle = torch.zeros_like(prev_r)
            particle_local = torch.zeros_like(valid_disk)
            if emitter_pos is not None and emitter_radius2 is not None:
                seg = next_xyz - prev_xyz
                seg2 = torch.clamp(torch.sum(seg * seg, dim=-1), min=1.0e-9)
                alpha_best = torch.full_like(prev_r, 2.0)
                for em_idx in range(int(emitter_pos.shape[0])):
                    em_pos = emitter_pos[em_idx]
                    em_r2 = emitter_radius2[em_idx]
                    to_emitter = em_pos.view(1, 3) - prev_xyz
                    alpha_seg = torch.clamp(torch.sum(to_emitter * seg, dim=-1) / seg2, min=0.0, max=1.0)
                    xyz_particle = prev_xyz + alpha_seg.unsqueeze(-1) * seg
                    delta = xyz_particle - em_pos.view(1, 3)
                    dist2 = torch.sum(delta * delta, dim=-1)
                    r_particle = self._ks_radius_from_xyz(xyz_particle)
                    local_hit = (dist2 <= em_r2) & torch.isfinite(dist2) & (r_particle > horizon_cut)
                    better = local_hit & (alpha_seg < alpha_best)
                    alpha_best = torch.where(better, alpha_seg, alpha_best)
                    particle_local = particle_local | local_hit
                alpha_particle = torch.where(particle_local, alpha_best, alpha_particle)

            if has_horizon:
                horizon_cross = ((prev_r > horizon_cut) & (next_r <= horizon_cut)) | (next_r <= horizon_cut)
                horizon_local = horizon_cross & (~valid_disk)
            else:
                horizon_local = torch.zeros_like(valid_disk)
            if self.is_wormhole:
                prev_abs = torch.abs(prev_r)
                next_abs = torch.abs(next_r)
                escape_cross = ((prev_abs < escape_r) & (next_abs >= escape_r)) | (next_abs >= escape_r)
            else:
                escape_cross = ((prev_r < escape_r) & (next_r >= escape_r)) | (next_r >= escape_r)
            escape_local = escape_cross & (~valid_disk) & (~horizon_local)

            alpha_disk_eff = torch.where(valid_disk, alpha_disk, torch.full_like(prev_r, 2.0))
            if has_horizon:
                denom_h = prev_r - next_r
                alpha_h = torch.where(
                    torch.abs(denom_h) > 1.0e-9,
                    (prev_r - horizon_cut) / denom_h,
                    torch.zeros_like(prev_r),
                )
                alpha_h = torch.clamp(alpha_h, min=0.0, max=1.0)
                alpha_h_eff = torch.where(horizon_local, alpha_h, torch.full_like(prev_r, 2.0))
            else:
                alpha_h_eff = torch.full_like(prev_r, 2.0)

            denom_e = next_r - prev_r
            alpha_e = torch.where(
                torch.abs(denom_e) > 1.0e-9,
                (escape_r - prev_r) / denom_e,
                torch.zeros_like(prev_r),
            )
            alpha_e = torch.clamp(alpha_e, min=0.0, max=1.0)
            alpha_e_eff = torch.where(escape_local, alpha_e, torch.full_like(prev_r, 2.0))

            alpha_p_eff = torch.where(particle_local, alpha_particle, torch.full_like(prev_r, 2.0))

            event_alpha = torch.full_like(prev_r, 2.0)
            event_code = torch.zeros_like(prev_r, dtype=torch.int64)

            select_particle = alpha_p_eff < event_alpha
            event_alpha = torch.where(select_particle, alpha_p_eff, event_alpha)
            event_code = torch.where(select_particle, torch.ones_like(event_code), event_code)

            select_disk = alpha_disk_eff < event_alpha
            event_alpha = torch.where(select_disk, alpha_disk_eff, event_alpha)
            event_code = torch.where(select_disk, torch.full_like(event_code, 2), event_code)

            select_horizon = alpha_h_eff < event_alpha
            event_alpha = torch.where(select_horizon, alpha_h_eff, event_alpha)
            event_code = torch.where(select_horizon, torch.full_like(event_code, 3), event_code)

            select_escape = alpha_e_eff < event_alpha
            event_alpha = torch.where(select_escape, alpha_e_eff, event_alpha)
            event_code = torch.where(select_escape, torch.full_like(event_code, 4), event_code)

            any_event = event_code > 0
            if not bool(any_event.any()):
                continue

            state_disk = prev_state + alpha_disk.unsqueeze(-1) * (next_state - prev_state)
            disk_xyz = state_disk[:, 1:4]
            disk_p_t = state_disk[:, 4]
            disk_p_xyz = state_disk[:, 5:8]
            _, disk_theta, disk_phi = self._cartesian_to_bl(disk_xyz)
            _, disk_p_theta, disk_p_phi = self._cartesian_covector_to_bl(
                disk_xyz,
                disk_p_xyz,
                r=r_cross,
                theta=disk_theta,
                phi=disk_phi,
            )

            state_particle = prev_state + alpha_particle.unsqueeze(-1) * (next_state - prev_state)
            particle_xyz = state_particle[:, 1:4]
            particle_p_t = state_particle[:, 4]
            particle_p_xyz = state_particle[:, 5:8]
            pr_p, pth_p, pphi_p = self._cartesian_covector_to_bl(particle_xyz, particle_p_xyz)

            sel_particle = event_code == 1
            if bool(sel_particle.any()):
                hit_idx = idx[sel_particle]
                hit_emitter[hit_idx] = True
                p_t_particle[hit_idx] = particle_p_t[sel_particle]
                p_r_particle[hit_idx] = pr_p[sel_particle]
                p_theta_particle[hit_idx] = pth_p[sel_particle]
                p_phi_particle[hit_idx] = pphi_p[sel_particle]

            sel_disk = event_code == 2
            if bool(sel_disk.any()):
                hit_idx = idx[sel_disk]
                hit_disk[hit_idx] = True
                r_emit[hit_idx] = r_cross[sel_disk]
                p_t_emit[hit_idx] = disk_p_t[sel_disk]
                p_phi_emit[hit_idx] = disk_p_phi[sel_disk]
                p_theta_emit[hit_idx] = disk_p_theta[sel_disk]
                disk_vertical_weight[hit_idx] = vertical_local[sel_disk]
                disk_hit_order[hit_idx] = order_local[sel_disk].to(dtype=self.dtype)
                t_emit[hit_idx] = state_disk[:, 0][sel_disk]
                phi_emit[hit_idx] = phi_cross[sel_disk]

            sel_horizon = event_code == 3
            if bool(sel_horizon.any()):
                hit_horizon[idx[sel_horizon]] = True

            sel_escape = event_code == 4
            if bool(sel_escape.any()):
                escaped[idx[sel_escape]] = True

            active[idx[any_event]] = False

        if bool(active.any()):
            escaped[active] = True

        if cfg.enforce_black_hole_shadow and self.has_horizon:
            absorb_cut = horizon_cut * cfg.shadow_absorb_radius_factor
            captured = (r_min <= absorb_cut) & (~hit_disk) & (~hit_emitter)
            if bool(captured.any()):
                hit_horizon[captured] = True
                escaped[captured] = False

        sky_xyz = state[:, 1:4]
        sky_norm = torch.clamp(torch.linalg.norm(sky_xyz, dim=-1), min=1.0e-8)
        sky_theta = torch.acos(torch.clamp(sky_xyz[:, 2] / sky_norm, min=-1.0, max=1.0))
        sky_theta = torch.clamp(sky_theta, min=THETA_EPS, max=math.pi - THETA_EPS)
        sky_phi = torch.remainder(torch.atan2(sky_xyz[:, 1], sky_xyz[:, 0]), 2.0 * self.pi)
        sky_side = torch.ones_like(sky_phi)
        return (
            hit_disk,
            hit_emitter,
            hit_horizon,
            escaped,
            r_emit,
            p_t_emit,
            p_phi_emit,
            p_theta_emit,
            disk_vertical_weight,
            disk_hit_order,
            t_emit,
            phi_emit,
            p_t_particle,
            p_r_particle,
            p_theta_particle,
            p_phi_particle,
            sky_theta,
            sky_phi,
            sky_side,
            steps_used,
        )

    def _trace_mps_optimized(
        self,
        x_pixel_offset: float = 0.0,
        y_pixel_offset: float = 0.0,
        emitter: EmitterInput = None,
        row_start: int = 0,
        row_end: int | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        int,
    ]:
        """Fast path for MPS: fixed-step RK4 with lightweight event interpolation."""
        cfg = self.config
        state = self._camera_rays(
            x_pixel_offset=x_pixel_offset,
            y_pixel_offset=y_pixel_offset,
            row_start=row_start,
            row_end=row_end,
        )
        n = state.shape[0]

        active = torch.ones(n, dtype=torch.bool, device=self.device)
        hit_disk = torch.zeros(n, dtype=torch.bool, device=self.device)
        hit_emitter = torch.zeros(n, dtype=torch.bool, device=self.device)
        hit_horizon = torch.zeros(n, dtype=torch.bool, device=self.device)
        escaped = torch.zeros(n, dtype=torch.bool, device=self.device)
        r_min = state[:, 1].clone()

        r_emit = torch.zeros(n, dtype=self.dtype, device=self.device)
        p_t_emit = torch.zeros(n, dtype=self.dtype, device=self.device)
        p_phi_emit = torch.zeros(n, dtype=self.dtype, device=self.device)
        p_theta_emit = torch.zeros(n, dtype=self.dtype, device=self.device)
        disk_vertical_weight = torch.zeros(n, dtype=self.dtype, device=self.device)
        disk_hit_order = torch.zeros(n, dtype=self.dtype, device=self.device)
        disk_cross_count = torch.zeros(n, dtype=torch.int16, device=self.device)
        t_emit = torch.zeros(n, dtype=self.dtype, device=self.device)
        phi_emit = torch.zeros(n, dtype=self.dtype, device=self.device)
        p_t_particle = torch.zeros(n, dtype=self.dtype, device=self.device)
        p_r_particle = torch.zeros(n, dtype=self.dtype, device=self.device)
        p_theta_particle = torch.zeros(n, dtype=self.dtype, device=self.device)
        p_phi_particle = torch.zeros(n, dtype=self.dtype, device=self.device)

        emitters = self._normalize_emitters(emitter)
        emitter_pos: torch.Tensor | None = None
        emitter_radius2: torch.Tensor | None = None
        if emitters:
            em_r = torch.as_tensor([em.r for em in emitters], dtype=self.dtype, device=self.device)
            em_theta = torch.as_tensor([em.theta for em in emitters], dtype=self.dtype, device=self.device)
            em_phi = torch.as_tensor([em.phi for em in emitters], dtype=self.dtype, device=self.device)
            emitter_pos = self._bl_to_cartesian(em_r, em_theta, em_phi)
            emitter_radius2 = torch.as_tensor(
                [em.radius * em.radius for em in emitters],
                dtype=self.dtype,
                device=self.device,
            )

        has_horizon = self.has_horizon
        horizon_cut = (
            self.horizon * 1.0005
            if has_horizon
            else torch.as_tensor(-1.0e30, dtype=self.dtype, device=self.device)
        )
        step_h = float(cfg.step_size)
        touch_eps = torch.as_tensor(2.0e-4, dtype=self.dtype, device=self.device)
        escape_r = torch.as_tensor(cfg.escape_radius, dtype=self.dtype, device=self.device)

        steps_used = 0
        for step in range(cfg.max_steps):
            if not bool(active.any()):
                break

            steps_used = step + 1
            idx = torch.nonzero(active, as_tuple=False).squeeze(-1)
            prev_state = state[idx]
            next_state, deriv_prev = self._rk4_step_with_k1(prev_state, step_h)
            if cfg.adaptive_integrator and cfg.adaptive_event_aware:
                prev_r0 = prev_state[:, 1]
                prev_theta0 = prev_state[:, 2]
                event_scale = self._event_aware_step_factor_bl(prev_r0, prev_theta0, horizon_cut)
                near_event = event_scale < 0.28
                if bool(near_event.any()):
                    near_pos = torch.nonzero(near_event, as_tuple=False).squeeze(-1)
                    near_prev = prev_state[near_pos]
                    mid_state, near_k1 = self._rk4_step_with_k1(near_prev, 0.5 * step_h)
                    near_next, _ = self._rk4_step_with_k1(mid_state, 0.5 * step_h)
                    next_state[near_pos] = near_next
                    deriv_prev[near_pos] = near_k1

            finite = torch.isfinite(next_state).all(dim=-1)
            if bool((~finite).any()):
                bad_idx = idx[~finite]
                hit_horizon[bad_idx] = True
                escaped[bad_idx] = False
                active[bad_idx] = False

            if not bool(finite.any()):
                continue

            idx = idx[finite]
            prev_state = prev_state[finite]
            next_state = next_state[finite]
            deriv_prev = deriv_prev[finite]
            state[idx] = next_state

            prev_r = prev_state[:, 1]
            next_r = next_state[:, 1]
            prev_theta = prev_state[:, 2]
            next_theta = next_state[:, 2]
            r_min[idx] = torch.minimum(r_min[idx], torch.minimum(prev_r, next_r))

            deriv_next = self._rhs_fn(next_state)
            step_used = torch.full((next_state.shape[0],), step_h, dtype=self.dtype, device=self.device)
            if cfg.thick_disk and cfg.disk_thickness_ratio > 0.0:
                prev_z = prev_r * torch.cos(prev_theta)
                next_z = next_r * torch.cos(next_theta)
                prev_h, prev_dh = self._disk_half_thickness_and_slope(prev_r)
                next_h, next_dh = self._disk_half_thickness_and_slope(next_r)
                prev_q = torch.abs(prev_z) - prev_h
                next_q = torch.abs(next_z) - next_h
                cross_disk = (prev_q * next_q <= 0.0) | (prev_q <= touch_eps) | (next_q <= touch_eps)

                dr_prev = deriv_prev[:, 1]
                dr_next = deriv_next[:, 1]
                dth_prev = deriv_prev[:, 2]
                dth_next = deriv_next[:, 2]
                dz_prev = dr_prev * torch.cos(prev_theta) - prev_r * torch.sin(prev_theta) * dth_prev
                dz_next = dr_next * torch.cos(next_theta) - next_r * torch.sin(next_theta) * dth_next
                sign_eps_prev = 0.15 * prev_h + 1.0e-4
                sign_eps_next = 0.15 * next_h + 1.0e-4
                sign_prev = prev_z / torch.sqrt(prev_z * prev_z + sign_eps_prev * sign_eps_prev)
                sign_next = next_z / torch.sqrt(next_z * next_z + sign_eps_next * sign_eps_next)
                dq_prev = sign_prev * dz_prev - prev_dh * dr_prev
                dq_next = sign_next * dz_next - next_dh * dr_next

                alpha_disk = self._refine_event_alpha(
                    prev_q,
                    next_q,
                    dq_prev,
                    dq_next,
                    step_used,
                    torch.zeros_like(prev_q),
                )
                alpha_disk = torch.where(prev_q <= 0.0, torch.zeros_like(alpha_disk), alpha_disk)
                alpha_disk = torch.clamp(alpha_disk, min=0.0, max=1.0)

                h_mid = 0.5 * (prev_h + next_h)
                soft_band = torch.clamp(
                    torch.as_tensor(cfg.disk_vertical_softness, dtype=self.dtype, device=self.device) * h_mid,
                    min=1.0e-4,
                )
                min_q = torch.minimum(prev_q, next_q)
                near_surface = (~cross_disk) & (min_q > 0.0) & (min_q <= soft_band)
                if cfg.vertical_transition_mode == "snap":
                    alpha_near = torch.where(prev_q <= next_q, torch.zeros_like(alpha_disk), torch.ones_like(alpha_disk))
                else:
                    alpha_near = torch.clamp(
                        prev_q / torch.clamp(prev_q + next_q, min=1.0e-8),
                        min=0.0,
                        max=1.0,
                    )
                alpha_disk = torch.where(near_surface, alpha_near, alpha_disk)
                vertical_local = torch.where(
                    cross_disk,
                    torch.ones_like(alpha_disk),
                    torch.where(
                        near_surface,
                        torch.exp(-torch.pow(min_q / torch.clamp(soft_band, min=1.0e-6), 2.0)),
                        torch.zeros_like(alpha_disk),
                    ),
                )
                disk_candidate = cross_disk | near_surface
            else:
                prev_equ = prev_theta - self.equatorial
                next_equ = next_theta - self.equatorial
                cross_disk = (prev_equ * next_equ <= 0.0) | (torch.abs(prev_equ) < touch_eps) | (torch.abs(next_equ) < touch_eps)
                alpha_disk = self._refine_event_alpha(
                    prev_theta,
                    next_theta,
                    deriv_prev[:, 2],
                    deriv_next[:, 2],
                    step_used,
                    self.equatorial,
                )
                alpha_disk = torch.clamp(alpha_disk, min=0.0, max=1.0)
                vertical_local = torch.where(cross_disk, torch.ones_like(alpha_disk), torch.zeros_like(alpha_disk))
                disk_candidate = cross_disk
            r_cross = self._hermite_interp(prev_r, next_r, deriv_prev[:, 1], deriv_next[:, 1], step_used, alpha_disk)

            valid_disk = (
                disk_candidate
                & (r_cross >= cfg.disk_inner_radius)
                & (r_cross <= cfg.disk_outer_radius)
                & (r_cross > horizon_cut)
            )
            if not self.enable_disk:
                valid_disk = torch.zeros_like(valid_disk)
            order_local = disk_cross_count[idx] + valid_disk.to(dtype=disk_cross_count.dtype)
            if cfg.multi_hit_disk:
                order_local = torch.clamp(order_local, min=0, max=int(cfg.max_disk_crossings))
            disk_cross_count[idx] = order_local
            alpha_particle = torch.zeros_like(prev_r)
            particle_local = torch.zeros_like(valid_disk)
            if emitter_pos is not None and emitter_radius2 is not None:
                prev_xyz = self._bl_to_cartesian(prev_r, prev_theta, prev_state[:, 3])
                next_xyz = self._bl_to_cartesian(next_r, next_theta, next_state[:, 3])
                seg = next_xyz - prev_xyz
                seg2 = torch.clamp(torch.sum(seg * seg, dim=-1), min=1.0e-9)
                alpha_best = torch.full_like(prev_r, 2.0)
                for em_idx in range(int(emitter_pos.shape[0])):
                    em_pos = emitter_pos[em_idx]
                    em_r2 = emitter_radius2[em_idx]
                    to_emitter = em_pos.view(1, 3) - prev_xyz
                    alpha_seg = torch.clamp(torch.sum(to_emitter * seg, dim=-1) / seg2, min=0.0, max=1.0)
                    closest = prev_xyz + seg * alpha_seg.unsqueeze(-1)
                    delta = closest - em_pos.view(1, 3)
                    dist2 = torch.sum(delta * delta, dim=-1)
                    local_hit = (dist2 <= em_r2) & torch.isfinite(dist2)
                    better = local_hit & (alpha_seg < alpha_best)
                    alpha_best = torch.where(better, alpha_seg, alpha_best)
                    particle_local = particle_local | local_hit
                alpha_particle = torch.where(particle_local, alpha_best, alpha_particle)

            if has_horizon:
                horizon_cross = ((prev_r > horizon_cut) & (next_r <= horizon_cut)) | (next_r <= horizon_cut)
            else:
                horizon_cross = torch.zeros_like(valid_disk)
            if self.is_wormhole:
                prev_abs = torch.abs(prev_r)
                next_abs = torch.abs(next_r)
                escape_cross = ((prev_abs < escape_r) & (next_abs >= escape_r)) | (next_abs >= escape_r)
            else:
                escape_cross = ((prev_r < escape_r) & (next_r >= escape_r)) | (next_r >= escape_r)

            alpha_disk_eff = torch.where(valid_disk, alpha_disk, torch.full_like(prev_r, 2.0))
            alpha_h = torch.clamp((prev_r - horizon_cut) / torch.clamp(prev_r - next_r, min=1.0e-9), min=0.0, max=1.0)
            alpha_h_eff = torch.where(horizon_cross, alpha_h, torch.full_like(prev_r, 2.0))
            alpha_e = torch.clamp((escape_r - prev_r) / torch.clamp(next_r - prev_r, min=1.0e-9), min=0.0, max=1.0)
            alpha_e_eff = torch.where(escape_cross, alpha_e, torch.full_like(prev_r, 2.0))
            alpha_p_eff = torch.where(particle_local, alpha_particle, torch.full_like(prev_r, 2.0))

            event_alpha = torch.full_like(prev_r, 2.0)
            event_code = torch.zeros_like(prev_r, dtype=torch.int64)

            select_particle = alpha_p_eff < event_alpha
            event_alpha = torch.where(select_particle, alpha_p_eff, event_alpha)
            event_code = torch.where(select_particle, torch.ones_like(event_code), event_code)

            select_disk = alpha_disk_eff < event_alpha
            event_alpha = torch.where(select_disk, alpha_disk_eff, event_alpha)
            event_code = torch.where(select_disk, torch.full_like(event_code, 2), event_code)

            select_horizon = alpha_h_eff < event_alpha
            event_alpha = torch.where(select_horizon, alpha_h_eff, event_alpha)
            event_code = torch.where(select_horizon, torch.full_like(event_code, 3), event_code)

            select_escape = alpha_e_eff < event_alpha
            event_alpha = torch.where(select_escape, alpha_e_eff, event_alpha)
            event_code = torch.where(select_escape, torch.full_like(event_code, 4), event_code)

            any_event = event_code > 0
            if not bool(any_event.any()):
                continue

            p_t_interp = prev_state[:, 4] + event_alpha * (next_state[:, 4] - prev_state[:, 4])
            p_r_interp = prev_state[:, 5] + event_alpha * (next_state[:, 5] - prev_state[:, 5])
            p_theta_interp = prev_state[:, 6] + event_alpha * (next_state[:, 6] - prev_state[:, 6])
            p_phi_interp = prev_state[:, 7] + event_alpha * (next_state[:, 7] - prev_state[:, 7])
            t_interp = prev_state[:, 0] + event_alpha * (next_state[:, 0] - prev_state[:, 0])
            if self._mt_shortest_arc_enabled():
                phi_interp = self._interp_phi_shortest_arc(
                    prev_state[:, 3],
                    next_state[:, 3],
                    event_alpha,
                )
            else:
                phi_interp = prev_state[:, 3] + event_alpha * (next_state[:, 3] - prev_state[:, 3])
            r_interp = prev_state[:, 1] + event_alpha * (next_state[:, 1] - prev_state[:, 1])

            sel_particle = event_code == 1
            if bool(sel_particle.any()):
                hit_idx = idx[sel_particle]
                hit_emitter[hit_idx] = True
                p_t_particle[hit_idx] = p_t_interp[sel_particle]
                p_r_particle[hit_idx] = p_r_interp[sel_particle]
                p_theta_particle[hit_idx] = p_theta_interp[sel_particle]
                p_phi_particle[hit_idx] = p_phi_interp[sel_particle]

            sel_disk = event_code == 2
            if bool(sel_disk.any()):
                hit_idx = idx[sel_disk]
                hit_disk[hit_idx] = True
                r_emit[hit_idx] = r_interp[sel_disk]
                p_t_emit[hit_idx] = p_t_interp[sel_disk]
                p_phi_emit[hit_idx] = p_phi_interp[sel_disk]
                p_theta_emit[hit_idx] = p_theta_interp[sel_disk]
                disk_vertical_weight[hit_idx] = vertical_local[sel_disk]
                disk_hit_order[hit_idx] = order_local[sel_disk].to(dtype=self.dtype)
                t_emit[hit_idx] = t_interp[sel_disk]
                phi_emit[hit_idx] = phi_interp[sel_disk]

            sel_horizon = event_code == 3
            if bool(sel_horizon.any()):
                hit_horizon[idx[sel_horizon]] = True

            sel_escape = event_code == 4
            if bool(sel_escape.any()):
                escaped[idx[sel_escape]] = True

            active[idx[any_event]] = False

        if bool(active.any()):
            escaped[active] = True

        if cfg.enforce_black_hole_shadow and self.has_horizon:
            absorb_cut = horizon_cut * cfg.shadow_absorb_radius_factor
            captured = (r_min <= absorb_cut) & (~hit_disk) & (~hit_emitter)
            if bool(captured.any()):
                hit_horizon[captured] = True
                escaped[captured] = False

        sky_theta, sky_phi, sky_side = self._sky_angles_from_state(state)
        return (
            hit_disk,
            hit_emitter,
            hit_horizon,
            escaped,
            r_emit,
            p_t_emit,
            p_phi_emit,
            p_theta_emit,
            disk_vertical_weight,
            disk_hit_order,
            t_emit,
            phi_emit,
            p_t_particle,
            p_r_particle,
            p_theta_particle,
            p_phi_particle,
            sky_theta,
            sky_phi,
            sky_side,
            steps_used,
        )

    def _shade(
        self,
        hit_disk: torch.Tensor,
        hit_emitter: torch.Tensor,
        hit_horizon: torch.Tensor,
        escaped: torch.Tensor,
        r_emit: torch.Tensor,
        p_t_emit: torch.Tensor,
        p_phi_emit: torch.Tensor,
        p_theta_emit: torch.Tensor,
        disk_vertical_weight: torch.Tensor,
        disk_hit_order: torch.Tensor,
        t_emit: torch.Tensor,
        phi_emit: torch.Tensor,
        p_t_particle: torch.Tensor,
        p_r_particle: torch.Tensor,
        p_theta_particle: torch.Tensor,
        p_phi_particle: torch.Tensor,
        sky_theta: torch.Tensor,
        sky_phi: torch.Tensor,
        sky_side: torch.Tensor,
        emitter: EmitterInput = None,
    ) -> torch.Tensor:
        cfg = self.config
        n = hit_disk.shape[0]
        rgb = torch.zeros((n, 3), dtype=self.dtype, device=self.device)
        emitter_ref: PointEmitter | None = None
        emitters = self._normalize_emitters(emitter)
        if emitters:
            emitter_ref = emitters[0]

        if bool(escaped.any()):
            escaped_idx = torch.nonzero(escaped, as_tuple=False).squeeze(-1)
            if cfg.enable_star_background:
                rgb[escaped_idx] = self._star_background(
                    sky_theta[escaped_idx],
                    sky_phi[escaped_idx],
                    side_sign=sky_side[escaped_idx],
                )
            elif cfg.background_mode == "darkspace":
                rgb[escaped_idx] = torch.zeros((escaped_idx.shape[0], 3), dtype=self.dtype, device=self.device)
            else:
                sky = torch.tensor([0.012, 0.018, 0.030], dtype=self.dtype, device=self.device)
                rgb[escaped_idx] = sky

        if emitter_ref is not None and bool(hit_emitter.any()):
            em_idx = torch.nonzero(hit_emitter, as_tuple=False).squeeze(-1)
            p_t = p_t_particle[em_idx]
            p_r = p_r_particle[em_idx]
            p_th = p_theta_particle[em_idx]
            p_phi = p_phi_particle[em_idx]

            u_t = torch.as_tensor(emitter_ref.u_t, dtype=self.dtype, device=self.device)
            u_r = torch.as_tensor(emitter_ref.u_r, dtype=self.dtype, device=self.device)
            u_th = torch.as_tensor(emitter_ref.u_theta, dtype=self.dtype, device=self.device)
            u_phi = torch.as_tensor(emitter_ref.u_phi, dtype=self.dtype, device=self.device)

            denom = -(u_t * p_t + u_r * p_r + u_th * p_th + u_phi * p_phi)
            g_factor = (-p_t) / torch.clamp(denom, min=1.0e-8)
            g_factor = torch.clamp(g_factor, min=0.0, max=3.5)

            base = torch.as_tensor(emitter_ref.color_rgb, dtype=self.dtype, device=self.device).unsqueeze(0).repeat(em_idx.shape[0], 1)
            blue_shift = torch.clamp(g_factor - 1.0, min=0.0, max=1.5)
            red_shift = torch.clamp(1.0 - g_factor, min=0.0, max=1.0)
            color = base.clone()
            color[:, 2] = torch.clamp(color[:, 2] + 0.30 * blue_shift, min=0.0, max=1.5)
            color[:, 0] = torch.clamp(color[:, 0] + 0.28 * red_shift, min=0.0, max=1.5)
            if cfg.enable_emitter_polarization and cfg.magnetic_field_strength != 0.0:
                angle = torch.as_tensor(cfg.faraday_rotation_strength * cfg.magnetic_field_strength, dtype=self.dtype, device=self.device)
                angle = angle / torch.clamp(g_factor, min=0.25)
                c = torch.cos(angle)
                s = torch.sin(angle)
                rr = color[:, 0].clone()
                bb = color[:, 2].clone()
                color[:, 0] = torch.clamp(rr * c - bb * s, min=0.0, max=1.8)
                color[:, 2] = torch.clamp(rr * s + bb * c, min=0.0, max=1.8)

            intensity = torch.clamp(
                torch.as_tensor(emitter_ref.intensity, dtype=self.dtype, device=self.device) * torch.pow(g_factor, 2.0),
                min=0.0,
                max=8.0,
            )
            # Soft highlight mapping avoids a hard white blob while preserving Doppler tint.
            emitter_rgb = color * intensity.unsqueeze(-1)
            rgb[em_idx] = 1.0 - torch.exp(-emitter_rgb)

        if bool(hit_disk.any()):
            disk_idx = torch.nonzero(hit_disk, as_tuple=False).squeeze(-1)
            emission_gain = torch.as_tensor(cfg.disk_emission_gain, dtype=self.dtype, device=self.device)
            r = torch.clamp(r_emit[disk_idx], min=cfg.disk_inner_radius)
            p_t = p_t_emit[disk_idx]
            p_phi = p_phi_emit[disk_idx]
            p_theta_disk = torch.abs(p_theta_emit[disk_idx])
            vertical_w = torch.clamp(disk_vertical_weight[disk_idx], min=0.0, max=1.0)
            vertical_gain = torch.clamp(0.32 + 0.68 * vertical_w, min=0.0, max=1.0)
            order = torch.clamp(disk_hit_order[disk_idx], min=1.0)
            if cfg.multi_hit_disk:
                # Rays that cross the disk multiple times are usually higher-order
                # lensed images (photon-ring vicinity). Boost gently, not explosively.
                order_gain = 1.0 + cfg.lensing_order_strength * torch.pow(torch.log2(order + 1.0), cfg.lensing_order_gamma)
            else:
                order_gain = torch.ones_like(order)
            t_disk = t_emit[disk_idx]
            phi_disk = phi_emit[disk_idx]

            r_profile = r
            if cfg.disk_structure_mode == "concentric_annuli":
                rin = torch.as_tensor(cfg.disk_inner_radius, dtype=self.dtype, device=self.device)
                rout = torch.as_tensor(cfg.disk_outer_radius, dtype=self.dtype, device=self.device)
                span_profile = torch.clamp(rout - rin, min=1.0e-6)
                n_ann = max(4, int(cfg.disk_annuli_count))
                n_ann_t = torch.as_tensor(float(n_ann), dtype=self.dtype, device=self.device)
                dr = span_profile / n_ann_t
                u = torch.clamp((r - rin) / span_profile, min=0.0, max=1.0 - 1.0e-7)
                ring_idx = torch.floor(u * n_ann_t)
                r_ring = rin + (ring_idx + 0.5) * dr
                ring_blend = torch.clamp(
                    torch.as_tensor(cfg.disk_annuli_blend, dtype=self.dtype, device=self.device),
                    min=0.0,
                    max=1.0,
                )
                r_profile = torch.clamp((1.0 - ring_blend) * r + ring_blend * r_ring, min=rin, max=rout)

            theta = torch.full_like(r_profile, math.pi * 0.5)
            metric = metric_components(
                r_profile,
                theta,
                cfg.spin,
                cfg.metric_model,
                cfg.charge,
                cfg.cosmological_constant,
                cfg.wormhole_throat_radius,
                cfg.wormhole_length_scale,
                cfg.wormhole_lensing_scale,
            )

            omega = (
                self._disk_diffrot_omega(r_profile, metric=metric)
                if cfg.enable_disk_differential_rotation
                else self._disk_keplerian_omega(r_profile)
            )
            u_t = torch.rsqrt(torch.clamp(-(metric.g_tt + 2.0 * metric.g_tphi * omega + metric.g_phiphi * omega * omega), min=1.0e-8))

            denom = -(p_t * u_t + p_phi * omega * u_t)
            g_factor = (-p_t) / torch.clamp(denom, min=1.0e-8)
            g_factor = torch.clamp(g_factor, min=0.0, max=6.0)
            beaming = torch.pow(torch.clamp(g_factor, min=0.2, max=6.0), cfg.disk_beaming_strength)
            mu = p_theta_disk / torch.clamp(p_theta_disk + 0.20, min=1.0e-6)
            occlusion = (1.0 - cfg.disk_self_occlusion_strength) + cfg.disk_self_occlusion_strength * torch.sqrt(torch.clamp(mu, min=0.0, max=1.0))
            relativistic_gain = torch.pow(g_factor, 3.0) * beaming * occlusion

            span = cfg.disk_outer_radius - cfg.disk_inner_radius
            x = torch.clamp((r_profile - cfg.disk_inner_radius) / (span + 1.0e-6), min=0.0, max=1.0)

            inner_width = 0.05 * span + 1.0e-3
            outer_width = 0.10 * span + 1.0e-3
            inner_rim = torch.exp(-torch.pow((r_profile - cfg.disk_inner_radius) / inner_width, 2.0))
            outer_rim = torch.exp(-torch.pow((cfg.disk_outer_radius - r_profile) / outer_width, 2.0))
            body = torch.pow(1.0 - x, 0.35)
            layer_complexity = torch.clamp(
                0.52 * torch.pow(1.0 - x, 0.66)
                + 0.24 * torch.pow(torch.clamp(g_factor / 4.0, min=0.0, max=1.5), 0.9)
                + 0.16 * inner_rim
                + 0.08 * torch.clamp(order / torch.clamp(torch.as_tensor(float(max(1, cfg.max_disk_crossings)), dtype=self.dtype, device=self.device), min=1.0), min=0.0, max=1.0),
                min=0.0,
                max=1.0,
            )
            volume_factor = self._disk_volume_emission_factor(r_profile, p_theta_disk)

            disk_model = cfg.disk_model if cfg.physical_disk_model else "legacy"
            if disk_model == "physical_nt":
                rin = torch.as_tensor(cfg.disk_inner_radius, dtype=self.dtype, device=self.device)
                if self._disk_flux_lut_main is not None:
                    rr_lut, flux_lut = self._disk_flux_lut_main
                    r_sample = torch.clamp(r_profile, min=rr_lut[0], max=rr_lut[-1])
                    flux = self._sample_flux_lut(r_sample, rr_lut, flux_lut)
                else:
                    if cfg.disk_radial_profile == "nt_page_thorne":
                        flux = self._novikov_thorne_flux_profile_page_thorne(r_profile, rin)
                    else:
                        flux = self._novikov_thorne_flux_profile(r_profile, rin)
                flux_ref = torch.as_tensor(self._disk_flux_reference, dtype=self.dtype, device=self.device)
                flux_norm = flux / torch.clamp(flux_ref, min=1.0e-8)
                edge_weight = 0.22 + body + cfg.inner_edge_boost * inner_rim + cfg.outer_edge_boost * outer_rim
                intensity = torch.clamp(flux_norm * edge_weight * relativistic_gain, min=0.0, max=40.0)
                intensity = intensity * vertical_gain
                intensity = intensity * order_gain
                intensity = intensity * volume_factor

                temp_profile = torch.pow(torch.clamp(flux_norm, min=1.0e-9), 0.25)
                temp_kelvin = cfg.disk_temperature_inner * temp_profile
                f_col = torch.as_tensor(cfg.disk_color_correction, dtype=self.dtype, device=self.device)
                temp_kelvin = temp_kelvin * f_col
                temp_kelvin = temp_kelvin * (0.85 + 0.35 * torch.pow(torch.clamp(g_factor / 3.5, min=0.0, max=1.5), 1.1))
                color_bb = self._blackbody_rgb(temp_kelvin)
                heat_nt = torch.clamp(
                    torch.pow(1.0 - x, 0.46)
                    + 0.28 * torch.pow(torch.clamp(g_factor / 4.0, min=0.0, max=1.5), 0.9)
                    + 0.20 * inner_rim,
                    min=0.0,
                    max=1.0,
                )
                color_plasma = self._plasma_palette(heat_nt)
                plasma_warmth = torch.as_tensor(cfg.disk_plasma_warmth, dtype=self.dtype, device=self.device)
                color = color_bb * (1.0 - plasma_warmth) + color_plasma * plasma_warmth
                if float(cfg.disk_plasma_warmth) > 0.0:
                    blue_cut = 1.0 - 0.22 * plasma_warmth
                    color = torch.stack([color[:, 0], color[:, 1], color[:, 2] * blue_cut], dim=-1)
                    ion_sheen_nt = torch.stack([0.30 * inner_rim, 0.14 * inner_rim, 0.04 * inner_rim], dim=-1)
                    color = color + plasma_warmth * ion_sheen_nt
                if cfg.disk_palette == "interstellar_warm":
                    warm_color = self._interstellar_warm_palette(x)
                    color = 0.45 * color + 0.55 * warm_color
                if cfg.disk_layered_palette:
                    layered = self._disk_layered_palette_color(
                        x,
                        phi_disk=phi_disk,
                        t_disk=t_disk,
                        layer_complexity=layer_complexity,
                        omega_layer=omega,
                    )
                    layer_mix = torch.as_tensor(cfg.disk_layer_mix, dtype=self.dtype, device=self.device)
                    color = (1.0 - layer_mix) * color + layer_mix * layered
                if cfg.disk_segmented_palette:
                    seg_color = self._disk_segmented_color(
                        x,
                        phi_disk=phi_disk,
                        t_disk=t_disk,
                        omega=omega,
                    )
                    seg_mix = torch.as_tensor(cfg.disk_segmented_mix, dtype=self.dtype, device=self.device)
                    color = (1.0 - seg_mix) * color + seg_mix * seg_color
                if cfg.enable_disk_differential_rotation and cfg.disk_diffrot_visual_mode != "layer_phase":
                    phase_vis = torch.remainder(phi_disk - omega * t_disk, 2.0 * self.pi)
                    if cfg.disk_structure_mode == "concentric_annuli":
                        n_vis = max(4, int(cfg.disk_annuli_count))
                    else:
                        n_vis = max(4, int(cfg.disk_layer_count))
                    layer_key_vis = torch.floor(torch.clamp(x, min=0.0, max=1.0 - 1.0e-7) * float(n_vis))
                    _, tile_gain_vis = self._disk_diffrot_phase_terms(phase_vis, layer_key_vis, x)
                    color = color * torch.clamp(0.88 + 0.12 * tile_gain_vis, min=0.5, max=1.5).unsqueeze(-1)
                color = color * (0.90 + 0.10 * torch.sqrt(torch.clamp(volume_factor, min=0.0)).unsqueeze(-1))
                dilution = 1.0 / torch.clamp(torch.pow(f_col, 4.0), min=1.0)
                disk_rgb = color * (dilution * 1.05 * intensity.unsqueeze(-1))
            else:
                radial_term = torch.clamp(1.0 - torch.sqrt(torch.clamp(cfg.disk_inner_radius / r_profile, max=1.0)), min=0.0)
                emissivity = torch.pow(r_profile / cfg.disk_inner_radius, -cfg.emissivity_index) * radial_term
                intensity = emissivity * relativistic_gain
                intensity = intensity * (0.35 + body + cfg.inner_edge_boost * inner_rim + cfg.outer_edge_boost * outer_rim)

                # Discontinuous azimuthal plasma sectors in a co-rotating frame to visualize disk motion.
                omega = (
                    self._disk_diffrot_omega(r_profile, metric=metric)
                    if cfg.enable_disk_differential_rotation
                    else self._disk_keplerian_omega(r_profile)
                )
                phase = torch.remainder(phi_disk - omega * t_disk, 2.0 * self.pi)
                if cfg.enable_disk_differential_rotation:
                    if cfg.disk_structure_mode == "concentric_annuli":
                        n_ann = max(4, int(cfg.disk_annuli_count))
                        layer_key = torch.floor(torch.clamp(x, min=0.0, max=1.0 - 1.0e-7) * float(n_ann))
                    else:
                        layer_key = torch.floor(torch.clamp(x, min=0.0, max=1.0) * 8.0)
                    phase, tile_gain = self._disk_diffrot_phase_terms(phase, layer_key, x)
                else:
                    tile_gain = torch.ones_like(phase)
                sector_wave = torch.sin(14.0 * phase + 9.0 * (1.0 - x))
                sector_mask = (sector_wave > 0.0).to(self.dtype)
                fine_band = (torch.sin(22.0 * x + 1.6 * phase) > 0.35).to(self.dtype)
                discontinuity = torch.clamp(0.18 + 0.82 * sector_mask * (0.65 + 0.35 * fine_band), min=0.0, max=1.0)
                if cfg.enable_disk_differential_rotation and cfg.disk_diffrot_visual_mode != "layer_phase":
                    discontinuity = discontinuity * tile_gain
                intensity = intensity * (0.35 + 1.25 * discontinuity)
                intensity = intensity * vertical_gain
                intensity = intensity * order_gain
                intensity = intensity * volume_factor

                intensity = torch.clamp(intensity, min=0.0, max=30.0)
                heat = torch.clamp(torch.pow(1.0 - x, 0.42) + 0.20 * torch.pow(torch.clamp(g_factor / 3.0, min=0.0, max=1.0), 1.2), min=0.0, max=1.0)
                phase_wave = 0.5 + 0.5 * torch.sin(14.0 * phase + 0.9 * torch.sin(2.0 * phase))
                if cfg.enable_disk_differential_rotation and cfg.disk_diffrot_visual_mode == "hybrid":
                    phase_wave = torch.clamp(0.65 * phase_wave + 0.35 * tile_gain, min=0.0, max=1.0)
                hard_switch = (phase_wave > 0.52).to(self.dtype).unsqueeze(-1)
                heat_hot = torch.clamp(heat + 0.22 * discontinuity + 0.18 * phase_wave, min=0.0, max=1.0)
                heat_cool = torch.clamp(heat - 0.18 + 0.08 * (1.0 - phase_wave), min=0.0, max=1.0)
                color_hot = self._plasma_palette(heat_hot)
                color_cool = self._plasma_palette(heat_cool)
                color = color_cool * (1.0 - hard_switch) + color_hot * hard_switch
                rim = torch.exp(-torch.pow((r_profile - cfg.disk_inner_radius) / (0.06 * span + 1.0e-3), 2.0))
                ion_sheen = torch.stack([0.55 * rim, 0.34 * rim, 0.14 * rim], dim=-1)
                white_core = torch.pow(torch.clamp(g_factor / 4.5, min=0.0, max=1.0), 2.0).unsqueeze(-1)
                color = color + ion_sheen + 0.40 * white_core
                color = color * (0.90 + 0.10 * torch.sqrt(torch.clamp(volume_factor, min=0.0)).unsqueeze(-1))
                disk_rgb = color * (0.95 * intensity.unsqueeze(-1))
            disk_rgb = disk_rgb * emission_gain
            rgb[disk_idx] = disk_rgb

        shadow = torch.tensor([0.0, 0.0, 0.0], dtype=self.dtype, device=self.device)
        rgb[hit_horizon] = shadow

        rgb = torch.nan_to_num(rgb, nan=0.0, posinf=1.0, neginf=0.0)
        rgb = self._tone_map(rgb)
        rgb = torch.pow(torch.clamp(rgb, min=0.0, max=1.0), 1.0 / 2.2)
        return rgb

    def _tone_map(self, rgb_linear: torch.Tensor) -> torch.Tensor:
        rgb = torch.clamp(rgb_linear, min=0.0) * float(self.config.tone_exposure)
        if self.config.tone_mapper == "aces":
            # Narkowicz 2015 approximation for ACES filmic curve.
            a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
            mapped = (rgb * (a * rgb + b)) / torch.clamp(rgb * (c * rgb + d) + e, min=1.0e-8)
        elif self.config.tone_mapper == "filmic":
            # Hable/Uncharted2 filmic curve with configurable white point.
            A, B, C, D, E, Fv = 0.15, 0.50, 0.10, 0.20, 0.02, 0.30
            mapped = ((rgb * (A * rgb + C * B) + D * E) / torch.clamp(rgb * (A * rgb + B) + D * Fv, min=1.0e-8)) - (E / Fv)
            w = float(self.config.tone_white_point)
            white = ((w * (A * w + C * B) + D * E) / max(1.0e-8, (w * (A * w + B) + D * Fv))) - (E / Fv)
            mapped = mapped / max(1.0e-8, white)
        else:
            mapped = rgb / (1.0 + rgb)

        roll = float(self.config.tone_highlight_rolloff)
        if roll > 0.0:
            mapped = mapped / (1.0 + roll * torch.pow(torch.clamp(mapped, min=0.0), 1.35))
        return torch.clamp(mapped, min=0.0, max=1.0)

    def _apply_postprocess_pipeline(self, rgb: torch.Tensor) -> torch.Tensor:
        pipeline = self.config.postprocess_pipeline
        strength = float(self.config.gargantua_look_strength)
        if pipeline == "off" or strength <= 0.0:
            return torch.clamp(rgb, min=0.0, max=1.0)
        if pipeline != "gargantua":
            return torch.clamp(rgb, min=0.0, max=1.0)
        return self._postprocess_gargantua(rgb, strength)

    def _postprocess_wormhole_seam(self, img: np.ndarray) -> np.ndarray:
        """Remove the vertical seam artifact from wormhole renders.

        The seam appears at the critical impact parameter boundary (b = b₀)
        where rays transition from local to remote background. It is detected
        automatically as the column with the highest horizontal gradient in the
        background region (excluding the wormhole shadow area), then smoothed
        with a horizontal Gaussian blend using a cosine-window mask.

        Parameters controlled by config fields:
          wormhole_seam_sigma      – Gaussian sigma in pixels (default 30)
          wormhole_seam_blend_half – half-width of the blend band in pixels (default 60)
        """
        h, w = img.shape[:2]
        arr = img.astype(np.float32)

        sigma = float(self.config.wormhole_seam_sigma)
        blend_half = int(self.config.wormhole_seam_blend_half)

        # --- Seam detection ---
        # Use only the peripheral rows (top+bottom quarters) where the background
        # is most visible and the wormhole shadow does not dominate.
        row_bands = np.concatenate([
            arr[:h // 4, :, :],
            arr[3 * h // 4:, :, :],
        ], axis=0)
        horiz_grad = np.mean(np.abs(np.diff(row_bands, axis=1)), axis=(0, 2))

        # Exclude a central exclusion zone around x=w//2 (wormhole shadow rim)
        excl_half = int(w * 0.14)
        cx = w // 2
        lo_excl = max(0, cx - excl_half)
        hi_excl = min(w - 2, cx + excl_half)
        horiz_grad[lo_excl:hi_excl] = 0.0
        # Also exclude the first and last 5% of the image width (edge artifacts)
        margin = max(1, w // 20)
        horiz_grad[:margin] = 0.0
        horiz_grad[-margin:] = 0.0

        seam_col = int(np.argmax(horiz_grad))

        # --- Horizontal Gaussian blur (1D kernel, reflect padding) ---
        radius = int(4.0 * sigma + 0.5)
        kx = np.arange(-radius, radius + 1, dtype=np.float64)
        kernel = np.exp(-0.5 * (kx / sigma) ** 2)
        kernel = (kernel / kernel.sum()).astype(np.float32)

        padded = np.pad(arr, ((0, 0), (radius, radius), (0, 0)), mode="reflect")
        blurred = np.zeros_like(arr)
        for i, k in enumerate(kernel):
            blurred += k * padded[:, i : i + w, :]

        # --- Cosine-window blend mask (0 at band edges → 1 at seam_col) ---
        x_lo = max(0, seam_col - blend_half)
        x_hi = min(w, seam_col + blend_half + 1)
        xs = np.arange(x_lo, x_hi, dtype=np.float32)
        alpha = 0.5 * (1.0 - np.cos(np.pi * (xs - x_lo) / max(1.0, x_hi - x_lo - 1)))
        alpha = alpha[np.newaxis, :, np.newaxis]  # (1, band_w, 1)

        result = arr.copy()
        result[:, x_lo:x_hi, :] = (
            (1.0 - alpha) * arr[:, x_lo:x_hi, :]
            + alpha * blurred[:, x_lo:x_hi, :]
        )
        return np.clip(result, 0.0, 255.0).astype(np.uint8)

    def _postprocess_gargantua(self, rgb: torch.Tensor, strength: float) -> torch.Tensor:
        s = max(0.0, min(2.0, float(strength)))
        x = torch.clamp(rgb, min=0.0, max=1.0)

        luma = 0.2126 * x[:, :, 0] + 0.7152 * x[:, :, 1] + 0.0722 * x[:, :, 2]
        warm_vec = torch.tensor(
            [1.0 + 0.08 * s, 1.0 + 0.03 * s, 1.0 - 0.12 * s],
            dtype=self.dtype,
            device=self.device,
        )
        x = torch.clamp(x * warm_vec.view(1, 1, 3), min=0.0, max=1.0)

        hmask = torch.pow(torch.clamp(luma, min=0.0, max=1.0), 0.70).unsqueeze(-1)
        hi_tint = torch.tensor([1.05, 1.01, 0.93], dtype=self.dtype, device=self.device)
        hi_mix = 0.25 * s
        x = (1.0 - hi_mix * hmask) * x + (hi_mix * hmask) * (x * hi_tint.view(1, 1, 3))
        x = torch.clamp(x, min=0.0, max=1.0)

        x_nchw = x.permute(2, 0, 1).unsqueeze(0)
        luma_n = 0.2126 * x_nchw[:, 0:1, :, :] + 0.7152 * x_nchw[:, 1:2, :, :] + 0.0722 * x_nchw[:, 2:3, :, :]
        thr = 0.60
        bright = torch.clamp((luma_n - thr) / max(1.0e-6, 1.0 - thr), min=0.0, max=1.0)
        bright_rgb = x_nchw * bright
        blur_a = F.avg_pool2d(bright_rgb, kernel_size=5, stride=1, padding=2)
        blur_b = F.avg_pool2d(bright_rgb, kernel_size=11, stride=1, padding=5)
        bloom = 0.65 * blur_a + 0.35 * blur_b
        x_nchw = torch.clamp(x_nchw + (0.22 * s) * bloom, min=0.0, max=1.0)

        contrast = 1.0 + 0.18 * s
        x_nchw = torch.clamp((x_nchw - 0.5) * contrast + 0.5, min=0.0, max=1.0)
        x_nchw = torch.pow(torch.clamp(x_nchw, min=0.0, max=1.0), 1.0 / (1.0 + 0.08 * s))
        x = x_nchw.squeeze(0).permute(1, 2, 0)

        h, w = x.shape[0], x.shape[1]
        yy = torch.linspace(-1.0, 1.0, h, dtype=self.dtype, device=self.device).view(h, 1)
        xx = torch.linspace(-1.0, 1.0, w, dtype=self.dtype, device=self.device).view(1, w)
        rr = torch.sqrt(xx * xx + yy * yy)
        vignette = torch.clamp(1.0 - (0.18 * s) * torch.pow(rr, 1.65), min=0.65, max=1.0)
        x = x * vignette.unsqueeze(-1)

        ring = torch.exp(-torch.pow((rr - 0.30) / 0.12, 2.0))
        equator = torch.exp(-torch.pow(yy / 0.26, 2.0)).expand_as(rr)
        halo = ring * equator
        halo_col = torch.tensor([1.0, 0.92, 0.78], dtype=self.dtype, device=self.device)
        x = torch.clamp(x + (0.08 * s) * halo.unsqueeze(-1) * halo_col.view(1, 1, 3), min=0.0, max=1.0)

        luma2 = 0.2126 * x[:, :, 0] + 0.7152 * x[:, :, 1] + 0.0722 * x[:, :, 2]
        sat = 1.0 + 0.10 * s
        x = torch.clamp(luma2.unsqueeze(-1) + (x - luma2.unsqueeze(-1)) * sat, min=0.0, max=1.0)
        return x

    def _destripe_meridian(
        self,
        rgb: torch.Tensor,
        escaped_mask: torch.Tensor | None = None,
        force: bool = False,
    ) -> torch.Tensor:
        h, w = rgb.shape[0], rgb.shape[1]
        if w < 6:
            return rgb

        # Detect strongest discontinuity across adjacent columns (with periodic wrap).
        edge = torch.mean(torch.abs(rgb - torch.roll(rgb, shifts=1, dims=1)), dim=(0, 2))
        edge_peak = torch.max(edge)
        edge_med = torch.median(edge)
        if not bool(torch.isfinite(edge_peak)):
            return rgb
        # Ignore normal texture variation; only treat outlier seam columns.
        if (not force) and bool(edge_peak <= (edge_med * 1.45 + 8.0e-4)):
            return rgb

        seam_col = int(torch.argmax(edge).item())
        out = rgb.clone()
        base = rgb

        if escaped_mask is not None and escaped_mask.numel() == h * w:
            mask2d = escaped_mask.reshape(h, w)
        else:
            mask2d = torch.ones((h, w), dtype=torch.bool, device=rgb.device)

        offsets = (-3, -2, -1, 0, 1, 2, 3)
        strengths = (0.25, 0.45, 0.70, 0.90, 0.70, 0.45, 0.25)
        for off, alpha in zip(offsets, strengths):
            c = (seam_col + off) % w
            l = (c - 1) % w
            r = (c + 1) % w
            blended = 0.5 * (base[:, l, :] + base[:, r, :])
            softened = (1.0 - alpha) * base[:, c, :] + alpha * blended
            mask = mask2d[:, c].unsqueeze(-1)
            out[:, c, :] = torch.where(mask, softened, out[:, c, :])

        return out

    def _soften_center_columns(self, rgb: torch.Tensor) -> torch.Tensor:
        h, w = rgb.shape[0], rgb.shape[1]
        if w < 6:
            return rgb
        out = rgb.clone()
        centers = {w // 2}
        if (w % 2) == 0:
            centers.add((w // 2) - 1)
        for c in centers:
            for off, alpha in ((-1, 0.25), (0, 0.55), (1, 0.25)):
                col = (c + off) % w
                left = (col - 1) % w
                right = (col + 1) % w
                blend = 0.5 * (rgb[:, left, :] + rgb[:, right, :])
                out[:, col, :] = (1.0 - alpha) * rgb[:, col, :] + alpha * blend
        return out

    def _meridian_destripe_passes(self) -> int:
        """Return how many seam-destripe passes to apply on the final RGB."""
        if self.is_wormhole:
            # Morris-Thorne is more prone to vertical seam artifacts around the
            # local/remote sky transition, so apply a stronger default cleanup.
            return 2
        if self.config.destripe_meridian:
            return 1
        return 0

    def _wormhole_mt_beam_offsets(self) -> list[float]:
        samples = max(0, min(4, int(self.config.wormhole_mt_beam_samples)))
        if samples <= 0:
            return []
        jitter = float(self.config.wormhole_mt_beam_jitter)
        # Symmetric horizontal jitter pattern around the camera center ray.
        base = [1.0, -1.0, 0.5, -0.5, 0.75, -0.75, 0.25, -0.25]
        count = min(len(base), 2 * samples)
        return [jitter * base[i] for i in range(count)]

    def _build_wormhole_mt_beam_mask(
        self,
        rgb: torch.Tensor,
        escaped_mask: torch.Tensor | None = None,
        hit_horizon_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = int(rgb.shape[0])
        w = int(rgb.shape[1])
        if (not self.is_wormhole) or w < 12 or h < 12:
            return torch.zeros((h, w), dtype=torch.bool, device=rgb.device)

        edge = torch.mean(torch.abs(rgb - torch.roll(rgb, shifts=1, dims=1)), dim=(0, 2))
        finite = torch.isfinite(edge)
        if not bool(torch.any(finite)):
            return torch.zeros((h, w), dtype=torch.bool, device=rgb.device)
        edge = torch.where(finite, edge, torch.zeros_like(edge))

        edge_med = float(torch.median(edge).item())
        edge_peak = float(torch.max(edge).item())
        if (not math.isfinite(edge_peak)) or edge_peak <= 0.0:
            return torch.zeros((h, w), dtype=torch.bool, device=rgb.device)

        threshold_ratio = float(self.config.wormhole_mt_beam_threshold)
        threshold_value = max(1.0e-6, edge_med * threshold_ratio)
        seam_cols = edge >= threshold_value

        if not bool(torch.any(seam_cols)):
            fallback_gate = edge_med * max(1.15, 0.80 * threshold_ratio) + 1.0e-4
            if edge_peak <= fallback_gate:
                return torch.zeros((h, w), dtype=torch.bool, device=rgb.device)
            seam_cols = torch.zeros((w,), dtype=torch.bool, device=rgb.device)
            seam_cols[int(torch.argmax(edge).item())] = True

        seam_indices = torch.nonzero(seam_cols, as_tuple=False).squeeze(-1)
        if int(seam_indices.numel()) > 3:
            topk = torch.topk(edge, k=3, largest=True).indices
            seam_indices = topk

        band_halfwidth = int(self.config.wormhole_mt_beam_band_halfwidth)
        col_mask = torch.zeros((w,), dtype=torch.bool, device=rgb.device)
        for seam_col_t in seam_indices:
            seam_col = int(seam_col_t.item())
            for off in range(-band_halfwidth, band_halfwidth + 1):
                col_mask[(seam_col + off) % w] = True
        mask2d = col_mask.unsqueeze(0).expand(h, w)

        trace_mask: torch.Tensor | None = None
        if escaped_mask is not None and escaped_mask.numel() == h * w:
            trace_mask = escaped_mask.reshape(h, w)
        if hit_horizon_mask is not None and hit_horizon_mask.numel() == h * w:
            hit2d = hit_horizon_mask.reshape(h, w)
            trace_mask = hit2d if trace_mask is None else (trace_mask | hit2d)
        if trace_mask is not None and bool(torch.any(trace_mask)):
            mask2d = mask2d & trace_mask

        return mask2d

    def _adaptive_row_scales_from_preview(
        self,
        preview_rgb: torch.Tensor,
        min_scale: float,
        quantile: float,
    ) -> torch.Tensor:
        lum = (
            0.2126 * preview_rgb[:, :, 0]
            + 0.7152 * preview_rgb[:, :, 1]
            + 0.0722 * preview_rgb[:, :, 2]
        )
        dx = torch.abs(lum[:, 1:] - lum[:, :-1])
        dy = torch.abs(lum[1:, :] - lum[:-1, :])
        grad = torch.zeros_like(lum)
        grad[:, :-1] = grad[:, :-1] + dx
        grad[:-1, :] = grad[:-1, :] + dy
        row_energy = grad.mean(dim=1)
        if int(row_energy.numel()) == 0:
            return torch.ones(0, dtype=self.dtype, device=self.device)

        q = float(torch.quantile(row_energy, float(quantile)).item())
        if (not math.isfinite(q)) or q <= 1.0e-9:
            return torch.ones_like(row_energy)

        normalized = torch.clamp(row_energy / q, min=0.0, max=1.0)
        row_scale = float(min_scale) + (1.0 - float(min_scale)) * normalized
        row_scale = torch.clamp(row_scale, min=float(min_scale), max=1.0)

        if int(row_scale.numel()) > 1:
            expanded = row_scale.clone()
            expanded[1:] = torch.maximum(expanded[1:], row_scale[:-1])
            expanded[:-1] = torch.maximum(expanded[:-1], row_scale[1:])
            row_scale = torch.clamp(expanded, min=float(min_scale), max=1.0)
        return row_scale

    def _roi_supersample_offsets(self) -> list[tuple[float, float]]:
        j = float(self.config.roi_supersample_jitter)
        base = [
            (+j, +j),
            (-j, +j),
            (+j, -j),
            (-j, -j),
            (+0.5 * j, -0.5 * j),
            (-0.5 * j, +0.5 * j),
            (+0.75 * j, 0.0),
            (0.0, +0.75 * j),
        ]
        count = max(1, min(len(base), int(self.config.roi_supersample_samples)))
        return base[:count]

    def _build_roi_supersample_mask(self, rgb: torch.Tensor) -> torch.Tensor:
        h = int(rgb.shape[0])
        w = int(rgb.shape[1])
        if h < 64 or w < 64:
            return torch.zeros((h, w), dtype=torch.bool, device=self.device)

        lum = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
        gx = torch.zeros_like(lum)
        gy = torch.zeros_like(lum)
        gx[:, :-1] = torch.abs(lum[:, 1:] - lum[:, :-1])
        gy[:-1, :] = torch.abs(lum[1:, :] - lum[:-1, :])
        grad = torch.maximum(gx, gy)

        try:
            q = float(torch.quantile(grad.reshape(-1), float(self.config.roi_supersample_threshold)).item())
        except Exception:
            q = 0.0
        grad_mask = grad >= max(q, 1.0e-5)

        yy = torch.linspace(-1.0, 1.0, h, dtype=self.dtype, device=self.device).view(h, 1)
        xx = torch.linspace(-1.0, 1.0, w, dtype=self.dtype, device=self.device).view(1, w)
        rr = torch.sqrt(xx * xx + yy * yy)
        ring = torch.abs(rr - 0.34) <= 0.12
        core = rr <= 0.18
        roi = grad_mask | ring | core

        roi_f = roi.to(self.dtype).unsqueeze(0).unsqueeze(0)
        roi_f = F.max_pool2d(roi_f, kernel_size=5, stride=1, padding=2)
        return roi_f[0, 0] > 0.0

    def _finalize_rgb_cuda_graph(self, rgb: torch.Tensor) -> torch.Tensor:
        def _fallback(x: torch.Tensor) -> torch.Tensor:
            post = self._apply_postprocess_pipeline(x)
            return torch.clamp(post * 255.0, min=0.0, max=255.0).to(torch.uint8)

        if self.device.type != "cuda":
            return _fallback(rgb)
        if self._cuda_finalize_graph_failed:
            return _fallback(rgb)

        use_graph = bool(self.config.cuda_graph_finalize)
        if (not use_graph) or (not hasattr(torch.cuda, "CUDAGraph")):
            return _fallback(rgb)

        if (not self._cuda_finalize_graph_ready) or self._cuda_finalize_static_in is None or self._cuda_finalize_static_out is None:
            try:
                static_in = torch.empty_like(rgb)
                stream = torch.cuda.Stream(device=self.device)
                stream.wait_stream(torch.cuda.current_stream(device=self.device))
                with torch.cuda.stream(stream):
                    static_in.copy_(rgb)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        post = self._apply_postprocess_pipeline(static_in)
                        static_out = torch.clamp(post * 255.0, min=0.0, max=255.0).to(torch.uint8)
                torch.cuda.current_stream(device=self.device).wait_stream(stream)
                self._cuda_finalize_graph = graph
                self._cuda_finalize_static_in = static_in
                self._cuda_finalize_static_out = static_out
                self._cuda_finalize_graph_ready = True
            except Exception:
                self._cuda_finalize_graph_failed = True
                return _fallback(rgb)

        assert self._cuda_finalize_graph is not None
        assert self._cuda_finalize_static_in is not None
        assert self._cuda_finalize_static_out is not None
        self._cuda_finalize_static_in.copy_(rgb)
        self._cuda_finalize_graph.replay()
        return self._cuda_finalize_static_out.clone()

    def _format_eta(self, seconds: float) -> str:
        secs = max(0, int(round(seconds)))
        hours, rem = divmod(secs, 3600)
        minutes, sec = divmod(rem, 60)
        if hours > 0:
            return f"{hours:d}:{minutes:02d}:{sec:02d}"
        return f"{minutes:02d}:{sec:02d}"

    def _print_row_progress(
        self,
        rows_done: int,
        rows_total: int,
        start_time: float | None = None,
        eta_seconds: float | None = None,
        finalize: bool = False,
    ) -> None:
        if rows_total <= 0:
            return
        ratio = max(0.0, min(1.0, float(rows_done) / float(rows_total)))
        bar_width = 30
        filled = int(round(ratio * bar_width))
        empty = max(0, bar_width - filled)
        fill_char, empty_char = self._progress_bar_chars()
        bar = (fill_char * filled) + (empty_char * empty)
        eta_txt = ""
        if eta_seconds is not None:
            eta_txt = f" ETA ~{self._format_eta(max(0.0, float(eta_seconds)))}"
        elif start_time is not None:
            if rows_done >= rows_total:
                eta_txt = " ETA ~00:00"
            elif rows_done > 0:
                elapsed = max(0.0, time.perf_counter() - float(start_time))
                remaining = elapsed * float(rows_total - rows_done) / float(rows_done)
                eta_txt = f" ETA ~{self._format_eta(remaining)}"
        msg = f"Render rows [{bar}] {ratio * 100.0:6.2f}% ({rows_done:4d}/{rows_total:4d}){eta_txt}"
        if sys.stdout.isatty():
            print(f"\r{msg}", end="\n" if finalize else "", flush=True)
            return
        if finalize:
            print(msg, flush=True)

    def _progress_bar_chars(self) -> tuple[str, str]:
        encoding = (sys.stdout.encoding or "").lower()
        if "utf" in encoding:
            return "█", "░"
        return "#", "-"

    def _resolve_progress_backend(self) -> str:
        backend = str(getattr(self.config, "progress_backend", "manual")).lower()
        if backend == "auto":
            if tqdm_auto is not None and sys.stdout.isatty():
                return "tqdm"
            return "manual"
        if backend == "tqdm" and tqdm_auto is None:
            if sys.stdout.isatty():
                print("tqdm backend requested but tqdm is not installed; falling back to manual progress.", flush=True)
            return "manual"
        return backend

    def _clear_device_cache(self) -> None:
        """Best-effort cache cleanup to limit long-run memory growth on GPU backends."""
        try:
            if self.device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            if self.device.type == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except Exception:
            pass

    @torch.inference_mode()
    def render(
        self,
        x_pixel_offset: float = 0.0,
        y_pixel_offset: float = 0.0,
        emitter: EmitterInput = None,
        row_progress_callback: Callable[[int, int], None] | None = None,
    ) -> RenderOutput:
        emitter_list = self._normalize_emitters(emitter)
        emitter_payload: EmitterInput
        if not emitter_list:
            emitter_payload = None
        elif len(emitter_list) == 1:
            emitter_payload = emitter_list[0]
        else:
            emitter_payload = emitter_list

        amp_ctx = nullcontext()
        if self.config.mixed_precision and self.device.type in {"cuda", "mps"}:
            try:
                amp_ctx = torch.autocast(device_type=self.device.type, dtype=torch.float16)
            except Exception:
                amp_ctx = nullcontext()

        if self.config.coordinate_system in {"kerr_schild", "generalized_doran"}:
            trace_fn = self._trace_kerr_schild
        else:
            if self.is_wormhole and bool(self.config.wormhole_mt_force_reference_trace):
                trace_fn = self._trace
            elif emitter_payload is not None:
                if self.use_mps_optimized_kernel and self.config.allow_mps_emitter_fastpath:
                    trace_fn = self._trace_mps_optimized
                else:
                    trace_fn = self._trace
            else:
                trace_fn = self._trace_mps_optimized if self.use_mps_optimized_kernel else self._trace

        h = self.config.height
        w = self.config.width
        tile_rows_cfg = int(self.config.render_tile_rows)
        if tile_rows_cfg > 0:
            tile_rows = min(h, tile_rows_cfg)
        elif self.use_mps_optimized_kernel and self.config.mps_auto_chunking and h >= 480:
            # MPS path benefits from moderate row chunking to avoid very large kernels
            # and keep compile/replay overhead predictable.
            target_chunks = 12 if h >= 432 else 8
            tile_rows = max(1, min(h, math.ceil(h / target_chunks)))
        elif row_progress_callback is not None and h > 1:
            # Keep row callbacks reasonably frequent for external progress sinks.
            target_chunks = 12 if h >= 600 else 10
            tile_rows = max(1, min(h, math.ceil(h / target_chunks)))
        elif self.config.adaptive_spatial_sampling and h > 1:
            # Adaptive per-row scheduling needs multiple row blocks even when the
            # progress bar is disabled.
            tile_rows = max(1, min(h, math.ceil(h / 8)))
        elif self.config.show_progress_bar and h > 1:
            # Keep chunks coarse to reduce per-tile overhead while preserving progress updates.
            target_chunks = 4 if self.config.meridian_supersample else 6
            tile_rows = max(1, min(h, math.ceil(h / target_chunks)))
        else:
            tile_rows = h
        use_tiling = tile_rows < h
        progress_enabled = bool(self.config.show_progress_bar and threading.current_thread() is threading.main_thread())
        progress_backend = self._resolve_progress_backend() if progress_enabled else "manual"
        use_tqdm_progress = bool(progress_enabled and progress_backend == "tqdm" and tqdm_auto is not None)
        use_manual_progress = bool(progress_enabled and not use_tqdm_progress)
        progress_start = time.perf_counter()
        rows_done = 0
        use_click_progress = False
        progress_print_lock = threading.Lock()
        progress_state_lock = threading.Lock()
        progress_stop = threading.Event()
        eta_base_seconds: float | None = None
        eta_base_time = progress_start

        def _notify_row_progress(rows_value: int) -> None:
            if row_progress_callback is None:
                return
            try:
                rows_clamped = max(0, min(h, int(rows_value)))
                row_progress_callback(rows_clamped, h)
            except Exception:
                # Progress callback is best-effort and must never break rendering.
                pass

        def _emit_progress(rows_now: int, eta_now: float | None, finalize_now: bool = False) -> None:
            with progress_print_lock:
                self._print_row_progress(
                    rows_done=rows_now,
                    rows_total=h,
                    start_time=progress_start if eta_now is None else None,
                    eta_seconds=eta_now,
                    finalize=finalize_now,
                )

        def _recompute_eta(now_ts: float, rows_now: int) -> tuple[float | None, float]:
            if rows_now <= 0 or rows_now >= h:
                return (0.0 if rows_now >= h else None), now_ts
            elapsed = max(1.0e-6, now_ts - progress_start)
            remaining = elapsed * float(h - rows_now) / float(rows_now)
            return remaining, now_ts

        def _eta_ticker() -> None:
            while not progress_stop.wait(1.0):
                if not use_manual_progress:
                    continue
                with progress_state_lock:
                    rows_now = rows_done
                    eta_base = eta_base_seconds
                    eta_time = eta_base_time
                if rows_now <= 0 or rows_now >= h or eta_base is None:
                    continue
                eta_live = max(0.0, eta_base - max(0.0, time.perf_counter() - eta_time))
                _emit_progress(rows_now=rows_now, eta_now=eta_live, finalize_now=False)

        ticker_thread: threading.Thread | None = None
        if use_manual_progress:
            eta_base_seconds, eta_base_time = _recompute_eta(progress_start, rows_done)
            _emit_progress(rows_now=0, eta_now=eta_base_seconds, finalize_now=False)
            ticker_thread = threading.Thread(target=_eta_ticker, name="kerrtrace-progress-eta", daemon=True)
            ticker_thread.start()
        _notify_row_progress(rows_done)

        def _shade_from_trace(trace: tuple[torch.Tensor, ...]) -> torch.Tensor:
            with amp_ctx:
                return self._shade(*trace[:-1], emitter=emitter_payload)

        cfg_render_base = self.config
        row_blocks: list[tuple[int, int]]
        if use_tiling:
            row_blocks = [(rs, min(h, rs + tile_rows)) for rs in range(0, h, tile_rows)]
        else:
            row_blocks = [(0, h)]

        low_memory_requested = bool(cfg_render_base.low_memory_spool)
        low_memory_reasons: list[str] = []
        if low_memory_requested:
            if bool(cfg_render_base.adaptive_spatial_sampling) and len(row_blocks) > 1:
                low_memory_reasons.append("adaptive_spatial_sampling requires a full-frame preview")
            if (
                str(cfg_render_base.postprocess_pipeline) == "gargantua"
                and float(cfg_render_base.gargantua_look_strength) > 0.0
            ):
                low_memory_reasons.append("gargantua postprocess requires full-frame neighborhood filters")
            if int(self._meridian_destripe_passes()) > 0:
                low_memory_reasons.append("meridian destripe requires full-frame seam detection")
            if bool(self.is_wormhole and cfg_render_base.wormhole_seam_remove):
                low_memory_reasons.append("wormhole seam postprocess requires full-frame analysis")
            if bool(
                self.is_wormhole
                and cfg_render_base.wormhole_mt_beam_supersampling
                and int(cfg_render_base.wormhole_mt_beam_samples) > 0
            ):
                low_memory_reasons.append("wormhole beam supersampling requires full-frame refinement masks")
            if low_memory_reasons:
                print(
                    "Low-memory spool requested but disabled: "
                    + "; ".join(low_memory_reasons)
                    + ". Falling back to standard in-memory render.",
                    flush=True,
                )
        use_low_memory_spool = bool(low_memory_requested and (not low_memory_reasons))

        hit_disk: torch.Tensor | None = None
        hit_emitter: torch.Tensor | None = None
        hit_horizon: torch.Tensor | None = None
        escaped: torch.Tensor | None = None
        rgb: torch.Tensor | None = None
        steps_used = 0

        spool_rgb_u8: np.memmap | None = None
        spool_path: Path | None = None
        disk_hits_count = 0
        horizon_hits_count = 0
        escaped_count = 0
        cache_flush_stride = 4 if self.device.type in {"cuda", "mps"} else 0
        if use_low_memory_spool:
            try:
                with tempfile.NamedTemporaryFile(prefix="kerrtrace_frame_spool_", suffix=".rgb8", delete=False) as tmp:
                    spool_path = Path(tmp.name)
                spool_rgb_u8 = np.memmap(spool_path, dtype=np.uint8, mode="w+", shape=(h, w, 3))
            except Exception as exc:
                use_low_memory_spool = False
                if low_memory_requested:
                    print(
                        f"Low-memory spool unavailable ({exc}); falling back to standard in-memory render.",
                        flush=True,
                    )
        if not use_low_memory_spool:
            hit_disk = torch.zeros(h * w, dtype=torch.bool, device=self.device)
            hit_emitter = torch.zeros(h * w, dtype=torch.bool, device=self.device)
            hit_horizon = torch.zeros(h * w, dtype=torch.bool, device=self.device)
            escaped = torch.zeros(h * w, dtype=torch.bool, device=self.device)
            rgb = torch.zeros((h, w, 3), dtype=self.dtype, device=self.device)

        def _trace_and_shade_block(
            row_start: int,
            row_end: int,
            use_meridian: bool,
            x_offset_extra: float = 0.0,
            y_offset_extra: float = 0.0,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
            rows = row_end - row_start
            if use_meridian:
                trace_a = trace_fn(
                    x_pixel_offset=x_pixel_offset + x_offset_extra - 0.35,
                    y_pixel_offset=y_pixel_offset + y_offset_extra,
                    emitter=emitter_payload,
                    row_start=row_start,
                    row_end=row_end,
                )
                trace_b = trace_fn(
                    x_pixel_offset=x_pixel_offset + x_offset_extra + 0.35,
                    y_pixel_offset=y_pixel_offset + y_offset_extra,
                    emitter=emitter_payload,
                    row_start=row_start,
                    row_end=row_end,
                )
                rgb_a = _shade_from_trace(trace_a).reshape(rows, w, 3)
                rgb_b = _shade_from_trace(trace_b).reshape(rows, w, 3)
                return (
                    0.5 * (rgb_a + rgb_b),
                    trace_a[0],
                    trace_a[1],
                    trace_a[2],
                    trace_a[3],
                    int(max(trace_a[-1], trace_b[-1])),
                )

            trace = trace_fn(
                x_pixel_offset=x_pixel_offset + x_offset_extra,
                y_pixel_offset=y_pixel_offset + y_offset_extra,
                emitter=emitter_payload,
                row_start=row_start,
                row_end=row_end,
            )
            return (
                _shade_from_trace(trace).reshape(rows, w, 3),
                trace[0],
                trace[1],
                trace[2],
                trace[3],
                int(trace[-1]),
            )

        row_step_overrides: list[int] | None = None
        adaptive_spatial_enabled = bool(
            cfg_render_base.adaptive_spatial_sampling
            and len(row_blocks) > 1
            and h >= 96
            and w >= 96
            and int(cfg_render_base.max_steps) > 80
        )
        if adaptive_spatial_enabled:
            preview_steps = max(
                16,
                min(int(cfg_render_base.max_steps) - 1, int(cfg_render_base.adaptive_spatial_preview_steps)),
            )
            if preview_steps < int(cfg_render_base.max_steps):
                try:
                    self.config = replace(cfg_render_base, max_steps=int(preview_steps))
                    preview_rgb = torch.zeros((h, w, 3), dtype=self.dtype, device=self.device)
                    for row_start, row_end in row_blocks:
                        block_rgb, _, _, _, _, _ = _trace_and_shade_block(
                            row_start=row_start,
                            row_end=row_end,
                            use_meridian=False,
                        )
                        preview_rgb[row_start:row_end, :, :] = block_rgb
                    row_scales = self._adaptive_row_scales_from_preview(
                        preview_rgb=preview_rgb,
                        min_scale=float(cfg_render_base.adaptive_spatial_min_scale),
                        quantile=float(cfg_render_base.adaptive_spatial_quantile),
                    )
                    row_step_overrides = []
                    for row_start, row_end in row_blocks:
                        if row_end <= row_start:
                            row_step_overrides.append(int(cfg_render_base.max_steps))
                            continue
                        block_scale = float(torch.max(row_scales[row_start:row_end]).item())
                        target_steps = int(round(float(cfg_render_base.max_steps) * block_scale))
                        target_steps = max(64, min(int(cfg_render_base.max_steps), target_steps))
                        row_step_overrides.append(target_steps)
                    if progress_enabled and row_step_overrides:
                        mean_scale = float(np.mean(row_step_overrides)) / max(1.0, float(cfg_render_base.max_steps))
                        print(
                            (
                                "Adaptive spatial sampling enabled: "
                                f"preview_steps={preview_steps}, mean_step_scale={mean_scale:.3f}"
                            ),
                            flush=True,
                        )
                except Exception:
                    row_step_overrides = None
                finally:
                    self.config = cfg_render_base

        if use_tqdm_progress:
            progress_ctx = tqdm_auto(
                total=h,
                desc="Render rows",
                unit="row",
                smoothing=0.3,
                mininterval=0.1,
                dynamic_ncols=True,
                leave=True,
                disable=not sys.stdout.isatty(),
            )
        elif use_click_progress:
            fill_char, empty_char = self._progress_bar_chars()
            progress_ctx = click.progressbar(
                length=h,
                label="Render rows",
                width=30,
                show_eta=True,
                show_percent=True,
                show_pos=True,
                fill_char=fill_char,
                empty_char=empty_char,
                file=sys.stdout,
            )
        else:
            progress_ctx = nullcontext(None)

        try:
            with progress_ctx as progress_bar:
                for block_index, (row_start, row_end) in enumerate(row_blocks):
                    if row_step_overrides is not None:
                        target_steps = int(row_step_overrides[block_index])
                        if int(self.config.max_steps) != target_steps:
                            self.config = replace(cfg_render_base, max_steps=target_steps)
                    elif self.config is not cfg_render_base:
                        self.config = cfg_render_base

                    (
                        block_rgb,
                        local_hit_disk,
                        local_hit_emitter,
                        local_hit_horizon,
                        local_escaped,
                        local_steps_used,
                    ) = _trace_and_shade_block(
                        row_start=row_start,
                        row_end=row_end,
                        use_meridian=bool(cfg_render_base.meridian_supersample),
                    )
                    steps_used = max(steps_used, int(local_steps_used))
                    if use_low_memory_spool:
                        assert spool_rgb_u8 is not None
                        disk_hits_count += int((local_hit_disk | local_hit_emitter).sum().item())
                        horizon_hits_count += int(local_hit_horizon.sum().item())
                        escaped_count += int(local_escaped.sum().item())
                        block_u8 = torch.clamp(
                            self._apply_postprocess_pipeline(block_rgb) * 255.0,
                            min=0.0,
                            max=255.0,
                        ).to(torch.uint8)
                        spool_rgb_u8[row_start:row_end, :, :] = (
                            block_u8.detach().to(device="cpu").contiguous().numpy()
                        )
                        if cache_flush_stride > 0 and (
                            ((block_index + 1) % cache_flush_stride == 0) or (row_end >= h)
                        ):
                            self._clear_device_cache()
                    else:
                        assert rgb is not None
                        assert hit_disk is not None and hit_emitter is not None
                        assert hit_horizon is not None and escaped is not None
                        rgb[row_start:row_end, :, :] = block_rgb
                        start = row_start * w
                        end = row_end * w
                        hit_disk[start:end] = local_hit_disk
                        hit_emitter[start:end] = local_hit_emitter
                        hit_horizon[start:end] = local_hit_horizon
                        escaped[start:end] = local_escaped
                    rows_done += (row_end - row_start)
                    if progress_bar is not None:
                        progress_bar.update(row_end - row_start)
                    elif use_manual_progress:
                        now_ts = time.perf_counter()
                        with progress_state_lock:
                            eta_base_seconds, eta_base_time = _recompute_eta(now_ts, rows_done)
                            eta_now = eta_base_seconds
                        _emit_progress(
                            rows_now=rows_done,
                            eta_now=eta_now,
                            finalize_now=rows_done >= h,
                        )
                    _notify_row_progress(rows_done)
        finally:
            self.config = cfg_render_base

        if ticker_thread is not None:
            progress_stop.set()
            ticker_thread.join(timeout=1.5)
        _notify_row_progress(h)

        if use_low_memory_spool:
            self._last_rgb_float = None
            out: np.ndarray
            try:
                assert spool_rgb_u8 is not None
                spool_rgb_u8.flush()
                out = np.asarray(spool_rgb_u8, dtype=np.uint8).copy()
            finally:
                if spool_rgb_u8 is not None:
                    del spool_rgb_u8
                if spool_path is not None:
                    try:
                        spool_path.unlink(missing_ok=True)
                    except Exception:
                        pass
            image = Image.fromarray(out, mode="RGB")
            stats = RenderStats(
                total_rays=self.config.width * self.config.height,
                disk_hits=int(disk_hits_count),
                horizon_hits=int(horizon_hits_count),
                escaped=int(escaped_count),
                steps_used=steps_used,
            )
            return RenderOutput(image=image, stats=stats)

        assert rgb is not None
        assert hit_disk is not None and hit_emitter is not None
        assert hit_horizon is not None and escaped is not None

        # ROI supersampling disabled: in realistic KNdS/GKS profiles this mode
        # improved local detail but introduced a substantial render-time penalty.
        roi_enabled = False
        if roi_enabled:
            try:
                self.config = cfg_render_base
                roi_mask = self._build_roi_supersample_mask(rgb)
                roi_rows = torch.nonzero(roi_mask.any(dim=1), as_tuple=False).squeeze(-1)
                if int(roi_rows.numel()) > 0:
                    row_ranges: list[tuple[int, int]] = []
                    start_row = int(roi_rows[0].item())
                    prev_row = start_row
                    for row_t in roi_rows[1:]:
                        row = int(row_t.item())
                        if row == prev_row + 1:
                            prev_row = row
                            continue
                        row_ranges.append((start_row, prev_row + 1))
                        start_row = row
                        prev_row = row
                    row_ranges.append((start_row, prev_row + 1))

                    offsets = self._roi_supersample_offsets()
                    for row_start, row_end in row_ranges:
                        mask_block = roi_mask[row_start:row_end, :].unsqueeze(-1)
                        if not bool(mask_block.any()):
                            continue
                        accum = rgb[row_start:row_end, :, :].clone()
                        for x_off, y_off in offsets:
                            block_rgb, _, _, _, _, local_steps_used = _trace_and_shade_block(
                                row_start=row_start,
                                row_end=row_end,
                                use_meridian=bool(cfg_render_base.meridian_supersample),
                                x_offset_extra=float(x_off),
                                y_offset_extra=float(y_off),
                            )
                            accum = accum + block_rgb
                            steps_used = max(steps_used, int(local_steps_used))
                        refined = accum / float(1 + len(offsets))
                        rgb[row_start:row_end, :, :] = torch.where(
                            mask_block,
                            refined,
                            rgb[row_start:row_end, :, :],
                        )
            except Exception:
                pass
            finally:
                self.config = cfg_render_base

        beam_enabled = bool(
            self.is_wormhole
            and cfg_render_base.wormhole_mt_beam_supersampling
            and int(cfg_render_base.wormhole_mt_beam_samples) > 0
        )
        if beam_enabled:
            try:
                self.config = cfg_render_base
                beam_mask = self._build_wormhole_mt_beam_mask(
                    rgb,
                    escaped_mask=escaped,
                    hit_horizon_mask=hit_horizon,
                )
                if bool(torch.any(beam_mask)):
                    offsets = self._wormhole_mt_beam_offsets()
                    if offsets:
                        accum = rgb.clone()
                        for x_off in offsets:
                            for block_index, (row_start, row_end) in enumerate(row_blocks):
                                if row_step_overrides is not None:
                                    target_steps = int(row_step_overrides[block_index])
                                    if int(self.config.max_steps) != target_steps:
                                        self.config = replace(cfg_render_base, max_steps=target_steps)
                                elif self.config is not cfg_render_base:
                                    self.config = cfg_render_base
                                block_rgb, _, _, _, _, local_steps_used = _trace_and_shade_block(
                                    row_start=row_start,
                                    row_end=row_end,
                                    use_meridian=bool(cfg_render_base.meridian_supersample),
                                    x_offset_extra=float(x_off),
                                    y_offset_extra=0.0,
                                )
                                accum[row_start:row_end, :, :] = accum[row_start:row_end, :, :] + block_rgb
                                steps_used = max(steps_used, int(local_steps_used))
                        refined = accum / float(1 + len(offsets))
                        rgb = torch.where(beam_mask.unsqueeze(-1), refined, rgb)
            except Exception:
                pass
            finally:
                self.config = cfg_render_base

        destripe_passes = int(self._meridian_destripe_passes())
        if destripe_passes > 0:
            seam_mask = escaped | hit_horizon
            for _ in range(destripe_passes):
                rgb = self._destripe_meridian(
                    rgb,
                    escaped_mask=seam_mask,
                    force=False,
                )

        try:
            self._last_rgb_float = self._apply_postprocess_pipeline(rgb).detach().to(device="cpu", dtype=torch.float32).contiguous()
        except Exception:
            self._last_rgb_float = None
        out_t = self._finalize_rgb_cuda_graph(rgb)
        out = out_t.cpu().contiguous().numpy()
        if self.is_wormhole and self.config.wormhole_seam_remove:
            out = self._postprocess_wormhole_seam(out)
        image = Image.fromarray(out, mode="RGB")

        stats = RenderStats(
            total_rays=self.config.width * self.config.height,
            disk_hits=int((hit_disk | hit_emitter).sum().item()),
            horizon_hits=int(hit_horizon.sum().item()),
            escaped=int(escaped.sum().item()),
            steps_used=steps_used,
        )
        return RenderOutput(image=image, stats=stats)

    def render_to_file(self, output_path: str | Path | None = None, emitter: EmitterInput = None) -> RenderStats:
        target = Path(output_path or self.config.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        result = self.render(emitter=emitter)
        suffix = target.suffix.lower()
        if suffix in {".hdr", ".exr"}:
            if self._last_rgb_float is None:
                float_rgb = np.asarray(result.image.convert("RGB"), dtype=np.float32) / 255.0
            else:
                float_rgb = self._last_rgb_float.numpy()
            if suffix == ".hdr":
                if iio is not None:
                    try:
                        iio.imwrite(str(target), float_rgb.astype(np.float32, copy=False))
                    except Exception:
                        _write_radiance_hdr(target, float_rgb)
                else:
                    _write_radiance_hdr(target, float_rgb)
            else:
                if iio is None:
                    raise RuntimeError(
                        "EXR output requires imageio backend support. Install with: pip install imageio[opencv]"
                    )
                try:
                    iio.imwrite(str(target), float_rgb.astype(np.float32, copy=False))
                except Exception as exc:
                    raise RuntimeError(
                        f"Failed to write .exr output. Ensure imageio backend supports EXR: {exc}"
                    ) from exc
        else:
            result.image.save(target)
        return result.stats
