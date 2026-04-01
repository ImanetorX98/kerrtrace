from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import logging
import math
from itertools import product
from typing import NamedTuple
import numpy as np

import torch

logger = logging.getLogger(__name__)

THETA_EPS = 1.0e-4
SIN2_EPS = 1.0e-7
DIV_EPS = 1.0e-10


class MetricModel(str, Enum):
    SCHWARZSCHILD = "schwarzschild"
    KERR = "kerr"
    REISSNER_NORDSTROM = "reissner_nordstrom"
    KERR_NEWMAN = "kerr_newman"
    SCHWARZSCHILD_DE_SITTER = "schwarzschild_de_sitter"
    KERR_DE_SITTER = "kerr_de_sitter"
    REISSNER_NORDSTROM_DE_SITTER = "reissner_nordstrom_de_sitter"
    KERR_NEWMAN_DE_SITTER = "kerr_newman_de_sitter"
    MORRIS_THORNE = "morris_thorne"
    DNEG_WORMHOLE = "dneg_wormhole"


METRIC_MODELS = {m.value for m in MetricModel}


class MetricComponents(NamedTuple):
    g_tt: torch.Tensor
    g_tphi: torch.Tensor
    g_rr: torch.Tensor
    g_thth: torch.Tensor
    g_phiphi: torch.Tensor


class InverseMetricComponents(NamedTuple):
    gtt: torch.Tensor
    gtphi: torch.Tensor
    grr: torch.Tensor
    gthth: torch.Tensor
    gphiphi: torch.Tensor


@dataclass(frozen=True)
class ISCOResult:
    spin: float
    charge: float
    cosmological_constant: float
    radius: float


def canonical_metric_model(metric_model: str) -> str:
    model = metric_model.strip().lower().replace("-", "_")
    if model not in METRIC_MODELS:
        raise ValueError(f"Unknown metric_model: {metric_model}")
    return model


def effective_metric_parameters(metric_model: str, spin: float, charge: float, cosmological_constant: float) -> tuple[float, float, float]:
    model = canonical_metric_model(metric_model)

    rotating = model in {"kerr", "kerr_newman", "kerr_de_sitter", "kerr_newman_de_sitter"}
    charged = model in {"reissner_nordstrom", "kerr_newman", "reissner_nordstrom_de_sitter", "kerr_newman_de_sitter"}
    de_sitter = model in {"schwarzschild_de_sitter", "kerr_de_sitter", "reissner_nordstrom_de_sitter", "kerr_newman_de_sitter"}

    a = float(spin) if rotating else 0.0
    q = float(charge) if charged else 0.0
    lmb = float(cosmological_constant) if de_sitter else 0.0
    return a, q, lmb


def _dneg_areal_radius(
    ell: torch.Tensor,
    rho: torch.Tensor,
    half_len: torch.Tensor,
    lensing_scale: torch.Tensor,
) -> torch.Tensor:
    """Areal radius r(ℓ) for the Dneg wormhole metric (James et al. 2015, AJP).

    Parameters
    ----------
    ell         : proper radial coordinate (negative on far side)
    rho         : throat radius (ρ)
    half_len    : half-length of the cylindrical interior (a); r = ρ for |ℓ| ≤ a
    lensing_scale : lensing parameter M; controls how quickly r grows outside the throat

    r(ℓ) = ρ                                        for |ℓ| ≤ a
    r(ℓ) = ρ + (2M/π)[x·arctan(x) − ½·ln(1+x²)]   for |ℓ| > a
           where x = 2(|ℓ| − a) / (πM)
    """
    abs_ell = torch.abs(ell)
    outer = torch.clamp(abs_ell - half_len, min=0.0)
    M_safe = torch.clamp(lensing_scale, min=1.0e-8)
    x = 2.0 * outer / (math.pi * M_safe)
    r_outer = rho + (2.0 * M_safe / math.pi) * (x * torch.atan(x) - 0.5 * torch.log1p(x * x))
    return torch.where(abs_ell <= half_len, rho * torch.ones_like(ell), r_outer)


def horizon_radii(
    spin: float,
    metric_model: str = "kerr",
    charge: float = 0.0,
    cosmological_constant: float = 0.0,
) -> list[float]:
    if canonical_metric_model(metric_model) in {"morris_thorne", "dneg_wormhole"}:
        return []
    a, q, lmb = effective_metric_parameters(metric_model, spin, charge, cosmological_constant)

    if abs(lmb) < 1.0e-14:
        disc = 1.0 - a * a - q * q
        if disc < 0.0:
            return []
        s = math.sqrt(max(0.0, disc))
        roots = [1.0 - s, 1.0 + s]
        return sorted(r for r in roots if r > 0.0)

    coeffs = [
        -lmb / 3.0,
        0.0,
        1.0 - (lmb * a * a) / 3.0,
        -2.0,
        a * a + q * q,
    ]
    roots = np.roots(np.array(coeffs, dtype=np.float64))
    return sorted(float(z.real) for z in roots if abs(z.imag) < 1.0e-8 and z.real > 0.0)


def event_horizon_radius(
    spin: float,
    metric_model: str = "kerr",
    charge: float = 0.0,
    cosmological_constant: float = 0.0,
) -> float:
    model = canonical_metric_model(metric_model)
    if model == "morris_thorne":
        raise ValueError("Morris-Thorne wormhole has no event horizon")
    if model == "dneg_wormhole":
        raise ValueError("Dneg wormhole has no event horizon")
    _, _, lmb = effective_metric_parameters(metric_model, spin, charge, cosmological_constant)
    real_pos = horizon_radii(spin, metric_model, charge, cosmological_constant)
    if not real_pos:
        raise ValueError("No positive real horizon root found for this metric configuration")

    # For de Sitter (Lambda > 0), largest root is typically cosmological horizon.
    if lmb > 0.0 and len(real_pos) >= 2:
        return real_pos[-2]
    return real_pos[-1]


def _safe_divisor(x: torch.Tensor, eps: float = DIV_EPS) -> torch.Tensor:
    return torch.where(torch.abs(x) < eps, torch.where(x >= 0.0, torch.full_like(x, eps), torch.full_like(x, -eps)), x)


def _knds_common(
    r: torch.Tensor,
    theta: torch.Tensor,
    metric_model: str,
    spin: float,
    charge: float,
    cosmological_constant: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    a_scalar, q_scalar, lmb_scalar = effective_metric_parameters(metric_model, spin, charge, cosmological_constant)
    a = torch.as_tensor(a_scalar, dtype=r.dtype, device=r.device)
    q2 = torch.as_tensor(q_scalar * q_scalar, dtype=r.dtype, device=r.device)
    lmb = torch.as_tensor(lmb_scalar, dtype=r.dtype, device=r.device)

    cos_th = torch.cos(theta)
    sin_th = torch.sin(theta)
    sin2 = torch.clamp(sin_th * sin_th, min=SIN2_EPS)

    r2 = r * r
    a2 = a * a
    sigma = r2 + a2 * cos_th * cos_th

    delta_r = (r2 + a2) * (1.0 - (lmb * r2) / 3.0) - 2.0 * r + q2
    delta_theta = 1.0 + (lmb * a2 / 3.0) * cos_th * cos_th
    xi = 1.0 + (lmb * a2 / 3.0)

    return sigma, delta_r, delta_theta, sin2, a, torch.as_tensor(xi, dtype=r.dtype, device=r.device)


def metric_components(
    r: torch.Tensor,
    theta: torch.Tensor,
    spin: float,
    metric_model: str = "kerr",
    charge: float = 0.0,
    cosmological_constant: float = 0.0,
    wormhole_throat_radius: float = 1.0,
    wormhole_length_scale: float = 1.0,
    wormhole_lensing_scale: float = 1.0,
) -> MetricComponents:
    model = canonical_metric_model(metric_model)
    if model == "morris_thorne":
        b0 = torch.as_tensor(max(1.0e-6, float(wormhole_throat_radius)), dtype=r.dtype, device=r.device)
        length_scale = torch.as_tensor(max(1.0e-6, float(wormhole_length_scale)), dtype=r.dtype, device=r.device)
        inv_len = 1.0 / length_scale
        b02 = b0 * b0
        sin2 = torch.clamp(torch.sin(theta) * torch.sin(theta), min=SIN2_EPS)
        l_eff = r * inv_len
        rho2 = l_eff * l_eff + b02
        rho2_safe = torch.clamp(rho2, min=DIV_EPS)
        g_tt = -torch.ones_like(r)
        g_tphi = torch.zeros_like(r)
        g_rr = torch.ones_like(r) * (inv_len * inv_len)
        g_thth = rho2_safe
        g_phiphi = rho2_safe * sin2
        return MetricComponents(g_tt=g_tt, g_tphi=g_tphi, g_rr=g_rr, g_thth=g_thth, g_phiphi=g_phiphi)

    if model == "dneg_wormhole":
        # Dneg three-parameter wormhole (James et al. 2015, AJP arXiv:1502.03809).
        # ds² = -dt² + dℓ² + r(ℓ)²(dθ² + sin²θ dφ²)
        # where r(ℓ) is the Dneg areal radius.  Here the BL coordinate r IS the
        # proper radial distance ℓ (signed: negative on the far side).
        # wormhole_throat_radius  = ρ  (throat radius)
        # wormhole_length_scale   = a  (half-length of cylindrical interior)
        # wormhole_lensing_scale  = M  (lensing parameter)
        rho = torch.as_tensor(max(1.0e-6, float(wormhole_throat_radius)), dtype=r.dtype, device=r.device)
        half_len = torch.as_tensor(max(0.0, float(wormhole_length_scale)), dtype=r.dtype, device=r.device)
        M_lens = torch.as_tensor(max(1.0e-6, float(wormhole_lensing_scale)), dtype=r.dtype, device=r.device)
        sin2 = torch.clamp(torch.sin(theta) * torch.sin(theta), min=SIN2_EPS)
        r_areal = _dneg_areal_radius(r, rho, half_len, M_lens)
        r2 = torch.clamp(r_areal * r_areal, min=DIV_EPS)
        g_tt = -torch.ones_like(r)
        g_tphi = torch.zeros_like(r)
        g_rr = torch.ones_like(r)
        g_thth = r2
        g_phiphi = r2 * sin2
        return MetricComponents(g_tt=g_tt, g_tphi=g_tphi, g_rr=g_rr, g_thth=g_thth, g_phiphi=g_phiphi)

    sigma, delta_r, delta_theta, sin2, a, xi = _knds_common(r, theta, metric_model, spin, charge, cosmological_constant)
    sigma_safe = torch.clamp(sigma, min=DIV_EPS)
    delta_r_safe = _safe_divisor(delta_r)
    delta_theta_safe = _safe_divisor(delta_theta)
    xi_safe = torch.where(torch.abs(xi) < DIV_EPS, torch.full_like(xi, DIV_EPS), xi)

    r2pa2 = r * r + a * a
    xi2 = xi_safe * xi_safe

    g_tt = (-delta_r + a * a * delta_theta * sin2) / sigma_safe
    g_tphi = (a * sin2 * (delta_r - r2pa2 * delta_theta)) / (sigma_safe * xi_safe)
    g_rr = sigma_safe / delta_r_safe
    g_thth = sigma_safe / delta_theta_safe
    g_phiphi = sin2 * ((r2pa2 * r2pa2) * delta_theta - a * a * sin2 * delta_r) / (sigma_safe * xi2)
    return MetricComponents(g_tt=g_tt, g_tphi=g_tphi, g_rr=g_rr, g_thth=g_thth, g_phiphi=g_phiphi)


def inverse_metric_components(
    r: torch.Tensor,
    theta: torch.Tensor,
    spin: float,
    metric_model: str = "kerr",
    charge: float = 0.0,
    cosmological_constant: float = 0.0,
    wormhole_throat_radius: float = 1.0,
    wormhole_length_scale: float = 1.0,
    wormhole_lensing_scale: float = 1.0,
) -> InverseMetricComponents:
    model = canonical_metric_model(metric_model)
    if model == "morris_thorne":
        b0 = torch.as_tensor(max(1.0e-6, float(wormhole_throat_radius)), dtype=r.dtype, device=r.device)
        length_scale = torch.as_tensor(max(1.0e-6, float(wormhole_length_scale)), dtype=r.dtype, device=r.device)
        b02 = b0 * b0
        sin2 = torch.clamp(torch.sin(theta) * torch.sin(theta), min=SIN2_EPS)
        l_eff = r / length_scale
        rho2 = torch.clamp(l_eff * l_eff + b02, min=DIV_EPS)
        gtt = -torch.ones_like(r)
        gtphi = torch.zeros_like(r)
        grr = torch.ones_like(r) * (length_scale * length_scale)
        gthth = 1.0 / rho2
        gphiphi = 1.0 / torch.clamp(rho2 * sin2, min=DIV_EPS)
        return InverseMetricComponents(gtt=gtt, gtphi=gtphi, grr=grr, gthth=gthth, gphiphi=gphiphi)

    if model == "dneg_wormhole":
        rho = torch.as_tensor(max(1.0e-6, float(wormhole_throat_radius)), dtype=r.dtype, device=r.device)
        half_len = torch.as_tensor(max(0.0, float(wormhole_length_scale)), dtype=r.dtype, device=r.device)
        M_lens = torch.as_tensor(max(1.0e-6, float(wormhole_lensing_scale)), dtype=r.dtype, device=r.device)
        sin2 = torch.clamp(torch.sin(theta) * torch.sin(theta), min=SIN2_EPS)
        r_areal = _dneg_areal_radius(r, rho, half_len, M_lens)
        r2 = torch.clamp(r_areal * r_areal, min=DIV_EPS)
        gtt = -torch.ones_like(r)
        gtphi = torch.zeros_like(r)
        grr = torch.ones_like(r)
        gthth = 1.0 / r2
        gphiphi = 1.0 / torch.clamp(r2 * sin2, min=DIV_EPS)
        return InverseMetricComponents(gtt=gtt, gtphi=gtphi, grr=grr, gthth=gthth, gphiphi=gphiphi)

    sigma, delta_r, delta_theta, sin2, a, xi = _knds_common(r, theta, metric_model, spin, charge, cosmological_constant)
    sigma_safe = torch.clamp(sigma, min=DIV_EPS)
    delta_r_safe = _safe_divisor(delta_r)
    delta_theta_safe = _safe_divisor(delta_theta)
    xi_safe = torch.where(torch.abs(xi) < DIV_EPS, torch.full_like(xi, DIV_EPS), xi)

    r2pa2 = r * r + a * a
    den = _safe_divisor(sigma_safe * delta_r_safe * delta_theta_safe)
    xi2 = xi_safe * xi_safe

    gtt = -xi2 * ((r2pa2 * r2pa2) * delta_theta - a * a * sin2 * delta_r) / den
    gtphi = xi2 * a * (delta_r - r2pa2 * delta_theta) / den
    grr = delta_r / sigma_safe
    gthth = delta_theta / sigma_safe
    gphiphi = xi2 * (delta_r - a * a * sin2 * delta_theta) / (den * sin2)
    return InverseMetricComponents(gtt=gtt, gtphi=gtphi, grr=grr, gthth=gthth, gphiphi=gphiphi)


def inverse_metric_derivatives(
    r: torch.Tensor,
    theta: torch.Tensor,
    spin: float,
    metric_model: str = "kerr",
    charge: float = 0.0,
    cosmological_constant: float = 0.0,
    wormhole_throat_radius: float = 1.0,
    wormhole_length_scale: float = 1.0,
    wormhole_lensing_scale: float = 1.0,
    r_rel_eps: float = 1.0e-4,
    theta_eps: float = 1.0e-5,
) -> tuple[InverseMetricComponents, InverseMetricComponents]:
    model = canonical_metric_model(metric_model)
    one = torch.ones_like(r)
    eps_r = r_rel_eps * torch.maximum(r.abs(), one)

    if model in {"morris_thorne", "dneg_wormhole"}:
        r_plus = r + eps_r
        r_minus = r - eps_r
    else:
        r_plus = torch.clamp(r + eps_r, min=1.0e-3)
        r_minus = torch.clamp(r - eps_r, min=1.0e-3)

    th_plus = torch.clamp(theta + theta_eps, min=THETA_EPS, max=math.pi - THETA_EPS)
    th_minus = torch.clamp(theta - theta_eps, min=THETA_EPS, max=math.pi - THETA_EPS)

    inv_r_plus = inverse_metric_components(
        r_plus,
        theta,
        spin,
        metric_model,
        charge,
        cosmological_constant,
        wormhole_throat_radius,
        wormhole_length_scale,
        wormhole_lensing_scale,
    )
    inv_r_minus = inverse_metric_components(
        r_minus,
        theta,
        spin,
        metric_model,
        charge,
        cosmological_constant,
        wormhole_throat_radius,
        wormhole_length_scale,
        wormhole_lensing_scale,
    )
    dr = torch.clamp(r_plus - r_minus, min=1.0e-9)

    d_r = InverseMetricComponents(
        gtt=(inv_r_plus.gtt - inv_r_minus.gtt) / dr,
        gtphi=(inv_r_plus.gtphi - inv_r_minus.gtphi) / dr,
        grr=(inv_r_plus.grr - inv_r_minus.grr) / dr,
        gthth=(inv_r_plus.gthth - inv_r_minus.gthth) / dr,
        gphiphi=(inv_r_plus.gphiphi - inv_r_minus.gphiphi) / dr,
    )

    inv_th_plus = inverse_metric_components(
        r,
        th_plus,
        spin,
        metric_model,
        charge,
        cosmological_constant,
        wormhole_throat_radius,
        wormhole_length_scale,
        wormhole_lensing_scale,
    )
    inv_th_minus = inverse_metric_components(
        r,
        th_minus,
        spin,
        metric_model,
        charge,
        cosmological_constant,
        wormhole_throat_radius,
        wormhole_length_scale,
        wormhole_lensing_scale,
    )
    dth = torch.clamp(th_plus - th_minus, min=1.0e-9)

    d_theta = InverseMetricComponents(
        gtt=(inv_th_plus.gtt - inv_th_minus.gtt) / dth,
        gtphi=(inv_th_plus.gtphi - inv_th_minus.gtphi) / dth,
        grr=(inv_th_plus.grr - inv_th_minus.grr) / dth,
        gthth=(inv_th_plus.gthth - inv_th_minus.gthth) / dth,
        gphiphi=(inv_th_plus.gphiphi - inv_th_minus.gphiphi) / dth,
    )

    return d_r, d_theta


def _scalar_inv_and_derivs(
    r: float,
    spin: float,
    metric_model: str,
    charge: float,
    cosmological_constant: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    r_val = float(r)
    if r_val <= 0.0:
        raise ValueError("r must be positive")

    dtype = torch.float64
    device = torch.device("cpu")
    theta = torch.tensor([math.pi * 0.5], dtype=dtype, device=device)
    rr = torch.tensor([r_val], dtype=dtype, device=device)

    inv = inverse_metric_components(rr, theta, spin, metric_model, charge, cosmological_constant)
    d_r, _ = inverse_metric_derivatives(
        rr,
        theta,
        spin,
        metric_model,
        charge,
        cosmological_constant,
        r_rel_eps=1.0e-5,
        theta_eps=1.0e-5,
    )

    h = max(1.0e-4, abs(r_val) * 2.0e-4)
    rp = torch.tensor([r_val + h], dtype=dtype, device=device)
    rm = torch.tensor([max(1.0e-6, r_val - h)], dtype=dtype, device=device)

    d_rp, _ = inverse_metric_derivatives(
        rp,
        theta,
        spin,
        metric_model,
        charge,
        cosmological_constant,
        r_rel_eps=1.0e-5,
        theta_eps=1.0e-5,
    )
    d_rm, _ = inverse_metric_derivatives(
        rm,
        theta,
        spin,
        metric_model,
        charge,
        cosmological_constant,
        r_rel_eps=1.0e-5,
        theta_eps=1.0e-5,
    )

    dr_span = float((rp - rm).item())
    d2_gtt = float(((d_rp.gtt - d_rm.gtt) / dr_span).item())
    d2_gtphi = float(((d_rp.gtphi - d_rm.gtphi) / dr_span).item())
    d2_gphiphi = float(((d_rp.gphiphi - d_rm.gphiphi) / dr_span).item())

    inv_vals = (float(inv.gtt.item()), float(inv.gtphi.item()), float(inv.gphiphi.item()))
    d1_vals = (float(d_r.gtt.item()), float(d_r.gtphi.item()), float(d_r.gphiphi.item()))
    d2_vals = (d2_gtt, d2_gtphi, d2_gphiphi)
    return inv_vals, d1_vals, d2_vals


def _choose_orbit_branch(
    inv_vals: tuple[float, float, float],
    d1_vals: tuple[float, float, float],
    spin: float,
    prograde: bool,
) -> tuple[float, float, float] | None:
    gtt, gtphi, gphiphi = inv_vals
    dgtt, dgtphi, dgphiphi = d1_vals

    qa = dgphiphi
    qb = -2.0 * dgtphi
    qc = dgtt
    roots: list[float] = []

    if abs(qa) < 1.0e-12:
        if abs(qb) < 1.0e-14:
            return None
        roots = [(-qc) / qb]
    else:
        disc = qb * qb - 4.0 * qa * qc
        if disc < 0.0:
            return None
        sdisc = math.sqrt(max(0.0, disc))
        roots = [(-qb + sdisc) / (2.0 * qa), (-qb - sdisc) / (2.0 * qa)]

    candidates: list[tuple[float, float, float]] = []
    for lam in roots:
        den = gtt - 2.0 * gtphi * lam + gphiphi * lam * lam
        if den >= -1.0e-12:
            continue
        e2 = -1.0 / den
        if not math.isfinite(e2) or e2 <= 0.0:
            continue
        e = math.sqrt(e2)
        l = lam * e
        ut_den = -gtt + gtphi * lam
        if abs(ut_den) < 1.0e-12:
            continue
        omega = (-gtphi + gphiphi * lam) / ut_den
        if not (math.isfinite(e) and math.isfinite(l) and math.isfinite(omega)):
            continue
        candidates.append((e, l, omega))

    if not candidates:
        return None

    # Prograde/retrograde selection follows the sign of the orbital angular
    # velocity relative to black-hole spin (if spin=0, use +/- convention).
    spin_sign = 1.0 if spin >= 0.0 else -1.0
    preferred_sign = spin_sign if prograde else -spin_sign
    if abs(spin) < 1.0e-12:
        preferred_sign = 1.0 if prograde else -1.0

    def rank(item: tuple[float, float, float]) -> tuple[int, float]:
        _, _, om = item
        same_sign = 0 if (om == 0.0 or math.copysign(1.0, om) == preferred_sign) else 1
        return (same_sign, -abs(om))

    return sorted(candidates, key=rank)[0]


def _isco_stability_value(
    r: float,
    spin: float,
    metric_model: str,
    charge: float,
    cosmological_constant: float,
    prograde: bool,
) -> float | None:
    inv_vals, d1_vals, d2_vals = _scalar_inv_and_derivs(r, spin, metric_model, charge, cosmological_constant)
    branch = _choose_orbit_branch(inv_vals, d1_vals, spin=spin, prograde=prograde)
    if branch is None:
        return None

    e, l, _ = branch
    d2_gtt, d2_gtphi, d2_gphiphi = d2_vals
    stability = d2_gtt * e * e - 2.0 * d2_gtphi * e * l + d2_gphiphi * l * l
    if not math.isfinite(stability):
        return None
    return stability


def isco_radius_general(
    spin: float,
    metric_model: str = "kerr",
    charge: float = 0.0,
    cosmological_constant: float = 0.0,
    prograde: bool = True,
    r_min: float | None = None,
    r_max: float | None = None,
    samples: int = 720,
    tol: float = 1.0e-6,
    max_iter: int = 80,
) -> float:
    """
    Numerically estimate the equatorial ISCO radius for the selected metric.

    The routine supports the full set of metric models already implemented in
    KerrTrace and works with (spin, charge, cosmological_constant) combinations.
    """
    model = canonical_metric_model(metric_model)
    if model in {"morris_thorne", "dneg_wormhole"}:
        raise ValueError("ISCO is not defined for wormhole metrics")
    horizon = event_horizon_radius(spin, model, charge, cosmological_constant)
    roots = horizon_radii(spin, model, charge, cosmological_constant)
    _, _, lmb = effective_metric_parameters(model, spin, charge, cosmological_constant)

    lo = max(1.0e-6, horizon * 1.01)
    if r_min is not None:
        lo = max(lo, float(r_min))

    if r_max is not None:
        hi = float(r_max)
    else:
        hi = 180.0
        if lmb > 0.0 and len(roots) >= 2:
            hi = min(hi, roots[-1] * 0.98)

    if hi <= lo * (1.0 + 1.0e-6):
        raise ValueError("Invalid ISCO search interval")

    ns = max(64, int(samples))
    if hi / lo > 1.25:
        rs = np.geomspace(lo, hi, num=ns)
    else:
        rs = np.linspace(lo, hi, num=ns)

    values: list[tuple[float, float]] = []
    for r in rs:
        stab = _isco_stability_value(float(r), spin, model, charge, cosmological_constant, prograde)
        if stab is None:
            continue
        values.append((float(r), float(stab)))

    if len(values) < 3:
        raise ValueError("Unable to determine ISCO: no valid circular-orbit branch in search range")

    bracket: tuple[float, float] | None = None
    for (r0, s0), (r1, s1) in zip(values[:-1], values[1:]):
        # For our Hamiltonian sign convention, stable circular orbits have
        # positive marginal-stability indicator.
        if s0 < 0.0 and s1 >= 0.0:
            bracket = (r0, r1)
            break

    if bracket is None:
        best_r, _ = min(values, key=lambda item: abs(item[1]))
        logger.warning("ISCO bracket not found, using nearest zero-stability sample at r=%.4f", best_r)
        return best_r

    a, b = bracket
    fa = _isco_stability_value(a, spin, model, charge, cosmological_constant, prograde)
    fb = _isco_stability_value(b, spin, model, charge, cosmological_constant, prograde)
    if fa is None or fb is None:
        return 0.5 * (a + b)

    for _ in range(max(1, int(max_iter))):
        mid = 0.5 * (a + b)
        fm = _isco_stability_value(mid, spin, model, charge, cosmological_constant, prograde)
        if fm is None:
            mid = math.nextafter(mid, a)
            fm = _isco_stability_value(mid, spin, model, charge, cosmological_constant, prograde)
            if fm is None:
                break

        if abs(fm) <= tol or (b - a) <= tol * max(1.0, mid):
            return float(mid)

        if fa * fm > 0.0:
            a, fa = mid, fm
        else:
            b, fb = mid, fm

    return float(0.5 * (a + b))


def isco_radius_grid(
    spins: list[float] | tuple[float, ...],
    charges: list[float] | tuple[float, ...],
    cosmological_constants: list[float] | tuple[float, ...],
    metric_model: str = "kerr_newman_de_sitter",
    prograde: bool = True,
) -> list[ISCOResult]:
    """Compute ISCO radii for the Cartesian product of (a, Q, Lambda) values."""
    results: list[ISCOResult] = []
    for a, q, lmb in product(spins, charges, cosmological_constants):
        r_isco = isco_radius_general(
            spin=float(a),
            metric_model=metric_model,
            charge=float(q),
            cosmological_constant=float(lmb),
            prograde=prograde,
        )
        results.append(
            ISCOResult(
                spin=float(a),
                charge=float(q),
                cosmological_constant=float(lmb),
                radius=float(r_isco),
            )
        )
    return results
