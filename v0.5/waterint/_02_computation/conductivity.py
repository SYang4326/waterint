from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


# Exact SI definitions (2019 SI).
ELEMENTARY_CHARGE_C = 1.602176634e-19
BOLTZMANN_J_PER_K = 1.380649e-23
ANGSTROM3_TO_M3 = 1.0e-30


@dataclass(frozen=True)
class ConductivityResult:
    """Nernst--Einstein conductivity derived from a carrier MSD."""

    time_ps: np.ndarray
    msd_a2: np.ndarray
    fit_msd_a2: np.ndarray
    fit_mask: np.ndarray
    slope_a2_per_ps: float
    intercept_a2: float
    diffusion_a2_per_ps: float
    diffusion_m2_per_s: float
    conductivity_s_per_m: float
    sheet_conductance_s: float | None
    carrier_count: int
    volume_a3: float
    dimensionality: int
    csv_path: Path | None = None
    msd_csv_path: Path | None = None
    png_path: Path | None = None
    metadata_path: Path | None = None


def compute_nernst_einstein_conductivity(
    time_ps: np.ndarray,
    msd_a2: np.ndarray,
    *,
    carrier_count: int,
    volume_a3: float,
    temperature_k: float,
    charge_e: float,
    dimensions: int,
    fit_range_ps: tuple[float, float],
    sheet_thickness_a: float | None = None,
) -> ConductivityResult:
    """Fit an MSD's diffusive region and apply the Nernst--Einstein relation.

    ``sigma = (N/V) q^2 D / (k_B T)``.  The MSD must represent a fixed set
    of independently counted carrier identities; correlated charge transport
    requires a Green--Kubo current calculation instead.
    """

    time = np.asarray(time_ps, dtype=float)
    msd = np.asarray(msd_a2, dtype=float)
    if time.ndim != 1 or msd.ndim != 1 or time.shape != msd.shape:
        raise ValueError("time_ps and msd_a2 must be one-dimensional arrays of equal length.")
    if time.size < 3:
        raise ValueError("Conductivity needs at least three MSD lag points.")
    if carrier_count <= 0:
        raise ValueError("carrier_count must be positive.")
    if volume_a3 <= 0:
        raise ValueError("Conductivity volume must be positive.")
    if temperature_k <= 0:
        raise ValueError("conductivity.temperature_K must be positive.")
    if charge_e == 0:
        raise ValueError("conductivity.charge_e must be non-zero.")
    if dimensions not in {2, 3}:
        raise ValueError("Conductivity dimensionality must be 2 or 3.")
    fit_start, fit_stop = (float(fit_range_ps[0]), float(fit_range_ps[1]))
    if not fit_stop > fit_start >= 0:
        raise ValueError("conductivity.fit_range_ps must satisfy 0 <= start < stop.")
    fit_mask = (time >= fit_start) & (time <= fit_stop) & np.isfinite(msd)
    if int(np.count_nonzero(fit_mask)) < 2:
        raise ValueError("conductivity.fit_range_ps must contain at least two finite MSD points.")

    slope, intercept = np.polyfit(time[fit_mask], msd[fit_mask], deg=1)
    if slope <= 0:
        raise ValueError("The fitted MSD slope is not positive; choose a diffusive fit range.")
    diffusion_a2_per_ps = float(slope) / (2.0 * dimensions)
    # 1 A^2 / ps = 1e-8 m^2 / s.
    diffusion_m2_per_s = diffusion_a2_per_ps * 1.0e-8
    number_density_m3 = carrier_count / (volume_a3 * ANGSTROM3_TO_M3)
    charge_c = charge_e * ELEMENTARY_CHARGE_C
    conductivity = number_density_m3 * charge_c * charge_c * diffusion_m2_per_s / (BOLTZMANN_J_PER_K * temperature_k)
    fit_msd = float(slope) * time + float(intercept)
    sheet = None if sheet_thickness_a is None else conductivity * float(sheet_thickness_a) * 1.0e-10
    return ConductivityResult(
        time_ps=time,
        msd_a2=msd,
        fit_msd_a2=fit_msd,
        fit_mask=fit_mask,
        slope_a2_per_ps=float(slope),
        intercept_a2=float(intercept),
        diffusion_a2_per_ps=diffusion_a2_per_ps,
        diffusion_m2_per_s=diffusion_m2_per_s,
        conductivity_s_per_m=float(conductivity),
        sheet_conductance_s=None if sheet is None else float(sheet),
        carrier_count=int(carrier_count),
        volume_a3=float(volume_a3),
        dimensionality=dimensions,
    )
