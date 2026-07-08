from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import math

import numpy as np

from waterint._02_computation._native import density_histogram_edges


DEFAULT_PROFILE_MASSES_AMU = {
    "O2-": 15.999,
    "OH-": 17.007,
    "H2O": 18.015,
    "H3O+": 19.023,
    "O_other": 15.999,
}
AMU_PER_A3_TO_G_PER_CM3 = 1.66053906660


@dataclass
class DensityState:
    bin_edges: np.ndarray
    counts_by_label: dict[str, np.ndarray]
    frames: int = 0
    selected_atoms_total: dict[str, int] | None = None


@dataclass(frozen=True)
class DensityResult:
    bin_centers: np.ndarray
    profiles: dict[str, dict[str, np.ndarray]]
    frames: int
    selected_atoms_total: dict[str, int]
    csv_path: Path | None = None
    png_path: Path | None = None
    metadata_path: Path | None = None


def new_density_state(labels: list[str], bin_edges: np.ndarray) -> DensityState:
    return DensityState(
        bin_edges=bin_edges,
        counts_by_label={label: np.zeros(len(bin_edges) - 1, dtype=float) for label in labels},
        selected_atoms_total={label: 0 for label in labels},
    )


def accumulate_density_frame(
    state: DensityState,
    coordinates_by_label: dict[str, np.ndarray],
    *,
    backend: str = "auto",
) -> None:
    if state.selected_atoms_total is None:
        state.selected_atoms_total = {label: 0 for label in state.counts_by_label}
    for label, values in coordinates_by_label.items():
        if label not in state.counts_by_label:
            continue
        state.selected_atoms_total[label] += int(values.size)
        hist = histogram_counts(values, state.bin_edges, backend=backend)
        state.counts_by_label[label] += hist
    state.frames += 1


def compute_density_profile(
    coordinates_by_frame: Iterable[dict[str, np.ndarray]],
    *,
    labels: list[str],
    bin_edges: np.ndarray,
    cell: tuple[float, float, float],
    axis: int,
    normalization_cfg: Any = None,
    backend: str = "auto",
) -> DensityResult:
    state = new_density_state(labels, bin_edges)
    for coordinates_by_label in coordinates_by_frame:
        accumulate_density_frame(state, coordinates_by_label, backend=backend)
    return finalize_density_state(
        state,
        cell=cell,
        axis=axis,
        normalization_cfg=normalization_cfg,
    )


def histogram_counts(values: np.ndarray, bin_edges: np.ndarray, *, backend: str = "auto") -> np.ndarray:
    backend = str(backend).lower()
    if backend not in {"auto", "python", "cpp"}:
        raise ValueError("density.backend must be auto, python, or cpp.")

    if backend in {"auto", "cpp"}:
        cpp_counts = histogram_counts_cpp(values, bin_edges)
        if cpp_counts is not None:
            return cpp_counts
        if backend == "cpp":
            raise RuntimeError("C++ density backend is not available.")

    hist, _ = np.histogram(values, bins=bin_edges)
    return hist.astype(float, copy=False)


def histogram_counts_cpp(values: np.ndarray, bin_edges: np.ndarray) -> np.ndarray | None:
    if bin_edges.ndim != 1 or bin_edges.size < 2:
        raise ValueError("bin_edges must be a one-dimensional array with at least two entries.")
    return density_histogram_edges(values, bin_edges=bin_edges)


def finalize_density_state(
    state: DensityState,
    *,
    cell: tuple[float, float, float],
    axis: int,
    normalization_cfg: Any = None,
) -> DensityResult:
    if state.frames == 0:
        raise ValueError("Cannot finalize density profile with zero frames.")
    bin_widths = np.diff(state.bin_edges)
    if not np.allclose(bin_widths, bin_widths[0]):
        raise ValueError("Density normalization currently requires uniform bin spacing.")
    bin_centers = 0.5 * (state.bin_edges[:-1] + state.bin_edges[1:])
    profiles = {
        label: {
            "counts_per_frame": counts / state.frames,
            "density": normalize_density_counts(
                counts=counts,
                frames=state.frames,
                cell=cell,
                axis=axis,
                bin_width=float(bin_widths[0]),
                normalization_cfg=normalization_cfg,
                label=label,
            ),
        }
        for label, counts in state.counts_by_label.items()
    }
    return DensityResult(
        bin_centers=bin_centers,
        profiles=profiles,
        frames=state.frames,
        selected_atoms_total=state.selected_atoms_total or {},
    )


def normalize_density_counts(
    *,
    counts: np.ndarray,
    frames: int,
    cell: tuple[float, float, float],
    axis: int,
    bin_width: float,
    normalization_cfg: Any,
    label: str,
) -> np.ndarray:
    if normalization_cfg is None:
        normalization_cfg = {}
    if not isinstance(normalization_cfg, dict):
        raise ValueError("normalization must be a mapping.")
    norm_type = str(normalization_cfg.get("type", "number_density"))
    unit = str(normalization_cfg.get("unit", "")).lower()
    if norm_type == "counts_per_frame":
        return counts / frames
    if norm_type not in {"number_density", "mass_density"}:
        raise ValueError("normalization.type must be number_density, mass_density, or counts_per_frame.")

    perpendicular_lengths = [cell[i] for i in range(3) if i != axis]
    slab_volume = perpendicular_lengths[0] * perpendicular_lengths[1] * bin_width
    if not math.isfinite(slab_volume) or slab_volume <= 0:
        raise ValueError("Computed slab volume must be positive.")
    number_density = counts / frames / slab_volume
    if norm_type == "number_density" and unit not in {"g_cm3", "g/cm3", "g/cm^3"}:
        return number_density

    mass_amu = profile_mass_amu(label, normalization_cfg)
    return number_density * mass_amu * AMU_PER_A3_TO_G_PER_CM3


def profile_mass_amu(label: str, normalization_cfg: dict[str, Any]) -> float:
    masses = normalization_cfg.get("masses_amu", {})
    if masses is None:
        masses = {}
    if not isinstance(masses, dict):
        raise ValueError("normalization.masses_amu must be a mapping.")
    if label in masses:
        mass = float(masses[label])
    elif label in DEFAULT_PROFILE_MASSES_AMU:
        mass = DEFAULT_PROFILE_MASSES_AMU[label]
    else:
        mass = normalization_cfg.get("mass_amu", None)
        if mass is None:
            raise ValueError(
                f"Need mass for profile {label!r}. Set normalization.mass_amu or normalization.masses_amu."
            )
        mass = float(mass)
    if mass <= 0:
        raise ValueError("Profile mass must be positive.")
    return mass
