from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from waterint._01_core.cell import minimum_image
from waterint._02_computation._native import msd_sums


@dataclass(frozen=True)
class MsdResult:
    lag_frames: np.ndarray
    time_ps: np.ndarray
    msd_a2: np.ndarray
    samples: np.ndarray
    selected_atoms: int
    dimensionality: str
    csv_path: Path | None = None
    png_path: Path | None = None
    metadata_path: Path | None = None


def compute_msd(
    positions: np.ndarray,
    *,
    cell_vectors: np.ndarray,
    pbc: tuple[bool, bool, bool],
    timestep_ps: float,
    max_lag_frames: int | None = None,
    origin_stride: int = 1,
    dimensionality: str = "3d",
    plane_normal_axis: int = 2,
    backend: str = "auto",
) -> MsdResult:
    """Compute a multiple-time-origin MSD for a fixed set of atom identities."""

    positions_array = np.asarray(positions, dtype=float)
    vectors_array = np.asarray(cell_vectors, dtype=float)
    if positions_array.ndim != 3 or positions_array.shape[2] != 3:
        raise ValueError("positions must have shape (n_frames, n_atoms, 3).")
    if positions_array.shape[0] < 2:
        raise ValueError("MSD requires at least two frames.")
    if positions_array.shape[1] == 0:
        raise ValueError("MSD selection found no atoms.")
    if vectors_array.shape != (positions_array.shape[0], 3, 3):
        raise ValueError("cell_vectors must have shape (n_frames, 3, 3).")
    if timestep_ps <= 0:
        raise ValueError("msd.timestep_ps must be positive.")
    mode = {2: "2d", 3: "3d"}.get(dimensionality, str(dimensionality).lower())
    if mode not in {"2d", "3d"}:
        raise ValueError("msd.dimensionality must be 2d or 3d.")
    if plane_normal_axis not in {0, 1, 2}:
        raise ValueError("msd.plane_normal_axis must be x, y, or z.")
    max_lag = positions_array.shape[0] - 1 if max_lag_frames is None else int(max_lag_frames)
    if not 1 <= max_lag < positions_array.shape[0]:
        raise ValueError("msd.max_lag_frames must be between 1 and n_frames - 1.")
    if origin_stride <= 0:
        raise ValueError("msd.origin_stride must be positive.")

    mode_backend = str(backend).lower()
    if mode_backend not in {"auto", "python", "cpp"}:
        raise ValueError("msd.backend must be auto, python, or cpp.")
    values: tuple[np.ndarray, np.ndarray] | None = None
    if mode_backend in {"auto", "cpp"}:
        values = msd_sums(
            positions_array,
            cell_vectors=vectors_array,
            pbc=pbc,
            max_lag=max_lag,
            origin_stride=origin_stride,
            dimensions=2 if mode == "2d" else 3,
            excluded_axis=plane_normal_axis,
        )
        if values is None and mode_backend == "cpp":
            raise RuntimeError("C++ MSD backend is not available.")
    if values is None:
        values = _msd_sums_python(
            positions_array,
            cell_vectors=vectors_array,
            pbc=pbc,
            max_lag=max_lag,
            origin_stride=origin_stride,
            dimensions=2 if mode == "2d" else 3,
            excluded_axis=plane_normal_axis,
        )
    sums, samples = values
    lag_frames = np.arange(max_lag + 1, dtype=int)
    msd = np.divide(sums, samples, out=np.zeros_like(sums), where=samples > 0)
    return MsdResult(
        lag_frames=lag_frames,
        time_ps=lag_frames.astype(float) * float(timestep_ps),
        msd_a2=msd,
        samples=samples,
        selected_atoms=positions_array.shape[1],
        dimensionality=mode,
    )


def _msd_sums_python(
    positions: np.ndarray,
    *,
    cell_vectors: np.ndarray,
    pbc: tuple[bool, bool, bool],
    max_lag: int,
    origin_stride: int,
    dimensions: int,
    excluded_axis: int,
) -> tuple[np.ndarray, np.ndarray]:
    unwrapped = np.empty_like(positions)
    unwrapped[0] = positions[0]
    for frame_index in range(1, positions.shape[0]):
        step = minimum_image(
            positions[frame_index] - positions[frame_index - 1],
            cell_vectors=cell_vectors[frame_index],
            pbc=pbc,
        )
        unwrapped[frame_index] = unwrapped[frame_index - 1] + step
    sums = np.zeros(max_lag + 1, dtype=float)
    samples = np.zeros(max_lag + 1, dtype=np.int64)
    for lag in range(max_lag + 1):
        origins = np.arange(0, positions.shape[0] - lag, origin_stride)
        displacement = unwrapped[origins + lag] - unwrapped[origins]
        if dimensions == 2:
            displacement[..., excluded_axis] = 0.0
        sums[lag] = float(np.sum(displacement * displacement))
        samples[lag] = displacement.shape[0] * displacement.shape[1]
    return sums, samples
