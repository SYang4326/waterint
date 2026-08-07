from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy.fftpack import dct

from waterint.chemistry import oxygen_hydrogen_neighbors_by_species
from waterint._02_computation._native import sfg_ssvvcf
from waterint._00_io.common import TrajectoryFrame
from waterint._01_core.coordinates import fixed_reference_value, slab_surface_reference
from waterint._01_core.selection import SelectionContext, element_indices, element_mask


AU2INVCM = 219474.62909500976
FS2AU = 41.34137300000000


@dataclass(frozen=True)
class SfgResult:
    mode: str
    cf_paths: dict[str, Path]
    ft_paths: dict[str, Path]
    png_paths: dict[str, Path]
    metadata_path: Path


@dataclass(frozen=True)
class SsvvcfResult:
    time_ps: np.ndarray
    corr: np.ndarray
    counts: np.ndarray
    zrefs: np.ndarray
    frames: int
    velocity_source: str


@dataclass
class _Segment:
    hydrogen_index: int
    oxygen_index: int
    mu: list[float] = field(default_factory=list)
    stretch: list[float] = field(default_factory=list)


def load_cf(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.loadtxt(path, comments="#")
    if data.ndim == 1:
        data = data[None, :]
    if data.shape[1] < 2:
        raise ValueError(f"{path} must contain at least time and correlation columns.")
    time_ps = data[:, 0].astype(float)
    corr = data[:, 1].astype(float)
    if data.shape[1] >= 3:
        counts = data[:, 2].astype(np.int64)
    else:
        counts = np.ones_like(time_ps, dtype=np.int64)
    if time_ps.size < 2:
        raise ValueError(f"{path} must contain at least two rows.")
    return time_ps, corr, counts


def write_cf(
    path: str | Path,
    time_ps: np.ndarray,
    corr: np.ndarray,
    counts: np.ndarray,
    *,
    time_label: str = "ps",
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"# time_{time_label}  C_avg  count\n")
        for time_value, corr_value, count in zip(time_ps, corr, counts):
            handle.write(f"{time_value:12.8f} {corr_value: .12e} {int(count)}\n")


def combine_cf(paths: list[str | Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not paths:
        raise ValueError("No CF paths were provided.")
    ref_time: np.ndarray | None = None
    weighted_sum: np.ndarray | None = None
    count_sum: np.ndarray | None = None

    for path in paths:
        time_ps, corr, counts = load_cf(path)
        if ref_time is None:
            ref_time = time_ps
            weighted_sum = np.zeros_like(corr, dtype=float)
            count_sum = np.zeros_like(counts, dtype=np.int64)
        else:
            if time_ps.shape != ref_time.shape or np.max(np.abs(time_ps - ref_time)) > 1e-9:
                raise ValueError(f"CF time grid mismatch: {path}")
        weighted_sum += corr * counts
        count_sum += counts

    assert ref_time is not None
    assert weighted_sum is not None
    assert count_sum is not None
    corr_avg = np.zeros_like(weighted_sum)
    mask = count_sum > 0
    corr_avg[mask] = weighted_sum[mask] / count_sum[mask]
    return ref_time, corr_avg, count_sum


def compute_ft(
    time: np.ndarray,
    corr: np.ndarray,
    *,
    time_unit: str = "ps",
    nzeros: int = 2000,
) -> tuple[np.ndarray, np.ndarray]:
    if time.size < 2:
        raise ValueError("Need at least two time points to compute FT.")
    dt = float(time[1] - time[0])
    if dt <= 0:
        raise ValueError("Time grid must be increasing.")
    if time_unit == "ps":
        dt_fs = dt * 1000.0
    elif time_unit == "fs":
        dt_fs = dt
    else:
        raise ValueError("time_unit must be ps or fs.")
    step_au = dt_fs * FS2AU

    signal_in = np.asarray(corr, dtype=float)
    if nzeros > 0:
        signal_in = np.concatenate((signal_in, np.zeros(int(nzeros), dtype=float)))
    signal = np.real(dct(signal_in, type=1))
    freq = np.linspace(0.0, 1.0 / (2.0 * step_au), signal.size)
    freq *= AU2INVCM * 2.0 * np.pi
    return freq, signal


def write_ft(path: str | Path, freq_cm: np.ndarray, signal: np.ndarray, *, frequency_label: str = "cm^-1") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = f"frequency_{frequency_label} signal"
    np.savetxt(path, np.column_stack((freq_cm, signal)), fmt="%.18e", header=header)


def load_ft(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, comments="#")
    if data.ndim == 1:
        data = data[None, :]
    if data.shape[1] < 2:
        raise ValueError(f"{path} must contain at least frequency and signal columns.")
    return data[:, 0], data[:, 1]


def write_zrefs(path: str | Path, frames: list[TrajectoryFrame], zrefs: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# frame_idx  step  zref\n")
        for frame, zref in zip(frames, zrefs):
            step = frame.step if frame.step is not None else frame.index
            handle.write(f"{frame.index} {step} {zref:.8f}\n")


def compute_ssvvcf_from_frames(
    frames: list[TrajectoryFrame],
    *,
    cell: tuple[float, float, float],
    sfg_cfg: dict[str, Any],
    context: SelectionContext,
) -> SsvvcfResult:
    """Compute trajectory-mode ssVVCF with a Python or optional C++ backend."""

    if len(frames) < 3:
        raise ValueError("SFG trajectory mode needs at least three frames for finite-difference velocities.")

    dt_ps = float(sfg_cfg.get("dt_ps", _infer_dt_ps(frames)))
    if dt_ps <= 0:
        raise ValueError("sfg.dt_ps must be positive.")
    lag_ps = float(sfg_cfg.get("lag_ps", dt_ps * min(200, max(1, len(frames) - 1))))
    max_lag = int(round(lag_ps / dt_ps))
    if max_lag < 1:
        raise ValueError("sfg.lag_ps must be at least one frame.")

    pbc = pbc_flags(sfg_cfg.get("pbc", [True, True, False]))
    hydrogen_symbol = str(sfg_cfg.get("hydrogen_symbol", "H"))
    oxygen_symbol = str(sfg_cfg.get("oxygen_symbol", "O"))
    oh_cutoff = float(sfg_cfg.get("oh_cutoff", sfg_cfg.get("bond_cutoff", 1.25)))
    oh_assignment = str(sfg_cfg.get("oh_assignment", "cutoff")).lower()
    if oh_assignment not in {"cutoff", "nearest_oxygen"}:
        raise ValueError("sfg.oh_assignment must be cutoff or nearest_oxygen.")
    neighbor_method = str(sfg_cfg.get("neighbor_method", "auto"))
    neighbor_workers = int(sfg_cfg.get("neighbor_workers", 1))
    oxygen_chunk_size = int(sfg_cfg.get("oxygen_chunk_size", 2048))
    mu_mode = str(sfg_cfg.get("mu_mode", "full")).lower()
    if mu_mode not in {"full", "stretch"}:
        raise ValueError("sfg.mu_mode must be full or stretch.")
    symmetrize = bool(sfg_cfg.get("symmetrize", True))
    flip_sign = bool(sfg_cfg.get("flip_sign", False))
    backend = str(sfg_cfg.get("backend", "auto")).lower()
    if backend not in {"auto", "python", "cpp"}:
        raise ValueError("sfg.backend must be auto, python, or cpp.")
    trajectory_velocities = trajectory_velocities_from_frames(frames, sfg_cfg=sfg_cfg)

    zrefs = zref_series(frames, sfg_cfg, context)
    first = frames[0]
    oxygen_indices = element_indices(first, {oxygen_symbol}, context)
    hydrogen_indices = element_indices(first, {hydrogen_symbol}, context)
    if oxygen_indices.size == 0 or hydrogen_indices.size == 0:
        raise ValueError("SFG trajectory mode could not find O/H atoms. Check input.type_map and sfg symbols.")

    fixed_topology = selection_is_fixed(frames, first)
    if backend == "cpp" and not fixed_topology:
        raise ValueError("sfg.backend: cpp requires fixed atom types/order across trajectory frames.")
    if backend in {"auto", "cpp"} and fixed_topology:
        positions = contiguous_frame_positions(frames)
        native_result = sfg_ssvvcf(
            positions,
            trajectory_velocities,
            oxygen_indices,
            hydrogen_indices,
            zrefs,
            dt_ps=dt_ps,
            max_lag=max_lag,
            oh_cutoff=oh_cutoff,
            oh_assignment=oh_assignment,
            cell=cell,
            pbc=pbc,
            mu_mode=mu_mode,
            symmetrize=symmetrize,
            flip_sign=flip_sign,
            duplicate_policy=(
                "nearest"
                if oh_assignment == "nearest_oxygen"
                else str(sfg_cfg.get("duplicate_hydrogen_policy", "nearest"))
            ),
            window=sfg_cfg.get("window"),
        )
        if native_result is not None:
            sums, counts, _stage_seconds = native_result
            corr = np.zeros_like(sums)
            mask = counts > 0
            corr[mask] = sums[mask] / counts[mask]
            time_ps = np.arange(max_lag + 1, dtype=float) * dt_ps
            return SsvvcfResult(
                time_ps=time_ps,
                corr=corr,
                counts=counts,
                zrefs=zrefs,
                frames=len(frames),
                velocity_source="trajectory" if trajectory_velocities is not None else "finite_difference",
            )
        if backend == "cpp":
            raise RuntimeError("C++ SFG backend is not available.")

    velocities, velocity_source = sfg_velocities_from_frames(frames, sfg_cfg=sfg_cfg, cell=cell, pbc=pbc)
    segments = build_sfg_segments_python(
        frames,
        velocities=velocities,
        zrefs=zrefs,
        cell=cell,
        sfg_cfg=sfg_cfg,
        context=context,
        pbc=pbc,
        oxygen_symbol=oxygen_symbol,
        hydrogen_symbol=hydrogen_symbol,
        oh_cutoff=oh_cutoff,
        oh_assignment=oh_assignment,
        neighbor_method=neighbor_method,
        neighbor_workers=neighbor_workers,
        oxygen_chunk_size=oxygen_chunk_size,
        mu_mode=mu_mode,
        flip_sign=flip_sign,
    )
    sums, counts = correlate_sfg_segments_python(
        segments,
        max_lag=max_lag,
        symmetrize=symmetrize,
    )
    corr = np.zeros_like(sums)
    mask = counts > 0
    corr[mask] = sums[mask] / counts[mask]
    time_ps = np.arange(max_lag + 1, dtype=float) * dt_ps
    return SsvvcfResult(
        time_ps=time_ps,
        corr=corr,
        counts=counts,
        zrefs=zrefs,
        frames=len(frames),
        velocity_source=velocity_source,
    )


def build_sfg_segments_python(
    frames: list[TrajectoryFrame],
    *,
    velocities: np.ndarray,
    zrefs: np.ndarray,
    cell: tuple[float, float, float],
    sfg_cfg: dict[str, Any],
    context: SelectionContext,
    pbc: tuple[bool, bool, bool],
    oxygen_symbol: str,
    hydrogen_symbol: str,
    oh_cutoff: float,
    oh_assignment: str,
    neighbor_method: str,
    neighbor_workers: int,
    oxygen_chunk_size: int,
    mu_mode: str,
    flip_sign: bool,
    stage_timings: dict[str, float] | None = None,
) -> list[_Segment]:
    """Assign O-H bonds and build continuous per-bond signal segments."""

    active_oxygen_by_h: dict[int, int] = {}
    active_segments: dict[int, _Segment] = {}
    segments: list[_Segment] = []
    for frame_index, frame in enumerate(frames):
        assignment_start = time.perf_counter() if stage_timings is not None else 0.0
        assigned = assign_sfg_hydrogens_for_frame(
            frame,
            cell=cell,
            pbc=pbc,
            context=context,
            oxygen_symbol=oxygen_symbol,
            hydrogen_symbol=hydrogen_symbol,
            oh_cutoff=oh_cutoff,
            oh_assignment=oh_assignment,
            neighbor_method=neighbor_method,
            neighbor_workers=neighbor_workers,
            oxygen_chunk_size=oxygen_chunk_size,
            duplicate_policy=(
                "nearest"
                if oh_assignment == "nearest_oxygen"
                else str(sfg_cfg.get("duplicate_hydrogen_policy", "nearest"))
            ),
        )
        if stage_timings is not None:
            stage_timings["O-H assignment"] += time.perf_counter() - assignment_start
            signal_start = time.perf_counter()
        append_sfg_frame_signals(
            frame_index,
            frame,
            assigned=assigned,
            velocities=velocities,
            zrefs=zrefs,
            cell=cell,
            pbc=pbc,
            sfg_cfg=sfg_cfg,
            mu_mode=mu_mode,
            flip_sign=flip_sign,
            active_oxygen_by_h=active_oxygen_by_h,
            active_segments=active_segments,
            segments=segments,
        )
        if stage_timings is not None:
            stage_timings["Segment signal construction"] += time.perf_counter() - signal_start

    segments.extend(active_segments.values())
    return segments


def assign_sfg_hydrogens_for_frame(
    frame: TrajectoryFrame,
    *,
    cell: tuple[float, float, float],
    pbc: tuple[bool, bool, bool],
    context: SelectionContext,
    oxygen_symbol: str,
    hydrogen_symbol: str,
    oh_cutoff: float,
    oh_assignment: str,
    neighbor_method: str,
    neighbor_workers: int,
    oxygen_chunk_size: int,
    duplicate_policy: str,
) -> dict[int, int]:
    """Assign H to O atoms with an optional no-cutoff nearest-O fallback."""

    oxygen_indices = element_indices(frame, {oxygen_symbol}, context)
    hydrogen_indices = element_indices(frame, {hydrogen_symbol}, context)
    neighbors_by_species = oxygen_hydrogen_neighbors_by_species(
        frame.symbols,
        frame.positions,
        oxygen_symbol=oxygen_symbol,
        hydrogen_symbol=hydrogen_symbol,
        oh_cutoff=oh_cutoff,
        neighbor_method=neighbor_method,
        neighbor_workers=neighbor_workers,
        oxygen_chunk_size=oxygen_chunk_size,
        oxygen_indices=oxygen_indices,
        hydrogen_indices=hydrogen_indices,
        cell=cell,
        pbc=pbc,
    )
    assigned = assigned_hydrogens_from_neighbors(
        neighbors_by_species,
        positions=frame.positions,
        cell=cell,
        pbc=pbc,
        duplicate_policy=duplicate_policy,
    )
    if oh_assignment == "nearest_oxygen":
        assigned = assign_unmatched_hydrogens_to_nearest_oxygen(
            assigned,
            oxygen_indices=oxygen_indices,
            hydrogen_indices=hydrogen_indices,
            positions=frame.positions,
            cell=cell,
            pbc=pbc,
        )
    return assigned


def append_sfg_frame_signals(
    frame_index: int,
    frame: TrajectoryFrame,
    *,
    assigned: dict[int, int],
    velocities: np.ndarray,
    zrefs: np.ndarray,
    cell: tuple[float, float, float],
    pbc: tuple[bool, bool, bool],
    sfg_cfg: dict[str, Any],
    mu_mode: str,
    flip_sign: bool,
    active_oxygen_by_h: dict[int, int],
    active_segments: dict[int, _Segment],
    segments: list[_Segment],
) -> None:
    """Update O-H segments and append this frame's mu/stretch signals."""

    missing_hydrogens = set(active_segments) - set(assigned)
    for hydrogen_index in missing_hydrogens:
        segments.append(active_segments.pop(hydrogen_index))
        active_oxygen_by_h.pop(hydrogen_index, None)

    positions = frame.positions
    for hydrogen_index, oxygen_index in assigned.items():
        previous_oxygen = active_oxygen_by_h.get(hydrogen_index)
        if previous_oxygen != oxygen_index:
            if hydrogen_index in active_segments:
                segments.append(active_segments.pop(hydrogen_index))
            active_oxygen_by_h[hydrogen_index] = oxygen_index
            active_segments[hydrogen_index] = _Segment(hydrogen_index, oxygen_index)

        oxygen_position = positions[oxygen_index]
        zprime = float(oxygen_position[2] - zrefs[frame_index])
        fwin = window_factor(zprime, sfg_cfg)
        r_oh = minimum_image(positions[hydrogen_index] - oxygen_position, cell=cell, pbc=pbc)
        r_norm = float(np.linalg.norm(r_oh))
        if r_norm <= 1e-12:
            continue
        vrel = velocities[frame_index, hydrogen_index] - velocities[frame_index, oxygen_index]
        stretch = float(np.dot(vrel, r_oh) / r_norm)
        cos_theta = float(r_oh[2] / r_norm)
        mu_stretch = fwin * stretch * cos_theta
        mu_full = fwin * float(vrel[2])
        mu = mu_stretch if mu_mode == "stretch" else mu_full
        if flip_sign:
            mu *= -1.0
        segment = active_segments[hydrogen_index]
        segment.mu.append(float(mu))
        segment.stretch.append(stretch)


def correlate_sfg_segments_python(
    segments: list[_Segment],
    *,
    max_lag: int,
    symmetrize: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Accumulate lag correlations for all continuous O-H segments."""

    sums = np.zeros(max_lag + 1, dtype=float)
    counts = np.zeros(max_lag + 1, dtype=np.int64)
    for segment in segments:
        accumulate_segment(segment, sums=sums, counts=counts, max_lag=max_lag, symmetrize=symmetrize)
    return sums, counts


def contiguous_frame_positions(frames: list[TrajectoryFrame]) -> np.ndarray:
    """Reuse a contiguous NPZ position block or stack other frame sources.

    NPZ frames are views into one loaded ``(frames, atoms, 3)`` array. Reusing
    that block avoids a second full-trajectory allocation before entering C++.
    """

    first = np.asarray(frames[0].positions)
    root = first.base
    if isinstance(root, np.ndarray) and root.ndim == 3 and root.shape[1:] == first.shape:
        indices = [int(frame.index) for frame in frames]
        if indices == list(range(indices[0], indices[0] + len(indices))):
            view = root[indices[0] : indices[0] + len(indices)]
            if (
                view.dtype == np.float64
                and view.flags.c_contiguous
                and np.shares_memory(view[0], frames[0].positions)
                and np.shares_memory(view[-1], frames[-1].positions)
            ):
                return view
    return np.ascontiguousarray(np.stack([frame.positions for frame in frames]), dtype=np.float64)


def selection_is_fixed(frames: list[TrajectoryFrame], first: TrajectoryFrame) -> bool:
    """Check the fixed atom-type/order contract required by the C++ kernel."""

    position_shape = np.asarray(first.positions).shape
    if any(np.asarray(frame.positions).shape != position_shape for frame in frames[1:]):
        return False
    if first.types is not None:
        first_types = np.asarray(first.types)
        if first_types.shape != position_shape[:1]:
            return False
        root = first_types.base
        if isinstance(root, np.ndarray) and root.ndim == 2 and root.shape[1:] == first_types.shape:
            indices = [int(frame.index) for frame in frames]
            if indices == list(range(indices[0], indices[0] + len(indices))):
                view = root[indices[0] : indices[0] + len(indices)]
                if (
                    view.shape[0] == len(frames)
                    and np.shares_memory(view[0], first_types)
                    and all(
                        frame.types is not None and np.shares_memory(view[offset], frame.types)
                        for offset, frame in enumerate(frames)
                    )
                ):
                    return bool(np.all(view == view[0]))
        return all(frame.types is not None and np.array_equal(frame.types, first.types) for frame in frames[1:])
    if any(frame.types is not None for frame in frames[1:]):
        return False
    if all(frame.symbols is first.symbols for frame in frames[1:]):
        return True
    return all(frame.symbols == first.symbols for frame in frames[1:])


def finite_difference_velocities(
    frames: list[TrajectoryFrame],
    *,
    dt_ps: float,
    cell: tuple[float, float, float],
    pbc: tuple[bool, bool, bool],
) -> np.ndarray:
    positions = [frame.positions for frame in frames]
    velocities = np.zeros((len(frames), positions[0].shape[0], 3), dtype=float)
    for i in range(len(frames)):
        if i == 0:
            velocities[i] = minimum_image(positions[1] - positions[0], cell=cell, pbc=pbc) / dt_ps
        elif i == len(frames) - 1:
            velocities[i] = minimum_image(positions[-1] - positions[-2], cell=cell, pbc=pbc) / dt_ps
        else:
            velocities[i] = minimum_image(positions[i + 1] - positions[i - 1], cell=cell, pbc=pbc) / (2.0 * dt_ps)
    return velocities


def sfg_velocities_from_frames(
    frames: list[TrajectoryFrame],
    *,
    sfg_cfg: dict[str, Any],
    cell: tuple[float, float, float],
    pbc: tuple[bool, bool, bool],
) -> tuple[np.ndarray, str]:
    """Return velocities in A/ps from a dump or the finite-difference fallback."""

    trajectory_velocities = trajectory_velocities_from_frames(frames, sfg_cfg=sfg_cfg)
    if trajectory_velocities is not None:
        return trajectory_velocities, "trajectory"

    dt_ps = float(sfg_cfg.get("dt_ps", _infer_dt_ps(frames)))
    return finite_difference_velocities(frames, dt_ps=dt_ps, cell=cell, pbc=pbc), "finite_difference"


def trajectory_velocities_from_frames(
    frames: list[TrajectoryFrame],
    *,
    sfg_cfg: dict[str, Any],
) -> np.ndarray | None:
    """Return supplied dump velocities in internal A/ps, if selected and complete."""

    source = str(sfg_cfg.get("velocity_source", "auto")).lower()
    if source not in {"auto", "trajectory", "finite_difference"}:
        raise ValueError("sfg.velocity_source must be auto, trajectory, or finite_difference.")
    if source == "finite_difference":
        return None

    supplied = [frame.velocities for frame in frames]
    if not all(velocity is not None for velocity in supplied):
        if source == "trajectory":
            raise ValueError("sfg.velocity_source: trajectory requires vx, vy, and vz in every trajectory frame.")
        return None

    velocities = np.ascontiguousarray(np.stack(supplied), dtype=float)
    position_shape = (len(frames),) + np.asarray(frames[0].positions).shape
    if velocities.shape != position_shape or not np.isfinite(velocities).all():
        raise ValueError("Trajectory velocities must be finite arrays with one (atoms, 3) row per frame.")
    unit = str(sfg_cfg.get("trajectory_velocity_unit", "A/ps")).lower().replace("angstrom", "a")
    if unit in {"a/ps", "a ps^-1"}:
        return velocities
    if unit in {"a/fs", "a fs^-1"}:
        return velocities * 1000.0
    raise ValueError("sfg.trajectory_velocity_unit must be A/ps or A/fs.")


def assigned_hydrogens_from_neighbors(
    neighbors_by_species: dict[str, list[tuple[int, np.ndarray]]],
    *,
    positions: np.ndarray,
    cell: tuple[float, float, float],
    pbc: tuple[bool, bool, bool],
    duplicate_policy: str,
) -> dict[int, int]:
    duplicate_policy = duplicate_policy.lower()
    if duplicate_policy not in {"nearest", "error"}:
        raise ValueError("sfg.duplicate_hydrogen_policy must be nearest or error.")

    assigned: dict[int, tuple[int, float]] = {}
    for neighbors in neighbors_by_species.values():
        for oxygen_index, hydrogen_indices in neighbors:
            for hydrogen_index in hydrogen_indices:
                hydrogen_index = int(hydrogen_index)
                oxygen_index = int(oxygen_index)
                vector = minimum_image(positions[hydrogen_index] - positions[oxygen_index], cell=cell, pbc=pbc)
                distance2 = float(np.dot(vector, vector))
                if hydrogen_index in assigned:
                    if duplicate_policy == "error":
                        raise ValueError(
                            f"Hydrogen atom {hydrogen_index} is assigned to more than one oxygen. "
                            "Set sfg.duplicate_hydrogen_policy: nearest, lower sfg.oh_cutoff, "
                            "or inspect the trajectory geometry."
                        )
                    if distance2 >= assigned[hydrogen_index][1]:
                        continue
                assigned[hydrogen_index] = (oxygen_index, distance2)
    return {hydrogen_index: oxygen_index for hydrogen_index, (oxygen_index, _distance2) in assigned.items()}


def assign_unmatched_hydrogens_to_nearest_oxygen(
    assigned: dict[int, int],
    *,
    oxygen_indices: np.ndarray,
    hydrogen_indices: np.ndarray,
    positions: np.ndarray,
    cell: tuple[float, float, float],
    pbc: tuple[bool, bool, bool],
) -> dict[int, int]:
    """Give every remaining H its nearest O without imposing an O-H cutoff."""

    for hydrogen_index in hydrogen_indices:
        hydrogen_index = int(hydrogen_index)
        if hydrogen_index in assigned:
            continue
        vectors = minimum_image(
            positions[hydrogen_index] - positions[oxygen_indices], cell=cell, pbc=pbc
        )
        nearest_local = int(np.argmin(np.einsum("ij,ij->i", vectors, vectors)))
        assigned[hydrogen_index] = int(oxygen_indices[nearest_local])
    return assigned


def accumulate_segment(
    segment: _Segment,
    *,
    sums: np.ndarray,
    counts: np.ndarray,
    max_lag: int,
    symmetrize: bool,
) -> None:
    mu = np.asarray(segment.mu, dtype=float)
    stretch = np.asarray(segment.stretch, dtype=float)
    length = mu.size
    if length < 2:
        return
    lag_max = min(max_lag, length - 1)
    for lag in range(lag_max + 1):
        n = length - lag
        corr = float(np.dot(mu[:n], stretch[lag : lag + n]))
        if symmetrize:
            corr = (corr + float(np.dot(stretch[:n], mu[lag : lag + n]))) * 2.0
            sums[lag] += corr
            counts[lag] += 2 * n
        else:
            sums[lag] += corr
            counts[lag] += n


def zref_series(frames: list[TrajectoryFrame], sfg_cfg: dict[str, Any], context: SelectionContext) -> np.ndarray:
    if sfg_cfg.get("zref_file"):
        data = np.loadtxt(sfg_cfg["zref_file"], comments="#")
        if data.ndim == 1:
            values = data
        else:
            col = int(sfg_cfg.get("zref_col", 0))
            values = data[:, col - 1 if col > 0 else -1]
        if values.size < len(frames):
            raise ValueError("sfg.zref_file has fewer values than trajectory frames.")
        return np.asarray(values[: len(frames)], dtype=float)

    reference_cfg = sfg_cfg.get("reference", {})
    if isinstance(reference_cfg, dict):
        fixed = fixed_reference_value(reference_cfg)
        if fixed is not None:
            return np.full(len(frames), fixed, dtype=float)
        if reference_cfg.get("species"):
            return np.asarray(
                [slab_surface_reference(frame, 2, 1.0, reference_cfg, context) for frame in frames],
                dtype=float,
            )

    return np.full(len(frames), float(sfg_cfg.get("z_ref0", 0.0)), dtype=float)


def window_factor(zprime: float, sfg_cfg: dict[str, Any]) -> float:
    if "window" not in sfg_cfg:
        return 1.0
    window = sfg_cfg["window"]
    if not isinstance(window, dict):
        raise ValueError("sfg.window must be a mapping.")
    z1 = float(window.get("z1", 0.0))
    z2 = float(window.get("z2", 0.0))
    ramp = float(window.get("ramp", 0.0))
    mode = int(window.get("mode", 1))
    flip = bool(window.get("flip", False))
    if mode == 1:
        return slab_interface_decay(zprime, z1, z2, ramp, flip)
    if mode == 2:
        return top_hat_ramp(zprime, z1, z2, ramp)
    raise ValueError("sfg.window.mode must be 1 or 2.")


def slab_interface_decay(z: float, z1: float, z2: float, ramp: float, flip: bool) -> float:
    if z2 < z1:
        z1, z2 = z2, z1
    width = z2 - z1
    if width <= 0:
        return 1.0 if ((z > z1) if flip else (z <= z1)) else 0.0
    ramp = min(max(ramp, 0.0), width)
    if not flip:
        if z <= z1:
            return 1.0
        if z >= z2:
            return 0.0
        if ramp <= 0 or z <= z2 - ramp:
            return 1.0
        return ramp_sin01((z2 - z) / ramp)
    if z <= z1:
        return 0.0
    if ramp <= 0 or z >= z1 + ramp:
        return 1.0
    return ramp_sin01((z - z1) / ramp)


def top_hat_ramp(z: float, z1: float, z2: float, ramp: float) -> float:
    if z2 < z1:
        z1, z2 = z2, z1
    if z < z1 or z > z2:
        return 0.0
    if ramp <= 0:
        return 1.0
    ramp = min(ramp, 0.5 * (z2 - z1))
    if z < z1 + ramp:
        return ramp_sin01((z - z1) / ramp)
    if z > z2 - ramp:
        return ramp_sin01((z2 - z) / ramp)
    return 1.0


def ramp_sin01(value: float) -> float:
    if value <= 0:
        return 0.0
    if value >= 1:
        return 1.0
    return float(np.sin(0.5 * np.pi * value))


def minimum_image(vectors: np.ndarray, *, cell: tuple[float, float, float], pbc: tuple[bool, bool, bool]) -> np.ndarray:
    out = np.asarray(vectors, dtype=float).copy()
    cell_array = np.asarray(cell, dtype=float)
    for axis, enabled in enumerate(pbc):
        if enabled:
            length = cell_array[axis]
            if length <= 0:
                raise ValueError("Cell lengths must be positive when sfg.pbc is enabled.")
            out[..., axis] -= np.rint(out[..., axis] / length) * length
    return out


def pbc_flags(value: Any) -> tuple[bool, bool, bool]:
    if isinstance(value, bool):
        return (value, value, value)
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError("sfg.pbc must be a boolean or a list of three booleans.")
    return tuple(bool(item) for item in value)  # type: ignore[return-value]


def _infer_dt_ps(frames: list[TrajectoryFrame]) -> float:
    if frames[0].step is not None and frames[1].step is not None:
        delta = frames[1].step - frames[0].step
        if delta > 0:
            # LAMMPS timesteps are unitless here; users should set dt_ps for production.
            return 1.0
    return 1.0
