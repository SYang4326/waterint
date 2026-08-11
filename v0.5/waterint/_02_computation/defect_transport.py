from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from waterint._01_core.cell import minimum_image


ELEMENTARY_CHARGE_C = 1.602176634e-19
ANGSTROM_M = 1.0e-10
PICOSECOND_S = 1.0e-12
BOLTZMANN_J_PER_K = 1.380649e-23


@dataclass(frozen=True)
class DefectSegment:
    track_id: int
    frame_indices: np.ndarray
    atom_indices: np.ndarray
    positions_a: np.ndarray


@dataclass(frozen=True)
class DefectTrackingResult:
    segments: tuple[DefectSegment, ...]
    carrier_counts: np.ndarray
    births: np.ndarray
    deaths: np.ndarray
    matched: np.ndarray
    charge_current_ea_per_ps: np.ndarray
    timestep_ps: float
    charge_e: float

    @property
    def mean_carriers(self) -> float:
        return float(np.mean(self.carrier_counts))


@dataclass(frozen=True)
class DefectMsdResult:
    lag_frames: np.ndarray
    time_ps: np.ndarray
    msd_a2: np.ndarray
    samples: np.ndarray
    segments: int
    mean_carriers: float
    dimensionality: str


@dataclass(frozen=True)
class GreenKuboResult:
    conductivity_s_per_m: float
    conductivity_std_s_per_m: float
    component_conductivity_s_per_m: np.ndarray
    component_std_s_per_m: np.ndarray


@dataclass
class _ActiveTrack:
    track_id: int
    frame_indices: list[int]
    atom_indices: list[int]
    positions: list[np.ndarray]
    last_wrapped: np.ndarray
    last_unwrapped: np.ndarray


def track_defects(
    positions_by_frame: list[np.ndarray],
    atom_indices_by_frame: list[np.ndarray],
    *,
    cell_vectors: np.ndarray,
    pbc: tuple[bool, bool, bool],
    timestep_ps: float,
    gate_a: float,
    charge_e: float = -1.0,
) -> DefectTrackingResult:
    """Track framewise defect positions with gated Hungarian assignment."""

    if len(positions_by_frame) < 2:
        raise ValueError("Defect tracking requires at least two frames.")
    if len(atom_indices_by_frame) != len(positions_by_frame):
        raise ValueError("Defect positions and atom indices must have equal frame counts.")
    vectors = np.asarray(cell_vectors, dtype=float)
    if vectors.shape != (len(positions_by_frame), 3, 3):
        raise ValueError("cell_vectors must have shape (n_frames, 3, 3).")
    if timestep_ps <= 0:
        raise ValueError("defect_tracking.timestep_ps must be positive.")
    if gate_a <= 0:
        raise ValueError("defect_tracking.gate_A must be positive.")

    counts = np.asarray([len(frame) for frame in positions_by_frame], dtype=np.int64)
    births = np.zeros(len(positions_by_frame), dtype=np.int64)
    deaths = np.zeros(len(positions_by_frame), dtype=np.int64)
    matched = np.zeros(len(positions_by_frame), dtype=np.int64)
    current = np.zeros((len(positions_by_frame) - 1, 3), dtype=float)
    completed: list[DefectSegment] = []
    active: list[_ActiveTrack] = []
    next_track_id = 0

    first_positions = _positions_array(positions_by_frame[0])
    first_indices = _indices_array(atom_indices_by_frame[0], len(first_positions))
    for position, atom_index in zip(first_positions, first_indices):
        active.append(_new_track(next_track_id, 0, int(atom_index), position))
        next_track_id += 1
    gate2 = float(gate_a) ** 2
    for frame_index in range(1, len(positions_by_frame)):
        current_positions = _positions_array(positions_by_frame[frame_index])
        current_indices = _indices_array(atom_indices_by_frame[frame_index], len(current_positions))
        assignments, displacements = _match_active_tracks(
            active,
            current_positions,
            cell_vectors=vectors[frame_index],
            pbc=pbc,
            gate2=gate2,
        )
        used_current: set[int] = set()
        next_active: list[_ActiveTrack] = []
        for active_index, track in enumerate(active):
            candidate = assignments[active_index]
            if candidate < 0:
                completed.append(_finish_track(track))
                deaths[frame_index] += 1
                continue
            displacement = displacements[active_index]
            track.last_wrapped = current_positions[candidate].copy()
            track.last_unwrapped = track.last_unwrapped + displacement
            track.frame_indices.append(frame_index)
            track.atom_indices.append(int(current_indices[candidate]))
            track.positions.append(track.last_unwrapped.copy())
            next_active.append(track)
            used_current.add(candidate)
            matched[frame_index] += 1
            current[frame_index - 1] += float(charge_e) * displacement / float(timestep_ps)

        for candidate, (position, atom_index) in enumerate(zip(current_positions, current_indices)):
            if candidate in used_current:
                continue
            next_active.append(_new_track(next_track_id, frame_index, int(atom_index), position))
            next_track_id += 1
            births[frame_index] += 1
        active = next_active

    completed.extend(_finish_track(track) for track in active)
    return DefectTrackingResult(
        segments=tuple(completed),
        carrier_counts=counts,
        births=births,
        deaths=deaths,
        matched=matched,
        charge_current_ea_per_ps=current,
        timestep_ps=float(timestep_ps),
        charge_e=float(charge_e),
    )


def compute_defect_msd(
    tracking: DefectTrackingResult,
    *,
    max_lag_frames: int | None = None,
    origin_stride: int = 1,
    dimensionality: str = "3d",
    plane_normal_axis: int = 2,
) -> DefectMsdResult:
    """Compute lifetime-aware MSD using only origins within continuous tracks."""

    if origin_stride <= 0:
        raise ValueError("defect_msd.origin_stride must be positive.")
    mode = {2: "2d", 3: "3d"}.get(dimensionality, str(dimensionality).lower())
    if mode not in {"2d", "3d"}:
        raise ValueError("defect_msd.dimensionality must be 2d or 3d.")
    if plane_normal_axis not in {0, 1, 2}:
        raise ValueError("defect_msd.plane_normal_axis must be x, y, or z.")
    longest = max((len(segment.frame_indices) for segment in tracking.segments), default=0)
    if longest < 2:
        raise ValueError("Defect MSD requires at least one segment with two frames.")
    max_lag = longest - 1 if max_lag_frames is None else min(int(max_lag_frames), longest - 1)
    if max_lag < 1:
        raise ValueError("defect_msd.max_lag_frames must be positive.")

    sums = np.zeros(max_lag + 1, dtype=float)
    samples = np.zeros(max_lag + 1, dtype=np.int64)
    for segment in tracking.segments:
        positions = np.asarray(segment.positions_a, dtype=float)
        for lag in range(min(max_lag, len(positions) - 1) + 1):
            origins = np.arange(0, len(positions) - lag, origin_stride, dtype=int)
            displacement = positions[origins + lag] - positions[origins]
            if mode == "2d":
                displacement[:, plane_normal_axis] = 0.0
            sums[lag] += float(np.sum(displacement * displacement))
            samples[lag] += len(origins)
    msd = np.divide(sums, samples, out=np.full_like(sums, np.nan), where=samples > 0)
    lags = np.arange(max_lag + 1, dtype=int)
    return DefectMsdResult(
        lag_frames=lags,
        time_ps=lags.astype(float) * tracking.timestep_ps,
        msd_a2=msd,
        samples=samples,
        segments=len(tracking.segments),
        mean_carriers=tracking.mean_carriers,
        dimensionality=mode,
    )


def compute_green_kubo_conductivity(
    tracking: DefectTrackingResult,
    *,
    volume_a3: float,
    temperature_k: float,
    dimensionality: str,
    plane_normal_axis: int = 2,
) -> GreenKuboResult:
    """Estimate collective defect conductivity with the optional STACIE package."""

    try:
        from stacie import ExpPolyModel, compute_spectrum, estimate_acint
    except ImportError as exc:
        raise RuntimeError(
            "Green-Kubo defect conductivity requires STACIE; install it with "
            "`python -m pip install stacie` or add its src directory to PYTHONPATH."
        ) from exc
    if volume_a3 <= 0 or temperature_k <= 0:
        raise ValueError("Green-Kubo conductivity requires positive volume and temperature.")
    mode = str(dimensionality).lower()
    axes = [axis for axis in range(3) if mode != "2d" or axis != plane_normal_axis]
    if mode not in {"2d", "3d"}:
        raise ValueError("Green-Kubo dimensionality must be 2d or 3d.")
    current_si = (
        tracking.charge_current_ea_per_ps[:, axes].T
        * ELEMENTARY_CHARGE_C
        * ANGSTROM_M
        / PICOSECOND_S
    )
    current_si -= np.mean(current_si, axis=1, keepdims=True)
    prefactor = 1.0 / (volume_a3 * 1.0e-30 * BOLTZMANN_J_PER_K * temperature_k)
    values = []
    errors = []
    for component in current_si:
        spectrum = compute_spectrum(
            component[None, :],
            prefactors=prefactor,
            timestep=tracking.timestep_ps * PICOSECOND_S,
            include_zero_freq=False,
        )
        estimate = estimate_acint(spectrum, ExpPolyModel([0, 1, 2]), verbose=False)
        values.append(float(np.asarray(estimate.acint)))
        errors.append(float(np.asarray(estimate.acint_std)))
    component_values = np.asarray(values)
    component_errors = np.asarray(errors)
    return GreenKuboResult(
        conductivity_s_per_m=float(np.mean(component_values)),
        conductivity_std_s_per_m=float(np.sqrt(np.sum(component_errors**2)) / len(component_errors)),
        component_conductivity_s_per_m=component_values,
        component_std_s_per_m=component_errors,
    )


def _match_active_tracks(
    active: list[_ActiveTrack],
    current: np.ndarray,
    *,
    cell_vectors: np.ndarray,
    pbc: tuple[bool, bool, bool],
    gate2: float,
) -> tuple[np.ndarray, np.ndarray]:
    if not active:
        return np.empty(0, dtype=int), np.empty((0, 3), dtype=float)
    if len(current) == 0:
        return np.full(len(active), -1, dtype=int), np.zeros((len(active), 3), dtype=float)
    previous = np.asarray([track.last_wrapped for track in active])
    delta = current[None, :, :] - previous[:, None, :]
    delta = minimum_image(delta, cell_vectors=cell_vectors, pbc=pbc)
    squared = np.sum(delta * delta, axis=2)
    unmatched = gate2 * (1.0 + 1.0e-6)
    cost = np.full((len(active), len(current) + len(active)), unmatched, dtype=float)
    cost[:, : len(current)] = np.where(squared <= gate2, squared, unmatched + 1.0e6 + squared)
    rows, columns = linear_sum_assignment(cost)
    assignments = np.full(len(active), -1, dtype=int)
    displacements = np.zeros((len(active), 3), dtype=float)
    for row, column in zip(rows, columns):
        if column < len(current) and squared[row, column] <= gate2:
            assignments[row] = int(column)
            displacements[row] = delta[row, column]
    return assignments, displacements


def _new_track(track_id: int, frame_index: int, atom_index: int, position: np.ndarray) -> _ActiveTrack:
    wrapped = np.asarray(position, dtype=float).copy()
    return _ActiveTrack(
        track_id=track_id,
        frame_indices=[frame_index],
        atom_indices=[atom_index],
        positions=[wrapped.copy()],
        last_wrapped=wrapped,
        last_unwrapped=wrapped.copy(),
    )


def _finish_track(track: _ActiveTrack) -> DefectSegment:
    return DefectSegment(
        track_id=track.track_id,
        frame_indices=np.asarray(track.frame_indices, dtype=np.int64),
        atom_indices=np.asarray(track.atom_indices, dtype=np.int64),
        positions_a=np.asarray(track.positions, dtype=float),
    )


def _positions_array(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.size == 0:
        return np.empty((0, 3), dtype=float)
    if result.ndim != 2 or result.shape[1] != 3:
        raise ValueError("Each defect-position frame must have shape (n_defects, 3).")
    return result


def _indices_array(value: np.ndarray, expected: int) -> np.ndarray:
    result = np.asarray(value, dtype=np.int64)
    if result.shape != (expected,):
        raise ValueError("Each defect-index frame must have shape (n_defects,).")
    return result
