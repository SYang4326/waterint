from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from waterint.chemistry import oxygen_hydrogen_neighbors_by_species
from waterint.common import element_indices, element_mask, iter_frames, parse_cell, resolve_path, selection_context
from waterint.config import require_mapping
from waterint.io.common import TrajectoryFrame
from waterint.sfg.processing import write_cf
from waterint.units import unit_system_from_config


@dataclass(frozen=True)
class SsvvcfResult:
    time_ps: np.ndarray
    corr: np.ndarray
    counts: np.ndarray
    cf_path: Path
    zref_path: Path
    frames: int


@dataclass
class _Segment:
    hydrogen_index: int
    oxygen_index: int
    mu: list[float] = field(default_factory=list)
    stretch: list[float] = field(default_factory=list)


def compute_ssvvcf_from_trajectory(config: dict[str, Any], output_prefix: Path) -> SsvvcfResult:
    input_cfg = require_mapping(config, "input")
    system_cfg = require_mapping(config, "system")
    sfg_cfg = require_mapping(config, "sfg")
    units = unit_system_from_config(config)
    sfg_internal = dict(sfg_cfg)
    if isinstance(sfg_cfg.get("window"), dict):
        window = dict(sfg_cfg["window"])
        for key in ("z1", "z2", "ramp"):
            if key in window:
                window[key] = units.input_length(float(window[key]))
        sfg_internal["window"] = window
    traj_path = resolve_path(config, input_cfg["trajectory"])
    configured_cell = parse_cell(system_cfg.get("cell", "auto"), units)
    context = selection_context(input_cfg)

    frames = list(iter_frames(traj_path, input_cfg, units))
    if len(frames) < 3:
        raise ValueError("SFG trajectory mode needs at least three frames for finite-difference velocities.")
    cell = configured_cell or frames[0].cell
    if cell is None:
        raise ValueError("No cell information was available. Set system.cell manually.")

    if "dt" in sfg_cfg:
        dt_ps = units.input_time_ps(float(sfg_cfg["dt"]))
    else:
        dt_ps = float(sfg_cfg.get("dt_ps", _infer_dt_ps(frames)))
    if dt_ps <= 0:
        raise ValueError("sfg.dt_ps must be positive.")
    if "lag" in sfg_cfg:
        lag_ps = units.input_time_ps(float(sfg_cfg["lag"]))
    else:
        lag_ps = float(sfg_cfg.get("lag_ps", dt_ps * min(200, max(1, len(frames) - 1))))
    max_lag = int(round(lag_ps / dt_ps))
    if max_lag < 1:
        raise ValueError("sfg.lag_ps must be at least one frame.")

    pbc = _pbc_flags(sfg_cfg.get("pbc", [True, True, False]))
    hydrogen_symbol = str(sfg_cfg.get("hydrogen_symbol", "H"))
    oxygen_symbol = str(sfg_cfg.get("oxygen_symbol", "O"))
    oh_cutoff = units.input_length(float(sfg_cfg.get("oh_cutoff", sfg_cfg.get("bond_cutoff", 1.25))))
    neighbor_method = str(sfg_cfg.get("neighbor_method", "auto"))
    neighbor_workers = int(sfg_cfg.get("neighbor_workers", 1))
    oxygen_chunk_size = int(sfg_cfg.get("oxygen_chunk_size", 2048))
    mu_mode = str(sfg_cfg.get("mu_mode", "full")).lower()
    if mu_mode not in {"full", "stretch"}:
        raise ValueError("sfg.mu_mode must be full or stretch.")
    symmetrize = bool(sfg_cfg.get("symmetrize", True))
    flip_sign = bool(sfg_cfg.get("flip_sign", False))

    cf_path = Path(str(output_prefix) + ".dat")
    zref_path = Path(str(output_prefix) + "_zref.dat")
    zrefs = _zref_series(frames, sfg_internal, context)
    _write_zrefs(zref_path, frames, units.output_length(zrefs), units.length_label)

    first = frames[0]
    oxygen_indices = element_indices(first, {oxygen_symbol}, context)
    hydrogen_indices = element_indices(first, {hydrogen_symbol}, context)
    if oxygen_indices.size == 0 or hydrogen_indices.size == 0:
        raise ValueError("SFG trajectory mode could not find O/H atoms. Check input.type_map and sfg symbols.")

    active_oxygen_by_h: dict[int, int] = {}
    active_segments: dict[int, _Segment] = {}
    segments: list[_Segment] = []

    velocities = _finite_difference_velocities(frames, dt_ps=dt_ps, cell=cell, pbc=pbc)
    for frame_index, frame in enumerate(frames):
        positions = frame.positions
        frame_oxygen_indices = element_indices(frame, {oxygen_symbol}, context)
        frame_hydrogen_indices = element_indices(frame, {hydrogen_symbol}, context)
        neighbors_by_species = oxygen_hydrogen_neighbors_by_species(
            frame.symbols,
            positions,
            oxygen_symbol=oxygen_symbol,
            hydrogen_symbol=hydrogen_symbol,
            oh_cutoff=oh_cutoff,
            neighbor_method=neighbor_method,
            neighbor_workers=neighbor_workers,
            oxygen_chunk_size=oxygen_chunk_size,
            oxygen_indices=frame_oxygen_indices,
            hydrogen_indices=frame_hydrogen_indices,
            cell=cell,
            pbc=pbc,
        )
        assigned = _assigned_hydrogens_from_neighbors(
            neighbors_by_species,
            positions=positions,
            cell=cell,
            pbc=pbc,
            duplicate_policy=str(sfg_cfg.get("duplicate_hydrogen_policy", "nearest")),
        )
        missing_hydrogens = set(active_segments) - set(assigned)
        for hydrogen_index in missing_hydrogens:
            segments.append(active_segments.pop(hydrogen_index))
            active_oxygen_by_h.pop(hydrogen_index, None)

        for hydrogen_index, oxygen_index in assigned.items():
            previous_oxygen = active_oxygen_by_h.get(hydrogen_index)
            if previous_oxygen != oxygen_index:
                if hydrogen_index in active_segments:
                    segments.append(active_segments.pop(hydrogen_index))
                active_oxygen_by_h[hydrogen_index] = oxygen_index
                active_segments[hydrogen_index] = _Segment(hydrogen_index, oxygen_index)

            oxygen_position = positions[oxygen_index]
            zprime = float(oxygen_position[2] - zrefs[frame_index])
            fwin = _window_factor(zprime, sfg_internal)
            r_oh = _minimum_image(positions[hydrogen_index] - oxygen_position, cell=cell, pbc=pbc)
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

    segments.extend(active_segments.values())
    sums = np.zeros(max_lag + 1, dtype=float)
    counts = np.zeros(max_lag + 1, dtype=np.int64)
    for segment in segments:
        _accumulate_segment(segment, sums=sums, counts=counts, max_lag=max_lag, symmetrize=symmetrize)

    corr = np.zeros_like(sums)
    mask = counts > 0
    corr[mask] = sums[mask] / counts[mask]
    time_ps = np.arange(max_lag + 1, dtype=float) * dt_ps
    write_cf(cf_path, units.output_time(time_ps), corr, counts, time_label=units.time_label)
    return SsvvcfResult(time_ps=time_ps, corr=corr, counts=counts, cf_path=cf_path, zref_path=zref_path, frames=len(frames))


def _finite_difference_velocities(
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
            velocities[i] = _minimum_image(positions[1] - positions[0], cell=cell, pbc=pbc) / dt_ps
        elif i == len(frames) - 1:
            velocities[i] = _minimum_image(positions[-1] - positions[-2], cell=cell, pbc=pbc) / dt_ps
        else:
            velocities[i] = _minimum_image(positions[i + 1] - positions[i - 1], cell=cell, pbc=pbc) / (2.0 * dt_ps)
    return velocities


def _assigned_hydrogens_from_neighbors(
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
                vector = _minimum_image(positions[hydrogen_index] - positions[oxygen_index], cell=cell, pbc=pbc)
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


def _accumulate_segment(
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
        corr = float(np.dot(mu[:n], stretch[lag:lag + n]))
        if symmetrize:
            corr = (corr + float(np.dot(stretch[:n], mu[lag:lag + n]))) * 2.0
            sums[lag] += corr
            counts[lag] += 2 * n
        else:
            sums[lag] += corr
            counts[lag] += n


def _zref_series(frames: list[TrajectoryFrame], sfg_cfg: dict[str, Any], context: dict[str, Any]) -> np.ndarray:
    if sfg_cfg.get("zref_file"):
        data = np.loadtxt(sfg_cfg["zref_file"], comments="#")
        if data.ndim == 1:
            values = data
        else:
            col = int(sfg_cfg.get("zref_col", 0))
            values = data[:, col - 1 if col > 0 else -1]
        if values.size < len(frames):
            raise ValueError("sfg.zref_file has fewer values than trajectory frames.")
        return np.asarray(values[:len(frames)], dtype=float)

    reference_cfg = sfg_cfg.get("reference", {})
    if isinstance(reference_cfg, dict) and reference_cfg.get("species"):
        species = {str(item) for item in reference_cfg["species"]}
        surface = str(reference_cfg.get("surface", "mean")).lower()
        values = []
        for frame in frames:
            mask = element_mask(frame, species, context)
            z = frame.positions[mask, 2]
            if z.size == 0:
                raise ValueError(f"SFG reference selection found no atoms: {species}")
            if surface == "max":
                values.append(float(np.max(z)))
            elif surface == "min":
                values.append(float(np.min(z)))
            elif surface == "mean":
                values.append(float(np.mean(z)))
            else:
                raise ValueError("sfg.reference.surface must be max, min, or mean.")
        return np.asarray(values, dtype=float)

    return np.full(len(frames), float(sfg_cfg.get("z_ref0", 0.0)), dtype=float)


def _write_zrefs(path: Path, frames: list[TrajectoryFrame], zrefs: np.ndarray, length_label: str = "A") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"# frame_idx  step  zref_{length_label}\n")
        for frame, zref in zip(frames, zrefs):
            step = frame.step if frame.step is not None else frame.index
            handle.write(f"{frame.index} {step} {zref:.8f}\n")


def _window_factor(zprime: float, sfg_cfg: dict[str, Any]) -> float:
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
        return _slab_interface_decay(zprime, z1, z2, ramp, flip)
    if mode == 2:
        return _top_hat_ramp(zprime, z1, z2, ramp)
    raise ValueError("sfg.window.mode must be 1 or 2.")


def _slab_interface_decay(z: float, z1: float, z2: float, ramp: float, flip: bool) -> float:
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
        return _ramp_sin01((z2 - z) / ramp)
    if z <= z1:
        return 0.0
    if ramp <= 0 or z >= z1 + ramp:
        return 1.0
    return _ramp_sin01((z - z1) / ramp)


def _top_hat_ramp(z: float, z1: float, z2: float, ramp: float) -> float:
    if z2 < z1:
        z1, z2 = z2, z1
    if z < z1 or z > z2:
        return 0.0
    if ramp <= 0:
        return 1.0
    ramp = min(ramp, 0.5 * (z2 - z1))
    if z < z1 + ramp:
        return _ramp_sin01((z - z1) / ramp)
    if z > z2 - ramp:
        return _ramp_sin01((z2 - z) / ramp)
    return 1.0


def _ramp_sin01(value: float) -> float:
    if value <= 0:
        return 0.0
    if value >= 1:
        return 1.0
    return float(np.sin(0.5 * np.pi * value))


def _minimum_image(vectors: np.ndarray, *, cell: tuple[float, float, float], pbc: tuple[bool, bool, bool]) -> np.ndarray:
    out = np.asarray(vectors, dtype=float).copy()
    cell_array = np.asarray(cell, dtype=float)
    for axis, enabled in enumerate(pbc):
        if enabled:
            out[..., axis] -= np.rint(out[..., axis] / cell_array[axis]) * cell_array[axis]
    return out


def _pbc_flags(value: Any) -> tuple[bool, bool, bool]:
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
