from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from waterint.io.common import TrajectoryFrame
from waterint.io.lammpstrj import read_lammpstrj
from waterint.io.npz import read_npz
from waterint.io.xyz import read_xyz


AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


def parse_axis(value: Any) -> tuple[str, int, float]:
    axis_label = str(value).strip().lower()
    sign = -1.0 if axis_label.startswith("-") else 1.0
    bare_axis = axis_label[1:] if axis_label.startswith("-") else axis_label
    if bare_axis not in AXIS_INDEX:
        raise ValueError("axis must be one of x, y, z, -x, -y, -z.")
    return axis_label, AXIS_INDEX[bare_axis], sign


def parse_range(value: Any, *, name: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{name} must be [min, max].")
    range_min, range_max = float(value[0]), float(value[1])
    if not range_max > range_min:
        raise ValueError(f"{name} max must be larger than min.")
    return range_min, range_max


def parse_cell(value: Any) -> tuple[float, float, float] | None:
    if value is None or str(value).lower() == "auto":
        return None
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError("system.cell must be [Lx, Ly, Lz] in Angstrom or auto.")
    cell = tuple(float(v) for v in value)
    if any(v <= 0 for v in cell):
        raise ValueError("system.cell values must be positive.")
    return cell


def resolve_path(config: dict[str, Any], path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return Path(config["_config_dir"]) / path


def selection_context(input_cfg: dict[str, Any]) -> dict[str, Any]:
    raw_type_map = input_cfg.get("type_map", {})
    symbol_to_types: dict[str, list[int]] = {}
    if isinstance(raw_type_map, dict):
        for raw_type, raw_symbol in raw_type_map.items():
            symbol_to_types.setdefault(str(raw_symbol), []).append(int(raw_type))
    return {"symbol_to_types": symbol_to_types}


def element_indices(
    frame: TrajectoryFrame,
    species_set: set[str],
    context: dict[str, Any],
) -> np.ndarray:
    return np.where(element_mask(frame, species_set, context))[0]


def element_mask(
    frame: TrajectoryFrame,
    species_set: set[str],
    context: dict[str, Any],
) -> np.ndarray:
    symbol_to_types = context.get("symbol_to_types", {})
    if frame.types is not None and symbol_to_types:
        type_ids: list[int] = []
        for species in species_set:
            type_ids.extend(symbol_to_types.get(species, []))
        if type_ids:
            return np.isin(frame.types, type_ids)
    return np.isin(np.asarray(frame.symbols), list(species_set))


def iter_frames(traj_path: Path, input_cfg: dict[str, Any]):
    fmt = str(input_cfg.get("format", "xyz")).lower()
    stride = int(input_cfg.get("stride", 1))
    max_frames_raw = input_cfg.get("max_frames", None)
    max_frames = None if max_frames_raw in {None, 0, "all"} else int(max_frames_raw)
    start_timestep_raw = input_cfg.get("start_timestep", None)
    start_timestep = None if start_timestep_raw in {None, ""} else int(start_timestep_raw)
    if stride <= 0:
        raise ValueError("input.stride must be > 0.")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("input.max_frames must be positive, 0, 'all', or omitted.")

    if fmt == "xyz":
        frames = read_xyz(traj_path)
    elif fmt == "lammpstrj":
        yield from read_lammpstrj(
            traj_path,
            type_map=input_cfg.get("type_map", {}),
            start_timestep=start_timestep,
            stride=stride,
            max_frames=max_frames,
        )
        return
    elif fmt == "npz":
        frames = read_npz(traj_path, type_map=input_cfg.get("type_map", {}))
    else:
        raise ValueError("input.format must be xyz, lammpstrj, or npz.")

    yielded = 0
    for frame in frames:
        if start_timestep is not None:
            if frame.step is None:
                raise ValueError("input.start_timestep requires trajectory frames with timestep information.")
            if frame.step < start_timestep:
                continue
        if frame.index % stride != 0:
            continue
        yield frame
        yielded += 1
        if max_frames is not None and yielded >= max_frames:
            return


def reference_value(
    frame: TrajectoryFrame,
    axis: int,
    reference_cfg: dict[str, Any],
    context: dict[str, Any],
) -> float:
    ref_type = str(reference_cfg.get("type", "element_mean"))
    if ref_type != "element_mean":
        raise ValueError("Only reference.type: element_mean is implemented.")

    ref_species = reference_cfg.get("species")
    if not ref_species or not isinstance(ref_species, list):
        raise ValueError("reference.species must be a non-empty list.")
    mask = element_mask(frame, {str(item) for item in ref_species}, context)
    if not np.any(mask):
        raise ValueError(f"Reference selection found no atoms: {ref_species}")
    return float(np.mean(frame.positions[mask, axis]))


def slab_reference_value(
    frame: TrajectoryFrame,
    axis: int,
    reference_cfg: dict[str, Any],
    axis_sign: float,
    context: dict[str, Any],
) -> float:
    ref_type = str(reference_cfg.get("type", "slab_surface"))
    if ref_type not in {"slab_surface", "element_surface"}:
        raise ValueError("relative_to_slab requires reference.type: slab_surface.")

    slab_species = reference_cfg.get("species")
    if not slab_species or not isinstance(slab_species, list):
        raise ValueError("reference.species must list slab atom symbols, e.g. ['Mg'].")

    mask = element_mask(frame, {str(item) for item in slab_species}, context)
    values = frame.positions[mask, axis]
    if values.size == 0:
        raise ValueError(f"Slab reference selection found no atoms: {slab_species}")

    surface = str(reference_cfg.get("surface", "auto")).lower()
    if surface == "auto":
        surface = "max" if axis_sign > 0 else "min"
    if surface == "max":
        return float(np.max(values))
    if surface == "min":
        return float(np.min(values))
    if surface == "mean":
        return float(np.mean(values))
    raise ValueError("reference.surface must be auto, max, min, or mean.")
