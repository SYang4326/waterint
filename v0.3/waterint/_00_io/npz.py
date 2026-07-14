from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np

from waterint._00_io.common import TrajectoryFrame
from waterint._00_io.lammpstrj import read_lammpstrj


def read_npz(path: str | Path, *, type_map: dict | None = None) -> Iterator[TrajectoryFrame]:
    cache_path = Path(path)
    symbol_by_type = _normalize_type_map(type_map or {})
    with np.load(cache_path) as data:
        positions = data["positions"]
        types = data["types"]
        cells = data["cells"]
        steps = data["steps"]
        velocities = data["velocities"] if "velocities" in data else None
        shared_symbols = None
        first_types = types[0] if types.shape[0] else None
        if first_types is not None and np.all(types == first_types):
            shared_symbols = _symbols_from_types(first_types, symbol_by_type)
        for frame_index in range(positions.shape[0]):
            frame_types = types[frame_index]
            symbols = shared_symbols if shared_symbols is not None else _symbols_from_types(frame_types, symbol_by_type)
            yield TrajectoryFrame(
                index=frame_index,
                comment=f"timestep {int(steps[frame_index])}",
                symbols=symbols,
                positions=positions[frame_index],
                cell=tuple(float(v) for v in cells[frame_index]),
                step=int(steps[frame_index]),
                types=frame_types,
                velocities=None if velocities is None else velocities[frame_index],
            )


def write_npz_from_lammpstrj(
    *,
    trajectory_path: str | Path,
    output_path: str | Path,
    type_map: dict | None = None,
    start_timestep: int | None = None,
    stride: int = 1,
    max_frames: int | None = None,
) -> Path:
    frames = list(
        read_lammpstrj(
            trajectory_path,
            type_map=type_map,
            start_timestep=start_timestep,
            stride=stride,
            max_frames=max_frames,
        )
    )
    if not frames:
        raise ValueError(f"No frames found in trajectory: {trajectory_path}")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    positions = np.stack([frame.positions for frame in frames])
    types = np.stack([frame.types for frame in frames])
    cells = np.asarray([frame.cell for frame in frames], dtype=float)
    steps = np.asarray([frame.step if frame.step is not None else frame.index for frame in frames], dtype=int)
    velocity_frames = [frame.velocities for frame in frames]
    if any(velocity is not None for velocity in velocity_frames) and not all(velocity is not None for velocity in velocity_frames):
        raise ValueError("Cannot write an NPZ trajectory with velocities missing from only some frames.")
    payload = {"positions": positions, "types": types, "cells": cells, "steps": steps}
    if velocity_frames and velocity_frames[0] is not None:
        payload["velocities"] = np.stack(velocity_frames)
    np.savez(output, **payload)
    return output


def _normalize_type_map(raw: dict) -> dict[int, str]:
    out: dict[int, str] = {}
    for key, value in raw.items():
        out[int(key)] = str(value)
    return out


def _symbols_from_types(types: np.ndarray, symbol_by_type: dict[int, str]) -> list[str]:
    return [symbol_by_type.get(int(type_id), str(int(type_id))) for type_id in types]
