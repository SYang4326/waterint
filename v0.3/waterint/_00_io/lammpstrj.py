from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import numpy as np

from waterint._00_io.common import TrajectoryFrame


def read_lammpstrj(
    path: str | Path,
    *,
    type_map: dict[Any, str] | None = None,
    start_timestep: int | None = None,
    stride: int = 1,
    max_frames: int | None = None,
    reader: str = "python",
) -> Iterator[TrajectoryFrame]:
    """Stream orthorhombic LAMMPS dump frames.

    Supported atom columns include `id type x y z` and optional `vx vy vz`.
    Triclinic tilt factors are not interpreted yet.
    """
    dump_path = Path(path)
    symbol_by_type = _normalize_type_map(type_map or {})
    if stride <= 0:
        raise ValueError("stride must be > 0.")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be positive or None.")
    mode = str(reader).lower()
    if mode not in {"python", "auto", "cpp"}:
        raise ValueError("LAMMPS dump reader must be python, auto, or cpp.")
    if mode in {"auto", "cpp"} and start_timestep is None and stride == 1:
        native_frames = _read_lammpstrj_cpp(dump_path, symbol_by_type=symbol_by_type, max_frames=max_frames)
        if native_frames is not None:
            yield from native_frames
            return
        if mode == "cpp":
            raise RuntimeError("C++ LAMMPS dump reader is unavailable. Use input.reader: auto or python.")
    elif mode == "cpp":
        raise ValueError("input.reader: cpp for lammpstrj currently requires start_timestep unset and stride: 1.")

    with dump_path.open("r", encoding="utf-8", errors="replace") as handle:
        frame_index = 0
        yielded = 0
        shared_symbols = None
        shared_types = None
        velocity_columns_expected: bool | None = None
        while True:
            line = handle.readline()
            if not line:
                return
            if line.strip() != "ITEM: TIMESTEP":
                continue

            timestep = int(handle.readline().strip())

            number_header = handle.readline().strip()
            if number_header != "ITEM: NUMBER OF ATOMS":
                raise ValueError(f"Expected ITEM: NUMBER OF ATOMS in {dump_path}, got {number_header!r}")
            natoms = int(handle.readline().strip())

            box_header = handle.readline().strip()
            if not box_header.startswith("ITEM: BOX BOUNDS"):
                raise ValueError(f"Expected ITEM: BOX BOUNDS in {dump_path}, got {box_header!r}")
            bounds = [_parse_bound_line(handle.readline()) for _ in range(3)]
            cell = tuple(hi - lo for lo, hi in bounds)

            atoms_header = handle.readline().strip()
            if not atoms_header.startswith("ITEM: ATOMS"):
                raise ValueError(f"Expected ITEM: ATOMS in {dump_path}, got {atoms_header!r}")
            columns = atoms_header.split()[2:]
            col_index = {name: i for i, name in enumerate(columns)}
            required = ["type", "x", "y", "z"]
            missing = [name for name in required if name not in col_index]
            if missing:
                raise ValueError(f"LAMMPS dump missing required atom columns: {missing}")
            velocity_columns = ["vx", "vy", "vz"]
            present_velocity_columns = [name for name in velocity_columns if name in col_index]
            if present_velocity_columns and len(present_velocity_columns) != len(velocity_columns):
                raise ValueError("LAMMPS dump must include vx, vy, and vz together.")
            has_velocities = bool(present_velocity_columns)
            if velocity_columns_expected is None:
                velocity_columns_expected = has_velocities
            elif has_velocities != velocity_columns_expected:
                raise ValueError("LAMMPS dump velocity columns must be present in every frame or none.")
            velocities = np.empty((natoms, 3), dtype=float) if present_velocity_columns else None

            should_yield = (
                (start_timestep is None or timestep >= start_timestep)
                and frame_index % stride == 0
            )
            if not should_yield:
                _skip_lines(handle, natoms)
                frame_index += 1
                continue

            symbols: list[str] = []
            types = np.empty(natoms, dtype=int)
            positions = np.empty((natoms, 3), dtype=float)
            for atom_i in range(natoms):
                fields = handle.readline().split()
                if len(fields) < len(columns):
                    raise ValueError(f"Bad atom row in frame {frame_index}: {fields!r}")
                type_id = int(float(fields[col_index["type"]]))
                types[atom_i] = type_id
                positions[atom_i] = [
                    float(fields[col_index["x"]]),
                    float(fields[col_index["y"]]),
                    float(fields[col_index["z"]]),
                ]
                if velocities is not None:
                    velocities[atom_i] = [
                        float(fields[col_index["vx"]]),
                        float(fields[col_index["vy"]]),
                        float(fields[col_index["vz"]]),
                    ]
            if shared_types is None:
                shared_types = types.copy()
                shared_symbols = _symbols_from_types(types, symbol_by_type)
            if shared_symbols is not None and np.array_equal(types, shared_types):
                symbols = shared_symbols
            else:
                symbols = _symbols_from_types(types, symbol_by_type)

            yield TrajectoryFrame(
                index=frame_index,
                comment=f"timestep {timestep}",
                symbols=symbols,
                positions=positions,
                cell=cell,
                step=timestep,
                types=types,
                velocities=velocities,
            )
            yielded += 1
            if max_frames is not None and yielded >= max_frames:
                return
            frame_index += 1


def _parse_bound_line(line: str) -> tuple[float, float]:
    fields = line.split()
    if len(fields) < 2:
        raise ValueError(f"Bad BOX BOUNDS row: {line!r}")
    return float(fields[0]), float(fields[1])


def _normalize_type_map(raw: dict[Any, str]) -> dict[int, str]:
    out: dict[int, str] = {}
    for key, value in raw.items():
        out[int(key)] = str(value)
    return out


def _symbols_from_types(types: np.ndarray, symbol_by_type: dict[int, str]) -> list[str]:
    return [symbol_by_type.get(int(type_id), str(int(type_id))) for type_id in types]


def _read_lammpstrj_cpp(
    path: Path,
    *,
    symbol_by_type: dict[int, str],
    max_frames: int | None,
) -> Iterator[TrajectoryFrame] | None:
    from waterint._02_computation._native import read_lammpstrj_file

    loaded = read_lammpstrj_file(path, max_frames=max_frames)
    if loaded is None:
        return None
    positions, types, cells, steps, velocities = loaded
    shared_symbols = _symbols_from_types(types[0], symbol_by_type)

    def iter_loaded() -> Iterator[TrajectoryFrame]:
        for frame_index in range(positions.shape[0]):
            frame_types = types[frame_index]
            symbols = shared_symbols if np.array_equal(frame_types, types[0]) else _symbols_from_types(frame_types, symbol_by_type)
            step = int(steps[frame_index])
            yield TrajectoryFrame(
                index=frame_index,
                comment=f"timestep {step}",
                symbols=symbols,
                positions=positions[frame_index],
                cell=tuple(float(v) for v in cells[frame_index]),
                step=step,
                types=frame_types,
                velocities=None if velocities is None else velocities[frame_index],
            )

    return iter_loaded()


def _skip_lines(handle, count: int) -> None:
    for _ in range(count):
        if not handle.readline():
            raise ValueError("Unexpected end of file while skipping LAMMPS atom rows.")
