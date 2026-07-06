from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import numpy as np

from waterint.io.common import TrajectoryFrame


def read_lammpstrj(
    path: str | Path,
    *,
    type_map: dict[Any, str] | None = None,
    start_timestep: int | None = None,
    stride: int = 1,
    max_frames: int | None = None,
) -> Iterator[TrajectoryFrame]:
    """Stream orthorhombic LAMMPS dump frames.

    Supported atom columns include `id type x y z`, with optional extra columns.
    Triclinic tilt factors are not interpreted yet.
    """
    dump_path = Path(path)
    symbol_by_type = _normalize_type_map(type_map or {})
    if stride <= 0:
        raise ValueError("stride must be > 0.")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be positive or None.")

    with dump_path.open("r", encoding="utf-8", errors="replace") as handle:
        frame_index = 0
        yielded = 0
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
                symbols.append(symbol_by_type.get(type_id, str(type_id)))
                positions[atom_i] = [
                    float(fields[col_index["x"]]),
                    float(fields[col_index["y"]]),
                    float(fields[col_index["z"]]),
                ]

            yield TrajectoryFrame(
                index=frame_index,
                comment=f"timestep {timestep}",
                symbols=symbols,
                positions=positions,
                cell=cell,
                step=timestep,
                types=types,
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


def _skip_lines(handle, count: int) -> None:
    for _ in range(count):
        if not handle.readline():
            raise ValueError("Unexpected end of file while skipping LAMMPS atom rows.")
