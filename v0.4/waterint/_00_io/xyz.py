from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np

from waterint._00_io.common import TrajectoryFrame


def read_xyz(
    path: str | Path,
    *,
    reader: str = "python",
    max_frames: int | None = None,
) -> Iterator[TrajectoryFrame]:
    mode = str(reader).lower()
    if mode not in {"python", "auto", "cpp"}:
        raise ValueError("XYZ reader must be python, auto, or cpp.")
    if mode in {"auto", "cpp"}:
        native_frames = _read_xyz_cpp(path, max_frames=max_frames)
        if native_frames is not None:
            yield from native_frames
            return
        if mode == "cpp":
            raise RuntimeError("C++ XYZ reader is unavailable. Use input.reader: auto or python.")
    yield from _read_xyz_python(path)


def _read_xyz_python(path: str | Path) -> Iterator[TrajectoryFrame]:
    xyz_path = Path(path)
    with xyz_path.open("r", encoding="utf-8") as handle:
        frame_index = 0
        while True:
            natoms_line = handle.readline()
            if not natoms_line:
                return
            if not natoms_line.strip():
                continue
            try:
                natoms = int(natoms_line.strip())
            except ValueError as exc:
                raise ValueError(f"Bad XYZ atom-count line in {xyz_path}: {natoms_line!r}") from exc

            comment = handle.readline()
            if not comment:
                raise ValueError(f"Unexpected EOF after atom count in {xyz_path}")

            symbols: list[str] = []
            positions = np.empty((natoms, 3), dtype=float)
            for atom_index in range(natoms):
                row = handle.readline()
                if not row:
                    raise ValueError(f"Unexpected EOF inside frame {frame_index} in {xyz_path}")
                fields = row.split()
                if len(fields) < 4:
                    raise ValueError(f"Bad XYZ atom row in {xyz_path}: {row!r}")
                symbols.append(fields[0])
                try:
                    positions[atom_index] = [float(fields[1]), float(fields[2]), float(fields[3])]
                except ValueError as exc:
                    raise ValueError(f"Bad XYZ coordinate row in {xyz_path}: {row!r}") from exc

            yield TrajectoryFrame(
                index=frame_index,
                comment=comment.rstrip("\n"),
                symbols=symbols,
                positions=positions,
            )
            frame_index += 1


def _read_xyz_cpp(path: str | Path, *, max_frames: int | None) -> Iterator[TrajectoryFrame] | None:
    from waterint._02_computation._native import read_xyz_file

    loaded = read_xyz_file(path, max_frames=max_frames)
    if loaded is None:
        return None
    positions, symbols = loaded

    def iter_loaded() -> Iterator[TrajectoryFrame]:
        for frame_index in range(positions.shape[0]):
            yield TrajectoryFrame(
                index=frame_index,
                comment="",
                symbols=symbols,
                positions=positions[frame_index],
            )

    return iter_loaded()
