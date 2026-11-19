"""Streaming block reader for orthogonal LAMMPS dumps (id type x y z [vx vy vz]); C++ parser, first-frame atom order."""
from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from waterint.backends.cpu.native import lammps_block_library


@dataclass
class LammpsBlock:
    ids: np.ndarray
    types: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray | None
    steps: np.ndarray
    boxes: np.ndarray          # [n, 6] xlo xhi ylo yhi zlo zhi


def lammpstrj_natoms(path: str | Path) -> int:
    with open(path, "r", encoding="utf-8") as handle:
        for _ in range(3):
            handle.readline()
        return int(handle.readline())


def _ptr(array, ctype):
    return None if array is None else array.ctypes.data_as(ctypes.POINTER(ctype))


def iter_lammpstrj_blocks(path: str | Path, *, block_frames: int = 1000, max_frames: int | None = None) -> Iterator[LammpsBlock]:
    lib = lammps_block_library()
    nat = lammpstrj_natoms(path)
    handle = lib.waterint_lmp_open(str(path).encode())
    if not handle:
        message = lib.waterint_sc_error()
        raise OSError(message.decode() if message else f"cannot open {path}")
    try:
        ids = np.zeros(nat, dtype=np.int32)
        types = np.zeros(nat, dtype=np.int16)
        done = 0
        while max_frames is None or done < max_frames:
            want = block_frames if max_frames is None else min(block_frames, max_frames - done)
            pos = np.empty((want, nat, 3), dtype=np.float64)
            vel = np.empty((want, nat, 3), dtype=np.float64)
            steps = np.empty(want, dtype=np.int64)
            boxes = np.empty((want, 6), dtype=np.float64)
            nread = ctypes.c_int64(0)
            has_vel = ctypes.c_int(0)
            status = lib.waterint_lmp_read(handle, want, nat, _ptr(ids, ctypes.c_int32), _ptr(types, ctypes.c_int16),
                                           _ptr(pos, ctypes.c_double), _ptr(vel, ctypes.c_double), _ptr(steps, ctypes.c_int64),
                                           _ptr(boxes, ctypes.c_double), ctypes.byref(nread), ctypes.byref(has_vel))
            if status != 0:
                message = lib.waterint_sc_error()
                raise RuntimeError(f"LAMMPS reader: {message.decode() if message else status}")
            n = nread.value
            if n == 0:
                return
            done += n
            yield LammpsBlock(ids.copy(), types.copy(), pos[:n], vel[:n] if has_vel.value else None, steps[:n], boxes[:n])
            if n < want:
                return
    finally:
        lib.waterint_lmp_close(handle)
