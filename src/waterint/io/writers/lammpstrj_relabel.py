"""Relabel stream for LAMMPS dumps: read frames (repeated timesteps skipped), write them back with new types.

    with RelabelStream(paths, out_path, type_map) as rs:
        for frame, ids in rs:            # TrajectoryFrame (symbols from type_map) + atom IDs
            rs.write(new_types)          # every token except the type column is copied verbatim

C++ implementation (backends/cpu/lmp_stream.cpp) with a pure-Python fallback of identical text semantics
(whitespace-split tokens re-joined by single spaces, as the v0.6 relabeler did).
"""
from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

from waterint.io.frame import TrajectoryFrame


class RelabelStream:
    def __init__(self, paths: Sequence[str | Path], out_path: str | Path | None, type_map: dict[int, str], *,
                 backend: str = "auto"):
        self.paths = [str(p) for p in paths]
        self.out_path = None if out_path is None else str(out_path)
        self.type_map = {int(k): str(v) for k, v in type_map.items()}
        self.duplicates = 0
        self._lib = None
        if backend in {"auto", "cpu", "cpp"}:
            from waterint.backends.cpu.native import native_library
            lib = native_library()
            if lib is not None:
                self._bind(lib)
            elif backend in {"cpu", "cpp"}:
                raise RuntimeError("C++ relabel stream unavailable.")
        self.backend = "cpu" if self._lib is not None else "python"

    # ------------------------------------------------------------------------------------------- native
    def _bind(self, lib) -> None:
        vp = ctypes.c_void_p
        lib.waterint_lmps_error.restype = ctypes.c_char_p
        lib.waterint_lmps_open.restype = vp
        lib.waterint_lmps_open.argtypes = [ctypes.POINTER(ctypes.c_char_p), ctypes.c_int, ctypes.c_char_p]
        lib.waterint_lmps_next.restype = ctypes.c_int
        lib.waterint_lmps_next.argtypes = [vp, ctypes.c_longlong, ctypes.POINTER(ctypes.c_longlong), ctypes.POINTER(ctypes.c_longlong),
                                           ctypes.POINTER(ctypes.c_longlong), ctypes.POINTER(ctypes.c_int),
                                           ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                           ctypes.POINTER(ctypes.c_double)]
        lib.waterint_lmps_write.restype = ctypes.c_int
        lib.waterint_lmps_write.argtypes = [vp, ctypes.POINTER(ctypes.c_int)]
        lib.waterint_lmps_duplicates.restype = ctypes.c_longlong
        lib.waterint_lmps_duplicates.argtypes = [vp]
        lib.waterint_lmps_close.argtypes = [vp]
        self._lib = lib

    def _error(self) -> str:
        msg = self._lib.waterint_lmps_error()
        return msg.decode() if msg else "unknown error"

    def __enter__(self) -> "RelabelStream":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __iter__(self) -> Iterator[tuple[TrajectoryFrame, np.ndarray]]:
        return self._iter_native() if self._lib is not None else self._iter_python()

    def _frame(self, index: int, step: int, types, pos, bounds, tilt) -> TrajectoryFrame:
        """Frame from raw BOX BOUNDS (+ tilt). Tilted boxes: LAMMPS bounding box -> origin/lengths and cell vectors."""
        symbols = [self.type_map.get(int(t), str(int(t))) for t in types]
        xy, xz, yz = (float(v) for v in tilt)
        xlo, xhi, ylo, yhi, zlo, zhi = (float(v) for v in bounds)
        if xy or xz or yz:
            xlo -= min(0.0, xy, xz, xy + xz)
            xhi -= max(0.0, xy, xz, xy + xz)
            ylo -= min(0.0, yz)
            yhi -= max(0.0, yz)
            lx, ly, lz = xhi - xlo, yhi - ylo, zhi - zlo
            vectors = np.array([[lx, 0.0, 0.0], [xy, ly, 0.0], [xz, yz, lz]])
            return TrajectoryFrame(index=index, comment=f"timestep {step}", symbols=symbols, positions=pos, cell=(lx, ly, lz),
                                   step=int(step), types=np.asarray(types, dtype=int), cell_vectors=vectors,
                                   cell_origin=(xlo, ylo, zlo), triclinic=True)
        return TrajectoryFrame(index=index, comment=f"timestep {step}", symbols=symbols, positions=pos,
                               cell=(xhi - xlo, yhi - ylo, zhi - zlo), step=int(step), types=np.asarray(types, dtype=int),
                               cell_origin=(xlo, ylo, zlo))

    def _iter_native(self):
        lib = self._lib
        arr = (ctypes.c_char_p * len(self.paths))(*[p.encode() for p in self.paths])
        self._h = lib.waterint_lmps_open(arr, len(self.paths), (self.out_path or "").encode())
        if not self._h:
            raise OSError(self._error())
        from waterint.io.readers.lammpstrj_blocks import lammpstrj_natoms
        cap = lammpstrj_natoms(self.paths[0])
        ids = np.zeros(cap, dtype=np.int64)
        types = np.zeros(cap, dtype=np.intc)
        bounds = np.zeros(6)
        tilt = np.zeros(3)
        n, step = ctypes.c_longlong(0), ctypes.c_longlong(0)
        index = 0
        p = lambda a, t: a.ctypes.data_as(ctypes.POINTER(t))
        while True:
            pos = np.zeros((cap, 3))
            r = lib.waterint_lmps_next(self._h, cap, ctypes.byref(n), ctypes.byref(step), p(ids, ctypes.c_longlong),
                                       p(types, ctypes.c_int), p(pos, ctypes.c_double), p(bounds, ctypes.c_double),
                                       p(tilt, ctypes.c_double))
            if r == 0:
                break
            if r < 0:
                raise RuntimeError(f"relabel stream: {self._error()}")
            yield self._frame(index, step.value, types[: n.value].copy(), pos[: n.value], bounds.copy(), tilt.copy()), ids[: n.value].copy()
            index += 1
        self.duplicates = int(lib.waterint_lmps_duplicates(self._h))

    def _iter_python(self):
        seen: set[str] = set()
        self._out = open(self.out_path, "w", encoding="utf-8", newline="\n") if self.out_path else None
        index = 0
        for path in self.paths:
            with open(path, "r", encoding="utf-8") as fh:
                while True:
                    line = fh.readline()
                    if not line:
                        break
                    if not line.startswith("ITEM: TIMESTEP"):
                        continue
                    header = [line.rstrip("\n")] + [fh.readline().rstrip("\n") for _ in range(8)]
                    cols = header[8].split()[2:]
                    natoms = int(header[3])
                    rows = [fh.readline().split() for _ in range(natoms)]
                    key = header[1].split()[0] if header[1].split() else header[1]
                    if key in seen:
                        self.duplicates += 1
                        continue
                    seen.add(key)
                    c = {name: k for k, name in enumerate(cols)}
                    cx = [c.get(a, c.get(a + "u")) for a in ("x", "y", "z")]
                    self._cur = (header, rows, c["type"])
                    ids = np.array([int(r[c["id"]]) for r in rows], dtype=np.int64)
                    types = np.array([int(r[c["type"]]) for r in rows])
                    pos = np.array([[float(r[k]) for k in cx] for r in rows])
                    bounds = np.array([float(v) for row in header[5:8] for v in row.split()[:2]])
                    tilt = np.array([float(row.split()[2]) if len(row.split()) > 2 else 0.0 for row in header[5:8]])
                    yield self._frame(index, int(key), types, pos, bounds, tilt), ids
                    index += 1

    def write(self, new_types: np.ndarray) -> None:
        codes = np.ascontiguousarray(new_types, dtype=np.intc)
        if self._lib is not None:
            if self._lib.waterint_lmps_write(self._h, codes.ctypes.data_as(ctypes.POINTER(ctypes.c_int))) != 0:
                raise RuntimeError(f"relabel stream write: {self._error()}")
            return
        header, rows, ct = self._cur
        out = self._out
        for h in header:
            out.write(h + "\n")
        for r, code in zip(rows, codes):
            r[ct] = str(int(code))
            out.write(" ".join(r) + "\n")

    def close(self) -> None:
        if self._lib is not None and getattr(self, "_h", None):
            self.duplicates = int(self._lib.waterint_lmps_duplicates(self._h))
            self._lib.waterint_lmps_close(self._h)
            self._h = None
        if getattr(self, "_out", None):
            self._out.close()
            self._out = None
