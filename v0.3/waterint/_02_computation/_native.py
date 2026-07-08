from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sysconfig
from pathlib import Path

import numpy as np


class _XYZData(ctypes.Structure):
    _fields_ = [
        ("n_frames", ctypes.c_size_t),
        ("n_atoms", ctypes.c_size_t),
        ("positions", ctypes.POINTER(ctypes.c_double)),
        ("symbols", ctypes.POINTER(ctypes.c_char)),
        ("symbols_size", ctypes.c_size_t),
        ("error", ctypes.POINTER(ctypes.c_char)),
    ]


class _LammpstrjData(ctypes.Structure):
    _fields_ = [
        ("n_frames", ctypes.c_size_t),
        ("n_atoms", ctypes.c_size_t),
        ("positions", ctypes.POINTER(ctypes.c_double)),
        ("types", ctypes.POINTER(ctypes.c_int64)),
        ("cells", ctypes.POINTER(ctypes.c_double)),
        ("steps", ctypes.POINTER(ctypes.c_int64)),
        ("error", ctypes.POINTER(ctypes.c_char)),
    ]


_LIB_CACHE: ctypes.CDLL | None = None
_LOAD_ATTEMPTED = False
_LAST_ERROR: str | None = None


def density_histogram_edges(
    values: np.ndarray,
    *,
    bin_edges: np.ndarray,
) -> np.ndarray | None:
    lib = native_library()
    if lib is None:
        return None

    values_array = np.ascontiguousarray(values, dtype=np.float64)
    edges_array = np.ascontiguousarray(bin_edges, dtype=np.float64)
    if edges_array.ndim != 1 or edges_array.size < 2:
        raise ValueError("bin_edges must be a one-dimensional array with at least two entries.")
    counts = np.zeros(edges_array.size - 1, dtype=np.float64)
    status = lib.waterint_density_histogram_edges(
        values_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_size_t(values_array.size),
        edges_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_size_t(counts.size),
        counts.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    if status != 0:
        raise RuntimeError(f"C++ density histogram failed with status {status}.")
    return counts


def count_hydrogen_neighbors(
    oxygen_positions: np.ndarray,
    hydrogen_positions: np.ndarray,
    *,
    cutoff: float,
    cell: tuple[float, float, float] | None = None,
    pbc: tuple[bool, bool, bool] | None = None,
) -> np.ndarray | None:
    lib = native_library()
    if lib is None:
        return None

    oxygen_array = np.ascontiguousarray(oxygen_positions, dtype=np.float64)
    hydrogen_array = np.ascontiguousarray(hydrogen_positions, dtype=np.float64)
    if oxygen_array.ndim != 2 or oxygen_array.shape[1] != 3:
        raise ValueError("oxygen_positions must have shape (n, 3).")
    if hydrogen_array.ndim != 2 or hydrogen_array.shape[1] != 3:
        raise ValueError("hydrogen_positions must have shape (n, 3).")

    counts = np.zeros(oxygen_array.shape[0], dtype=np.int64)
    cell_pointer = None
    pbc_pointer = None
    if pbc is not None and any(pbc):
        if cell is None:
            raise ValueError("cell is required when pbc is enabled.")
        cell_array = np.ascontiguousarray(cell, dtype=np.float64)
        pbc_array = np.ascontiguousarray([1 if flag else 0 for flag in pbc], dtype=np.uint8)
        cell_pointer = cell_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        pbc_pointer = pbc_array.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))

    status = lib.waterint_count_hydrogen_neighbors(
        oxygen_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_size_t(oxygen_array.shape[0]),
        hydrogen_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_size_t(hydrogen_array.shape[0]),
        ctypes.c_double(float(cutoff)),
        cell_pointer,
        pbc_pointer,
        counts.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
    )
    if status != 0:
        raise RuntimeError(f"C++ O-H neighbor count failed with status {status}.")
    return counts


def classify_oxygen_by_h_count_compact(
    oxygen_positions: np.ndarray,
    hydrogen_positions: np.ndarray,
    oxygen_indices: np.ndarray,
    *,
    cutoff: float,
    cell: tuple[float, float, float] | None = None,
    pbc: tuple[bool, bool, bool] | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    lib = native_library()
    if lib is None:
        return None

    oxygen_positions_array = np.ascontiguousarray(oxygen_positions, dtype=np.float64)
    hydrogen_positions_array = np.ascontiguousarray(hydrogen_positions, dtype=np.float64)
    oxygen_indices_array = np.ascontiguousarray(oxygen_indices, dtype=np.int64)
    if oxygen_positions_array.ndim != 2 or oxygen_positions_array.shape[1] != 3:
        raise ValueError("oxygen_positions must have shape (n, 3).")
    if hydrogen_positions_array.ndim != 2 or hydrogen_positions_array.shape[1] != 3:
        raise ValueError("hydrogen_positions must have shape (n, 3).")
    if oxygen_indices_array.ndim != 1 or oxygen_indices_array.size != oxygen_positions_array.shape[0]:
        raise ValueError("oxygen_indices must be one-dimensional and match oxygen_positions.")

    label_counts = np.zeros(5, dtype=np.int64)
    grouped_indices = np.zeros((5, oxygen_indices_array.size), dtype=np.int64)
    cell_pointer = None
    pbc_pointer = None
    if pbc is not None and any(pbc):
        if cell is None:
            raise ValueError("cell is required when pbc is enabled.")
        cell_array = np.ascontiguousarray(cell, dtype=np.float64)
        pbc_array = np.ascontiguousarray([1 if flag else 0 for flag in pbc], dtype=np.uint8)
        cell_pointer = cell_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        pbc_pointer = pbc_array.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))

    status = lib.waterint_classify_oxygen_by_h_count_compact(
        oxygen_positions_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_size_t(oxygen_positions_array.shape[0]),
        hydrogen_positions_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_size_t(hydrogen_positions_array.shape[0]),
        oxygen_indices_array.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        ctypes.c_double(float(cutoff)),
        cell_pointer,
        pbc_pointer,
        label_counts.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        grouped_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
    )
    if status != 0:
        raise RuntimeError(f"C++ compact oxygen species classification failed with status {status}.")
    return label_counts, grouped_indices


def read_xyz_file(
    path: str | Path,
    *,
    max_frames: int | None = None,
) -> tuple[np.ndarray, list[str]] | None:
    lib = native_library()
    if lib is None:
        return None

    limit = 0 if max_frames in {None, 0} else int(max_frames)
    if limit < 0:
        raise ValueError("max_frames must be positive, 0, or None.")

    data = _XYZData()
    status = lib.waterint_read_xyz_file(
        os.fsencode(Path(path)),
        ctypes.c_size_t(limit),
        ctypes.byref(data),
    )
    try:
        if status != 0:
            message = ""
            if data.error:
                message = ctypes.string_at(data.error).decode("utf-8", errors="replace")
            raise RuntimeError(f"C++ XYZ reader failed with status {status}: {message}")
        if not data.positions or not data.symbols:
            raise RuntimeError("C++ XYZ reader returned empty buffers.")

        positions_view = np.ctypeslib.as_array(
            data.positions,
            shape=(int(data.n_frames), int(data.n_atoms), 3),
        )
        positions = positions_view.copy()
        raw_symbols = ctypes.string_at(data.symbols, int(data.symbols_size)).decode(
            "utf-8",
            errors="replace",
        )
        symbols = raw_symbols.splitlines()
        if len(symbols) != int(data.n_atoms):
            raise RuntimeError(
                f"C++ XYZ reader returned {len(symbols)} symbols for {int(data.n_atoms)} atoms."
            )
        return positions, symbols
    finally:
        lib.waterint_free_xyz_data(ctypes.byref(data))


def read_lammpstrj_file(
    path: str | Path,
    *,
    max_frames: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    lib = native_library()
    if lib is None:
        return None

    limit = 0 if max_frames in {None, 0} else int(max_frames)
    if limit < 0:
        raise ValueError("max_frames must be positive, 0, or None.")

    data = _LammpstrjData()
    status = lib.waterint_read_lammpstrj_file(
        os.fsencode(Path(path)),
        ctypes.c_size_t(limit),
        ctypes.byref(data),
    )
    try:
        if status != 0:
            message = ""
            if data.error:
                message = ctypes.string_at(data.error).decode("utf-8", errors="replace")
            raise RuntimeError(f"C++ LAMMPS dump reader failed with status {status}: {message}")
        if not data.positions or not data.types or not data.cells or not data.steps:
            raise RuntimeError("C++ LAMMPS dump reader returned empty buffers.")

        n_frames = int(data.n_frames)
        n_atoms = int(data.n_atoms)
        positions = np.ctypeslib.as_array(data.positions, shape=(n_frames, n_atoms, 3)).copy()
        types = np.ctypeslib.as_array(data.types, shape=(n_frames, n_atoms)).copy()
        cells = np.ctypeslib.as_array(data.cells, shape=(n_frames, 3)).copy()
        steps = np.ctypeslib.as_array(data.steps, shape=(n_frames,)).copy()
        return positions, types, cells, steps
    finally:
        lib.waterint_free_lammpstrj_data(ctypes.byref(data))


def native_library() -> ctypes.CDLL | None:
    global _LIB_CACHE, _LOAD_ATTEMPTED, _LAST_ERROR
    if _LIB_CACHE is not None:
        return _LIB_CACHE
    if _LOAD_ATTEMPTED:
        return None
    _LOAD_ATTEMPTED = True

    library_path = _library_path()
    if not library_path.exists():
        try:
            build_native_library(library_path)
        except Exception as exc:
            _LAST_ERROR = str(exc)
            return None

    try:
        lib = ctypes.CDLL(str(library_path))
        lib.waterint_density_histogram_edges.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_double),
        ]
        lib.waterint_density_histogram_edges.restype = ctypes.c_int
        lib.waterint_count_hydrogen_neighbors.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
            ctypes.c_double,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_int64),
        ]
        lib.waterint_count_hydrogen_neighbors.restype = ctypes.c_int
        lib.waterint_classify_oxygen_by_h_count_compact.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_double,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_int64),
        ]
        lib.waterint_classify_oxygen_by_h_count_compact.restype = ctypes.c_int
        lib.waterint_read_xyz_file.argtypes = [
            ctypes.c_char_p,
            ctypes.c_size_t,
            ctypes.POINTER(_XYZData),
        ]
        lib.waterint_read_xyz_file.restype = ctypes.c_int
        lib.waterint_free_xyz_data.argtypes = [ctypes.POINTER(_XYZData)]
        lib.waterint_free_xyz_data.restype = None
        lib.waterint_read_lammpstrj_file.argtypes = [
            ctypes.c_char_p,
            ctypes.c_size_t,
            ctypes.POINTER(_LammpstrjData),
        ]
        lib.waterint_read_lammpstrj_file.restype = ctypes.c_int
        lib.waterint_free_lammpstrj_data.argtypes = [ctypes.POINTER(_LammpstrjData)]
        lib.waterint_free_lammpstrj_data.restype = None
    except Exception as exc:
        _LAST_ERROR = str(exc)
        return None

    _LIB_CACHE = lib
    return _LIB_CACHE


def build_native_library(output_path: Path | None = None) -> Path:
    compiler = os.environ.get("CXX") or shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        raise RuntimeError("No C++ compiler was found. Set CXX or install clang++/g++.")

    if output_path is None:
        output_path = _library_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    source_dir = Path(__file__).resolve().parents[1] / "_cpp"
    sources = sorted(source_dir.glob("*.cpp"))
    if not sources:
        raise RuntimeError(f"No C++ source files were found in {source_dir}")

    command = [compiler, "-std=c++17", "-O3"]
    if sysconfig.get_platform().startswith("macosx"):
        command.append("-dynamiclib")
    else:
        command.extend(["-shared", "-fPIC"])
    command.extend(str(source) for source in sources)
    command.extend(["-o", str(output_path)])
    subprocess.run(command, check=True, capture_output=True, text=True)
    return output_path


def native_status() -> dict[str, str | bool]:
    lib = native_library()
    return {
        "available": lib is not None,
        "library": str(_library_path()),
        "last_error": _LAST_ERROR or "",
    }


def _library_path() -> Path:
    suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    return Path(__file__).resolve().parents[1] / "_native" / f"waterint_native{suffix}"
