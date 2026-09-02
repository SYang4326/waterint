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
        ("velocities", ctypes.POINTER(ctypes.c_double)),
        ("types", ctypes.POINTER(ctypes.c_int64)),
        ("cells", ctypes.POINTER(ctypes.c_double)),
        ("cell_vectors", ctypes.POINTER(ctypes.c_double)),
        ("cell_origins", ctypes.POINTER(ctypes.c_double)),
        ("triclinic", ctypes.POINTER(ctypes.c_uint8)),
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


def msd_sums(
    positions: np.ndarray,
    *,
    cell_vectors: np.ndarray,
    pbc: tuple[bool, bool, bool],
    max_lag: int,
    origin_stride: int,
    dimensions: int,
    excluded_axis: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    lib = native_library()
    if lib is None:
        return None
    positions_array = np.ascontiguousarray(positions, dtype=np.float64)
    vectors_array = np.ascontiguousarray(cell_vectors, dtype=np.float64)
    pbc_array = np.ascontiguousarray([1 if flag else 0 for flag in pbc], dtype=np.uint8)
    if positions_array.ndim != 3 or positions_array.shape[2] != 3:
        raise ValueError("positions must have shape (n_frames, n_atoms, 3).")
    if vectors_array.shape != (positions_array.shape[0], 3, 3):
        raise ValueError("cell_vectors must contain one 3x3 matrix per frame.")
    sums = np.zeros(max_lag + 1, dtype=np.float64)
    samples = np.zeros(max_lag + 1, dtype=np.int64)
    status = lib.waterint_msd_sums(
        positions_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_size_t(positions_array.shape[0]),
        ctypes.c_size_t(positions_array.shape[1]),
        vectors_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        pbc_array.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        ctypes.c_size_t(max_lag),
        ctypes.c_size_t(origin_stride),
        ctypes.c_int(dimensions),
        ctypes.c_int(excluded_axis),
        sums.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        samples.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
    )
    if status != 0:
        raise RuntimeError(f"C++ MSD kernel failed with status {status}.")
    return sums, samples


def rdf_histogram(
    positions: np.ndarray,
    first_indices: np.ndarray,
    second_indices: np.ndarray,
    *,
    r_max: float,
    n_bins: int,
    cell_vectors: np.ndarray | None,
    pbc: tuple[bool, bool, bool],
    same_selection: bool,
) -> np.ndarray | None:
    lib = native_library()
    if lib is None:
        return None
    positions_array = np.ascontiguousarray(positions, dtype=np.float64)
    first_array = np.ascontiguousarray(first_indices, dtype=np.int64)
    second_array = np.ascontiguousarray(second_indices, dtype=np.int64)
    if positions_array.ndim != 2 or positions_array.shape[1] != 3:
        raise ValueError("positions must have shape (n_atoms, 3).")
    if first_array.ndim != 1 or second_array.ndim != 1:
        raise ValueError("RDF indices must be one-dimensional arrays.")
    vectors_pointer = None
    pbc_pointer = None
    if cell_vectors is not None and any(pbc):
        vectors_array = np.ascontiguousarray(cell_vectors, dtype=np.float64)
        if vectors_array.shape != (3, 3):
            raise ValueError("cell_vectors must have shape (3, 3).")
        pbc_array = np.ascontiguousarray([1 if flag else 0 for flag in pbc], dtype=np.uint8)
        vectors_pointer = vectors_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        pbc_pointer = pbc_array.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    counts = np.zeros(n_bins, dtype=np.float64)
    status = lib.waterint_rdf_histogram(
        positions_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_size_t(positions_array.shape[0]),
        first_array.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        ctypes.c_size_t(first_array.size),
        second_array.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        ctypes.c_size_t(second_array.size),
        ctypes.c_int(1 if same_selection else 0),
        ctypes.c_double(float(r_max)),
        ctypes.c_size_t(n_bins),
        vectors_pointer,
        pbc_pointer,
        counts.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    if status != 0:
        raise RuntimeError(f"C++ RDF kernel failed with status {status}.")
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


def classify_oxygen_by_h_count_nearest(
    oxygen_positions: np.ndarray,
    hydrogen_positions: np.ndarray,
    oxygen_indices: np.ndarray,
    *,
    cutoff: float,
    cell: tuple[float, float, float] | None = None,
    cell_vectors: np.ndarray | None = None,
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
    cell_vectors_pointer = None
    pbc_pointer = None
    if pbc is not None and any(pbc):
        pbc_array = np.ascontiguousarray([1 if flag else 0 for flag in pbc], dtype=np.uint8)
        pbc_pointer = pbc_array.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        if cell_vectors is not None:
            cell_vectors_array = np.ascontiguousarray(cell_vectors, dtype=np.float64)
            if cell_vectors_array.shape != (3, 3):
                raise ValueError("cell_vectors must have shape (3, 3).")
            cell_vectors_pointer = cell_vectors_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        elif cell is not None:
            cell_array = np.ascontiguousarray(cell, dtype=np.float64)
            cell_pointer = cell_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        else:
            raise ValueError("cell or cell_vectors is required when pbc is enabled.")

    status = lib.waterint_classify_oxygen_by_h_count_nearest(
        oxygen_positions_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_size_t(oxygen_positions_array.shape[0]),
        hydrogen_positions_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_size_t(hydrogen_positions_array.shape[0]),
        oxygen_indices_array.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        ctypes.c_double(float(cutoff)),
        cell_pointer,
        cell_vectors_pointer,
        pbc_pointer,
        label_counts.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        grouped_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
    )
    if status != 0:
        raise RuntimeError(f"C++ nearest oxygen species classification failed with status {status}.")
    return label_counts, grouped_indices


def hydrogen_neighbor_matrix(
    oxygen_positions: np.ndarray,
    hydrogen_positions: np.ndarray,
    *,
    cutoff: float,
    cell: tuple[float, float, float] | None = None,
    pbc: tuple[bool, bool, bool] | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Call the shared C++ cell list and return O-H counts plus local H indices.

    The matrix starts with four columns and is retried with a larger capacity
    only if an unusual oxygen has more than four hydrogen neighbors.
    """

    lib = native_library()
    if lib is None:
        return None

    oxygen_array = np.ascontiguousarray(oxygen_positions, dtype=np.float64)
    hydrogen_array = np.ascontiguousarray(hydrogen_positions, dtype=np.float64)
    if oxygen_array.ndim != 2 or oxygen_array.shape[1] != 3:
        raise ValueError("oxygen_positions must have shape (n, 3).")
    if hydrogen_array.ndim != 2 or hydrogen_array.shape[1] != 3:
        raise ValueError("hydrogen_positions must have shape (n, 3).")
    if oxygen_array.shape[0] == 0:
        return np.empty(0, dtype=np.int64), np.empty((0, 0), dtype=np.int64)
    if hydrogen_array.shape[0] == 0:
        return np.zeros(oxygen_array.shape[0], dtype=np.int64), np.empty((oxygen_array.shape[0], 0), dtype=np.int64)

    cell_pointer = None
    pbc_pointer = None
    if pbc is not None and any(pbc):
        if cell is None:
            raise ValueError("cell is required when pbc is enabled.")
        cell_array = np.ascontiguousarray(cell, dtype=np.float64)
        pbc_array = np.ascontiguousarray([1 if flag else 0 for flag in pbc], dtype=np.uint8)
        cell_pointer = cell_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        pbc_pointer = pbc_array.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))

    capacity = 4
    while True:
        counts = np.zeros(oxygen_array.shape[0], dtype=np.int64)
        neighbors = np.full((oxygen_array.shape[0], capacity), -1, dtype=np.int64)
        required = ctypes.c_size_t(0)
        status = lib.waterint_hydrogen_neighbor_matrix(
            oxygen_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_size_t(oxygen_array.shape[0]),
            hydrogen_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_size_t(hydrogen_array.shape[0]),
            ctypes.c_double(float(cutoff)),
            cell_pointer,
            pbc_pointer,
            ctypes.c_size_t(capacity),
            counts.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            neighbors.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            ctypes.byref(required),
        )
        if status == 0:
            return counts, neighbors
        if status == 3 and required.value > capacity:
            # C++ reports the required row width instead of truncating neighbors.
            capacity = int(required.value)
            continue
        raise RuntimeError(f"C++ O-H neighbor-list kernel failed with status {status}.")


def proton_hbond_accumulate(donor_positions: np.ndarray, acceptor_positions: np.ndarray, hydrogen_positions: np.ndarray, hydrogen_counts: np.ndarray, hydrogen_matrix: np.ndarray, *, cell: np.ndarray, pbc: tuple[bool, bool, bool], oo_range: tuple[float, float], angle_min: float, delta_edges: np.ndarray, oo_edges: np.ndarray, hist: np.ndarray) -> tuple[np.ndarray, int, int] | None:
    """Accumulate H-bond-qualified pairs in the native proton-sharing kernel."""
    lib = native_library()
    if lib is None:
        return None
    d=np.ascontiguousarray(donor_positions,dtype=np.float64); a=np.ascontiguousarray(acceptor_positions,dtype=np.float64); h=np.ascontiguousarray(hydrogen_positions,dtype=np.float64)
    counts=np.ascontiguousarray(hydrogen_counts,dtype=np.int64); matrix=np.ascontiguousarray(hydrogen_matrix,dtype=np.int64); de=np.ascontiguousarray(delta_edges,dtype=np.float64); oe=np.ascontiguousarray(oo_edges,dtype=np.float64); output=np.ascontiguousarray(hist,dtype=np.float64).copy(); lengths=np.ascontiguousarray(cell,dtype=np.float64); flags=np.ascontiguousarray([int(value) for value in pbc],dtype=np.uint8); pairs=np.array(0,dtype=np.int64); acceptors=np.array(0,dtype=np.int64)
    status=lib.waterint_proton_hbond_accumulate(d.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),ctypes.c_size_t(len(d)),a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),ctypes.c_size_t(len(a)),h.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),ctypes.c_size_t(len(h)),counts.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),matrix.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),ctypes.c_size_t(matrix.shape[1]),lengths.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),flags.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),ctypes.c_double(oo_range[0]),ctypes.c_double(oo_range[1]),ctypes.c_double(angle_min),de.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),ctypes.c_size_t(len(de)-1),oe.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),ctypes.c_size_t(len(oe)-1),output.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),pairs.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),acceptors.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)))
    if status != 0:
        raise RuntimeError(f"C++ proton H-bond kernel failed with status {status}.")
    return output, int(pairs), int(acceptors)


def nearest_oh_assignment(oxygen_positions: np.ndarray, hydrogen_positions: np.ndarray, *, cutoff: float, cell: np.ndarray, pbc: tuple[bool, bool, bool]) -> tuple[np.ndarray, np.ndarray] | None:
    """Perform unique nearest-O H assignment in C++ for an orthorhombic cell."""
    lib = native_library()
    if lib is None:
        return None
    oxygen=np.ascontiguousarray(oxygen_positions,dtype=np.float64); hydrogen=np.ascontiguousarray(hydrogen_positions,dtype=np.float64); capacity=max(4,len(hydrogen)); counts=np.zeros(len(oxygen),dtype=np.int64); matrix=np.full((len(oxygen),capacity),-1,dtype=np.int64); lengths=np.ascontiguousarray(cell,dtype=np.float64); flags=np.ascontiguousarray([int(value) for value in pbc],dtype=np.uint8)
    status=lib.waterint_nearest_oh_assignment(oxygen.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),ctypes.c_size_t(len(oxygen)),hydrogen.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),ctypes.c_size_t(len(hydrogen)),ctypes.c_double(cutoff),lengths.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),flags.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),ctypes.c_size_t(capacity),counts.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),matrix.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)))
    if status != 0:
        raise RuntimeError(f"C++ nearest O-H assignment failed with status {status}.")
    return counts, matrix


def hbond_geometry_counts(
    oxygen_positions: np.ndarray,
    hydrogen_positions: np.ndarray,
    *,
    hydrogen_counts: np.ndarray,
    hydrogen_matrix: np.ndarray,
    oo_cutoff: float,
    dha_angle_min: float,
    h_acceptor_cutoff: float | None,
    cell: tuple[float, float, float],
    pbc: tuple[bool, bool, bool],
    max_acceptors_per_hydrogen: bool,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Call hbond.cpp and return donated/accepted bond counts per oxygen.

    ``hydrogen_counts`` and ``hydrogen_matrix`` come from the shared O-H
    neighbor kernel. The wrapper converts NumPy arrays to contiguous buffers,
    invokes C++ through ctypes, and converts no scientific labels itself.
    """

    lib = native_library()
    if lib is None:
        return None

    oxygen_array = np.ascontiguousarray(oxygen_positions, dtype=np.float64)
    hydrogen_array = np.ascontiguousarray(hydrogen_positions, dtype=np.float64)
    if oxygen_array.ndim != 2 or oxygen_array.shape[1] != 3:
        raise ValueError("oxygen_positions must have shape (n, 3).")
    if hydrogen_array.ndim != 2 or hydrogen_array.shape[1] != 3:
        raise ValueError("hydrogen_positions must have shape (n, 3).")
    counts_array = np.ascontiguousarray(hydrogen_counts, dtype=np.int64)
    matrix_array = np.ascontiguousarray(hydrogen_matrix, dtype=np.int64)
    if counts_array.ndim != 1 or counts_array.size != oxygen_array.shape[0]:
        raise ValueError("hydrogen_counts must have one value per oxygen.")
    if matrix_array.ndim != 2 or matrix_array.shape[0] != oxygen_array.shape[0] or matrix_array.shape[1] == 0:
        raise ValueError("hydrogen_matrix must have a non-empty row for each oxygen.")
    if oo_cutoff <= 0.0:
        raise ValueError("hbond.oo_cutoff must be positive.")
    if not 0.0 <= dha_angle_min <= 180.0:
        raise ValueError("hbond.dha_angle_min must be between 0 and 180.")

    cell_array = np.ascontiguousarray(cell, dtype=np.float64)
    pbc_array = np.ascontiguousarray([1 if flag else 0 for flag in pbc], dtype=np.uint8)
    if cell_array.shape != (3,):
        raise ValueError("cell must contain three orthorhombic cell lengths.")
    if pbc_array.shape != (3,):
        raise ValueError("pbc must contain three boolean flags.")
    donor_counts = np.zeros(oxygen_array.shape[0], dtype=np.int64)
    acceptor_counts = np.zeros(oxygen_array.shape[0], dtype=np.int64)
    # NaN is the C ABI sentinel meaning that the optional H-A cutoff is disabled.
    h_acceptor_limit = float("nan") if h_acceptor_cutoff is None else float(h_acceptor_cutoff)
    status = lib.waterint_hbond_geometry_counts(
        oxygen_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_size_t(oxygen_array.shape[0]),
        hydrogen_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_size_t(hydrogen_array.shape[0]),
        counts_array.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        matrix_array.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        ctypes.c_size_t(matrix_array.shape[1]),
        ctypes.c_double(float(oo_cutoff)),
        ctypes.c_double(float(dha_angle_min)),
        ctypes.c_double(h_acceptor_limit),
        cell_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        pbc_array.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        ctypes.c_int(1 if max_acceptors_per_hydrogen else 0),
        donor_counts.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        acceptor_counts.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
    )
    if status != 0:
        raise RuntimeError(f"C++ H-bond kernel failed with status {status}.")
    return donor_counts, acceptor_counts


def sfg_ssvvcf(
    positions: np.ndarray,
    velocities: np.ndarray | None,
    oxygen_indices: np.ndarray,
    hydrogen_indices: np.ndarray,
    zrefs: np.ndarray,
    *,
    dt_ps: float,
    max_lag: int,
    oh_cutoff: float,
    oh_assignment: str,
    cell: tuple[float, float, float],
    pbc: tuple[bool, bool, bool],
    mu_mode: str,
    symmetrize: bool,
    flip_sign: bool,
    duplicate_policy: str,
    window: dict | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Call the C++ SFG kernel and return sums, counts, and stage timings."""

    lib = native_library()
    if lib is None:
        return None

    positions_array = np.ascontiguousarray(positions, dtype=np.float64)
    velocities_array = None if velocities is None else np.ascontiguousarray(velocities, dtype=np.float64)
    oxygen_array = np.ascontiguousarray(oxygen_indices, dtype=np.int64)
    hydrogen_array = np.ascontiguousarray(hydrogen_indices, dtype=np.int64)
    zref_array = np.ascontiguousarray(zrefs, dtype=np.float64)
    if positions_array.ndim != 3 or positions_array.shape[2] != 3:
        raise ValueError("positions must have shape (n_frames, n_atoms, 3).")
    if velocities_array is not None and velocities_array.shape != positions_array.shape:
        raise ValueError("velocities must have the same shape as positions.")
    if oxygen_array.ndim != 1 or hydrogen_array.ndim != 1:
        raise ValueError("oxygen_indices and hydrogen_indices must be one-dimensional.")
    if zref_array.shape != (positions_array.shape[0],):
        raise ValueError("zrefs must contain one value per frame.")
    if dt_ps <= 0.0 or max_lag < 1 or oh_cutoff <= 0.0:
        raise ValueError("dt_ps, max_lag, and oh_cutoff must be positive.")
    assignment = str(oh_assignment).lower()
    if assignment not in {"cutoff", "nearest_oxygen"}:
        raise ValueError("sfg.oh_assignment must be cutoff or nearest_oxygen.")

    mode = str(mu_mode).lower()
    if mode not in {"full", "stretch"}:
        raise ValueError("sfg.mu_mode must be full or stretch.")
    duplicate = str(duplicate_policy).lower()
    if duplicate not in {"nearest", "error"}:
        raise ValueError("sfg.duplicate_hydrogen_policy must be nearest or error.")
    cell_array = np.ascontiguousarray(cell, dtype=np.float64)
    pbc_array = np.ascontiguousarray([1 if flag else 0 for flag in pbc], dtype=np.uint8)
    if cell_array.shape != (3,) or pbc_array.shape != (3,):
        raise ValueError("cell and pbc must each contain three values.")

    window_enabled = window is not None
    window_cfg = {} if window is None else window
    window_mode = int(window_cfg.get("mode", 1))
    if window_enabled and window_mode not in {1, 2}:
        raise ValueError("sfg.window.mode must be 1 or 2.")
    sums = np.zeros(max_lag + 1, dtype=np.float64)
    counts = np.zeros(max_lag + 1, dtype=np.int64)
    stage_seconds = np.zeros(4, dtype=np.float64)
    status = lib.waterint_sfg_ssvvcf(
        positions_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        None if velocities_array is None else velocities_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_size_t(positions_array.shape[0]),
        ctypes.c_size_t(positions_array.shape[1]),
        oxygen_array.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        ctypes.c_size_t(oxygen_array.size),
        hydrogen_array.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        ctypes.c_size_t(hydrogen_array.size),
        zref_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_double(float(dt_ps)),
        ctypes.c_size_t(int(max_lag)),
        ctypes.c_double(float(oh_cutoff)),
        ctypes.c_int(1 if assignment == "nearest_oxygen" else 0),
        cell_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        pbc_array.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        ctypes.c_int(1 if mode == "stretch" else 0),
        ctypes.c_int(1 if symmetrize else 0),
        ctypes.c_int(1 if flip_sign else 0),
        ctypes.c_int(1 if duplicate == "error" else 0),
        ctypes.c_int(1 if window_enabled else 0),
        ctypes.c_int(window_mode),
        ctypes.c_double(float(window_cfg.get("z1", 0.0))),
        ctypes.c_double(float(window_cfg.get("z2", 0.0))),
        ctypes.c_double(float(window_cfg.get("ramp", 0.0))),
        ctypes.c_int(1 if bool(window_cfg.get("flip", False)) else 0),
        sums.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        counts.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        stage_seconds.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    if status == 4:
        raise ValueError(
            "A hydrogen is assigned to more than one oxygen. "
            "Use sfg.duplicate_hydrogen_policy: nearest or lower sfg.oh_cutoff."
        )
    if status != 0:
        raise RuntimeError(f"C++ SFG ssVVCF kernel failed with status {status}.")
    return sums, counts, stage_seconds


def sfg_layered_ssvvcf(
    positions: np.ndarray,
    velocities: np.ndarray | None,
    oxygen_indices: np.ndarray,
    hydrogen_indices: np.ndarray,
    zrefs: np.ndarray,
    *,
    dt_ps: float,
    max_lag: int,
    oh_cutoff: float,
    oh_assignment: str,
    cell: tuple[float, float, float],
    pbc: tuple[bool, bool, bool],
    mu_mode: str,
    symmetrize: bool,
    flip_sign: bool,
    duplicate_policy: str,
    channel_specs: dict[str, dict],
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Call the multi-channel C++ SFG kernel.

    Rows in the returned sums/counts arrays retain ``channel_specs`` order.
    Species membership is evaluated natively from the unique H-to-O assignment
    on every frame.
    """

    lib = native_library()
    if lib is None:
        return None

    positions_array = np.ascontiguousarray(positions, dtype=np.float64)
    velocities_array = None if velocities is None else np.ascontiguousarray(velocities, dtype=np.float64)
    oxygen_array = np.ascontiguousarray(oxygen_indices, dtype=np.int64)
    hydrogen_array = np.ascontiguousarray(hydrogen_indices, dtype=np.int64)
    zref_array = np.ascontiguousarray(zrefs, dtype=np.float64)
    if positions_array.ndim != 3 or positions_array.shape[2] != 3:
        raise ValueError("positions must have shape (n_frames, n_atoms, 3).")
    if velocities_array is not None and velocities_array.shape != positions_array.shape:
        raise ValueError("velocities must have the same shape as positions.")
    if oxygen_array.ndim != 1 or hydrogen_array.ndim != 1:
        raise ValueError("oxygen_indices and hydrogen_indices must be one-dimensional.")
    if zref_array.shape != (positions_array.shape[0],):
        raise ValueError("zrefs must contain one value per frame.")
    if dt_ps <= 0.0 or max_lag < 1 or oh_cutoff <= 0.0:
        raise ValueError("dt_ps, max_lag, and oh_cutoff must be positive.")
    if not channel_specs:
        raise ValueError("channel_specs must contain at least one layered SFG channel.")

    assignment = str(oh_assignment).lower()
    if assignment not in {"cutoff", "nearest_oxygen"}:
        raise ValueError("sfg.oh_assignment must be cutoff or nearest_oxygen.")
    mode = str(mu_mode).lower()
    if mode not in {"full", "stretch"}:
        raise ValueError("sfg.mu_mode must be full or stretch.")
    duplicate = str(duplicate_policy).lower()
    if duplicate not in {"nearest", "error"}:
        raise ValueError("sfg.duplicate_hydrogen_policy must be nearest or error.")

    species_code = {"all": 0, "O2-": 1, "OH-": 2, "H2O": 3, "H3O+": 4, "O_other": 5}
    window_enabled: list[int] = []
    window_modes: list[int] = []
    window_z1: list[float] = []
    window_z2: list[float] = []
    window_ramps: list[float] = []
    window_flips: list[int] = []
    species_codes: list[int] = []
    for spec in channel_specs.values():
        window = spec.get("window")
        window_cfg = {} if window is None else window
        window_mode = int(window_cfg.get("mode", 1))
        if window is not None and window_mode not in {1, 2}:
            raise ValueError("SFG layer window mode must be 1 or 2.")
        species = str(spec.get("species", "all"))
        if species not in species_code:
            raise ValueError(f"Unknown SFG oxygen species channel: {species}")
        window_enabled.append(0 if window is None else 1)
        window_modes.append(window_mode)
        window_z1.append(float(window_cfg.get("z1", 0.0)))
        window_z2.append(float(window_cfg.get("z2", 0.0)))
        window_ramps.append(float(window_cfg.get("ramp", 0.0)))
        window_flips.append(1 if bool(window_cfg.get("flip", False)) else 0)
        species_codes.append(species_code[species])

    cell_array = np.ascontiguousarray(cell, dtype=np.float64)
    pbc_array = np.ascontiguousarray([1 if flag else 0 for flag in pbc], dtype=np.uint8)
    enabled_array = np.ascontiguousarray(window_enabled, dtype=np.uint8)
    modes_array = np.ascontiguousarray(window_modes, dtype=np.int32)
    z1_array = np.ascontiguousarray(window_z1, dtype=np.float64)
    z2_array = np.ascontiguousarray(window_z2, dtype=np.float64)
    ramps_array = np.ascontiguousarray(window_ramps, dtype=np.float64)
    flips_array = np.ascontiguousarray(window_flips, dtype=np.uint8)
    species_array = np.ascontiguousarray(species_codes, dtype=np.int32)
    if cell_array.shape != (3,) or pbc_array.shape != (3,):
        raise ValueError("cell and pbc must each contain three values.")

    shape = (len(channel_specs), max_lag + 1)
    sums = np.zeros(shape, dtype=np.float64)
    counts = np.zeros(shape, dtype=np.int64)
    stage_seconds = np.zeros(4, dtype=np.float64)
    status = lib.waterint_sfg_layered_ssvvcf(
        positions_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        None if velocities_array is None else velocities_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_size_t(positions_array.shape[0]),
        ctypes.c_size_t(positions_array.shape[1]),
        oxygen_array.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        ctypes.c_size_t(oxygen_array.size),
        hydrogen_array.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        ctypes.c_size_t(hydrogen_array.size),
        zref_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_double(float(dt_ps)),
        ctypes.c_size_t(int(max_lag)),
        ctypes.c_double(float(oh_cutoff)),
        ctypes.c_int(1 if assignment == "nearest_oxygen" else 0),
        cell_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        pbc_array.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        ctypes.c_int(1 if mode == "stretch" else 0),
        ctypes.c_int(1 if symmetrize else 0),
        ctypes.c_int(1 if flip_sign else 0),
        ctypes.c_int(1 if duplicate == "error" else 0),
        ctypes.c_size_t(len(channel_specs)),
        enabled_array.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        modes_array.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        z1_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        z2_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ramps_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        flips_array.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        species_array.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        sums.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        counts.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        stage_seconds.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    if status == 4:
        raise ValueError(
            "A hydrogen is assigned to more than one oxygen. "
            "Use sfg.duplicate_hydrogen_policy: nearest or lower sfg.oh_cutoff."
        )
    if status != 0:
        raise RuntimeError(f"C++ layered SFG ssVVCF kernel failed with status {status}.")
    return sums, counts, stage_seconds


def accumulate_oh_orientation_from_neighbors(
    oxygen_positions: np.ndarray,
    hydrogen_positions: np.ndarray,
    neighbor_counts: np.ndarray,
    neighbor_matrix: np.ndarray,
    *,
    vector_mode: str,
    axis: int,
    axis_sign: float,
    reference: float,
    angle_axis_sign: float,
    z_edges: np.ndarray,
    angle_edges: np.ndarray,
    histograms: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    lib = native_library()
    if lib is None:
        return None

    oxygen_array = np.ascontiguousarray(oxygen_positions, dtype=np.float64)
    hydrogen_array = np.ascontiguousarray(hydrogen_positions, dtype=np.float64)
    counts_array = np.ascontiguousarray(neighbor_counts, dtype=np.int64)
    neighbors_array = np.ascontiguousarray(neighbor_matrix, dtype=np.int64)
    z_edges_array = np.ascontiguousarray(z_edges, dtype=np.float64)
    angle_edges_array = np.ascontiguousarray(angle_edges, dtype=np.float64)
    histogram_array = np.asarray(histograms)
    if oxygen_array.ndim != 2 or oxygen_array.shape[1] != 3:
        raise ValueError("oxygen_positions must have shape (n, 3).")
    if hydrogen_array.ndim != 2 or hydrogen_array.shape[1] != 3:
        raise ValueError("hydrogen_positions must have shape (n, 3).")
    if counts_array.ndim != 1 or counts_array.size != oxygen_array.shape[0]:
        raise ValueError("neighbor_counts must match oxygen_positions.")
    if neighbors_array.ndim != 2 or neighbors_array.shape[0] != oxygen_array.shape[0]:
        raise ValueError("neighbor_matrix must have one row per oxygen.")
    if z_edges_array.ndim != 1 or z_edges_array.size < 2:
        raise ValueError("z_edges must be one-dimensional with at least two values.")
    if angle_edges_array.ndim != 1 or angle_edges_array.size < 2:
        raise ValueError("angle_edges must be one-dimensional with at least two values.")
    expected_shape = (3, z_edges_array.size - 1, angle_edges_array.size - 1)
    if histogram_array.dtype != np.float64 or histogram_array.shape != expected_shape or not histogram_array.flags.c_contiguous:
        raise ValueError(f"histograms must be a C-contiguous float64 array with shape {expected_shape}.")

    mode = str(vector_mode).lower()
    if mode == "oh_bond":
        mode_code = 0
    elif mode in {"oh_bisector", "dipole"}:
        mode_code = 1
    else:
        raise ValueError("vector_mode must be oh_bond, oh_bisector, or dipole.")

    bond_counts = np.zeros(3, dtype=np.int64)
    sample_counts = np.zeros(3, dtype=np.int64)
    status = lib.waterint_accumulate_oh_orientation(
        oxygen_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_size_t(oxygen_array.shape[0]),
        hydrogen_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_size_t(hydrogen_array.shape[0]),
        counts_array.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        neighbors_array.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        ctypes.c_size_t(neighbors_array.shape[1]),
        ctypes.c_int(mode_code),
        ctypes.c_int(int(axis)),
        ctypes.c_double(float(axis_sign)),
        ctypes.c_double(float(reference)),
        ctypes.c_double(float(angle_axis_sign)),
        z_edges_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_size_t(z_edges_array.size - 1),
        angle_edges_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.c_size_t(angle_edges_array.size - 1),
        histogram_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        bond_counts.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        sample_counts.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
    )
    if status != 0:
        raise RuntimeError(f"C++ O-H orientation kernel failed with status {status}.")
    return bond_counts, sample_counts


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
    start_timestep: int | None = None,
    stride: int = 1,
    max_frames: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None] | None:
    lib = native_library()
    if lib is None:
        return None

    limit = 0 if max_frames in {None, 0} else int(max_frames)
    if limit < 0:
        raise ValueError("max_frames must be positive, 0, or None.")
    if stride <= 0:
        raise ValueError("stride must be > 0.")

    data = _LammpstrjData()
    status = lib.waterint_read_lammpstrj_file(
        os.fsencode(Path(path)),
        ctypes.c_size_t(limit),
        ctypes.c_int(0 if start_timestep is None else 1),
        ctypes.c_int64(0 if start_timestep is None else int(start_timestep)),
        ctypes.c_size_t(stride),
        ctypes.byref(data),
    )
    try:
        if status != 0:
            message = ""
            if data.error:
                message = ctypes.string_at(data.error).decode("utf-8", errors="replace")
            raise RuntimeError(f"C++ LAMMPS dump reader failed with status {status}: {message}")
        if (
            not data.positions or not data.types or not data.cells or not data.cell_vectors or
            not data.cell_origins or not data.triclinic or not data.steps
        ):
            raise RuntimeError("C++ LAMMPS dump reader returned empty buffers.")

        n_frames = int(data.n_frames)
        n_atoms = int(data.n_atoms)
        positions = np.ctypeslib.as_array(data.positions, shape=(n_frames, n_atoms, 3)).copy()
        velocities = (
            np.ctypeslib.as_array(data.velocities, shape=(n_frames, n_atoms, 3)).copy()
            if data.velocities
            else None
        )
        types = np.ctypeslib.as_array(data.types, shape=(n_frames, n_atoms)).copy()
        cells = np.ctypeslib.as_array(data.cells, shape=(n_frames, 3)).copy()
        cell_vectors = np.ctypeslib.as_array(data.cell_vectors, shape=(n_frames, 3, 3)).copy()
        cell_origins = np.ctypeslib.as_array(data.cell_origins, shape=(n_frames, 3)).copy()
        triclinic = np.ctypeslib.as_array(data.triclinic, shape=(n_frames,)).astype(bool, copy=True)
        steps = np.ctypeslib.as_array(data.steps, shape=(n_frames,)).copy()
        return positions, types, cells, cell_vectors, cell_origins, triclinic, steps, velocities
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
    if _native_library_needs_build(library_path):
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
        lib.waterint_msd_sums.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_int64),
        ]
        lib.waterint_msd_sums.restype = ctypes.c_int
        lib.waterint_rdf_histogram.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_double,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_double),
        ]
        lib.waterint_rdf_histogram.restype = ctypes.c_int
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
        lib.waterint_classify_oxygen_by_h_count_nearest.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_double,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_int64),
        ]
        lib.waterint_classify_oxygen_by_h_count_nearest.restype = ctypes.c_int
        lib.waterint_hydrogen_neighbor_matrix.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
            ctypes.c_double,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_size_t),
        ]
        lib.waterint_hydrogen_neighbor_matrix.restype = ctypes.c_int
        lib.waterint_hbond_geometry_counts.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_size_t,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_int64),
        ]
        lib.waterint_hbond_geometry_counts.restype = ctypes.c_int
        lib.waterint_proton_hbond_accumulate.argtypes = [ctypes.POINTER(ctypes.c_double),ctypes.c_size_t,ctypes.POINTER(ctypes.c_double),ctypes.c_size_t,ctypes.POINTER(ctypes.c_double),ctypes.c_size_t,ctypes.POINTER(ctypes.c_int64),ctypes.POINTER(ctypes.c_int64),ctypes.c_size_t,ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_uint8),ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.POINTER(ctypes.c_double),ctypes.c_size_t,ctypes.POINTER(ctypes.c_double),ctypes.c_size_t,ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_int64),ctypes.POINTER(ctypes.c_int64)]
        lib.waterint_proton_hbond_accumulate.restype = ctypes.c_int
        lib.waterint_nearest_oh_assignment.argtypes = [ctypes.POINTER(ctypes.c_double),ctypes.c_size_t,ctypes.POINTER(ctypes.c_double),ctypes.c_size_t,ctypes.c_double,ctypes.POINTER(ctypes.c_double),ctypes.POINTER(ctypes.c_uint8),ctypes.c_size_t,ctypes.POINTER(ctypes.c_int64),ctypes.POINTER(ctypes.c_int64)]
        lib.waterint_nearest_oh_assignment.restype = ctypes.c_int
        lib.waterint_sfg_ssvvcf.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_double,
            ctypes.c_size_t,
            ctypes.c_double,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_double),
        ]
        lib.waterint_sfg_ssvvcf.restype = ctypes.c_int
        lib.waterint_sfg_layered_ssvvcf.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_double,
            ctypes.c_size_t,
            ctypes.c_double,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_double),
        ]
        lib.waterint_sfg_layered_ssvvcf.restype = ctypes.c_int
        lib.waterint_accumulate_oh_orientation.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_int64),
        ]
        lib.waterint_accumulate_oh_orientation.restype = ctypes.c_int
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
            ctypes.c_int,
            ctypes.c_int64,
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
    package_dir = Path(__file__).resolve().parents[1]
    sources = _native_sources(package_dir)
    if not sources:
        raise RuntimeError(f"No C++ source files were found in {package_dir}")

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


def _native_sources(package_dir: Path) -> list[Path]:
    return sorted(
        source
        for source in package_dir.rglob("*.cpp")
        if "_native" not in source.relative_to(package_dir).parts
    )


def _native_library_needs_build(library_path: Path) -> bool:
    if not library_path.exists():
        return True
    package_dir = Path(__file__).resolve().parents[1]
    build_inputs = _native_sources(package_dir) + sorted(package_dir.rglob("*.hpp"))
    library_mtime = library_path.stat().st_mtime
    return any(path.stat().st_mtime > library_mtime for path in build_inputs)
