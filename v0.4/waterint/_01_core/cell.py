from __future__ import annotations

import numpy as np


def orthorhombic_cell_vectors(cell: tuple[float, float, float]) -> np.ndarray:
    cell_array = np.asarray(cell, dtype=float)
    if cell_array.shape != (3,):
        raise ValueError("cell must contain three lengths.")
    if np.any(cell_array <= 0):
        raise ValueError("cell lengths must be positive.")
    return np.diag(cell_array)


def minimum_image(
    vectors: np.ndarray,
    *,
    cell: tuple[float, float, float] | None = None,
    cell_vectors: np.ndarray | None = None,
    pbc: tuple[bool, bool, bool] | None = None,
) -> np.ndarray:
    if pbc is None or not any(pbc):
        return np.asarray(vectors, dtype=float)
    basis = _cell_matrix(cell=cell, cell_vectors=cell_vectors)
    if basis is None:
        raise ValueError("A cell or cell_vectors value is required when pbc is enabled.")
    out = np.asarray(vectors, dtype=float).copy()
    inv_basis = np.linalg.inv(basis)
    fractional = out @ inv_basis
    for axis, enabled in enumerate(pbc):
        if enabled:
            fractional[..., axis] -= np.rint(fractional[..., axis])
    return fractional @ basis


def cell_volume(*, cell: tuple[float, float, float] | None = None, cell_vectors: np.ndarray | None = None) -> float:
    basis = _cell_matrix(cell=cell, cell_vectors=cell_vectors)
    if basis is None:
        raise ValueError("cell or cell_vectors is required.")
    return float(abs(np.linalg.det(basis)))


def cross_section_area(
    *,
    cell: tuple[float, float, float] | None = None,
    cell_vectors: np.ndarray | None = None,
    axis: int,
) -> float:
    basis = _cell_matrix(cell=cell, cell_vectors=cell_vectors)
    if basis is None:
        raise ValueError("cell or cell_vectors is required.")
    vectors = basis
    if axis == 0:
        area = np.linalg.norm(np.cross(vectors[1], vectors[2]))
    elif axis == 1:
        area = np.linalg.norm(np.cross(vectors[0], vectors[2]))
    elif axis == 2:
        area = np.linalg.norm(np.cross(vectors[0], vectors[1]))
    else:
        raise ValueError("axis must be 0, 1, or 2.")
    return float(area)


def _cell_matrix(
    *,
    cell: tuple[float, float, float] | None,
    cell_vectors: np.ndarray | None,
) -> np.ndarray | None:
    if cell_vectors is not None:
        basis = np.asarray(cell_vectors, dtype=float)
        if basis.shape != (3, 3):
            raise ValueError("cell_vectors must have shape (3, 3).")
        return basis
    if cell is None:
        return None
    return orthorhombic_cell_vectors(cell)
