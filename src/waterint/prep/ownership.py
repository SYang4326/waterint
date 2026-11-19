"""Proton -> acceptor ownership: the ONE implementation every analysis uses.

Modes (OwnershipParams.mode):
  nearest            the canonical rule (backends/cpu/ownership.hpp, also used inside the SFG kernels):
                     candidates d <= cutoff; d_min = their minimum; tie set d <= d_min + tie_tolerance;
                     owner = lowest atom index of the tie set. cutoff = inf: every proton takes its nearest acceptor.
  all_within_cutoff  a proton is counted for every acceptor within cutoff (v0.6 orientation / H-bond rule)
Both have a Python reference and a C++ kernel (``ownership.nearest`` / ``ownership.all_within``).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from waterint.backends.registry import register, select
from waterint.io.cell import minimum_image
from waterint.system.profile import OwnershipParams


@dataclass(frozen=True)
class Ownership:
    acceptor_idx: np.ndarray      # global atom indices of acceptors (ascending)
    proton_idx: np.ndarray        # global atom indices of protons (ascending)
    counts: np.ndarray            # [n_acceptors] protons per acceptor
    matrix: np.ndarray            # [n_acceptors, width] local proton indices, -1 padded (v0.6 kernel layout)
    backend: str
    tied: np.ndarray | None = None  # [n_protons] size of the tie set (nearest mode; > 1 = tie resolved by index)

    def proton_lists(self) -> list[np.ndarray]:
        """Local proton indices per acceptor."""
        return [self.matrix[i, : int(self.counts[i])] for i in range(self.counts.size)]

    def owner(self) -> np.ndarray:
        """Global acceptor index of every proton (-1 = free). all_within_cutoff: the lowest-index acceptor."""
        out = np.full(self.proton_idx.size, -1, dtype=np.int64)
        for i in range(self.counts.size - 1, -1, -1):
            row = self.matrix[i, : int(self.counts[i])]
            out[row] = self.acceptor_idx[i]
        return out

    def pairs(self) -> np.ndarray:
        """(n, 2) global [acceptor, proton] pairs, acceptor-major."""
        rows = [np.column_stack([np.full(int(c), a), self.proton_idx[self.matrix[i, : int(c)]]])
                for i, (a, c) in enumerate(zip(self.acceptor_idx, self.counts)) if c]
        return np.concatenate(rows).astype(np.int64) if rows else np.empty((0, 2), dtype=np.int64)


def compute_ownership(
    positions: np.ndarray,
    acceptor_idx: np.ndarray,
    proton_idx: np.ndarray,
    params: OwnershipParams,
    *,
    cell: tuple[float, float, float] | None,
    cell_vectors: np.ndarray | None,
    triclinic: bool,
    backend: str = "auto",
) -> Ownership:
    acceptor_idx = np.asarray(acceptor_idx, dtype=np.int64)
    proton_idx = np.asarray(proton_idx, dtype=np.int64)
    if acceptor_idx.size == 0 or proton_idx.size == 0:
        return Ownership(acceptor_idx, proton_idx, np.zeros(acceptor_idx.size, dtype=np.int64),
                         np.empty((acceptor_idx.size, 0), dtype=np.int64), "python")
    pbc = params.pbc
    vectors = np.asarray(cell_vectors, dtype=float) if (triclinic and cell_vectors is not None) else None
    if pbc is not None and cell is None and vectors is None:
        raise ValueError("Ownership with pbc requires cell information in the frame (or set system.cell).")
    kernel = "ownership.nearest" if params.mode == "nearest" else "ownership.all_within"
    used, fn = select(kernel, backend)
    out = fn(np.ascontiguousarray(positions[acceptor_idx], dtype=float), np.ascontiguousarray(positions[proton_idx], dtype=float),
             cutoff=params.cutoff, cell=cell, cell_vectors=vectors, pbc=pbc, tie_tolerance=params.tie_tolerance)
    counts, matrix = out[0], out[1]
    return Ownership(acceptor_idx, proton_idx, counts, matrix, used, out[2] if len(out) > 2 else None)


# ---------------------------------------------------------------- kernels -----------------------------------
def _matrix_from_owner(owner: np.ndarray, n_acceptors: int) -> tuple[np.ndarray, np.ndarray]:
    counts = np.bincount(owner[owner >= 0], minlength=n_acceptors).astype(np.int64)
    width = max(4, int(counts.max()) if counts.size else 0)
    matrix = np.full((n_acceptors, width), -1, dtype=np.int64)
    fill = np.zeros(n_acceptors, dtype=np.int64)
    for h, o in enumerate(owner):           # protons in ascending order -> rows sorted
        if o >= 0:
            matrix[o, fill[o]] = h
            fill[o] += 1
    return counts, matrix


def _matrix_from_lists(lists: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    counts = np.asarray([len(x) for x in lists], dtype=np.int64)
    width = max(4, int(counts.max()) if counts.size else 0)
    matrix = np.full((len(lists), width), -1, dtype=np.int64)
    for i, row in enumerate(lists):
        matrix[i, : len(row)] = np.sort(np.asarray(row, dtype=np.int64))
    return counts, matrix


def nearest_owner_python(acc, pro, *, cutoff, tie_tolerance, cell, cell_vectors, pbc, chunk: int = 1024):
    """Python reference of the canonical rule (identical to ownership.hpp)."""
    owner = np.full(pro.shape[0], -1, dtype=np.int64)
    tied = np.zeros(pro.shape[0], dtype=np.int32)
    for start in range(0, pro.shape[0], chunk):
        d = pro[start:start + chunk, None, :] - acc[None, :, :]
        if pbc is not None:
            d = minimum_image(d, cell=cell, cell_vectors=cell_vectors, pbc=pbc)
        dist = np.sqrt(np.einsum("ijk,ijk->ij", d, d))
        cand = dist <= cutoff
        if not np.isfinite(cutoff):
            cand[:] = True
        dmin = np.where(cand, dist, np.inf).min(axis=1)
        tie = cand & (dist <= dmin[:, None] + tie_tolerance)
        has = np.isfinite(dmin)
        first = np.argmax(tie, axis=1)                       # lowest acceptor index in the tie set
        owner[start:start + chunk][has] = first[has]
        tied[start:start + chunk] = tie.sum(axis=1)
    return owner, tied


@register("ownership.nearest", "python")
def _nearest_python(acc, pro, *, cutoff, cell, cell_vectors, pbc, tie_tolerance=1.0e-8):
    owner, tied = nearest_owner_python(acc, pro, cutoff=cutoff, tie_tolerance=tie_tolerance, cell=cell,
                                       cell_vectors=cell_vectors, pbc=pbc)
    return (*_matrix_from_owner(owner, acc.shape[0]), tied)


@register("ownership.nearest", "cpu")
def _nearest_cpu(acc, pro, *, cutoff, cell, cell_vectors, pbc, tie_tolerance=1.0e-8):
    from waterint.backends.cpu.native import nearest_owner
    owner, tied = nearest_owner(acc, pro, cutoff=cutoff, tie_tolerance=tie_tolerance, fallback_nearest=not np.isfinite(cutoff),
                                cell=cell, cell_vectors=cell_vectors, pbc=pbc)
    return (*_matrix_from_owner(owner, acc.shape[0]), tied)


@register("ownership.all_within", "python")
def _all_python(acc, pro, *, cutoff, cell, cell_vectors, pbc, tie_tolerance=0.0, chunk: int = 2048):
    c2 = cutoff * cutoff
    lists: list[np.ndarray] = []
    for start in range(0, acc.shape[0], chunk):
        d = pro[:, None, :] - acc[None, start:start + chunk, :]
        if pbc is not None:
            d = minimum_image(d, cell=cell, cell_vectors=cell_vectors, pbc=pbc)
        r2 = np.einsum("ijk,ijk->ij", d, d)
        lists.extend(np.nonzero(r2[:, i] <= c2)[0] for i in range(r2.shape[1]))
    return _matrix_from_lists(lists)


@register("ownership.all_within", "cpu")
def _all_cpu(acc, pro, *, cutoff, cell, cell_vectors, pbc, tie_tolerance=0.0):
    if cell_vectors is not None and pbc is not None:
        return _all_python(acc, pro, cutoff=cutoff, cell=cell, cell_vectors=cell_vectors, pbc=pbc)
    from waterint.backends.cpu.native import hydrogen_neighbor_matrix
    out = hydrogen_neighbor_matrix(acc, pro, cutoff=cutoff, cell=cell if pbc is not None else None, pbc=pbc)
    counts, matrix = out
    if matrix.shape[1] < 4:   # keep a uniform minimum width for downstream kernels
        wide = np.full((matrix.shape[0], 4), -1, dtype=np.int64)
        wide[:, : matrix.shape[1]] = matrix
        matrix = wide
    return counts.astype(np.int64), matrix.astype(np.int64)
