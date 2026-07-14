from __future__ import annotations

from itertools import product

import numpy as np

from waterint._02_computation._native import classify_oxygen_by_h_count_compact
from waterint._02_computation._native import count_hydrogen_neighbors


OXYGEN_SPECIES_BY_H_COUNT = {
    0: "O2-",
    1: "OH-",
    2: "H2O",
    3: "H3O+",
}


class PythonCutoffNeighborSearch:
    """Reusable orthorhombic cell list implemented with Python and NumPy.

    Candidate atoms are binned once. A query inspects only its own bin and
    adjacent bins, then applies an exact minimum-image distance check. Returned
    indices are sorted to keep backend tie behavior deterministic.
    """

    def __init__(
        self,
        query_positions: np.ndarray,
        candidate_positions: np.ndarray,
        *,
        cutoff: float,
        cell: tuple[float, float, float] | None = None,
        pbc: tuple[bool, bool, bool] | None = None,
    ) -> None:
        self.query_positions = np.ascontiguousarray(query_positions, dtype=float)
        self.candidate_positions = np.ascontiguousarray(candidate_positions, dtype=float)
        if self.query_positions.ndim != 2 or self.query_positions.shape[1] != 3:
            raise ValueError("query_positions must have shape (n, 3).")
        if self.candidate_positions.ndim != 2 or self.candidate_positions.shape[1] != 3:
            raise ValueError("candidate_positions must have shape (n, 3).")
        if cutoff <= 0:
            raise ValueError("cutoff must be positive.")

        self.cutoff2 = float(cutoff) ** 2
        self.pbc = (False, False, False) if pbc is None else tuple(bool(flag) for flag in pbc)
        if len(self.pbc) != 3:
            raise ValueError("pbc must contain three boolean flags.")
        if any(self.pbc):
            if cell is None or len(cell) != 3 or any(float(length) <= 0 for length in cell):
                raise ValueError("A positive three-length cell is required when pbc is enabled.")
        self.cell = np.zeros(3, dtype=float) if cell is None else np.asarray(cell, dtype=float)

        self.origin = np.zeros(3, dtype=float)
        self.length = np.zeros(3, dtype=float)
        all_positions = np.vstack((self.query_positions, self.candidate_positions))
        for axis in range(3):
            if self.pbc[axis]:
                self.length[axis] = self.cell[axis]
                continue
            if all_positions.shape[0] == 0:
                lower = 0.0
                upper = 0.0
            else:
                lower = float(np.min(all_positions[:, axis]))
                upper = float(np.max(all_positions[:, axis]))
            self.origin[axis] = lower - cutoff
            self.length[axis] = max(float(cutoff), upper - lower + 2.0 * cutoff)

        self.bin_counts = np.maximum(1, np.floor(self.length / cutoff).astype(int))
        self.bin_width = self.length / self.bin_counts
        self.query_bins = self._position_bins(self.query_positions)
        candidate_bins = self._position_bins(self.candidate_positions)
        self.candidates_by_bin: dict[tuple[int, int, int], list[int]] = {}
        for candidate_index, raw_bin in enumerate(candidate_bins):
            bin_key = tuple(int(value) for value in raw_bin)
            self.candidates_by_bin.setdefault(bin_key, []).append(candidate_index)
        self.candidate_cache: dict[tuple[int, int, int], np.ndarray] = {}

    def collect_indices(self, query_index: int) -> np.ndarray:
        """Return sorted candidate indices within cutoff of one query atom."""

        center = tuple(int(value) for value in self.query_bins[query_index])
        candidates = self.candidate_cache.get(center)
        if candidates is None:
            candidate_lists = [
                self.candidates_by_bin[bin_key]
                for bin_key in self._neighbor_bins(center)
                if bin_key in self.candidates_by_bin
            ]
            candidates = (
                np.sort(np.concatenate([np.asarray(items, dtype=int) for items in candidate_lists]))
                if candidate_lists
                else np.empty(0, dtype=int)
            )
            self.candidate_cache[center] = candidates
        if candidates.size == 0:
            return candidates

        vectors = self.candidate_positions[candidates] - self.query_positions[query_index]
        vectors = self.minimum_image(vectors)
        distances2 = np.einsum("ij,ij->i", vectors, vectors)
        return candidates[distances2 <= self.cutoff2]

    def minimum_image(self, vectors: np.ndarray) -> np.ndarray:
        """Apply this search's minimum-image convention to displacement vectors."""

        out = np.asarray(vectors, dtype=float).copy()
        for axis, enabled in enumerate(self.pbc):
            if enabled:
                out[..., axis] -= np.rint(out[..., axis] / self.cell[axis]) * self.cell[axis]
        return out

    def _position_bins(self, positions: np.ndarray) -> np.ndarray:
        """Map Cartesian positions to integer cell-list bins."""

        shifted = np.asarray(positions, dtype=float) - self.origin
        for axis, enabled in enumerate(self.pbc):
            if enabled:
                shifted[:, axis] -= np.floor(shifted[:, axis] / self.length[axis]) * self.length[axis]
        bins = np.floor(shifted / self.bin_width).astype(int)
        return np.clip(bins, 0, self.bin_counts - 1)

    def _neighbor_bins(self, center: tuple[int, int, int]) -> list[tuple[int, int, int]]:
        """Return unique adjacent bins, wrapping only periodic axes."""

        neighbors: set[tuple[int, int, int]] = set()
        for offset in product((-1, 0, 1), repeat=3):
            values: list[int] = []
            valid = True
            for axis in range(3):
                value = center[axis] + offset[axis]
                if self.pbc[axis]:
                    value %= int(self.bin_counts[axis])
                elif value < 0 or value >= self.bin_counts[axis]:
                    valid = False
                    break
                values.append(int(value))
            if valid:
                neighbors.add((values[0], values[1], values[2]))
        return sorted(neighbors)


def classify_oxygen_by_h_count(
    symbols: list[str],
    positions: np.ndarray,
    *,
    oxygen_symbol: str = "O",
    hydrogen_symbol: str = "H",
    oh_cutoff: float = 1.25,
    neighbor_method: str = "auto",
    neighbor_workers: int = 1,
    oxygen_chunk_size: int = 2048,
    oxygen_indices: np.ndarray | None = None,
    hydrogen_indices: np.ndarray | None = None,
    cell: tuple[float, float, float] | None = None,
    pbc: tuple[bool, bool, bool] | None = None,
) -> dict[str, np.ndarray]:
    """Classify oxygen atoms by the number of hydrogens within a cutoff.

    This is a geometry-based first implementation. It is intentionally simple:
    no bonding history, no PBC minimum-image convention, and no special handling
    of shared protons. The default neighbor search uses scipy's cKDTree when it
    is available, with a chunked NumPy fallback for lighter installations.
    """
    if oh_cutoff <= 0:
        raise ValueError("oh_cutoff must be positive.")
    if oxygen_chunk_size <= 0:
        raise ValueError("oxygen_chunk_size must be positive.")

    if oxygen_indices is None or hydrogen_indices is None:
        symbols_array = np.asarray(symbols)
        if oxygen_indices is None:
            oxygen_indices = np.where(symbols_array == oxygen_symbol)[0]
        if hydrogen_indices is None:
            hydrogen_indices = np.where(symbols_array == hydrogen_symbol)[0]
    else:
        oxygen_indices = np.asarray(oxygen_indices, dtype=int)
        hydrogen_indices = np.asarray(hydrogen_indices, dtype=int)

    out: dict[str, list[int]] = {
        "O2-": [],
        "OH-": [],
        "H2O": [],
        "H3O+": [],
        "O_other": [],
    }
    if oxygen_indices.size == 0:
        return {key: np.asarray(value, dtype=int) for key, value in out.items()}

    if hydrogen_indices.size == 0:
        out["O2-"] = oxygen_indices.tolist()
        return {key: np.asarray(value, dtype=int) for key, value in out.items()}

    method = str(neighbor_method).lower()
    if method in {"auto", "cpp"}:
        oxygen_positions = positions[oxygen_indices]
        hydrogen_positions = positions[hydrogen_indices]
        native_classified = _classify_oxygen_by_h_count_compact(
            oxygen_positions=oxygen_positions,
            hydrogen_positions=hydrogen_positions,
            oxygen_indices=oxygen_indices,
            cutoff=oh_cutoff,
            cell=cell,
            pbc=pbc,
        )
        if native_classified is not None:
            return native_classified
        if method == "cpp":
            raise RuntimeError("C++ oxygen species classification backend is not available.")

    oxygen_positions = positions[oxygen_indices]
    hydrogen_positions = positions[hydrogen_indices]
    h_counts = _count_hydrogen_neighbors(
        oxygen_positions=oxygen_positions,
        hydrogen_positions=hydrogen_positions,
        cutoff=oh_cutoff,
        method=neighbor_method,
        workers=neighbor_workers,
        oxygen_chunk_size=oxygen_chunk_size,
        cell=cell,
        pbc=pbc,
    )

    for oxygen_index, h_count in zip(oxygen_indices, h_counts):
        label = OXYGEN_SPECIES_BY_H_COUNT.get(int(h_count), "O_other")
        out[label].append(int(oxygen_index))

    return {key: np.asarray(value, dtype=int) for key, value in out.items()}


def _classify_oxygen_by_h_count_compact(
    *,
    oxygen_positions: np.ndarray,
    hydrogen_positions: np.ndarray,
    oxygen_indices: np.ndarray,
    cutoff: float,
    cell: tuple[float, float, float] | None,
    pbc: tuple[bool, bool, bool] | None,
) -> dict[str, np.ndarray] | None:
    native_result = classify_oxygen_by_h_count_compact(
        oxygen_positions,
        hydrogen_positions,
        oxygen_indices,
        cutoff=cutoff,
        cell=cell,
        pbc=pbc,
    )
    if native_result is None:
        return None
    label_counts, grouped_indices = native_result
    return {
        "O2-": grouped_indices[0, : label_counts[0]].copy(),
        "OH-": grouped_indices[1, : label_counts[1]].copy(),
        "H2O": grouped_indices[2, : label_counts[2]].copy(),
        "H3O+": grouped_indices[3, : label_counts[3]].copy(),
        "O_other": grouped_indices[4, : label_counts[4]].copy(),
    }


def oxygen_hydrogen_pairs_by_species(
    symbols: list[str],
    positions: np.ndarray,
    *,
    oxygen_symbol: str = "O",
    hydrogen_symbol: str = "H",
    oh_cutoff: float = 1.25,
    neighbor_method: str = "auto",
    neighbor_workers: int = 1,
    oxygen_chunk_size: int = 2048,
    oxygen_indices: np.ndarray | None = None,
    hydrogen_indices: np.ndarray | None = None,
    cell: tuple[float, float, float] | None = None,
    pbc: tuple[bool, bool, bool] | None = None,
) -> dict[str, np.ndarray]:
    """Return O-H pairs grouped by the oxygen species label.

    Each returned array has shape (n_pairs, 2), with columns
    [oxygen_index, hydrogen_index]. Species labels use the same local O-H
    coordination definition as classify_oxygen_by_h_count.
    """
    if oh_cutoff <= 0:
        raise ValueError("oh_cutoff must be positive.")
    if oxygen_chunk_size <= 0:
        raise ValueError("oxygen_chunk_size must be positive.")

    if oxygen_indices is None or hydrogen_indices is None:
        symbols_array = np.asarray(symbols)
        if oxygen_indices is None:
            oxygen_indices = np.where(symbols_array == oxygen_symbol)[0]
        if hydrogen_indices is None:
            hydrogen_indices = np.where(symbols_array == hydrogen_symbol)[0]
    else:
        oxygen_indices = np.asarray(oxygen_indices, dtype=int)
        hydrogen_indices = np.asarray(hydrogen_indices, dtype=int)

    out: dict[str, list[tuple[int, int]]] = {
        "O2-": [],
        "OH-": [],
        "H2O": [],
        "H3O+": [],
        "O_other": [],
    }
    if oxygen_indices.size == 0 or hydrogen_indices.size == 0:
        return {key: np.empty((0, 2), dtype=int) for key in out}

    neighbor_lists = _hydrogen_neighbor_lists(
        oxygen_positions=positions[oxygen_indices],
        hydrogen_positions=positions[hydrogen_indices],
        cutoff=oh_cutoff,
        method=neighbor_method,
        workers=neighbor_workers,
        oxygen_chunk_size=oxygen_chunk_size,
        cell=cell,
        pbc=pbc,
    )
    for oxygen_index, local_hydrogen_indices in zip(oxygen_indices, neighbor_lists):
        label = OXYGEN_SPECIES_BY_H_COUNT.get(len(local_hydrogen_indices), "O_other")
        for local_hydrogen_index in local_hydrogen_indices:
            out[label].append((int(oxygen_index), int(hydrogen_indices[local_hydrogen_index])))

    return {key: np.asarray(value, dtype=int).reshape((-1, 2)) for key, value in out.items()}


def oxygen_hydrogen_neighbors_by_species(
    symbols: list[str],
    positions: np.ndarray,
    *,
    oxygen_symbol: str = "O",
    hydrogen_symbol: str = "H",
    oh_cutoff: float = 1.25,
    neighbor_method: str = "auto",
    neighbor_workers: int = 1,
    oxygen_chunk_size: int = 2048,
    oxygen_indices: np.ndarray | None = None,
    hydrogen_indices: np.ndarray | None = None,
    cell: tuple[float, float, float] | None = None,
    pbc: tuple[bool, bool, bool] | None = None,
) -> dict[str, list[tuple[int, np.ndarray]]]:
    """Return local O-H neighbor lists grouped by oxygen species label.

    Each value is a list of ``(oxygen_index, hydrogen_indices)`` entries. This
    keeps one record per oxygen atom, which is useful for molecular vectors
    such as an H2O O-H bisector.
    """
    if oh_cutoff <= 0:
        raise ValueError("oh_cutoff must be positive.")
    if oxygen_chunk_size <= 0:
        raise ValueError("oxygen_chunk_size must be positive.")

    if oxygen_indices is None or hydrogen_indices is None:
        symbols_array = np.asarray(symbols)
        if oxygen_indices is None:
            oxygen_indices = np.where(symbols_array == oxygen_symbol)[0]
        if hydrogen_indices is None:
            hydrogen_indices = np.where(symbols_array == hydrogen_symbol)[0]
    else:
        oxygen_indices = np.asarray(oxygen_indices, dtype=int)
        hydrogen_indices = np.asarray(hydrogen_indices, dtype=int)

    out: dict[str, list[tuple[int, np.ndarray]]] = {
        "O2-": [],
        "OH-": [],
        "H2O": [],
        "H3O+": [],
        "O_other": [],
    }
    if oxygen_indices.size == 0 or hydrogen_indices.size == 0:
        return out

    neighbor_lists = _hydrogen_neighbor_lists(
        oxygen_positions=positions[oxygen_indices],
        hydrogen_positions=positions[hydrogen_indices],
        cutoff=oh_cutoff,
        method=neighbor_method,
        workers=neighbor_workers,
        oxygen_chunk_size=oxygen_chunk_size,
        cell=cell,
        pbc=pbc,
    )
    for oxygen_index, local_hydrogen_indices in zip(oxygen_indices, neighbor_lists):
        label = OXYGEN_SPECIES_BY_H_COUNT.get(len(local_hydrogen_indices), "O_other")
        absolute_hydrogen_indices = hydrogen_indices[local_hydrogen_indices]
        out[label].append((int(oxygen_index), np.asarray(absolute_hydrogen_indices, dtype=int)))

    return out


def _count_hydrogen_neighbors(
    *,
    oxygen_positions: np.ndarray,
    hydrogen_positions: np.ndarray,
    cutoff: float,
    method: str,
    workers: int,
    oxygen_chunk_size: int,
    cell: tuple[float, float, float] | None,
    pbc: tuple[bool, bool, bool] | None,
) -> np.ndarray:
    method = str(method).lower()
    if method not in {"auto", "cpp", "kdtree", "matrix"}:
        raise ValueError("selection.neighbor_method must be auto, cpp, kdtree, or matrix.")

    use_pbc = _uses_pbc(pbc)
    if method in {"auto", "cpp"}:
        native_counts = count_hydrogen_neighbors(
            oxygen_positions,
            hydrogen_positions,
            cutoff=cutoff,
            cell=cell,
            pbc=pbc,
        )
        if native_counts is not None:
            return native_counts
        if method == "cpp":
            raise RuntimeError("C++ O-H neighbor count backend is not available.")

    if use_pbc and method == "kdtree":
        raise ValueError("selection.neighbor_method: kdtree does not support pbc-aware O-H assignment; use auto or matrix.")

    if method in {"auto", "kdtree"} and not use_pbc:
        try:
            return _count_hydrogen_neighbors_kdtree(
                oxygen_positions=oxygen_positions,
                hydrogen_positions=hydrogen_positions,
                cutoff=cutoff,
                workers=workers,
            )
        except ImportError:
            if method == "kdtree":
                raise

    return _count_hydrogen_neighbors_matrix(
        oxygen_positions=oxygen_positions,
        hydrogen_positions=hydrogen_positions,
        cutoff=cutoff,
        oxygen_chunk_size=oxygen_chunk_size,
        cell=cell,
        pbc=pbc,
    )


def _hydrogen_neighbor_lists(
    *,
    oxygen_positions: np.ndarray,
    hydrogen_positions: np.ndarray,
    cutoff: float,
    method: str,
    workers: int,
    oxygen_chunk_size: int,
    cell: tuple[float, float, float] | None,
    pbc: tuple[bool, bool, bool] | None,
) -> list[np.ndarray]:
    method = str(method).lower()
    if method not in {"auto", "cpp", "kdtree", "matrix"}:
        raise ValueError("selection.neighbor_method must be auto, cpp, kdtree, or matrix.")
    if method == "cpp":
        raise ValueError("selection.neighbor_method: cpp is currently available for species counts only; use auto or matrix for neighbor lists.")

    use_pbc = _uses_pbc(pbc)
    if use_pbc and method == "kdtree":
        raise ValueError("selection.neighbor_method: kdtree does not support pbc-aware O-H assignment; use auto or matrix.")

    if method in {"auto", "kdtree"} and not use_pbc:
        try:
            return _hydrogen_neighbor_lists_kdtree(
                oxygen_positions=oxygen_positions,
                hydrogen_positions=hydrogen_positions,
                cutoff=cutoff,
                workers=workers,
            )
        except ImportError:
            if method == "kdtree":
                raise

    return _hydrogen_neighbor_lists_matrix(
        oxygen_positions=oxygen_positions,
        hydrogen_positions=hydrogen_positions,
        cutoff=cutoff,
        oxygen_chunk_size=oxygen_chunk_size,
        cell=cell,
        pbc=pbc,
    )


def _hydrogen_neighbor_lists_kdtree(
    *,
    oxygen_positions: np.ndarray,
    hydrogen_positions: np.ndarray,
    cutoff: float,
    workers: int,
) -> list[np.ndarray]:
    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:
        raise ImportError(
            "selection.neighbor_method: kdtree requires scipy. "
            "Install scipy or use neighbor_method: matrix."
        ) from exc

    tree = cKDTree(hydrogen_positions)
    try:
        raw_lists = tree.query_ball_point(oxygen_positions, r=cutoff, workers=workers)
    except TypeError:
        raw_lists = tree.query_ball_point(oxygen_positions, r=cutoff)
    return [np.asarray(items, dtype=int) for items in raw_lists]


def _hydrogen_neighbor_lists_matrix(
    *,
    oxygen_positions: np.ndarray,
    hydrogen_positions: np.ndarray,
    cutoff: float,
    oxygen_chunk_size: int,
    cell: tuple[float, float, float] | None,
    pbc: tuple[bool, bool, bool] | None,
) -> list[np.ndarray]:
    cutoff2 = cutoff * cutoff
    out: list[np.ndarray] = []
    for start in range(0, oxygen_positions.shape[0], oxygen_chunk_size):
        stop = min(start + oxygen_chunk_size, oxygen_positions.shape[0])
        oxygen_chunk = oxygen_positions[start:stop]
        d = hydrogen_positions[:, None, :] - oxygen_chunk[None, :, :]
        d = _minimum_image(d, cell=cell, pbc=pbc)
        dist2 = np.sum(d * d, axis=2)
        out.extend(np.where(dist2[:, i] <= cutoff2)[0] for i in range(stop - start))
    return [np.asarray(items, dtype=int) for items in out]


def _count_hydrogen_neighbors_kdtree(
    *,
    oxygen_positions: np.ndarray,
    hydrogen_positions: np.ndarray,
    cutoff: float,
    workers: int,
) -> np.ndarray:
    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:
        raise ImportError(
            "selection.neighbor_method: kdtree requires scipy. "
            "Install scipy or use neighbor_method: matrix."
        ) from exc

    tree = cKDTree(hydrogen_positions)
    try:
        h_counts = tree.query_ball_point(
            oxygen_positions,
            r=cutoff,
            workers=workers,
            return_length=True,
        )
    except TypeError:
        neighbor_lists = tree.query_ball_point(oxygen_positions, r=cutoff)
        h_counts = np.fromiter((len(items) for items in neighbor_lists), dtype=int, count=len(neighbor_lists))
    return np.asarray(h_counts, dtype=int)


def _count_hydrogen_neighbors_matrix(
    *,
    oxygen_positions: np.ndarray,
    hydrogen_positions: np.ndarray,
    cutoff: float,
    oxygen_chunk_size: int,
    cell: tuple[float, float, float] | None,
    pbc: tuple[bool, bool, bool] | None,
) -> np.ndarray:
    cutoff2 = cutoff * cutoff
    h_counts = np.empty(oxygen_positions.shape[0], dtype=int)
    for start in range(0, oxygen_positions.shape[0], oxygen_chunk_size):
        stop = min(start + oxygen_chunk_size, oxygen_positions.shape[0])
        oxygen_chunk = oxygen_positions[start:stop]
        d = hydrogen_positions[:, None, :] - oxygen_chunk[None, :, :]
        d = _minimum_image(d, cell=cell, pbc=pbc)
        dist2 = np.sum(d * d, axis=2)
        h_counts[start:stop] = np.sum(dist2 <= cutoff2, axis=0)
    return h_counts


def _uses_pbc(pbc: tuple[bool, bool, bool] | None) -> bool:
    return bool(pbc is not None and any(pbc))


def _minimum_image(
    vectors: np.ndarray,
    *,
    cell: tuple[float, float, float] | None,
    pbc: tuple[bool, bool, bool] | None,
) -> np.ndarray:
    if cell is None or pbc is None or not any(pbc):
        return vectors
    out = np.asarray(vectors, dtype=float).copy()
    cell_array = np.asarray(cell, dtype=float)
    for axis, enabled in enumerate(pbc):
        if enabled:
            out[..., axis] -= np.rint(out[..., axis] / cell_array[axis]) * cell_array[axis]
    return out
