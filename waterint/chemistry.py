from __future__ import annotations

import numpy as np


OXYGEN_SPECIES_BY_H_COUNT = {
    0: "O2-",
    1: "OH-",
    2: "H2O",
    3: "H3O+",
}


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

    oxygen_positions = positions[oxygen_indices]
    hydrogen_positions = positions[hydrogen_indices]
    h_counts = _count_hydrogen_neighbors(
        oxygen_positions=oxygen_positions,
        hydrogen_positions=hydrogen_positions,
        cutoff=oh_cutoff,
        method=neighbor_method,
        workers=neighbor_workers,
        oxygen_chunk_size=oxygen_chunk_size,
    )

    for oxygen_index, h_count in zip(oxygen_indices, h_counts):
        label = OXYGEN_SPECIES_BY_H_COUNT.get(int(h_count), "O_other")
        out[label].append(int(oxygen_index))

    return {key: np.asarray(value, dtype=int) for key, value in out.items()}


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
) -> np.ndarray:
    method = str(method).lower()
    if method not in {"auto", "kdtree", "matrix"}:
        raise ValueError("selection.neighbor_method must be auto, kdtree, or matrix.")

    if method in {"auto", "kdtree"}:
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
    )


def _hydrogen_neighbor_lists(
    *,
    oxygen_positions: np.ndarray,
    hydrogen_positions: np.ndarray,
    cutoff: float,
    method: str,
    workers: int,
    oxygen_chunk_size: int,
) -> list[np.ndarray]:
    method = str(method).lower()
    if method not in {"auto", "kdtree", "matrix"}:
        raise ValueError("selection.neighbor_method must be auto, kdtree, or matrix.")

    if method in {"auto", "kdtree"}:
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
) -> list[np.ndarray]:
    cutoff2 = cutoff * cutoff
    out: list[np.ndarray] = []
    for start in range(0, oxygen_positions.shape[0], oxygen_chunk_size):
        stop = min(start + oxygen_chunk_size, oxygen_positions.shape[0])
        oxygen_chunk = oxygen_positions[start:stop]
        d = hydrogen_positions[:, None, :] - oxygen_chunk[None, :, :]
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
) -> np.ndarray:
    cutoff2 = cutoff * cutoff
    h_counts = np.empty(oxygen_positions.shape[0], dtype=int)
    for start in range(0, oxygen_positions.shape[0], oxygen_chunk_size):
        stop = min(start + oxygen_chunk_size, oxygen_positions.shape[0])
        oxygen_chunk = oxygen_positions[start:stop]
        d = hydrogen_positions[:, None, :] - oxygen_chunk[None, :, :]
        dist2 = np.sum(d * d, axis=2)
        h_counts[start:stop] = np.sum(dist2 <= cutoff2, axis=0)
    return h_counts
