from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from waterint._02_computation._native import rdf_histogram


@dataclass(frozen=True)
class RdfPairResult:
    name: str
    r_centers: np.ndarray
    g_r: np.ndarray
    counts: np.ndarray
    expected_counts: np.ndarray
    frames: int
    csv_path: Path | None = None
    png_path: Path | None = None


def new_histogram(n_bins: int) -> np.ndarray:
    if n_bins <= 0:
        raise ValueError("rdf.bins must be positive.")
    return np.zeros(n_bins, dtype=float)


def accumulate_rdf_frame(
    counts: np.ndarray,
    *,
    positions: np.ndarray,
    first_indices: np.ndarray,
    second_indices: np.ndarray,
    r_max: float,
    cell_vectors: np.ndarray | None,
    pbc: tuple[bool, bool, bool],
    same_selection: bool,
    backend: str = "auto",
) -> None:
    mode = str(backend).lower()
    if mode not in {"auto", "python", "cpp"}:
        raise ValueError("rdf.backend must be auto, python, or cpp.")
    if first_indices.size == 0 or second_indices.size == 0:
        return
    frame_counts = None
    if mode in {"auto", "cpp"}:
        frame_counts = rdf_histogram(
            positions,
            first_indices,
            second_indices,
            r_max=r_max,
            n_bins=counts.size,
            cell_vectors=cell_vectors,
            pbc=pbc,
            same_selection=same_selection,
        )
        if frame_counts is None and mode == "cpp":
            raise RuntimeError("C++ RDF backend is not available.")
    if frame_counts is None:
        frame_counts = _rdf_histogram_python(
            positions,
            first_indices,
            second_indices,
            r_max=r_max,
            n_bins=counts.size,
            cell_vectors=cell_vectors,
            pbc=pbc,
            same_selection=same_selection,
        )
    counts += frame_counts


def expected_rdf_counts(
    *,
    first_size: int,
    second_size: int,
    same_selection: bool,
    overlap_size: int,
    volume: float,
    r_edges: np.ndarray,
) -> np.ndarray:
    pair_count = first_size * (first_size - 1) / 2.0 if same_selection else first_size * second_size - overlap_size
    shell_volumes = 4.0 * np.pi / 3.0 * (r_edges[1:] ** 3 - r_edges[:-1] ** 3)
    return pair_count * shell_volumes / volume


def finalize_rdf_pair(
    *,
    name: str,
    counts: np.ndarray,
    expected_counts: np.ndarray,
    r_edges: np.ndarray,
    frames: int,
) -> RdfPairResult:
    if frames == 0:
        raise ValueError("Cannot finalize RDF with zero frames.")
    return RdfPairResult(
        name=name,
        r_centers=0.5 * (r_edges[:-1] + r_edges[1:]),
        g_r=np.divide(counts, expected_counts, out=np.zeros_like(counts), where=expected_counts > 0),
        counts=counts,
        expected_counts=expected_counts,
        frames=frames,
    )


def _rdf_histogram_python(
    positions: np.ndarray,
    first_indices: np.ndarray,
    second_indices: np.ndarray,
    *,
    r_max: float,
    n_bins: int,
    cell_vectors: np.ndarray | None,
    pbc: tuple[bool, bool, bool],
    same_selection: bool,
) -> np.ndarray:
    from waterint._01_core.cell import minimum_image

    first = positions[first_indices]
    second = positions[second_indices]
    delta = second[None, :, :] - first[:, None, :]
    if cell_vectors is not None and any(pbc):
        delta = minimum_image(delta, cell_vectors=cell_vectors, pbc=pbc)
    distances = np.linalg.norm(delta, axis=-1)
    if same_selection:
        upper = np.triu_indices(first_indices.size, k=1)
        distances = distances[upper]
    else:
        distances = distances[first_indices[:, None] != second_indices[None, :]]
    hist, _ = np.histogram(distances, bins=n_bins, range=(0.0, r_max))
    return hist.astype(float)
