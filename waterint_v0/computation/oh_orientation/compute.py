from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import math

import numpy as np


OXYGEN_SPECIES_ORDER = ("OH-", "H2O", "H3O+")


@dataclass
class OhOrientationState:
    z_edges: np.ndarray
    angle_edges: np.ndarray
    counts_by_species: dict[str, np.ndarray]
    bond_counts_total: dict[str, int]
    sample_counts_total: dict[str, int]
    frames: int = 0


@dataclass(frozen=True)
class OhOrientationResult:
    z_centers: np.ndarray
    angle_centers: np.ndarray
    histograms: dict[str, np.ndarray]
    frames: int
    bond_counts_total: dict[str, int]
    sample_counts_total: dict[str, int]
    csv_paths: dict[str, Path]
    png_paths: dict[str, Path]
    metadata_path: Path | None


def species_labels(selection_cfg: dict[str, Any]) -> list[str]:
    selected = selection_cfg.get("oxygen_species", list(OXYGEN_SPECIES_ORDER))
    if selected == "all":
        return list(OXYGEN_SPECIES_ORDER)
    if not isinstance(selected, list) or not selected:
        raise ValueError("selection.oxygen_species must be 'all' or a non-empty list.")
    labels = [str(item) for item in selected]
    unknown = [label for label in labels if label not in OXYGEN_SPECIES_ORDER]
    if unknown:
        raise ValueError(f"Unknown angle-z oxygen species labels: {unknown}")
    return labels


def new_oh_orientation_state(labels: list[str], z_edges: np.ndarray, angle_edges: np.ndarray) -> OhOrientationState:
    return OhOrientationState(
        z_edges=z_edges,
        angle_edges=angle_edges,
        counts_by_species={label: np.zeros((len(z_edges) - 1, len(angle_edges) - 1), dtype=float) for label in labels},
        bond_counts_total={label: 0 for label in labels},
        sample_counts_total={label: 0 for label in labels},
    )


def accumulate_angle_samples(
    state: OhOrientationState,
    label: str,
    z_values: np.ndarray,
    angle_values: np.ndarray,
    *,
    bond_count: int,
) -> None:
    if z_values.size == 0:
        return
    hist, _, _ = np.histogram2d(z_values, angle_values, bins=[state.z_edges, state.angle_edges])
    state.counts_by_species[label] += hist
    state.bond_counts_total[label] += int(bond_count)
    state.sample_counts_total[label] += int(z_values.size)


def pair_z_and_angles(
    *,
    positions: np.ndarray,
    pairs: np.ndarray,
    axis: int,
    axis_sign: float,
    reference: float,
    angle_axis_sign: float,
) -> tuple[np.ndarray, np.ndarray]:
    oxygen_positions = positions[pairs[:, 0]]
    hydrogen_positions = positions[pairs[:, 1]]
    vectors = hydrogen_positions - oxygen_positions
    norms = np.linalg.norm(vectors, axis=1)
    valid = norms > 0
    vectors = vectors[valid]
    norms = norms[valid]
    oxygen_positions = oxygen_positions[valid]
    cos_theta = angle_axis_sign * vectors[:, axis] / norms
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angles = np.degrees(np.arccos(cos_theta))
    z_values = axis_sign * (oxygen_positions[:, axis] - reference)
    return z_values, angles


def neighbor_bisector_z_and_angles(
    *,
    positions: np.ndarray,
    neighbors: list[tuple[int, np.ndarray]],
    axis: int,
    axis_sign: float,
    reference: float,
    angle_axis_sign: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    z_values: list[float] = []
    angle_values: list[float] = []
    bond_count = 0
    for oxygen_index, hydrogen_indices in neighbors:
        if hydrogen_indices.size == 0:
            continue
        oxygen_position = positions[oxygen_index]
        vectors = positions[hydrogen_indices] - oxygen_position
        norms = np.linalg.norm(vectors, axis=1)
        valid = norms > 0
        if not np.any(valid):
            continue
        unit_vectors = vectors[valid] / norms[valid, None]
        direction = np.sum(unit_vectors, axis=0)
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm <= 0:
            continue
        direction = direction / direction_norm
        cos_theta = angle_axis_sign * direction[axis]
        cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
        angle_values.append(float(np.degrees(np.arccos(cos_theta))))
        z_values.append(float(axis_sign * (oxygen_position[axis] - reference)))
        bond_count += int(np.count_nonzero(valid))
    return np.asarray(z_values, dtype=float), np.asarray(angle_values, dtype=float), bond_count


def finalize_oh_orientation_state(
    state: OhOrientationState,
    *,
    cell: tuple[float, float, float],
    axis: int,
    normalization_cfg: Any,
) -> OhOrientationResult:
    if state.frames == 0:
        raise ValueError("Cannot finalize angle-z histogram with zero frames.")
    z_widths = np.diff(state.z_edges)
    angle_widths = np.diff(state.angle_edges)
    if not np.allclose(z_widths, z_widths[0]) or not np.allclose(angle_widths, angle_widths[0]):
        raise ValueError("Angle-z normalization currently requires uniform bins.")
    histograms = {
        label: normalize_histogram(
            counts=counts,
            frames=state.frames,
            cell=cell,
            axis=axis,
            z_bin_width=float(z_widths[0]),
            angle_bin_width=float(angle_widths[0]),
            normalization_cfg=normalization_cfg,
        )
        for label, counts in state.counts_by_species.items()
    }
    return OhOrientationResult(
        z_centers=0.5 * (state.z_edges[:-1] + state.z_edges[1:]),
        angle_centers=0.5 * (state.angle_edges[:-1] + state.angle_edges[1:]),
        histograms=histograms,
        frames=state.frames,
        bond_counts_total=state.bond_counts_total,
        sample_counts_total=state.sample_counts_total,
        csv_paths={},
        png_paths={},
        metadata_path=None,
    )


def normalize_histogram(
    *,
    counts: np.ndarray,
    frames: int,
    cell: tuple[float, float, float],
    axis: int,
    z_bin_width: float,
    angle_bin_width: float,
    normalization_cfg: Any,
) -> np.ndarray:
    if normalization_cfg is None:
        normalization_cfg = {}
    if not isinstance(normalization_cfg, dict):
        raise ValueError("normalization must be a mapping.")
    norm_type = str(normalization_cfg.get("type", "counts_per_frame"))
    counts_per_frame = counts / frames
    if norm_type == "counts_per_frame":
        return counts_per_frame
    if norm_type != "number_density":
        raise ValueError("normalization.type must be counts_per_frame or number_density.")
    perpendicular_lengths = [cell[i] for i in range(3) if i != axis]
    volume_angle = perpendicular_lengths[0] * perpendicular_lengths[1] * z_bin_width * angle_bin_width
    if not math.isfinite(volume_angle) or volume_angle <= 0:
        raise ValueError("Computed z-angle normalization volume must be positive.")
    return counts_per_frame / volume_angle


AngleZState = OhOrientationState
AngleZResult = OhOrientationResult
new_angle_z_state = new_oh_orientation_state
finalize_angle_z_state = finalize_oh_orientation_state
