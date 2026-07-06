from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from waterint.chemistry import oxygen_hydrogen_neighbors_by_species
from waterint.io.common import TrajectoryFrame
from waterint.core.selection import SelectionContext, element_indices


OXYGEN_SPECIES_ORDER = ("OH-", "H2O", "H3O+")
DEFAULT_CLASSES_BY_SPECIES = {
    "OH-": ("DAAA", "DAA", "DA", "AAA", "AA", "A", "other"),
    "H2O": ("DDAA", "DDA", "DAA", "DA", "AA", "A", "other"),
    "H3O+": ("DDDA", "DDD", "DDA", "DD", "DA", "D", "other"),
}


@dataclass
class HbondState:
    classes: dict[str, list[str]]
    species_labels: list[str]
    counts: dict[str, dict[str, int]]
    raw_counts: dict[str, dict[str, int]]
    samples_total: dict[str, int]
    frames: int = 0


@dataclass(frozen=True)
class HbondResult:
    classes: dict[str, list[str]]
    species_labels: list[str]
    counts: dict[str, dict[str, int]]
    raw_counts: dict[str, dict[str, int]]
    fractions: dict[str, dict[str, float]]
    samples_total: dict[str, int]
    frames: int
    csv_path: Path | None
    raw_csv_path: Path | None
    png_path: Path | None
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
        raise ValueError(f"Unknown H-bond oxygen species labels: {unknown}")
    return labels


def classes_by_species(hbond_cfg: dict[str, Any], labels: list[str]) -> dict[str, list[str]]:
    raw = hbond_cfg.get("classes_by_species", hbond_cfg.get("classes"))
    if raw is None:
        return {species: list(DEFAULT_CLASSES_BY_SPECIES[species]) for species in labels}
    if isinstance(raw, list):
        classes = validate_classes(raw, name="hbond.classes")
        return {species: classes[:] for species in labels}
    if not isinstance(raw, dict):
        raise ValueError("hbond.classes_by_species must be a mapping from species label to class list.")
    out: dict[str, list[str]] = {}
    for species in labels:
        raw_classes = raw.get(species, DEFAULT_CLASSES_BY_SPECIES[species])
        out[species] = validate_classes(raw_classes, name=f"hbond.classes_by_species.{species}")
    return out


def validate_classes(raw: Any, *, name: str) -> list[str]:
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{name} must be a non-empty list.")
    classes = [str(item) for item in raw]
    if len(set(classes)) != len(classes):
        raise ValueError(f"{name} cannot contain duplicates.")
    return classes


def new_hbond_state(classes: dict[str, list[str]], labels: list[str]) -> HbondState:
    return HbondState(
        classes=classes,
        species_labels=labels,
        counts={species: {class_label: 0 for class_label in classes[species]} for species in labels},
        raw_counts={species: {} for species in labels},
        samples_total={species: 0 for species in labels},
    )


def accumulate_frame_counts(
    state: HbondState,
    frame_counts: dict[str, dict[str, int]],
    frame_raw_counts: dict[str, dict[str, int]],
    frame_samples: dict[str, int],
) -> None:
    for species in state.species_labels:
        state.samples_total[species] += frame_samples[species]
        for class_label in state.classes[species]:
            state.counts[species][class_label] += frame_counts[species][class_label]
        for class_label, value in frame_raw_counts[species].items():
            state.raw_counts[species][class_label] = state.raw_counts[species].get(class_label, 0) + value
    state.frames += 1


def frame_hbond_classes(
    *,
    frame: TrajectoryFrame,
    selection_cfg: dict[str, Any],
    hbond_cfg: dict[str, Any],
    classes: dict[str, list[str]],
    selected_species: list[str],
    context: SelectionContext,
    cell: tuple[float, float, float],
) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, int]], dict[str, int]]:
    oxygen_symbol = str(selection_cfg.get("oxygen_symbol", "O"))
    hydrogen_symbol = str(selection_cfg.get("hydrogen_symbol", "H"))
    oxygen_indices = element_indices(frame, {oxygen_symbol}, context)
    hydrogen_indices = element_indices(frame, {hydrogen_symbol}, context)
    neighbors_by_species = oxygen_hydrogen_neighbors_by_species(
        frame.symbols,
        frame.positions,
        oxygen_symbol=oxygen_symbol,
        hydrogen_symbol=hydrogen_symbol,
        oh_cutoff=float(selection_cfg.get("oh_cutoff", 1.25)),
        neighbor_method=str(selection_cfg.get("neighbor_method", "auto")),
        neighbor_workers=int(selection_cfg.get("neighbor_workers", 1)),
        oxygen_chunk_size=int(selection_cfg.get("oxygen_chunk_size", 2048)),
        oxygen_indices=oxygen_indices,
        hydrogen_indices=hydrogen_indices,
    )
    species_by_oxygen: dict[int, str] = {}
    donor_neighbors: list[tuple[int, np.ndarray]] = []
    for species, entries in neighbors_by_species.items():
        for oxygen_index, attached_hydrogens in entries:
            species_by_oxygen[oxygen_index] = species
            if attached_hydrogens.size:
                donor_neighbors.append((oxygen_index, attached_hydrogens))

    donor_counts = {int(oxygen_index): 0 for oxygen_index in oxygen_indices}
    acceptor_counts = {int(oxygen_index): 0 for oxygen_index in oxygen_indices}
    bonds = find_hbonds(
        positions=frame.positions,
        oxygen_indices=oxygen_indices,
        donor_neighbors=donor_neighbors,
        hbond_cfg=hbond_cfg,
        cell=cell,
    )
    for donor_oxygen, _hydrogen, acceptor_oxygen in bonds:
        donor_counts[donor_oxygen] = donor_counts.get(donor_oxygen, 0) + 1
        acceptor_counts[acceptor_oxygen] = acceptor_counts.get(acceptor_oxygen, 0) + 1

    counts = {species: {class_label: 0 for class_label in classes[species]} for species in selected_species}
    raw_counts = {species: {} for species in selected_species}
    samples = {species: 0 for species in selected_species}
    for oxygen_index, species in species_by_oxygen.items():
        if species not in counts:
            continue
        samples[species] += 1
        class_label = topology_label(donor_counts.get(oxygen_index, 0), acceptor_counts.get(oxygen_index, 0))
        raw_counts[species][class_label] = raw_counts[species].get(class_label, 0) + 1
        grouped_label = class_label
        if grouped_label not in counts[species]:
            grouped_label = "other" if "other" in counts[species] else grouped_label
        if grouped_label in counts[species]:
            counts[species][grouped_label] += 1
    return counts, raw_counts, samples


def find_hbonds(
    *,
    positions: np.ndarray,
    oxygen_indices: np.ndarray,
    donor_neighbors: list[tuple[int, np.ndarray]],
    hbond_cfg: dict[str, Any],
    cell: tuple[float, float, float],
) -> list[tuple[int, int, int]]:
    oo_cutoff = float(hbond_cfg.get("oo_cutoff", 3.5))
    angle_min = float(hbond_cfg.get("dha_angle_min", 150.0))
    h_acceptor_cutoff = optional_float(hbond_cfg.get("h_acceptor_cutoff"))
    pbc = pbc_flags(hbond_cfg.get("pbc", [True, True, True]))
    max_acceptors_per_hydrogen = bool(hbond_cfg.get("max_acceptors_per_hydrogen", True))
    if oo_cutoff <= 0:
        raise ValueError("hbond.oo_cutoff must be positive.")
    if not 0 <= angle_min <= 180:
        raise ValueError("hbond.dha_angle_min must be between 0 and 180.")

    oxygen_positions = positions[oxygen_indices]
    oxygen_index_by_local = {local: int(index) for local, index in enumerate(oxygen_indices)}
    bonds: list[tuple[int, int, int]] = []
    for donor_oxygen, attached_hydrogens in donor_neighbors:
        donor_position = positions[donor_oxygen]
        oo_vectors = minimum_image(oxygen_positions - donor_position, cell=cell, pbc=pbc)
        oo_distances = np.linalg.norm(oo_vectors, axis=1)
        candidate_local = np.where((oo_distances <= oo_cutoff) & (oxygen_indices != donor_oxygen))[0]
        if candidate_local.size == 0:
            continue
        for hydrogen_index in attached_hydrogens:
            hydrogen_position = positions[hydrogen_index]
            passing: list[tuple[float, int]] = []
            for local_index in candidate_local:
                acceptor_oxygen = oxygen_index_by_local[int(local_index)]
                acceptor_vector = minimum_image(
                    positions[acceptor_oxygen] - hydrogen_position,
                    cell=cell,
                    pbc=pbc,
                )
                hydrogen_acceptor_distance = float(np.linalg.norm(acceptor_vector))
                if h_acceptor_cutoff is not None and hydrogen_acceptor_distance > h_acceptor_cutoff:
                    continue
                donor_vector = minimum_image(donor_position - hydrogen_position, cell=cell, pbc=pbc)
                angle = angle_degrees(donor_vector, acceptor_vector)
                if angle >= angle_min:
                    passing.append((hydrogen_acceptor_distance, acceptor_oxygen))
            if max_acceptors_per_hydrogen and passing:
                _, acceptor_oxygen = min(passing, key=lambda item: item[0])
                bonds.append((int(donor_oxygen), int(hydrogen_index), int(acceptor_oxygen)))
            elif passing:
                bonds.extend((int(donor_oxygen), int(hydrogen_index), int(acceptor)) for _, acceptor in passing)
    return bonds


def finalize_hbond_state(state: HbondState) -> HbondResult:
    fractions = {
        species: {
            class_label: (state.counts[species][class_label] / state.samples_total[species] if state.samples_total[species] else 0.0)
            for class_label in state.classes[species]
        }
        for species in state.species_labels
    }
    return HbondResult(
        classes=state.classes,
        species_labels=state.species_labels,
        counts=state.counts,
        raw_counts=state.raw_counts,
        fractions=fractions,
        samples_total=state.samples_total,
        frames=state.frames,
        csv_path=None,
        raw_csv_path=None,
        png_path=None,
        metadata_path=None,
    )


def minimum_image(vectors: np.ndarray, *, cell: tuple[float, float, float], pbc: tuple[bool, bool, bool]) -> np.ndarray:
    out = np.asarray(vectors, dtype=float).copy()
    cell_array = np.asarray(cell, dtype=float)
    for axis, enabled in enumerate(pbc):
        if enabled:
            length = cell_array[axis]
            if length <= 0:
                raise ValueError("Cell lengths must be positive when hbond.pbc is enabled.")
            out[..., axis] -= np.rint(out[..., axis] / length) * length
    return out


def angle_degrees(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    norm_a = float(np.linalg.norm(vector_a))
    norm_b = float(np.linalg.norm(vector_b))
    if norm_a <= 0 or norm_b <= 0:
        return 0.0
    cosine = float(np.dot(vector_a, vector_b) / (norm_a * norm_b))
    cosine = float(np.clip(cosine, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def topology_label(donor_count: int, acceptor_count: int) -> str:
    if donor_count <= 0 and acceptor_count <= 0:
        return "none"
    return "D" * max(donor_count, 0) + "A" * max(acceptor_count, 0)


def pbc_flags(value: Any) -> tuple[bool, bool, bool]:
    if isinstance(value, bool):
        return (value, value, value)
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError("hbond.pbc must be a boolean or a list of three booleans.")
    return tuple(bool(item) for item in value)  # type: ignore[return-value]


def optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)
