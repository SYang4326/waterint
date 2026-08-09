from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from waterint.chemistry import PythonCutoffNeighborSearch, oxygen_hydrogen_neighbors_by_species
from waterint._02_computation._native import hbond_geometry_counts, hydrogen_neighbor_matrix
from waterint._00_io.common import TrajectoryFrame
from waterint._01_core.selection import SelectionContext, element_indices


OXYGEN_SPECIES_ORDER = ("OH-", "H2O", "H3O+")
OXYGEN_SPECIES_BY_H_COUNT = {1: "OH-", 2: "H2O", 3: "H3O+"}
DEFAULT_CLASSES_BY_SPECIES = {
    "OH-": ("DAAA", "DAA", "DA", "AAA", "AA", "A", "other"),
    "H2O": ("DDAA", "DDA", "DAA", "DA", "AA", "A", "other"),
    "H3O+": ("DDDA", "DDD", "DDA", "DD", "DA", "D", "other"),
}


@dataclass
class HbondState:
    """Mutable accumulators updated once per trajectory frame."""

    classes: dict[str, list[str]]
    species_labels: list[str]
    counts: dict[str, dict[str, int]]
    raw_counts: dict[str, dict[str, int]]
    samples_total: dict[str, int]
    frames: int = 0


@dataclass(frozen=True)
class HbondResult:
    """Final H-bond statistics and the paths written by the workflow."""

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
    """Read and validate the oxygen species included in the analysis."""

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
    """Resolve the displayed topology classes for each oxygen species."""

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
    """Require a non-empty class list without duplicate topology labels."""

    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{name} must be a non-empty list.")
    classes = [str(item) for item in raw]
    if len(set(classes)) != len(classes):
        raise ValueError(f"{name} cannot contain duplicates.")
    return classes


def new_hbond_state(classes: dict[str, list[str]], labels: list[str]) -> HbondState:
    """Create zero-filled accumulators for a new H-bond trajectory run."""

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
    """Add one frame's grouped and raw topology counts to the run state."""

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
    backend: str = "auto",
) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, int]], dict[str, int]]:
    """Select O/H atoms, dispatch a backend, and classify one frame.

    Both backends ultimately produce three arrays per oxygen: attached-H,
    donated-bond, and accepted-bond counts. The shared Python grouping step
    converts those arrays into labels such as DDAA or DAA.
    """

    oxygen_symbol = str(selection_cfg.get("oxygen_symbol", "O"))
    hydrogen_symbol = str(selection_cfg.get("hydrogen_symbol", "H"))
    oxygen_indices = element_indices(frame, {oxygen_symbol}, context)
    hydrogen_indices = element_indices(frame, {hydrogen_symbol}, context)
    if backend not in {"auto", "python", "cpp"}:
        raise ValueError("hbond.backend must be auto, python, or cpp.")
    if oxygen_indices.size == 0 or hydrogen_indices.size == 0:
        # Keep the output shape valid even when this frame has no analyzable O-H system.
        return frame_hbond_classes_from_counts(
            oxygen_indices=oxygen_indices,
            h_counts=np.zeros(oxygen_indices.size, dtype=int),
            donor_counts=np.zeros(oxygen_indices.size, dtype=int),
            acceptor_counts=np.zeros(oxygen_indices.size, dtype=int),
            classes=classes,
            selected_species=selected_species,
        )
    if backend in {"auto", "cpp"}:
        # Stage 1: shared C++ cell list assigns local hydrogen indices to each oxygen.
        native_neighbors = build_hbond_neighbor_matrix_cpp(
            oxygen_positions=frame.positions[oxygen_indices],
            hydrogen_positions=frame.positions[hydrogen_indices],
            cutoff=float(selection_cfg.get("oh_cutoff", 1.25)),
        )
        if native_neighbors is not None:
            h_counts, h_matrix = native_neighbors
            # Stage 2: hbond.cpp evaluates O-O candidates and D-H-A geometry.
            native_geometry = hbond_geometry_counts(
                frame.positions[oxygen_indices],
                frame.positions[hydrogen_indices],
                hydrogen_counts=h_counts,
                hydrogen_matrix=h_matrix,
                oo_cutoff=float(hbond_cfg.get("oo_cutoff", 3.5)),
                dha_angle_min=float(hbond_cfg.get("dha_angle_min", 150.0)),
                h_acceptor_cutoff=optional_float(hbond_cfg.get("h_acceptor_cutoff")),
                cell=cell,
                pbc=pbc_flags(hbond_cfg.get("pbc", [True, True, True])),
                max_acceptors_per_hydrogen=bool(hbond_cfg.get("max_acceptors_per_hydrogen", True)),
            )
            if native_geometry is None:
                if backend == "cpp":
                    raise RuntimeError("C++ H-bond backend is not available.")
            else:
                donor_counts, acceptor_counts = native_geometry
                # Labels and user-defined class grouping intentionally remain in Python.
                return frame_hbond_classes_from_counts(
                    oxygen_indices=oxygen_indices,
                    h_counts=h_counts,
                    donor_counts=donor_counts,
                    acceptor_counts=acceptor_counts,
                    classes=classes,
                    selected_species=selected_species,
                )
        if backend == "cpp":
            raise RuntimeError("C++ H-bond backend is not available.")

    # Explicit Python mode, or the fallback used when backend="auto" cannot load C++.
    return frame_hbond_classes_python(
        frame=frame,
        selection_cfg=selection_cfg,
        hbond_cfg=hbond_cfg,
        classes=classes,
        selected_species=selected_species,
        oxygen_indices=oxygen_indices,
        hydrogen_indices=hydrogen_indices,
        cell=cell,
    )


def build_hbond_neighbor_matrix_cpp(
    *,
    oxygen_positions: np.ndarray,
    hydrogen_positions: np.ndarray,
    cutoff: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Build the shared C++ O-H neighbor matrix also used by OH orientation.

    ``counts[i]`` is the number of H atoms attached to oxygen ``i`` and row
    ``matrix[i]`` stores their local indices in ``hydrogen_positions``.
    """

    return hydrogen_neighbor_matrix(oxygen_positions, hydrogen_positions, cutoff=cutoff)


def frame_hbond_classes_python(
    *,
    frame: TrajectoryFrame,
    selection_cfg: dict[str, Any],
    hbond_cfg: dict[str, Any],
    classes: dict[str, list[str]],
    selected_species: list[str],
    oxygen_indices: np.ndarray,
    hydrogen_indices: np.ndarray,
    cell: tuple[float, float, float],
) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, int]], dict[str, int]]:
    """Run the complete Python cell-list implementation for one frame."""

    oxygen_symbol = str(selection_cfg.get("oxygen_symbol", "O"))
    hydrogen_symbol = str(selection_cfg.get("hydrogen_symbol", "H"))
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
    h_counts_by_oxygen: dict[int, int] = {}
    donor_neighbors: list[tuple[int, np.ndarray]] = []
    # Convert species-grouped neighbor records into the donor list expected by find_hbonds().
    for entries in neighbors_by_species.values():
        for oxygen_index, attached_hydrogens in entries:
            h_counts_by_oxygen[oxygen_index] = int(attached_hydrogens.size)
            if attached_hydrogens.size:
                donor_neighbors.append((oxygen_index, attached_hydrogens))

    donor_counts_by_oxygen = {int(oxygen_index): 0 for oxygen_index in oxygen_indices}
    acceptor_counts_by_oxygen = {int(oxygen_index): 0 for oxygen_index in oxygen_indices}
    # The Python cell-list routine returns explicit (donor O, H, acceptor O) triples.
    bonds = find_hbonds(
        positions=frame.positions,
        oxygen_indices=oxygen_indices,
        donor_neighbors=donor_neighbors,
        hbond_cfg=hbond_cfg,
        cell=cell,
    )
    for donor_oxygen, _hydrogen, acceptor_oxygen in bonds:
        donor_counts_by_oxygen[donor_oxygen] = donor_counts_by_oxygen.get(donor_oxygen, 0) + 1
        acceptor_counts_by_oxygen[acceptor_oxygen] = acceptor_counts_by_oxygen.get(acceptor_oxygen, 0) + 1

    h_counts = np.asarray([h_counts_by_oxygen.get(int(index), 0) for index in oxygen_indices], dtype=int)
    return frame_hbond_classes_from_counts(
        oxygen_indices=oxygen_indices,
        h_counts=h_counts,
        donor_counts=np.asarray([donor_counts_by_oxygen.get(int(index), 0) for index in oxygen_indices], dtype=int),
        acceptor_counts=np.asarray([acceptor_counts_by_oxygen.get(int(index), 0) for index in oxygen_indices], dtype=int),
        classes=classes,
        selected_species=selected_species,
    )


def frame_hbond_classes_from_counts(
    *,
    oxygen_indices: np.ndarray,
    h_counts: np.ndarray,
    donor_counts: np.ndarray,
    acceptor_counts: np.ndarray,
    classes: dict[str, list[str]],
    selected_species: list[str],
) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, int]], dict[str, int]]:
    """Convert per-oxygen integer counts into species and topology classes.

    This function is shared by the Python and C++ routes so class names,
    ``other`` grouping, and sample totals cannot diverge between backends.
    """

    if not (oxygen_indices.size == h_counts.size == donor_counts.size == acceptor_counts.size):
        raise ValueError("H-bond count arrays must have one value per oxygen atom.")

    counts = {species: {class_label: 0 for class_label in classes[species]} for species in selected_species}
    raw_counts = {species: {} for species in selected_species}
    samples = {species: 0 for species in selected_species}
    for local_index in range(oxygen_indices.size):
        # Attached-H count identifies OH-, H2O, or H3O+.
        species = OXYGEN_SPECIES_BY_H_COUNT.get(int(h_counts[local_index]))
        if species not in counts:
            continue
        samples[species] += 1
        # Donor/acceptor counts identify the raw topology, for example 2D + 2A -> DDAA.
        class_label = topology_label(int(donor_counts[local_index]), int(acceptor_counts[local_index]))
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
    """Find H bonds with a Python cell list and vectorized local geometry.

    O-O cell-list queries replace a donor-to-all-oxygen scan. Geometry for all
    local D-H-A candidates in the frame is evaluated in one NumPy batch.
    """

    oo_cutoff, angle_min, h_acceptor_cutoff, pbc, max_acceptors = _hbond_parameters(hbond_cfg)
    oxygen_positions = np.ascontiguousarray(positions[oxygen_indices], dtype=float)
    oxygen_local_by_global = {int(index): local for local, index in enumerate(oxygen_indices)}
    acceptor_search = PythonCutoffNeighborSearch(
        oxygen_positions,
        oxygen_positions,
        cutoff=oo_cutoff,
        cell=cell,
        pbc=pbc,
    )

    donor_blocks: list[np.ndarray] = []
    hydrogen_blocks: list[np.ndarray] = []
    acceptor_blocks: list[np.ndarray] = []
    group_blocks: list[np.ndarray] = []
    group_index = 0
    for donor_oxygen, attached_hydrogens in donor_neighbors:
        donor_local = oxygen_local_by_global[int(donor_oxygen)]
        candidate_local = acceptor_search.collect_indices(donor_local)
        candidate_local = candidate_local[candidate_local != donor_local]
        if candidate_local.size == 0:
            group_index += attached_hydrogens.size
            continue
        for hydrogen_index in attached_hydrogens:
            n_candidates = candidate_local.size
            donor_blocks.append(np.full(n_candidates, int(donor_oxygen), dtype=int))
            hydrogen_blocks.append(np.full(n_candidates, int(hydrogen_index), dtype=int))
            acceptor_blocks.append(candidate_local)
            group_blocks.append(np.full(n_candidates, group_index, dtype=int))
            group_index += 1
    if not acceptor_blocks:
        return []

    # Evaluate all local D-H-A candidates in one vectorized NumPy batch.
    donor_global = np.concatenate(donor_blocks)
    hydrogen_global = np.concatenate(hydrogen_blocks)
    acceptor_local = np.concatenate(acceptor_blocks)
    groups = np.concatenate(group_blocks)
    acceptor_vectors = acceptor_search.minimum_image(
        oxygen_positions[acceptor_local] - positions[hydrogen_global]
    )
    donor_vectors = acceptor_search.minimum_image(
        positions[donor_global] - positions[hydrogen_global]
    )
    acceptor_distances = np.linalg.norm(acceptor_vectors, axis=1)
    donor_distances = np.linalg.norm(donor_vectors, axis=1)

    valid_norms = (acceptor_distances > 0.0) & (donor_distances > 0.0)
    cosines = np.ones(acceptor_distances.size, dtype=float)
    cosines[valid_norms] = np.einsum(
        "ij,ij->i",
        donor_vectors[valid_norms],
        acceptor_vectors[valid_norms],
    ) / (donor_distances[valid_norms] * acceptor_distances[valid_norms])
    angles = np.zeros(acceptor_distances.size, dtype=float)
    angles[valid_norms] = np.degrees(np.arccos(np.clip(cosines[valid_norms], -1.0, 1.0)))
    passing = angles >= angle_min
    if h_acceptor_cutoff is not None:
        passing &= acceptor_distances <= h_acceptor_cutoff
    if not np.any(passing):
        return []

    donor_global = donor_global[passing]
    hydrogen_global = hydrogen_global[passing]
    acceptor_local = acceptor_local[passing]
    acceptor_distances = acceptor_distances[passing]
    groups = groups[passing]
    if max_acceptors:
        # Sort by H group, H-A distance, then O index; take the first of each group.
        order = np.lexsort((acceptor_local, acceptor_distances, groups))
        sorted_groups = groups[order]
        first_in_group = np.r_[True, sorted_groups[1:] != sorted_groups[:-1]]
        selected = order[first_in_group]
        donor_global = donor_global[selected]
        hydrogen_global = hydrogen_global[selected]
        acceptor_local = acceptor_local[selected]

    acceptor_global = oxygen_indices[acceptor_local]
    return [
        (int(donor), int(hydrogen), int(acceptor))
        for donor, hydrogen, acceptor in zip(donor_global, hydrogen_global, acceptor_global)
    ]


def _hbond_parameters(
    hbond_cfg: dict[str, Any],
) -> tuple[float, float, float | None, tuple[bool, bool, bool], bool]:
    """Parse and validate the Python H-bond geometry parameters."""

    oo_cutoff = float(hbond_cfg.get("oo_cutoff", 3.5))
    angle_min = float(hbond_cfg.get("dha_angle_min", 150.0))
    h_acceptor_cutoff = optional_float(hbond_cfg.get("h_acceptor_cutoff"))
    pbc = pbc_flags(hbond_cfg.get("pbc", [True, True, True]))
    max_acceptors_per_hydrogen = bool(hbond_cfg.get("max_acceptors_per_hydrogen", True))
    if oo_cutoff <= 0:
        raise ValueError("hbond.oo_cutoff must be positive.")
    if not 0 <= angle_min <= 180:
        raise ValueError("hbond.dha_angle_min must be between 0 and 180.")
    return oo_cutoff, angle_min, h_acceptor_cutoff, pbc, max_acceptors_per_hydrogen


def finalize_hbond_state(state: HbondState) -> HbondResult:
    """Convert accumulated counts to fractions and freeze the result object."""

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


def topology_label(donor_count: int, acceptor_count: int) -> str:
    """Encode donor/acceptor counts as labels such as DDAA, DA, or none."""

    if donor_count <= 0 and acceptor_count <= 0:
        return "none"
    return "D" * max(donor_count, 0) + "A" * max(acceptor_count, 0)


def pbc_flags(value: Any) -> tuple[bool, bool, bool]:
    """Normalize one boolean or three booleans to orthorhombic PBC flags."""

    if isinstance(value, bool):
        return (value, value, value)
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError("hbond.pbc must be a boolean or a list of three booleans.")
    return tuple(bool(item) for item in value)  # type: ignore[return-value]


def optional_float(value: Any) -> float | None:
    """Parse an optional numeric config value while preserving null as None."""

    if value is None or value == "":
        return None
    return float(value)
