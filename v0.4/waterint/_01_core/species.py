from __future__ import annotations

from typing import Any

import numpy as np

from waterint.chemistry import classify_oxygen_by_h_count
from waterint._00_io.common import TrajectoryFrame
from waterint._01_core.selection import SelectionContext, element_indices


OXYGEN_SPECIES_ORDER = ("O2-", "OH-", "H2O", "H3O+", "O_other")


def oxygen_species_labels(selection_cfg: dict[str, Any]) -> list[str]:
    selected = selection_cfg.get("oxygen_species", "all")
    if selected == "all":
        return list(OXYGEN_SPECIES_ORDER)
    if not isinstance(selected, list) or not selected:
        raise ValueError("selection.oxygen_species must be 'all' or a non-empty list.")
    labels = [str(item) for item in selected]
    unknown = [label for label in labels if label not in OXYGEN_SPECIES_ORDER]
    if unknown:
        raise ValueError(f"Unknown oxygen species labels: {unknown}")
    return labels


def oxygen_species_indices(
    frame: TrajectoryFrame,
    selection_cfg: dict[str, Any],
    context: SelectionContext,
) -> dict[str, np.ndarray]:
    oxygen_symbol = str(selection_cfg.get("oxygen_symbol", "O"))
    hydrogen_symbol = str(selection_cfg.get("hydrogen_symbol", "H"))
    classified = classify_oxygen_by_h_count(
        frame.symbols,
        frame.positions,
        oxygen_symbol=oxygen_symbol,
        hydrogen_symbol=hydrogen_symbol,
        oh_cutoff=float(selection_cfg.get("oh_cutoff", 1.25)),
        neighbor_method=str(selection_cfg.get("neighbor_method", "auto")),
        neighbor_workers=int(selection_cfg.get("neighbor_workers", 1)),
        oxygen_chunk_size=int(selection_cfg.get("oxygen_chunk_size", 2048)),
        hydrogen_assignment=str(selection_cfg.get("hydrogen_assignment", "nearest")),
        oxygen_indices=element_indices(frame, {oxygen_symbol}, context),
        hydrogen_indices=element_indices(frame, {hydrogen_symbol}, context),
        cell=frame.cell,
        cell_vectors=frame.cell_vectors if frame.triclinic else None,
        pbc=_pbc_flags(selection_cfg.get("pbc", [True, True, False])),
    )
    return {label: classified[label] for label in oxygen_species_labels(selection_cfg)}


def _pbc_flags(value: Any) -> tuple[bool, bool, bool]:
    if isinstance(value, bool):
        return (value, value, value)
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError("selection.pbc must be a boolean or a list of three booleans.")
    return tuple(bool(item) for item in value)  # type: ignore[return-value]
