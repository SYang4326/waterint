"""v0.6 selection / coordinate / species helpers, kept ONLY for the SFG kernels (sfg_core, sfg_species_core).

SFG assigns every H to one O itself (an SFG-specific rule: within cutoff, otherwise the nearest O) and is ported
unchanged so its spectra stay bitwise identical. Folding that rule into prep.ownership is an open item
(docs/PHASE2_REPORT.md). New code must use waterint.prep instead.
"""
from __future__ import annotations

# ---- from v0.6 _01_core/selection.py

from dataclasses import dataclass
from typing import Any

import numpy as np

from waterint.io.frame import TrajectoryFrame


@dataclass(frozen=True)
class SelectionContext:
    symbol_to_types: dict[str, list[int]]

    @classmethod
    def from_input_config(cls, input_cfg: dict[str, Any]) -> "SelectionContext":
        raw_type_map = input_cfg.get("type_map", {})
        symbol_to_types: dict[str, list[int]] = {}
        if isinstance(raw_type_map, dict):
            for raw_type, raw_symbol in raw_type_map.items():
                symbol_to_types.setdefault(str(raw_symbol), []).append(int(raw_type))
        return cls(symbol_to_types=symbol_to_types)


def element_mask(
    frame: TrajectoryFrame,
    species: set[str],
    context: SelectionContext,
) -> np.ndarray:
    if frame.types is not None and context.symbol_to_types:
        type_ids: list[int] = []
        for symbol in species:
            type_ids.extend(context.symbol_to_types.get(symbol, []))
        if type_ids:
            return np.isin(frame.types, type_ids)
    return np.isin(np.asarray(frame.symbols), list(species))


def element_indices(
    frame: TrajectoryFrame,
    species: set[str],
    context: SelectionContext,
) -> np.ndarray:
    return np.where(element_mask(frame, species, context))[0]


# ---- from v0.6 _01_core/coordinates.py

from dataclasses import dataclass
from typing import Any

import numpy as np

from waterint.io.frame import TrajectoryFrame


AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


@dataclass(frozen=True)
class CoordinateSpec:
    label: str
    axis: int
    sign: float
    mode: str = "absolute"
    reference: dict[str, Any] | None = None


def parse_axis(value: Any) -> tuple[str, int, float]:
    axis_label = str(value).strip().lower()
    sign = -1.0 if axis_label.startswith("-") else 1.0
    bare_axis = axis_label[1:] if axis_label.startswith("-") else axis_label
    if bare_axis not in AXIS_INDEX:
        raise ValueError("axis must be one of x, y, z, -x, -y, -z.")
    return axis_label, AXIS_INDEX[bare_axis], sign


def coordinate_spec_from_config(coord_cfg: dict[str, Any]) -> CoordinateSpec:
    axis_label, axis, sign = parse_axis(coord_cfg.get("axis", "z"))
    mode = str(coord_cfg.get("mode", "absolute"))
    if mode not in {"absolute", "relative_to_reference", "relative_to_slab"}:
        raise ValueError("coordinate.mode must be absolute, relative_to_reference, or relative_to_slab.")
    reference = coord_cfg.get("reference", {})
    if mode != "absolute" and not isinstance(reference, dict):
        raise ValueError("coordinate.reference must be a mapping for relative coordinate modes.")
    return CoordinateSpec(label=axis_label, axis=axis, sign=sign, mode=mode, reference=reference)


def reference_for_frame(
    frame: TrajectoryFrame,
    spec: CoordinateSpec,
    context: SelectionContext,
) -> float:
    if spec.mode == "absolute":
        return 0.0
    reference_cfg = spec.reference or {}
    fixed = fixed_reference_value(reference_cfg)
    if fixed is not None:
        return fixed
    if spec.mode == "relative_to_reference":
        return element_mean_reference(frame, spec.axis, reference_cfg, context)
    if spec.mode == "relative_to_slab":
        return slab_surface_reference(frame, spec.axis, spec.sign, reference_cfg, context)
    raise ValueError(f"Unsupported coordinate mode: {spec.mode}")


def coordinate_values(
    frame: TrajectoryFrame,
    atom_indices: np.ndarray,
    spec: CoordinateSpec,
    context: SelectionContext,
) -> np.ndarray:
    reference = reference_for_frame(frame, spec, context)
    return spec.sign * (frame.positions[atom_indices, spec.axis] - reference)


def element_mean_reference(
    frame: TrajectoryFrame,
    axis: int,
    reference_cfg: dict[str, Any],
    context: SelectionContext,
) -> float:
    ref_type = str(reference_cfg.get("type", "element_mean"))
    if ref_type != "element_mean":
        raise ValueError("relative_to_reference requires reference.type: element_mean or fixed.")
    species = _species_list(reference_cfg, "reference.species")
    mask = element_mask(frame, set(species), context)
    if not np.any(mask):
        raise ValueError(f"Reference selection found no atoms: {species}")
    return float(np.mean(frame.positions[mask, axis]))


def slab_surface_reference(
    frame: TrajectoryFrame,
    axis: int,
    axis_sign: float,
    reference_cfg: dict[str, Any],
    context: SelectionContext,
) -> float:
    ref_type = str(reference_cfg.get("type", "slab_surface"))
    if ref_type not in {"slab_surface", "element_surface", "top_layer_mean"}:
        raise ValueError(
            "relative_to_slab requires reference.type: slab_surface, top_layer_mean, or fixed."
        )
    species = _species_list(reference_cfg, "reference.species")
    mask = element_mask(frame, set(species), context)
    values = frame.positions[mask, axis]
    if values.size == 0:
        raise ValueError(f"Slab reference selection found no atoms: {species}")

    surface = str(reference_cfg.get("surface", "auto")).lower()
    if surface == "auto":
        surface = "max" if axis_sign > 0 else "min"
    if surface == "max":
        if ref_type == "top_layer_mean":
            return _surface_layer_mean(values, surface="max", reference_cfg=reference_cfg)
        return float(np.max(values))
    if surface == "min":
        if ref_type == "top_layer_mean":
            return _surface_layer_mean(values, surface="min", reference_cfg=reference_cfg)
        return float(np.min(values))
    if surface == "mean":
        if ref_type == "top_layer_mean":
            raise ValueError("reference.type: top_layer_mean requires surface: max or min.")
        return float(np.mean(values))
    raise ValueError("reference.surface must be auto, max, min, or mean.")


def fixed_reference_value(reference_cfg: dict[str, Any]) -> float | None:
    ref_type = str(reference_cfg.get("type", "")).lower()
    if ref_type not in {"fixed", "fixed_value"}:
        return None
    if "value" not in reference_cfg:
        raise ValueError("reference.value is required when reference.type is fixed.")
    return float(reference_cfg["value"])


def _surface_layer_mean(
    values: np.ndarray,
    *,
    surface: str,
    reference_cfg: dict[str, Any],
) -> float:
    """Average selected slab atoms within one configured surface-layer thickness."""

    layer_width = float(reference_cfg.get("layer_width", reference_cfg.get("layer_width_A", 0.7)))
    if layer_width <= 0:
        raise ValueError("reference.layer_width must be positive for top_layer_mean.")
    edge = float(np.max(values) if surface == "max" else np.min(values))
    mask = values >= edge - layer_width if surface == "max" else values <= edge + layer_width
    return float(np.mean(values[mask]))


def _species_list(config: dict[str, Any], name: str) -> list[str]:
    value = config.get("species")
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty list.")
    return [str(item) for item in value]


# ---- from v0.6 _01_core/species.py

from typing import Any

import numpy as np

from waterint.prep.neighbors import classify_oxygen_by_h_count
from waterint.io.frame import TrajectoryFrame


OXYGEN_SPECIES_ORDER = ("O2-", "OH-", "H2O", "H3O+", "O_other")


def classify_assigned_oxygen_indices(
    oxygen_indices: np.ndarray,
    assigned_hydrogens: dict[int, int],
) -> dict[str, np.ndarray]:
    """Classify oxygen atoms from an already-resolved H-to-O assignment.

    SFG assigns each hydrogen once before constructing continuous bond segments.
    Reusing that assignment here keeps species channels consistent with the SFG
    correlation while avoiding a second neighbor search for every frame.
    """

    counts = {int(oxygen_index): 0 for oxygen_index in np.asarray(oxygen_indices, dtype=int)}
    for oxygen_index in assigned_hydrogens.values():
        oxygen_index = int(oxygen_index)
        if oxygen_index in counts:
            counts[oxygen_index] += 1

    result: dict[str, list[int]] = {label: [] for label in OXYGEN_SPECIES_ORDER}
    for oxygen_index, hydrogen_count in counts.items():
        if hydrogen_count == 0:
            label = "O2-"
        elif hydrogen_count == 1:
            label = "OH-"
        elif hydrogen_count == 2:
            label = "H2O"
        elif hydrogen_count >= 3:
            label = "H3O+"
        else:
            label = "O_other"
        result[label].append(oxygen_index)
    return {label: np.asarray(indices, dtype=int) for label, indices in result.items()}


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
