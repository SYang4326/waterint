from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from waterint.io.common import TrajectoryFrame
from waterint.core.selection import SelectionContext, element_mask


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
        raise ValueError("Only reference.type: element_mean is implemented.")
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
    if ref_type not in {"slab_surface", "element_surface"}:
        raise ValueError("relative_to_slab requires reference.type: slab_surface.")
    species = _species_list(reference_cfg, "reference.species")
    mask = element_mask(frame, set(species), context)
    values = frame.positions[mask, axis]
    if values.size == 0:
        raise ValueError(f"Slab reference selection found no atoms: {species}")

    surface = str(reference_cfg.get("surface", "auto")).lower()
    if surface == "auto":
        surface = "max" if axis_sign > 0 else "min"
    if surface == "max":
        return float(np.max(values))
    if surface == "min":
        return float(np.min(values))
    if surface == "mean":
        return float(np.mean(values))
    raise ValueError("reference.surface must be auto, max, min, or mean.")


def _species_list(config: dict[str, Any], name: str) -> list[str]:
    value = config.get("species")
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty list.")
    return [str(item) for item in value]
