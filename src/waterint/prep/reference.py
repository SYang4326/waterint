"""Coordinates along an axis, optionally relative to a per-frame reference plane (v0.6 semantics).

Coordinate config (per analysis, or the profile default):
  axis: z | -z | x ...
  mode: absolute | relative_to_reference | relative_to_slab
  reference: {type: element_mean | slab_surface | top_layer_mean | fixed, species: [..], surface: auto|max|min|mean,
              layer_width: 0.7, value: ...}
``species`` may contain ``@solid`` (the frame's solid atoms, see prep.solid).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

AXIS_INDEX = {"x": 0, "y": 1, "z": 2}

MaskFn = Callable[[tuple[str, ...]], np.ndarray]


@dataclass(frozen=True)
class CoordinateSpec:
    label: str
    axis: int
    sign: float
    mode: str = "absolute"
    reference_key: tuple = ()          # hashable form of the reference mapping (cache key)

    @property
    def reference(self) -> dict[str, Any]:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.reference_key}


def parse_axis(value: Any) -> tuple[str, int, float]:
    axis_label = str(value).strip().lower()
    sign = -1.0 if axis_label.startswith("-") else 1.0
    bare = axis_label[1:] if axis_label.startswith("-") else axis_label
    if bare not in AXIS_INDEX:
        raise ValueError("axis must be one of x, y, z, -x, -y, -z.")
    return axis_label, AXIS_INDEX[bare], sign


def _freeze(mapping: dict[str, Any]) -> tuple:
    return tuple(sorted((k, tuple(v) if isinstance(v, list) else v) for k, v in mapping.items()))


def coordinate_spec_from_config(coord_cfg: dict[str, Any]) -> CoordinateSpec:
    axis_label, axis, sign = parse_axis(coord_cfg.get("axis", "z"))
    mode = str(coord_cfg.get("mode", "absolute"))
    if mode not in {"absolute", "relative_to_reference", "relative_to_slab"}:
        raise ValueError("coordinate.mode must be absolute, relative_to_reference, or relative_to_slab.")
    reference = coord_cfg.get("reference", {}) or {}
    if mode != "absolute" and not isinstance(reference, dict):
        raise ValueError("coordinate.reference must be a mapping for relative coordinate modes.")
    return CoordinateSpec(label=axis_label, axis=axis, sign=sign, mode=mode, reference_key=_freeze(reference))


def coordinate_spec_from_profile(profile: Any) -> CoordinateSpec:
    """Default coordinate of a profile: z_rel relative to the profile reference plane."""
    ref = profile.reference or {}
    axis_label = "xyz"[profile.normal_axis]
    mode = str(ref.get("mode", "none"))
    if mode == "none":
        return coordinate_spec_from_config({"axis": axis_label, "mode": "absolute"})
    if mode == "fixed":
        return coordinate_spec_from_config({"axis": axis_label, "mode": "relative_to_reference",
                                            "reference": {"type": "fixed", "value": float(ref.get("value_A", 0.0))}})
    if mode == "mean":
        return coordinate_spec_from_config({"axis": axis_label, "mode": "relative_to_reference",
                                            "reference": {"type": "element_mean", "species": list(ref.get("elements", []))}})
    surface = "max" if mode == "top_layer" else "min"
    return coordinate_spec_from_config({"axis": axis_label, "mode": "relative_to_slab", "reference": {
        "type": "top_layer_mean", "species": list(ref.get("elements", [])), "surface": surface,
        "layer_width": float(ref.get("layer_width_A", 0.7))}})


def reference_value(positions: np.ndarray, spec: CoordinateSpec, mask_fn: MaskFn) -> float:
    if spec.mode == "absolute":
        return 0.0
    ref = spec.reference
    if str(ref.get("type", "")).lower() in {"fixed", "fixed_value"}:
        if "value" not in ref:
            raise ValueError("reference.value is required when reference.type is fixed.")
        return float(ref["value"])
    species = _species(ref)
    if spec.mode == "relative_to_reference":
        if str(ref.get("type", "element_mean")) != "element_mean":
            raise ValueError("relative_to_reference requires reference.type: element_mean or fixed.")
        mask = mask_fn(species)
        if not np.any(mask):
            raise ValueError(f"Reference selection found no atoms: {list(species)}")
        return float(np.mean(positions[mask, spec.axis]))
    ref_type = str(ref.get("type", "slab_surface"))
    if ref_type not in {"slab_surface", "element_surface", "top_layer_mean"}:
        raise ValueError("relative_to_slab requires reference.type: slab_surface, top_layer_mean, or fixed.")
    values = positions[mask_fn(species), spec.axis]
    if values.size == 0:
        raise ValueError(f"Slab reference selection found no atoms: {list(species)}")
    surface = str(ref.get("surface", "auto")).lower()
    if surface == "auto":
        surface = "max" if spec.sign > 0 else "min"
    if surface in {"max", "min"}:
        if ref_type == "top_layer_mean":
            width = float(ref.get("layer_width", ref.get("layer_width_A", 0.7)))
            if width <= 0:
                raise ValueError("reference.layer_width must be positive for top_layer_mean.")
            edge = float(np.max(values) if surface == "max" else np.min(values))
            sel = values >= edge - width if surface == "max" else values <= edge + width
            return float(np.mean(values[sel]))
        return float(np.max(values) if surface == "max" else np.min(values))
    if surface == "mean":
        if ref_type == "top_layer_mean":
            raise ValueError("reference.type: top_layer_mean requires surface: max or min.")
        return float(np.mean(values))
    raise ValueError("reference.surface must be auto, max, min, or mean.")


def _species(ref: dict[str, Any]) -> tuple[str, ...]:
    value = ref.get("species")
    if not isinstance(value, list) or not value:
        raise ValueError("reference.species must be a non-empty list.")
    return tuple(str(v) for v in value)
