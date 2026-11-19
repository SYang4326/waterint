"""Atom selectors shared by analyses (rdf pairs, msd, ...).

    {elements: [O]} | {types: [2]} | {oxygen_species: [H2O, OH-]} | {species: [...]}   (+ optional)
    layer: {coordinate: {axis, mode, reference}, range: [lo, hi]}       # half-open [lo, hi)
    ownership keys (oh_cutoff, hydrogen_assignment, pbc, oxygen_symbol, hydrogen_symbol) override the
    profile ownership for species selection.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from waterint.prep.labelled import LabelledFrame
from waterint.prep.reference import coordinate_spec_from_config
from waterint.system.profile import OwnershipParams

OWNERSHIP_KEYS = {"oh_cutoff": "cutoff", "hydrogen_assignment": "mode", "pbc": "pbc",
                  "oxygen_symbol": "acceptors", "hydrogen_symbol": "protons"}


def selector_ownership(selector: dict[str, Any], base: OwnershipParams) -> OwnershipParams:
    return base.replace({OWNERSHIP_KEYS[k]: v for k, v in selector.items() if k in OWNERSHIP_KEYS})


def select(lf: LabelledFrame, selector: dict[str, Any], base: OwnershipParams) -> np.ndarray:
    if not isinstance(selector, dict):
        raise ValueError("An analysis selector must be a mapping.")
    if "oxygen_species" in selector:
        grouped = lf.species(selector_ownership(selector, base), use_solid=bool(selector.get("use_solid", True)))
        requested = selector["oxygen_species"]
        labels = list(grouped) if requested == "all" else [str(v) for v in requested]
        parts = [grouped[label] for label in labels if label in grouped]
        indices = np.sort(np.concatenate(parts)) if parts else np.empty(0, dtype=int)
    else:
        raw_types = selector.get("types")
        if raw_types is None and "species" in selector and _numeric(selector["species"]):
            raw_types = selector["species"]
        if raw_types is not None:
            if lf.frame.types is None:
                raise ValueError("A type-based selector requires trajectory atom types.")
            indices = np.nonzero(np.isin(lf.frame.types, [int(v) for v in raw_types]))[0]
        else:
            elements = selector.get("elements", selector.get("species"))
            if not isinstance(elements, list) or not elements:
                raise ValueError("A selector needs elements: [O], types: [1], or oxygen_species: [H2O].")
            indices = lf.indices(tuple(str(v) for v in elements))
    layer = selector.get("layer")
    if layer is None:
        return indices
    if not isinstance(layer, dict):
        raise ValueError("selector.layer must be a mapping.")
    spec = coordinate_spec_from_config(layer.get("coordinate", layer))
    low, high = parse_range(layer.get("range"), "selector.layer.range")
    values = lf.coordinates(indices, spec)
    return indices[(values >= low) & (values < high)]


def parse_range(value: Any, name: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must be [min, max].")
    lo, hi = float(value[0]), float(value[1])
    if not hi > lo:
        raise ValueError(f"{name} max must be larger than min.")
    return lo, hi


def _numeric(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(isinstance(v, (int, float)) for v in value)
