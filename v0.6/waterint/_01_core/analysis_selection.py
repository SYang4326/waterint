from __future__ import annotations

import json
from typing import Any

import numpy as np

from waterint._00_io.common import TrajectoryFrame
from waterint._01_core.selection import SelectionContext, element_indices
from waterint._01_core.species import oxygen_species_indices


def analysis_indices(
    frame: TrajectoryFrame,
    selector: dict[str, Any],
    context: SelectionContext,
    *,
    defaults: dict[str, Any] | None = None,
    species_cache: dict[str, dict[str, np.ndarray]] | None = None,
) -> np.ndarray:
    """Return atom indices for element/type/oxygen-species analysis selectors."""

    if not isinstance(selector, dict):
        raise ValueError("An analysis selector must be a mapping.")
    merged = dict(defaults or {})
    merged.update(selector)

    if "oxygen_species" in merged:
        grouped = _oxygen_species_groups(frame, merged, context, species_cache)
        requested = merged["oxygen_species"]
        labels = list(grouped) if requested == "all" else [str(value) for value in requested]
        parts = [grouped[label] for label in labels if label in grouped]
        return np.sort(np.concatenate(parts)) if parts else np.empty(0, dtype=int)

    raw_types = merged.get("types")
    if raw_types is None and "species" in merged and _numeric_list(merged["species"]):
        raw_types = merged["species"]
    if raw_types is not None:
        if frame.types is None:
            raise ValueError("A type-based selector requires trajectory atom types.")
        if not isinstance(raw_types, list) or not raw_types:
            raise ValueError("selector.types must be a non-empty list.")
        return np.where(np.isin(frame.types, [int(value) for value in raw_types]))[0]

    elements = merged.get("elements", merged.get("species"))
    if not isinstance(elements, list) or not elements:
        raise ValueError("A selector needs elements: [O], types: [1], or oxygen_species: [H2O].")
    return element_indices(frame, {str(value) for value in elements}, context)


def _numeric_list(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(isinstance(item, (int, float)) for item in value)


def _oxygen_species_groups(
    frame: TrajectoryFrame,
    selector: dict[str, Any],
    context: SelectionContext,
    species_cache: dict[str, dict[str, np.ndarray]] | None,
) -> dict[str, np.ndarray]:
    if species_cache is None:
        return oxygen_species_indices(frame, selector, context)
    classification_cfg = {
        key: value
        for key, value in selector.items()
        if key not in {"oxygen_species", "layer", "label"}
    }
    key = json.dumps(classification_cfg, sort_keys=True, separators=(",", ":"), default=str)
    if key not in species_cache:
        all_species_cfg = dict(classification_cfg)
        all_species_cfg["oxygen_species"] = "all"
        species_cache[key] = oxygen_species_indices(frame, all_species_cfg, context)
    return species_cache[key]
