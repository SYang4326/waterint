from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from waterint._00_io.common import TrajectoryFrame


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
