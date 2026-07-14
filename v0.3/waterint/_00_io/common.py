from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrajectoryFrame:
    index: int
    comment: str
    symbols: list[str]
    positions: np.ndarray
    cell: tuple[float, float, float] | None = None
    step: int | None = None
    types: np.ndarray | None = None
    velocities: np.ndarray | None = None
