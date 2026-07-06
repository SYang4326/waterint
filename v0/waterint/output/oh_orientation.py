from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def value_label(normalization_cfg: Any, *, vector_mode: str = "oh_bond") -> str:
    sample_name = "O-H bond" if vector_mode == "oh_bond" else "molecular vector"
    if isinstance(normalization_cfg, dict) and str(normalization_cfg.get("type", "counts_per_frame")) == "number_density":
        return f"{sample_name} density (1/A^3/degree)"
    return f"{sample_name}s per frame"


def vector_label(vector_mode: str) -> str:
    if vector_mode == "oh_bond":
        return "O-H"
    if vector_mode == "dipole":
        return "dipole"
    return "O-H bisector"


def plot_angle_label(vector_mode: str) -> str:
    if vector_mode == "oh_bond":
        return r"OH angle to +z ($^\circ$)"
    if vector_mode == "dipole":
        return r"dipole angle to +z ($^\circ$)"
    return r"O-H bisector angle to +z ($^\circ$)"


def write_hist_csv(
    path: Path,
    z_centers: np.ndarray,
    angle_centers: np.ndarray,
    hist: np.ndarray,
    axis_label: str,
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"{axis_label}_center_A,angle_center_deg,value\n")
        for i, z_value in enumerate(z_centers):
            for j, angle_value in enumerate(angle_centers):
                handle.write(f"{z_value:.10g},{angle_value:.10g},{hist[i, j]:.10g}\n")
