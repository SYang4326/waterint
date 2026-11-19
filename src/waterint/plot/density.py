from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def density_ylabel(normalization_cfg: Any) -> str:
    if isinstance(normalization_cfg, dict) and normalization_cfg.get("type") == "counts_per_frame":
        return "counts per frame"
    if isinstance(normalization_cfg, dict):
        norm_type = str(normalization_cfg.get("type", "number_density"))
        unit = str(normalization_cfg.get("unit", "")).lower()
        if norm_type == "mass_density" or unit in {"g_cm3", "g/cm3", "g/cm^3"}:
            return "mass density (g/cm^3)"
    return "number density (1/A^3)"


def write_density_csv(
    path: Path,
    bin_centers: np.ndarray,
    profiles: dict[str, dict[str, np.ndarray]],
    axis_name: str,
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        columns = [f"{axis_name}_center_A"]
        for label in profiles:
            columns.extend([f"{label}_counts_per_frame", f"{label}_density"])
        handle.write(",".join(columns) + "\n")
        for i, x in enumerate(bin_centers):
            row = [f"{x:.10g}"]
            for profile in profiles.values():
                row.append(f"{profile['counts_per_frame'][i]:.10g}")
                row.append(f"{profile['density'][i]:.10g}")
            handle.write(",".join(row) + "\n")


def plot_density_profile(
    path: str | Path,
    x: np.ndarray,
    y: np.ndarray | dict[str, dict[str, np.ndarray]],
    xlabel: str,
    ylabel: str,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.0, 4.0), constrained_layout=True)
    if isinstance(y, dict):
        for label, profile in y.items():
            ax.plot(x, profile["density"], lw=1.8, label=label)
        ax.legend(frameon=False)
    else:
        ax.plot(x, y, color="#2563eb", lw=1.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, lw=0.5, alpha=0.35)
    fig.savefig(path, dpi=220)
    plt.close(fig)
