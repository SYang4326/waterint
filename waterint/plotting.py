from __future__ import annotations

from pathlib import Path

import numpy as np


def plot_density_profile(
    path: str | Path,
    x: np.ndarray,
    y: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.0, 4.0), constrained_layout=True)
    ax.plot(x, y, color="#2563eb", lw=1.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, lw=0.5, alpha=0.35)
    fig.savefig(path, dpi=220)
    plt.close(fig)
