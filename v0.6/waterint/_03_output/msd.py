from __future__ import annotations

from pathlib import Path

import numpy as np


def write_msd_csv(path: Path, time_ps: np.ndarray, lag_frames: np.ndarray, msd_a2: np.ndarray, samples: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("lag_frames,time_ps,msd_A2,samples\n")
        for lag, time, value, count in zip(lag_frames, time_ps, msd_a2, samples):
            handle.write(f"{lag},{time:.10g},{value:.10g},{count}\n")


def plot_msd(path: Path, time_ps: np.ndarray, msd_a2: np.ndarray, *, title: str, dimensionality: str, dpi: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.0, 4.0), constrained_layout=True)
    ax.plot(time_ps, msd_a2, color="#1f77b4", linewidth=1.8)
    ax.set_xlabel("Lag time (ps)")
    ax.set_ylabel(f"{dimensionality.upper()} MSD (A^2)")
    ax.set_title(title)
    ax.grid(True, linewidth=0.5, alpha=0.35)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
