from __future__ import annotations

from pathlib import Path

import numpy as np


def plot_carrier_msd(path: Path, lag_ps: np.ndarray, msd_A2: np.ndarray, support: np.ndarray, classes) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    t = lag_ps[1:]
    for c, name in enumerate(classes):
        ax[0].loglog(t, msd_A2[c, 1:], label=name)
        ax[1].semilogx(t, support[c, 1:], label=name)
    ax[0].set_xlabel("lag (ps)"); ax[0].set_ylabel("strict OH$^-$ site MSD$_{xy}$ (Å$^2$)"); ax[0].legend(frameon=False)
    ax[1].set_xlabel("lag (ps)"); ax[1].set_ylabel("identity support"); ax[1].set_ylim(0, 1.02)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
