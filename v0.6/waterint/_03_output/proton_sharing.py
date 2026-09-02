"""Output helpers for proton-sharing free-energy surfaces."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from waterint._02_computation.proton_sharing import ProtonSharingResult


def write_proton_sharing_csv(path: Path, result: ProtonSharingResult) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("delta_A,R_OO_A,free_energy_kJ_mol,probability,count\n")
        for delta_index, delta in enumerate(result.delta_centers_a):
            for oo_index, oo in enumerate(result.oo_centers_a):
                handle.write(f"{delta:.10g},{oo:.10g},{result.free_energy_kj_mol[delta_index, oo_index]:.10g},{result.probability[delta_index, oo_index]:.10g},{result.counts[delta_index, oo_index]}\n")


def write_shared_proton_csv(path: Path, result: ProtonSharingResult) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("s_A,rho_A,free_energy_kJ_mol,probability_density,count\n")
        for s_index, s in enumerate(result.shared_s_centers_a):
            for rho_index, rho in enumerate(result.shared_rho_centers_a):
                handle.write(f"{s:.10g},{rho:.10g},{result.shared_free_energy_kj_mol[s_index, rho_index]:.10g},{result.shared_probability_density[s_index, rho_index]:.10g},{result.shared_counts[s_index, rho_index]}\n")


def plot_proton_sharing(path: Path, result: ProtonSharingResult, *, title: str, dpi: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    _plot_surface(axes[0], result.oo_centers_a, result.delta_centers_a, result.free_energy_kj_mol, "R_OO (A)", "delta = r(O_d,H) - r(H,O_a) (A)", f"{title}: F(delta, R_OO)")
    _plot_surface(axes[1], result.shared_s_centers_a, result.shared_rho_centers_a, result.shared_free_energy_kj_mol.T, "s from O-O midpoint (A)", "rho from O-O axis (A)", f"{title}: F(s, rho | shared)")
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _plot_surface(axis, x: np.ndarray, y: np.ndarray, values: np.ndarray, xlabel: str, ylabel: str, title: str) -> None:
    finite = np.isfinite(values)
    masked = np.ma.masked_where(~finite, values)
    image = axis.pcolormesh(x, y, masked, shading="nearest", cmap="magma_r")
    axis.set(xlabel=xlabel, ylabel=ylabel, title=title)
    axis.figure.colorbar(image, ax=axis, label="F (kJ mol$^{-1}$)")
