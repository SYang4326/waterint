from __future__ import annotations

from pathlib import Path

import numpy as np

from waterint._02_computation.conductivity import ConductivityResult


def write_conductivity_csv(path: Path, result: ConductivityResult, *, temperature_k: float, charge_e: float, fit_range_ps: tuple[float, float]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write(
            "temperature_K,charge_e,dimensions,carrier_count,volume_A3,fit_start_ps,fit_stop_ps,"
            "msd_slope_A2_per_ps,diffusion_A2_per_ps,diffusion_m2_per_s,conductivity_S_per_m,"
            "conductivity_S_per_cm,sheet_conductance_S\n"
        )
        sheet = "" if result.sheet_conductance_s is None else f"{result.sheet_conductance_s:.10g}"
        handle.write(
            f"{temperature_k:.10g},{charge_e:.10g},{result.dimensionality},{result.carrier_count},"
            f"{result.volume_a3:.10g},{fit_range_ps[0]:.10g},{fit_range_ps[1]:.10g},"
            f"{result.slope_a2_per_ps:.10g},{result.diffusion_a2_per_ps:.10g},"
            f"{result.diffusion_m2_per_s:.10g},{result.conductivity_s_per_m:.10g},"
            f"{result.conductivity_s_per_m / 100.0:.10g},{sheet}\n"
        )


def write_conductivity_msd_csv(path: Path, result: ConductivityResult) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("time_ps,msd_A2,linear_fit_A2,in_fit_range\n")
        for time, msd, fitted, used in zip(result.time_ps, result.msd_a2, result.fit_msd_a2, result.fit_mask):
            handle.write(f"{time:.10g},{msd:.10g},{fitted:.10g},{int(used)}\n")


def plot_conductivity_msd(path: Path, result: ConductivityResult, *, title: str, dpi: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.2, 4.2), constrained_layout=True)
    ax.plot(result.time_ps, result.msd_a2, color="#1f77b4", linewidth=1.8, label="MSD")
    ax.plot(result.time_ps[result.fit_mask], result.fit_msd_a2[result.fit_mask], color="#d62728", linewidth=2.0, label="linear fit")
    ax.set_xlabel("Lag time (ps)")
    ax.set_ylabel(f"{result.dimensionality}D MSD (A$^2$)")
    ax.set_title(title)
    ax.grid(True, linewidth=0.5, alpha=0.35)
    ax.legend(frameon=False)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
