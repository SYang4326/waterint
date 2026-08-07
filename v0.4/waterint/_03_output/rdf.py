from __future__ import annotations

from pathlib import Path

import numpy as np

from waterint._02_computation.rdf import RdfPairResult


def write_rdf_csv(path: Path, result: RdfPairResult) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("r_A,g_r,pair_count,expected_pair_count\n")
        for radius, g_r, count, expected in zip(result.r_centers, result.g_r, result.counts, result.expected_counts):
            handle.write(f"{radius:.10g},{g_r:.10g},{count:.10g},{expected:.10g}\n")


def plot_rdf(path: Path, results: dict[str, RdfPairResult], *, title: str, dpi: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.2, 4.1), constrained_layout=True)
    for name, result in results.items():
        ax.plot(result.r_centers, result.g_r, linewidth=1.8, label=name)
    ax.set_xlabel("r (A)")
    ax.set_ylabel("g(r)")
    ax.set_title(title)
    ax.grid(True, linewidth=0.5, alpha=0.35)
    ax.legend(frameon=False)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
