"""CSV and heatmap output for simplified oxygen identity transitions."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np



def write_identity_transition_outputs(
    result: IdentityTransitionResult,
    *,
    directory: Path,
    prefix: str,
    title: str,
) -> tuple[Path, Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    csv_path = directory / f"{prefix}_matrix.csv"
    png_path = directory / f"{prefix}_matrix.png"
    pdf_path = directory / f"{prefix}_matrix.pdf"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["from_state", *result.states])
        for state, row in zip(result.states, result.counts):
            writer.writerow([state, *(int(value) for value in row)])

    fig, axis = plt.subplots(figsize=(6.4, 5.4), constrained_layout=True)
    image = axis.imshow(np.log10(result.counts + 1), cmap="YlOrRd", vmin=0)
    for row in range(result.counts.shape[0]):
        for column in range(result.counts.shape[1]):
            value = int(result.counts[row, column])
            if value:
                axis.text(column, row, str(value), ha="center", va="center", fontsize=9)
    axis.set(
        xticks=np.arange(len(result.states)),
        yticks=np.arange(len(result.states)),
        xticklabels=result.states,
        yticklabels=result.states,
        xlabel="identity after transition",
        ylabel="identity before transition",
        title=title,
    )
    colorbar = fig.colorbar(image, ax=axis)
    colorbar.set_label(r"$\log_{10}(N+1)$")
    fig.savefig(png_path, dpi=260)
    fig.savefig(pdf_path)
    plt.close(fig)
    return csv_path, png_path, pdf_path
