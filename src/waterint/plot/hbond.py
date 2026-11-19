from __future__ import annotations

from pathlib import Path

import numpy as np


def write_hbond_csv(
    path: Path,
    counts: dict[str, dict[str, int]],
    fractions: dict[str, dict[str, float]],
    samples_total: dict[str, int],
    classes: dict[str, list[str]],
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("species,hbond_class,count,total_oxygen_samples,fraction\n")
        for species in counts:
            for class_label in classes[species]:
                handle.write(
                    f"{species},{class_label},{counts[species][class_label]},"
                    f"{samples_total[species]},{fractions[species][class_label]:.10g}\n"
                )


def write_raw_hbond_csv(
    path: Path,
    raw_counts: dict[str, dict[str, int]],
    grouped_counts: dict[str, dict[str, int]],
    samples_total: dict[str, int],
    classes: dict[str, list[str]],
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("species,raw_hbond_class,grouped_hbond_class,count,total_oxygen_samples,fraction\n")
        for species in raw_counts:
            for raw_label, count in sorted(raw_counts[species].items(), key=lambda item: (-item[1], item[0])):
                grouped_label = raw_label if raw_label in classes[species] else "other"
                if grouped_label not in grouped_counts[species]:
                    grouped_label = ""
                fraction = count / samples_total[species] if samples_total[species] else 0.0
                handle.write(
                    f"{species},{raw_label},{grouped_label},{count},"
                    f"{samples_total[species]},{fraction:.10g}\n"
                )


def plot_hbond_fractions(
    *,
    path: str | Path,
    fractions: dict[str, dict[str, float]],
    classes: dict[str, list[str]],
    title: str = "",
    ylabel: str = "Fraction",
    dpi: int = 220,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    species_labels = list(fractions)
    if not species_labels:
        raise ValueError("No H-bond fractions to plot.")

    palette = {
        "OH-": "#1f9eea",
        "H2O": "#e50055",
        "H3O+": "#0aa35f",
        "O2-": "#7a7a7a",
        "O_other": "#8b5fbf",
    }
    hatches = ["", "///", "\\\\\\", "...", "xx"]
    group_gap = 0.85
    bar_width = 0.78
    x_positions: list[float] = []
    heights: list[float] = []
    colors: list[str] = []
    labels: list[str] = []
    hatch_values: list[str] = []
    group_centers: list[float] = []

    cursor = 0.0
    for species_index, species in enumerate(species_labels):
        start = cursor
        for class_label in classes[species]:
            x_positions.append(cursor)
            heights.append(float(fractions[species].get(class_label, 0.0)))
            colors.append(palette.get(species, "#4c78a8"))
            labels.append(class_label)
            hatch_values.append(hatches[species_index % len(hatches)])
            cursor += 1.0
        group_centers.append(0.5 * (start + cursor - 1.0))
        cursor += group_gap

    fig_width = max(7.0, 0.54 * len(x_positions) + 1.8)
    fig, ax = plt.subplots(figsize=(fig_width, 4.4), constrained_layout=True)
    for x, height, color, hatch in zip(x_positions, heights, colors, hatch_values):
        ax.bar(
            x,
            height,
            width=bar_width,
            color=color,
            edgecolor="#222222",
            linewidth=0.8,
            hatch=hatch,
            alpha=0.92,
        )

    ax.set_ylabel(ylabel)
    ax.set_ylim(0.0, max(1.0, float(np.ceil((max(heights) + 0.05) * 10.0) / 10.0)))
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=90)
    ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.35)
    ax.grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.16)
    ax.tick_params(direction="in", top=True, right=True)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    for center, species in zip(group_centers, species_labels):
        ax.text(
            center,
            -0.18,
            species,
            ha="center",
            va="top",
            transform=ax.get_xaxis_transform(),
            fontsize=10,
        )

    handles = [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor=palette.get(species, "#4c78a8"),
            edgecolor="#222222",
            hatch=hatches[index % len(hatches)],
            alpha=0.92,
        )
        for index, species in enumerate(species_labels)
    ]
    ax.legend(handles, species_labels, loc="upper right", frameon=True, framealpha=0.95)
    if title:
        ax.set_title(title)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
