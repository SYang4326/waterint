from __future__ import annotations

from pathlib import Path
import re

import numpy as np

from waterint.sfg.processing import load_ft


PROJECT_BIN_COLORS = {
    "0_1d5": "#FF6EC7",
    "1d5_2d8": "#4DA3FF",
    "2d8_4d0": "#B388FF",
    "4d0_5d5": "#FFC857",
    "4d0_30": "#FF5A36",
    "5d5_30": "#FF5A36",
    "all": "#000000",
}


def plot_spectrum(
    *,
    path: str | Path,
    freq_cm: np.ndarray,
    signal: np.ndarray,
    xmin: float = 0.0,
    xmax: float = 4500.0,
    flip: bool = True,
    title: str = "",
    frequency_label: str = "cm^-1",
    dpi: int = 220,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mask = (freq_cm >= xmin) & (freq_cm <= xmax)
    if not np.any(mask):
        mask = np.ones_like(freq_cm, dtype=bool)
    y = -signal if flip else signal

    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    ax.plot(freq_cm[mask], y[mask], linewidth=1.5, color="black")
    ax.axhline(0.0, linewidth=0.9, color="#777777")
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel(_frequency_axis_label(frequency_label))
    ax.set_ylabel("FT(CF) (a.u.)")
    if title:
        ax.set_title(title)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_overlay(
    *,
    path: str | Path,
    spectra: dict[str, Path],
    xmin: float = 2500.0,
    xmax: float = 4200.0,
    flip: bool = True,
    top_scale: float | None = 0.96,
    title: str = "",
    palette: str = "project",
    frequency_label: str = "cm^-1",
    dpi: int = 220,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    plotted = 0
    for label, spectrum_path in spectra.items():
        if not spectrum_path.exists():
            continue
        freq, signal = load_ft(spectrum_path)
        mask = (freq >= xmin) & (freq <= xmax)
        y = -signal if flip else signal
        color = _color_for_bin(label) if palette == "project" else None
        ax.plot(
            freq[mask],
            y[mask],
            label=format_bin_label(label),
            color=color,
            linewidth=2.4 if _normalize_bin_key(label) == "all" else 2.0,
        )
        plotted += 1
    if plotted == 0:
        raise ValueError("No SFG spectra were available to plot.")

    ax.axhline(0.0, color="#666666", linewidth=1.0)
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel(_frequency_axis_label(frequency_label))
    ax.set_ylabel("FT(CF) (a.u.)")
    if top_scale is not None:
        if top_scale == 0:
            raise ValueError("top_scale must be non-zero.")
        top = ax.secondary_xaxis("top", functions=(lambda x: x * top_scale, lambda x: x / top_scale))
        top.set_xlabel(f"rescaled {_frequency_axis_label(frequency_label)} ({top_scale:g}x)")
    ax.set_title(title or "SFG spectra from different bins")
    ax.legend(fontsize=8)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def format_bin_label(label: str) -> str:
    if label.lower() == "all":
        return "all"
    if "_" not in label:
        return label
    left, right = label.split("_", 1)

    def to_num(text: str) -> str:
        text = text.replace("d", ".")
        if text.startswith("m") and len(text) > 1 and text[1].isdigit():
            return "-" + text[1:]
        return text

    left = to_num(left)
    right = to_num(right)
    if re.fullmatch(r"-?\d+(?:\.\d+)?", left) and re.fullmatch(r"-?\d+(?:\.\d+)?", right):
        return f"z = {left}-{right}"
    return label


def _frequency_axis_label(unit: str) -> str:
    if unit == "THz":
        return "frequency (THz)"
    return "wavenumber (cm$^{-1}$)"


def _normalize_bin_key(label: str) -> str:
    return label.strip().lower().replace(".", "d").replace(" ", "")


def _color_for_bin(label: str) -> str | None:
    return PROJECT_BIN_COLORS.get(_normalize_bin_key(label))
