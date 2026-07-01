from __future__ import annotations

from pathlib import Path

import numpy as np


def plot_angle_z_histogram(
    *,
    path: str | Path,
    z_centers: np.ndarray,
    angle_centers: np.ndarray,
    hist: np.ndarray,
    title: str,
    z_label: str,
    value_label: str,
    log: bool = False,
    cmap: str = "turbo",
    style: str = "contour",
    orientation: str = "angle_z",
    invert_angle_axis: bool = False,
    figure_preset: str = "default",
    colorbar_mode: str = "auto",
    colormap_style: str = "turbo",
    yaxis_side: str = "left",
    show_y_label: bool = True,
    display_z_max: float | None = None,
    mask_threshold: float | None = None,
    smooth_sigma: float = 0.8,
    log_vmin: float | None = None,
    log_vmax: float | None = None,
    dpi: int = 220,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    import matplotlib.ticker as mticker
    from matplotlib.ticker import LogFormatterMathtext
    from matplotlib.ticker import MaxNLocator

    if figure_preset == "sharedz":
        _plot_sharedz_angle_z_histogram(
            path=path,
            z_centers=z_centers,
            angle_centers=angle_centers,
            hist=hist,
            z_label=z_label,
            log=log,
            style=style,
            cmap_name=cmap,
            colormap_style=colormap_style,
            colorbar_mode=colorbar_mode,
            yaxis_side=yaxis_side,
            show_y_label=show_y_label,
            display_z_max=display_z_max,
            mask_threshold=mask_threshold,
            smooth_sigma=smooth_sigma,
            log_vmin=log_vmin,
            log_vmax=log_vmax,
            dpi=dpi,
        )
        return

    if orientation not in {"angle_z", "z_angle"}:
        raise ValueError("angle-z plot orientation must be angle_z or z_angle.")
    if orientation == "z_angle":
        fig, ax = plt.subplots(figsize=(6.0, 5.0), constrained_layout=True)
    else:
        fig, ax = plt.subplots(figsize=(3.05, 4.6), constrained_layout=True)
    plot_data = np.asarray(hist, dtype=float)
    if smooth_sigma > 0:
        try:
            from scipy.ndimage import gaussian_filter

            plot_data = gaussian_filter(plot_data, sigma=float(smooth_sigma))
        except ImportError:
            pass

    if log:
        positive = plot_data[plot_data > 0]
        if positive.size:
            vmin = max(float(positive.min()), 1e-12) if log_vmin is None else max(float(log_vmin), 1e-12)
            vmax = float(positive.max()) if log_vmax is None else float(log_vmax)
            norm = LogNorm(vmin=vmin, vmax=vmax)
            plot_data = np.where(plot_data > 0, plot_data, np.nan)
        else:
            vmin = 1e-12 if log_vmin is None else max(float(log_vmin), 1e-12)
            norm = LogNorm(vmin=vmin, vmax=vmin * 10)
            plot_data = np.full_like(plot_data, vmin)
    else:
        norm = None

    angle_grid, z_grid = np.meshgrid(angle_centers, z_centers)
    if style == "imshow":
        if orientation == "z_angle":
            image_data = plot_data.T
            extent = [
                float(z_centers.min()),
                float(z_centers.max()),
                float(angle_centers.min()),
                float(angle_centers.max()),
            ]
        else:
            image_data = plot_data
            extent = [
                float(angle_centers.min()),
                float(angle_centers.max()),
                float(z_centers.min()),
                float(z_centers.max()),
            ]
        artist = ax.imshow(
            image_data,
            origin="lower",
            aspect="auto",
            extent=extent,
            interpolation="nearest",
            cmap=cmap,
            norm=norm,
        )
    elif style == "contour":
        levels = _levels(plot_data, log=log, norm=norm)
        if orientation == "z_angle":
            z_grid_for_plot, angle_grid_for_plot = np.meshgrid(z_centers, angle_centers)
            artist = ax.contourf(
                z_grid_for_plot,
                angle_grid_for_plot,
                plot_data.T,
                levels=levels,
                cmap=cmap,
                norm=norm,
                extend="max",
            )
            ax.contour(
                z_grid_for_plot,
                angle_grid_for_plot,
                plot_data.T,
                levels=levels,
                colors="k",
                linewidths=0.15,
                alpha=0.18,
                norm=norm,
            )
        else:
            artist = ax.contourf(angle_grid, z_grid, plot_data, levels=levels, cmap=cmap, norm=norm, extend="max")
            ax.contour(angle_grid, z_grid, plot_data, levels=levels, colors="k", linewidths=0.15, alpha=0.18, norm=norm)
    else:
        raise ValueError("angle-z plot style must be contour or imshow.")

    if orientation == "z_angle":
        cbar = fig.colorbar(artist, ax=ax)
    else:
        cbar = fig.colorbar(
            artist,
            ax=ax,
            orientation="horizontal",
            location="top",
            pad=0.08,
            fraction=0.08,
        )
    if log and norm is not None:
        ticks = _log_ticks(norm.vmin, norm.vmax)
        if ticks.size:
            cbar.set_ticks(ticks)
            cbar.formatter = LogFormatterMathtext(base=10)
            cbar.update_ticks()
    cbar.set_label(value_label, fontsize=9, labelpad=3)
    cbar.ax.tick_params(labelsize=8, pad=1)
    if orientation == "angle_z":
        cbar.ax.xaxis.set_ticks_position("top")
        cbar.ax.xaxis.set_label_position("top")
        ax.set_xlabel("OH angle to +z (degrees)")
        ax.set_ylabel(z_label.replace("coordinate", "coordinate"))
        ax.set_xlim(float(angle_centers.min()), float(angle_centers.max()))
        ax.set_ylim(float(z_centers.min()), float(z_centers.max()))
        ax.set_xticks([0, 60, 120, 180])
        ax.yaxis.set_major_locator(MaxNLocator(6))
    else:
        ax.set_xlabel(z_label.replace("coordinate", "coordinate"))
        ax.set_ylabel("OH angle to +z (degrees)")
        ax.set_xlim(float(z_centers.min()), float(z_centers.max()))
        ax.set_ylim(float(angle_centers.min()), float(angle_centers.max()))
        if invert_angle_axis:
            ax.invert_yaxis()
        ax.xaxis.set_major_locator(MaxNLocator(7))
        ax.set_yticks([0, 30, 60, 90, 120, 150, 180])
    if title:
        ax.set_title(title, fontsize=10, pad=4)
    ax.grid(True, color="#cfcfcf", linestyle="--", linewidth=0.8, alpha=0.65)
    ax.tick_params(direction="in", top=True, right=True, width=1.2, length=4)
    for spine in ax.spines.values():
        spine.set_linewidth(1.4)
        spine.set_color("black")
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _levels(data: np.ndarray, *, log: bool, norm: LogNorm | None) -> np.ndarray | int:
    positive = data[np.isfinite(data) & (data > 0)]
    if positive.size == 0:
        return 12
    if log and norm is not None:
        return np.logspace(np.log10(norm.vmin), np.log10(norm.vmax), 18)
    return np.linspace(float(np.nanmin(data)), float(np.nanmax(data)), 18)


def _log_ticks(vmin: float, vmax: float) -> np.ndarray:
    candidates = np.asarray([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0], dtype=float)
    ticks = candidates[(candidates >= vmin) & (candidates <= vmax)]
    if ticks.size > 3:
        ticks = ticks[::2]
    return ticks


def _plot_sharedz_angle_z_histogram(
    *,
    path: str | Path,
    z_centers: np.ndarray,
    angle_centers: np.ndarray,
    hist: np.ndarray,
    z_label: str,
    log: bool,
    style: str,
    cmap_name: str,
    colormap_style: str,
    colorbar_mode: str,
    yaxis_side: str,
    show_y_label: bool,
    display_z_max: float | None,
    mask_threshold: float | None,
    smooth_sigma: float,
    log_vmin: float | None,
    log_vmax: float | None,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    from matplotlib.colors import LogNorm

    old_rc = plt.rcParams.copy()
    plt.rcParams.update({
        "font.size": 15,
        "axes.labelsize": 15,
        "axes.titlesize": 21,
        "axes.grid": True,
        "axes.linewidth": 1.25,
        "xtick.top": True,
        "ytick.right": True,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.width": 1.25,
        "ytick.major.width": 1.25,
        "xtick.minor.width": 1.25,
        "ytick.minor.width": 1.25,
        "xtick.minor.visible": False,
        "ytick.minor.visible": False,
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "grid.linestyle": "dashed",
        "grid.linewidth": 1,
        "grid.alpha": 0.3,
        "legend.fontsize": 12,
        "mathtext.default": "regular",
    })

    try:
        plot_data = np.asarray(hist, dtype=float)
        if smooth_sigma > 0:
            try:
                from scipy.ndimage import gaussian_filter

                plot_data = gaussian_filter(plot_data, sigma=float(smooth_sigma))
            except ImportError:
                pass

        positive = plot_data[plot_data > 0]
        if positive.size:
            vmin = max(float(positive.min()), 1e-5) if log_vmin is None else max(float(log_vmin), 1e-12)
            vmax = float(positive.max()) if log_vmax is None else float(log_vmax)
        else:
            vmin = 1e-5 if log_vmin is None else max(float(log_vmin), 1e-12)
            vmax = vmin * 10 if log_vmax is None else float(log_vmax)
        norm = LogNorm(vmin=vmin, vmax=vmax) if log else None
        threshold = vmin if mask_threshold is None else float(mask_threshold)
        if log:
            plot_data = np.ma.masked_less_equal(plot_data, threshold)
        else:
            plot_data = np.ma.masked_invalid(plot_data)

        cmap = _sharedz_cmap(colormap_style, cmap_name)
        cmap.set_bad(color="white", alpha=1.0)
        dz = float(np.median(np.abs(np.diff(z_centers)))) if z_centers.size > 1 else 1.0
        da = float(np.median(np.abs(np.diff(angle_centers)))) if angle_centers.size > 1 else 1.0
        z_min = float(z_centers.min() - 0.5 * dz)
        z_max = float(display_z_max if display_z_max is not None else z_centers.max() + 0.5 * dz)
        angle_min = float(angle_centers.min() - 0.5 * da)
        angle_max = float(angle_centers.max() + 0.5 * da)

        fig_w_px, fig_h_px = 1050, 1706
        axis_left = 80 / fig_w_px
        axis_bottom = 201 / fig_h_px
        axis_width = 600 / fig_w_px
        axis_height = 1478 / fig_h_px
        fig = plt.figure(figsize=(fig_w_px / dpi, fig_h_px / dpi), dpi=dpi)
        ax = fig.add_axes([axis_left, axis_bottom, axis_width, axis_height])

        if style == "contour":
            angle_grid, z_grid = np.meshgrid(angle_centers, z_centers)
            levels = np.geomspace(vmin, vmax, 18) if log else np.linspace(plot_data.min(), plot_data.max(), 18)
            artist = ax.contourf(
                angle_grid,
                z_grid,
                plot_data,
                levels=levels,
                cmap=cmap,
                norm=norm,
                extend="max",
            )
            ax.contour(
                angle_grid,
                z_grid,
                plot_data,
                levels=levels,
                colors="#444444",
                linewidths=0.35,
                alpha=0.35,
                norm=norm,
            )
        else:
            artist = ax.imshow(
                plot_data + (1e-12 if log else 0.0),
                origin="lower",
                aspect="auto",
                extent=[angle_min, angle_max, z_min, float(z_centers.max() + 0.5 * dz)],
                interpolation="nearest",
                cmap=cmap,
                norm=norm,
            )

        ax.set_xlim(0, 180)
        ax.set_ylim(z_min, z_max)
        ax.set_xlabel(r"OH angle to +z ($^\circ$)", fontsize=12, labelpad=10)
        if show_y_label:
            ax.set_ylabel(z_label, labelpad=10)
        else:
            ax.set_ylabel("")
        if yaxis_side == "right":
            ax.yaxis.set_label_position("right")
            ax.yaxis.tick_right()
            ax.tick_params(axis="y", left=False, labelleft=False, right=True, labelright=True, pad=3, labelsize=15)
        else:
            ax.yaxis.set_label_position("left")
            ax.yaxis.tick_left()
            ax.tick_params(axis="y", left=True, labelleft=True, right=False, labelright=False, pad=3, labelsize=15)
        ax.xaxis.set_major_locator(mticker.FixedLocator([0, 60, 120, 180]))
        ax.xaxis.set_minor_locator(mticker.NullLocator())
        ax.tick_params(axis="x", labelsize=13, pad=3)
        if z_max <= 6.6:
            y_ticks = [0, 2, 4, 6]
        else:
            y_ticks = [0, 2, 4, 6, 8]
        ax.yaxis.set_major_locator(mticker.FixedLocator(y_ticks))
        ax.yaxis.set_minor_locator(mticker.NullLocator())
        ax.grid(True, linestyle="--", linewidth=1, alpha=0.3)

        if colorbar_mode == "auto":
            colorbar_mode = "right"
        if colorbar_mode == "right":
            ticks = [t for t in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1] if vmin <= t <= vmax]
            if len(ticks) < 2:
                ticks = [vmin, vmax]
            cax = fig.add_axes([900 / fig_w_px, axis_bottom + 0.24 * axis_height, 42 / fig_w_px, 0.48 * axis_height])
            cb = fig.colorbar(artist, cax=cax, orientation="vertical", ticks=ticks)
            cb.ax.yaxis.set_major_formatter(mticker.LogFormatterMathtext(base=10))
            cb.ax.yaxis.set_minor_locator(mticker.NullLocator())
            cb.ax.tick_params(labelsize=11, length=3, pad=2)
            cb.outline.set_linewidth(1.0)
            cb.ax.set_title("P", fontsize=11, pad=3)
        elif colorbar_mode == "none":
            pass
        else:
            raise ValueError("sharedz colorbar_mode must be right, none, or auto.")

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
    finally:
        plt.rcParams.update(old_rc)


def _sharedz_cmap(colormap_style: str, cmap_name: str):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    if colormap_style == "convention":
        colors = [
            (0.2549019607843137, 0.17254901960784313, 0.7411764705882353),
            (0.1803921568627451, 0.7333333333333333, 0.7529411764705882),
            (0.9607843137254902, 0.9137254901960784, 0.13725490196078433),
        ]
        return LinearSegmentedColormap.from_list("custom_blue_to_bright_yellow", colors, N=256)
    return plt.get_cmap(cmap_name)
