from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import json
import math

import numpy as np

from waterint.angle_z.plotting import plot_angle_z_histogram
from waterint.chemistry import oxygen_hydrogen_pairs_by_species
from waterint.common import (
    element_indices,
    iter_frames,
    parse_axis,
    parse_cell,
    parse_range,
    reference_value,
    resolve_path,
    selection_context,
    slab_reference_value,
)
from waterint.config import require_mapping


OXYGEN_SPECIES_ORDER = ("OH-", "H2O", "H3O+")


@dataclass(frozen=True)
class AngleZResult:
    z_centers: np.ndarray
    angle_centers: np.ndarray
    histograms: dict[str, np.ndarray]
    frames: int
    bond_counts_total: dict[str, int]
    csv_paths: dict[str, Path]
    png_paths: dict[str, Path]
    metadata_path: Path


def run_angle_z(config: dict[str, Any]) -> AngleZResult:
    input_cfg = require_mapping(config, "input")
    system_cfg = require_mapping(config, "system")
    selection_cfg = require_mapping(config, "selection")
    coord_cfg = require_mapping(config, "coordinate")
    angle_cfg = require_mapping(config, "angle")
    output_cfg = require_mapping(config, "output")

    traj_path = resolve_path(config, input_cfg["trajectory"])
    configured_cell = parse_cell(system_cfg.get("cell", "auto"))
    axis_label, axis, axis_sign = parse_axis(coord_cfg.get("axis", "z"))
    z_min, z_max = parse_range(coord_cfg.get("range"), name="coordinate.range")
    z_bins = int(coord_cfg.get("bins", 200))
    if z_bins <= 0:
        raise ValueError("coordinate.bins must be > 0.")

    angle_min, angle_max = parse_range(angle_cfg.get("range", [0.0, 180.0]), name="angle.range")
    angle_bins = int(angle_cfg.get("bins", 180))
    if angle_bins <= 0:
        raise ValueError("angle.bins must be > 0.")

    species_labels = _species_labels(selection_cfg)
    z_edges = np.linspace(z_min, z_max, z_bins + 1)
    angle_edges = np.linspace(angle_min, angle_max, angle_bins + 1)
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])
    angle_centers = 0.5 * (angle_edges[:-1] + angle_edges[1:])
    counts_by_species = {label: np.zeros((z_bins, angle_bins), dtype=float) for label in species_labels}
    bond_counts_total = {label: 0 for label in species_labels}

    mode = str(coord_cfg.get("mode", "absolute"))
    reference_cfg = coord_cfg.get("reference", {})
    if mode not in {"absolute", "relative_to_reference", "relative_to_slab"}:
        raise ValueError("coordinate.mode must be absolute, relative_to_reference, or relative_to_slab.")
    if mode != "absolute" and not isinstance(reference_cfg, dict):
        raise ValueError("coordinate.reference must be a mapping for relative coordinate modes.")

    context = selection_context(input_cfg)
    cell = configured_cell
    frames = 0
    for frame in iter_frames(traj_path, input_cfg):
        if cell is None:
            if frame.cell is None:
                raise ValueError("system.cell is auto, but the trajectory did not provide cell information.")
            cell = frame.cell

        reference = 0.0
        if mode == "relative_to_reference":
            reference = reference_value(frame, axis, reference_cfg, context)
        elif mode == "relative_to_slab":
            reference = slab_reference_value(frame, axis, reference_cfg, axis_sign, context)

        pairs_by_species = oxygen_hydrogen_pairs_by_species(
            frame.symbols,
            frame.positions,
            oxygen_symbol=str(selection_cfg.get("oxygen_symbol", "O")),
            hydrogen_symbol=str(selection_cfg.get("hydrogen_symbol", "H")),
            oh_cutoff=float(selection_cfg.get("oh_cutoff", 1.25)),
            neighbor_method=str(selection_cfg.get("neighbor_method", "auto")),
            neighbor_workers=int(selection_cfg.get("neighbor_workers", 1)),
            oxygen_chunk_size=int(selection_cfg.get("oxygen_chunk_size", 2048)),
            oxygen_indices=element_indices(frame, {str(selection_cfg.get("oxygen_symbol", "O"))}, context),
            hydrogen_indices=element_indices(frame, {str(selection_cfg.get("hydrogen_symbol", "H"))}, context),
        )
        for label in species_labels:
            pairs = pairs_by_species[label]
            if pairs.size == 0:
                continue
            z_values, angle_values = _pair_z_and_angles(
                positions=frame.positions,
                pairs=pairs,
                axis=axis,
                axis_sign=axis_sign,
                reference=reference,
                angle_axis_sign=float(angle_cfg.get("axis_sign", 1.0)),
            )
            hist, _, _ = np.histogram2d(z_values, angle_values, bins=[z_edges, angle_edges])
            counts_by_species[label] += hist
            bond_counts_total[label] += int(pairs.shape[0])
        frames += 1

    if frames == 0:
        raise ValueError(f"No frames found in trajectory: {traj_path}")
    if cell is None:
        raise ValueError("No cell information was available. Set system.cell manually.")

    normalization_cfg = config.get("normalization", {})
    histograms = {
        label: _normalize_histogram(
            counts=counts,
            frames=frames,
            cell=cell,
            axis=axis,
            z_bin_width=(z_max - z_min) / z_bins,
            angle_bin_width=(angle_max - angle_min) / angle_bins,
            normalization_cfg=normalization_cfg,
        )
        for label, counts in counts_by_species.items()
    }

    outdir = resolve_path(config, output_cfg.get("directory", "output"))
    outdir.mkdir(parents=True, exist_ok=True)
    prefix = str(output_cfg.get("prefix", "angle_z"))
    plot_enabled = bool(output_cfg.get("plot", True))
    csv_paths: dict[str, Path] = {}
    png_paths: dict[str, Path] = {}
    for label, hist in histograms.items():
        safe_label = _safe_label(label)
        csv_path = outdir / f"{prefix}_{safe_label}.csv"
        csv_paths[label] = csv_path
        _write_hist_csv(csv_path, z_centers, angle_centers, hist, axis_label)
        if plot_enabled:
            png_path = outdir / f"{prefix}_{safe_label}.png"
            png_paths[label] = png_path
            plot_angle_z_histogram(
                path=png_path,
                z_centers=z_centers,
                angle_centers=angle_centers,
                hist=hist,
                title=f"{label}: {output_cfg.get('title', f'O-H angle vs {axis_label}')}",
                z_label=str(output_cfg.get("z_label", f"{axis_label} coordinate (Angstrom)")),
                value_label=_value_label(normalization_cfg),
                log=bool(output_cfg.get("log", False)),
                cmap=str(output_cfg.get("cmap", "turbo")),
                style=str(output_cfg.get("style", "contour")),
                orientation=str(output_cfg.get("orientation", "angle_z")),
                invert_angle_axis=bool(output_cfg.get("invert_angle_axis", False)),
                figure_preset=str(output_cfg.get("figure_preset", "default")),
                colorbar_mode=str(output_cfg.get("colorbar_mode", "auto")),
                colormap_style=str(output_cfg.get("colormap_style", "turbo")),
                yaxis_side=str(output_cfg.get("yaxis_side", "left")),
                show_y_label=bool(output_cfg.get("show_y_label", True)),
                display_z_max=_optional_float(output_cfg.get("display_z_max")),
                mask_threshold=_optional_float(output_cfg.get("mask_threshold")),
                colorbar_height=float(output_cfg.get("colorbar_height", 0.36)),
                colorbar_width_px=float(output_cfg.get("colorbar_width_px", 30.0)),
                colorbar_x_px=float(output_cfg.get("colorbar_x_px", 900.0)),
                colorbar_center=float(output_cfg.get("colorbar_center", 0.50)),
                colorbar_top_pad_px=float(output_cfg.get("colorbar_top_pad_px", 38.0)),
                colorbar_top_height_px=float(output_cfg.get("colorbar_top_height_px", 34.0)),
                colorbar_tick_size=float(output_cfg.get("colorbar_tick_size", 9.0)),
                colorbar_title_size=float(output_cfg.get("colorbar_title_size", 10.0)),
                smooth_sigma=float(output_cfg.get("smooth_sigma", 0.8)),
                log_vmin=_optional_float(output_cfg.get("log_vmin")),
                log_vmax=_optional_float(output_cfg.get("log_vmax")),
                dpi=int(output_cfg.get("dpi", 220)),
            )

    metadata_path = outdir / f"{prefix}_metadata.json"
    _write_metadata(
        metadata_path,
        config=config,
        traj_path=traj_path,
        axis=axis_label,
        species_labels=species_labels,
        frames=frames,
        bond_counts_total=bond_counts_total,
        csv_paths=csv_paths,
        png_paths=png_paths,
    )

    return AngleZResult(
        z_centers=z_centers,
        angle_centers=angle_centers,
        histograms=histograms,
        frames=frames,
        bond_counts_total=bond_counts_total,
        csv_paths=csv_paths,
        png_paths=png_paths,
        metadata_path=metadata_path,
    )


def _species_labels(selection_cfg: dict[str, Any]) -> list[str]:
    selected = selection_cfg.get("oxygen_species", list(OXYGEN_SPECIES_ORDER))
    if selected == "all":
        return list(OXYGEN_SPECIES_ORDER)
    if not isinstance(selected, list) or not selected:
        raise ValueError("selection.oxygen_species must be 'all' or a non-empty list.")
    labels = [str(item) for item in selected]
    unknown = [label for label in labels if label not in OXYGEN_SPECIES_ORDER]
    if unknown:
        raise ValueError(f"Unknown angle-z oxygen species labels: {unknown}")
    return labels


def _pair_z_and_angles(
    *,
    positions: np.ndarray,
    pairs: np.ndarray,
    axis: int,
    axis_sign: float,
    reference: float,
    angle_axis_sign: float,
) -> tuple[np.ndarray, np.ndarray]:
    oxygen_positions = positions[pairs[:, 0]]
    hydrogen_positions = positions[pairs[:, 1]]
    vectors = hydrogen_positions - oxygen_positions
    norms = np.linalg.norm(vectors, axis=1)
    valid = norms > 0
    vectors = vectors[valid]
    norms = norms[valid]
    oxygen_positions = oxygen_positions[valid]
    cos_theta = angle_axis_sign * vectors[:, axis] / norms
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angles = np.degrees(np.arccos(cos_theta))
    z_values = axis_sign * (oxygen_positions[:, axis] - reference)
    return z_values, angles


def _normalize_histogram(
    *,
    counts: np.ndarray,
    frames: int,
    cell: tuple[float, float, float],
    axis: int,
    z_bin_width: float,
    angle_bin_width: float,
    normalization_cfg: Any,
) -> np.ndarray:
    if normalization_cfg is None:
        normalization_cfg = {}
    if not isinstance(normalization_cfg, dict):
        raise ValueError("normalization must be a mapping.")
    norm_type = str(normalization_cfg.get("type", "counts_per_frame"))
    counts_per_frame = counts / frames
    if norm_type == "counts_per_frame":
        return counts_per_frame
    if norm_type != "number_density":
        raise ValueError("normalization.type must be counts_per_frame or number_density.")

    perpendicular_lengths = [cell[i] for i in range(3) if i != axis]
    volume_angle = perpendicular_lengths[0] * perpendicular_lengths[1] * z_bin_width * angle_bin_width
    if not math.isfinite(volume_angle) or volume_angle <= 0:
        raise ValueError("Computed z-angle normalization volume must be positive.")
    return counts_per_frame / volume_angle


def _value_label(normalization_cfg: Any) -> str:
    if isinstance(normalization_cfg, dict) and str(normalization_cfg.get("type", "counts_per_frame")) == "number_density":
        return "bond density (1/A^3/degree)"
    return "O-H bonds per frame"


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _write_hist_csv(
    path: Path,
    z_centers: np.ndarray,
    angle_centers: np.ndarray,
    hist: np.ndarray,
    axis_label: str,
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"{axis_label}_center_A,angle_center_deg,value\n")
        for i, z_value in enumerate(z_centers):
            for j, angle_value in enumerate(angle_centers):
                handle.write(f"{z_value:.10g},{angle_value:.10g},{hist[i, j]:.10g}\n")


def _write_metadata(
    path: Path,
    config: dict[str, Any],
    traj_path: Path,
    axis: str,
    species_labels: list[str],
    frames: int,
    bond_counts_total: dict[str, int],
    csv_paths: dict[str, Path],
    png_paths: dict[str, Path],
) -> None:
    public_config = {key: value for key, value in config.items() if not key.startswith("_")}
    metadata = {
        "analysis_name": "angle_z",
        "package": "waterint",
        "config_file": config.get("_config_path"),
        "trajectory": str(traj_path),
        "axis": axis,
        "species_labels": species_labels,
        "frames": frames,
        "bond_counts_total": bond_counts_total,
        "outputs": {
            "csv": {label: str(path) for label, path in csv_paths.items()},
            "png": {label: str(path) for label, path in png_paths.items()},
        },
        "config": public_config,
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")


def _safe_label(label: str) -> str:
    return label.lower().replace("+", "plus").replace("-", "minus").replace(" ", "_")
