from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from waterint.angle_z.plotting import plot_angle_z_histogram
from waterint.chemistry import oxygen_hydrogen_neighbors_by_species, oxygen_hydrogen_pairs_by_species
from waterint.config import require_mapping
from waterint.io.common import TrajectoryFrame
from waterint_v0.computation.oh_orientation.compute import (
    OhOrientationResult,
    OhOrientationState,
    accumulate_angle_samples,
    finalize_oh_orientation_state,
    neighbor_bisector_z_and_angles,
    new_oh_orientation_state,
    pair_z_and_angles,
    species_labels,
)
from waterint_v0.core.coordinates import CoordinateSpec, coordinate_spec_from_config, reference_for_frame
from waterint_v0.core.selection import SelectionContext, element_indices
from waterint_v0.output.oh_orientation import plot_angle_label, value_label, vector_label, write_hist_csv
from waterint_v0.output.metadata import write_metadata
from waterint_v0.workflows.common import parse_cell, parse_range, required_workflow_sections, resolve_path
from waterint_v0.workflows.framewise import run_framewise_analysis


@dataclass
class OhOrientationWorkflowState:
    oh_orientation: OhOrientationState
    selection_cfg: dict[str, Any]
    angle_cfg: dict[str, Any]
    context: SelectionContext
    coordinate: CoordinateSpec
    species_labels: list[str]
    vector_mode: str
    normalization_cfg: Any


def run_oh_orientation(config: dict[str, Any]) -> OhOrientationResult:
    input_cfg, system_cfg, output_cfg = required_workflow_sections(config)
    selection_cfg = require_mapping(config, "selection")
    coord_cfg = require_mapping(config, "coordinate")
    angle_cfg = require_mapping(config, "angle")

    traj_path = resolve_path(config, input_cfg["trajectory"])
    configured_cell = parse_cell(system_cfg.get("cell", "auto"))
    coordinate = coordinate_spec_from_config(coord_cfg)
    z_min, z_max = parse_range(coord_cfg.get("range"), name="coordinate.range")
    z_bins = int(coord_cfg.get("bins", 200))
    if z_bins <= 0:
        raise ValueError("coordinate.bins must be > 0.")
    angle_min, angle_max = parse_range(angle_cfg.get("range", [0.0, 180.0]), name="angle.range")
    angle_bins = int(angle_cfg.get("bins", 180))
    if angle_bins <= 0:
        raise ValueError("angle.bins must be > 0.")
    vector_mode = str(angle_cfg.get("vector_mode", "oh_bond")).lower()
    if vector_mode not in {"oh_bond", "oh_bisector", "dipole"}:
        raise ValueError("angle.vector_mode must be oh_bond, oh_bisector, or dipole.")

    labels = species_labels(selection_cfg)
    state = OhOrientationWorkflowState(
        oh_orientation=new_oh_orientation_state(
            labels,
            np.linspace(z_min, z_max, z_bins + 1),
            np.linspace(angle_min, angle_max, angle_bins + 1),
        ),
        selection_cfg=selection_cfg,
        angle_cfg=angle_cfg,
        context=SelectionContext.from_input_config(input_cfg),
        coordinate=coordinate,
        species_labels=labels,
        vector_mode=vector_mode,
        normalization_cfg=config.get("normalization", {}),
    )
    framewise = run_framewise_analysis(
        traj_path=traj_path,
        input_cfg=input_cfg,
        configured_cell=configured_cell,
        state=state,
        accumulate=_accumulate_oh_orientation_frame,
        finalize=_finalize_oh_orientation,
    )
    result = framewise.result

    outdir = resolve_path(config, output_cfg.get("directory", "output"))
    outdir.mkdir(parents=True, exist_ok=True)
    prefix = str(output_cfg.get("prefix", "angle_z"))
    plot_enabled = bool(output_cfg.get("plot", True))
    csv_paths: dict[str, Path] = {}
    png_paths: dict[str, Path] = {}
    for label, hist in result.histograms.items():
        safe_label = safe_label_for_path(label)
        csv_path = outdir / f"{prefix}_{safe_label}.csv"
        csv_paths[label] = csv_path
        write_hist_csv(csv_path, result.z_centers, result.angle_centers, hist, coordinate.label)
        if plot_enabled:
            png_path = outdir / f"{prefix}_{safe_label}.png"
            png_paths[label] = png_path
            plot_angle_z_histogram(
                path=png_path,
                z_centers=result.z_centers,
                angle_centers=result.angle_centers,
                hist=hist,
                title=f"{label}: {output_cfg.get('title', f'{vector_label(vector_mode)} angle vs {coordinate.label}')}",
                z_label=str(output_cfg.get("z_label", f"{coordinate.label} coordinate (Angstrom)")),
                value_label=value_label(config.get("normalization", {}), vector_mode=vector_mode),
                angle_label=str(output_cfg.get("angle_label", plot_angle_label(vector_mode))),
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
                display_z_max=optional_float(output_cfg.get("display_z_max")),
                mask_threshold=optional_float(output_cfg.get("mask_threshold")),
                colorbar_height=float(output_cfg.get("colorbar_height", 0.36)),
                colorbar_width_px=float(output_cfg.get("colorbar_width_px", 30.0)),
                colorbar_x_px=float(output_cfg.get("colorbar_x_px", 900.0)),
                colorbar_center=float(output_cfg.get("colorbar_center", 0.50)),
                colorbar_top_pad_px=float(output_cfg.get("colorbar_top_pad_px", 38.0)),
                colorbar_top_height_px=float(output_cfg.get("colorbar_top_height_px", 34.0)),
                colorbar_tick_size=float(output_cfg.get("colorbar_tick_size", 9.0)),
                colorbar_title_size=float(output_cfg.get("colorbar_title_size", 10.0)),
                hide_first_colorbar_tick_label=bool(output_cfg.get("hide_first_colorbar_tick_label", False)),
                smooth_sigma=float(output_cfg.get("smooth_sigma", 0.8)),
                log_vmin=optional_float(output_cfg.get("log_vmin")),
                log_vmax=optional_float(output_cfg.get("log_vmax")),
                dpi=int(output_cfg.get("dpi", 220)),
            )

    metadata_path = outdir / f"{prefix}_metadata.json"
    write_metadata(
        metadata_path,
        {
            "analysis_name": "oh_orientation",
            "package": "waterint_v0",
            "config_file": config.get("_config_path"),
            "trajectory": str(traj_path),
            "axis": coordinate.label,
            "species_labels": labels,
            "frames": result.frames,
            "bond_counts_total": result.bond_counts_total,
            "sample_counts_total": result.sample_counts_total,
            "outputs": {
                "csv": {label: str(path) for label, path in csv_paths.items()},
                "png": {label: str(path) for label, path in png_paths.items()},
            },
            "config": {key: value for key, value in config.items() if not key.startswith("_")},
        },
    )
    return replace(result, csv_paths=csv_paths, png_paths=png_paths, metadata_path=metadata_path)


def _accumulate_oh_orientation_frame(
    state: OhOrientationWorkflowState,
    frame: TrajectoryFrame,
    _cell: tuple[float, float, float],
) -> None:
    reference = reference_for_frame(frame, state.coordinate, state.context)
    oxygen_symbol = str(state.selection_cfg.get("oxygen_symbol", "O"))
    hydrogen_symbol = str(state.selection_cfg.get("hydrogen_symbol", "H"))
    oxygen_indices = element_indices(frame, {oxygen_symbol}, state.context)
    hydrogen_indices = element_indices(frame, {hydrogen_symbol}, state.context)
    if state.vector_mode == "oh_bond":
        pairs_by_species = oxygen_hydrogen_pairs_by_species(
            frame.symbols,
            frame.positions,
            oxygen_symbol=oxygen_symbol,
            hydrogen_symbol=hydrogen_symbol,
            oh_cutoff=float(state.selection_cfg.get("oh_cutoff", 1.25)),
            neighbor_method=str(state.selection_cfg.get("neighbor_method", "auto")),
            neighbor_workers=int(state.selection_cfg.get("neighbor_workers", 1)),
            oxygen_chunk_size=int(state.selection_cfg.get("oxygen_chunk_size", 2048)),
            oxygen_indices=oxygen_indices,
            hydrogen_indices=hydrogen_indices,
        )
        for label in state.species_labels:
            pairs = pairs_by_species[label]
            if pairs.size == 0:
                continue
            z_values, angle_values = pair_z_and_angles(
                positions=frame.positions,
                pairs=pairs,
                axis=state.coordinate.axis,
                axis_sign=state.coordinate.sign,
                reference=reference,
                angle_axis_sign=float(state.angle_cfg.get("axis_sign", 1.0)),
            )
            accumulate_angle_samples(state.oh_orientation, label, z_values, angle_values, bond_count=pairs.shape[0])
    else:
        neighbors_by_species = oxygen_hydrogen_neighbors_by_species(
            frame.symbols,
            frame.positions,
            oxygen_symbol=oxygen_symbol,
            hydrogen_symbol=hydrogen_symbol,
            oh_cutoff=float(state.selection_cfg.get("oh_cutoff", 1.25)),
            neighbor_method=str(state.selection_cfg.get("neighbor_method", "auto")),
            neighbor_workers=int(state.selection_cfg.get("neighbor_workers", 1)),
            oxygen_chunk_size=int(state.selection_cfg.get("oxygen_chunk_size", 2048)),
            oxygen_indices=oxygen_indices,
            hydrogen_indices=hydrogen_indices,
        )
        for label in state.species_labels:
            z_values, angle_values, bond_count = neighbor_bisector_z_and_angles(
                positions=frame.positions,
                neighbors=neighbors_by_species[label],
                axis=state.coordinate.axis,
                axis_sign=state.coordinate.sign,
                reference=reference,
                angle_axis_sign=float(state.angle_cfg.get("axis_sign", 1.0)),
            )
            accumulate_angle_samples(state.oh_orientation, label, z_values, angle_values, bond_count=bond_count)
    state.oh_orientation.frames += 1


def _finalize_oh_orientation(
    state: OhOrientationWorkflowState,
    cell: tuple[float, float, float],
) -> OhOrientationResult:
    return finalize_oh_orientation_state(
        state.oh_orientation,
        cell=cell,
        axis=state.coordinate.axis,
        normalization_cfg=state.normalization_cfg,
    )


def optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def safe_label_for_path(label: str) -> str:
    return label.lower().replace("+", "plus").replace("-", "minus").replace(" ", "_")


run_angle_z = run_oh_orientation
