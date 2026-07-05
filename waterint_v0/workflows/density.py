from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from waterint.config import require_mapping
from waterint.density.plotting import plot_density_profile
from waterint.io.common import TrajectoryFrame
from waterint_v0.core.coordinates import CoordinateSpec, coordinate_spec_from_config, coordinate_values
from waterint_v0.core.selection import SelectionContext, element_indices
from waterint_v0.core.species import oxygen_species_indices, oxygen_species_labels
from waterint_v0.computation.density.compute import (
    DensityResult,
    DensityState,
    accumulate_density_frame,
    finalize_density_state,
    new_density_state,
)
from waterint_v0.output.density import density_ylabel, write_density_csv
from waterint_v0.output.metadata import write_metadata
from waterint_v0.workflows.common import parse_cell, parse_range, required_workflow_sections, resolve_path
from waterint_v0.workflows.framewise import run_framewise_analysis


def run_density(config: dict[str, Any]) -> DensityResult:
    input_cfg, system_cfg, output_cfg = required_workflow_sections(config)
    selection_cfg = require_mapping(config, "selection")
    coord_cfg = require_mapping(config, "coordinate")

    traj_path = resolve_path(config, input_cfg["trajectory"])
    configured_cell = parse_cell(system_cfg.get("cell", "auto"))
    coordinate = coordinate_spec_from_config(coord_cfg)
    range_min, range_max = parse_range(coord_cfg.get("range"), name="coordinate.range")
    bins = int(coord_cfg.get("bins", 200))
    if bins <= 0:
        raise ValueError("coordinate.bins must be > 0.")

    labels = density_profile_labels(selection_cfg)
    bin_edges = np.linspace(range_min, range_max, bins + 1)
    context = SelectionContext.from_input_config(input_cfg)
    state = DensityWorkflowState(
        density=new_density_state(labels, bin_edges),
        selection_cfg=selection_cfg,
        context=context,
        coordinate=coordinate,
        normalization_cfg=config.get("normalization", {}),
    )

    framewise = run_framewise_analysis(
        traj_path=traj_path,
        input_cfg=input_cfg,
        configured_cell=configured_cell,
        state=state,
        accumulate=_accumulate_density_frame,
        finalize=_finalize_density,
    )
    result = framewise.result

    outdir = resolve_path(config, output_cfg.get("directory", "output"))
    outdir.mkdir(parents=True, exist_ok=True)
    prefix = str(output_cfg.get("prefix", "density"))
    csv_path = outdir / f"{prefix}.csv"
    png_path = outdir / f"{prefix}.png" if bool(output_cfg.get("plot", True)) else None
    metadata_path = outdir / f"{prefix}_metadata.json"

    write_density_csv(csv_path, result.bin_centers, result.profiles, coordinate.label)
    if png_path is not None:
        plot_density_profile(
            path=png_path,
            x=result.bin_centers,
            y=result.profiles,
            xlabel=f"{coordinate.label} coordinate (Angstrom)",
            ylabel=density_ylabel(config.get("normalization", {})),
            title=str(output_cfg.get("title", "Density profile")),
        )
    write_metadata(
        metadata_path,
        {
            "analysis_name": "density",
            "package": "waterint_v0",
            "config_file": config.get("_config_path"),
            "trajectory": str(traj_path),
            "axis": coordinate.label,
            "profile_labels": labels,
            "frames": result.frames,
            "selected_atoms_total": result.selected_atoms_total,
            "outputs": {
                "csv": str(csv_path),
                "png": str(png_path) if png_path else None,
            },
            "config": {key: value for key, value in config.items() if not key.startswith("_")},
        },
    )

    return replace(result, csv_path=csv_path, png_path=png_path, metadata_path=metadata_path)


@dataclass
class DensityWorkflowState:
    density: DensityState
    selection_cfg: dict[str, Any]
    context: SelectionContext
    coordinate: CoordinateSpec
    normalization_cfg: Any


def density_profile_labels(selection_cfg: dict[str, Any]) -> list[str]:
    mode = str(selection_cfg.get("mode", "element"))
    if mode == "element":
        species = selection_cfg.get("species")
        if not species or not isinstance(species, list):
            raise ValueError("selection.species must be a non-empty list, e.g. ['O'].")
        return [str(selection_cfg.get("label", "_".join(sorted(str(item) for item in species))))]
    if mode == "oxygen_species":
        return oxygen_species_labels(selection_cfg)
    raise ValueError("selection.mode must be element or oxygen_species.")


def _accumulate_density_frame(
    state: DensityWorkflowState,
    frame: TrajectoryFrame,
    _cell: tuple[float, float, float],
) -> None:
    coordinates_by_label = {
        label: coordinate_values(frame, indices, state.coordinate, state.context)
        for label, indices in selected_indices_by_label(frame, state.selection_cfg, state.context).items()
    }
    accumulate_density_frame(state.density, coordinates_by_label)


def _finalize_density(
    state: DensityWorkflowState,
    cell: tuple[float, float, float],
) -> DensityResult:
    return finalize_density_state(
        state.density,
        cell=cell,
        axis=state.coordinate.axis,
        normalization_cfg=state.normalization_cfg,
    )


def selected_indices_by_label(
    frame: TrajectoryFrame,
    selection_cfg: dict[str, Any],
    context: SelectionContext,
) -> dict[str, np.ndarray]:
    mode = str(selection_cfg.get("mode", "element"))
    if mode == "element":
        species = selection_cfg.get("species")
        species_set = {str(item) for item in species}
        label = str(selection_cfg.get("label", "_".join(sorted(species_set))))
        return {label: element_indices(frame, species_set, context)}
    if mode == "oxygen_species":
        return oxygen_species_indices(frame, selection_cfg, context)
    raise ValueError("selection.mode must be element or oxygen_species.")
