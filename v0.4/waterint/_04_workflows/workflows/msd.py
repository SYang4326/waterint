from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from waterint.config import require_mapping
from waterint._00_io.common import TrajectoryFrame
from waterint._01_core.analysis_selection import analysis_indices
from waterint._01_core.cell import orthorhombic_cell_vectors
from waterint._01_core.coordinates import coordinate_spec_from_config, coordinate_values
from waterint._01_core.selection import SelectionContext
from waterint._02_computation.msd import MsdResult, compute_msd
from waterint._03_output.metadata import write_metadata
from waterint._03_output.msd import plot_msd, write_msd_csv
from waterint._04_workflows.workflows.common import iter_frames, parse_cell, parse_range, required_workflow_sections, resolve_path


def run_msd(config: dict[str, Any]) -> MsdResult:
    input_cfg, system_cfg, output_cfg = required_workflow_sections(config)
    msd_cfg = require_mapping(config, "msd")
    selection_cfg = require_mapping(config, "selection")
    traj_path = resolve_path(config, input_cfg["trajectory"])
    context = SelectionContext.from_input_config(input_cfg)
    configured_cell = parse_cell(system_cfg.get("cell", "auto"))
    first_frame: TrajectoryFrame | None = None
    first_types: np.ndarray | None = None
    selected: np.ndarray | None = None
    position_frames: list[np.ndarray] = []
    cell_vector_frames: list[np.ndarray] = []
    for frame in iter_frames(traj_path, input_cfg):
        if first_frame is None:
            first_frame = frame
            first_types = None if frame.types is None else frame.types.copy()
            selected = analysis_indices(frame, selection_cfg, context)
            selected = apply_initial_layer(frame, selected, msd_cfg, context)
            if selected.size == 0:
                raise ValueError("MSD selection found no atoms after applying selection/layer criteria.")
        elif first_types is not None and (frame.types is None or not np.array_equal(frame.types, first_types)):
            raise ValueError("MSD requires fixed atom ordering and types across the trajectory.")
        position_frames.append(frame.positions[selected])
        cell_vector_frames.append(frame_cell_vectors(frame, configured_cell))
    if first_frame is None or selected is None or len(position_frames) < 2:
        raise ValueError("MSD requires at least two trajectory frames.")

    positions = np.stack(position_frames)
    cell_vectors = np.stack(cell_vector_frames)
    dimensionality = msd_cfg.get("dimensionality", "3d")
    normal_axis = axis_from_value(msd_cfg.get("plane_normal_axis", "z"))
    result = compute_msd(
        positions,
        cell_vectors=cell_vectors,
        pbc=parse_pbc(msd_cfg.get("pbc", [True, True, True]), "msd.pbc"),
        timestep_ps=float(msd_cfg["timestep_ps"]),
        max_lag_frames=optional_positive_int(msd_cfg.get("max_lag_frames")),
        origin_stride=positive_int(msd_cfg.get("origin_stride", 1), "msd.origin_stride"),
        dimensionality=dimensionality,
        plane_normal_axis=normal_axis,
        backend=str(msd_cfg.get("backend", "auto")),
    )

    outdir = resolve_path(config, output_cfg.get("directory", "output"))
    outdir.mkdir(parents=True, exist_ok=True)
    prefix = str(output_cfg.get("prefix", "msd"))
    csv_path = outdir / f"{prefix}.csv"
    png_path = outdir / f"{prefix}.png" if bool(output_cfg.get("plot", True)) else None
    metadata_path = outdir / f"{prefix}_metadata.json"
    write_msd_csv(csv_path, result.time_ps, result.lag_frames, result.msd_a2, result.samples)
    if png_path is not None:
        plot_msd(png_path, result.time_ps, result.msd_a2, title=str(output_cfg.get("title", "Mean-squared displacement")), dimensionality=result.dimensionality, dpi=int(output_cfg.get("dpi", 220)))
    write_metadata(metadata_path, {
        "analysis_name": "msd", "package": "waterint", "config_file": config.get("_config_path"),
        "trajectory": str(traj_path), "frames": len(position_frames), "selected_atoms": int(selected.size),
        "selection_frame": int(first_frame.index), "outputs": {"csv": str(csv_path), "png": str(png_path) if png_path else None},
        "config": {key: value for key, value in config.items() if not key.startswith("_")},
    })
    return replace(result, csv_path=csv_path, png_path=png_path, metadata_path=metadata_path)


def apply_initial_layer(
    frame: TrajectoryFrame,
    selected: np.ndarray,
    msd_cfg: dict[str, Any],
    context: SelectionContext,
) -> np.ndarray:
    layer_cfg = msd_cfg.get("layer")
    if layer_cfg is None:
        return selected
    if not isinstance(layer_cfg, dict):
        raise ValueError("msd.layer must be a mapping when provided.")
    coordinate = coordinate_spec_from_config(require_mapping_like(layer_cfg, "coordinate"))
    low, high = parse_range(layer_cfg.get("range"), name="msd.layer.range")
    values = coordinate_values(frame, selected, coordinate, context)
    return selected[(values >= low) & (values < high)]


def frame_cell_vectors(frame: TrajectoryFrame, configured_cell: tuple[float, float, float] | None = None) -> np.ndarray:
    if frame.cell_vectors is not None:
        return np.asarray(frame.cell_vectors, dtype=float)
    if frame.cell is not None:
        return orthorhombic_cell_vectors(frame.cell)
    if configured_cell is not None:
        return orthorhombic_cell_vectors(configured_cell)
    raise ValueError("MSD PBC unwrapping requires cell information in every frame.")


def parse_pbc(value: Any, name: str) -> tuple[bool, bool, bool]:
    if isinstance(value, bool):
        return (value, value, value)
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{name} must be a boolean or a list of three booleans.")
    return tuple(bool(item) for item in value)  # type: ignore[return-value]


def axis_from_value(value: Any) -> int:
    labels = {"x": 0, "y": 1, "z": 2}
    label = str(value).lower().lstrip("-")
    if label not in labels:
        raise ValueError("msd.plane_normal_axis must be x, y, or z.")
    return labels[label]


def optional_positive_int(value: Any) -> int | None:
    if value in {None, "", 0, "all"}:
        return None
    result = int(value)
    if result <= 0:
        raise ValueError("msd.max_lag_frames must be positive, 0, 'all', or omitted.")
    return result


def positive_int(value: Any, name: str) -> int:
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive.")
    return result


def require_mapping_like(value: dict[str, Any], key: str) -> dict[str, Any]:
    child = value.get(key, value)
    if not isinstance(child, dict):
        raise ValueError(f"msd.layer.{key} must be a mapping.")
    return child
