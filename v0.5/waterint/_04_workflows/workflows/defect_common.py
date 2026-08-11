from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from waterint.config import require_mapping
from waterint._01_core.analysis_selection import analysis_indices
from waterint._01_core.selection import SelectionContext
from waterint._02_computation.defect_transport import (
    DefectMsdResult,
    DefectTrackingResult,
    compute_defect_msd,
    track_defects,
)
from waterint._04_workflows.workflows.common import iter_frames, resolve_path
from waterint._04_workflows.workflows.msd import (
    apply_initial_layer,
    axis_from_value,
    frame_cell_vectors,
    optional_positive_int,
    parse_pbc,
    positive_int,
)


@dataclass(frozen=True)
class DefectAnalysis:
    tracking: DefectTrackingResult
    msd: DefectMsdResult
    steps: np.ndarray
    cell_vectors: np.ndarray


def run_defect_analysis(config: dict[str, Any], *, charge_e: float = -1.0) -> DefectAnalysis:
    input_cfg = require_mapping(config, "input")
    system_cfg = require_mapping(config, "system")
    selection_cfg = require_mapping(config, "selection")
    tracking_cfg = require_mapping(config, "defect_tracking")
    msd_cfg = require_mapping(config, "defect_msd")
    if "oxygen_species" not in selection_cfg:
        raise ValueError("Dynamic defect tracking requires selection.oxygen_species.")

    context = SelectionContext.from_input_config(input_cfg)
    trajectory = resolve_path(config, input_cfg["trajectory"])
    configured_cell = system_cfg.get("cell", "auto")
    explicit_cell = None
    if str(configured_cell).lower() != "auto":
        explicit_cell = tuple(float(value) for value in configured_cell)
    positions_by_frame: list[np.ndarray] = []
    indices_by_frame: list[np.ndarray] = []
    vectors: list[np.ndarray] = []
    steps: list[int] = []
    layer = tracking_cfg.get("layer")
    for frame_number, frame in enumerate(iter_frames(trajectory, input_cfg)):
        selected = analysis_indices(frame, selection_cfg, context)
        if layer is not None:
            selected = apply_initial_layer(frame, selected, {"layer": layer}, context)
        positions_by_frame.append(np.asarray(frame.positions[selected], dtype=float))
        indices_by_frame.append(np.asarray(selected, dtype=np.int64))
        vectors.append(frame_cell_vectors(frame, explicit_cell))
        steps.append(frame_number if frame.step is None else int(frame.step))
    if len(positions_by_frame) < 2:
        raise ValueError("Dynamic defect analysis requires at least two trajectory frames.")

    timestep_ps = float(tracking_cfg["timestep_ps"])
    pbc = parse_pbc(tracking_cfg.get("pbc", [True, True, True]), "defect_tracking.pbc")
    tracking = track_defects(
        positions_by_frame,
        indices_by_frame,
        cell_vectors=np.asarray(vectors),
        pbc=pbc,
        timestep_ps=timestep_ps,
        gate_a=float(tracking_cfg.get("gate_A", 3.0)),
        charge_e=charge_e,
    )
    msd = compute_defect_msd(
        tracking,
        max_lag_frames=optional_positive_int(msd_cfg.get("max_lag_frames")),
        origin_stride=positive_int(msd_cfg.get("origin_stride", 1), "defect_msd.origin_stride"),
        frame_stride=positive_int(msd_cfg.get("frame_stride", 1), "defect_msd.frame_stride"),
        dimensionality=msd_cfg.get("dimensionality", "3d"),
        plane_normal_axis=axis_from_value(msd_cfg.get("plane_normal_axis", "z")),
    )
    return DefectAnalysis(
        tracking=tracking,
        msd=msd,
        steps=np.asarray(steps, dtype=np.int64),
        cell_vectors=np.asarray(vectors, dtype=float),
    )


def analysis_volume_from_cells(
    cells: np.ndarray,
    volume_cfg: dict[str, Any],
    *,
    default_axis: Any = "z",
) -> tuple[float, float | None]:
    mode = str(volume_cfg.get("mode", "cell")).lower()
    if mode == "cell":
        volumes = np.abs(np.linalg.det(cells))
        thickness = None
    elif mode == "slab":
        thickness = float(volume_cfg["thickness_A"])
        if thickness <= 0:
            raise ValueError("conductivity.volume.thickness_A must be positive.")
        axis = axis_from_value(volume_cfg.get("normal_axis", default_axis))
        other = [value for value in range(3) if value != axis]
        areas = np.linalg.norm(np.cross(cells[:, other[0], :], cells[:, other[1], :]), axis=1)
        volumes = areas * thickness
    else:
        raise ValueError("conductivity.volume.mode must be cell or slab.")
    if np.any(volumes <= 0):
        raise ValueError("Conductivity requires positive analysis volumes.")
    return float(np.mean(volumes)), thickness
