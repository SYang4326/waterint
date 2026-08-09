from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from waterint.config import require_mapping
from waterint._02_computation.conductivity import ConductivityResult, compute_nernst_einstein_conductivity
from waterint._03_output.conductivity import plot_conductivity_msd, write_conductivity_csv, write_conductivity_msd_csv
from waterint._03_output.metadata import write_metadata
from waterint._04_workflows.workflows.common import required_workflow_sections, resolve_path
from waterint._04_workflows.workflows.msd import frame_cell_vectors, positive_int, run_msd


def run_conductivity(config: dict[str, Any]) -> ConductivityResult:
    """Run a fixed-carrier Nernst--Einstein conductivity workflow."""

    input_cfg, system_cfg, output_cfg = required_workflow_sections(config)
    conductivity_cfg = require_mapping(config, "conductivity")
    msd_cfg = require_mapping(config, "msd")
    temperature_k = positive_float(conductivity_cfg.get("temperature_K"), "conductivity.temperature_K")
    charge_e = float(conductivity_cfg.get("charge_e", 0.0))
    fit_range = parse_fit_range(conductivity_cfg.get("fit_range_ps"))
    msd_result = run_msd(config)
    dimensions = 2 if msd_result.dimensionality == "2d" else 3
    volume_a3, thickness_a = analysis_volume(config, input_cfg, system_cfg, conductivity_cfg)
    result = compute_nernst_einstein_conductivity(
        msd_result.time_ps,
        msd_result.msd_a2,
        carrier_count=msd_result.selected_atoms,
        volume_a3=volume_a3,
        temperature_k=temperature_k,
        charge_e=charge_e,
        dimensions=dimensions,
        fit_range_ps=fit_range,
        sheet_thickness_a=thickness_a,
    )

    outdir = resolve_path(config, output_cfg.get("directory", "output"))
    outdir.mkdir(parents=True, exist_ok=True)
    prefix = str(output_cfg.get("prefix", "conductivity"))
    csv_path = outdir / f"{prefix}.csv"
    msd_csv_path = outdir / f"{prefix}_msd.csv"
    png_path = outdir / f"{prefix}.png" if bool(output_cfg.get("plot", True)) else None
    metadata_path = outdir / f"{prefix}_metadata.json"
    write_conductivity_csv(csv_path, result, temperature_k=temperature_k, charge_e=charge_e, fit_range_ps=fit_range)
    write_conductivity_msd_csv(msd_csv_path, result)
    if png_path is not None:
        plot_conductivity_msd(png_path, result, title=str(output_cfg.get("title", "Nernst-Einstein conductivity MSD fit")), dpi=int(output_cfg.get("dpi", 220)))
    write_metadata(metadata_path, {
        "analysis_name": "conductivity", "method": "Nernst-Einstein", "package": "waterint",
        "config_file": config.get("_config_path"), "temperature_K": temperature_k, "charge_e": charge_e,
        "carrier_count": result.carrier_count, "volume_A3": result.volume_a3,
        "diffusion_m2_per_s": result.diffusion_m2_per_s, "conductivity_S_per_m": result.conductivity_s_per_m,
        "sheet_conductance_S": result.sheet_conductance_s,
        "outputs": {"csv": str(csv_path), "msd_csv": str(msd_csv_path), "png": str(png_path) if png_path else None},
        "config": {key: value for key, value in config.items() if not key.startswith("_")},
    })
    return replace(result, csv_path=csv_path, msd_csv_path=msd_csv_path, png_path=png_path, metadata_path=metadata_path)


def analysis_volume(config: dict[str, Any], input_cfg: dict[str, Any], system_cfg: dict[str, Any], conductivity_cfg: dict[str, Any]) -> tuple[float, float | None]:
    """Return mean analysis volume and optional physical thickness for sheet G."""

    volume_cfg = conductivity_cfg.get("volume", {"mode": "cell"})
    if not isinstance(volume_cfg, dict):
        raise ValueError("conductivity.volume must be a mapping.")
    mode = str(volume_cfg.get("mode", "cell")).lower()
    if mode not in {"cell", "slab"}:
        raise ValueError("conductivity.volume.mode must be cell or slab.")
    if mode == "slab":
        thickness_a = positive_float(volume_cfg.get("thickness_A"), "conductivity.volume.thickness_A")
        axis = axis_from_value(volume_cfg.get("normal_axis", msd_cfg_axis(config)))
    else:
        thickness_a = None
        axis = 2
    configured_cell = system_cfg.get("cell", "auto")
    vectors = []
    from waterint._04_workflows.workflows.common import iter_frames

    for frame in iter_frames(resolve_path(config, input_cfg["trajectory"]), input_cfg):
        vectors.append(frame_cell_vectors(frame, None if str(configured_cell).lower() == "auto" else tuple(float(value) for value in configured_cell)))
    if not vectors:
        raise ValueError("Conductivity volume calculation found no trajectory frames.")
    cells = np.asarray(vectors, dtype=float)
    if mode == "cell":
        volumes = np.abs(np.linalg.det(cells))
    else:
        other_axes = [index for index in range(3) if index != axis]
        areas = np.linalg.norm(np.cross(cells[:, other_axes[0], :], cells[:, other_axes[1], :]), axis=1)
        volumes = areas * thickness_a
    if np.any(volumes <= 0):
        raise ValueError("Conductivity requires non-zero cell/slab volume in every frame.")
    return float(np.mean(volumes)), thickness_a


def parse_fit_range(value: Any) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError("conductivity.fit_range_ps is required and must be [start_ps, stop_ps].")
    return float(value[0]), float(value[1])


def positive_float(value: Any, name: str) -> float:
    if value is None:
        raise ValueError(f"{name} is required.")
    result = float(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive.")
    return result


def axis_from_value(value: Any) -> int:
    labels = {"x": 0, "y": 1, "z": 2}
    label = str(value).lower().lstrip("-")
    if label not in labels:
        raise ValueError("conductivity.volume.normal_axis must be x, y, or z.")
    return labels[label]


def msd_cfg_axis(config: dict[str, Any]) -> Any:
    return require_mapping(config, "msd").get("plane_normal_axis", "z")
