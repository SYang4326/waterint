from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import json
import math

import numpy as np

from waterint.config import require_mapping
from waterint.io.xyz import read_xyz
from waterint.plotting import plot_density_profile


AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


@dataclass(frozen=True)
class DensityResult:
    bin_centers: np.ndarray
    counts_per_frame: np.ndarray
    density: np.ndarray
    frames: int
    selected_atoms_total: int
    csv_path: Path
    png_path: Path | None
    metadata_path: Path


def run_density(config: dict[str, Any]) -> DensityResult:
    input_cfg = require_mapping(config, "input")
    system_cfg = require_mapping(config, "system")
    selection_cfg = require_mapping(config, "selection")
    coord_cfg = require_mapping(config, "coordinate")
    output_cfg = require_mapping(config, "output")

    fmt = input_cfg.get("format", "xyz")
    if fmt != "xyz":
        raise ValueError("Only input.format: xyz is implemented in this first version.")

    traj_path = _resolve_path(config, input_cfg["trajectory"])
    cell = _parse_cell(system_cfg.get("cell"))
    axis_name = str(coord_cfg.get("axis", "z")).lower()
    if axis_name not in AXIS_INDEX:
        raise ValueError("coordinate.axis must be one of x, y, z.")
    axis = AXIS_INDEX[axis_name]

    range_min, range_max = _parse_range(coord_cfg.get("range"))
    bins = int(coord_cfg.get("bins", 200))
    if bins <= 0:
        raise ValueError("coordinate.bins must be > 0.")

    species = selection_cfg.get("species")
    if not species or not isinstance(species, list):
        raise ValueError("selection.species must be a non-empty list, e.g. ['O'].")
    species_set = {str(item) for item in species}
    label = str(selection_cfg.get("label", "_".join(sorted(species_set))))

    mode = str(coord_cfg.get("mode", "absolute"))
    reference_cfg = coord_cfg.get("reference", {})
    if mode not in {"absolute", "relative_to_reference"}:
        raise ValueError("coordinate.mode must be absolute or relative_to_reference.")
    if mode == "relative_to_reference" and not isinstance(reference_cfg, dict):
        raise ValueError("coordinate.reference must be a mapping for relative_to_reference mode.")

    bin_edges = np.linspace(range_min, range_max, bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    counts = np.zeros(bins, dtype=float)

    frames = 0
    selected_atoms_total = 0
    for frame in read_xyz(traj_path):
        symbols = np.asarray(frame.symbols)
        selected_mask = np.isin(symbols, list(species_set))
        selected_positions = frame.positions[selected_mask]
        selected_atoms_total += int(selected_positions.shape[0])

        values = selected_positions[:, axis]
        if mode == "relative_to_reference":
            values = values - _reference_value(frame.symbols, frame.positions, axis, reference_cfg)

        hist, _ = np.histogram(values, bins=bin_edges)
        counts += hist
        frames += 1

    if frames == 0:
        raise ValueError(f"No frames found in trajectory: {traj_path}")

    counts_per_frame = counts / frames
    density = _normalize_density(
        counts=counts,
        frames=frames,
        cell=cell,
        axis=axis,
        bin_width=(range_max - range_min) / bins,
        normalization_cfg=config.get("normalization", {}),
    )

    outdir = _resolve_path(config, output_cfg.get("directory", "output"))
    outdir.mkdir(parents=True, exist_ok=True)
    prefix = str(output_cfg.get("prefix", f"density_{label}"))
    csv_path = outdir / f"{prefix}.csv"
    png_path = outdir / f"{prefix}.png" if bool(output_cfg.get("plot", True)) else None
    metadata_path = outdir / f"{prefix}_metadata.json"

    _write_density_csv(csv_path, bin_centers, counts_per_frame, density, axis_name)
    if png_path is not None:
        plot_density_profile(
            path=png_path,
            x=bin_centers,
            y=density,
            xlabel=f"{axis_name} coordinate (Angstrom)",
            ylabel=_density_ylabel(config.get("normalization", {})),
            title=str(output_cfg.get("title", f"Density profile: {label}")),
        )
    _write_metadata(
        metadata_path,
        config=config,
        traj_path=traj_path,
        axis=axis_name,
        label=label,
        frames=frames,
        selected_atoms_total=selected_atoms_total,
        csv_path=csv_path,
        png_path=png_path,
    )

    return DensityResult(
        bin_centers=bin_centers,
        counts_per_frame=counts_per_frame,
        density=density,
        frames=frames,
        selected_atoms_total=selected_atoms_total,
        csv_path=csv_path,
        png_path=png_path,
        metadata_path=metadata_path,
    )


def _resolve_path(config: dict[str, Any], path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return Path(config["_config_dir"]) / path


def _parse_cell(value: Any) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError("system.cell must be [Lx, Ly, Lz] in Angstrom.")
    cell = tuple(float(v) for v in value)
    if any(v <= 0 for v in cell):
        raise ValueError("system.cell values must be positive.")
    return cell


def _parse_range(value: Any) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError("coordinate.range must be [min, max].")
    range_min, range_max = float(value[0]), float(value[1])
    if not range_max > range_min:
        raise ValueError("coordinate.range max must be larger than min.")
    return range_min, range_max


def _reference_value(
    symbols: list[str],
    positions: np.ndarray,
    axis: int,
    reference_cfg: dict[str, Any],
) -> float:
    ref_type = str(reference_cfg.get("type", "element_mean"))
    if ref_type != "element_mean":
        raise ValueError("Only reference.type: element_mean is implemented.")

    ref_species = reference_cfg.get("species")
    if not ref_species or not isinstance(ref_species, list):
        raise ValueError("reference.species must be a non-empty list.")
    mask = np.isin(np.asarray(symbols), [str(item) for item in ref_species])
    if not np.any(mask):
        raise ValueError(f"Reference selection found no atoms: {ref_species}")
    return float(np.mean(positions[mask, axis]))


def _normalize_density(
    counts: np.ndarray,
    frames: int,
    cell: tuple[float, float, float],
    axis: int,
    bin_width: float,
    normalization_cfg: Any,
) -> np.ndarray:
    if normalization_cfg is None:
        normalization_cfg = {}
    if not isinstance(normalization_cfg, dict):
        raise ValueError("normalization must be a mapping.")
    norm_type = str(normalization_cfg.get("type", "number_density"))
    if norm_type == "counts_per_frame":
        return counts / frames
    if norm_type != "number_density":
        raise ValueError("normalization.type must be number_density or counts_per_frame.")

    perpendicular_lengths = [cell[i] for i in range(3) if i != axis]
    slab_volume = perpendicular_lengths[0] * perpendicular_lengths[1] * bin_width
    if not math.isfinite(slab_volume) or slab_volume <= 0:
        raise ValueError("Computed slab volume must be positive.")
    return counts / frames / slab_volume


def _density_ylabel(normalization_cfg: Any) -> str:
    if isinstance(normalization_cfg, dict) and normalization_cfg.get("type") == "counts_per_frame":
        return "counts per frame"
    return "number density (1/A^3)"


def _write_density_csv(
    path: Path,
    bin_centers: np.ndarray,
    counts_per_frame: np.ndarray,
    density: np.ndarray,
    axis_name: str,
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"{axis_name}_center_A,counts_per_frame,density\n")
        for x, count, rho in zip(bin_centers, counts_per_frame, density):
            handle.write(f"{x:.10g},{count:.10g},{rho:.10g}\n")


def _write_metadata(
    path: Path,
    config: dict[str, Any],
    traj_path: Path,
    axis: str,
    label: str,
    frames: int,
    selected_atoms_total: int,
    csv_path: Path,
    png_path: Path | None,
) -> None:
    public_config = {
        key: value
        for key, value in config.items()
        if not key.startswith("_")
    }
    metadata = {
        "analysis_name": "density",
        "package": "waterint",
        "config_file": config.get("_config_path"),
        "trajectory": str(traj_path),
        "axis": axis,
        "selection_label": label,
        "frames": frames,
        "selected_atoms_total": selected_atoms_total,
        "outputs": {
            "csv": str(csv_path),
            "png": str(png_path) if png_path else None,
        },
        "config": public_config,
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")
