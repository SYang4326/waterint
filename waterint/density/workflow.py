from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import json
import math

import numpy as np

from waterint.chemistry import classify_oxygen_by_h_count
from waterint.config import require_mapping
from waterint.io.common import TrajectoryFrame
from waterint.io.lammpstrj import read_lammpstrj
from waterint.io.npz import read_npz
from waterint.io.xyz import read_xyz
from waterint.density.plotting import plot_density_profile


AXIS_INDEX = {"x": 0, "y": 1, "z": 2}
OXYGEN_SPECIES_ORDER = ("O2-", "OH-", "H2O", "H3O+", "O_other")
DEFAULT_PROFILE_MASSES_AMU = {
    "O2-": 15.999,
    "OH-": 17.007,
    "H2O": 18.015,
    "H3O+": 19.023,
    "O_other": 15.999,
}
AMU_PER_A3_TO_G_PER_CM3 = 1.66053906660


@dataclass(frozen=True)
class DensityResult:
    bin_centers: np.ndarray
    profiles: dict[str, dict[str, np.ndarray]]
    frames: int
    selected_atoms_total: dict[str, int]
    csv_path: Path
    png_path: Path | None
    metadata_path: Path


def run_density(config: dict[str, Any]) -> DensityResult:
    input_cfg = require_mapping(config, "input")
    system_cfg = require_mapping(config, "system")
    selection_cfg = require_mapping(config, "selection")
    coord_cfg = require_mapping(config, "coordinate")
    output_cfg = require_mapping(config, "output")

    fmt = str(input_cfg.get("format", "xyz")).lower()
    if fmt not in {"xyz", "lammpstrj", "npz"}:
        raise ValueError("input.format must be xyz, lammpstrj, or npz.")

    traj_path = _resolve_path(config, input_cfg["trajectory"])
    configured_cell = _parse_cell(system_cfg.get("cell", "auto"))
    axis_label, axis, axis_sign = _parse_axis(coord_cfg.get("axis", "z"))

    range_min, range_max = _parse_range(coord_cfg.get("range"))
    bins = int(coord_cfg.get("bins", 200))
    if bins <= 0:
        raise ValueError("coordinate.bins must be > 0.")

    profile_labels = _profile_labels(selection_cfg)
    selection_context = _selection_context(input_cfg)

    mode = str(coord_cfg.get("mode", "absolute"))
    reference_cfg = coord_cfg.get("reference", {})
    if mode not in {"absolute", "relative_to_reference", "relative_to_slab"}:
        raise ValueError("coordinate.mode must be absolute, relative_to_reference, or relative_to_slab.")
    if mode == "relative_to_reference" and not isinstance(reference_cfg, dict):
        raise ValueError("coordinate.reference must be a mapping for relative_to_reference mode.")
    if mode == "relative_to_slab" and not isinstance(reference_cfg, dict):
        raise ValueError("coordinate.reference must be a mapping for relative_to_slab mode.")

    bin_edges = np.linspace(range_min, range_max, bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    counts_by_label = {
        label: np.zeros(bins, dtype=float)
        for label in profile_labels
    }

    frames = 0
    selected_atoms_total = {label: 0 for label in profile_labels}
    cell = configured_cell
    for frame in _iter_frames(traj_path, input_cfg):
        if cell is None:
            if frame.cell is None:
                raise ValueError("system.cell is auto, but the trajectory did not provide cell information.")
            cell = frame.cell

        reference = 0.0
        if mode == "relative_to_reference":
            reference = _reference_value(frame, axis, reference_cfg, selection_context)
        elif mode == "relative_to_slab":
            reference = _slab_reference_value(frame, axis, reference_cfg, axis_sign, selection_context)

        selected_indices_by_label = _selected_indices_by_label(frame, selection_cfg, selection_context)
        for label, selected_indices in selected_indices_by_label.items():
            selected_atoms_total[label] += int(selected_indices.size)
            values = axis_sign * (frame.positions[selected_indices, axis] - reference)
            hist, _ = np.histogram(values, bins=bin_edges)
            counts_by_label[label] += hist
        frames += 1

    if frames == 0:
        raise ValueError(f"No frames found in trajectory: {traj_path}")
    if cell is None:
        raise ValueError("No cell information was available. Set system.cell manually.")

    profiles: dict[str, dict[str, np.ndarray]] = {}
    for label, counts in counts_by_label.items():
        profiles[label] = {
            "counts_per_frame": counts / frames,
            "density": _normalize_density(
                counts=counts,
                frames=frames,
                cell=cell,
                axis=axis,
                bin_width=(range_max - range_min) / bins,
                normalization_cfg=config.get("normalization", {}),
                label=label,
            ),
        }

    outdir = _resolve_path(config, output_cfg.get("directory", "output"))
    outdir.mkdir(parents=True, exist_ok=True)
    prefix = str(output_cfg.get("prefix", "density"))
    csv_path = outdir / f"{prefix}.csv"
    png_path = outdir / f"{prefix}.png" if bool(output_cfg.get("plot", True)) else None
    metadata_path = outdir / f"{prefix}_metadata.json"

    _write_density_csv(csv_path, bin_centers, profiles, axis_label)
    if png_path is not None:
        plot_density_profile(
            path=png_path,
            x=bin_centers,
            y=profiles,
            xlabel=f"{axis_label} coordinate (Angstrom)",
            ylabel=_density_ylabel(config.get("normalization", {})),
            title=str(output_cfg.get("title", "Density profile")),
        )
    _write_metadata(
        metadata_path,
        config=config,
        traj_path=traj_path,
        axis=axis_label,
        profile_labels=profile_labels,
        frames=frames,
        selected_atoms_total=selected_atoms_total,
        csv_path=csv_path,
        png_path=png_path,
    )

    return DensityResult(
        bin_centers=bin_centers,
        profiles=profiles,
        frames=frames,
        selected_atoms_total=selected_atoms_total,
        csv_path=csv_path,
        png_path=png_path,
        metadata_path=metadata_path,
    )


def _parse_axis(value: Any) -> tuple[str, int, float]:
    axis_label = str(value).strip().lower()
    sign = -1.0 if axis_label.startswith("-") else 1.0
    bare_axis = axis_label[1:] if axis_label.startswith("-") else axis_label
    if bare_axis not in AXIS_INDEX:
        raise ValueError("coordinate.axis must be one of x, y, z, -x, -y, -z.")
    return axis_label, AXIS_INDEX[bare_axis], sign


def _iter_frames(traj_path: Path, input_cfg: dict[str, Any]):
    fmt = str(input_cfg.get("format", "xyz")).lower()
    stride = int(input_cfg.get("stride", 1))
    max_frames_raw = input_cfg.get("max_frames", None)
    max_frames = None if max_frames_raw in {None, 0, "all"} else int(max_frames_raw)
    start_timestep_raw = input_cfg.get("start_timestep", None)
    start_timestep = None if start_timestep_raw in {None, ""} else int(start_timestep_raw)
    if stride <= 0:
        raise ValueError("input.stride must be > 0.")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("input.max_frames must be positive, 0, 'all', or omitted.")

    if fmt == "xyz":
        frames = read_xyz(traj_path)
    elif fmt == "lammpstrj":
        yield from read_lammpstrj(
            traj_path,
            type_map=input_cfg.get("type_map", {}),
            start_timestep=start_timestep,
            stride=stride,
            max_frames=max_frames,
        )
        return
    elif fmt == "npz":
        frames = read_npz(traj_path, type_map=input_cfg.get("type_map", {}))
    else:
        raise ValueError("input.format must be xyz, lammpstrj, or npz.")

    yielded = 0
    for frame in frames:
        if start_timestep is not None:
            if frame.step is None:
                raise ValueError("input.start_timestep requires trajectory frames with timestep information.")
            if frame.step < start_timestep:
                continue
        if frame.index % stride != 0:
            continue
        yield frame
        yielded += 1
        if max_frames is not None and yielded >= max_frames:
            return


def _profile_labels(selection_cfg: dict[str, Any]) -> list[str]:
    mode = str(selection_cfg.get("mode", "element"))
    if mode == "element":
        species = selection_cfg.get("species")
        if not species or not isinstance(species, list):
            raise ValueError("selection.species must be a non-empty list, e.g. ['O'].")
        return [str(selection_cfg.get("label", "_".join(sorted(str(item) for item in species))))]

    if mode == "oxygen_species":
        selected = selection_cfg.get("oxygen_species", "all")
        if selected == "all":
            return list(OXYGEN_SPECIES_ORDER)
        if not isinstance(selected, list) or not selected:
            raise ValueError("selection.oxygen_species must be 'all' or a non-empty list.")
        labels = [str(item) for item in selected]
        unknown = [label for label in labels if label not in OXYGEN_SPECIES_ORDER]
        if unknown:
            raise ValueError(f"Unknown oxygen species labels: {unknown}")
        return labels

    raise ValueError("selection.mode must be element or oxygen_species.")


def _selection_context(input_cfg: dict[str, Any]) -> dict[str, Any]:
    raw_type_map = input_cfg.get("type_map", {})
    symbol_to_types: dict[str, list[int]] = {}
    if isinstance(raw_type_map, dict):
        for raw_type, raw_symbol in raw_type_map.items():
            symbol_to_types.setdefault(str(raw_symbol), []).append(int(raw_type))
    return {"symbol_to_types": symbol_to_types}


def _selected_indices_by_label(
    frame: TrajectoryFrame,
    selection_cfg: dict[str, Any],
    selection_context: dict[str, Any],
) -> dict[str, np.ndarray]:
    mode = str(selection_cfg.get("mode", "element"))
    if mode == "element":
        species = selection_cfg.get("species")
        species_set = {str(item) for item in species}
        label = str(selection_cfg.get("label", "_".join(sorted(species_set))))
        mask = _element_mask(frame, species_set, selection_context)
        return {label: np.where(mask)[0]}

    if mode == "oxygen_species":
        classified = classify_oxygen_by_h_count(
            frame.symbols,
            frame.positions,
            oxygen_symbol=str(selection_cfg.get("oxygen_symbol", "O")),
            hydrogen_symbol=str(selection_cfg.get("hydrogen_symbol", "H")),
            oh_cutoff=float(selection_cfg.get("oh_cutoff", 1.25)),
            neighbor_method=str(selection_cfg.get("neighbor_method", "auto")),
            neighbor_workers=int(selection_cfg.get("neighbor_workers", 1)),
            oxygen_chunk_size=int(selection_cfg.get("oxygen_chunk_size", 2048)),
            oxygen_indices=_element_indices(
                frame,
                {str(selection_cfg.get("oxygen_symbol", "O"))},
                selection_context,
            ),
            hydrogen_indices=_element_indices(
                frame,
                {str(selection_cfg.get("hydrogen_symbol", "H"))},
                selection_context,
            ),
        )
        return {
            label: classified[label]
            for label in _profile_labels(selection_cfg)
        }

    raise ValueError("selection.mode must be element or oxygen_species.")


def _element_indices(
    frame: TrajectoryFrame,
    species_set: set[str],
    selection_context: dict[str, Any],
) -> np.ndarray:
    return np.where(_element_mask(frame, species_set, selection_context))[0]


def _element_mask(
    frame: TrajectoryFrame,
    species_set: set[str],
    selection_context: dict[str, Any],
) -> np.ndarray:
    symbol_to_types = selection_context.get("symbol_to_types", {})
    if frame.types is not None and symbol_to_types:
        type_ids: list[int] = []
        for species in species_set:
            type_ids.extend(symbol_to_types.get(species, []))
        if type_ids:
            return np.isin(frame.types, type_ids)
    return np.isin(np.asarray(frame.symbols), list(species_set))


def _resolve_path(config: dict[str, Any], path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return Path(config["_config_dir"]) / path


def _parse_cell(value: Any) -> tuple[float, float, float] | None:
    if value is None or str(value).lower() == "auto":
        return None
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError("system.cell must be [Lx, Ly, Lz] in Angstrom or auto.")
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
    frame: TrajectoryFrame,
    axis: int,
    reference_cfg: dict[str, Any],
    selection_context: dict[str, Any],
) -> float:
    ref_type = str(reference_cfg.get("type", "element_mean"))
    if ref_type != "element_mean":
        raise ValueError("Only reference.type: element_mean is implemented.")

    ref_species = reference_cfg.get("species")
    if not ref_species or not isinstance(ref_species, list):
        raise ValueError("reference.species must be a non-empty list.")
    mask = _element_mask(frame, {str(item) for item in ref_species}, selection_context)
    if not np.any(mask):
        raise ValueError(f"Reference selection found no atoms: {ref_species}")
    return float(np.mean(frame.positions[mask, axis]))


def _slab_reference_value(
    frame: TrajectoryFrame,
    axis: int,
    reference_cfg: dict[str, Any],
    axis_sign: float,
    selection_context: dict[str, Any],
) -> float:
    ref_type = str(reference_cfg.get("type", "slab_surface"))
    if ref_type not in {"slab_surface", "element_surface"}:
        raise ValueError("relative_to_slab requires reference.type: slab_surface.")

    slab_species = reference_cfg.get("species")
    if not slab_species or not isinstance(slab_species, list):
        raise ValueError("reference.species must list slab atom symbols, e.g. ['Mg'].")

    mask = _element_mask(frame, {str(item) for item in slab_species}, selection_context)
    values = frame.positions[mask, axis]
    if values.size == 0:
        raise ValueError(f"Slab reference selection found no atoms: {slab_species}")

    surface = str(reference_cfg.get("surface", "auto")).lower()
    if surface == "auto":
        surface = "max" if axis_sign > 0 else "min"
    if surface == "max":
        return float(np.max(values))
    if surface == "min":
        return float(np.min(values))
    if surface == "mean":
        return float(np.mean(values))
    raise ValueError("reference.surface must be auto, max, min, or mean.")


def _normalize_density(
    counts: np.ndarray,
    frames: int,
    cell: tuple[float, float, float],
    axis: int,
    bin_width: float,
    normalization_cfg: Any,
    label: str,
) -> np.ndarray:
    if normalization_cfg is None:
        normalization_cfg = {}
    if not isinstance(normalization_cfg, dict):
        raise ValueError("normalization must be a mapping.")
    norm_type = str(normalization_cfg.get("type", "number_density"))
    unit = str(normalization_cfg.get("unit", "")).lower()
    if norm_type == "counts_per_frame":
        return counts / frames
    if norm_type not in {"number_density", "mass_density"}:
        raise ValueError("normalization.type must be number_density, mass_density, or counts_per_frame.")

    perpendicular_lengths = [cell[i] for i in range(3) if i != axis]
    slab_volume = perpendicular_lengths[0] * perpendicular_lengths[1] * bin_width
    if not math.isfinite(slab_volume) or slab_volume <= 0:
        raise ValueError("Computed slab volume must be positive.")
    number_density = counts / frames / slab_volume
    if norm_type == "number_density" and unit not in {"g_cm3", "g/cm3", "g/cm^3"}:
        return number_density

    mass_amu = _profile_mass_amu(label, normalization_cfg)
    return number_density * mass_amu * AMU_PER_A3_TO_G_PER_CM3


def _profile_mass_amu(label: str, normalization_cfg: dict[str, Any]) -> float:
    masses = normalization_cfg.get("masses_amu", {})
    if masses is None:
        masses = {}
    if not isinstance(masses, dict):
        raise ValueError("normalization.masses_amu must be a mapping.")
    if label in masses:
        mass = float(masses[label])
    elif label in DEFAULT_PROFILE_MASSES_AMU:
        mass = DEFAULT_PROFILE_MASSES_AMU[label]
    else:
        mass = normalization_cfg.get("mass_amu", None)
        if mass is None:
            raise ValueError(
                f"Need mass for profile {label!r}. Set normalization.mass_amu or normalization.masses_amu."
            )
        mass = float(mass)
    if mass <= 0:
        raise ValueError("Profile mass must be positive.")
    return mass


def _density_ylabel(normalization_cfg: Any) -> str:
    if isinstance(normalization_cfg, dict) and normalization_cfg.get("type") == "counts_per_frame":
        return "counts per frame"
    if isinstance(normalization_cfg, dict):
        norm_type = str(normalization_cfg.get("type", "number_density"))
        unit = str(normalization_cfg.get("unit", "")).lower()
        if norm_type == "mass_density" or unit in {"g_cm3", "g/cm3", "g/cm^3"}:
            return "mass density (g/cm^3)"
    return "number density (1/A^3)"


def _write_density_csv(
    path: Path,
    bin_centers: np.ndarray,
    profiles: dict[str, dict[str, np.ndarray]],
    axis_name: str,
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        columns = [f"{axis_name}_center_A"]
        for label in profiles:
            columns.extend([f"{label}_counts_per_frame", f"{label}_density"])
        handle.write(",".join(columns) + "\n")
        for i, x in enumerate(bin_centers):
            row = [f"{x:.10g}"]
            for profile in profiles.values():
                row.append(f"{profile['counts_per_frame'][i]:.10g}")
                row.append(f"{profile['density'][i]:.10g}")
            handle.write(",".join(row) + "\n")


def _write_metadata(
    path: Path,
    config: dict[str, Any],
    traj_path: Path,
    axis: str,
    profile_labels: list[str],
    frames: int,
    selected_atoms_total: dict[str, int],
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
        "profile_labels": profile_labels,
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
