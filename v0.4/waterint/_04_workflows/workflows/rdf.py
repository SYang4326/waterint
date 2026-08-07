from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Any

import numpy as np

from waterint.config import require_mapping
from waterint._00_io.common import TrajectoryFrame
from waterint._01_core.analysis_selection import analysis_indices
from waterint._01_core.cell import cell_volume, orthorhombic_cell_vectors
from waterint._01_core.coordinates import coordinate_spec_from_config, coordinate_values
from waterint._01_core.selection import SelectionContext
from waterint._02_computation.rdf import RdfPairResult, accumulate_rdf_frame, expected_rdf_counts, finalize_rdf_pair, new_histogram
from waterint._03_output.metadata import write_metadata
from waterint._03_output.rdf import plot_rdf, write_rdf_csv
from waterint._04_workflows.workflows.common import iter_frames, parse_cell, parse_range, required_workflow_sections, resolve_path
from waterint._04_workflows.workflows.msd import parse_pbc


@dataclass(frozen=True)
class RdfResult:
    pairs: dict[str, RdfPairResult]
    metadata_path: Path | None = None


@dataclass
class _PairState:
    name: str
    first_cfg: dict[str, Any]
    second_cfg: dict[str, Any]
    same_selection: bool
    counts: np.ndarray
    expected: np.ndarray


def run_rdf(config: dict[str, Any]) -> RdfResult:
    input_cfg, system_cfg, output_cfg = required_workflow_sections(config)
    rdf_cfg = require_mapping(config, "rdf")
    traj_path = resolve_path(config, input_cfg["trajectory"])
    r_max = float(rdf_cfg.get("r_max", 8.0))
    bins = int(rdf_cfg.get("bins", 200))
    if r_max <= 0 or bins <= 0:
        raise ValueError("rdf.r_max and rdf.bins must be positive.")
    r_edges = np.linspace(0.0, r_max, bins + 1)
    pbc = parse_pbc(rdf_cfg.get("pbc", [True, True, True]), "rdf.pbc")
    pairs = parse_pairs(rdf_cfg)
    states = [
        _PairState(name, first, second, selectors_equivalent(first, second), new_histogram(bins), new_histogram(bins))
        for name, first, second in pairs
    ]
    context = SelectionContext.from_input_config(input_cfg)
    configured_cell = parse_cell(system_cfg.get("cell", "auto"))
    frame_count = 0
    for frame in iter_frames(traj_path, input_cfg):
        vectors = frame_cell_vectors(frame, configured_cell)
        volume = cell_volume(cell_vectors=vectors)
        species_cache: dict[str, dict[str, np.ndarray]] = {}
        selector_cache: dict[str, np.ndarray] = {}
        for state in states:
            first = select_with_layer(frame, state.first_cfg, context, species_cache, selector_cache)
            second = select_with_layer(frame, state.second_cfg, context, species_cache, selector_cache)
            if state.same_selection and not np.array_equal(first, second):
                state.same_selection = False
            overlap_size = int(np.intersect1d(first, second, assume_unique=True).size)
            accumulate_rdf_frame(state.counts, positions=frame.positions, first_indices=first, second_indices=second, r_max=r_max, cell_vectors=vectors, pbc=pbc, same_selection=state.same_selection, backend=str(rdf_cfg.get("backend", "auto")))
            state.expected += expected_rdf_counts(first_size=first.size, second_size=second.size, same_selection=state.same_selection, overlap_size=overlap_size, volume=volume, r_edges=r_edges)
        frame_count += 1
    if frame_count == 0:
        raise ValueError(f"No frames found in trajectory: {traj_path}")
    result_pairs = {state.name: finalize_rdf_pair(name=state.name, counts=state.counts, expected_counts=state.expected, r_edges=r_edges, frames=frame_count) for state in states}

    outdir = resolve_path(config, output_cfg.get("directory", "output"))
    outdir.mkdir(parents=True, exist_ok=True)
    prefix = str(output_cfg.get("prefix", "rdf"))
    written: dict[str, RdfPairResult] = {}
    for name, pair in result_pairs.items():
        safe_name = name.lower().replace(" ", "_").replace("-", "_")
        csv_path = outdir / f"{prefix}_{safe_name}.csv"
        write_rdf_csv(csv_path, pair)
        written[name] = replace(pair, csv_path=csv_path)
    png_path = outdir / f"{prefix}.png" if bool(output_cfg.get("plot", True)) else None
    plot_mode = str(output_cfg.get("plot_mode", "combined")).lower()
    if plot_mode not in {"combined", "separate", "both"}:
        raise ValueError("output.plot_mode must be combined, separate, or both.")
    if png_path is not None and plot_mode in {"combined", "both"}:
        plot_rdf(png_path, written, title=str(output_cfg.get("title", "Radial distribution functions")), dpi=int(output_cfg.get("dpi", 220)))
        written = {name: replace(pair, png_path=png_path) for name, pair in written.items()}
    if bool(output_cfg.get("plot", True)) and plot_mode in {"separate", "both"}:
        separate_paths: dict[str, Path] = {}
        for name, pair in written.items():
            safe_name = name.lower().replace(" ", "_").replace("-", "_")
            separate_path = outdir / f"{prefix}_{safe_name}.png"
            plot_rdf(separate_path, {name: pair}, title=name, dpi=int(output_cfg.get("dpi", 220)))
            separate_paths[name] = separate_path
        written = {name: replace(pair, png_path=separate_paths[name]) for name, pair in written.items()}
    metadata_path = outdir / f"{prefix}_metadata.json"
    write_metadata(metadata_path, {
        "analysis_name": "rdf", "package": "waterint", "config_file": config.get("_config_path"),
        "trajectory": str(traj_path), "frames": frame_count,
        "outputs": {
            "csv": {name: str(pair.csv_path) for name, pair in written.items()},
            "png": {name: str(pair.png_path) if pair.png_path else None for name, pair in written.items()},
        },
        "config": {key: value for key, value in config.items() if not key.startswith("_")},
    })
    return RdfResult(pairs=written, metadata_path=metadata_path)


def parse_pairs(rdf_cfg: dict[str, Any]) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    raw_pairs = rdf_cfg.get("pairs")
    if raw_pairs is None:
        return [("O-O", {"elements": ["O"]}, {"elements": ["O"]}), ("O-H", {"elements": ["O"]}, {"elements": ["H"]})]
    if not isinstance(raw_pairs, list) or not raw_pairs:
        raise ValueError("rdf.pairs must be a non-empty list.")
    parsed = []
    for item in raw_pairs:
        if not isinstance(item, dict):
            raise ValueError("Each rdf.pairs item must be a mapping.")
        name = str(item.get("name", ""))
        first = item.get("first")
        second = item.get("second")
        if not name or not isinstance(first, dict) or not isinstance(second, dict):
            raise ValueError("Each RDF pair needs name, first, and second mappings.")
        parsed.append((name, first, second))
    return parsed


def select_with_layer(
    frame: TrajectoryFrame,
    selector: dict[str, Any],
    context: SelectionContext,
    species_cache: dict[str, dict[str, np.ndarray]],
    selector_cache: dict[str, np.ndarray],
) -> np.ndarray:
    key = json.dumps(selector, sort_keys=True, separators=(",", ":"), default=str)
    if key in selector_cache:
        return selector_cache[key]
    indices = analysis_indices(frame, selector, context, species_cache=species_cache)
    layer = selector.get("layer")
    if layer is None:
        selector_cache[key] = indices
        return indices
    if not isinstance(layer, dict):
        raise ValueError("RDF selector.layer must be a mapping.")
    coordinate = coordinate_spec_from_config(layer.get("coordinate", layer))
    low, high = parse_range(layer.get("range"), name="rdf selector.layer.range")
    values = coordinate_values(frame, indices, coordinate, context)
    selected = indices[(values >= low) & (values < high)]
    selector_cache[key] = selected
    return selected


def selectors_equivalent(first: dict[str, Any], second: dict[str, Any]) -> bool:
    return first == second


def frame_cell_vectors(frame: TrajectoryFrame, configured_cell: tuple[float, float, float] | None = None) -> np.ndarray:
    if frame.cell_vectors is not None:
        return np.asarray(frame.cell_vectors, dtype=float)
    if frame.cell is not None:
        return orthorhombic_cell_vectors(frame.cell)
    if configured_cell is not None:
        return orthorhombic_cell_vectors(configured_cell)
    raise ValueError("RDF PBC normalization requires cell information in every frame.")
