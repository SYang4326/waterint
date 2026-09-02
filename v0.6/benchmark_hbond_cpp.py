from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from waterint.config import load_config
from waterint._00_io.common import TrajectoryFrame
from waterint._01_core.selection import SelectionContext, element_indices
from waterint._02_computation.hbond import (
    accumulate_frame_counts,
    build_hbond_neighbor_matrix_cpp,
    classes_by_species,
    frame_hbond_classes_from_counts,
    find_hbonds,
    new_hbond_state,
    species_labels,
)
from waterint._02_computation._native import hbond_geometry_counts
from waterint.chemistry import oxygen_hydrogen_neighbors_by_species
from waterint._04_workflows.workflows.common import iter_frames, parse_cell, resolve_path


STAGE_NAMES = (
    "Read NPZ frames",
    "Select O/H atoms",
    "O-H species assignment",
    "H-bond geometry",
    "Topology grouping",
    "Other overhead",
)

ROUTES = ("python", "cpp")


def main() -> None:
    root = Path(__file__).resolve().parent
    config = load_config(root / "example/mgo_hbond/config_oh_h2o_h3o_npz.yaml")
    outputs = {}
    for route in ROUTES:
        result, timings = run_profile(config, route=route)
        outputs[route] = {"result": result, "timings": timings}
        print(f"{route}: {timings['Total']:.2f} s")

    reference = outputs[ROUTES[0]]["result"]
    if any(outputs[route]["result"] != reference for route in ROUTES[1:]):
        raise RuntimeError("The Python and C++ H-bond routes produced different results.")
    print("exact result comparison: passed")
    print(json.dumps(outputs, indent=2, sort_keys=True))


def run_profile(config: dict[str, Any], *, route: str) -> tuple[dict[str, Any], dict[str, float]]:
    if route not in ROUTES:
        raise ValueError(f"route must be one of {ROUTES}.")
    input_cfg = config["input"]
    selection_cfg = config["selection"]
    hbond_cfg = copy.deepcopy(config["hbond"])
    hbond_cfg["backend"] = route
    labels = species_labels(selection_cfg)
    classes = classes_by_species(hbond_cfg, labels)
    state = new_hbond_state(classes, labels)
    context = SelectionContext.from_input_config(input_cfg)
    traj_path = resolve_path(config, input_cfg["trajectory"])
    cell = parse_cell(config["system"].get("cell", "auto"))
    timings = {name: 0.0 for name in STAGE_NAMES}
    start = time.perf_counter()
    frames = iter_frames(traj_path, input_cfg)
    while True:
        before_read = time.perf_counter()
        try:
            frame = next(frames)
        except StopIteration:
            break
        timings["Read NPZ frames"] += time.perf_counter() - before_read
        if cell is None:
            if frame.cell is None:
                raise ValueError("No cell available.")
            cell = frame.cell
        _accumulate_profiled_frame(
            state,
            frame,
            selection_cfg,
            hbond_cfg,
            classes,
            labels,
            context,
            cell,
            timings,
            route=route,
        )
    timings["Other overhead"] += time.perf_counter() - start - sum(timings.values())
    timings["Total"] = time.perf_counter() - start
    result = {
        "counts": state.counts,
        "raw_counts": state.raw_counts,
        "samples_total": state.samples_total,
        "frames": state.frames,
    }
    return result, timings


def _accumulate_profiled_frame(
    state,
    frame: TrajectoryFrame,
    selection_cfg: dict[str, Any],
    hbond_cfg: dict[str, Any],
    classes: dict[str, list[str]],
    labels: list[str],
    context: SelectionContext,
    cell: tuple[float, float, float],
    timings: dict[str, float],
    route: str,
) -> None:
    # The kernel is split here only for benchmark attribution; the normal workflow stays unchanged.
    before_select = time.perf_counter()
    oxygen_indices = element_indices(frame, {str(selection_cfg.get("oxygen_symbol", "O"))}, context)
    hydrogen_indices = element_indices(frame, {str(selection_cfg.get("hydrogen_symbol", "H"))}, context)
    timings["Select O/H atoms"] += time.perf_counter() - before_select

    if route == "python":
        before_neighbors = time.perf_counter()
        neighbors_by_species = oxygen_hydrogen_neighbors_by_species(
            frame.symbols,
            frame.positions,
            oxygen_symbol=str(selection_cfg.get("oxygen_symbol", "O")),
            hydrogen_symbol=str(selection_cfg.get("hydrogen_symbol", "H")),
            oh_cutoff=float(selection_cfg.get("oh_cutoff", 1.25)),
            neighbor_method=str(selection_cfg.get("neighbor_method", "auto")),
            neighbor_workers=int(selection_cfg.get("neighbor_workers", 1)),
            oxygen_chunk_size=int(selection_cfg.get("oxygen_chunk_size", 2048)),
            oxygen_indices=oxygen_indices,
            hydrogen_indices=hydrogen_indices,
        )
        h_counts_by_oxygen: dict[int, int] = {}
        donors = []
        for entries in neighbors_by_species.values():
            for oxygen_index, attached_hydrogens in entries:
                h_counts_by_oxygen[oxygen_index] = int(attached_hydrogens.size)
                if attached_hydrogens.size:
                    donors.append((oxygen_index, attached_hydrogens))
        h_counts = np.asarray([h_counts_by_oxygen.get(int(index), 0) for index in oxygen_indices], dtype=int)
        timings["O-H species assignment"] += time.perf_counter() - before_neighbors
        before_geometry = time.perf_counter()
        bonds = find_hbonds(
            positions=frame.positions,
            oxygen_indices=oxygen_indices,
            donor_neighbors=donors,
            hbond_cfg=hbond_cfg,
            cell=cell,
        )
        donor_counts = np.zeros(oxygen_indices.size, dtype=int)
        acceptor_counts = np.zeros(oxygen_indices.size, dtype=int)
        local_oxygen = {int(index): local for local, index in enumerate(oxygen_indices)}
        for donor, _hydrogen, acceptor in bonds:
            donor_counts[local_oxygen[donor]] += 1
            acceptor_counts[local_oxygen[acceptor]] += 1
        timings["H-bond geometry"] += time.perf_counter() - before_geometry
    else:
        before_neighbors = time.perf_counter()
        neighbor_data = build_hbond_neighbor_matrix_cpp(
            oxygen_positions=frame.positions[oxygen_indices],
            hydrogen_positions=frame.positions[hydrogen_indices],
            cutoff=float(selection_cfg.get("oh_cutoff", 1.25)),
        )
        timings["O-H species assignment"] += time.perf_counter() - before_neighbors
        if neighbor_data is None:
            raise RuntimeError("C++ O-H neighbor backend is not available.")
        h_counts, h_matrix = neighbor_data
        before_geometry = time.perf_counter()
        geometry_counts = hbond_geometry_counts(
            frame.positions[oxygen_indices],
            frame.positions[hydrogen_indices],
            hydrogen_counts=h_counts,
            hydrogen_matrix=h_matrix,
            oo_cutoff=float(hbond_cfg.get("oo_cutoff", 3.5)),
            dha_angle_min=float(hbond_cfg.get("dha_angle_min", 150.0)),
            h_acceptor_cutoff=hbond_cfg.get("h_acceptor_cutoff"),
            cell=cell,
            pbc=tuple(bool(value) for value in hbond_cfg.get("pbc", [True, True, True])),
            max_acceptors_per_hydrogen=bool(hbond_cfg.get("max_acceptors_per_hydrogen", True)),
        )
        timings["H-bond geometry"] += time.perf_counter() - before_geometry
        if geometry_counts is None:
            raise RuntimeError("C++ H-bond geometry backend is not available.")
        donor_counts, acceptor_counts = geometry_counts
    before_grouping_kernel = time.perf_counter()
    frame_counts, frame_raw_counts, frame_samples = frame_hbond_classes_from_counts(
        oxygen_indices=oxygen_indices,
        h_counts=h_counts,
        donor_counts=donor_counts,
        acceptor_counts=acceptor_counts,
        classes=classes,
        selected_species=labels,
    )
    timings["Topology grouping"] += time.perf_counter() - before_grouping_kernel

    before_grouping = time.perf_counter()
    accumulate_frame_counts(state, frame_counts, frame_raw_counts, frame_samples)
    timings["Topology grouping"] += time.perf_counter() - before_grouping


if __name__ == "__main__":
    main()
