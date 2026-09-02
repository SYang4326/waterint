"""Config-driven H-bond-filtered, CN-resolved proton-sharing workflow."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from waterint.config import require_mapping
from waterint._01_core.coordinates import coordinate_spec_from_config, coordinate_values
from waterint._01_core.selection import SelectionContext, element_indices
from waterint._02_computation import _native
from waterint._02_computation.proton_sharing_hbond import (
    ProtonSharingHbondState, accumulate_python_pairs, free_energy_from_counts,
    new_proton_sharing_hbond_state,
)
from waterint._04_workflows.workflows.common import iter_frames, parse_cell, parse_range, required_workflow_sections, resolve_path
from waterint._04_workflows.workflows.msd import frame_cell_vectors, parse_pbc
from waterint._04_workflows.workflows.proton_sharing import assigned_hydrogens


@dataclass(frozen=True)
class ProtonHbondWorkflowResult:
    output_directory: Path
    metadata_path: Path
    frames: int
    l1_l1_pairs: int
    l1_l2_pairs: int


def run_proton_sharing_hbond(config: dict[str, Any]) -> ProtonHbondWorkflowResult:
    input_cfg, system_cfg, output_cfg = required_workflow_sections(config)
    selection_cfg = require_mapping(config, "selection")
    analysis_cfg = require_mapping(config, "proton_sharing_hbond")
    backend = str(analysis_cfg.get("backend", "auto")).lower()
    if backend not in {"auto", "python", "cpp"}:
        raise ValueError("proton_sharing_hbond.backend must be auto, python, or cpp.")
    state = new_proton_sharing_hbond_state(
        parse_range(analysis_cfg.get("delta_range_A", [-1.5, 1.5]), name="proton_sharing_hbond.delta_range_A"),
        parse_range(analysis_cfg.get("oo_range_A", [2.2, 3.2]), name="proton_sharing_hbond.oo_range_A"),
        int(analysis_cfg.get("delta_bins", 120)), int(analysis_cfg.get("oo_bins", 100)),
    )
    context = SelectionContext.from_input_config(input_cfg)
    coordinate = coordinate_spec_from_config(require_mapping(analysis_cfg, "coordinate"))
    pbc = parse_pbc(selection_cfg.get("pbc", [True, True, False]), "selection.pbc")
    configured_cell = parse_cell(system_cfg.get("cell", "auto"))
    l1_range = parse_range(analysis_cfg.get("l1_range", [1.5, 2.8]), name="proton_sharing_hbond.l1_range")
    l2_range = parse_range(analysis_cfg.get("l2_range", [2.8, 4.0]), name="proton_sharing_hbond.l2_range")
    oo_range = parse_range(analysis_cfg.get("oo_range_A", [2.2, 3.2]), name="proton_sharing_hbond.oo_range_A")
    angle_min = float(analysis_cfg.get("dha_angle_min_deg", 150.0))
    oh_cutoff = float(analysis_cfg.get("oh_assignment_cutoff_A", 1.5))
    use_cpp = backend != "python"
    previous_step = None
    trajectories = input_cfg.get("trajectories", [input_cfg["trajectory"]] if "trajectory" in input_cfg else None)
    if not isinstance(trajectories, list) or not trajectories:
        raise ValueError("input.trajectory or non-empty input.trajectories is required.")

    for trajectory in trajectories:
        for frame in iter_frames(resolve_path(config, trajectory), input_cfg):
            if frame.step is not None and frame.step == previous_step:
                continue
            previous_step = frame.step
            oxygen_indices = element_indices(frame, {str(selection_cfg.get("oxygen_symbol", "O"))}, context)
            hydrogen_indices = element_indices(frame, {str(selection_cfg.get("hydrogen_symbol", "H"))}, context)
            vectors = frame_cell_vectors(frame, configured_cell)
            cell = _orthorhombic_lengths(vectors)
            oxygen_positions, hydrogen_positions = frame.positions[oxygen_indices], frame.positions[hydrogen_indices]
            z_values = coordinate_values(frame, oxygen_indices, coordinate, context)
            assignment = _native.nearest_oh_assignment(oxygen_positions, hydrogen_positions, cutoff=oh_cutoff, cell=cell, pbc=pbc) if use_cpp else None
            if use_cpp and assignment is None:
                if backend == "cpp":
                    raise RuntimeError("C++ backend is unavailable.")
                use_cpp = False
            if use_cpp:
                coordination, hydrogen_matrix = assignment
            else:
                hydrogen_lists = assigned_hydrogens(oxygen_positions, hydrogen_positions, vectors, pbc, oh_cutoff)
                coordination = np.asarray([len(items) for items in hydrogen_lists], dtype=int)
            donor_l1 = (coordination == 2) & (z_values >= l1_range[0]) & (z_values < l1_range[1])
            acceptor_l1 = (coordination == 1) & (z_values >= l1_range[0]) & (z_values < l1_range[1])
            donor_l2 = (coordination == 2) & (z_values >= l2_range[0]) & (z_values < l2_range[1])
            acceptor_l2 = (coordination == 1) & (z_values >= l2_range[0]) & (z_values < l2_range[1])

            def accumulate(donor_mask, acceptor_mask, *, l1_l1: bool, reverse: bool = False) -> None:
                nonlocal state
                if not np.any(donor_mask) or not np.any(acceptor_mask):
                    return
                if use_cpp:
                    histogram = np.zeros_like(state.l1_l1_counts_by_cn)
                    result = _native.proton_hbond_accumulate(
                        oxygen_positions[donor_mask], oxygen_positions[acceptor_mask], hydrogen_positions,
                        coordination[donor_mask], hydrogen_matrix[donor_mask], cell=cell, pbc=pbc,
                        oo_range=oo_range, angle_min=angle_min, delta_edges=state.delta_edges_a,
                        oo_edges=state.oo_edges_a, hist=histogram,
                    )
                    if result is None:
                        raise RuntimeError("C++ backend became unavailable during frame accumulation.")
                    histogram, pairs, acceptors = result
                    if l1_l1:
                        state.l1_l1_counts_by_cn += histogram
                        state.l1_l1_pair_samples += pairs
                        state.l1_l1_acceptor_samples += acceptors
                    else:
                        values = histogram.sum(axis=0)
                        state.l1_l2_counts += values[::-1] if reverse else values
                        state.l1_l2_pair_samples += pairs
                else:
                    donor_hydrogens = [hydrogen_positions[hydrogen_lists[i]] for i in np.flatnonzero(donor_mask)]
                    accumulate_python_pairs(state, oxygen_positions[donor_mask], oxygen_positions[acceptor_mask], donor_hydrogens, vectors, pbc, is_l1_l1=l1_l1, oo_range_a=oo_range, angle_min_deg=angle_min, reverse_delta=reverse)

            if bool(analysis_cfg.get("include_l1_l1", True)):
                accumulate(donor_l1, acceptor_l1, l1_l1=True)
            if bool(analysis_cfg.get("include_l1_l2", True)):
                accumulate(donor_l1, acceptor_l2, l1_l1=False)
                accumulate(donor_l2, acceptor_l1, l1_l1=False, reverse=True)
            state.frames += 1

    if state.frames == 0:
        raise ValueError("No trajectory frames found.")
    if bool(analysis_cfg.get("l1_l1_delta_symmetrize", True)):
        state.l1_l1_counts_by_cn = (state.l1_l1_counts_by_cn + state.l1_l1_counts_by_cn[:, ::-1, :]) / 2.0
    output_directory = resolve_path(config, output_cfg.get("directory", "output"))
    output_directory.mkdir(parents=True, exist_ok=True)
    prefix = str(output_cfg.get("prefix", "proton_hbond"))
    surfaces = {"l1_l1_all": state.l1_l1_counts_by_cn[1:].sum(axis=0), "l1_l2": state.l1_l2_counts}
    if bool(analysis_cfg.get("classify_l1_acceptor_cn", True)):
        surfaces.update({f"l1_l1_CN{cn}": state.l1_l1_counts_by_cn[int(cn)] for cn in analysis_cfg.get("cn_values", [1, 2, 3, 4])})
    surface_metadata = {}
    for name, counts in surfaces.items():
        free_energy, probability = free_energy_from_counts(counts, float(analysis_cfg.get("temperature_K", 300.0)))
        path = output_directory / f"{prefix}_{name}.npz"
        np.savez_compressed(path, delta_edges_a=state.delta_edges_a, oo_edges_a=state.oo_edges_a, counts=counts, probability=probability, free_energy_kj_mol=free_energy)
        surface_metadata[name] = {"file": str(path), "pair_samples": int(counts.sum())}
    metadata_path = output_directory / f"{prefix}_metadata.json"
    metadata_path.write_text(json.dumps({"analysis_name": "proton-sharing-hbond", "backend": "cpp" if use_cpp else "python", "frames": state.frames, "pair_weighting": "unit weight per retained pair; global pooled normalization", "criteria": {"O_H_assignment_cutoff_A": oh_cutoff, "R_OO_A": list(oo_range), "D_H_A_angle_min_deg": angle_min}, "CN_definition": "distinct qualifying L1 H2O donor O atoms per L1 OH- acceptor in the same frame", "l1_l1_delta_symmetrized": bool(analysis_cfg.get("l1_l1_delta_symmetrize", True)), "l1_l1_pairs_before_symmetrization": state.l1_l1_pair_samples, "l1_l2_pairs": state.l1_l2_pair_samples, "surfaces": surface_metadata}, indent=2) + "\n")
    return ProtonHbondWorkflowResult(output_directory, metadata_path, state.frames, state.l1_l1_pair_samples, state.l1_l2_pair_samples)


def _orthorhombic_lengths(cell_vectors: np.ndarray) -> np.ndarray:
    if not np.allclose(cell_vectors, np.diag(np.diag(cell_vectors))):
        raise ValueError("proton-sharing-hbond C++ backend currently requires orthorhombic cells.")
    return np.diag(cell_vectors)
