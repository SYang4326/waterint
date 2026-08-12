"""Workflow for a layer- and species-selective proton-sharing FES."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from waterint.config import require_mapping
from waterint._01_core.coordinates import coordinate_spec_from_config, coordinate_values
from waterint._01_core.selection import SelectionContext, element_indices
from waterint._02_computation.proton_sharing import ProtonSharingResult, proton_sharing_surface
from waterint._03_output.metadata import write_metadata
from waterint._03_output.proton_sharing import plot_proton_sharing, write_proton_sharing_csv, write_shared_proton_csv
from waterint._04_workflows.workflows.common import iter_frames, parse_cell, parse_range, required_workflow_sections, resolve_path
from waterint._04_workflows.workflows.msd import frame_cell_vectors, parse_pbc


@dataclass(frozen=True)
class ProtonSharingWorkflowResult:
    fes_csv_path: Path
    shared_csv_path: Path
    png_path: Path | None
    metadata_path: Path
    pair_samples: int
    shared_samples: int


def run_proton_sharing(config: dict[str, Any]) -> ProtonSharingWorkflowResult:
    input_cfg, system_cfg, output_cfg = required_workflow_sections(config)
    selection_cfg = require_mapping(config, "selection")
    sharing_cfg = require_mapping(config, "proton_sharing")
    donor_cfg = require_mapping(sharing_cfg, "donor")
    acceptor_cfg = require_mapping(sharing_cfg, "acceptor")
    include_swapped_state = bool(sharing_cfg.get("include_swapped_state", True))
    if include_swapped_state:
        for key, default in (("delta_range_A", [-1.5, 1.5]), ("shared_s_range_A", [-1.5, 1.5])):
            lower, upper = parse_range(sharing_cfg.get(key, default), name=f"proton_sharing.{key}")
            if not np.isclose(lower, -upper):
                raise ValueError(f"proton_sharing.{key} must be symmetric about zero when include_swapped_state is true.")
    context = SelectionContext.from_input_config(input_cfg)
    trajectory = resolve_path(config, input_cfg["trajectory"])
    configured_cell = parse_cell(system_cfg.get("cell", "auto"))
    pbc = parse_pbc(selection_cfg.get("pbc", [True, True, True]), "selection.pbc")
    all_pair_counts = None
    all_shared_counts = None
    pair_samples = shared_samples = frames = 0
    for frame in iter_frames(trajectory, input_cfg):
        oxygen, hydrogens, coordination, z_values = classify_frame(
            frame, context, selection_cfg, sharing_cfg, configured_cell
        )
        donor_mask = species_layer_mask(coordination, z_values, donor_cfg)
        acceptor_mask = species_layer_mask(coordination, z_values, acceptor_cfg)
        cell = frame_cell_vectors(frame, configured_cell)
        hydrogen_lists = assigned_hydrogens(
            frame.positions[oxygen], frame.positions[hydrogens], cell, pbc,
            float(selection_cfg.get("oh_cutoff", 1.25)),
        )
        donor_hydrogens = (
            [frame.positions[hydrogens][hydrogen_lists[index]] for index in np.flatnonzero(donor_mask)]
        )
        frame_result = surface_for_selection(
            frame.positions[oxygen][donor_mask], frame.positions[oxygen][acceptor_mask], donor_hydrogens,
            cell, pbc, sharing_cfg,
        )
        frame_pair_counts, frame_shared_counts = frame_result.counts, frame_result.shared_counts
        frame_pair_samples, frame_shared_samples = frame_result.pair_samples, frame_result.shared_samples
        if include_swapped_state:
            # Select the exchanged state: H2O remains the proton donor, now on
            # the opposite layer, while OH- is the acceptor on the original
            # donor layer. The resulting L2 -> L1 coordinate is sign-reversed.
            swapped_donor_cfg = {**acceptor_cfg, "species": donor_cfg["species"]}
            swapped_acceptor_cfg = {**donor_cfg, "species": acceptor_cfg["species"]}
            swapped_donor_mask = species_layer_mask(coordination, z_values, swapped_donor_cfg)
            swapped_acceptor_mask = species_layer_mask(coordination, z_values, swapped_acceptor_cfg)
            swapped_hydrogens = [
                frame.positions[hydrogens][hydrogen_lists[index]] for index in np.flatnonzero(swapped_donor_mask)
            ]
            swapped = surface_for_selection(
                frame.positions[oxygen][swapped_donor_mask], frame.positions[oxygen][swapped_acceptor_mask],
                swapped_hydrogens, cell, pbc, sharing_cfg,
            )
            # The swapped calculation is L2 -> L1.  Reverse delta and s to
            # preserve the reported L1 -> L2 coordinate convention.
            frame_pair_counts = frame_pair_counts + swapped.counts[::-1, :]
            frame_shared_counts = frame_shared_counts + swapped.shared_counts[::-1, :]
            frame_pair_samples += swapped.pair_samples
            frame_shared_samples += swapped.shared_samples
        all_pair_counts = frame_pair_counts if all_pair_counts is None else all_pair_counts + frame_pair_counts
        all_shared_counts = frame_shared_counts if all_shared_counts is None else all_shared_counts + frame_shared_counts
        pair_samples += frame_pair_samples
        shared_samples += frame_shared_samples
        frames += 1
    if not frames:
        raise ValueError("Proton-sharing analysis found no trajectory frames.")
    result = proton_sharing_surface_from_counts(
        all_pair_counts, all_shared_counts, sharing_cfg, pair_samples, shared_samples
    )
    outdir = resolve_path(config, output_cfg.get("directory", "output"))
    outdir.mkdir(parents=True, exist_ok=True)
    prefix = str(output_cfg.get("prefix", "proton_sharing"))
    fes_path, shared_path = outdir / f"{prefix}.csv", outdir / f"{prefix}_shared.csv"
    png_path = outdir / f"{prefix}.png" if bool(output_cfg.get("plot", True)) else None
    metadata_path = outdir / f"{prefix}_metadata.json"
    write_proton_sharing_csv(fes_path, result)
    write_shared_proton_csv(shared_path, result)
    if png_path is not None:
        plot_proton_sharing(png_path, result, title=str(output_cfg.get("title", "Proton-sharing free-energy surface")), dpi=int(output_cfg.get("dpi", 220)))
    write_metadata(metadata_path, {"analysis_name": "proton-sharing", "package": "waterint", "config_file": config.get("_config_path"), "frames": frames, "pair_samples": result.pair_samples, "shared_samples": result.shared_samples, "outputs": {"fes_csv": str(fes_path), "shared_csv": str(shared_path), "png": None if png_path is None else str(png_path)}, "config": {key: value for key, value in config.items() if not key.startswith("_")}})
    return ProtonSharingWorkflowResult(fes_path, shared_path, png_path, metadata_path, result.pair_samples, result.shared_samples)


def classify_frame(frame, context: SelectionContext, selection_cfg: dict[str, Any], sharing_cfg: dict[str, Any], configured_cell=None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    oxygen = element_indices(frame, {str(selection_cfg.get("oxygen_symbol", "O"))}, context)
    hydrogens = element_indices(frame, {str(selection_cfg.get("hydrogen_symbol", "H"))}, context)
    vectors = frame_cell_vectors(frame, configured_cell)
    pbc = parse_pbc(selection_cfg.get("pbc", [True, True, True]), "selection.pbc")
    assignments = assigned_hydrogens(frame.positions[oxygen], frame.positions[hydrogens], vectors, pbc, float(selection_cfg.get("oh_cutoff", 1.25)))
    coordination = np.asarray([len(items) for items in assignments], dtype=int)
    coordinate = coordinate_spec_from_config(require_mapping(sharing_cfg, "coordinate"))
    z_values = coordinate_values(frame, oxygen, coordinate, context)
    return oxygen, hydrogens, coordination, z_values


def assigned_hydrogens(oxygen_positions: np.ndarray, hydrogen_positions: np.ndarray, cell_vectors: np.ndarray, pbc: tuple[bool, bool, bool], cutoff: float) -> list[list[int]]:
    """Assign each H once to its nearest O inside the cutoff.

    The common orthorhombic case uses periodically imaged KD-tree sites rather
    than constructing the full H-by-O distance matrix.  The matrix path keeps
    the same minimum-image semantics for arbitrary cell vectors.
    """
    assignments = [[] for _ in range(len(oxygen_positions))]
    if not len(oxygen_positions) or not len(hydrogen_positions):
        return assignments
    if np.allclose(cell_vectors, np.diag(np.diag(cell_vectors))):
        try:
            from scipy.spatial import cKDTree

            shifts = np.asarray([
                [ix * cell_vectors[0, 0], iy * cell_vectors[1, 1], iz * cell_vectors[2, 2]]
                for ix in (-1, 0, 1) if pbc[0] or ix == 0
                for iy in (-1, 0, 1) if pbc[1] or iy == 0
                for iz in (-1, 0, 1) if pbc[2] or iz == 0
            ])
            sites = np.concatenate([oxygen_positions + shift for shift in shifts])
            original = np.tile(np.arange(len(oxygen_positions)), len(shifts))
            distances, nearest = cKDTree(sites).query(hydrogen_positions, distance_upper_bound=cutoff)
            for h_index, (distance, site_index) in enumerate(zip(distances, nearest)):
                if np.isfinite(distance):
                    assignments[int(original[int(site_index)])].append(h_index)
            return assignments
        except ImportError:
            pass
    fractional = (hydrogen_positions[:, None, :] - oxygen_positions[None, :, :]) @ np.linalg.inv(cell_vectors)
    for axis, enabled in enumerate(pbc):
        if enabled:
            fractional[..., axis] -= np.rint(fractional[..., axis])
    distances = np.linalg.norm(fractional @ cell_vectors, axis=2)
    closest = np.argmin(distances, axis=1)
    valid = np.min(distances, axis=1) <= cutoff
    for oxygen_index in range(len(oxygen_positions)):
        assignments[oxygen_index] = np.flatnonzero(valid & (closest == oxygen_index)).tolist()
    return assignments


def species_layer_mask(coordination: np.ndarray, z_values: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    species = str(cfg["species"])
    expected = {"OH-": 1, "H2O": 2}.get(species)
    if expected is None:
        raise ValueError("proton_sharing donor/acceptor species must be OH- or H2O.")
    low, high = parse_range(cfg["range"], name="proton_sharing donor/acceptor range")
    return (coordination == expected) & (z_values >= low) & (z_values < high)


def surface_for_selection(donors, acceptors, donor_hydrogens, cell, pbc, cfg: dict[str, Any]) -> ProtonSharingResult:
    return proton_sharing_surface(
        donor_positions=donors,
        acceptor_positions=acceptors,
        donor_hydrogens=donor_hydrogens,
        cell_vectors=cell,
        pbc=pbc,
        oo_range_a=parse_range(cfg.get("oo_range_A", [2.3, 3.6]), name="proton_sharing.oo_range_A"),
        delta_range_a=parse_range(cfg.get("delta_range_A", [-1.5, 1.5]), name="proton_sharing.delta_range_A"),
        delta_bins=int(cfg.get("delta_bins", 120)),
        oo_bins=int(cfg.get("oo_bins", 130)),
        shared_delta_max_a=float(cfg.get("shared_delta_max_A", 0.15)),
        shared_s_range_a=parse_range(cfg.get("shared_s_range_A", [-1.5, 1.5]), name="proton_sharing.shared_s_range_A"),
        shared_rho_range_a=parse_range(cfg.get("shared_rho_range_A", [0.0, 1.5]), name="proton_sharing.shared_rho_range_A"),
        shared_s_bins=int(cfg.get("shared_s_bins", 120)),
        shared_rho_bins=int(cfg.get("shared_rho_bins", 75)),
        temperature_k=float(cfg["temperature_K"]),
    )


def accumulate_frames(donors, acceptors, hydrogens, cells, pbc, cfg: dict[str, Any]) -> ProtonSharingResult:
    all_pair_counts = None
    all_shared_counts = None
    pair_samples = shared_samples = 0
    for donor, acceptor, donor_h, cell in zip(donors, acceptors, hydrogens, cells):
        result = proton_sharing_surface(donor_positions=donor, acceptor_positions=acceptor, donor_hydrogens=donor_h, cell_vectors=cell, pbc=pbc, oo_range_a=parse_range(cfg.get("oo_range_A", [2.3, 3.6]), name="proton_sharing.oo_range_A"), delta_range_a=parse_range(cfg.get("delta_range_A", [-1.5, 1.5]), name="proton_sharing.delta_range_A"), delta_bins=int(cfg.get("delta_bins", 120)), oo_bins=int(cfg.get("oo_bins", 130)), shared_delta_max_a=float(cfg.get("shared_delta_max_A", 0.15)), shared_s_range_a=parse_range(cfg.get("shared_s_range_A", [-1.5, 1.5]), name="proton_sharing.shared_s_range_A"), shared_rho_range_a=parse_range(cfg.get("shared_rho_range_A", [0.0, 1.5]), name="proton_sharing.shared_rho_range_A"), shared_s_bins=int(cfg.get("shared_s_bins", 120)), shared_rho_bins=int(cfg.get("shared_rho_bins", 75)), temperature_k=float(cfg["temperature_K"]))
        all_pair_counts = result.counts if all_pair_counts is None else all_pair_counts + result.counts
        all_shared_counts = result.shared_counts if all_shared_counts is None else all_shared_counts + result.shared_counts
        pair_samples += result.pair_samples
        shared_samples += result.shared_samples
    # One final conversion prevents averaging per-frame free energies.
    return proton_sharing_surface_from_counts(all_pair_counts, all_shared_counts, cfg, pair_samples, shared_samples)


def proton_sharing_surface_from_counts(pair_counts, shared_counts, cfg, pair_samples, shared_samples) -> ProtonSharingResult:
    from waterint._02_computation.proton_sharing import free_energy_from_probability, normalized_probability
    delta_edges = np.linspace(*parse_range(cfg.get("delta_range_A", [-1.5, 1.5]), name="proton_sharing.delta_range_A"), int(cfg.get("delta_bins", 120)) + 1)
    oo_edges = np.linspace(*parse_range(cfg.get("oo_range_A", [2.3, 3.6]), name="proton_sharing.oo_range_A"), int(cfg.get("oo_bins", 130)) + 1)
    s_edges = np.linspace(*parse_range(cfg.get("shared_s_range_A", [-1.5, 1.5]), name="proton_sharing.shared_s_range_A"), int(cfg.get("shared_s_bins", 120)) + 1)
    rho_edges = np.linspace(*parse_range(cfg.get("shared_rho_range_A", [0.0, 1.5]), name="proton_sharing.shared_rho_range_A"), int(cfg.get("shared_rho_bins", 75)) + 1)
    rho = (rho_edges[:-1] + rho_edges[1:]) / 2
    jacobian = 2 * np.pi * rho[None, :] * np.diff(s_edges)[:, None] * np.diff(rho_edges)[None, :]
    probability = normalized_probability(pair_counts)
    shared_probability = normalized_probability(np.divide(shared_counts, jacobian, out=np.zeros_like(shared_counts, dtype=float), where=jacobian > 0))
    temperature = float(cfg["temperature_K"])
    return ProtonSharingResult((delta_edges[:-1] + delta_edges[1:]) / 2, (oo_edges[:-1] + oo_edges[1:]) / 2, free_energy_from_probability(probability, temperature), probability, pair_counts, (s_edges[:-1] + s_edges[1:]) / 2, rho, free_energy_from_probability(shared_probability, temperature), shared_probability, shared_counts, pair_samples, shared_samples)
