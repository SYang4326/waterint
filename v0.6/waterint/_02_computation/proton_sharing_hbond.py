"""Pure numerical helpers for H-bond-filtered proton-sharing PES analysis."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .proton_sharing import minimum_image

KJ_MOL_K = 0.008314462618
CN_MAX = 4


@dataclass
class ProtonSharingHbondState:
    delta_edges_a: np.ndarray
    oo_edges_a: np.ndarray
    l1_l1_counts_by_cn: np.ndarray
    l1_l2_counts: np.ndarray
    l1_l1_pair_samples: int = 0
    l1_l2_pair_samples: int = 0
    l1_l1_acceptor_samples: int = 0
    frames: int = 0


def new_proton_sharing_hbond_state(delta_range_a: tuple[float, float], oo_range_a: tuple[float, float], delta_bins: int, oo_bins: int) -> ProtonSharingHbondState:
    delta_edges = np.linspace(*delta_range_a, int(delta_bins) + 1)
    oo_edges = np.linspace(*oo_range_a, int(oo_bins) + 1)
    return ProtonSharingHbondState(delta_edges, oo_edges, np.zeros((CN_MAX + 1, delta_bins, oo_bins)), np.zeros((delta_bins, oo_bins)))


def free_energy_from_counts(counts: np.ndarray, temperature_k: float) -> tuple[np.ndarray, np.ndarray]:
    probability = np.asarray(counts, dtype=float)
    total = probability.sum()
    if total:
        probability = probability / total
    free_energy = np.full_like(probability, np.nan)
    present = probability > 0
    if np.any(present):
        free_energy[present] = -KJ_MOL_K * temperature_k * np.log(probability[present])
        free_energy[present] -= np.nanmin(free_energy[present])
    return free_energy, probability


def accumulate_python_pairs(state: ProtonSharingHbondState, donor_positions: np.ndarray, acceptor_positions: np.ndarray, donor_hydrogens: list[np.ndarray], cell_vectors: np.ndarray, pbc: tuple[bool, bool, bool], *, is_l1_l1: bool, oo_range_a: tuple[float, float], angle_min_deg: float, reverse_delta: bool = False) -> None:
    """Reference path: retain every passing pair at a unit statistical weight."""
    for acceptor in acceptor_positions:
        pairs = _qualified_pairs(donor_positions, acceptor, donor_hydrogens, cell_vectors, pbc, oo_range_a, angle_min_deg)
        if is_l1_l1:
            cn = min(len(pairs), CN_MAX)
            if cn:
                state.l1_l1_acceptor_samples += 1
            for delta, oo in pairs:
                _add_sample(state.l1_l1_counts_by_cn[cn], state, delta, oo)
                state.l1_l1_pair_samples += 1
        else:
            for delta, oo in pairs:
                _add_sample(state.l1_l2_counts, state, -delta if reverse_delta else delta, oo)
                state.l1_l2_pair_samples += 1


def _qualified_pairs(donors, acceptor, donor_hydrogens, cell, pbc, oo_range, angle_min):
    pairs: list[tuple[float, float]] = []
    for donor, hydrogens in zip(donors, donor_hydrogens):
        if len(hydrogens) == 0:
            continue
        oo = float(np.linalg.norm(minimum_image(acceptor - donor, cell, pbc)))
        if not oo_range[0] <= oo < oo_range[1]:
            continue
        donor_vectors = minimum_image(np.asarray(hydrogens) - donor, cell, pbc)
        acceptor_vectors = minimum_image(np.asarray(hydrogens) - acceptor, cell, pbc)
        donor_distances = np.linalg.norm(donor_vectors, axis=1)
        acceptor_distances = np.linalg.norm(acceptor_vectors, axis=1)
        valid = (donor_distances > 0) & (acceptor_distances > 0)
        cosines = np.ones(len(donor_distances))
        cosines[valid] = np.einsum("ij,ij->i", donor_vectors[valid], acceptor_vectors[valid]) / (donor_distances[valid] * acceptor_distances[valid])
        passing = np.flatnonzero(valid & (np.degrees(np.arccos(np.clip(cosines, -1.0, 1.0))) >= angle_min))
        if passing.size:
            best = int(passing[np.argmin((donor_distances + acceptor_distances)[passing])])
            pairs.append((float(donor_distances[best] - acceptor_distances[best]), oo))
    return pairs


def _add_sample(histogram: np.ndarray, state: ProtonSharingHbondState, delta: float, oo: float) -> None:
    delta_bin = np.searchsorted(state.delta_edges_a, delta, side="right") - 1
    oo_bin = np.searchsorted(state.oo_edges_a, oo, side="right") - 1
    if 0 <= delta_bin < histogram.shape[0] and 0 <= oo_bin < histogram.shape[1]:
        histogram[delta_bin, oo_bin] += 1
