"""Local-coordinate proton-sharing free-energy surfaces for selected O-O pairs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ProtonSharingResult:
    delta_centers_a: np.ndarray
    oo_centers_a: np.ndarray
    free_energy_kj_mol: np.ndarray
    probability: np.ndarray
    counts: np.ndarray
    shared_s_centers_a: np.ndarray
    shared_rho_centers_a: np.ndarray
    shared_free_energy_kj_mol: np.ndarray
    shared_probability_density: np.ndarray
    shared_counts: np.ndarray
    pair_samples: int
    shared_samples: int


def proton_sharing_surface(
    *,
    donor_positions: np.ndarray,
    acceptor_positions: np.ndarray,
    donor_hydrogens: list[np.ndarray],
    cell_vectors: np.ndarray,
    pbc: tuple[bool, bool, bool],
    oo_range_a: tuple[float, float],
    delta_range_a: tuple[float, float],
    delta_bins: int,
    oo_bins: int,
    shared_delta_max_a: float,
    shared_s_range_a: tuple[float, float],
    shared_rho_range_a: tuple[float, float],
    shared_s_bins: int,
    shared_rho_bins: int,
    temperature_k: float,
) -> ProtonSharingResult:
    """Accumulate pairwise proton-transfer and shared-proton surfaces.

    Each donor-acceptor pair contributes the donor H that minimizes
    r(O_d,H)+r(H,O_a). Coordinates use the donor O as origin and the O_d--O_a
    vector as local z; `s` is measured from the O-O midpoint and `rho` is the
    perpendicular distance. All distances respect the supplied PBC convention.
    """
    _validate_inputs(
        donor_positions, acceptor_positions, donor_hydrogens, cell_vectors,
        oo_range_a, delta_range_a, delta_bins, oo_bins, shared_delta_max_a,
        shared_s_range_a, shared_rho_range_a, shared_s_bins, shared_rho_bins,
        temperature_k,
    )
    delta_edges = np.linspace(*delta_range_a, delta_bins + 1)
    oo_edges = np.linspace(*oo_range_a, oo_bins + 1)
    s_edges = np.linspace(*shared_s_range_a, shared_s_bins + 1)
    rho_edges = np.linspace(*shared_rho_range_a, shared_rho_bins + 1)
    pair_counts = np.zeros((delta_bins, oo_bins), dtype=np.int64)
    shared_counts = np.zeros((shared_s_bins, shared_rho_bins), dtype=np.int64)
    pair_samples = shared_samples = 0
    if _VECTORISED:
        pair_samples, shared_samples = _accumulate_vectorised(
            donor_positions, acceptor_positions, donor_hydrogens, cell_vectors, pbc, oo_range_a, shared_delta_max_a,
            delta_edges, oo_edges, s_edges, rho_edges, pair_counts, shared_counts)
        donor_positions = donor_positions[:0]                  # skip the reference loop below

    for donor, hydrogens in zip(donor_positions, donor_hydrogens):
        if len(hydrogens) == 0:
            continue
        for acceptor in acceptor_positions:
            oo_vector = minimum_image(acceptor - donor, cell_vectors, pbc)
            oo_distance = float(np.linalg.norm(oo_vector))
            if not oo_range_a[0] <= oo_distance < oo_range_a[1]:
                continue
            h_vectors = minimum_image(np.asarray(hydrogens) - donor, cell_vectors, pbc)
            h_donor = np.linalg.norm(h_vectors, axis=1)
            h_acceptor_vectors = minimum_image(np.asarray(hydrogens) - acceptor, cell_vectors, pbc)
            h_acceptor = np.linalg.norm(h_acceptor_vectors, axis=1)
            best = int(np.argmin(h_donor + h_acceptor))
            delta = float(h_donor[best] - h_acceptor[best])
            delta_index = np.searchsorted(delta_edges, delta, side="right") - 1
            oo_index = np.searchsorted(oo_edges, oo_distance, side="right") - 1
            if 0 <= delta_index < delta_bins and 0 <= oo_index < oo_bins:
                pair_counts[delta_index, oo_index] += 1
                pair_samples += 1
            if abs(delta) > shared_delta_max_a:
                continue
            direction = oo_vector / oo_distance
            h_vector = h_vectors[best]
            s = float(np.dot(h_vector, direction) - 0.5 * oo_distance)
            rho_vector = h_vector - np.dot(h_vector, direction) * direction
            rho = float(np.linalg.norm(rho_vector))
            s_index = np.searchsorted(s_edges, s, side="right") - 1
            rho_index = np.searchsorted(rho_edges, rho, side="right") - 1
            if 0 <= s_index < shared_s_bins and 0 <= rho_index < shared_rho_bins:
                shared_counts[s_index, rho_index] += 1
                shared_samples += 1

    probability = normalized_probability(pair_counts)
    free_energy = free_energy_from_probability(probability, temperature_k)
    # The s-rho histogram is a projection of a cylindrical 3D density. Divide
    # by 2*pi*rho*ds*drho before taking -kT ln P to remove the geometric Jacobian.
    ds = np.diff(s_edges)[:, None]
    rho_centers = (rho_edges[:-1] + rho_edges[1:]) / 2.0
    drho = np.diff(rho_edges)[None, :]
    jacobian = 2.0 * np.pi * rho_centers[None, :] * ds * drho
    shared_density = np.divide(shared_counts, jacobian, out=np.zeros_like(shared_counts, dtype=float), where=jacobian > 0)
    shared_probability = normalized_probability(shared_density)
    shared_free_energy = free_energy_from_probability(shared_probability, temperature_k)
    return ProtonSharingResult(
        delta_centers_a=(delta_edges[:-1] + delta_edges[1:]) / 2.0,
        oo_centers_a=(oo_edges[:-1] + oo_edges[1:]) / 2.0,
        free_energy_kj_mol=free_energy,
        probability=probability,
        counts=pair_counts,
        shared_s_centers_a=(s_edges[:-1] + s_edges[1:]) / 2.0,
        shared_rho_centers_a=rho_centers,
        shared_free_energy_kj_mol=shared_free_energy,
        shared_probability_density=shared_probability,
        shared_counts=shared_counts,
        pair_samples=pair_samples,
        shared_samples=shared_samples,
    )


_VECTORISED = True     # False = original per-pair loop (reference; identical counts, checked in tests)


def _accumulate_vectorised(donors, acceptors, donor_hydrogens, cell_vectors, pbc, oo_range, shared_delta_max,
                           delta_edges, oo_edges, s_edges, rho_edges, pair_counts, shared_counts) -> tuple[int, int]:
    """All donor-acceptor pairs of one frame at once. Same pair rule, binning and PBC convention as the loop."""
    keep = [i for i, h in enumerate(donor_hydrogens) if len(h)]
    if not keep or not len(acceptors):
        return 0, 0
    donors = np.asarray(donors, dtype=float)[keep]
    acceptors = np.asarray(acceptors, dtype=float)
    kmax = max(len(donor_hydrogens[i]) for i in keep)
    H = np.full((len(keep), kmax, 3), np.nan)
    for j, i in enumerate(keep):
        H[j, :len(donor_hydrogens[i])] = np.asarray(donor_hydrogens[i], dtype=float)
    oo = minimum_image(acceptors[None, :, :] - donors[:, None, :], cell_vectors, pbc)          # (D, A, 3)
    r_oo = np.linalg.norm(oo, axis=-1)
    d_idx, a_idx = np.nonzero((oo_range[0] <= r_oo) & (r_oo < oo_range[1]))
    if d_idx.size == 0:
        return 0, 0
    hv = minimum_image(H - donors[:, None, :], cell_vectors, pbc)                                # (D, K, 3)
    hd = np.linalg.norm(hv, axis=-1)                                                            # (D, K)
    ha = np.linalg.norm(minimum_image(H[d_idx] - acceptors[a_idx][:, None, :], cell_vectors, pbc), axis=-1)  # (P, K)
    tot = np.where(np.isnan(hd[d_idx] + ha), np.inf, hd[d_idx] + ha)
    best = np.argmin(tot, axis=1)
    rows = np.arange(d_idx.size)
    delta = hd[d_idx, best] - ha[rows, best]
    dist = r_oo[d_idx, a_idx]
    di = np.searchsorted(delta_edges, delta, side="right") - 1
    oi = np.searchsorted(oo_edges, dist, side="right") - 1
    ok = (di >= 0) & (di < pair_counts.shape[0]) & (oi >= 0) & (oi < pair_counts.shape[1])
    np.add.at(pair_counts, (di[ok], oi[ok]), 1)
    sh = np.abs(delta) <= shared_delta_max
    direction = oo[d_idx[sh], a_idx[sh]] / dist[sh][:, None]
    h = hv[d_idx[sh], best[sh]]
    proj = np.einsum("ij,ij->i", h, direction)
    s = proj - 0.5 * dist[sh]
    rho = np.linalg.norm(h - proj[:, None] * direction, axis=-1)
    si = np.searchsorted(s_edges, s, side="right") - 1
    ri = np.searchsorted(rho_edges, rho, side="right") - 1
    ok2 = (si >= 0) & (si < shared_counts.shape[0]) & (ri >= 0) & (ri < shared_counts.shape[1])
    np.add.at(shared_counts, (si[ok2], ri[ok2]), 1)
    return int(ok.sum()), int(ok2.sum())


def minimum_image(vectors: np.ndarray, cell_vectors: np.ndarray, pbc: tuple[bool, bool, bool]) -> np.ndarray:
    inverse = np.linalg.inv(cell_vectors)
    fractional = np.asarray(vectors, dtype=float) @ inverse
    for axis, enabled in enumerate(pbc):
        if enabled:
            fractional[..., axis] -= np.rint(fractional[..., axis])
    return fractional @ cell_vectors


def normalized_probability(values: np.ndarray) -> np.ndarray:
    total = float(np.sum(values))
    return np.asarray(values, dtype=float) / total if total > 0 else np.zeros_like(values, dtype=float)


def free_energy_from_probability(probability: np.ndarray, temperature_k: float) -> np.ndarray:
    result = np.full_like(probability, np.nan, dtype=float)
    positive = probability > 0
    if np.any(positive):
        result[positive] = -0.008314462618 * temperature_k * np.log(probability[positive])
        result[positive] -= np.nanmin(result[positive])
    return result


def _validate_inputs(
    donor_positions: np.ndarray, acceptor_positions: np.ndarray, donor_hydrogens: list[np.ndarray], cell_vectors: np.ndarray,
    oo_range: tuple[float, float], delta_range: tuple[float, float], delta_bins: int, oo_bins: int, shared_delta: float,
    s_range: tuple[float, float], rho_range: tuple[float, float], s_bins: int, rho_bins: int, temperature: float,
) -> None:
    if donor_positions.ndim != 2 or donor_positions.shape[1] != 3 or acceptor_positions.ndim != 2 or acceptor_positions.shape[1] != 3:
        raise ValueError("donor_positions and acceptor_positions must have shape (n, 3).")
    if len(donor_positions) != len(donor_hydrogens) or cell_vectors.shape != (3, 3):
        raise ValueError("Each donor requires a hydrogen list and cell_vectors must have shape (3, 3).")
    if not (oo_range[1] > oo_range[0] > 0 and delta_range[1] > delta_range[0] and s_range[1] > s_range[0] and rho_range[1] > rho_range[0] >= 0):
        raise ValueError("All ranges must be strictly increasing; O-O lower bound must be positive.")
    if min(delta_bins, oo_bins, s_bins, rho_bins) <= 0 or shared_delta <= 0 or temperature <= 0:
        raise ValueError("Bin counts, shared_delta_max_A, and temperature_K must be positive.")
