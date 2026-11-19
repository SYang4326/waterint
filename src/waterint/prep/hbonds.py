"""Geometric hydrogen bonds: the ONE definition used by every analysis.

A bond D-H...A exists when R(D,A) <= oo_cutoff, angle(D-H-A) >= dha_angle_min and (optionally)
R(H,A) <= h_acceptor_cutoff. Donor H come from prep.ownership. With max_acceptors_per_hydrogen each H
donates at most one bond (to the closest qualifying acceptor). Kernels: ``hbonds.counts`` (python, cpu).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from waterint.backends.registry import register, select
from waterint.prep.neighbors import PythonCutoffNeighborSearch
from waterint.prep.ownership import Ownership


@dataclass(frozen=True)
class HBondParams:
    oo_cutoff: float = 3.5
    dha_angle_min: float = 150.0
    h_acceptor_cutoff: float | None = None
    pbc: tuple[bool, bool, bool] = (True, True, True)
    max_acceptors_per_hydrogen: bool = True

    @classmethod
    def from_config(cls, cfg: dict[str, Any] | None) -> "HBondParams":
        cfg = cfg or {}
        pbc = cfg.get("pbc", [True, True, True])
        pbc = (pbc,) * 3 if isinstance(pbc, bool) else tuple(bool(v) for v in pbc)
        hac = cfg.get("h_acceptor_cutoff")
        params = cls(float(cfg.get("oo_cutoff", 3.5)), float(cfg.get("dha_angle_min", 150.0)),
                     None if hac in (None, "") else float(hac), pbc,  # type: ignore[arg-type]
                     bool(cfg.get("max_acceptors_per_hydrogen", True)))
        if params.oo_cutoff <= 0:
            raise ValueError("hbond.oo_cutoff must be positive.")
        if not 0 <= params.dha_angle_min <= 180:
            raise ValueError("hbond.dha_angle_min must be between 0 and 180.")
        return params


def hbond_counts(positions: np.ndarray, own: Ownership, params: HBondParams, *,
                 cell: tuple[float, float, float], backend: str = "auto") -> tuple[np.ndarray, np.ndarray, str]:
    """Donated and accepted bond counts per acceptor (in ``own.acceptor_idx`` order)."""
    n = own.acceptor_idx.size
    if n == 0 or own.proton_idx.size == 0:
        return np.zeros(n, dtype=np.int64), np.zeros(n, dtype=np.int64), "python"
    used, fn = select("hbonds.counts", backend)
    donors, acceptors = fn(positions, own, params, cell)
    return donors, acceptors, used


@register("hbonds.counts", "cpu")
def _counts_cpu(positions, own: Ownership, params: HBondParams, cell):
    from waterint.backends.cpu.native import hbond_geometry_counts
    out = hbond_geometry_counts(
        positions[own.acceptor_idx], positions[own.proton_idx],
        hydrogen_counts=own.counts, hydrogen_matrix=own.matrix,
        oo_cutoff=params.oo_cutoff, dha_angle_min=params.dha_angle_min, h_acceptor_cutoff=params.h_acceptor_cutoff,
        cell=cell, pbc=params.pbc, max_acceptors_per_hydrogen=params.max_acceptors_per_hydrogen)
    donors, acceptors = out
    return np.asarray(donors, dtype=np.int64), np.asarray(acceptors, dtype=np.int64)


@register("hbonds.counts", "python")
def _counts_python(positions, own: Ownership, params: HBondParams, cell):
    triples = find_hbonds(positions, own, params, cell)
    local = {int(a): i for i, a in enumerate(own.acceptor_idx)}
    donors = np.zeros(own.acceptor_idx.size, dtype=np.int64)
    acceptors = np.zeros(own.acceptor_idx.size, dtype=np.int64)
    for d, _h, a in triples:
        donors[local[d]] += 1
        acceptors[local[a]] += 1
    return donors, acceptors


def find_hbonds(positions: np.ndarray, own: Ownership, params: HBondParams,
                cell: tuple[float, float, float]) -> list[tuple[int, int, int]]:
    """Explicit (donor O, H, acceptor O) global-index triples (Python cell list, vectorised geometry)."""
    oxygen_indices = own.acceptor_idx
    oxygen_positions = np.ascontiguousarray(positions[oxygen_indices], dtype=float)
    search = PythonCutoffNeighborSearch(oxygen_positions, oxygen_positions, cutoff=params.oo_cutoff, cell=cell, pbc=params.pbc)
    donor_b, hyd_b, acc_b, grp_b = [], [], [], []
    group = 0
    for i, row in enumerate(own.proton_lists()):
        if row.size == 0:
            continue
        candidates = search.collect_indices(i)
        candidates = candidates[candidates != i]
        if candidates.size == 0:
            group += row.size
            continue
        for h_local in row:
            n = candidates.size
            donor_b.append(np.full(n, int(oxygen_indices[i])))
            hyd_b.append(np.full(n, int(own.proton_idx[h_local])))
            acc_b.append(candidates)
            grp_b.append(np.full(n, group))
            group += 1
    if not acc_b:
        return []
    donor = np.concatenate(donor_b); hyd = np.concatenate(hyd_b); acc = np.concatenate(acc_b); grp = np.concatenate(grp_b)
    va = search.minimum_image(oxygen_positions[acc] - positions[hyd])
    vd = search.minimum_image(positions[donor] - positions[hyd])
    ra = np.linalg.norm(va, axis=1); rd = np.linalg.norm(vd, axis=1)
    ok = (ra > 0) & (rd > 0)
    cos = np.ones(ra.size)
    cos[ok] = np.einsum("ij,ij->i", vd[ok], va[ok]) / (rd[ok] * ra[ok])
    ang = np.zeros(ra.size)
    ang[ok] = np.degrees(np.arccos(np.clip(cos[ok], -1.0, 1.0)))
    passing = ang >= params.dha_angle_min
    if params.h_acceptor_cutoff is not None:
        passing &= ra <= params.h_acceptor_cutoff
    if not np.any(passing):
        return []
    donor, hyd, acc, ra, grp = donor[passing], hyd[passing], acc[passing], ra[passing], grp[passing]
    if params.max_acceptors_per_hydrogen:
        order = np.lexsort((acc, ra, grp))
        sg = grp[order]
        first = np.r_[True, sg[1:] != sg[:-1]]
        sel = order[first]
        donor, hyd, acc = donor[sel], hyd[sel], acc[sel]
    return [(int(d), int(h), int(oxygen_indices[a])) for d, h, a in zip(donor, hyd, acc)]
