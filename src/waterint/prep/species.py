"""Element selection, solid identity and acceptor species labels (all driven by the SystemProfile)."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from waterint.io.frame import TrajectoryFrame
from waterint.prep.ownership import Ownership
from waterint.system.profile import SystemProfile

SOLID_TOKEN = "@solid"


def element_mask(frame: TrajectoryFrame, elements: Sequence[str], type_map: dict[int, str]) -> np.ndarray:
    """Atoms whose element is in ``elements``; LAMMPS types are mapped through ``type_map`` when given."""
    wanted = {str(e) for e in elements}
    if frame.types is not None and type_map:
        type_ids = [t for t, sym in type_map.items() if sym in wanted]
        if type_ids:
            return np.isin(frame.types, type_ids)
    return np.isin(np.asarray(frame.symbols), list(wanted))


def frame_symbols(frame: TrajectoryFrame, type_map: dict[int, str]) -> np.ndarray:
    if frame.types is not None and type_map:
        lookup = np.array([type_map.get(int(t), str(t)) for t in np.unique(frame.types)])
        return lookup[np.searchsorted(np.unique(frame.types), frame.types)]
    return np.asarray(frame.symbols)


def dynamic_solid_mask(symbols: np.ndarray, own: Ownership, profile: SystemProfile) -> np.ndarray:
    """Dynamic solid: every atom that is neither proton nor acceptor, plus acceptors that own no proton."""
    roles = set(profile.ownership.protons) | set(profile.ownership.acceptors)
    mask = ~np.isin(symbols, list(roles))
    mask[own.acceptor_idx[own.counts == 0]] = True
    return mask


def frozen_solid_mask(positions: np.ndarray, symbols: np.ndarray, profile: SystemProfile, z_ref: float) -> np.ndarray:
    """Frozen solid (v0.6 rule): acceptors of the solid elements with z <= z_ref + max_z_rel at frame 0."""
    solid = profile.solid
    elements = list(solid.get("elements", []))
    mask = np.isin(symbols, elements) & (positions[:, profile.normal_axis] <= z_ref + float(solid.get("max_z_rel_A", 0.7)))
    expected = solid.get("expected_count")
    if expected is not None and int(mask.sum()) != int(expected):
        raise ValueError(f"solid selection found {int(mask.sum())} atoms, profile expects {expected}.")
    return mask


def species_by_label(own: Ownership, profile_free: Sequence[str], profile_solid: Sequence[str] | None,
                     solid_mask: np.ndarray | None) -> dict[str, np.ndarray]:
    """Acceptor indices grouped by species label (acceptor order preserved within each label).
    Free acceptors use the free table by proton count; acceptors in ``solid_mask`` use the solid table."""
    free = list(profile_free)
    solid_tab = list(profile_solid) if profile_solid else None
    labels = free + [l for l in (solid_tab or []) if l not in free]
    nh_free = np.minimum(own.counts, len(free) - 1)
    lab = np.asarray(free, dtype=object)[nh_free]
    if solid_mask is not None and solid_tab is not None:
        in_solid = solid_mask[own.acceptor_idx]
        nh_s = np.minimum(own.counts[in_solid], len(solid_tab) - 1)
        lab[in_solid] = np.asarray(solid_tab, dtype=object)[nh_s]
    return {label: own.acceptor_idx[lab == label] for label in labels}
