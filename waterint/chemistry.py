from __future__ import annotations

import numpy as np


OXYGEN_SPECIES_BY_H_COUNT = {
    0: "O2-",
    1: "OH-",
    2: "H2O",
    3: "H3O+",
}


def classify_oxygen_by_h_count(
    symbols: list[str],
    positions: np.ndarray,
    *,
    oxygen_symbol: str = "O",
    hydrogen_symbol: str = "H",
    oh_cutoff: float = 1.25,
) -> dict[str, np.ndarray]:
    """Classify oxygen atoms by the number of hydrogens within a cutoff.

    This is a geometry-based first implementation. It is intentionally simple:
    no bonding history, no PBC minimum-image convention, and no special handling
    of shared protons. Later modules can replace this with a richer species
    assignment protocol while keeping the density API stable.
    """
    if oh_cutoff <= 0:
        raise ValueError("oh_cutoff must be positive.")

    symbols_array = np.asarray(symbols)
    oxygen_indices = np.where(symbols_array == oxygen_symbol)[0]
    hydrogen_indices = np.where(symbols_array == hydrogen_symbol)[0]

    out: dict[str, list[int]] = {
        "O2-": [],
        "OH-": [],
        "H2O": [],
        "H3O+": [],
        "O_other": [],
    }
    if oxygen_indices.size == 0:
        return {key: np.asarray(value, dtype=int) for key, value in out.items()}

    if hydrogen_indices.size == 0:
        out["O2-"] = oxygen_indices.tolist()
        return {key: np.asarray(value, dtype=int) for key, value in out.items()}

    oxygen_positions = positions[oxygen_indices]
    hydrogen_positions = positions[hydrogen_indices]
    d = hydrogen_positions[:, None, :] - oxygen_positions[None, :, :]
    dist2 = np.sum(d * d, axis=2)
    h_counts = np.sum(dist2 <= oh_cutoff * oh_cutoff, axis=0)

    for oxygen_index, h_count in zip(oxygen_indices, h_counts):
        label = OXYGEN_SPECIES_BY_H_COUNT.get(int(h_count), "O_other")
        out[label].append(int(oxygen_index))

    return {key: np.asarray(value, dtype=int) for key, value in out.items()}
