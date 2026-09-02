from __future__ import annotations

import numpy as np

from waterint._02_computation.proton_sharing_hbond import (
    free_energy_from_counts,
    new_proton_sharing_hbond_state,
)


def test_free_energy_normalizes_global_counts():
    counts = np.array([[1.0, 3.0], [0.0, 0.0]])
    free_energy, probability = free_energy_from_counts(counts, 300.0)
    np.testing.assert_allclose(probability, [[0.25, 0.75], [0.0, 0.0]])
    assert np.isclose(np.nanmin(free_energy), 0.0)


def test_state_has_cn_one_to_four_histograms():
    state = new_proton_sharing_hbond_state((-1.5, 1.5), (2.2, 3.2), 12, 10)
    assert state.l1_l1_counts_by_cn.shape == (5, 12, 10)
    assert state.l1_l2_counts.shape == (12, 10)
