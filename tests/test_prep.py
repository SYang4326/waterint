"""prep layer: ownership/H-bond kernels agree across backends; species, solid and reference behave."""
from __future__ import annotations

import numpy as np
import pytest

from synthetic import TYPE_MAP, make_frames
from waterint.backends import registry as kr
from waterint.io.frame import TrajectoryFrame
from waterint.prep.hbonds import HBondParams
from waterint.prep.labelled import LabelledFrame, PrepContext
from waterint.system.profile import OwnershipParams, load_system_profile

needs_cpu = pytest.mark.skipif(not kr.cpu_available(), reason="native library not available")


def _lf(profile="generic_oxide_water", seed=0, backend="auto", triclinic=False):
    frames, types, box = make_frames(n_frames=1, seed=seed)
    vectors = np.diag(box)
    if triclinic:
        vectors = np.array([[box[0], 0, 0], [1.5, box[1], 0], [0, 0, box[2]]])
    frame = TrajectoryFrame(index=0, comment="", symbols=[TYPE_MAP[t] for t in types], positions=frames[0],
                            cell=tuple(box), types=types, cell_vectors=vectors if triclinic else None, triclinic=triclinic)
    prof = load_system_profile(profile)
    return LabelledFrame(frame, PrepContext(profile=prof, type_map=dict(TYPE_MAP), backend=backend))


@needs_cpu
@pytest.mark.parametrize("mode", ["nearest", "all_within_cutoff"])
@pytest.mark.parametrize("pbc", [(True, True, False), None])
def test_ownership_python_equals_cpu(mode, pbc):
    for seed in range(3):
        params = OwnershipParams(cutoff=1.25, mode=mode, pbc=pbc)
        a = _lf(seed=seed, backend="python").ownership(params)
        b = _lf(seed=seed, backend="cpu").ownership(params)
        assert a.backend == "python" and b.backend == "cpu"
        np.testing.assert_array_equal(a.counts, b.counts)
        for ra, rb in zip(a.proton_lists(), b.proton_lists()):
            np.testing.assert_array_equal(np.sort(ra), np.sort(rb))


@needs_cpu
def test_ownership_triclinic_nearest_python_equals_cpu():
    params = OwnershipParams(cutoff=1.25, mode="nearest", pbc=(True, True, False))
    a = _lf(backend="python", triclinic=True).ownership(params)
    b = _lf(backend="cpu", triclinic=True).ownership(params)
    np.testing.assert_array_equal(a.owner(), b.owner())


def test_pbc_matters_for_boundary_molecules():
    lf = _lf(backend="python")
    with_pbc = lf.ownership(OwnershipParams(cutoff=1.25, mode="nearest", pbc=(True, True, False)))
    no_pbc = lf.ownership(OwnershipParams(cutoff=1.25, mode="nearest", pbc=None))
    assert with_pbc.counts.sum() > no_pbc.counts.sum()      # boundary O-H bonds are only found with PBC


def test_every_proton_owned_once_in_nearest_mode():
    own = _lf(backend="python").ownership()
    owner = own.owner()
    assert (owner >= 0).all()
    assert own.counts.sum() == own.proton_idx.size


def test_species_and_dynamic_solid():
    lf = _lf()
    species = lf.species()
    n_o = lf.indices(("O",)).size
    assert sum(v.size for v in species.values()) == n_o
    assert species["OH-"].size >= 2 and species["H3O+"].size == 1
    solid = lf.solid_mask()
    assert solid[lf.indices(("Mg",))].all()                   # cations are solid
    assert not solid[species["H2O"]].any()                     # water is not
    z_ref = lf.reference()
    top_solid = lf.positions[solid, 2].max()
    assert top_solid - 0.7 <= z_ref <= top_solid


@needs_cpu
def test_hbond_counts_python_equals_cpu():
    hb = HBondParams()
    params = OwnershipParams(cutoff=1.25, mode="all_within_cutoff", pbc=None)
    for seed in range(3):
        a = _lf(seed=seed, backend="python").hbond_counts(hb, params)
        b = _lf(seed=seed, backend="cpu").hbond_counts(hb, params)
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])


def test_prep_products_are_cached():
    lf = _lf()
    assert lf.ownership() is lf.ownership()
    assert lf.species() is lf.species()


def test_backend_request_errors_are_explicit(monkeypatch):
    with pytest.raises(RuntimeError):
        kr.select("density.histogram", "gpu")
    with pytest.raises(ValueError):
        kr.normalize("fortran")
