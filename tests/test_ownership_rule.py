"""The canonical ownership rule (ownership.hpp / prep.ownership) and its users (SFG, relabel)."""
from __future__ import annotations

import numpy as np
import pytest

from synthetic import TYPE_MAP, make_frames, write_lammpstrj
from waterint.backends import registry as kr
from waterint.prep.ownership import nearest_owner_python

needs_cpu = pytest.mark.skipif(not kr.cpu_available(), reason="native library not available")
CELL = (10.0, 10.0, 30.0)


def _both(acc, pro, **kw):
    out = [nearest_owner_python(acc, pro, cell=CELL, cell_vectors=None, pbc=(True, True, False), **kw)]
    if kr.cpu_available():
        from waterint.backends.cpu.native import nearest_owner
        out.append(nearest_owner(acc, pro, cell=CELL, pbc=(True, True, False), fallback_nearest=not np.isfinite(kw["cutoff"]), **kw))
    return out


def test_exact_tie_goes_to_lowest_index():
    acc = np.array([[5.0, 5.0, 5.0], [3.0, 5.0, 5.0], [4.0, 6.0, 5.0]])
    pro = np.array([[4.0, 5.0, 5.0]])            # 1.0 A from acceptors 0 and 1, 1.0 A from 2 as well
    for owner, tied in _both(acc, pro, cutoff=1.5, tie_tolerance=1e-8):
        assert owner[0] == 0 and tied[0] == 3


def test_tolerance_window():
    acc = np.array([[3.0, 5.0, 5.0], [5.0 + 5e-9, 5.0, 5.0]])
    pro = np.array([[4.0, 5.0, 5.0]])            # acceptor 1 is 5e-9 A farther: tie within 1e-8, lowest index wins
    for owner, tied in _both(acc, pro, cutoff=1.5, tie_tolerance=1e-8):
        assert owner[0] == 0 and tied[0] == 2
    for owner, tied in _both(acc, pro, cutoff=1.5, tie_tolerance=1e-10):
        assert owner[0] == 0 and tied[0] == 1


def test_cutoff_and_no_cutoff():
    acc = np.array([[1.0, 1.0, 5.0]])
    pro = np.array([[1.0, 1.0, 7.0]])            # 2 A away
    for owner, _ in _both(acc, pro, cutoff=1.5, tie_tolerance=1e-8):
        assert owner[0] == -1
    for owner, _ in _both(acc, pro, cutoff=np.inf, tie_tolerance=1e-8):
        assert owner[0] == 0


def test_periodic_image_counts():
    acc = np.array([[0.2, 5.0, 5.0], [8.5, 5.0, 5.0]])
    pro = np.array([[9.7, 5.0, 5.0]])            # 0.5 A to acceptor 0 through x, 1.2 A to acceptor 1
    for owner, _ in _both(acc, pro, cutoff=1.5, tie_tolerance=1e-8):
        assert owner[0] == 0


@needs_cpu
def test_random_python_equals_cpu():
    from waterint.backends.cpu.native import nearest_owner
    rng = np.random.default_rng(3)
    for _ in range(5):
        acc = rng.uniform(0, 10, size=(60, 3)) * [1, 1, 3]
        pro = rng.uniform(0, 10, size=(120, 3)) * [1, 1, 3]
        acc = np.round(acc, 3)                    # quantised coordinates make near-ties possible
        pro = np.round(pro, 3)
        for cutoff in (1.5, np.inf):
            a = nearest_owner_python(acc, pro, cutoff=cutoff, tie_tolerance=1e-8, cell=CELL, cell_vectors=None, pbc=(True, True, False))
            b = nearest_owner(acc, pro, cutoff=cutoff, tie_tolerance=1e-8, fallback_nearest=not np.isfinite(cutoff), cell=CELL,
                              pbc=(True, True, False))
            np.testing.assert_array_equal(a[0], b[0])
            np.testing.assert_array_equal(a[1], b[1])


def test_sfg_assignment_is_prep_ownership():
    from waterint.analysis.spectra._legacy_selection import SelectionContext
    from waterint.analysis.spectra.sfg_core import assign_sfg_hydrogens_for_frame
    from waterint.io.frame import TrajectoryFrame
    frames, types, box = make_frames(n_frames=1)
    frame = TrajectoryFrame(index=0, comment="", symbols=[TYPE_MAP[t] for t in types], positions=frames[0], cell=tuple(box), types=types)
    ctx = SelectionContext.from_input_config({"type_map": TYPE_MAP})
    got = assign_sfg_hydrogens_for_frame(frame, cell=tuple(box), pbc=(True, True, False), context=ctx, oxygen_symbol="O",
                                         hydrogen_symbol="H", oh_cutoff=1.25, oh_assignment="nearest_oxygen", neighbor_method="auto",
                                         neighbor_workers=1, oxygen_chunk_size=2048, duplicate_policy="nearest")
    o = np.flatnonzero(types == 3)
    h = np.flatnonzero(types == 1)
    owner, _ = nearest_owner_python(frames[0][o], frames[0][h], cutoff=np.inf, tie_tolerance=1e-8, cell=tuple(box),
                                    cell_vectors=None, pbc=(True, True, False))
    assert got == {int(h[i]): int(o[k]) for i, k in enumerate(owner)}


@needs_cpu
def test_relabel_stream_cpu_equals_python(tmp_path):
    from waterint.io.writers.lammpstrj_relabel import RelabelStream
    frames, types, box = make_frames(n_frames=4)
    src = write_lammpstrj(tmp_path / "in.lammpstrj", frames, types, box)
    outs = {}
    for be in ("cpu", "python"):
        out = tmp_path / f"{be}.lammpstrj"
        with RelabelStream([src, src], out, TYPE_MAP, backend=be) as rs:   # second copy: all timesteps repeated
            for frame, ids in rs:
                rs.write(np.where(frame.types == 3, 7, frame.types))
        assert rs.duplicates == 4
        outs[be] = out.read_bytes()
    assert outs["cpu"] == outs["python"]
    assert outs["cpu"].replace(b" 7 ", b" 3 ") == src.read_bytes()


def test_relabel_codes_follow_species(tmp_path):
    from waterint.pipeline.run import run
    frames, types, box = make_frames(n_frames=3)
    src = write_lammpstrj(tmp_path / "in.lammpstrj", frames, types, box)
    prof = {"extends": "mgo_water", "solid": {"elements": ["O"], "max_z_rel_A": 0.7, "mode": "frozen"},
            "expected_counts": {}}
    run({"input": {"trajectory": str(src), "format": "lammpstrj", "type_map": TYPE_MAP}, "system": {"profile": prof},
         "output": {"directory": str(tmp_path / "out"), "plot": False}, "analyses": [{"type": "relabel", "name": "rl"}]})
    tm = [l.split("\t") for l in (tmp_path / "out/rl_type_map.tsv").read_text().splitlines()[1:]]
    names = {int(c): n for c, n, _ in tm}
    assert names[1] == "H" and names[2] == "lattice_Mg" and names[3] == "Os" and names[9] == "L1_OH" and names[27] == "other_region_O_other"
    audit = (tmp_path / "out/rl_audit.csv").read_text().splitlines()
    head = audit[0].split(",")
    row = dict(zip(head, audit[1].split(",")))
    n_o = int(np.sum(types == 3))
    o_cols = [n for c, n in names.items() if c >= 3]
    assert sum(int(row[c]) for c in o_cols) == n_o
    assert int(row["n_H_total"]) == int(np.sum(types == 1))


def _tilted_copy(src, dst, xy=1.5):
    """Rewrite a synthetic dump as a tilted LAMMPS box (bounding-box convention); atoms stay inside the cell."""
    out = []
    for block in src.read_text().split("ITEM: TIMESTEP\n")[1:]:
        lines = block.splitlines()
        b = [list(map(float, l.split())) for l in lines[4:7]]
        lines[3] = "ITEM: BOX BOUNDS xy xz yz pp pp pp"
        lines[4] = f"{b[0][0] + min(0.0, xy)} {b[0][1] + max(0.0, xy)} {xy}"
        lines[5] = f"{b[1][0]} {b[1][1]} 0.0"
        lines[6] = f"{b[2][0]} {b[2][1]} 0.0"
        out.append("ITEM: TIMESTEP\n" + "\n".join(lines) + "\n")
    dst.write_text("".join(out))
    return dst


@needs_cpu
def test_relabel_stream_tilted_box(tmp_path):
    from waterint.io.writers.lammpstrj_relabel import RelabelStream
    frames, types, box = make_frames(n_frames=2)
    src = _tilted_copy(write_lammpstrj(tmp_path / "o.lammpstrj", frames, types, box), tmp_path / "t.lammpstrj")
    for be in ("cpu", "python"):
        with RelabelStream([src], tmp_path / f"{be}.lammpstrj", TYPE_MAP, backend=be) as rs:
            for frame, _ids in rs:
                assert frame.triclinic and np.allclose(frame.cell_vectors[1], [1.5, box[1], 0.0])
                assert np.allclose(frame.cell, box)
                rs.write(frame.types)
        assert (tmp_path / f"{be}.lammpstrj").read_bytes() == src.read_bytes()


def test_msd_accepts_species_type_codes(tmp_path):
    """Relabelled inputs change O type codes with the species; MSD must compare elements, not raw types."""
    from waterint.pipeline.run import run
    frames, types, box = make_frames(n_frames=6)
    t2 = [types.copy() for _ in frames]
    t2[3][np.flatnonzero(types == 3)[-1]] = 4             # one O changes its code (OH- -> H2O style)
    path = tmp_path / "r.lammpstrj"
    with path.open("w") as fh:
        for f, (p, tt) in enumerate(zip(frames, t2)):
            fh.write(f"ITEM: TIMESTEP\n{f}\nITEM: NUMBER OF ATOMS\n{len(tt)}\nITEM: BOX BOUNDS pp pp pp\n0 {box[0]}\n0 {box[1]}\n0 {box[2]}\n")
            fh.write("ITEM: ATOMS id type x y z\n" + "".join(f"{i + 1} {int(t)} {r[0]:.6f} {r[1]:.6f} {r[2]:.6f}\n" for i, (t, r) in enumerate(zip(tt, p))))
    res = run({"input": {"trajectory": str(path), "format": "lammpstrj", "type_map": {**TYPE_MAP, 4: "O"}},
               "system": {"profile": "generic_oxide_water"}, "output": {"directory": str(tmp_path / "o"), "plot": False},
               "analyses": [{"type": "msd", "name": "m", "selection": {"elements": ["O"]}, "msd": {"timestep_ps": 0.05}}]})
    assert np.all(np.isfinite(res["m"].arrays["msd_a2"]))
