"""pipeline: one pass with several analyses == separate runs; backends agree; outputs and metadata exist."""
from __future__ import annotations

import json

import numpy as np
import pytest

from synthetic import TYPE_MAP
from waterint.backends import registry as kr
from waterint.pipeline.run import run

ANALYSES = [
    {"type": "density", "name": "dens_O", "selection": {"mode": "element", "species": ["O"]},
     "coordinate": {"range": [-3, 20], "bins": 46}},
    {"type": "density", "name": "dens_sp", "selection": {"mode": "oxygen_species", "oxygen_species": "all"},
     "coordinate": {"range": [-3, 20], "bins": 46}},
    {"type": "orientation", "name": "ori", "coordinate": {"range": [-3, 20], "bins": 23}, "angle": {"bins": 18}},
    {"type": "hbond", "name": "hb"},
    {"type": "rdf", "name": "rdf", "rdf": {"r_max": 6.0, "bins": 60, "pbc": [True, True, False]}},
    {"type": "msd", "name": "msd", "selection": {"oxygen_species": ["H2O"]}, "msd": {"timestep_ps": 0.01, "pbc": [True, True, False]}},
]


def _config(traj, outdir, analyses, backend="auto", profile="generic_oxide_water"):
    analyses = [dict(a, backend=backend) for a in analyses]
    return {"input": {"trajectory": str(traj), "format": "lammpstrj", "type_map": TYPE_MAP},
            "system": {"profile": profile}, "prep": {"backend": backend},
            "output": {"directory": str(outdir), "plot": False}, "analyses": analyses}


def _arrays(results):
    return {(name, k): v for name, r in results.items() for k, v in r.arrays.items()}


def test_one_pass_equals_separate_runs(synthetic_traj, tmp_path):
    together = _arrays(run(_config(synthetic_traj, tmp_path / "all", ANALYSES)))
    for a in ANALYSES:
        alone = _arrays(run(_config(synthetic_traj, tmp_path / a["name"], [a])))
        for key, value in alone.items():
            np.testing.assert_array_equal(value, together[key], err_msg=str(key))


@pytest.mark.skipif(not kr.cpu_available(), reason="native library not available")
def test_python_and_cpu_backends_agree(synthetic_traj, tmp_path):
    py = _arrays(run(_config(synthetic_traj, tmp_path / "py", ANALYSES, backend="python")))
    cpu = _arrays(run(_config(synthetic_traj, tmp_path / "cpu", ANALYSES, backend="cpu")))
    assert py.keys() == cpu.keys()
    for key in py:
        np.testing.assert_allclose(py[key], cpu[key], rtol=1e-10, atol=1e-12, err_msg=str(key))


def test_outputs_and_metadata(synthetic_traj, tmp_path):
    results = run(_config(synthetic_traj, tmp_path, ANALYSES))
    summary = json.loads((tmp_path / "waterint_run.json").read_text())
    assert summary["frames"] == 8 and summary["analyses"] == [a["name"] for a in ANALYSES]
    for name, result in results.items():
        meta = json.loads((tmp_path / f"{name}_metadata.json").read_text())
        assert meta["frames"] == 8 and meta["profile"] == "generic_oxide_water"
        assert (tmp_path / f"{name}_result.npz").exists()
    dens = results["dens_sp"]
    total = sum(dens.arrays[f"counts_per_frame__{i}"].sum() for i in range(len(dens.meta["profile_labels"])))
    total_o = results["dens_O"].arrays["counts_per_frame__0"].sum()
    assert total == pytest.approx(total_o)                    # species partition all O


def test_segments_are_concatenated_and_deduplicated(tmp_path):
    from synthetic import make_frames, write_lammpstrj
    frames, types, box = make_frames(n_frames=6)
    a = write_lammpstrj(tmp_path / "a.lammpstrj", frames[:4], types, box)
    b_frames = frames[3:]
    b = tmp_path / "b.lammpstrj"
    write_lammpstrj(b, b_frames, types, box)
    text = b.read_text().replace("TIMESTEP\n0\n", "TIMESTEP\n30\n").replace("TIMESTEP\n10\n", "TIMESTEP\n40\n").replace("TIMESTEP\n20\n", "TIMESTEP\n50\n")
    b.write_text(text)
    from waterint.io.trajectory import open_trajectory
    steps = [f.step for f in open_trajectory({"trajectory": [str(a), str(b)], "format": "lammpstrj"})]
    assert steps == [0, 10, 20, 30, 40, 50]


def test_plots_are_written(synthetic_traj, tmp_path):
    cfg = _config(synthetic_traj, tmp_path, ANALYSES[:1] + ANALYSES[3:5])
    cfg["output"]["plot"] = True
    run(cfg)
    assert (tmp_path / "dens_O.png").exists() and (tmp_path / "hb.png").exists() and (tmp_path / "rdf.png").exists()
