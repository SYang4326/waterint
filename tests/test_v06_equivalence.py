"""v0.7 (through v0.6-config migration) reproduces the v0.6 reference implementation.

The v0.6 reference is an external checkout (WATERINT_V06: the directory that contains the v0.6 `waterint/` package,
e.g. `v0/` of tag v0.6). Tests are skipped when it is not available.

The v0.6 package runs in a subprocess (both packages are called ``waterint``); CSV outputs are compared:
integer-like columns exactly, floats to rtol 1e-12.
"""
from __future__ import annotations

import copy
import csv
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from synthetic import TYPE_MAP
from waterint.pipeline.migrate import migrate_v06
from waterint.pipeline.run import run

ROOT = Path(__file__).resolve().parents[1]
V06 = Path(os.environ.get("WATERINT_V06", ROOT / "_v06_reference"))
pytestmark = pytest.mark.skipif(not (V06 / "waterint" / "cli.py").exists(), reason="v0.6 reference not available")
SLAB_REF = {"axis": "z", "mode": "relative_to_slab",
            "reference": {"type": "top_layer_mean", "species": ["Mg"], "surface": "max", "layer_width": 0.7}}

CASES = {
    "density_element": ("density", {"selection": {"mode": "element", "species": ["O"]},
                                    "coordinate": {"axis": "z", "range": [0, 25], "bins": 50}}),
    "density_species_mass": ("density", {"selection": {"mode": "oxygen_species", "oxygen_species": "all", "oh_cutoff": 1.25},
                                         "coordinate": {**SLAB_REF, "range": [-3, 20], "bins": 46},
                                         "normalization": {"type": "mass_density", "unit": "g_cm3"}}),
    "orientation_bond": ("oh-orientation", {"selection": {"oh_cutoff": 1.25},
                                            "coordinate": {**SLAB_REF, "range": [-3, 20], "bins": 23},
                                            "angle": {"bins": 36, "vector_mode": "oh_bond"}}),
    "orientation_bisector": ("oh-orientation", {"selection": {"oxygen_species": ["H2O"]},
                                                "coordinate": {"axis": "z", "range": [0, 25], "bins": 25},
                                                "angle": {"bins": 18, "vector_mode": "oh_bisector"},
                                                "normalization": {"type": "number_density"}}),
    "hbond": ("hbond", {"selection": {"oxygen_species": "all"},
                        "hbond": {"oo_cutoff": 3.5, "dha_angle_min": 150.0, "pbc": [True, True, False]}}),
    "rdf": ("rdf", {"rdf": {"r_max": 6.0, "bins": 60, "pbc": [True, True, False], "pairs": [
        {"name": "O-O", "first": {"elements": ["O"]}, "second": {"elements": ["O"]}},
        {"name": "H2O-OH", "first": {"oxygen_species": ["H2O"]}, "second": {"oxygen_species": ["OH-"]}},
        {"name": "L-O-H", "first": {"elements": ["O"], "layer": {"coordinate": SLAB_REF, "range": [1.0, 6.0]}},
         "second": {"elements": ["H"]}}]}}),
    "msd_2d_layer": ("msd", {"selection": {"elements": ["O"]},
                             "msd": {"timestep_ps": 0.01, "dimensionality": "2d", "pbc": [True, True, False],
                                     "layer": {"coordinate": {"axis": "z"}, "range": [5.0, 25.0]}}}),
    "msd_species": ("msd", {"selection": {"oxygen_species": ["H2O"]}, "msd": {"timestep_ps": 0.01, "max_lag_frames": 5}}),
}


def _v06_config(traj: Path, outdir: Path, body: dict) -> dict:
    cfg = {"input": {"trajectory": str(traj), "format": "lammpstrj", "type_map": dict(TYPE_MAP)},
           "system": {"cell": "auto"}, "output": {"directory": str(outdir), "prefix": "out", "plot": False}}
    cfg.update(copy.deepcopy(body))
    return cfg


def _run_v06(module: str, cfg_path: Path) -> None:
    env = dict(os.environ, PYTHONPATH=str(V06))
    code = "import sys; from waterint.cli import main; sys.exit(main(sys.argv[1:]))"
    proc = subprocess.run([sys.executable, "-c", code, module, "--config", str(cfg_path)], env=env,
                          capture_output=True, text=True, cwd=str(cfg_path.parent))
    assert proc.returncode == 0, proc.stderr[-2000:]


def _read_csv(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open(encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    return rows[0], rows[1:]


def _compare(a: Path, b: Path) -> float:
    ha, ra = _read_csv(a)
    hb, rb = _read_csv(b)
    assert ha == hb, f"{a.name}: header differs"
    assert len(ra) == len(rb), f"{a.name}: row count differs"
    worst = 0.0
    for x, y in zip(ra, rb):
        for u, v in zip(x, y):
            try:
                fu, fv = float(u), float(v)
            except ValueError:
                assert u == v, f"{a.name}: {u!r} != {v!r}"
                continue
            if fu != fv:
                rel = abs(fu - fv) / max(abs(fu), abs(fv))
                worst = max(worst, rel)
    return worst


@pytest.mark.parametrize("case", sorted(CASES))
def test_v07_matches_v06(case, synthetic_traj, tmp_path):
    module, body = CASES[case]
    d6, d7 = tmp_path / "v06", tmp_path / "v07"
    cfg = _v06_config(synthetic_traj, d6, body)
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    _run_v06(module, cfg_path)
    cfg7 = _v06_config(synthetic_traj, d7, body)
    cfg7["_config_dir"] = str(tmp_path)
    run(migrate_v06(module, cfg7))
    csv6 = sorted(p.name for p in d6.glob("*.csv"))
    csv7 = sorted(p.name for p in d7.glob("*.csv"))
    assert csv6 and csv6 == csv7
    for name in csv6:
        worst = _compare(d6 / name, d7 / name)
        assert worst <= 1e-12, f"{case}/{name}: max relative difference {worst:.3g}"
