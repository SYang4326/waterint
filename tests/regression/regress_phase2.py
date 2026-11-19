"""Phase-2 regression: v0.6-ext (golden) vs v0.7 for the ported modules, on MgO LAMMPS and qnpz data.

    python tests/regression/regress_phase2.py --root $WATERINT_REGRESSION_ROOT [--only name,...]

Every case runs the v0.6 module CLI of the untouched baseline (WATERINT_V06) and the v0.7 migrated config. Chained cases
(events -> carrier-track -> carrier-msd, relabel -> identity-transition, site-current -> site-conductance, sfg-species
-> aggregate) feed each version its own upstream outputs. All output files are compared:
text tables (csv/dat/tsv/txt) numerically, npz array by array, relabelled dumps byte by byte, json/png/pdf skipped.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from waterint.pipeline.migrate import migrate_v06  # noqa: E402
from waterint.pipeline.run import run  # noqa: E402

V06 = os.environ.get("WATERINT_V06", "")
REG = Path(os.environ.get("WATERINT_REGRESSION_ROOT", "regression_data"))   # see tests/regression/README.md
MGO = str(REG / "data/mgo_2000f.lammpstrj")
QA = os.environ.get("WATERINT_QNPZ_A", "")
QB = os.environ.get("WATERINT_QNPZ_B", "")
TMAP = {1: "H", 2: "Mg", 3: "O"}
SLAB = {"mode": "relative_to_slab", "axis": "z", "reference": {"type": "top_layer_mean", "species": ["Mg"], "surface": "max", "layer_width": 0.7}}
LMP = {"trajectory": MGO, "format": "lammpstrj", "type_map": TMAP, "reader": "cpp"}


def cases(out: dict[str, Path]) -> list[tuple[str, str, dict]]:
    """(name, module, v0.6 config without output). `out[name]` is the output dir of an earlier case (per version)."""
    return [
        ("defect_msd", "defect-msd", {"input": LMP, "system": {"cell": "auto"},
            "selection": {"oxygen_species": ["OH-"], "hydrogen_assignment": "nearest", "oh_cutoff": 1.25},
            "defect_tracking": {"timestep_ps": 0.004, "gate_A": 3.0, "pbc": [True, True, False],
                                "layer": {"coordinate": SLAB, "range": [-0.5, 4.0]}},
            "defect_msd": {"frame_stride": 2, "dimensionality": "2d", "max_lag_frames": 500, "origin_stride": 2}}),
        ("defect_conductivity", "defect-conductivity", {"input": LMP, "system": {"cell": "auto"},
            "selection": {"oxygen_species": ["OH-"], "hydrogen_assignment": "nearest", "oh_cutoff": 1.25},
            "defect_tracking": {"timestep_ps": 0.004, "gate_A": 3.0, "pbc": [True, True, False],
                                "layer": {"coordinate": SLAB, "range": [-0.5, 4.0]}},
            "defect_msd": {"frame_stride": 2, "dimensionality": "2d", "max_lag_frames": 500, "origin_stride": 2},
            "defect_conductivity": {"estimator": "both", "temperature_K": 300.0, "fit_range_ps": [0.4, 1.6],
                                    "volume": {"mode": "slab", "normal_axis": "z", "thickness_A": 4.5}}}),
        ("conductivity", "conductivity", {"input": LMP, "system": {"cell": "auto"}, "selection": {"elements": ["O"]},
            "msd": {"timestep_ps": 0.004, "dimensionality": "2d", "pbc": [True, True, False], "max_lag_frames": 400,
                    "layer": {"coordinate": {**SLAB, "reference": {"type": "slab_surface", "species": ["Mg"], "surface": "max"}},
                              "range": [0.0, 4.0]}},
            "conductivity": {"temperature_K": 300.0, "charge_e": -1.0, "fit_range_ps": [0.2, 0.8],
                             "volume": {"mode": "slab", "normal_axis": "z", "thickness_A": 4.0}}}),
        ("proton_sharing", "proton-sharing", {"input": LMP, "system": {"cell": "auto"},
            "selection": {"oh_cutoff": 1.25, "pbc": [True, True, False]},
            "proton_sharing": {"temperature_K": 300.0, "coordinate": SLAB, "donor": {"species": "H2O", "range": [1.5, 2.8]},
                               "acceptor": {"species": "OH-", "range": [2.8, 4.0]}, "include_swapped_state": True}}),
        ("proton_sharing_hbond", "proton-sharing-hbond", {"input": LMP, "system": {"cell": "auto"},
            "selection": {"pbc": [True, True, False]},
            "proton_sharing_hbond": {"temperature_K": 300.0, "coordinate": SLAB, "l1_range": [1.5, 2.8], "l2_range": [2.8, 4.0],
                                     "oh_assignment_cutoff_A": 1.5}}),
        ("sfg_trajectory", "sfg", {"input": {**LMP, "max_frames": 400}, "system": {"cell": "auto"},
            "sfg": {"mode": "trajectory", "backend": "auto", "velocity_source": "auto", "dt_ps": 0.004, "lag_ps": 0.2,
                    "pbc": [True, True, False], "reference": {"species": ["Mg"], "surface": "max"},
                    "window": {"mode": 1, "z1": 0.0, "z2": 8.0, "ramp": 0.5, "flip": True}, "mu_mode": "full",
                    "symmetrize": True, "sticky_A": 0.0, "nzeros": 2000}}),
        ("relabel", "relabel", {"input": {"trajectories": [MGO], "format": "lammpstrj"}, "relabel": {"max_frames": 300}}),
        ("identity_transition", "identity-transition", {"input": {"trajectory": str(out["relabel"] / "out.lammpstrj"),
                                                                  "format": "lammpstrj"},
            "identity_transition": {"type_codes": "standard", "minimum_residence_frames": 2}}),
        ("site_current_lmp", "site-current", {"input": {"trajectory": MGO, "format": "lammpstrj", "dt_ps": 0.004, "dstep": 8},
            "site_current": {"block_frames": 500}, "events": {"enabled": True}}),
        ("carrier_track_lmp", "carrier-track", {"input": {"events": [str(out["site_current_lmp"] / "out_events.npz")]}}),
        ("carrier_msd_lmp", "carrier-msd", {"input": {"trajectory": MGO, "format": "lammpstrj", "dt_ps": 0.004,
                                                      "carrier_tracks": str(out["carrier_track_lmp"] / "out.npz")},
            "carrier_msd": {"full_lag_ps": 1.0, "max_lag_ps": 4.0}}),
        ("site_current_q", "site-current", {"input": {"format": "gpumd-qnpz", "segments": [QA], "max_frames": 20000},
            "events": {"enabled": True}}),
        ("carrier_track_q", "carrier-track", {"input": {"events": [str(out["site_current_q"] / "out_events.npz")]},
            "carrier_track": {"positive": True}}),
        ("carrier_msd_q", "carrier-msd", {"input": {"format": "gpumd-qnpz", "segments": [QA], "max_frames": 20000, "dt_ps": 0.008,
                                                    "carrier_tracks": str(out["carrier_track_q"] / "out.npz")},
            "carrier_msd": {"full_lag_ps": 2.0, "max_lag_ps": 40.0}}),
        ("site_conductance_q", "site-conductance", {"runs": {"r0": [str(out["site_current_q"] / "out.npz")],
                                                             "r1": [str(out["site_current_q"] / "out.npz")]},
            "conductance": {"temperature_K": 300.0, "zones": True, "helfand_windows_ps": [[1, 5], [5, 20]]}}),
        ("hb_correlation_q", "hb-correlation", {"input": {"format": "gpumd-qnpz", "segments": [QA], "max_frames": 30000},
            "hb_correlation": {"window_frames": 10000, "block_frames": 10000, "msd_stride_frames": 10, "msd_max_lag_frames": 500,
                               "backend": "cpu"}}),
        ("sfg_species_q", "sfg-species", {"input": {"format": "gpumd-qnpz", "segments": [QB], "first_frame": 250, "n_frames": 2000},
            "sfg_species": {"block_frames": 1000, "n_lag": 250, "free_oh": True}}),
        ("sfg_species_q2", "sfg-species", {"input": {"format": "gpumd-qnpz", "segments": [QB], "first_frame": 0, "n_frames": 1500},
            "sfg_species": {"block_frames": 1500, "n_lag": 250, "free_oh": True}}),
        ("sfg_species_aggregate", "sfg-species-aggregate", {"input": {"runs": [str(out["sfg_species_q"] / "out.npz"),
                                                                               str(out["sfg_species_q2"] / "out.npz")]}}),
    ]


class Lazy(dict):
    def __init__(self, base: Path):
        super().__init__()
        self.base = base

    def __missing__(self, key):
        return self.base / key


def run_v06(module: str, cfg: dict, workdir: Path) -> float:
    workdir.mkdir(parents=True, exist_ok=True)
    path = workdir / "config.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    env = dict(os.environ, PYTHONPATH=V06)
    code = "import sys; from waterint.cli import main; sys.exit(main(sys.argv[1:]))"
    t0 = time.perf_counter()
    proc = subprocess.run([sys.executable, "-c", code, module, "--config", str(path)], env=env, capture_output=True,
                          text=True, cwd=str(workdir))
    if proc.returncode != 0:
        raise RuntimeError(f"v0.6 {module} failed: {proc.stderr[-2500:]}")
    return time.perf_counter() - t0


def _num(token: str):
    try:
        return float(token)
    except ValueError:
        return None


def compare_text(a: Path, b: Path, rep: dict) -> None:
    la, lb = a.read_text(encoding="utf-8").split("\n"), b.read_text(encoding="utf-8").split("\n")
    if len(la) != len(lb):
        rep["problems"].append(f"{a.name}: {len(la)} vs {len(lb)} lines")
        return
    for x, y in zip(la, lb):
        ta, tb = x.replace(",", " ").split(), y.replace(",", " ").split()
        if len(ta) != len(tb):
            rep["problems"].append(f"{a.name}: token count differs in {x[:60]!r}")
            return
        for u, v in zip(ta, tb):
            rep["values"] += 1
            fu, fv = _num(u), _num(v)
            if fu is None or fv is None:
                if u != v:
                    rep["diff"] += 1
                    rep["problems"].append(f"{a.name}: {u!r} != {v!r}")
            elif fu != fv and not (np.isnan(fu) and np.isnan(fv)):
                rep["diff"] += 1
                rep["max_rel"] = max(rep["max_rel"], abs(fu - fv) / max(abs(fu), abs(fv)))


def compare_arrays(name: str, x: np.ndarray, y: np.ndarray, rep: dict) -> None:
    if x.shape != y.shape or x.dtype.kind != y.dtype.kind:
        rep["problems"].append(f"{name}: shape/dtype {x.shape}/{x.dtype} vs {y.shape}/{y.dtype}")
        return
    rep["values"] += int(x.size)
    if x.dtype.kind in "fc":
        both_nan = np.isnan(x) & np.isnan(y)
        d = (x != y) & ~both_nan
        rep["diff"] += int(d.sum())
        if d.any():
            rel = np.abs(x[d] - y[d]) / np.maximum(np.abs(x[d]), np.abs(y[d]))
            rep["max_rel"] = max(rep["max_rel"], float(np.max(rel)))
    elif x.dtype.kind == "V":
        for field in x.dtype.names or ():
            compare_arrays(f"{name}.{field}", x[field], y[field], rep)
    else:
        if not np.array_equal(x, y):
            n = int(np.sum(x != y)) if x.dtype.kind not in "OU" else 1
            rep["diff"] += n
            rep["problems"].append(f"{name}: {n} non-numeric/integer values differ")


def compare_dirs(d6: Path, d7: Path) -> dict:
    rep = {"files": 0, "values": 0, "diff": 0, "max_rel": 0.0, "problems": [], "only_v06": [], "only_v07": []}
    skip = {".json", ".png", ".pdf", ".yaml", ".log"}
    f6 = {p.relative_to(d6) for p in d6.rglob("*") if p.is_file() and p.suffix not in skip and ".tmp" not in p.name}
    f7 = {p.relative_to(d7) for p in d7.rglob("*") if p.is_file() and p.suffix not in skip and ".tmp" not in p.name}
    f7 = {p for p in f7 if not p.name.endswith("_result.npz")}
    rep["only_v06"] = sorted(map(str, f6 - f7))
    rep["only_v07"] = sorted(map(str, f7 - f6))
    for rel in sorted(f6 & f7):
        a, b = d6 / rel, d7 / rel
        rep["files"] += 1
        if a.suffix in {".csv", ".dat", ".tsv", ".txt"}:
            compare_text(a, b, rep)
        elif a.suffix == ".npz":
            with np.load(a, allow_pickle=False) as za, np.load(b, allow_pickle=False) as zb:
                if set(za.files) != set(zb.files):
                    rep["problems"].append(f"{rel}: keys {sorted(set(za.files) ^ set(zb.files))}")
                for k in sorted(set(za.files) & set(zb.files)):
                    compare_arrays(f"{rel}:{k}", za[k], zb[k], rep)
        elif a.suffix == ".npy":
            compare_arrays(str(rel), np.load(a), np.load(b), rep)
        else:
            same = a.read_bytes() == b.read_bytes()
            rep["values"] += 1
            if not same:
                rep["diff"] += 1
                rep["problems"].append(f"{rel}: bytes differ")
    rep["problems"] = rep["problems"][:12]
    return rep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(REG))
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    root = Path(args.root)
    only = set(filter(None, args.only.split(",")))
    o6, o7 = Lazy(root / "golden2"), Lazy(root / "v07_2")
    report = {"date": dt.date.today().isoformat(), "v06": V06, "cases": {}}
    names6 = {n: (m, c) for n, m, c in cases(o6)}
    names7 = {n: (m, c) for n, m, c in cases(o7)}
    for name in names6:
        if only and name not in only:
            continue
        module, c6 = names6[name]
        _m, c7 = names7[name]
        d6, d7 = o6[name], o7[name]
        cfg6 = copy.deepcopy(c6)
        cfg6["output"] = {"directory": str(d6), "prefix": "out", "plot": False}
        cfg7 = copy.deepcopy(c7)
        cfg7["output"] = {"directory": str(d7), "prefix": "out", "plot": False}
        cfg7["_config_dir"] = str(d7)
        try:
            t6 = run_v06(module, cfg6, d6)
            t0 = time.perf_counter()
            run(migrate_v06(module, cfg7))
            t7 = time.perf_counter() - t0
            rep = {"module": module, "v06_s": round(t6, 2), "v07_s": round(t7, 2), **compare_dirs(d6, d7)}
        except Exception as exc:     # report and continue with the other cases
            rep = {"module": module, "error": f"{type(exc).__name__}: {str(exc)[-1500:]}"}
        report["cases"][name] = rep
        if "error" in rep:
            print(f"{name:24s} ERROR {rep['error'][:300]}", flush=True)
        else:
            print(f"{name:24s} v0.6 {rep['v06_s']:7.1f}s v0.7 {rep['v07_s']:7.1f}s files {rep['files']:3d} "
                  f"values {rep['values']:>10d} diff {rep['diff']:>6d} max_rel {rep['max_rel']:.2e} "
                  f"only6 {rep['only_v06']} problems {rep['problems'][:3]}", flush=True)
    (root / "report").mkdir(parents=True, exist_ok=True)
    (root / "report" / f"regression_phase2_{report['date']}.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
