"""Real-data regression + benchmark: v0.6-ext (golden) vs v0.7 on MgO-water and TiO2-water slices.

    python tests/regression/regress_real.py --root $WATERINT_REGRESSION_ROOT [--frames 2000]

For every case the untouched baseline package (WATERINT_V06) runs the
v0.6 module CLI; v0.7 runs the migrated config. CSVs are compared (max relative difference). Then the
benchmark times the v0.6 sequential runs against ONE v0.7 pipeline containing all analyses of the system.
Writes <root>/report/regression_<date>.json and .md.
"""
from __future__ import annotations

import argparse
import copy
import csv
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from waterint.pipeline.migrate import migrate_v06  # noqa: E402
from waterint.pipeline.run import run  # noqa: E402
from waterint.analysis.base import Result  # noqa: E402

V06 = os.environ.get("WATERINT_V06", "")
SYSTEMS = {
    "mgo": {"source": os.environ.get("WATERINT_MGO_DUMP", ""),          # MgO(100)-water film, LAMMPS dump
            "lines_per_frame": 1905, "type_map": {1: "H", 2: "Mg", 3: "O"}, "cation": "Mg"},
    "tio2": {"source": os.environ.get("WATERINT_TIO2_DUMP", ""),        # rutile TiO2(110)-water, relabelled dump
             "lines_per_frame": 606, "type_map": {1: "H", 2: "O", 3: "Ti", 4: "O", 5: "O", 6: "O", 7: "O"}, "cation": "Ti"},
}


def cases(cation: str) -> dict[str, tuple[str, dict]]:
    slab = {"axis": "z", "mode": "relative_to_slab",
            "reference": {"type": "top_layer_mean", "species": [cation], "surface": "max", "layer_width": 0.7}}
    return {
        "density_species": ("density", {"selection": {"mode": "oxygen_species", "oxygen_species": "all"},
                                        "coordinate": {**slab, "range": [-3, 30], "bins": 330},
                                        "normalization": {"type": "mass_density", "unit": "g_cm3"}}),
        "density_H": ("density", {"selection": {"mode": "element", "species": ["H"]},
                                  "coordinate": {**slab, "range": [-3, 30], "bins": 330}}),
        "orientation_bond": ("oh-orientation", {"selection": {"oh_cutoff": 1.25}, "coordinate": {**slab, "range": [0, 15], "bins": 75},
                                                "angle": {"bins": 90, "vector_mode": "oh_bond"}}),
        "orientation_bisector": ("oh-orientation", {"selection": {"oxygen_species": ["H2O"]},
                                                    "coordinate": {**slab, "range": [0, 15], "bins": 75},
                                                    "angle": {"bins": 90, "vector_mode": "oh_bisector"}}),
        "hbond": ("hbond", {"selection": {"oxygen_species": "all"}, "hbond": {"pbc": [True, True, False]}}),
        "rdf": ("rdf", {"rdf": {"r_max": 8.0, "bins": 160, "pbc": [True, True, False], "pairs": [
            {"name": "O-O", "first": {"elements": ["O"]}, "second": {"elements": ["O"]}},
            {"name": "H2O-OH", "first": {"oxygen_species": ["H2O"]}, "second": {"oxygen_species": ["OH-"]}},
            {"name": "L1-O-O", "first": {"elements": ["O"], "layer": {"coordinate": slab, "range": [1.5, 2.8]}},
             "second": {"elements": ["O"]}}]}}),
        "msd_water_2d": ("msd", {"selection": {"oxygen_species": ["H2O"]},
                                 "msd": {"timestep_ps": 0.001, "dimensionality": "2d", "pbc": [True, True, False],
                                         "max_lag_frames": 500, "layer": {"coordinate": slab, "range": [1.5, 30.0]}}}),
        "msd_H_3d": ("msd", {"selection": {"elements": ["H"]}, "msd": {"timestep_ps": 0.001, "max_lag_frames": 500,
                                                                     "pbc": [True, True, False]}}),
    }


def slice_trajectory(src: str, dst: Path, frames: int, lines_per_frame: int) -> Path:
    if dst.exists():
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp")
    with open(src, "rb") as fin, open(tmp, "wb") as fout:
        for _ in range(frames * lines_per_frame):
            line = fin.readline()
            if not line:
                break
            fout.write(line)
    tmp.rename(dst)
    return dst


def base_config(traj: Path, outdir: Path, type_map: dict, frames: int, body: dict) -> dict:
    cfg = {"input": {"trajectory": str(traj), "format": "lammpstrj", "type_map": dict(type_map), "max_frames": frames,
                     "reader": "cpp"},
           "system": {"cell": "auto"}, "output": {"directory": str(outdir), "prefix": "out", "plot": False}}
    cfg.update(copy.deepcopy(body))
    return cfg


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
        raise RuntimeError(f"v0.6 {module} failed: {proc.stderr[-1500:]}")
    return time.perf_counter() - t0


def compare_dirs(d6: Path, d7: Path) -> dict:
    csv6 = sorted(p.name for p in d6.glob("*.csv"))
    csv7 = sorted(p.name for p in d7.glob("*.csv"))
    out = {"files": csv6, "same_files": csv6 == csv7, "max_rel": 0.0, "n_values": 0, "n_diff": 0}
    for name in csv6:
        if name not in csv7:
            continue
        with open(d6 / name) as a, open(d7 / name) as b:
            ra, rb = list(csv.reader(a)), list(csv.reader(b))
        if ra[0] != rb[0] or len(ra) != len(rb):
            out["same_files"] = False
            continue
        for x, y in zip(ra[1:], rb[1:]):
            for u, v in zip(x, y):
                out["n_values"] += 1
                try:
                    fu, fv = float(u), float(v)
                except ValueError:
                    out["n_diff"] += int(u != v)
                    continue
                if fu != fv:
                    out["n_diff"] += 1
                    out["max_rel"] = max(out["max_rel"], abs(fu - fv) / max(abs(fu), abs(fv)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("WATERINT_REGRESSION_ROOT", "regression_data"))
    ap.add_argument("--frames", type=int, default=2000)
    ap.add_argument("--systems", default="mgo,tio2")
    args = ap.parse_args()
    root = Path(args.root)
    report = {"date": dt.date.today().isoformat(), "v06": V06, "frames": args.frames, "systems": {}}
    for sysname in args.systems.split(","):
        s = SYSTEMS[sysname]
        traj = slice_trajectory(s["source"], root / "data" / f"{sysname}_{args.frames}f.lammpstrj", args.frames,
                                s["lines_per_frame"])
        rep = {"trajectory": str(traj), "cases": {}, "v06_seconds": 0.0}
        analyses7 = []
        for case, (module, body) in cases(s["cation"]).items():
            d6 = root / "golden" / sysname / case
            d7 = root / "v07" / sysname / case
            t6 = run_v06(module, base_config(traj, d6, s["type_map"], args.frames, body), d6)
            cfg7 = base_config(traj, d7, s["type_map"], args.frames, body)
            cfg7["_config_dir"] = str(d7)
            mig = migrate_v06(module, cfg7)
            t0 = time.perf_counter()
            run(mig)
            t7 = time.perf_counter() - t0
            cmp = compare_dirs(d6, d7)
            rep["cases"][case] = {"module": module, "v06_s": round(t6, 2), "v07_s": round(t7, 2), **cmp}
            rep["v06_seconds"] += t6
            a = dict(mig["analyses"][0]); a["name"] = case
            analyses7.append(a)
            print(f"[{sysname}] {case:22s} v0.6 {t6:7.1f}s  v0.7 {t7:7.1f}s  max_rel {cmp['max_rel']:.2e}  "
                  f"diff {cmp['n_diff']}/{cmp['n_values']}", flush=True)
        one = {"input": base_config(traj, root, s["type_map"], args.frames, {})["input"],
               "system": {"profile": migrate_v06("density", {"input": {}})["system"]["profile"]},
               "output": {"directory": str(root / "v07_onepass" / sysname), "plot": False}, "analyses": analyses7,
               "_config_dir": str(root)}
        t0 = time.perf_counter()
        results = run(one)
        rep["v07_onepass_seconds"] = round(time.perf_counter() - t0, 2)
        rep["v06_seconds"] = round(rep["v06_seconds"], 2)
        import numpy as np
        identical = {}
        for case, result in results.items():
            sep = Result.load(root / "v07" / sysname / case / migrate_name(case, sysname, root))
            identical[case] = all(np.array_equal(sep.arrays[k], result.arrays[k]) for k in result.arrays)
        rep["onepass_equals_separate"] = identical
        report["systems"][sysname] = rep
        print(f"[{sysname}] v0.6 sequential {rep['v06_seconds']:.1f}s   v0.7 one pass {rep['v07_onepass_seconds']:.1f}s", flush=True)
    out = root / "report"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"regression_{report['date']}.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: {c: (v['max_rel'], v['n_diff']) for c, v in r['cases'].items()} for k, r in report['systems'].items()}))
    return 0


def migrate_name(case: str, sysname: str, root: Path) -> str:
    meta = sorted((root / "v07" / sysname / case).glob("*_metadata.json"))
    return meta[0].name[: -len("_metadata.json")]


if __name__ == "__main__":
    sys.exit(main())
