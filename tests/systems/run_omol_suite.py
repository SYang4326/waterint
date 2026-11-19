"""Multi-system test: 24 oxides + 3 carbonates (MACE-MH-1/OMOL) with the generic oxide profile only.

A  reproduction: the 2026-08-17 OH-/H2O mass-density profiles (all_within_cutoff ownership, same production frames)
   compared bin by bin with the archived density.csv of every case.
B  canonical run (generic_oxide_water, nearest ownership), one pass: species and element densities, O-H orientation,
   H-bond topology, RDF, 2D water MSD, OH- defect MSD, and a relabelled dump (dynamic solid). Sanity checks:
   species partition all O, H conservation, z_ref spread, finite results.

    python tests/systems/run_omol_suite.py [--cases a,b] [--workers 4] [--stride_b 2]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

OMOL = Path(os.environ.get("WATERINT_OMOL_CASES", "omol_cases"))   # dir with cases/<case>/ and relabel_configs/
OUT = Path(os.environ.get("WATERINT_REGRESSION_ROOT", "regression_data")) / "systems"
O_VARIANTS = {"O", "O_lattice", "OH", "H2O", "H3O"}


def case_inputs(case: str) -> dict:
    cdir = OMOL / "cases" / case
    rel = json.loads((cdir / "relabelled_trajectory_metadata.json").read_text())
    meta = json.loads((cdir / "metadata.json").read_text())
    type_map = {int(code): ("O" if name in O_VARIANTS else name) for name, code in rel["type_map"].items()}
    segs = [int(s) for s in meta["completed_production_segments"]]
    start = 1000 * (min(segs) - 1)                 # relabelled: 1001 frames in seg 1, 1000 per later seg (dedup)
    n = int(meta["production_frames"])
    traj = cdir / "relabelled_trajectory.lammpstrj"
    steps = []
    with open(traj, "rb") as fh:
        for line in fh:
            if line.startswith(b"ITEM: TIMESTEP"):
                steps.append(int(next(fh)))
    if start + n > len(steps):
        raise ValueError(f"{case}: trajectory has {len(steps)} frames, need {start + n}")
    return {"traj": str(traj), "type_map": type_map, "start_step": steps[start], "n": n, "meta": meta,
            "elements": sorted(set(type_map.values())), "frames_in_file": len(steps)}


def ref_meta_area(case: str) -> float:
    return json.loads((OMOL / "cases" / case / "metadata.json").read_text())["cross_section_area_A2"]


def run_case(case: str, stride_b: int) -> dict:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from waterint.analysis.base import Result
    from waterint.io.trajectory import open_trajectory
    from waterint.pipeline.run import run
    from waterint.prep.labelled import LabelledFrame, PrepContext
    from waterint.system.profile import load_system_profile

    ci = case_inputs(case)
    out = OUT / case
    rep: dict = {"case": case, "elements": ci["elements"], "frames": ci["n"]}
    base_in = {"trajectory": ci["traj"], "format": "lammpstrj", "type_map": ci["type_map"], "reader": "cpp",
               "start_timestep": ci["start_step"], "max_frames": ci["n"]}
    dens_sp = {"type": "density", "selection": {"mode": "oxygen_species", "oxygen_species": ["OH-", "H2O"]},
               "coordinate": {"range": [-2.0, 32.0], "bins": 340}, "normalization": {"type": "mass_density", "unit": "g_cm3"}}

    # ---------------------------------------------------------------- A: reproduction of the 2026-08-17 profiles
    t0 = time.perf_counter()
    res = run({"input": base_in, "system": {"profile": "generic_oxide_water"}, "prep": {"ownership": {"mode": "all_within_cutoff"}},
               "output": {"directory": str(out / "A"), "plot": False}, "analyses": [{**dens_sp, "name": "dens"}]})
    rep["A_seconds"] = round(time.perf_counter() - t0, 1)
    a = res["dens"].arrays
    ref = np.genfromtxt(OMOL / "cases" / case / "density.csv", delimiter=",", names=True)
    labels = res["dens"].meta["profile_labels"]
    oh, h2o = a[f"density__{labels.index('OH-')}"], a[f"density__{labels.index('H2O')}"]
    scale = max(float(np.max(np.abs(ref["H2O_mass_density_g_cm3"]))), 1e-12)
    rep["A_frames_match"] = res["dens"].meta.get("frames", None)
    rep["A_max_abs_diff_H2O"] = float(np.max(np.abs(h2o - ref["H2O_mass_density_g_cm3"])))
    rep["A_max_abs_diff_OH"] = float(np.max(np.abs(oh - ref["OHminus_mass_density_g_cm3"])))
    rep["A_max_rel_diff"] = max(rep["A_max_abs_diff_H2O"], rep["A_max_abs_diff_OH"]) / scale
    rep["A_identical"] = bool(np.array_equal(h2o, ref["H2O_mass_density_g_cm3"]) and np.array_equal(oh, ref["OHminus_mass_density_g_cm3"]))
    # The 2026-08-17 script built bin edges with np.arange (v0.7: np.linspace); with 6-digit coordinates some z - z_ref
    # values fall exactly on an edge. Recount with the arange edges to test the classification itself.
    from dataclasses import replace
    prof = load_system_profile("generic_oxide_water")
    ctx = PrepContext(profile=replace(prof, ownership=prof.ownership.replace({"mode": "all_within_cutoff"})), type_map=ci["type_map"])
    edges = np.arange(-2.0, 32.0 + 0.05, 0.1)
    lin = np.linspace(-2.0, 32.0, 341)
    counts = {"OH-": np.zeros(340), "H2O": np.zeros(340)}
    on_edge = 0
    for frame in open_trajectory({k: v for k, v in base_in.items()}):
        lf = LabelledFrame(frame, ctx)
        sp = lf.species()
        for lab in counts:
            v = lf.coordinates(sp[lab])
            counts[lab] += np.histogram(v, edges)[0]
            on_edge += int(np.isin(v, edges).sum() + np.isin(v, lin).sum())
    area = float(ref_meta_area(case))
    unit = {"OH-": 17.007, "H2O": 18.015}
    ok = True
    for lab, col in (("OH-", "OHminus_mass_density_g_cm3"), ("H2O", "H2O_mass_density_g_cm3")):
        expect = np.round(ref[col] / (unit[lab] / (area * 0.1) * 1.66053906660 / ci["n"]))
        ok &= bool(np.array_equal(counts[lab], expect))
    rep["A_counts_identical_with_0817_edges"] = ok
    rep["A_values_exactly_on_bin_edges"] = on_edge

    # ---------------------------------------------------------------- B: canonical one-pass suite
    b_in = {**base_in, "stride": stride_b, "max_frames": (ci["n"] + stride_b - 1) // stride_b}
    analyses = [
        {**dens_sp, "name": "dens_species", "selection": {"mode": "oxygen_species", "oxygen_species": "all"},
         "coordinate": {"range": [-30.0, 60.0], "bins": 900}},
        {"type": "density", "name": "dens_O", "selection": {"mode": "element", "species": ["O"]},
         "coordinate": {"range": [-30.0, 60.0], "bins": 900}, "normalization": {"type": "counts_per_frame"}},
        {"type": "orientation", "name": "ori", "coordinate": {"range": [-2.0, 12.0], "bins": 70}, "angle": {"bins": 36}},
        {"type": "hbond", "name": "hb", "hbond": {"pbc": [True, True, False]}},
        {"type": "rdf", "name": "rdf", "rdf": {"r_max": 6.0, "bins": 120, "pbc": [True, True, False], "pairs": [
            {"name": "O-O", "first": {"elements": ["O"]}, "second": {"elements": ["O"]}},
            {"name": "H2O-H2O", "first": {"oxygen_species": ["H2O"]}, "second": {"oxygen_species": ["H2O"]}}]}},
        {"type": "msd", "name": "msd_water", "selection": {"oxygen_species": ["H2O"]},
         "msd": {"timestep_ps": 0.05 * stride_b, "dimensionality": "2d", "pbc": [True, True, False], "max_lag_frames": 200}},
    ]
    t0 = time.perf_counter()
    try:
        resb = run({"input": b_in, "system": {"profile": "generic_oxide_water"}, "output": {"directory": str(out / "B"), "plot": True},
                    "analyses": analyses})
        rep["B_seconds"] = round(time.perf_counter() - t0, 1)
        finite = {n: all(np.all(np.isfinite(v)) for v in r.arrays.values() if np.asarray(v).dtype.kind == "f") for n, r in resb.items()}
        rep["B_all_finite"] = all(finite.values())
        sp = resb["dens_species"]
        total_species = sum(sp.arrays[f"counts_per_frame__{i}"].sum() for i in range(len(sp.meta["profile_labels"])))
        rep["B_species_per_frame"] = {l: round(float(sp.arrays[f"counts_per_frame__{i}"].sum()), 3)
                                      for i, l in enumerate(sp.meta["profile_labels"])}
        rep["B_O_per_frame"] = round(float(resb["dens_O"].arrays["counts_per_frame__0"].sum()), 3)
        rep["B_species_partition_ok"] = abs(total_species - rep["B_O_per_frame"]) < 1e-6
        hb = resb["hb"].meta
        rep["B_hbond_samples"] = hb["samples_total"]
        rep["B_water_msd_A2_at_lag200"] = round(float(resb["msd_water"].arrays["msd_a2"][-1]), 3)
        rep["B_rdf_OO_first_peak_A"] = round(float(resb["rdf"].arrays["r_centers"][np.argmax(resb["rdf"].arrays["g_r__0"])]), 3)
    except Exception as exc:
        rep["B_error"] = f"{type(exc).__name__}: {exc}"
        rep["B_traceback"] = traceback.format_exc()[-1500:]

    # ---------------------------------------------------------------- prep sanity on sampled frames
    prof = load_system_profile("generic_oxide_water")
    ctx = PrepContext(profile=prof, type_map=ci["type_map"])
    traj = open_trajectory({"trajectory": ci["traj"], "format": "lammpstrj", "type_map": ci["type_map"], "reader": "cpp",
                            "start_timestep": ci["start_step"], "stride": 250, "max_frames": 21})
    zrefs, h_ok, free_h, ties, n_h = [], True, 0, 0, 0
    for frame in traj:
        lf = LabelledFrame(frame, ctx)
        own = lf.ownership()
        n_h = own.proton_idx.size
        h_ok &= int(own.counts.sum()) + int(np.sum(own.owner() < 0)) == n_h
        free_h += int(np.sum(own.owner() < 0))
        ties += int(np.sum(own.tied > 1)) if own.tied is not None else 0
        zrefs.append(lf.reference())
    rep["prep_H_conserved"] = bool(h_ok)
    rep["prep_free_H_in_21_frames"] = free_h
    rep["prep_ties_in_21_frames"] = ties
    rep["prep_zref_mean_A"] = round(float(np.mean(zrefs)), 4)
    rep["prep_zref_spread_A"] = round(float(np.ptp(zrefs)), 4)
    rep["zref_vs_0817_A"] = round(float(np.mean(zrefs)) - float(ci["meta"]["mean_top_solid_z_A"]), 4)

    # ---------------------------------------------------------------- relabel (dynamic solid) on 200 frames
    t0 = time.perf_counter()
    try:
        run({"input": {"trajectory": ci["traj"], "format": "lammpstrj", "type_map": ci["type_map"]},
             "system": {"profile": {"extends": "generic_oxide_water", "lammps_type_map": ci["type_map"]}},
             "output": {"directory": str(out / "relabel"), "plot": False},
             "analyses": [{"type": "relabel", "name": "rl", "relabel": {"max_frames": 200}}]})
        audit = list(csv.DictReader(open(out / "relabel" / "rl_audit.csv")))
        tm = [l.split("\t")[1] for l in (out / "relabel" / "rl_type_map.tsv").read_text().splitlines()[1:]]
        o_labels = [n for n in tm if "_" in n]
        n_o = int(round(rep.get("B_O_per_frame", 0)))
        rep["relabel_seconds"] = round(time.perf_counter() - t0, 1)
        rep["relabel_O_partition_ok"] = all(sum(int(r[c]) for c in o_labels) == n_o for r in audit) if n_o else None
        last = audit[-1]
        rep["relabel_last_frame_labels"] = {c: int(last[c]) for c in o_labels if int(last[c])}
    except Exception as exc:
        rep["relabel_error"] = f"{type(exc).__name__}: {exc}"
    return rep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--stride_b", type=int, default=2)
    args = ap.parse_args()
    cases = sorted(p.name for p in (OMOL / "cases").iterdir() if p.is_dir())
    if args.cases:
        cases = [c for c in cases if c in set(args.cases.split(","))]
    OUT.mkdir(parents=True, exist_ok=True)
    reports = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(run_case, c, args.stride_b): c for c in cases}
        for fut in as_completed(futs):
            c = futs[fut]
            try:
                r = fut.result()
            except Exception as exc:
                r = {"case": c, "error": f"{type(exc).__name__}: {exc}"}
            reports[c] = r
            print(json.dumps({k: r.get(k) for k in ("case", "A_counts_identical_with_0817_edges", "A_values_exactly_on_bin_edges", "A_max_rel_diff", "B_species_partition_ok",
                                                     "B_all_finite", "prep_H_conserved", "relabel_O_partition_ok",
                                                     "B_error", "relabel_error", "error")}, default=str), flush=True)
            (OUT / "suite_report.json").write_text(json.dumps(reports, indent=1, default=str) + "\n", encoding="utf-8")
    print("SUITE_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
