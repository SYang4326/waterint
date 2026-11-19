"""Two v0.7-internal checks on real data (see tests/regression/README.md):

1. Block.frames() == frame readers, field by field (lammpstrj and gpumd-qnpz), so stream analyses fed from blocks in a
   mixed run see exactly the frames they would see alone; plus one mixed run (block + stream) == separate runs.
2. hb-correlation gpu backend vs cpu backend (C_PT / C_HB / FES counts exact; orientation within a few counts).

    WATERINT_CUDA_ARCH=sm_120 python tests/regression/check_blocks_gpu.py      # CUDA toolkit in the environment
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from waterint.analysis.base import Result  # noqa: E402
from waterint.io.trajectory import open_trajectory  # noqa: E402
from waterint.pipeline.run import run  # noqa: E402

REG = Path(os.environ.get("WATERINT_REGRESSION_ROOT", "regression_data"))   # see tests/regression/README.md
MGO = str(REG / "data/mgo_2000f.lammpstrj")
QA = os.environ.get("WATERINT_QNPZ_A", "")      # GPUMD qnpz run, 8 fs frames
QB = os.environ.get("WATERINT_QNPZ_B", "")      # GPUMD qnpz run with velocities, 4 fs frames
TMAP = {1: "H", 2: "Mg", 3: "O"}
FIELDS = ("index", "symbols", "positions", "cell", "step", "types", "velocities", "cell_vectors", "cell_origin", "triclinic")


def same(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.array_equal(np.asarray(a), np.asarray(b))
    return a == b


def frames_vs_blocks(input_cfg: dict, n: int, velocities: bool) -> dict:
    t = open_trajectory(input_cfg)
    ref = []
    for f in t.iter_frames():
        ref.append(f)
        if len(ref) == n:
            break
    got = []
    for b in t.iter_blocks(300, velocities=velocities):
        got.extend(b.frames())
        if len(got) >= n:
            break
    got = got[:n]
    bad = {}
    for f, g in zip(ref, got):
        for field in FIELDS:
            if not same(getattr(f, field), getattr(g, field)):
                bad[field] = bad.get(field, 0) + 1
    return {"frames": len(ref), "mismatching_fields": bad}


def main() -> int:
    rep: dict = {}
    rep["lammpstrj"] = frames_vs_blocks({"trajectory": MGO, "format": "lammpstrj", "type_map": TMAP, "reader": "cpp"}, 700, True)
    rep["qnpz_vel"] = frames_vs_blocks({"trajectory": QB, "format": "gpumd-qnpz", "type_map": TMAP, "dstep": 8}, 700, True)
    rep["qnpz_window"] = frames_vs_blocks({"trajectory": QA, "format": "gpumd-qnpz", "type_map": TMAP, "dstep": 16,
                                           "first_frame": 7300, "max_frames": 600, "velocities": False}, 600, False)
    print(json.dumps(rep, indent=1), flush=True)

    # mixed run: site-current (block) + density/msd (stream) in one pass == separate runs
    out = REG / "mixed"
    base_in = {"trajectory": MGO, "format": "lammpstrj", "type_map": TMAP, "reader": "cpp", "dt_ps": 0.004, "dstep": 8}
    sc = {"type": "site-current", "name": "sc", "events": {"enabled": True}}
    dens = {"type": "density", "name": "dens", "selection": {"mode": "oxygen_species", "oxygen_species": "all"},
            "coordinate": {"axis": "z", "mode": "relative_to_slab",
                           "reference": {"type": "top_layer_mean", "species": ["Mg"], "surface": "max", "layer_width": 0.7},
                           "range": [-3, 30], "bins": 330}}
    msd = {"type": "msd", "name": "msd", "selection": {"oxygen_species": ["H2O"]},
           "msd": {"timestep_ps": 0.004, "dimensionality": "2d", "pbc": [True, True, False], "max_lag_frames": 300}}
    prof = "mgo_water"
    cfg = lambda d, a: {"input": base_in, "system": {"profile": prof}, "output": {"directory": str(out / d), "plot": False}, "analyses": a}
    t0 = time.perf_counter(); together = run(cfg("together", [sc, dens, msd])); t_tog = time.perf_counter() - t0
    t0 = time.perf_counter()
    alone = {**run(cfg("sc", [sc])), **run(cfg("dens", [dens])), **run(cfg("msd", [msd]))}
    t_alone = time.perf_counter() - t0
    mixed = {}
    for name in ("dens", "msd"):
        mixed[name] = all(np.array_equal(together[name].arrays[k], alone[name].arrays[k]) for k in alone[name].arrays)
    with np.load(out / "together/sc.npz") as a, np.load(out / "sc/sc.npz") as b:
        mixed["sc"] = all(np.array_equal(a[k], b[k]) for k in b.files)
    rep["mixed_run_equals_separate"] = mixed
    rep["mixed_seconds"] = {"one_pass": round(t_tog, 2), "separate": round(t_alone, 2)}
    print(json.dumps({k: rep[k] for k in ("mixed_run_equals_separate", "mixed_seconds")}), flush=True)

    # GPU vs CPU hb-correlation
    from waterint.backends import registry as kr
    if kr.gpu_available():
        hb = {"window_frames": 10000, "block_frames": 10000, "msd_stride_frames": 10, "msd_max_lag_frames": 500}
        q_in = {"trajectory": [QA], "format": "gpumd-qnpz", "type_map": TMAP, "max_frames": 30000}
        times = {}
        for be in ("cpu", "gpu"):
            t0 = time.perf_counter()
            run({"input": q_in, "system": {"profile": prof}, "output": {"directory": str(REG / f"hbc_{be}"), "plot": False},
                 "analyses": [{"type": "hb-correlation", "name": "hbc", "backend": be, "hb_correlation": hb}]})
            times[be] = round(time.perf_counter() - t0, 2)
        diffs = {}
        with np.load(REG / "hbc_cpu/hbc_summary.npz") as a, np.load(REG / "hbc_gpu/hbc_summary.npz") as b:
            for k in a.files:
                x, y = a[k], b[k]
                if x.dtype.kind in "fiu" and x.shape == y.shape:
                    d = np.abs(x.astype(float) - y.astype(float))
                    diffs[k] = {"n_diff": int((d > 0).sum()), "size": int(x.size), "max_abs": float(d.max()) if d.size else 0.0}
        rep["gpu_vs_cpu"] = {"seconds": times, "summary_arrays_with_differences": {k: v for k, v in diffs.items() if v["n_diff"]},
                             "identical_arrays": sorted(k for k, v in diffs.items() if not v["n_diff"])}
    else:
        rep["gpu_vs_cpu"] = "gpu backend unavailable"
    print(json.dumps(rep.get("gpu_vs_cpu"), indent=1), flush=True)
    (REG / "report" / "check_blocks_gpu.json").write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
