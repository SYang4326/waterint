"""v0.7 profile-driven relabel vs the v0.6 MgO-standard relabel executable (see tests/regression/README.md).

Compares: relabelled dump (bytes), z_ref (values), frozen lattice-O IDs, per-frame audit columns shared by both
(label counts by name, H totals, ties, other_region). Since every O's code depends on its owned-H count, an identical
dump also cross-checks prep's canonical ownership against the v0.6 rule on every frame.

    python tests/regression/check_relabel.py [--frames 2000]
"""
from __future__ import annotations

import argparse
import csv
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

V06 = os.environ.get("WATERINT_V06", "")
REG = Path(os.environ.get("WATERINT_REGRESSION_ROOT", "regression_data"))   # see tests/regression/README.md
MGO = str(REG / "data/mgo_2000f.lammpstrj")


def read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=2000)
    args = ap.parse_args()
    d6, d7 = REG / "relabel_v06", REG / "relabel_v07"
    cfg = {"input": {"trajectories": [MGO], "format": "lammpstrj"}, "relabel": {"max_frames": args.frames}}
    d6.mkdir(parents=True, exist_ok=True)
    c6 = dict(cfg, output={"directory": str(d6), "prefix": "rl", "plot": False})
    (d6 / "config.yaml").write_text(yaml.safe_dump(c6), encoding="utf-8")
    t0 = time.perf_counter()
    subprocess.run([sys.executable, "-c", "import sys; from waterint.cli import main; sys.exit(main(sys.argv[1:]))", "relabel",
                    "--config", str(d6 / "config.yaml")], env=dict(os.environ, PYTHONPATH=V06), check=True, capture_output=True)
    t6 = time.perf_counter() - t0
    c7 = dict(cfg, output={"directory": str(d7), "prefix": "rl", "plot": False}, _config_dir=str(d7))
    t0 = time.perf_counter()
    run(migrate_v06("relabel", c7))
    t7 = time.perf_counter() - t0
    rep: dict = {"frames": args.frames, "seconds": {"v06": round(t6, 1), "v07": round(t7, 1)}}
    rep["dump_identical"] = (d6 / "rl.lammpstrj").read_bytes() == (d7 / "rl.lammpstrj").read_bytes()
    z6, z7 = read_csv(d6 / "rl_zref.csv"), read_csv(d7 / "rl_zref.csv")
    rep["zref_identical"] = [(r["step"], r["z_ref_A"]) for r in z6] == [(r["step"], r["z_ref_A"]) for r in z7]
    ids = lambda p: [int(l) for l in p.read_text().splitlines() if l and not l.startswith("#")]
    rep["lattice_ids_identical"] = ids(d6 / "rl.lammpstrj.lattice_O_ids.txt") == ids(d7 / "rl.lammpstrj.lattice_O_ids.txt")
    a6, a7 = read_csv(d6 / "rl_audit.csv"), read_csv(d7 / "rl_audit.csv")
    shared = [c for c in a7[0] if c in a6[0] and c not in {"z_ref_A"}]
    bad = {c: sum(1 for x, y in zip(a6, a7) if x[c] != y[c]) for c in shared}
    rep["audit_rows"] = (len(a6), len(a7))
    rep["audit_shared_columns"] = len(shared)
    rep["audit_differences"] = {c: n for c, n in bad.items() if n}
    m6 = json.loads((d6 / "rl_metadata.json").read_text())
    m7 = json.loads((d7 / "rl_metadata.json").read_text())
    rep["label_totals_identical"] = m6["label_counts_total"] == m7["label_counts_total"]
    rep["ties"] = (m6["audit_totals"]["nearest_O_tie"], m7["audit_totals"]["nearest_O_tie"])
    rep["unassigned_H"] = (m6["audit_totals"]["n_unassigned_H"], m7["audit_totals"]["n_unassigned_H"])
    print(json.dumps(rep, indent=1))
    (REG / "report" / "check_relabel.json").write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
