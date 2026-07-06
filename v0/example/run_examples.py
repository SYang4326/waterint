from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from waterint.config import load_config
from waterint.workflows.density import run_density
from waterint.workflows.hbond import run_hbond
from waterint.workflows.oh_orientation import run_oh_orientation
from waterint.workflows.sfg import run_sfg


EXAMPLE_ROOT = Path(__file__).resolve().parent

RUNS = [
    ("density", run_density, "mgo_density/config_oh_h2o_npz.yaml"),
    ("oh_orientation", run_oh_orientation, "mgo_oh_orientation/config_oh_h2o_h3o_npz.yaml"),
    ("hbond", run_hbond, "mgo_hbond/config_oh_h2o_h3o_npz.yaml"),
    ("sfg", run_sfg, "mgo_sfg/config_100ps_npz.yaml"),
]

DEFAULT_ANALYSES = {"density", "oh_orientation", "sfg"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run MgO-water example workflows.")
    parser.add_argument(
        "--include-hbond",
        action="store_true",
        help="Also run the full 100 ps H-bond example. This can take about an hour on a laptop.",
    )
    parser.add_argument(
        "--only",
        choices=["density", "oh_orientation", "hbond", "sfg"],
        help="Run only one analysis.",
    )
    args = parser.parse_args()

    selected = _selected_runs(include_hbond=args.include_hbond, only=args.only)
    failures: list[tuple[str, str]] = []
    for index, (name, runner, rel_config) in enumerate(selected, 1):
        config_path = EXAMPLE_ROOT / rel_config
        start = time.perf_counter()
        print(f"RUN  {index:02d} {name:14s} {rel_config}", flush=True)
        try:
            result = runner(load_config(config_path))
        except Exception as exc:  # pragma: no cover - command-line reporting
            failures.append((rel_config, f"{type(exc).__name__}: {exc}"))
            print(f"FAIL {index:02d} {name:14s} {rel_config}: {failures[-1][1]}", flush=True)
            continue
        elapsed = time.perf_counter() - start
        print(f"PASS {index:02d} {name:14s} {rel_config} ({elapsed:.2f}s)", flush=True)
        _print_outputs(result)

    if failures:
        print("\nFailures:")
        for rel_config, message in failures:
            print(f"- {rel_config}: {message}")
        return 1
    return 0


def _selected_runs(*, include_hbond: bool, only: str | None) -> list[tuple[str, object, str]]:
    if only is not None:
        return [run for run in RUNS if run[0] == only]
    analyses = set(DEFAULT_ANALYSES)
    if include_hbond:
        analyses.add("hbond")
    return [run for run in RUNS if run[0] in analyses]


def _print_outputs(result: object) -> None:
    for attr in ("csv_path", "raw_csv_path", "png_path", "metadata_path"):
        path = getattr(result, attr, None)
        if path:
            print(f"  {attr}: {path}")
    for attr in ("csv_paths", "png_paths", "cf_paths", "ft_paths"):
        paths = getattr(result, attr, None)
        if isinstance(paths, dict):
            for label, path in paths.items():
                print(f"  {attr}[{label}]: {path}")


if __name__ == "__main__":
    raise SystemExit(main())
