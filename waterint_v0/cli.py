from __future__ import annotations

import argparse

from waterint.config import load_config
from waterint_v0.workflows.oh_orientation import run_angle_z, run_oh_orientation
from waterint_v0.workflows.density import run_density
from waterint_v0.workflows.hbond import run_hbond
from waterint_v0.workflows.sfg import run_sfg


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="waterint-v0")
    subparsers = parser.add_subparsers(dest="command", required=True)

    density_parser = subparsers.add_parser("density", help="Run the v0 density workflow.")
    density_parser.add_argument("--config", required=True, help="YAML config file.")

    angle_z_parser = subparsers.add_parser("angle-z", help="Run the v0 O-H angle vs coordinate workflow.")
    angle_z_parser.add_argument("--config", required=True, help="YAML config file.")

    oh_orientation_parser = subparsers.add_parser("oh-orientation", help="Run the v0 O-H orientation workflow.")
    oh_orientation_parser.add_argument("--config", required=True, help="YAML config file.")

    hbond_parser = subparsers.add_parser("hbond", help="Run the v0 H-bond topology workflow.")
    hbond_parser.add_argument("--config", required=True, help="YAML config file.")

    sfg_parser = subparsers.add_parser("sfg", help="Run the v0 SFG workflow.")
    sfg_parser.add_argument("--config", required=True, help="YAML config file.")

    args = parser.parse_args(argv)
    config = load_config(args.config)

    if args.command == "density":
        result = run_density(config)
        print(f"Wrote: {result.csv_path}")
        if result.png_path:
            print(f"Wrote: {result.png_path}")
        print(f"Wrote: {result.metadata_path}")
        return 0

    if args.command == "angle-z":
        result = run_angle_z(config)
        for path in result.csv_paths.values():
            print(f"Wrote: {path}")
        for path in result.png_paths.values():
            print(f"Wrote: {path}")
        print(f"Wrote: {result.metadata_path}")
        return 0

    if args.command == "oh-orientation":
        result = run_oh_orientation(config)
        for path in result.csv_paths.values():
            print(f"Wrote: {path}")
        for path in result.png_paths.values():
            print(f"Wrote: {path}")
        print(f"Wrote: {result.metadata_path}")
        return 0

    if args.command == "hbond":
        result = run_hbond(config)
        print(f"Wrote: {result.csv_path}")
        print(f"Wrote: {result.raw_csv_path}")
        if result.png_path:
            print(f"Wrote: {result.png_path}")
        print(f"Wrote: {result.metadata_path}")
        return 0

    if args.command == "sfg":
        result = run_sfg(config)
        for path in result.cf_paths.values():
            print(f"Wrote: {path}")
        for path in result.ft_paths.values():
            print(f"Wrote: {path}")
        for path in result.png_paths.values():
            print(f"Wrote: {path}")
        print(f"Wrote: {result.metadata_path}")
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
