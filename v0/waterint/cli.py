from __future__ import annotations

import argparse

from waterint.config import load_config
from waterint.io.npz import write_npz_from_lammpstrj
from waterint.workflows.oh_orientation import run_angle_z, run_oh_orientation
from waterint.workflows.density import run_density
from waterint.workflows.hbond import run_hbond
from waterint.workflows.sfg import run_sfg


def _parse_type_map_entries(entries: list[str]) -> dict[int, str]:
    type_map: dict[int, str] = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"Bad --type-map entry {entry!r}; expected TYPE=SYMBOL.")
        raw_type, symbol = entry.split("=", 1)
        type_map[int(raw_type)] = symbol
    return type_map


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="waterint")
    subparsers = parser.add_subparsers(dest="command", required=True)

    density_parser = subparsers.add_parser("density", help="Run a density profile workflow.")
    density_parser.add_argument("--config", required=True, help="YAML config file.")

    angle_z_parser = subparsers.add_parser("angle-z", help="Run an O-H angle vs coordinate workflow.")
    angle_z_parser.add_argument("--config", required=True, help="YAML config file.")

    oh_orientation_parser = subparsers.add_parser("oh-orientation", help="Run an O-H orientation workflow.")
    oh_orientation_parser.add_argument("--config", required=True, help="YAML config file.")

    hbond_parser = subparsers.add_parser("hbond", help="Run an H-bond topology workflow.")
    hbond_parser.add_argument("--config", required=True, help="YAML config file.")

    sfg_parser = subparsers.add_parser("sfg", help="Run an SFG workflow.")
    sfg_parser.add_argument("--config", required=True, help="YAML config file.")

    convert_parser = subparsers.add_parser("convert-lammpstrj", help="Convert a LAMMPS dump trajectory to WaterInt NPZ.")
    convert_parser.add_argument("--input", required=True, help="Input LAMMPS dump trajectory.")
    convert_parser.add_argument("--output", required=True, help="Output NPZ trajectory cache.")
    convert_parser.add_argument("--type-map", nargs="*", default=[], help="Atom type map entries, e.g. 1=H 2=Mg 3=O.")
    convert_parser.add_argument("--start-timestep", type=int, default=None)
    convert_parser.add_argument("--stride", type=int, default=1)
    convert_parser.add_argument("--max-frames", default=None, help="Maximum frames to convert, or 'all'.")

    args = parser.parse_args(argv)

    if args.command == "density":
        result = run_density(load_config(args.config))
        print(f"Wrote: {result.csv_path}")
        if result.png_path:
            print(f"Wrote: {result.png_path}")
        print(f"Wrote: {result.metadata_path}")
        return 0

    if args.command == "angle-z":
        result = run_angle_z(load_config(args.config))
        for path in result.csv_paths.values():
            print(f"Wrote: {path}")
        for path in result.png_paths.values():
            print(f"Wrote: {path}")
        print(f"Wrote: {result.metadata_path}")
        return 0

    if args.command == "oh-orientation":
        result = run_oh_orientation(load_config(args.config))
        for path in result.csv_paths.values():
            print(f"Wrote: {path}")
        for path in result.png_paths.values():
            print(f"Wrote: {path}")
        print(f"Wrote: {result.metadata_path}")
        return 0

    if args.command == "hbond":
        result = run_hbond(load_config(args.config))
        print(f"Wrote: {result.csv_path}")
        print(f"Wrote: {result.raw_csv_path}")
        if result.png_path:
            print(f"Wrote: {result.png_path}")
        print(f"Wrote: {result.metadata_path}")
        return 0

    if args.command == "sfg":
        result = run_sfg(load_config(args.config))
        for path in result.cf_paths.values():
            print(f"Wrote: {path}")
        for path in result.ft_paths.values():
            print(f"Wrote: {path}")
        for path in result.png_paths.values():
            print(f"Wrote: {path}")
        print(f"Wrote: {result.metadata_path}")
        return 0

    if args.command == "convert-lammpstrj":
        max_frames = None if args.max_frames in {None, "all", "0"} else int(args.max_frames)
        output = write_npz_from_lammpstrj(
            trajectory_path=args.input,
            output_path=args.output,
            type_map=_parse_type_map_entries(args.type_map),
            start_timestep=args.start_timestep,
            stride=args.stride,
            max_frames=max_frames,
        )
        print(f"Wrote: {output}")
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
