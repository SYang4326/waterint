from __future__ import annotations

import argparse

from waterint.angle_z import run_angle_z
from waterint.batch import run_batch
from waterint.config import load_config
from waterint.density import run_density
from waterint.hbond import run_hbond
from waterint.io.npz import write_npz_from_lammpstrj
from waterint.sfg import run_sfg


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

    density_parser = subparsers.add_parser("density", help="Run a 1D density profile analysis.")
    density_parser.add_argument("--config", required=True, help="YAML config file.")

    angle_z_parser = subparsers.add_parser("angle-z", help="Run an O-H angle vs coordinate 2D histogram.")
    angle_z_parser.add_argument("--config", required=True, help="YAML config file.")

    oh_orientation_parser = subparsers.add_parser(
        "oh-orientation",
        help="Run an O-H orientation vs coordinate 2D histogram.",
    )
    oh_orientation_parser.add_argument("--config", required=True, help="YAML config file.")

    hbond_parser = subparsers.add_parser("hbond", help="Run H-bond topology analysis by oxygen species.")
    hbond_parser.add_argument("--config", required=True, help="YAML config file.")

    sfg_parser = subparsers.add_parser("sfg", help="Run SFG CF postprocessing and plotting.")
    sfg_parser.add_argument("--config", required=True, help="YAML config file.")

    batch_parser = subparsers.add_parser("batch", help="Run multiple WaterInt tasks from a batch config.")
    batch_parser.add_argument("--config", required=True, help="Batch YAML config file.")
    batch_parser.add_argument("--dry-run", action="store_true", help="Validate and list tasks without running analyses.")
    batch_error = batch_parser.add_mutually_exclusive_group()
    batch_error.add_argument("--continue-on-error", action="store_true", help="Continue after failed tasks.")
    batch_error.add_argument("--stop-on-error", action="store_true", help="Stop at the first failed task.")

    subparsers.add_parser("ui", help="Start the local WaterInt desktop UI.")

    web_ui_parser = subparsers.add_parser("web-ui", help="Start the optional local browser UI.")
    web_ui_parser.add_argument("--host", default="127.0.0.1", help="Host interface for the UI server.")
    web_ui_parser.add_argument("--port", type=int, default=8765, help="Port for the UI server.")
    web_ui_parser.add_argument("--no-open", action="store_true", help="Do not open a browser automatically.")

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

    if args.command in {"angle-z", "oh-orientation"}:
        result = run_angle_z(load_config(args.config))
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

    if args.command == "batch":
        continue_on_error = None
        if args.continue_on_error:
            continue_on_error = True
        elif args.stop_on_error:
            continue_on_error = False
        result = run_batch(args.config, dry_run=args.dry_run, continue_on_error=continue_on_error)
        for task in result.tasks:
            print(f"[{task.status}] {task.name} ({task.module}) {task.config_path}")
            if task.error:
                print(f"  Error: {task.error}")
            for artifact in task.artifacts:
                print(f"  Wrote: {artifact}")
        if result.summary_path:
            print(f"Wrote summary: {result.summary_path}")
        return 1 if result.failed else 0

    if args.command == "ui":
        from waterint.ui.desktop import run_desktop_ui

        run_desktop_ui()
        return 0

    if args.command == "web-ui":
        from waterint.ui.server import run_ui_server

        run_ui_server(args.host, args.port, open_browser=not args.no_open)
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
