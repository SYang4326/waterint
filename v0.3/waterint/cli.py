from __future__ import annotations

import argparse

from waterint.config import load_config
from waterint._00_io.npz import write_npz_from_lammpstrj
from waterint._04_workflows.registry.registry import iter_analysis_modules


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

    for module in iter_analysis_modules():
        module.add_cli_parser(subparsers)

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

    if hasattr(args, "analysis_module"):
        module = args.analysis_module
        result = module.run_config(load_config(args.config))
        module.print_outputs(result)
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

    if args.command == "ui":
        from waterint._05_ui.desktop import run_desktop_ui

        run_desktop_ui()
        return 0

    if args.command == "web-ui":
        from waterint._05_ui.server import run_ui_server

        run_ui_server(args.host, args.port, open_browser=not args.no_open)
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
