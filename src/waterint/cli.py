"""Command line: `waterint run cfg.yaml`, `waterint <analysis> --config v06.yaml`, `waterint list|backends|profile`."""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any


def main(argv: list[str] | None = None) -> int:
    from waterint import registry
    parser = argparse.ArgumentParser(prog="waterint", description="Water/solid-interface trajectory analysis.")
    sub = parser.add_subparsers(dest="command", required=True)
    p_run = sub.add_parser("run", help="Run a v0.7 pipeline config (several analyses in one pass).")
    p_run.add_argument("config")
    sub.add_parser("list", help="List analyses and their parameters (JSON).")
    sub.add_parser("backends", help="Show kernel backends and their availability.")
    p_prof = sub.add_parser("profile", help="Show a system profile.")
    p_prof.add_argument("name")
    sub.add_parser("ui", help="Start the local desktop UI (tkinter).")
    p_web = sub.add_parser("web-ui", help="Start the local browser UI.")
    p_web.add_argument("--host", default="127.0.0.1")
    p_web.add_argument("--port", type=int, default=8765)
    p_web.add_argument("--no-open", action="store_true", help="Do not open a browser automatically.")
    p_conv = sub.add_parser("convert-lammpstrj", help="Convert a LAMMPS dump to a WaterInt NPZ trajectory cache.")
    p_conv.add_argument("--input", required=True)
    p_conv.add_argument("--output", required=True)
    p_conv.add_argument("--type-map", nargs="*", default=[], help="TYPE=SYMBOL entries, e.g. 1=H 2=Mg 3=O")
    p_conv.add_argument("--start-timestep", type=int, default=None)
    p_conv.add_argument("--stride", type=int, default=1)
    p_conv.add_argument("--max-frames", default=None, help="maximum frames, or 'all'")
    for name in sorted(set(registry.ANALYSES) | set(registry.ALIASES)):
        p = sub.add_parser(name, help=f"Run one analysis from a (v0.6-style) config: {registry.resolve_type(name)}.")
        p.add_argument("--config", required=True)
    args = parser.parse_args(argv)

    if args.command == "ui":
        from waterint.ui.desktop import run_desktop_ui
        run_desktop_ui()
        return 0
    if args.command == "web-ui":
        from waterint.ui.server import run_ui_server
        run_ui_server(args.host, args.port, open_browser=not args.no_open)
        return 0
    if args.command == "convert-lammpstrj":
        from waterint.io.readers.npz import write_npz_from_lammpstrj
        type_map = {}
        for entry in args.type_map:
            if "=" not in entry:
                parser.error(f"bad --type-map entry {entry!r}; expected TYPE=SYMBOL")
            k, v = entry.split("=", 1)
            type_map[int(k)] = v
        max_frames = None if args.max_frames in {None, "all", "0"} else int(args.max_frames)
        out = write_npz_from_lammpstrj(trajectory_path=args.input, output_path=args.output, type_map=type_map,
                                       start_timestep=args.start_timestep, stride=args.stride, max_frames=max_frames)
        print(f"Wrote: {out}")
        return 0

    from waterint.pipeline.configfile import load_config
    if args.command == "run":
        from waterint.pipeline.run import run
        _report(run(load_config(args.config)))
    elif args.command == "list":
        print(json.dumps(registry.describe(), indent=2, default=str))
    elif args.command == "backends":
        from waterint.backends import registry as kr
        for backend in kr.BACKENDS:
            print(f"{backend:7s} {kr.backend_status(backend)}")
        for kernel, impls in kr.describe().items():
            print(f"  {kernel:28s} {', '.join(impls)}")
    elif args.command == "profile":
        from waterint.system.profile import load_profile
        print(json.dumps(load_profile(args.name), indent=2, default=str))
    else:
        from waterint.pipeline.migrate import migrate_v06
        from waterint.pipeline.run import run
        _report(run(migrate_v06(args.command, load_config(args.config))))
    return 0


def _report(results: dict[str, Any]) -> None:
    for name, result in results.items():
        for kind, value in result.outputs.items():
            values = value.values() if isinstance(value, dict) else [value]
            for v in values:
                print(f"Wrote ({name}/{kind}): {v}")


if __name__ == "__main__":
    sys.exit(main())
