from __future__ import annotations

import argparse

from waterint.config import load_config
from waterint.density import run_density


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="waterint")
    subparsers = parser.add_subparsers(dest="command", required=True)

    density_parser = subparsers.add_parser("density", help="Run a 1D density profile analysis.")
    density_parser.add_argument("--config", required=True, help="YAML config file.")

    args = parser.parse_args(argv)

    if args.command == "density":
        result = run_density(load_config(args.config))
        print(f"Wrote: {result.csv_path}")
        if result.png_path:
            print(f"Wrote: {result.png_path}")
        print(f"Wrote: {result.metadata_path}")
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
