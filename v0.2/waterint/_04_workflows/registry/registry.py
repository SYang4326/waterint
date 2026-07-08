from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from waterint._04_workflows.registry.analysis_module import AnalysisModule
from waterint._04_workflows.workflows.density import run_density
from waterint._04_workflows.workflows.hbond import run_hbond
from waterint._04_workflows.workflows.oh_orientation import run_oh_orientation
from waterint._04_workflows.workflows.sfg import run_sfg


def print_density_outputs(result: Any) -> None:
    print(f"Wrote: {result.csv_path}")
    if result.png_path:
        print(f"Wrote: {result.png_path}")
    print(f"Wrote: {result.metadata_path}")


def print_oh_orientation_outputs(result: Any) -> None:
    for path in result.csv_paths.values():
        print(f"Wrote: {path}")
    for path in result.png_paths.values():
        print(f"Wrote: {path}")
    print(f"Wrote: {result.metadata_path}")


def print_hbond_outputs(result: Any) -> None:
    print(f"Wrote: {result.csv_path}")
    print(f"Wrote: {result.raw_csv_path}")
    if result.png_path:
        print(f"Wrote: {result.png_path}")
    print(f"Wrote: {result.metadata_path}")


def print_sfg_outputs(result: Any) -> None:
    for path in result.cf_paths.values():
        print(f"Wrote: {path}")
    for path in result.ft_paths.values():
        print(f"Wrote: {path}")
    for path in result.png_paths.values():
        print(f"Wrote: {path}")
    print(f"Wrote: {result.metadata_path}")


ANALYSIS_MODULES: tuple[AnalysisModule, ...] = (
    AnalysisModule(
        name="density",
        help="Run a density profile workflow.",
        run=run_density,
        print_outputs=print_density_outputs,
        description="Project selected atoms or oxygen species onto one coordinate and write a density profile.",
    ),
    AnalysisModule(
        name="oh-orientation",
        help="Run an O-H orientation workflow.",
        run=run_oh_orientation,
        print_outputs=print_oh_orientation_outputs,
        aliases=("angle-z",),
        description="Accumulate O-H angle distributions along a selected coordinate.",
    ),
    AnalysisModule(
        name="hbond",
        help="Run an H-bond topology workflow.",
        run=run_hbond,
        print_outputs=print_hbond_outputs,
        description="Classify hydrogen-bond topology by oxygen species.",
    ),
    AnalysisModule(
        name="sfg",
        help="Run an SFG workflow.",
        run=run_sfg,
        print_outputs=print_sfg_outputs,
        description="Run SFG correlation-function and spectrum workflows.",
    ),
)


def iter_analysis_modules() -> Iterable[AnalysisModule]:
    return iter(ANALYSIS_MODULES)


def get_analysis_module(command: str) -> AnalysisModule:
    for module in ANALYSIS_MODULES:
        if command in module.command_names:
            return module
    raise KeyError(command)
