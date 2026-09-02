from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from waterint._04_workflows.registry.analysis_module import AnalysisModule
from waterint._04_workflows.workflows.density import run_density
from waterint._04_workflows.workflows.hbond import run_hbond
from waterint._04_workflows.workflows.oh_orientation import run_oh_orientation
from waterint._04_workflows.workflows.msd import run_msd
from waterint._04_workflows.workflows.conductivity import run_conductivity
from waterint._04_workflows.workflows.defect_conductivity import run_defect_conductivity
from waterint._04_workflows.workflows.defect_msd import run_defect_msd
from waterint._04_workflows.workflows.rdf import run_rdf
from waterint._04_workflows.workflows.sfg import run_sfg
from waterint._04_workflows.workflows.proton_sharing import run_proton_sharing
from waterint._04_workflows.workflows.proton_sharing_hbond import run_proton_sharing_hbond


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


def print_msd_outputs(result: Any) -> None:
    print(f"Wrote: {result.csv_path}")
    if result.png_path:
        print(f"Wrote: {result.png_path}")
    print(f"Wrote: {result.metadata_path}")


def print_conductivity_outputs(result: Any) -> None:
    print(f"Wrote: {result.csv_path}")
    print(f"Wrote: {result.msd_csv_path}")
    if result.png_path:
        print(f"Wrote: {result.png_path}")
    print(f"Wrote: {result.metadata_path}")


def print_defect_msd_outputs(result: Any) -> None:
    print(f"Wrote: {result.csv_path}")
    print(f"Wrote: {result.tracks_csv_path}")
    print(f"Wrote: {result.events_csv_path}")
    if result.png_path:
        print(f"Wrote: {result.png_path}")
    print(f"Wrote: {result.metadata_path}")


def print_defect_conductivity_outputs(result: Any) -> None:
    print(f"Wrote: {result.csv_path}")
    print(f"Wrote: {result.msd_csv_path}")
    print(f"Wrote: {result.tracks_csv_path}")
    print(f"Wrote: {result.events_csv_path}")
    print(f"Wrote: {result.current_csv_path}")
    if result.png_path:
        print(f"Wrote: {result.png_path}")
    print(f"Wrote: {result.metadata_path}")


def print_rdf_outputs(result: Any) -> None:
    for pair in result.pairs.values():
        print(f"Wrote: {pair.csv_path}")
    png_paths = {pair.png_path for pair in result.pairs.values() if pair.png_path is not None}
    for path in sorted(png_paths):
        print(f"Wrote: {path}")
    print(f"Wrote: {result.metadata_path}")


def print_proton_sharing_outputs(result: Any) -> None:
    print(f"Wrote: {result.fes_csv_path}")
    print(f"Wrote: {result.shared_csv_path}")
    if result.png_path:
        print(f"Wrote: {result.png_path}")
    print(f"Wrote: {result.metadata_path}")


def print_proton_sharing_hbond_outputs(result: Any) -> None:
    print(f"Wrote outputs: {result.output_directory}")
    print(f"Wrote: {result.metadata_path}")
    print(f"Frames: {result.frames}; L1-L1 pairs: {result.l1_l1_pairs}; L1-L2 pairs: {result.l1_l2_pairs}")


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
    AnalysisModule(
        name="msd",
        help="Run a fixed-atom mean-squared-displacement workflow.",
        run=run_msd,
        print_outputs=print_msd_outputs,
        aliases=("atom-msd",),
        description="Compute 2D or 3D multiple-time-origin MSD for a fixed stable-atom selection.",
    ),
    AnalysisModule(
        name="defect-msd",
        help="Run a dynamically tracked defect MSD workflow.",
        run=run_defect_msd,
        print_outputs=print_defect_msd_outputs,
        description="Classify defects per frame, track them with Hungarian assignment, and compute lifetime-aware MSD.",
    ),
    AnalysisModule(
        name="conductivity",
        help="Run a fixed-carrier Nernst-Einstein conductivity workflow.",
        run=run_conductivity,
        print_outputs=print_conductivity_outputs,
        aliases=("conductivity-ne",),
        description="Fit a fixed stable-carrier MSD; not valid for identity-changing proton defects.",
    ),
    AnalysisModule(
        name="defect-conductivity",
        help="Run dynamically tracked defect conductivity workflows.",
        run=run_defect_conductivity,
        print_outputs=print_defect_conductivity_outputs,
        aliases=("conductivity-defect",),
        description="Compare dynamic-defect MSD Nernst-Einstein with collective STACIE Green-Kubo conductivity.",
    ),
    AnalysisModule(
        name="rdf",
        help="Run a radial-distribution-function workflow.",
        run=run_rdf,
        print_outputs=print_rdf_outputs,
        description="Compute element-, type-, species-, and layer-selective radial distribution functions.",
    ),
    AnalysisModule(
        name="proton-sharing",
        help="Run a proton-sharing free-energy-surface workflow.",
        run=run_proton_sharing,
        print_outputs=print_proton_sharing_outputs,
        aliases=("proton-fes",),
        description="Accumulate F(delta, R_OO) and a shared-proton local-coordinate F(s, rho) for selected donor-acceptor layers.",
    ),
    AnalysisModule(
        name="proton-sharing-hbond",
        help="Run H-bond-filtered, CN-resolved proton-sharing PES.",
        run=run_proton_sharing_hbond,
        print_outputs=print_proton_sharing_hbond_outputs,
        aliases=("proton-fes-hbond",),
        description="Accumulate globally pooled unit-pair F(delta, R_OO) surfaces with optional L1 acceptor H-bond coordination classes.",
    ),
)


def iter_analysis_modules() -> Iterable[AnalysisModule]:
    return iter(ANALYSIS_MODULES)


def get_analysis_module(command: str) -> AnalysisModule:
    for module in ANALYSIS_MODULES:
        if command in module.command_names:
            return module
    raise KeyError(command)
