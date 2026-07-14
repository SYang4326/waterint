from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from waterint.config import require_mapping
from waterint._00_io.common import TrajectoryFrame
from waterint._01_core.selection import SelectionContext
from waterint._02_computation.hbond import (
    HbondResult,
    HbondState,
    accumulate_frame_counts,
    classes_by_species,
    finalize_hbond_state,
    frame_hbond_classes,
    new_hbond_state,
    species_labels,
)
from waterint._03_output.hbond import plot_hbond_fractions, write_hbond_csv, write_raw_hbond_csv
from waterint._03_output.metadata import write_metadata
from waterint._04_workflows.workflows.common import parse_cell, required_workflow_sections, resolve_path
from waterint._04_workflows.workflows.framewise import run_framewise_analysis


@dataclass
class HbondWorkflowState:
    """Configuration and accumulators carried through the frame loop."""

    hbond: HbondState
    selection_cfg: dict[str, Any]
    hbond_cfg: dict[str, Any]
    context: SelectionContext
    backend: str


def run_hbond(config: dict[str, Any]) -> HbondResult:
    """Run the config-driven H-bond analysis and write all requested outputs.

    This workflow owns orchestration only: it resolves config and paths,
    streams trajectory frames through the computation module, then delegates
    CSV/plot/metadata writing to the output layer.
    """

    input_cfg, system_cfg, output_cfg = required_workflow_sections(config)
    selection_cfg = require_mapping(config, "selection")
    hbond_cfg = require_mapping(config, "hbond")

    traj_path = resolve_path(config, input_cfg["trajectory"])
    labels = species_labels(selection_cfg)
    classes = classes_by_species(hbond_cfg, labels)
    backend = str(hbond_cfg.get("backend", "auto")).lower()
    if backend not in {"auto", "python", "cpp"}:
        raise ValueError("hbond.backend must be auto, python, or cpp.")
    state = HbondWorkflowState(
        hbond=new_hbond_state(classes, labels),
        selection_cfg=selection_cfg,
        hbond_cfg=hbond_cfg,
        context=SelectionContext.from_input_config(input_cfg),
        backend=backend,
    )
    # The common framewise runner handles trajectory I/O, stride, frame limits, and cell resolution.
    framewise = run_framewise_analysis(
        traj_path=traj_path,
        input_cfg=input_cfg,
        configured_cell=parse_cell(system_cfg.get("cell", "auto")),
        state=state,
        accumulate=_accumulate_hbond_frame,
        finalize=_finalize_hbond,
    )
    result = framewise.result

    outdir = resolve_path(config, output_cfg.get("directory", "output"))
    outdir.mkdir(parents=True, exist_ok=True)
    prefix = str(output_cfg.get("prefix", "hbond"))
    csv_path = outdir / f"{prefix}.csv"
    raw_csv_path = outdir / f"{prefix}_raw.csv"
    png_path = outdir / f"{prefix}.png" if bool(output_cfg.get("plot", True)) else None
    metadata_path = outdir / f"{prefix}_metadata.json"

    # Computation is complete here; the remaining work only serializes or plots the result.
    write_hbond_csv(csv_path, result.counts, result.fractions, result.samples_total, result.classes)
    write_raw_hbond_csv(raw_csv_path, result.raw_counts, result.counts, result.samples_total, result.classes)
    if png_path is not None:
        plot_species_labels = [
            species
            for species in result.species_labels
            if result.samples_total[species] > 0 or bool(output_cfg.get("show_empty_species", False))
        ]
        if not plot_species_labels:
            plot_species_labels = result.species_labels
        plot_hbond_fractions(
            path=png_path,
            fractions={species: result.fractions[species] for species in plot_species_labels},
            classes={species: result.classes[species] for species in plot_species_labels},
            title=str(output_cfg.get("title", "H-bond topology fractions")),
            ylabel=str(output_cfg.get("ylabel", "Fraction")),
            dpi=int(output_cfg.get("dpi", 220)),
        )
    write_metadata(
        metadata_path,
        {
            "analysis_name": "hbond",
            "package": "waterint",
            "config_file": config.get("_config_path"),
            "trajectory": str(traj_path),
            "frames": result.frames,
            "species_labels": result.species_labels,
            "classes": result.classes,
            "samples_total": result.samples_total,
            "outputs": {
                "csv": str(csv_path),
                "raw_csv": str(raw_csv_path),
                "png": str(png_path) if png_path is not None else None,
            },
            "config": {key: value for key, value in config.items() if not key.startswith("_")},
        },
    )
    return replace(result, csv_path=csv_path, raw_csv_path=raw_csv_path, png_path=png_path, metadata_path=metadata_path)


def _accumulate_hbond_frame(
    state: HbondWorkflowState,
    frame: TrajectoryFrame,
    cell: tuple[float, float, float],
) -> None:
    """Analyze one frame and merge its counts into the workflow state."""

    frame_counts, frame_raw_counts, frame_samples = frame_hbond_classes(
        frame=frame,
        selection_cfg=state.selection_cfg,
        hbond_cfg=state.hbond_cfg,
        classes=state.hbond.classes,
        selected_species=state.hbond.species_labels,
        context=state.context,
        cell=cell,
        backend=state.backend,
    )
    accumulate_frame_counts(state.hbond, frame_counts, frame_raw_counts, frame_samples)


def _finalize_hbond(
    state: HbondWorkflowState,
    _cell: tuple[float, float, float],
) -> HbondResult:
    """Finish the framewise run by calculating topology fractions."""

    return finalize_hbond_state(state.hbond)
