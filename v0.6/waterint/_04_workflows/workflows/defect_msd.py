from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from waterint.config import require_mapping
from waterint._03_output.defect_transport import (
    write_defect_events_csv,
    write_defect_msd_csv,
    write_defect_tracks_csv,
)
from waterint._03_output.metadata import write_metadata
from waterint._03_output.msd import plot_msd
from waterint._04_workflows.workflows.common import resolve_path
from waterint._04_workflows.workflows.defect_common import run_defect_analysis


@dataclass(frozen=True)
class DefectMsdWorkflowResult:
    csv_path: Path
    tracks_csv_path: Path
    events_csv_path: Path
    png_path: Path | None
    metadata_path: Path
    mean_carriers: float
    segments: int


def run_defect_msd(config: dict[str, Any]) -> DefectMsdWorkflowResult:
    output_cfg = require_mapping(config, "output")
    analysis = run_defect_analysis(config)
    outdir = resolve_path(config, output_cfg.get("directory", "output"))
    outdir.mkdir(parents=True, exist_ok=True)
    prefix = str(output_cfg.get("prefix", "defect_msd"))
    csv_path = outdir / f"{prefix}.csv"
    tracks_path = outdir / f"{prefix}_tracks.csv"
    events_path = outdir / f"{prefix}_events.csv"
    png_path = outdir / f"{prefix}.png" if bool(output_cfg.get("plot", True)) else None
    metadata_path = outdir / f"{prefix}_metadata.json"
    write_defect_msd_csv(csv_path, analysis.msd)
    write_defect_tracks_csv(tracks_path, analysis.tracking, analysis.steps)
    write_defect_events_csv(events_path, analysis.tracking, analysis.steps)
    if png_path is not None:
        plot_msd(
            png_path,
            analysis.msd.time_ps,
            analysis.msd.msd_a2,
            title=str(output_cfg.get("title", "Dynamic defect MSD")),
            dimensionality=analysis.msd.dimensionality,
            dpi=int(output_cfg.get("dpi", 220)),
        )
    write_metadata(
        metadata_path,
        {
            "analysis_name": "defect-msd",
            "method": "framewise species classification and gated Hungarian tracking",
            "package": "waterint",
            "config_file": config.get("_config_path"),
            "frames": int(len(analysis.tracking.carrier_counts)),
            "initial_carriers": int(analysis.tracking.carrier_counts[0]),
            "mean_carriers": analysis.tracking.mean_carriers,
            "segments": len(analysis.tracking.segments),
            "births": int(analysis.tracking.births.sum()),
            "deaths": int(analysis.tracking.deaths.sum()),
            "outputs": {
                "csv": str(csv_path),
                "tracks_csv": str(tracks_path),
                "events_csv": str(events_path),
                "png": None if png_path is None else str(png_path),
            },
            "config": {key: value for key, value in config.items() if not key.startswith("_")},
        },
    )
    return DefectMsdWorkflowResult(
        csv_path=csv_path,
        tracks_csv_path=tracks_path,
        events_csv_path=events_path,
        png_path=png_path,
        metadata_path=metadata_path,
        mean_carriers=analysis.tracking.mean_carriers,
        segments=len(analysis.tracking.segments),
    )
