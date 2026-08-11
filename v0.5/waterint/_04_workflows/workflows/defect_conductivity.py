from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from waterint.config import require_mapping
from waterint._02_computation.conductivity import (
    ConductivityResult,
    compute_nernst_einstein_conductivity,
)
from waterint._02_computation.defect_transport import (
    GreenKuboResult,
    compute_green_kubo_conductivity,
)
from waterint._03_output.conductivity import plot_conductivity_msd, write_conductivity_msd_csv
from waterint._03_output.defect_transport import (
    write_defect_current_csv,
    write_defect_events_csv,
    write_defect_msd_csv,
    write_defect_tracks_csv,
)
from waterint._03_output.metadata import write_metadata
from waterint._04_workflows.workflows.common import resolve_path
from waterint._04_workflows.workflows.defect_common import (
    analysis_volume_from_cells,
    run_defect_analysis,
)
from waterint._04_workflows.workflows.msd import axis_from_value


@dataclass(frozen=True)
class DefectConductivityWorkflowResult:
    csv_path: Path
    msd_csv_path: Path
    tracks_csv_path: Path
    events_csv_path: Path
    current_csv_path: Path
    png_path: Path | None
    metadata_path: Path
    nernst_einstein: ConductivityResult | None
    green_kubo: GreenKuboResult | None


def run_defect_conductivity(config: dict[str, Any]) -> DefectConductivityWorkflowResult:
    conductivity_cfg = require_mapping(config, "defect_conductivity")
    output_cfg = require_mapping(config, "output")
    charge_e = float(conductivity_cfg.get("charge_e", -1.0))
    temperature_k = float(conductivity_cfg["temperature_K"])
    estimator = str(conductivity_cfg.get("estimator", "green_kubo")).lower()
    if estimator not in {"nernst_einstein", "green_kubo", "both"}:
        raise ValueError("defect_conductivity.estimator must be nernst_einstein, green_kubo, or both.")
    analysis = run_defect_analysis(config, charge_e=charge_e)
    volume_cfg = conductivity_cfg.get("volume", {"mode": "cell"})
    if not isinstance(volume_cfg, dict):
        raise ValueError("defect_conductivity.volume must be a mapping.")
    msd_cfg = require_mapping(config, "defect_msd")
    volume_a3, thickness_a = analysis_volume_from_cells(
        analysis.cell_vectors,
        volume_cfg,
        default_axis=msd_cfg.get("plane_normal_axis", "z"),
    )
    dimensions = 2 if analysis.msd.dimensionality == "2d" else 3
    ne_result = None
    if estimator in {"nernst_einstein", "both"}:
        fit_range = conductivity_cfg.get("fit_range_ps")
        if not isinstance(fit_range, list) or len(fit_range) != 2:
            raise ValueError("defect_conductivity.fit_range_ps is required for defect Nernst-Einstein.")
        ne_result = compute_nernst_einstein_conductivity(
            analysis.msd.time_ps,
            analysis.msd.msd_a2,
            carrier_count=analysis.tracking.mean_carriers,
            volume_a3=volume_a3,
            temperature_k=temperature_k,
            charge_e=charge_e,
            dimensions=dimensions,
            fit_range_ps=(float(fit_range[0]), float(fit_range[1])),
            sheet_thickness_a=thickness_a,
        )
    gk_result = None
    if estimator in {"green_kubo", "both"}:
        gk_result = compute_green_kubo_conductivity(
            analysis.tracking,
            volume_a3=volume_a3,
            temperature_k=temperature_k,
            dimensionality=analysis.msd.dimensionality,
            plane_normal_axis=axis_from_value(msd_cfg.get("plane_normal_axis", "z")),
        )

    outdir = resolve_path(config, output_cfg.get("directory", "output"))
    outdir.mkdir(parents=True, exist_ok=True)
    prefix = str(output_cfg.get("prefix", "defect_conductivity"))
    csv_path = outdir / f"{prefix}.csv"
    msd_path = outdir / f"{prefix}_msd.csv"
    tracks_path = outdir / f"{prefix}_tracks.csv"
    events_path = outdir / f"{prefix}_events.csv"
    current_path = outdir / f"{prefix}_current.csv"
    png_path = outdir / f"{prefix}.png" if bool(output_cfg.get("plot", True)) and ne_result is not None else None
    metadata_path = outdir / f"{prefix}_metadata.json"
    _write_summary(
        csv_path,
        ne_result=ne_result,
        green_kubo=gk_result,
        temperature_k=temperature_k,
        charge_e=charge_e,
        mean_carriers=analysis.tracking.mean_carriers,
        volume_a3=volume_a3,
        thickness_a=thickness_a,
    )
    write_defect_msd_csv(msd_path, analysis.msd)
    write_defect_tracks_csv(tracks_path, analysis.tracking, analysis.steps)
    write_defect_events_csv(events_path, analysis.tracking, analysis.steps)
    write_defect_current_csv(current_path, analysis.tracking, analysis.steps)
    if png_path is not None and ne_result is not None:
        plot_conductivity_msd(
            png_path,
            ne_result,
            title=str(output_cfg.get("title", "Dynamic defect MSD and Nernst-Einstein fit")),
            dpi=int(output_cfg.get("dpi", 220)),
        )
    write_metadata(
        metadata_path,
        {
            "analysis_name": "defect-conductivity",
            "recommended_estimator": "collective Green-Kubo",
            "tracking_method": "framewise species classification and gated Hungarian assignment",
            "package": "waterint",
            "config_file": config.get("_config_path"),
            "frames": int(len(analysis.tracking.carrier_counts)),
            "initial_carriers": int(analysis.tracking.carrier_counts[0]),
            "mean_carriers": analysis.tracking.mean_carriers,
            "segments": len(analysis.tracking.segments),
            "births": int(analysis.tracking.births.sum()),
            "deaths": int(analysis.tracking.deaths.sum()),
            "volume_A3": volume_a3,
            "nernst_einstein_S_per_m": None if ne_result is None else ne_result.conductivity_s_per_m,
            "green_kubo_S_per_m": None if gk_result is None else gk_result.conductivity_s_per_m,
            "green_kubo_std_S_per_m": None if gk_result is None else gk_result.conductivity_std_s_per_m,
            "outputs": {
                "csv": str(csv_path),
                "msd_csv": str(msd_path),
                "tracks_csv": str(tracks_path),
                "events_csv": str(events_path),
                "current_csv": str(current_path),
                "png": None if png_path is None else str(png_path),
            },
            "config": {key: value for key, value in config.items() if not key.startswith("_")},
        },
    )
    return DefectConductivityWorkflowResult(
        csv_path=csv_path,
        msd_csv_path=msd_path,
        tracks_csv_path=tracks_path,
        events_csv_path=events_path,
        current_csv_path=current_path,
        png_path=png_path,
        metadata_path=metadata_path,
        nernst_einstein=ne_result,
        green_kubo=gk_result,
    )


def _write_summary(
    path: Path,
    *,
    ne_result: ConductivityResult | None,
    green_kubo: GreenKuboResult | None,
    temperature_k: float,
    charge_e: float,
    mean_carriers: float,
    volume_a3: float,
    thickness_a: float | None,
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write(
            "estimator,temperature_K,charge_e,mean_carriers,volume_A3,conductivity_S_per_m,"
            "statistical_std_S_per_m,conductivity_S_per_cm,sheet_conductance_S\n"
        )
        if ne_result is not None:
            sheet = "" if ne_result.sheet_conductance_s is None else f"{ne_result.sheet_conductance_s:.10g}"
            handle.write(
                f"defect_nernst_einstein,{temperature_k:.10g},{charge_e:.10g},{mean_carriers:.10g},"
                f"{volume_a3:.10g},{ne_result.conductivity_s_per_m:.10g},,"
                f"{ne_result.conductivity_s_per_m / 100.0:.10g},{sheet}\n"
            )
        if green_kubo is not None:
            sheet = "" if thickness_a is None else f"{green_kubo.conductivity_s_per_m * thickness_a * 1.0e-10:.10g}"
            handle.write(
                f"defect_green_kubo,{temperature_k:.10g},{charge_e:.10g},{mean_carriers:.10g},"
                f"{volume_a3:.10g},{green_kubo.conductivity_s_per_m:.10g},"
                f"{green_kubo.conductivity_std_s_per_m:.10g},{green_kubo.conductivity_s_per_m / 100.0:.10g},{sheet}\n"
            )
