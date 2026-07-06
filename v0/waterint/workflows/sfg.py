from __future__ import annotations

from pathlib import Path
from typing import Any

from waterint.config import require_mapping
from waterint.output.sfg_plotting import plot_overlay, plot_spectrum
from waterint.computation.sfg.processing import combine_cf, compute_ft, load_cf, write_cf, write_ft
from waterint.output.metadata import write_metadata
from waterint.computation.sfg.result import SfgResult
from waterint.computation.sfg.trajectory import compute_ssvvcf_from_trajectory
from waterint.workflows.common import resolve_path


def run_sfg(config: dict[str, Any]) -> SfgResult:
    sfg_cfg = require_mapping(config, "sfg")
    output_cfg = require_mapping(config, "output")
    mode = str(sfg_cfg.get("mode", "single"))
    if mode == "single":
        return run_single(config, sfg_cfg, output_cfg)
    if mode == "combine_bins":
        return run_combine_bins(config, sfg_cfg, output_cfg)
    if mode == "trajectory":
        return run_trajectory(config, sfg_cfg, output_cfg)
    raise ValueError("sfg.mode must be single, combine_bins, or trajectory.")


def run_single(config: dict[str, Any], sfg_cfg: dict[str, Any], output_cfg: dict[str, Any]) -> SfgResult:
    cf_path = resolve_path(config, sfg_cfg["cf"])
    outdir = resolve_path(config, output_cfg.get("directory", "output"))
    outdir.mkdir(parents=True, exist_ok=True)
    prefix = str(output_cfg.get("prefix", cf_path.with_suffix("").name))
    ft_path = outdir / f"{prefix}_FT.dat"
    png_path = outdir / f"{prefix}_FT.png" if bool(output_cfg.get("plot", True)) else None
    metadata_path = outdir / f"{prefix}_metadata.json"

    time_ps, corr, _counts = load_cf(cf_path)
    freq, signal = compute_ft(
        time_ps,
        corr,
        time_unit=str(sfg_cfg.get("time_unit", "ps")),
        nzeros=int(sfg_cfg.get("nzeros", 2000)),
    )
    write_ft(ft_path, freq, signal)
    png_paths: dict[str, Path] = {}
    if png_path is not None:
        plot_spectrum(
            path=png_path,
            freq_cm=freq,
            signal=signal,
            xmin=float(output_cfg.get("xmin", 0.0)),
            xmax=float(output_cfg.get("xmax", 4500.0)),
            flip=bool(output_cfg.get("flip", True)),
            title=str(output_cfg.get("title", prefix)),
            dpi=int(output_cfg.get("dpi", 220)),
        )
        png_paths["spectrum"] = png_path

    result = SfgResult("single", {"input": cf_path}, {"spectrum": ft_path}, png_paths, metadata_path)
    write_sfg_metadata(metadata_path, config=config, result=result)
    return result


def run_combine_bins(config: dict[str, Any], sfg_cfg: dict[str, Any], output_cfg: dict[str, Any]) -> SfgResult:
    input_dir = resolve_path(config, sfg_cfg["input_directory"])
    outdir = resolve_path(config, output_cfg.get("directory", input_dir))
    outdir.mkdir(parents=True, exist_ok=True)
    runs = string_list(sfg_cfg.get("runs"), name="sfg.runs")
    bins = string_list(sfg_cfg.get("bins"), name="sfg.bins")
    cf_prefix = str(sfg_cfg.get("cf_prefix", "ssVVCF"))
    include_nh1 = bool(sfg_cfg.get("include_nh1", True))
    skip_missing_nh1 = bool(sfg_cfg.get("skip_missing_nh1", True))
    nzeros = int(sfg_cfg.get("nzeros", 2000))

    cf_paths: dict[str, Path] = {}
    ft_paths: dict[str, Path] = {}
    png_paths: dict[str, Path] = {}
    all_overlay_inputs: dict[str, Path] = {}
    nh1_overlay_inputs: dict[str, Path] = {}

    for bin_label in bins:
        paths = [overall_path(input_dir, cf_prefix, bin_label, run) for run in runs]
        missing = [path for path in paths if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing SFG CF inputs for bin {bin_label}: {missing}")
        time_ps, corr, counts = combine_cf(paths)
        combined_cf = outdir / f"combined_{bin_label}_cf.dat"
        write_cf(combined_cf, time_ps, corr, counts)
        freq, signal = compute_ft(time_ps, corr, time_unit="ps", nzeros=nzeros)
        ft_path = outdir / f"combined_{bin_label}_FT.dat"
        write_ft(ft_path, freq, signal)
        cf_paths[f"{bin_label}:all"] = combined_cf
        ft_paths[f"{bin_label}:all"] = ft_path
        all_overlay_inputs[bin_label] = ft_path

        if include_nh1:
            nh1_paths = [nh1_path(input_dir, cf_prefix, bin_label, run) for run in runs]
            existing = [path for path in nh1_paths if path.exists()]
            if len(existing) == len(runs):
                time_ps_nh1, corr_nh1, counts_nh1 = combine_cf(nh1_paths)
                combined_nh1 = outdir / f"combined_{bin_label}_cf_nh1.dat"
                write_cf(combined_nh1, time_ps_nh1, corr_nh1, counts_nh1)
                freq_nh1, signal_nh1 = compute_ft(time_ps_nh1, corr_nh1, time_unit="ps", nzeros=nzeros)
                ft_nh1 = outdir / f"combined_{bin_label}_FT_nh1.dat"
                write_ft(ft_nh1, freq_nh1, signal_nh1)
                cf_paths[f"{bin_label}:nh1"] = combined_nh1
                ft_paths[f"{bin_label}:nh1"] = ft_nh1
                nh1_overlay_inputs[bin_label] = ft_nh1
            elif not skip_missing_nh1:
                missing_nh1 = [path for path in nh1_paths if not path.exists()]
                raise FileNotFoundError(f"Missing SFG nh1 inputs for bin {bin_label}: {missing_nh1}")

    if bool(output_cfg.get("plot", True)):
        overlay_all = outdir / str(output_cfg.get("overlay_all", "FT_bins_all.png"))
        plot_overlay(
            path=overlay_all,
            spectra=all_overlay_inputs,
            xmin=float(output_cfg.get("xmin", 2500.0)),
            xmax=float(output_cfg.get("xmax", 4200.0)),
            flip=bool(output_cfg.get("flip", True)),
            top_scale=optional_float(output_cfg.get("top_scale", 0.96)),
            title=str(output_cfg.get("title", "Overall FT across bins")),
            palette=str(output_cfg.get("palette", "project")),
            dpi=int(output_cfg.get("dpi", 220)),
        )
        png_paths["overlay_all"] = overlay_all
        if nh1_overlay_inputs:
            overlay_nh1 = outdir / str(output_cfg.get("overlay_nh1", "FT_bins_nh1.png"))
            plot_overlay(
                path=overlay_nh1,
                spectra=nh1_overlay_inputs,
                xmin=float(output_cfg.get("xmin", 2500.0)),
                xmax=float(output_cfg.get("xmax", 4200.0)),
                flip=bool(output_cfg.get("flip", True)),
                top_scale=optional_float(output_cfg.get("top_scale", 0.96)),
                title=str(output_cfg.get("title_nh1", "Hydroxide FT across bins")),
                palette=str(output_cfg.get("palette", "project")),
                dpi=int(output_cfg.get("dpi", 220)),
            )
            png_paths["overlay_nh1"] = overlay_nh1

    metadata_path = outdir / f"{str(output_cfg.get('prefix', 'sfg'))}_metadata.json"
    result = SfgResult("combine_bins", cf_paths, ft_paths, png_paths, metadata_path)
    write_sfg_metadata(metadata_path, config=config, result=result)
    return result


def run_trajectory(config: dict[str, Any], sfg_cfg: dict[str, Any], output_cfg: dict[str, Any]) -> SfgResult:
    outdir = resolve_path(config, output_cfg.get("directory", "output"))
    outdir.mkdir(parents=True, exist_ok=True)
    prefix = str(output_cfg.get("prefix", "sfg_trajectory"))
    output_prefix = outdir / prefix
    calc = compute_ssvvcf_from_trajectory(config, output_prefix)

    ft_path = outdir / f"{prefix}_FT.dat"
    freq, signal = compute_ft(
        calc.time_ps,
        calc.corr,
        time_unit="ps",
        nzeros=int(sfg_cfg.get("nzeros", 2000)),
    )
    write_ft(ft_path, freq, signal)

    png_paths: dict[str, Path] = {}
    png_path = outdir / f"{prefix}_FT.png" if bool(output_cfg.get("plot", True)) else None
    if png_path is not None:
        plot_spectrum(
            path=png_path,
            freq_cm=freq,
            signal=signal,
            xmin=float(output_cfg.get("xmin", 0.0)),
            xmax=float(output_cfg.get("xmax", 4500.0)),
            flip=bool(output_cfg.get("flip", True)),
            title=str(output_cfg.get("title", prefix)),
            dpi=int(output_cfg.get("dpi", 220)),
        )
        png_paths["spectrum"] = png_path

    result = SfgResult(
        "trajectory",
        {"ssvvcf": calc.cf_path, "zref": calc.zref_path},
        {"spectrum": ft_path},
        png_paths,
        outdir / f"{prefix}_metadata.json",
    )
    write_sfg_metadata(result.metadata_path, config=config, result=result)
    return result


def write_sfg_metadata(path: Path, *, config: dict[str, Any], result: SfgResult) -> None:
    write_metadata(
        path,
        {
            "analysis_name": "sfg",
            "package": "waterint",
            "mode": result.mode,
            "config_file": config.get("_config_path"),
            "outputs": {
                "cf": {label: str(path) for label, path in result.cf_paths.items()},
                "ft": {label: str(path) for label, path in result.ft_paths.items()},
                "png": {label: str(path) for label, path in result.png_paths.items()},
            },
            "config": {key: value for key, value in config.items() if not key.startswith("_")},
        },
    )


def overall_path(outdir: Path, prefix: str, bin_label: str, run: str) -> Path:
    return outdir / f"{prefix}_{bin_label}_{run}.dat"


def nh1_path(outdir: Path, prefix: str, bin_label: str, run: str) -> Path:
    return outdir / f"{prefix}_{bin_label}_{run}_cf_nh1.dat"


def string_list(value: Any, *, name: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty list.")
    return [str(item) for item in value]


def optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)
