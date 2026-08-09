from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from waterint.config import require_mapping
from waterint._00_io.common import TrajectoryFrame
from waterint._01_core.selection import SelectionContext
from waterint._03_output.sfg import plot_overlay, plot_spectrum
from waterint._02_computation.sfg import (
    SfgResult,
    combine_cf,
    compute_ft,
    compute_layered_ssvvcf_from_frames,
    compute_ssvvcf_from_frames,
    load_cf,
    write_cf,
    write_ft,
    write_zrefs,
)
from waterint._03_output.metadata import write_metadata
from waterint._04_workflows.workflows.common import iter_frames, parse_cell, resolve_path


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
    species_tokens = combined_species_tokens(sfg_cfg)
    complete_partition = set(species_tokens) == set(ALL_SPECIES_TOKENS)
    # A complete partition is normally requested specifically to make a
    # publication contribution plot. Silently dropping a species would defeat
    # that guarantee, whereas legacy nh1-only combinations remain permissive.
    default_skip_missing = False if complete_partition else bool(sfg_cfg.get("skip_missing_nh1", True))
    skip_missing_species = bool(sfg_cfg.get("skip_missing_species", default_skip_missing))
    validate_closure = bool(sfg_cfg.get("validate_species_closure", True))
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

        species_corr: dict[str, np.ndarray] = {}
        species_ft: dict[str, np.ndarray] = {}
        for token in species_tokens:
            paths = [species_path(input_dir, cf_prefix, bin_label, run, token) for run in runs]
            existing = [path for path in paths if path.exists()]
            if len(existing) == len(runs):
                species_time, species_curve, species_counts = combine_cf(paths)
                if species_time.shape != time_ps.shape or np.max(np.abs(species_time - time_ps)) > 1e-9:
                    raise ValueError(f"SFG species time grid mismatch for {bin_label}:{token}")
                combined_species = outdir / f"combined_{bin_label}_cf_{token}.dat"
                write_cf(combined_species, species_time, species_curve, species_counts)
                species_frequency, species_signal = compute_ft(
                    species_time, species_curve, time_unit="ps", nzeros=nzeros
                )
                if species_frequency.shape != freq.shape or np.max(np.abs(species_frequency - freq)) > 1e-9:
                    raise ValueError(f"SFG species frequency grid mismatch for {bin_label}:{token}")
                ft_species = outdir / f"combined_{bin_label}_FT_{token}.dat"
                write_ft(ft_species, species_frequency, species_signal)
                cf_paths[f"{bin_label}:{token}"] = combined_species
                ft_paths[f"{bin_label}:{token}"] = ft_species
                species_corr[token] = species_curve
                species_ft[token] = species_signal
                if token == "nh1":
                    nh1_overlay_inputs[bin_label] = ft_species
            elif not skip_missing_species:
                missing = [path for path in paths if not path.exists()]
                raise FileNotFoundError(f"Missing SFG species inputs for {bin_label}:{token}: {missing}")

        if validate_closure and complete_partition:
            missing = [token for token in ALL_SPECIES_TOKENS if token not in species_corr]
            if missing:
                if not skip_missing_species:
                    raise FileNotFoundError(f"Missing SFG species inputs required for closure in {bin_label}: {missing}")
            else:
                verify_additive_species_closure(
                    bin_label,
                    corr,
                    species_corr,
                    signal,
                    species_ft,
                )

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
    input_cfg = require_mapping(config, "input")
    system_cfg = require_mapping(config, "system")
    traj_path = resolve_path(config, input_cfg["trajectory"])
    frames = list(iter_frames(traj_path, input_cfg))
    configured_cell = parse_cell(system_cfg.get("cell", "auto"))
    cell = configured_cell or (frames[0].cell if frames else None)
    if cell is None:
        raise ValueError("No cell information was available. Set system.cell manually.")

    outdir = resolve_path(config, output_cfg.get("directory", "output"))
    outdir.mkdir(parents=True, exist_ok=True)
    prefix = str(output_cfg.get("prefix", "sfg_trajectory"))
    output_prefix = outdir / prefix
    cf_path = Path(str(output_prefix) + ".dat")
    zref_path = Path(str(output_prefix) + "_zref.dat")
    compute_cfg = dict(sfg_cfg)
    if compute_cfg.get("zref_file"):
        compute_cfg["zref_file"] = resolve_path(config, compute_cfg["zref_file"])
    if "layer_bins" in compute_cfg:
        return run_layered_trajectory(
            config=config,
            input_cfg=input_cfg,
            frames=frames,
            cell=cell,
            sfg_cfg=compute_cfg,
            output_cfg=output_cfg,
            outdir=outdir,
            prefix=prefix,
        )
    calc = compute_ssvvcf_from_frames(
        frames,
        cell=cell,
        sfg_cfg=compute_cfg,
        context=SelectionContext.from_input_config(input_cfg),
    )
    write_cf(cf_path, calc.time_ps, calc.corr, calc.counts)
    write_zrefs(zref_path, frames, calc.zrefs)

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
        {"ssvvcf": cf_path, "zref": zref_path},
        {"spectrum": ft_path},
        png_paths,
        outdir / f"{prefix}_metadata.json",
    )
    write_sfg_metadata(result.metadata_path, config=config, result=result, velocity_source=calc.velocity_source)
    return result


def run_layered_trajectory(
    *,
    config: dict[str, Any],
    input_cfg: dict[str, Any],
    frames: list[TrajectoryFrame],
    cell: tuple[float, float, float],
    sfg_cfg: dict[str, Any],
    output_cfg: dict[str, Any],
    outdir: Path,
    prefix: str,
) -> SfgResult:
    """Write one all-OH and species CF/FT pair for every configured layer."""

    requested_backend = str(sfg_cfg.get("backend", "auto")).lower()
    if requested_backend not in {"auto", "python", "cpp"}:
        raise ValueError("sfg.backend must be auto, python, or cpp.")
    calc = compute_layered_ssvvcf_from_frames(
        frames,
        cell=cell,
        sfg_cfg=sfg_cfg,
        context=SelectionContext.from_input_config(input_cfg),
    )
    run_label = output_cfg.get("run_label")
    filename_prefix = f"{prefix}_"
    cf_paths: dict[str, Path] = {}
    ft_paths: dict[str, Path] = {}
    png_paths: dict[str, Path] = {}
    for channel_name, channel in calc.channels.items():
        layer, species = channel_name.split(":", 1)
        stem = f"{filename_prefix}{layer}"
        if run_label:
            stem += f"_{run_label}"
        if species != "all":
            stem += f"_cf_{species_token(species)}"
        cf_path = outdir / f"{stem}.dat"
        write_cf(cf_path, channel.time_ps, channel.corr, channel.counts)
        cf_paths[channel_name] = cf_path
        frequency, signal = compute_ft(
            channel.time_ps,
            channel.corr,
            time_unit="ps",
            nzeros=int(sfg_cfg.get("nzeros", 2000)),
        )
        ft_path = outdir / f"{stem}_FT.dat"
        write_ft(ft_path, frequency, signal)
        ft_paths[channel_name] = ft_path
        if bool(output_cfg.get("plot", True)):
            png_path = outdir / f"{stem}_FT.png"
            plot_spectrum(
                path=png_path,
                freq_cm=frequency,
                signal=signal,
                xmin=float(output_cfg.get("xmin", 0.0)),
                xmax=float(output_cfg.get("xmax", 4500.0)),
                flip=bool(output_cfg.get("flip", True)),
                title=f"{output_cfg.get('title', prefix)}: {channel_name}",
                dpi=int(output_cfg.get("dpi", 220)),
            )
            png_paths[channel_name] = png_path

    zref_path = outdir / f"{prefix}_zref.dat"
    write_zrefs(zref_path, frames, calc.zrefs)
    cf_paths["zref"] = zref_path
    result = SfgResult(
        "trajectory_layered",
        cf_paths,
        ft_paths,
        png_paths,
        outdir / f"{prefix}_metadata.json",
    )
    write_sfg_metadata(result.metadata_path, config=config, result=result, velocity_source=calc.velocity_source)
    return result


def species_token(species: str) -> str:
    """Keep the established hydroxide filename while making other names safe."""

    if species == "OH-":
        return "nh1"
    return species.lower().replace("+", "plus").replace("-", "minus").replace("_", "")


# Output tokens deliberately retain ``nh1`` for OH- so existing Fig. 2 files
# remain usable. The other tokens are the filenames written by
# ``run_layered_trajectory``.
ALL_SPECIES_TOKENS = ("o2minus", "nh1", "h2o", "h3oplus", "oother")
_SPECIES_TOKEN_ALIASES = {
    "O2-": "o2minus",
    "o2minus": "o2minus",
    "OH-": "nh1",
    "nh1": "nh1",
    "H2O": "h2o",
    "h2o": "h2o",
    "H3O+": "h3oplus",
    "h3oplus": "h3oplus",
    "O_other": "oother",
    "oother": "oother",
}


def combined_species_tokens(sfg_cfg: dict[str, Any]) -> list[str]:
    """Resolve requested species CFs for a multi-run SFG combination.

    ``species_channels: all`` is the publication-safe choice: it requests the
    complete dynamic oxygen-species partition and enables an exact closure
    check against the all-OH result. Omitting it preserves the historical
    ``include_nh1`` behaviour.
    """

    raw = sfg_cfg.get("species_channels")
    if raw is None:
        return ["nh1"] if bool(sfg_cfg.get("include_nh1", True)) else []
    if raw == "all":
        return list(ALL_SPECIES_TOKENS)
    if not isinstance(raw, list):
        raise ValueError("sfg.species_channels must be 'all' or a list of oxygen species labels.")
    tokens: list[str] = []
    for value in raw:
        label = str(value)
        if label == "all":
            tokens.extend(ALL_SPECIES_TOKENS)
            continue
        try:
            tokens.append(_SPECIES_TOKEN_ALIASES[label])
        except KeyError as exc:
            raise ValueError(f"Unknown SFG species channel for combination: {label!r}") from exc
    if len(tokens) != len(set(tokens)):
        raise ValueError("sfg.species_channels cannot contain duplicate species labels.")
    return tokens


def species_path(input_dir: Path, prefix: str, bin_label: str, run: str, token: str) -> Path:
    return input_dir / f"{prefix}_{bin_label}_{run}_cf_{token}.dat"


def verify_additive_species_closure(
    bin_label: str,
    total_corr: np.ndarray,
    species_corr: dict[str, np.ndarray],
    total_ft: np.ndarray,
    species_ft: dict[str, np.ndarray],
) -> None:
    """Reject conditional/per-peak-normalized data masquerading as contributions."""

    corr_difference = np.sum([species_corr[token] for token in ALL_SPECIES_TOKENS], axis=0) - total_corr
    ft_difference = np.sum([species_ft[token] for token in ALL_SPECIES_TOKENS], axis=0) - total_ft
    corr_error = float(np.max(np.abs(corr_difference)))
    ft_error = float(np.max(np.abs(ft_difference)))
    if not (
        np.allclose(corr_difference, 0.0, rtol=1e-10, atol=1e-12)
        and np.allclose(ft_difference, 0.0, rtol=1e-10, atol=1e-12)
    ):
        raise ValueError(
            f"SFG species do not add to all-OH for bin {bin_label!r} "
            f"(max CF error {corr_error:.3e}; max FT error {ft_error:.3e}). "
            "Use trajectory outputs generated with species_normalization: additive; "
            "do not combine conditional or independently peak-normalized species spectra."
        )


def write_sfg_metadata(
    path: Path,
    *,
    config: dict[str, Any],
    result: SfgResult,
    velocity_source: str | None = None,
) -> None:
    metadata = {
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
    }
    if velocity_source is not None:
        metadata["velocity_source"] = velocity_source
    write_metadata(path, metadata)


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
