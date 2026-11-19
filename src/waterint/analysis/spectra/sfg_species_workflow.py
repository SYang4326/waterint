"""Config workflows: species-resolved SFG on long runs (``sfg-species``) and the equal-run summary
(``sfg-species-aggregate``). Definitions: ``waterint.analysis.spectra.sfg_species_core``.

sfg-species:
    input:
      format: gpumd-qnpz            # segment dirs with velocities (only the needed chunks are decoded); or lammpstrj
      segments: [run/seg_02]        # frames are numbered over the listed segments
      first_frame: 25000            # e.g. 5.1 ns into seg_02 ... (global index over the listed segments)
      n_frames: 125000
      # trajectory: dump.lammpstrj  # lammpstrj (needs vx vy vz in A/ps)
    sfg_species:
      block_frames: 25000           # independent blocks (pairs across a block edge are lost, ~1 ps per block)
      n_lag: 250
      free_oh: true                 # also the top-surface free-OH peak -> frequency scale 3700 / nu_free
    output: {directory: sfg, prefix: run00}
  -> <prefix>_block<k>.npz per block, <prefix>.npz (blocks summed: numerators and counts, then channels / spectra)

sfg-species-aggregate:
    input: {runs: [sfg/run00.npz, sfg/run01.npz, ...], reference: null}   # reference: spectra npz to overlay
    sfg_species: {scale: auto}      # auto = equal-run mean of the runs' free-OH scales, a number, or 1
    output: {directory: sfg, prefix: summary}
  -> <prefix>.npz (per-run, mean, SD, SEM), <prefix>_peaks.txt, <prefix>.png
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from waterint.analysis.spectra.sfg_species_core import (
    N_LAG, free_oh_peak, gibbs_top_surface, layered_block, species_channels, species_sfg_config, spectra, zref_top_layer_mean,
)
from waterint.analysis.spectra._workflow_utils import resolve_path
from waterint.analysis.spectra._workflow_utils import require_mapping


@dataclass(frozen=True)
class SfgSpeciesWorkflowResult:
    npz_path: Path
    block_paths: list
    metadata_path: Path
    frames: int


@dataclass(frozen=True)
class SfgSpeciesAggregateResult:
    npz_path: Path
    png_path: Path
    peaks_path: Path
    runs: int


def _save(path, keys, nums, counts, zref, frames, n_lag, dt, top=None):
    names, C, den_full, den_z04 = species_channels(keys, nums, counts)
    S, freq = spectra(C, dt)
    extra = {}
    if top is not None:
        num_t, cnt_t, zg = top
        k, nu, scale = free_oh_peak(num_t / cnt_t, dt)
        extra = dict(free_oh_numerator=num_t, free_oh_counts=cnt_t, z_gds=np.array(zg), free_oh_peak_index=np.array(k),
                     free_oh_peak_cm1=np.array(nu), scale=np.array(scale))
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, channel_names=np.array(names), lag_ps=np.arange(n_lag) * dt, cf=C, spectrum=S, freq_cm1=freq,
                        full_system_denominator=den_full, z0_4_denominator=den_z04, all_channel_keys=np.array(keys),
                        all_numerators=nums, all_counts=counts, zref=zref, frames=np.array(frames), **extra)
    tmp.replace(path)
    return None if "scale" not in extra else float(extra["scale"])


def run_sfg_species_aggregate(config: dict[str, Any]) -> SfgSpeciesAggregateResult:
    input_cfg = require_mapping(config, "input")
    output_cfg = require_mapping(config, "output")
    sc = dict(config.get("sfg_species") or {})
    files = [resolve_path(config, str(p)) for p in input_cfg["runs"]]
    zs = [np.load(f) for f in files]
    names = [str(x) for x in zs[0]["channel_names"]]
    freq = zs[0]["freq_cm1"]
    S = np.array([z["spectrum"] for z in zs])
    C = np.array([z["cf"] for z in zs])
    n = len(zs)
    mean, sd = S.mean(axis=0), (S.std(axis=0, ddof=1) if n > 1 else np.zeros_like(S[0]))
    sem = sd / np.sqrt(n)
    sv = sc.get("scale", "auto")
    if sv == "auto":
        sc_runs = [float(z["scale"]) for z in zs if "scale" in z.files]
        scale = float(np.mean(sc_runs)) if sc_runs else 1.0
    else:
        scale = float(sv)
    out_dir = resolve_path(config, output_cfg.get("directory", "."))
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = str(output_cfg.get("prefix", "sfg_species_summary"))
    npz_path, png_path, peaks_path = out_dir / f"{prefix}.npz", out_dir / f"{prefix}.png", out_dir / f"{prefix}_peaks.txt"
    np.savez_compressed(npz_path, channel_names=np.array(names), freq_cm1=freq, freq_scaled_cm1=freq * scale, scale=np.array(scale),
                        per_run_spectrum=S, per_run_cf=C, mean=mean, sd=sd, sem=sem, runs=np.array([str(f) for f in files]),
                        lag_ps=zs[0]["lag_ps"])
    fs = freq * scale
    w = (fs >= 2500) & (fs <= 4150)
    lines = [f"{n} runs; frequency scale {scale:.10f}; peaks (2500-4150 cm-1, scaled axis) of the equal-run mean: channel  max(+) @  min(-) @"]
    for i, c in enumerate(names):
        m = mean[i, w]
        lines.append(f"  {c:14s} {m.max():+9.3g} @ {fs[w][m.argmax()]:6.0f}   {m.min():+9.3g} @ {fs[w][m.argmin()]:6.0f}")
    peaks_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ref = input_cfg.get("reference")
    R = np.load(resolve_path(config, str(ref)), allow_pickle=True) if ref else None
    fig, axs = plt.subplots(2, 3, figsize=(11, 6), sharex=True)
    for i, (ax, c) in enumerate(zip(axs.ravel(), names)):
        ax.axhline(0, color="0.7", lw=0.8)
        ax.plot(fs[w], mean[i, w], color="#1f5fa8", lw=1.6, label=f"{output_cfg.get('label', 'this work')} (n={n})")
        ax.fill_between(fs[w], (mean - sem)[i, w], (mean + sem)[i, w], color="#1f5fa8", alpha=0.25, lw=0)
        if R is not None:
            rn = [str(x) for x in R["channels"]]
            if c in rn:
                j = rn.index(c); fr = R["frequency_cm1"]; wr = (fr >= 2500) & (fr <= 4150)
                ax.plot(fr[wr], R["classical_ft_mean"][j][wr], color="#c0504d", lw=1.2, label="reference")
        ax.set_title(c.replace("_z0_4", "").replace("_", " "), fontsize=10)
    for ax in axs[1]:
        ax.set_xlabel(f"wavenumber (cm$^{{-1}}$, x {scale:.4f})", fontsize=9)
    axs[0, 0].legend(fontsize=8, frameon=False)
    fig.suptitle("Species-resolved SFG, 0 ≤ z$_{rel}$ < 4 Å", fontsize=11)
    fig.tight_layout(); fig.savefig(png_path, dpi=200); plt.close(fig)
    return SfgSpeciesAggregateResult(npz_path, png_path, peaks_path, n)
