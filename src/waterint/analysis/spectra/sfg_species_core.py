"""Species-resolved SFG (ssVVCF), array path for long runs.

Channels (optical window 0 <= z_rel < 4 A; z_rel from the mean z of the Mg within 0.70 A of the highest Mg):
  total_z0_4 = all O-H in [0, 4); Os_H = O with one H in [0, 1.5) (lattice O); L1_OH / L1_H2O = O with 1 / 2 H in
  [1.5, 2.8); L2_OH = O with 1 H in [2.8, 4.0); residual = total - the four. H owner = nearest O within 1.50 A.
  Every channel's raw numerator is divided by the common full-system O-H denominator.
Kernel: waterint layered ssVVCF (``sfg.cpp``), layers z04 / s / l1 / l2 / all, species channels OH- and H2O, additive
normalisation, trajectory velocities (A/ps), symmetrised, 250 lags at 4 fs.
Spectrum: append 2000 zeros, DCT-I, no taper; freq_k = k / [2 (N - 1) dt c].
Free-OH scale: frequency axis x 3700 / nu_free, nu_free = extremum of |S| in
3750-4100 cm^-1 of all O-H in [z_G - 3, z_G + 10) A, z_G = Gibbs dividing surface of the film's top (half the O plateau
density between z_rel 8 and 14 A).
The array path (positions/velocities of O and H only, no per-frame objects) equals the frame path
(``compute_layered_ssvvcf_from_frames``) to 4e-17 with identical counts.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.fft import dct

from waterint.backends.cpu.native import sfg_layered_ssvvcf
from waterint.analysis.spectra.sfg_core import layered_channel_specs

DT_PS, N_LAG, N_ZERO = 0.004, 250, 2000
C_CM_PER_PS = 0.0299792458


def species_sfg_config(dt_ps: float = DT_PS, n_lag: int = N_LAG, reference=()) -> dict:
    return {"mode": "trajectory", "backend": "cpp", "velocity_source": "trajectory", "trajectory_velocity_unit": "A/ps",
            "hydrogen_symbol": "H", "oxygen_symbol": "O", "oh_assignment": "nearest_oxygen", "oh_cutoff": 1.5,
            "dt_ps": dt_ps, "lag_ps": dt_ps * (n_lag - 1), "pbc": [True, True, False],
            "reference": {"type": "top_layer_mean", "species": list(reference), "surface": "max", "layer_width": 0.7},
            "layer_bins": [{"label": "z04", "window": {"mode": 2, "z1": 0.0, "z2": 4.0, "ramp": 0.0}},
                           {"label": "s", "window": {"mode": 2, "z1": 0.0, "z2": 1.5, "ramp": 0.0}},
                           {"label": "l1", "window": {"mode": 2, "z1": 1.5, "z2": 2.8, "ramp": 0.0}},
                           {"label": "l2", "window": {"mode": 2, "z1": 2.8, "z2": 4.0, "ramp": 0.0}},
                           {"label": "all"}],
            "species_channels": ["OH-", "H2O"], "species_normalization": "additive", "mu_mode": "full", "symmetrize": True}


SPECIES_CHANNELS = {"total_z0_4": "z04:all", "Os_H_z0_4": "s:nh1", "L1_OH_z0_4": "l1:nh1", "L1_H2O_z0_4": "l1:H2O",
                    "L2_OH_z0_4": "l2:nh1"}


def zref_top_layer_mean(z_mg: np.ndarray, width: float = 0.7) -> np.ndarray:
    """waterint _surface_layer_mean per frame (np.mean of the Mg within `width` of the highest Mg)."""
    zmax = z_mg.max(axis=1)
    return np.array([float(np.mean(row[row >= top - width])) for row, top in zip(z_mg, zmax)])


@dataclass
class SpeciesBlock:
    keys: list[str]
    numerators: np.ndarray      # raw numerators per kernel channel (corr * count)
    counts: np.ndarray          # additive denominators (the layer's all-O-H counts)
    zref: np.ndarray
    frames: int
    stage_seconds: list[float]


def layered_block(P: np.ndarray, V: np.ndarray, o_idx: np.ndarray, h_idx: np.ndarray, zref: np.ndarray,
                  cell: tuple[float, float, float], cfg: dict, n_lag: int) -> SpeciesBlock:
    """One contiguous block (O/H positions wrapped in x, y; velocities A/ps) through the native layered kernel."""
    specs = layered_channel_specs(cfg)
    r = sfg_layered_ssvvcf(P, V, o_idx, h_idx, zref, dt_ps=float(cfg["dt_ps"]), max_lag=n_lag - 1,
                           oh_cutoff=float(cfg["oh_cutoff"]), oh_assignment="nearest_oxygen", cell=cell, pbc=(True, True, False),
                           mu_mode="full", symmetrize=True, flip_sign=False, duplicate_policy="nearest", channel_specs=specs)
    if r is None:
        raise RuntimeError("native layered SFG kernel unavailable")
    keys = list(specs)
    sums = np.asarray(r[0], dtype=float)
    sel = np.asarray(r[1])
    cnt = np.array([sel[keys.index(k.split(":")[0] + ":all")] for k in keys])
    return SpeciesBlock(keys, sums, cnt, zref, P.shape[0], [float(x) for x in r[2]])


def spectra(cf: np.ndarray, dt_ps: float = DT_PS, n_zero: int = N_ZERO) -> tuple[np.ndarray, np.ndarray]:
    S = dct(np.concatenate([cf, np.zeros(cf.shape[:-1] + (n_zero,))], axis=-1), type=1, axis=-1)
    freq = np.arange(S.shape[-1]) / (2 * (S.shape[-1] - 1) * dt_ps * C_CM_PER_PS)
    return S, freq


def species_channels(keys: list[str], nums: np.ndarray, counts: np.ndarray) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    ch = dict(zip(keys, range(len(keys))))
    den_full = counts[ch["all:all"]].astype(float)
    den_z04 = counts[ch["z04:all"]].astype(float)
    cf = {lib: nums[ch[w]] / den_full for lib, w in SPECIES_CHANNELS.items()}
    cf["residual_z0_4"] = cf["total_z0_4"] - sum(cf[k] for k in SPECIES_CHANNELS if k != "total_z0_4")
    names = list(cf)
    return names, np.array([cf[k] for k in names]), den_full, den_z04


def gibbs_top_surface(z_rel_O_frames: list[np.ndarray]) -> tuple[float, np.ndarray, np.ndarray, float]:
    """z_G of the film's top from O z_rel samples of several frames (every max(1, n // 200)-th frame):
    first 0.25 A bin centre above 14 A with O density below half the plateau (mean density of 8-14 A)."""
    z = np.concatenate(z_rel_O_frames)
    h, e = np.histogram(z, bins=np.arange(0, 40.01, 0.25))
    c = 0.5 * (e[1:] + e[:-1])
    rho = h / len(z_rel_O_frames)
    plateau = rho[(c >= 8) & (c < 14)].mean()
    above = np.where((c > 14) & (rho < 0.5 * plateau))[0]
    if above.size == 0:
        raise ValueError("no water/vacuum surface found above 14 A")
    return float(c[above[0]]), c, rho, float(plateau)


def free_oh_peak(cf: np.ndarray, dt_ps: float = DT_PS) -> tuple[int, float, float]:
    S, f = spectra(cf, dt_ps)
    w = (f >= 3750) & (f <= 4100)
    k = int(np.where(w)[0][np.argmax(np.abs(S[w]))])
    return k, float(f[k]), 3700.0 / float(f[k])
