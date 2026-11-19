"""Sheet conductance from charge-moment increments (STACIE Green-Kubo and collective Helfand).

STACIE estimator settings: joint x/y sequences, each centred by its
time mean, ExpPolyModel([0, 1, 2]), CV2LCriterion, zero frequency excluded, rng seed 42,
neff 15-1000, cutoff spacing 0.5, switch exponent 8, nonlinear budget 100, criterion_high 100.
G_sheet = (1 / (A kB T)) * integral <J(t) J(0)> dt  (per in-plane component, S).

The collective Helfand estimator gives the same G from the slope of the charge-moment MSD,
G = slope / (4 A kB T) for the xy sum; with several current components it gives the full
matrix G_ab (cross terms included), which sums exactly to the total.
"""
from __future__ import annotations

import importlib.metadata
from pathlib import Path

import numpy as np
from scipy.fft import irfft, next_fast_len, rfft

K_B = 1.380649e-23
E_ANGSTROM_TO_C_M = 1.602176634e-29
E_ANGSTROM_PER_PS_TO_A_M = 1.602176634e-17      # J_SI = dM / dt_ps * this
FIT_KW = dict(neff_min=15, neff_max=1000, fcut_min=None, fcut_max=None, fcut_spacing=0.5, switch_exponent=8.0,
              nonlinear_budget=100, criterion_high=100)
RNG_SEED = 42
MODEL = "ExpPolyModel([0,1,2])"


def stacie_version() -> dict:
    import stacie
    return {"version": importlib.metadata.version("stacie"), "module_path": str(Path(stacie.__file__).resolve().parent)}


def sheet_prefactor(area_A2: float, temperature_K: float) -> float:
    return 1.0 / (area_A2 * 1e-20 * K_B * temperature_K)


def current_sequences(dM_eA: np.ndarray, dt_ps: float) -> tuple[np.ndarray, np.ndarray]:
    """(n, >=2) increments -> centred (2, n) current sequences in A*m, plus their removed means."""
    j = np.asarray(dM_eA[:, :2], dtype=np.float64) / dt_ps * E_ANGSTROM_PER_PS_TO_A_M
    seq = j.T.copy()
    means = seq.mean(axis=1)
    seq -= means[:, None]
    return seq, means


def fit_stacie(sequences, prefactors, timestep_s: float, *, fcut_max_Hz: float | None = None, label: str = "fit",
               max_fit_fraction_of_nyquist: float = 0.1) -> tuple[dict, dict]:
    from stacie import CV2LCriterion, ExpPolyModel, compute_spectrum, estimate_acint
    kwargs = dict(FIT_KW)
    kwargs["fcut_max"] = fcut_max_Hz
    result = compute_spectrum(sequences, prefactors=prefactors, timestep=timestep_s, include_zero_freq=False)
    est = estimate_acint(result, ExpPolyModel([0, 1, 2]), cutoff_criterion=CV2LCriterion(),
                         rng=np.random.default_rng(RNG_SEED), verbose=False, **kwargs)
    hist = est.history
    if not hist:
        raise RuntimeError(f"Empty STACIE cutoff history for {label}")
    maxw = max(hist, key=lambda h: float(h["fcut_weight"]))
    flags = []
    if maxw is hist[0] or maxw is hist[-1]:
        flags.append("maximum_cutoff_weight_at_search_boundary")
    if float(est.props["cost_zscore"]) >= 2:
        flags.append("fit_cost_zscore_ge_2")
    if float(est.props["criterion_zscore"]) >= 2:
        flags.append("cutoff_criterion_zscore_ge_2")
    nyquist = float(result.freqs[-1])
    highest = float(result.freqs[est.ncut - 1])
    if not highest < max_fit_fraction_of_nyquist * nyquist:
        raise RuntimeError(f"{label}: fitted band reaches >= {max_fit_fraction_of_nyquist:.0%} of Nyquist ({highest=}, {nyquist=})")
    arrays = {
        "frequency_THz": result.freqs / 1e12, "spectrum_amplitude_S": result.amplitudes, "spectrum_ndofs": result.ndofs,
        "fit_frequency_THz": result.freqs[: est.ncut] / 1e12, "fit_amplitude_S": np.asarray(est.props["amplitudes_model"][0], float),
        "cutoff_frequency_THz": np.array([h["fcut"] for h in hist]) / 1e12, "cutoff_weight": np.array([h["fcut_weight"] for h in hist]),
        "cutoff_acint_S": np.array([h["acint"] for h in hist]), "cutoff_acint_std_S": np.array([h["acint_std"] for h in hist]),
    }
    record = {
        "label": label, "G_S": float(est.acint), "G_nS": float(est.acint) * 1e9, "fit_std_S": float(est.acint_std),
        "fit_std_is_within_fit_only": True, "mean_fcut_THz": float(est.fcut / 1e12),
        "max_weight_fcut_THz": float(maxw["fcut"]) / 1e12, "cutoff_scan_count": len(hist), "ncut": int(est.ncut),
        "neff": float(est.neff), "cost_zscore": float(est.props["cost_zscore"]),
        "criterion_zscore": float(est.props["criterion_zscore"]), "fit_flags": flags, "model": MODEL,
        "kwargs": {**FIT_KW, "fcut_max": fcut_max_Hz, "cutoff_criterion": "CV2LCriterion()", "rng_seed": RNG_SEED},
        "timestep_s": timestep_s, "include_zero_freq": False, "nyquist_THz": nyquist / 1e12,
        "highest_fit_fraction_of_nyquist": highest / nyquist,
    }
    return record, arrays


def _cross_msd(a: np.ndarray, b: np.ndarray, max_lag: int) -> np.ndarray:
    """<(A(t+k)-A(t)) . (B(t+k)-B(t))> over all origins, for (n+1, d) moments A, B (FFT, O(n log n))."""
    n = a.shape[0]
    nfft = next_fast_len(2 * n)
    corr = irfft(np.conj(rfft(a, nfft, axis=0)) * rfft(b, nfft, axis=0), nfft, axis=0)[: max_lag + 1].sum(axis=1)
    corr_ba = irfft(np.conj(rfft(b, nfft, axis=0)) * rfft(a, nfft, axis=0), nfft, axis=0)[: max_lag + 1].sum(axis=1)
    lag = np.arange(max_lag + 1)
    prod = np.sum(a * b, axis=1)
    cs = np.r_[0.0, np.cumsum(prod)]
    # sum_t [A(t+k)B(t+k) + A(t)B(t)] - sum_t [A(t)B(t+k) + A(t+k)B(t)]
    same = (cs[n] - cs[lag]) + cs[n - lag]
    return (same - corr - corr_ba) / (n - lag)


def helfand_matrix(increments: np.ndarray, dt_ps: float, area_A2: float, temperature_K: float,
                   windows_ps=((20.0, 60.0), (20.0, 200.0), (200.0, 1000.0)), max_lag_ps: float | None = None) -> dict:
    """increments: (K, n, 2) charge-moment increments of K components (e*A). Returns G_ab per lag window (nS)
    from linear fits of the cross MSD; sum_ab G_ab equals the Helfand G of the summed current."""
    inc = np.asarray(increments, dtype=np.float64)
    if inc.ndim == 2:
        inc = inc[None]
    K, n = inc.shape[0], inc.shape[1]
    max_lag_ps = max(w[1] for w in windows_ps) if max_lag_ps is None else max_lag_ps
    max_lag = min(n - 1, int(round(max_lag_ps / dt_ps)))
    moments = [np.vstack([np.zeros(2), np.cumsum(inc[k, :, :2], axis=0)]) for k in range(K)]
    moments = [m - m.mean(axis=0) for m in moments]
    lag_ps = np.arange(max_lag + 1) * dt_ps
    msd = np.zeros((K, K, max_lag + 1))
    for a in range(K):
        for b in range(a, K):
            msd[a, b] = msd[b, a] = _cross_msd(moments[a], moments[b], max_lag)
    scale = E_ANGSTROM_TO_C_M ** 2 / 1e-12 / (4 * area_A2 * 1e-20 * K_B * temperature_K)
    out = {"lag_ps": lag_ps, "msd_e2A2": msd, "windows": {}}
    for lo, hi in windows_ps:
        mask = (lag_ps >= lo) & (lag_ps <= hi)
        if mask.sum() < 3:
            continue
        G = np.zeros((K, K))
        for a in range(K):
            for b in range(K):
                G[a, b] = np.polyfit(lag_ps[mask], msd[a, b, mask], 1)[0] * scale
        out["windows"][f"{lo:g}_{hi:g}ps"] = G * 1e9
    return out


def equal_run_stats(values_S) -> dict:
    v = np.asarray(values_S, dtype=float)
    n = int(v.size)
    sd = float(v.std(ddof=1)) if n > 1 else float("nan")
    return {"n_runs": n, "mean_nS": float(v.mean() * 1e9), "sd_nS": sd * 1e9, "sem_nS": sd / np.sqrt(n) * 1e9 if n > 1 else float("nan"),
            "values_nS": (v * 1e9).tolist(), "statistic": "equal-run mean, sample SD (ddof=1), SEM = SD/sqrt(n)"}
