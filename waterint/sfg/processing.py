from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.fftpack import dct


AU2INVCM = 219474.62909500976
FS2AU = 41.34137300000000


def load_cf(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.loadtxt(path, comments="#")
    if data.ndim == 1:
        data = data[None, :]
    if data.shape[1] < 2:
        raise ValueError(f"{path} must contain at least time and correlation columns.")
    time_ps = data[:, 0].astype(float)
    corr = data[:, 1].astype(float)
    if data.shape[1] >= 3:
        counts = data[:, 2].astype(np.int64)
    else:
        counts = np.ones_like(time_ps, dtype=np.int64)
    if time_ps.size < 2:
        raise ValueError(f"{path} must contain at least two rows.")
    return time_ps, corr, counts


def write_cf(path: str | Path, time_ps: np.ndarray, corr: np.ndarray, counts: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# time_ps  C_avg  count\n")
        for time_value, corr_value, count in zip(time_ps, corr, counts):
            handle.write(f"{time_value:12.8f} {corr_value: .12e} {int(count)}\n")


def combine_cf(paths: list[str | Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not paths:
        raise ValueError("No CF paths were provided.")
    ref_time: np.ndarray | None = None
    weighted_sum: np.ndarray | None = None
    count_sum: np.ndarray | None = None

    for path in paths:
        time_ps, corr, counts = load_cf(path)
        if ref_time is None:
            ref_time = time_ps
            weighted_sum = np.zeros_like(corr, dtype=float)
            count_sum = np.zeros_like(counts, dtype=np.int64)
        else:
            if time_ps.shape != ref_time.shape or np.max(np.abs(time_ps - ref_time)) > 1e-9:
                raise ValueError(f"CF time grid mismatch: {path}")
        weighted_sum += corr * counts
        count_sum += counts

    assert ref_time is not None
    assert weighted_sum is not None
    assert count_sum is not None
    corr_avg = np.zeros_like(weighted_sum)
    mask = count_sum > 0
    corr_avg[mask] = weighted_sum[mask] / count_sum[mask]
    return ref_time, corr_avg, count_sum


def compute_ft(
    time: np.ndarray,
    corr: np.ndarray,
    *,
    time_unit: str = "ps",
    nzeros: int = 2000,
) -> tuple[np.ndarray, np.ndarray]:
    if time.size < 2:
        raise ValueError("Need at least two time points to compute FT.")
    dt = float(time[1] - time[0])
    if dt <= 0:
        raise ValueError("Time grid must be increasing.")
    if time_unit == "ps":
        dt_fs = dt * 1000.0
    elif time_unit == "fs":
        dt_fs = dt
    else:
        raise ValueError("time_unit must be ps or fs.")
    step_au = dt_fs * FS2AU

    signal_in = np.asarray(corr, dtype=float)
    if nzeros > 0:
        signal_in = np.concatenate((signal_in, np.zeros(int(nzeros), dtype=float)))
    signal = np.real(dct(signal_in, type=1))
    freq = np.linspace(0.0, 1.0 / (2.0 * step_au), signal.size)
    freq *= AU2INVCM * 2.0 * np.pi
    return freq, signal


def write_ft(path: str | Path, freq_cm: np.ndarray, signal: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.column_stack((freq_cm, signal)), fmt="%.18e")


def load_ft(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, comments="#")
    if data.ndim == 1:
        data = data[None, :]
    if data.shape[1] < 2:
        raise ValueError(f"{path} must contain at least frequency and signal columns.")
    return data[:, 0], data[:, 1]
