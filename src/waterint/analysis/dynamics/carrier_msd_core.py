"""Strict OH-defect (carrier) site-path xy MSD.

Definition: per carrier track (``carrier_track``) the site path is
    site(birth) = position of the carrier O,  site(t) = site(t-1) + mic_xy(pos[O(t)](t) - pos[O(t-1)](t-1))
on every saved frame, and
    MSD_xy(k) = sum_tracks sum_t |r(t+k) - r(t)|^2 / sum_tracks max(L - k, 0)
with all valid same-track origins (tracks are never joined across a birth, death or gap), dx^2 + dy^2 (no 1/2), no
drift subtraction. Identity support S(k) = valid pairs / at-risk origins, at-risk = origins t with t <= T_last - k.
It is a survival-conditioned defect self-MSD, not an unconditional diffusion coefficient and not a conductivity.
Origin classes by the carrier z_rel at the origin (named unions of profile windows,
weights w(t) in {0, 1}: sum_t w(t) |r(t+k) - r(t)|^2 via FFT cross-correlations), and lags beyond 200 ps.
The masked estimator equals brute-force sums to 1e-12.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from waterint.analysis.dynamics.carrier_track_core import schedule_starts


def zref_top_layer(z_ref_atoms: np.ndarray, width: float) -> np.ndarray:
    """Top-layer z_ref per frame: np.mean(z[z >= max(z) - width]) over the reference atoms; z: (frames, atoms)."""
    zmax = z_ref_atoms.max(axis=1, keepdims=True)
    mask = z_ref_atoms >= zmax - width
    return np.array([np.mean(row[m]) for row, m in zip(z_ref_atoms, mask)])


class CarrierSitePaths:
    """Rebuild the per-frame carrier site of every track from position blocks pushed in time order.

    Rows of track i live at offset[i] .. offset[i + 1] - 1 (one per frame, birth .. death). With ``storage`` the
    arrays are .npy memory maps in that directory (a 50 ns run has ~2.5e8 carrier-frames)."""

    def __init__(self, tracks: np.ndarray, changes: np.ndarray, first_frame: int, last_frame: int,
                 storage: Path | None = None) -> None:
        self.birth = tracks["birth_frame"].astype(np.int64)
        self.death = tracks["death_frame"].astype(np.int64)
        self.n = len(tracks)
        self.offset = np.r_[0, np.cumsum(self.death - self.birth + 1)].astype(np.int64)
        self.changes = changes
        self.starts = schedule_starts(changes, self.n)
        self.first_frame, self.last_frame = int(first_frame), int(last_frame)
        rows = int(self.offset[-1])
        if storage is None:
            self.xy = np.zeros((rows, 2))
            self.zrel = np.zeros(rows, dtype=np.float32)
        else:
            storage = Path(storage)
            storage.mkdir(parents=True, exist_ok=True)
            self.xy = np.lib.format.open_memmap(storage / "site_xy.npy", "w+", np.float64, (rows, 2))
            self.zrel = np.lib.format.open_memmap(storage / "z_rel.npy", "w+", np.float32, (rows,))
        self._last = np.zeros((self.n, 2))
        self._unw = np.zeros((self.n, 2))
        self.done = np.zeros(self.n, dtype=np.int64)
        self.frames_seen = 0
        self.nh_checks = 0
        self.nh_not_one = 0

    def push(self, g0: int, pos: np.ndarray, lx: float, ly: float, zref: np.ndarray,
             o_index: np.ndarray | None = None, h_index: np.ndarray | None = None, cutoff: float = 1.5) -> None:
        """pos: (frames, atoms, 3) for global frames g0 .. g0 + len - 1; atom id = index + 1 (waterint convention).
        With o_index / h_index the first frame of the block is checked: every tracked carrier O owns exactly one H."""
        c0, c1 = g0, g0 + pos.shape[0]
        self.frames_seen = max(self.frames_seen, c1)
        ch = self.changes
        for i in np.flatnonzero((self.birth < c1) & (self.death >= c0)):
            a, b = max(self.birth[i], c0), min(self.death[i], c1 - 1)
            f = np.arange(a, b + 1)
            lo, hi = self.starts[i], self.starts[i + 1]
            O = ch["O"][lo:hi][np.searchsorted(ch["frame"][lo:hi], f, side="right") - 1]
            p = pos[f - c0, O - 1]
            dd = p[:, :2] - np.vstack([self._last[i][None], p[:-1, :2]])
            dd[:, 0] -= lx * np.rint(dd[:, 0] / lx)
            dd[:, 1] -= ly * np.rint(dd[:, 1] / ly)
            if a == self.birth[i]:
                dd[0] = p[0, :2]
                seq = np.cumsum(dd, axis=0)                     # sequential sums (site += delta)
            else:
                seq = np.cumsum(np.vstack([self._unw[i][None], dd]), axis=0)[1:]
            r0 = self.offset[i] + a - self.birth[i]
            self.xy[r0:r0 + len(f)] = seq
            self.zrel[r0:r0 + len(f)] = (p[:, 2] - zref[f - c0]).astype(np.float32)
            self._unw[i], self._last[i] = seq[-1], p[-1, :2]
            self.done[i] += len(f)
        if o_index is not None and h_index is not None:
            fc = max(c0, self.first_frame)
            if fc <= min(c1 - 1, self.last_frame):
                X = pos[fc - c0]
                dv = X[h_index][:, None, :] - X[o_index][None, :, :]
                dv[..., 0] -= lx * np.rint(dv[..., 0] / lx)
                dv[..., 1] -= ly * np.rint(dv[..., 1] / ly)
                d2 = np.einsum("ijk,ijk->ij", dv, dv)
                j = d2.argmin(axis=1)
                own = o_index[j[d2[np.arange(len(h_index)), j] <= cutoff * cutoff]]
                nown = np.bincount(own, minlength=X.shape[0])
                for i in np.flatnonzero((self.birth <= fc) & (self.death >= fc)):
                    lo, hi = self.starts[i], self.starts[i + 1]
                    o = ch["O"][lo:hi][np.searchsorted(ch["frame"][lo:hi], fc, side="right") - 1] - 1
                    self.nh_checks += 1
                    self.nh_not_one += int(nown[o] != 1)

    def complete(self) -> np.ndarray:
        return self.done == self.offset[1:] - self.offset[:-1]


def default_lag_grid(full_lag: int, max_lag: int) -> np.ndarray:
    """Every frame up to full_lag, then every 250 frames to 250000, every 2500 to 2.5e6, every 25000 beyond."""
    g = set(range(full_lag + 1)) | set(range(250, 250001, 250)) | set(range(2500, 2500001, 2500)) | set(range(25000, 25000001, 25000))
    return np.array(sorted(x for x in g if x <= max_lag), dtype=np.int64)


@dataclass
class CarrierMsdResult:
    classes: list[str]
    lag_frames: np.ndarray
    numerator_A2: np.ndarray          # (classes, lags)
    valid_pair_count: np.ndarray      # (classes, lags)
    at_risk_origin_count: np.ndarray  # (classes, lags)
    carrier_frames: np.ndarray        # (classes,)
    frames: int
    tracks: int
    extra: dict = field(default_factory=dict)

    @property
    def msd_A2(self) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore"):
            return self.numerator_A2 / self.valid_pair_count

    @property
    def identity_support(self) -> np.ndarray:
        with np.errstate(invalid="ignore", divide="ignore"):
            return self.valid_pair_count / self.at_risk_origin_count


def strict_carrier_msd(birth: np.ndarray, death: np.ndarray, offset: np.ndarray, xy: np.ndarray, zrel: np.ndarray,
                       first_frame: int, last_frame: int, class_edges: dict[str, list[tuple[float, float]]],
                       full_lag: int = 50000, max_lag: int = 2500000) -> CarrierMsdResult:
    """Estimator over all origins (class 'all') plus origin classes; class_edges: name -> list of half-open z_rel ranges."""
    tlast = int(last_frame) - int(first_frame)
    kmax = min(int(max_lag), tlast)
    names = ["all"] + list(class_edges)
    nc = len(names)
    num = np.zeros((nc, kmax + 1))
    cnt = np.zeros((nc, kmax + 1), dtype=np.int64)
    hist = np.zeros((nc, tlast + 1), dtype=np.int64)
    for i in range(len(birth)):
        a = np.asarray(xy[offset[i]:offset[i + 1]], dtype=np.float64)
        z = np.asarray(zrel[offset[i]:offset[i + 1]])
        n = len(a)
        b0 = int(birth[i]) - int(first_frame)
        m = min(kmax, n - 1) + 1
        lag = np.arange(m)
        ws = [None]
        for rng in class_edges.values():
            w = np.zeros(n, dtype=bool)
            for lo, hi in rng:
                w |= (z >= lo) & (z < hi)
            ws.append(w)
        nfft = 1 << (2 * n - 2).bit_length() if n > 1 else 1
        fx_cache = {}
        for c, w in enumerate(ws):
            wv = np.ones(n, dtype=bool) if w is None else w
            if not wv.any():
                continue
            hist[c, b0:b0 + n] += wv
            cw = np.r_[0, np.cumsum(wv, dtype=np.int64)]
            cnt[c, :m] += cw[n - lag]
            if n == 1:
                continue
            for d in range(2):
                x = a[:, d] - a[0, d]
                if d not in fx_cache:
                    fx_cache[d] = np.fft.rfft(x, n=nfft)
                fx = fx_cache[d]
                if w is None:                                   # FFT segment numerator
                    corr = np.fft.irfft(fx * np.conjugate(fx), n=nfft)[:m]
                    pre = np.r_[0.0, np.cumsum(x * x)]
                    t = pre[n - lag] + (pre[n] - pre[lag]) - 2.0 * corr
                else:                                           # sum_t w(t) [x(t+k)^2 + x(t)^2 - 2 x(t) x(t+k)]
                    wf = wv.astype(np.float64)
                    t1 = np.fft.irfft(np.conjugate(np.fft.rfft(wf, n=nfft)) * np.fft.rfft(x * x, n=nfft), n=nfft)[:m]
                    t3 = np.fft.irfft(np.conjugate(np.fft.rfft(wf * x, n=nfft)) * fx, n=nfft)[:m]
                    pre = np.r_[0.0, np.cumsum(wf * x * x)]
                    t = t1 + pre[n - lag] - 2.0 * t3
                t[0] = 0.0
                t[cw[n - lag] == 0] = 0.0                       # no valid origin at this lag: exactly zero, not FFT noise
                num[c, :m] += t
    lags = default_lag_grid(int(full_lag), kmax)
    ch = np.cumsum(hist, axis=1)
    risk = ch[:, tlast - lags]
    return CarrierMsdResult(names, lags, num[:, lags], cnt[:, lags], risk, hist.sum(axis=1), tlast + 1, len(birth))
