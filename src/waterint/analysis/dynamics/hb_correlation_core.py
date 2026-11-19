"""One-pass H-bond / proton-transfer analysis: Python binding of the streaming C++ kernel ``hb_correlation.cpp``.

Definitions:
  * C_PT / C_HB / proton-sharing FES (seeded geometric definitions). H owner = nearest O within
    1.50 A (ties 1e-8 A -> lowest ID); eligible edge = O-O <= 3.5 A and cos(D-H...A) <= -0.5 (broad), strict = 2.2 <= R_OO
    < 3.2 A and angle >= 150 deg; primary acceptor = nearest eligible A by H...A; seeds every 50 frames, lags to 25000 frames;
    channels A L1 OH- -> L1 H2O, B L1 H2O -> L2 OH-, C L1 H2O -> L1 OH-, D L2 H2O -> L1 OH-; layers relative to the mean z of
    the 72 highest Mg (z_ref).
  * density, O-H orientation, per-frame species counts (z_ref2 = Mg within 0.70 A of the highest Mg).
  * xy MSD of fixed atoms (mobile O, H) with classes by the state at the origin
    (dx^2 + dy^2, every strided frame an origin, no drift subtraction), sparse lags.
Lattice O: O below z_ref(first frame) + 1.0 A (frozen at the first frame). All results per block of ``block`` frames.
The CPU kernel and the GPU kernel give identical counts on the same frames.
"""
from __future__ import annotations

import ctypes
import json
from pathlib import Path

import numpy as np

from waterint.backends.cpu import native as _native

_P = ctypes.c_void_p
_BOUND = False
KM = np.array([15.999, 17.00734, 18.01528, 19.023, 15.999])          # O, OH, H2O, H3O, >=4H (molar masses of the O-centred species)
CHANNELS = ["A L1OH->L1H2O", "B L1H2O->L2OH", "C L1H2O->L1OH", "D L2H2O->L1OH"]
MSD_GROUPS = {"O_mobile_all": list(range(0, 7)), "H_all": list(range(7, 14)), "O_L1": [1], "O_L2": [2], "O_bulk": [3],
              "H_owner_L1": [8], "H_owner_L2": [9], "H_owner_bulk": [10], "H_OsH": [7]}


def _lib():
    global _BOUND
    lib = _native.native_library()
    if lib is None:
        raise RuntimeError(f"WaterInt native library unavailable: {getattr(_native, '_LAST_ERROR', None)}")
    if not _BOUND:
        dp = ctypes.POINTER(ctypes.c_double)
        lib.waterint_hbc_error.restype = ctypes.c_char_p
        lib.waterint_hbc_create.restype = _P
        lib.waterint_hbc_create.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int16),
                                            ctypes.POINTER(ctypes.c_uint8), ctypes.c_int64, ctypes.c_double, ctypes.c_double,
                                            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        lib.waterint_hbc_push.restype = ctypes.c_int
        lib.waterint_hbc_push.argtypes = [_P, ctypes.c_int, dp, dp, dp]
        lib.waterint_hbc_finish.restype = ctypes.c_int
        lib.waterint_hbc_finish.argtypes = [_P, ctypes.c_char_p, ctypes.c_double]
        lib.waterint_hbc_threads.restype = ctypes.c_int
        lib.waterint_hbc_threads.argtypes = [_P]
        lib.waterint_hbc_free.argtypes = [_P]
        _BOUND = True
    return lib


def _dp(a):
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


class HBCorrelationGeometry:
    """Geometry of one run, fixed at its first frame.

    ``reference``: element(s) of the reference layer (profile reference elements); z_ref = mean z of the
    ``top_n`` highest of them (first-frame selection); z_ref2 = those within ``layer_width_A`` of the highest.
    ``acceptor`` / ``proton``: the O-like and H-like elements (profile ownership).
    """

    def __init__(self, species, first_positions: np.ndarray, top_n: int = 72, lattice_offset_A: float = 1.0,
                 layer_width_A: float = 0.70, *, reference, acceptor: str = "O", proton: str = "H") -> None:
        sp = np.asarray(species)
        self.oh = np.flatnonzero((sp == acceptor) | (sp == proton))
        self.mg = np.flatnonzero(np.isin(sp, list(reference)))
        if self.mg.size < top_n:
            raise ValueError(f"need at least {top_n} reference atoms ({list(reference)}) for z_ref")
        self.top = self.mg[np.argsort(first_positions[self.mg, 2])[-top_n:]]
        zr0 = first_positions[self.top, 2].mean()
        self.ids = (self.oh + 1).astype(np.int32)
        self.types = np.where(sp[self.oh] == proton, 1, 3).astype(np.int16)
        self.lattice = ((sp[self.oh] == acceptor) & (first_positions[self.oh, 2] < zr0 + lattice_offset_A)).astype(np.uint8)
        self.width = float(layer_width_A)

    def zrefs(self, pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        zm = pos[:, self.mg, 2]
        msk = zm >= zm.max(axis=1, keepdims=True) - self.width
        zr2 = np.cumsum(np.where(msk, zm, 0.0), axis=1)[:, -1] / msk.sum(axis=1)     # sequential sum (fixed summation order)
        return pos[:, self.top, 2].mean(axis=1), zr2


class HBCorrelationKernel:
    def __init__(self, geom: HBCorrelationGeometry, n_frames: int, lx: float, ly: float, *, window: int = 100000,
                 block: int = 250000, msd_stride: int = 10, msd_maxlag: int = 250000) -> None:
        self._lib = _lib()
        self.geom = geom
        self._keep = [np.ascontiguousarray(geom.ids), np.ascontiguousarray(geom.types), np.ascontiguousarray(geom.lattice)]
        self._h = self._lib.waterint_hbc_create(len(geom.ids), self._keep[0].ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                                                self._keep[1].ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
                                                self._keep[2].ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), int(n_frames),
                                                float(lx), float(ly), int(window), int(block), int(msd_stride), int(msd_maxlag))
        if not self._h:
            raise ValueError(self._lib.waterint_hbc_error().decode())
        self.threads = int(self._lib.waterint_hbc_threads(self._h))

    def push(self, pos: np.ndarray) -> None:
        """pos: (frames, all atoms, 3) in time order."""
        zr, zr2 = self.geom.zrefs(pos)
        x = np.ascontiguousarray(pos[:, self.geom.oh, :], dtype=np.float64)
        zr, zr2 = np.ascontiguousarray(zr), np.ascontiguousarray(zr2)
        if self._lib.waterint_hbc_push(self._h, x.shape[0], _dp(zr), _dp(zr2), _dp(x)) != 0:
            raise RuntimeError(f"hb-correlation kernel: {self._lib.waterint_hbc_error().decode()}")

    def finish(self, out_dir: Path, read_seconds: float = 0.0) -> None:
        if self._lib.waterint_hbc_finish(self._h, str(out_dir).encode(), float(read_seconds)) != 0:
            raise RuntimeError(f"hb-correlation kernel: {self._lib.waterint_hbc_error().decode()}")

    def close(self) -> None:
        if getattr(self, "_h", None):
            self._lib.waterint_hbc_free(self._h)
            self._h = None

    def __del__(self) -> None:  # pragma: no cover
        try:
            self.close()
        except Exception:
            pass


def summarize(raw: Path, dt_ps: float, area_A2: float) -> dict[str, np.ndarray]:
    """Raw kernel counts -> physical quantities."""
    D = Path(raw)
    S = json.loads((D / "kernel_summary.json").read_text())
    nb, nf, blk = S["blocks"], S["frames"], S["block_frames"]
    bf = np.fromfile(D / "block_frames.u64", dtype="<u8").astype(float)
    mass = np.concatenate([KM] * 5)
    dens = np.fromfile(D / "density.u32", dtype="<u4").reshape(nb, 25, 1240).astype(float)
    to_gcm3 = mass * 1.66053907 / (area_A2 * 0.025)
    out: dict[str, np.ndarray] = {
        "z_density_A": -1 + (np.arange(1240) + 0.5) * 0.025,
        "density_g_cm3": dens.sum(0) / nf * to_gcm3[:, None],
        "density_blocks_g_cm3": dens / bf[:, None, None] * to_gcm3[None, :, None],
        "population_per_frame": np.fromfile(D / "population.u64", dtype="<u8").reshape(nb, 25).astype(float) / bf[:, None],
        "orientation_bonds_per_frame": np.fromfile(D / "orientation.u32", dtype="<u4").reshape(nb, 5, 360, 360).astype(float).sum(0) / nf,
        "z_orientation_A": -1 + (np.arange(360) + 0.5) * 0.025, "angle_deg": (np.arange(360) + 0.5) * 0.5,
    }
    spec = np.fromfile(D / "species.i16", dtype="<i2").reshape(nf, 4)
    out["species_per_frame_mean"] = spec.mean(0)
    out["species_block_mean"] = np.array([spec[i * blk:(i + 1) * blk].mean(0) for i in range(nb)])
    lags = np.fromfile(D / "lag_frames.bin", dtype="<i4")
    nl = len(lags)
    numb = np.fromfile(D / "numerators_blocks.bin", dtype="<u8").reshape(nb, 2, 4, 2, nl).astype(float)
    num = numb.sum(0)
    den = num[:, :, 0, 0]
    with np.errstate(invalid="ignore", divide="ignore"):
        out["C"] = num / den[:, :, None, None]
        out["C_blocks"] = numb / numb[:, :, :, 0:1, 0:1]
    out.update(lag_ps=lags * dt_ps, seed_denominators=den, seed_denominators_blocks=numb[:, :, :, 0, 0])
    fes_c = np.fromfile(D / "fes_counts_blocks.bin", dtype="<u4").reshape(nb, 2, 2, 280, 350).astype(float).sum(0)
    kT = 8.617333262e-5 * 300 * 1000
    with np.errstate(divide="ignore", invalid="ignore"):
        fes = -kT * np.log(fes_c / fes_c.sum(axis=(2, 3), keepdims=True))
        fes -= np.nanmin(np.where(np.isfinite(fes), fes, np.nan), axis=(2, 3), keepdims=True)
    out.update(fes_meV=fes, fes_counts=fes_c, delta_A=-3.5 + (np.arange(280) + 0.5) * 0.025, R_OO_A=(np.arange(350) + 0.5) * 0.01)
    hbd = np.fromfile(D / "hbond_donated.u64", dtype="<u8").reshape(nb, 2, 25, 4).astype(float).sum(0)
    hba = np.fromfile(D / "hbond_accepted.u64", dtype="<u8").reshape(nb, 2, 25, 8).astype(float).sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        out["hbond_donated_mean_per_H"] = (hbd * np.arange(4)).sum(-1) / hbd.sum(-1)
        out["hbond_accepted_mean_per_O"] = (hba * np.arange(8)).sum(-1) / hba.sum(-1)
    out.update(hbond_donated_hist=hbd, hbond_accepted_hist=hba)
    ml = np.fromfile(D / "msd_lag_frames.i32", dtype="<i4")
    nml = len(ml)
    ms = S.get("msd_stride_frames", 10)
    mn = np.fromfile(D / "msd_numerators.f64", dtype="<f8").reshape(nb, 14, nml, 2)
    mc = np.fromfile(D / "msd_counts.u64", dtype="<u8").reshape(nb, 14, nml).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        out.update(msd_lag_ps=ml * ms * dt_ps, msd_groups=np.array(list(MSD_GROUPS)),
                   msd_A2=np.array([mn[:, g].sum(axis=(0, 1, 3)) / mc[:, g].sum(axis=(0, 1)) for g in MSD_GROUPS.values()]),
                   msd_pairs=np.array([mc[:, g].sum(axis=(0, 1)) for g in MSD_GROUPS.values()]),
                   msd_blocks_A2=np.array([mn[:, g].sum(axis=(1, 3)) / mc[:, g].sum(axis=1) for g in MSD_GROUPS.values()]))
    out.update(frames=np.array(nf), block_frames=np.array(blk), area_A2=np.array(area_A2), dt_ps=np.array(dt_ps))
    return out
