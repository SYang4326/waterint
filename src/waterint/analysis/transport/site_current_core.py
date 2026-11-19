"""Complete formal-ion site current: Python binding of the streaming C++ kernel.

The kernel (``site_current.cpp``) streams frames that are
pushed in blocks so any reader (LAMMPS text, GPUMD quantized npz) can feed it.
Charges, roles, cutoff and periodicity come from the resolved system profile.
"""
from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from waterint.backends.cpu import native as _native

E_ANGSTROM_PER_PS_TO_SI = 1.602176634e-17  # e*A/ps -> C*m/s (A*m)

FRAME_DTYPE = np.dtype([
    ("frame", "<i8"), ("step", "<i8"), ("time_ps", "<f8"), ("B", "<f8", (3,)), ("assigned", "<i4"),
    ("free", "<i4"), ("ties", "<i4"), ("formal_q", "<i4"), ("max_assigned_dist", "<f8"),
], align=False)
INTERVAL_DTYPE = np.dtype([
    ("frame_start", "<i8"), ("step_start", "<i8"), ("time_mid", "<f8"), ("dMref", "<f8", (3,)), ("dB", "<f8", (3,)),
    ("dMsite", "<f8", (3,)), ("motion", "<f8", (3,)), ("transfer", "<f8", (3,)), ("dI", "<f8", (3,)),
    ("residual", "<f8", (3,)), ("assigned0", "<i4"), ("free0", "<i4"), ("ties0", "<i4"), ("assigned1", "<i4"),
    ("free1", "<i4"), ("ties1", "<i4"), ("owner_changes", "<i4"), ("crossings", "<i4", (3,)),
    ("max_mic_component", "<f8"), ("max_mic_norm", "<f8"), ("max_raw_vel", "<f8"), ("max_raw_vxy_dt_over_L", "<f8"),
], align=False)
assert FRAME_DTYPE.itemsize == 72 and INTERVAL_DTYPE.itemsize == 264
EVENTS_FORMAT = "waterint-events-1"
EVENT_DTYPE = np.dtype([
    ("frame", "<i8"), ("h_id", "<i4"), ("donor_id", "<i4"), ("acceptor_id", "<i4"), ("donor_row", "i1"), ("acceptor_row", "i1"),
    ("donor_nh0", "<i2"), ("donor_nh1", "<i2"), ("acceptor_nh0", "<i2"), ("acceptor_nh1", "<i2"),
    ("donor_zrel", "<f4"), ("acceptor_zrel", "<f4"), ("hop", "<f4", (3,)), ("d_new", "<f4"),
], align=False)
SITE_DTYPE = np.dtype([("frame", "<i8"), ("id", "<i4"), ("row", "i1"), ("nh", "i1"), ("u", "<f4", (3,))], align=False)
assert EVENT_DTYPE.itemsize == 54 and SITE_DTYPE.itemsize == 26
STATS_FIELDS = ("max_box_drift_A", "max_MIC_step_component_A", "max_MIC_step_norm_A", "max_raw_velocity_component",
                "max_raw_velocity_xy_dt_over_Lxy", "xy_image_crossings_x", "xy_image_crossings_y", "frames",
                "first_step", "last_step", "formal_charge_e", "owner_changes", "tie_H_samples", "free_H_samples",
                "nlist_frames", "full_search_frames", "nlist_rebuilds", "nlist_audits", "nlist_fallbacks")

_P = ctypes.c_void_p
_BOUND = False


def _lib():
    global _BOUND
    lib = _native.native_library()
    if lib is None:
        raise RuntimeError(f"WaterInt native library unavailable: {getattr(_native, '_LAST_ERROR', None)}")
    if not _BOUND:
        dp = ctypes.POINTER(ctypes.c_double)
        lib.waterint_sc_error.restype = ctypes.c_char_p
        lib.waterint_sc_create.restype = _P
        lib.waterint_sc_create.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int16), ctypes.POINTER(ctypes.c_int32), dp,
                                           ctypes.c_int64, ctypes.c_int64, ctypes.c_double, ctypes.c_double]
        lib.waterint_sc_create2.restype = _P
        lib.waterint_sc_create2.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int8), dp, dp,
                                            ctypes.POINTER(ctypes.c_uint8), ctypes.c_double, ctypes.c_double, ctypes.c_int64,
                                            ctypes.c_int64, ctypes.c_double, ctypes.c_double]
        lib.waterint_sc_set_decomposition.restype = ctypes.c_int
        lib.waterint_sc_set_decomposition.argtypes = [_P, ctypes.POINTER(ctypes.c_uint8), ctypes.c_int, ctypes.c_double, ctypes.c_double,
                                                      ctypes.POINTER(ctypes.c_int32), ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int32),
                                                      ctypes.POINTER(ctypes.c_int32), ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                                      ctypes.c_int, ctypes.c_int, dp]
        lib.waterint_sc_set_events.restype = ctypes.c_int
        lib.waterint_sc_set_events.argtypes = [_P, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_uint8)]
        lib.waterint_sc_pending.argtypes = [_P, ctypes.POINTER(ctypes.c_int64)]
        lib.waterint_sc_take.argtypes = [_P, _P, ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int32), _P]
        lib.waterint_sc_free.argtypes = [_P]
        lib.waterint_sc_push.restype = ctypes.c_int
        lib.waterint_sc_push.argtypes = [_P, ctypes.c_int64, ctypes.c_int, dp, dp, ctypes.POINTER(ctypes.c_int64), dp, _P, _P,
                                         ctypes.POINTER(ctypes.c_int64), dp, dp]
        lib.waterint_sc_stats.argtypes = [_P, dp]
        lib.waterint_sc_configure.restype = ctypes.c_int
        lib.waterint_sc_configure.argtypes = [_P, ctypes.c_int, ctypes.c_double, ctypes.c_int]
        lib.waterint_lmp_open.restype = _P
        lib.waterint_lmp_open.argtypes = [ctypes.c_char_p]
        lib.waterint_lmp_close.argtypes = [_P]
        lib.waterint_lmp_read.restype = ctypes.c_int
        lib.waterint_lmp_read.argtypes = [_P, ctypes.c_int64, ctypes.c_int, ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int16),
                                          dp, dp, ctypes.POINTER(ctypes.c_int64), dp, ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int)]
        _BOUND = True
    return lib


def _ptr(array: np.ndarray | None, ctype):
    return None if array is None else array.ctypes.data_as(ctypes.POINTER(ctype))


def _error(lib) -> str:
    message = lib.waterint_sc_error()
    return message.decode() if message else "unknown error"


class SiteCurrentAccumulator:
    """Streaming site/formal current; push frame blocks in time order.

    The resolved system ``profile`` (waterint.system.profile.resolve) defines charges, roles, cutoff and periodicity.
    """

    def __init__(self, *, ids: np.ndarray, box_bounds: np.ndarray, first_step: int, dstep: int, dt_ps: float,
                 types: np.ndarray | None = None, profile=None, t0_ps: float = 0.0, neighbor_list: bool = True,
                 skin_A: float = 1.0, audit_every: int = 1000) -> None:
        self._lib = _lib()
        self.ids = np.ascontiguousarray(ids, dtype=np.int32)
        self.natoms = int(self.ids.size)
        self.box = np.ascontiguousarray(np.asarray(box_bounds, dtype=np.float64).reshape(6))
        self.dt_ps = float(dt_ps)
        self.cells = 0
        self.groups: list[str] = []
        self.windows: list[str] = []
        self.last_decomposition = None
        self.last_zref = None
        if profile is None:
            raise ValueError("SiteCurrentAccumulator needs a resolved system profile (waterint.system.profile.resolve).")
        if True:
            if len(profile.species) != self.natoms:
                raise ValueError("profile and ids describe different atom counts.")
            self._keep = [np.ascontiguousarray(profile.role, dtype=np.int8), np.ascontiguousarray(profile.qf, dtype=np.float64),
                          np.ascontiguousarray(profile.pbc, dtype=np.uint8)]
            self._h = self._lib.waterint_sc_create2(self.natoms, _ptr(self.ids, ctypes.c_int32), _ptr(self._keep[0], ctypes.c_int8),
                                                    _ptr(self._keep[1], ctypes.c_double), _ptr(self.box, ctypes.c_double),
                                                    _ptr(self._keep[2], ctypes.c_uint8), float(profile.cutoff), float(profile.tie_tol),
                                                    int(first_step), int(dstep), self.dt_ps, float(t0_ps))
        if not self._h:
            raise ValueError(_error(self._lib))
        # Verlet candidate list for the H-owner search: exact (rebuilt whenever max|dU_H| + max|dU_O| > skin),
        # audited against the full grid search every `audit_every` frames (mismatch -> error), and switched off
        # automatically for a while when most frames need a rebuild (coarse sampling).
        if self._lib.waterint_sc_configure(self._h, int(bool(neighbor_list)), float(skin_A), int(audit_every)) != 0:
            raise ValueError(_error(self._lib))

    def set_decomposition(self, profile, solid: np.ndarray) -> None:
        """Enable the (group, z-window) decomposition of dM_site defined by the profile."""
        rows = profile.acceptor_rows(solid)
        arrs = [np.ascontiguousarray(profile.ref_mask, dtype=np.uint8), np.ascontiguousarray(rows, dtype=np.int32),
                np.ascontiguousarray(profile.table, dtype=np.int32), np.ascontiguousarray(profile.other_group, dtype=np.int32),
                np.ascontiguousarray(profile.edges, dtype=np.float64)]
        self._keep_dec = arrs
        status = self._lib.waterint_sc_set_decomposition(
            self._h, _ptr(arrs[0], ctypes.c_uint8), int(profile.ref_mode), float(profile.ref_width), float(profile.ref_fixed),
            _ptr(arrs[1], ctypes.c_int32), int(profile.table.shape[0]), int(profile.nh_max), _ptr(arrs[2], ctypes.c_int32),
            _ptr(arrs[3], ctypes.c_int32),
            int(profile.free_h_group), int(profile.transfer_group), int(profile.image_group), len(profile.group_names),
            len(profile.window_names), _ptr(arrs[4], ctypes.c_double) if arrs[4].size else None)
        if status != 0:
            raise ValueError(_error(self._lib))
        self.groups = list(profile.group_names)
        self.windows = list(profile.window_names) + ["outside"]
        self.cells = len(self.groups) * len(self.windows)

    def set_events(self, profile, tracked: list[str], counts_stride: int = 10, sites_stride: int = 10) -> None:
        """Record H owner changes (every interval), species counts per (group, window) and the unwrapped positions of
        tracked acceptor species (e.g. OH-, H3O+, Os_H) every `counts_stride` / `sites_stride` frames."""
        if not self.cells:
            raise ValueError("call set_decomposition first (events use the profile species rows and windows).")
        wanted = set(tracked)
        table = np.array([[profile.label_of(r, n) in wanted for n in range(profile.nh_max + 1)]
                          for r in range(len(profile.row_names))], dtype=np.uint8).reshape(-1)
        self._keep_ev = table
        if self._lib.waterint_sc_set_events(self._h, 1, int(counts_stride), int(sites_stride), _ptr(table, ctypes.c_uint8)) != 0:
            raise ValueError(_error(self._lib))

    def take_events(self) -> dict[str, np.ndarray]:
        """Pending event records since the last call (and clears them)."""
        n = np.zeros(3, dtype=np.int64)
        self._lib.waterint_sc_pending(self._h, _ptr(n, ctypes.c_int64))
        events = np.zeros(int(n[0]), dtype=EVENT_DTYPE)
        frames = np.zeros(int(n[1]), dtype=np.int64)
        counts = np.zeros((int(n[1]), self.cells), dtype=np.int32)
        sites = np.zeros(int(n[2]), dtype=SITE_DTYPE)
        self._lib.waterint_sc_take(self._h, events.ctypes.data_as(_P), _ptr(frames, ctypes.c_int64), _ptr(counts, ctypes.c_int32),
                                   sites.ctypes.data_as(_P))
        return {"events": events, "count_frames": frames,
                "counts": counts.reshape(-1, len(self.groups), len(self.windows)), "sites": sites}

    def push(self, positions: np.ndarray, steps: np.ndarray, *, velocities: np.ndarray | None = None,
             boxes: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Returns (frame records, interval records). With a decomposition, ``last_decomposition`` holds the
        (n_intervals, groups, windows + 1, 3) charge-moment increments and ``last_zref`` the per-frame z_ref."""
        pos = np.ascontiguousarray(positions, dtype=np.float64)
        if pos.ndim != 3 or pos.shape[1:] != (self.natoms, 3):
            raise ValueError(f"positions must have shape (n_frames, {self.natoms}, 3).")
        nf = pos.shape[0]
        st = np.ascontiguousarray(steps, dtype=np.int64)
        if st.shape != (nf,):
            raise ValueError("steps must have one entry per frame.")
        vel = None if velocities is None else np.ascontiguousarray(velocities, dtype=np.float64)
        if vel is not None and vel.shape != pos.shape:
            raise ValueError("velocities must match positions.")
        bx = None if boxes is None else np.ascontiguousarray(boxes, dtype=np.float64).reshape(nf, 6)
        frames = np.zeros(nf, dtype=FRAME_DTYPE)
        intervals = np.zeros(nf, dtype=INTERVAL_DTYPE)
        n_int = ctypes.c_int64(0)
        dec = np.zeros((nf, self.cells, 3), dtype=np.float64) if self.cells else None
        zref = np.zeros(nf, dtype=np.float64) if self.cells else None
        status = self._lib.waterint_sc_push(self._h, nf, self.natoms, _ptr(pos, ctypes.c_double), _ptr(vel, ctypes.c_double),
                                            _ptr(st, ctypes.c_int64), _ptr(bx, ctypes.c_double),
                                            frames.ctypes.data_as(_P), intervals.ctypes.data_as(_P), ctypes.byref(n_int),
                                            _ptr(dec, ctypes.c_double), _ptr(zref, ctypes.c_double))
        if status != 0:
            raise RuntimeError(f"site-current kernel: {_error(self._lib)}")
        ni = n_int.value
        if dec is not None:
            self.last_decomposition = dec[:ni].reshape(ni, len(self.groups), len(self.windows), 3)
            self.last_zref = zref
        return frames, intervals[:ni]

    def stats(self) -> dict[str, float]:
        out = np.zeros(len(STATS_FIELDS), dtype=np.float64)
        self._lib.waterint_sc_stats(self._h, _ptr(out, ctypes.c_double))
        return {name: float(value) for name, value in zip(STATS_FIELDS, out)}

    def close(self) -> None:
        if getattr(self, "_h", None):
            self._lib.waterint_sc_free(self._h)
            self._h = None

    def __del__(self) -> None:  # pragma: no cover - best effort
        try:
            self.close()
        except Exception:
            pass
