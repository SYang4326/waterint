"""Strict carrier tracking and carrier site-path MSD.

carrier-track  (post)   owner-change event tables of site-current (events: enabled) -> tracks + carrier-O schedule
carrier-msd    (block)  site paths rebuilt every frame from positions + the tracks -> strict same-track xy MSD
The chain site-current -> carrier-track -> carrier-msd needs two passes (carrier-msd must know the tracks before it
reads positions): run site-current (+ carrier-track) first, then carrier-msd with ``carrier_msd.carrier_tracks``.
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from waterint.analysis.base import Analysis, Result
from waterint.analysis.dynamics.carrier_msd_core import CarrierSitePaths, strict_carrier_msd, zref_top_layer
from waterint.analysis.dynamics.carrier_msd_workflow import DEFAULT_CLASSES, _class_edges
from waterint.analysis.dynamics.carrier_track_core import track_event_tables
from waterint.analysis.dynamics.carrier_track_workflow import _sha, run_carrier_track, save_tracking


def _event_files(analysis: Analysis, raw: Any) -> list[str]:
    """Event tables: file paths, or the name of a site-current analysis that ran earlier in this run."""
    items = [raw] if isinstance(raw, (str, Path)) else list(raw or [])
    out: list[str] = []
    for item in items:
        res = analysis.run.results.get(str(item))
        if res is not None:
            if "events" not in res.outputs:
                raise ValueError(f"analysis {item!r} wrote no event table (set events.enabled: true).")
            out.append(res.outputs["events"])
        else:
            out.append(str(analysis.run.path(item)))
    if not out:
        raise ValueError("carrier tracking needs event tables (files or a site-current analysis name).")
    return out


class CarrierTrack(Analysis):
    type = "carrier-track"
    kind = "post"
    v06_name = "carrier-track"
    description = ("Track OH- charge carriers through proton-transfer events: direct hops and relay chains of up to 4 protons "
                   "(unique shortest path), with births and deaths, from the owner-change event tables of site-current; "
                   "writes the tracks and each track's carrier-O schedule.")
    params = {"carrier_track.events": ("required", "event tables in time order, or a site-current analysis name"),
              "carrier_track.positive": (False, "also track the +1 family (H3O+ / Os_H)")}

    def __init__(self, cfg: dict[str, Any], run):
        super().__init__(cfg, run)
        self.ct = dict(cfg.get("carrier_track") or {})

    def finalize(self) -> Result:
        legacy = {"input": {"events": _event_files(self, self.ct.get("events"))},
                  "carrier_track": {"positive": bool(self.ct.get("positive", False))},
                  "output": {"directory": str(self.run.output_dir), "prefix": self.name,
                             "verbose": bool(self.run.output_cfg.get("verbose", False))}}
        res = run_carrier_track(legacy)
        self.frames = res.frames
        self._meta_path = res.metadata_path
        return Result(self.type, self.name, {}, json.loads(res.metadata_path.read_text(encoding="utf-8")),
                      outputs={"npz": str(res.npz_path)})

    def write(self, result: Result, outdir: Path) -> dict[str, Any]:
        return {"npz": str(outdir / f"{self.name}.npz")}

    def metadata(self, result: Result) -> dict[str, Any]:
        return result.meta            # v0.6 metadata layout (written by the tracker), outputs added by save_result


class CarrierMsd(Analysis):
    type = "carrier-msd"
    kind = "block"
    v06_name = "carrier-msd"
    description = ("Same-track xy MSD of OH- carriers along their site paths (followed through every proton transfer), with "
                   "the identity-survival function and origin classes by z window.")
    params = {"carrier_msd.carrier_tracks": (None, "tracks file of carrier-track (or carrier_msd.events to track here)"),
              "carrier_msd.full_lag_ps": (200.0, "every saved frame up to this lag"), "carrier_msd.max_lag_ps": (10000.0, "longest lag (ps); sparse lags beyond full_lag_ps"),
              "carrier_msd.origin_classes": ("interface / bulk", "unions of profile windows (state at the origin)"),
              "carrier_msd.storage": (None, "directory for memory-mapped site paths (long runs)"),
              "input.dt_ps": (0.004, "time between saved frames")}

    def __init__(self, cfg: dict[str, Any], run):
        super().__init__(cfg, run)
        self.cm = dict(cfg.get("carrier_msd") or {})
        self.t0 = time.time()
        self.raw_profile = self.profile.raw
        ref = self.raw_profile.get("reference", {})
        if str(ref.get("mode", "top_layer")) != "top_layer" or "@solid" in ref.get("elements", []):
            raise ValueError("carrier-msd needs a top_layer reference plane from explicit reference elements.")
        self.width = float(ref.get("layer_width_A", 0.70))
        self.ref_el = list(ref.get("elements", []))
        self.acceptor, self.proton = self.profile.ownership.acceptors[0], self.profile.ownership.protons[0]
        opts = dict(run.input_spec.options) if run.input_spec is not None else {}
        self.opts = opts
        self.dt_ps = float(opts.get("dt_ps", 0.004))
        out = run.output_dir
        out.mkdir(parents=True, exist_ok=True)
        if self.cm.get("carrier_tracks"):
            tp = run.path(self.cm["carrier_tracks"])
            with np.load(tp) as z:
                tracks, changes, f0, fl = z["tracks"], z["changes"], int(z["first_frame"]), int(z["last_frame"])
            self.track_src = {"carrier_tracks": str(tp), "sha256": _sha(tp)}
        else:
            res = track_event_tables([Path(p) for p in _event_files(self, self.cm.get("events"))])
            tracks, changes, f0, fl = res.tracks, res.changes, res.first_frame, res.last_frame
            tp = out / f"{self.name}_carrier_tracks.npz"
            save_tracking(tp, res)
            self.track_src = {"tracked_from_events": True, "carrier_tracks": str(tp), "summary": res.summary}
        self.f0, self.fl = f0, fl
        storage = self.cm.get("storage")
        self.paths = CarrierSitePaths(tracks, changes, f0, fl, run.path(storage) if storage else None)
        self.cutoff = float(self.raw_profile.get("h_assignment", {}).get("cutoff_A", 1.5))
        self.t_read = 0.0

    def update_block(self, block) -> None:
        t = time.time()
        pos, sym = block.positions, np.asarray(block.symbols)
        if block.boxes is not None:                           # lammpstrj: atom id = index + 1
            if not np.array_equal(block.ids, np.arange(1, block.ids.size + 1)):
                order = np.argsort(block.ids)
                if not np.array_equal(block.ids[order], np.arange(1, block.ids.size + 1)):
                    raise ValueError("atom ids must be 1..N")
                pos, sym = pos[:, order], sym[order]
            lx, ly = float(block.boxes[0][1] - block.boxes[0][0]), float(block.boxes[0][3] - block.boxes[0][2])
            g = block.first + int(self.opts.get("first_frame_index", 0))
        else:
            lx, ly = float(block.cell[0]), float(block.cell[1])
            g = block.first
        zref = zref_top_layer(pos[:, np.isin(sym, self.ref_el), 2], self.width)
        self.paths.push(g, pos, lx, ly, zref, np.flatnonzero(sym == self.acceptor), np.flatnonzero(sym == self.proton), self.cutoff)
        self.t_read += time.time() - t

    def finalize(self) -> Result:
        p, fl, f0, dt_ps = self.paths, self.fl, self.f0, self.dt_ps
        if p.frames_seen <= fl:
            raise RuntimeError(f"positions end at frame {p.frames_seen - 1}, the tracks run to {fl}")
        if not p.complete().all():
            raise RuntimeError(f"{int((~p.complete()).sum())} tracks were not fully rebuilt")
        if p.nh_not_one:
            raise RuntimeError(f"tracked carrier O without exactly one H at {p.nh_not_one} of {p.nh_checks} checks: "
                               "positions and event tables do not belong together")
        classes = dict(self.cm.get("origin_classes") or DEFAULT_CLASSES)
        edges = _class_edges(self.raw_profile, classes)
        t_msd = time.time()
        res = strict_carrier_msd(p.birth, p.death, p.offset, p.xy, p.zrel, f0, fl, edges,
                                 full_lag=int(round(float(self.cm.get("full_lag_ps", 200.0)) / dt_ps)),
                                 max_lag=int(round(float(self.cm.get("max_lag_ps", 10000.0)) / dt_ps)))
        t_msd = time.time() - t_msd
        self._res, self._classes, self._edges = res, classes, edges
        meta = {
            "analysis": "waterint carrier-msd",
            "definition": "strict same-track OH-defect site-path xy MSD; dx^2 + dy^2, all "
                          "valid same-track origins, pair-weighted within the run; support = valid pairs / at-risk origins",
            "origin_classes": {"all": "all origins", **{k: {"windows": v, "z_rel_ranges_A": edges[k]} for k, v in classes.items()}},
            "tracks": self.track_src, "positions": {"trajectory": self.run.trajectory}, "frames": res.frames,
            "first_frame": f0, "last_frame": fl,
            "carrier_frames_per_class": dict(zip(res.classes, map(int, res.carrier_frames))),
            "checks": {"tracks_complete": int(p.complete().sum()), "carrier_nH_checks": p.nh_checks, "carrier_nH_not_one": p.nh_not_one},
            "lag_grid": {"dt_ps": dt_ps, "full_lag_ps": float(self.cm.get("full_lag_ps", 200.0)),
                         "max_lag_ps": float(res.lag_frames[-1] * dt_ps)},
            "profile": {"id": self.raw_profile.get("profile_id"), "path": self.raw_profile.get("_path")},
            "elapsed_s": {"read_and_paths": self.t_read, "msd": t_msd, "total": time.time() - self.t0},
            "interpretation_limit": "survival-conditioned defect self-MSD; not an unconditional diffusion coefficient, not a conductivity"}
        return Result(self.type, self.name, {}, meta)

    def write(self, result: Result, outdir: Path) -> dict[str, Any]:
        res, dt_ps = self._res, self.dt_ps
        npz_path, csv_path = outdir / f"{self.name}.npz", outdir / f"{self.name}.csv"
        tmp = npz_path.with_name(npz_path.name + ".tmp.npz")
        np.savez_compressed(tmp, classes=np.array(res.classes), lag_frames=res.lag_frames, lag_time_ps=res.lag_frames * dt_ps,
                            numerator_A2=res.numerator_A2, valid_pair_count=res.valid_pair_count,
                            at_risk_origin_count=res.at_risk_origin_count, msd_xy_A2=res.msd_A2,
                            identity_support_fraction=res.identity_support, carrier_frames=res.carrier_frames,
                            frames=np.array(res.frames), tracks=np.array(res.tracks), dt_ps=np.array(dt_ps))
        tmp.replace(npz_path)
        with open(csv_path, "w", newline="", encoding="utf-8") as handle:
            w = csv.writer(handle)
            w.writerow(["lag_ps"] + [f"msd_xy_A2_{c}" for c in res.classes] + [f"support_{c}" for c in res.classes]
                       + [f"pairs_{c}" for c in res.classes])
            for j, k in enumerate(res.lag_frames):
                w.writerow([f"{k * dt_ps:.6g}"] + [f"{v:.8g}" for v in res.msd_A2[:, j]] + [f"{v:.6g}" for v in res.identity_support[:, j]]
                           + [int(v) for v in res.valid_pair_count[:, j]])
        return {"npz": str(npz_path), "csv": str(csv_path)}

    def plot(self, result: Result, outdir: Path) -> dict[str, Any]:
        from waterint.plot.carrier import plot_carrier_msd
        path = outdir / f"{self.name}.png"
        plot_carrier_msd(path, self._res.lag_frames * self.dt_ps, self._res.msd_A2, self._res.identity_support, self._res.classes)
        return {"png": str(path)}

    def metadata(self, result: Result) -> dict[str, Any]:
        return result.meta
