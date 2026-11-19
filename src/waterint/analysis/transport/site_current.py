"""Complete formal-ion site current (block analysis, streaming C++ kernel).

    dM_ref  = sum_i q_i MIC(dr_i)                 (formal atomic reference current)
    B       = sum_assigned H MIC(r_H - r_owner)
    dM_site = dM_ref - dB                         (complete site current, primary)
Charges, proton roles, ownership cutoff/ties, periodicity, solid, windows and current groups all come from the
system profile (it needs formal_charges / species tables, e.g. mgo_water). With profile windows
and current_groups the exact decomposition of dM_site into (group, z-window) cells is written; events (owner changes,
tracked-site positions) feed carrier-track / carrier-msd.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from waterint.analysis.base import Analysis, Result
from waterint.analysis.transport.site_current_core import E_ANGSTROM_PER_PS_TO_SI, EVENTS_FORMAT, SiteCurrentAccumulator
from waterint.system import profile as wprof

KERNEL_SOURCE = Path(__file__).resolve().parents[2] / "backends" / "cpu" / "site_current.cpp"


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


class _DecCollector:
    """Sums the (group, window) xy increments over `n` consecutive intervals (exact moment subsampling)."""

    def __init__(self, n: int) -> None:
        self.n, self.parts, self.site = n, [], []
        self.carry: np.ndarray | None = None
        self.carry_site: np.ndarray | None = None

    def add(self, dec: np.ndarray, dmsite: np.ndarray) -> None:
        d, m = dec[..., :2], dmsite[:, :2]
        if self.carry is not None:
            d, m = np.concatenate([self.carry, d]), np.concatenate([self.carry_site, m])
        usable = (d.shape[0] // self.n) * self.n
        self.carry, self.carry_site = d[usable:].copy(), m[usable:].copy()
        if usable:
            self.parts.append(d[:usable].reshape(-1, self.n, *d.shape[1:]).sum(axis=1))
            self.site.append(m[:usable].reshape(-1, self.n, 2).sum(axis=1))


class _Collector:
    def __init__(self, level: str, coarsen: int) -> None:
        self.level, self.coarsen = level, coarsen
        self.frames: list[np.ndarray] = []
        self.intervals: list[np.ndarray] = []
        self.compact: dict[str, list[np.ndarray]] = {k: [] for k in ("start", "tmid", "dMref", "dMsite", "och", "micn", "free1", "ties1")}
        self._carry: np.ndarray | None = None
        self.n_frames = self.n_intervals = 0

    def add(self, frames: np.ndarray, intervals: np.ndarray) -> None:
        self.n_frames += frames.size
        self.n_intervals += intervals.size
        if self.level == "full":
            self.frames.append(frames)
            self.intervals.append(intervals)
            return
        block = intervals if self._carry is None else np.concatenate([self._carry, intervals])
        n = self.coarsen
        usable = (block.size // n) * n
        self._carry = block[usable:].copy()
        if usable == 0:
            return
        g = block[:usable].reshape(-1, n)
        c = self.compact
        c["start"].append(g["frame_start"][:, 0])
        c["tmid"].append(0.5 * (g["time_mid"][:, 0] + g["time_mid"][:, -1]))
        c["dMref"].append(g["dMref"].sum(axis=1) if n > 1 else g["dMref"][:, 0])
        c["dMsite"].append(g["dMsite"].sum(axis=1) if n > 1 else g["dMsite"][:, 0])
        c["och"].append(g["owner_changes"].sum(axis=1).astype(np.int32))
        c["micn"].append(g["max_mic_norm"].max(axis=1))
        c["free1"].append(g["free1"][:, -1])
        c["ties1"].append(g["ties1"][:, -1])

    def dropped(self) -> int:
        return 0 if self._carry is None else int(self._carry.size)


class SiteCurrent(Analysis):
    type = "site-current"
    kind = "block"
    v06_name = "site-current"
    description = ("Per-interval charge-moment increments dM_site = dM_ref - dB (formal charges and proton ownership from the "
                   "profile) from LAMMPS dumps or GPUMD quantized-npz segments; input for STACIE / Helfand sheet conductance.")
    params = {"site_current.decomposition": (True, "write the (group, window) decomposition (needs profile windows/groups)"),
              "site_current.neighbor_list / skin_A / audit_every": ((True, 1.0, 1000), "H-owner search acceleration (exact)"),
              "site_current.boundary_max_step_A": (1.0, "max MIC step at a segment boundary (qnpz)"),
              "site_current.output_level": ("compact", "compact | full"), "site_current.coarsen": (1, "intervals per compact row"),
              "site_current.decomposition_coarsen": (10, "intervals per decomposition row"),
              "events.enabled": (False, "write owner-change events and tracked-site positions"),
              "input (lammpstrj)": ("dt_ps 0.004, dstep 8", "time between saved frames / MD steps per frame"),
              "input (gpumd-qnpz)": ("seed_segment, first_frame_index, solid_from, md_timestep_fs, velocities",
                                     "segment chaining (last frame of the previous segment), frozen-solid source, MD timestep")}

    def __init__(self, cfg: dict[str, Any], run):
        super().__init__(cfg, run)
        sc = self.sc = dict(cfg.get("site_current") or {})
        self.level = str(sc.get("output_level", cfg.get("level", "compact"))).lower()
        self.coarsen = int(sc.get("coarsen", cfg.get("coarsen", 1)))
        if self.level not in {"full", "compact"}:
            raise ValueError("output level must be full or compact.")
        if self.coarsen < 1 or (self.level == "full" and self.coarsen != 1):
            raise ValueError("coarsen must be >= 1 and is only allowed with level: compact.")
        self.boundary_limit = float(sc.get("boundary_max_step_A", 1.0))
        self.nl_opts = {"neighbor_list": bool(sc.get("neighbor_list", True)), "skin_A": float(sc.get("skin_A", 1.0)),
                        "audit_every": int(sc.get("audit_every", 1000))}
        self.profile_raw = run.prep.profile.raw
        if "formal_charges" not in self.profile_raw:
            raise ValueError(f"site-current needs a profile with formal_charges (profile {run.prep.profile.profile_id!r} has none).")
        self.want_dec = bool(sc.get("decomposition", True)) and bool(self.profile_raw.get("windows")) \
            and bool(self.profile_raw.get("current_groups"))
        self.dec_n = int(sc.get("decomposition_coarsen", cfg.get("decomposition_coarsen", 10)))
        self.ev_cfg = dict(cfg.get("events") or {})
        self.want_ev = bool(self.ev_cfg.get("enabled", False))
        if self.want_ev and not self.want_dec:
            raise ValueError("events need the decomposition (profile windows and current_groups, site_current.decomposition: true).")
        spec = run.input_spec
        if spec is None or spec.format not in {"lammpstrj", "gpumd-qnpz"}:
            raise ValueError("site-current reads input.format lammpstrj or gpumd-qnpz.")
        self.fmt = spec.format
        self.opts = dict(spec.options)
        self.needs_velocities = bool(self.opts.get("velocities", False))
        self.collector = _Collector(self.level, self.coarsen)
        self.dcol = _DecCollector(self.dec_n) if self.want_dec else None
        self.ev_parts: dict[str, list[np.ndarray]] = {"events": [], "count_frames": [], "counts": [], "sites": []}
        self.zrefs: list[np.ndarray] = []
        self.acc: SiteCurrentAccumulator | None = None
        self.frame_offset = 0
        self.boundary_steps: list[dict[str, Any]] = []
        self.seg_first: dict[int, int] = {}
        self.t_start = time.time()
        self.source: dict[str, Any] = {"format": self.fmt}

    # -------------------------------------------------------------------------------------------------- setup
    def _init(self, block) -> None:
        raw = self.profile_raw
        if self.fmt == "lammpstrj":
            tmap = {int(k): str(v) for k, v in (raw.get("lammps_type_map") or {}).items()} or dict(self.run.prep.type_map)
            species = [tmap[int(t)] for t in block.types]
            self.rp = wprof.resolve(raw, species)
            self.dt_ps = float(self.opts.get("dt_ps", 0.004))
            dstep = int(self.opts.get("dstep", 8))
            self.acc = SiteCurrentAccumulator(profile=self.rp, ids=block.ids, box_bounds=block.boxes[0],
                                              first_step=int(block.steps[0]), dstep=dstep, dt_ps=self.dt_ps, **self.nl_opts)
            self.ids, self.types, self.box = block.ids, block.types.astype(np.int16), block.boxes[0]
            first_pos = block.positions[0]
            self.frame_offset = 0
        else:
            self.rp = wprof.resolve(raw, block.symbols)
            inv = {str(v): int(k) for k, v in (raw.get("lammps_type_map") or {}).items()}
            self.types = np.array([inv.get(sp, -1) for sp in self.rp.species], dtype=np.int16)
            self.ids = np.arange(1, len(self.rp.species) + 1, dtype=np.int32)
            cell = block.cell
            self.box = np.array([0.0, cell[0], 0.0, cell[1], 0.0, cell[2]])
            self.dt_ps = float(block.meta["dt_ps"])
            dstep = int(block.meta["dstep"])
            start_index = block.first
            self.acc = SiteCurrentAccumulator(profile=self.rp, ids=self.ids, box_bounds=self.box, first_step=start_index * dstep,
                                              dstep=dstep, dt_ps=self.dt_ps, t0_ps=start_index * self.dt_ps, **self.nl_opts)
            self.frame_offset = start_index
            first_pos = block.positions[0]
            sf = self.opts.get("solid_from")
            if sf and self.want_dec:
                ref = self.run.open_trajectory({"trajectory": sf, "format": "gpumd-qnpz"})
                first_pos = next(ref.iter_blocks()).positions[0]
                self.source["solid_frozen_from"] = f"{sf} frame 0"
            else:
                self.source["solid_frozen_from"] = "first frame given to this job" + (" (seed segment last frame)" if block.seed else "")
        if self.want_dec:
            solid = wprof.solid_mask(self.rp, first_pos, ids=self.ids, solid_ids=self.opts.get("solid_ids"))
            self.acc.set_decomposition(self.rp, solid)
        if self.want_ev:
            tracked = self.ev_cfg.get("tracked_species", raw.get("tracked_species", []))
            self.acc.set_events(self.rp, list(tracked), int(self.ev_cfg.get("counts_stride", 10)), int(self.ev_cfg.get("sites_stride", 10)))

    # -------------------------------------------------------------------------------------------------- stream
    def update_block(self, block) -> None:
        if self.acc is None:
            self._init(block)
        if self.fmt == "lammpstrj":
            fr, ir = self.acc.push(block.positions, block.steps, velocities=block.velocities, boxes=block.boxes)
        else:
            fr, ir = self.acc.push(block.positions, block.steps, velocities=None if block.seed else block.velocities)
        self.collector.add(fr, ir)
        if self.want_dec:
            if not block.seed:
                self.dcol.add(self.acc.last_decomposition, ir["dMsite"])
            self.zrefs.append(self.acc.last_zref)
        if self.want_ev and not block.seed:
            self._take_events()
        if self.fmt == "gpumd-qnpz" and not block.seed:
            self.seg_first.setdefault(block.segment, block.first)
            for s_idx, idx in self.seg_first.items():
                if idx > self.frame_offset and block.first <= idx < block.first + block.n \
                        and not any(b["segment"] == s_idx for b in self.boundary_steps):
                    rec = ir[ir["frame_start"] == idx - 1 - self.frame_offset]
                    if rec.size:
                        self.boundary_steps.append({"segment": s_idx, "first_frame_index": idx,
                                                    "max_MIC_step_norm_A": float(rec["max_mic_norm"][0])})

    def _take_events(self) -> None:
        got = self.acc.take_events()
        for key in self.ev_parts:
            self.ev_parts[key].append(got[key])

    # -------------------------------------------------------------------------------------------------- output
    def finalize(self) -> Result:
        if self.acc is None:
            raise RuntimeError(f"No frames in {self.run.trajectory}.")
        bad = [b for b in self.boundary_steps if b["max_MIC_step_norm_A"] > self.boundary_limit]
        if bad:
            raise RuntimeError(f"Segment discontinuity: boundary step above {self.boundary_limit} A: {bad}")
        outdir = self.run.output_dir
        outdir.mkdir(parents=True, exist_ok=True)
        prefix = self.name
        rp, dt_ps, off = self.rp, self.dt_ps, self.frame_offset
        events_meta = None
        if self.want_ev:
            self._take_events()
            cat = {k: (np.concatenate(v) if v else None) for k, v in self.ev_parts.items()}
            ev = (cat["events"] if cat["events"] is not None else np.zeros(0)).copy()
            sites = (cat["sites"] if cat["sites"] is not None else np.zeros(0)).copy()
            ev["frame"] += off
            sites["frame"] += off
            count_frames = cat["count_frames"] + off
            ev_path = outdir / f"{prefix}_events.npz"
            tmp_ev = ev_path.with_name(ev_path.name + ".tmp.npz")
            tracked = list(self.ev_cfg.get("tracked_species", self.profile_raw.get("tracked_species", [])))
            np.savez_compressed(
                tmp_ev, format=np.array(EVENTS_FORMAT), events=ev, count_frames=count_frames, counts=cat["counts"], sites=sites,
                groups=np.array(rp.group_names), windows=np.array(list(rp.window_names) + ["outside"]),
                window_edges_A=rp.edges.reshape(-1, 2), row_names=np.array(rp.row_names),
                row_labels=np.array([[rp.label_of(r, n) for n in range(rp.nh_max + 1)] for r in range(len(rp.row_names))]),
                dt_ps=np.array(dt_ps), tracked_species=np.array(tracked))
            tmp_ev.replace(ev_path)
            events_meta = {"path": str(ev_path), "format": EVENTS_FORMAT, "owner_change_events": int(ev.size),
                           "count_rows": int(count_frames.size), "tracked_site_records": int(sites.size),
                           "counts_stride": int(self.ev_cfg.get("counts_stride", 10)),
                           "sites_stride": int(self.ev_cfg.get("sites_stride", 10)), "frames_are_global_indices": True,
                           "sha256": _sha(ev_path)}
        stats = self.acc.stats()
        self.acc.close()
        box6 = np.asarray(self.box, dtype=float).reshape(3, 2)
        area = float(np.prod(box6[:2, 1] - box6[:2, 0]))
        lz = float(box6[2, 1] - box6[2, 0])
        types, ids, col = self.types, self.ids, self.collector
        if self.level == "full":
            fr, ir = np.concatenate(col.frames), np.concatenate(col.intervals)
            arrays = {
                "frame": fr["frame"] + off, "step": fr["step"], "time_ps": fr["time_ps"], "B_over_e_A": fr["B"],
                "assigned_H_count": fr["assigned"], "free_H_count": fr["free"], "tie_H_count": fr["ties"],
                "formal_charge_e": fr["formal_q"], "max_assigned_H_O_distance_A": fr["max_assigned_dist"],
                "interval_start_frame": ir["frame_start"] + off, "step_start": ir["step_start"], "time_mid_ps": ir["time_mid"],
                "delta_M_ref_eA": ir["dMref"], "delta_B_eA": ir["dB"], "delta_M_site_eA": ir["dMsite"],
                "delta_M_site_motion_eA": ir["motion"], "delta_M_owner_transfer_eA": ir["transfer"],
                "delta_image_correction_eA": ir["dI"], "edge_decomposition_residual_eA": ir["residual"],
                "J_formal_eA_per_ps": ir["dMref"] / dt_ps, "J_site_eA_per_ps": ir["dMsite"] / dt_ps,
                "J_formal_SI_Cm_per_s": ir["dMref"] / dt_ps * E_ANGSTROM_PER_PS_TO_SI,
                "J_site_SI_Cm_per_s": ir["dMsite"] / dt_ps * E_ANGSTROM_PER_PS_TO_SI,
                "owner_change_count": ir["owner_changes"], "xy_image_crossing_count": ir["crossings"][:, :2],
                "max_MIC_step_component_A": ir["max_mic_component"], "max_MIC_step_norm_A": ir["max_mic_norm"],
                "max_abs_raw_velocity_component": ir["max_raw_vel"], "max_abs_velocity_xy_dt_over_Lxy": ir["max_raw_vxy_dt_over_L"],
                "atom_id": ids, "atom_type": types, "box_bounds_A": box6, "xy_area_A2": np.array(area), "Lz_A": np.array(lz),
                "dt_ps": np.array(dt_ps)}
        else:
            c = col.compact
            cat = lambda k, dtype=None, shape=(0,): np.concatenate(c[k]) if c[k] else np.zeros(shape, dtype=dtype or float)
            dt_eff = dt_ps * self.coarsen
            dMref, dMsite = cat("dMref", shape=(0, 3)), cat("dMsite", shape=(0, 3))
            arrays = {
                "interval_start_frame": cat("start", np.int64) + off, "time_mid_ps": cat("tmid"),
                "delta_M_ref_eA": dMref, "delta_M_site_eA": dMsite,
                "J_formal_eA_per_ps": dMref / dt_eff, "J_site_eA_per_ps": dMsite / dt_eff,
                "owner_change_count": cat("och", np.int32), "max_MIC_step_norm_A": cat("micn"),
                "free_H_count_end": cat("free1", np.int32), "tie_H_count_end": cat("ties1", np.int32),
                "atom_type": types, "box_bounds_A": box6, "xy_area_A2": np.array(area), "Lz_A": np.array(lz),
                "dt_ps": np.array(dt_eff), "coarsen": np.array(self.coarsen)}
        dec_meta = None
        if self.want_dec:
            d = self.dcol
            dec = np.concatenate(d.parts) if d.parts else np.zeros((0, len(rp.group_names), len(rp.window_names) + 1, 2))
            dsite = np.concatenate(d.site) if d.site else np.zeros((0, 2))
            closure = float(np.max(np.abs(dec.sum(axis=(1, 2)) - dsite))) if dec.size else 0.0
            scale = 1.0 + (float(np.max(np.abs(dsite))) if dsite.size else 0.0)
            if closure > 1e-6 * scale:
                raise RuntimeError(f"decomposition closure failed after coarsening: {closure}")
            zr = np.concatenate(self.zrefs) if self.zrefs else np.zeros(0)
            arrays.update({
                "decomposition_xy_eA": dec.astype(np.float32), "decomposition_dt_ps": np.array(dt_ps * self.dec_n),
                "decomposition_groups": np.array(rp.group_names), "decomposition_windows": np.array(list(rp.window_names) + ["outside"]),
                "decomposition_window_edges_A": rp.edges.reshape(-1, 2), "z_ref_A": zr.astype(np.float32)})
            dec_meta = {"groups": rp.group_names, "windows": list(rp.window_names) + ["outside"], "coarsen": self.dec_n,
                        "intervals_written": int(dec.shape[0]),
                        "intervals_dropped": int(0 if d.carry is None else d.carry.shape[0]),
                        "max_closure_residual_eA_float64": closure, "stored_dtype": "float32 (sums computed in float64)",
                        "rules": "acceptor motion split half/half between its classes at the interval ends; windows by interval-midpoint "
                                 "z - z_ref; transfer by the donor/acceptor midpoint; image and free-H by the H midpoint"}
        for key, value in arrays.items():
            if key != "decomposition_window_edges_A" and value.dtype.kind in "fc" and not np.all(np.isfinite(value)):
                raise RuntimeError(f"nonfinite output array {key}")
        npz_path = outdir / f"{prefix}.npz"
        tmp = npz_path.with_name(npz_path.name + ".tmp.npz")
        np.savez_compressed(tmp, **arrays)
        tmp.replace(npz_path)
        meta = {
            "analysis": "waterint site-current",
            "definition": "complete formal-ion site current dM_site = dM_ref - dB (waterint site_current kernel)",
            "kernel_source_sha256": _sha(KERNEL_SOURCE) if KERNEL_SOURCE.is_file() else None,
            "source": {**self.source, "trajectory": self.run.trajectory}, "output_level": self.level, "coarsen": self.coarsen,
            "frames": col.n_frames, "intervals": col.n_intervals,
            "compact_intervals_written": None if self.level == "full" else int(arrays["delta_M_site_eA"].shape[0]),
            "intervals_dropped_by_coarsen": col.dropped(), "dt_ps": dt_ps, "box_bounds_A": box6.tolist(),
            "xy_area_A2": area, "Lz_A": lz, "atom_count": int(len(types)),
            "kernel_stats": stats, "neighbor_list": self.nl_opts, "segment_boundary_steps": self.boundary_steps,
            "profile": {"id": rp.profile_id, "path": self.profile_raw.get("_path")},
            "decomposition": dec_meta, "events": events_meta,
            "units": {"delta_M": "e*angstrom", "J": "e*angstrom/ps", "SI_factor_e_angstrom_per_ps_to_A_m": E_ANGSTROM_PER_PS_TO_SI},
            "npz_sha256": _sha(npz_path), "elapsed_s": time.time() - self.t_start,
            "warnings": ["bounded saved-frame MIC image reconstruction; within-step whole-cell aliasing cannot be excluded by saved samples",
                         "formal ionic reference is not automatically the electronic/physical current",
                         "no conductivity fit or DC equivalence claim",
                         "with seeded independent segments the motion/transfer/image decomposition depends on the unwrap origin; "
                         "dM_site and dM_ref do not"]}
        self._paths = {"npz": str(npz_path), **({"events": events_meta["path"]} if events_meta else {})}
        return Result(self.type, self.name, {}, meta)

    def write(self, result: Result, outdir: Path) -> dict[str, Any]:
        return dict(self._paths)
