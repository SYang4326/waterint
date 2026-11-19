"""Config workflow: sheet conductance from `site-current` outputs (one or more segment files per run).

    runs: {run_name: [site_current.npz, ...] | "glob/pattern/*/site_current.npz"}
    conductance: {current: site | formal | both, temperature_K: 300, time_window_ps: [t0, t1] | null,
                  fcut_max_THz: null, pooled: true, helfand_windows_ps: [[20, 60], [20, 200], [200, 1000]],
                  coarsen: 1,                       # sum n consecutive increments on load (exact; e.g. 10 = 40 fs)
                  decomposition: {groups: [...] | all, windows: [...] | all, combine_windows: false},
                  zones: {interface: [below_L1, L1, L2], bulk: [bulk]} | true | null}
`zones`: z-zone partition of G from the profile-window decomposition, all current groups summed per zone,
the remaining windows as `rest`; Helfand matrix over the zones (cross terms included) and the additive partition
G_zone = row sum (own term + all cross terms with the other zones), sum over zones = total. `true` = the MgO default
interface z_rel < 4 A (below_L1, L1, L2) / bulk 4-30 A (G is not split into L1 / L2 separately).
Segments of a run are concatenated in interval order; gaps or overlaps are errors. Each run is fitted on
its own (primary statistic: equal-run mean, SD ddof=1, SEM); the pooled-spectrum fit is a separate estimator.
"""
from __future__ import annotations

import glob
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from waterint.analysis.transport import conductance_core as cd
from waterint.analysis.spectra._workflow_utils import resolve_path
from waterint.analysis.spectra._workflow_utils import require_mapping


@dataclass(frozen=True)
class SiteConductanceResult:
    summary_path: Path
    runs: int


def _paths(config, spec) -> list[Path]:
    items = [spec] if isinstance(spec, (str, Path)) else list(spec)
    out: list[Path] = []
    for item in items:
        p = str(resolve_path(config, str(item)))
        hits = sorted(glob.glob(p)) if any(ch in p for ch in "*?[") else [p]
        out.extend(Path(h) for h in hits)
    if not out:
        raise FileNotFoundError(f"No site-current files for {spec!r}.")
    return out


DEFAULT_ZONES = {"interface": ["below_L1", "L1", "L2"], "bulk": ["bulk"]}


def _coarsen_run(run: dict, n: int) -> dict:
    """Exact moment subsampling: sums of n consecutive increments (identical to summing the increments of the coarser grid)."""
    nb = run["start"].size // n
    out = dict(run)
    out.update(start=run["start"][: nb * n: n], tmid=run["tmid"][: nb * n].reshape(nb, n).mean(axis=1),
               site=run["site"][: nb * n].reshape(nb, n, 3).sum(axis=1), formal=run["formal"][: nb * n].reshape(nb, n, 3).sum(axis=1),
               dt_ps=run["dt_ps"] * n, coarsen=n, intervals_dropped_by_coarsen=int(run["start"].size - nb * n))
    return out


def load_run(paths: list[Path], want_dec: bool, coarsen: int = 1) -> dict:
    parts = []
    for p in paths:
        with np.load(p) as z:
            d = {k: z[k] for k in ("interval_start_frame", "time_mid_ps", "delta_M_site_eA", "delta_M_ref_eA", "xy_area_A2", "dt_ps")}
            # coarsened files hold exact sums of consecutive increments; dt_ps already is the coarsened step
            if want_dec:
                if "decomposition_xy_eA" not in z.files:
                    raise ValueError(f"{p} has no decomposition.")
                d.update({k: z[k] for k in ("decomposition_xy_eA", "decomposition_dt_ps", "decomposition_groups", "decomposition_windows")})
            parts.append(d)
    parts.sort(key=lambda d: int(d["interval_start_frame"][0]))
    dt = float(parts[0]["dt_ps"]); area = float(parts[0]["xy_area_A2"])
    for a, b in zip(parts, parts[1:]):
        # interval index step between records (1 for native files, n for files coarsened by n)
        step = int(np.median(np.diff(a["interval_start_frame"]))) if a["interval_start_frame"].size > 1 else 1
        if int(b["interval_start_frame"][0]) != int(a["interval_start_frame"][-1]) + step:
            raise ValueError(f"segment gap/overlap: interval {int(a['interval_start_frame'][-1])} -> {int(b['interval_start_frame'][0])}")
        if float(b["dt_ps"]) != dt or float(b["xy_area_A2"]) != area:
            raise ValueError("segments differ in dt or area")
    run = {"dt_ps": dt, "area_A2": area,
           "start": np.concatenate([d["interval_start_frame"] for d in parts]),
           "tmid": np.concatenate([d["time_mid_ps"] for d in parts]),
           "site": np.concatenate([d["delta_M_site_eA"] for d in parts]),
           "formal": np.concatenate([d["delta_M_ref_eA"] for d in parts]), "files": [str(p) for p in paths]}
    if want_dec:
        run.update({"dec": np.concatenate([d["decomposition_xy_eA"] for d in parts]).astype(np.float64),
                    "dec_dt_ps": float(parts[0]["decomposition_dt_ps"]),
                    "groups": [str(g) for g in parts[0]["decomposition_groups"]],
                    "windows": [str(w) for w in parts[0]["decomposition_windows"]]})
    return _coarsen_run(run, coarsen) if coarsen > 1 else run


def run_site_conductance(config: dict[str, Any]) -> SiteConductanceResult:
    runs_cfg = require_mapping(config, "runs")
    c = dict(config.get("conductance") or {})
    out_cfg = require_mapping(config, "output")
    out = resolve_path(config, out_cfg.get("directory", "."))
    out.mkdir(parents=True, exist_ok=True)
    prefix = str(out_cfg.get("prefix", "conductance"))
    T = float(c.get("temperature_K", 300.0))
    which = str(c.get("current", "site"))
    currents = ["site", "formal"] if which == "both" else [which]
    tw = c.get("time_window_ps")
    fcut = c.get("fcut_max_THz")
    fcut_Hz = None if fcut is None else float(fcut) * 1e12
    hw = [tuple(map(float, w)) for w in c.get("helfand_windows_ps", [[20, 60], [20, 200], [200, 1000]])]
    dec_cfg = c.get("decomposition") or None
    zones = c.get("zones")
    zones = DEFAULT_ZONES if zones is True else (dict(zones) if zones else None)
    coarsen = int(c.get("coarsen", 1))
    zone_rows: dict[str, list] = {}
    summary: dict[str, Any] = {"stacie": cd.stacie_version(), "temperature_K": T, "time_window_ps": tw, "fcut_max_THz": fcut,
                               "estimator": f"STACIE {cd.MODEL}, joint x/y, centred", "runs": {}}
    per_current: dict[str, list[float]] = {k: [] for k in currents}
    pooled_inputs: dict[str, tuple[list, list]] = {k: ([], []) for k in currents}
    arrays: dict[str, np.ndarray] = {}
    for name, spec in runs_cfg.items():
        run = load_run(_paths(config, spec), want_dec=bool(dec_cfg) or bool(zones), coarsen=coarsen)
        sel = np.ones(run["tmid"].size, dtype=bool)
        if tw is not None:
            sel = (run["tmid"] >= float(tw[0])) & (run["tmid"] < float(tw[1]))
        pref = cd.sheet_prefactor(run["area_A2"], T)
        rec_run: dict[str, Any] = {"files": run["files"], "intervals": int(sel.sum()), "area_A2": run["area_A2"], "dt_ps": run["dt_ps"]}
        for cur in currents:
            seq, means = cd.current_sequences(run[cur][sel], run["dt_ps"])
            rec, arr = cd.fit_stacie(seq, pref, run["dt_ps"] * 1e-12, fcut_max_Hz=fcut_Hz, label=f"{name}_{cur}")
            rec["removed_component_means_A_m"] = means.tolist()
            helf = cd.helfand_matrix(run[cur][sel][None, :, :2], run["dt_ps"], run["area_A2"], T, hw)
            rec["helfand_G_nS"] = {k: float(v[0, 0]) for k, v in helf["windows"].items()}
            rec_run[cur] = rec
            per_current[cur].append(rec["G_S"])
            pooled_inputs[cur][0].append(seq)
            pooled_inputs[cur][1].append(pref)
            for k, v in arr.items():
                arrays[f"{name}/{cur}/{k}"] = v
        if dec_cfg:
            rec_run["decomposition"] = _decomposition(run, sel, dec_cfg, T, hw, fcut_Hz, name, arrays)
        if zones:
            rec_run["zones"] = _zones(run, sel, zones, T, hw)
            for key, rows in rec_run["zones"]["partition_nS"].items():
                zone_rows.setdefault(key, []).append([rows[z] for z in rec_run["zones"]["zones"]] + [rec_run["zones"]["total_nS"][key]])
        summary["runs"][name] = rec_run
    summary["equal_run"] = {cur: cd.equal_run_stats(v) for cur, v in per_current.items()}
    if zones:
        names = list(zones) + ["rest", "total"]
        summary["zones_equal_run"] = {
            key: {z: cd.equal_run_stats(np.asarray(v)[:, i] * 1e-9) for i, z in enumerate(names)} for key, v in zone_rows.items()}
        summary["zones_definition"] = {"zones": zones, "rest": "all other windows (incl. outside)", "partition": "row sums of the Helfand matrix"}
    if bool(c.get("pooled", True)) and len(runs_cfg) > 1:
        summary["pooled"] = {}
        for cur in currents:
            seqs, prefs = pooled_inputs[cur]
            nmin = min(q.shape[1] for q in seqs)
            # STACIE pools equal-length sequences: truncate every run to the shortest and re-centre
            seqs = [q[:, :nmin] - q[:, :nmin].mean(axis=1, keepdims=True) for q in seqs]
            rec, arr = cd.fit_stacie(seqs, prefs, list(summary["runs"].values())[0]["dt_ps"] * 1e-12, fcut_max_Hz=fcut_Hz, label=f"pooled_{cur}")
            rec["note"] = "pooled-spectrum fit: separate nonlinear estimator; within-fit std is not a between-run SEM"
            rec["pooled_intervals_per_run"] = int(nmin)
            summary["pooled"][cur] = rec
    np.savez_compressed(out / f"{prefix}_arrays.npz", **{k.replace("/", "__"): v for k, v in arrays.items()})
    path = out / f"{prefix}.json"
    path.write_text(json.dumps(summary, indent=1, default=float) + "\n", encoding="utf-8")
    return SiteConductanceResult(path, len(runs_cfg))


def _decomposition(run, sel, dec_cfg, T, hw, fcut_Hz, name, arrays) -> dict:
    """Helfand G_ab over selected (group, window) components + STACIE of each diagonal component."""
    n = int(round(run["dec_dt_ps"] / run["dt_ps"]))
    # time selection at decomposition resolution: block k covers intervals [k n, (k+1) n)
    blocks = sel[: (sel.size // n) * n].reshape(-1, n).all(axis=1)
    dec = run["dec"][: blocks.size][blocks]                       # (m, G, W, 2)
    groups, windows = run["groups"], run["windows"]
    gsel = groups if dec_cfg.get("groups", "all") == "all" else list(dec_cfg["groups"])
    wsel = windows if dec_cfg.get("windows", "all") == "all" else list(dec_cfg["windows"])
    comps, labels = [], []
    if bool(dec_cfg.get("combine_windows", False)):
        for g in gsel:
            comps.append(dec[:, groups.index(g), [windows.index(w) for w in wsel]].sum(axis=1)); labels.append(g)
    else:
        for g in gsel:
            for w in wsel:
                comps.append(dec[:, groups.index(g), windows.index(w)]); labels.append(f"{g}@{w}")
    comps = np.stack(comps)                                        # (K, m, 2)
    total = dec.sum(axis=(1, 2))
    helf = cd.helfand_matrix(np.concatenate([comps, total[None]]), run["dec_dt_ps"], run["area_A2"], T, hw)
    pref = cd.sheet_prefactor(run["area_A2"], T)
    diag = {}
    for k, lab in enumerate(labels):
        if not np.any(comps[k]):
            diag[lab] = None
            continue
        seq, _ = cd.current_sequences(comps[k], run["dec_dt_ps"])
        try:
            rec, _ = cd.fit_stacie(seq, pref, run["dec_dt_ps"] * 1e-12, fcut_max_Hz=fcut_Hz, label=f"{name}_{lab}")
            diag[lab] = {"G_nS": rec["G_nS"], "fit_std_nS": rec["fit_std_S"] * 1e9, "fit_flags": rec["fit_flags"]}
        except RuntimeError as err:
            diag[lab] = {"error": str(err)}
    for key, G in helf["windows"].items():
        arrays[f"{name}/decomposition/helfand_{key}"] = G
    return {"components": labels, "selected_groups": gsel, "selected_windows": wsel, "dt_ps": run["dec_dt_ps"],
            "helfand_G_nS": {k: {"matrix_components_plus_total": G.tolist(),
                                 "sum_of_component_block": float(G[:-1, :-1].sum()), "total": float(G[-1, -1])}
                             for k, G in helf["windows"].items()},
            "stacie_diagonal": diag,
            "note": "component conductances are not additive without cross terms; the Helfand matrix includes them"}


def _zones(run, sel, zones, T, hw) -> dict:
    """Helfand matrix over z zones (all groups summed per zone) and its additive row-sum partition."""
    n = int(round(run["dec_dt_ps"] / (run["dt_ps"] / run.get("coarsen", 1))))
    nb_sel = sel.size * run.get("coarsen", 1)                     # selection back at native interval resolution
    native_sel = np.repeat(sel, run.get("coarsen", 1))[:nb_sel]
    blocks = native_sel[: (native_sel.size // n) * n].reshape(-1, n).all(axis=1)
    dec = run["dec"][: blocks.size][blocks]                        # (m, G, W, 2)
    windows = run["windows"]
    missing = [w for ws in zones.values() for w in ws if w not in windows]
    if missing:
        raise ValueError(f"zones use windows {missing} that the decomposition does not have ({windows})")
    comps = [dec[:, :, [windows.index(w) for w in ws]].sum(axis=(1, 2)) for ws in zones.values()]
    total = dec.sum(axis=(1, 2))
    comps.append(total - sum(comps))
    helf = cd.helfand_matrix(np.stack(comps), run["dec_dt_ps"], run["area_A2"], T, hw)
    labels = list(zones) + ["rest"]
    return {"zones": labels, "dt_ps": run["dec_dt_ps"], "intervals": int(dec.shape[0]),
            "matrix_nS": {k: G.tolist() for k, G in helf["windows"].items()},
            "partition_nS": {k: dict(zip(labels, map(float, G.sum(axis=1)))) for k, G in helf["windows"].items()},
            "own_term_nS": {k: dict(zip(labels, map(float, np.diag(G)))) for k, G in helf["windows"].items()},
            "total_nS": {k: float(G.sum()) for k, G in helf["windows"].items()}}
