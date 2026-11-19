"""Dynamic proton defects and conductivity.

defect-msd           framewise species selection, gated Hungarian tracking, lifetime-aware MSD
defect-conductivity  + Nernst-Einstein (defect MSD) and collective Green-Kubo (recommended)
conductivity         fixed stable carriers: MSD + Nernst-Einstein (not for identity-changing defects)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from waterint.analysis.base import Analysis, Result
from waterint.analysis.dynamics.msd import Msd
from waterint.analysis.selectors import parse_range
from waterint.analysis.selectors import select as select_atoms
from waterint.analysis.transport.conductivity_core import compute_nernst_einstein_conductivity
from waterint.analysis.transport.defect_core import compute_defect_msd, compute_green_kubo_conductivity, track_defects
from waterint.prep.labelled import LabelledFrame
from waterint.prep.reference import coordinate_spec_from_config


def _axis(value: Any, name: str = "plane_normal_axis") -> int:
    label = str(value).lower().lstrip("-")
    if label not in {"x", "y", "z"}:
        raise ValueError(f"{name} must be x, y, or z.")
    return "xyz".index(label)


def _pbc(value: Any) -> tuple[bool, bool, bool]:
    return (value,) * 3 if isinstance(value, bool) else tuple(bool(v) for v in value)  # type: ignore[return-value]


def _pos_int(value: Any, name: str) -> int:
    v = int(value)
    if v <= 0:
        raise ValueError(f"{name} must be positive.")
    return v


def _opt_int(value: Any) -> int | None:
    return None if value in {None, "", 0, "all"} else _pos_int(value, "max_lag_frames")


def analysis_volume(cells: np.ndarray, volume_cfg: dict[str, Any], default_axis: Any = "z") -> tuple[float, float | None]:
    mode = str(volume_cfg.get("mode", "cell")).lower()
    if mode == "cell":
        volumes, thickness = np.abs(np.linalg.det(cells)), None
    elif mode == "slab":
        thickness = float(volume_cfg["thickness_A"]) if "thickness_A" in volume_cfg else None
        if thickness is None or thickness <= 0:
            raise ValueError("conductivity.volume.thickness_A must be positive.")
        axis = _axis(volume_cfg.get("normal_axis", default_axis), "conductivity.volume.normal_axis")
        other = [v for v in range(3) if v != axis]
        volumes = np.linalg.norm(np.cross(cells[:, other[0], :], cells[:, other[1], :]), axis=1) * thickness
    else:
        raise ValueError("conductivity.volume.mode must be cell or slab.")
    if np.any(volumes <= 0):
        raise ValueError("Conductivity requires positive analysis volumes.")
    return float(np.mean(volumes)), thickness


class DefectMsd(Analysis):
    type = "defect-msd"
    v06_name = "defect-msd"
    description = "Classify defects per frame, track them with Hungarian assignment, and compute lifetime-aware MSD."
    params = {
        "selection.oxygen_species": ("required", "defect species, e.g. [OH-]"),
        "defect_tracking.timestep_ps": ("required", "time between analysed frames (ps)"),
        "defect_tracking.gate_A": (3.0, "maximum hop distance for track continuation (A)"),
        "defect_tracking.pbc": ([True, True, True], "periodic axes"),
        "defect_tracking.layer": (None, "{coordinate, range}: per-frame layer filter"),
        "defect_msd.dimensionality": ("3d", "2d | 3d"), "defect_msd.plane_normal_axis": ("z", "excluded axis for 2d"),
        "defect_msd.max_lag_frames": (None, "maximum lag (frames); none = all"),
        "defect_msd.origin_stride": (1, "origin stride (frames)"),
        "defect_msd.frame_stride": (1, "use every n-th tracked frame"),
    }
    charge_e = -1.0

    def __init__(self, cfg: dict[str, Any], run):
        super().__init__(cfg, run)
        self.selection = cfg.get("selection")
        if not isinstance(self.selection, dict) or "oxygen_species" not in self.selection:
            raise ValueError("Dynamic defect tracking requires selection.oxygen_species.")
        self.tracking_cfg = dict(cfg.get("defect_tracking") or {})
        self.msd_cfg = dict(cfg.get("defect_msd") or {})
        if "timestep_ps" not in self.tracking_cfg:
            raise ValueError("defect_tracking.timestep_ps is required.")
        layer = self.tracking_cfg.get("layer")
        self.layer = None
        if layer is not None:
            if not isinstance(layer, dict):
                raise ValueError("defect_tracking.layer must be a mapping.")
            self.layer = (coordinate_spec_from_config(layer.get("coordinate", layer)), parse_range(layer.get("range"), "layer.range"))
        self.positions: list[np.ndarray] = []
        self.indices: list[np.ndarray] = []
        self.vectors: list[np.ndarray] = []
        self.steps: list[int] = []

    def update(self, lf: LabelledFrame) -> None:
        selected = select_atoms(lf, self.selection, self.ownership_params())
        if self.layer is not None:
            spec, (lo, hi) = self.layer
            values = lf.coordinates(selected, spec)
            selected = selected[(values >= lo) & (values < hi)]
        self.positions.append(np.asarray(lf.positions[selected], dtype=float))
        self.indices.append(np.asarray(selected, dtype=np.int64))
        self.vectors.append(lf.cell_vectors)
        self.steps.append(len(self.steps) if lf.frame.step is None else int(lf.frame.step))

    def _track(self):
        if len(self.positions) < 2:
            raise ValueError("Dynamic defect analysis requires at least two trajectory frames.")
        tracking = track_defects(self.positions, self.indices, cell_vectors=np.asarray(self.vectors),
                                 pbc=_pbc(self.tracking_cfg.get("pbc", [True, True, True])),
                                 timestep_ps=float(self.tracking_cfg["timestep_ps"]),
                                 gate_a=float(self.tracking_cfg.get("gate_A", 3.0)), charge_e=self.charge_e)
        msd = compute_defect_msd(
            tracking, max_lag_frames=_opt_int(self.msd_cfg.get("max_lag_frames")),
            origin_stride=_pos_int(self.msd_cfg.get("origin_stride", 1), "defect_msd.origin_stride"),
            frame_stride=_pos_int(self.msd_cfg.get("frame_stride", 1), "defect_msd.frame_stride"),
            dimensionality=self.msd_cfg.get("dimensionality", "3d"),
            plane_normal_axis=_axis(self.msd_cfg.get("plane_normal_axis", "z")))
        return tracking, msd

    def finalize(self) -> Result:
        tracking, msd = self._track()
        self._objects = (tracking, msd, np.asarray(self.steps, dtype=np.int64))
        return Result(self.type, self.name, {"time_ps": msd.time_ps, "msd_a2": msd.msd_a2}, self._tracking_meta(tracking))

    def _tracking_meta(self, tracking) -> dict[str, Any]:
        return {"method": "framewise species classification and gated Hungarian tracking",
                "initial_carriers": int(tracking.carrier_counts[0]), "mean_carriers": tracking.mean_carriers,
                "segments": len(tracking.segments), "births": int(tracking.births.sum()), "deaths": int(tracking.deaths.sum())}

    def write(self, result: Result, outdir: Path) -> dict[str, Any]:
        from waterint.plot.defect_transport import write_defect_events_csv, write_defect_msd_csv, write_defect_tracks_csv
        tracking, msd, steps = self._objects
        paths = {"csv": outdir / f"{self.name}.csv", "tracks_csv": outdir / f"{self.name}_tracks.csv",
                 "events_csv": outdir / f"{self.name}_events.csv"}
        write_defect_msd_csv(paths["csv"], msd)
        write_defect_tracks_csv(paths["tracks_csv"], tracking, steps)
        write_defect_events_csv(paths["events_csv"], tracking, steps)
        return {k: str(v) for k, v in paths.items()}

    def plot(self, result: Result, outdir: Path) -> dict[str, Any]:
        from waterint.plot.msd import plot_msd
        _tracking, msd, _steps = self._objects
        path = outdir / f"{self.name}.png"
        out = self.run.output_cfg
        plot_msd(path, msd.time_ps, msd.msd_a2, title=str(out.get("title", "Dynamic defect MSD")),
                 dimensionality=msd.dimensionality, dpi=int(out.get("dpi", 220)))
        return {"png": str(path)}


class DefectConductivity(DefectMsd):
    type = "defect-conductivity"
    v06_name = "defect-conductivity"
    description = "Compare dynamic-defect MSD Nernst-Einstein with collective STACIE Green-Kubo conductivity."
    params = {**DefectMsd.params,
              "defect_conductivity.temperature_K": ("required", "temperature (K)"),
              "defect_conductivity.charge_e": (-1.0, "carrier charge (e)"),
              "defect_conductivity.estimator": ("green_kubo", "nernst_einstein | green_kubo | both"),
              "defect_conductivity.fit_range_ps": (None, "[start, stop] for Nernst-Einstein"),
              "defect_conductivity.volume": ({"mode": "cell"}, "cell | slab (+ thickness_A, normal_axis)")}

    def __init__(self, cfg: dict[str, Any], run):
        super().__init__(cfg, run)
        self.cond = dict(cfg.get("defect_conductivity") or {})
        if "temperature_K" not in self.cond:
            raise ValueError("defect_conductivity.temperature_K is required.")
        self.charge_e = float(self.cond.get("charge_e", -1.0))
        self.estimator = str(self.cond.get("estimator", "green_kubo")).lower()
        if self.estimator not in {"nernst_einstein", "green_kubo", "both"}:
            raise ValueError("defect_conductivity.estimator must be nernst_einstein, green_kubo, or both.")

    def finalize(self) -> Result:
        tracking, msd = self._track()
        steps = np.asarray(self.steps, dtype=np.int64)
        vol_cfg = self.cond.get("volume", {"mode": "cell"})
        if not isinstance(vol_cfg, dict):
            raise ValueError("defect_conductivity.volume must be a mapping.")
        volume, thickness = analysis_volume(np.asarray(self.vectors, dtype=float), vol_cfg, self.msd_cfg.get("plane_normal_axis", "z"))
        temperature = float(self.cond["temperature_K"])
        ne = gk = None
        if self.estimator in {"nernst_einstein", "both"}:
            fit = self.cond.get("fit_range_ps")
            if not isinstance(fit, list) or len(fit) != 2:
                raise ValueError("defect_conductivity.fit_range_ps is required for defect Nernst-Einstein.")
            ne = compute_nernst_einstein_conductivity(
                msd.time_ps, msd.msd_a2, carrier_count=tracking.mean_carriers, volume_a3=volume, temperature_k=temperature,
                charge_e=self.charge_e, dimensions=2 if msd.dimensionality == "2d" else 3,
                fit_range_ps=(float(fit[0]), float(fit[1])), sheet_thickness_a=thickness)
        if self.estimator in {"green_kubo", "both"}:
            gk = compute_green_kubo_conductivity(tracking, volume_a3=volume, temperature_k=temperature,
                                                 dimensionality=msd.dimensionality,
                                                 plane_normal_axis=_axis(self.msd_cfg.get("plane_normal_axis", "z")))
        self._objects = (tracking, msd, steps)
        self._cond = (ne, gk, temperature, volume, thickness)
        meta = {**self._tracking_meta(tracking), "recommended_estimator": "collective Green-Kubo", "volume_A3": volume,
                "nernst_einstein_S_per_m": None if ne is None else ne.conductivity_s_per_m,
                "green_kubo_S_per_m": None if gk is None else gk.conductivity_s_per_m,
                "green_kubo_std_S_per_m": None if gk is None else gk.conductivity_std_s_per_m}
        return Result(self.type, self.name, {"time_ps": msd.time_ps, "msd_a2": msd.msd_a2}, meta)

    def write(self, result: Result, outdir: Path) -> dict[str, Any]:
        from waterint.plot.defect_transport import (write_defect_current_csv, write_defect_events_csv, write_defect_msd_csv,
                                                    write_defect_tracks_csv)
        tracking, msd, steps = self._objects
        ne, gk, temperature, volume, thickness = self._cond
        p = {k: outdir / f"{self.name}{suffix}.csv" for k, suffix in
             (("csv", ""), ("msd_csv", "_msd"), ("tracks_csv", "_tracks"), ("events_csv", "_events"), ("current_csv", "_current"))}
        with p["csv"].open("w", encoding="utf-8") as h:
            h.write("estimator,temperature_K,charge_e,mean_carriers,volume_A3,conductivity_S_per_m,"
                    "statistical_std_S_per_m,conductivity_S_per_cm,sheet_conductance_S\n")
            mc = tracking.mean_carriers
            if ne is not None:
                sheet = "" if ne.sheet_conductance_s is None else f"{ne.sheet_conductance_s:.10g}"
                h.write(f"defect_nernst_einstein,{temperature:.10g},{self.charge_e:.10g},{mc:.10g},{volume:.10g},"
                        f"{ne.conductivity_s_per_m:.10g},,{ne.conductivity_s_per_m / 100.0:.10g},{sheet}\n")
            if gk is not None:
                sheet = "" if thickness is None else f"{gk.conductivity_s_per_m * thickness * 1.0e-10:.10g}"
                h.write(f"defect_green_kubo,{temperature:.10g},{self.charge_e:.10g},{mc:.10g},{volume:.10g},"
                        f"{gk.conductivity_s_per_m:.10g},{gk.conductivity_std_s_per_m:.10g},"
                        f"{gk.conductivity_s_per_m / 100.0:.10g},{sheet}\n")
        write_defect_msd_csv(p["msd_csv"], msd)
        write_defect_tracks_csv(p["tracks_csv"], tracking, steps)
        write_defect_events_csv(p["events_csv"], tracking, steps)
        write_defect_current_csv(p["current_csv"], tracking, steps)
        return {k: str(v) for k, v in p.items()}

    def plot(self, result: Result, outdir: Path) -> dict[str, Any]:
        ne = self._cond[0]
        if ne is None:
            return {}
        from waterint.plot.conductivity import plot_conductivity_msd
        path = outdir / f"{self.name}.png"
        out = self.run.output_cfg
        plot_conductivity_msd(path, ne, title=str(out.get("title", "Dynamic defect MSD and Nernst-Einstein fit")),
                              dpi=int(out.get("dpi", 220)))
        return {"png": str(path)}


class Conductivity(Msd):
    type = "conductivity"
    v06_name = "conductivity"
    description = "Fit a fixed stable-carrier MSD; not valid for identity-changing proton defects."
    params = {**Msd.params,
              "conductivity.temperature_K": ("required", "temperature (K)"),
              "conductivity.charge_e": (0.0, "carrier charge (e)"),
              "conductivity.fit_range_ps": ("required", "[start, stop] of the linear MSD fit"),
              "conductivity.volume": ({"mode": "cell"}, "cell | slab (+ thickness_A, normal_axis)")}

    def __init__(self, cfg: dict[str, Any], run):
        if isinstance(cfg.get("selection"), dict) and "oxygen_species" in cfg["selection"]:
            raise ValueError("Fixed-carrier conductivity cannot use selection.oxygen_species because proton transfer "
                             "changes the carrier oxygen identity. Use defect-conductivity instead.")
        super().__init__(cfg, run)
        self.cond = dict(cfg.get("conductivity") or {})
        if self.cond.get("temperature_K") is None or float(self.cond["temperature_K"]) <= 0:
            raise ValueError("conductivity.temperature_K must be positive.")
        fit = self.cond.get("fit_range_ps")
        if not isinstance(fit, list) or len(fit) != 2:
            raise ValueError("conductivity.fit_range_ps is required and must be [start_ps, stop_ps].")
        self.fit = (float(fit[0]), float(fit[1]))

    def finalize(self) -> Result:
        msd_result = super().finalize()
        vol_cfg = self.cond.get("volume", {"mode": "cell"})
        if not isinstance(vol_cfg, dict):
            raise ValueError("conductivity.volume must be a mapping.")
        default_axis = self.msd.get("plane_normal_axis", "z") if str(vol_cfg.get("mode", "cell")) == "slab" else "z"
        volume, thickness = analysis_volume(np.asarray(self.vec, dtype=float), vol_cfg, default_axis)
        a = msd_result.arrays
        res = compute_nernst_einstein_conductivity(
            a["time_ps"], a["msd_a2"], carrier_count=msd_result.meta["selected_atoms"], volume_a3=volume,
            temperature_k=float(self.cond["temperature_K"]), charge_e=float(self.cond.get("charge_e", 0.0)),
            dimensions=2 if msd_result.meta["dimensionality"] == "2d" else 3, fit_range_ps=self.fit, sheet_thickness_a=thickness)
        self._res = res
        meta = {**msd_result.meta, "method": "Nernst-Einstein", "temperature_K": float(self.cond["temperature_K"]),
                "charge_e": float(self.cond.get("charge_e", 0.0)), "carrier_count": res.carrier_count, "volume_A3": res.volume_a3,
                "diffusion_m2_per_s": res.diffusion_m2_per_s, "conductivity_S_per_m": res.conductivity_s_per_m,
                "sheet_conductance_S": res.sheet_conductance_s}
        return Result(self.type, self.name, a, meta)

    def write(self, result: Result, outdir: Path) -> dict[str, Any]:
        from waterint.plot.conductivity import write_conductivity_csv, write_conductivity_msd_csv
        csv, msd_csv = outdir / f"{self.name}.csv", outdir / f"{self.name}_msd.csv"
        write_conductivity_csv(csv, self._res, temperature_k=float(self.cond["temperature_K"]),
                               charge_e=float(self.cond.get("charge_e", 0.0)), fit_range_ps=self.fit)
        write_conductivity_msd_csv(msd_csv, self._res)
        return {"csv": str(csv), "msd_csv": str(msd_csv)}

    def plot(self, result: Result, outdir: Path) -> dict[str, Any]:
        from waterint.plot.conductivity import plot_conductivity_msd
        path = outdir / f"{self.name}.png"
        out = self.run.output_cfg
        plot_conductivity_msd(path, self._res, title=str(out.get("title", "Nernst-Einstein conductivity MSD fit")),
                              dpi=int(out.get("dpi", 220)))
        return {"png": str(path)}
