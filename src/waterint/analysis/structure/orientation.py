"""O-H orientation: 2D histogram of (coordinate of O, angle of an O-H vector to the axis) per acceptor species.

vector_mode: oh_bond (every O-H bond), oh_bisector / dipole (unit-vector sum of the O-H bonds).
Species are those of the profile free table with 1, 2 or 3 protons (OH-, H2O, H3O+ by default).
Kernel ``orientation.accumulate`` (python, cpu).
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from waterint.analysis.base import Analysis, Result
from waterint.analysis.selectors import OWNERSHIP_KEYS, parse_range
from waterint.backends.registry import register, select
from waterint.prep.labelled import LabelledFrame
from waterint.prep.ownership import Ownership


@register("orientation.accumulate", "cpu")
def _acc_cpu(positions, own: Ownership, *, vector_mode, axis, axis_sign, reference, angle_axis_sign, z_edges, angle_edges, hist):
    from waterint.backends.cpu.native import accumulate_oh_orientation_from_neighbors
    out = accumulate_oh_orientation_from_neighbors(
        positions[own.acceptor_idx], positions[own.proton_idx], own.counts, own.matrix, vector_mode=vector_mode,
        axis=axis, axis_sign=axis_sign, reference=reference, angle_axis_sign=angle_axis_sign,
        z_edges=z_edges, angle_edges=angle_edges, histograms=hist)
    bonds, samples = out
    return np.asarray(bonds, dtype=np.int64), np.asarray(samples, dtype=np.int64)


@register("orientation.accumulate", "python")
def _acc_python(positions, own: Ownership, *, vector_mode, axis, axis_sign, reference, angle_axis_sign, z_edges, angle_edges, hist):
    bonds = np.zeros(3, dtype=np.int64)
    samples = np.zeros(3, dtype=np.int64)
    rows = own.proton_lists()
    for k, nh in enumerate((1, 2, 3)):
        sel = np.nonzero(own.counts == nh)[0]
        if sel.size == 0:
            continue
        z_vals: list[np.ndarray] = []
        a_vals: list[np.ndarray] = []
        for i in sel:
            o = positions[own.acceptor_idx[i]]
            vec = positions[own.proton_idx[rows[i]]] - o
            norms = np.linalg.norm(vec, axis=1)
            ok = norms > 0
            if vector_mode == "oh_bond":
                cos = np.clip(angle_axis_sign * vec[ok, axis] / norms[ok], -1.0, 1.0)
                a_vals.append(np.degrees(np.arccos(cos)))
                z_vals.append(np.full(cos.size, axis_sign * (o[axis] - reference)))
                bonds[k] += int(ok.sum())
            else:
                if not np.any(ok):
                    continue
                d = np.sum(vec[ok] / norms[ok, None], axis=0)
                dn = float(np.linalg.norm(d))
                if dn <= 0:
                    continue
                cos = float(np.clip(angle_axis_sign * d[axis] / dn, -1.0, 1.0))
                a_vals.append(np.array([math.degrees(math.acos(cos))]))
                z_vals.append(np.array([axis_sign * (o[axis] - reference)]))
                bonds[k] += int(ok.sum())
        if z_vals:
            z = np.concatenate(z_vals)
            a = np.concatenate(a_vals)
            hist[k] += np.histogram2d(z, a, bins=[z_edges, angle_edges])[0]
            samples[k] += z.size
    return bonds, samples


class Orientation(Analysis):
    type = "orientation"
    v06_name = "oh-orientation"
    description = "Accumulate O-H angle distributions along a selected coordinate."
    params = {
        "selection.oxygen_species": ("species with 1-3 protons", "subset of the profile labels for nH = 1, 2, 3"),
        "coordinate": ("profile default", "axis / mode / reference (default: the profile reference plane)"),
        "coordinate.range": ("required", "[lo, hi] of the coordinate (A)"),
        "coordinate.bins": (200, "number of coordinate bins"),
        "angle.range": ([0.0, 180.0], "angle range (deg)"), "angle.bins": (180, "angle bins"),
        "angle.vector_mode": ("oh_bond", "oh_bond | oh_bisector | dipole"),
        "angle.axis_sign": (1.0, "+1 angle to +axis, -1 to -axis"),
        "normalization.type": ("counts_per_frame", "counts_per_frame | number_density"),
        "backend": ("auto", "auto | python | cpu"),
    }

    def __init__(self, cfg: dict[str, Any], run):
        super().__init__(cfg, run)
        selection = dict(cfg.get("selection") or {})
        coord_cfg = dict(cfg.get("coordinate") or {})
        angle_cfg = dict(cfg.get("angle") or {})
        self.own = self.ownership_params({OWNERSHIP_KEYS[k]: v for k, v in selection.items() if k in OWNERSHIP_KEYS})
        self.coord = self.coordinate(coord_cfg)
        zlo, zhi = parse_range(coord_cfg.get("range"), "coordinate.range")
        zb = int(coord_cfg.get("bins", 200))
        alo, ahi = parse_range(angle_cfg.get("range", [0.0, 180.0]), "angle.range")
        ab = int(angle_cfg.get("bins", 180))
        if zb <= 0 or ab <= 0:
            raise ValueError("coordinate.bins and angle.bins must be > 0.")
        self.z_edges = np.linspace(zlo, zhi, zb + 1)
        self.angle_edges = np.linspace(alo, ahi, ab + 1)
        self.vector_mode = str(angle_cfg.get("vector_mode", "oh_bond")).lower()
        if self.vector_mode not in {"oh_bond", "oh_bisector", "dipole"}:
            raise ValueError("angle.vector_mode must be oh_bond, oh_bisector, or dipole.")
        self.angle_axis_sign = float(angle_cfg.get("axis_sign", 1.0))
        free = self.profile.free_species
        self.label_by_nh = {nh: free[nh] for nh in (1, 2, 3) if nh < len(free) - 1}  # last entry = "nH or more"
        requested = selection.get("oxygen_species", "all")
        self.labels = list(self.label_by_nh.values()) if requested == "all" else [str(x) for x in requested]
        unknown = [l for l in self.labels if l not in self.label_by_nh.values()]
        if unknown:
            raise ValueError(f"Unknown angle-z oxygen species labels: {unknown} (allowed: {list(self.label_by_nh.values())}).")
        self.slot = {label: nh - 1 for nh, label in self.label_by_nh.items()}
        self.normalization = cfg.get("normalization") or {}
        backend = cfg.get("backend", angle_cfg.get("backend", "auto"))
        self.backend, self.kernel = select("orientation.accumulate", backend)
        self.backends_used["orientation.accumulate"] = self.backend
        self.hist = np.zeros((3, zb, ab), dtype=float)
        self.bonds = np.zeros(3, dtype=np.int64)
        self.samples = np.zeros(3, dtype=np.int64)

    def update(self, lf: LabelledFrame) -> None:
        own = lf.ownership(self.own)
        if own.acceptor_idx.size == 0 or own.proton_idx.size == 0:
            return
        bonds, samples = self.kernel(lf.positions, own, vector_mode=self.vector_mode, axis=self.coord.axis,
                                     axis_sign=self.coord.sign, reference=lf.reference(self.coord),
                                     angle_axis_sign=self.angle_axis_sign, z_edges=self.z_edges,
                                     angle_edges=self.angle_edges, hist=self.hist)
        self.bonds += bonds
        self.samples += samples

    def finalize(self) -> Result:
        if self.frames == 0:
            raise ValueError("Cannot finalize angle-z histogram with zero frames.")
        zw, aw = np.diff(self.z_edges), np.diff(self.angle_edges)
        if not np.allclose(zw, zw[0]) or not np.allclose(aw, aw[0]):
            raise ValueError("Angle-z normalization currently requires uniform bins.")
        arrays = {"z_centers": 0.5 * (self.z_edges[:-1] + self.z_edges[1:]),
                  "angle_centers": 0.5 * (self.angle_edges[:-1] + self.angle_edges[1:])}
        for i, label in enumerate(self.labels):
            arrays[f"hist__{i}"] = self._normalize(self.hist[self.slot[label]], float(zw[0]), float(aw[0]))
        return Result(self.type, self.name, arrays, {
            "axis": self.coord.label, "species_labels": self.labels, "vector_mode": self.vector_mode,
            "bond_counts_total": {l: int(self.bonds[self.slot[l]]) for l in self.labels},
            "sample_counts_total": {l: int(self.samples[self.slot[l]]) for l in self.labels}})

    def _normalize(self, counts: np.ndarray, zw: float, aw: float) -> np.ndarray:
        norm_type = str(self.normalization.get("type", "counts_per_frame"))
        per_frame = counts / self.frames
        if norm_type == "counts_per_frame":
            return per_frame
        if norm_type != "number_density":
            raise ValueError("normalization.type must be counts_per_frame or number_density.")
        lengths = [self.first_cell[i] for i in range(3) if i != self.coord.axis]
        volume = lengths[0] * lengths[1] * zw * aw
        if not math.isfinite(volume) or volume <= 0:
            raise ValueError("Computed z-angle normalization volume must be positive.")
        return per_frame / volume

    def write(self, result: Result, outdir: Path) -> dict[str, Any]:
        from waterint.plot.orientation import write_hist_csv
        paths = {}
        for i, label in enumerate(result.meta["species_labels"]):
            path = outdir / f"{self.name}_{safe_label(label)}.csv"
            write_hist_csv(path, result.arrays["z_centers"], result.arrays["angle_centers"], result.arrays[f"hist__{i}"],
                           result.meta["axis"])
            paths[label] = str(path)
        return {"csv": paths}

    def plot(self, result: Result, outdir: Path) -> dict[str, Any]:
        from waterint.plot.orientation import plot_angle_label, plot_angle_z_histogram, value_label, vector_label
        out = {**self.run.output_cfg, **(self.cfg.get("plot") or {})}
        paths = {}
        axis, vm = result.meta["axis"], result.meta["vector_mode"]
        opt = lambda k: None if out.get(k) in (None, "") else float(out[k])
        for i, label in enumerate(result.meta["species_labels"]):
            path = outdir / f"{self.name}_{safe_label(label)}.png"
            plot_angle_z_histogram(
                path=path, z_centers=result.arrays["z_centers"], angle_centers=result.arrays["angle_centers"],
                hist=result.arrays[f"hist__{i}"], title=f"{label}: {out.get('title', f'{vector_label(vm)} angle vs {axis}')}",
                z_label=str(out.get("z_label", f"{axis} coordinate (Angstrom)")), value_label=value_label(self.normalization, vector_mode=vm),
                angle_label=str(out.get("angle_label", plot_angle_label(vm))), log=bool(out.get("log", False)),
                cmap=str(out.get("cmap", "turbo")), style=str(out.get("style", "contour")),
                orientation=str(out.get("orientation", "angle_z")), invert_angle_axis=bool(out.get("invert_angle_axis", False)),
                figure_preset=str(out.get("figure_preset", "default")), colorbar_mode=str(out.get("colorbar_mode", "auto")),
                colormap_style=str(out.get("colormap_style", "turbo")), yaxis_side=str(out.get("yaxis_side", "left")),
                show_y_label=bool(out.get("show_y_label", True)), display_z_max=opt("display_z_max"),
                mask_threshold=opt("mask_threshold"), colorbar_height=float(out.get("colorbar_height", 0.36)),
                colorbar_width_px=float(out.get("colorbar_width_px", 30.0)), colorbar_x_px=float(out.get("colorbar_x_px", 900.0)),
                colorbar_center=float(out.get("colorbar_center", 0.50)), colorbar_top_pad_px=float(out.get("colorbar_top_pad_px", 38.0)),
                colorbar_top_height_px=float(out.get("colorbar_top_height_px", 34.0)),
                colorbar_tick_size=float(out.get("colorbar_tick_size", 9.0)), colorbar_title_size=float(out.get("colorbar_title_size", 10.0)),
                hide_first_colorbar_tick_label=bool(out.get("hide_first_colorbar_tick_label", False)),
                smooth_sigma=float(out.get("smooth_sigma", 0.8)), log_vmin=opt("log_vmin"), log_vmax=opt("log_vmax"),
                dpi=int(out.get("dpi", 220)))
            paths[label] = str(path)
        return {"png": paths}


def safe_label(label: str) -> str:
    return label.lower().replace("+", "plus").replace("-", "minus").replace(" ", "_")
