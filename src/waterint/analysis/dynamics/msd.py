"""2D/3D multiple-time-origin MSD of a fixed atom selection (chosen at the first frame, optional layer)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from waterint.analysis.base import Analysis, Result
from waterint.analysis.selectors import parse_range
from waterint.analysis.selectors import select as select_atoms
from waterint.backends.registry import register, select
from waterint.io.cell import minimum_image
from waterint.prep.labelled import LabelledFrame
from waterint.prep.reference import coordinate_spec_from_config


@register("msd.sums", "cpu")
def _msd_cpu(positions, *, cell_vectors, pbc, max_lag, origin_stride, dimensions, excluded_axis):
    from waterint.backends.cpu.native import msd_sums
    return msd_sums(positions, cell_vectors=cell_vectors, pbc=pbc, max_lag=max_lag, origin_stride=origin_stride,
                    dimensions=dimensions, excluded_axis=excluded_axis)


@register("msd.sums", "python")
def _msd_python(positions, *, cell_vectors, pbc, max_lag, origin_stride, dimensions, excluded_axis):
    unwrapped = np.empty_like(positions)
    unwrapped[0] = positions[0]
    for f in range(1, positions.shape[0]):
        unwrapped[f] = unwrapped[f - 1] + minimum_image(positions[f] - positions[f - 1], cell_vectors=cell_vectors[f], pbc=pbc)
    sums = np.zeros(max_lag + 1)
    samples = np.zeros(max_lag + 1, dtype=np.int64)
    for lag in range(max_lag + 1):
        origins = np.arange(0, positions.shape[0] - lag, origin_stride)
        d = unwrapped[origins + lag] - unwrapped[origins]
        if dimensions == 2:
            d[..., excluded_axis] = 0.0
        sums[lag] = float(np.sum(d * d))
        samples[lag] = d.shape[0] * d.shape[1]
    return sums, samples


class Msd(Analysis):
    type = "msd"
    v06_name = "msd"
    description = "Compute 2D or 3D multiple-time-origin MSD for a fixed stable-atom selection."
    params = {
        "selection": ("required", "selector (elements/types/oxygen_species), chosen at the first frame"),
        "msd.timestep_ps": ("required", "time between analysed frames (ps)"),
        "msd.dimensionality": ("3d", "2d | 3d"), "msd.plane_normal_axis": ("z", "excluded axis for 2d"),
        "msd.max_lag_frames": (None, "maximum lag (frames)"), "msd.origin_stride": (1, "origin stride (frames)"),
        "msd.layer": (None, "{coordinate, range}: keep atoms in this layer at the first frame"),
        "msd.pbc": ([True, True, True], "periodic axes for unwrapping"),
        "backend": ("auto", "auto | python | cpu"),
    }

    def __init__(self, cfg: dict[str, Any], run):
        super().__init__(cfg, run)
        self.msd = dict(cfg.get("msd") or {})
        if "timestep_ps" not in self.msd or float(self.msd["timestep_ps"]) <= 0:
            raise ValueError("msd.timestep_ps must be positive.")
        self.selection = cfg.get("selection")
        if not isinstance(self.selection, dict):
            raise ValueError("Config section 'selection' is required and must be a mapping.")
        pbc = self.msd.get("pbc", [True, True, True])
        self.pbc = (pbc,) * 3 if isinstance(pbc, bool) else tuple(bool(v) for v in pbc)
        dim = self.msd.get("dimensionality", "3d")
        self.dim = {2: "2d", 3: "3d"}.get(dim, str(dim).lower())
        if self.dim not in {"2d", "3d"}:
            raise ValueError("msd.dimensionality must be 2d or 3d.")
        label = str(self.msd.get("plane_normal_axis", "z")).lower().lstrip("-")
        if label not in {"x", "y", "z"}:
            raise ValueError("msd.plane_normal_axis must be x, y, or z.")
        self.normal = "xyz".index(label)
        self.backend, self.kernel = select("msd.sums", cfg.get("backend", self.msd.get("backend", "auto")))
        self.backends_used["msd.sums"] = self.backend
        self.selected: np.ndarray | None = None
        self.first_symbols: np.ndarray | None = None
        self.first_index = 0
        self.pos: list[np.ndarray] = []
        self.vec: list[np.ndarray] = []

    def update(self, lf: LabelledFrame) -> None:
        if self.selected is None:
            self.first_index = lf.index
            self.first_symbols = np.asarray(lf.symbols).copy()
            selected = select_atoms(lf, self.selection, self.ownership_params())
            layer = self.msd.get("layer")
            if layer is not None:
                if not isinstance(layer, dict):
                    raise ValueError("msd.layer must be a mapping when provided.")
                spec = coordinate_spec_from_config(layer.get("coordinate", layer))
                lo, hi = parse_range(layer.get("range"), "msd.layer.range")
                values = lf.coordinates(selected, spec)
                selected = selected[(values >= lo) & (values < hi)]
            if selected.size == 0:
                raise ValueError("MSD selection found no atoms after applying selection/layer criteria.")
            self.selected = selected
        elif not np.array_equal(np.asarray(lf.symbols), self.first_symbols):
            # elements, not raw type codes: relabelled trajectories change O type codes with the species
            raise ValueError("MSD requires fixed atom ordering (the same element sequence in every frame).")
        self.pos.append(lf.positions[self.selected])
        self.vec.append(lf.cell_vectors)

    def finalize(self) -> Result:
        if self.selected is None or len(self.pos) < 2:
            raise ValueError("MSD requires at least two trajectory frames.")
        positions = np.stack(self.pos)
        vectors = np.stack(self.vec)
        raw = self.msd.get("max_lag_frames")
        max_lag = positions.shape[0] - 1 if raw in {None, "", 0, "all"} else int(raw)
        if not 1 <= max_lag < positions.shape[0]:
            raise ValueError("msd.max_lag_frames must be between 1 and n_frames - 1.")
        stride = int(self.msd.get("origin_stride", 1))
        if stride <= 0:
            raise ValueError("msd.origin_stride must be positive.")
        sums, samples = self.kernel(positions, cell_vectors=vectors, pbc=self.pbc, max_lag=max_lag, origin_stride=stride,
                                    dimensions=2 if self.dim == "2d" else 3, excluded_axis=self.normal)
        lag = np.arange(max_lag + 1, dtype=int)
        msd = np.divide(sums, samples, out=np.zeros_like(sums), where=samples > 0)
        return Result(self.type, self.name, {"lag_frames": lag, "time_ps": lag * float(self.msd["timestep_ps"]),
                                             "msd_a2": msd, "samples": samples},
                      {"selected_atoms": int(self.selected.size), "selection_frame": int(self.first_index),
                       "dimensionality": self.dim})

    def write(self, result: Result, outdir: Path) -> dict[str, Any]:
        from waterint.plot.msd import write_msd_csv
        a = result.arrays
        path = outdir / f"{self.name}.csv"
        write_msd_csv(path, a["time_ps"], a["lag_frames"], a["msd_a2"], a["samples"])
        return {"csv": str(path)}

    def plot(self, result: Result, outdir: Path) -> dict[str, Any]:
        from waterint.plot.msd import plot_msd
        out = self.run.output_cfg
        path = outdir / f"{self.name}.png"
        plot_msd(path, result.arrays["time_ps"], result.arrays["msd_a2"], title=str(out.get("title", "Mean-squared displacement")),
                 dimensionality=result.meta["dimensionality"], dpi=int(out.get("dpi", 220)))
        return {"png": str(path)}
