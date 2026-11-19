"""Radial distribution functions for element-, type-, species- and layer-selective pairs."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from waterint.analysis.base import Analysis, Result
from waterint.analysis.selectors import select as select_atoms
from waterint.backends.registry import register, select
from waterint.io.cell import cell_volume, minimum_image
from waterint.prep.labelled import LabelledFrame


@register("rdf.histogram", "cpu")
def _rdf_cpu(positions, first, second, *, r_max, n_bins, cell_vectors, pbc, same_selection):
    from waterint.backends.cpu.native import rdf_histogram
    return rdf_histogram(positions, first, second, r_max=r_max, n_bins=n_bins, cell_vectors=cell_vectors, pbc=pbc,
                         same_selection=same_selection)


@register("rdf.histogram", "python")
def _rdf_python(positions, first, second, *, r_max, n_bins, cell_vectors, pbc, same_selection):
    delta = positions[second][None, :, :] - positions[first][:, None, :]
    if cell_vectors is not None and any(pbc):
        delta = minimum_image(delta, cell_vectors=cell_vectors, pbc=pbc)
    dist = np.linalg.norm(delta, axis=-1)
    if same_selection:
        dist = dist[np.triu_indices(first.size, k=1)]
    else:
        dist = dist[first[:, None] != second[None, :]]
    return np.histogram(dist, bins=n_bins, range=(0.0, r_max))[0].astype(float)


@dataclass
class _Pair:
    name: str
    first: dict[str, Any]
    second: dict[str, Any]
    same: bool
    counts: np.ndarray
    expected: np.ndarray


class Rdf(Analysis):
    type = "rdf"
    v06_name = "rdf"
    description = "Compute element-, type-, species-, and layer-selective radial distribution functions."
    params = {
        "rdf.r_max": (8.0, "maximum distance (A)"), "rdf.bins": (200, "number of bins"),
        "rdf.pbc": ([True, True, True], "periodic axes for distances"),
        "rdf.pairs": ("O-O and O-H", "list of {name, first: selector, second: selector}"),
        "backend": ("auto", "auto | python | cpu"),
    }

    def __init__(self, cfg: dict[str, Any], run):
        super().__init__(cfg, run)
        rdf = dict(cfg.get("rdf") or {})
        self.r_max = float(rdf.get("r_max", 8.0))
        bins = int(rdf.get("bins", 200))
        if self.r_max <= 0 or bins <= 0:
            raise ValueError("rdf.r_max and rdf.bins must be positive.")
        self.edges = np.linspace(0.0, self.r_max, bins + 1)
        pbc = rdf.get("pbc", [True, True, True])
        self.pbc = (pbc,) * 3 if isinstance(pbc, bool) else tuple(bool(v) for v in pbc)
        raw = rdf.get("pairs") or [{"name": "O-O", "first": {"elements": ["O"]}, "second": {"elements": ["O"]}},
                                   {"name": "O-H", "first": {"elements": ["O"]}, "second": {"elements": ["H"]}}]
        self.pairs = []
        for item in raw:
            if not isinstance(item, dict) or not item.get("name") or not isinstance(item.get("first"), dict) \
                    or not isinstance(item.get("second"), dict):
                raise ValueError("Each RDF pair needs name, first, and second mappings.")
            self.pairs.append(_Pair(str(item["name"]), item["first"], item["second"], item["first"] == item["second"],
                                    np.zeros(bins), np.zeros(bins)))
        self.base_own = self.ownership_params()
        self.backend, self.kernel = select("rdf.histogram", cfg.get("backend", rdf.get("backend", "auto")))
        self.backends_used["rdf.histogram"] = self.backend

    def update(self, lf: LabelledFrame) -> None:
        vectors = lf.cell_vectors
        volume = cell_volume(cell_vectors=vectors)
        shells = 4.0 * np.pi / 3.0 * (self.edges[1:] ** 3 - self.edges[:-1] ** 3)
        for p in self.pairs:
            first = select_atoms(lf, p.first, self.base_own)
            second = select_atoms(lf, p.second, self.base_own)
            if p.same and not np.array_equal(first, second):
                p.same = False
            overlap = int(np.intersect1d(first, second, assume_unique=True).size)
            if first.size and second.size:
                p.counts += self.kernel(lf.positions, first, second, r_max=self.r_max, n_bins=p.counts.size,
                                        cell_vectors=vectors, pbc=self.pbc, same_selection=p.same)
            n_pairs = first.size * (first.size - 1) / 2.0 if p.same else first.size * second.size - overlap
            p.expected += n_pairs * shells / volume

    def finalize(self) -> Result:
        if self.frames == 0:
            raise ValueError("Cannot finalize RDF with zero frames.")
        arrays = {"r_centers": 0.5 * (self.edges[:-1] + self.edges[1:])}
        for i, p in enumerate(self.pairs):
            arrays[f"g_r__{i}"] = np.divide(p.counts, p.expected, out=np.zeros_like(p.counts), where=p.expected > 0)
            arrays[f"counts__{i}"] = p.counts
            arrays[f"expected__{i}"] = p.expected
        return Result(self.type, self.name, arrays, {"pairs": [p.name for p in self.pairs]})

    def _pair_views(self, result: Result) -> dict[str, Any]:
        from types import SimpleNamespace
        a = result.arrays
        return {name: SimpleNamespace(r_centers=a["r_centers"], g_r=a[f"g_r__{i}"], counts=a[f"counts__{i}"],
                                      expected_counts=a[f"expected__{i}"]) for i, name in enumerate(result.meta["pairs"])}

    def write(self, result: Result, outdir: Path) -> dict[str, Any]:
        from waterint.plot.rdf import write_rdf_csv
        paths = {}
        for name, view in self._pair_views(result).items():
            path = outdir / f"{self.name}_{name.lower().replace(' ', '_').replace('-', '_')}.csv"
            write_rdf_csv(path, view)
            paths[name] = str(path)
        return {"csv": paths}

    def plot(self, result: Result, outdir: Path) -> dict[str, Any]:
        from waterint.plot.rdf import plot_rdf
        out = self.run.output_cfg
        mode = str(out.get("plot_mode", "combined")).lower()
        views = self._pair_views(result)
        paths: dict[str, Any] = {}
        if mode in {"combined", "both"}:
            path = outdir / f"{self.name}.png"
            plot_rdf(path, views, title=str(out.get("title", "Radial distribution functions")), dpi=int(out.get("dpi", 220)))
            paths["combined"] = str(path)
        if mode in {"separate", "both"}:
            for name, view in views.items():
                path = outdir / f"{self.name}_{name.lower().replace(' ', '_').replace('-', '_')}.png"
                plot_rdf(path, {name: view}, title=name, dpi=int(out.get("dpi", 220)))
                paths[name] = str(path)
        return {"png": paths}
