"""Density profile along one coordinate for elements or acceptor species."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from waterint.analysis.base import Analysis, Result
from waterint.analysis.selectors import OWNERSHIP_KEYS, parse_range
from waterint.backends.registry import register, select
from waterint.io.cell import cross_section_area
from waterint.prep.labelled import LabelledFrame

DEFAULT_MASSES_AMU = {"O2-": 15.999, "OH-": 17.007, "H2O": 18.015, "H3O+": 19.023, "O_other": 15.999}
AMU_PER_A3_TO_G_PER_CM3 = 1.66053906660


@register("density.histogram", "python")
def _hist_python(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.histogram(values, bins=edges)[0].astype(float, copy=False)


@register("density.histogram", "cpu")
def _hist_cpu(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    from waterint.backends.cpu.native import density_histogram_edges
    return density_histogram_edges(values, bin_edges=edges)


class Density(Analysis):
    type = "density"
    v06_name = "density"
    description = "Project selected atoms or acceptor species onto one coordinate and write a density profile."
    params = {
        "selection.mode": ("element", "element | oxygen_species"),
        "selection.species": (None, "element list for mode element, e.g. [O]"),
        "selection.oxygen_species": ("all", "species labels for mode oxygen_species"),
        "coordinate": ("profile default", "axis / mode / reference (default: the profile reference plane)"),
        "coordinate.range": ("required", "[lo, hi] of the coordinate (A)"),
        "coordinate.bins": (200, "number of coordinate bins"),
        "normalization.type": ("number_density", "number_density | mass_density | counts_per_frame"),
        "backend": ("auto", "auto | python | cpu"),
    }

    def __init__(self, cfg: dict[str, Any], run):
        super().__init__(cfg, run)
        self.selection = dict(cfg.get("selection") or {"mode": "element", "species": ["O"]})
        coord_cfg = dict(cfg.get("coordinate") or {})
        self.coord = self.coordinate(coord_cfg)
        lo, hi = parse_range(coord_cfg.get("range"), "coordinate.range")
        bins = int(coord_cfg.get("bins", 200))
        if bins <= 0:
            raise ValueError("coordinate.bins must be > 0.")
        self.edges = np.linspace(lo, hi, bins + 1)
        self.normalization = cfg.get("normalization") or {}
        self.backend, self.hist = select("density.histogram", cfg.get("backend", "auto"))
        self.backends_used["density.histogram"] = self.backend
        mode = str(self.selection.get("mode", "element"))
        if mode == "element":
            species = self.selection.get("species")
            if not species or not isinstance(species, list):
                raise ValueError("selection.species must be a non-empty list, e.g. ['O'].")
            self.elements = tuple(str(s) for s in species)
            self.labels = [str(self.selection.get("label", "_".join(sorted(self.elements))))]
        elif mode == "oxygen_species":
            self.own = self.ownership_params({OWNERSHIP_KEYS[k]: v for k, v in self.selection.items() if k in OWNERSHIP_KEYS})
            requested = self.selection.get("oxygen_species", "all")
            all_labels = self.profile.species_labels
            if requested == "all":
                self.labels = list(self.profile.free_species) if not self.profile.solid_species else all_labels
            else:
                self.labels = [str(x) for x in requested]
                unknown = [l for l in self.labels if l not in all_labels]
                if unknown:
                    raise ValueError(f"Unknown oxygen species labels: {unknown} (profile has {all_labels}).")
        else:
            raise ValueError("selection.mode must be element or oxygen_species.")
        self.mode = mode
        self.counts = {label: np.zeros(bins, dtype=float) for label in self.labels}
        self.selected_total = {label: 0 for label in self.labels}

    def update(self, lf: LabelledFrame) -> None:
        if self.mode == "element":
            groups = {self.labels[0]: lf.indices(self.elements)}
        else:
            species = lf.species(self.own)
            groups = {label: species.get(label, np.empty(0, dtype=int)) for label in self.labels}
        for label, idx in groups.items():
            values = lf.coordinates(idx, self.coord)
            self.selected_total[label] += int(values.size)
            self.counts[label] += self.hist(values, self.edges)

    def finalize(self) -> Result:
        if self.frames == 0:
            raise ValueError("Cannot finalize density profile with zero frames.")
        if self.first_cell is None and self.first_cell_vectors is None:
            raise ValueError("No cell information was available. Set system.cell manually.")
        widths = np.diff(self.edges)
        if not np.allclose(widths, widths[0]):
            raise ValueError("Density normalization currently requires uniform bin spacing.")
        centers = 0.5 * (self.edges[:-1] + self.edges[1:])
        arrays: dict[str, np.ndarray] = {"bin_centers": centers, "bin_edges": self.edges}
        for i, label in enumerate(self.labels):
            arrays[f"counts_per_frame__{i}"] = self.counts[label] / self.frames
            arrays[f"density__{i}"] = self._normalize(self.counts[label], label, float(widths[0]))
        return Result(self.type, self.name, arrays, {
            "axis": self.coord.label, "profile_labels": self.labels, "selected_atoms_total": self.selected_total})

    def _normalize(self, counts: np.ndarray, label: str, bin_width: float) -> np.ndarray:
        cfg = self.normalization
        norm_type = str(cfg.get("type", "number_density"))
        unit = str(cfg.get("unit", "")).lower()
        if norm_type == "counts_per_frame":
            return counts / self.frames
        if norm_type not in {"number_density", "mass_density"}:
            raise ValueError("normalization.type must be number_density, mass_density, or counts_per_frame.")
        area = cross_section_area(cell=self.first_cell, cell_vectors=self.first_cell_vectors, axis=self.coord.axis)
        volume = area * bin_width
        if not math.isfinite(volume) or volume <= 0:
            raise ValueError("Computed slab volume must be positive.")
        number = counts / self.frames / volume
        if norm_type == "number_density" and unit not in {"g_cm3", "g/cm3", "g/cm^3"}:
            return number
        masses = cfg.get("masses_amu") or {}
        mass = masses.get(label, DEFAULT_MASSES_AMU.get(label, cfg.get("mass_amu")))
        if mass is None:
            raise ValueError(f"Need mass for profile {label!r}. Set normalization.mass_amu or normalization.masses_amu.")
        if float(mass) <= 0:
            raise ValueError("Profile mass must be positive.")
        return number * float(mass) * AMU_PER_A3_TO_G_PER_CM3

    # -- output ----------------------------------------------------------------------------------
    def _profiles(self, result: Result) -> dict[str, dict[str, np.ndarray]]:
        return {label: {"counts_per_frame": result.arrays[f"counts_per_frame__{i}"], "density": result.arrays[f"density__{i}"]}
                for i, label in enumerate(result.meta["profile_labels"])}

    def write(self, result: Result, outdir: Path) -> dict[str, Any]:
        from waterint.plot.density import write_density_csv
        path = outdir / f"{self.name}.csv"
        write_density_csv(path, result.arrays["bin_centers"], self._profiles(result), result.meta["axis"])
        return {"csv": str(path)}

    def plot(self, result: Result, outdir: Path) -> dict[str, Any]:
        from waterint.plot.density import density_ylabel, plot_density_profile
        path = outdir / f"{self.name}.png"
        out = self.run.output_cfg
        plot_density_profile(path=path, x=result.arrays["bin_centers"], y=self._profiles(result),
                             xlabel=f"{result.meta['axis']} coordinate (Angstrom)", ylabel=density_ylabel(self.normalization),
                             title=str(self.cfg.get("title", out.get("title", "Density profile"))))
        return {"png": str(path)}
