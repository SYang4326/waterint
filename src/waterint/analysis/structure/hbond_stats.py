"""H-bond topology statistics (e.g. DDAA, DAA) per acceptor species.

Bonds come from prep.hbonds (one geometric definition); this analysis only classifies and counts.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from waterint.analysis.base import Analysis, Result
from waterint.analysis.selectors import OWNERSHIP_KEYS
from waterint.prep.hbonds import HBondParams
from waterint.prep.labelled import LabelledFrame

DEFAULT_CLASSES_BY_NH = {
    1: ("DAAA", "DAA", "DA", "AAA", "AA", "A", "other"),
    2: ("DDAA", "DDA", "DAA", "DA", "AA", "A", "other"),
    3: ("DDDA", "DDD", "DDA", "DD", "DA", "D", "other"),
}


def topology_label(donors: int, acceptors: int) -> str:
    if donors <= 0 and acceptors <= 0:
        return "none"
    return "D" * max(donors, 0) + "A" * max(acceptors, 0)


class HBondStats(Analysis):
    type = "hbond"
    v06_name = "hbond"
    description = "Classify hydrogen-bond topology by acceptor species."
    params = {
        "selection.oxygen_species": ("species with 1-3 protons", "subset of profile labels for nH = 1, 2, 3"),
        "hbond.oo_cutoff": (3.5, "donor-acceptor distance cutoff (A)"),
        "hbond.dha_angle_min": (150.0, "minimum D-H...A angle (deg)"),
        "hbond.h_acceptor_cutoff": (None, "optional H...A cutoff (A)"),
        "hbond.pbc": ([True, True, True], "periodic axes for O-O search"),
        "hbond.max_acceptors_per_hydrogen": (True, "each H donates at most one bond"),
        "hbond.classes_by_species": ("defaults", "topology classes shown per species; others -> 'other'"),
        "backend": ("auto", "prep backend for ownership + H-bond geometry"),
    }

    def __init__(self, cfg: dict[str, Any], run):
        super().__init__(cfg, run)
        selection = dict(cfg.get("selection") or {})
        hb_cfg = dict(cfg.get("hbond") or {})
        self.own = self.ownership_params({OWNERSHIP_KEYS[k]: v for k, v in selection.items() if k in OWNERSHIP_KEYS})
        self.hb = HBondParams.from_config(hb_cfg)
        free = self.profile.free_species
        self.label_by_nh = {nh: free[nh] for nh in (1, 2, 3) if nh < len(free) - 1}
        self.nh_by_label = {l: nh for nh, l in self.label_by_nh.items()}
        requested = selection.get("oxygen_species", "all")
        self.labels = list(self.label_by_nh.values()) if requested == "all" else [str(x) for x in requested]
        unknown = [l for l in self.labels if l not in self.nh_by_label]
        if unknown:
            raise ValueError(f"Unknown H-bond oxygen species labels: {unknown}")
        raw = hb_cfg.get("classes_by_species", hb_cfg.get("classes"))
        self.classes: dict[str, list[str]] = {}
        for label in self.labels:
            default = list(DEFAULT_CLASSES_BY_NH[self.nh_by_label[label]])
            if raw is None:
                cls = default
            elif isinstance(raw, list):
                cls = [str(x) for x in raw]
            elif isinstance(raw, dict):
                cls = [str(x) for x in raw.get(label, default)]
            else:
                raise ValueError("hbond.classes_by_species must be a mapping from species label to class list.")
            if not cls or len(set(cls)) != len(cls):
                raise ValueError(f"hbond classes for {label} must be a non-empty list without duplicates.")
            self.classes[label] = cls
        self.counts = {l: {c: 0 for c in self.classes[l]} for l in self.labels}
        self.raw = {l: {} for l in self.labels}
        self.samples = {l: 0 for l in self.labels}

    def update(self, lf: LabelledFrame) -> None:
        own = lf.ownership(self.own)
        if own.acceptor_idx.size == 0 or own.proton_idx.size == 0:
            return
        donors, acceptors = lf.hbond_counts(self.hb, self.own)
        for nh_val, d, a in zip(own.counts, donors, acceptors):
            label = self.label_by_nh.get(int(nh_val))
            if label not in self.counts:
                continue
            self.samples[label] += 1
            topo = topology_label(int(d), int(a))
            self.raw[label][topo] = self.raw[label].get(topo, 0) + 1
            grouped = topo if topo in self.counts[label] else ("other" if "other" in self.counts[label] else None)
            if grouped is not None:
                self.counts[label][grouped] += 1

    def finalize(self) -> Result:
        fractions = {l: {c: (self.counts[l][c] / self.samples[l] if self.samples[l] else 0.0) for c in self.classes[l]}
                     for l in self.labels}
        arrays = {f"counts__{i}": np.array([self.counts[l][c] for c in self.classes[l]], dtype=np.int64)
                  for i, l in enumerate(self.labels)}
        return Result(self.type, self.name, arrays, {
            "species_labels": self.labels, "classes": self.classes, "counts": self.counts, "raw_counts": self.raw,
            "fractions": fractions, "samples_total": self.samples})

    def write(self, result: Result, outdir: Path) -> dict[str, Any]:
        from waterint.plot.hbond import write_hbond_csv, write_raw_hbond_csv
        m = result.meta
        csv, raw = outdir / f"{self.name}.csv", outdir / f"{self.name}_raw.csv"
        write_hbond_csv(csv, m["counts"], m["fractions"], m["samples_total"], m["classes"])
        write_raw_hbond_csv(raw, m["raw_counts"], m["counts"], m["samples_total"], m["classes"])
        return {"csv": str(csv), "raw_csv": str(raw)}

    def plot(self, result: Result, outdir: Path) -> dict[str, Any]:
        from waterint.plot.hbond import plot_hbond_fractions
        m = result.meta
        out = self.run.output_cfg
        show = [s for s in m["species_labels"] if m["samples_total"][s] > 0 or bool(out.get("show_empty_species", False))]
        show = show or m["species_labels"]
        path = outdir / f"{self.name}.png"
        plot_hbond_fractions(path=path, fractions={s: m["fractions"][s] for s in show}, classes={s: m["classes"][s] for s in show},
                             title=str(out.get("title", "H-bond topology fractions")), ylabel=str(out.get("ylabel", "Fraction")),
                             dpi=int(out.get("dpi", 220)))
        return {"png": str(path)}
