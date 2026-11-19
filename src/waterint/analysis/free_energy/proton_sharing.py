"""Proton-sharing free-energy surfaces.

proton-sharing        F(delta, R_OO) and the shared-proton local-coordinate F(s, rho) for donor/acceptor species in
                      chosen layers (optionally including the exchanged state)
proton-sharing-hbond  H-bond-filtered, CN-resolved F(delta, R_OO) for same-layer (L1-L1) and L1-L2 pairs
Proton ownership comes from prep (nearest acceptor within the configured cutoff).
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from waterint.analysis.base import Analysis, Result
from waterint.analysis.free_energy.proton_sharing_core import (ProtonSharingResult, free_energy_from_probability,
                                                                normalized_probability, proton_sharing_surface)
from waterint.analysis.free_energy.proton_sharing_hbond_core import (accumulate_python_pairs, free_energy_from_counts,
                                                                      new_proton_sharing_hbond_state)
from waterint.analysis.selectors import parse_range
from waterint.backends.registry import normalize, register, select
from waterint.prep.labelled import LabelledFrame
from waterint.prep.reference import coordinate_spec_from_config
from waterint.system.profile import OwnershipParams

SPECIES_NH = {"OH-": 1, "H2O": 2}


def _pbc(value: Any) -> tuple[bool, bool, bool]:
    return (value,) * 3 if isinstance(value, bool) else tuple(bool(v) for v in value)  # type: ignore[return-value]


def _own(selection: dict[str, Any], cutoff_key: str, default_cutoff: float, default_pbc) -> OwnershipParams:
    cutoff = float(selection.get(cutoff_key, default_cutoff))
    return OwnershipParams(protons=(str(selection.get("hydrogen_symbol", "H")),),
                           acceptors=(str(selection.get("oxygen_symbol", "O")),),
                           cutoff=cutoff if cutoff > 0 else math.inf, mode="nearest",
                           pbc=_pbc(selection.get("pbc", default_pbc)))


class ProtonSharing(Analysis):
    type = "proton-sharing"
    v06_name = "proton-sharing"
    description = "Accumulate F(delta, R_OO) and a shared-proton local-coordinate F(s, rho) for selected donor-acceptor layers."
    params = {"proton_sharing.donor / acceptor": ("required", "{species: OH- | H2O, range: [lo, hi]} on the coordinate"),
              "proton_sharing.coordinate": ("required", "coordinate spec for the layer ranges"),
              "proton_sharing.temperature_K": ("required", "temperature for F = -kT ln P (K)"),
              "proton_sharing.include_swapped_state": (True, "also count the donor/acceptor-swapped pairs (delta -> -delta)"),
              "selection.oh_cutoff": (1.25, "ownership cutoff (A); <= 0: nearest O without cutoff"),
              "selection.pbc": ([True, True, True], "periodic axes")}

    def __init__(self, cfg: dict[str, Any], run):
        super().__init__(cfg, run)
        self.sel = dict(cfg.get("selection") or {})
        self.sc = dict(cfg.get("proton_sharing") or {})
        for key in ("donor", "acceptor", "coordinate"):
            if not isinstance(self.sc.get(key), dict):
                raise ValueError(f"proton_sharing.{key} is required and must be a mapping.")
        self.swapped = bool(self.sc.get("include_swapped_state", True))
        if self.swapped:
            for key in ("delta_range_A", "shared_s_range_A"):
                lo, hi = parse_range(self.sc.get(key, [-1.5, 1.5]), f"proton_sharing.{key}")
                if not np.isclose(lo, -hi):
                    raise ValueError(f"proton_sharing.{key} must be symmetric about zero when include_swapped_state is true.")
        self.own = _own(self.sel, "oh_cutoff", 1.25, [True, True, True])
        self.pbc = _pbc(self.sel.get("pbc", [True, True, True]))
        self.coord = coordinate_spec_from_config(self.sc["coordinate"])
        self.pairs = self.shared = None
        self.pair_samples = self.shared_samples = 0

    def _mask(self, coordination, z, cfg) -> np.ndarray:
        expected = SPECIES_NH.get(str(cfg["species"]))
        if expected is None:
            raise ValueError("proton_sharing donor/acceptor species must be OH- or H2O.")
        lo, hi = parse_range(cfg["range"], "proton_sharing donor/acceptor range")
        return (coordination == expected) & (z >= lo) & (z < hi)

    def _surface(self, donors, acceptors, hydrogens, cell) -> ProtonSharingResult:
        c = self.sc
        return proton_sharing_surface(
            donor_positions=donors, acceptor_positions=acceptors, donor_hydrogens=hydrogens, cell_vectors=cell, pbc=self.pbc,
            oo_range_a=parse_range(c.get("oo_range_A", [2.3, 3.6]), "oo_range_A"),
            delta_range_a=parse_range(c.get("delta_range_A", [-1.5, 1.5]), "delta_range_A"),
            delta_bins=int(c.get("delta_bins", 120)), oo_bins=int(c.get("oo_bins", 130)),
            shared_delta_max_a=float(c.get("shared_delta_max_A", 0.15)),
            shared_s_range_a=parse_range(c.get("shared_s_range_A", [-1.5, 1.5]), "shared_s_range_A"),
            shared_rho_range_a=parse_range(c.get("shared_rho_range_A", [0.0, 1.5]), "shared_rho_range_A"),
            shared_s_bins=int(c.get("shared_s_bins", 120)), shared_rho_bins=int(c.get("shared_rho_bins", 75)),
            temperature_k=float(c["temperature_K"]))

    def update(self, lf: LabelledFrame) -> None:
        own = lf.ownership(self.own)
        rows = own.proton_lists()
        coordination = own.counts.astype(int)
        z = lf.coordinates(own.acceptor_idx, self.coord)
        opos, hpos = lf.positions[own.acceptor_idx], lf.positions[own.proton_idx]
        cell = lf.cell_vectors
        donor_cfg, acceptor_cfg = self.sc["donor"], self.sc["acceptor"]
        dm, am = self._mask(coordination, z, donor_cfg), self._mask(coordination, z, acceptor_cfg)
        r = self._surface(opos[dm], opos[am], [hpos[rows[i]] for i in np.flatnonzero(dm)], cell)
        pc, sc, ps, ss = r.counts, r.shared_counts, r.pair_samples, r.shared_samples
        if self.swapped:
            sd = self._mask(coordination, z, {**acceptor_cfg, "species": donor_cfg["species"]})
            sa = self._mask(coordination, z, {**donor_cfg, "species": acceptor_cfg["species"]})
            s = self._surface(opos[sd], opos[sa], [hpos[rows[i]] for i in np.flatnonzero(sd)], cell)
            pc, sc = pc + s.counts[::-1, :], sc + s.shared_counts[::-1, :]
            ps, ss = ps + s.pair_samples, ss + s.shared_samples
        self.pairs = pc if self.pairs is None else self.pairs + pc
        self.shared = sc if self.shared is None else self.shared + sc
        self.pair_samples += ps
        self.shared_samples += ss

    def finalize(self) -> Result:
        if self.frames == 0:
            raise ValueError("Proton-sharing analysis found no trajectory frames.")
        c = self.sc
        de = np.linspace(*parse_range(c.get("delta_range_A", [-1.5, 1.5]), "delta_range_A"), int(c.get("delta_bins", 120)) + 1)
        oe = np.linspace(*parse_range(c.get("oo_range_A", [2.3, 3.6]), "oo_range_A"), int(c.get("oo_bins", 130)) + 1)
        se = np.linspace(*parse_range(c.get("shared_s_range_A", [-1.5, 1.5]), "shared_s_range_A"), int(c.get("shared_s_bins", 120)) + 1)
        re = np.linspace(*parse_range(c.get("shared_rho_range_A", [0.0, 1.5]), "shared_rho_range_A"), int(c.get("shared_rho_bins", 75)) + 1)
        rho = (re[:-1] + re[1:]) / 2
        jac = 2 * np.pi * rho[None, :] * np.diff(se)[:, None] * np.diff(re)[None, :]
        prob = normalized_probability(self.pairs)
        sprob = normalized_probability(np.divide(self.shared, jac, out=np.zeros_like(self.shared, dtype=float), where=jac > 0))
        t = float(c["temperature_K"])
        self._res = ProtonSharingResult((de[:-1] + de[1:]) / 2, (oe[:-1] + oe[1:]) / 2, free_energy_from_probability(prob, t), prob,
                                        self.pairs, (se[:-1] + se[1:]) / 2, rho, free_energy_from_probability(sprob, t), sprob,
                                        self.shared, self.pair_samples, self.shared_samples)
        return Result(self.type, self.name, {"pair_counts": self.pairs, "shared_counts": self.shared},
                      {"pair_samples": self.pair_samples, "shared_samples": self.shared_samples})

    def write(self, result: Result, outdir: Path) -> dict[str, Any]:
        from waterint.plot.proton_sharing import write_proton_sharing_csv, write_shared_proton_csv
        fes, shared = outdir / f"{self.name}.csv", outdir / f"{self.name}_shared.csv"
        write_proton_sharing_csv(fes, self._res)
        write_shared_proton_csv(shared, self._res)
        return {"fes_csv": str(fes), "shared_csv": str(shared)}

    def plot(self, result: Result, outdir: Path) -> dict[str, Any]:
        from waterint.plot.proton_sharing import plot_proton_sharing
        path = outdir / f"{self.name}.png"
        out = self.run.output_cfg
        plot_proton_sharing(path, self._res, title=str(out.get("title", "Proton-sharing free-energy surface")), dpi=int(out.get("dpi", 220)))
        return {"png": str(path)}


@register("proton_hbond.accumulate", "cpu")
def _phb_cpu(**kw):
    from waterint.backends.cpu.native import proton_hbond_accumulate
    return proton_hbond_accumulate(**kw)


class ProtonSharingHbond(Analysis):
    type = "proton-sharing-hbond"
    v06_name = "proton-sharing-hbond"
    description = "Accumulate globally pooled unit-pair F(delta, R_OO) surfaces with optional L1 acceptor H-bond coordination classes."
    params = {"proton_sharing_hbond.coordinate": ("required", "coordinate spec of the layer ranges"),
              "proton_sharing_hbond.l1_range / l2_range": ("profile windows L1 / L2", "layer ranges on the coordinate"),
              "proton_sharing_hbond.oo_range_A": ([2.2, 3.2], "O-O distance range of a pair (A)"),
              "proton_sharing_hbond.dha_angle_min_deg": (150.0, "minimum D-H...A angle (deg)"),
              "proton_sharing_hbond.oh_assignment_cutoff_A": (1.5, "ownership cutoff (A)"),
              "selection.pbc": ([True, True, False], "periodic axes"), "backend": ("auto", "auto | python | cpu")}

    def __init__(self, cfg: dict[str, Any], run):
        super().__init__(cfg, run)
        self.sel = dict(cfg.get("selection") or {})
        a = self.ac = dict(cfg.get("proton_sharing_hbond") or {})
        if not isinstance(a.get("coordinate"), dict):
            raise ValueError("proton_sharing_hbond.coordinate is required.")
        self.coord = coordinate_spec_from_config(a["coordinate"])
        windows = {w[0]: (w[1], w[2]) for w in self.profile.windows}
        self.l1 = self._layer("l1_range", windows.get("L1"))
        self.l2 = self._layer("l2_range", windows.get("L2"))
        self.oo = parse_range(a.get("oo_range_A", [2.2, 3.2]), "proton_sharing_hbond.oo_range_A")
        self.angle = float(a.get("dha_angle_min_deg", 150.0))
        self.pbc = _pbc(self.sel.get("pbc", [True, True, False]))
        self.own = _own({**self.sel, "oh_assignment_cutoff_A": a.get("oh_assignment_cutoff_A", 1.5)},
                        "oh_assignment_cutoff_A", 1.5, [True, True, False])
        self.state = new_proton_sharing_hbond_state(
            parse_range(a.get("delta_range_A", [-1.5, 1.5]), "delta_range_A"), self.oo,
            int(a.get("delta_bins", 120)), int(a.get("oo_bins", 100)))
        want = cfg.get("backend", a.get("backend", "auto"))
        self.use_cpu = False
        if normalize(want) != "python":
            try:
                self.backend, self.kernel = select("proton_hbond.accumulate", want)
                self.use_cpu = True
            except RuntimeError:
                if normalize(want) == "cpu":
                    raise
        self.backends_used["proton_hbond.accumulate"] = "cpu" if self.use_cpu else "python"
        self.prev_step = None

    def _layer(self, key: str, default) -> tuple[float, float]:
        value = self.ac.get(key, list(default) if default else None)
        if value is None:
            raise ValueError(f"proton_sharing_hbond.{key} is required (the profile has no {key[:2].upper()} window).")
        return parse_range(value, f"proton_sharing_hbond.{key}")

    def update(self, lf: LabelledFrame) -> None:
        if lf.frame.step is not None and lf.frame.step == self.prev_step:
            self.frames -= 1          # duplicate restart frame: not counted
            return
        self.prev_step = lf.frame.step
        own = lf.ownership(self.own)
        st = self.state
        coordination = own.counts.astype(np.int64)
        z = lf.coordinates(own.acceptor_idx, self.coord)
        opos, hpos = lf.positions[own.acceptor_idx], lf.positions[own.proton_idx]
        vectors = lf.cell_vectors
        if self.use_cpu and not np.allclose(vectors, np.diag(np.diag(vectors))):
            raise ValueError("proton-sharing-hbond C++ backend currently requires orthorhombic cells.")
        cell = np.diag(vectors)
        inside = lambda r: (z >= r[0]) & (z < r[1])
        d1, a1 = (coordination == 2) & inside(self.l1), (coordination == 1) & inside(self.l1)
        d2, a2 = (coordination == 2) & inside(self.l2), (coordination == 1) & inside(self.l2)
        rows = None if self.use_cpu else own.proton_lists()

        def accumulate(dm, am, *, l1_l1: bool, reverse: bool = False) -> None:
            if not np.any(dm) or not np.any(am):
                return
            if self.use_cpu:
                hist = np.zeros_like(st.l1_l1_counts_by_cn)
                hist, pairs, acceptors = self.kernel(
                    donor_positions=opos[dm], acceptor_positions=opos[am], hydrogen_positions=hpos,
                    hydrogen_counts=coordination[dm], hydrogen_matrix=own.matrix[dm], cell=cell, pbc=self.pbc,
                    oo_range=self.oo, angle_min=self.angle, delta_edges=st.delta_edges_a, oo_edges=st.oo_edges_a, hist=hist)
                if l1_l1:
                    st.l1_l1_counts_by_cn += hist
                    st.l1_l1_pair_samples += pairs
                    st.l1_l1_acceptor_samples += acceptors
                else:
                    values = hist.sum(axis=0)
                    st.l1_l2_counts += values[::-1] if reverse else values
                    st.l1_l2_pair_samples += pairs
            else:
                donor_h = [hpos[rows[i]] for i in np.flatnonzero(dm)]
                accumulate_python_pairs(st, opos[dm], opos[am], donor_h, vectors, self.pbc, is_l1_l1=l1_l1,
                                        oo_range_a=self.oo, angle_min_deg=self.angle, reverse_delta=reverse)

        if bool(self.ac.get("include_l1_l1", True)):
            accumulate(d1, a1, l1_l1=True)
        if bool(self.ac.get("include_l1_l2", True)):
            accumulate(d1, a2, l1_l1=False)
            accumulate(d2, a1, l1_l1=False, reverse=True)
        st.frames += 1

    def finalize(self) -> Result:
        st = self.state
        if st.frames == 0:
            raise ValueError("No trajectory frames found.")
        if bool(self.ac.get("l1_l1_delta_symmetrize", True)):
            st.l1_l1_counts_by_cn = (st.l1_l1_counts_by_cn + st.l1_l1_counts_by_cn[:, ::-1, :]) / 2.0
        self._surfaces = {"l1_l1_all": st.l1_l1_counts_by_cn[1:].sum(axis=0), "l1_l2": st.l1_l2_counts}
        if bool(self.ac.get("classify_l1_acceptor_cn", True)):
            self._surfaces.update({f"l1_l1_CN{cn}": st.l1_l1_counts_by_cn[int(cn)] for cn in self.ac.get("cn_values", [1, 2, 3, 4])})
        return Result(self.type, self.name, {}, {
            "backend": "cpp" if self.use_cpu else "python", "frames": st.frames,
            "pair_weighting": "unit weight per retained pair; global pooled normalization",
            "criteria": {"O_H_assignment_cutoff_A": self.own.cutoff, "R_OO_A": list(self.oo), "D_H_A_angle_min_deg": self.angle},
            "CN_definition": "distinct qualifying L1 H2O donor O atoms per L1 OH- acceptor in the same frame",
            "l1_l1_delta_symmetrized": bool(self.ac.get("l1_l1_delta_symmetrize", True)),
            "l1_l1_pairs_before_symmetrization": st.l1_l1_pair_samples, "l1_l2_pairs": st.l1_l2_pair_samples})

    def write(self, result: Result, outdir: Path) -> dict[str, Any]:
        st = self.state
        out = {}
        for name, counts in self._surfaces.items():
            fe, prob = free_energy_from_counts(counts, float(self.ac.get("temperature_K", 300.0)))
            path = outdir / f"{self.name}_{name}.npz"
            np.savez_compressed(path, delta_edges_a=st.delta_edges_a, oo_edges_a=st.oo_edges_a, counts=counts,
                                probability=prob, free_energy_kj_mol=fe)
            out[name] = str(path)
            result.meta.setdefault("surfaces", {})[name] = {"file": str(path), "pair_samples": int(counts.sum())}
        return {"surfaces": out}
