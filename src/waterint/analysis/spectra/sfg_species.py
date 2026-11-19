"""Species-resolved SFG on a frame window, and its equal-run aggregate.

sfg-species            (block) frames = the input window (input.first_frame, input.max_frames; qnpz decodes only the
                       chunks of the window); split into independent blocks of sfg_species.block_frames
sfg-species-aggregate  (post)  equal-run mean / SD / SEM of several sfg-species outputs (+ free-OH frequency scale)
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from waterint.analysis.base import Analysis, Result
from waterint.analysis.spectra.sfg_species_core import (N_LAG, gibbs_top_surface, layered_block, species_sfg_config,
                                                        zref_top_layer_mean)
from waterint.analysis.spectra.sfg_species_workflow import _save, run_sfg_species_aggregate


class SfgSpecies(Analysis):
    type = "sfg-species"
    kind = "block"
    needs_velocities = True
    v06_name = "sfg-species"
    description = ("Species- and layer-resolved SFG (surface O-H, interfacial OH- and H2O by layer) from velocity correlations "
                   "with the layered native kernel; the frame window is split into independent blocks that are summed.")
    params = {"sfg_species.block_frames": ("all", "independent blocks (pairs across a block edge are lost)"),
              "sfg_species.n_lag": (N_LAG, "correlation lags"), "sfg_species.free_oh": (False, "top-surface free-OH peak"),
              "sfg_species.dt_ps": (0.004, "time between frames"),
              "input.first_frame / max_frames": ("0 / all", "the frame window")}

    def __init__(self, cfg: dict[str, Any], run):
        super().__init__(cfg, run)
        self.sc = dict(cfg.get("sfg_species") or {})
        self.n_lag = int(self.sc.get("n_lag", N_LAG))
        self.want_top = bool(self.sc.get("free_oh", False))
        self.dt = float(self.sc.get("dt_ps", 0.004))
        ref = self.profile.reference or {}
        self.reference = [e for e in ref.get("elements", []) if e != "@solid"]
        if not self.reference:
            raise ValueError("sfg-species needs explicit reference elements in the profile (reference.elements).")
        own = self.profile.ownership
        self.acceptor, self.proton = own.acceptors[0], own.protons[0]
        self.nb = self.sc.get("block_frames")
        self.buf: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        self.buffered = 0
        self.blocks: list[Path] = []
        self.timing: list[dict[str, Any]] = []
        self.tot = None
        self.window_first: int | None = None
        self.t0 = time.time()

    def update_block(self, block) -> None:
        if block.velocities is None:
            raise ValueError("sfg-species needs velocities in the trajectory.")
        sym = np.asarray(block.symbols)
        if self.window_first is None:
            self.window_first = block.first
            self.oh = np.flatnonzero((sym == self.acceptor) | (sym == self.proton))
            self.ref_idx = np.flatnonzero(np.isin(sym, self.reference))
            so = sym[self.oh]
            self.oi, self.hi = np.flatnonzero(so == self.acceptor), np.flatnonzero(so == self.proton)
            self.cell = block.cell if block.cell is not None else tuple(block.boxes[0, 1::2] - block.boxes[0, 0::2])
            self.wrap = block.boxes is None            # qnpz positions are wrapped in x, y (v0.6 _qnpz_arrays)
        self.buf.append((block.positions[:, self.oh], block.velocities[:, self.oh], block.positions[:, self.ref_idx, 2]))
        self.buffered += block.n
        nb = int(self.nb) if self.nb else None
        while nb is not None and self.buffered >= nb:
            self._process(nb)

    def _take(self, m: int):
        parts, got = [], 0
        while got < m:
            P, V, Z = self.buf[0]
            k = min(m - got, P.shape[0])
            parts.append((P[:k], V[:k], Z[:k]))
            if k == P.shape[0]:
                self.buf.pop(0)
            else:
                self.buf[0] = (P[k:], V[k:], Z[k:])
            got += k
        self.buffered -= m
        return (np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]), np.concatenate([p[2] for p in parts]))

    def _process(self, m: int) -> None:
        tr = time.time()
        P, V, zmg = self._take(m)
        P = P.copy()
        if self.wrap:
            P[:, :, 0] %= self.cell[0]
            P[:, :, 1] %= self.cell[1]
        cfg = species_sfg_config(self.dt, self.n_lag, self.reference)
        zref = zref_top_layer_mean(zmg, 0.7)
        tk = time.time()
        blk = layered_block(P, V, self.oi, self.hi, zref, self.cell, cfg, self.n_lag)
        top = None
        if self.want_top:
            step = max(1, m // 200)
            zg, *_ = gibbs_top_surface([P[j, self.oi, 2] - zref[j] for j in range(0, m, step)])
            tcfg = dict(cfg)
            tcfg["species_channels"] = ["OH-"]
            tcfg["layer_bins"] = [{"label": "top", "window": {"mode": 2, "z1": zg - 3.0, "z2": zg + 10.0, "ramp": 0.0}}]
            tb = layered_block(P, V, self.oi, self.hi, zref, self.cell, tcfg, self.n_lag)
            j = tb.keys.index("top:all")
            top = (tb.numerators[j], tb.counts[j].astype(float), zg)
        out = self.run.output_dir
        out.mkdir(parents=True, exist_ok=True)
        bp = out / f"{self.name}_block{len(self.blocks):03d}.npz"
        _save(bp, blk.keys, blk.numerators, blk.counts, zref, m, self.n_lag, self.dt, top)
        self.blocks.append(bp)
        b0 = self.window_first + sum(t["frames"] for t in self.timing)
        self.timing.append({"first_frame": b0, "frames": m, "read_s": tk - tr, "kernel_s": time.time() - tk})
        if self.tot is None:
            self.tot = [blk.keys, blk.numerators.copy(), blk.counts.copy(), [zref], m, [] if top is None else [top]]
        else:
            if blk.keys != self.tot[0]:
                raise RuntimeError("channel keys changed between blocks")
            self.tot[1] += blk.numerators
            self.tot[2] += blk.counts
            self.tot[3] += [zref]
            self.tot[4] += m
            if top is not None:
                self.tot[5] += [top]

    def finalize(self) -> Result:
        if self.buffered:
            self._process(self.buffered)
        if self.tot is None:
            raise RuntimeError("sfg-species received no frames.")
        tot = self.tot
        top_tot = None
        if self.want_top:
            top_tot = (sum(t[0] for t in tot[5]), sum(t[1] for t in tot[5]), float(np.mean([t[2] for t in tot[5]])))
        npz = self.run.output_dir / f"{self.name}.npz"
        scale = _save(npz, tot[0], tot[1], tot[2], np.concatenate(tot[3]), tot[4], self.n_lag, self.dt, top_tot)
        n = int(tot[4])
        meta = {"analysis": "waterint sfg-species",
                "definition": "species-resolved channels (see sfg_species_core); waterint layered ssVVCF kernel",
                "source": {"trajectory": self.run.trajectory}, "first_frame": self.window_first, "frames": n,
                "block_frames": int(self.nb) if self.nb else n, "n_lag": self.n_lag, "dt_ps": self.dt, "free_oh_scale": scale,
                "blocks": [str(p) for p in self.blocks], "timing": self.timing, "elapsed_s": time.time() - self.t0,
                "note": "blocks are independent: O-H pairs whose lag window crosses a block edge are not counted (~1 ps per block)"}
        return Result(self.type, self.name, {}, meta, outputs={"npz": str(npz)})

    def write(self, result: Result, outdir: Path) -> dict[str, Any]:
        return {"npz": str(outdir / f"{self.name}.npz"), "blocks": [str(p) for p in self.blocks]}

    def metadata(self, result: Result) -> dict[str, Any]:
        return result.meta


class SfgSpeciesAggregate(Analysis):
    type = "sfg-species-aggregate"
    kind = "post"
    v06_name = "sfg-species-aggregate"
    description = "Combine per-run sfg-species spectra (equal run weights) with the free-OH frequency scale."
    params = {"runs": ("required", "sfg-species npz files or sfg-species analysis names"),
              "reference": (None, "reference spectra (npz) to overlay in the comparison plot"),
              "sfg_species.scale": ("auto", "auto = mean free-OH scale of the runs | number")}

    def __init__(self, cfg: dict[str, Any], run):
        super().__init__(cfg, run)
        if not cfg.get("runs"):
            raise ValueError("sfg-species-aggregate needs runs: [...].")

    def finalize(self) -> Result:
        runs = []
        for item in self.cfg["runs"]:
            res = self.run.results.get(str(item))
            runs.append(res.outputs["npz"] if res is not None else str(self.run.path(item)))
        legacy = {"input": {"runs": runs, "reference": None if not self.cfg.get("reference") else str(self.run.path(self.cfg["reference"]))},
                  "sfg_species": dict(self.cfg.get("sfg_species") or {}),
                  "output": {"directory": str(self.run.output_dir), "prefix": self.name,
                             "label": self.run.output_cfg.get("label", "this work")}}
        res = run_sfg_species_aggregate(legacy)
        self.frames = 0
        return Result(self.type, self.name, {}, {"runs": runs},
                      outputs={"npz": str(res.npz_path), "png": str(res.png_path), "peaks": str(res.peaks_path)})

    def write(self, result: Result, outdir: Path) -> dict[str, Any]:
        return dict(result.outputs)
