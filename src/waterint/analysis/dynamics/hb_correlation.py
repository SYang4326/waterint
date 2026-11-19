"""One streaming pass for C_PT / C_HB / proton-sharing FES, species density, O-H orientation, species counts,
H-bond statistics and fixed-atom xy MSD (definitions: hb_correlation_core).

Backends (kernel ``hb_correlation.stream``): cpu (C++/OpenMP, in process) and gpu (CUDA kernel in a separate
process, built with nvcc on first use; WATERINT_CUDA_ARCH, default sm_90). ``auto`` = cpu: the GPU kernel gives
identical C_PT / C_HB / FES counts but orientation histograms can differ by a few counts in 1e7 (GPU acos), so the
GPU is used only when requested.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from waterint.analysis.base import Analysis, Result
from waterint.analysis.dynamics.hb_correlation_core import HBCorrelationGeometry, HBCorrelationKernel, summarize
from waterint.backends.registry import register, select


@register("hb_correlation.stream", "cpu")
def _hbc_cpu(geom, n_frames, lx, ly, raw_dir, **kw):
    return HBCorrelationKernel(geom, n_frames, lx, ly, **kw)


@register("hb_correlation.stream", "gpu")
def _hbc_gpu(geom, n_frames, lx, ly, raw_dir, binary=None, **kw):
    from waterint.backends.gpu.hb_correlation import HBCorrelationGPUProcess
    return HBCorrelationGPUProcess(geom, n_frames, lx, ly, raw_dir, binary=binary, **kw)


class HbCorrelation(Analysis):
    type = "hb-correlation"
    kind = "block"
    v06_name = "hb-correlation"
    description = ("Proton-transfer and H-bond correlation functions C_PT(t), C_HB(t) from seeded geometric definitions, "
                   "proton-sharing free energy, species density, O-H orientation, H-bond statistics and fixed-atom xy MSD "
                   "in one streaming pass, results per block; cpu or gpu backend.")
    params = {"hb_correlation.window_frames": (100000, "frames per window (ring = window + 25000)"),
              "hb_correlation.block_frames": (250000, "frames per result block"),
              "hb_correlation.msd_stride_frames": (10, "MSD frame stride (origins and lags)"),
              "hb_correlation.msd_max_lag_frames": (250000, "longest MSD lag (frames)"),
              "hb_correlation.top_reference_atoms": (72, "z_ref = mean z of the N highest reference atoms (alias top_mg)"),
              "hb_correlation.lattice_offset_A": (1.0, "lattice acceptor: z < z_ref(frame 0) + offset"),
              "backend": ("auto (= cpu)", "cpu | gpu"), "hb_correlation.gpu_binary": (None, "prebuilt CUDA executable"),
              "input.dt_ps": (0.004, "LAMMPS input: time between saved frames")}

    def __init__(self, cfg: dict[str, Any], run):
        super().__init__(cfg, run)
        self.hc = dict(cfg.get("hb_correlation") or {})
        want = str(cfg.get("backend", self.hc.get("backend", "cpu"))).lower()
        self.backend, self.factory = select("hb_correlation.stream", "cpu" if want == "auto" else want)
        self.backends_used["hb_correlation.stream"] = self.backend
        ref = self.profile.reference or {}
        self.reference = [e for e in ref.get("elements", []) if e != "@solid"]
        if not self.reference:
            raise ValueError("hb-correlation needs explicit reference elements in the profile (reference.elements).")
        self.width = float(ref.get("layer_width_A", 0.70))
        self.top_n = int(self.hc.get("top_reference_atoms", self.hc.get("top_mg", 72)))
        self.kernel = None
        self.t0 = time.time()
        self.t_wait = 0.0
        self.raw = run.output_dir / f"{self.name}_raw"
        spec = run.input_spec
        self.opts = dict(spec.options) if spec is not None else {}

    def update_block(self, block) -> None:
        pos, sym = block.positions, list(block.symbols)
        if block.boxes is not None:          # lammpstrj: atom id order
            order = np.argsort(block.ids)
            if not np.array_equal(block.ids[order], np.arange(1, block.ids.size + 1)):
                raise ValueError("atom ids must be 1..N")
            pos, sym = pos[:, order], [sym[i] for i in order]
        if self.kernel is None:
            if block.boxes is not None:
                b = block.boxes[0]
                self.cell = [float(b[1] - b[0]), float(b[3] - b[2])]
                self.dt_ps = float(self.opts.get("dt_ps", 0.004))
            else:
                self.cell = [float(block.cell[0]), float(block.cell[1])]
                self.dt_ps = float(block.meta["dt_ps"])
            self.n_frames = int(self.run.count_frames())
            own = self.profile.ownership
            self.geom = HBCorrelationGeometry(sym, pos[0], top_n=self.top_n,
                                              lattice_offset_A=float(self.hc.get("lattice_offset_A", 1.0)),
                                              layer_width_A=self.width, reference=self.reference,
                                              acceptor=own.acceptors[0], proton=own.protons[0])
            self.raw.mkdir(parents=True, exist_ok=True)
            kw = dict(window=int(self.hc.get("window_frames", 100000)), block=int(self.hc.get("block_frames", 250000)),
                      msd_stride=int(self.hc.get("msd_stride_frames", 10)), msd_maxlag=int(self.hc.get("msd_max_lag_frames", 250000)))
            if self.backend == "gpu":
                kw["binary"] = self.hc.get("gpu_binary")
            self.kernel = self.factory(self.geom, self.n_frames, self.cell[0], self.cell[1], self.raw, **kw)
        self.kernel.push(pos)

    def finalize(self) -> Result:
        if self.kernel is None:
            raise RuntimeError("no frames read")
        self.kernel.finish(self.raw, self.t_wait)
        threads = self.kernel.threads
        self.kernel.close()
        area = self.cell[0] * self.cell[1]
        self._summary = summarize(self.raw, self.dt_ps, area)
        ks = json.loads((self.raw / "kernel_summary.json").read_text())
        meta = {
            "analysis": "waterint hb-correlation",
            "definition": "seeded geometric C_PT / C_HB and proton-sharing FES; species density, orientation and counts; "
                          "fixed-atom xy MSD at sparse lags (see waterint.analysis.dynamics.hb_correlation_core)",
            "source": {"trajectory": self.run.trajectory}, "frames": self.n_frames, "dt_ps": self.dt_ps, "cell_xy_A": self.cell[:2],
            "threads": threads, "backend": self.backend,
            "geometry": {"atoms_O_H": int(len(self.geom.oh)), "lattice_O": int(self.geom.lattice.sum()), "top_mg": self.top_n,
                         "reference_elements": self.reference, "lattice_offset_A": float(self.hc.get("lattice_offset_A", 1.0))},
            "kernel_summary": ks, "raw_directory": str(self.raw),
            "elapsed_s": {"total": time.time() - self.t0, "reading": "see waterint_run.json seconds_read (shared pass)"},
            "species_per_frame_mean": dict(zip(["Os_H", "L1_OH", "L2_OH", "bulk_OH"], map(float, self._summary["species_per_frame_mean"])))}
        return Result(self.type, self.name, {}, meta)

    def write(self, result: Result, outdir: Path) -> dict[str, Any]:
        path = outdir / f"{self.name}_summary.npz"
        np.savez_compressed(path, **self._summary)
        result.meta["summary_npz"] = str(path)
        return {"summary_npz": str(path), "raw_directory": str(self.raw)}

    def plot(self, result: Result, outdir: Path) -> dict[str, Any]:
        from waterint.plot.hb_correlation import plot_overview
        path = outdir / f"{self.name}_overview.png"
        plot_overview(path, self._summary, self.raw, str(self.run.output_cfg.get("title", self.name)))
        return {"png": str(path)}

    def metadata(self, result: Result) -> dict[str, Any]:
        return result.meta
