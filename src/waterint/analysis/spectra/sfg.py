"""SFG: ssVVCF correlation functions and spectra.

sfg.mode: trajectory    (stream: frames from the shared pass; layered when sfg.layer_bins is given)
          single        (post: FT of an existing correlation file sfg.cf)
          combine_bins  (post: combine per-bin correlation files)
The numerics and file layout are the v0.6 ones (sfg_core / sfg_workflow), so spectra stay bitwise identical.
OPEN ITEM: SFG assigns H to O with its own rule (every H belongs to one O-H bond); see _legacy_selection.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from waterint.analysis.base import Analysis, Result
from waterint.analysis.spectra import sfg_workflow as wf
from waterint.prep.labelled import LabelledFrame


class Sfg(Analysis):
    type = "sfg"
    v06_name = "sfg"
    needs_velocities = True
    description = "Run SFG correlation-function and spectrum workflows."
    params = {"sfg.mode": ("single", "single | combine_bins | trajectory"),
              "sfg.layer_bins": (None, "layered trajectory mode: z windows"),
              "sfg.nzeros": (2000, "zero padding of the FT"), "sfg.backend": ("auto", "auto | python | cpp")}

    def __init__(self, cfg: dict[str, Any], run):
        super().__init__(cfg, run)
        self.sfg = dict(cfg.get("sfg") or {})
        self.mode = str(self.sfg.get("mode", "single"))
        if self.mode not in {"single", "combine_bins", "trajectory"}:
            raise ValueError("sfg.mode must be single, combine_bins, or trajectory.")
        self.kind = "stream" if self.mode == "trajectory" else "post"
        self.collected: list = []

    def update(self, lf: LabelledFrame) -> None:
        self.collected.append(lf.frame)

    def _legacy_config(self) -> dict[str, Any]:
        out = {**self.run.output_cfg, **(self.cfg.get("output") or {})}
        out["directory"] = str(self.run.output_dir)
        out["prefix"] = self.name
        cfg = {"sfg": copy.deepcopy(self.sfg), "output": out, "_config_dir": str(self.run.config_dir),
               "system": {"cell": list(self.run.prep.configured_cell) if self.run.prep.configured_cell else "auto"}}
        if self.run.input_spec is not None:
            cfg["input"] = self.run.input_spec.legacy_input_cfg()
        return cfg

    def finalize(self) -> Result:
        cfg = self._legacy_config()
        self.run.output_dir.mkdir(parents=True, exist_ok=True)
        if self.mode == "single":
            res = wf.run_single(cfg, cfg["sfg"], cfg["output"])
        elif self.mode == "combine_bins":
            res = wf.run_combine_bins(cfg, cfg["sfg"], cfg["output"])
        else:
            if not self.collected:
                raise ValueError("SFG trajectory mode found no frames.")
            cfg["_frames"] = self.collected
            res = wf.run_trajectory(cfg, cfg["sfg"], cfg["output"])
            self.collected = []
        self._res = res
        return Result(self.type, self.name, {}, {"mode": self.mode})

    def metadata(self, result: Result) -> dict[str, Any]:
        # the v0.6 workflow already wrote <prefix>_metadata.json; keep it, store the v0.7 record beside it
        import json
        base = super().metadata(result)
        Path(f"{self.run.output_dir / self.name}_waterint.json").write_text(json.dumps(base, indent=2, default=str) + "\n",
                                                                            encoding="utf-8")
        meta_path = Path(self._res.metadata_path)
        return json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else base

    def write(self, result: Result, outdir: Path) -> dict[str, Any]:
        r = self._res
        conv = lambda d: {str(k): str(v) for k, v in (d or {}).items()}
        return {"cf": conv(getattr(r, "cf_paths", {})), "ft": conv(getattr(r, "ft_paths", {})), "png": conv(getattr(r, "png_paths", {}))}
