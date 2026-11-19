"""Sheet conductance from site-current outputs (post analysis): STACIE per run (equal-run mean / SEM) and pooled,
collective Helfand, the (group, window) decomposition and the interface / bulk zone partition (profile windows)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from waterint.analysis.base import Analysis, Result
from waterint.analysis.transport.site_conductance_workflow import run_site_conductance


class SiteConductance(Analysis):
    type = "site-conductance"
    kind = "post"
    v06_name = "site-conductance"
    description = ("Sheet conductance from site currents: STACIE Green-Kubo fit per run (equal-run mean and SEM) and of the "
                   "pooled runs, collective Helfand estimate, exact interface / bulk partition from the site-current "
                   "decomposition.")
    params = {"runs": ("required", "{run: [site_current.npz, ...] | glob | site-current analysis name}"),
              "conductance.current": ("site", "site | formal | both"), "conductance.temperature_K": (300.0, "temperature (K)"),
              "conductance.time_window_ps": (None, "[t0, t1] part of each run to use (ps); none = all"), "conductance.coarsen": (1, "sum n consecutive increments"),
              "conductance.zones": (None, "true = interface (below_L1, L1, L2) / bulk (bulk) profile windows")}

    def __init__(self, cfg: dict[str, Any], run):
        super().__init__(cfg, run)
        if not isinstance(cfg.get("runs"), dict) or not cfg["runs"]:
            raise ValueError("site-conductance needs runs: {name: files | site-current analysis name}.")

    def _resolve(self, spec: Any) -> Any:
        items = [spec] if isinstance(spec, (str, Path)) else list(spec)
        out = []
        for item in items:
            res = self.run.results.get(str(item))
            out.append(res.outputs["npz"] if res is not None else str(self.run.path(item)))
        return out

    def finalize(self) -> Result:
        legacy = {"runs": {name: self._resolve(spec) for name, spec in self.cfg["runs"].items()},
                  "conductance": dict(self.cfg.get("conductance") or {}),
                  "output": {"directory": str(self.run.output_dir), "prefix": self.name}, "_config_dir": str(self.run.config_dir)}
        res = run_site_conductance(legacy)
        return Result(self.type, self.name, {}, json.loads(res.summary_path.read_text(encoding="utf-8")),
                      outputs={"json": str(res.summary_path)})

    def write(self, result: Result, outdir: Path) -> dict[str, Any]:
        return {"json": str(outdir / f"{self.name}.json"), "arrays": str(outdir / f"{self.name}_arrays.npz")}
