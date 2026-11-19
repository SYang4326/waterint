"""Oxygen identity transitions (O_bone / O2- / OH / H2O / H3O+ ...) from relabelled-trajectory type codes."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from waterint.analysis.base import Analysis, Result
from waterint.analysis.dynamics.identity_transition_core import (STANDARD_TYPE_CODES, IdentityTransitionAccumulator,
                                                                  identity_states_for, type_codes_to_identity_states)
from waterint.prep.labelled import LabelledFrame


class IdentityTransition(Analysis):
    type = "identity-transition"
    v06_name = "identity-transition"
    description = "Write an O_bone/OH/H2O/H3O+ transition-count matrix and heatmap."
    params = {"identity_transition.type_codes": ("standard", "state -> LAMMPS type code(s) of a relabelled trajectory"),
              "identity_transition.minimum_residence_frames": (1, "both contiguous states must last this many frames")}

    def __init__(self, cfg: dict[str, Any], run):
        super().__init__(cfg, run)
        tcfg = dict(cfg.get("identity_transition") or {})
        raw = tcfg.get("type_codes", "standard")
        if raw == "standard":
            raw = STANDARD_TYPE_CODES
        if not isinstance(raw, dict):
            raise ValueError("identity_transition.type_codes must be 'standard' or a mapping.")
        self.type_codes = {str(k): int(v) if isinstance(v, int) else [int(x) for x in v] for k, v in raw.items()}
        self.states = identity_states_for(self.type_codes)
        self.oxygen_types = {c for v in self.type_codes.values() for c in ([v] if isinstance(v, int) else v)}
        self.min_res = int(tcfg.get("minimum_residence_frames", 1))
        if self.min_res <= 0:
            raise ValueError("identity_transition.minimum_residence_frames must be positive.")
        self.acc: IdentityTransitionAccumulator | None = None
        self.oxygen: np.ndarray | None = None
        self.seen: set[int] = set()
        self.duplicates = 0

    def update(self, lf: LabelledFrame) -> None:
        frame = lf.frame
        if frame.step is None or frame.types is None:
            raise ValueError("identity-transition needs a LAMMPS trajectory with timesteps and type codes.")
        if int(frame.step) in self.seen:
            self.duplicates += 1
            return
        self.seen.add(int(frame.step))
        types = np.asarray(frame.types, dtype=np.int64)
        if self.oxygen is None:
            self.oxygen = np.where(np.isin(types, list(self.oxygen_types)))[0]
            if self.oxygen.size == 0:
                raise ValueError("No O identity type codes occur in the trajectory.")
            self.acc = IdentityTransitionAccumulator(n_oxygen=int(self.oxygen.size), minimum_residence_frames=self.min_res,
                                                     states=self.states)
        elif not np.all(np.isin(types[self.oxygen], list(self.oxygen_types))):
            raise ValueError("O identity type membership changed; atom ordering/types are not compatible.")
        self.acc.add(type_codes_to_identity_states(types[self.oxygen], type_codes=self.type_codes))

    def finalize(self) -> Result:
        if self.acc is None:
            raise RuntimeError("No frames were read.")
        self._res = self.acc.finish()
        r = self._res
        return Result(self.type, self.name, {}, {
            "duplicate_timesteps_skipped": self.duplicates, "states": list(r.states), "type_codes": self.type_codes,
            "minimum_residence_frames": self.min_res, "transition_count": r.transitions,
            "definition": "Off-diagonal transitions between adjacent per-oxygen state segments; both contiguous states "
                          "must meet minimum_residence_frames."})

    def write(self, result: Result, outdir: Path) -> dict[str, Any]:
        from waterint.plot.identity_transition import write_identity_transition_outputs
        csv, png, pdf = write_identity_transition_outputs(self._res, directory=outdir, prefix=self.name,
                                                          title=str(self.run.output_cfg.get("title", "Oxygen identity transitions")))
        return {"csv": str(csv), "png": str(png), "pdf": str(pdf)}
