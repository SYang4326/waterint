"""Config workflow: strict OH-defect (carrier) site-path xy MSD (definition: carrier_msd_core).

    profile: mgo_water                                # z_ref (reference elements, top layer width) and z windows
    input:
      format: gpumd-qnpz                              # or lammpstrj (trajectory: ...)
      segments: [run/seg_00, run/seg_01, ...]         # the frames the event tables were made from, same order
      carrier_tracks: out/carrier.npz                 # from carrier-track; or events: [...] to track here
      prefetch: 3                                     # qnpz chunks decoded ahead in threads
    carrier_msd:
      full_lag_ps: 200                                # every saved frame up to here, sparse beyond
      max_lag_ps: 10000
      origin_classes: {interface: [below_L1, L1, L2], bulk: [bulk]}   # unions of profile windows (state at the origin)
      storage: null                                   # directory for memory-mapped site paths (long runs)
    output: {directory: out, prefix: carrier_msd, plot: true}
Outputs: <prefix>.npz (numerators, valid pairs, at-risk origins per class and lag), <prefix>.csv, <prefix>_metadata.json,
optionally <prefix>.png. The positions are read once; frame indices must match the event tables (global indices).
"""
from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from waterint.system import profile as wprof
from waterint.analysis.dynamics.carrier_msd_core import CarrierSitePaths, strict_carrier_msd, zref_top_layer
from waterint.analysis.dynamics.carrier_track_core import track_event_tables
from waterint.analysis.dynamics.carrier_track_workflow import _sha, event_paths, save_tracking
from waterint.analysis.spectra._workflow_utils import resolve_path
from waterint.analysis.spectra._workflow_utils import require_mapping

DEFAULT_CLASSES = {"interface": ["below_L1", "L1", "L2"], "bulk": ["bulk"]}


@dataclass(frozen=True)
class CarrierMsdWorkflowResult:
    npz_path: Path
    csv_path: Path
    png_path: Path | None
    metadata_path: Path
    tracks: int
    frames: int


def _class_edges(profile: dict, classes: dict[str, list[str]]) -> dict[str, list[tuple[float, float]]]:
    win = {str(w[0]): (float(w[1]), float(w[2])) for w in profile.get("windows", [])}
    out = {}
    for name, members in classes.items():
        missing = [m for m in members if m not in win]
        if missing:
            raise ValueError(f"origin class {name}: windows {missing} are not in the profile ({list(win)})")
        out[str(name)] = [win[m] for m in members]
    return out


