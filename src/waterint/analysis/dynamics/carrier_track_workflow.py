"""Config workflow: strict OH- carrier tracking on site-current event tables.

    input:
      events: [run/seg_00/pass_events.npz, run/seg_01/pass_events.npz, ...]   # site-current (events: enabled), in order
    carrier_track:
      positive: false        # true: also the +1 family H3O+ / Os_H (moves, conversions, pair creation / recombination)
    output:
      directory: out
      prefix: carrier
Writes <prefix>.npz (tracks, carrier-O schedule, birth/death reasons) and <prefix>_metadata.json. The schedule is the
input of ``carrier-msd``. See ``waterint.analysis.dynamics.carrier_track_core`` for the rules.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from waterint.analysis.dynamics.carrier_track_core import track_event_tables
from waterint.analysis.spectra._workflow_utils import resolve_path
from waterint.analysis.spectra._workflow_utils import require_mapping


@dataclass(frozen=True)
class CarrierTrackWorkflowResult:
    npz_path: Path
    metadata_path: Path
    tracks: int
    frames: int


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def event_paths(config: dict[str, Any], input_cfg: dict[str, Any]) -> list[Path]:
    raw = input_cfg.get("events")
    if isinstance(raw, (str, Path)):
        raw = [raw]
    if not raw:
        raise ValueError("input.events must list site-current *_events.npz tables of one run, in time order.")
    return [resolve_path(config, str(p)) for p in raw]


def save_tracking(path: Path, result) -> None:
    tmp = path.with_name(path.name + ".tmp.npz")
    extra = {}
    if result.extra:
        extra = {"positive_tracks": result.extra["positive_tracks"], "pairs": result.extra["pairs"],
                 "pair_type_names": np.array(["creation", "recombination"])}
    np.savez_compressed(tmp, tracks=result.tracks, changes=result.changes, birth_reason=result.birth_reason,
                        death_reason=result.death_reason, first_frame=np.array(result.first_frame),
                        last_frame=np.array(result.last_frame), format=np.array("waterint-carrier-tracks-1"), **extra)
    tmp.replace(path)


def run_carrier_track(config: dict[str, Any]) -> CarrierTrackWorkflowResult:
    input_cfg = require_mapping(config, "input")
    output_cfg = require_mapping(config, "output")
    files = event_paths(config, input_cfg)
    out_dir = resolve_path(config, output_cfg.get("directory", "."))
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = str(output_cfg.get("prefix", "carrier"))
    t0 = time.time()
    verbose = bool(output_cfg.get("verbose", True))
    positive = bool((config.get("carrier_track") or {}).get("positive", False))
    result = track_event_tables(files, log=print if verbose else None, positive=positive)
    npz_path, meta_path = out_dir / f"{prefix}.npz", out_dir / f"{prefix}_metadata.json"
    save_tracking(npz_path, result)
    life = (result.tracks["death_frame"] - result.tracks["birth_frame"]).astype(float)
    metadata = {
        "analysis": "waterint carrier-track",
        "definition": "strict OH- carrier tracking (direct + relay <= 4 H, unique shortest path) on waterint-events-1 tables",
        "inputs": [{"path": str(f), "sha256": _sha(f)} for f in files],
        "summary": result.summary,
        "positive_family": None if not result.extra else result.extra["positive_summary"],
        "lifetime_frames": {"median": float(np.median(life)) if life.size else None,
                            "persistent_tracks_ge_250_frames": int(np.sum(life >= 250))},
        "outputs": {"npz": str(npz_path), "npz_sha256": _sha(npz_path)},
        "elapsed_s": time.time() - t0,
        "limits": ["regions come from the event z_rel (midpoint of the two frames); initial carriers have region -1",
                   "per-frame site paths need positions: run carrier-msd with this file"],
    }
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return CarrierTrackWorkflowResult(npz_path, meta_path, int(result.summary["tracks"]), int(result.summary["frames"]))
