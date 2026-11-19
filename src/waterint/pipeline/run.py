"""One-pass pipeline: read each frame once, prepare it once (LabelledFrame), feed every analysis.

v0.7 config:
    input:    {trajectory: path | [segments], format, type_map, stride, max_frames, ...}
    system:   {profile: name | path | mapping (optionally with `extends`), cell: auto | [Lx, Ly, Lz]}
    prep:     {backend: auto | python | cpu, ownership: {cutoff, mode, pbc}}   # overrides of the profile
    output:   {directory, plot: true, ...plot options}
    analyses: [{type: density, name: ..., <analysis sections>}, ...]
"""
from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from waterint import registry
from waterint.analysis.base import Result, RunContext, save_result
from waterint.io.blocks import count_frames
from waterint.io.trajectory import open_trajectory
from waterint.prep.labelled import LabelledFrame, PrepContext
from waterint.system.profile import load_system_profile


def run(config: dict[str, Any]) -> dict[str, Result]:
    cfg = validate(config)
    base = Path(cfg.get("_config_dir", "."))
    profile = load_system_profile(cfg["system"].get("profile"))
    prep_cfg = cfg.get("prep") or {}
    if prep_cfg.get("ownership"):
        profile = replace(profile, ownership=profile.ownership.replace(prep_cfg["ownership"]))
    trajectory = open_trajectory(cfg["input"], base) if cfg.get("input") else None
    type_map = dict(trajectory.spec.type_map) if trajectory is not None else {}
    prep = PrepContext(profile=profile, type_map=type_map, backend=str(prep_cfg.get("backend", "auto")),
                       configured_cell=parse_cell(cfg["system"].get("cell", "auto")))
    out_cfg = dict(cfg.get("output") or {})
    outdir = Path(out_cfg.get("directory", "output"))
    outdir = outdir if outdir.is_absolute() else base / outdir
    run_ctx = RunContext(prep=prep, trajectory=trajectory.label if trajectory else "", config_dir=base, output_dir=outdir,
                         output_cfg=out_cfg, config_path=cfg.get("_config_path"),
                         input_spec=trajectory.spec if trajectory else None,
                         open_trajectory=lambda input_cfg: open_trajectory(input_cfg, base),
                         count_frames=(lambda: count_frames(trajectory.spec)) if trajectory else None)
    analyses = [registry.get(a["type"])(a, run_ctx) for a in cfg["analyses"]]
    names = [a.name for a in analyses]
    if len(set(names)) != len(names):
        raise ValueError(f"Analysis names must be unique (they are output prefixes): {names}")
    streaming = [a for a in analyses if a.kind in {"stream", "block"}]
    if streaming and trajectory is None:
        raise ValueError("Config section 'input' is required for analyses that read a trajectory.")

    t0 = time.perf_counter()
    read_s = 0.0
    frames = 0
    stream = [a for a in streaming if a.kind == "stream"]
    blocks = [a for a in streaming if a.kind == "block"]
    if blocks:
        it = trajectory.iter_blocks(int(cfg["input"].get("block_frames", 1000)),
                                    velocities=any(a.needs_velocities for a in streaming))
        while True:
            tr = time.perf_counter()
            block = next(it, None)
            read_s += time.perf_counter() - tr
            if block is None:
                break
            for analysis in blocks:
                analysis.step_block(block)
            if stream and not block.seed:
                for frame in block.frames():
                    lf = LabelledFrame(frame, prep)
                    for analysis in stream:
                        analysis.step(lf)
            frames += 0 if block.seed else block.n
    elif stream:
        it = iter(trajectory)
        while True:
            tr = time.perf_counter()
            frame = next(it, None)
            read_s += time.perf_counter() - tr
            if frame is None:
                break
            lf = LabelledFrame(frame, prep)
            for analysis in stream:
                analysis.step(lf)
            frames += 1
    if streaming and frames == 0:
        raise ValueError(f"No frames found in trajectory: {trajectory.label}")

    results: dict[str, Result] = {}
    plot = bool(out_cfg.get("plot", True))
    for analysis in analyses:               # config order; post analyses see earlier results
        t1 = time.perf_counter()
        result = analysis.finalize()
        if analysis.kind == "post":
            analysis.seconds += time.perf_counter() - t1
        results[analysis.name] = save_result(analysis, result, outdir, plot=plot)
        run_ctx.results[analysis.name] = results[analysis.name]
    summary = {"trajectory": run_ctx.trajectory, "frames": frames, "profile": profile.profile_id,
               "seconds_total": round(time.perf_counter() - t0, 3), "seconds_read": round(read_s, 3),
               "seconds_per_analysis_incl_prep": {a.name: round(a.seconds, 3) for a in analyses},
               "prep_backends": prep.backends_used, "analyses": names}
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "waterint_run.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return results


def validate(config: dict[str, Any]) -> dict[str, Any]:
    if "analyses" not in config:
        raise ValueError("Config section 'analyses' is required (v0.7 schema; v0.6 configs: `waterint <module> --config`).")
    if not isinstance(config["analyses"], list) or not config["analyses"]:
        raise ValueError("analyses must be a non-empty list.")
    for item in config["analyses"]:
        if not isinstance(item, dict) or "type" not in item:
            raise ValueError("Every analyses entry needs a type.")
        registry.resolve_type(item["type"])
    config.setdefault("system", {})
    return config


def parse_cell(value: Any) -> tuple[float, float, float] | None:
    if value is None or str(value).lower() == "auto":
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError("system.cell must be [Lx, Ly, Lz] in Angstrom or auto.")
    cell = tuple(float(v) for v in value)
    if any(v <= 0 for v in cell):
        raise ValueError("system.cell values must be positive.")
    return cell  # type: ignore[return-value]
