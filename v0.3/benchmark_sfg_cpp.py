from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any

import numpy as np

from waterint.config import load_config
from waterint._01_core.selection import SelectionContext, element_indices
from waterint._02_computation._native import sfg_ssvvcf
from waterint._02_computation.sfg import (
    build_sfg_segments_python,
    contiguous_frame_positions,
    correlate_sfg_segments_python,
    pbc_flags,
    sfg_velocities_from_frames,
    trajectory_velocities_from_frames,
    zref_series,
)
from waterint._04_workflows.workflows.common import iter_frames, parse_cell, resolve_path


STAGE_NAMES = (
    "Read NPZ frames",
    "Reference + O/H selection",
    "Velocity preparation",
    "O-H assignment",
    "Segment signal construction",
    "Segment correlation",
    "Finalize correlation",
    "Other overhead",
)


def main() -> None:
    root = Path(__file__).resolve().parent
    config = load_config(root / "example/mgo_sfg/config_100ps_npz_full.yaml")
    outputs = {}
    for backend in ("python", "cpp"):
        result, timings = run_profile(config, backend=backend)
        outputs[backend] = {"result": result, "timings": timings}
        print(f"{backend}: {timings['Total']:.2f} s")

    python = outputs["python"]["result"]
    cpp = outputs["cpp"]["result"]
    if python["counts"] != cpp["counts"] or python["zrefs"] != cpp["zrefs"]:
        raise RuntimeError("Python and C++ SFG integer/reference results differ.")
    corr_difference = float(np.max(np.abs(np.asarray(python["corr"]) - np.asarray(cpp["corr"]))))
    if corr_difference > 1.0e-10:
        raise RuntimeError(f"Python and C++ SFG correlations differ by {corr_difference}.")
    print(f"frames: {python['frames']}")
    print("counts and zrefs: exact")
    print(f"maximum correlation difference: {corr_difference:.3e}")
    print_timing_table(outputs)


def print_timing_table(outputs: dict[str, dict[str, Any]]) -> None:
    """Print only the non-overlapping stage timings, not full result arrays."""

    print("\n| Stage | Python (s) | C++ (s) | Saved (s) | Speedup |")
    print("|---|---:|---:|---:|---:|")
    for stage in (*STAGE_NAMES, "Total"):
        python_seconds = float(outputs["python"]["timings"][stage])
        cpp_seconds = float(outputs["cpp"]["timings"][stage])
        speedup = python_seconds / cpp_seconds if cpp_seconds > 0.0 else float("inf")
        print(
            f"| {stage} | {python_seconds:.2f} | {cpp_seconds:.2f} | "
            f"{python_seconds - cpp_seconds:.2f} | {speedup:.2f}x |"
        )


def run_profile(config: dict[str, Any], *, backend: str) -> tuple[dict[str, Any], dict[str, float]]:
    if backend not in {"python", "cpp"}:
        raise ValueError("backend must be python or cpp.")
    input_cfg = config["input"]
    sfg_cfg = copy.deepcopy(config["sfg"])
    context = SelectionContext.from_input_config(input_cfg)
    timings = {name: 0.0 for name in STAGE_NAMES}
    total_start = time.perf_counter()

    read_start = time.perf_counter()
    traj_path = resolve_path(config, input_cfg["trajectory"])
    frames = list(iter_frames(traj_path, input_cfg))
    timings["Read NPZ frames"] = time.perf_counter() - read_start
    if len(frames) < 3:
        raise ValueError("SFG benchmark requires at least three frames.")
    cell = parse_cell(config["system"].get("cell", "auto")) or frames[0].cell
    if cell is None:
        raise ValueError("No cell information is available.")

    setup_start = time.perf_counter()
    dt_ps = float(sfg_cfg["dt_ps"])
    max_lag = int(round(float(sfg_cfg["lag_ps"]) / dt_ps))
    pbc = pbc_flags(sfg_cfg.get("pbc", [True, True, False]))
    oxygen_symbol = str(sfg_cfg.get("oxygen_symbol", "O"))
    hydrogen_symbol = str(sfg_cfg.get("hydrogen_symbol", "H"))
    oxygen_indices = element_indices(frames[0], {oxygen_symbol}, context)
    hydrogen_indices = element_indices(frames[0], {hydrogen_symbol}, context)
    zrefs = zref_series(frames, sfg_cfg, context)
    timings["Reference + O/H selection"] = time.perf_counter() - setup_start

    if backend == "python":
        velocity_start = time.perf_counter()
        velocities, _velocity_source = sfg_velocities_from_frames(
            frames,
            sfg_cfg=sfg_cfg,
            cell=cell,
            pbc=pbc,
        )
        timings["Velocity preparation"] = time.perf_counter() - velocity_start

        segment_start = time.perf_counter()
        segments = build_sfg_segments_python(
            frames,
            velocities=velocities,
            zrefs=zrefs,
            cell=cell,
            sfg_cfg=sfg_cfg,
            context=context,
            pbc=pbc,
            oxygen_symbol=oxygen_symbol,
            hydrogen_symbol=hydrogen_symbol,
            oh_cutoff=float(sfg_cfg.get("oh_cutoff", sfg_cfg.get("bond_cutoff", 1.25))),
            neighbor_method=str(sfg_cfg.get("neighbor_method", "auto")),
            neighbor_workers=int(sfg_cfg.get("neighbor_workers", 1)),
            oxygen_chunk_size=int(sfg_cfg.get("oxygen_chunk_size", 2048)),
            mu_mode=str(sfg_cfg.get("mu_mode", "full")).lower(),
            flip_sign=bool(sfg_cfg.get("flip_sign", False)),
            stage_timings=timings,
        )
        segment_wall = time.perf_counter() - segment_start
        timings["Other overhead"] += segment_wall - (
            timings["O-H assignment"] + timings["Segment signal construction"]
        )

        correlation_start = time.perf_counter()
        sums, counts = correlate_sfg_segments_python(
            segments,
            max_lag=max_lag,
            symmetrize=bool(sfg_cfg.get("symmetrize", True)),
        )
        timings["Segment correlation"] = time.perf_counter() - correlation_start
    else:
        supplied_velocities = trajectory_velocities_from_frames(frames, sfg_cfg=sfg_cfg)
        native_start = time.perf_counter()
        native_result = sfg_ssvvcf(
            contiguous_frame_positions(frames),
            supplied_velocities,
            oxygen_indices,
            hydrogen_indices,
            zrefs,
            dt_ps=dt_ps,
            max_lag=max_lag,
            oh_cutoff=float(sfg_cfg.get("oh_cutoff", sfg_cfg.get("bond_cutoff", 1.25))),
            cell=cell,
            pbc=pbc,
            mu_mode=str(sfg_cfg.get("mu_mode", "full")),
            symmetrize=bool(sfg_cfg.get("symmetrize", True)),
            flip_sign=bool(sfg_cfg.get("flip_sign", False)),
            duplicate_policy=str(sfg_cfg.get("duplicate_hydrogen_policy", "nearest")),
            window=sfg_cfg.get("window"),
        )
        native_wall = time.perf_counter() - native_start
        if native_result is None:
            raise RuntimeError("C++ SFG backend is not available.")
        sums, counts, native_stages = native_result
        timings["O-H assignment"] = float(native_stages[0])
        timings["Velocity preparation"] = float(native_stages[1])
        timings["Segment signal construction"] = float(native_stages[2])
        timings["Segment correlation"] = float(native_stages[3])
        timings["Other overhead"] += native_wall - float(np.sum(native_stages))

    finalize_start = time.perf_counter()
    corr = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    timings["Finalize correlation"] = time.perf_counter() - finalize_start
    timings["Other overhead"] += time.perf_counter() - total_start - sum(timings.values())
    timings["Total"] = time.perf_counter() - total_start
    result = {
        "frames": len(frames),
        "counts": counts.tolist(),
        "corr": corr.tolist(),
        "zrefs": zrefs.tolist(),
    }
    return result, timings


if __name__ == "__main__":
    main()
