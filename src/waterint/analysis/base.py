"""Analysis contract.

    class MyAnalysis(Analysis):
        type = "my-analysis"
        description = "..."
        params = {"bins": (200, "number of bins"), ...}      # documented defaults (docs generator)
        def update(self, lf: LabelledFrame) -> None: ...     # kind "stream": once per frame
        def update_block(self, block) -> None: ...            # kind "block": arrays of many frames (batch kernels)
        # kind "post": no trajectory; finalize() reads inputs (results of earlier analyses or files)
        def finalize(self) -> Result: ...
        def write(self, result, outdir) -> dict               # optional: extra CSVs (npz + metadata always)
        def plot(self, result, outdir) -> list[Path]          # optional: figures from the Result only

Analyses never read trajectories, never re-implement neighbour search / ownership / species / H bonds /
reference planes (they ask the LabelledFrame), and never compute physics in ``plot``.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from waterint.prep.labelled import LabelledFrame, PrepContext
from waterint.prep.reference import CoordinateSpec, coordinate_spec_from_config
from waterint.system.profile import OwnershipParams


@dataclass
class RunContext:
    prep: PrepContext
    trajectory: str
    config_dir: Path
    output_dir: Path
    output_cfg: dict[str, Any] = field(default_factory=dict)
    config_path: str | None = None
    input_spec: Any = None                                     # io.trajectory.TrajectorySpec (None for post-only runs)
    results: dict[str, "Result"] = field(default_factory=dict)  # results of analyses finished earlier in this run
    open_trajectory: Any = None                                 # callable(input_cfg) -> io Trajectory (pipeline-provided)
    count_frames: Any = None                                    # callable() -> frames of this run's input (pipeline-provided)

    def path(self, value: str | Path) -> Path:
        p = Path(value)
        return p if p.is_absolute() else self.config_dir / p


@dataclass
class Result:
    analysis: str
    name: str
    arrays: dict[str, np.ndarray]
    meta: dict[str, Any]
    outputs: dict[str, Any] = field(default_factory=dict)

    def save_npz(self, path: Path) -> None:
        np.savez_compressed(path, **{k: np.asarray(v) for k, v in self.arrays.items()})

    @classmethod
    def load(cls, prefix_path: Path) -> "Result":
        meta = json.loads(Path(f"{prefix_path}_metadata.json").read_text(encoding="utf-8"))
        with np.load(f"{prefix_path}_result.npz", allow_pickle=False) as data:
            arrays = {k: data[k] for k in data.files}
        return cls(analysis=meta["analysis_name"], name=meta.get("name", ""), arrays=arrays, meta=meta,
                   outputs=meta.get("outputs", {}))


class Analysis:
    type: ClassVar[str] = ""
    kind: ClassVar[str] = "stream"          # stream | block | post
    needs_velocities: ClassVar[bool] = False
    description: ClassVar[str] = ""
    params: ClassVar[dict[str, tuple[Any, str]]] = {}
    v06_name: ClassVar[str | None] = None   # v0.6 module name, for config migration

    def __init__(self, cfg: dict[str, Any], run: RunContext):
        self.cfg = cfg
        self.run = run
        self.name = str(cfg.get("name") or cfg.get("prefix") or self.type)
        self.frames = 0
        self.first_cell: tuple[float, float, float] | None = None
        self.first_cell_vectors: np.ndarray | None = None
        self.seconds = 0.0
        self.backends_used: dict[str, str] = {}

    # -- helpers shared by analyses -------------------------------------------------------------
    @property
    def profile(self):
        return self.run.prep.profile

    def ownership_params(self, overrides: dict[str, Any] | None = None) -> OwnershipParams:
        """Profile ownership, optionally overridden by ``cfg.ownership`` (and ``overrides``)."""
        params = self.profile.ownership.replace(self.cfg.get("ownership"))
        return params.replace(overrides)

    def coordinate(self, coord_cfg: dict[str, Any] | None) -> CoordinateSpec:
        if not coord_cfg or (set(coord_cfg) <= {"range", "bins"}):
            return self.run.prep.default_coordinate
        return coordinate_spec_from_config(coord_cfg)

    def observe_frame(self, lf: LabelledFrame) -> None:
        if self.frames == 0:
            self.first_cell = self.run.prep.configured_cell or lf.frame.cell
            if lf.frame.cell_vectors is not None:
                self.first_cell_vectors = np.asarray(lf.frame.cell_vectors, dtype=float)

    def step(self, lf: LabelledFrame) -> None:
        t0 = time.perf_counter()
        self.observe_frame(lf)
        self.update(lf)
        self.frames += 1
        self.seconds += time.perf_counter() - t0

    def step_block(self, block) -> None:
        t0 = time.perf_counter()
        self.update_block(block)
        self.frames += 0 if block.seed else block.n
        self.seconds += time.perf_counter() - t0

    # -- contract ---------------------------------------------------------------------------------
    def update(self, lf: LabelledFrame) -> None:
        raise NotImplementedError

    def update_block(self, block) -> None:
        raise NotImplementedError

    def finalize(self) -> Result:
        raise NotImplementedError

    def write(self, result: Result, outdir: Path) -> dict[str, Any]:
        return {}

    def plot(self, result: Result, outdir: Path) -> dict[str, Any]:
        return {}

    def metadata(self, result: Result) -> dict[str, Any]:
        return {
            "analysis_name": self.type, "name": self.name, "package": "waterint", "version": _version(),
            "config_file": self.run.config_path, "trajectory": self.run.trajectory, "frames": self.frames,
            "profile": self.profile.profile_id, "backends": self.backends_used,
            "prep_backends": dict(self.run.prep.backends_used), "seconds": round(self.seconds, 3),
            "config": self.cfg, **result.meta,
        }


def _version() -> str:
    from waterint import __version__
    return __version__


def save_result(analysis: Analysis, result: Result, outdir: Path, *, plot: bool = True) -> Result:
    outdir.mkdir(parents=True, exist_ok=True)
    prefix = outdir / analysis.name
    outputs: dict[str, Any] = {}
    if result.arrays:
        result.save_npz(Path(f"{prefix}_result.npz"))
        outputs["npz"] = f"{prefix}_result.npz"
    outputs.update(analysis.write(result, outdir))
    if plot:
        outputs.update(analysis.plot(result, outdir))
    result.outputs = outputs
    meta = analysis.metadata(result)
    meta["outputs"] = outputs
    Path(f"{prefix}_metadata.json").write_text(json.dumps(_jsonable(meta), indent=2) + "\n", encoding="utf-8")
    return result


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items() if not str(k).startswith("_")}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value
