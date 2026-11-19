"""Single entry point for reading trajectories.

Every analysis receives frames from :class:`Trajectory`; no analysis opens files itself.
Single-file reading keeps the exact v0.6 semantics (stride, max_frames, start_timestep);
a list of files (restart segments) is concatenated with duplicate timesteps removed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from waterint.io.frame import TrajectoryFrame

FORMATS = ("xyz", "lammpstrj", "npz", "gpumd-qnpz")


@dataclass(frozen=True)
class TrajectorySpec:
    paths: tuple[Path, ...]
    format: str = "xyz"
    type_map: dict[int, str] = field(default_factory=dict)
    stride: int = 1
    max_frames: int | None = None
    start_timestep: int | None = None
    reader: str = "python"
    options: dict[str, Any] = field(default_factory=dict)  # format-specific (qnpz: first_frame, prefetch, ...)

    @classmethod
    def from_config(cls, input_cfg: dict[str, Any], base_dir: str | Path | None = None) -> "TrajectorySpec":
        raw = input_cfg.get("trajectory", input_cfg.get("trajectories"))
        if raw is None:
            raise ValueError("input.trajectory is required.")
        items = raw if isinstance(raw, list) else [raw]
        base = Path(base_dir) if base_dir is not None else None
        paths = tuple((Path(p) if Path(p).is_absolute() or base is None else base / p) for p in items)
        fmt = str(input_cfg.get("format", "xyz")).lower()
        if fmt == "qnpz":
            fmt = "gpumd-qnpz"
        if fmt not in FORMATS:
            raise ValueError(f"input.format must be one of {FORMATS}.")
        stride = int(input_cfg.get("stride", 1))
        if stride <= 0:
            raise ValueError("input.stride must be > 0.")
        max_raw = input_cfg.get("max_frames")
        max_frames = None if max_raw in {None, 0, "all"} else int(max_raw)
        if max_frames is not None and max_frames <= 0:
            raise ValueError("input.max_frames must be positive, 0, 'all', or omitted.")
        start_raw = input_cfg.get("start_timestep")
        start = None if start_raw in {None, ""} else int(start_raw)
        known = {"trajectory", "trajectories", "format", "type_map", "stride", "max_frames", "start_timestep", "reader"}
        options = {k: v for k, v in input_cfg.items() if k not in known}
        type_map = {int(k): str(v) for k, v in (input_cfg.get("type_map") or {}).items()}
        return cls(paths=paths, format=fmt, type_map=type_map, stride=stride, max_frames=max_frames,
                   start_timestep=start, reader=str(input_cfg.get("reader", "python")), options=options)

    def legacy_input_cfg(self) -> dict[str, Any]:
        cfg: dict[str, Any] = {"format": self.format, "type_map": dict(self.type_map), "stride": self.stride,
                               "max_frames": self.max_frames, "start_timestep": self.start_timestep,
                               "reader": self.reader}
        cfg.update(self.options)
        return cfg


class Trajectory:
    """Iterable over frames of one trajectory (one file, a qnpz run, or a list of restart segments)."""

    def __init__(self, spec: TrajectorySpec):
        self.spec = spec

    @property
    def label(self) -> str:
        return str(self.spec.paths[0]) if len(self.spec.paths) == 1 else f"{self.spec.paths[0]} (+{len(self.spec.paths) - 1} segments)"

    def __iter__(self) -> Iterator[TrajectoryFrame]:
        return self.iter_frames()

    def iter_frames(self) -> Iterator[TrajectoryFrame]:
        if len(self.spec.paths) == 1:
            yield from _iter_single(self.spec.paths[0], self.spec.legacy_input_cfg())
            return
        yield from self._iter_segments()

    def iter_blocks(self, block_frames: int = 1000, *, velocities: bool = False):
        """Arrays of many frames at once (see waterint.io.blocks)."""
        from waterint.io.blocks import iter_blocks
        yield from iter_blocks(self.spec, block_frames=block_frames, velocities=velocities)

    def first_frame(self) -> TrajectoryFrame:
        return next(iter(self.iter_frames()))

    def _iter_segments(self) -> Iterator[TrajectoryFrame]:
        """Concatenate restart segments: drop frames whose timestep was already seen, then apply
        start_timestep / stride / max_frames to the deduplicated sequence."""
        raw_cfg = self.spec.legacy_input_cfg()
        raw_cfg.update({"stride": 1, "max_frames": None, "start_timestep": None})
        last_step: int | None = None
        index = 0
        yielded = 0
        for path in self.spec.paths:
            for frame in _iter_single(path, raw_cfg):
                if frame.step is not None and last_step is not None and frame.step <= last_step:
                    continue
                if frame.step is not None:
                    last_step = frame.step
                if self.spec.start_timestep is not None and (frame.step is None or frame.step < self.spec.start_timestep):
                    continue
                keep = index % self.spec.stride == 0
                index += 1
                if not keep:
                    continue
                yield _with_index(frame, yielded)
                yielded += 1
                if self.spec.max_frames is not None and yielded >= self.spec.max_frames:
                    return


def open_trajectory(input_cfg: dict[str, Any], base_dir: str | Path | None = None) -> Trajectory:
    return Trajectory(TrajectorySpec.from_config(input_cfg, base_dir))


def _with_index(frame: TrajectoryFrame, index: int) -> TrajectoryFrame:
    from dataclasses import replace
    return replace(frame, index=index)


def _iter_single(traj_path: Path, input_cfg: dict[str, Any]) -> Iterator[TrajectoryFrame]:
    """v0.6 ``iter_frames`` semantics for one path (kept bitwise for regression)."""
    fmt = str(input_cfg.get("format", "xyz")).lower()
    stride = int(input_cfg.get("stride", 1) or 1)
    max_frames = input_cfg.get("max_frames")
    max_frames = None if max_frames in {None, 0, "all"} else int(max_frames)
    start_timestep = input_cfg.get("start_timestep")
    start_timestep = None if start_timestep in {None, ""} else int(start_timestep)

    if fmt == "xyz":
        from waterint.io.readers.xyz import read_xyz
        reader_max = max_frames * stride if (start_timestep is None and max_frames is not None) else None
        frames = read_xyz(traj_path, reader=str(input_cfg.get("reader", "python")), max_frames=reader_max)
    elif fmt == "lammpstrj":
        from waterint.io.readers.lammpstrj import read_lammpstrj
        yield from read_lammpstrj(traj_path, type_map=input_cfg.get("type_map", {}), start_timestep=start_timestep,
                                  stride=stride, max_frames=max_frames, reader=str(input_cfg.get("reader", "python")))
        return
    elif fmt == "npz":
        from waterint.io.readers.npz import read_npz
        frames = read_npz(traj_path, type_map=input_cfg.get("type_map", {}))
    elif fmt == "gpumd-qnpz":
        from waterint.io.readers.gpumd_qnpz import iter_trajectory_frames
        if start_timestep is not None:
            raise ValueError("gpumd-qnpz input uses input.first_frame, not start_timestep.")
        yield from iter_trajectory_frames(
            traj_path, type_map=input_cfg.get("type_map") or None, first_frame=int(input_cfg.get("first_frame", 0)),
            max_frames=max_frames, stride=stride, velocities=bool(input_cfg.get("velocities", True)),
            wrap_xy=bool(input_cfg.get("wrap_xy", True)), dstep=int(input_cfg.get("dstep", 8)),
            max_segments=input_cfg.get("max_segments"), prefetch=int(input_cfg.get("prefetch", 2)))
        return
    else:
        raise ValueError("input.format must be xyz, lammpstrj, npz, or gpumd-qnpz.")

    yielded = 0
    for frame in frames:
        if start_timestep is not None:
            if frame.step is None:
                raise ValueError("input.start_timestep requires trajectory frames with timestep information.")
            if frame.step < start_timestep:
                continue
        if frame.index % stride != 0:
            continue
        yield frame
        yielded += 1
        if max_frames is not None and yielded >= max_frames:
            return
