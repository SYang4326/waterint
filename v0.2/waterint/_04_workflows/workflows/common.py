from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

from waterint.config import require_mapping
from waterint._00_io.common import TrajectoryFrame
from waterint._00_io.lammpstrj import read_lammpstrj
from waterint._00_io.npz import read_npz
from waterint._00_io.xyz import read_xyz


def resolve_path(config: dict[str, Any], path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return Path(config["_config_dir"]) / path


def parse_cell(value: Any) -> tuple[float, float, float] | None:
    if value is None or str(value).lower() == "auto":
        return None
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError("system.cell must be [Lx, Ly, Lz] in Angstrom or auto.")
    cell = tuple(float(v) for v in value)
    if any(v <= 0 for v in cell):
        raise ValueError("system.cell values must be positive.")
    return cell


def parse_range(value: Any, *, name: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{name} must be [min, max].")
    range_min, range_max = float(value[0]), float(value[1])
    if not range_max > range_min:
        raise ValueError(f"{name} max must be larger than min.")
    return range_min, range_max


def iter_frames(traj_path: Path, input_cfg: dict[str, Any]) -> Iterator[TrajectoryFrame]:
    fmt = str(input_cfg.get("format", "xyz")).lower()
    stride = int(input_cfg.get("stride", 1))
    max_frames_raw = input_cfg.get("max_frames", None)
    max_frames = None if max_frames_raw in {None, 0, "all"} else int(max_frames_raw)
    start_timestep_raw = input_cfg.get("start_timestep", None)
    start_timestep = None if start_timestep_raw in {None, ""} else int(start_timestep_raw)
    if stride <= 0:
        raise ValueError("input.stride must be > 0.")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("input.max_frames must be positive, 0, 'all', or omitted.")

    if fmt == "xyz":
        frames = read_xyz(traj_path)
    elif fmt == "lammpstrj":
        yield from read_lammpstrj(
            traj_path,
            type_map=input_cfg.get("type_map", {}),
            start_timestep=start_timestep,
            stride=stride,
            max_frames=max_frames,
        )
        return
    elif fmt == "npz":
        frames = read_npz(traj_path, type_map=input_cfg.get("type_map", {}))
    else:
        raise ValueError("input.format must be xyz, lammpstrj, or npz.")

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


def required_workflow_sections(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    input_cfg = require_mapping(config, "input")
    system_cfg = require_mapping(config, "system")
    output_cfg = require_mapping(config, "output")
    return input_cfg, system_cfg, output_cfg
