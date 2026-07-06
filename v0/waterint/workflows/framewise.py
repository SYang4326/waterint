from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Generic, TypeVar

from waterint.io.common import TrajectoryFrame
from waterint.workflows.common import iter_frames


StateT = TypeVar("StateT")
ResultT = TypeVar("ResultT")


@dataclass(frozen=True)
class FramewiseRunInfo:
    traj_path: Path
    frames: int
    cell: tuple[float, float, float]


@dataclass(frozen=True)
class FramewiseResult(Generic[ResultT]):
    result: ResultT
    info: FramewiseRunInfo


def run_framewise_analysis(
    *,
    traj_path: Path,
    input_cfg: dict[str, Any],
    configured_cell: tuple[float, float, float] | None,
    state: StateT,
    accumulate: Callable[[StateT, TrajectoryFrame, tuple[float, float, float]], None],
    finalize: Callable[[StateT, tuple[float, float, float]], ResultT],
) -> FramewiseResult[ResultT]:
    frames = 0
    cell = configured_cell
    for frame in iter_frames(traj_path, input_cfg):
        if cell is None:
            if frame.cell is None:
                raise ValueError("system.cell is auto, but the trajectory did not provide cell information.")
            cell = frame.cell
        accumulate(state, frame, cell)
        frames += 1

    if frames == 0:
        raise ValueError(f"No frames found in trajectory: {traj_path}")
    if cell is None:
        raise ValueError("No cell information was available. Set system.cell manually.")

    return FramewiseResult(
        result=finalize(state, cell),
        info=FramewiseRunInfo(traj_path=traj_path, frames=frames, cell=cell),
    )
