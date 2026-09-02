from __future__ import annotations

from pathlib import Path

import numpy as np

from waterint._02_computation.defect_transport import DefectMsdResult, DefectTrackingResult


def write_defect_msd_csv(path: Path, result: DefectMsdResult) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("lag_frames,time_ps,defect_msd_A2,segment_origin_samples\n")
        for lag, time, value, samples in zip(
            result.lag_frames, result.time_ps, result.msd_a2, result.samples
        ):
            handle.write(f"{lag},{time:.10g},{value:.10g},{samples}\n")


def write_defect_tracks_csv(path: Path, tracking: DefectTrackingResult, steps: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("track_id,frame,timestep,atom_index,x_unwrapped_A,y_unwrapped_A,z_unwrapped_A\n")
        for segment in sorted(tracking.segments, key=lambda item: item.track_id):
            for frame, atom, position in zip(
                segment.frame_indices, segment.atom_indices, segment.positions_a
            ):
                handle.write(
                    f"{segment.track_id},{frame},{int(steps[frame])},{atom},"
                    f"{position[0]:.10g},{position[1]:.10g},{position[2]:.10g}\n"
                )


def write_defect_events_csv(path: Path, tracking: DefectTrackingResult, steps: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("frame,timestep,carrier_count,matched,births,deaths\n")
        for frame in range(len(tracking.carrier_counts)):
            handle.write(
                f"{frame},{int(steps[frame])},{tracking.carrier_counts[frame]},"
                f"{tracking.matched[frame]},{tracking.births[frame]},{tracking.deaths[frame]}\n"
            )


def write_defect_current_csv(path: Path, tracking: DefectTrackingResult, steps: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("interval,start_timestep,stop_timestep,Jx_eA_per_ps,Jy_eA_per_ps,Jz_eA_per_ps\n")
        for interval, current in enumerate(tracking.charge_current_ea_per_ps):
            handle.write(
                f"{interval},{int(steps[interval])},{int(steps[interval + 1])},"
                f"{current[0]:.10g},{current[1]:.10g},{current[2]:.10g}\n"
            )
