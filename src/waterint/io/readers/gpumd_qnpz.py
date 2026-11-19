"""Reader for GPUMD quantized-npz trajectories (``md_xyz2npz.py`` format).

One converted segment is a directory with ``meta.json`` and ``chunk_00000.npz`` ...;
each chunk holds ``d`` (int32, (nf, N, 3)) position deltas in units of ``quant_A``
whose first row is absolute (format v2: same integers stored as q0/Lq/k/dS/vS
byte planes, decoded here to identical arrays), optionally ``v`` (int32 velocities in units of
``quant_vel_A_per_fs``) and ``e`` (energies). Decoding matches
``md_xyz2npz.iter_frames``: ``positions = cumsum(d, axis=0) / (1 / quant_A)``.
Velocities are returned in A/ps (LAMMPS metal units).

I/O: chunks are opened with a large read buffer (GPFS single-stream throughput grows
with request size: ~17 MB/s at 1 MB, ~44 MB/s at 256 MB on a parallel file system) and can be decoded
ahead in background threads (file reads and zlib release the GIL), so reading overlaps
with the consumer's computation. The decoded values do not depend on either setting.
"""
from __future__ import annotations

import json
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

READ_BUFFER = 64 << 20
FORMAT_V1 = "int32 deltas within chunk, first row absolute"
FORMAT_V2 = ("int32 deltas within chunk, first row absolute; v2 members q0, Lq, k (image jumps), "
             "dS/vS = uint8 byte planes (W, N, 3, nf) of atom-major int16/int32 arrays")


def _from_planes(p: np.ndarray) -> np.ndarray:
    """qnpz v2 byte planes (W, N, 3, nf) uint8 -> (nf, N, 3) int64 (W = 2 int16 or 4 int32, little-endian)."""
    if p.dtype != np.uint8 or p.ndim != 4 or p.shape[0] not in (2, 4):
        raise ValueError(f"bad byte-plane array {p.dtype} {p.shape}")
    t = np.ascontiguousarray(np.moveaxis(p, 0, -1)).view("<i2" if p.shape[0] == 2 else "<i4")[..., 0]
    return np.ascontiguousarray(t.transpose(2, 0, 1)).astype(np.int64)


def _deltas(data) -> np.ndarray:
    """int32 deltas d (nf, N, 3), first row absolute, from a v1 (d) or v2 (q0, Lq, k, dS) chunk."""
    if "dS" not in data.files:
        return data["d"]
    q0 = data["q0"].astype(np.int64); lq = data["Lq"].astype(np.int64); k = data["k"].astype(np.int64)
    du = _from_planes(data["dS"])
    if du.shape != k.shape or du.shape[1:] != q0.shape:
        raise ValueError(f"inconsistent v2 chunk: du {du.shape} k {k.shape} q0 {q0.shape}")
    d = np.empty((du.shape[0] + 1,) + q0.shape, dtype=np.int64)
    d[0] = q0
    d[1:] = du + k * lq
    return d.astype(np.int32)


@dataclass(frozen=True)
class QnpzSegment:
    path: Path
    meta: dict

    @property
    def n_frames(self) -> int:
        return int(self.meta["n_frames"])

    @property
    def n_chunks(self) -> int:
        return int(self.meta["n_chunks"])


def open_segment(path: str | Path) -> QnpzSegment:
    root = Path(path)
    meta_path = root / "meta.json"
    if not meta_path.is_file() and (root / "npz_4fs" / "meta.json").is_file():
        root = root / "npz_4fs"
        meta_path = root / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    for key in ("species", "cell", "dt_ps", "quant_A", "n_frames", "n_chunks"):
        if key not in meta:
            raise ValueError(f"{meta_path} lacks {key!r}.")
    if meta.get("format", FORMAT_V1) not in (FORMAT_V1, FORMAT_V2):
        raise ValueError(f"Unsupported qnpz format in {meta_path}: {meta.get('format')}")
    return QnpzSegment(root, meta)


def check_compatible(segments: Sequence[QnpzSegment]) -> None:
    first = segments[0].meta
    for seg in segments[1:]:
        for key in ("species", "cell", "dt_ps", "quant_A"):
            if seg.meta[key] != first[key]:
                raise ValueError(f"Segment {seg.path} differs from {segments[0].path} in {key}.")


def species_types(species: Sequence[str], type_map: dict[str, int]) -> np.ndarray:
    try:
        return np.asarray([type_map[s] for s in species], dtype=np.int16)
    except KeyError as error:
        raise ValueError(f"Species {error} has no type code.") from None


def decode_chunk(segment: QnpzSegment, index: int, *, with_velocities: bool) -> tuple[np.ndarray, np.ndarray | None]:
    q_scale = 1.0 / float(segment.meta["quant_A"])  # == md_xyz2npz.Q (1000.0)
    with open(segment.path / f"chunk_{index:05d}.npz", "rb", buffering=READ_BUFFER) as handle, np.load(handle) as data:
        positions = np.cumsum(_deltas(data), axis=0) / q_scale
        velocities = None
        vkey = "v" if "v" in data.files else "vS" if "vS" in data.files else None
        if with_velocities and vkey is not None:
            qv = 1.0 / float(segment.meta["quant_vel_A_per_fs"])
            v = data["v"] if vkey == "v" else _from_planes(data["vS"]).astype(np.int32)
            velocities = v / qv * 1000.0  # A/fs -> A/ps
    return positions, velocities


def iter_blocks(segments: Sequence[QnpzSegment], *, with_velocities: bool = False, max_frames: int | None = None,
                prefetch: int = 2) -> Iterator[tuple[int, int, np.ndarray, np.ndarray | None]]:
    """Yield (segment_index, first_global_frame, positions, velocities) chunk by chunk, in order.

    ``prefetch`` > 0 decodes that many chunks ahead in background threads; output is identical to prefetch=0.
    """
    check_compatible(segments)
    jobs = [(si, seg, ic) for si, seg in enumerate(segments) for ic in range(seg.n_chunks)]
    pool = ThreadPoolExecutor(max_workers=prefetch) if prefetch > 0 else None
    pending: deque = deque()
    submitted = 0

    def submit_until(limit: int) -> None:
        nonlocal submitted
        while submitted < min(limit, len(jobs)):
            _, seg, ic = jobs[submitted]
            if pool is None:
                pending.append(decode_chunk(seg, ic, with_velocities=with_velocities))
            else:
                pending.append(pool.submit(decode_chunk, seg, ic, with_velocities=with_velocities))
            submitted += 1

    global_frame = 0
    seen = [0] * len(segments)
    try:
        for k, (si, seg, ic) in enumerate(jobs):
            submit_until(k + 1 + max(prefetch, 0))
            item = pending.popleft()
            pos, vel = item if pool is None else item.result()
            seen[si] += pos.shape[0]
            if max_frames is not None and global_frame + pos.shape[0] > max_frames:
                keep = max_frames - global_frame
                pos = pos[:keep]
                vel = None if vel is None else vel[:keep]
            if pos.shape[0]:
                yield si, global_frame, pos, vel
            global_frame += pos.shape[0]
            if max_frames is not None and global_frame >= max_frames:
                return
            if ic == seg.n_chunks - 1 and seen[si] != seg.n_frames:
                raise ValueError(f"{seg.path}: chunks hold {seen[si]} frames, meta.json says {seg.n_frames}.")
    finally:
        if pool is not None:
            for fut in pending:
                fut.cancel()
            pool.shutdown(wait=True)


def last_frame(segment: QnpzSegment) -> np.ndarray:
    """Positions of the final frame of a segment (used to seed the boundary interval)."""
    pos, _ = decode_chunk(segment, segment.n_chunks - 1, with_velocities=False)
    return pos[-1]


def run_segments(path: str | Path) -> list[QnpzSegment]:
    """A segment directory (meta.json or npz_4fs/meta.json) or a run directory holding seg_* segment directories."""
    root = Path(path)
    if (root / "meta.json").is_file() or (root / "npz_4fs" / "meta.json").is_file():
        return [open_segment(root)]
    segs = sorted(p for p in root.glob("seg_*") if p.is_dir())
    if not segs:
        raise FileNotFoundError(f"{root}: neither a qnpz segment nor a run directory with seg_* segments")
    return [open_segment(p) for p in segs]


def iter_trajectory_frames(path: str | Path, *, type_map: dict | None = None, first_frame: int = 0, max_frames: int | None = None,
                           stride: int = 1, velocities: bool = True, wrap_xy: bool = True, dstep: int = 8,
                           max_segments: int | None = None, prefetch: int = 2):
    """TrajectoryFrame stream for the frame-based workflows (density, hbond, sfg, msd, rdf, ...).

    Frames are numbered over the segments in order; only the chunks that contain frames first_frame + k * stride are
    decoded (prefetched in threads). x, y are wrapped into the cell (as LAMMPS dumps); velocities in A/ps when stored."""
    from waterint.io.frame import TrajectoryFrame
    segs = run_segments(path)
    if max_segments is not None:
        segs = segs[:int(max_segments)]
    check_compatible(segs)
    meta = segs[0].meta
    species = list(meta["species"])
    c = meta["cell"]
    cell = (float(c[0]), float(c[1]), float(c[2])) if len(c) == 3 else (float(c[0]), float(c[4]), float(c[8]))
    if type_map:
        inv = {str(v): int(k) for k, v in type_map.items()}
    else:  # no type map: number the elements in order of first appearance (symbols carry the identity)
        inv = {}
        for s in species:
            inv.setdefault(str(s), len(inv) + 1)
    missing = sorted(set(species) - set(inv))
    if missing:
        raise ValueError(f"species {missing} have no type in input.type_map")
    types = np.array([inv[s] for s in species], dtype=np.int64)
    jobs = []
    g = 0
    for s in segs:
        ch = int(s.meta["chunk"])
        for ic in range(s.n_chunks):
            c0, c1 = g + ic * ch, min(g + (ic + 1) * ch, g + s.n_frames)
            last = None if max_frames is None else first_frame + (max_frames - 1) * stride
            if c1 <= first_frame or (last is not None and c0 > last):
                continue
            jobs.append((s, ic, c0))
        g += s.n_frames
    pool = ThreadPoolExecutor(max_workers=prefetch) if prefetch > 0 else None
    want_v = bool(velocities) and bool(meta.get("has_vel", True))
    futures = deque()
    k = 0
    try:
        for j in range(min(prefetch, len(jobs))):
            futures.append(pool.submit(decode_chunk, jobs[j][0], jobs[j][1], with_velocities=want_v))
        for j, (s, ic, c0) in enumerate(jobs):
            if pool is not None:
                pos, vel = futures.popleft().result()
                if j + prefetch < len(jobs):
                    nx = jobs[j + prefetch]
                    futures.append(pool.submit(decode_chunk, nx[0], nx[1], with_velocities=want_v))
            else:
                pos, vel = decode_chunk(s, ic, with_velocities=want_v)
            for r in range(pos.shape[0]):
                gi = c0 + r
                if gi < first_frame or (gi - first_frame) % stride:
                    continue
                p = pos[r].copy()
                if wrap_xy:
                    p[:, 0] %= cell[0]
                    p[:, 1] %= cell[1]
                yield TrajectoryFrame(index=gi, comment=f"qnpz frame {gi}", symbols=species, positions=p, cell=cell,
                                      step=gi * dstep, types=types, velocities=None if vel is None else vel[r].astype(np.float64))
                k += 1
                if max_frames is not None and k >= max_frames:
                    return
    finally:
        if pool is not None:
            for f in futures:
                f.cancel()
            pool.shutdown(wait=True)
