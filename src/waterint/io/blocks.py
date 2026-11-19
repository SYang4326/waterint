"""Block access to trajectories: arrays of many frames at once, for batch (C++/GPU) kernels.

Every format can be read as blocks:
  lammpstrj   C++ streaming parser (first-frame atom order), files of a list concatenated with timestep dedup
  gpumd-qnpz  native chunks of one or more segments (run directories expand to their seg_* segments),
              optionally seeded with the last frame of the previous segment (``seed_segment``)
  xyz / npz   frames batched into blocks
``Block.frames()`` turns a block into TrajectoryFrames identical to the frame readers', so frame analyses
and block analyses can share one pass.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from waterint.io.cell import orthorhombic_cell_vectors
from waterint.io.frame import TrajectoryFrame


@dataclass
class Block:
    first: int                          # global index of the first frame in this block
    positions: np.ndarray               # [n, natoms, 3] float64
    steps: np.ndarray                   # [n] int64 MD steps
    symbols: list[str]
    types: np.ndarray | None = None     # [natoms]
    ids: np.ndarray | None = None       # [natoms]
    velocities: np.ndarray | None = None
    boxes: np.ndarray | None = None     # [n, 6] lo/hi bounds (lammpstrj)
    cell: tuple[float, float, float] | None = None   # orthorhombic lengths (qnpz)
    segment: int = 0
    seed: bool = False                  # the seed frame (last frame of the previous segment)
    wrap_xy: bool = False               # wrap x, y into the cell when converting to frames (qnpz frame semantics)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(self.positions.shape[0])

    def frames(self) -> list[TrajectoryFrame]:
        if "frames" in self.meta:
            return list(self.meta["frames"])
        out = []
        for r in range(self.n):
            p = self.positions[r]
            if self.boxes is not None:
                lo = self.boxes[r, 0::2]
                cell = tuple(float(v) for v in (self.boxes[r, 1::2] - lo))
                origin = tuple(float(v) for v in lo)
                p = p.copy()
            else:
                cell, origin = self.cell, None
                p = p.copy()
                if self.wrap_xy and cell is not None:
                    p[:, 0] %= cell[0]
                    p[:, 1] %= cell[1]
            gi = self.first + r
            vel = None if self.velocities is None else np.asarray(self.velocities[r], dtype=np.float64)
            out.append(TrajectoryFrame(
                index=gi, comment=self.meta.get("comment_fmt", "frame {i}").format(i=gi, step=int(self.steps[r])),
                symbols=self.symbols, positions=p, cell=cell, step=int(self.steps[r]),
                types=None if self.types is None else np.asarray(self.types, dtype=int), velocities=vel,
                cell_vectors=None if self.boxes is None else orthorhombic_cell_vectors(cell),
                cell_origin=origin, triclinic=False))
        return out


def iter_blocks(spec: Any, *, block_frames: int = 1000, velocities: bool = False) -> Iterator[Block]:
    """Blocks of a TrajectorySpec (stride and start_timestep are applied; max_frames counts kept frames)."""
    fmt = spec.format
    if fmt == "lammpstrj":
        yield from _apply_selection(_lammpstrj_blocks(spec, block_frames), spec, raw_counts=False)
    elif fmt == "gpumd-qnpz":
        yield from _apply_selection(_qnpz_blocks(spec, velocities), spec, raw_counts=False)
    else:
        yield from _frame_blocks(spec, block_frames)


# ------------------------------------------------------------------------------------------------- lammpstrj
def _lammpstrj_blocks(spec, block_frames: int) -> Iterator[Block]:
    from waterint.io.readers.lammpstrj_blocks import iter_lammpstrj_blocks
    tmap = spec.type_map
    last_step = None
    index = 0
    for path in spec.paths:
        for b in iter_lammpstrj_blocks(path, block_frames=block_frames):
            keep = np.ones(b.steps.size, dtype=bool)
            if last_step is not None and len(spec.paths) > 1:   # restart segments: drop repeated timesteps
                keep &= b.steps > last_step
            if not keep.any():
                continue
            symbols = [tmap.get(int(t), str(int(t))) for t in b.types]
            n = int(keep.sum())
            yield Block(first=index, positions=b.positions[keep], steps=b.steps[keep].astype(np.int64), symbols=symbols,
                        types=b.types.astype(np.int64), ids=b.ids, velocities=None if b.velocities is None else b.velocities[keep],
                        boxes=b.boxes[keep], meta={"comment_fmt": "timestep {step}", "source": str(path)})
            index += n
            last_step = int(b.steps[keep][-1])


# ------------------------------------------------------------------------------------------------- gpumd-qnpz
def qnpz_segments(spec) -> list[Any]:
    from waterint.io.readers import gpumd_qnpz
    segs = []
    for p in spec.paths:
        p = Path(p)
        segs.extend([gpumd_qnpz.open_segment(p)] if (p / "meta.json").is_file() else gpumd_qnpz.run_segments(p))
    if spec.options.get("max_segments") is not None:
        segs = segs[: int(spec.options["max_segments"])]
    gpumd_qnpz.check_compatible(segs)
    return segs


def qnpz_dstep(spec, meta: dict) -> int:
    if spec.options.get("dstep") is not None:
        return int(spec.options["dstep"])
    return int(round(float(meta["dt_ps"]) * 1000.0 / float(spec.options.get("md_timestep_fs", 0.5))))


def _qnpz_blocks(spec, velocities: bool) -> Iterator[Block]:
    from waterint.io.readers import gpumd_qnpz
    segs = qnpz_segments(spec)
    meta = segs[0].meta
    species = [str(s) for s in meta["species"]]
    inv = {str(v): int(k) for k, v in spec.type_map.items()} if spec.type_map else {}
    if not inv:
        for s in species:
            inv.setdefault(s, len(inv) + 1)
    types = np.array([inv.get(s, -1) for s in species], dtype=np.int64)
    c = meta["cell"]
    cell = (float(c[0]), float(c[1]), float(c[2])) if len(c) == 3 else (float(c[0]), float(c[4]), float(c[8]))
    dstep = qnpz_dstep(spec, meta)
    seed_path = spec.options.get("seed_segment")
    seed = gpumd_qnpz.open_segment(Path(seed_path)) if seed_path else None
    if seed is not None:
        gpumd_qnpz.check_compatible([seed, *segs])
    first_index = int(spec.options.get("first_frame_index", 1 if seed is not None else 0))
    common = dict(symbols=species, types=types, ids=np.arange(1, len(species) + 1, dtype=np.int32), cell=cell,
                  wrap_xy=bool(spec.options.get("wrap_xy", True)))
    m = {"dt_ps": float(meta["dt_ps"]), "dstep": dstep, "quant_A": meta.get("quant_A"), "comment_fmt": "qnpz frame {i}"}
    if seed is not None:
        if first_index < 1:
            raise ValueError("input.first_frame_index must be >= 1 when seed_segment is given.")
        yield Block(first=first_index - 1, positions=gpumd_qnpz.last_frame(seed)[None].astype(np.float64),
                    steps=np.array([(first_index - 1) * dstep], dtype=np.int64), seed=True, segment=-1, meta=m, **common)
    want_v = bool(velocities) and bool(meta.get("has_vel", True))
    lo = int(spec.options.get("first_frame", 0))
    hi = None if spec.max_frames is None else lo + (spec.max_frames - 1) * spec.stride + 1
    for si, g0, pos, vel in _qnpz_chunks(segs, want_v, int(spec.options.get("prefetch", 2)), lo, hi):
        first = first_index + g0
        steps = (first + np.arange(pos.shape[0], dtype=np.int64)) * dstep
        yield Block(first=first, positions=pos, steps=steps, velocities=vel, segment=si, meta=m, **common)


def _qnpz_chunks(segs, want_v: bool, prefetch: int, lo: int, hi: int | None):
    """(segment, first global frame, positions, velocities) of the chunks overlapping frames [lo, hi); chunks outside
    the window are never decoded. Decoding runs ``prefetch`` chunks ahead in threads (output identical to prefetch=0)."""
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor
    from waterint.io.readers import gpumd_qnpz
    jobs, g = [], 0
    for si, s in enumerate(segs):
        ch = int(s.meta["chunk"])
        for ic in range(s.n_chunks):
            c0, c1 = g + ic * ch, min(g + (ic + 1) * ch, g + s.n_frames)
            if c1 > lo and (hi is None or c0 < hi):
                jobs.append((si, s, ic, c0))
        g += s.n_frames
    pool = ThreadPoolExecutor(max_workers=prefetch) if prefetch > 0 else None
    pending: deque = deque()
    try:
        for k in range(len(jobs)):
            while len(pending) < (prefetch if pool else 0) + 1 and k + len(pending) < len(jobs):
                _si, s, ic, _c0 = jobs[k + len(pending)]
                pending.append(pool.submit(gpumd_qnpz.decode_chunk, s, ic, with_velocities=want_v) if pool
                               else gpumd_qnpz.decode_chunk(s, ic, with_velocities=want_v))
            item = pending.popleft()
            pos, vel = item.result() if pool else item
            si, s, ic, c0 = jobs[k]
            yield si, c0, pos, vel
    finally:
        if pool is not None:
            for fut in pending:
                fut.cancel()
            pool.shutdown(wait=True)


# ------------------------------------------------------------------------------------------------- frames
def _frame_blocks(spec, block_frames: int) -> Iterator[Block]:
    from waterint.io.trajectory import Trajectory
    batch: list[TrajectoryFrame] = []
    for frame in Trajectory(spec).iter_frames():
        batch.append(frame)
        if len(batch) == block_frames:
            yield _block_from_frames(batch)
            batch = []
    if batch:
        yield _block_from_frames(batch)


def _block_from_frames(frames: list[TrajectoryFrame]) -> Block:
    """Batch of reader frames; ``frames()`` returns the original frame objects (exact frame semantics)."""
    f0 = frames[0]
    vel = np.stack([f.velocities for f in frames]) if all(f.velocities is not None for f in frames) else None
    return Block(first=f0.index, positions=np.stack([f.positions for f in frames]),
                 steps=np.array([f.step if f.step is not None else f.index for f in frames], dtype=np.int64),
                 symbols=list(f0.symbols), types=f0.types, velocities=vel, cell=f0.cell, meta={"frames": frames})


def _apply_selection(blocks: Iterator[Block], spec, *, raw_counts: bool) -> Iterator[Block]:
    """Apply start_timestep / stride / max_frames to raw blocks; frame indices count raw frames (v0.6 semantics)."""
    stride, start, max_frames = spec.stride, spec.start_timestep, spec.max_frames
    offset = int(spec.options.get("first_frame", 0)) if spec.format == "gpumd-qnpz" else 0
    if stride == 1 and start is None and max_frames is None and offset == 0:
        yield from blocks
        return
    kept = 0
    for b in blocks:
        idx = b.first + np.arange(b.n)
        keep = ((idx >= offset) & ((idx - offset) % stride == 0)) if not b.seed else np.ones(b.n, bool)
        if start is not None:
            keep &= b.steps >= start
        if max_frames is not None and not b.seed:
            room = max_frames - kept
            if room <= 0:
                return
            pos = np.nonzero(keep)[0]
            if pos.size > room:
                keep[pos[room:]] = False
        if not keep.any():
            continue
        sel = np.nonzero(keep)[0]
        if not b.seed:
            kept += sel.size
        if sel.size == b.n:
            yield b
            continue
        if stride != 1:
            raise ValueError("Block analyses need contiguous frames: use input.stride: 1 (subsample inside the analysis).")
        yield Block(first=b.first + int(sel[0]), positions=b.positions[sel], steps=b.steps[sel], symbols=b.symbols,
                    types=b.types, ids=b.ids, velocities=None if b.velocities is None else b.velocities[sel],
                    boxes=None if b.boxes is None else b.boxes[sel], cell=b.cell, segment=b.segment, seed=b.seed,
                    wrap_xy=b.wrap_xy, meta=b.meta)


def count_frames(spec) -> int:
    """Number of frames the block iterator will yield (without the seed frame); cheap for qnpz (metadata) and a
    byte scan for LAMMPS dumps."""
    if spec.format == "gpumd-qnpz":
        n = sum(s.n_frames for s in qnpz_segments(spec)) - int(spec.options.get("first_frame", 0))
    elif spec.format == "lammpstrj":
        n = 0
        for path in spec.paths:
            tail = b""
            with open(path, "rb") as handle:
                for chunk in iter(lambda: handle.read(64 << 20), b""):
                    buf = tail + chunk
                    n += buf.count(b"ITEM: TIMESTEP")
                    tail = buf[-13:]
    else:
        from waterint.io.trajectory import Trajectory
        n = sum(1 for _ in Trajectory(spec).iter_frames())
        return n
    n = max(n, 0)
    if spec.stride > 1:
        n = (n + spec.stride - 1) // spec.stride
    return n if spec.max_frames is None else min(n, spec.max_frames)
