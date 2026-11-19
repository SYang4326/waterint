"""Deterministic synthetic slab + water trajectories for tests (no external data needed).

Rock-salt-like slab (cation 'Mg' / lattice O) at the bottom, water above with a few OH-, one H3O+,
surface O-H groups, and molecules straddling the periodic x/y boundaries. Atom types: 1 H, 2 Mg, 3 O.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

TYPE_MAP = {1: "H", 2: "Mg", 3: "O"}


def _unit(rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(size=3)
    return v / np.linalg.norm(v)


def make_system(seed: int = 0, n_water: int = 40, box=(12.6, 12.6, 30.0)):
    rng = np.random.default_rng(seed)
    lx, ly, lz = box
    a = lx / 6
    pos, typ, mol = [], [], []
    for layer, z in enumerate((1.0, 3.1)):
        for i in range(6):
            for j in range(6):
                cation = (i + j + layer) % 2 == 0
                pos.append([i * a + 0.3, j * a + 0.3, z]); typ.append(2 if cation else 3); mol.append(-1)
    lattice_o = [k for k, t in enumerate(typ) if t == 3 and pos[k][2] > 3.0]
    for k in lattice_o[:3]:                                   # surface Os-H
        pos.append([pos[k][0], pos[k][1], pos[k][2] + 0.98]); typ.append(1); mol.append(-1)
    placed: list[np.ndarray] = []
    for w in range(n_water):
        while True:                                           # keep O-O >= 2.6 A (minimum image in x, y)
            o = np.array([rng.uniform(0, lx), rng.uniform(0, ly), rng.uniform(5.0, 20.0)])
            if w < 4:                                         # straddle the x / y boundaries
                o[w % 2] = 0.2 if w < 2 else lx - 0.2
            ok = True
            for q in placed:
                d = o - q
                d[0] -= lx * np.rint(d[0] / lx); d[1] -= ly * np.rint(d[1] / ly)
                ok = ok and float(np.linalg.norm(d)) >= 2.6
            if ok:
                break
        placed.append(o.copy())
        n_h = 1 if w in (5, 6) else (3 if w == 7 else 2)       # OH-, OH-, H3O+
        pos.append(o.tolist()); typ.append(3); mol.append(w)
        dirs = [_unit(rng)]
        while len(dirs) < n_h:
            d = _unit(rng)
            if all(np.dot(d, e) < 0.2 for e in dirs):
                dirs.append(d)
        for d in dirs:
            h = o + 0.97 * d
            h[0] %= lx; h[1] %= ly
            pos.append(h.tolist()); typ.append(1); mol.append(w)
    return np.asarray(pos), np.asarray(typ), np.asarray(box), np.asarray(mol)


def make_frames(n_frames: int = 8, seed: int = 0, noise: float = 0.05):
    pos0, typ, box, mol = make_system(seed)
    rng = np.random.default_rng(seed + 1)
    frames = []
    n_mol = int(mol.max()) + 1
    mol_drift = np.zeros((n_mol, 3))
    for _ in range(n_frames):
        mol_drift += rng.normal(scale=0.15, size=(n_mol, 3)) * np.array([1, 1, 0.3])   # molecules move as a whole
        drift = np.where((mol >= 0)[:, None], mol_drift[np.maximum(mol, 0)], 0.0)
        p = pos0 + drift + rng.normal(scale=noise, size=pos0.shape)
        p[:, 0] %= box[0]; p[:, 1] %= box[1]
        frames.append(p)
    return frames, typ, box


def write_lammpstrj(path: Path, frames, types, box, *, dstep: int = 10) -> Path:
    path = Path(path)
    with path.open("w", encoding="utf-8") as fh:
        for f, p in enumerate(frames):
            fh.write(f"ITEM: TIMESTEP\n{f * dstep}\nITEM: NUMBER OF ATOMS\n{len(types)}\n")
            fh.write(f"ITEM: BOX BOUNDS pp pp pp\n0 {box[0]}\n0 {box[1]}\n0 {box[2]}\nITEM: ATOMS id type x y z\n")
            for i, (t, r) in enumerate(zip(types, p)):
                fh.write(f"{i + 1} {int(t)} {r[0]:.6f} {r[1]:.6f} {r[2]:.6f}\n")
    return path
