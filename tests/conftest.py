from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


@pytest.fixture(scope="session")
def synthetic_traj(tmp_path_factory):
    from synthetic import make_frames, write_lammpstrj
    frames, types, box = make_frames(n_frames=8)
    return write_lammpstrj(tmp_path_factory.mktemp("traj") / "slab_water.lammpstrj", frames, types, box)
