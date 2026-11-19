"""Architecture rules of docs/ARCHITECTURE_v0.7.md, enforced as tests."""
from __future__ import annotations

import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "waterint"
LAYER_FORBIDDEN = {
    # layer: modules it must not import
    "io": ("waterint.system", "waterint.prep", "waterint.analysis", "waterint.pipeline", "waterint.plot"),
    "system": ("waterint.prep", "waterint.analysis", "waterint.pipeline", "waterint.plot"),
    "prep": ("waterint.analysis", "waterint.pipeline", "waterint.plot", "waterint.io.readers"),
    "analysis": ("waterint.pipeline", "waterint.io.readers", "waterint.io.trajectory"),
    "plot": ("waterint.prep", "waterint.analysis", "waterint.pipeline", "waterint.io", "waterint.backends"),
}


def _py_files(sub: str):
    return [p for p in (SRC / sub).rglob("*.py")]


def test_layer_dependencies():
    bad = []
    for layer, forbidden in LAYER_FORBIDDEN.items():
        for path in _py_files(layer):
            text = path.read_text(encoding="utf-8")
            for mod in forbidden:
                if re.search(rf"^\s*(from|import)\s+{re.escape(mod)}\b", text, re.M):
                    bad.append(f"{path.relative_to(SRC)} imports {mod}")
    assert not bad, bad


def test_no_system_constants_in_code():
    """Element names of a specific solid must live in profiles, not in code (quoted literals; prose may mention MgO).
    Exempt: the C++ kernels ported unchanged (site_current/sfg/hb_correlation .cpp/.cu) and the standalone
    MgO-standard relabel executable (documented as MgO-only)."""
    pattern = re.compile(r"""["'](Mg|Ti|Al|Si|Zn|Ca|Fe)["']""")
    exempt_stems = {"site_current", "sfg", "hb_correlation", "hb_correlation_gpu", "relabel_canonical"}
    hits = []
    for path in list(SRC.rglob("*.py")) + list(SRC.rglob("*.cpp")) + list(SRC.rglob("*.cu")):
        if "profiles" in path.parts or path.stem in exempt_stems:
            continue
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if pattern.search(line) and not line.lstrip().startswith(("#", "//")):
                hits.append(f"{path.relative_to(SRC)}:{n}: {line.strip()}")
    assert not hits, hits


TRAJECTORY_READING = re.compile(r"decode_chunk|open_segment|iter_lammpstrj_blocks|read_lammpstrj\(|read_xyz\(|read_npz\(|iter_frames\(")


def test_analyses_do_not_read_trajectories_or_plot():
    """Analyses get frames/blocks from the pipeline (post analyses may load result files of other analyses).
    Plotting lives in waterint.plot; *_workflow.py modules (v0.6 output orchestration reused unchanged) are listed
    as known exceptions for plotting."""
    for path in _py_files("analysis"):
        text = path.read_text(encoding="utf-8")
        assert not TRAJECTORY_READING.search(text), path
        if not path.name.endswith("_workflow.py"):
            assert "import matplotlib" not in text, path


def test_core_imports_without_gpu():
    import importlib
    import sys
    sys.modules.pop("waterint.backends.gpu", None)
    importlib.import_module("waterint.registry")
    assert "waterint.backends.gpu" not in sys.modules
