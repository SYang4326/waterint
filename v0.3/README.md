# WaterInt v0.3

WaterInt v0.3 is an experimental performance line based on v0.2. It keeps the v0.2 registry architecture and adds optional C++ backends for density histogramming and O-H neighbor counting.

Documentation website:

https://water-interface-analysis.syang4326m.workers.dev/

## What Changed From v0.2

- Added `waterint/_cpp/` for small C++ kernels.
- Added `waterint/_02_computation/_native.py` to compile and load native code with the Python standard library.
- Added `density.backend: auto | python | cpp`.
- Added `selection.neighbor_method: cpp` for O-H neighbor counting used by oxygen-species classification.
- Added experimental `input.reader: auto | python | cpp` for standard XYZ and common orthorhombic LAMMPS dump files.
- Kept Python as the public API and fallback path.

The density histogram backend is useful as a minimal native-code test. The O-H neighbor-count backend is the first backend that materially speeds up the MgO-water density workflow, because oxygen-species classification is a real hotspot.

## C++ Backend

Use the backend field in a density config:

```yaml
density:
  backend: auto

selection:
  neighbor_method: cpp
```

Backend modes:

- `auto`: use C++ if it compiles and loads, otherwise use Python.
- `python`: force the NumPy implementation.
- `cpp`: require the C++ backend and raise an error if unavailable.

For `selection.neighbor_method`, `auto` tries C++ first for species counts and falls back to the previous Python/scipy paths if native compilation is unavailable. `cpp` requires the C++ neighbor-count backend.

The native library is built on demand into:

```text
waterint/_native/
```

This directory is ignored by Git. The C++ source remains readable in:

```text
waterint/_cpp/density.cpp
waterint/_cpp/chemistry.cpp
waterint/_cpp/xyz.cpp
waterint/_cpp/lammpstrj.cpp
```

For standard XYZ input, the experimental reader can be selected with:

```yaml
input:
  format: xyz
  reader: cpp
```

For common LAMMPS dump input, use the same field:

```yaml
input:
  format: lammpstrj
  reader: cpp
```

The C++ text readers are intentionally narrow. The XYZ reader supports fixed-topology XYZ files with rows of `symbol x y z`. The LAMMPS dump reader supports fixed-atom-count orthorhombic dumps with `type`, `x`, `y`, and `z` atom columns. They read selected frames into memory before yielding `TrajectoryFrame` objects, so they are faster than the Python streaming readers but can use more memory for very long trajectories.

## Structure

```text
_00_io/            trajectory readers and NPZ cache support
_01_core/          shared coordinate, selection, and species helpers
_02_computation/   Python computation APIs and optional native adapter
_03_output/        one output helper file per analysis module
_04_workflows/
  registry/             AnalysisModule definition, registry, and adding-module guide
  workflows/            config-driven run_<method>() orchestration
_05_ui/            local UI code
_cpp/              optional C++ kernels
_native/           local compiled libraries, ignored by Git
```

## Installation

The repository root currently installs the active package line selected by the root `pyproject.toml`. To test v0.3 directly without changing the root package metadata:

```bash
cd /path/to/waterint
PYTHONPATH=v0.3 python -m waterint.cli --help
```

To install v0.3 as the active editable package, update the root packaging metadata to use `v0.3` instead of `v0`, then run:

```bash
python -m pip install -e .
```

## Current Modules

- `density`: density profiles along an absolute, reference-relative, or slab-relative coordinate.
- `oh-orientation`: O-H angle distributions resolved along a coordinate.
- `hbond`: hydrogen-bond topology classification by oxygen species.
- `sfg`: trajectory-based interfacial O-H correlation and spectrum workflows.

`angle-z` is kept as a CLI alias for `oh-orientation`.

## Usage

```bash
PYTHONPATH=v0.3 python -m waterint.cli density --config v0.3/example/mgo_density/config_oh_h2o_npz.yaml
PYTHONPATH=v0.3 python -m waterint.cli oh-orientation --config v0.3/example/mgo_oh_orientation/config_oh_h2o_h3o_npz.yaml
PYTHONPATH=v0.3 python -m waterint.cli hbond --config v0.3/example/mgo_hbond/config_oh_h2o_h3o_npz.yaml
PYTHONPATH=v0.3 python -m waterint.cli sfg --config v0.3/example/mgo_sfg/config_100ps_npz.yaml
```

The MgO-water examples expect the shared 100 ps trajectory cache at:

```text
v0.3/example/shared/input/dump.MgO_water_2x2_equil.last100ps.unique.waterint.npz
```

The trajectory cache is large and should be distributed through Git LFS, a release asset, or an external data link rather than normal Git history.
