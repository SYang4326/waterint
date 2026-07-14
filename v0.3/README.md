# WaterInt v0.3

WaterInt v0.3 is an experimental performance line based on v0.2. It keeps the v0.2 registry architecture and adds optional C++ backends for density, O-H orientation, H-bond, and trajectory-based SFG kernels.

Documentation website:

https://water-interface-analysis.syang4326m.workers.dev/

## What Changed From v0.2

- Added small C++ kernels beside the Python modules that own them.
- Added `waterint/_02_computation/_native.py` to compile and load native code with the Python standard library.
- Added `density.backend: auto | python | cpp`.
- Added `angle.backend: auto | python | cpp` for O-H orientation.
- Added `hbond.backend: auto | python | cpp` for H-bond geometry.
- Added `sfg.backend: auto | python | cpp` for trajectory-mode ssVVCF construction.
- Added `selection.neighbor_method: cpp` for O-H neighbor counting used by oxygen-species classification.
- Added experimental `input.reader: auto | python | cpp` for standard XYZ and common orthorhombic LAMMPS dump files.
- Kept Python as the public API and fallback path.

The chemistry layer provides corresponding shared Python and C++ cell-list implementations. Density uses count-only queries, while O-H orientation and H-bond analysis reuse full O-H neighbor identities. H-bond analysis uses the same spatial-search abstraction for O-O acceptor candidates.

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

O-H orientation uses a separate backend field:

```yaml
angle:
  backend: auto
  vector_mode: oh_bond
```

The C++ orientation path supports `oh_bond`, `oh_bisector`, and `dipole`. It reuses the shared C++ O-H cell list, calculates orientation vectors, and accumulates the z-angle histogram without materializing Python pair lists. `auto` falls back to the existing Python/scipy implementation when native compilation is unavailable.

H-bond analysis uses the same backend convention:

```yaml
hbond:
  backend: auto
```

The native route reuses the shared O-H neighbor matrix and accelerates O-O candidate search and D-H-A geometry. Python still handles config validation, topology labels, grouping, and output.

When `hbond.backend: python` is selected, WaterInt uses a Python cell list and frame-level NumPy geometry batch by default. This replaces the original donor-to-all-oxygen scan while retaining a pure Python/scipy/NumPy calculation route.

Trajectory-mode SFG follows the same convention:

```yaml
sfg:
  mode: trajectory
  backend: auto
```

The native SFG path reuses the shared C++ O-H cell list and accelerates finite-difference velocities, O-H assignment, continuous bond-segment construction, and segment correlation. Python retains config handling, trajectory loading, z-reference calculation, Fourier transformation, plotting, and file output. The C++ path requires fixed atom types and ordering; `auto` falls back to Python for variable-topology trajectories.

For a LAMMPS dump containing all of `vx vy vz`, SFG can use the stored velocities directly instead of estimating them from positions:

```yaml
input:
  format: lammpstrj

sfg:
  velocity_source: auto              # auto, trajectory, or finite_difference
  trajectory_velocity_unit: A/ps     # use A/fs for LAMMPS real units
```

`auto` uses trajectory velocities only when every selected frame has all three velocity columns; otherwise it uses the existing finite-difference route. `trajectory` requires those columns and fails clearly when they are absent. WaterInt converts `A/fs` to its internal `A/ps` convention. `convert-lammpstrj` preserves velocity arrays in an NPZ cache, so the same option also works after conversion.

The native library is built on demand into:

```text
waterint/_native/
```

This directory is ignored by Git. The C++ source remains beside its corresponding Python module:

```text
waterint/_02_computation/density.py          + density.cpp
waterint/chemistry.py                        + chemistry.cpp
waterint/_02_computation/oh_orientation.py   + oh_orientation.cpp
waterint/_02_computation/hbond.py            + hbond.cpp
waterint/_02_computation/sfg.py              + sfg.cpp
waterint/_00_io/xyz.py                       + xyz.cpp
waterint/_00_io/lammpstrj.py                 + lammpstrj.cpp
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
_02_computation/   Python computation APIs, local C++ kernels, and native adapter
_03_output/        one output helper file per analysis module
_04_workflows/
  registry/             AnalysisModule definition, registry, and adding-module guide
  workflows/            config-driven run_<method>() orchestration
_05_ui/            local UI code
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
