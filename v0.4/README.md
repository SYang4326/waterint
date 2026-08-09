# WaterInt v0.4

WaterInt v0.4 is an experimental performance line based on v0.3. It keeps the registry architecture and adds optional C++ backends for MSD and RDF alongside density, O-H orientation, H-bond, and trajectory-based SFG.

Documentation website:

https://water-interface-analysis.syang4326m.workers.dev/

## What Changed From v0.3

- Added small C++ kernels beside the Python modules that own them.
- Added `waterint/_02_computation/_native.py` to compile and load native code with the Python standard library.
- Added `density.backend: auto | python | cpp`.
- Added `angle.backend: auto | python | cpp` for O-H orientation.
- Added `hbond.backend: auto | python | cpp` for H-bond geometry.
- Added `sfg.backend: auto | python | cpp` for trajectory-mode ssVVCF construction.
- Added `selection.neighbor_method: cpp` for O-H neighbor counting used by oxygen-species classification.
- Added experimental `input.reader: auto | python | cpp` for standard XYZ and common orthorhombic LAMMPS dump files.
- Added `msd`: 2D/3D, PBC-unwrapped multiple-time-origin MSD with an optional initial-frame layer selection.
- Added `rdf`: O-O/O-H defaults plus element, LAMMPS type, oxygen-species, and coordinate-layer selective pair RDFs.
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

### Layer- and species-resolved SFG

Set `sfg.layer_bins` to write a separate ssVVCF and spectrum for each named
z window. Every layer always includes an all-O-H channel. `species_channels`
adds channels selected from the existing oxygen-species labels (`O2-`, `OH-`,
`H2O`, `H3O+`, and `O_other`) using the same H-to-O assignment as SFG. The
legacy hydroxide filename token is `nh1` and means one assigned H per oxygen;
it is a species label, not a layer number.

```yaml
sfg:
  mode: trajectory
  backend: auto
  layer_bins:
    - label: 0_1d5
      window: {mode: 2, z1: 0.0, z2: 1.5, ramp: 0.0}
    - label: 1d5_2d8
      window: {mode: 2, z1: 1.5, z2: 2.8, ramp: 0.0}
    - label: all
  species_channels: [OH-]

output:
  prefix: ssvvcf
  run_label: 900ps
```

This writes `ssvvcf_0_1d5_900ps.dat` for all O-H bonds and
`ssvvcf_0_1d5_900ps_cf_nh1.dat` for OH-. These names are accepted directly by
the existing `sfg.mode: combine_bins` workflow with `cf_prefix: ssvvcf`.
For fixed-topology trajectories, layered channels use the native multi-channel
kernel with one per-frame layer/species mask and accumulator per output. The
`nh1` mask is dynamic: C++ resolves the unique H-to-O assignment on every frame
and selects an O-H bond only when that oxygen has exactly one assigned H at
that frame. `backend: auto` falls back to the equivalent Python segment path
when native compilation or fixed topology is unavailable. The included
runnable configuration is `example/mgo_sfg/config_layered_sfg_quick.yaml`.

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
waterint/_02_computation/msd.py              + msd.cpp
waterint/_02_computation/rdf.py              + rdf.cpp
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

The repository root currently installs the active package line selected by the root `pyproject.toml`. To test v0.4 directly without changing the root package metadata:

```bash
cd /path/to/waterint
PYTHONPATH=v0.4 python -m waterint.cli --help
```

To install v0.4 as the active editable package, update the root packaging metadata to use `v0.4` instead of `v0`, then run:

```bash
python -m pip install -e .
```

## Current Modules

- `density`: density profiles along an absolute, reference-relative, or slab-relative coordinate.
- `oh-orientation`: O-H angle distributions resolved along a coordinate.
- `hbond`: hydrogen-bond topology classification by oxygen species.
- `sfg`: trajectory-based interfacial O-H correlation and spectrum workflows.
- `msd`: PBC-unwrapped 2D/3D mean-squared displacement for a fixed atom selection.
- `rdf`: radial distribution functions for configurable atom or oxygen-species pairs.

`angle-z` is kept as a CLI alias for `oh-orientation`.

## Usage

```bash
PYTHONPATH=v0.4 python -m waterint.cli density --config v0.4/example/mgo_density/config_oh_h2o_npz.yaml
PYTHONPATH=v0.4 python -m waterint.cli oh-orientation --config v0.4/example/mgo_oh_orientation/config_oh_h2o_h3o_npz.yaml
PYTHONPATH=v0.4 python -m waterint.cli hbond --config v0.4/example/mgo_hbond/config_oh_h2o_h3o_npz.yaml
PYTHONPATH=v0.4 python -m waterint.cli sfg --config v0.4/example/mgo_sfg/config_100ps_npz.yaml
PYTHONPATH=v0.4 python -m waterint.cli msd --config v0.4/example/mgo_msd/config_h2o_layer.yaml
PYTHONPATH=v0.4 python -m waterint.cli rdf --config v0.4/example/mgo_rdf/config_oo_oh.yaml
```

## MSD

`msd` follows a fixed set of atom identities. `selection` chooses those atoms in the first frame. When `msd.layer` is present, it further retains only atoms whose first-frame coordinate lies in that interval. The coordinate uses the existing absolute/reference/slab-relative definition, so an interfacial layer can be expressed relative to a moving slab surface. Multiple time origins are then averaged for 2D or 3D displacement.

```yaml
selection:
  elements: [O]

msd:
  backend: auto
  timestep_ps: 0.005
  dimensionality: "2d"
  plane_normal_axis: z
  pbc: [true, true, false]
  max_lag_frames: 2000
  origin_stride: 10  # analyze every tenth eligible time origin
  layer:
    coordinate:
      mode: relative_to_slab
      axis: z
      reference:
        type: slab_surface
        species: [Mg]
        surface: max
    range: [0.0, 4.0]
```

## RDF

When `rdf.pairs` is omitted, WaterInt writes O-O and O-H RDFs. A pair can independently select `elements`, `types`, or `oxygen_species`; it can also attach a coordinate layer. Species and RDF layer selection are evaluated on each frame, so they remain meaningful when proton transfer changes a water oxygen's instantaneous species label. RDF normalization is accumulated frame by frame using each selection's actual population and cell volume.

For a relabelled interface analysis, use one source selector per chemical and
spatial group, then pair it with `elements: [O]` and `elements: [H]`. The
MgO-water example at `example/mgo_rdf/config_relabelled_oxygen_groups.yaml`
defines Mg-O/Mg-H plus lattice O (`O2-` under the O-H coordination criterion),
layer-1 OH-/H2O, layer-2 OH-, and bulk H2O this way.

```yaml
rdf:
  backend: auto
  r_max: 8.0
  bins: 200
  pbc: [true, true, true]
  pairs:
    - name: H2O-OH
      first:
        oxygen_species: [H2O]
        oxygen_symbol: O
        hydrogen_symbol: H
        hydrogen_assignment: nearest
      second:
        oxygen_species: [OH-]
        oxygen_symbol: O
        hydrogen_symbol: H
        hydrogen_assignment: nearest
```

The MgO-water examples expect the shared 100 ps trajectory cache at:

```text
v0.4/example/shared/input/dump.MgO_water_2x2_equil.last100ps.unique.waterint.npz
```

The trajectory cache is large and should be distributed through Git LFS, a release asset, or an external data link rather than normal Git history.
