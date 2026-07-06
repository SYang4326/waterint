# WaterInt

WaterInt is a Python toolkit for reproducible analysis of water-containing
molecular simulations, with an emphasis on interfacial water systems.

## Current Version

The active code is in [`v0/`](v0/). The outer folder name records the released
version line; inside it, the Python package is still named `waterint`, so imports
and commands do not need to change when a future `v1/` is added.

The earlier prototype package named `waterint/` has been archived locally and is
not part of the GitHub source tree.

Public website:

```text
https://syang4326.github.io/water-interface-analysis-site/
```

## Install

From this repository root:

```bash
python -m pip install -e .
```

This installs the package from `v0/waterint` and provides the command:

```bash
waterint --help
```

If an older conda/pip build environment has trouble with editable install build
isolation, use:

```bash
python -m pip install -e . --no-build-isolation
```

## Quick Start

Run the MgO-water density example:

```bash
waterint density --config v0/example/mgo_density/config_oh_h2o_npz.yaml
```

Run OH-orientation or H-bond examples:

```bash
waterint oh-orientation --config v0/example/mgo_oh_orientation/config_oh_h2o_h3o_npz.yaml
waterint hbond --config v0/example/mgo_hbond/config_oh_h2o_h3o_npz.yaml
```

The large shared MgO trajectory cache is intentionally not stored in normal Git
history. See [`v0/example/README.md`](v0/example/README.md) for the expected
trajectory path and example notes.

## What Is Included

- `density`: oxygen-species and element density profiles.
- `oh-orientation`: z-resolved O-H orientation histograms.
- `hbond`: hydrogen-bond topology fractions by oxygen species.
- `sfg`: SFG/ssVVCF trajectory and postprocessing workflows.
- `core`: reusable selection, species, and coordinate-reference helpers.
- `computation`: numerical analysis routines.
- `workflows`: config-driven orchestration.
- `output`: CSV, plot, and metadata writers.

## Version Notes

- Version line: `v0`
- Last reorganized: 2026-07-06
- Main package import: `waterint`
- CLI command: `waterint`
