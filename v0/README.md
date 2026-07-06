# WaterInt v0

WaterInt v0 is the current version line of the WaterInt package, a Python toolkit for reproducible analysis of water-containing molecular simulations. It focuses on interfacial and confined-water workflows, with config-driven analyses for density profiles, O-H orientation, hydrogen-bond topology, and SFG-related correlation functions.

Documentation website:

https://water-interface-analysis.syang4326m.workers.dev/

## Installation

Install WaterInt from the repository root, not from this `v0/` directory. The root packaging files point Python to the active source code in `v0/waterint`.

```bash
git clone https://github.com/SYang4326/waterint.git
cd waterint
python -m pip install -e .
```

After installation, check that the command-line interface is available:

```bash
waterint --help
```

If an older conda or pip environment has trouble with editable-install build isolation, use:

```bash
python -m pip install -e . --no-build-isolation
```

The folder is named `v0` for repository versioning, but the installed Python package is named `waterint`. User code should therefore use stable imports such as:

```python
from waterint.workflows.density import run_density
```

## Structure

```text
v0/
  README.md        notes for the current version line
  example/         representative MgO-water configs, outputs, and run script
  waterint/        Python package installed as waterint
    core/          shared coordinate, selection, and species helpers
    io/            trajectory readers and NPZ cache support
    computation/   numerical kernels for each analysis module
    workflows/     config-driven orchestration for user-facing commands
    output/        CSV, metadata, and plotting helpers
    ui/            local UI code when enabled in the package checkout
```

The main analysis workflows are:

- `density`: density profiles along an absolute, reference-relative, or slab-relative coordinate.
- `oh-orientation`: O-H angle distributions resolved along a coordinate.
- `hbond`: hydrogen-bond topology classification by oxygen species.
- `sfg`: trajectory-based interfacial O-H correlation and spectrum workflows.

## Usage

Run a workflow by passing a YAML config to the corresponding `waterint` command:

```bash
waterint density --config v0/example/mgo_density/config_oh_h2o_npz.yaml
```

Other example commands:

```bash
waterint oh-orientation --config v0/example/mgo_oh_orientation/config_oh_h2o_h3o_npz.yaml
waterint hbond --config v0/example/mgo_hbond/config_oh_h2o_h3o_npz.yaml
waterint sfg --config v0/example/mgo_sfg/config_100ps_npz.yaml
```

You can also run the publication-facing example set from the repository root:

```bash
python v0/example/run_examples.py
```

The full H-bond example is slower, so it is opt-in:

```bash
python v0/example/run_examples.py --include-hbond
```

## Example Data

The MgO-water examples expect the shared 100 ps trajectory cache at:

```text
v0/example/shared/input/dump.MgO_water_2x2_equil.last100ps.unique.waterint.npz
```

This `.npz` file is large, so it should be distributed through Git LFS, a release asset, or an external data link rather than normal Git history.

## Version Information

- Version line: `v0`
- Last updated: 2026-07-06
- Package name: `waterint`
- CLI command: `waterint`
- Public documentation: https://water-interface-analysis.syang4326m.workers.dev/
