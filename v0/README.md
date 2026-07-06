# WaterInt v0

This directory contains the current WaterInt v0 code line.

The directory is named `v0` for versioning, but the Python package inside is
named `waterint`. User-facing commands and source imports should therefore use
stable names such as:

```python
from waterint.workflows.density import run_density
```

and:

```bash
waterint density --config v0/example/mgo_density/config_oh_h2o_npz.yaml
```

## Version Information

- Version line: `v0`
- Last updated: 2026-07-06
- Package name: `waterint`
- Public website: https://syang4326.github.io/water-interface-analysis-site/

## Package Structure

```text
waterint/
  core/          reusable simulation-domain helpers
  computation/   numerical analysis modules
  workflows/     config-driven orchestration
  output/        CSV, metadata, and plotting helpers
  io/            trajectory readers and NPZ cache support
  config.py      YAML config loading and validation helpers
  chemistry.py   oxygen/hydrogen neighbor and species helpers
```

The main analysis workflows are:

- `density`: density profiles along a selected coordinate.
- `oh-orientation`: O-H orientation distributions relative to a coordinate.
- `hbond`: hydrogen-bond topology classification.
- `sfg`: SFG/ssVVCF trajectory and postprocessing workflows.

## Install

Install from the repository root, not from this subdirectory:

```bash
python -m pip install -e .
```

The root `pyproject.toml` points setuptools at `v0/waterint`.

## Examples

The publication-facing MgO-water examples are in [`example/`](example/).

```bash
waterint density --config v0/example/mgo_density/config_oh_h2o_npz.yaml
waterint oh-orientation --config v0/example/mgo_oh_orientation/config_oh_h2o_h3o_npz.yaml
waterint hbond --config v0/example/mgo_hbond/config_oh_h2o_h3o_npz.yaml
```

The shared 100 ps MgO trajectory cache is large and should be distributed via a
release asset, Git LFS, or an external data link rather than ordinary Git
history.
