# WaterInt

WaterInt is a Python toolkit for reproducible analysis of water-containing molecular simulations, with a focus on interfacial water and confined water systems. It organizes common analyses such as density profiles, O-H orientation, hydrogen-bond topology, and SFG-related workflows around explicit YAML configs, reusable computation modules, and standard output files.

Documentation website:

https://water-interface-analysis.syang4326m.workers.dev/

## Install

Clone the repository and install it from the repository root:

```bash
git clone https://github.com/SYang4326/waterint.git
cd waterint
python -m pip install -e .
```

This installs the active package from [`v0.6/waterint`](v0.6/waterint), while the import name and command remain stable:

```bash
waterint --help
```

If an older conda or pip build environment has trouble with editable-install build isolation, use:

```bash
python -m pip install -e . --no-build-isolation
```

## Usage

Run an analysis by passing a YAML config to a module command:

```bash
waterint density --config v0.6/example/mgo_density/config_oh_h2o_npz.yaml
```

Other current module commands include:

```bash
waterint oh-orientation --config v0.6/example/mgo_oh_orientation/config_oh_h2o_h3o_npz.yaml
waterint hbond --config v0.6/example/mgo_hbond/config_oh_h2o_h3o_npz.yaml
waterint sfg --config v0.6/example/mgo_sfg/config_100ps_npz.yaml
```

The shared MgO-water 100 ps trajectory cache is large, so it is not intended to be stored in ordinary Git history. See [`v0.6/example/README.md`](v0.6/example/README.md) for the expected example-data layout.

## Package Structure

The repository keeps the current version line in [`v0.6/`](v0.6/), but the Python package inside is named `waterint`. This means future version directories can be added without forcing user code to use versioned package names.

```text
v0.6/
  README.md        version-line notes
  example/         publication-facing example configs and outputs
  waterint/        Python package installed as waterint
    core/          shared simulation-domain helpers
    io/            trajectory readers and cache support
    computation/   numerical analysis modules
    workflows/     config-driven orchestration
    output/        CSV, metadata, and plotting utilities
```

The main modules are:

- `density`: density profiles along an absolute or surface-relative coordinate.
- `oh-orientation`: O-H orientation distributions resolved along a coordinate.
- `hbond`: hydrogen-bond topology classification by oxygen species.
- `sfg`: trajectory-based interfacial O-H correlation and spectrum workflows.

Coordinate references can be recomputed from selected atoms, such as a slab surface, or supplied as a fixed value when the reference plane is already known.

## Version Notes

- Active editable-install version line: `v0.6`
- Experimental architecture version line: [`v0.2`](v0.2/README.md)
- Last reorganized: 2026-07-06
- Python package import: `waterint`
- CLI command: `waterint`
