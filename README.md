# WaterInt

WaterInt is a Python-first toolkit for reproducible analysis of water-containing molecular simulations.

The first implemented workflow is a config-driven density profile from XYZ trajectories. The intended scope is broader than a single MgO/water project: bulk water, water/material interfaces, nanoconfined water, and selected ions or molecular species should all use the same analysis engine with different configuration files.

## Current Status

Implemented:

- XYZ trajectory reader
- 1D density profile along `x`, `y`, or `z`
- element-based selection
- absolute coordinates or coordinates relative to a reference element mean
- CSV, PNG, and metadata JSON outputs
- command-line entry point: `waterint density --config config.yaml`

Planned:

- OH bond orientation
- H-bond analysis
- richer trajectory formats
- selection language beyond element names
- nanoconfinement/interfacial presets

## Install Locally

From this directory:

```bash
python3 -m pip install -e .
```

## Run The Example

Without installing:

```bash
python -m waterint.cli density --config examples/density_xyz/config.yaml
```

After editable installation:

```bash
waterint density --config examples/density_xyz/config.yaml
```

Expected outputs:

```text
examples/density_xyz/output/density_water_O.csv
examples/density_xyz/output/density_water_O.png
examples/density_xyz/output/density_water_O_metadata.json
```

## Repository Placement

Keep this package as an independent Git repository, separate from the static website prototype:

```text
20_projects/
  waterint/                         # Python package and source code
  water-interface-analysis-site/    # documentation website prototype
```

Once the package stabilizes, the website can either link to this repository or be moved into a docs system inside this repository.
