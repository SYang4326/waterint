# WaterInt

WaterInt is a Python-first toolkit for reproducible analysis of water-containing molecular simulations.

The first implemented workflow is a config-driven density profile from XYZ or LAMMPS dump trajectories. The intended scope is broader than a single MgO/water project: bulk water, water/material interfaces, nanoconfined water, and selected ions or molecular species should all use the same analysis engine with different configuration files.

## Current Status

Implemented:

- XYZ trajectory reader
- LAMMPS dump (`lammpstrj`) trajectory reader with automatic orthorhombic cell extraction
- WaterInt NPZ trajectory cache for faster repeated analysis of large LAMMPS dump files
- 1D density profile along `x`, `y`, `z`, `-x`, `-y`, or `-z`
- O-H bond angle vs coordinate 2D histograms for `OH-`, `H2O`, and `H3O+`
- H-bond topology fractions for oxygen species, with species-specific topology classes
- SFG/ssVVCF postprocessing: combine correlation functions, Fourier transform, and plot spectra
- element-based selection
- configurable coordinate range and bin count
- absolute coordinates, coordinates relative to an element mean, or coordinates relative to a slab surface
- oxygen-species density profiles based on local O-H coordination: `O2-`, `OH-`, `H2O`, `H3O+`
- fast O-H neighbor counting with scipy `cKDTree` when available
- CSV, PNG, and metadata JSON outputs
- command-line entry point: `waterint density --config config.yaml`
- command-line entry point: `waterint angle-z --config config.yaml`
- command-line entry point: `waterint hbond --config config.yaml`
- command-line entry point: `waterint sfg --config config.yaml`

Planned:

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

Run an oxygen-species density example relative to a slab reference:

```bash
python -m waterint.cli density --config examples/density_xyz/config_oxygen_species.yaml
```

The species classifier currently uses a simple O-H distance cutoff. This is useful as a first robust protocol, but it is not yet a full proton-sharing or bond-history model.
For large trajectories, oxygen-species classification uses `selection.neighbor_method: auto` by default. This uses scipy `cKDTree` when scipy is installed, and falls back to a chunked NumPy distance search otherwise. You can set `selection.neighbor_workers: -1` to let scipy use all available CPU threads for the neighbor search.

Run a LAMMPS dump example with explicit atom-type mapping:

```bash
python -m waterint.cli density --config examples/density_lammpstrj/config_oxygen_species.yaml
```

For `lammpstrj` inputs, set `input.type_map` when atom types are numeric. `system.cell: auto` reads the cell lengths from `ITEM: BOX BOUNDS`.

Run an O-H bond angle vs z example:

```bash
python -m waterint.cli angle-z --config examples/angle_z_lammpstrj/config.yaml
```

This writes one 2D histogram per requested oxygen species. The current angle convention is the O-to-H bond vector angle to the configured coordinate axis, in degrees. The coordinate position is the oxygen position, using the same absolute/reference/slab-relative coordinate modes as the density workflow.

Run an H-bond topology example:

```bash
python -m waterint.cli hbond --config examples/hbond_lammpstrj/config.yaml
```

This classifies each requested oxygen species by donated and accepted H-bonds. For example, `DDAA` means two donated and two accepted H-bonds, while `DA` means one donated and one accepted H-bond. The default topology labels are species-specific: `OH-` does not show impossible double-donor classes, and `H3O+` uses classes such as `DDDA` and `DDD`. The first implementation uses an O-O cutoff plus a D-H-A angle criterion; both are set in the config.
The grouped CSV is accompanied by a raw topology CSV, which lists every observed donor/acceptor label before uncommon labels are merged into `other`.

Run an SFG postprocessing example:

```bash
python -m waterint.cli sfg --config examples/sfg_cf/config_combine_bins.yaml
```

This module covers both a Python trajectory-to-spectrum path and the original postprocessing path. In trajectory mode, WaterInt assigns each H to its nearest O, builds continuous O-H segments, computes ssVVCF from the stretch velocity and dipole-velocity proxy, writes the CF, performs a DCT-based FT using the existing unit convention, and plots the spectrum. The first Python calculator uses finite-difference velocities from positions, so production runs should set the real `sfg.dt_ps` explicitly and use sufficiently dense trajectory frames.

```bash
python -m waterint.cli sfg --config examples/sfg_trajectory/config.yaml
```

For repeated analysis of the same large LAMMPS dump, first convert it to a WaterInt NPZ cache:

```bash
waterint convert-lammpstrj \
  --input dump.lammpstrj \
  --output dump.waterint.npz \
  --type-map 1=H 2=Mg 3=O
```

Then use the cache in a density config:

```yaml
input:
  trajectory: dump.waterint.npz
  format: npz
  type_map:
    1: H
    2: Mg
    3: O
```

The conversion is an upfront cost, but it avoids repeatedly parsing the same text trajectory for density, orientation, and future analysis modules.

## Repository Placement

Keep this package as an independent Git repository, separate from the static website prototype:

```text
20_projects/
  waterint/                         # Python package and source code
  water-interface-analysis-site/    # documentation website prototype
```

Once the package stabilizes, the website can either link to this repository or be moved into a docs system inside this repository.
