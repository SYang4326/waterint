# WaterInt

WaterInt analyses molecular-dynamics trajectories of water at solid interfaces (oxides, carbonates, metals, ...).
One pipeline reads a trajectory once, prepares every frame once (H ownership, oxygen species, reference plane,
layers, hydrogen bonds) and feeds all requested analyses in the same pass: density, O-H orientation, H-bond
topology, RDF, MSD, conductivity (Nernst-Einstein, Green-Kubo, site currents), SFG spectra, proton-sharing free
energies, identity transitions, carrier tracking, and relabelled trajectories.

Documentation: https://water-interface-analysis.syang4326m.workers.dev/

## Install

```bash
git clone https://github.com/SYang4326/waterint.git
cd waterint
python -m pip install -e .
```

Core dependencies: numpy, scipy, matplotlib, PyYAML. A C++ compiler is optional (hot kernels compile on first use
and are cached; without one, the Python reference implementation runs). CUDA is optional and only used by analyses
that offer a `gpu` backend. `pip install -e .[transport]` adds STACIE for Green-Kubo fits.

## Use

One pipeline, several analyses:

```yaml
# run.yaml
input:
  trajectory: dump.lammpstrj
  format: lammpstrj
  type_map: {1: H, 2: Mg, 3: O}
system:
  profile: mgo_water     # or generic_oxide_water for any oxide/water slab
output:
  directory: out
analyses:
  - {type: density, name: rho_species, selection: {mode: oxygen_species, oxygen_species: all},
     coordinate: {range: [-1.0, 30.0], bins: 310}}
  - {type: hbond, name: hb}
  - {type: msd, name: msd_o, selection: {elements: [O]}, msd: {timestep_ps: 0.01, dimensionality: 2d}}
```

```bash
waterint run run.yaml
```

Single analyses with v0.6-style configs keep working (`waterint density --config cfg.yaml`); the examples in
`examples/` use this form. Other commands: `waterint list` (analyses and parameters), `waterint backends`,
`waterint profile <name>`, `waterint convert-lammpstrj`, `waterint ui`, `waterint web-ui`.

## Layout

```
src/waterint/
  io/         readers (lammpstrj, xyz, npz, GPUMD qnpz), blocks, writers
  system/     SystemProfile + profiles/*.yaml (everything system-specific)
  prep/       ownership, species, reference plane, H bonds -> LabelledFrame
  analysis/   one Analysis class per method (stream | block | post)
  backends/   kernel registry: python reference, optional cpu (C++/OpenMP), optional gpu (CUDA)
  pipeline/   one-pass run, v0.6 config migration
  plot/       figures from results only
  registry.py the analysis list used by the CLI, the UI and the docs generator
  ui/         local desktop and browser UI
tools/gen_docs.py   registry -> docs site pages
```

Design: `docs/ARCHITECTURE_v0.7.md`. Verification: every analysis reproduces v0.6 on MgO-water and TiO2-water
trajectories (integer counts exactly, floats bit for bit, apart from one 1e-14 OpenMP summation-order difference), and
27 MACE trajectories of oxide, carbonate and metal interfaces pass conservation and consistency checks with one generic
profile. Earlier versions (v0.2-v0.6) are in the git history; tag `v0.6` is the last v0.6 release.

## Tests

```bash
python -m pytest tests
```

`tests/test_v06_equivalence.py` compares against a v0.6 checkout (`WATERINT_V06`) and is skipped without one.
Real-data regression scripts: `tests/regression/README.md`.
