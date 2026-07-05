# waterint_v0 architecture sketch

`waterint_v0` is an experimental refactor of the original `waterint` package.
The goal is to keep scientific calculations separate from configuration,
trajectory loading, plotting, and file output.

## Layers

- `core/`: small reusable domain helpers.
  - `selection.py`: element/type-map selection.
  - `coordinates.py`: axis parsing and coordinate references.
  - `species.py`: oxygen-species classification adapters.
- `computation/`: numerical computation modules.
  - `density/`: density-specific histogram and normalization code.
  - `oh_orientation/`: O-H orientation vs coordinate histograms. This replaces
    the older `angle_z` naming; `angle_z` remains only as a compatibility alias.
  - `hbond/`: H-bond topology classification and accumulation.
  - `sfg/`: SFG result types and trajectory backend.
- `workflows/`: orchestration code that connects config, input, core helpers,
  analysis code, and output.
  - `framewise.py` is the reusable frame-by-frame runner.
  - `density.py`, `oh_orientation.py`, and `hbond.py` are adapters around that runner.
  - `angle_z.py` is a compatibility wrapper around `oh_orientation.py`.
  - `sfg.py` uses a separate workflow style because some SFG modes are
    file-postprocessing jobs rather than framewise trajectory analyses.
- `output/`: file writers and output labels.

## Design rule

Analysis code should receive already-prepared scientific data and return
analysis results. Workflow code is allowed to know about config files, paths,
trajectory iteration, plotting, and metadata.

For framewise analyses, this means:

```text
config -> workflow -> framewise runner -> core helpers -> analysis compute -> output
```

## Current coverage

- `density`: refactored into core selection/coordinates, density compute,
  workflow, and output.
- `oh_orientation`: refactored into framewise workflow plus O-H orientation
  histogram compute. The old `angle_z` workflow name is still available as a
  compatibility alias.
- `hbond`: refactored into framewise workflow plus topology compute.
- `sfg`: has a v0 workflow wrapper for all current modes. `single` and
  `combine_bins` are file-processing workflows. `trajectory` uses the v0
  `sfg/trajectory.py` backend, which has been moved onto v0 core/workflow
  helpers while preserving the legacy algorithm.

## Import direction

Keep dependencies one-way:

```text
workflows -> core / computation / output
computation -> core only when unavoidable
output -> analysis data structures or plain arrays
core -> no workflow/output imports
```

Analysis subpackages should not import workflows from their `__init__.py`.
That avoids circular imports and keeps "compute" code usable without the config
or CLI layer.
