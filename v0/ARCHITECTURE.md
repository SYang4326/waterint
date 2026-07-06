# WaterInt v0 Architecture

WaterInt v0 separates scientific computation from configuration, trajectory
loading, plotting, and file output. The goal is to keep analysis modules
reusable while keeping command-line workflows convenient.

## Layers

- `core/`: reusable domain helpers.
  - `selection.py`: element/type-map selection.
  - `coordinates.py`: axis parsing and coordinate references.
  - `species.py`: oxygen-species classification adapters.
- `computation/`: numerical computation modules.
  - `density/`: density histogram and normalization code.
  - `oh_orientation/`: O-H orientation vs coordinate histograms.
  - `hbond/`: H-bond topology classification and accumulation.
  - `sfg/`: SFG result types, correlation processing, and trajectory backend.
- `workflows/`: orchestration code that connects configs, trajectory iteration,
  core helpers, computation, and output.
- `output/`: CSV writers, plotting helpers, and metadata writers.
- `io/`: trajectory readers and NPZ cache support.

## Design Rule

Computation code should receive prepared scientific data and return analysis
results. Workflow code is allowed to know about config files, paths, trajectory
iteration, plotting, and metadata.

For framewise analyses:

```text
config -> workflow -> framewise runner -> core helpers -> computation -> output
```

## Import Direction

Keep dependencies one-way:

```text
workflows -> core / computation / output
computation -> core only when unavoidable
output -> plain arrays or result data
core -> no workflow/output imports
```

The package directory is `v0/waterint`, but internal imports use `waterint.*`.
The version lives at the directory level, not in function names or import paths.
