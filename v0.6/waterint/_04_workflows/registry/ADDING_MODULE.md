# Adding A New Analysis Module

This note describes the v0.2 workflow for adding a new WaterInt analysis method. The goal is to keep scientific calculation, config orchestration, output writing, and CLI registration separate.

Assume the new method is called `diffusion`.

## 1. Define The Scientific Contract

Before writing code, decide what the method means.

Write down:

- physical question answered by the method
- required trajectory information
- required selection or species definition
- module-specific config section
- output files
- result object fields

For example:

```text
method: diffusion
question: how fast selected species move over time
config section: diffusion
outputs: msd.csv, msd.png, metadata.json
main quantity: MSD(t) = <|r(t0 + t) - r(t0)|^2>
```

## 2. Add Pure Computation Code

Create:

```text
waterint/_02_computation/diffusion.py
```

This file should contain numerical functions and result dataclasses. It should avoid YAML parsing, command-line parsing, file paths, plotting, and metadata writing.

Example shape:

```python
from dataclasses import dataclass

import numpy as np


@dataclass
class DiffusionResult:
    times: np.ndarray
    msd: np.ndarray
    csv_path: object | None = None
    png_path: object | None = None
    metadata_path: object | None = None


def compute_msd(positions: np.ndarray, times: np.ndarray) -> DiffusionResult:
    ...
```

Rule of thumb:

```text
_02_computation = math, arrays, result objects
```

## 3. Add The Workflow

Create:

```text
waterint/_04_workflows/workflows/diffusion.py
```

The workflow connects config, trajectory reading, selections, computation, output files, and metadata.

Example shape:

```python
from typing import Any

from waterint.config import require_mapping
from waterint._02_computation.diffusion import DiffusionResult, compute_msd
from waterint._03_output.metadata import write_metadata
from waterint._04_workflows.workflows.common import required_workflow_sections, resolve_path


def run_diffusion(config: dict[str, Any]) -> DiffusionResult:
    input_cfg, system_cfg, output_cfg = required_workflow_sections(config)
    diffusion_cfg = require_mapping(config, "diffusion")

    # 1. resolve paths
    # 2. read trajectory or iterate frames
    # 3. prepare selected positions
    # 4. call compute_msd(...)
    # 5. write outputs and metadata
    # 6. return result
```

If the method is framewise, reuse:

```text
waterint/_04_workflows/workflows/framewise.py
```

Rule of thumb:

```text
_04_workflows/workflows = config + IO + core helpers + computation + output
```

## 4. Add Output Helpers If Needed

For simple outputs, the workflow can write files directly. If output logic will be reused or needs styling, add:

```text
waterint/_03_output/diffusion.py
```

Keep plotting and CSV formatting here, not in `_02_computation`.

## 5. Register The Method

Open:

```text
waterint/_04_workflows/registry/registry.py
```

Import the workflow:

```python
from waterint._04_workflows.workflows.diffusion import run_diffusion
```

Add an output printer:

```python
def print_diffusion_outputs(result):
    print(f"Wrote: {result.csv_path}")
    if result.png_path:
        print(f"Wrote: {result.png_path}")
    print(f"Wrote: {result.metadata_path}")
```

Add an `AnalysisModule` entry to `ANALYSIS_MODULES`:

```python
AnalysisModule(
    name="diffusion",
    help="Run a diffusion analysis workflow.",
    run=run_diffusion,
    print_outputs=print_diffusion_outputs,
    description="Compute mean-squared displacement and diffusion-related quantities.",
)
```

After this, the CLI should expose:

```bash
PYTHONPATH=v0.2 python -m waterint.cli diffusion --help
```

## 6. Add An Example Config

Create:

```text
v0.2/example/mgo_diffusion/config.yaml
```

Example shape:

```yaml
input:
  trajectory: ../shared/input/dump.MgO_water_2x2_equil.last100ps.unique.waterint.npz
  format: npz
  type_map:
    1: H
    2: Mg
    3: O

system:
  cell: auto

selection:
  mode: oxygen_species
  oxygen_species: [H2O]
  oxygen_symbol: O
  hydrogen_symbol: H
  oh_cutoff: 1.25

diffusion:
  timestep: 0.005
  max_lag: 10.0
  unwrap: true

output:
  directory: output
  prefix: mgo_h2o_diffusion
  plot: true
```

## 7. Add Tests

At minimum, add a pure computation test:

```text
tests/test_diffusion.py
```

Use tiny arrays and no real trajectory file when possible.

Also add a registry test:

```python
from waterint._04_workflows.registry.registry import get_analysis_module


def test_diffusion_registered():
    module = get_analysis_module("diffusion")
    assert module.name == "diffusion"
```

If practical, add one small workflow test that checks output files are written.

## 8. Update Documentation

Update:

```text
v0.2/README.md
v0.2/ARCHITECTURE.md
```

If the public website documents this method, add a module page with:

```text
Overview
Theory
Example Config
Run Command
Result
Module Parameters
Outputs
Common Pitfalls
```

## 9. Verify

Run:

```bash
PYTHONPATH=v0.2 python -m waterint.cli --help
PYTHONPATH=v0.2 python -m waterint.cli diffusion --help
python -m pytest -q
```

If an example exists:

```bash
PYTHONPATH=v0.2 python -m waterint.cli diffusion \
  --config v0.2/example/mgo_diffusion/config.yaml
```

## Checklist

- Added `_02_computation/<method>.py`
- Added `_04_workflows/workflows/<method>.py`
- Added `_03_output/<method>.py` if the method writes CSV, plots, or other result files
- Registered method in `_04_workflows/registry/registry.py`
- Added example config
- Added tests
- Updated docs
- Verified CLI help and test suite
