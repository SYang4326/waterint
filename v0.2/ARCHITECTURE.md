# WaterInt v0.2 Architecture

WaterInt v0.2 keeps the v0 separation between input, domain helpers, scientific computation, output, workflow orchestration, and UI code. The directory prefixes follow the normal data flow: read data first, interpret it, compute quantities, write results, expose workflows, then provide interfaces.

## Layers

- `_00_io/`: trajectory readers and NPZ cache support.
- `_01_core/`: reusable domain helpers.
  - `selection.py`: element/type-map selection.
  - `coordinates.py`: axis parsing and coordinate references.
  - `species.py`: oxygen-species classification adapters.
- `_02_computation/`: flat numerical computation files.
  - `density.py`
  - `oh_orientation.py`
  - `hbond.py`
  - `sfg.py`
- `_03_output/`: one output helper file per analysis module, plus shared metadata writers.
- `_04_workflows/`: config-driven orchestration and workflow registration.
  - `registry/`: `AnalysisModule` and the central workflow registry.
  - `workflows/`: config-driven `run_<method>()` implementations.
- `_05_ui/`: local UI code.

The leading underscore keeps the directories valid Python package names. The numeric prefix keeps the source tree sorted by the main data flow.

## Workflow Registry

In v0, adding a new method required editing the scientific code and then manually wiring the method into the CLI. In v0.2, the registry lives inside `_04_workflows/registry/`, because it describes how workflow entry points are exposed to users.

```text
cli.py -> _04_workflows/registry/registry.py -> AnalysisModule -> _04_workflows/workflows/<method>.py
```

This means a new method has one explicit registration point. The CLI does not need a new `if args.command == ...` branch for every analysis method.

The step-by-step implementation guide lives in `_04_workflows/registry/ADDING_MODULE.md`.

## Module Contract

A standard module should provide:

```text
waterint/_02_computation/<method>.py    pure numerical functions and result objects
waterint/_04_workflows/workflows/<method>.py
                                          run_<method>(config) workflow entry point
waterint/_04_workflows/registry/registry.py
                                          AnalysisModule registration
v0.2/example/<method>/                  example config and expected outputs
tests/test_<method>.py                  focused regression tests
```

The computation layer should not know about YAML, command-line parsing, file paths, or plotting. Those concerns belong in workflows, output helpers, and workflow registration.

## Adding A Method

Example:

```python
from waterint._04_workflows.registry.analysis_module import AnalysisModule
from waterint._04_workflows.workflows.diffusion import run_diffusion


def print_diffusion_outputs(result):
    print(f"Wrote: {result.csv_path}")
    if result.png_path:
        print(f"Wrote: {result.png_path}")
    print(f"Wrote: {result.metadata_path}")


DIFFUSION = AnalysisModule(
    name="diffusion",
    help="Run a diffusion analysis workflow.",
    run=run_diffusion,
    print_outputs=print_diffusion_outputs,
)
```

Then add `DIFFUSION` to `ANALYSIS_MODULES` in `_04_workflows/registry/registry.py`. The command appears in `waterint --help` without adding another branch to `cli.py`.

## Import Direction

Keep dependencies mostly one-way:

```text
_04_workflows -> _00_io / _01_core / _02_computation / _03_output
_03_output -> result data and arrays
_02_computation -> arrays, result dataclasses, and minimal core/io helpers when needed
_01_core -> _00_io dataclasses only when needed
_00_io -> no workflow/output imports
```

The version lives at the directory level. Internal imports remain `waterint.*`, not versioned package names.
