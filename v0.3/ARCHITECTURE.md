# WaterInt v0.3 Architecture

WaterInt v0.3 keeps the v0.2 separation between input, domain helpers, scientific computation, output, workflow orchestration, and UI code. It adds a small native-kernel layer for optional C++ acceleration.

## Layers

- `_00_io/`: trajectory readers and NPZ cache support.
- `_01_core/`: reusable domain helpers.
  - `selection.py`: element/type-map selection.
  - `coordinates.py`: axis parsing and coordinate references.
  - `species.py`: oxygen-species classification adapters.
- `_02_computation/`: Python computation APIs, result objects, and `_native.py`.
  - `density.py`
  - `oh_orientation.py`
  - `hbond.py`
  - `sfg.py`
- `_03_output/`: one output helper file per analysis module, plus shared metadata writers.
- `_04_workflows/`: config-driven orchestration and workflow registration.
  - `registry/`: `AnalysisModule` and the central workflow registry.
  - `workflows/`: config-driven `run_<method>()` implementations.
- `_05_ui/`: local UI code.
- `_cpp/`: optional C++ source files for narrow hot kernels.
- `_native/`: locally compiled dynamic libraries, ignored by Git.

## Native Kernel Policy

C++ code should accelerate only narrow hot kernels. The public computation API remains Python, and every native path needs a Python fallback.

The v0.3 prototype adds:

```text
waterint/_cpp/density.cpp
waterint/_02_computation/_native.py
waterint/_02_computation/density.py
```

`density.backend` controls density histogramming:

```yaml
density:
  backend: auto   # auto, python, or cpp
```

`selection.neighbor_method` can require the C++ O-H neighbor-count backend:

```yaml
selection:
  neighbor_method: cpp
```

The native library is generated on demand under `waterint/_native/`. Normal users can still run without compiling C++ code because `auto` falls back to Python.

## Performance Finding

The C++ density histogram kernel is faster than NumPy histogramming on large synthetic arrays and produces identical bin counts. On the full MgO-water density workflow, however, density binning is a small fraction of the runtime.

The C++ O-H neighbor-count kernel is the first backend with a real end-to-end speedup. On the MgO-water density workflow it reduced wall time from about 71.6 s to 51.5 s while producing identical CSV output. The remaining hotspots are NPZ frame reconstruction, selection masks, coordinate references, and Python-level species bookkeeping.

## Workflow Registry

```text
cli.py -> _04_workflows/registry/registry.py -> AnalysisModule -> _04_workflows/workflows/<method>.py
```

A new method has one explicit registration point. The step-by-step implementation guide lives in `_04_workflows/registry/ADDING_MODULE.md`.

## Module Contract

A standard module should provide:

```text
waterint/_02_computation/<method>.py    pure numerical functions and result objects
waterint/_04_workflows/workflows/<method>.py
                                          run_<method>(config) workflow entry point
waterint/_04_workflows/registry/registry.py
                                          AnalysisModule registration
v0.3/example/<method>/                  example config and expected outputs
tests/test_<method>.py                  focused regression tests
```

The computation layer should not know about YAML, command-line parsing, file paths, or plotting. Those concerns belong in workflows, output helpers, and workflow registration.

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
