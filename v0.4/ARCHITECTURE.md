# WaterInt v0.4 Architecture

WaterInt v0.4 keeps the v0.3 separation between input, domain helpers, scientific computation, output, workflow orchestration, and UI code. It adds C++-accelerated MSD and RDF modules.

## Layers

- `_00_io/`: trajectory readers and NPZ cache support.
- `_01_core/`: reusable domain helpers.
  - `selection.py`: element/type-map selection.
  - `coordinates.py`: axis parsing and coordinate references.
  - `species.py`: oxygen-species classification adapters.
- `_02_computation/`: Python computation APIs, result objects, `_native.py`, and local C++ kernels.
  - `density.py`
  - `oh_orientation.py`
  - `hbond.py`
  - `sfg.py`
  - `msd.py`
  - `rdf.py`
- `_03_output/`: one output helper file per analysis module, plus shared metadata writers.
- `_04_workflows/`: config-driven orchestration and workflow registration.
  - `registry/`: `AnalysisModule` and the central workflow registry.
  - `workflows/`: config-driven `run_<method>()` implementations.
- `_05_ui/`: local UI code.
- `_native/`: locally compiled dynamic libraries, ignored by Git.

## Native Kernel Policy

C++ code should accelerate only narrow hot kernels. The public computation API remains Python, and every native path needs a Python fallback.

The v0.4 native layer includes:

```text
waterint/_02_computation/density.cpp
waterint/chemistry.cpp
waterint/chemistry.hpp
waterint/_02_computation/oh_orientation.cpp
waterint/_02_computation/hbond.cpp
waterint/_02_computation/sfg.cpp
waterint/_02_computation/msd.cpp
waterint/_02_computation/rdf.cpp
waterint/_02_computation/_native.py
waterint/_02_computation/density.py
waterint/_02_computation/oh_orientation.py
waterint/_02_computation/hbond.py
waterint/_02_computation/sfg.py
waterint/_02_computation/msd.py
waterint/_02_computation/rdf.py
```

Each C++ source file lives beside the Python module it accelerates. The native loader discovers package-local `*.cpp` files and compiles them into one optional library, keeping installation simple while preserving module ownership.

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

`angle.backend` controls the fused O-H orientation kernel:

```yaml
angle:
  backend: auto   # auto, python, or cpp
```

The reusable chemistry kernel returns an O-H neighbor matrix built with the same cell list used by density. The orientation-specific kernel consumes that matrix to calculate bond or bisector directions and accumulate a two-dimensional z-angle histogram. Config parsing, coordinate references, normalization, and output remain in Python.

`hbond.backend` controls the H-bond kernel:

```yaml
hbond:
  backend: auto   # auto, python, or cpp
```

H-bond analysis consumes the same shared O-H neighbor matrix as orientation. Its local C++ kernel uses `CutoffNeighborSearch` from `chemistry.hpp` for O-O acceptor candidates and evaluates D-H-A geometry. The pure Python route uses the corresponding `PythonCutoffNeighborSearch` from `chemistry.py` and batches local geometry with NumPy. Python retains topology labels, custom class grouping, workflow orchestration, and output in both routes.

`sfg.backend` controls the trajectory-mode ssVVCF kernel:

```yaml
sfg:
  mode: trajectory
  backend: auto   # auto, python, or cpp
```

The SFG C++ kernel uses `CutoffNeighborSearch` for O-H assignment and owns direct trajectory-velocity ingestion or finite-difference velocities, continuous O-H segment management, windowed signal construction, and FFT-based segment correlation. Python owns config and trajectory orchestration, z-reference calculation, final normalization, Fourier transformation, plotting, and output. The native route requires fixed atom types and ordering; `auto` uses the Python implementation when that contract is not satisfied. LAMMPS readers retain `vx/vy/vz` when all three fields are present, and NPZ conversion preserves them.

`sfg.layer_bins` enables multi-channel layer/species ssVVCF. The layered C++
ABI keeps one continuous H-O segment with a per-frame window/species mask for
every requested output channel and accumulates each channel separately.
Species membership is derived from the unique H-to-O assignment on every
frame, so `nh1` follows proton transfer rather than a fixed atom selection.
The existing single-window ABI remains available for compatibility. Python
implements the same mask semantics and is the fallback for variable topology
or unavailable native compilation.

The native library is generated on demand under `waterint/_native/`. Normal users can still run without compiling C++ code because `auto` falls back to Python.

## Performance Finding

The C++ density histogram kernel is faster than NumPy histogramming on large synthetic arrays and produces identical bin counts. On the full MgO-water density workflow, however, density binning is a small fraction of the runtime.

On the full MgO-water density workflow, the optimized NPZ reader and C++ cell-list kernels reduce the measured runtime from about 49.8 s to 18.7 s while producing identical CSV output.

On the full 100 ps O-H bond orientation workflow, the C++ neighbor-list and orientation kernels reduce the measured analysis runtime from 133.05 s to 21.98 s. Histograms and bond/sample totals match the Python/scipy route exactly. See `BENCHMARK_OH_ORIENTATION_CPP.md` for the stage breakdown.

On the full 100 ps H-bond workflow, the Python cell list reduces runtime from 6706.48 s to 1542.96 s. Shared neighbor-list and C++ geometry kernels reduce it further to 66.67 s. All three routes preserve raw topology counts exactly. See `BENCHMARK_HBOND_CPP.md` for the stage breakdown.

On the full 100 ps trajectory-mode SFG workflow, the native kernel reduces staged runtime from 1738.42 s to 30.20 s, a 57.57x end-to-end speedup. Counts and z-references match exactly, and the maximum absolute correlation difference is `6.846e-13`. See `BENCHMARK_SFG_CPP.md` for the stage breakdown.

`msd` and `rdf` follow the same native-backend convention. The C++ MSD kernel performs PBC-aware trajectory unwrapping followed by multiple-time-origin displacement accumulation. The C++ RDF kernel accumulates pair distances with full triclinic minimum-image handling. Python owns config, selection, coordinate references, normalization, and output, and remains the fallback path.

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
v0.4/example/<method>/                  example config and expected outputs
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
