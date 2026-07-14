# SFG C++ Backend Benchmark

Date: 2026-07-13

System: macOS 14.4.1, Apple Silicon, Apple clang 15.0, Python 3.9.7, NumPy 1.26.4, SciPy 1.13.1

## Implementation

The trajectory-mode SFG backend keeps the public workflow in Python and moves the repeated numerical core into the module-local C++ kernel:

```text
Python NPZ reader and config
  -> Python z-reference and fixed O/H selection
  -> C++ finite-difference velocities
  -> shared C++ cell-list O-H assignment
  -> C++ continuous O-H segment signals
  -> C++ FFT segment correlation
  -> Python normalization, FT, plotting, and output
```

The C++ implementation lives beside its Python API in `_02_computation/sfg.cpp`. It reuses `CutoffNeighborSearch` from `chemistry.hpp`; it does not add a second SFG-specific cell list.

```yaml
sfg:
  mode: trajectory
  backend: auto   # auto, python, or cpp
```

`auto` uses C++ when the native library is available and atom types/order are fixed. It otherwise falls back to Python. `cpp` requires the native path, while `python` provides an explicit reference route.

## Full 100 ps Benchmark

Input:

- MgO-water NPZ trajectory
- 20001 frames
- 1896 atoms per frame
- O-H cutoff: 1.25 Angstrom
- frame interval: 0.005 ps
- correlation lag: 0.995 ps
- Mg surface z-reference and interface window enabled
- FT, plotting, and file output excluded from staged timing

The rows below are mutually exclusive wall-time stages. `Other overhead` contains dispatch, array validation, and timing bookkeeping not included in another row.

| Stage | Python (s) | C++ (s) | Saved (s) | Speedup |
|---|---:|---:|---:|---:|
| Read NPZ frames | 1.59 | 2.15 | -0.56 | 0.74x |
| Reference + O/H selection | 1.95 | 1.96 | -0.01 | 1.00x |
| Finite-difference velocities | 1.62 | 0.91 | 0.71 | 1.79x |
| O-H assignment | 1144.99 | 8.88 | 1136.11 | 128.99x |
| Segment signal construction | 560.42 | 1.09 | 559.34 | 516.46x |
| Segment correlation | 27.77 | 12.88 | 14.89 | 2.16x |
| Finalize correlation | 0.00 | 0.00 | 0.00 | 2.24x |
| Other overhead | 0.08 | 2.34 | -2.26 | 0.03x |
| **Total** | **1738.42** | **30.20** | **1708.23** | **57.57x** |

The real `backend: auto` CLI run, including NPZ loading, the native kernel, FT, data output, metadata, and PNG rendering, completed in 36.53 s.

## Correctness

The complete Python and C++ runs used the same 20001 frames and produced:

```text
time grid: exact
correlation counts: exact
z-reference series: exact
maximum absolute correlation difference: 6.846e-13
allclose(rtol=1e-10, atol=1e-10): true
```

The native code preserves the WaterInt trajectory-mode definitions for O-H cutoff assignment, duplicate-H handling, minimum-image PBC, `full` and `stretch` signals, both window modes, sign flipping, symmetrization, and bond-segment switching.

## Validation

Focused tests cover:

- `full` and `stretch` signal modes
- symmetrized and unsymmetrized correlations
- interface and top-hat windows
- sign flipping and PBC
- O-H segment switching
- duplicate-H errors
- variable-topology rejection and automatic Python fallback
- density, O-H orientation, and H-bond native regressions

Run:

```bash
PYTHONPATH=v0.3 python -m pytest -q \
  tests/test_v03_sfg_cpp.py \
  tests/test_v03_hbond_cpp.py \
  tests/test_v03_oh_orientation_cpp.py \
  tests/test_v03_density_cpp.py \
  tests/test_v02_registry.py
```
