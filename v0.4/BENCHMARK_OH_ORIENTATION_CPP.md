# O-H Orientation C++ Backend Benchmark

Date: 2026-07-10

System: local macOS / Apple clang / Python 3.9

## Implementation

The O-H orientation backend reuses the v0.3 native infrastructure and chemistry cell list already used by density:

```text
NPZ reader
  -> Python coordinate reference and O/H selection
  -> C++ cell-list O-H neighbor matrix
  -> C++ bond or bisector orientation
  -> C++ z-angle histogram accumulation
  -> Python normalization and output
```

The public configuration remains Python-driven:

```yaml
angle:
  backend: auto   # auto, python, or cpp
  vector_mode: oh_bond
```

`auto` uses the C++ path when the native library can be compiled and loaded, otherwise it falls back to the existing Python/scipy path. The native kernel also supports `oh_bisector` and `dipole`.

## Full 100 ps Benchmark

Input:

- MgO-water NPZ trajectory
- 20001 frames
- 1896 atoms per frame
- `vector_mode: oh_bond`
- 180 z bins and 180 angle bins
- plotting disabled during stage timing
- CSV and metadata output enabled

| Step | Python/scipy route | C++ route | Saved |
|---|---:|---:|---:|
| Setup/config | 0.00 s | 0.01 s | -0.01 s |
| Read NPZ frames | 4.33 s | 1.99 s | 2.33 s |
| Reference + O/H selection | 11.42 s | 6.40 s | 5.02 s |
| O-H neighbors + species | 95.85 s | 9.15 s | 86.70 s |
| Vector/angle + 2D histogram | 20.74 s | 4.15 s | 16.59 s |
| Finalize/normalization | 0.11 s | 0.00 s | 0.10 s |
| Write CSV/metadata | 0.39 s | 0.21 s | 0.18 s |
| Other overhead | 0.21 s | 0.06 s | 0.15 s |
| Total | 133.05 s | 21.98 s | 111.06 s |

The C++ route is about 6.05x faster end to end for this example.

## Correctness

The Python/scipy and C++ routes produced identical raw and normalized histograms:

```text
max_abs_diff: 0.0
exact: true
frames: 20001
bond_counts_equal: true
sample_counts_equal: true
bond_counts_total:
  OH-: 1367400
  H2O: 15463810
  H3O+: 0
```

The three CSV files produced by the real example CLI were byte-for-byte identical to the benchmark C++ output. The full CLI run, including CSV, metadata, and three PNG plots, completed in 25.5 s.

## Validation

Focused tests cover:

- C++ neighbor matrix versus the Python matrix implementation
- `oh_bond` histogram and counters
- `oh_bisector` histogram and counters
- current `dipole` behavior
- Python fallback behavior through the existing workflow

Run:

```bash
python -m pytest -q tests/test_v03_oh_orientation_cpp.py tests/test_v03_density_cpp.py
```
