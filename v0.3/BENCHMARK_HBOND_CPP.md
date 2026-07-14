# H-Bond Python and C++ Cell-List Benchmark

Date: 2026-07-13

System: local macOS / Apple clang / Python 3.9

## Implementation

The benchmark compares three implementations of the same H-bond definition:

```text
legacy Python
  -> scipy O-H assignment
  -> each donor scans every oxygen
  -> Python loops evaluate D-H-A geometry

cell-list Python
  -> scipy O-H assignment
  -> Python O-O cell list
  -> NumPy evaluates local D-H-A candidates in one frame-level batch

cell-list C++
  -> shared C++ O-H cell list
  -> shared C++ O-O cell list
  -> C++ evaluates local D-H-A candidates
```

`chemistry.py` owns `PythonCutoffNeighborSearch` and `chemistry.hpp` owns its C++ counterpart, `CutoffNeighborSearch`. H-bond analysis consumes these shared spatial-search implementations instead of maintaining module-specific cell lists.

```yaml
hbond:
  backend: auto   # auto, python, or cpp
```

`auto` uses C++ when the native library is available and otherwise falls back to Python. `python` and `cpp` force one route for validation or benchmarking.

`backend: python` uses the Python cell-list implementation. The original all-oxygen scan is not retained in v0.3; its timings in this document were measured before replacement, and the implementation remains available in the v0.2 history.

## Full 100 ps Benchmark

Input:

- MgO-water NPZ trajectory
- 20001 frames
- 1896 atoms per frame
- O-H cutoff: 1.25 Angstrom
- O-O cutoff: 3.5 Angstrom
- minimum D-H-A angle: 150 degrees
- plotting and file output excluded from staged timing

| Step | Original Python | Cell-list Python | Cell-list C++ |
|---|---:|---:|---:|
| Read NPZ frames | 2.07 s | 2.08 s | 2.37 s |
| Select O/H atoms | 6.46 s | 6.96 s | 4.51 s |
| O-H species assignment | 82.54 s | 84.10 s | 9.96 s |
| H-bond geometry | 6588.80 s | 1421.92 s | 28.01 s |
| Topology grouping | 21.79 s | 22.69 s | 21.64 s |
| Other overhead | 4.83 s | 5.21 s | 0.19 s |
| **Total** | **6706.48 s** | **1542.96 s** | **66.67 s** |

The Python cell list is approximately 4.35x faster than the original Python implementation. The C++ route is approximately 100.6x faster than the original and 23.14x faster than the Python cell-list route. The real C++ CLI run including CSV, raw CSV, metadata, and PNG output completed in 65.59 s.

## Correctness

All three complete runs matched exactly:

```text
frames: 20001
grouped counts equal: true
raw topology counts equal: true
sample counts equal: true
OH- samples: 1367400
H2O samples: 7731905
H3O+ samples: 0
```

The implementation deliberately preserves the existing semantics: O-H species assignment is non-periodic, while H-bond D-H-A geometry uses `hbond.pbc`.

## Validation

Current v0.3 tests cover:

- C++ counts versus Python geometry
- Python cell-list queries versus brute-force distances on random systems
- periodic acceptors
- nearest-acceptor tie ordering
- multiple acceptors per hydrogen
- complete Python and C++ NPZ workflows
- density and O-H orientation regression tests after sharing the cell list

Run:

```bash
python -m pytest -q tests/test_v03_hbond_cpp.py tests/test_v03_density_cpp.py tests/test_v03_oh_orientation_cpp.py tests/test_v02_registry.py
```
