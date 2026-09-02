# Density C++ Backend Benchmark

Date: 2026-07-08

System: local macOS / Apple clang 15 / Python 3.9

## What Was Tested

v0.3 adds optional C++ backends for:

- density histogram binning
- O-H neighbor counting for oxygen-species classification

```yaml
density:
  backend: auto

selection:
  neighbor_method: cpp
```

The C++ code is intentionally narrow. It replaces small kernels while the Python computation modules remain the public API and fallback path.

## Correctness

The C++ backend was compared against the Python/NumPy backend on the MgO-water full 100 ps density example.

Result:

```text
same_header True
shape_py (310, 5)
shape_cpp (310, 5)
max_abs_diff 0.0
allclose_exact True
```

The Python and C++ workflow CSV outputs are exactly identical after using the full `bin_edges` array in the C++ density kernel and matching the previous O-H neighbor-count behavior.

## Density Histogram Only: Full Workflow Timing

Command shape:

```bash
/usr/bin/time -p env PYTHONPATH=v0.3 python -m waterint.cli density --config <benchmark-config>
```

Settings:

- trajectory: MgO-water 100 ps NPZ cache
- output plotting: disabled
- output: CSV + metadata only

Timing:

```text
backend: python
real 69.78 s
user 68.94 s
sys   2.46 s

backend: cpp
real 77.35 s
user 71.07 s
sys   2.87 s
```

The full workflow does not speed up from the density histogram backend alone. This is expected after profiling: density binning is not the dominant cost for this example.

## O-H Neighbor Count: Full Workflow Timing

Command shape:

```bash
/usr/bin/time -p env PYTHONPATH=v0.3 python -m waterint.cli density --config <benchmark-config>
```

Settings:

- trajectory: MgO-water 100 ps NPZ cache
- output plotting: disabled
- density histogram backend: `cpp`
- old route: `selection.neighbor_method: kdtree`
- new route: `selection.neighbor_method: cpp`

Timing:

```text
neighbor_method: kdtree
real 71.55 s
user 67.54 s
sys   1.73 s

neighbor_method: cpp
real 51.48 s
user 50.20 s
sys   2.69 s
```

The C++ neighbor-count backend gives an end-to-end speedup of about 1.39x on this MgO-water density workflow.

Correctness:

```text
same_header True
shape (310, 5) (310, 5)
max_abs_diff 0.0
exact True
```

The CSV output is exactly identical to the previous route.

## Kernel Timing

Synthetic histogram benchmark:

- values: 2,000,000 random coordinates
- bins: 310
- calls: 20

Timing:

```text
python: 4.1207 s total, 0.20603 s/call
cpp:    2.2055 s total, 0.11027 s/call
max_abs_diff 0.0
```

The C++ kernel is about 1.9x faster than NumPy histogramming in this strict-edge implementation.

An earlier uniform-bin formula was about 18x faster on the synthetic kernel, but it produced small bin-boundary differences on real workflow output. v0.3 uses the strict `bin_edges` implementation because identical output is more important than raw microbenchmark speed.

## Profiling Result

Profiling the original full MgO-water density workflow showed the main costs:

```text
oxygen_species_indices / classify_oxygen_by_h_count / neighbor search: ~50 s
NPZ frame iteration and reconstruction:                             ~29 s
coordinate/reference selection:                                      ~5-9 s
np.histogram density binning:                                        ~3 s
```

After adding the C++ O-H neighbor-count kernel, profiling shows the next major costs are NPZ frame reconstruction, selection masks, coordinate references, and Python-level species bookkeeping. A future optimization could expose a fuller native species-classification API that returns grouped oxygen indices directly.

## Experimental XYZ Reader Timing

v0.3 also includes an experimental C++ reader for fixed-topology standard XYZ files:

```yaml
input:
  format: xyz
  reader: cpp
```

Benchmark input:

- generated standard XYZ from the MgO-water NPZ cache
- 2000 frames
- 1896 atoms per frame
- XYZ size: 135 MB
- row format: `symbol x y z`

Reader-only timing before optimizing NPZ symbol reuse:

| Reader | Best time | Mean time | Frames |
|---|---:|---:|---:|
| Python XYZ reader | 6.81 s | 6.97 s | 2000 |
| C++ XYZ reader | 1.68 s | 1.74 s | 2000 |
| Current NPZ reader | 3.05 s | 3.27 s | 2000 |

The C++ XYZ reader is about 4x faster than the current Python XYZ reader on this text-parsing benchmark. It is also faster than the current NPZ reader for the same 2000-frame slice, which indicates that Python-side frame reconstruction and symbol mapping are now visible costs.

After adding fixed-topology symbol reuse to the NPZ reader:

| Reader | Best time | Mean time | Frames |
|---|---:|---:|---:|
| Python XYZ reader | 6.78 s | 6.90 s | 2000 |
| C++ XYZ reader | 1.69 s | 1.81 s | 2000 |
| Optimized NPZ reader | 1.33 s | 1.53 s | 2000 |

The practical recommendation is therefore:

1. Use `waterint convert` to create a binary NPZ cache for repeated analysis.
2. Use `input.reader: cpp` when analyzing a standard XYZ file directly and conversion is not convenient.
3. Keep `input.reader: python` as the most memory-frugal streaming fallback.

End-to-end density timing on the same 2000-frame XYZ sample:

| Workflow | Time | Frames | Correctness check |
|---|---:|---:|---|
| Python XYZ reader + C++ computation kernels | 13.85 s | 2000 | same selected-atom totals |
| C++ XYZ reader + C++ computation kernels | 8.61 s | 2000 | same selected-atom totals |

This is a useful speedup, but it is not a replacement for a binary trajectory cache. The current C++ XYZ reader reads the selected XYZ frames into memory before yielding frames to Python, so large all-frame XYZ runs can use more memory than the Python streaming reader.

## MgO-Water 100 ps Input Reader Timing

The real MgO-water 100 ps trajectory available locally is a LAMMPS dump plus a WaterInt NPZ cache:

```text
dump.MgO_water_2x2_equil.last100ps.unique.lammpstrj
dump.MgO_water_2x2_equil.last100ps.unique.waterint.npz
```

Trajectory size:

- 20001 frames
- 1896 atoms per frame
- LAMMPS dump: 1.9 GB
- NPZ cache: 1.1 GB

Reader-only timing, consuming every frame and touching coordinates:

| Reader | Time | Frames | Notes |
|---|---:|---:|---|
| Python LAMMPS dump reader | 89.61 s | 20001 | streaming text parser |
| C++ LAMMPS dump reader | 25.95-26.74 s | 20001 | experimental in-memory parser |
| Optimized NPZ reader | 1.56-1.82 s | 20001 | binary cache with fixed-topology symbol reuse |

Checksums, frame counts, atom counts, and first/last timesteps matched across all three routes:

```text
frames: 20001
atoms/frame: 1896
first timestep: 2436220
last timestep: 2636220
```

The C++ LAMMPS dump reader gives about a 3.4x reader-only speedup over the Python LAMMPS dump reader. The NPZ cache is still about 16x faster than the C++ LAMMPS dump reader, so the recommended workflow remains:

1. Convert large text trajectories to NPZ once.
2. Use NPZ for repeated analysis.
3. Use the C++ text reader when analyzing a text trajectory directly.

## Density Workflow Timing After Fast NPZ Reader

After adding fixed-topology symbol reuse to `read_npz`, the full MgO-water 100 ps density workflow was re-timed with the same non-overlapping stage timers as before.

Settings:

- trajectory: MgO-water 100 ps NPZ cache
- frames: 20001
- atoms per frame: 1896
- output plotting: disabled
- output: CSV + metadata only
- Python/scipy route: `selection.neighbor_method: kdtree`, `density.backend: python`
- C++ route: `selection.neighbor_method: cpp`, `density.backend: cpp`
- native library was pre-built before timing

Correctness:

```text
header_same: true
shape: 310 x 5
max_abs_diff: 0.0
exact: true
selected_atoms_total:
  OH-: 1367400
  H2O: 7731905
```

Timing:

| Step | Python/scipy route | C++ route | Saved |
|---|---:|---:|---:|
| Setup/config | 0.00 s | 0.00 s | -0.00 s |
| Read NPZ frames | 1.98 s | 1.69 s | 0.30 s |
| O species classification | 41.98 s | 25.65 s | 16.33 s |
| Coordinate projection/reference | 4.17 s | 4.26 s | -0.10 s |
| Histogram accumulation | 2.42 s | 1.49 s | 0.93 s |
| Finalize/normalization | 0.00 s | 0.00 s | 0.00 s |
| Write CSV/metadata | 0.01 s | 0.01 s | -0.00 s |
| Other overhead | 0.03 s | 0.03 s | 0.00 s |
| Total | 50.60 s | 33.14 s | 17.46 s |

Compared with the previous benchmark table, `Read NPZ frames` drops from about 21 s to about 2 s because `symbols` are now generated once for fixed-topology NPZ trajectories instead of once per frame. After this change, oxygen-species classification is again the dominant cost.

## Density Workflow Timing After Compact C++ Species Classification

The C++ oxygen-species path was then tightened further. The previous native path returned only per-oxygen H counts; Python still looped over oxygen atoms to build the `O2-`, `OH-`, `H2O`, `H3O+`, and `O_other` index arrays. The new compact kernel keeps the fast contiguous O/H coordinate arrays but returns the grouped oxygen indices directly.

Correctness:

```text
shape: 310 x 5
max_abs_diff: 0.0
exact: true
selected_atoms_total:
  OH-: 1367400
  H2O: 7731905
```

Timing:

| Step | Python/scipy route | C++ route | Saved |
|---|---:|---:|---:|
| Setup/config | 0.00 s | 0.00 s | -0.00 s |
| Read NPZ frames | 2.00 s | 1.58 s | 0.42 s |
| O species classification | 42.13 s | 17.39 s | 24.74 s |
| Coordinate projection/reference | 4.12 s | 3.79 s | 0.33 s |
| Histogram accumulation | 2.39 s | 1.27 s | 1.12 s |
| Finalize/normalization | 0.00 s | 0.00 s | 0.00 s |
| Write CSV/metadata | 0.00 s | 0.00 s | 0.00 s |
| Other overhead | 0.03 s | 0.03 s | 0.01 s |
| Total | 50.67 s | 24.05 s | 26.62 s |

This is about a 2.1x end-to-end speedup over the Python/scipy route for the MgO-water density example, with identical CSV output.

## Density Workflow Timing After C++ Cell List O-H Search

The compact C++ species-classification kernel was then changed from a direct `N_O x N_H` search to a cell-list search. Hydrogen atoms are binned into cutoff-sized spatial cells, and each oxygen checks only neighboring cells.

Correctness:

```text
shape: 310 x 5
max_abs_diff: 0.0
exact: true
selected_atoms_total:
  OH-: 1367400
  H2O: 7731905
```

Timing:

| Step | Python/scipy route | C++ route | Saved |
|---|---:|---:|---:|
| Setup/config | 0.00 s | 0.00 s | -0.00 s |
| Read NPZ frames | 1.93 s | 1.49 s | 0.45 s |
| O species classification | 41.44 s | 12.07 s | 29.36 s |
| Coordinate projection/reference | 4.05 s | 3.80 s | 0.25 s |
| Histogram accumulation | 2.36 s | 1.28 s | 1.08 s |
| Finalize/normalization | 0.00 s | 0.00 s | 0.00 s |
| Write CSV/metadata | 0.01 s | 0.01 s | 0.00 s |
| Other overhead | 0.03 s | 0.03 s | 0.01 s |
| Total | 49.82 s | 18.68 s | 31.15 s |

This is about a 2.7x end-to-end speedup over the Python/scipy route for this density example. Compared with the earlier compact direct-search kernel, the C++ route drops from about 24.05 s to about 18.68 s.
