# Real-data regression scripts

These scripts compare v0.7 against the v0.6 reference on real trajectories and are not part of `pytest`. They need
data that is not distributed with the package; point them at it with environment variables:

| variable | used by | meaning |
|---|---|---|
| `WATERINT_V06` | all, `tests/test_v06_equivalence.py` | directory containing the v0.6 `waterint/` package (e.g. `v0/` of tag `v0.6`) |
| `WATERINT_REGRESSION_ROOT` | all | working directory: `data/mgo_2000f.lammpstrj` (2000-frame MgO-water LAMMPS dump), golden outputs, reports |
| `WATERINT_QNPZ_A`, `WATERINT_QNPZ_B` | `regress_phase2.py`, `check_blocks_gpu.py` | GPUMD qnpz runs (8 fs frames; 4 fs frames with velocities) |
| `WATERINT_MGO_DUMP`, `WATERINT_TIO2_DUMP` | `regress_real.py` | MgO(100)-water and rutile TiO2(110)-water LAMMPS dumps |
| `WATERINT_OMOL_CASES` | `tests/systems/run_omol_suite.py` | directory with `cases/<case>/relabelled_trajectory.lammpstrj` and `relabel_configs/<case>.json` |
| `WATERINT_CACHE` | all | build cache of the C++ kernels |

```bash
python tests/regression/regress_real.py --frames 2000
python tests/regression/regress_phase2.py [--only name,...]
python tests/regression/check_relabel.py
python tests/regression/check_blocks_gpu.py          # needs CUDA
python tests/systems/run_omol_suite.py --workers 4
```
