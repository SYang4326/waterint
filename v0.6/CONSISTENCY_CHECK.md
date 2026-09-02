# Proton-sharing H-bond backend consistency

The `proton-sharing-hbond` workflow has two calculation paths. `backend: python`
uses the existing WaterInt nearest-O assignment and NumPy geometry; `backend: cpp`
uses `proton_sharing_hbond.cpp` for assignment, D-H-A filtering, CN
classification, and histogram accumulation. Both paths use the same contract:

* nearest unique O assignment with an O-H cutoff of 1.50 A;
* 2.20 <= R_OO < 3.20 A and D-H-A >= 150 degrees;
* every passing donor/acceptor pair is retained;
* CN is the number of distinct passing donor O atoms per L1 OH- acceptor;
* unit pair weights and global pooled normalization.

The native source compiles with the WaterInt C++ build command (`-std=c++17 -O3 -shared -fPIC`).
A direct ABI smoke test on one donor/acceptor pair gives one retained pair and
one acceptor with CN=1.

## Anneal verification (PASS)

Input: replica 00 anneal trajectory, first 100 frames, with the exact slab
reference and ranges in `waterint_hbond_cn_check.yaml`. The automated report is
at `.../proton_sharing_fes/output/waterint_hbond_cn_consistency/CONSISTENCY_CHECK.json`.

| Surface | Python pairs | C++ pairs | max abs. bin difference |
| --- | ---: | ---: | ---: |
| L1-L1 all | 1866 | 1866 | 0 |
| L1-L1 CN1 | 217 | 217 | 0 |
| L1-L1 CN2 | 1280 | 1280 | 0 |
| L1-L1 CN3 | 369 | 369 | 0 |
| L1-L1 CN4 | 0 | 0 | 0 |
| L1-L2 | 3849 | 3849 | 0 |

The production parser/native smoke test also passed on production replica 00,
segment 01 (two frames): 37 L1-L1 and 76 L1-L2 retained pairs.
