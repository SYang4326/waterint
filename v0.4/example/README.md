# waterint MgO-water examples

This directory is a publication-facing example set for the refactored
`waterint` package. It intentionally keeps one representative MgO-water
system instead of many small toy cases, so the examples show the same interface
being reused across the main analysis workflows.

The shared 100 ps trajectory cache is stored once at:

```text
shared/input/dump.MgO_water_2x2_equil.last100ps.unique.waterint.npz
```

The density, OH-orientation, and H-bond configs use the full 100 ps trajectory
cache (`20001` frames). The MSD and RDF default examples are deliberately
bounded quick runs; their sibling configs are ready for the full trajectory.
The default SFG example also remains bounded to a short window so it runs on
installations that fall back to Python:

- `mgo_density/`: OH- and H2O mass density relative to the MgO slab surface.
- `mgo_oh_orientation/`: z-resolved OH-bond orientation distributions for OH-,
  H2O, and H3O+.
- `mgo_hbond/`: hydrogen-bond topology fractions by oxygen species.
- `mgo_msd/`: layer-selected 2D oxygen MSD relative to the MgO surface.
- `mgo_rdf/`: O-O/O-H and species-selective RDF configurations.
- `mgo_rdf/config_relabelled_oxygen_groups.yaml`: per-frame relabelled
  Mg, lattice O, layer-1 OH-/H2O, layer-2 OH-, and bulk H2O source groups.
  It writes Mg-O/Mg-H plus each oxygen group's O-O/O-H RDFs.
- `mgo_sfg/`: trajectory-based SFG response on a short frame window by default.

`mgo_sfg/config_100ps_npz_full.yaml` is included for the full 100 ps SFG run,
but it is not part of the default example command. On the benchmark machine,
the full C++-assisted CLI run took 36.5 s; the pure Python analysis baseline
took about 29 minutes.

Run the default set from the repository root. This includes the full density
and OH-orientation examples, plus the short SFG example:

```bash
PYTHONPATH=v0.4 python v0.4/example/run_examples.py
```

The full H-bond example is slower, so it is opt-in:

```bash
PYTHONPATH=v0.4 python v0.4/example/run_examples.py --include-hbond
```

Run a single workflow:

```bash
PYTHONPATH=v0.4 python v0.4/example/run_examples.py --only density
PYTHONPATH=v0.4 python v0.4/example/run_examples.py --only oh_orientation
PYTHONPATH=v0.4 python v0.4/example/run_examples.py --only hbond
PYTHONPATH=v0.4 python v0.4/example/run_examples.py --only msd
PYTHONPATH=v0.4 python v0.4/example/run_examples.py --only rdf
PYTHONPATH=v0.4 python v0.4/example/run_examples.py --only sfg
```

Run one workflow manually:

```bash
PYTHONPATH=v0.4 python -m waterint.cli density --config v0.4/example/mgo_density/config_oh_h2o_npz.yaml
PYTHONPATH=v0.4 python -m waterint.cli oh-orientation --config v0.4/example/mgo_oh_orientation/config_oh_h2o_h3o_npz.yaml
PYTHONPATH=v0.4 python -m waterint.cli hbond --config v0.4/example/mgo_hbond/config_oh_h2o_h3o_npz.yaml
PYTHONPATH=v0.4 python -m waterint.cli msd --config v0.4/example/mgo_msd/config_h2o_layer.yaml
PYTHONPATH=v0.4 python -m waterint.cli rdf --config v0.4/example/mgo_rdf/config_oo_oh.yaml
PYTHONPATH=v0.4 python -m waterint.cli rdf --config v0.4/example/mgo_rdf/config_relabelled_oxygen_groups.yaml
PYTHONPATH=v0.4 python -m waterint.cli sfg --config v0.4/example/mgo_sfg/config_100ps_npz.yaml
```

Run the complete 100 ps SFG example:

```bash
PYTHONPATH=v0.4 python -m waterint.cli sfg --config v0.4/example/mgo_sfg/config_100ps_npz_full.yaml
```

The shared `.npz` file is about 1.1 GB. For GitHub publication, keep it outside
normal git history, for example with Git LFS, a release asset, or an external
download link used by the example page.

## Relabelled Oxygen RDF Example

`mgo_rdf/config_relabelled_oxygen_groups.yaml` labels source oxygen atoms in
every sampled frame using the existing nearest-O O-H assignment and the MgO
surface reference:

- `lattice_O`: `O2-` under the O-H coordination criterion.
- `layer1_OH` and `layer1_H2O`: `0.7 <= z_rel < 2.5 A`.
- `layer2_OH`: `2.5 <= z_rel < 4.0 A`.
- `bulk_H2O`: `4.0 <= z_rel < 30.0 A`.

Each group writes both source-O to all-O and source-O to all-H RDFs. The
configuration also writes Mg-O and Mg-H. The
included configuration samples every tenth saved frame (`2001` frames over
the 100 ps trajectory); set `input.stride: 1` for all `20001` frames. On this
machine, the sampled run takes about 67 seconds, whereas the full run is
expected to take roughly 10 minutes with the current C++ kernels.
