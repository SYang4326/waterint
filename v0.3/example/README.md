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
cache (`20001` frames). The default SFG example remains bounded to a short
window so it also runs on installations that fall back to Python:

- `mgo_density/`: OH- and H2O mass density relative to the MgO slab surface.
- `mgo_oh_orientation/`: z-resolved OH-bond orientation distributions for OH-,
  H2O, and H3O+.
- `mgo_hbond/`: hydrogen-bond topology fractions by oxygen species.
- `mgo_sfg/`: trajectory-based SFG response on a short frame window by default.

`mgo_sfg/config_100ps_npz_full.yaml` is included for the full 100 ps SFG run,
but it is not part of the default example command. On the benchmark machine,
the full C++-assisted CLI run took 36.5 s; the pure Python analysis baseline
took about 29 minutes.

Run the default set from the repository root. This includes the full density
and OH-orientation examples, plus the short SFG example:

```bash
PYTHONPATH=v0.3 python v0.3/example/run_examples.py
```

The full H-bond example is slower, so it is opt-in:

```bash
PYTHONPATH=v0.3 python v0.3/example/run_examples.py --include-hbond
```

Run a single workflow:

```bash
PYTHONPATH=v0.3 python v0.3/example/run_examples.py --only density
PYTHONPATH=v0.3 python v0.3/example/run_examples.py --only oh_orientation
PYTHONPATH=v0.3 python v0.3/example/run_examples.py --only hbond
PYTHONPATH=v0.3 python v0.3/example/run_examples.py --only sfg
```

Run one workflow manually:

```bash
PYTHONPATH=v0.3 python -m waterint.cli density --config v0.3/example/mgo_density/config_oh_h2o_npz.yaml
PYTHONPATH=v0.3 python -m waterint.cli oh-orientation --config v0.3/example/mgo_oh_orientation/config_oh_h2o_h3o_npz.yaml
PYTHONPATH=v0.3 python -m waterint.cli hbond --config v0.3/example/mgo_hbond/config_oh_h2o_h3o_npz.yaml
PYTHONPATH=v0.3 python -m waterint.cli sfg --config v0.3/example/mgo_sfg/config_100ps_npz.yaml
```

Run the complete 100 ps SFG example:

```bash
PYTHONPATH=v0.3 python -m waterint.cli sfg --config v0.3/example/mgo_sfg/config_100ps_npz_full.yaml
```

The shared `.npz` file is about 1.1 GB. For GitHub publication, keep it outside
normal git history, for example with Git LFS, a release asset, or an external
download link used by the example page.
