# WaterInt v0.2

WaterInt v0.2 is an experimental version line based on v0. Its main design change is a workflow registry, so adding a new analysis method requires less editing in unrelated files.

Documentation website:

https://water-interface-analysis.syang4326m.workers.dev/

## What Changed From v0

The scientific organization remains the same:

```text
_00_io/            trajectory readers and NPZ cache support
_01_core/          shared coordinate, selection, and species helpers
_02_computation/   flat numerical computation files
_03_output/        CSV, metadata, and plotting helpers
_04_workflows/
  registry/             AnalysisModule definition, registry, and adding-module guide
  workflows/            config-driven run_<method>() orchestration
_05_ui/            local UI code
```

The leading underscore keeps these directories valid Python package names while the numeric prefix keeps the source tree sorted by data flow.

The registry records the module command name, help text, workflow entry point, aliases, and output-printing function. The CLI reads this registry to create subcommands automatically.
The local UI code is present in this version line, but the first v0.2 refactor wires the registry into the CLI only. A next iteration can make the UI build its module list from the same registry.

For the detailed checklist, see
[`waterint/_04_workflows/registry/ADDING_MODULE.md`](waterint/_04_workflows/registry/ADDING_MODULE.md).

## Installation

The repository root currently installs the active package line selected by the root `pyproject.toml`. To test v0.2 directly without changing the root package metadata, run commands with `PYTHONPATH`:

```bash
cd /path/to/waterint
PYTHONPATH=v0.2 python -m waterint.cli --help
```

To install v0.2 as the active editable package, update the root packaging metadata to use `v0.2` instead of `v0`:

```toml
[tool.setuptools.packages.find]
where = ["v0.2"]
include = ["waterint*"]
```

and in `setup.py`:

```python
package_dir={"": "v0.2"}
packages=find_packages(where="v0.2", include=["waterint", "waterint.*"])
```

Then install from the repository root:

```bash
python -m pip install -e .
```

## Adding A New Module

A new analysis method should usually add three pieces:

```text
waterint/_02_computation/<method>.py    pure numerical implementation
waterint/_04_workflows/workflows/<method>.py
                                          config-driven run_<method>() entry point
waterint/_04_workflows/registry/registry.py
                                          one AnalysisModule registration
```

Example registry entry:

```python
from waterint._04_workflows.registry.analysis_module import AnalysisModule
from waterint._04_workflows.workflows.diffusion import run_diffusion


def print_diffusion_outputs(result):
    print(f"Wrote: {result.csv_path}")
    if result.png_path:
        print(f"Wrote: {result.png_path}")
    print(f"Wrote: {result.metadata_path}")


AnalysisModule(
    name="diffusion",
    help="Run a diffusion analysis workflow.",
    run=run_diffusion,
    print_outputs=print_diffusion_outputs,
)
```

After registration, the CLI can expose the method without adding a new `if args.command == ...` block.

## Current Modules

- `density`: density profiles along an absolute, reference-relative, or slab-relative coordinate.
- `oh-orientation`: O-H angle distributions resolved along a coordinate.
- `hbond`: hydrogen-bond topology classification by oxygen species.
- `sfg`: trajectory-based interfacial O-H correlation and spectrum workflows.

`angle-z` is kept as a CLI alias for `oh-orientation`.

## Usage

```bash
PYTHONPATH=v0.2 python -m waterint.cli density --config v0.2/example/mgo_density/config_oh_h2o_npz.yaml
PYTHONPATH=v0.2 python -m waterint.cli oh-orientation --config v0.2/example/mgo_oh_orientation/config_oh_h2o_h3o_npz.yaml
PYTHONPATH=v0.2 python -m waterint.cli hbond --config v0.2/example/mgo_hbond/config_oh_h2o_h3o_npz.yaml
PYTHONPATH=v0.2 python -m waterint.cli sfg --config v0.2/example/mgo_sfg/config_100ps_npz.yaml
```

The MgO-water examples expect the shared 100 ps trajectory cache at:

```text
v0.2/example/shared/input/dump.MgO_water_2x2_equil.last100ps.unique.waterint.npz
```

The trajectory cache is large and should be distributed through Git LFS, a release asset, or an external data link rather than normal Git history.
