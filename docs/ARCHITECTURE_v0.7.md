# WaterInt v0.7 architecture (design, 2026-11-17)

Design specification of the v0.7 refactor (2026-11-17). Baseline: the extended v0.6 tree, which is
the numerical reference. Not yet implemented: `system/autodetect.py` (`waterint profile detect`), the on-disk prep cache.

## 1. Goals

1. **One coherent architecture**: read → prepare (neighbours, H ownership, species, reference plane, layers, H bonds) → analyse (density, orientation, H-bond statistics, RDF, MSD, conductance, SFG, ...) → plot. Each step has exactly one implementation that every analysis reuses.
2. **System-agnostic**: changing the solid (MgO, TiO2, Al2O3, carbonates, metals, ...) means changing a profile, never the code. No element name, charge, cutoff or geometry constant is hard-coded outside profiles/defaults.
3. **Fast without being complicated**: read once, prepare once, feed many analyses in one pass; C++ only in hot paths; numpy elsewhere.
4. **Backends are options, not dependencies**: every heavy kernel has a Python reference, an optional CPU (C++/OpenMP) and an optional GPU (CUDA) implementation behind one signature. The core package imports and runs without a compiler or CUDA.
5. **Web docs stay in sync automatically**: module list, parameters and config schema are generated from the registry (`tools/gen_docs.py`) for the documentation site, https://water-interface-analysis.syang4326m.workers.dev/.
6. **Tested on many systems**: MgO regression against v0.6-ext + 27 MACE-MH-1/OMOL oxide/carbonate/metal interface trajectories.

## 2. Package layout

```
waterint/
  io/                 reading only
    frame.py          Frame (positions, symbols/types, velocities, cell, step, time), Cell (orthorhombic/triclinic, pbc)
    readers/          lammpstrj (py + cpp), xyz (py + cpp), npz, gpumd_qnpz (prefetch), events tables
    trajectory.py     open_trajectory(spec) -> Trajectory: multi-file/segment concatenation, timestep dedup,
                      range/stride, iter_frames(), iter_chunks(n) (arrays [n, natom, 3]) for batch kernels
  system/             what the system is
    profile.py        SystemProfile dataclass + load/validate/resolve (all system-specific numbers)
    autodetect.py     draft profile from frame 0 (elements, solid, slab normal/side, reference layer, type map)
    profiles/*.yaml   mgo_water_standard, mgo_water_zwindows_library, generic_oxide_water, ...
  prep/               per-frame preparation, computed once and shared
    neighbors.py      one cell-list/neighbour API (minimum image, triclinic, open z)
    ownership.py      H -> acceptor ownership (nearest within cutoff, tie rule), free H
    species.py        acceptor species labels by (solid?, nH) from the profile tables
    solid.py          solid identity: mode frozen (frame 0, v0.6 behaviour) | dynamic (element/coordination rule)
    reference.py      z_ref per frame, z_rel, named windows/layers
    hbonds.py         ONE geometric H-bond definition (R_OO, angle or R_HA criteria from profile/config)
    labelled.py       LabelledFrame = Frame + lazily computed, cached prep products
    events.py         event tables (waterint-events-1) write/read; prep cache on disk for repeated runs
  analysis/           the science; each analysis is an Analysis subclass
    base.py           Analysis protocol (see 3), Result (save/load, metadata), kinds: stream | post
    structure/        density, orientation, hbond_stats, rdf, angle_z
    dynamics/         msd, defect_msd, identity_transition, carrier_track, carrier_msd, hb_correlation
    transport/        conductivity, defect_conductivity, site_current, site_conductance
    spectra/          sfg, sfg_species (+ aggregate)
    free_energy/      proton_sharing, proton_sharing_hbond
    relabel.py        relabelled-trajectory writer (it is an analysis that writes a trajectory)
  backends/
    registry.py       kernel registry: name -> {python, cpu, gpu}; select(requested, available) with fallback + log
    cpu/              *.cpp + build.py (on-demand compile, cache dir, -fopenmp when supported; from v0.6 _native.py)
    gpu/              optional: *.cu + build/loader (nvcc on demand or prebuilt); lazily imported; never required
  pipeline/
    config.py         config schema (dataclasses or dict schema) + validation + v0.6-config migration
    run.py            run(config): trajectory -> union of prep requirements -> all stream analyses in ONE pass
                      -> post analyses (DAG by declared inputs) -> results -> optional plots
  plot/               one module per Result type; reads saved results only; never computes physics
  registry.py         module metadata (name, description, kind, requires, config schema, backends, outputs)
                      used by CLI, UI and the docs generator
  cli.py              `waterint run cfg.yaml`, `waterint <module> --config cfg.yaml` (v0.6 names kept),
                      `waterint profile detect traj`, `waterint docs build`
  ui/                 existing desktop/server UI, re-pointed to registry + pipeline
tools/gen_docs.py     registry -> site pages (module reference + config schema)
tests/                unit (per layer), backend equivalence, regression (vs v0.6-ext golden), systems (oxides)
```

Rules:
- Dependency direction is strictly `io -> system -> prep -> analysis -> pipeline -> cli/ui`; `plot` depends only on Result files; `backends` is a leaf used by io/prep/analysis. No analysis reads files or decodes trajectories itself; no analysis plots.
- No analysis re-implements neighbour search, ownership, species, reference plane or H-bond criteria. If an analysis needs a variant, it is a parameter of the prep step.
- No system constants in code. Defaults that are not system-specific (bin widths, etc.) live in the analysis config schema with documented defaults.

## 3. Analysis contract

```python
class Analysis:
    name: str                      # registry name, e.g. "density"
    kind: Literal["stream", "post"]
    requires: set[str]             # prep products: {"ownership","species","reference","windows","hbonds","velocities",...}
    def __init__(self, cfg, profile, run_info): ...
    def update(self, lf: LabelledFrame) -> None          # stream: called once per frame (or update_chunk for batched kernels)
    def finalize(self) -> Result
    # post analyses: inputs: list[str] (results of other analyses or files); run(inputs) -> Result
```
- Time-correlation analyses (MSD, conductance, SFG, hb-correlation) store compact per-frame data in `update` and correlate in `finalize` (or stream with fixed memory where v0.6 already does).
- Kinds (added in phase 2): `stream` (per frame, LabelledFrame), `block` (`update_block(block)`: arrays of many frames for
  batch C++/GPU kernels; `io/blocks.py`), `post` (no trajectory; `finalize()` reads results of earlier analyses in the
  run by name, or files). A run with any block analysis reads blocks and feeds stream analyses from `Block.frames()`,
  which equal the frame readers' frames: one pass either way.
- `Result` = arrays + metadata (config, profile id, versions, backend actually used, frame range, timing) saved as npz/csv/json; plots are produced from saved Results.

## 4. Prep products and caching

`LabelledFrame` computes products on first access, in dependency order: neighbours -> ownership -> species (needs solid) -> reference/windows -> hbonds. The pipeline asks each analysis for `requires` and computes only the union. Optional disk cache of per-frame labels/events (`prep.cache: path`) so repeated analyses on long runs skip prep.

## 5. Backends

- Every heavy kernel registers `python` (reference, always present), optionally `cpu`, optionally `gpu`.
- Config: `backend: auto | python | cpu | gpu` per analysis (and `prep.backend`). `auto` = gpu if registered+available+worth it (batch size), else cpu, else python. An explicit request that cannot be met is an error; `auto` fallbacks are logged in Result metadata.
- GPU code lives only in `backends/gpu/`, is imported lazily, and receives chunks (`iter_chunks`) to amortise transfers. Candidates: prep (neighbours+ownership+species), hb_correlation (exists), SFG correlation. Not density/MSD.
- Equivalence tests: python vs cpu vs gpu on fixed inputs (bitwise for integer counts where possible, else documented tolerance).

## 6. Generality

- Profile fields (from v0.6 `profiles/`): type map, formal charges, protons, acceptors, h_assignment cutoff + tie rule, pbc, surface normal axis (new; default z) and side (top/bottom/both, new), reference (elements/mode/layer width), solid (mode frozen|dynamic, elements, max_z_rel, expected counts), species tables, windows, tracked species, current groups.
- `generic_oxide_water.yaml`: solid = all non-H/non-O atoms + O with zero H (dynamic), reference = highest 0.7 A layer of solid atoms (the rule used for the 27 OMOL density profiles, 2026-08-17).
- `autodetect` writes a draft profile from frame 0 for a new system; users confirm it.
- MgO-specific rules (carrier relay rules, SFG windows, z windows) are profile/config parameters, documented as such.

## 7. Compatibility and verification

- Golden outputs: run v0.6-ext on a fixed MgO test set (density, orientation, hbond, rdf, msd, defect-msd, conductivity, defect-conductivity, site-current, site-conductance, carrier-track, carrier-msd, hb-correlation, sfg, sfg-species, proton-sharing(+hbond), relabel, identity-transition) and store them under `$WATERINT_REGRESSION_ROOT/golden/` with the exact configs.
- Every ported analysis must reproduce its golden output (integer counts exact; floats to 1e-12 relative unless a documented reason, e.g. summation order, then <= 1e-10).
- v0.6 configs keep working through `pipeline/config.py` migration (or are converted once by `waterint migrate-config`).
- System suite: all structure analyses + species/identity analyses run on the 27 OMOL cases with only the generic profile; sanity checks: atom/H conservation, species counts sum to O count, density integral = atom count, no NaN.
- Performance benchmark kept in `tests/benchmarks/`: one-pass multi-analysis vs v0.6 sequential runs on the same data.

## 8. Phases

1. Foundation: io, system, prep, backends registry + cpu build, analysis base, pipeline, plot base, registry/CLI; port density, orientation, hbond_stats, rdf, msd; golden data + regression harness.
2. Port the rest in parallel groups: (A) transport + defect + identity; (B) carrier + hb_correlation(+GPU) + proton_sharing; (C) spectra.
3. Generality + docs: generic profile + autodetect, OMOL 27-case suite, docs generator -> docs site, UI re-pointing, remove the old `v0.6/` tree.

Status (2026-11-19): phases 1-3 done except autodetect and the prep disk cache; v0.7 is the main line.
