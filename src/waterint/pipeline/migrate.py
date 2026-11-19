"""Convert a v0.6 single-module config into a v0.7 pipeline config with identical numerics.

v0.6 modules used different, implicit conventions; migration makes them explicit:
  proton ownership   density / rdf / msd / defect species: nearest O within selection.oh_cutoff (1.25), pbc [T, T, F]
                     oh-orientation / hbond: every O within cutoff counts the H (all_within_cutoff), NO pbc
  system profile     site-current / carrier-msd / hb-correlation / sfg-species: the mgo_water profile (or `profile:`)
                     all other modules: the generic water profile (no solid, absolute coordinates)
  inputs             segments / trajectories -> trajectory list; n_frames -> max_frames; events / carrier_tracks /
                     runs move into the analysis section
The v0.6 behaviour is reproduced; new configs should rely on the profile instead.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from waterint import registry
from waterint.system.profile import GENERIC_PROFILE, load_profile

DEFAULT_PREFIX = {"density": "density", "orientation": "angle_z", "hbond": "hbond", "rdf": "rdf", "msd": "msd",
                  "defect-msd": "defect_msd", "defect-conductivity": "defect_conductivity", "conductivity": "conductivity",
                  "identity-transition": "identity_transition", "proton-sharing": "proton_sharing",
                  "proton-sharing-hbond": "proton_hbond", "relabel": "relabelled", "sfg": "sfg_trajectory",
                  "site-current": "site_current", "site-conductance": "conductance", "carrier-track": "carrier",
                  "carrier-msd": "carrier_msd", "hb-correlation": "hbc", "sfg-species": "sfg_species",
                  "sfg-species-aggregate": "sfg_species_summary"}
PROFILE_MODULES = {"site-current", "carrier-msd", "hb-correlation", "sfg-species", "relabel"}
DEFAULT_MGO_PROFILE = "mgo_water"
DEFAULT_FORMAT = {"site-current": "lammpstrj", "carrier-msd": "gpumd-qnpz", "hb-correlation": "gpumd-qnpz",
                  "sfg-species": "gpumd-qnpz", "identity-transition": "lammpstrj", "relabel": "lammpstrj"}
LEGACY_IGNORED_SELECTION = {"hydrogen_assignment", "pbc", "neighbor_method", "neighbor_workers", "oxygen_chunk_size"}


def migrate_v06(module: str, config: dict[str, Any]) -> dict[str, Any]:
    kind = registry.resolve_type(module)
    old = copy.deepcopy(config)
    out_cfg = dict(old.get("output") or {})
    in_cfg = dict(old.get("input") or {})
    sections = {k: v for k, v in old.items() if k not in {"input", "system", "output", "profile"} and not k.startswith("_")}
    prefix = out_cfg.get("prefix")
    if prefix is None and kind == "sfg" and str((sections.get("sfg") or {}).get("mode", "single")) == "single":
        prefix = Path(str(sections["sfg"]["cf"])).with_suffix("").name
    analysis: dict[str, Any] = {"type": kind, "name": str(prefix or DEFAULT_PREFIX[kind]), **sections}
    prep: dict[str, Any] = {}
    system = dict(old.get("system") or {})

    # -- system profile ---------------------------------------------------------------------------------------------
    if kind in PROFILE_MODULES:
        prof_name = old.get("profile") or DEFAULT_MGO_PROFILE
        system["profile"] = prof_name
        if "type_map" not in in_cfg:
            tmap = load_profile(prof_name).get("lammps_type_map")
            if tmap:
                in_cfg["type_map"] = {int(k): str(v) for k, v in tmap.items()}
    else:
        system["profile"] = dict(GENERIC_PROFILE)

    # -- inputs -----------------------------------------------------------------------------------------------------
    if "segments" in in_cfg:
        seg = in_cfg.pop("segments")
        in_cfg["trajectory"] = seg if isinstance(seg, list) else [seg]
    if "trajectories" in in_cfg and "trajectory" not in in_cfg:
        in_cfg["trajectory"] = in_cfg.pop("trajectories")
    if kind in DEFAULT_FORMAT:
        in_cfg.setdefault("format", DEFAULT_FORMAT[kind])
    if in_cfg.get("format") == "gpumd-qnpz" and kind in PROFILE_MODULES:
        in_cfg.setdefault("wrap_xy", False)       # block analyses get raw positions (v0.6 iter_blocks / decode_chunk)

    # -- per-module conventions -------------------------------------------------------------------------------------
    if kind == "density":
        analysis["backend"] = (sections.get("density") or {}).get("backend", "auto")
        sel = dict(analysis.get("selection") or {})
        sel.setdefault("hydrogen_assignment", "nearest")
        analysis["selection"] = sel
    elif kind in {"orientation", "hbond"}:
        analysis["selection"] = {k: v for k, v in (analysis.get("selection") or {}).items() if k not in LEGACY_IGNORED_SELECTION}
        analysis["ownership"] = {"mode": "all_within_cutoff", "pbc": "none"}
        if kind == "orientation":
            analysis["backend"] = (sections.get("angle") or {}).get("backend", "auto")
        else:
            prep["backend"] = (sections.get("hbond") or {}).get("backend", "auto")
    elif kind in {"rdf", "msd", "conductivity"}:
        key = {"rdf": "rdf", "msd": "msd", "conductivity": "msd"}[kind]
        analysis["backend"] = (sections.get(key) or {}).get("backend", "auto")
    elif kind == "identity-transition":
        for key in ("stride", "max_frames", "start_timestep"):   # v0.6 read every frame of the dump
            in_cfg.pop(key, None)
    elif kind == "proton-sharing-hbond":
        ph = dict(analysis.get("proton_sharing_hbond") or {})
        ph.setdefault("l1_range", [1.5, 2.8])
        ph.setdefault("l2_range", [2.8, 4.0])
        analysis["proton_sharing_hbond"] = ph
    elif kind == "site-current":
        sc = dict(analysis.get("site_current") or {})
        for key, new in (("level", "output_level"), ("coarsen", "coarsen"), ("decomposition_coarsen", "decomposition_coarsen")):
            if key in out_cfg:
                sc.setdefault(new, out_cfg.pop(key))
        if "block_frames" in sc:
            in_cfg.setdefault("block_frames", sc["block_frames"])
        analysis["site_current"] = sc
    elif kind == "carrier-track":
        ct = dict(analysis.get("carrier_track") or {})
        ct["events"] = in_cfg.pop("events")
        analysis["carrier_track"] = ct
        in_cfg = {}
    elif kind == "carrier-msd":
        cm = dict(analysis.get("carrier_msd") or {})
        for key in ("carrier_tracks", "events"):
            if key in in_cfg:
                cm[key] = in_cfg.pop(key)
        analysis["carrier_msd"] = cm
    elif kind == "sfg-species":
        if "n_frames" in in_cfg:
            in_cfg["max_frames"] = in_cfg.pop("n_frames")
        sp = dict(analysis.get("sfg_species") or {})
        if "dt_ps" in in_cfg:
            sp.setdefault("dt_ps", in_cfg["dt_ps"])
        analysis["sfg_species"] = sp
    elif kind == "sfg-species-aggregate":
        analysis["runs"] = in_cfg.pop("runs")
        if in_cfg.get("reference"):
            analysis["reference"] = in_cfg.pop("reference")
        in_cfg = {}
    elif kind == "relabel":
        rc = dict(analysis.get("relabel") or {})
        if rc.get("lattice_ids"):
            raise ValueError("relabel.lattice_ids (an ID manifest) is not supported by the v0.7 relabel yet.")
        # v0.6 options name the type numbers of the proton / reference-cation / acceptor elements of the profile
        base = load_profile(system["profile"])
        inv = {str(v): int(k) for k, v in (base.get("lammps_type_map") or {}).items()}
        roles = {"h_type": base.get("protons", ["H"])[0], "o_type": base.get("acceptors", ["O"])[0],
                 "mg_type": (base.get("reference") or {}).get("elements", [None])[0]}
        in_cfg["type_map"] = {int(rc.pop(opt, inv.get(el))): el for opt, el in roles.items() if el is not None}
        over = {}
        if "oh_cutoff" in rc:
            over["cutoff_A"] = float(rc.pop("oh_cutoff"))
        if "tie_tolerance" in rc:
            over["tie_tolerance_A"] = float(rc.pop("tie_tolerance"))
        width = rc.pop("top_layer_width", None)
        for key in ("expected_mg_count", "expected_lattice_o_count", "progress_every", "definition"):
            rc.pop(key, None)
        if over or width is not None:
            prof = load_profile(system["profile"])
            if over:
                prof["h_assignment"] = {**prof.get("h_assignment", {}), **over}
            if width is not None:
                prof["reference"] = {**prof.get("reference", {}), "layer_width_A": float(width)}
            system["profile"] = {k: v for k, v in prof.items() if not k.startswith("_")}
        analysis["relabel"] = rc
    elif kind == "site-conductance":
        in_cfg = {}
    elif kind == "sfg":
        if str((sections.get("sfg") or {}).get("mode", "single")) != "trajectory":
            in_cfg = {}

    new = {"system": system, "output": out_cfg, "analyses": [analysis]}
    if in_cfg:
        new["input"] = in_cfg
    if prep:
        new["prep"] = prep
    for key in ("_config_path", "_config_dir"):
        if key in old:
            new[key] = old[key]
    return new
