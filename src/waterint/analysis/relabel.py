"""Relabelled LAMMPS trajectory: every acceptor (O) gets a type code for its species, profile-driven.

Classification uses prep only: canonical proton ownership (profile h_assignment), solid identity (profile solid,
frozen at the first frame), z_ref (profile reference). Type-code catalog, generated from the profile:
    non-acceptor elements in type_map order (types renumbered 1..k; H/Mg keep 1/2 for MgO-water)
    solid acceptor labels (profile species.solid table, by nH)
    <region>_<label> for every region (profile ``relabel.regions`` windows, default all windows) x free-species table
    other_region_<label> (acceptors outside the regions)
With the MgO standard profile this is the v0.6 catalog (codes 1-27) and the relabelled dump is byte-identical.
Outputs: <name>.lammpstrj, <name>.lammpstrj.lattice_O_ids.txt (frozen solid acceptor IDs; frozen solids only), <name>_type_map.tsv,
<name>_audit.csv (per frame), <name>_zref.csv, <name>_warnings.csv, <name>_metadata.json.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from waterint.analysis.base import Analysis, Result
from waterint.io.writers.lammpstrj_relabel import RelabelStream
from waterint.prep.labelled import LabelledFrame
from waterint.prep.reference import reference_value


def _token(label: str, names: dict[str, str]) -> str:
    return names.get(label, label.replace("+", "plus").replace("-", "minus"))


def build_catalog(profile, type_map: dict[int, str], regions: list[str], names: dict[str, str]):
    """[(code, name, definition)], element->code for non-acceptors, solid label->code, (region, free label)->code."""
    acceptors = set(profile.ownership.acceptors)
    cat, elem_code = [], {}
    for t in sorted(type_map):
        el = type_map[t]
        if el in acceptors or el in elem_code:
            continue
        elem_code[el] = len(cat) + 1
        cat.append((elem_code[el], names.get(el, el), f"input element {el}; not an acceptor"))
    solid_code = {}
    for nh, label in enumerate(profile.solid_species or ()):
        if label in solid_code:
            continue
        solid_code[label] = len(cat) + 1
        more = " or more" if nh == len(profile.solid_species) - 1 else ""
        cat.append((solid_code[label], label, f"solid acceptor with {nh}{more} owned protons"))
    windows = {w[0]: (w[1], w[2]) for w in profile.windows}
    region_code = {}
    for region in [*regions, "other_region"]:
        for nh, label in enumerate(profile.free_species):
            if (region, label) in region_code:
                continue
            region_code[(region, label)] = len(cat) + 1
            where = (f"z_rel in [{windows[region][0]:.2f},{windows[region][1]:.2f})" if region != "other_region"
                     else "z_rel outside the regions")
            more = " or more" if nh == len(profile.free_species) - 1 else ""
            cat.append((region_code[(region, label)], f"{region}_{_token(label, names)}",
                        f"non-solid acceptor, {where}, nH={nh}{more}"))
    return cat, elem_code, solid_code, region_code


class Relabel(Analysis):
    type = "relabel"
    kind = "post"
    v06_name = "relabel"
    description = ("Concatenate and timestep-deduplicate LAMMPS dumps and write a relabelled dump: every acceptor gets a type "
                   "code for (solid label | region x species), from the profile.")
    params = {"relabel.regions": ("profile relabel.regions or all windows", "named profile windows used as regions"),
              "relabel.max_frames": (0, "stop after this many frames (0 = all)"),
              "relabel.solid_escape_z": (1.5, "solid atoms with z_rel >= this are counted as escaped (warning)"),
              "relabel.fail_on_solid_escape": (False, "abort when the solid does not stay intact"),
              "backend": ("auto", "relabel stream: auto | cpu | python")}

    def __init__(self, cfg: dict[str, Any], run):
        super().__init__(cfg, run)
        self.rc = dict(cfg.get("relabel") or {})
        spec = run.input_spec
        if spec is None or spec.format != "lammpstrj":
            raise ValueError("relabel reads and writes LAMMPS dumps; set input.format: lammpstrj.")
        self.inputs = [Path(p) for p in spec.paths]
        missing = [str(p) for p in self.inputs if not p.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing input trajectory files: {missing}")
        prof = self.profile
        rel = dict(prof.raw.get("relabel") or {})
        self.regions = list(self.rc.get("regions", rel.get("regions", [w[0] for w in prof.windows])))
        unknown = [r for r in self.regions if r not in {w[0] for w in prof.windows}]
        if unknown:
            raise ValueError(f"relabel regions {unknown} are not windows of profile {prof.profile_id}.")
        self.names = dict(rel.get("names") or {})
        self.type_map = dict(run.prep.type_map)
        if not self.type_map:
            raise ValueError("relabel needs a type map (input.type_map or profile lammps_type_map).")

    def finalize(self) -> Result:
        prof, prep = self.profile, self.run.prep
        out = self.run.output_dir
        out.mkdir(parents=True, exist_ok=True)
        base = out / self.name
        traj = Path(f"{base}.lammpstrj")
        cat, elem_code, solid_code, region_code = build_catalog(prof, self.type_map, self.regions, self.names)
        windows = [(w[0], w[1], w[2]) for w in prof.windows if w[0] in self.regions]
        ref_spec = prep.default_coordinate
        ref_elements = tuple(e for e in prof.reference.get("elements", []) if e != "@solid")
        escape_z = float(self.rc.get("solid_escape_z", 1.5))
        solid_tab = np.array([solid_code[l] for l in (prof.solid_species or ("",))] if prof.solid_species else [0], dtype=np.int64)
        region_tab = np.array([[region_code[(r, l)] for l in prof.free_species] for r in [*(w[0] for w in windows), "other_region"]],
                              dtype=np.int64)
        max_frames = int(self.rc.get("max_frames", 0))
        t0 = time.time()
        totals = {"n_H_total": 0, "n_H_assigned": 0, "n_unassigned_H": 0, "nearest_O_tie": 0, "other_region_O": 0,
                  "solid_acceptor_escape_samples": 0, "reference_atom_escape_samples": 0}
        label_tot = {c[0]: 0 for c in cat}
        frames = 0
        with RelabelStream(self.inputs, traj, self.type_map, backend=str(self.cfg.get("backend", "auto"))) as rs, \
                open(f"{base}_audit.csv", "w", encoding="utf-8", newline="\n") as audit, \
                open(f"{base}_zref.csv", "w", encoding="utf-8", newline="\n") as zf, \
                open(f"{base}_warnings.csv", "w", encoding="utf-8", newline="\n") as warn:
            audit.write("frame_idx,step,z_ref_A,n_H_total,n_H_assigned,n_unassigned_H,nearest_O_tie,other_region_O,"
                        + ",".join(c[1] for c in cat) + "\n")
            zf.write("frame_idx,step,z_ref_A\n")
            warn.write("warning,frame_idx,step,atom_id,oxygen_id,distance_A,z_rel_A,nH,region\n")
            for frame, ids in rs:
                lf = LabelledFrame(frame, prep)
                own = lf.ownership()
                solid = lf.solid_mask()
                if frames == 0 and prof.solid.get("mode") == "frozen":
                    lattice_ids = np.sort(ids[np.flatnonzero(solid)])
                    Path(f"{traj}.lattice_O_ids.txt").write_text(
                        "# frozen solid acceptor atom IDs (profile solid definition)\n" + "".join(f"{i}\n" for i in lattice_ids),
                        encoding="utf-8")
                zref = reference_value(lf.positions, ref_spec, lambda els: lf.mask(els))
                zf.write(f"{frames},{frame.step},{zref:.12g}\n")
                codes = np.zeros(len(ids), dtype=np.int64)
                for el, code in elem_code.items():
                    codes[lf.mask((el,))] = code
                owner = own.owner()
                unassigned = int(np.sum(owner < 0))
                ties = int(np.sum(own.tied > 1)) if own.tied is not None else 0
                for h in own.proton_idx[owner < 0]:
                    warn.write(f"unassigned_H,{frames},{frame.step},{ids[h]},,,,,\n")
                if own.tied is not None:
                    for k in np.flatnonzero(own.tied > 1):
                        warn.write(f"nearest_O_tie,{frames},{frame.step},{ids[own.proton_idx[k]]},{ids[owner[k]]},,,,\n")
                zrel = lf.positions[:, prof.normal_axis] - zref
                acc, nh = own.acceptor_idx, own.counts.astype(np.int64)
                za = zrel[acc]
                is_solid = solid[acc] if prof.solid_species else np.zeros(acc.size, dtype=bool)
                region_i = np.full(acc.size, len(windows), dtype=np.int64)          # default: other_region
                for wi in range(len(windows) - 1, -1, -1):
                    region_i[(za >= windows[wi][1]) & (za < windows[wi][2])] = wi
                free_codes = region_tab[region_i, np.minimum(nh, region_tab.shape[1] - 1)]
                codes[acc] = np.where(is_solid, solid_tab[np.minimum(nh, solid_tab.size - 1)], free_codes)
                other = int(np.sum(~is_solid & (region_i == len(windows))))
                escaped = np.flatnonzero(is_solid & (za >= escape_z))
                totals["solid_acceptor_escape_samples"] += int(escaped.size)
                for k in escaped:
                    warn.write(f"solid_acceptor_escaped,{frames},{frame.step},{ids[acc[k]]},,,{za[k]:.12g},{int(nh[k])},solid\n")
                ref_idx = lf.indices(ref_elements)
                esc = int(np.sum(zrel[ref_idx] >= escape_z))
                totals["reference_atom_escape_samples"] += esc
                if esc and bool(self.rc.get("fail_on_solid_escape", False)):
                    raise RuntimeError(f"Solid integrity violated at timestep {frame.step}.")
                if np.any(codes == 0):
                    raise RuntimeError(f"atoms without a type code at timestep {frame.step} (type map incomplete?)")
                rs.write(codes)
                counts = np.bincount(codes, minlength=len(cat) + 1)
                nh_total = int(own.proton_idx.size)
                row = [frames, frame.step, f"{zref:.12g}", nh_total, nh_total - unassigned, unassigned, ties, other]
                audit.write(",".join(map(str, row)) + "," + ",".join(str(int(counts[c[0]])) for c in cat) + "\n")
                for c in cat:
                    label_tot[c[0]] += int(counts[c[0]])
                for key, val in (("n_H_total", nh_total), ("n_H_assigned", nh_total - unassigned), ("n_unassigned_H", unassigned),
                                 ("nearest_O_tie", ties), ("other_region_O", other)):
                    totals[key] += val
                frames += 1
                if max_frames and frames >= max_frames:
                    break
            duplicates = rs.duplicates
            stream_backend = rs.backend
        with open(f"{base}_type_map.tsv", "w", encoding="utf-8", newline="\n") as tm:
            tm.write("type_code\tcanonical_label\tdefinition\n")
            for code, name, definition in cat:
                tm.write(f"{code}\t{name}\t{definition}\n")
        self.frames = frames
        intact = totals["solid_acceptor_escape_samples"] == 0 and totals["reference_atom_escape_samples"] == 0
        meta = {"tool": "waterint-v0.7-relabel", "profile": prof.profile_id, "inputs": [str(p) for p in self.inputs],
                "duplicate_timesteps_skipped": duplicates, "output": str(traj), "frames_processed": frames,
                "regions": self.regions, "hydrogen_assignment": {"rule": "canonical nearest (prep.ownership)",
                "cutoff_A": prof.ownership.cutoff, "tie_tolerance_A": prof.ownership.tie_tolerance},
                "solid_integrity": {"escape_z_rel_A": escape_z, **{k: totals[k] for k in totals if "escape" in k}, "intact": intact},
                "audit_totals": {k: v for k, v in totals.items() if "escape" not in k},
                "label_counts_total": {name: label_tot[code] for code, name, _ in cat},
                "stream_backend": stream_backend, "elapsed_s": round(time.time() - t0, 3)}
        if not intact:
            import warnings
            warnings.warn("Solid integrity violated (see <name>_warnings.csv); species labels assume an intact solid.",
                          RuntimeWarning, stacklevel=2)
        return Result(self.type, self.name, {}, meta, outputs={"lammpstrj": str(traj)})

    def write(self, result: Result, outdir: Path) -> dict[str, Any]:
        base = outdir / self.name
        return {"lammpstrj": f"{base}.lammpstrj", "type_map": f"{base}_type_map.tsv", "audit": f"{base}_audit.csv",
                "zref": f"{base}_zref.csv", "warnings": f"{base}_warnings.csv"}
