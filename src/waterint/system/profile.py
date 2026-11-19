"""System profiles: all system-specific definitions (charges, H ownership, reference plane, solid, species,
windows, current groups) live in a YAML profile, so one code path serves every interface system."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml

PROFILE_DIR = Path(__file__).resolve().parent / "profiles"
DEFAULT_PROFILE = "mgo_water"
# names used before v0.7 (still accepted)
PROFILE_ALIASES = {"mgo_water_standard_2026-11-05": "mgo_water",
                   "mgo_water_zwindows_library_2026-11-06": "mgo_water_zwindows"}
SPECIAL_TERMS = ("free_H", "transfer", "image")
REF_MODES = {"none": 0, "top_layer": 1, "bottom_layer": 2, "fixed": 3, "mean": 4}


def load_profile(name_or_path: str | Path | dict | None = None) -> dict:
    if isinstance(name_or_path, dict):
        return name_or_path
    key = DEFAULT_PROFILE if name_or_path is None else str(name_or_path)
    key = PROFILE_ALIASES.get(key, key)
    path = Path(key)
    if not path.is_file():
        path = PROFILE_DIR / f"{key}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"WaterInt profile {name_or_path!r} not found (looked for {path}).")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data.setdefault("profile_id", path.stem)
    data["_path"] = str(path)
    return data


@dataclass
class ResolvedProfile:
    profile_id: str
    species: list[str]
    role: np.ndarray            # int8: 1 proton, 2 acceptor, 0 fixed charge
    qf: np.ndarray              # float64 formal charges
    pbc: np.ndarray             # uint8[3]
    cutoff: float
    tie_tol: float
    ref_mask: np.ndarray        # uint8
    ref_mode: int
    ref_width: float
    ref_fixed: float
    group_names: list[str]
    nh_max: int
    table: np.ndarray           # int32 [n_rows, nh_max + 1]: group of an acceptor by (row, nH)
    row_names: list[str]        # "<element>" / "<element>:solid"
    row_labels: list[list[str]] # species label by (row, nH)
    other_group: np.ndarray     # int32 [nat], -1 for non-fixed atoms
    free_h_group: int
    transfer_group: int
    image_group: int
    window_names: list[str]
    edges: np.ndarray           # float64 [2 * W]
    solid_elements: list[str]
    solid_max_zrel: float | None
    solid_expected: int | None

    def acceptor_rows(self, solid: np.ndarray) -> np.ndarray:
        """Species-table row of every atom (-1 for non-acceptors), using the frozen solid mask."""
        rows = np.full(len(self.species), -1, dtype=np.int32)
        index = {name: k for k, name in enumerate(self.row_names)}
        for i, (el, r) in enumerate(zip(self.species, self.role)):
            if r == 2:
                key = f"{el}:solid" if solid[i] and f"{el}:solid" in index else el
                rows[i] = index[key]
        return rows

    def label_of(self, row: int, nh: int) -> str:
        labels = self.row_labels[row]
        return labels[min(nh, len(labels) - 1)]


def _label_table(table: dict) -> tuple[list[str], int]:
    keys = sorted(int(k) for k in table)
    if keys != list(range(len(keys))):
        raise ValueError("species tables must have consecutive nH keys 0..n (the last one means 'n or more').")
    return [str(table[k] if k in table else table[str(k)]) for k in keys], keys[-1]


def resolve(profile: dict, species: Sequence[str]) -> ResolvedProfile:
    charges = {str(k): float(v) for k, v in profile["formal_charges"].items()}
    protons, acceptors = set(profile.get("protons", ["H"])), set(profile["acceptors"])
    species = [str(s) for s in species]
    missing = sorted(set(species) - set(charges))
    if missing:
        raise ValueError(f"Profile {profile['profile_id']} has no formal charge for {missing}.")
    role = np.array([1 if s in protons else (2 if s in acceptors else 0) for s in species], dtype=np.int8)
    qf = np.array([charges[s] for s in species], dtype=np.float64)
    for el, n in (profile.get("expected_counts") or {}).items():
        if species.count(el) != int(n):
            raise ValueError(f"Profile expects {n} {el}, trajectory has {species.count(el)}.")
    h = profile.get("h_assignment", {})
    ref = profile.get("reference") or {"mode": "none"}
    mode = REF_MODES[str(ref.get("mode", "none"))]
    ref_mask = np.array([s in set(ref.get("elements", [])) for s in species], dtype=np.uint8)
    # species tables: either {acceptor: {...}, solid: {...}} for every acceptor element, or per element
    # {O: {free: {...}, solid: {...}}, N: {free: {...}}}; the largest nH key means "this many or more"
    spec = profile["species"]
    per_element = {}
    if "acceptor" in spec:
        for el in sorted(acceptors):
            per_element[el] = {"free": spec["acceptor"], "solid": spec.get("solid")}
    else:
        for el in sorted(acceptors):
            if el not in spec:
                raise ValueError(f"profile species table has no entry for acceptor element {el}.")
            per_element[el] = {"free": spec[el]["free"], "solid": spec[el].get("solid")}
    row_names, row_labels = [], []
    for el, t in per_element.items():
        row_names.append(el); row_labels.append(_label_table(t["free"])[0])
        if t.get("solid"):
            row_names.append(f"{el}:solid"); row_labels.append(_label_table(t["solid"])[0])
    nh_max = max(len(l) for l in row_labels) - 1
    acc_labels = sorted({l for labels in row_labels for l in labels})
    sol_labels: list[str] = []
    groups = profile.get("current_groups") or {}
    group_names = list(groups)
    owner: dict[str, int] = {}
    for gi, (name, members) in enumerate(groups.items()):
        for m in members:
            if m in owner:
                raise ValueError(f"current_groups: {m!r} appears in two groups.")
            owner[str(m)] = gi
    fixed_elements = sorted({s for s, r in zip(species, role) if r == 0})
    needed = set(acc_labels) | set(sol_labels) | set(SPECIAL_TERMS) | set(fixed_elements)
    if groups:
        lost = sorted(needed - set(owner))
        if lost:
            raise ValueError(f"current_groups must place every label/element/term; missing {lost}.")
    get = lambda label: owner.get(label, 0)
    table = np.array([[get(labels[min(n, len(labels) - 1)]) for n in range(nh_max + 1)] for labels in row_labels], dtype=np.int32)
    other_group = np.array([get(s) if r == 0 else -1 for s, r in zip(species, role)], dtype=np.int32)
    windows = profile.get("windows") or []
    names = [str(w[0]) for w in windows]
    edges = np.array([[float(w[1]), float(w[2])] for w in windows], dtype=np.float64).reshape(-1)
    for k in range(len(windows)):
        if not edges[2 * k] < edges[2 * k + 1]:
            raise ValueError(f"window {names[k]!r} must have lo < hi.")
    solid = profile.get("solid") or {}
    return ResolvedProfile(
        profile_id=str(profile["profile_id"]), species=species, role=role, qf=qf,
        pbc=np.array([bool(x) for x in profile.get("pbc", [True, True, False])], dtype=np.uint8),
        cutoff=float(h.get("cutoff_A", 1.5)), tie_tol=float(h.get("tie_tolerance_A", 1e-8)),
        ref_mask=ref_mask, ref_mode=mode, ref_width=float(ref.get("layer_width_A", 0.0)), ref_fixed=float(ref.get("value_A", 0.0)),
        group_names=group_names or ["total"], nh_max=nh_max, table=table, row_names=row_names, row_labels=row_labels,
        other_group=other_group,
        free_h_group=get("free_H"), transfer_group=get("transfer"), image_group=get("image"),
        window_names=names, edges=edges,
        solid_elements=list(solid.get("elements", [])), solid_max_zrel=solid.get("max_z_rel_A"),
        solid_expected=solid.get("expected_count"),
    )


def reference_height(rp: ResolvedProfile, positions: np.ndarray) -> float:
    """Same rule as the C++ kernel (used once, for freezing the solid at the first frame)."""
    if rp.ref_mode == 0:
        return 0.0
    if rp.ref_mode == 3:
        return rp.ref_fixed
    z = positions[rp.ref_mask.astype(bool), 2]
    if rp.ref_mode == 4:
        return float(z.mean())
    if rp.ref_mode == 1:
        return float(z[z >= z.max() - rp.ref_width].mean())
    return float(z[z <= z.min() + rp.ref_width].mean())


def solid_mask(rp: ResolvedProfile, first_positions: np.ndarray, ids: np.ndarray | None = None,
               solid_ids: Sequence[int] | None = None) -> np.ndarray:
    """Solid acceptors frozen at the first frame (or from an explicit ID list)."""
    nat = len(rp.species)
    mask = np.zeros(nat, dtype=np.uint8)
    if solid_ids is not None:
        idset = set(int(i) for i in solid_ids)
        ids = np.arange(1, nat + 1) if ids is None else ids
        mask[[k for k in range(nat) if int(ids[k]) in idset]] = 1
    elif rp.solid_elements and rp.solid_max_zrel is not None:
        zref = reference_height(rp, first_positions)
        sel = np.array([s in set(rp.solid_elements) for s in rp.species]) & (first_positions[:, 2] <= zref + float(rp.solid_max_zrel))
        mask[sel] = 1
    if rp.solid_expected is not None and int(mask.sum()) != int(rp.solid_expected):
        raise ValueError(f"solid selection found {int(mask.sum())} atoms, profile expects {rp.solid_expected}.")
    return mask


# ---------------------------------------------------------------------------------------------
# v0.7: SystemProfile = the system description used by prep/ (ownership, species, reference, solid).
# ---------------------------------------------------------------------------------------------
AXES = {"x": 0, "y": 1, "z": 2}
DEFAULT_FREE_SPECIES = ("O2-", "OH-", "H2O", "H3O+", "O_other")  # nH = 0, 1, 2, 3, >=4


def _pbc3(value: Any, default: tuple[bool, bool, bool] | None) -> tuple[bool, bool, bool] | None:
    if value is None:
        return default
    if value is False or value == "none":
        return None
    if value is True:
        return (True, True, True)
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError("pbc must be a boolean, 'none', or a list of three booleans.")
    flags = tuple(bool(v) for v in value)
    return flags if any(flags) else None  # type: ignore[return-value]


@dataclass(frozen=True)
class OwnershipParams:
    """How protons are assigned to acceptors. One definition used by every analysis.

    mode: ``nearest`` (each proton belongs to its nearest acceptor within cutoff; exact ties -> lowest index)
    or ``all_within_cutoff`` (a proton counts for every acceptor within cutoff; the v0.6 orientation/H-bond rule).
    pbc: periodic axes used for O-H distances; ``None`` = no periodic images.
    """
    protons: tuple[str, ...] = ("H",)
    acceptors: tuple[str, ...] = ("O",)
    cutoff: float = 1.25                     # A; math.inf = no cutoff (every proton takes its nearest acceptor)
    mode: str = "nearest"
    pbc: tuple[bool, bool, bool] | None = (True, True, False)
    tie_tolerance: float = 1.0e-8            # A; nearest mode: acceptors within d_min + tol tie -> lowest atom index

    def __post_init__(self) -> None:
        if self.tie_tolerance < 0:
            raise ValueError("ownership tie_tolerance must be >= 0.")
        if self.cutoff <= 0:
            raise ValueError("ownership cutoff must be positive.")
        if self.mode not in {"nearest", "all_within_cutoff"}:
            raise ValueError("ownership mode must be nearest or all_within_cutoff.")

    def replace(self, overrides: dict[str, Any] | None) -> "OwnershipParams":
        if not overrides:
            return self
        from dataclasses import replace as dc_replace
        kw: dict[str, Any] = {}
        for key, value in overrides.items():
            if key in {"cutoff", "oh_cutoff", "cutoff_A"}:
                kw["cutoff"] = float("inf") if value in (None, "none", "inf") else float(value)
            elif key in {"tie_tolerance", "tie_tolerance_A"}:
                kw["tie_tolerance"] = float(value)
            elif key in {"mode", "hydrogen_assignment"}:
                kw["mode"] = str(value)
            elif key == "pbc":
                kw["pbc"] = _pbc3(value, None)
            elif key in {"protons", "hydrogen_symbol"}:
                kw["protons"] = tuple([value] if isinstance(value, str) else value)
            elif key in {"acceptors", "oxygen_symbol"}:
                kw["acceptors"] = tuple([value] if isinstance(value, str) else value)
            else:
                raise ValueError(f"Unknown ownership parameter {key!r}.")
        return dc_replace(self, **kw)


@dataclass(frozen=True)
class SystemProfile:
    profile_id: str
    type_map: dict[int, str]
    ownership: OwnershipParams
    pbc: tuple[bool, bool, bool]
    normal_axis: int
    reference: dict[str, Any]           # mode none|top_layer|bottom_layer|mean|fixed, elements, layer_width_A, value_A
    solid: dict[str, Any]               # mode none|frozen|dynamic (+ elements, max_z_rel_A, expected_count)
    free_species: tuple[str, ...]       # acceptor label by nH (last = "this many or more")
    solid_species: tuple[str, ...] | None
    windows: tuple[tuple[str, float, float], ...]
    raw: dict[str, Any]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SystemProfile":
        pbc = _pbc3(data.get("pbc"), (True, True, False)) or (False, False, False)
        h = data.get("h_assignment") or {}
        ownership = OwnershipParams(
            protons=tuple(data.get("protons", ["H"])),
            acceptors=tuple(data.get("acceptors", ["O"])),
            cutoff=float(h.get("cutoff_A", 1.25)),
            mode=str(h.get("mode", "nearest")),
            tie_tolerance=float(h.get("tie_tolerance_A", 1.0e-8)),
            pbc=pbc if any(pbc) else None,
        )
        spec = data.get("species") or {}
        free = tuple(_label_table(spec["acceptor"])[0]) if "acceptor" in spec else DEFAULT_FREE_SPECIES
        solid_tab = tuple(_label_table(spec["solid"])[0]) if spec.get("solid") else None
        solid = dict(data.get("solid") or {})
        solid.setdefault("mode", "frozen" if solid.get("elements") else "none")
        windows = tuple((str(w[0]), float(w[1]), float(w[2])) for w in (data.get("windows") or []))
        axis = data.get("normal_axis", "z")
        return cls(
            profile_id=str(data.get("profile_id", "inline")),
            type_map={int(k): str(v) for k, v in (data.get("lammps_type_map") or data.get("type_map") or {}).items()},
            ownership=ownership, pbc=pbc,
            normal_axis=AXES[str(axis).lower()] if not isinstance(axis, int) else int(axis),
            reference=dict(data.get("reference") or {"mode": "none"}),
            solid=solid, free_species=free, solid_species=solid_tab, windows=windows, raw=dict(data),
        )

    def species_label(self, nh: int, *, solid: bool = False) -> str:
        table = self.solid_species if (solid and self.solid_species) else self.free_species
        return table[min(int(nh), len(table) - 1)]

    @property
    def species_labels(self) -> list[str]:
        out = list(self.free_species)
        for label in self.solid_species or ():
            if label not in out:
                out.append(label)
        return out


GENERIC_PROFILE = {
    "profile_id": "generic_water",
    "description": "Water only: O accepts H (nearest O within 1.25 A), x/y periodic, no solid, absolute coordinates.",
    "protons": ["H"], "acceptors": ["O"],
    "h_assignment": {"cutoff_A": 1.25, "mode": "nearest"},
    "pbc": [True, True, False],
    "reference": {"mode": "none"},
    "species": {"acceptor": {0: "O2-", 1: "OH-", 2: "H2O", 3: "H3O+", 4: "O_other"}},
}


def load_system_profile(spec: str | Path | dict | None = None) -> SystemProfile:
    """Profile by name (bundled), path, inline mapping, or None (generic water profile)."""
    if spec is None:
        return SystemProfile.from_dict(GENERIC_PROFILE)
    if isinstance(spec, dict) and "extends" in spec:
        base = load_profile(spec["extends"])
        merged = {**base, **{k: v for k, v in spec.items() if k != "extends"}}
        return SystemProfile.from_dict(merged)
    return SystemProfile.from_dict(load_profile(spec))
