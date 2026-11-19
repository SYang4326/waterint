"""Registry of analyses: the single source for the CLI, the UI and the docs generator."""
from __future__ import annotations

from typing import Any

from waterint.analysis.base import Analysis
from waterint.analysis.dynamics.carrier import CarrierMsd, CarrierTrack
from waterint.analysis.dynamics.hb_correlation import HbCorrelation
from waterint.analysis.dynamics.identity_transition import IdentityTransition
from waterint.analysis.dynamics.msd import Msd
from waterint.analysis.free_energy.proton_sharing import ProtonSharing, ProtonSharingHbond
from waterint.analysis.relabel import Relabel
from waterint.analysis.spectra.sfg import Sfg
from waterint.analysis.spectra.sfg_species import SfgSpecies, SfgSpeciesAggregate
from waterint.analysis.structure.density import Density
from waterint.analysis.structure.hbond_stats import HBondStats
from waterint.analysis.structure.orientation import Orientation
from waterint.analysis.structure.rdf import Rdf
from waterint.analysis.transport.defect import Conductivity, DefectConductivity, DefectMsd
from waterint.analysis.transport.site_conductance import SiteConductance
from waterint.analysis.transport.site_current import SiteCurrent

ANALYSES: dict[str, type[Analysis]] = {cls.type: cls for cls in (
    Density, Orientation, HBondStats, Rdf, Msd,
    DefectMsd, DefectConductivity, Conductivity, IdentityTransition,
    ProtonSharing, ProtonSharingHbond, Sfg, SfgSpecies, SfgSpeciesAggregate,
    SiteCurrent, SiteConductance, CarrierTrack, CarrierMsd, HbCorrelation, Relabel)}
ALIASES = {"oh-orientation": "orientation", "angle-z": "orientation", "angle_z": "orientation", "oh_orientation": "orientation",
           "h-bond": "hbond", "hbond-stats": "hbond"}
GROUPS = {"structure": ["density", "orientation", "hbond", "rdf"],
          "dynamics": ["msd", "defect-msd", "identity-transition", "carrier-track", "carrier-msd", "hb-correlation"],
          "transport": ["conductivity", "defect-conductivity", "site-current", "site-conductance"],
          "spectra": ["sfg", "sfg-species", "sfg-species-aggregate"],
          "free energy": ["proton-sharing", "proton-sharing-hbond"],
          "trajectory tools": ["relabel"]}


def resolve_type(name: str) -> str:
    key = str(name).strip().lower()
    key = ALIASES.get(key, key)
    if key not in ANALYSES:
        raise KeyError(f"Unknown analysis {name!r}. Available: {sorted(ANALYSES)}")
    return key


def get(name: str) -> type[Analysis]:
    return ANALYSES[resolve_type(name)]


def describe() -> list[dict[str, Any]]:
    from waterint.backends.registry import describe as kernels
    k = kernels()
    group_of = {n: g for g, names in GROUPS.items() for n in names}
    out = []
    for name, cls in sorted(ANALYSES.items()):
        prefix = name.replace("-", "_")
        out.append({"name": name, "group": group_of.get(name, "other"), "kind": cls.kind, "description": cls.description,
                    "aliases": sorted(a for a, t in ALIASES.items() if t == name),
                    "params": {p: {"default": d, "help": h} for p, (d, h) in cls.params.items()},
                    "kernels": {kn: b for kn, b in k.items() if kn.split(".")[0] in {name, prefix, "ownership", "hbonds"}}})
    return out
