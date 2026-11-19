"""LabelledFrame: one frame plus lazily computed, cached prep products.

Every analysis asks the LabelledFrame instead of recomputing: element indices, ownership, species,
H-bond counts, reference plane and coordinates. Products are memoised per frame by their parameters,
so N analyses that share a definition trigger one computation; different definitions coexist.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from waterint.io.cell import orthorhombic_cell_vectors
from waterint.io.frame import TrajectoryFrame
from waterint.prep.hbonds import HBondParams, hbond_counts
from waterint.prep.ownership import Ownership, compute_ownership
from waterint.prep.reference import CoordinateSpec, coordinate_spec_from_profile, reference_value
from waterint.prep.species import (SOLID_TOKEN, dynamic_solid_mask, element_mask, frame_symbols, frozen_solid_mask,
                                   species_by_label)
from waterint.system.profile import OwnershipParams, SystemProfile


@dataclass
class PrepContext:
    """Run-level prep settings and state shared by all frames."""
    profile: SystemProfile
    type_map: dict[int, str] = field(default_factory=dict)
    backend: str = "auto"                                  # prep kernels (ownership, H bonds)
    configured_cell: tuple[float, float, float] | None = None
    frozen_solid: np.ndarray | None = None                  # filled at the first frame (solid.mode: frozen)
    timings: dict[str, float] = field(default_factory=dict)
    backends_used: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.type_map:
            self.type_map = dict(self.profile.type_map)

    @property
    def default_coordinate(self) -> CoordinateSpec:
        return coordinate_spec_from_profile(self.profile)


class LabelledFrame:
    def __init__(self, frame: TrajectoryFrame, ctx: PrepContext):
        self.frame = frame
        self.ctx = ctx
        self._cache: dict[Any, Any] = {}

    # ---------------------------------------------------------------- frame data
    @property
    def positions(self) -> np.ndarray:
        return self.frame.positions

    @property
    def index(self) -> int:
        return self.frame.index

    @property
    def cell(self) -> tuple[float, float, float] | None:
        return self.frame.cell if self.frame.cell is not None else self.ctx.configured_cell

    @property
    def cell_vectors(self) -> np.ndarray:
        if self.frame.cell_vectors is not None:
            return np.asarray(self.frame.cell_vectors, dtype=float)
        if self.cell is None:
            raise ValueError("No cell information: the trajectory has none and system.cell is not set.")
        return orthorhombic_cell_vectors(self.cell)

    def _memo(self, key: Any, fn):
        if key not in self._cache:
            self._cache[key] = fn()
        return self._cache[key]

    # ---------------------------------------------------------------- selection
    @property
    def symbols(self) -> np.ndarray:
        return self._memo("symbols", lambda: frame_symbols(self.frame, self.ctx.type_map))

    def mask(self, elements: Sequence[str]) -> np.ndarray:
        elements = tuple(str(e) for e in elements)

        def build() -> np.ndarray:
            plain = tuple(e for e in elements if e != SOLID_TOKEN)
            m = element_mask(self.frame, plain, self.ctx.type_map) if plain else np.zeros(len(self.positions), bool)
            if SOLID_TOKEN in elements:
                m = m | self.solid_mask()
            return m
        return self._memo(("mask", elements), build)

    def indices(self, elements: Sequence[str]) -> np.ndarray:
        return self._memo(("idx", tuple(elements)), lambda: np.nonzero(self.mask(elements))[0])

    # ---------------------------------------------------------------- ownership / species / solid
    def ownership(self, params: OwnershipParams | None = None) -> Ownership:
        params = params or self.ctx.profile.ownership

        def build() -> Ownership:
            own = compute_ownership(self.positions, self.indices(params.acceptors), self.indices(params.protons), params,
                                    cell=self.cell, cell_vectors=self.frame.cell_vectors, triclinic=self.frame.triclinic,
                                    backend=self.ctx.backend)
            self.ctx.backends_used[f"ownership.{params.mode}"] = own.backend
            return own
        return self._memo(("own", params), build)

    def solid_mask(self) -> np.ndarray:
        prof = self.ctx.profile
        mode = str(prof.solid.get("mode", "none"))
        if mode == "none":
            return np.zeros(len(self.positions), dtype=bool)
        if mode == "dynamic":
            return self._memo("solid", lambda: dynamic_solid_mask(self.symbols, self.ownership(), prof))
        if mode == "frozen":
            if self.ctx.frozen_solid is None:
                ref = prof.reference
                elements = [e for e in ref.get("elements", []) if e != SOLID_TOKEN]
                spec = self.ctx.default_coordinate
                if any(e == SOLID_TOKEN for e in ref.get("elements", [])) or not elements:
                    raise ValueError("solid.mode frozen needs a reference plane from explicit elements.")
                z_ref = reference_value(self.positions, spec, lambda els: self.mask(els))
                self.ctx.frozen_solid = frozen_solid_mask(self.positions, self.symbols, prof, z_ref)
            return self.ctx.frozen_solid
        raise ValueError(f"Unknown solid.mode {mode!r} (none | frozen | dynamic).")

    def species(self, params: OwnershipParams | None = None, *, use_solid: bool = True) -> dict[str, np.ndarray]:
        """Acceptor indices by species label (profile tables; solid acceptors get the solid table)."""
        params = params or self.ctx.profile.ownership
        prof = self.ctx.profile

        def build() -> dict[str, np.ndarray]:
            solid = self.solid_mask() if (use_solid and prof.solid_species) else None
            return species_by_label(self.ownership(params), prof.free_species, prof.solid_species, solid)
        return self._memo(("species", params, use_solid), build)

    # ---------------------------------------------------------------- hydrogen bonds
    def hbond_counts(self, hb: HBondParams, params: OwnershipParams | None = None) -> tuple[np.ndarray, np.ndarray]:
        """(donated, accepted) bond counts per acceptor of ``ownership(params)``."""
        params = params or self.ctx.profile.ownership

        def build():
            if self.cell is None:
                raise ValueError("H-bond analysis requires cell information.")
            donors, acceptors, used = hbond_counts(self.positions, self.ownership(params), hb, cell=self.cell,
                                                   backend=self.ctx.backend)
            self.ctx.backends_used["hbonds.counts"] = used
            return donors, acceptors
        return self._memo(("hb", params, hb), build)

    # ---------------------------------------------------------------- coordinates
    def reference(self, spec: CoordinateSpec | None = None) -> float:
        spec = spec or self.ctx.default_coordinate
        return self._memo(("ref", spec), lambda: reference_value(self.positions, spec, lambda els: self.mask(els)))

    def coordinates(self, indices: np.ndarray, spec: CoordinateSpec | None = None) -> np.ndarray:
        spec = spec or self.ctx.default_coordinate
        return spec.sign * (self.positions[indices, spec.axis] - self.reference(spec))
