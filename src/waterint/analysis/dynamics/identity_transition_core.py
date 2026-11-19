"""Oxygen identity-transition counting independent of trajectory IO."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


IDENTITY_STATES = ("O_bone", "O2-", "OH", "H2O", "H3O+", "O_other")
REQUIRED_STATES = ("O_bone", "OH", "H2O", "H3O+")

# Codes written by the standard relabel (type_map.tsv): lattice O 3-7; non-lattice
# O by region L1/L2/bulk/other_region, each O2-/OH/H2O/H3O+/O_other.
STANDARD_TYPE_CODES = {
    "O_bone": [3, 4, 5, 6, 7],
    "O2-": [8, 13, 18, 23],
    "OH": [9, 14, 19, 24],
    "H2O": [10, 15, 20, 25],
    "H3O+": [11, 16, 21, 26],
    "O_other": [12, 17, 22, 27],
}


@dataclass(frozen=True)
class IdentityTransitionResult:
    """Non-self transitions between temporally persistent oxygen identities."""

    states: tuple[str, ...]
    counts: np.ndarray
    frames: int
    transitions: int


class IdentityTransitionAccumulator:
    """Streaming per-oxygen state-segment accumulator.

    A transition is counted only after both adjacent contiguous states meet the
    configured residence threshold. This removes one- or few-frame nearest-O
    assignment rattling without modifying any instantaneous identity label.
    """

    def __init__(self, *, n_oxygen: int, minimum_residence_frames: int, states: tuple[str, ...] = IDENTITY_STATES) -> None:
        if n_oxygen <= 0:
            raise ValueError("n_oxygen must be positive.")
        if minimum_residence_frames <= 0:
            raise ValueError("minimum_residence_frames must be positive.")
        self.minimum_residence_frames = int(minimum_residence_frames)
        self.states = tuple(states)
        self.counts = np.zeros((len(self.states), len(self.states)), dtype=np.int64)
        self.frames = 0
        self._current_state = np.full(n_oxygen, -1, dtype=np.int8)
        self._current_length = np.zeros(n_oxygen, dtype=np.int64)
        self._previous_state = np.full(n_oxygen, -1, dtype=np.int8)
        self._previous_length = np.zeros(n_oxygen, dtype=np.int64)

    def add(self, states: np.ndarray) -> None:
        states = np.asarray(states, dtype=np.int8)
        if states.shape != self._current_state.shape:
            raise ValueError("Every frame must contain the same oxygen identities.")
        if self.frames == 0:
            self._current_state[:] = states
            self._current_length[:] = 1
            self.frames = 1
            return
        changed = states != self._current_state
        self._current_length[~changed] += 1
        for index in np.flatnonzero(changed):
            self._complete_segment(int(index), int(self._current_state[index]), int(self._current_length[index]))
            self._current_state[index] = states[index]
            self._current_length[index] = 1
        self.frames += 1

    def finish(self) -> IdentityTransitionResult:
        for index in range(self._current_state.size):
            self._complete_segment(int(index), int(self._current_state[index]), int(self._current_length[index]))
        return IdentityTransitionResult(
            states=self.states,
            counts=self.counts.copy(),
            frames=self.frames,
            transitions=int(self.counts.sum()),
        )

    def _complete_segment(self, index: int, state: int, length: int) -> None:
        previous_state = int(self._previous_state[index])
        previous_length = int(self._previous_length[index])
        if (
            previous_state >= 0
            and previous_state != state
            and previous_length >= self.minimum_residence_frames
            and length >= self.minimum_residence_frames
        ):
            self.counts[previous_state, state] += 1
        self._previous_state[index] = state
        self._previous_length[index] = length


def identity_states_for(type_codes: dict[str, int | list[int]]) -> tuple[str, ...]:
    """States in canonical order: the four required ones plus any optional ones given."""
    unknown = set(type_codes) - set(IDENTITY_STATES)
    missing = set(REQUIRED_STATES) - set(type_codes)
    if unknown or missing:
        raise ValueError(f"identity_transition.type_codes: unknown {sorted(unknown)}, missing {sorted(missing)}; allowed {list(IDENTITY_STATES)}.")
    return tuple(label for label in IDENTITY_STATES if label in type_codes)


def type_codes_to_identity_states(types: np.ndarray, *, type_codes: dict[str, int | list[int]]) -> np.ndarray:
    """Map oxygen LAMMPS types to compact identity-state indices (order of identity_states_for)."""
    states = identity_states_for(type_codes)
    types = np.asarray(types, dtype=np.int64)
    result = np.full(types.shape, -1, dtype=np.int8)
    for index, label in enumerate(states):
        raw_codes = type_codes[label]
        codes = [int(raw_codes)] if isinstance(raw_codes, int) else [int(value) for value in raw_codes]
        result[np.isin(types, codes)] = index
    if np.any(result < 0):
        unknown = sorted(set(int(value) for value in types[result < 0]))
        raise ValueError(f"Oxygen types not covered by identity_transition.type_codes: {unknown}")
    return result
