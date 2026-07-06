from __future__ import annotations

import unittest

import numpy as np

from waterint.core.coordinates import coordinate_spec_from_config, coordinate_values, reference_for_frame
from waterint.core.selection import SelectionContext
from waterint.io.common import TrajectoryFrame


def _frame() -> TrajectoryFrame:
    return TrajectoryFrame(
        index=0,
        comment="fixed reference",
        symbols=["O", "O"],
        positions=np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 3.5]]),
        cell=(10.0, 10.0, 10.0),
    )


def _context() -> SelectionContext:
    return SelectionContext(symbol_to_types={})


class CoordinateReferenceTests(unittest.TestCase):
    def test_fixed_slab_reference_does_not_require_slab_atoms(self):
        spec = coordinate_spec_from_config(
            {
                "mode": "relative_to_slab",
                "axis": "z",
                "reference": {"type": "fixed", "value": 2.0},
            }
        )

        self.assertEqual(reference_for_frame(_frame(), spec, _context()), 2.0)
        np.testing.assert_allclose(coordinate_values(_frame(), np.asarray([0, 1]), spec, _context()), [-1.0, 1.5])

    def test_fixed_reference_alias_respects_negative_axis(self):
        spec = coordinate_spec_from_config(
            {
                "mode": "relative_to_slab",
                "axis": "-z",
                "reference": {"type": "fixed_value", "value": 2.0},
            }
        )

        np.testing.assert_allclose(coordinate_values(_frame(), np.asarray([0, 1]), spec, _context()), [1.0, -1.5])

    def test_fixed_reference_works_for_relative_to_reference(self):
        spec = coordinate_spec_from_config(
            {
                "mode": "relative_to_reference",
                "axis": "z",
                "reference": {"type": "fixed", "value": 1.5},
            }
        )

        self.assertEqual(reference_for_frame(_frame(), spec, _context()), 1.5)


if __name__ == "__main__":
    unittest.main()
