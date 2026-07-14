from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
V03 = ROOT / "v0.3"


class V03CoordinateReferenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self._old_path = list(sys.path)
        self._old_modules = {
            name: module
            for name, module in sys.modules.items()
            if name == "waterint" or name.startswith("waterint.")
        }
        for name in list(self._old_modules):
            sys.modules.pop(name, None)
        sys.path.insert(0, str(V03))

    def tearDown(self) -> None:
        for name in [name for name in sys.modules if name == "waterint" or name.startswith("waterint.")]:
            sys.modules.pop(name, None)
        sys.modules.update(self._old_modules)
        sys.path[:] = self._old_path

    def test_top_layer_mean_uses_only_atoms_near_selected_surface(self):
        common = importlib.import_module("waterint._00_io.common")
        coordinates = importlib.import_module("waterint._01_core.coordinates")
        selection = importlib.import_module("waterint._01_core.selection")
        frame = common.TrajectoryFrame(
            index=0,
            comment="Mg slab",
            symbols=["Mg", "Mg", "Mg"],
            positions=np.asarray([[0.0, 0.0, 10.0], [0.0, 0.0, 9.6], [0.0, 0.0, 5.0]]),
            cell=(10.0, 10.0, 12.0),
            types=np.asarray([2, 2, 2], dtype=int),
        )
        context = selection.SelectionContext.from_input_config({"type_map": {2: "Mg"}})
        spec = coordinates.coordinate_spec_from_config(
            {
                "mode": "relative_to_slab",
                "axis": "z",
                "reference": {
                    "type": "top_layer_mean",
                    "species": ["Mg"],
                    "surface": "max",
                    "layer_width": 0.7,
                },
            }
        )

        self.assertAlmostEqual(coordinates.reference_for_frame(frame, spec, context), 9.8)


if __name__ == "__main__":
    unittest.main()
