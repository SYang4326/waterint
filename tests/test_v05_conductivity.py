from __future__ import annotations

import importlib
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
V05 = ROOT / "v0.5"


class V05ConductivityTests(unittest.TestCase):
    def setUp(self) -> None:
        self._old_path = list(sys.path)
        self._old_modules = {name: module for name, module in sys.modules.items() if name == "waterint" or name.startswith("waterint.")}
        for name in list(self._old_modules):
            sys.modules.pop(name, None)
        sys.path.insert(0, str(V05))

    def tearDown(self) -> None:
        for name in [name for name in sys.modules if name == "waterint" or name.startswith("waterint.")]:
            sys.modules.pop(name, None)
        sys.modules.update(self._old_modules)
        sys.path[:] = self._old_path

    def test_nernst_einstein_conversion_uses_msd_slope_and_absolute_charge(self) -> None:
        conductivity = importlib.import_module("waterint._02_computation.conductivity")
        result = conductivity.compute_nernst_einstein_conductivity(
            np.asarray([0.0, 1.0, 2.0, 3.0]),
            np.asarray([0.0, 1.0, 4.0, 9.0]),
            carrier_count=2,
            volume_a3=1000.0,
            temperature_k=300.0,
            charge_e=-1.0,
            dimensions=2,
            fit_range_ps=(1.0, 3.0),
            sheet_thickness_a=5.0,
        )
        self.assertAlmostEqual(result.slope_a2_per_ps, 4.0)
        self.assertAlmostEqual(result.diffusion_a2_per_ps, 1.0)
        expected = 2 / (1000.0e-30) * conductivity.ELEMENTARY_CHARGE_C**2 * 1.0e-8 / (conductivity.BOLTZMANN_J_PER_K * 300.0)
        self.assertAlmostEqual(result.conductivity_s_per_m, expected)
        self.assertAlmostEqual(result.sheet_conductance_s, expected * 5.0e-10)

    def test_workflow_writes_summary_and_msd_fit_outputs(self) -> None:
        workflow = importlib.import_module("waterint._04_workflows.workflows.conductivity")
        positions = np.asarray([[[0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]], [[2.0, 0.0, 0.0]], [[3.0, 0.0, 0.0]]])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.savez(root / "trajectory.npz", positions=positions, types=np.asarray([[3]] * 4), cells=np.asarray([[10.0, 10.0, 10.0]] * 4), steps=np.arange(4))
            config = {
                "_config_dir": str(root),
                "input": {"trajectory": "trajectory.npz", "format": "npz", "type_map": {3: "O"}},
                "system": {"cell": "auto"},
                "selection": {"elements": ["O"]},
                "msd": {"timestep_ps": 1.0, "max_lag_frames": 3, "pbc": [False, False, False], "dimensionality": "2d"},
                "conductivity": {"temperature_K": 300.0, "charge_e": -1.0, "fit_range_ps": [1.0, 3.0], "volume": {"mode": "slab", "normal_axis": "z", "thickness_A": 5.0}},
                "output": {"directory": "output", "prefix": "conductivity", "plot": False},
            }
            result = workflow.run_conductivity(config)
            self.assertTrue(result.csv_path.exists())
            self.assertTrue(result.msd_csv_path.exists())
            self.assertTrue(result.metadata_path.exists())
            self.assertAlmostEqual(result.diffusion_a2_per_ps, 1.0)
            self.assertAlmostEqual(result.volume_a3, 500.0)
            self.assertIn("conductivity_S_per_m", result.csv_path.read_text(encoding="utf-8"))

    def test_conductivity_is_registered(self) -> None:
        registry = importlib.import_module("waterint._04_workflows.registry.registry")
        self.assertEqual(registry.get_analysis_module("conductivity").name, "conductivity")
        self.assertEqual(registry.get_analysis_module("conductivity-ne").name, "conductivity")

    def test_fixed_conductivity_rejects_dynamic_oxygen_species(self) -> None:
        workflow = importlib.import_module("waterint._04_workflows.workflows.conductivity")
        config = {
            "input": {},
            "system": {},
            "selection": {"oxygen_species": ["OH-"]},
            "msd": {},
            "conductivity": {},
            "output": {},
        }
        with self.assertRaisesRegex(ValueError, "defect-conductivity"):
            workflow.run_conductivity(config)


if __name__ == "__main__":
    unittest.main()
