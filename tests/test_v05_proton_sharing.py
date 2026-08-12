from __future__ import annotations

import importlib
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
V05 = ROOT / "v0.5"


class V05ProtonSharingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.old_path = list(sys.path)
        self.old_modules = {name: module for name, module in sys.modules.items() if name == "waterint" or name.startswith("waterint.")}
        for name in list(self.old_modules):
            sys.modules.pop(name, None)
        sys.path.insert(0, str(V05))

    def tearDown(self) -> None:
        for name in [name for name in sys.modules if name == "waterint" or name.startswith("waterint.")]:
            sys.modules.pop(name, None)
        sys.modules.update(self.old_modules)
        sys.path[:] = self.old_path

    def test_delta_zero_and_shared_coordinate_are_accumulated(self) -> None:
        module = importlib.import_module("waterint._02_computation.proton_sharing")
        result = module.proton_sharing_surface(
            donor_positions=np.asarray([[0.0, 0.0, 0.0]]),
            acceptor_positions=np.asarray([[2.4, 0.0, 0.0]]),
            donor_hydrogens=[np.asarray([[1.2, 0.0, 0.0]])],
            cell_vectors=np.diag([10.0, 10.0, 10.0]),
            pbc=(True, True, False),
            oo_range_a=(2.0, 3.0), delta_range_a=(-1.0, 1.0), delta_bins=20, oo_bins=10,
            shared_delta_max_a=0.2, shared_s_range_a=(-1.0, 1.0), shared_rho_range_a=(0.0, 1.0),
            shared_s_bins=20, shared_rho_bins=10, temperature_k=300.0,
        )
        self.assertEqual(result.pair_samples, 1)
        self.assertEqual(result.shared_samples, 1)
        delta_index, oo_index = np.unravel_index(np.argmax(result.counts), result.counts.shape)
        self.assertAlmostEqual(result.delta_centers_a[delta_index], 0.05, places=12)
        self.assertAlmostEqual(result.oo_centers_a[oo_index], 2.45, places=12)
        self.assertTrue(np.isfinite(result.free_energy_kj_mol[delta_index, oo_index]))
        self.assertEqual(int(result.shared_counts.sum()), 1)

    def test_workflow_writes_l1_h2o_to_l2_oh_surface_and_is_registered(self) -> None:
        workflow = importlib.import_module("waterint._04_workflows.workflows.proton_sharing")
        registry = importlib.import_module("waterint._04_workflows.registry.registry")
        self.assertEqual(registry.get_analysis_module("proton-fes").name, "proton-sharing")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Mg establishes z_ref=0; O(0) is L1 H2O and O(1) is L2 OH-.
            positions = np.asarray([[
                [0.0, 0.0, 0.0], [0.0, 0.0, 1.8], [2.0, 0.0, 2.8],
                [1.0, 0.0, 2.3], [-0.8, 0.0, 1.8], [3.0, 0.0, 2.8],
            ]] * 3, dtype=float)
            types = np.repeat(np.asarray([[2, 3, 3, 1, 1, 1]]), 3, axis=0)
            np.savez(root / "trajectory.npz", positions=positions, types=types, cells=np.repeat(np.asarray([[10.0, 10.0, 10.0]]), 3, axis=0), steps=np.arange(3))
            config = {
                "_config_dir": str(root),
                "input": {"trajectory": "trajectory.npz", "format": "npz", "type_map": {1: "H", 2: "Mg", 3: "O"}},
                "system": {"cell": "auto"},
                "selection": {"oxygen_symbol": "O", "hydrogen_symbol": "H", "oh_cutoff": 1.25, "pbc": [True, True, False]},
                "proton_sharing": {
                    "temperature_K": 300.0,
                    "coordinate": {"mode": "relative_to_slab", "axis": "z", "reference": {"type": "top_layer_mean", "species": ["Mg"], "surface": "max", "layer_width": 0.7}},
                    "donor": {"species": "H2O", "range": [1.5, 2.8]}, "acceptor": {"species": "OH-", "range": [2.8, 4.0]},
                    "oo_range_A": [2.0, 3.0], "delta_range_A": [-1.5, 1.5], "delta_bins": 30, "oo_bins": 20,
                    "shared_delta_max_A": 0.2, "shared_s_range_A": [-1.5, 1.5], "shared_rho_range_A": [0.0, 1.5], "shared_s_bins": 30, "shared_rho_bins": 15,
                },
                "output": {"directory": "output", "prefix": "sharing", "plot": False},
            }
            result = workflow.run_proton_sharing(config)
            self.assertEqual(result.pair_samples, 3)
            self.assertEqual(result.shared_samples, 3)
            self.assertTrue(result.fes_csv_path.exists())
            self.assertTrue(result.shared_csv_path.exists())
            self.assertIn("delta_A,R_OO_A", result.fes_csv_path.read_text(encoding="utf-8"))

    def test_swapped_state_uses_reversed_species(self) -> None:
        workflow = importlib.import_module("waterint._04_workflows.workflows.proton_sharing")
        # The only selected pair is the exchanged L2-H2O / L1-OH- state.
        positions = np.asarray([[
            [0.0, 0.0, 1.8], [2.4, 0.0, 1.8],
            [1.2, 0.0, 1.8], [0.0, 0.0, 2.8], [2.4, 0.0, 0.6],
        ]], dtype=float)
        types = np.asarray([[3, 3, 1, 1, 1]])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.savez(root / "trajectory.npz", positions=positions, types=types, cells=np.asarray([[10.0, 10.0, 10.0]]), steps=np.asarray([0]))
            config = {
                "_config_dir": str(root),
                "input": {"trajectory": "trajectory.npz", "format": "npz", "type_map": {1: "H", 3: "O"}},
                "system": {"cell": "auto"},
                "selection": {"oxygen_symbol": "O", "hydrogen_symbol": "H", "oh_cutoff": 1.25, "pbc": [True, True, False]},
                "proton_sharing": {
                    "temperature_K": 300.0,
                    "coordinate": {"mode": "relative_to_slab", "axis": "z", "reference": {"type": "fixed", "value": 0.0}},
                    "donor": {"species": "H2O", "range": [1.5, 2.8]}, "acceptor": {"species": "OH-", "range": [1.5, 2.8]},
                    "include_swapped_state": True, "oo_range_A": [2.0, 3.0], "delta_range_A": [-1.5, 1.5], "delta_bins": 30, "oo_bins": 20,
                    "shared_delta_max_A": 0.2, "shared_s_range_A": [-1.5, 1.5], "shared_rho_range_A": [0.0, 1.5], "shared_s_bins": 30, "shared_rho_bins": 15,
                },
                "output": {"directory": "output", "prefix": "sharing", "plot": False},
            }
            result = workflow.run_proton_sharing(config)
            self.assertEqual(result.pair_samples, 2)


if __name__ == "__main__":
    unittest.main()
