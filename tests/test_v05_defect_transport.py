from __future__ import annotations

import importlib
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
V05 = ROOT / "v0.5"


class V05DefectTransportTests(unittest.TestCase):
    def setUp(self) -> None:
        self._old_path = list(sys.path)
        self._old_modules = {
            name: module
            for name, module in sys.modules.items()
            if name == "waterint" or name.startswith("waterint.")
        }
        for name in list(self._old_modules):
            sys.modules.pop(name, None)
        sys.path.insert(0, str(V05))

    def tearDown(self) -> None:
        for name in [name for name in sys.modules if name == "waterint" or name.startswith("waterint.")]:
            sys.modules.pop(name, None)
        sys.modules.update(self._old_modules)
        sys.path[:] = self._old_path

    def test_hungarian_tracking_follows_pbc_defect_and_records_death(self) -> None:
        module = importlib.import_module("waterint._02_computation.defect_transport")
        positions = [
            np.asarray([[9.5, 0.0, 0.0], [5.0, 0.0, 0.0]]),
            np.asarray([[0.5, 0.0, 0.0], [6.0, 0.0, 0.0]]),
            np.asarray([[1.5, 0.0, 0.0]]),
        ]
        atom_indices = [np.asarray([10, 20]), np.asarray([11, 20]), np.asarray([11])]
        cells = np.repeat(np.diag([10.0, 10.0, 10.0])[None, :, :], 3, axis=0)
        tracking = module.track_defects(
            positions,
            atom_indices,
            cell_vectors=cells,
            pbc=(True, False, False),
            timestep_ps=1.0,
            gate_a=2.0,
            charge_e=-1.0,
        )
        np.testing.assert_array_equal(tracking.carrier_counts, [2, 2, 1])
        np.testing.assert_array_equal(tracking.births, [0, 0, 0])
        np.testing.assert_array_equal(tracking.deaths, [0, 0, 1])
        np.testing.assert_allclose(tracking.charge_current_ea_per_ps[:, 0], [-2.0, -1.0])
        longest = max(tracking.segments, key=lambda item: len(item.frame_indices))
        np.testing.assert_array_equal(longest.atom_indices, [10, 11, 11])
        np.testing.assert_allclose(longest.positions_a[:, 0], [9.5, 10.5, 11.5])

        msd = module.compute_defect_msd(
            tracking,
            max_lag_frames=2,
            dimensionality="2d",
            plane_normal_axis=2,
        )
        np.testing.assert_allclose(msd.msd_a2, [0.0, 1.0, 4.0])
        np.testing.assert_array_equal(msd.samples, [5, 3, 1])

        strided_msd = module.compute_defect_msd(
            tracking,
            max_lag_frames=1,
            frame_stride=2,
            dimensionality="2d",
            plane_normal_axis=2,
        )
        np.testing.assert_allclose(strided_msd.time_ps, [0.0, 2.0])
        np.testing.assert_allclose(strided_msd.msd_a2, [0.0, 4.0])
        np.testing.assert_array_equal(strided_msd.samples, [3, 1])

    def test_defect_workflows_classify_each_frame_and_write_tracks(self) -> None:
        msd_workflow = importlib.import_module("waterint._04_workflows.workflows.defect_msd")
        conductivity_workflow = importlib.import_module(
            "waterint._04_workflows.workflows.defect_conductivity"
        )
        positions = np.asarray(
            [
                [[0.0, 0, 0], [3.0, 0, 0], [0.9, 0, 0], [2.1, 0, 0], [3.9, 0, 0]],
                [[0.5, 0, 0], [3.5, 0, 0], [1.4, 0, 0], [2.6, 0, 0], [4.4, 0, 0]],
                [[1.0, 0, 0], [4.0, 0, 0], [0.1, 0, 0], [1.9, 0, 0], [4.9, 0, 0]],
                [[1.5, 0, 0], [4.5, 0, 0], [0.6, 0, 0], [2.4, 0, 0], [5.4, 0, 0]],
            ],
            dtype=float,
        )
        types = np.repeat(np.asarray([[3, 3, 1, 1, 1]]), 4, axis=0)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.savez(
                root / "trajectory.npz",
                positions=positions,
                types=types,
                cells=np.repeat(np.asarray([[10.0, 10.0, 10.0]]), 4, axis=0),
                steps=np.arange(4),
            )
            base = {
                "_config_dir": str(root),
                "input": {
                    "trajectory": "trajectory.npz",
                    "format": "npz",
                    "type_map": {1: "H", 3: "O"},
                },
                "system": {"cell": "auto"},
                "selection": {
                    "oxygen_species": ["OH-"],
                    "oxygen_symbol": "O",
                    "hydrogen_symbol": "H",
                    "hydrogen_assignment": "nearest",
                    "oh_cutoff": 1.25,
                },
                "defect_tracking": {
                    "timestep_ps": 1.0,
                    "gate_A": 4.0,
                    "pbc": [False, False, False],
                },
                "defect_msd": {
                    "dimensionality": "2d",
                    "plane_normal_axis": "z",
                    "max_lag_frames": 3,
                },
                "output": {"directory": "output", "plot": False},
            }
            msd_result = msd_workflow.run_defect_msd(
                {**base, "output": {**base["output"], "prefix": "defect_msd"}}
            )
            self.assertTrue(msd_result.csv_path.exists())
            self.assertTrue(msd_result.tracks_csv_path.exists())
            self.assertIn("track_id", msd_result.tracks_csv_path.read_text(encoding="utf-8"))
            self.assertEqual(msd_result.mean_carriers, 1.0)

            conductivity_result = conductivity_workflow.run_defect_conductivity(
                {
                    **base,
                    "defect_conductivity": {
                        "estimator": "nernst_einstein",
                        "temperature_K": 300.0,
                        "charge_e": -1.0,
                        "fit_range_ps": [1.0, 3.0],
                        "volume": {"mode": "slab", "normal_axis": "z", "thickness_A": 4.5},
                    },
                    "output": {**base["output"], "prefix": "defect_conductivity"},
                }
            )
            self.assertIsNotNone(conductivity_result.nernst_einstein)
            self.assertIsNone(conductivity_result.green_kubo)
            self.assertIn("defect_nernst_einstein", conductivity_result.csv_path.read_text())

    def test_green_kubo_fits_cartesian_components_jointly(self) -> None:
        module = importlib.import_module("waterint._02_computation.defect_transport")
        positions = [
            np.asarray([[float(frame), 2.0 * frame, 0.0]])
            for frame in range(4)
        ]
        tracking = module.track_defects(
            positions,
            [np.asarray([0])] * 4,
            cell_vectors=np.repeat(np.diag([20.0, 20.0, 20.0])[None, :, :], 4, axis=0),
            pbc=(False, False, False),
            timestep_ps=1.0,
            gate_a=4.0,
        )
        sequence_counts = []
        fake_stacie = types.ModuleType("stacie")
        fake_stacie.ExpPolyModel = lambda orders: tuple(orders)

        def compute_spectrum(sequences, **kwargs):
            sequence_counts.append(sequences.shape[0])
            return sequences.shape[0]

        def estimate_acint(spectrum, model, verbose=False):
            return types.SimpleNamespace(acint=float(spectrum), acint_std=0.1 * spectrum)

        fake_stacie.compute_spectrum = compute_spectrum
        fake_stacie.estimate_acint = estimate_acint
        old_stacie = sys.modules.get("stacie")
        sys.modules["stacie"] = fake_stacie
        try:
            result = module.compute_green_kubo_conductivity(
                tracking,
                volume_a3=1000.0,
                temperature_k=300.0,
                dimensionality="2d",
                plane_normal_axis=2,
            )
        finally:
            if old_stacie is None:
                sys.modules.pop("stacie", None)
            else:
                sys.modules["stacie"] = old_stacie

        self.assertEqual(sequence_counts, [2, 1, 1])
        self.assertEqual(result.conductivity_s_per_m, 2.0)
        np.testing.assert_allclose(result.component_conductivity_s_per_m, [1.0, 1.0])

    def test_dynamic_defect_modules_are_registered(self) -> None:
        registry = importlib.import_module("waterint._04_workflows.registry.registry")
        self.assertEqual(registry.get_analysis_module("atom-msd").name, "msd")
        self.assertEqual(registry.get_analysis_module("defect-msd").name, "defect-msd")
        self.assertEqual(
            registry.get_analysis_module("conductivity-defect").name, "defect-conductivity"
        )


if __name__ == "__main__":
    unittest.main()
