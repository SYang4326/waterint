from __future__ import annotations

import importlib
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
V04 = ROOT / "v0.4"


class V04MsdRdfTests(unittest.TestCase):
    def setUp(self) -> None:
        self._old_path = list(sys.path)
        self._old_modules = {name: module for name, module in sys.modules.items() if name == "waterint" or name.startswith("waterint.")}
        for name in list(self._old_modules):
            sys.modules.pop(name, None)
        sys.path.insert(0, str(V04))

    def tearDown(self) -> None:
        for name in [name for name in sys.modules if name == "waterint" or name.startswith("waterint.")]:
            sys.modules.pop(name, None)
        sys.modules.update(self._old_modules)
        sys.path[:] = self._old_path

    def test_msd_unwraps_periodic_motion_and_2d_excludes_normal_axis(self) -> None:
        msd = importlib.import_module("waterint._02_computation.msd")
        positions = np.asarray(
            [
                [[9.0, 1.0, 0.0]],
                [[0.0, 2.0, 1.0]],
                [[1.0, 3.0, 2.0]],
            ],
            dtype=float,
        )
        vectors = np.repeat(np.diag([10.0, 10.0, 10.0])[None, :, :], 3, axis=0)
        result_3d = msd.compute_msd(positions, cell_vectors=vectors, pbc=(True, False, False), timestep_ps=1.0, max_lag_frames=2, dimensionality="3d", backend="python")
        result_2d = msd.compute_msd(positions, cell_vectors=vectors, pbc=(True, False, False), timestep_ps=1.0, max_lag_frames=2, dimensionality="2d", plane_normal_axis=2, backend="python")
        np.testing.assert_allclose(result_3d.msd_a2, [0.0, 3.0, 12.0])
        np.testing.assert_allclose(result_2d.msd_a2, [0.0, 2.0, 8.0])

    def test_cpp_msd_matches_python_for_triclinic_cell(self) -> None:
        msd = importlib.import_module("waterint._02_computation.msd")
        native = importlib.import_module("waterint._02_computation._native")
        if not native.native_status()["available"]:
            self.skipTest("C++ compiler is unavailable.")
        positions = np.asarray([[[8.9, 8.8, 0.0]], [[0.4, 0.3, 0.0]], [[1.9, 1.8, 0.0]]])
        basis = np.asarray([[10.0, 0.0, 0.0], [2.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
        vectors = np.repeat(basis[None, :, :], positions.shape[0], axis=0)
        kwargs = dict(cell_vectors=vectors, pbc=(True, True, False), timestep_ps=0.5, max_lag_frames=2, origin_stride=2, dimensionality="3d")
        python_result = msd.compute_msd(positions, backend="python", **kwargs)
        cpp_result = msd.compute_msd(positions, backend="cpp", **kwargs)
        np.testing.assert_allclose(cpp_result.msd_a2, python_result.msd_a2)
        np.testing.assert_array_equal(cpp_result.samples, python_result.samples)

    def test_rdf_same_selection_does_not_include_self_or_double_count(self) -> None:
        rdf = importlib.import_module("waterint._02_computation.rdf")
        positions = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        counts = rdf.new_histogram(4)
        indices = np.asarray([0, 1, 2])
        rdf.accumulate_rdf_frame(counts, positions=positions, first_indices=indices, second_indices=indices, r_max=4.0, cell_vectors=None, pbc=(False, False, False), same_selection=True, backend="python")
        np.testing.assert_array_equal(counts, [0.0, 1.0, 1.0, 1.0])

    def test_cpp_rdf_matches_python_with_pbc(self) -> None:
        rdf = importlib.import_module("waterint._02_computation.rdf")
        native = importlib.import_module("waterint._02_computation._native")
        if not native.native_status()["available"]:
            self.skipTest("C++ compiler is unavailable.")
        positions = np.asarray([[9.8, 0.0, 0.0], [0.2, 0.0, 0.0], [5.0, 0.0, 0.0]])
        indices = np.asarray([0, 1, 2])
        kwargs = dict(positions=positions, first_indices=indices, second_indices=indices, r_max=5.0, cell_vectors=np.diag([10.0, 10.0, 10.0]), pbc=(True, False, False), same_selection=True)
        python_counts = rdf.new_histogram(10)
        cpp_counts = rdf.new_histogram(10)
        rdf.accumulate_rdf_frame(python_counts, backend="python", **kwargs)
        rdf.accumulate_rdf_frame(cpp_counts, backend="cpp", **kwargs)
        np.testing.assert_array_equal(cpp_counts, python_counts)
        self.assertEqual(cpp_counts[0], 1.0)

    def test_cpp_rdf_cell_list_matches_python_for_random_triclinic_selection(self) -> None:
        rdf = importlib.import_module("waterint._02_computation.rdf")
        native = importlib.import_module("waterint._02_computation._native")
        if not native.native_status()["available"]:
            self.skipTest("C++ compiler is unavailable.")
        rng = np.random.default_rng(20260807)
        basis = np.asarray([[14.0, 0.0, 0.0], [2.0, 12.0, 0.0], [1.0, 0.8, 15.0]])
        positions = rng.random((120, 3)) @ basis
        first = np.arange(0, 70, dtype=int)
        second = np.arange(40, 120, dtype=int)
        kwargs = dict(positions=positions, first_indices=first, second_indices=second, r_max=5.5, cell_vectors=basis, pbc=(True, True, True), same_selection=False)
        python_counts = rdf.new_histogram(110)
        cpp_counts = rdf.new_histogram(110)
        rdf.accumulate_rdf_frame(python_counts, backend="python", **kwargs)
        rdf.accumulate_rdf_frame(cpp_counts, backend="cpp", **kwargs)
        np.testing.assert_array_equal(cpp_counts, python_counts)

    def test_msd_and_rdf_are_registered(self) -> None:
        registry = importlib.import_module("waterint._04_workflows.registry.registry")
        self.assertEqual(registry.get_analysis_module("msd").name, "msd")
        self.assertEqual(registry.get_analysis_module("rdf").name, "rdf")

    def test_workflows_write_outputs_from_npz(self) -> None:
        msd_workflow = importlib.import_module("waterint._04_workflows.workflows.msd")
        rdf_workflow = importlib.import_module("waterint._04_workflows.workflows.rdf")
        positions = np.asarray(
            [
                [[9.6, 0.0, 1.0], [0.4, 0.0, 1.0], [2.0, 0.0, 1.0]],
                [[0.1, 0.0, 1.0], [0.9, 0.0, 1.0], [2.2, 0.0, 1.0]],
                [[0.6, 0.0, 1.0], [1.4, 0.0, 1.0], [2.4, 0.0, 1.0]],
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.savez(
                root / "trajectory.npz",
                positions=positions,
                types=np.asarray([[3, 3, 1]] * 3),
                cells=np.asarray([[10.0, 10.0, 10.0]] * 3),
                steps=np.asarray([0, 1, 2]),
            )
            base = {
                "_config_dir": str(root),
                "input": {"trajectory": "trajectory.npz", "format": "npz", "type_map": {1: "H", 3: "O"}},
                "system": {"cell": "auto"},
                "selection": {"elements": ["O"]},
                "output": {"directory": "output", "plot": False},
            }
            msd_config = {**base, "msd": {"timestep_ps": 0.1, "max_lag_frames": 2, "pbc": [True, False, False]}, "output": {**base["output"], "prefix": "msd"}}
            msd_result = msd_workflow.run_msd(msd_config)
            self.assertTrue(msd_result.csv_path.exists())
            self.assertTrue(msd_result.metadata_path.exists())

            rdf_config = {**base, "rdf": {"r_max": 5.0, "bins": 20, "pbc": [True, False, False]}, "output": {**base["output"], "prefix": "rdf"}}
            rdf_result = rdf_workflow.run_rdf(rdf_config)
            self.assertIn("O-O", rdf_result.pairs)
            self.assertTrue(rdf_result.pairs["O-O"].csv_path.exists())
            self.assertTrue(rdf_result.metadata_path.exists())


if __name__ == "__main__":
    unittest.main()
