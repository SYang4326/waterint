from __future__ import annotations

import importlib
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
V03 = ROOT / "v0.3"


class V03OhOrientationCppTests(unittest.TestCase):
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

    def test_cpp_neighbor_matrix_matches_matrix_lists(self):
        chemistry = importlib.import_module("waterint.chemistry")
        native = importlib.import_module("waterint._02_computation._native")
        if not native.native_status()["available"]:
            self.skipTest("C++ compiler or native backend is not available.")

        oxygen_positions, hydrogen_positions, _, _ = self._orientation_system()
        cpp = native.hydrogen_neighbor_matrix(oxygen_positions, hydrogen_positions, cutoff=1.25)
        self.assertIsNotNone(cpp)
        counts, matrix = cpp
        python_lists = chemistry._hydrogen_neighbor_lists(
            oxygen_positions=oxygen_positions,
            hydrogen_positions=hydrogen_positions,
            cutoff=1.25,
            method="matrix",
            workers=1,
            oxygen_chunk_size=8,
            cell=None,
            pbc=None,
        )

        np.testing.assert_array_equal(counts, np.asarray([1, 2, 3]))
        for oxygen_index, expected in enumerate(python_lists):
            np.testing.assert_array_equal(matrix[oxygen_index, : counts[oxygen_index]], expected)

    def test_cpp_neighbor_matrix_expands_for_high_coordination(self):
        native = importlib.import_module("waterint._02_computation._native")
        if not native.native_status()["available"]:
            self.skipTest("C++ compiler or native backend is not available.")

        oxygen_positions = np.asarray([[0.0, 0.0, 0.0]])
        hydrogen_positions = np.asarray(
            [
                [0.5, 0.0, 0.0],
                [-0.5, 0.0, 0.0],
                [0.0, 0.5, 0.0],
                [0.0, -0.5, 0.0],
                [0.0, 0.0, 0.5],
                [0.0, 0.0, -0.5],
            ]
        )
        counts, matrix = native.hydrogen_neighbor_matrix(oxygen_positions, hydrogen_positions, cutoff=1.0)

        np.testing.assert_array_equal(counts, np.asarray([6]))
        self.assertGreaterEqual(matrix.shape[1], 6)
        np.testing.assert_array_equal(matrix[0, :6], np.arange(6))

    def test_cpp_neighbor_matrix_supports_periodic_boundaries(self):
        native = importlib.import_module("waterint._02_computation._native")
        if not native.native_status()["available"]:
            self.skipTest("C++ compiler or native backend is not available.")

        counts, matrix = native.hydrogen_neighbor_matrix(
            np.asarray([[9.8, 0.0, 0.0]]),
            np.asarray([[0.2, 0.0, 0.0]]),
            cutoff=0.5,
            cell=(10.0, 10.0, 10.0),
            pbc=(True, False, False),
        )

        np.testing.assert_array_equal(counts, np.asarray([1]))
        np.testing.assert_array_equal(matrix[0, :1], np.asarray([0]))

    def test_cpp_oh_bond_histogram_matches_python(self):
        self._assert_cpp_matches_python("oh_bond")

    def test_cpp_bisector_histogram_matches_python(self):
        self._assert_cpp_matches_python("oh_bisector")

    def test_cpp_dipole_histogram_matches_current_python_behavior(self):
        self._assert_cpp_matches_python("dipole")

    def test_python_and_cpp_workflows_match_on_npz(self):
        workflow = importlib.import_module("waterint._04_workflows.workflows.oh_orientation")
        native = importlib.import_module("waterint._02_computation._native")
        if not native.native_status()["available"]:
            self.skipTest("C++ compiler or native backend is not available.")

        oxygen_positions, hydrogen_positions, positions, _ = self._orientation_system()
        frame_positions = np.stack([positions, positions + np.asarray([0.0, 0.0, 0.1])])
        atom_types = np.asarray([3] * oxygen_positions.shape[0] + [1] * hydrogen_positions.shape[0], dtype=int)
        frame_types = np.stack([atom_types, atom_types])

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            trajectory = tmp_path / "orientation.npz"
            np.savez(
                trajectory,
                positions=frame_positions,
                types=frame_types,
                cells=np.asarray([[12.0, 12.0, 12.0], [12.0, 12.0, 12.0]]),
                steps=np.asarray([0, 1], dtype=int),
            )
            results = {}
            for backend in ["python", "cpp"]:
                output = tmp_path / backend
                config = {
                    "_config_dir": str(tmp_path),
                    "_config_path": str(tmp_path / f"{backend}.yaml"),
                    "input": {
                        "trajectory": str(trajectory),
                        "format": "npz",
                        "type_map": {1: "H", 3: "O"},
                        "max_frames": "all",
                        "stride": 1,
                    },
                    "system": {"cell": "auto"},
                    "selection": {
                        "oxygen_species": ["OH-", "H2O", "H3O+"],
                        "oxygen_symbol": "O",
                        "hydrogen_symbol": "H",
                        "oh_cutoff": 1.25,
                        "neighbor_method": "matrix",
                    },
                    "coordinate": {"mode": "absolute", "axis": "z", "range": [0.0, 5.0], "bins": 10},
                    "angle": {
                        "backend": backend,
                        "vector_mode": "oh_bond",
                        "range": [0.0, 180.0],
                        "bins": 18,
                        "axis_sign": 1.0,
                    },
                    "normalization": {"type": "counts_per_frame"},
                    "output": {"directory": str(output), "prefix": "orientation", "plot": False},
                }
                results[backend] = workflow.run_oh_orientation(config)

        python_result = results["python"]
        cpp_result = results["cpp"]
        self.assertEqual(cpp_result.frames, python_result.frames)
        self.assertEqual(cpp_result.bond_counts_total, python_result.bond_counts_total)
        self.assertEqual(cpp_result.sample_counts_total, python_result.sample_counts_total)
        for label in python_result.histograms:
            np.testing.assert_array_equal(cpp_result.histograms[label], python_result.histograms[label])

    def _assert_cpp_matches_python(self, vector_mode: str) -> None:
        chemistry = importlib.import_module("waterint.chemistry")
        orientation = importlib.import_module("waterint._02_computation.oh_orientation")
        native = importlib.import_module("waterint._02_computation._native")
        if not native.native_status()["available"]:
            self.skipTest("C++ compiler or native backend is not available.")

        oxygen_positions, hydrogen_positions, positions, symbols = self._orientation_system()
        oxygen_indices = np.arange(oxygen_positions.shape[0], dtype=int)
        hydrogen_indices = np.arange(oxygen_positions.shape[0], positions.shape[0], dtype=int)
        labels = ["OH-", "H2O", "H3O+"]
        z_edges = np.linspace(0.0, 5.0, 11)
        angle_edges = np.linspace(0.0, 180.0, 19)
        python_state = orientation.new_oh_orientation_state(labels, z_edges, angle_edges)
        cpp_state = orientation.new_oh_orientation_state(labels, z_edges, angle_edges)

        if vector_mode == "oh_bond":
            pairs_by_species = chemistry.oxygen_hydrogen_pairs_by_species(
                symbols,
                positions,
                oh_cutoff=1.25,
                neighbor_method="matrix",
                oxygen_chunk_size=8,
                oxygen_indices=oxygen_indices,
                hydrogen_indices=hydrogen_indices,
            )
            for label in labels:
                pairs = pairs_by_species[label]
                if pairs.size == 0:
                    continue
                z_values, angle_values = orientation.pair_z_and_angles(
                    positions=positions,
                    pairs=pairs,
                    axis=2,
                    axis_sign=1.0,
                    reference=0.0,
                    angle_axis_sign=1.0,
                )
                orientation.accumulate_angle_samples(
                    python_state,
                    label,
                    z_values,
                    angle_values,
                    bond_count=pairs.shape[0],
                )
        else:
            neighbors_by_species = chemistry.oxygen_hydrogen_neighbors_by_species(
                symbols,
                positions,
                oh_cutoff=1.25,
                neighbor_method="matrix",
                oxygen_chunk_size=8,
                oxygen_indices=oxygen_indices,
                hydrogen_indices=hydrogen_indices,
            )
            for label in labels:
                z_values, angle_values, bond_count = orientation.neighbor_bisector_z_and_angles(
                    positions=positions,
                    neighbors=neighbors_by_species[label],
                    axis=2,
                    axis_sign=1.0,
                    reference=0.0,
                    angle_axis_sign=1.0,
                )
                orientation.accumulate_angle_samples(
                    python_state,
                    label,
                    z_values,
                    angle_values,
                    bond_count=bond_count,
                )

        neighbor_data = orientation.build_oh_neighbor_matrix_cpp(
            oxygen_positions=oxygen_positions,
            hydrogen_positions=hydrogen_positions,
            cutoff=1.25,
        )
        self.assertIsNotNone(neighbor_data)
        neighbor_counts, neighbor_matrix = neighbor_data
        used_cpp = orientation.accumulate_oh_orientation_cpp(
            cpp_state,
            oxygen_positions=oxygen_positions,
            hydrogen_positions=hydrogen_positions,
            neighbor_counts=neighbor_counts,
            neighbor_matrix=neighbor_matrix,
            vector_mode=vector_mode,
            axis=2,
            axis_sign=1.0,
            reference=0.0,
            angle_axis_sign=1.0,
        )

        self.assertTrue(used_cpp)
        np.testing.assert_array_equal(cpp_state.counts_buffer, python_state.counts_buffer)
        self.assertEqual(cpp_state.bond_counts_total, python_state.bond_counts_total)
        self.assertEqual(cpp_state.sample_counts_total, python_state.sample_counts_total)

    @staticmethod
    def _orientation_system() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
        oxygen_positions = np.asarray(
            [
                [0.0, 0.0, 1.0],
                [4.0, 0.0, 2.0],
                [8.0, 0.0, 3.0],
            ],
            dtype=float,
        )
        hydrogen_positions = np.asarray(
            [
                [0.0, 0.0, 2.0],
                [4.8660254038, 0.0, 2.5],
                [3.1339745962, 0.0, 2.5],
                [8.8, 0.0, 3.6],
                [7.6, 0.6928203230, 3.6],
                [7.6, -0.6928203230, 3.6],
            ],
            dtype=float,
        )
        positions = np.vstack([oxygen_positions, hydrogen_positions])
        symbols = ["O"] * oxygen_positions.shape[0] + ["H"] * hydrogen_positions.shape[0]
        return oxygen_positions, hydrogen_positions, positions, symbols


if __name__ == "__main__":
    unittest.main()
