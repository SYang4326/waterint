from __future__ import annotations

import importlib
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
V03 = ROOT / "v0.3"


class V03HbondCppTests(unittest.TestCase):
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

    def test_cpp_counts_match_python_geometry(self):
        positions, oxygen_indices, hydrogen_indices = self._hbond_system()
        self._assert_native_matches_python(
            positions,
            oxygen_indices,
            hydrogen_indices,
            cell=(12.0, 12.0, 12.0),
            pbc=(False, False, False),
            max_acceptors=True,
        )

    def test_cpp_counts_match_python_with_periodic_acceptor(self):
        positions = np.asarray(
            [
                [9.7, 0.0, 0.0],
                [0.3, 0.0, 0.0],
                [9.0, 0.0, 0.0],
            ],
            dtype=float,
        )
        self._assert_native_matches_python(
            positions,
            np.asarray([0, 1]),
            np.asarray([2]),
            cell=(10.0, 10.0, 10.0),
            pbc=(True, False, False),
            max_acceptors=True,
        )

    def test_cpp_preserves_first_acceptor_tie_and_multiple_acceptor_mode(self):
        positions = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [2.8, 1.0, 0.0],
                [2.8, -1.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=float,
        )
        oxygen_indices = np.asarray([0, 1, 2])
        hydrogen_indices = np.asarray([3])
        for max_acceptors in [True, False]:
            with self.subTest(max_acceptors=max_acceptors):
                native, expected = self._native_and_python_counts(
                    positions,
                    oxygen_indices,
                    hydrogen_indices,
                    cell=(12.0, 12.0, 12.0),
                    pbc=(False, False, False),
                    max_acceptors=max_acceptors,
                    angle_min=140.0,
                )
                for actual, wanted in zip(native, expected):
                    np.testing.assert_array_equal(actual, wanted)
        native, _ = self._native_and_python_counts(
            positions,
            oxygen_indices,
            hydrogen_indices,
            cell=(12.0, 12.0, 12.0),
            pbc=(False, False, False),
            max_acceptors=True,
            angle_min=140.0,
        )
        np.testing.assert_array_equal(native[2], np.asarray([0, 1, 0]))

    def test_cpp_hydrogen_acceptor_cutoff_matches_python(self):
        positions, oxygen_indices, hydrogen_indices = self._hbond_system()
        native, expected = self._native_and_python_counts(
            positions,
            oxygen_indices,
            hydrogen_indices,
            cell=(12.0, 12.0, 12.0),
            pbc=(False, False, False),
            max_acceptors=True,
            h_acceptor_cutoff=1.9,
        )
        for actual, wanted in zip(native, expected):
            np.testing.assert_array_equal(actual, wanted)

    def test_python_cutoff_celllist_matches_bruteforce_random_systems(self):
        chemistry = importlib.import_module("waterint.chemistry")
        rng = np.random.default_rng(20260713)
        for pbc, cell in [
            ((False, False, False), (10.0, 10.0, 10.0)),
            ((True, True, False), (4.0, 5.0, 10.0)),
            ((True, True, True), (2.0, 2.0, 2.0)),
        ]:
            queries = rng.uniform(0.0, np.asarray(cell), size=(20, 3))
            candidates = rng.uniform(0.0, np.asarray(cell), size=(35, 3))
            search = chemistry.PythonCutoffNeighborSearch(
                queries,
                candidates,
                cutoff=1.25,
                cell=cell,
                pbc=pbc,
            )
            for query_index in range(queries.shape[0]):
                vectors = candidates - queries[query_index]
                for axis, enabled in enumerate(pbc):
                    if enabled:
                        vectors[:, axis] -= np.rint(vectors[:, axis] / cell[axis]) * cell[axis]
                expected = np.where(np.einsum("ij,ij->i", vectors, vectors) <= 1.25**2)[0]
                with self.subTest(pbc=pbc, query=query_index):
                    np.testing.assert_array_equal(search.collect_indices(query_index), expected)

    def test_native_library_rebuilds_when_header_is_newer(self):
        native = importlib.import_module("waterint._02_computation._native")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            library = tmp_path / "native.so"
            library.write_bytes(b"library")
            os.utime(library, (1.0, 1.0))
            self.assertTrue(native._native_library_needs_build(library))

    def test_python_and_cpp_workflows_match_on_npz(self):
        workflow = importlib.import_module("waterint._04_workflows.workflows.hbond")
        native = importlib.import_module("waterint._02_computation._native")
        if not native.native_status()["available"]:
            self.skipTest("C++ compiler or native backend is not available.")

        positions, _, _ = self._hbond_system()
        frame_positions = np.stack([positions, positions + np.asarray([0.0, 0.0, 0.1])])
        atom_types = np.asarray([3, 3, 3, 3, 1, 1, 1, 1, 1, 1], dtype=int)
        frame_types = np.stack([atom_types, atom_types])
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            trajectory = tmp_path / "hbond.npz"
            np.savez(
                trajectory,
                positions=frame_positions,
                types=frame_types,
                cells=np.asarray([[12.0, 12.0, 12.0], [12.0, 12.0, 12.0]]),
                steps=np.asarray([0, 1], dtype=int),
            )
            results = {}
            for backend in ["python", "cpp"]:
                config = self._workflow_config(tmp_path, trajectory, backend)
                results[backend] = workflow.run_hbond(config)

        python_result = results["python"]
        cpp_result = results["cpp"]
        self.assertEqual(cpp_result.frames, python_result.frames)
        self.assertEqual(cpp_result.counts, python_result.counts)
        self.assertEqual(cpp_result.raw_counts, python_result.raw_counts)
        self.assertEqual(cpp_result.samples_total, python_result.samples_total)
        self.assertEqual(cpp_result.fractions, python_result.fractions)

    def _assert_native_matches_python(self, *args, **kwargs) -> None:
        native, expected = self._native_and_python_counts(*args, **kwargs)
        for actual, wanted in zip(native, expected):
            np.testing.assert_array_equal(actual, wanted)

    def _native_and_python_counts(
        self,
        positions,
        oxygen_indices,
        hydrogen_indices,
        *,
        cell,
        pbc,
        max_acceptors,
        angle_min=150.0,
        h_acceptor_cutoff=None,
    ):
        native_module = importlib.import_module("waterint._02_computation._native")
        hbond_module = importlib.import_module("waterint._02_computation.hbond")
        chemistry = importlib.import_module("waterint.chemistry")
        if not native_module.native_status()["available"]:
            self.skipTest("C++ compiler or native backend is not available.")

        oxygen_positions = positions[oxygen_indices]
        hydrogen_positions = positions[hydrogen_indices]
        neighbor_data = native_module.hydrogen_neighbor_matrix(
            oxygen_positions,
            hydrogen_positions,
            cutoff=1.25,
        )
        self.assertIsNotNone(neighbor_data)
        h_counts, h_matrix = neighbor_data
        geometry_counts = native_module.hbond_geometry_counts(
            oxygen_positions,
            hydrogen_positions,
            hydrogen_counts=h_counts,
            hydrogen_matrix=h_matrix,
            oo_cutoff=3.5,
            dha_angle_min=angle_min,
            h_acceptor_cutoff=h_acceptor_cutoff,
            cell=cell,
            pbc=pbc,
            max_acceptors_per_hydrogen=max_acceptors,
        )
        self.assertIsNotNone(geometry_counts)
        donor_counts_native, acceptor_counts_native = geometry_counts
        native = (h_counts, donor_counts_native, acceptor_counts_native)
        neighbor_lists = chemistry._hydrogen_neighbor_lists(
            oxygen_positions=oxygen_positions,
            hydrogen_positions=hydrogen_positions,
            cutoff=1.25,
            method="matrix",
            workers=1,
            oxygen_chunk_size=16,
            cell=None,
            pbc=None,
        )
        donors = [
            (int(oxygen_indices[index]), hydrogen_indices[neighbors])
            for index, neighbors in enumerate(neighbor_lists)
            if neighbors.size
        ]
        bonds = hbond_module.find_hbonds(
            positions=positions,
            oxygen_indices=oxygen_indices,
            donor_neighbors=donors,
            hbond_cfg={
                "oo_cutoff": 3.5,
                "dha_angle_min": angle_min,
                "h_acceptor_cutoff": h_acceptor_cutoff,
                "max_acceptors_per_hydrogen": max_acceptors,
                "pbc": list(pbc),
            },
            cell=cell,
        )
        donor_counts = np.zeros(oxygen_indices.size, dtype=np.int64)
        acceptor_counts = np.zeros(oxygen_indices.size, dtype=np.int64)
        local = {int(index): slot for slot, index in enumerate(oxygen_indices)}
        for donor, _hydrogen, acceptor in bonds:
            donor_counts[local[donor]] += 1
            acceptor_counts[local[acceptor]] += 1
        expected = (
            np.asarray([len(items) for items in neighbor_lists]),
            donor_counts,
            acceptor_counts,
        )
        return native, expected

    @staticmethod
    def _hbond_system():
        positions = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [2.8, 0.0, 0.0],
                [5.6, 0.0, 0.0],
                [8.4, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [-0.8, 0.0, 0.0],
                [3.8, 0.0, 0.0],
                [2.8, 0.9, 0.0],
                [6.6, 0.0, 0.0],
                [5.6, 0.9, 0.0],
            ],
            dtype=float,
        )
        return positions, np.arange(4, dtype=int), np.arange(4, 10, dtype=int)

    @staticmethod
    def _workflow_config(tmp_path: Path, trajectory: Path, backend: str):
        return {
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
            "hbond": {
                "backend": backend,
                "oo_cutoff": 3.5,
                "dha_angle_min": 150.0,
                "h_acceptor_cutoff": None,
                "max_acceptors_per_hydrogen": True,
                "pbc": [False, False, False],
            },
            "output": {"directory": str(tmp_path / backend), "prefix": "hbond", "plot": False},
        }


if __name__ == "__main__":
    unittest.main()
