from __future__ import annotations

import importlib
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
V03 = ROOT / "v0.3"


class V03DensityCppTests(unittest.TestCase):
    def setUp(self) -> None:
        self._old_path = list(sys.path)
        self._old_modules = {name: module for name, module in sys.modules.items() if name == "waterint" or name.startswith("waterint.")}
        for name in list(self._old_modules):
            sys.modules.pop(name, None)
        sys.path.insert(0, str(V03))

    def tearDown(self) -> None:
        for name in [name for name in sys.modules if name == "waterint" or name.startswith("waterint.")]:
            sys.modules.pop(name, None)
        sys.modules.update(self._old_modules)
        sys.path[:] = self._old_path

    def test_cpp_histogram_matches_numpy_histogram(self):
        density = importlib.import_module("waterint._02_computation.density")
        native = importlib.import_module("waterint._02_computation._native")
        if not native.native_status()["available"]:
            self.skipTest("C++ compiler or native density backend is not available.")

        values = np.asarray([-1.0, 0.0, 0.1, 0.9, 1.0, 1.999, 2.0, 2.1, np.nan])
        edges = np.linspace(0.0, 2.0, 5)
        expected, _ = np.histogram(values, bins=edges)
        actual = density.histogram_counts(values, edges, backend="cpp")

        np.testing.assert_array_equal(actual, expected.astype(float))

    def test_auto_histogram_matches_python_histogram(self):
        density = importlib.import_module("waterint._02_computation.density")

        rng = np.random.default_rng(20260708)
        values = rng.normal(loc=5.0, scale=2.0, size=10000)
        edges = np.linspace(-2.0, 12.0, 141)
        python_counts = density.histogram_counts(values, edges, backend="python")
        auto_counts = density.histogram_counts(values, edges, backend="auto")

        np.testing.assert_array_equal(auto_counts, python_counts)

    def test_cpp_neighbor_counts_match_matrix_counts(self):
        chemistry = importlib.import_module("waterint.chemistry")
        native = importlib.import_module("waterint._02_computation._native")
        if not native.native_status()["available"]:
            self.skipTest("C++ compiler or native backend is not available.")

        oxygen_positions = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [9.8, 0.0, 0.0],
            ],
            dtype=float,
        )
        hydrogen_positions = np.asarray(
            [
                [0.8, 0.0, 0.0],
                [2.7, 0.0, 0.0],
                [0.2, 0.0, 0.0],
                [6.0, 0.0, 0.0],
            ],
            dtype=float,
        )

        cpp_counts = chemistry._count_hydrogen_neighbors(
            oxygen_positions=oxygen_positions,
            hydrogen_positions=hydrogen_positions,
            cutoff=1.0,
            method="cpp",
            workers=1,
            oxygen_chunk_size=2,
            cell=(10.0, 10.0, 10.0),
            pbc=(True, False, False),
        )
        matrix_counts = chemistry._count_hydrogen_neighbors(
            oxygen_positions=oxygen_positions,
            hydrogen_positions=hydrogen_positions,
            cutoff=1.0,
            method="matrix",
            workers=1,
            oxygen_chunk_size=2,
            cell=(10.0, 10.0, 10.0),
            pbc=(True, False, False),
        )

        np.testing.assert_array_equal(cpp_counts, matrix_counts)
        np.testing.assert_array_equal(cpp_counts, np.asarray([2, 1, 2]))

    def test_cpp_oxygen_species_classification_matches_matrix(self):
        chemistry = importlib.import_module("waterint.chemistry")
        native = importlib.import_module("waterint._02_computation._native")
        if not native.native_status()["available"]:
            self.skipTest("C++ compiler or native backend is not available.")

        symbols = ["O", "O", "O", "H", "H", "H", "H"]
        positions = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [9.8, 0.0, 0.0],
                [0.8, 0.0, 0.0],
                [2.7, 0.0, 0.0],
                [0.2, 0.0, 0.0],
                [6.0, 0.0, 0.0],
            ],
            dtype=float,
        )
        kwargs = dict(
            symbols=symbols,
            positions=positions,
            oxygen_symbol="O",
            hydrogen_symbol="H",
            oh_cutoff=1.0,
            oxygen_indices=np.asarray([0, 1, 2], dtype=int),
            hydrogen_indices=np.asarray([3, 4, 5, 6], dtype=int),
            cell=(10.0, 10.0, 10.0),
            pbc=(True, False, False),
        )
        cpp = chemistry.classify_oxygen_by_h_count(neighbor_method="cpp", neighbor_workers=1, oxygen_chunk_size=2, **kwargs)
        matrix = chemistry.classify_oxygen_by_h_count(neighbor_method="matrix", neighbor_workers=1, oxygen_chunk_size=2, **kwargs)

        for label in ["O2-", "OH-", "H2O", "H3O+", "O_other"]:
            np.testing.assert_array_equal(cpp[label], matrix[label])
        np.testing.assert_array_equal(cpp["OH-"], np.asarray([1]))
        np.testing.assert_array_equal(cpp["H2O"], np.asarray([0, 2]))

    def test_cpp_species_classification_matches_matrix_random_nonperiodic(self):
        chemistry = importlib.import_module("waterint.chemistry")
        native = importlib.import_module("waterint._02_computation._native")
        if not native.native_status()["available"]:
            self.skipTest("C++ compiler or native backend is not available.")

        rng = np.random.default_rng(20260708)
        oxygen_positions = rng.uniform(-2.0, 8.0, size=(25, 3))
        hydrogen_positions = rng.uniform(-2.0, 8.0, size=(50, 3))
        positions = np.vstack([oxygen_positions, hydrogen_positions])
        symbols = ["O"] * oxygen_positions.shape[0] + ["H"] * hydrogen_positions.shape[0]
        oxygen_indices = np.arange(oxygen_positions.shape[0], dtype=int)
        hydrogen_indices = np.arange(oxygen_positions.shape[0], positions.shape[0], dtype=int)
        kwargs = dict(
            symbols=symbols,
            positions=positions,
            oxygen_symbol="O",
            hydrogen_symbol="H",
            oh_cutoff=1.25,
            oxygen_indices=oxygen_indices,
            hydrogen_indices=hydrogen_indices,
            cell=None,
            pbc=None,
        )

        cpp = chemistry.classify_oxygen_by_h_count(neighbor_method="cpp", neighbor_workers=1, oxygen_chunk_size=8, **kwargs)
        matrix = chemistry.classify_oxygen_by_h_count(neighbor_method="matrix", neighbor_workers=1, oxygen_chunk_size=8, **kwargs)

        for label in ["O2-", "OH-", "H2O", "H3O+", "O_other"]:
            np.testing.assert_array_equal(cpp[label], matrix[label])

    def test_cli_help_imports_v03(self):
        cli = importlib.import_module("waterint.cli")

        with self.assertRaises(SystemExit) as caught:
            cli.main(["density", "--help"])
        self.assertEqual(caught.exception.code, 0)

    def test_cpp_xyz_reader_matches_python_reader(self):
        xyz = importlib.import_module("waterint._00_io.xyz")
        native = importlib.import_module("waterint._02_computation._native")
        if not native.native_status()["available"]:
            self.skipTest("C++ compiler or native backend is not available.")

        content = """3
frame 0
O 0.0 0.1 0.2
H 0.8 0.1 0.2
Mg 2.0 2.1 2.2
3
frame 1
O 0.2 0.3 0.4
H 0.9 0.3 0.4
Mg 2.2 2.3 2.4
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tiny.xyz"
            path.write_text(content, encoding="utf-8")
            python_frames = list(xyz.read_xyz(path, reader="python"))
            cpp_frames = list(xyz.read_xyz(path, reader="cpp"))

        self.assertEqual(len(cpp_frames), len(python_frames))
        self.assertEqual(cpp_frames[0].symbols, python_frames[0].symbols)
        for actual, expected in zip(cpp_frames, python_frames):
            np.testing.assert_allclose(actual.positions, expected.positions)

    def test_cpp_lammpstrj_reader_matches_python_reader(self):
        lammpstrj = importlib.import_module("waterint._00_io.lammpstrj")
        native = importlib.import_module("waterint._02_computation._native")
        if not native.native_status()["available"]:
            self.skipTest("C++ compiler or native backend is not available.")

        content = """ITEM: TIMESTEP
10
ITEM: NUMBER OF ATOMS
3
ITEM: BOX BOUNDS pp pp ff
0 10
0 11
-1 12
ITEM: ATOMS id type x y z vx vy vz
1 3 0.0 0.1 0.2 1.0 2.0 3.0
2 1 0.8 0.1 0.2 4.0 5.0 6.0
3 2 2.0 2.1 2.2 7.0 8.0 9.0
ITEM: TIMESTEP
20
ITEM: NUMBER OF ATOMS
3
ITEM: BOX BOUNDS pp pp ff
0 10
0 11
-1 12
ITEM: ATOMS id type x y z vx vy vz
1 3 0.2 0.3 0.4 -1.0 -2.0 -3.0
2 1 0.9 0.3 0.4 -4.0 -5.0 -6.0
3 2 2.2 2.3 2.4 -7.0 -8.0 -9.0
"""
        type_map = {1: "H", 2: "Mg", 3: "O"}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tiny.lammpstrj"
            path.write_text(content, encoding="utf-8")
            python_frames = list(lammpstrj.read_lammpstrj(path, type_map=type_map, reader="python"))
            cpp_frames = list(lammpstrj.read_lammpstrj(path, type_map=type_map, reader="cpp"))

        self.assertEqual(len(cpp_frames), len(python_frames))
        for actual, expected in zip(cpp_frames, python_frames):
            self.assertEqual(actual.symbols, expected.symbols)
            self.assertEqual(actual.cell, expected.cell)
            self.assertEqual(actual.step, expected.step)
            np.testing.assert_array_equal(actual.types, expected.types)
            np.testing.assert_allclose(actual.positions, expected.positions)
            self.assertIsNotNone(actual.velocities)
            self.assertIsNotNone(expected.velocities)
            np.testing.assert_allclose(actual.velocities, expected.velocities)

    def test_npz_reader_reuses_symbols_for_fixed_topology(self):
        npz = importlib.import_module("waterint._00_io.npz")

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fixed_topology.npz"
            positions = np.asarray(
                [
                    [[0.0, 0.1, 0.2], [1.0, 1.1, 1.2]],
                    [[0.2, 0.3, 0.4], [1.2, 1.3, 1.4]],
                ],
                dtype=float,
            )
            types = np.asarray([[1, 2], [1, 2]], dtype=int)
            cells = np.asarray([[10.0, 10.0, 10.0], [10.0, 10.0, 10.0]], dtype=float)
            steps = np.asarray([10, 20], dtype=int)
            np_module = importlib.import_module("numpy")
            np_module.savez(path, positions=positions, types=types, cells=cells, steps=steps)

            frames = list(npz.read_npz(path, type_map={1: "H", 2: "O"}))

        self.assertEqual(frames[0].symbols, ["H", "O"])
        self.assertIs(frames[0].symbols, frames[1].symbols)
        np.testing.assert_allclose(frames[1].positions, positions[1])

    def test_npz_conversion_preserves_lammpstrj_velocities(self):
        npz = importlib.import_module("waterint._00_io.npz")

        content = """ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
2
ITEM: BOX BOUNDS pp pp pp
0 10
0 10
0 10
ITEM: ATOMS id type x y z vx vy vz
1 3 0.0 0.0 0.0 1.0 2.0 3.0
2 1 1.0 0.0 0.0 4.0 5.0 6.0
ITEM: TIMESTEP
1
ITEM: NUMBER OF ATOMS
2
ITEM: BOX BOUNDS pp pp pp
0 10
0 10
0 10
ITEM: ATOMS id type x y z vx vy vz
1 3 0.1 0.0 0.0 -1.0 -2.0 -3.0
2 1 1.1 0.0 0.0 -4.0 -5.0 -6.0
"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dump_path = root / "input.lammpstrj"
            npz_path = root / "input.npz"
            dump_path.write_text(content, encoding="utf-8")
            npz.write_npz_from_lammpstrj(
                trajectory_path=dump_path,
                output_path=npz_path,
                type_map={1: "H", 3: "O"},
            )
            frames = list(npz.read_npz(npz_path, type_map={1: "H", 3: "O"}))

        self.assertEqual(len(frames), 2)
        np.testing.assert_allclose(frames[0].velocities, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        np.testing.assert_allclose(frames[1].velocities, [[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0]])


if __name__ == "__main__":
    unittest.main()
