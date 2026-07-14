from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
V03 = ROOT / "v0.3"


class V03SfgCppTests(unittest.TestCase):
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

    def test_cpp_matches_python_for_full_and_stretch_modes(self):
        for mode in ["full", "stretch"]:
            for symmetrize in [False, True]:
                with self.subTest(mode=mode, symmetrize=symmetrize):
                    python, cpp = self._run_both(
                        mu_mode=mode,
                        symmetrize=symmetrize,
                        flip_sign=True,
                        window={"mode": 1, "z1": 0.0, "z2": 3.0, "ramp": 0.5, "flip": True},
                    )
                    self._assert_results_equal(python, cpp)

    def test_cpp_matches_python_for_top_hat_window_and_segment_switch(self):
        python, cpp = self._run_both(
            mu_mode="full",
            symmetrize=True,
            flip_sign=False,
            window={"mode": 2, "z1": -0.2, "z2": 1.5, "ramp": 0.3, "flip": False},
        )
        self._assert_results_equal(python, cpp)

    def test_cpp_duplicate_error_matches_python(self):
        sfg = importlib.import_module("waterint._02_computation.sfg")
        native = importlib.import_module("waterint._02_computation._native")
        if not native.native_status()["available"]:
            self.skipTest("C++ compiler or native backend is not available.")
        frames, context = self._duplicate_frames()
        config = self._config(backend="cpp", window=None)
        config["oh_cutoff"] = 1.1
        config["duplicate_hydrogen_policy"] = "error"
        with self.assertRaises(ValueError):
            sfg.compute_ssvvcf_from_frames(frames, cell=(10.0, 10.0, 10.0), sfg_cfg=config, context=context)
        config["backend"] = "python"
        with self.assertRaises(ValueError):
            sfg.compute_ssvvcf_from_frames(frames, cell=(10.0, 10.0, 10.0), sfg_cfg=config, context=context)

    def test_cpp_rejects_variable_topology_and_auto_falls_back(self):
        sfg = importlib.import_module("waterint._02_computation.sfg")
        frames, context = self._frames()
        frames[2] = type(frames[2])(
            index=frames[2].index,
            comment=frames[2].comment,
            symbols=["O", "H", "H", "H"],
            positions=frames[2].positions,
            cell=frames[2].cell,
            step=frames[2].step,
            types=np.asarray([3, 1, 1, 1], dtype=int),
        )
        config = self._config(backend="cpp", window=None)
        with self.assertRaises(ValueError):
            sfg.compute_ssvvcf_from_frames(frames, cell=(10.0, 10.0, 10.0), sfg_cfg=config, context=context)
        config["backend"] = "auto"
        result = sfg.compute_ssvvcf_from_frames(
            frames,
            cell=(10.0, 10.0, 10.0),
            sfg_cfg=config,
            context=context,
        )
        self.assertEqual(result.frames, len(frames))

    def test_cpp_rejects_changed_types_when_symbols_object_is_shared(self):
        sfg = importlib.import_module("waterint._02_computation.sfg")
        frames, context = self._frames()
        shared_symbols = frames[0].symbols
        changed_types = frames[2].types.copy()
        changed_types[0] = 1
        frames[2] = type(frames[2])(
            index=frames[2].index,
            comment=frames[2].comment,
            symbols=shared_symbols,
            positions=frames[2].positions,
            cell=frames[2].cell,
            step=frames[2].step,
            types=changed_types,
        )
        self.assertTrue(all(frame.symbols is shared_symbols for frame in frames))
        config = self._config(backend="cpp", window=None)
        with self.assertRaises(ValueError):
            sfg.compute_ssvvcf_from_frames(
                frames,
                cell=(10.0, 10.0, 10.0),
                sfg_cfg=config,
                context=context,
            )

    def test_trajectory_velocities_override_finite_difference_in_both_backends(self):
        sfg = importlib.import_module("waterint._02_computation.sfg")
        frames, context = self._velocity_frames(velocity_scale=1.0)
        results = {}
        for backend in ["python", "cpp"]:
            config = self._config(backend=backend, window=None)
            config.update({"velocity_source": "auto", "trajectory_velocity_unit": "A/ps", "symmetrize": False})
            results[backend] = sfg.compute_ssvvcf_from_frames(
                frames,
                cell=(10.0, 10.0, 10.0),
                sfg_cfg=config,
                context=context,
            )
        self._assert_results_equal(results["python"], results["cpp"])
        self.assertEqual(results["python"].velocity_source, "trajectory")
        self.assertGreater(abs(results["python"].corr[0]), 1.0)

        finite_difference_config = self._config(backend="python", window=None)
        finite_difference_config.update({"velocity_source": "finite_difference", "symmetrize": False})
        finite_difference = sfg.compute_ssvvcf_from_frames(
            frames,
            cell=(10.0, 10.0, 10.0),
            sfg_cfg=finite_difference_config,
            context=context,
        )
        self.assertEqual(finite_difference.velocity_source, "finite_difference")
        np.testing.assert_allclose(finite_difference.corr, 0.0)

    def test_trajectory_velocity_afs_unit_converts_to_internal_aps(self):
        sfg = importlib.import_module("waterint._02_computation.sfg")
        frames, context = self._velocity_frames(velocity_scale=0.001)
        config = self._config(backend="python", window=None)
        config.update({"velocity_source": "trajectory", "trajectory_velocity_unit": "A/fs", "symmetrize": False})
        result = sfg.compute_ssvvcf_from_frames(
            frames,
            cell=(10.0, 10.0, 10.0),
            sfg_cfg=config,
            context=context,
        )
        self.assertEqual(result.velocity_source, "trajectory")
        self.assertAlmostEqual(result.corr[0], 6.0)

    def test_trajectory_velocity_source_requires_velocities(self):
        sfg = importlib.import_module("waterint._02_computation.sfg")
        frames, context = self._frames()
        config = self._config(backend="python", window=None)
        config["velocity_source"] = "trajectory"
        with self.assertRaisesRegex(ValueError, "requires vx, vy, and vz"):
            sfg.compute_ssvvcf_from_frames(
                frames,
                cell=(10.0, 10.0, 10.0),
                sfg_cfg=config,
                context=context,
            )

    def test_nearest_oxygen_assigns_hydrogen_outside_cutoff_in_both_backends(self):
        sfg = importlib.import_module("waterint._02_computation.sfg")
        native = importlib.import_module("waterint._02_computation._native")
        if not native.native_status()["available"]:
            self.skipTest("C++ compiler or native backend is not available.")
        frames, context = self._out_of_cutoff_frames()

        cutoff_config = self._config(backend="python", window=None)
        cutoff_config.update({"oh_cutoff": 1.25, "oh_assignment": "cutoff", "symmetrize": False})
        cutoff_result = sfg.compute_ssvvcf_from_frames(
            frames, cell=(10.0, 10.0, 10.0), sfg_cfg=cutoff_config, context=context
        )
        self.assertEqual(int(cutoff_result.counts[0]), 0)

        results = {}
        for backend in ["python", "cpp"]:
            config = self._config(backend=backend, window=None)
            config.update({"oh_cutoff": 1.25, "oh_assignment": "nearest_oxygen", "symmetrize": False})
            results[backend] = sfg.compute_ssvvcf_from_frames(
                frames, cell=(10.0, 10.0, 10.0), sfg_cfg=config, context=context
            )
        self.assertGreater(int(results["python"].counts[0]), 0)
        self._assert_results_equal(results["python"], results["cpp"])

    def test_top_layer_mean_reference_averages_surface_layer(self):
        sfg = importlib.import_module("waterint._02_computation.sfg")
        common = importlib.import_module("waterint._00_io.common")
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
        zrefs = sfg.zref_series(
            [frame],
            {"reference": {"type": "top_layer_mean", "species": ["Mg"], "surface": "max", "layer_width": 0.7}},
            context,
        )
        np.testing.assert_allclose(zrefs, [9.8])

    def _run_both(self, *, mu_mode, symmetrize, flip_sign, window):
        sfg = importlib.import_module("waterint._02_computation.sfg")
        native = importlib.import_module("waterint._02_computation._native")
        if not native.native_status()["available"]:
            self.skipTest("C++ compiler or native backend is not available.")
        frames, context = self._frames()
        results = {}
        for backend in ["python", "cpp"]:
            config = self._config(backend=backend, window=window)
            config.update(
                {
                    "mu_mode": mu_mode,
                    "symmetrize": symmetrize,
                    "flip_sign": flip_sign,
                }
            )
            results[backend] = sfg.compute_ssvvcf_from_frames(
                frames,
                cell=(10.0, 10.0, 10.0),
                sfg_cfg=config,
                context=context,
            )
        return results["python"], results["cpp"]

    def _assert_results_equal(self, python, cpp):
        self.assertEqual(cpp.frames, python.frames)
        np.testing.assert_array_equal(cpp.time_ps, python.time_ps)
        np.testing.assert_array_equal(cpp.counts, python.counts)
        np.testing.assert_array_equal(cpp.zrefs, python.zrefs)
        np.testing.assert_allclose(cpp.corr, python.corr, rtol=1e-12, atol=1e-12)

    @staticmethod
    def _config(*, backend: str, window):
        config = {
            "backend": backend,
            "hydrogen_symbol": "H",
            "oxygen_symbol": "O",
            "oh_cutoff": 1.25,
            "neighbor_method": "matrix",
            "duplicate_hydrogen_policy": "nearest",
            "dt_ps": 0.1,
            "lag_ps": 0.4,
            "pbc": [True, True, False],
            "z_ref0": 0.0,
            "mu_mode": "full",
            "symmetrize": True,
            "flip_sign": False,
        }
        if window is not None:
            config["window"] = window
        return config

    @staticmethod
    def _frames():
        common = importlib.import_module("waterint._00_io.common")
        selection = importlib.import_module("waterint._01_core.selection")
        symbols = ["O", "O", "H", "H"]
        types = np.asarray([3, 3, 1, 1], dtype=int)
        frames = []
        for index in range(8):
            oxygen_0 = np.asarray([9.7 + 0.01 * index, 0.0, 0.5])
            oxygen_1 = np.asarray([2.8, 0.0, 1.0])
            hydrogen_0 = oxygen_0 + np.asarray([0.7 + 0.02 * np.sin(index), 0.0, 0.1 * index])
            if index < 4:
                hydrogen_1 = oxygen_1 + np.asarray([0.75 + 0.02 * index, 0.0, -0.05 * index])
            else:
                hydrogen_1 = oxygen_0 + np.asarray([-0.75, 0.0, 0.02 * index])
            positions = np.vstack((oxygen_0, oxygen_1, hydrogen_0, hydrogen_1))
            frames.append(
                common.TrajectoryFrame(
                    index=index,
                    comment=f"frame {index}",
                    symbols=symbols,
                    positions=positions,
                    cell=(10.0, 10.0, 10.0),
                    step=index,
                    types=types,
                )
            )
        return frames, selection.SelectionContext.from_input_config({"type_map": {1: "H", 3: "O"}})

    @staticmethod
    def _duplicate_frames():
        common = importlib.import_module("waterint._00_io.common")
        selection = importlib.import_module("waterint._01_core.selection")
        positions = np.asarray([[0.0, 0.0, 0.0], [1.6, 0.0, 0.0], [0.8, 0.0, 0.0]])
        symbols = ["O", "O", "H"]
        types = np.asarray([3, 3, 1], dtype=int)
        frames = [
            common.TrajectoryFrame(
                index=index,
                comment=f"frame {index}",
                symbols=symbols,
                positions=positions.copy(),
                cell=(10.0, 10.0, 10.0),
                step=index,
                types=types,
            )
            for index in range(3)
        ]
        return frames, selection.SelectionContext.from_input_config({"type_map": {1: "H", 3: "O"}})

    @staticmethod
    def _out_of_cutoff_frames():
        common = importlib.import_module("waterint._00_io.common")
        selection = importlib.import_module("waterint._01_core.selection")
        symbols = ["O", "O", "H"]
        types = np.asarray([3, 3, 1], dtype=int)
        positions = np.asarray([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [1.6, 0.0, 0.0]])
        frames = [
            common.TrajectoryFrame(
                index=index,
                comment=f"frame {index}",
                symbols=symbols,
                positions=positions.copy(),
                cell=(10.0, 10.0, 10.0),
                step=index,
                types=types,
            )
            for index in range(4)
        ]
        return frames, selection.SelectionContext.from_input_config({"type_map": {1: "H", 3: "O"}})

    @staticmethod
    def _velocity_frames(*, velocity_scale: float):
        common = importlib.import_module("waterint._00_io.common")
        selection = importlib.import_module("waterint._01_core.selection")
        symbols = ["O", "H"]
        types = np.asarray([3, 1], dtype=int)
        positions = np.asarray([[0.0, 0.0, 0.5], [1.0, 0.0, 0.5]])
        velocities = np.asarray([[0.0, 0.0, 0.0], [2.0 * velocity_scale, 0.0, 3.0 * velocity_scale]])
        frames = [
            common.TrajectoryFrame(
                index=index,
                comment=f"frame {index}",
                symbols=symbols,
                positions=positions.copy(),
                cell=(10.0, 10.0, 10.0),
                step=index,
                types=types,
                velocities=velocities.copy(),
            )
            for index in range(3)
        ]
        return frames, selection.SelectionContext.from_input_config({"type_map": {1: "H", 3: "O"}})


if __name__ == "__main__":
    unittest.main()
