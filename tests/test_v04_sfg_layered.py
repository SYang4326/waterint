from __future__ import annotations

import importlib
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
V04 = ROOT / "v0.4"


class V04LayeredSfgTests(unittest.TestCase):
    def setUp(self) -> None:
        self._old_path = list(sys.path)
        self._old_modules = {
            name: module
            for name, module in sys.modules.items()
            if name == "waterint" or name.startswith("waterint.")
        }
        for name in list(self._old_modules):
            sys.modules.pop(name, None)
        sys.path.insert(0, str(V04))

    def tearDown(self) -> None:
        for name in [name for name in sys.modules if name == "waterint" or name.startswith("waterint.")]:
            sys.modules.pop(name, None)
        sys.modules.update(self._old_modules)
        sys.path[:] = self._old_path

    @staticmethod
    def _frames() -> list:
        trajectory_frame = importlib.import_module("waterint._00_io.common").TrajectoryFrame
        symbols = ["O", "O", "H", "H", "H"]
        positions = np.asarray(
            [[0.0, 0.0, 1.0], [5.0, 0.0, 3.0], [0.0, 0.0, 1.9], [5.0, 0.0, 3.9], [5.0, 0.0, 2.1]],
            dtype=float,
        )
        velocities = np.asarray(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 2.0], [0.0, 0.0, 3.0]],
            dtype=float,
        )
        return [
            trajectory_frame(
                index=index,
                comment="",
                symbols=symbols,
                positions=positions.copy(),
                cell=(20.0, 20.0, 20.0),
                step=index,
                velocities=velocities.copy(),
            )
            for index in range(5)
        ]

    def test_layered_channels_reuse_assignment_and_masks(self) -> None:
        sfg = importlib.import_module("waterint._02_computation.sfg")
        native = importlib.import_module("waterint._02_computation._native")
        selection_context = importlib.import_module("waterint._01_core.selection").SelectionContext
        if not native.native_status()["available"]:
            self.skipTest("C++ compiler or native backend is not available.")
        cfg = {
            "dt_ps": 0.1,
            "lag_ps": 0.2,
            "velocity_source": "trajectory",
            "trajectory_velocity_unit": "A/ps",
            "oh_assignment": "nearest_oxygen",
            "oh_cutoff": 1.25,
            "z_ref0": 0.0,
            "pbc": [False, False, False],
            "layer_bins": [
                {"label": "low", "window": {"mode": 2, "z1": 0.0, "z2": 2.0, "ramp": 0.0}},
                {"label": "high", "window": {"mode": 2, "z1": 2.0, "z2": 4.0, "ramp": 0.0}},
                {"label": "all"},
            ],
            "species_channels": ["OH-"],
            "symmetrize": True,
        }
        results = {}
        for backend in ["python", "cpp"]:
            results[backend] = sfg.compute_layered_ssvvcf_from_frames(
                self._frames(),
                cell=(20.0, 20.0, 20.0),
                sfg_cfg={**cfg, "backend": backend},
                context=selection_context({}),
            )
        result = results["python"]
        self.assertEqual(set(result.channels), {"low:all", "low:nh1", "high:all", "high:nh1", "all:all", "all:nh1"})
        self.assertEqual(result.channels["low:all"].counts[0], 10)
        self.assertEqual(result.channels["high:all"].counts[0], 20)
        self.assertEqual(result.channels["low:nh1"].counts[0], 10)
        self.assertEqual(result.channels["high:nh1"].counts[0], 20)
        self.assertEqual(result.channels["all:all"].counts[0], 30)
        self.assertEqual(result.channels["all:nh1"].counts[0], 30)
        self.assertEqual(result.channels["low:nh1"].selected_counts[0], 10)
        self.assertEqual(result.channels["high:nh1"].selected_counts[0], 0)
        self.assertEqual(result.channels["all:nh1"].selected_counts[0], 10)
        for channel_name in result.channels:
            np.testing.assert_array_equal(
                results["cpp"].channels[channel_name].counts,
                results["python"].channels[channel_name].counts,
            )
            np.testing.assert_allclose(
                results["cpp"].channels[channel_name].corr,
                results["python"].channels[channel_name].corr,
                rtol=1e-12,
                atol=1e-12,
            )

    def test_layered_config_validation(self) -> None:
        sfg = importlib.import_module("waterint._02_computation.sfg")
        with self.assertRaisesRegex(ValueError, "requires a window"):
            sfg.layered_channel_specs({"layer_bins": [{"label": "low"}]})
        with self.assertRaisesRegex(ValueError, "Unknown SFG oxygen species"):
            sfg.layered_channel_specs({"layer_bins": [{"label": "all"}], "species_channels": ["CO2"]})
        with self.assertRaisesRegex(ValueError, "all layer must omit"):
            sfg.layered_channel_specs({"layer_bins": [{"label": "all", "window": {}}]})

    def test_cpp_tracks_dynamic_nh1_membership(self) -> None:
        sfg = importlib.import_module("waterint._02_computation.sfg")
        native = importlib.import_module("waterint._02_computation._native")
        selection_context = importlib.import_module("waterint._01_core.selection").SelectionContext
        if not native.native_status()["available"]:
            self.skipTest("C++ compiler or native backend is not available.")
        cfg = {
            "dt_ps": 0.1,
            "lag_ps": 0.2,
            "velocity_source": "trajectory",
            "trajectory_velocity_unit": "A/ps",
            "oh_assignment": "nearest_oxygen",
            "oh_cutoff": 1.25,
            "z_ref0": 0.0,
            "pbc": [False, False, False],
            "layer_bins": [{"label": "all"}],
            "species_channels": "all",
            "symmetrize": True,
        }
        results = {}
        for backend in ["python", "cpp"]:
            results[backend] = sfg.compute_layered_ssvvcf_from_frames(
                self._dynamic_frames(),
                cell=(20.0, 20.0, 20.0),
                sfg_cfg={**cfg, "backend": backend},
                context=selection_context({}),
            )
        self.assertEqual(int(results["python"].channels["all:all"].counts[0]), 24)
        self.assertEqual(int(results["python"].channels["all:nh1"].counts[0]), 24)
        self.assertEqual(int(results["python"].channels["all:H2O"].counts[0]), 24)
        self.assertEqual(int(results["python"].channels["all:nh1"].selected_counts[0]), 12)
        self.assertEqual(int(results["python"].channels["all:H2O"].selected_counts[0]), 12)
        for channel_name in results["python"].channels:
            np.testing.assert_array_equal(
                results["cpp"].channels[channel_name].counts,
                results["python"].channels[channel_name].counts,
            )
            np.testing.assert_allclose(
                results["cpp"].channels[channel_name].corr,
                results["python"].channels[channel_name].corr,
                rtol=1e-12,
                atol=1e-12,
            )
        species_names = ["all:O2-", "all:nh1", "all:H2O", "all:H3O+", "all:O_other"]
        species_corr = sum(results["python"].channels[name].corr for name in species_names)
        np.testing.assert_allclose(
            species_corr,
            results["python"].channels["all:all"].corr,
            rtol=1e-12,
            atol=1e-12,
        )
        total_frequency, total_signal = sfg.compute_ft(
            results["python"].channels["all:all"].time_ps,
            results["python"].channels["all:all"].corr,
            nzeros=20,
        )
        species_signals = []
        for name in species_names:
            frequency, signal = sfg.compute_ft(
                results["python"].channels[name].time_ps,
                results["python"].channels[name].corr,
                nzeros=20,
            )
            np.testing.assert_array_equal(frequency, total_frequency)
            species_signals.append(signal)
        np.testing.assert_allclose(sum(species_signals), total_signal, rtol=1e-12, atol=1e-12)

        conditional = sfg.compute_layered_ssvvcf_from_frames(
            self._dynamic_frames(),
            cell=(20.0, 20.0, 20.0),
            sfg_cfg={**cfg, "backend": "python", "species_normalization": "conditional"},
            context=selection_context({}),
        )
        self.assertEqual(int(conditional.channels["all:nh1"].counts[0]), 12)

    @staticmethod
    def _dynamic_frames() -> list:
        trajectory_frame = importlib.import_module("waterint._00_io.common").TrajectoryFrame
        frames = []
        symbols = ["O", "O", "H", "H"]
        velocities = np.asarray(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 2.0]],
            dtype=float,
        )
        for index in range(6):
            hydrogen_1 = [5.0, 0.0, 3.9] if index < 3 else [0.0, 0.0, 1.1]
            positions = np.asarray(
                [[0.0, 0.0, 1.0], [5.0, 0.0, 3.0], [0.0, 0.0, 1.9], hydrogen_1],
                dtype=float,
            )
            frames.append(
                trajectory_frame(
                    index=index,
                    comment="",
                    symbols=symbols,
                    positions=positions,
                    cell=(20.0, 20.0, 20.0),
                    step=index,
                    velocities=velocities.copy(),
                )
            )
        return frames

    def test_layered_workflow_writes_combine_bins_compatible_names(self) -> None:
        workflow = importlib.import_module("waterint._04_workflows.workflows.sfg")
        frames = self._frames()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.savez(
                root / "trajectory.npz",
                positions=np.stack([frame.positions for frame in frames]),
                types=np.asarray([[3, 3, 1, 1, 1]] * len(frames)),
                cells=np.asarray([[20.0, 20.0, 20.0]] * len(frames)),
                steps=np.arange(len(frames)),
                velocities=np.stack([frame.velocities for frame in frames]),
            )
            config = {
                "_config_dir": str(root),
                "input": {"trajectory": "trajectory.npz", "format": "npz", "type_map": {1: "H", 3: "O"}},
                "system": {"cell": "auto"},
                "sfg": {
                    "mode": "trajectory",
                    "backend": "auto",
                    "dt_ps": 0.1,
                    "lag_ps": 0.2,
                    "velocity_source": "trajectory",
                    "oh_assignment": "nearest_oxygen",
                    "pbc": [False, False, False],
                    "layer_bins": [
                        {"label": "low", "window": {"mode": 2, "z1": 0.0, "z2": 2.0, "ramp": 0.0}},
                        {"label": "all"},
                    ],
                    "species_channels": ["OH-"],
                },
                "output": {"directory": "output", "prefix": "ssvvcf", "run_label": "test", "plot": False},
            }
            result = workflow.run_sfg(config)
            self.assertEqual(result.mode, "trajectory_layered")
            self.assertTrue((root / "output/ssvvcf_low_test.dat").exists())
            self.assertTrue((root / "output/ssvvcf_low_test_cf_nh1.dat").exists())

            combine_config = {
                "_config_dir": str(root),
                "sfg": {
                    "mode": "combine_bins",
                    "input_directory": "output",
                    "runs": ["test"],
                    "bins": ["low"],
                    "cf_prefix": "ssvvcf",
                    "include_nh1": True,
                },
                "output": {"directory": "combined", "prefix": "combined", "plot": False},
            }
            combined = workflow.run_sfg(combine_config)
            self.assertTrue(combined.cf_paths["low:all"].exists())
            self.assertTrue(combined.cf_paths["low:nh1"].exists())

    def test_combine_bins_checks_complete_species_partition(self) -> None:
        workflow = importlib.import_module("waterint._04_workflows.workflows.sfg")
        sfg = importlib.import_module("waterint._02_computation.sfg")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            time = np.asarray([0.0, 0.1, 0.2])
            total = np.asarray([4.0, 2.0, 1.0])
            parts = {
                "o2minus": np.asarray([0.0, 0.0, 0.0]),
                "nh1": np.asarray([1.0, 0.5, 0.25]),
                "h2o": np.asarray([3.0, 1.5, 0.75]),
                "h3oplus": np.asarray([0.0, 0.0, 0.0]),
                "oother": np.asarray([0.0, 0.0, 0.0]),
            }
            sfg.write_cf(root / "ssvvcf_all_test.dat", time, total, np.asarray([8, 6, 4]))
            for token, curve in parts.items():
                # Additive channels must retain the all-OH denominator.
                sfg.write_cf(root / f"ssvvcf_all_test_cf_{token}.dat", time, curve, np.asarray([8, 6, 4]))
            config = {
                "_config_dir": str(root),
                "sfg": {
                    "mode": "combine_bins",
                    "input_directory": ".",
                    "runs": ["test"],
                    "bins": ["all"],
                    "cf_prefix": "ssvvcf",
                    "species_channels": "all",
                    "nzeros": 20,
                },
                "output": {"directory": "combined", "plot": False},
            }
            result = workflow.run_sfg(config)
            self.assertIn("all:h2o", result.cf_paths)
            self.assertIn("all:nh1", result.ft_paths)

            sfg.write_cf(root / "ssvvcf_all_test_cf_h2o.dat", time, parts["h2o"] * 1.1, np.asarray([8, 6, 4]))
            with self.assertRaisesRegex(ValueError, "do not combine conditional"):
                workflow.run_sfg({**config, "output": {"directory": "invalid", "plot": False}})


if __name__ == "__main__":
    unittest.main()
