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
        selection_context = importlib.import_module("waterint._01_core.selection").SelectionContext
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
        result = sfg.compute_layered_ssvvcf_from_frames(
            self._frames(), cell=(20.0, 20.0, 20.0), sfg_cfg=cfg, context=selection_context({})
        )
        self.assertEqual(set(result.channels), {"low:all", "low:nh1", "high:all", "high:nh1", "all:all", "all:nh1"})
        self.assertEqual(result.channels["low:all"].counts[0], 10)
        self.assertEqual(result.channels["high:all"].counts[0], 20)
        self.assertEqual(result.channels["low:nh1"].counts[0], 10)
        self.assertEqual(result.channels["high:nh1"].counts[0], 0)
        self.assertEqual(result.channels["all:all"].counts[0], 30)
        self.assertEqual(result.channels["all:nh1"].counts[0], 10)

    def test_layered_config_validation(self) -> None:
        sfg = importlib.import_module("waterint._02_computation.sfg")
        with self.assertRaisesRegex(ValueError, "requires a window"):
            sfg.layered_channel_specs({"layer_bins": [{"label": "low"}]})
        with self.assertRaisesRegex(ValueError, "Unknown SFG oxygen species"):
            sfg.layered_channel_specs({"layer_bins": [{"label": "all"}], "species_channels": ["CO2"]})
        with self.assertRaisesRegex(ValueError, "all layer must omit"):
            sfg.layered_channel_specs({"layer_bins": [{"label": "all", "window": {}}]})

    def test_layered_cpp_backend_is_rejected_clearly(self) -> None:
        workflow = importlib.import_module("waterint._04_workflows.workflows.sfg")
        with self.assertRaisesRegex(ValueError, "requires the Python backend"):
            workflow.run_layered_trajectory(
                config={},
                input_cfg={},
                frames=[],
                cell=(20.0, 20.0, 20.0),
                sfg_cfg={"backend": "cpp"},
                output_cfg={},
                outdir=Path("."),
                prefix="test",
            )

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


if __name__ == "__main__":
    unittest.main()
