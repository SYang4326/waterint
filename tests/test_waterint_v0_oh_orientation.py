from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from waterint.angle_z import run_angle_z as run_angle_z_legacy
from waterint.config import load_config
from waterint_v0.workflows.oh_orientation import run_oh_orientation


class WaterintV0OhOrientationTests(unittest.TestCase):
    def test_angle_z_matches_legacy_workflow(self):
        root = Path(__file__).resolve().parents[1]
        with TemporaryDirectory() as tmpdir:
            config = load_config(root / "examples/angle_z_lammpstrj/config.yaml")
            config["output"]["directory"] = str(Path(tmpdir) / "v0")
            config["output"]["prefix"] = "angle_z_v0"

            legacy_config = load_config(root / "examples/angle_z_lammpstrj/config.yaml")
            legacy_config["output"]["directory"] = str(Path(tmpdir) / "legacy")
            legacy_config["output"]["prefix"] = "angle_z_v0_legacy"

            result = run_oh_orientation(config)
            legacy = run_angle_z_legacy(legacy_config)

            self.assertEqual(result.frames, legacy.frames)
            self.assertEqual(result.bond_counts_total, legacy.bond_counts_total)
            self.assertEqual(result.sample_counts_total, legacy.sample_counts_total)
            np.testing.assert_allclose(result.z_centers, legacy.z_centers)
            np.testing.assert_allclose(result.angle_centers, legacy.angle_centers)
            for label in result.histograms:
                np.testing.assert_allclose(result.histograms[label], legacy.histograms[label])
                self.assertTrue(result.csv_paths[label].exists())
                self.assertTrue(result.png_paths[label].exists())
            self.assertTrue(result.metadata_path.exists())

    def test_angle_z_bisector_matches_legacy_workflow(self):
        root = Path(__file__).resolve().parents[1]
        with TemporaryDirectory() as tmpdir:
            config = load_config(root / "examples/angle_z_lammpstrj/config.yaml")
            config["angle"]["vector_mode"] = "oh_bisector"
            config["output"]["directory"] = str(Path(tmpdir) / "v0")
            config["output"]["prefix"] = "angle_z_v0_bisector"

            legacy_config = load_config(root / "examples/angle_z_lammpstrj/config.yaml")
            legacy_config["angle"]["vector_mode"] = "oh_bisector"
            legacy_config["output"]["directory"] = str(Path(tmpdir) / "legacy")
            legacy_config["output"]["prefix"] = "angle_z_v0_bisector_legacy"

            result = run_oh_orientation(config)
            legacy = run_angle_z_legacy(legacy_config)

            self.assertEqual(result.bond_counts_total, legacy.bond_counts_total)
            self.assertEqual(result.sample_counts_total, legacy.sample_counts_total)
            for label in result.histograms:
                np.testing.assert_allclose(result.histograms[label], legacy.histograms[label])


if __name__ == "__main__":
    unittest.main()
