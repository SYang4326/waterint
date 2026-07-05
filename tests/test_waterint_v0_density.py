from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from waterint.config import load_config
from waterint.density import run_density as run_density_legacy
from waterint_v0.workflows.density import run_density


class WaterintV0DensityTests(unittest.TestCase):
    def test_density_example_matches_legacy_workflow(self):
        root = Path(__file__).resolve().parents[1]
        with TemporaryDirectory() as tmpdir:
            config = load_config(root / "examples/density_xyz/config.yaml")
            config["output"]["directory"] = str(Path(tmpdir) / "v0")
            config["output"]["prefix"] = "density_water_O_v0"

            legacy_config = load_config(root / "examples/density_xyz/config.yaml")
            legacy_config["output"]["directory"] = str(Path(tmpdir) / "legacy")
            legacy_config["output"]["prefix"] = "density_water_O_v0_legacy"

            result = run_density(config)
            legacy = run_density_legacy(legacy_config)

            self.assertEqual(result.frames, legacy.frames)
            self.assertEqual(result.selected_atoms_total, legacy.selected_atoms_total)
            np.testing.assert_allclose(result.bin_centers, legacy.bin_centers)
            np.testing.assert_allclose(
                result.profiles["water_O"]["density"],
                legacy.profiles["water_O"]["density"],
            )
            self.assertTrue(result.csv_path.exists())
            self.assertTrue(result.png_path.exists())
            self.assertTrue(result.metadata_path.exists())

    def test_oxygen_species_density_matches_legacy_workflow(self):
        root = Path(__file__).resolve().parents[1]
        with TemporaryDirectory() as tmpdir:
            config = load_config(root / "examples/density_xyz/config_oxygen_species.yaml")
            config["output"]["directory"] = str(Path(tmpdir) / "v0")
            config["output"]["prefix"] = "density_oxygen_species_v0"

            legacy_config = load_config(root / "examples/density_xyz/config_oxygen_species.yaml")
            legacy_config["output"]["directory"] = str(Path(tmpdir) / "legacy")
            legacy_config["output"]["prefix"] = "density_oxygen_species_v0_legacy"

            result = run_density(config)
            legacy = run_density_legacy(legacy_config)

            self.assertEqual(result.frames, legacy.frames)
            self.assertEqual(result.selected_atoms_total, legacy.selected_atoms_total)
            self.assertEqual(set(result.profiles), set(legacy.profiles))
            for label in result.profiles:
                np.testing.assert_allclose(
                    result.profiles[label]["density"],
                    legacy.profiles[label]["density"],
                )


if __name__ == "__main__":
    unittest.main()
