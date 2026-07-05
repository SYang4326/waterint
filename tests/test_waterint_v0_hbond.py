from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from waterint.config import load_config
from waterint.hbond import run_hbond as run_hbond_legacy
from waterint_v0.workflows.hbond import run_hbond


class WaterintV0HbondTests(unittest.TestCase):
    def test_hbond_matches_legacy_workflow(self):
        root = Path(__file__).resolve().parents[1]
        with TemporaryDirectory() as tmpdir:
            config = load_config(root / "examples/hbond_lammpstrj/config.yaml")
            config["output"]["directory"] = str(Path(tmpdir) / "v0")
            config["output"]["prefix"] = "hbond_v0"

            legacy_config = load_config(root / "examples/hbond_lammpstrj/config.yaml")
            legacy_config["output"]["directory"] = str(Path(tmpdir) / "legacy")
            legacy_config["output"]["prefix"] = "hbond_v0_legacy"

            result = run_hbond(config)
            legacy = run_hbond_legacy(legacy_config)

            self.assertEqual(result.frames, legacy.frames)
            self.assertEqual(result.classes, legacy.classes)
            self.assertEqual(result.counts, legacy.counts)
            self.assertEqual(result.raw_counts, legacy.raw_counts)
            self.assertEqual(result.samples_total, legacy.samples_total)
            self.assertEqual(result.fractions, legacy.fractions)
            self.assertTrue(result.csv_path.exists())
            self.assertTrue(result.raw_csv_path.exists())
            self.assertTrue(result.png_path.exists())
            self.assertTrue(result.metadata_path.exists())

    def test_hbond_species_specific_classes_match_legacy(self):
        root = Path(__file__).resolve().parents[1]
        with TemporaryDirectory() as tmpdir:
            config = load_config(root / "examples/hbond_lammpstrj/config.yaml")
            config["selection"]["oxygen_species"] = ["OH-", "H2O", "H3O+"]
            config["output"]["plot"] = False
            config["output"]["directory"] = str(Path(tmpdir) / "v0")

            legacy_config = load_config(root / "examples/hbond_lammpstrj/config.yaml")
            legacy_config["selection"]["oxygen_species"] = ["OH-", "H2O", "H3O+"]
            legacy_config["output"]["plot"] = False
            legacy_config["output"]["directory"] = str(Path(tmpdir) / "legacy")

            self.assertEqual(run_hbond(config).classes, run_hbond_legacy(legacy_config).classes)


if __name__ == "__main__":
    unittest.main()
