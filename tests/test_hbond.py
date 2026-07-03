from pathlib import Path
import shutil
import unittest

from waterint.config import load_config
from waterint.hbond import run_hbond


class HbondTests(unittest.TestCase):
    def test_hbond_example_classifies_h2o_topologies(self):
        root = Path(__file__).resolve().parents[1]
        output = root / "examples/hbond_lammpstrj/output"
        if output.exists():
            shutil.rmtree(output)

        result = run_hbond(load_config(root / "examples/hbond_lammpstrj/config.yaml"))

        self.assertEqual(result.frames, 1)
        self.assertEqual(result.samples_total["H2O"], 3)
        self.assertEqual(result.counts["H2O"]["DDAA"], 1)
        self.assertEqual(result.counts["H2O"]["DA"], 2)
        self.assertEqual(result.counts["H2O"]["DDA"], 0)
        self.assertEqual(result.counts["H2O"]["DAA"], 0)
        self.assertEqual(result.counts["H2O"]["AA"], 0)
        self.assertEqual(result.counts["H2O"]["A"], 0)
        self.assertEqual(result.counts["H2O"]["other"], 0)
        self.assertAlmostEqual(result.fractions["H2O"]["DDAA"], 1.0 / 3.0)
        self.assertAlmostEqual(result.fractions["H2O"]["DA"], 2.0 / 3.0)
        self.assertEqual(result.raw_counts["H2O"]["DDAA"], 1)
        self.assertEqual(result.raw_counts["H2O"]["DA"], 2)
        self.assertTrue(result.csv_path.exists())
        self.assertTrue(result.raw_csv_path.exists())
        self.assertTrue(result.png_path.exists())
        self.assertTrue(result.metadata_path.exists())

    def test_hbond_default_classes_are_species_specific(self):
        root = Path(__file__).resolve().parents[1]
        config = load_config(root / "examples/hbond_lammpstrj/config.yaml")
        config["selection"]["oxygen_species"] = ["OH-", "H2O", "H3O+"]
        config["output"]["plot"] = False

        result = run_hbond(config)

        self.assertIn("DAAA", result.classes["OH-"])
        self.assertIn("AAA", result.classes["OH-"])
        self.assertNotIn("DDAA", result.classes["OH-"])
        self.assertIn("DDAA", result.classes["H2O"])
        self.assertNotIn("DDAA", result.classes["H3O+"])
        self.assertNotIn("DAA", result.classes["H3O+"])
        self.assertIn("DDDA", result.classes["H3O+"])
        self.assertIn("DDD", result.classes["H3O+"])


if __name__ == "__main__":
    unittest.main()
