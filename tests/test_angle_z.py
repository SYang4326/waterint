from pathlib import Path
import shutil
import unittest

from waterint.angle_z import run_angle_z
from waterint.config import load_config


class AngleZTests(unittest.TestCase):
    def test_angle_z_example_writes_species_outputs(self):
        root = Path(__file__).resolve().parents[1]
        output = root / "examples/angle_z_lammpstrj/output"
        if output.exists():
            shutil.rmtree(output)

        result = run_angle_z(load_config(root / "examples/angle_z_lammpstrj/config.yaml"))

        self.assertEqual(result.frames, 2)
        self.assertEqual(result.bond_counts_total["OH-"], 2)
        self.assertEqual(result.bond_counts_total["H2O"], 8)
        self.assertEqual(result.bond_counts_total["H3O+"], 6)
        for label in ["OH-", "H2O", "H3O+"]:
            self.assertTrue(result.csv_paths[label].exists())
            self.assertTrue(result.png_paths[label].exists())
            self.assertGreater(float(result.histograms[label].sum()), 0.0)
        self.assertTrue(result.metadata_path.exists())

    def test_angle_z_bisector_counts_one_vector_per_oxygen(self):
        root = Path(__file__).resolve().parents[1]
        config = load_config(root / "examples/angle_z_lammpstrj/config.yaml")
        config["angle"]["vector_mode"] = "oh_bisector"
        config["output"]["directory"] = str(root / "examples/angle_z_lammpstrj/output_bisector")
        config["output"]["prefix"] = "angle_z_bisector"
        output = Path(config["output"]["directory"])
        if output.exists():
            shutil.rmtree(output)

        result = run_angle_z(config)

        self.assertEqual(result.bond_counts_total["OH-"], 2)
        self.assertEqual(result.bond_counts_total["H2O"], 8)
        self.assertEqual(result.bond_counts_total["H3O+"], 6)
        self.assertEqual(result.sample_counts_total["OH-"], 2)
        self.assertEqual(result.sample_counts_total["H2O"], 4)
        self.assertEqual(result.sample_counts_total["H3O+"], 2)
        self.assertEqual(float(result.histograms["H2O"].sum()), 2.0)


if __name__ == "__main__":
    unittest.main()
