from pathlib import Path
import shutil
import unittest

from waterint.config import load_config
from waterint.density import run_density


class DensityTests(unittest.TestCase):
    def test_density_example_writes_outputs(self):
        root = Path(__file__).resolve().parents[1]
        output = root / "examples/density_xyz/output"
        if output.exists():
            shutil.rmtree(output)

        result = run_density(load_config(root / "examples/density_xyz/config.yaml"))

        self.assertEqual(result.frames, 2)
        self.assertEqual(result.selected_atoms_total["water_O"], 4)
        self.assertTrue(result.csv_path.exists())
        self.assertTrue(result.png_path.exists())
        self.assertTrue(result.metadata_path.exists())

    def test_oxygen_species_density_writes_multiple_profiles(self):
        root = Path(__file__).resolve().parents[1]
        result = run_density(load_config(root / "examples/density_xyz/config_oxygen_species.yaml"))

        self.assertEqual(result.frames, 2)
        self.assertIn("H2O", result.profiles)
        self.assertIn("O2-", result.profiles)
        self.assertEqual(result.selected_atoms_total["H2O"], 4)
        self.assertEqual(result.selected_atoms_total["O2-"], 0)
        text = result.csv_path.read_text()
        self.assertIn("H2O_density", text)
        self.assertIn("OH-_density", text)

    def test_signed_axis_is_supported(self):
        root = Path(__file__).resolve().parents[1]
        config = load_config(root / "examples/density_xyz/config.yaml")
        config["coordinate"]["axis"] = "-z"
        config["coordinate"]["range"] = [-10.0, 0.0]
        config["output"]["prefix"] = "density_water_O_minus_z"

        result = run_density(config)

        self.assertAlmostEqual(float(result.bin_centers[0]), -9.75)
        self.assertAlmostEqual(float(result.bin_centers[-1]), -0.25)
        self.assertEqual(result.selected_atoms_total["water_O"], 4)

    def test_lammpstrj_auto_cell_and_type_map(self):
        root = Path(__file__).resolve().parents[1]
        result = run_density(load_config(root / "examples/density_lammpstrj/config_oxygen_species.yaml"))

        self.assertEqual(result.frames, 2)
        self.assertEqual(result.selected_atoms_total["H2O"], 2)
        self.assertEqual(result.selected_atoms_total["O2-"], 2)
        text = result.csv_path.read_text()
        self.assertIn("H2O_density", text)
        self.assertIn("O2-_density", text)

    def test_lammpstrj_start_timestep(self):
        root = Path(__file__).resolve().parents[1]
        config = load_config(root / "examples/density_lammpstrj/config_oxygen_species.yaml")
        config["input"]["start_timestep"] = 1
        config["output"]["prefix"] = "density_lammpstrj_after_step1"

        result = run_density(config)

        self.assertEqual(result.frames, 1)
        self.assertEqual(result.selected_atoms_total["H2O"], 1)
        self.assertEqual(result.selected_atoms_total["O2-"], 1)

    def test_mass_density_g_cm3_for_h2o(self):
        root = Path(__file__).resolve().parents[1]
        config = load_config(root / "examples/density_lammpstrj/config_oxygen_species.yaml")
        config["selection"]["oxygen_species"] = ["H2O"]
        config["normalization"] = {"type": "mass_density", "unit": "g/cm^3"}
        config["output"]["prefix"] = "density_lammpstrj_h2o_mass_density"

        result = run_density(config)

        density = result.profiles["H2O"]["density"]
        self.assertGreater(float(density.max()), 0.0)
        text = result.csv_path.read_text()
        self.assertIn("H2O_density", text)


if __name__ == "__main__":
    unittest.main()
