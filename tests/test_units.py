from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from waterint.config import load_config
from waterint.density import run_density
from waterint.sfg import run_sfg
from waterint.units import CM1_PER_THz, unit_system_from_config


class UnitSystemTests(unittest.TestCase):
    def test_lammps_style_unit_conversions(self):
        self.assertEqual(unit_system_from_config({}).style.name, "metal")
        units = unit_system_from_config(
            {
                "units": {
                    "style": "real",
                    "output": {
                        "length": "nm",
                        "time": "ns",
                        "frequency": "THz",
                    },
                }
            }
        )
        self.assertEqual(units.style.length_unit, "A")
        self.assertAlmostEqual(units.input_time_ps(1.0), 0.001)
        self.assertAlmostEqual(float(units.output_length([10.0])[0]), 1.0)
        self.assertAlmostEqual(float(units.output_time([1000.0])[0]), 1.0)
        self.assertAlmostEqual(float(units.output_frequency([CM1_PER_THz])[0]), 1.0)

    def test_density_output_length_and_number_density_units(self):
        root = Path(__file__).resolve().parents[1]
        config = load_config(root / "examples/density_xyz/config.yaml")
        with tempfile.TemporaryDirectory() as tmp:
            config["output"]["directory"] = tmp
            config["output"]["plot"] = False
            config["units"] = {
                "style": "metal",
                "output": {
                    "length": "nm",
                    "number_density": "1/nm^3",
                },
            }
            result = run_density(config)
            data = np.loadtxt(result.csv_path, delimiter=",", skiprows=1)
            self.assertAlmostEqual(float(data[0, 0]), float(result.bin_centers[0]) / 10.0)
            self.assertAlmostEqual(float(data[0, 2]), float(result.profiles["water_O"]["density"][0]) * 1000.0)

    def test_density_output_mass_density_units(self):
        root = Path(__file__).resolve().parents[1]
        config = load_config(root / "examples/density_xyz/config.yaml")
        with tempfile.TemporaryDirectory() as tmp:
            config["output"]["directory"] = tmp
            config["output"]["plot"] = False
            config["normalization"] = {
                "type": "mass_density",
                "mass_amu": 18.015,
            }
            config["units"] = {
                "style": "metal",
                "output": {
                    "mass_density": "kg/m^3",
                },
            }
            result = run_density(config)
            data = np.loadtxt(result.csv_path, delimiter=",", skiprows=1)
            self.assertAlmostEqual(float(data[0, 2]), float(result.profiles["water_O"]["density"][0]) * 1000.0)

    def test_sfg_frequency_output_unit(self):
        root = Path(__file__).resolve().parents[1]
        config = load_config(root / "examples/sfg_trajectory/config.yaml")
        with tempfile.TemporaryDirectory() as tmp:
            config["output"]["directory"] = tmp
            config["output"]["plot"] = False
            config["units"] = {
                "style": "metal",
                "output": {
                    "frequency": "THz",
                },
            }
            result = run_sfg(config)
            text = result.ft_paths["spectrum"].read_text(encoding="utf-8")
            self.assertIn("frequency_THz", text.splitlines()[0])


if __name__ == "__main__":
    unittest.main()
