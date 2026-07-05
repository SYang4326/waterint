from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from waterint.config import load_config
from waterint.sfg import run_sfg as run_sfg_legacy
from waterint.sfg.processing import load_cf
from waterint_v0.workflows.sfg import run_sfg


class WaterintV0SfgTests(unittest.TestCase):
    def test_sfg_single_matches_legacy_outputs(self):
        root = Path(__file__).resolve().parents[1]
        with TemporaryDirectory() as tmpdir:
            config = load_config(root / "examples/sfg_cf/config_single.yaml")
            config["output"]["directory"] = str(Path(tmpdir) / "v0")
            legacy_config = load_config(root / "examples/sfg_cf/config_single.yaml")
            legacy_config["output"]["directory"] = str(Path(tmpdir) / "legacy")

            result = run_sfg(config)
            legacy = run_sfg_legacy(legacy_config)

            self.assertEqual(result.mode, legacy.mode)
            self.assertTrue(result.ft_paths["spectrum"].exists())
            self.assertTrue(result.png_paths["spectrum"].exists())
            self.assertTrue(result.metadata_path.exists())

    def test_sfg_combine_bins_matches_legacy_average(self):
        root = Path(__file__).resolve().parents[1]
        with TemporaryDirectory() as tmpdir:
            config = load_config(root / "examples/sfg_cf/config_combine_bins.yaml")
            config["output"]["directory"] = str(Path(tmpdir) / "v0")

            result = run_sfg(config)

            time_ps, corr, counts = load_cf(result.cf_paths["0_1d5:all"])
            self.assertTrue(np.allclose(time_ps, [0.0, 0.001, 0.002, 0.003, 0.004]))
            self.assertEqual(int(counts[0]), 30)
            self.assertAlmostEqual(float(corr[0]), (1.0 * 10 + 0.9 * 20) / 30)
            self.assertTrue(result.ft_paths["0_1d5:all"].exists())
            self.assertTrue(result.ft_paths["all:all"].exists())
            self.assertTrue(result.png_paths["overlay_all"].exists())
            self.assertTrue(result.metadata_path.exists())

    def test_sfg_trajectory_matches_legacy_shape(self):
        root = Path(__file__).resolve().parents[1]
        with TemporaryDirectory() as tmpdir:
            config = load_config(root / "examples/sfg_trajectory/config.yaml")
            config["output"]["directory"] = str(Path(tmpdir) / "v0")
            config["output"]["prefix"] = "sfg_trajectory_v0"

            legacy_config = load_config(root / "examples/sfg_trajectory/config.yaml")
            legacy_config["output"]["directory"] = str(Path(tmpdir) / "legacy")
            legacy_config["output"]["prefix"] = "sfg_trajectory_v0_legacy"

            result = run_sfg(config)
            legacy = run_sfg_legacy(legacy_config)

            self.assertEqual(result.mode, legacy.mode)
            self.assertTrue(result.cf_paths["ssvvcf"].exists())
            self.assertTrue(result.cf_paths["zref"].exists())
            self.assertTrue(result.ft_paths["spectrum"].exists())
            self.assertTrue(result.png_paths["spectrum"].exists())
            _time_ps, corr, counts = load_cf(result.cf_paths["ssvvcf"])
            self.assertGreater(int(counts[0]), 0)
            self.assertGreater(float(np.max(np.abs(corr))), 0.0)


if __name__ == "__main__":
    unittest.main()
