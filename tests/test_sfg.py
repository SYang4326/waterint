from pathlib import Path
import shutil
import unittest

import numpy as np

from waterint.config import load_config
from waterint.sfg import run_sfg
from waterint.sfg.processing import load_cf


class SfgTests(unittest.TestCase):
    def test_sfg_single_writes_ft_and_plot(self):
        root = Path(__file__).resolve().parents[1]
        output = root / "examples/sfg_cf/output"
        if output.exists():
            shutil.rmtree(output)

        result = run_sfg(load_config(root / "examples/sfg_cf/config_single.yaml"))

        self.assertEqual(result.mode, "single")
        self.assertTrue(result.ft_paths["spectrum"].exists())
        self.assertTrue(result.png_paths["spectrum"].exists())
        self.assertTrue(result.metadata_path.exists())

    def test_sfg_combine_bins_uses_count_weighted_average(self):
        root = Path(__file__).resolve().parents[1]
        output = root / "examples/sfg_cf/output"
        if output.exists():
            shutil.rmtree(output)

        result = run_sfg(load_config(root / "examples/sfg_cf/config_combine_bins.yaml"))

        combined = result.cf_paths["0_1d5:all"]
        time_ps, corr, counts = load_cf(combined)
        self.assertTrue(np.allclose(time_ps, [0.0, 0.001, 0.002, 0.003, 0.004]))
        self.assertEqual(int(counts[0]), 30)
        self.assertAlmostEqual(float(corr[0]), (1.0 * 10 + 0.9 * 20) / 30)
        self.assertTrue(result.ft_paths["0_1d5:all"].exists())
        self.assertTrue(result.ft_paths["all:all"].exists())
        self.assertTrue(result.png_paths["overlay_all"].exists())
        self.assertTrue(result.metadata_path.exists())

    def test_sfg_trajectory_writes_ssvvcf_ft_and_plot(self):
        root = Path(__file__).resolve().parents[1]
        output = root / "examples/sfg_trajectory/output"
        if output.exists():
            shutil.rmtree(output)

        result = run_sfg(load_config(root / "examples/sfg_trajectory/config.yaml"))

        self.assertEqual(result.mode, "trajectory")
        self.assertTrue(result.cf_paths["ssvvcf"].exists())
        self.assertTrue(result.cf_paths["zref"].exists())
        self.assertTrue(result.ft_paths["spectrum"].exists())
        self.assertTrue(result.png_paths["spectrum"].exists())
        _time_ps, corr, counts = load_cf(result.cf_paths["ssvvcf"])
        self.assertGreater(int(counts[0]), 0)
        self.assertGreater(float(np.max(np.abs(corr))), 0.0)


if __name__ == "__main__":
    unittest.main()
