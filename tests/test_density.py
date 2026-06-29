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
        self.assertEqual(result.selected_atoms_total, 4)
        self.assertTrue(result.csv_path.exists())
        self.assertTrue(result.png_path.exists())
        self.assertTrue(result.metadata_path.exists())


if __name__ == "__main__":
    unittest.main()
