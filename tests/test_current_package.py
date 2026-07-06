from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from waterint.cli import main
from waterint.computation.density import compute_density_profile
from waterint.config import load_config
from waterint.workflows.density import run_density


class CurrentPackageTests(unittest.TestCase):
    def test_stable_package_imports(self):
        import waterint

        self.assertTrue(callable(waterint.run_density))
        self.assertTrue(callable(waterint.run_oh_orientation))
        self.assertTrue(callable(waterint.run_hbond))
        self.assertTrue(callable(waterint.run_sfg))

    def test_cli_help_uses_stable_command_name(self):
        with self.assertRaises(SystemExit) as caught:
            main(["--help"])
        self.assertEqual(caught.exception.code, 0)

    def test_density_compute_is_independent_of_config_files(self):
        result = compute_density_profile(
            [{"O": np.asarray([0.2, 0.8, 1.4])}, {"O": np.asarray([0.3, 1.2])}],
            labels=["O"],
            bin_edges=np.asarray([0.0, 1.0, 2.0]),
            cell=(10.0, 10.0, 10.0),
            axis=2,
        )

        self.assertEqual(result.frames, 2)
        np.testing.assert_allclose(result.profiles["O"]["counts_per_frame"], [1.5, 1.0])
        np.testing.assert_allclose(result.profiles["O"]["density"], [0.015, 0.01])

    def test_density_workflow_runs_minimal_xyz_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            xyz = root / "input.xyz"
            xyz.write_text(
                """3
frame 0
O 0.0 0.0 1.0
H 0.0 0.0 1.8
H 0.0 0.8 1.0
3
frame 1
O 0.0 0.0 2.0
H 0.0 0.0 2.8
H 0.0 0.8 2.0
""",
                encoding="utf-8",
            )
            config_path = root / "density.yaml"
            config_path.write_text(
                """
input:
  trajectory: input.xyz
  format: xyz

system:
  cell: [10.0, 10.0, 10.0]

selection:
  mode: element
  species: [O]
  label: oxygen

coordinate:
  mode: absolute
  axis: z
  range: [0.0, 4.0]
  bins: 4

output:
  directory: output
  prefix: density_test
  plot: false
""",
                encoding="utf-8",
            )

            result = run_density(load_config(config_path))

            self.assertEqual(result.frames, 2)
            self.assertEqual(result.selected_atoms_total["oxygen"], 2)
            self.assertTrue(result.csv_path.exists())
            self.assertTrue(result.metadata_path.exists())


if __name__ == "__main__":
    unittest.main()
