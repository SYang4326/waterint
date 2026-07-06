from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from waterint.batch import expand_batch_tasks, load_batch_config, run_batch


class BatchTests(unittest.TestCase):
    def test_expand_batch_repeat_tasks(self):
        root = Path(__file__).resolve().parents[1]
        config = load_batch_config(root / "examples/batch_total.yaml")
        tasks = expand_batch_tasks(config)

        self.assertEqual([task.name for task in tasks], ["density_xyz", "density_lammpstrj", "hbond_small", "sfg_small"])
        self.assertEqual([task.module for task in tasks], ["density", "density", "hbond", "sfg"])

    def test_batch_dry_run_writes_summary_without_running(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            batch_path = Path(tmp) / "batch.yaml"
            batch_path.write_text(
                f"""
output:
  summary: summary.json
tasks:
  - name: density_test
    module: density
    config: {root / "examples/density_xyz/config.yaml"}
""",
                encoding="utf-8",
            )

            result = run_batch(batch_path, dry_run=True)
            self.assertEqual(result.tasks[0].status, "dry_run")
            self.assertTrue(result.summary_path.exists())
            summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["tasks_total"], 1)
            self.assertTrue(summary["dry_run"])

    def test_batch_runs_small_density_task(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            batch_path = Path(tmp) / "batch.yaml"
            batch_path.write_text(
                f"""
output:
  summary: summary.json
tasks:
  - name: density_test
    module: density
    config: {root / "examples/density_xyz/config.yaml"}
    overrides:
      output:
        directory: {tmp}/density_output
        plot: false
""",
                encoding="utf-8",
            )

            result = run_batch(batch_path)
            self.assertEqual(result.tasks[0].status, "complete")
            self.assertTrue((Path(tmp) / "density_output/density_water_O.csv").exists())
            self.assertTrue(result.summary_path.exists())

    def test_batch_repeat_overrides_are_expanded(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            batch_path = Path(tmp) / "batch.yaml"
            batch_path.write_text(
                f"""
tasks:
  - name: density_{{id}}
    module: density
    defaults:
      overrides:
        output:
          plot: false
    repeat:
      - id: a
        config: {root / "examples/density_xyz/config.yaml"}
        overrides:
          output:
            directory: {tmp}/a
      - id: b
        config: {root / "examples/density_xyz/config.yaml"}
        overrides:
          output:
            directory: {tmp}/b
""",
                encoding="utf-8",
            )

            result = run_batch(batch_path)
            self.assertEqual([task.status for task in result.tasks], ["complete", "complete"])
            self.assertTrue((Path(tmp) / "a/density_water_O.csv").exists())
            self.assertTrue((Path(tmp) / "b/density_water_O.csv").exists())


if __name__ == "__main__":
    unittest.main()
