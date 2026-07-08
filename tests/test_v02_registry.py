from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V02 = ROOT / "v0.2"


class V02RegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        self._old_path = list(sys.path)
        self._old_modules = {name: module for name, module in sys.modules.items() if name == "waterint" or name.startswith("waterint.")}
        for name in list(self._old_modules):
            sys.modules.pop(name, None)
        sys.path.insert(0, str(V02))

    def tearDown(self) -> None:
        for name in [name for name in sys.modules if name == "waterint" or name.startswith("waterint.")]:
            sys.modules.pop(name, None)
        sys.modules.update(self._old_modules)
        sys.path[:] = self._old_path

    def test_registry_contains_current_analysis_modules(self):
        registry = importlib.import_module("waterint._04_workflows.registry.registry")

        names = {module.name for module in registry.iter_analysis_modules()}
        self.assertEqual(names, {"density", "oh-orientation", "hbond", "sfg"})
        self.assertEqual(registry.get_analysis_module("angle-z").name, "oh-orientation")

    def test_cli_help_is_built_from_registry(self):
        cli = importlib.import_module("waterint.cli")

        with self.assertRaises(SystemExit) as caught:
            cli.main(["--help"])
        self.assertEqual(caught.exception.code, 0)

    def test_registered_modules_have_config_cli_argument(self):
        cli = importlib.import_module("waterint.cli")

        for command in ["density", "oh-orientation", "angle-z", "hbond", "sfg"]:
            with self.subTest(command=command):
                with self.assertRaises(SystemExit) as caught:
                    cli.main([command, "--help"])
                self.assertEqual(caught.exception.code, 0)

    def test_sfg_computation_uses_single_module(self):
        sfg = importlib.import_module("waterint._02_computation.sfg")

        self.assertEqual(sfg.SfgResult.__module__, "waterint._02_computation.sfg")
        self.assertTrue(hasattr(sfg, "compute_ssvvcf_from_frames"))
        for old_module in [
            "waterint._02_computation.sfg_processing",
            "waterint._02_computation.sfg_result",
            "waterint._02_computation.sfg_trajectory",
        ]:
            with self.subTest(old_module=old_module):
                with self.assertRaises(ModuleNotFoundError):
                    importlib.import_module(old_module)

    def test_output_layer_uses_one_file_per_module(self):
        output_dir = V02 / "waterint" / "_03_output"

        files = {path.name for path in output_dir.glob("*.py")}
        self.assertEqual(
            files,
            {
                "__init__.py",
                "density.py",
                "hbond.py",
                "metadata.py",
                "oh_orientation.py",
                "sfg.py",
            },
        )
        self.assertFalse(any(name.endswith("_plotting.py") for name in files))


if __name__ == "__main__":
    unittest.main()
