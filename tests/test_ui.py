from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from waterint.ui.core import collect_artifacts, parse_ui_config, type_map_yaml, yaml_string
import waterint.ui.desktop as desktop
from waterint.ui.desktop import EXAMPLES, MODULES
from waterint.ui.server import _static_path


class UITests(unittest.TestCase):
    def test_parse_ui_config_adds_config_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            base_dir = Path(tmp)
            config = parse_ui_config(
                """
input:
  trajectory: input.xyz
output:
  directory: output
""",
                base_dir,
            )

            self.assertEqual(config["_config_dir"], str(base_dir.resolve()))
            self.assertEqual(config["_config_path"], str((base_dir / "waterint-ui-config.yaml").resolve()))
            self.assertEqual(config["input"]["trajectory"], "input.xyz")

    def test_parse_ui_config_rejects_empty_or_non_mapping_yaml(self):
        with tempfile.TemporaryDirectory() as tmp:
            base_dir = Path(tmp)
            with self.assertRaises(ValueError):
                parse_ui_config("", base_dir)
            with self.assertRaises(ValueError):
                parse_ui_config("- not\n- a\n- mapping\n", base_dir)

    def test_collect_artifacts_reads_result_paths_and_mappings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = root / "profile.csv"
            png_path = root / "profile.png"
            metadata_path = root / "metadata.json"
            for path in (csv_path, png_path, metadata_path):
                path.write_text("x", encoding="utf-8")
            result = SimpleNamespace(
                csv_path=csv_path,
                png_paths={"figure": png_path},
                metadata_path=metadata_path,
            )

            artifacts = collect_artifacts(result)
            names = {item["name"] for item in artifacts}
            kinds = {item["name"]: item["kind"] for item in artifacts}

            self.assertEqual(names, {"profile.csv", "profile.png", "metadata.json"})
            self.assertEqual(kinds["profile.png"], "image")
            self.assertEqual(kinds["profile.csv"], "file")

    def test_static_path_stays_inside_static_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            static_dir = Path(tmp) / "static"
            static_dir.mkdir()
            (static_dir / "app.js").write_text("", encoding="utf-8")

            self.assertEqual(_static_path(static_dir, "app.js"), (static_dir / "app.js").resolve())
            self.assertIsNone(_static_path(static_dir, "../server.py"))

    def test_type_map_yaml_accepts_colon_and_equals_lines(self):
        self.assertEqual(type_map_yaml("1: H\n2=O"), ["    1: H", "    2: O"])

    def test_yaml_string_is_inline_yaml_scalar(self):
        self.assertEqual(yaml_string("path/to/trajectory.npz"), '"path/to/trajectory.npz"')
        self.assertNotIn("...", yaml_string("path/to/trajectory.npz"))

    def test_desktop_examples_cover_all_modules(self):
        self.assertEqual(set(EXAMPLES), set(MODULES))

    def test_desktop_generated_yaml_is_parseable(self):
        app = desktop.WaterIntDesktopApp()
        try:
            config_yaml = app.yaml_text.get("1.0", "end")
            config = parse_ui_config(config_yaml, Path.cwd())

            self.assertEqual(config["input"]["format"], "npz")
            self.assertIn("trajectory", config["input"])
        finally:
            app.destroy()

    def test_desktop_run_worker_preserves_error_message(self):
        messages: list[str] = []
        original_showerror = desktop.messagebox.showerror
        desktop.messagebox.showerror = lambda _title, message: messages.append(message)
        try:
            app = desktop.WaterIntDesktopApp()
            app.after = lambda _delay, callback: callback()
            app._run_worker("density", "", Path.cwd())

            self.assertEqual(app.status_var.get(), "Analysis failed")
            self.assertEqual(messages, ["Config YAML is empty."])
        finally:
            desktop.messagebox.showerror = original_showerror
            try:
                app.destroy()
            except Exception:
                pass


if __name__ == "__main__":
    unittest.main()
