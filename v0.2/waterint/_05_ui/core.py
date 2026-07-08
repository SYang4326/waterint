from __future__ import annotations

import uuid
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from waterint._04_workflows.workflows.oh_orientation import run_angle_z
from waterint._04_workflows.workflows.density import run_density
from waterint._04_workflows.workflows.hbond import run_hbond
from waterint._04_workflows.workflows.sfg import run_sfg


RUNNERS = {
    "density": run_density,
    "oh-orientation": run_angle_z,
    "hbond": run_hbond,
    "sfg": run_sfg,
}


@dataclass(frozen=True)
class AnalysisRun:
    module: str
    config_path: Path
    artifacts: list[dict[str, Any]]


def parse_ui_config(config_yaml: str, base_dir: Path) -> dict[str, Any]:
    if not config_yaml.strip():
        raise ValueError("Config YAML is empty.")
    config = yaml.safe_load(config_yaml)
    if not isinstance(config, dict):
        raise ValueError("Config YAML must define a mapping.")
    config["_config_dir"] = str(base_dir.resolve())
    config["_config_path"] = str((base_dir / "waterint-ui-config.yaml").resolve())
    return config


def run_analysis(module: str, config_yaml: str, base_dir: Path) -> AnalysisRun:
    if module not in RUNNERS:
        raise ValueError(f"Unknown module: {module}")
    config = parse_ui_config(config_yaml, base_dir)
    output_cfg = config.setdefault("output", {})
    if not isinstance(output_cfg, dict):
        raise ValueError("output must be a mapping.")
    outdir = resolve_output_dir(config, base_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    config_path = outdir / "waterint_ui_config.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(public_config(config), handle, sort_keys=False)

    result = RUNNERS[module](config)
    artifacts = collect_artifacts(result)
    artifacts.insert(0, artifact("config", config_path))
    return AnalysisRun(module=module, config_path=config_path, artifacts=artifacts)


def collect_artifacts(result: Any) -> list[dict[str, Any]]:
    paths: list[tuple[str, Path]] = []
    for attr in ("csv_path", "raw_csv_path", "png_path", "metadata_path"):
        value = getattr(result, attr, None)
        if value:
            paths.append((attr.removesuffix("_path"), Path(value)))
    for attr in ("csv_paths", "png_paths", "cf_paths", "ft_paths"):
        mapping = getattr(result, attr, None)
        if isinstance(mapping, dict):
            for label, value in mapping.items():
                paths.append((f"{attr.removesuffix('_paths')}:{label}", Path(value)))
    return [artifact(label, path) for label, path in paths if path.exists()]


def artifact(label: str, path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    kind = "image" if suffix in {".png", ".jpg", ".jpeg", ".gif", ".webp"} else "file"
    return {
        "id": uuid.uuid4().hex[:10],
        "label": label,
        "name": path.name,
        "path": str(path.resolve()),
        "kind": kind,
        "size": path.stat().st_size if path.exists() else 0,
    }


def resolve_base_dir(raw: Any, cwd: Path) -> Path:
    if raw in {None, ""}:
        return cwd
    path = Path(str(raw)).expanduser()
    return path if path.is_absolute() else cwd / path


def resolve_output_dir(config: dict[str, Any], base_dir: Path) -> Path:
    output_cfg = config.get("output", {})
    directory = "waterint_ui_output"
    if isinstance(output_cfg, dict):
        directory = str(output_cfg.get("directory", directory))
    path = Path(directory).expanduser()
    return path if path.is_absolute() else base_dir / path


def public_config(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if not key.startswith("_")}


def comma_list(text: str) -> list[str]:
    return [item.strip() for item in text.replace("[", "").replace("]", "").split(",") if item.strip()]


def type_map_from_text(text: str) -> dict[int, str]:
    type_map: dict[int, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if ":" in line:
            raw_key, raw_value = line.split(":", 1)
        elif "=" in line:
            raw_key, raw_value = line.split("=", 1)
        else:
            raise ValueError(f"Bad type-map line {raw_line!r}; expected '1: H' or '1=H'.")
        type_map[int(raw_key.strip())] = raw_value.strip()
    return type_map


def type_map_yaml(text: str) -> list[str]:
    mapping = type_map_from_text(text)
    return [f"    {key}: {value}" for key, value in sorted(mapping.items())]


def yaml_string(text: str) -> str:
    return json.dumps(str(text))
