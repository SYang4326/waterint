"""Small config helpers used by the reused v0.6 SFG workflow code."""
from __future__ import annotations

from pathlib import Path
from typing import Any


def require_mapping(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Config section '{key}' is required and must be a mapping.")
    return value


def resolve_path(config: dict[str, Any], path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else Path(config.get("_config_dir", ".")) / path


def parse_cell(value: Any) -> tuple[float, float, float] | None:
    if value is None or str(value).lower() == "auto":
        return None
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError("system.cell must be [Lx, Ly, Lz] in Angstrom or auto.")
    return tuple(float(v) for v in value)  # type: ignore[return-value]
