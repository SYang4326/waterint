from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SfgResult:
    mode: str
    cf_paths: dict[str, Path]
    ft_paths: dict[str, Path]
    png_paths: dict[str, Path]
    metadata_path: Path
