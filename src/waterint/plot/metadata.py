from __future__ import annotations

from pathlib import Path
from typing import Any

import json


def write_metadata(path: Path, metadata: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")
