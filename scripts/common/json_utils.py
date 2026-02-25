# scripts/common/json_utils.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any, ascii_flag: bool = False, compact: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        if compact:
            json.dump(payload, handle, ensure_ascii=ascii_flag, separators=(",", ":"))
        else:
            json.dump(payload, handle, ensure_ascii=ascii_flag, indent=2)
