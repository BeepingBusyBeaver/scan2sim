# scripts/common/path_utils.py
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional


def natural_key_text(text: str):
    return [int(token) if token.isdigit() else token.lower() for token in re.split(r"(\d+)", text)]


def natural_key_path(path: Path):
    return natural_key_text(path.name)


def extract_numeric_suffix(path_or_name: str) -> Optional[int]:
    name = Path(path_or_name).name
    match = re.search(r"(\d+)(?=\.json$)", name)
    if not match:
        return None
    return int(match.group(1))
