# scripts/common/angle_utils.py
from __future__ import annotations


def wrap180(angle_deg: float) -> float:
    return ((angle_deg + 180.0) % 360.0) - 180.0
