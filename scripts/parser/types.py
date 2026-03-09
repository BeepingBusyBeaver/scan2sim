from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np


@dataclass(frozen=True)
class PartBox:
    name: str
    valid: bool
    num_points: int
    min_xyz: np.ndarray
    max_xyz: np.ndarray
    center_xyz: np.ndarray
    size_xyz: np.ndarray

    @staticmethod
    def empty(name: str) -> "PartBox":
        zeros = np.zeros((3,), dtype=np.float64)
        return PartBox(
            name=name,
            valid=False,
            num_points=0,
            min_xyz=zeros.copy(),
            max_xyz=zeros.copy(),
            center_xyz=zeros.copy(),
            size_xyz=zeros.copy(),
        )

    @staticmethod
    def from_points(name: str, points: np.ndarray, min_points: int) -> "PartBox":
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"Part '{name}' points must be Nx3.")
        count = int(points.shape[0])
        if count < int(min_points):
            return PartBox.empty(name)

        min_xyz = np.min(points, axis=0).astype(np.float64)
        max_xyz = np.max(points, axis=0).astype(np.float64)
        center_xyz = ((min_xyz + max_xyz) * 0.5).astype(np.float64)
        size_xyz = (max_xyz - min_xyz).astype(np.float64)
        return PartBox(
            name=name,
            valid=True,
            num_points=count,
            min_xyz=min_xyz,
            max_xyz=max_xyz,
            center_xyz=center_xyz,
            size_xyz=size_xyz,
        )

    def to_jsonable(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "valid": self.valid,
            "num_points": self.num_points,
            "min_xyz": [float(v) for v in self.min_xyz.tolist()],
            "max_xyz": [float(v) for v in self.max_xyz.tolist()],
            "center_xyz": [float(v) for v in self.center_xyz.tolist()],
            "size_xyz": [float(v) for v in self.size_xyz.tolist()],
        }

