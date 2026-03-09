from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

import numpy as np
import open3d as o3d


@dataclass(frozen=True)
class BodyFrame:
    center_world: np.ndarray
    axes_world: np.ndarray
    extents: np.ndarray

    def world_to_body(self, points_world: np.ndarray) -> np.ndarray:
        return (points_world - self.center_world[None, :]) @ self.axes_world

    def body_to_world(self, points_body: np.ndarray) -> np.ndarray:
        return points_body @ self.axes_world.T + self.center_world[None, :]

    def to_jsonable(self) -> Dict[str, Any]:
        return {
            "center_world": [float(v) for v in self.center_world.tolist()],
            "axes_world": [[float(c) for c in row] for row in self.axes_world.tolist()],
            "extents": [float(v) for v in self.extents.tolist()],
        }


def _normalize(vec: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(vec))
    if n <= 1e-9:
        return vec.copy()
    return vec / n


def _axis_spread(points: np.ndarray, axis: np.ndarray, center: np.ndarray) -> float:
    proj = (points - center[None, :]) @ axis
    return float(np.std(proj))


def estimate_body_frame(points_world: np.ndarray, config: Mapping[str, Any] | None = None) -> Tuple[BodyFrame, Dict[str, Any]]:
    if points_world.ndim != 2 or points_world.shape[1] != 3:
        raise ValueError("Input points must be Nx3.")
    if points_world.shape[0] == 0:
        raise ValueError("Input points are empty.")

    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    if isinstance(config, Mapping):
        wu = config.get("world_up")
        if wu is not None:
            try:
                wu_arr = np.asarray(wu, dtype=np.float64).reshape(-1)
                if wu_arr.size == 3:
                    world_up = wu_arr
            except Exception:
                pass
    world_up = _normalize(world_up)
    if float(np.linalg.norm(world_up)) <= 1e-9:
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points_world.astype(np.float64))
    hull, _ = cloud.compute_convex_hull()
    obb = hull.get_oriented_bounding_box()

    center = np.asarray(obb.center, dtype=np.float64)
    obb_extents = np.asarray(obb.extent, dtype=np.float64)
    obb_axes = np.asarray(obb.R, dtype=np.float64)
    raw_axes = [_normalize(obb_axes[:, i]) for i in range(3)]

    up_scores = [abs(float(np.dot(axis, world_up))) for axis in raw_axes]
    y_idx = int(np.argmax(np.array(up_scores, dtype=np.float64)))
    y_axis = raw_axes[y_idx]
    if float(np.dot(y_axis, world_up)) < 0.0:
        y_axis = -y_axis

    remain = [idx for idx in range(3) if idx != y_idx]
    spread0 = _axis_spread(points_world, raw_axes[remain[0]], center)
    spread1 = _axis_spread(points_world, raw_axes[remain[1]], center)
    x_idx = remain[0] if spread0 >= spread1 else remain[1]
    z_idx = remain[1] if x_idx == remain[0] else remain[0]

    x_axis = raw_axes[x_idx]
    z_axis = _normalize(np.cross(x_axis, y_axis))
    if float(np.linalg.norm(z_axis)) <= 1e-9:
        z_axis = raw_axes[z_idx]
    x_axis = _normalize(np.cross(y_axis, z_axis))
    z_axis = _normalize(np.cross(x_axis, y_axis))

    axes = np.stack([x_axis, y_axis, z_axis], axis=1).astype(np.float64)
    extents = obb_extents[np.array([x_idx, y_idx, z_idx], dtype=np.int64)]
    frame = BodyFrame(center_world=center, axes_world=axes, extents=extents)
    info = {
        "obb_center": [float(v) for v in center.tolist()],
        "obb_extent": [float(v) for v in obb_extents.tolist()],
        "body_extent": [float(v) for v in extents.tolist()],
        "vertical_axis_index": int(y_idx),
        "horizontal_axis_index": int(x_idx),
        "depth_axis_index": int(z_idx),
    }
    return frame, info
