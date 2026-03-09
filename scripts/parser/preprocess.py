from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

import numpy as np
import open3d as o3d


def _default_preprocess_config() -> Dict[str, Any]:
    return {
        "enabled": True,
        "voxel_size": 0.02,
        "radius_outlier": {
            "enabled": True,
            "nb_points": 8,
            "radius": 0.08,
        },
        "statistical_outlier": {
            "enabled": True,
            "nb_neighbors": 20,
            "std_ratio": 2.0,
        },
        "hidden_point_filter": {
            "enabled": False,
            "radius_scale": 4.0,
            "camera_offset": [0.0, 0.0, 1.0],
        },
    }


def _merge_dict(base: Dict[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = _merge_dict(dict(out[key]), value)
        else:
            out[key] = value
    return out


def preprocess_human_points(points: np.ndarray, config: Mapping[str, Any] | None) -> Tuple[np.ndarray, Dict[str, Any]]:
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Input points must be Nx3.")
    valid = np.isfinite(points).all(axis=1)
    points = points[valid]
    if points.shape[0] == 0:
        raise ValueError("Input points are empty after finite filtering.")

    preprocess_cfg = _default_preprocess_config()
    if config is not None:
        preprocess_cfg = _merge_dict(preprocess_cfg, dict(config))
    if not bool(preprocess_cfg.get("enabled", True)):
        return points.astype(np.float64), {"enabled": False, "input_points": int(points.shape[0]), "output_points": int(points.shape[0])}

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    stats: Dict[str, Any] = {"enabled": True, "input_points": int(points.shape[0])}

    voxel_size = float(preprocess_cfg.get("voxel_size", 0.02))
    if voxel_size > 1e-6:
        cloud = cloud.voxel_down_sample(voxel_size)
    stats["after_voxel"] = int(len(cloud.points))

    radius_cfg = preprocess_cfg.get("radius_outlier", {})
    if isinstance(radius_cfg, Mapping) and bool(radius_cfg.get("enabled", True)) and len(cloud.points) > 0:
        nb_points = int(radius_cfg.get("nb_points", 8))
        radius = float(radius_cfg.get("radius", 0.08))
        cloud, _ = cloud.remove_radius_outlier(nb_points=max(nb_points, 1), radius=max(radius, 1e-6))
    stats["after_radius_outlier"] = int(len(cloud.points))

    stat_cfg = preprocess_cfg.get("statistical_outlier", {})
    if isinstance(stat_cfg, Mapping) and bool(stat_cfg.get("enabled", True)) and len(cloud.points) > 0:
        nb_neighbors = int(stat_cfg.get("nb_neighbors", 20))
        std_ratio = float(stat_cfg.get("std_ratio", 2.0))
        cloud, _ = cloud.remove_statistical_outlier(
            nb_neighbors=max(nb_neighbors, 1),
            std_ratio=max(std_ratio, 1e-6),
        )
    stats["after_statistical_outlier"] = int(len(cloud.points))

    hpr_cfg = preprocess_cfg.get("hidden_point_filter", {})
    if isinstance(hpr_cfg, Mapping) and bool(hpr_cfg.get("enabled", False)) and len(cloud.points) > 0:
        pts = np.asarray(cloud.points, dtype=np.float64)
        center = np.mean(pts, axis=0)
        radius = max(float(np.linalg.norm(np.max(pts, axis=0) - np.min(pts, axis=0))), 1e-3)
        radius_scale = float(hpr_cfg.get("radius_scale", 4.0))
        camera_offset = np.array(hpr_cfg.get("camera_offset", [0.0, 0.0, 1.0]), dtype=np.float64)
        if camera_offset.shape != (3,):
            camera_offset = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        camera_location = center + camera_offset * radius * max(radius_scale, 1.0)
        _, visible = cloud.hidden_point_removal(camera_location.tolist(), radius * max(radius_scale, 1.0))
        cloud = cloud.select_by_index(visible)
    stats["after_hidden_point_filter"] = int(len(cloud.points))

    out = np.asarray(cloud.points, dtype=np.float64)
    if out.shape[0] == 0:
        out = points.astype(np.float64)
    stats["output_points"] = int(out.shape[0])
    return out, stats

