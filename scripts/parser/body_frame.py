# scripts/parser/body_frame.py
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


def _parse_vector3(value: Any, default: np.ndarray) -> np.ndarray:
    try:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
        if arr.size == 3:
            return arr
    except Exception:
        pass
    return default


def _parse_axes_world(value: Any) -> np.ndarray | None:
    try:
        arr = np.asarray(value, dtype=np.float64)
    except Exception:
        return None
    if arr.shape == (3, 3):
        return arr
    flat = arr.reshape(-1)
    if flat.size == 9:
        return flat.reshape(3, 3)
    return None


def _apply_horizontal_hint(
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    z_axis: np.ndarray,
    config: Mapping[str, Any] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    hint_raw = None
    mode = "align"
    allow_xz_swap = True
    min_alignment = 0.0
    if isinstance(config, Mapping):
        hint_raw = config.get("horizontal_hint")
        mode_raw = config.get("horizontal_mode")
        if isinstance(mode_raw, str) and mode_raw.strip():
            mode = mode_raw.strip().lower()
        allow_xz_swap = bool(config.get("horizontal_allow_xz_swap", True))
        min_alignment = float(np.clip(float(config.get("horizontal_min_alignment", 0.0)), 0.0, 1.0))
    if hint_raw is None:
        return x_axis, y_axis, z_axis, {"enabled": False}

    hint_vec = _parse_vector3(hint_raw, np.zeros((3,), dtype=np.float64))
    hint_vec = _normalize(hint_vec)
    if float(np.linalg.norm(hint_vec)) <= 1e-9:
        return x_axis, y_axis, z_axis, {"enabled": True, "applied": False, "reason": "invalid_hint"}

    hint_proj = hint_vec - y_axis * float(np.dot(hint_vec, y_axis))
    hint_proj = _normalize(hint_proj)
    if float(np.linalg.norm(hint_proj)) <= 1e-9:
        return x_axis, y_axis, z_axis, {"enabled": True, "applied": False, "reason": "parallel_to_up"}

    if mode in {"lock", "fixed", "strict"}:
        x_axis = hint_proj
        z_axis = _normalize(np.cross(x_axis, y_axis))
        if float(np.linalg.norm(z_axis)) <= 1e-9:
            return x_axis, y_axis, z_axis, {"enabled": True, "applied": False, "reason": "lock_cross_failed"}
        x_axis = _normalize(np.cross(y_axis, z_axis))
        z_axis = _normalize(np.cross(x_axis, y_axis))
        info = {
            "enabled": True,
            "applied": True,
            "mode": "lock",
            "hint": [float(v) for v in hint_vec.tolist()],
            "hint_projected": [float(v) for v in hint_proj.tolist()],
            "swapped_xz": False,
            "flipped_sign": False,
            "final_alignment": float(np.dot(x_axis, hint_proj)),
        }
        return x_axis, y_axis, z_axis, info

    score_x = abs(float(np.dot(x_axis, hint_proj)))
    score_z = abs(float(np.dot(z_axis, hint_proj)))
    swapped = False
    if allow_xz_swap and (score_z > score_x):
        x_axis, z_axis = z_axis, -x_axis
        swapped = True

    flipped = False
    if float(np.dot(x_axis, hint_proj)) < 0.0:
        x_axis = -x_axis
        z_axis = -z_axis
        flipped = True

    x_axis = _normalize(x_axis)
    z_axis = _normalize(np.cross(x_axis, y_axis))
    if float(np.linalg.norm(z_axis)) <= 1e-9:
        z_axis = _normalize(z_axis)
    x_axis = _normalize(np.cross(y_axis, z_axis))
    z_axis = _normalize(np.cross(x_axis, y_axis))
    final_alignment = float(np.dot(x_axis, hint_proj))
    if abs(final_alignment) < min_alignment:
        return x_axis, y_axis, z_axis, {
            "enabled": True,
            "applied": False,
            "mode": "align",
            "reason": "below_min_alignment",
            "min_alignment": float(min_alignment),
            "final_alignment": float(final_alignment),
            "score_x_before": float(score_x),
            "score_z_before": float(score_z),
            "swapped_xz": bool(swapped),
            "flipped_sign": bool(flipped),
        }
    info = {
        "enabled": True,
        "applied": True,
        "mode": "align",
        "hint": [float(v) for v in hint_vec.tolist()],
        "hint_projected": [float(v) for v in hint_proj.tolist()],
        "score_x_before": float(score_x),
        "score_z_before": float(score_z),
        "swapped_xz": bool(swapped),
        "flipped_sign": bool(flipped),
        "allow_xz_swap": bool(allow_xz_swap),
        "final_alignment": float(final_alignment),
    }
    return x_axis, y_axis, z_axis, info


def _orthogonality_error(axes: np.ndarray) -> float:
    gram = axes.T @ axes
    return float(np.max(np.abs(gram - np.eye(3, dtype=np.float64))))


def _fit_center_and_extents(points_world: np.ndarray, center: np.ndarray, axes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    projected = (points_world - center[None, :]) @ axes
    min_xyz = np.min(projected, axis=0)
    max_xyz = np.max(projected, axis=0)
    center_offset = 0.5 * (min_xyz + max_xyz)
    center_refined = center + axes @ center_offset
    projected_refined = (points_world - center_refined[None, :]) @ axes
    min_refined = np.min(projected_refined, axis=0)
    max_refined = np.max(projected_refined, axis=0)
    extents = np.maximum(max_refined - min_refined, 1e-6)
    return center_refined, extents


def _robust_span(points_world: np.ndarray, axis: np.ndarray, q_lo: float, q_hi: float) -> float:
    proj = points_world @ axis
    lo = float(np.quantile(proj, q_lo))
    hi = float(np.quantile(proj, q_hi))
    return max(hi - lo, 1e-9)


def _resolve_world_up(points_world: np.ndarray, config: Mapping[str, Any] | None) -> Tuple[np.ndarray, Dict[str, Any]]:
    hint_default = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    world_up_hint = hint_default.copy()
    mode = "auto"
    auto_up_cfg: Mapping[str, Any] = {}

    if isinstance(config, Mapping):
        world_up_hint = _parse_vector3(config.get("world_up"), hint_default)
        mode_raw = config.get("world_up_mode")
        if isinstance(mode_raw, str) and mode_raw.strip():
            mode = mode_raw.strip().lower()
        auto_up_raw = config.get("auto_up")
        if isinstance(auto_up_raw, Mapping):
            auto_up_cfg = auto_up_raw

    world_up_hint = _normalize(world_up_hint)
    if float(np.linalg.norm(world_up_hint)) <= 1e-9:
        world_up_hint = hint_default.copy()

    if mode in {"fixed", "config", "manual"}:
        return world_up_hint, {"mode": mode, "source": "config_world_up"}

    q_lo, q_hi = 0.05, 0.95
    quantiles = auto_up_cfg.get("span_quantiles")
    if isinstance(quantiles, (list, tuple)) and len(quantiles) == 2:
        try:
            q_lo = float(quantiles[0])
            q_hi = float(quantiles[1])
        except Exception:
            q_lo, q_hi = 0.05, 0.95
    q_lo = float(np.clip(q_lo, 0.0, 0.49))
    q_hi = float(np.clip(q_hi, q_lo + 1e-3, 1.0))

    candidates_raw = auto_up_cfg.get("candidate_axes")
    candidates: list[np.ndarray] = []
    if isinstance(candidates_raw, (list, tuple)):
        for item in candidates_raw:
            axis = _parse_vector3(item, np.zeros((3,), dtype=np.float64))
            axis = _normalize(axis)
            if float(np.linalg.norm(axis)) > 1e-9:
                candidates.append(axis)
    if not candidates:
        candidates = [
            np.array([1.0, 0.0, 0.0], dtype=np.float64),
            np.array([0.0, 1.0, 0.0], dtype=np.float64),
            np.array([0.0, 0.0, 1.0], dtype=np.float64),
        ]

    align_weight = float(auto_up_cfg.get("align_weight", 0.10))
    spans: list[float] = []
    scores: list[float] = []
    for axis in candidates:
        span = _robust_span(points_world, axis, q_lo=q_lo, q_hi=q_hi)
        align = abs(float(np.dot(axis, world_up_hint)))
        score = span * (1.0 + align_weight * align)
        spans.append(span)
        scores.append(score)
    best_idx = int(np.argmax(np.asarray(scores, dtype=np.float64)))
    world_up = candidates[best_idx].copy()
    if float(np.dot(world_up, world_up_hint)) < 0.0:
        world_up = -world_up

    info = {
        "mode": mode,
        "source": "auto_up",
        "hint": [float(v) for v in world_up_hint.tolist()],
        "selected_index": int(best_idx),
        "selected_axis": [float(v) for v in world_up.tolist()],
        "spans": [float(v) for v in spans],
        "scores": [float(v) for v in scores],
        "span_quantiles": [q_lo, q_hi],
        "align_weight": float(align_weight),
    }
    return world_up, info


def estimate_body_frame(points_world: np.ndarray, config: Mapping[str, Any] | None = None) -> Tuple[BodyFrame, Dict[str, Any]]:
    if points_world.ndim != 2 or points_world.shape[1] != 3:
        raise ValueError("Input points must be Nx3.")
    if points_world.shape[0] == 0:
        raise ValueError("Input points are empty.")

    world_up, world_up_info = _resolve_world_up(points_world, config)
    continuity_cfg: Mapping[str, Any] = {}
    prev_axes_world: np.ndarray | None = None
    continuity_enabled = True
    if isinstance(config, Mapping):
        continuity_raw = config.get("continuity")
        if isinstance(continuity_raw, Mapping):
            continuity_cfg = continuity_raw
        continuity_enabled = bool(continuity_cfg.get("enabled", True))
        prev_axes_world = _parse_axes_world(continuity_cfg.get("prev_axes_world"))
        if prev_axes_world is None:
            prev_axes_world = _parse_axes_world(config.get("prev_axes_world"))
    world_up = _normalize(world_up)
    if float(np.linalg.norm(world_up)) <= 1e-9:
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points_world.astype(np.float64))
    hull, _ = cloud.compute_convex_hull()
    obb = hull.get_oriented_bounding_box()

    center_obb = np.asarray(obb.center, dtype=np.float64)
    obb_extents = np.asarray(obb.extent, dtype=np.float64)
    obb_axes = np.asarray(obb.R, dtype=np.float64)
    raw_axes = [_normalize(obb_axes[:, i]) for i in range(3)]

    up_scores = [abs(float(np.dot(axis, world_up))) for axis in raw_axes]
    y_idx = int(np.argmax(np.array(up_scores, dtype=np.float64)))
    y_axis = raw_axes[y_idx]
    if float(np.dot(y_axis, world_up)) < 0.0:
        y_axis = -y_axis

    remain = [idx for idx in range(3) if idx != y_idx]
    spread0 = _axis_spread(points_world, raw_axes[remain[0]], center_obb)
    spread1 = _axis_spread(points_world, raw_axes[remain[1]], center_obb)
    x_idx = remain[0] if spread0 >= spread1 else remain[1]
    z_idx = remain[1] if x_idx == remain[0] else remain[0]

    x_axis = raw_axes[x_idx]
    z_axis = _normalize(np.cross(x_axis, y_axis))
    if float(np.linalg.norm(z_axis)) <= 1e-9:
        z_axis = raw_axes[z_idx]
    x_axis = _normalize(np.cross(y_axis, z_axis))
    z_axis = _normalize(np.cross(x_axis, y_axis))

    x_axis, y_axis, z_axis, horizontal_hint_info = _apply_horizontal_hint(
        x_axis,
        y_axis,
        z_axis,
        config,
    )

    continuity_axis_flips = {"x": False, "y": False, "z": False}
    if continuity_enabled and prev_axes_world is not None and prev_axes_world.shape == (3, 3):
        prev_x = _normalize(prev_axes_world[:, 0])
        prev_y = _normalize(prev_axes_world[:, 1])
        prev_z = _normalize(prev_axes_world[:, 2])

        if float(np.dot(y_axis, prev_y)) < 0.0:
            y_axis = -y_axis
            continuity_axis_flips["y"] = True
        if float(np.dot(y_axis, world_up)) < 0.0:
            y_axis = -y_axis
            continuity_axis_flips["y"] = not continuity_axis_flips["y"]

        if float(np.dot(x_axis, prev_x)) < 0.0:
            x_axis = -x_axis
            continuity_axis_flips["x"] = True
        z_axis = _normalize(np.cross(x_axis, y_axis))
        if float(np.linalg.norm(z_axis)) <= 1e-9:
            z_axis = raw_axes[z_idx]
        z_axis = _normalize(z_axis)
        if float(np.dot(z_axis, prev_z)) < 0.0:
            z_axis = -z_axis
            x_axis = -x_axis
            continuity_axis_flips["z"] = True
            continuity_axis_flips["x"] = not continuity_axis_flips["x"]
        x_axis = _normalize(np.cross(y_axis, z_axis))
        z_axis = _normalize(np.cross(x_axis, y_axis))

    axes = np.stack([x_axis, y_axis, z_axis], axis=1).astype(np.float64)
    det_axes = float(np.linalg.det(axes))
    if det_axes < 0.0:
        z_axis = -z_axis
        continuity_axis_flips["z"] = not continuity_axis_flips["z"]
        axes = np.stack([x_axis, y_axis, z_axis], axis=1).astype(np.float64)
        det_axes = float(np.linalg.det(axes))

    center, extents = _fit_center_and_extents(points_world, center_obb, axes)
    axes = np.stack([x_axis, y_axis, z_axis], axis=1).astype(np.float64)
    frame = BodyFrame(center_world=center, axes_world=axes, extents=extents)
    spread_xyz = np.std((points_world - center[None, :]) @ axes, axis=0)
    info = {
        "obb_center": [float(v) for v in center_obb.tolist()],
        "body_center": [float(v) for v in center.tolist()],
        "obb_extent": [float(v) for v in obb_extents.tolist()],
        "body_extent": [float(v) for v in extents.tolist()],
        "vertical_axis_index": int(y_idx),
        "horizontal_axis_index": int(x_idx),
        "depth_axis_index": int(z_idx),
        "axis_norms": [float(np.linalg.norm(axes[:, i])) for i in range(3)],
        "orthogonality_error": _orthogonality_error(axes),
        "handedness_determinant": det_axes,
        "up_alignment_cosine": float(np.dot(y_axis, world_up)),
        "world_up_info": world_up_info,
        "axis_spread": [float(v) for v in spread_xyz.tolist()],
        "continuity_enabled": bool(continuity_enabled),
        "continuity_prev_provided": bool(prev_axes_world is not None),
        "continuity_axis_flips": continuity_axis_flips,
        "horizontal_hint_info": horizontal_hint_info,
    }
    return frame, info
