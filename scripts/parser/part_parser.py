from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Tuple

import numpy as np
import yaml

from scripts.parser.types import PartBox


DEFAULT_PART_ORDER: Tuple[str, ...] = (
    "head",
    "torso",
    "left_upper_arm",
    "left_lower_arm",
    "right_upper_arm",
    "right_lower_arm",
    "left_thigh",
    "left_shin",
    "right_thigh",
    "right_shin",
)


def _default_config() -> Dict[str, Any]:
    return {
        "parser": {
            "left_is_positive_x": True,
            "min_points_per_part": 8,
            "torso": {
                "y_range": [0.46, 0.78],
                "x_scale": 1.2,
            },
            "head": {
                "y_range": [0.80, 1.02],
                "x_scale": 1.25,
            },
            "arm": {
                "y_range": [0.38, 0.82],
                "side_min_scale": 0.65,
            },
            "leg": {
                "y_range": [0.00, 0.44],
                "side_min_scale": 0.15,
            },
            "continuity": {
                "enabled": True,
                "blend_with_prev": 0.25,
            },
        }
    }


def load_part_parser_config(path: Path | None) -> Dict[str, Any]:
    cfg = _default_config()
    if path is None:
        return cfg
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise TypeError(f"Part parser config root must be object: {path}")
    for key, value in loaded.items():
        if key != "parser":
            cfg[key] = value
            continue
        if value is None:
            continue
        if not isinstance(value, dict):
            raise TypeError(f"'parser' section must be object: {path}")
        for parser_key, parser_value in value.items():
            if isinstance(parser_value, dict) and isinstance(cfg["parser"].get(parser_key), dict):
                cfg["parser"][parser_key].update(parser_value)
            else:
                cfg["parser"][parser_key] = parser_value
    return cfg


def _empty_points() -> np.ndarray:
    return np.zeros((0, 3), dtype=np.float64)


def _normalize_y(points: np.ndarray) -> Tuple[np.ndarray, float, float]:
    y = points[:, 1]
    y_min = float(np.min(y))
    y_max = float(np.max(y))
    height = max(y_max - y_min, 1e-6)
    y_norm = (y - y_min) / height
    return y_norm, y_min, y_max


def _torso_anchor(points: np.ndarray, y_norm: np.ndarray, torso_cfg: Mapping[str, Any]) -> Tuple[np.ndarray, float]:
    y_range = torso_cfg.get("y_range", [0.46, 0.78])
    x_scale = float(torso_cfg.get("x_scale", 1.2))
    y_lo, y_hi = float(y_range[0]), float(y_range[1])
    mid_mask = (y_norm >= y_lo) & (y_norm <= y_hi)
    mid_points = points[mid_mask]
    if mid_points.shape[0] < 20:
        mid_points = points
    center = np.median(mid_points, axis=0).astype(np.float64)
    abs_x = np.abs(mid_points[:, 0] - center[0])
    torso_half_width = float(np.percentile(abs_x, 75)) if mid_points.shape[0] > 0 else 0.08
    torso_half_width = max(torso_half_width, 0.06) * max(x_scale, 0.4)
    return center, torso_half_width


def _split_side(points: np.ndarray, center_x: float, left_is_positive_x: bool) -> Tuple[np.ndarray, np.ndarray]:
    delta = points[:, 0] - center_x
    if left_is_positive_x:
        left = points[delta >= 0.0]
        right = points[delta < 0.0]
    else:
        left = points[delta <= 0.0]
        right = points[delta > 0.0]
    return left, right


def _principal_axis(points: np.ndarray) -> np.ndarray:
    if points.shape[0] < 3:
        return np.array([0.0, -1.0, 0.0], dtype=np.float64)
    centered = points - np.mean(points, axis=0, keepdims=True)
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    vec = eigvecs[:, int(np.argmax(eigvals))]
    if vec[1] > 0.0:
        vec = -vec
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 1e-8 else np.array([0.0, -1.0, 0.0], dtype=np.float64)


def _assign_proximal_distal(
    points: np.ndarray,
    anchor: np.ndarray,
    prev_prox: np.ndarray | None,
    prev_dist: np.ndarray | None,
) -> Tuple[np.ndarray, np.ndarray]:
    if points.shape[0] < 2:
        return points, _empty_points()
    axis = _principal_axis(points)
    proj = points @ axis
    threshold = float(np.median(proj))
    g0 = points[proj <= threshold]
    g1 = points[proj > threshold]
    if g0.shape[0] == 0 or g1.shape[0] == 0:
        half = points.shape[0] // 2
        g0 = points[:half]
        g1 = points[half:]

    c0 = np.mean(g0, axis=0)
    c1 = np.mean(g1, axis=0)
    prox0 = float(np.linalg.norm(c0 - anchor))
    prox1 = float(np.linalg.norm(c1 - anchor))
    proximal, distal = (g0, g1) if prox0 <= prox1 else (g1, g0)

    if prev_prox is not None and prev_dist is not None and proximal.shape[0] > 0 and distal.shape[0] > 0:
        cp = np.mean(proximal, axis=0)
        cd = np.mean(distal, axis=0)
        keep_cost = float(np.linalg.norm(cp - prev_prox) + np.linalg.norm(cd - prev_dist))
        swap_cost = float(np.linalg.norm(cd - prev_prox) + np.linalg.norm(cp - prev_dist))
        if swap_cost + 1e-6 < keep_cost:
            proximal, distal = distal, proximal

    return proximal, distal


def _prev_center(prev_state: Mapping[str, Any] | None, name: str) -> np.ndarray | None:
    if not isinstance(prev_state, Mapping):
        return None
    prev_parts = prev_state.get("part_centers")
    if not isinstance(prev_parts, Mapping):
        return None
    value = prev_parts.get(name)
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    return np.array(value, dtype=np.float64)


def parse_parts_from_points(
    points_body: np.ndarray,
    config: Mapping[str, Any],
    prev_state: Mapping[str, Any] | None = None,
) -> Tuple[Dict[str, PartBox], Dict[str, Any], Dict[str, np.ndarray], Dict[str, Any]]:
    if points_body.ndim != 2 or points_body.shape[1] != 3:
        raise ValueError("Input points must be Nx3 in body frame.")
    if points_body.shape[0] == 0:
        raise ValueError("Input points are empty.")
    finite_mask = np.isfinite(points_body).all(axis=1)
    points = points_body[finite_mask]
    if points.shape[0] == 0:
        raise ValueError("Input points contain no finite values.")

    parser_cfg = config.get("parser", {}) if isinstance(config, Mapping) else {}
    if not isinstance(parser_cfg, Mapping):
        raise TypeError("config.parser must be object.")

    left_is_positive_x = bool(parser_cfg.get("left_is_positive_x", True))
    min_points_per_part = int(parser_cfg.get("min_points_per_part", 8))
    continuity_cfg = parser_cfg.get("continuity", {})
    if not isinstance(continuity_cfg, Mapping):
        continuity_cfg = {}
    continuity_enabled = bool(continuity_cfg.get("enabled", True))
    blend_prev = float(continuity_cfg.get("blend_with_prev", 0.25))

    y_norm, y_min, y_max = _normalize_y(points)
    torso_center, torso_half_width = _torso_anchor(
        points,
        y_norm,
        parser_cfg.get("torso", {}) if isinstance(parser_cfg.get("torso"), Mapping) else {},
    )

    head_cfg = parser_cfg.get("head", {}) if isinstance(parser_cfg.get("head"), Mapping) else {}
    head_y = head_cfg.get("y_range", [0.8, 1.02])
    head_x_scale = float(head_cfg.get("x_scale", 1.25))

    torso_cfg = parser_cfg.get("torso", {}) if isinstance(parser_cfg.get("torso"), Mapping) else {}
    torso_y = torso_cfg.get("y_range", [0.46, 0.78])

    arm_cfg = parser_cfg.get("arm", {}) if isinstance(parser_cfg.get("arm"), Mapping) else {}
    arm_y = arm_cfg.get("y_range", [0.38, 0.82])
    arm_side_min = float(arm_cfg.get("side_min_scale", 0.65))

    leg_cfg = parser_cfg.get("leg", {}) if isinstance(parser_cfg.get("leg"), Mapping) else {}
    leg_y = leg_cfg.get("y_range", [0.0, 0.44])
    leg_side_min = float(leg_cfg.get("side_min_scale", 0.15))

    torso_mask = (
        (y_norm >= float(torso_y[0]))
        & (y_norm <= float(torso_y[1]))
        & (np.abs(points[:, 0] - torso_center[0]) <= torso_half_width)
    )
    torso_points = points[torso_mask]
    if torso_points.shape[0] < min_points_per_part:
        torso_points = points[(y_norm >= float(torso_y[0])) & (y_norm <= float(torso_y[1]))]
    if torso_points.shape[0] > 0:
        torso_center = np.mean(torso_points, axis=0)

    head_mask = (
        (y_norm >= float(head_y[0]))
        & (y_norm <= float(head_y[1]))
        & (np.abs(points[:, 0] - torso_center[0]) <= torso_half_width * head_x_scale)
    )
    head_points = points[head_mask]

    arm_mask = (y_norm >= float(arm_y[0])) & (y_norm <= float(arm_y[1]))
    arm_points = points[arm_mask]
    leg_mask = (y_norm >= float(leg_y[0])) & (y_norm <= float(leg_y[1]))
    leg_points = points[leg_mask]

    left_arm_all, right_arm_all = _split_side(arm_points, torso_center[0], left_is_positive_x)
    left_leg_all, right_leg_all = _split_side(leg_points, torso_center[0], left_is_positive_x)

    left_arm_all = left_arm_all[np.abs(left_arm_all[:, 0] - torso_center[0]) >= torso_half_width * arm_side_min]
    right_arm_all = right_arm_all[np.abs(right_arm_all[:, 0] - torso_center[0]) >= torso_half_width * arm_side_min]
    left_leg_all = left_leg_all[np.abs(left_leg_all[:, 0] - torso_center[0]) >= torso_half_width * leg_side_min]
    right_leg_all = right_leg_all[np.abs(right_leg_all[:, 0] - torso_center[0]) >= torso_half_width * leg_side_min]

    left_upper, left_lower = _assign_proximal_distal(
        left_arm_all,
        anchor=torso_center,
        prev_prox=_prev_center(prev_state, "left_upper_arm") if continuity_enabled else None,
        prev_dist=_prev_center(prev_state, "left_lower_arm") if continuity_enabled else None,
    )
    right_upper, right_lower = _assign_proximal_distal(
        right_arm_all,
        anchor=torso_center,
        prev_prox=_prev_center(prev_state, "right_upper_arm") if continuity_enabled else None,
        prev_dist=_prev_center(prev_state, "right_lower_arm") if continuity_enabled else None,
    )
    left_thigh, left_shin = _assign_proximal_distal(
        left_leg_all,
        anchor=torso_center,
        prev_prox=_prev_center(prev_state, "left_thigh") if continuity_enabled else None,
        prev_dist=_prev_center(prev_state, "left_shin") if continuity_enabled else None,
    )
    right_thigh, right_shin = _assign_proximal_distal(
        right_leg_all,
        anchor=torso_center,
        prev_prox=_prev_center(prev_state, "right_thigh") if continuity_enabled else None,
        prev_dist=_prev_center(prev_state, "right_shin") if continuity_enabled else None,
    )

    part_points: Dict[str, np.ndarray] = {
        "head": head_points,
        "torso": torso_points,
        "left_upper_arm": left_upper,
        "left_lower_arm": left_lower,
        "right_upper_arm": right_upper,
        "right_lower_arm": right_lower,
        "left_thigh": left_thigh,
        "left_shin": left_shin,
        "right_thigh": right_thigh,
        "right_shin": right_shin,
    }

    boxes: Dict[str, PartBox] = {}
    part_centers: Dict[str, np.ndarray] = {}
    for name in DEFAULT_PART_ORDER:
        pts = part_points.get(name, _empty_points())
        box = PartBox.from_points(name, pts, min_points=min_points_per_part)
        prev_center = _prev_center(prev_state, name) if continuity_enabled else None
        if box.valid and prev_center is not None and 0.0 < blend_prev < 1.0:
            blend_center = (1.0 - blend_prev) * box.center_xyz + blend_prev * prev_center
            box = PartBox(
                name=box.name,
                valid=box.valid,
                num_points=box.num_points,
                min_xyz=box.min_xyz,
                max_xyz=box.max_xyz,
                center_xyz=blend_center,
                size_xyz=box.size_xyz,
            )
        boxes[name] = box
        if box.valid:
            part_centers[name] = box.center_xyz

    info = {
        "input_points": int(points.shape[0]),
        "valid_parts": int(sum(1 for box in boxes.values() if box.valid)),
        "torso_center": [float(v) for v in torso_center.tolist()],
        "torso_half_width": float(torso_half_width),
        "y_min": float(y_min),
        "y_max": float(y_max),
        "left_is_positive_x": left_is_positive_x,
        "continuity_enabled": continuity_enabled,
    }
    next_state = {"part_centers": {k: [float(c) for c in v.tolist()] for k, v in part_centers.items()}}
    return boxes, info, part_points, next_state
