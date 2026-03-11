# scripts/parser/relation_features.py
from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

from scripts.parser.types import PartBox


DEFAULT_RELATION_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("torso", "head"),
    ("torso", "left_upper_arm"),
    ("left_upper_arm", "left_lower_arm"),
    ("torso", "right_upper_arm"),
    ("right_upper_arm", "right_lower_arm"),
    ("torso", "left_thigh"),
    ("left_thigh", "left_shin"),
    ("torso", "right_thigh"),
    ("right_thigh", "right_shin"),
    ("left_thigh", "right_thigh"),
    ("left_shin", "right_shin"),
)


def _resolve_pairs(config: Mapping[str, Any] | None) -> Sequence[Tuple[str, str]]:
    if not config:
        return DEFAULT_RELATION_PAIRS
    rel_cfg = config.get("relations", {}) if isinstance(config, Mapping) else {}
    if not isinstance(rel_cfg, Mapping):
        return DEFAULT_RELATION_PAIRS
    pairs_raw = rel_cfg.get("pairs")
    if not isinstance(pairs_raw, list) or not pairs_raw:
        return DEFAULT_RELATION_PAIRS
    pairs = []
    for item in pairs_raw:
        if not isinstance(item, Mapping):
            raise TypeError("Each relation pair must be object.")
        a = str(item.get("a", "")).strip()
        b = str(item.get("b", "")).strip()
        if not a or not b:
            raise ValueError("Relation pair requires non-empty 'a' and 'b'.")
        pairs.append((a, b))
    return tuple(pairs)


def _safe_center(parts: Mapping[str, PartBox], names: Sequence[str]) -> np.ndarray | None:
    centers = [parts[name].center_xyz for name in names if name in parts and parts[name].valid]
    if not centers:
        return None
    return np.mean(np.stack(centers, axis=0), axis=0)


def _interval_overlap_ratio(a_min: float, a_max: float, b_min: float, b_max: float) -> float:
    inter = max(0.0, min(a_max, b_max) - max(a_min, b_min))
    a_len = max(a_max - a_min, 1e-8)
    b_len = max(b_max - b_min, 1e-8)
    denom = max(min(a_len, b_len), 1e-8)
    return float(inter / denom)


def _axis_gap(a_min: float, a_max: float, b_min: float, b_max: float) -> float:
    if a_max < b_min:
        return float(b_min - a_max)
    if b_max < a_min:
        return float(a_min - b_max)
    return 0.0


def _pair_box_metrics(box_a: PartBox, box_b: PartBox, norm_scale: float) -> Dict[str, float]:
    delta = (box_b.center_xyz - box_a.center_xyz).astype(np.float64)
    center_dist = float(np.linalg.norm(delta))

    overlap_x = _interval_overlap_ratio(box_a.min_xyz[0], box_a.max_xyz[0], box_b.min_xyz[0], box_b.max_xyz[0])
    overlap_y = _interval_overlap_ratio(box_a.min_xyz[1], box_a.max_xyz[1], box_b.min_xyz[1], box_b.max_xyz[1])
    overlap_z = _interval_overlap_ratio(box_a.min_xyz[2], box_a.max_xyz[2], box_b.min_xyz[2], box_b.max_xyz[2])

    gap_x = _axis_gap(box_a.min_xyz[0], box_a.max_xyz[0], box_b.min_xyz[0], box_b.max_xyz[0])
    gap_y = _axis_gap(box_a.min_xyz[1], box_a.max_xyz[1], box_b.min_xyz[1], box_b.max_xyz[1])
    gap_z = _axis_gap(box_a.min_xyz[2], box_a.max_xyz[2], box_b.min_xyz[2], box_b.max_xyz[2])
    surface_gap = float(np.linalg.norm(np.array([gap_x, gap_y, gap_z], dtype=np.float64)))
    contact = float(np.exp(-surface_gap / max(norm_scale * 0.2, 1e-6)))

    return {
        "dx": float(delta[0] / norm_scale),
        "dy": float(delta[1] / norm_scale),
        "dz": float(delta[2] / norm_scale),
        "dist": float(center_dist / norm_scale),
        "overlap_x": overlap_x,
        "overlap_y": overlap_y,
        "overlap_z": overlap_z,
        "surface_gap": float(surface_gap / norm_scale),
        "contact": contact,
    }


def _flatten_features(pair_features: Mapping[str, Mapping[str, Any]], derived: Mapping[str, float]) -> Dict[str, float]:
    flat: Dict[str, float] = {k: float(v) for k, v in derived.items()}
    for pair_key, item in pair_features.items():
        if not isinstance(item, Mapping):
            continue
        if not bool(item.get("valid", False)):
            continue
        for metric in ("dx", "dy", "dz", "dist", "overlap_x", "overlap_y", "overlap_z", "surface_gap", "contact"):
            if metric in item:
                flat[f"pair.{pair_key}.{metric}"] = float(item[metric])
    return flat


def compute_relation_features(
    parts: Mapping[str, PartBox],
    config: Mapping[str, Any] | None = None,
    prev_state: Mapping[str, Any] | None = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    valid_boxes = [box for box in parts.values() if box.valid]
    if not valid_boxes:
        raise ValueError("At least one valid part is required for relation feature extraction.")

    all_points = []
    for box in valid_boxes:
        all_points.extend([box.min_xyz, box.max_xyz])
    arr = np.stack(all_points, axis=0)
    body_height = max(float(np.max(arr[:, 1]) - np.min(arr[:, 1])), 1e-6)
    body_width = max(float(np.max(arr[:, 0]) - np.min(arr[:, 0])), 1e-6)
    norm_scale = max(body_height, body_width, 1e-6)

    pairs = _resolve_pairs(config)
    pair_features: Dict[str, Any] = {}
    for a_name, b_name in pairs:
        key = f"{a_name}__{b_name}"
        box_a = parts.get(a_name)
        box_b = parts.get(b_name)
        if box_a is None or box_b is None or (not box_a.valid) or (not box_b.valid):
            pair_features[key] = {"valid": False}
            continue
        pair_features[key] = {"valid": True, **_pair_box_metrics(box_a, box_b, norm_scale)}

    hip_center = _safe_center(parts, ["left_thigh", "right_thigh"])
    shoulder_center = _safe_center(parts, ["left_upper_arm", "right_upper_arm"])
    torso = parts.get("torso")
    derived: Dict[str, float] = {}

    if torso is not None and torso.valid and hip_center is not None:
        derived["pelvis_x_proxy"] = float((torso.center_xyz[2] - hip_center[2]) / norm_scale)
        derived["pelvis_y_proxy"] = float((torso.center_xyz[1] - hip_center[1]) / norm_scale)
        derived["pelvis_z_proxy"] = float((torso.center_xyz[0] - hip_center[0]) / norm_scale)

    def add_pair_metric(feature_name: str, pair_key: str, metric: str, default: float = 0.0) -> None:
        item = pair_features.get(pair_key, {})
        if isinstance(item, Mapping) and bool(item.get("valid", False)) and metric in item:
            derived[feature_name] = float(item[metric])
        else:
            derived[feature_name] = float(default)

    add_pair_metric("right_hip_x_proxy", "torso__right_thigh", "dy")
    add_pair_metric("right_hip_z_proxy", "torso__right_thigh", "dx")
    add_pair_metric("left_hip_x_proxy", "torso__left_thigh", "dy")
    add_pair_metric("left_hip_z_proxy", "torso__left_thigh", "dx")
    add_pair_metric("right_knee_overlap_y_proxy", "right_thigh__right_shin", "overlap_y")
    add_pair_metric("left_knee_overlap_y_proxy", "left_thigh__left_shin", "overlap_y")
    add_pair_metric("right_shoulder_x_proxy", "torso__right_upper_arm", "dy")
    add_pair_metric("right_shoulder_z_proxy", "torso__right_upper_arm", "dx")
    add_pair_metric("left_shoulder_x_proxy", "torso__left_upper_arm", "dy")
    add_pair_metric("left_shoulder_z_proxy", "torso__left_upper_arm", "dx")
    add_pair_metric("right_elbow_y_proxy", "right_upper_arm__right_lower_arm", "overlap_y")
    add_pair_metric("left_elbow_y_proxy", "left_upper_arm__left_lower_arm", "overlap_y")

    if shoulder_center is not None and torso is not None and torso.valid:
        lateral_offset = float((shoulder_center[0] - torso.center_xyz[0]) / norm_scale)
        derived["shoulder_lateral_offset_proxy"] = lateral_offset

    flattened = _flatten_features(pair_features, derived)

    prev_flat = {}
    if isinstance(prev_state, Mapping):
        pf = prev_state.get("flat_features")
        if isinstance(pf, Mapping):
            prev_flat = pf
    deltas: Dict[str, float] = {}
    for key, value in flattened.items():
        prev_value = prev_flat.get(key)
        if isinstance(prev_value, (float, int)):
            deltas[f"{key}_delta"] = float(value - float(prev_value))
    flattened_with_delta = dict(flattened)
    flattened_with_delta.update(deltas)

    relation_obj = {
        "norm_scale": float(norm_scale),
        "pair_features": pair_features,
        "derived_features": derived,
        "flat_features": flattened_with_delta,
    }
    next_state = {"flat_features": flattened}
    return relation_obj, next_state
