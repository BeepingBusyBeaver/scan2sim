# scripts/parser/attention_refiner.py
from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

import numpy as np

from scripts.parser.types import PartBox

_PARENT_CHILD_LINKS: Tuple[Tuple[str, str], ...] = (
    ("left_lower_arm", "left_upper_arm"),
    ("right_lower_arm", "right_upper_arm"),
    ("left_upper_arm", "torso"),
    ("right_upper_arm", "torso"),
    ("left_shin", "left_thigh"),
    ("right_shin", "right_thigh"),
    ("left_thigh", "torso"),
    ("right_thigh", "torso"),
    ("head", "torso"),
)

_LIMB_PARTS: Tuple[str, ...] = (
    "left_upper_arm",
    "left_lower_arm",
    "right_upper_arm",
    "right_lower_arm",
    "left_thigh",
    "left_shin",
    "right_thigh",
    "right_shin",
)


def _softmax_rows(logits: np.ndarray) -> np.ndarray:
    if logits.ndim != 2:
        raise ValueError("logits must be 2D (N, C)")
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_v = np.exp(shifted)
    denom = np.sum(exp_v, axis=1, keepdims=True)
    denom = np.maximum(denom, 1e-12)
    return exp_v / denom


def _box_membership(points: np.ndarray, box: PartBox) -> np.ndarray:
    if (not box.valid) or points.shape[0] == 0:
        return np.zeros((points.shape[0],), dtype=bool)
    return np.all((points >= box.min_xyz[None, :]) & (points <= box.max_xyz[None, :]), axis=1)


def _box_volume(box: PartBox) -> float:
    if not box.valid:
        return 0.0
    size = np.maximum(np.asarray(box.size_xyz, dtype=np.float64), 0.0)
    return float(size[0] * size[1] * size[2])


def _box_iou_axis_aligned(a: PartBox, b: PartBox) -> float:
    if (not a.valid) or (not b.valid):
        return 0.0
    inter_min = np.maximum(np.asarray(a.min_xyz, dtype=np.float64), np.asarray(b.min_xyz, dtype=np.float64))
    inter_max = np.minimum(np.asarray(a.max_xyz, dtype=np.float64), np.asarray(b.max_xyz, dtype=np.float64))
    inter_size = np.maximum(inter_max - inter_min, 0.0)
    inter_vol = float(inter_size[0] * inter_size[1] * inter_size[2])
    if inter_vol <= 0.0:
        return 0.0
    va = _box_volume(a)
    vb = _box_volume(b)
    union = max(va + vb - inter_vol, 1e-9)
    return float(inter_vol / union)


def _axis_gap(child: PartBox, parent: PartBox) -> float:
    if (not child.valid) or (not parent.valid):
        return 0.0
    sep = np.maximum(np.asarray(child.min_xyz, dtype=np.float64) - np.asarray(parent.max_xyz, dtype=np.float64), 0.0)
    sep2 = np.maximum(np.asarray(parent.min_xyz, dtype=np.float64) - np.asarray(child.max_xyz, dtype=np.float64), 0.0)
    gap = np.maximum(sep, sep2)
    return float(np.linalg.norm(gap))


def _structural_score(parts: Mapping[str, PartBox]) -> float:
    valid_count = float(sum(1 for p in parts.values() if p.valid))
    gap_penalty = 0.0
    for child_name, parent_name in _PARENT_CHILD_LINKS:
        child = parts.get(child_name)
        parent = parts.get(parent_name)
        if child is None or parent is None:
            continue
        gap_penalty += _axis_gap(child, parent)

    torso = parts.get("torso")
    torso_vol = _box_volume(torso) if torso is not None else 0.0
    limb_penalty = 0.0
    if torso_vol > 1e-8:
        for name in _LIMB_PARTS:
            part = parts.get(name)
            if part is None or (not part.valid):
                continue
            ratio = _box_volume(part) / torso_vol
            if ratio > 2.0:
                limb_penalty += (ratio - 2.0)

    return float(valid_count - 2.0 * gap_penalty - 0.5 * limb_penalty)


def refine_parts_with_attention(
    *,
    points_body: np.ndarray,
    parts: Mapping[str, PartBox],
    part_points: Mapping[str, np.ndarray],
    config: Mapping[str, Any] | None,
    parser_cfg: Mapping[str, Any] | None = None,
) -> Tuple[Dict[str, PartBox], Dict[str, np.ndarray], Dict[str, Any]]:
    cfg = dict(config or {})
    enabled = bool(cfg.get("enabled", False))
    if not enabled:
        return dict(parts), {k: np.asarray(v, dtype=np.float64) for k, v in part_points.items()}, {"enabled": False, "applied": False}
    if points_body.ndim != 2 or points_body.shape[1] != 3 or points_body.shape[0] == 0:
        return dict(parts), {k: np.asarray(v, dtype=np.float64) for k, v in part_points.items()}, {"enabled": True, "applied": False, "reason": "invalid_points"}

    parser_cfg_local = dict(parser_cfg or {})
    box_post_cfg = parser_cfg_local.get("box_postprocess", {})
    if not isinstance(box_post_cfg, Mapping):
        box_post_cfg = {}
    min_points_per_part = int(cfg.get("min_points_per_part", parser_cfg_local.get("min_points_per_part", 8)))
    allow_small_parts = bool(cfg.get("allow_small_parts", box_post_cfg.get("allow_small_parts", True)))
    small_part_min_size = float(cfg.get("small_part_min_size", box_post_cfg.get("small_part_min_size", 0.025)))
    robust_quantile = float(cfg.get("robust_quantile", box_post_cfg.get("robust_quantile", 0.03)))
    min_keep_ratio = float(cfg.get("min_keep_ratio", 0.45))
    preserve_invalid = bool(cfg.get("preserve_invalid", True))
    guard_cfg = cfg.get("guard", {})
    if not isinstance(guard_cfg, Mapping):
        guard_cfg = {}

    temperature = float(cfg.get("temperature", 0.55))
    in_box_bonus = float(cfg.get("in_box_bonus", 0.35))
    y_prior_weight = float(cfg.get("y_prior_weight", 0.15))
    min_confidence = float(cfg.get("min_confidence", 0.45))
    iterations = max(1, int(cfg.get("iterations", 2)))
    guard_enabled = bool(guard_cfg.get("enabled", True))
    guard_min_iou = float(guard_cfg.get("min_iou", 0.10))
    guard_max_size_ratio = float(guard_cfg.get("max_size_ratio", 2.0))
    guard_max_center_shift = float(guard_cfg.get("max_center_shift_norm", 1.5))
    guard_global_tolerance = float(guard_cfg.get("global_tolerance", 0.25))

    points = np.asarray(points_body, dtype=np.float64)

    candidate_names = [
        name
        for name, box in parts.items()
        if getattr(box, "valid", False) and name in part_points and np.asarray(part_points[name]).shape[0] > 0
    ]
    if len(candidate_names) < 2:
        return dict(parts), {k: np.asarray(v, dtype=np.float64) for k, v in part_points.items()}, {
            "enabled": True,
            "applied": False,
            "reason": "insufficient_candidates",
            "candidate_parts": int(len(candidate_names)),
        }

    centers = []
    scales = []
    y_means = []
    y_stds = []
    in_box_masks = []
    for name in candidate_names:
        pts = np.asarray(part_points[name], dtype=np.float64)
        center = np.mean(pts, axis=0)
        dist = np.linalg.norm(pts - center[None, :], axis=1)
        scale = max(float(np.percentile(dist, 60.0)), 0.03)
        centers.append(center)
        scales.append(scale)
        y_means.append(float(np.mean(pts[:, 1])))
        y_stds.append(max(float(np.std(pts[:, 1])), 0.05))
        in_box_masks.append(_box_membership(points, parts[name]))

    centers_arr = np.stack(centers, axis=0)
    scales_arr = np.asarray(scales, dtype=np.float64)
    y_means_arr = np.asarray(y_means, dtype=np.float64)
    y_stds_arr = np.asarray(y_stds, dtype=np.float64)
    in_box = np.stack(in_box_masks, axis=1).astype(np.float64)

    diff0 = points[:, None, :] - centers_arr[None, :, :]
    d2_base = np.sum(diff0 * diff0, axis=2)
    base_assign = np.argmin(d2_base, axis=1).astype(np.int32)
    assign = base_assign.copy()
    conf = np.ones((points.shape[0],), dtype=np.float64)

    for _ in range(iterations):
        diff = points[:, None, :] - centers_arr[None, :, :]
        d2 = np.sum(diff * diff, axis=2)
        scale2 = np.maximum(scales_arr * scales_arr, 1e-6)
        logits = -(d2 / scale2[None, :]) / max(temperature, 1e-4)
        y_delta = np.abs(points[:, 1][:, None] - y_means_arr[None, :]) / y_stds_arr[None, :]
        logits = logits - max(y_prior_weight, 0.0) * y_delta
        logits = logits + max(in_box_bonus, 0.0) * in_box
        probs = _softmax_rows(logits)
        assign = np.argmax(probs, axis=1).astype(np.int32)
        conf = np.max(probs, axis=1)
        low_conf = conf < max(min_confidence, 0.0)
        if np.any(low_conf):
            assign[low_conf] = base_assign[low_conf]

        for part_idx in range(len(candidate_names)):
            sel = assign == part_idx
            if int(np.count_nonzero(sel)) < 3:
                continue
            pts = points[sel]
            center = np.mean(pts, axis=0)
            dist = np.linalg.norm(pts - center[None, :], axis=1)
            centers_arr[part_idx] = center
            scales_arr[part_idx] = max(float(np.percentile(dist, 60.0)), 0.03)
            y_means_arr[part_idx] = float(np.mean(pts[:, 1]))
            y_stds_arr[part_idx] = max(float(np.std(pts[:, 1])), 0.05)

    refined_part_points: Dict[str, np.ndarray] = {
        name: np.asarray(points_arr, dtype=np.float64)
        for name, points_arr in part_points.items()
    }
    reassigned_points = 0
    for part_idx, name in enumerate(candidate_names):
        selected = points[assign == part_idx]
        orig = refined_part_points.get(name, np.zeros((0, 3), dtype=np.float64))
        min_keep = max(min_points_per_part, int(np.ceil(orig.shape[0] * max(min_keep_ratio, 0.0))))
        if selected.shape[0] >= min_keep:
            refined_part_points[name] = selected
            reassigned_points += int(selected.shape[0])

    refined_parts: Dict[str, PartBox] = {}
    for name, box in parts.items():
        pts = refined_part_points.get(name, np.zeros((0, 3), dtype=np.float64))
        new_box = PartBox.from_points(
            name,
            pts,
            min_points=min_points_per_part,
            allow_small=allow_small_parts,
            min_size=small_part_min_size,
            robust_quantile=robust_quantile,
        )
        if preserve_invalid and (not new_box.valid) and box.valid:
            refined_parts[name] = box
        else:
            refined_parts[name] = new_box

    reverted_parts = 0
    if guard_enabled:
        for name in candidate_names:
            old_box = parts.get(name)
            new_box = refined_parts.get(name)
            if old_box is None or new_box is None:
                continue
            if (not old_box.valid) or (not new_box.valid):
                continue
            old_vol = _box_volume(old_box)
            new_vol = _box_volume(new_box)
            size_ratio = max(new_vol / max(old_vol, 1e-9), old_vol / max(new_vol, 1e-9))
            iou = _box_iou_axis_aligned(old_box, new_box)
            old_norm = max(float(np.linalg.norm(old_box.size_xyz)), 1e-6)
            center_shift = float(np.linalg.norm(new_box.center_xyz - old_box.center_xyz) / old_norm)
            if (iou < guard_min_iou) or (size_ratio > guard_max_size_ratio) or (center_shift > guard_max_center_shift):
                refined_parts[name] = old_box
                refined_part_points[name] = np.asarray(part_points[name], dtype=np.float64)
                reverted_parts += 1

        before_score = _structural_score(parts)
        after_score = _structural_score(refined_parts)
        if (after_score + guard_global_tolerance) < before_score:
            return dict(parts), {k: np.asarray(v, dtype=np.float64) for k, v in part_points.items()}, {
                "enabled": True,
                "applied": False,
                "reason": "global_guard_reject",
                "score_before": float(before_score),
                "score_after": float(after_score),
                "global_tolerance": float(guard_global_tolerance),
            }
    else:
        before_score = _structural_score(parts)
        after_score = _structural_score(refined_parts)

    meta = {
        "enabled": True,
        "applied": True,
        "candidate_parts": int(len(candidate_names)),
        "iterations": int(iterations),
        "temperature": float(temperature),
        "in_box_bonus": float(in_box_bonus),
        "y_prior_weight": float(y_prior_weight),
        "min_confidence": float(min_confidence),
        "reassigned_points": int(reassigned_points),
        "guard_enabled": bool(guard_enabled),
        "guard_reverted_parts": int(reverted_parts),
        "score_before": float(before_score),
        "score_after": float(after_score),
    }
    return refined_parts, refined_part_points, meta
