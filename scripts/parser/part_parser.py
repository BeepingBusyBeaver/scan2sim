# scripts/parser/part_parser.py
from __future__ import annotations

import heapq
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Tuple

import numpy as np
import open3d as o3d
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
                "z_scale": 2.0,
                "x_percentile": 75.0,
                "z_percentile": 80.0,
                "limb_exclusion": {
                    "enabled": True,
                    "assign_to_torso": True,
                    "y_pad": 0.05,
                    "x_scale": 1.55,
                    "z_scale": 1.30,
                },
            },
            "torso_arm_salvage": {
                "enabled": True,
                "y_margin": 0.03,
                "edge_percentile": 0.72,
                "min_lateral": 0.055,
                "max_ratio": 0.28,
                "min_points": 12,
                "max_shoulder_dist": 0.42,
            },
            "head": {
                "y_range": [0.80, 1.02],
                "x_scale": 1.25,
                "z_scale": 1.6,
                "center_keep_quantile": 0.65,
                "max_radius": 1.15,
                "top_band_scale": 0.12,
                "keep_drop_scale": 0.30,
                "core_x_scale": 2.2,
                "core_z_scale": 2.4,
                "core_min_keep_ratio": 0.35,
            },
            "arm": {
                "y_range": [0.38, 0.82],
                "side_min_scale": 0.65,
            },
            "leg": {
                "y_range": [0.00, 0.44],
                "side_min_scale": 0.15,
            },
            "limb_assignment": {
                "shoulder_quantile": 0.80,
                "hip_quantile": 0.20,
                "arm_vertical_penalty": 0.90,
                "leg_vertical_penalty": 0.90,
                "min_side_scale": 0.08,
                "y_prior_weight": 0.70,
                "top_arm_seed_fraction": 0.12,
                "bottom_leg_seed_fraction": 0.18,
                "cluster_eps": 0.085,
                "cluster_min_points": 6,
                "seed_conf_fraction": 0.10,
                "geodesic_k": 10,
                "geodesic_radius_scale": 1.8,
                "geodesic_score_weight": 0.35,
            },
            "arm_absence_guard": {
                "enabled": True,
                "min_arm_points": 10,
                "min_side_ratio": 0.16,
                "min_median_y_norm": 0.42,
                "max_shoulder_attach_dist": 0.24,
                "attach_quantile": 0.30,
            },
            "segment_split": {
                "arm_prox_quantile": 0.55,
                "leg_prox_quantile": 0.50,
                "arm_prox_reclaim": {
                    "enabled": True,
                    "prox_quantile": 0.85,
                    "margin": 0.015,
                    "max_move_ratio": 0.35,
                },
            },
            "boundary_refine": {
                "enabled": True,
                "unresolved_max_dist": 0.09,
                "head_cap": {
                    "enabled": True,
                    "y_margin_low": 0.04,
                    "x_scale": 1.35,
                    "z_scale": 1.95,
                },
                "torso_arm_reclaim": {
                    "enabled": True,
                    "y_margin": 0.03,
                    "lateral_x_scale": 0.70,
                    "frontal_z_scale": 0.70,
                    "max_arm_attach_dist": 0.11,
                    "arm_vs_torso_margin": 0.008,
                },
                "torso_leg_reclaim": {
                    "enabled": True,
                    "y_margin": 0.03,
                    "max_leg_attach_dist": 0.12,
                    "leg_vs_torso_margin": 0.004,
                },
            },
            "part_grouping": {
                "enabled": True,
                "cluster_eps": 0.070,
                "cluster_min_points": 6,
                "noise_attach_radius": 0.045,
                "min_keep_ratio": 0.50,
                "secondary_center_radius": 0.10,
                "proximal_min_keep_ratio": 0.40,
                "proximal_max_anchor_quantile": 0.88,
                "distal_min_keep_ratio": 0.35,
                "distal_secondary_center_radius": 0.06,
                "distal_noise_attach_radius": 0.025,
                "distal_fallback_to_input": True,
                "distal_max_anchor_quantile": 0.90,
            },
            "hierarchy": {
                "head_arm_guard": {
                    "enabled": True,
                    "y_pad": 0.03,
                    "radius_scale_x": 1.25,
                    "radius_scale_y": 1.15,
                    "radius_scale_z": 1.25,
                },
                "arm_torso_guard": {
                    "enabled": True,
                    "y_pad": 0.05,
                    "x_scale": 1.45,
                    "z_scale": 1.65,
                    "shoulder_keep_radius_scale": 0.95,
                    "shoulder_outward_min": 0.015,
                    "min_keep_ratio": 0.45,
                },
            },
            "temporal_box_stabilization": {
                "enabled": True,
                "center_alpha": 0.65,
                "size_alpha": 0.55,
                "max_center_step_ratio": 0.85,
                "max_size_delta_ratio": 0.35,
                "min_box_diag": 0.08,
            },
            "box_postprocess": {
                "enabled": True,
                "contact_mode": "mixed",
                "allow_small_parts": True,
                "small_part_min_size": 0.025,
                "robust_quantile": 0.03,
                "contact_margin": 0.0,
                "forbidden_margin": 0.003,
                "limb_shape": {
                    "enabled": True,
                    "align_to_links": True,
                    "direction_blend": 0.70,
                    "leg_direction_blend_scale": 0.55,
                    "use_torso_secondary_arms": True,
                    "use_torso_secondary_legs": False,
                    "max_turn_deg_default": 45.0,
                    "max_turn_deg": {
                        "left_upper_arm": 45.0,
                        "right_upper_arm": 45.0,
                        "left_lower_arm": 55.0,
                        "right_lower_arm": 55.0,
                        "left_thigh": 24.0,
                        "right_thigh": 24.0,
                        "left_shin": 32.0,
                        "right_shin": 32.0,
                    },
                    "quantile": 0.04,
                    "min_long_axis": 0.085,
                    "min_aspect_ratio": 1.45,
                    "max_cross_axis": 0.16,
                },
                "hard_contact": {
                    "enabled": True,
                    "mode": "shift",
                    "iterations": 2,
                    "min_overlap": 0.004,
                    "max_expand_per_axis": 0.05,
                    "projection_contact": True,
                    "projection_max_shift": 0.12,
                    "links": [
                        {"child": "head", "parent": "torso", "min_overlap": 0.002},
                        {"child": "left_upper_arm", "parent": "torso", "min_overlap": 0.003},
                        {"child": "left_lower_arm", "parent": "left_upper_arm"},
                        {"child": "right_upper_arm", "parent": "torso", "min_overlap": 0.003},
                        {"child": "right_lower_arm", "parent": "right_upper_arm"},
                        {"child": "left_thigh", "parent": "torso", "min_overlap": 0.003},
                        {"child": "left_shin", "parent": "left_thigh"},
                        {"child": "right_thigh", "parent": "torso", "min_overlap": 0.003},
                        {"child": "right_shin", "parent": "right_thigh"},
                    ],
                },
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


def _torso_anchor(points: np.ndarray, y_norm: np.ndarray, torso_cfg: Mapping[str, Any]) -> Tuple[np.ndarray, float, float]:
    y_range = torso_cfg.get("y_range", [0.46, 0.78])
    x_scale = float(torso_cfg.get("x_scale", 1.2))
    z_scale = float(torso_cfg.get("z_scale", 2.0))
    x_percentile = float(torso_cfg.get("x_percentile", 75.0))
    z_percentile = float(torso_cfg.get("z_percentile", 80.0))
    y_lo, y_hi = float(y_range[0]), float(y_range[1])
    mid_mask = (y_norm >= y_lo) & (y_norm <= y_hi)
    mid_points = points[mid_mask]
    if mid_points.shape[0] < 20:
        mid_points = points
    center = np.median(mid_points, axis=0).astype(np.float64)
    abs_x = np.abs(mid_points[:, 0] - center[0])
    abs_z = np.abs(mid_points[:, 2] - center[2])
    torso_half_width = float(np.percentile(abs_x, x_percentile)) if mid_points.shape[0] > 0 else 0.08
    torso_half_width = max(torso_half_width, 0.06) * max(x_scale, 0.4)
    torso_half_depth = float(np.percentile(abs_z, z_percentile)) if mid_points.shape[0] > 0 else 0.05
    torso_half_depth = max(torso_half_depth, 0.04) * max(z_scale, 0.4)
    return center, torso_half_width, torso_half_depth


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
    proximal, distal = _swap_with_prev_state(proximal, distal, prev_prox, prev_dist)
    return proximal, distal


def _swap_with_prev_state(
    proximal: np.ndarray,
    distal: np.ndarray,
    prev_prox: np.ndarray | None,
    prev_dist: np.ndarray | None,
) -> Tuple[np.ndarray, np.ndarray]:
    if prev_prox is None or prev_dist is None or proximal.shape[0] == 0 or distal.shape[0] == 0:
        return proximal, distal
    cp = np.mean(proximal, axis=0)
    cd = np.mean(distal, axis=0)
    keep_cost = float(np.linalg.norm(cp - prev_prox) + np.linalg.norm(cd - prev_dist))
    swap_cost = float(np.linalg.norm(cd - prev_prox) + np.linalg.norm(cp - prev_dist))
    min_improve_ratio = 0.05
    if swap_cost + 1e-6 < keep_cost * (1.0 - min_improve_ratio):
        return distal, proximal
    return proximal, distal


def _assign_proximal_distal_by_anchor(
    points: np.ndarray,
    anchor: np.ndarray,
    quantile: float,
    prev_prox: np.ndarray | None,
    prev_dist: np.ndarray | None,
) -> Tuple[np.ndarray, np.ndarray]:
    if points.shape[0] < 2:
        return points, _empty_points()
    q = float(np.clip(quantile, 0.2, 0.8))
    d = np.linalg.norm(points - anchor[None, :], axis=1)
    n_points = int(points.shape[0])
    seed_n = max(1, min(n_points - 1, int(np.ceil(n_points * 0.12))))
    proximal_seed = np.argsort(d)[:seed_n]

    k_geo = max(3, min(12, int(np.ceil(np.sqrt(max(n_points, 1)))) + 2))
    radius_geo = max(float(np.quantile(d, 0.60)) * 0.55, 0.03)
    graph = _build_neighbor_graph(points, k=k_geo, radius=radius_geo)
    geo_dist = _multi_source_dijkstra(graph, proximal_seed.tolist())
    geo_valid = np.isfinite(geo_dist)
    proximal = _empty_points()
    distal = _empty_points()
    if int(np.count_nonzero(geo_valid)) >= max(4, int(0.6 * n_points)):
        geo_threshold = float(np.quantile(geo_dist[geo_valid], q))
        proximal = points[geo_dist <= geo_threshold]
        distal = points[geo_dist > geo_threshold]

    if proximal.shape[0] == 0 or distal.shape[0] == 0:
        distal_dir = _estimate_distal_direction(points, anchor, d)
        proj = (points - anchor[None, :]) @ distal_dir
        threshold = float(np.quantile(proj, q))
        proximal = points[proj <= threshold]
        distal = points[proj > threshold]
        if proximal.shape[0] == 0 or distal.shape[0] == 0:
            order = np.argsort(proj)
            split_idx = max(1, min(points.shape[0] - 1, int(np.ceil(points.shape[0] * q))))
            proximal = points[order[:split_idx]]
            distal = points[order[split_idx:]]
    proximal, distal = _swap_with_prev_state(proximal, distal, prev_prox, prev_dist)
    proximal, distal = _enforce_anchor_order(proximal, distal, anchor)
    return proximal, distal


def _estimate_distal_direction(points: np.ndarray, anchor: np.ndarray, dists: np.ndarray) -> np.ndarray:
    vectors = points - anchor[None, :]
    mean_vec = np.mean(vectors, axis=0)
    if float(np.linalg.norm(mean_vec)) <= 1e-8:
        mean_vec = _principal_axis(points)

    centered = points - np.mean(points, axis=0, keepdims=True)
    if centered.shape[0] >= 3:
        cov = np.cov(centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        pca_axis = eigvecs[:, int(np.argmax(eigvals))]
    else:
        pca_axis = mean_vec
    if float(np.dot(pca_axis, mean_vec)) < 0.0:
        pca_axis = -pca_axis

    k = max(3, int(np.ceil(points.shape[0] * 0.20)))
    far_idx = np.argsort(dists)[-k:]
    far_vec = np.mean(vectors[far_idx], axis=0)
    if float(np.linalg.norm(far_vec)) <= 1e-8:
        far_vec = mean_vec

    blend = 0.55 * pca_axis + 0.45 * far_vec
    norm = float(np.linalg.norm(blend))
    if norm <= 1e-8:
        blend = mean_vec
        norm = float(np.linalg.norm(blend))
    if norm <= 1e-8:
        return np.array([0.0, -1.0, 0.0], dtype=np.float64)
    return blend / norm


def _refine_head_points_from_top_core(
    head_points: np.ndarray,
    *,
    y_min: float,
    y_max: float,
    torso_half_width: float,
    torso_half_depth: float,
    cfg: Mapping[str, Any],
    min_points: int,
) -> np.ndarray:
    if head_points.shape[0] < max(6, int(min_points)):
        return head_points
    height = max(float(y_max - y_min), 1e-6)
    top_band_scale = float(np.clip(float(cfg.get("top_band_scale", 0.12)), 0.03, 0.40))
    keep_drop_scale = float(np.clip(float(cfg.get("keep_drop_scale", 0.30)), 0.08, 0.55))
    core_x_scale = max(float(cfg.get("core_x_scale", 2.2)), 0.6)
    core_z_scale = max(float(cfg.get("core_z_scale", 2.4)), 0.6)
    core_min_keep_ratio = float(np.clip(float(cfg.get("core_min_keep_ratio", 0.35)), 0.10, 0.95))

    y_top = float(np.max(head_points[:, 1]))
    band_lo = y_top - top_band_scale * height
    core = head_points[head_points[:, 1] >= band_lo]
    if core.shape[0] == 0:
        core = head_points
    core_center = np.mean(core, axis=0)
    core_abs_x = np.abs(core[:, 0] - core_center[0])
    core_abs_z = np.abs(core[:, 2] - core_center[2])
    rx = max(float(np.percentile(core_abs_x, 90.0)), max(torso_half_width * 0.14, 0.03)) * core_x_scale
    rz = max(float(np.percentile(core_abs_z, 90.0)), max(torso_half_depth * 0.14, 0.03)) * core_z_scale
    y_floor = y_top - keep_drop_scale * height
    dx = (head_points[:, 0] - core_center[0]) / max(rx, 1e-6)
    dz = (head_points[:, 2] - core_center[2]) / max(rz, 1e-6)
    keep = (head_points[:, 1] >= y_floor) & ((dx * dx + dz * dz) <= 1.0)
    refined = head_points[keep]
    min_keep = max(int(min_points), int(np.ceil(head_points.shape[0] * core_min_keep_ratio)))
    if refined.shape[0] < min_keep:
        return head_points
    return refined


def _enforce_anchor_order(
    proximal: np.ndarray,
    distal: np.ndarray,
    anchor: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    if proximal.shape[0] == 0 or distal.shape[0] == 0:
        return proximal, distal
    d_prox = np.linalg.norm(proximal - anchor[None, :], axis=1)
    d_dist = np.linalg.norm(distal - anchor[None, :], axis=1)
    prox_score = float(np.median(d_prox))
    dist_score = float(np.median(d_dist))
    if prox_score <= dist_score:
        return proximal, distal
    return distal, proximal


def _apply_arm_absence_guard(
    arm_points: np.ndarray,
    leg_points: np.ndarray,
    *,
    shoulder_anchor: np.ndarray,
    side_total_points: int,
    y_min: float,
    y_max: float,
    cfg: Mapping[str, Any],
) -> Tuple[np.ndarray, np.ndarray, bool]:
    if arm_points.shape[0] == 0:
        return arm_points, leg_points, False
    if not bool(cfg.get("enabled", True)):
        return arm_points, leg_points, False

    min_arm_points = max(int(cfg.get("min_arm_points", 10)), 1)
    min_side_ratio = float(np.clip(float(cfg.get("min_side_ratio", 0.16)), 0.0, 1.0))
    min_median_y_norm = float(np.clip(float(cfg.get("min_median_y_norm", 0.42)), 0.0, 1.0))
    max_shoulder_attach_dist = max(float(cfg.get("max_shoulder_attach_dist", 0.24)), 0.02)
    attach_quantile = float(np.clip(float(cfg.get("attach_quantile", 0.30)), 0.0, 1.0))

    denom = max(int(side_total_points), 1)
    arm_ratio = float(arm_points.shape[0]) / float(denom)
    height = max(float(y_max - y_min), 1e-6)
    arm_y_norm = (arm_points[:, 1] - float(y_min)) / height
    median_y_norm = float(np.median(arm_y_norm))
    d_shoulder = np.linalg.norm(arm_points - shoulder_anchor[None, :], axis=1)
    attach_dist = float(np.quantile(d_shoulder, attach_quantile))

    weak_arm = (arm_points.shape[0] < min_arm_points) or (arm_ratio < min_side_ratio) or (median_y_norm < min_median_y_norm)
    detached = attach_dist > max_shoulder_attach_dist
    if not (weak_arm and detached):
        return arm_points, leg_points, False
    merged_leg = np.concatenate([leg_points, arm_points], axis=0) if leg_points.shape[0] > 0 else arm_points
    return _empty_points(), merged_leg, True


def _reclaim_arm_proximal_points(
    upper: np.ndarray,
    lower: np.ndarray,
    *,
    shoulder_anchor: np.ndarray,
    cfg: Mapping[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    if upper.shape[0] == 0 or lower.shape[0] == 0:
        return upper, lower
    if not bool(cfg.get("enabled", True)):
        return upper, lower
    prox_quantile = float(np.clip(float(cfg.get("prox_quantile", 0.85)), 0.50, 0.98))
    margin = max(float(cfg.get("margin", 0.015)), 0.0)
    max_move_ratio = float(np.clip(float(cfg.get("max_move_ratio", 0.35)), 0.0, 0.95))

    d_upper = np.linalg.norm(upper - shoulder_anchor[None, :], axis=1)
    d_lower = np.linalg.norm(lower - shoulder_anchor[None, :], axis=1)
    prox_thr = float(np.quantile(d_upper, prox_quantile)) + margin
    move_mask = d_lower <= prox_thr
    move_idx = np.flatnonzero(move_mask)
    if move_idx.size == 0:
        return upper, lower
    max_move = max(1, int(np.floor(lower.shape[0] * max_move_ratio)))
    if move_idx.size > max_move:
        sorted_idx = move_idx[np.argsort(d_lower[move_idx])]
        move_idx = sorted_idx[:max_move]
    keep_mask = np.ones((lower.shape[0],), dtype=bool)
    keep_mask[move_idx] = False
    moved = lower[move_idx]
    lower_new = lower[keep_mask]
    upper_new = np.concatenate([upper, moved], axis=0)
    return upper_new, lower_new


def _side_selector(points: np.ndarray, center_x: float, *, left: bool, left_is_positive_x: bool) -> np.ndarray:
    if left:
        if left_is_positive_x:
            return points[:, 0] >= center_x
        return points[:, 0] <= center_x
    if left_is_positive_x:
        return points[:, 0] < center_x
    return points[:, 0] > center_x


def _estimate_side_anchor(
    torso_points: np.ndarray,
    torso_center: np.ndarray,
    torso_half_width: float,
    *,
    left: bool,
    left_is_positive_x: bool,
    top: bool,
    y_quantile: float,
) -> np.ndarray:
    x_sign = 1.0 if (left == left_is_positive_x) else -1.0
    if torso_points.shape[0] == 0:
        y_fallback = torso_center[1] + (0.15 if top else -0.15)
        return np.array(
            [torso_center[0] + x_sign * torso_half_width * (0.75 if top else 0.55), y_fallback, torso_center[2]],
            dtype=np.float64,
        )
    side_mask = _side_selector(torso_points, torso_center[0], left=left, left_is_positive_x=left_is_positive_x)
    side_points = torso_points[side_mask]
    if side_points.shape[0] == 0:
        side_points = torso_points
    y = side_points[:, 1]
    q = float(np.clip(y_quantile, 0.0, 1.0))
    y_thr = float(np.quantile(y, q))
    band = side_points[y >= y_thr] if top else side_points[y <= y_thr]
    if band.shape[0] == 0:
        band = side_points
    return np.mean(band, axis=0).astype(np.float64)


def _split_arm_leg_by_anchors(
    side_points: np.ndarray,
    side_y_norm: np.ndarray,
    shoulder_anchor: np.ndarray,
    hip_anchor: np.ndarray,
    arm_vertical_penalty: float,
    leg_vertical_penalty: float,
    arm_y_range: Tuple[float, float],
    leg_y_range: Tuple[float, float],
    y_prior_weight: float,
    top_arm_seed_fraction: float,
    bottom_leg_seed_fraction: float,
    cluster_eps: float,
    cluster_min_points: int,
    seed_conf_fraction: float,
    geodesic_k: int,
    geodesic_radius_scale: float,
    geodesic_score_weight: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if side_points.shape[0] == 0:
        return _empty_points(), _empty_points()
    d_shoulder = np.linalg.norm(side_points - shoulder_anchor[None, :], axis=1)
    d_hip = np.linalg.norm(side_points - hip_anchor[None, :], axis=1)
    arm_score = d_shoulder + float(arm_vertical_penalty) * np.maximum(0.0, hip_anchor[1] - side_points[:, 1])
    leg_score = d_hip + float(leg_vertical_penalty) * np.maximum(0.0, side_points[:, 1] - shoulder_anchor[1])

    arm_lo, arm_hi = float(arm_y_range[0]), float(arm_y_range[1])
    leg_lo, leg_hi = float(leg_y_range[0]), float(leg_y_range[1])
    y_arm_penalty = np.maximum(arm_lo - side_y_norm, 0.0) + np.maximum(side_y_norm - arm_hi, 0.0)
    y_leg_penalty = np.maximum(leg_lo - side_y_norm, 0.0) + np.maximum(side_y_norm - leg_hi, 0.0)
    arm_score = arm_score + float(y_prior_weight) * y_arm_penalty
    leg_score = leg_score + float(y_prior_weight) * y_leg_penalty

    n_points = int(side_points.shape[0])
    sorted_idx = np.argsort(side_y_norm)
    top_n = max(1, min(n_points, int(np.ceil(n_points * float(np.clip(top_arm_seed_fraction, 0.0, 0.5))))))
    bottom_n = max(1, min(n_points, int(np.ceil(n_points * float(np.clip(bottom_leg_seed_fraction, 0.0, 0.5))))))

    score_gap = leg_score - arm_score
    conf_n = max(1, min(n_points, int(np.ceil(n_points * float(np.clip(seed_conf_fraction, 0.02, 0.35))))))
    arm_conf_idx = np.argsort(score_gap)[-conf_n:]
    leg_conf_idx = np.argsort(score_gap)[:conf_n]

    arm_seed = np.unique(np.concatenate([sorted_idx[-top_n:], arm_conf_idx], axis=0))
    leg_seed = np.unique(np.concatenate([sorted_idx[:bottom_n], leg_conf_idx], axis=0))
    overlap = np.intersect1d(arm_seed, leg_seed, assume_unique=False)
    if overlap.size > 0:
        keep_arm = overlap[score_gap[overlap] >= 0.0]
        keep_leg = overlap[score_gap[overlap] < 0.0]
        arm_seed = np.unique(np.concatenate([arm_seed, keep_arm], axis=0))
        leg_seed = np.unique(np.concatenate([leg_seed, keep_leg], axis=0))
        arm_seed = arm_seed[~np.isin(arm_seed, keep_leg)]
        leg_seed = leg_seed[~np.isin(leg_seed, keep_arm)]

    if arm_seed.size == 0 or leg_seed.size == 0:
        y_mid = 0.5 * (float(shoulder_anchor[1]) + float(hip_anchor[1]))
        arm_mask = side_points[:, 1] >= y_mid
        return side_points[arm_mask], side_points[~arm_mask]

    graph = _build_neighbor_graph(
        side_points,
        k=max(3, int(geodesic_k)),
        radius=max(float(cluster_eps) * float(geodesic_radius_scale), 1e-3),
    )
    arm_geo = _multi_source_dijkstra(graph, arm_seed.tolist())
    leg_geo = _multi_source_dijkstra(graph, leg_seed.tolist())
    arm_geo = _normalize_scores(arm_geo)
    leg_geo = _normalize_scores(leg_geo)
    arm_unary = _normalize_scores(arm_score)
    leg_unary = _normalize_scores(leg_score)

    blend = float(np.clip(geodesic_score_weight, 0.0, 1.0))
    arm_total = arm_geo + blend * arm_unary
    leg_total = leg_geo + blend * leg_unary
    arm_mask = arm_total <= leg_total
    arm_mask[arm_seed] = True
    arm_mask[leg_seed] = False

    dbscan_labels = _cluster_labels(side_points, eps=float(cluster_eps), min_points=int(cluster_min_points))
    unique_labels = [label for label in np.unique(dbscan_labels) if int(label) >= 0]
    for label in unique_labels:
        cluster_mask = dbscan_labels == label
        if not np.any(cluster_mask):
            continue
        arm_ratio = float(np.mean(arm_mask[cluster_mask]))
        if arm_ratio >= 0.75:
            arm_mask[cluster_mask] = True
        elif arm_ratio <= 0.25:
            arm_mask[cluster_mask] = False

    if int(np.count_nonzero(arm_mask)) == 0 or int(np.count_nonzero(~arm_mask)) == 0:
        y_mid = 0.5 * (float(shoulder_anchor[1]) + float(hip_anchor[1]))
        arm_mask = side_points[:, 1] >= y_mid
    return side_points[arm_mask], side_points[~arm_mask]


def _cluster_labels(points: np.ndarray, eps: float, min_points: int) -> np.ndarray:
    if points.shape[0] == 0:
        return np.zeros((0,), dtype=np.int32)
    if points.shape[0] < max(int(min_points), 3):
        return np.full((points.shape[0],), -1, dtype=np.int32)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    labels = np.asarray(
        cloud.cluster_dbscan(
            eps=max(float(eps), 1e-4),
            min_points=max(int(min_points), 2),
            print_progress=False,
        ),
        dtype=np.int32,
    )
    if labels.shape[0] != points.shape[0]:
        return np.full((points.shape[0],), -1, dtype=np.int32)
    return labels


def _filter_part_component_by_anchor(
    points: np.ndarray,
    anchor: np.ndarray,
    *,
    eps: float,
    min_points: int,
    noise_attach_radius: float,
    min_keep_ratio: float,
    secondary_center_radius: float,
    fallback_to_input: bool = True,
) -> np.ndarray:
    if points.shape[0] <= max(3, int(min_points)):
        return points
    n_total = int(points.shape[0])
    labels = _cluster_labels(points, eps=max(float(eps), 1e-4), min_points=max(int(min_points), 2))
    valid_labels = [label for label in np.unique(labels) if int(label) >= 0]
    if not valid_labels:
        return points

    cluster_rows: list[tuple[int, float, int, np.ndarray, np.ndarray]] = []
    for label in valid_labels:
        cluster = points[labels == label]
        if cluster.shape[0] == 0:
            continue
        dist = np.linalg.norm(cluster - anchor[None, :], axis=1)
        anchor_med = float(np.median(dist))
        score = anchor_med - 0.002 * float(cluster.shape[0])
        cluster_rows.append((int(label), score, int(cluster.shape[0]), np.mean(cluster, axis=0), cluster))
    if not cluster_rows:
        return points

    cluster_rows.sort(key=lambda row: row[1])
    best_label, best_score, _, _, best_cluster = cluster_rows[0]
    selected = best_cluster
    if selected.shape[0] == 0:
        return points

    target_keep = max(int(min_points), int(np.ceil(n_total * float(np.clip(min_keep_ratio, 0.30, 0.90)))))
    if selected.shape[0] < target_keep and float(secondary_center_radius) > 0.0:
        selected_center = np.mean(selected, axis=0)
        anchor_margin = max(float(eps) * 1.75, 0.03)
        for label, score, _, cluster_center, cluster in cluster_rows[1:]:
            if selected.shape[0] >= target_keep:
                break
            center_dist = float(np.linalg.norm(cluster_center - selected_center))
            if center_dist > float(secondary_center_radius):
                continue
            if score > (best_score + anchor_margin):
                continue
            selected = np.concatenate([selected, cluster], axis=0)
            selected_center = np.mean(selected, axis=0)

    if float(noise_attach_radius) > 0.0:
        noise_points = points[labels < 0]
        if noise_points.shape[0] > 0:
            d = np.linalg.norm(noise_points[:, None, :] - selected[None, :, :], axis=2)
            d_min = np.min(d, axis=1)
            attach = noise_points[d_min <= float(noise_attach_radius)]
            if attach.shape[0] > 0:
                selected = np.concatenate([selected, attach], axis=0)

    min_required = max(int(min_points), int(np.ceil(n_total * 0.35)))
    if selected.shape[0] < min_required:
        if fallback_to_input:
            return points
        min_selected = max(2, min(int(min_points), 4))
        if selected.shape[0] >= min_selected:
            return selected
        return _empty_points()
    return selected


def _trim_far_points_from_anchor(
    points: np.ndarray,
    anchor: np.ndarray,
    *,
    max_anchor_quantile: float,
    min_keep_ratio: float,
    min_points: int,
) -> np.ndarray:
    if points.shape[0] <= max(3, int(min_points)):
        return points
    d = np.linalg.norm(points - anchor[None, :], axis=1)
    q = float(np.clip(max_anchor_quantile, 0.50, 0.99))
    threshold = float(np.quantile(d, q))
    keep = d <= (threshold + 1e-9)
    filtered = points[keep]
    min_keep = max(int(min_points), int(np.ceil(points.shape[0] * float(np.clip(min_keep_ratio, 0.20, 0.95)))))
    if filtered.shape[0] < min_keep:
        return points
    return filtered


def _exclude_head_zone_from_limbs(
    limb_points: np.ndarray,
    head_points: np.ndarray,
    *,
    torso_half_width: float,
    torso_half_depth: float,
    y_pad: float,
    radius_scale_x: float,
    radius_scale_y: float,
    radius_scale_z: float,
) -> np.ndarray:
    if limb_points.shape[0] == 0 or head_points.shape[0] < 3:
        return limb_points
    head_center = np.mean(head_points, axis=0)
    head_min_y = float(np.min(head_points[:, 1]))

    rx_raw = float(np.percentile(np.abs(head_points[:, 0] - head_center[0]), 90.0))
    ry_raw = float(np.percentile(np.abs(head_points[:, 1] - head_center[1]), 90.0))
    rz_raw = float(np.percentile(np.abs(head_points[:, 2] - head_center[2]), 90.0))
    rx = max(rx_raw, torso_half_width * 0.24, 0.03) * max(float(radius_scale_x), 0.5)
    ry = max(ry_raw, 0.045) * max(float(radius_scale_y), 0.5)
    rz = max(rz_raw, torso_half_depth * 0.24, 0.03) * max(float(radius_scale_z), 0.5)

    dx = (limb_points[:, 0] - head_center[0]) / max(rx, 1e-6)
    dy = (limb_points[:, 1] - head_center[1]) / max(ry, 1e-6)
    dz = (limb_points[:, 2] - head_center[2]) / max(rz, 1e-6)
    in_ellipsoid = (dx * dx + dy * dy + dz * dz) <= 1.0
    high_enough = limb_points[:, 1] >= (head_min_y - max(float(y_pad), 0.0))
    keep_mask = ~(in_ellipsoid & high_enough)
    filtered = limb_points[keep_mask]
    if filtered.shape[0] < max(2, int(0.55 * limb_points.shape[0])):
        return limb_points
    return filtered


def _exclude_torso_zone_from_arms(
    arm_points: np.ndarray,
    shoulder_anchor: np.ndarray,
    torso_center: np.ndarray,
    torso_half_width: float,
    torso_half_depth: float,
    *,
    y_range: Tuple[float, float],
    y_min: float,
    y_max: float,
    y_pad: float,
    x_scale: float,
    z_scale: float,
    shoulder_keep_radius_scale: float,
    shoulder_outward_min: float,
    min_keep_ratio: float,
) -> np.ndarray:
    if arm_points.shape[0] == 0:
        return arm_points
    height = max(float(y_max - y_min), 1e-6)
    y_norm = (arm_points[:, 1] - float(y_min)) / height
    y_lo = float(y_range[0]) - max(float(y_pad), 0.0)
    y_hi = float(y_range[1]) + max(float(y_pad), 0.0)
    in_y = (y_norm >= y_lo) & (y_norm <= y_hi)
    in_x = np.abs(arm_points[:, 0] - float(torso_center[0])) <= max(float(torso_half_width), 1e-6) * max(float(x_scale), 0.5)
    in_z = np.abs(arm_points[:, 2] - float(torso_center[2])) <= max(float(torso_half_depth), 1e-6) * max(float(z_scale), 0.5)
    near_torso = in_y & in_x & in_z
    if not np.any(near_torso):
        return arm_points

    shoulder_keep_radius = max(max(float(torso_half_width), float(torso_half_depth)) * max(float(shoulder_keep_radius_scale), 0.1), 0.06)
    d_shoulder = np.linalg.norm(arm_points - shoulder_anchor[None, :], axis=1)
    near_shoulder = d_shoulder <= shoulder_keep_radius

    outward_vec = shoulder_anchor - torso_center
    outward_vec[1] = 0.0
    outward_norm = float(np.linalg.norm(outward_vec))
    if outward_norm <= 1e-8:
        outward_unit = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        outward_unit = outward_vec / outward_norm
    outward_proj = (arm_points - shoulder_anchor[None, :]) @ outward_unit
    shoulder_keep = near_shoulder & (outward_proj >= float(shoulder_outward_min))

    remove = near_torso & (~shoulder_keep)
    filtered = arm_points[~remove]
    min_keep = max(2, int(np.ceil(arm_points.shape[0] * float(np.clip(min_keep_ratio, 0.10, 0.95)))))
    if filtered.shape[0] < min_keep:
        return arm_points
    return filtered


def _build_neighbor_graph(points: np.ndarray, k: int, radius: float) -> list[list[tuple[int, float]]]:
    n_points = int(points.shape[0])
    if n_points == 0:
        return []
    k_eff = max(1, min(k, n_points - 1)) if n_points > 1 else 0
    diff = points[:, None, :] - points[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    adjacency: list[dict[int, float]] = [dict() for _ in range(n_points)]

    for idx in range(n_points):
        if k_eff > 0:
            knn = np.argpartition(dist[idx], k_eff)[: k_eff + 1]
            knn = knn[knn != idx]
        else:
            knn = np.zeros((0,), dtype=np.int64)
        radial = np.flatnonzero((dist[idx] <= radius) & (np.arange(n_points) != idx))
        neighbors = np.unique(np.concatenate([knn, radial], axis=0))
        if neighbors.size == 0 and n_points > 1:
            nearest = int(np.argmin(np.where(np.arange(n_points) == idx, np.inf, dist[idx])))
            neighbors = np.array([nearest], dtype=np.int64)
        for n_idx in neighbors.tolist():
            weight = float(max(dist[idx, n_idx], 1e-6))
            prev = adjacency[idx].get(int(n_idx))
            if prev is None or weight < prev:
                adjacency[idx][int(n_idx)] = weight
            prev_rev = adjacency[int(n_idx)].get(idx)
            if prev_rev is None or weight < prev_rev:
                adjacency[int(n_idx)][idx] = weight

    graph: list[list[tuple[int, float]]] = []
    for row in adjacency:
        graph.append([(neighbor, weight) for neighbor, weight in row.items()])
    return graph


def _multi_source_dijkstra(graph: list[list[tuple[int, float]]], seeds: list[int]) -> np.ndarray:
    n_points = len(graph)
    out = np.full((n_points,), np.inf, dtype=np.float64)
    if n_points == 0:
        return out
    heap: list[tuple[float, int]] = []
    for seed in seeds:
        if 0 <= int(seed) < n_points:
            out[int(seed)] = 0.0
            heapq.heappush(heap, (0.0, int(seed)))
    while heap:
        current_dist, node = heapq.heappop(heap)
        if current_dist > out[node] + 1e-12:
            continue
        for neigh, weight in graph[node]:
            next_dist = current_dist + float(weight)
            if next_dist + 1e-12 < out[neigh]:
                out[neigh] = next_dist
                heapq.heappush(heap, (next_dist, neigh))
    return out


def _normalize_scores(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    finite_mask = np.isfinite(values)
    if not np.any(finite_mask):
        return np.ones_like(values, dtype=np.float64)
    finite = values[finite_mask]
    lo = float(np.min(finite))
    hi = float(np.max(finite))
    out = np.ones_like(values, dtype=np.float64)
    if hi - lo <= 1e-9:
        out[finite_mask] = 0.0
        return out
    out[finite_mask] = (values[finite_mask] - lo) / (hi - lo)
    out[~finite_mask] = 1.0
    return out


def _point_lookup_exact(points: np.ndarray) -> Dict[Tuple[float, float, float], list[int]]:
    lookup: Dict[Tuple[float, float, float], list[int]] = {}
    for idx, row in enumerate(points):
        key = (float(row[0]), float(row[1]), float(row[2]))
        lookup.setdefault(key, []).append(int(idx))
    return lookup


def _assign_labels_from_part_points(
    points: np.ndarray,
    part_points: Mapping[str, np.ndarray],
) -> np.ndarray:
    labels = np.full((points.shape[0],), -1, dtype=np.int32)
    name_to_idx = {name: idx for idx, name in enumerate(DEFAULT_PART_ORDER)}
    centers: Dict[str, np.ndarray] = {}
    for name in DEFAULT_PART_ORDER:
        pts = np.asarray(part_points.get(name, _empty_points()), dtype=np.float64)
        if pts.shape[0] > 0:
            centers[name] = np.mean(pts, axis=0)

    lookup = _point_lookup_exact(points)
    for name in DEFAULT_PART_ORDER:
        pts = np.asarray(part_points.get(name, _empty_points()), dtype=np.float64)
        if pts.shape[0] == 0:
            continue
        cand_idx = int(name_to_idx[name])
        cand_center = centers.get(name)
        if cand_center is None:
            continue
        for row in pts:
            key = (float(row[0]), float(row[1]), float(row[2]))
            targets = lookup.get(key)
            if not targets:
                continue
            for point_idx in targets:
                prev_idx = int(labels[point_idx])
                if prev_idx < 0:
                    labels[point_idx] = cand_idx
                    continue
                if prev_idx == cand_idx:
                    continue
                prev_name = DEFAULT_PART_ORDER[prev_idx]
                prev_center = centers.get(prev_name)
                if prev_center is None:
                    labels[point_idx] = cand_idx
                    continue
                point = points[point_idx]
                cand_dist = float(np.linalg.norm(point - cand_center))
                prev_dist = float(np.linalg.norm(point - prev_center))
                if cand_dist < prev_dist:
                    labels[point_idx] = cand_idx
    return labels


def _nearest_distance_to_cloud(point: np.ndarray, cloud: np.ndarray) -> float:
    if cloud.shape[0] == 0:
        return float("inf")
    d = np.linalg.norm(cloud - point[None, :], axis=1)
    return float(np.min(d))


def _rebuild_part_points_from_labels(points: np.ndarray, labels: np.ndarray) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for idx, name in enumerate(DEFAULT_PART_ORDER):
        out[name] = points[labels == idx]
    return out


def _refine_part_boundaries(
    *,
    points: np.ndarray,
    y_norm: np.ndarray,
    labels: np.ndarray,
    part_points: Mapping[str, np.ndarray],
    torso_center: np.ndarray,
    torso_half_width: float,
    torso_half_depth: float,
    arm_y: Tuple[float, float],
    leg_y: Tuple[float, float],
    head_y: Tuple[float, float],
    left_is_positive_x: bool,
    cfg: Mapping[str, Any],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    if points.shape[0] == 0:
        return labels, {"enabled": False, "reason": "empty_points"}
    enabled = bool(cfg.get("enabled", True))
    if not enabled:
        return labels, {"enabled": False}

    label_out = labels.copy()
    arrays = {name: np.asarray(part_points.get(name, _empty_points()), dtype=np.float64) for name in DEFAULT_PART_ORDER}
    valid_names = [name for name in DEFAULT_PART_ORDER if arrays[name].shape[0] > 0]
    if not valid_names:
        return label_out, {"enabled": True, "reason": "no_valid_parts"}
    name_to_idx = {name: idx for idx, name in enumerate(DEFAULT_PART_ORDER)}

    head_cap_cfg = cfg.get("head_cap", {}) if isinstance(cfg.get("head_cap"), Mapping) else {}
    head_cap_enabled = bool(head_cap_cfg.get("enabled", True))
    head_cap_y_margin = float(head_cap_cfg.get("y_margin_low", 0.04))
    head_cap_x_scale = float(head_cap_cfg.get("x_scale", 1.35))
    head_cap_z_scale = float(head_cap_cfg.get("z_scale", 1.95))

    unresolved_max_dist = float(cfg.get("unresolved_max_dist", 0.09))
    reclaim_cfg = cfg.get("torso_arm_reclaim", {}) if isinstance(cfg.get("torso_arm_reclaim"), Mapping) else {}
    reclaim_enabled = bool(reclaim_cfg.get("enabled", True))
    reclaim_y_margin = float(reclaim_cfg.get("y_margin", 0.03))
    reclaim_lateral_x_scale = float(reclaim_cfg.get("lateral_x_scale", 0.70))
    reclaim_frontal_z_scale = float(reclaim_cfg.get("frontal_z_scale", 0.70))
    reclaim_max_arm_attach_dist = float(reclaim_cfg.get("max_arm_attach_dist", 0.11))
    reclaim_margin = float(reclaim_cfg.get("arm_vs_torso_margin", 0.008))
    leg_reclaim_cfg = cfg.get("torso_leg_reclaim", {}) if isinstance(cfg.get("torso_leg_reclaim"), Mapping) else {}
    leg_reclaim_enabled = bool(leg_reclaim_cfg.get("enabled", True))
    leg_reclaim_y_margin = float(leg_reclaim_cfg.get("y_margin", 0.03))
    leg_reclaim_max_attach_dist = float(leg_reclaim_cfg.get("max_leg_attach_dist", 0.12))
    leg_reclaim_margin = float(leg_reclaim_cfg.get("leg_vs_torso_margin", 0.004))

    head_assigned = 0
    unresolved_assigned = 0
    torso_reclaimed = 0
    torso_leg_reclaimed = 0

    if head_cap_enabled and "head" in name_to_idx:
        head_idx = name_to_idx["head"]
        unresolved_idx = np.flatnonzero(label_out < 0)
        if unresolved_idx.size > 0:
            y_low = float(head_y[0]) - max(head_cap_y_margin, 0.0)
            rx = max(torso_half_width * max(head_cap_x_scale, 0.5), 0.05)
            rz = max(torso_half_depth * max(head_cap_z_scale, 0.5), 0.05)
            for idx in unresolved_idx.tolist():
                if float(y_norm[idx]) < y_low:
                    continue
                dx = (float(points[idx, 0]) - float(torso_center[0])) / rx
                dz = (float(points[idx, 2]) - float(torso_center[2])) / rz
                if (dx * dx + dz * dz) <= 1.0:
                    label_out[idx] = int(head_idx)
                    head_assigned += 1

    unresolved_idx = np.flatnonzero(label_out < 0)
    if unresolved_idx.size > 0:
        for idx in unresolved_idx.tolist():
            point = points[idx]
            best_name: str | None = None
            best_dist = float("inf")
            for name in valid_names:
                dist = _nearest_distance_to_cloud(point, arrays[name])
                if dist < best_dist:
                    best_dist = dist
                    best_name = name
            if best_name is not None and best_dist <= max(unresolved_max_dist, 0.0):
                label_out[idx] = int(name_to_idx[best_name])
                unresolved_assigned += 1

    if reclaim_enabled and "torso" in name_to_idx:
        torso_idx = name_to_idx["torso"]
        if valid_names:
            arm_lo = float(arm_y[0]) - max(reclaim_y_margin, 0.0)
            arm_hi = float(arm_y[1]) + max(reclaim_y_margin, 0.0)
            lateral_thr = max(float(torso_half_width) * max(reclaim_lateral_x_scale, 0.1), 0.02)
            frontal_thr = max(float(torso_half_depth) * max(reclaim_frontal_z_scale, 0.1), 0.02)
            torso_indices = np.flatnonzero(label_out == torso_idx)
            for idx in torso_indices.tolist():
                if not (arm_lo <= float(y_norm[idx]) <= arm_hi):
                    continue
                lateral = abs(float(points[idx, 0]) - float(torso_center[0]))
                frontal = abs(float(points[idx, 2]) - float(torso_center[2]))
                if (lateral < lateral_thr) and (frontal < frontal_thr):
                    continue
                if lateral >= lateral_thr:
                    is_left = (
                        float(points[idx, 0]) >= float(torso_center[0])
                        if left_is_positive_x
                        else float(points[idx, 0]) <= float(torso_center[0])
                    )
                    if is_left:
                        candidates = ["left_upper_arm", "left_lower_arm"]
                    else:
                        candidates = ["right_upper_arm", "right_lower_arm"]
                else:
                    candidates = [
                        "left_upper_arm",
                        "left_lower_arm",
                        "right_upper_arm",
                        "right_lower_arm",
                    ]
                best_arm_name: str | None = None
                best_arm_dist = float("inf")
                point = points[idx]
                for name in candidates:
                    cloud = arrays.get(name, _empty_points())
                    if cloud.shape[0] == 0:
                        continue
                    dist = _nearest_distance_to_cloud(point, cloud)
                    if dist < best_arm_dist:
                        best_arm_dist = dist
                        best_arm_name = name
                if best_arm_name is None:
                    continue
                if best_arm_dist > max(reclaim_max_arm_attach_dist, 0.0):
                    continue
                d_torso_center = float(np.linalg.norm(point - torso_center))
                if best_arm_dist + float(reclaim_margin) < d_torso_center:
                    label_out[idx] = int(name_to_idx[best_arm_name])
                    torso_reclaimed += 1

    if leg_reclaim_enabled and "torso" in name_to_idx:
        torso_idx = name_to_idx["torso"]
        leg_lo = float(leg_y[0]) - max(leg_reclaim_y_margin, 0.0)
        leg_hi = float(leg_y[1]) + max(leg_reclaim_y_margin, 0.0)
        torso_indices = np.flatnonzero(label_out == torso_idx)
        for idx in torso_indices.tolist():
            if not (leg_lo <= float(y_norm[idx]) <= leg_hi):
                continue
            point = points[idx]
            is_left = (
                float(point[0]) >= float(torso_center[0])
                if left_is_positive_x
                else float(point[0]) <= float(torso_center[0])
            )
            side_candidates = (
                ["left_thigh", "left_shin"]
                if is_left
                else ["right_thigh", "right_shin"]
            )
            all_candidates = side_candidates + [name for name in ["left_thigh", "left_shin", "right_thigh", "right_shin"] if name not in side_candidates]
            best_leg_name: str | None = None
            best_leg_dist = float("inf")
            for name in all_candidates:
                cloud = arrays.get(name, _empty_points())
                if cloud.shape[0] == 0:
                    continue
                dist = _nearest_distance_to_cloud(point, cloud)
                if dist < best_leg_dist:
                    best_leg_dist = dist
                    best_leg_name = name
            if best_leg_name is None:
                continue
            if best_leg_dist > max(leg_reclaim_max_attach_dist, 0.0):
                continue
            d_torso_center = float(np.linalg.norm(point - torso_center))
            if best_leg_dist + float(leg_reclaim_margin) < d_torso_center:
                label_out[idx] = int(name_to_idx[best_leg_name])
                torso_leg_reclaimed += 1

    return label_out, {
        "enabled": True,
        "head_cap_assigned": int(head_assigned),
        "unresolved_assigned": int(unresolved_assigned),
        "torso_arm_reclaimed": int(torso_reclaimed),
        "torso_leg_reclaimed": int(torso_leg_reclaimed),
        "unresolved_after": int(np.count_nonzero(label_out < 0)),
    }


_PARENT_CHILD_LINKS: Tuple[Tuple[str, str], ...] = (
    ("head", "torso"),
    ("left_upper_arm", "torso"),
    ("left_lower_arm", "left_upper_arm"),
    ("right_upper_arm", "torso"),
    ("right_lower_arm", "right_upper_arm"),
    ("left_thigh", "torso"),
    ("left_shin", "left_thigh"),
    ("right_thigh", "torso"),
    ("right_shin", "right_thigh"),
)

_SHIFT_CONTACT_LINKS: Tuple[Tuple[str, str], ...] = (
    ("left_lower_arm", "left_upper_arm"),
    ("right_lower_arm", "right_upper_arm"),
)

_FORBIDDEN_CONTACTS: Dict[str, Tuple[str, ...]] = {
    "left_lower_arm": ("torso",),
    "right_lower_arm": ("torso",),
    "left_shin": ("torso",),
    "right_shin": ("torso",),
}


def _build_box_from_bounds(
    name: str,
    num_points: int,
    min_xyz: np.ndarray,
    max_xyz: np.ndarray,
    template: PartBox | None = None,
) -> PartBox:
    return PartBox.from_bounds(
        name,
        num_points=int(num_points),
        min_xyz=min_xyz,
        max_xyz=max_xyz,
        template=template,
    )


def _touch_child_to_parent_expand(child: PartBox, parent: PartBox, margin: float) -> PartBox:
    if not child.valid or not parent.valid:
        return child
    cmin = child.min_xyz.copy()
    cmax = child.max_xyz.copy()
    pmin = parent.min_xyz
    pmax = parent.max_xyz
    for axis in range(3):
        if cmax[axis] + margin < pmin[axis]:
            cmax[axis] = pmin[axis] + margin
        elif pmax[axis] + margin < cmin[axis]:
            cmin[axis] = pmax[axis] - margin
    cmax = np.maximum(cmax, cmin + 1e-6)
    return _build_box_from_bounds(child.name, child.num_points, cmin, cmax, template=child)


def _touch_child_to_parent_shift(child: PartBox, parent: PartBox, margin: float) -> PartBox:
    if not child.valid or not parent.valid:
        return child
    cmin = child.min_xyz.copy()
    cmax = child.max_xyz.copy()
    pmin = parent.min_xyz
    pmax = parent.max_xyz
    shift = np.zeros((3,), dtype=np.float64)
    for axis in range(3):
        if cmax[axis] + margin < pmin[axis]:
            shift[axis] = pmin[axis] - (cmax[axis] + margin)
        elif pmax[axis] + margin < cmin[axis]:
            shift[axis] = (pmax[axis] + margin) - cmin[axis]
    if np.any(np.abs(shift) > 0.0):
        cmin = cmin + shift
        cmax = cmax + shift
    cmax = np.maximum(cmax, cmin + 1e-6)
    return _build_box_from_bounds(child.name, child.num_points, cmin, cmax, template=child)


def _boxes_overlap_depth(child: PartBox, other: PartBox) -> np.ndarray:
    return np.minimum(child.max_xyz, other.max_xyz) - np.maximum(child.min_xyz, other.min_xyz)


def _separate_forbidden(child: PartBox, other: PartBox, margin: float) -> PartBox:
    if not child.valid or not other.valid:
        return child
    overlap = _boxes_overlap_depth(child, other)
    if np.any(overlap <= 0.0):
        return child
    axis = int(np.argmin(overlap))
    shift = float(overlap[axis] + margin)
    cmin = child.min_xyz.copy()
    cmax = child.max_xyz.copy()
    if child.center_xyz[axis] <= other.center_xyz[axis]:
        cmin[axis] -= shift
        cmax[axis] -= shift
    else:
        cmin[axis] += shift
        cmax[axis] += shift
    return _build_box_from_bounds(child.name, child.num_points, cmin, cmax, template=child)


def _enforce_pair_min_overlap(
    child: PartBox,
    parent: PartBox,
    *,
    min_overlap: float,
    max_expand_per_axis: float,
) -> PartBox:
    if not child.valid or not parent.valid:
        return child
    min_overlap_val = max(float(min_overlap), 0.0)
    if min_overlap_val <= 0.0:
        return child
    max_expand = max(float(max_expand_per_axis), 1e-6)
    cmin = child.min_xyz.copy()
    cmax = child.max_xyz.copy()
    pmin = parent.min_xyz
    pmax = parent.max_xyz
    ccenter = child.center_xyz
    pcenter = parent.center_xyz
    for axis in range(3):
        overlap = float(min(cmax[axis], pmax[axis]) - max(cmin[axis], pmin[axis]))
        if overlap >= min_overlap_val:
            continue
        deficit = min(min_overlap_val - overlap, max_expand)
        if ccenter[axis] <= pcenter[axis]:
            cmax[axis] += deficit
        else:
            cmin[axis] -= deficit
    cmax = np.maximum(cmax, cmin + 1e-6)
    return _build_box_from_bounds(child.name, child.num_points, cmin, cmax, template=child)


def _parse_hard_contact_links(raw_links: Any) -> list[tuple[str, str, float | None]]:
    out: list[tuple[str, str, float | None]] = []
    if not isinstance(raw_links, list):
        return out
    for row in raw_links:
        if isinstance(row, Mapping):
            child = row.get("child")
            parent = row.get("parent")
            if not isinstance(child, str) or not isinstance(parent, str):
                continue
            min_overlap_raw = row.get("min_overlap")
            if min_overlap_raw is None:
                min_overlap_val = None
            else:
                try:
                    min_overlap_val = float(min_overlap_raw)
                except Exception:
                    min_overlap_val = None
            out.append((child, parent, min_overlap_val))
            continue
        if isinstance(row, (list, tuple)) and len(row) >= 2:
            child = row[0]
            parent = row[1]
            if not isinstance(child, str) or not isinstance(parent, str):
                continue
            min_overlap_val = None
            if len(row) >= 3:
                try:
                    min_overlap_val = float(row[2])
                except Exception:
                    min_overlap_val = None
            out.append((child, parent, min_overlap_val))
    return out


def _normalize_vector(vec: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-8:
        fb = np.asarray(fallback, dtype=np.float64)
        fb_norm = float(np.linalg.norm(fb))
        if fb_norm <= 1e-8:
            return np.array([1.0, 0.0, 0.0], dtype=np.float64)
        return fb / fb_norm
    return vec / norm


def _orthonormal_axes_from_primary(primary: np.ndarray, hint: np.ndarray) -> np.ndarray:
    axis0 = _normalize_vector(np.asarray(primary, dtype=np.float64), np.array([1.0, 0.0, 0.0], dtype=np.float64))
    hint_vec = np.asarray(hint, dtype=np.float64)
    axis1 = hint_vec - axis0 * float(np.dot(hint_vec, axis0))
    if float(np.linalg.norm(axis1)) <= 1e-8:
        trial = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if abs(float(np.dot(trial, axis0))) > 0.9:
            trial = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        axis1 = trial - axis0 * float(np.dot(trial, axis0))
    axis1 = _normalize_vector(axis1, np.array([0.0, 1.0, 0.0], dtype=np.float64))
    axis2 = np.cross(axis0, axis1)
    axis2 = _normalize_vector(axis2, np.array([0.0, 0.0, 1.0], dtype=np.float64))
    axis1 = np.cross(axis2, axis0)
    axis1 = _normalize_vector(axis1, np.array([0.0, 1.0, 0.0], dtype=np.float64))
    axes = np.stack([axis0, axis1, axis2], axis=1)
    if float(np.linalg.det(axes)) < 0.0:
        axes[:, 2] = -axes[:, 2]
    return axes


def _partbox_from_obb(
    *,
    name: str,
    num_points: int,
    center: np.ndarray,
    axes: np.ndarray,
    size: np.ndarray,
) -> PartBox:
    half = 0.5 * np.maximum(np.asarray(size, dtype=np.float64), 1e-6)
    local = np.array(
        [
            [-half[0], -half[1], -half[2]],
            [half[0], -half[1], -half[2]],
            [half[0], half[1], -half[2]],
            [-half[0], half[1], -half[2]],
            [-half[0], -half[1], half[2]],
            [half[0], -half[1], half[2]],
            [half[0], half[1], half[2]],
            [-half[0], half[1], half[2]],
        ],
        dtype=np.float64,
    )
    corners = local @ axes.T + center[None, :]
    min_xyz = np.min(corners, axis=0).astype(np.float64)
    max_xyz = np.max(corners, axis=0).astype(np.float64)
    center_xyz = 0.5 * (min_xyz + max_xyz)
    size_xyz = max_xyz - min_xyz
    return PartBox(
        name=name,
        valid=True,
        num_points=int(num_points),
        min_xyz=min_xyz,
        max_xyz=max_xyz,
        center_xyz=center_xyz,
        size_xyz=size_xyz.astype(np.float64),
        obb_center_xyz=np.asarray(center, dtype=np.float64),
        obb_size_xyz=np.maximum(np.asarray(size, dtype=np.float64), 1e-6),
        obb_axes_xyz=np.asarray(axes, dtype=np.float64),
        obb_corners_xyz=corners.astype(np.float64),
    )


def _enforce_limb_shape_boxes(
    boxes: Dict[str, PartBox],
    part_points: Mapping[str, np.ndarray],
    *,
    cfg: Mapping[str, Any] | None,
    min_points: int,
) -> Dict[str, PartBox]:
    shape_cfg = cfg if isinstance(cfg, Mapping) else {}
    if not bool(shape_cfg.get("enabled", True)):
        return boxes
    align_to_links = bool(shape_cfg.get("align_to_links", True))
    direction_blend = float(np.clip(float(shape_cfg.get("direction_blend", 0.70)), 0.0, 1.0))
    leg_direction_blend_scale = float(np.clip(float(shape_cfg.get("leg_direction_blend_scale", 0.55)), 0.0, 1.0))
    use_torso_secondary_arms = bool(shape_cfg.get("use_torso_secondary_arms", True))
    use_torso_secondary_legs = bool(shape_cfg.get("use_torso_secondary_legs", False))
    max_turn_deg_default = max(float(shape_cfg.get("max_turn_deg_default", 45.0)), 1.0)
    max_turn_cfg_raw = shape_cfg.get("max_turn_deg")
    max_turn_cfg = max_turn_cfg_raw if isinstance(max_turn_cfg_raw, Mapping) else {}
    quantile = float(np.clip(float(shape_cfg.get("quantile", 0.04)), 0.0, 0.20))
    min_long_axis = max(float(shape_cfg.get("min_long_axis", 0.085)), 0.01)
    min_aspect_ratio = max(float(shape_cfg.get("min_aspect_ratio", 1.20)), 1.0)
    max_cross_axis = max(float(shape_cfg.get("max_cross_axis", 0.16)), 0.02)

    arm_secondary = "torso" if use_torso_secondary_arms else ""
    leg_secondary = "torso" if use_torso_secondary_legs else ""
    link_hints: Dict[str, Tuple[str, str]] = {
        "left_upper_arm": ("left_lower_arm", arm_secondary),
        "right_upper_arm": ("right_lower_arm", arm_secondary),
        "left_lower_arm": ("left_upper_arm", ""),
        "right_lower_arm": ("right_upper_arm", ""),
        "left_thigh": ("left_shin", leg_secondary),
        "right_thigh": ("right_shin", leg_secondary),
        "left_shin": ("left_thigh", ""),
        "right_shin": ("right_thigh", ""),
    }
    leg_parts = {"left_thigh", "right_thigh", "left_shin", "right_shin"}
    out = dict(boxes)
    for name, (primary_link, secondary_link) in link_hints.items():
        box = out.get(name)
        pts = np.asarray(part_points.get(name, _empty_points()), dtype=np.float64)
        if box is None or (not box.valid) or pts.shape[0] < max(int(min_points), 4):
            continue

        center_seed = np.mean(pts, axis=0)
        axes_seed = np.asarray(box.obb_axes_xyz, dtype=np.float64)
        size_seed = np.maximum(np.asarray(box.obb_size_xyz, dtype=np.float64), 1e-6)
        long_idx = int(np.argmax(size_seed))
        axis_long = axes_seed[:, long_idx]

        target_dirs: list[np.ndarray] = []
        if align_to_links:
            for target_name in (primary_link, secondary_link):
                if not target_name:
                    continue
                target_box = out.get(target_name)
                if target_box is None or not target_box.valid:
                    continue
                vec = np.asarray(target_box.center_xyz, dtype=np.float64) - center_seed
                if float(np.linalg.norm(vec)) > 1e-6:
                    target_dirs.append(vec)
        if target_dirs:
            max_turn_deg = max_turn_deg_default
            if name in max_turn_cfg:
                try:
                    max_turn_deg = max(float(max_turn_cfg[name]), 1.0)
                except Exception:
                    max_turn_deg = max_turn_deg_default
            target = np.mean(np.stack(target_dirs, axis=0), axis=0)
            target = _normalize_vector(target, axis_long)
            if float(np.dot(axis_long, target)) < 0.0:
                axis_long = -axis_long
            local_blend = direction_blend * (leg_direction_blend_scale if name in leg_parts else 1.0)
            local_blend = float(np.clip(local_blend, 0.0, 1.0))
            angle = float(np.arccos(np.clip(float(np.dot(axis_long, target)), -1.0, 1.0)))
            max_turn_rad = float(np.deg2rad(max_turn_deg))
            if angle > 1e-6 and max_turn_rad > 0.0:
                local_blend *= min(1.0, max_turn_rad / angle)
            axis_long = _normalize_vector((1.0 - local_blend) * axis_long + local_blend * target, axis_long)

        hint_axis = axes_seed[:, (long_idx + 1) % 3]
        axes_new = _orthonormal_axes_from_primary(axis_long, hint_axis)
        proj = (pts - center_seed[None, :]) @ axes_new
        if quantile > 0.0 and pts.shape[0] >= max(10, int(min_points)):
            lo = np.quantile(proj, quantile, axis=0)
            hi = np.quantile(proj, 1.0 - quantile, axis=0)
        else:
            lo = np.min(proj, axis=0)
            hi = np.max(proj, axis=0)
        ext = np.maximum(hi - lo, 1e-6)
        center_local = 0.5 * (lo + hi)
        center_new = center_seed + axes_new @ center_local
        long_size = max(float(ext[0]), min_long_axis, float(max(ext[1], ext[2])) * min_aspect_ratio)
        cross_1 = min(float(ext[1]), max_cross_axis)
        cross_2 = min(float(ext[2]), max_cross_axis)
        size_new = np.array([long_size, max(cross_1, 1e-6), max(cross_2, 1e-6)], dtype=np.float64)
        out[name] = _partbox_from_obb(
            name=name,
            num_points=int(box.num_points),
            center=center_new,
            axes=axes_new,
            size=size_new,
        )
    return out


def _projected_half_extent(box: PartBox, direction: np.ndarray) -> float:
    if not box.valid:
        return 0.0
    unit = _normalize_vector(np.asarray(direction, dtype=np.float64), np.array([1.0, 0.0, 0.0], dtype=np.float64))
    axes = np.asarray(box.obb_axes_xyz, dtype=np.float64)
    half = 0.5 * np.maximum(np.asarray(box.obb_size_xyz, dtype=np.float64), 1e-6)
    proj = np.abs(axes.T @ unit)
    return float(np.sum(proj * half))


def _shift_box_obb(box: PartBox, shift: np.ndarray) -> PartBox:
    if not box.valid:
        return box
    delta = np.asarray(shift, dtype=np.float64)
    if float(np.linalg.norm(delta)) <= 1e-9:
        return box
    corners = np.asarray(box.obb_corners_xyz, dtype=np.float64) + delta[None, :]
    min_xyz = np.min(corners, axis=0).astype(np.float64)
    max_xyz = np.max(corners, axis=0).astype(np.float64)
    center_xyz = 0.5 * (min_xyz + max_xyz)
    size_xyz = max_xyz - min_xyz
    return PartBox(
        name=box.name,
        valid=box.valid,
        num_points=box.num_points,
        min_xyz=min_xyz,
        max_xyz=max_xyz,
        center_xyz=center_xyz,
        size_xyz=size_xyz.astype(np.float64),
        obb_center_xyz=np.asarray(box.obb_center_xyz, dtype=np.float64) + delta,
        obb_size_xyz=np.asarray(box.obb_size_xyz, dtype=np.float64).copy(),
        obb_axes_xyz=np.asarray(box.obb_axes_xyz, dtype=np.float64).copy(),
        obb_corners_xyz=corners.astype(np.float64),
    )


def _enforce_pair_projection_contact(
    child: PartBox,
    parent: PartBox,
    *,
    min_overlap: float,
    max_shift: float,
) -> PartBox:
    if not child.valid or not parent.valid:
        return child
    child_center = np.asarray(child.obb_center_xyz, dtype=np.float64)
    parent_center = np.asarray(parent.obb_center_xyz, dtype=np.float64)
    direction = child_center - parent_center
    center_dist = float(np.linalg.norm(direction))
    if center_dist <= 1e-8:
        direction = np.asarray(child.obb_axes_xyz, dtype=np.float64)[:, 0]
        center_dist = float(np.linalg.norm(direction))
        if center_dist <= 1e-8:
            direction = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            center_dist = 1.0
    unit = direction / center_dist
    child_half = _projected_half_extent(child, unit)
    parent_half = _projected_half_extent(parent, unit)
    required_dist = max(child_half + parent_half - max(float(min_overlap), 0.0), 0.0)
    gap = center_dist - required_dist
    if gap <= 1e-6:
        return child
    shift_mag = min(gap, max(float(max_shift), 0.0))
    if shift_mag <= 1e-8:
        return child
    return _shift_box_obb(child, -unit * shift_mag)


def _enforce_hard_contact_links(
    boxes: Dict[str, PartBox],
    *,
    contact_margin: float,
    hard_contact_cfg: Mapping[str, Any] | None,
) -> Dict[str, PartBox]:
    cfg = hard_contact_cfg if isinstance(hard_contact_cfg, Mapping) else {}
    if not bool(cfg.get("enabled", True)):
        return boxes
    mode = str(cfg.get("mode", "shift")).strip().lower()
    iterations = max(1, int(cfg.get("iterations", 1)))
    min_overlap_default = max(float(cfg.get("min_overlap", 0.0)), 0.0)
    max_expand_per_axis = max(float(cfg.get("max_expand_per_axis", 0.05)), 1e-6)
    projection_contact_enabled = bool(cfg.get("projection_contact", True))
    projection_max_shift = max(float(cfg.get("projection_max_shift", 0.12)), 0.0)

    links = _parse_hard_contact_links(cfg.get("links"))
    if not links:
        links = [
            ("left_lower_arm", "left_upper_arm", None),
            ("right_lower_arm", "right_upper_arm", None),
            ("left_shin", "left_thigh", None),
            ("right_shin", "right_thigh", None),
        ]

    out = dict(boxes)
    for _ in range(iterations):
        for child_name, parent_name, link_min_overlap in links:
            child = out.get(child_name)
            parent = out.get(parent_name)
            if child is None or parent is None:
                continue
            if mode in ("expand", "mixed"):
                child_adj = _touch_child_to_parent_expand(child, parent, margin=contact_margin)
            else:
                child_adj = _touch_child_to_parent_shift(child, parent, margin=contact_margin)
            overlap_target = min_overlap_default if link_min_overlap is None else max(float(link_min_overlap), 0.0)
            child_adj = _enforce_pair_min_overlap(
                child_adj,
                parent,
                min_overlap=overlap_target,
                max_expand_per_axis=max_expand_per_axis,
            )
            if projection_contact_enabled:
                child_adj = _enforce_pair_projection_contact(
                    child_adj,
                    parent,
                    min_overlap=overlap_target,
                    max_shift=projection_max_shift,
                )
            out[child_name] = child_adj
    return out


def _enforce_box_hierarchy(
    boxes: Dict[str, PartBox],
    *,
    contact_margin: float,
    forbidden_margin: float,
    contact_mode: str = "mixed",
    hard_contact_cfg: Mapping[str, Any] | None = None,
) -> Dict[str, PartBox]:
    out = dict(boxes)
    mode = str(contact_mode).strip().lower()
    shift_all = mode in ("shift", "shift_all", "all_shift")
    shift_links = set(_PARENT_CHILD_LINKS if shift_all else _SHIFT_CONTACT_LINKS)
    for _ in range(2):
        for child_name, parent_name in _PARENT_CHILD_LINKS:
            child = out.get(child_name)
            parent = out.get(parent_name)
            if child is None or parent is None:
                continue
            if (child_name, parent_name) in shift_links:
                out[child_name] = _touch_child_to_parent_shift(child, parent, margin=contact_margin)
            else:
                out[child_name] = _touch_child_to_parent_expand(child, parent, margin=contact_margin)
        for child_name, forbidden_list in _FORBIDDEN_CONTACTS.items():
            child = out.get(child_name)
            if child is None or not child.valid:
                continue
            for forbidden_name in forbidden_list:
                other = out.get(forbidden_name)
                if other is None or not other.valid:
                    continue
                child = _separate_forbidden(child, other, margin=forbidden_margin)
            out[child_name] = child
    for child_name, parent_name in _PARENT_CHILD_LINKS:
        child = out.get(child_name)
        parent = out.get(parent_name)
        if child is None or parent is None:
            continue
        if (child_name, parent_name) in shift_links:
            out[child_name] = _touch_child_to_parent_shift(child, parent, margin=contact_margin)
        else:
            out[child_name] = _touch_child_to_parent_expand(child, parent, margin=contact_margin)
    out = _enforce_hard_contact_links(
        out,
        contact_margin=contact_margin,
        hard_contact_cfg=hard_contact_cfg,
    )
    return out


def _stabilize_boxes_temporal(
    boxes: Dict[str, PartBox],
    *,
    prev_state: Mapping[str, Any] | None,
    cfg: Mapping[str, Any],
) -> Tuple[Dict[str, PartBox], Dict[str, Any]]:
    enabled = bool(cfg.get("enabled", True))
    if not enabled:
        return boxes, {"enabled": False, "applied_parts": 0}
    prev_boxes = prev_state.get("part_boxes") if isinstance(prev_state, Mapping) else None
    if not isinstance(prev_boxes, Mapping):
        return boxes, {"enabled": True, "applied_parts": 0, "reason": "no_prev_state"}

    center_alpha = float(np.clip(float(cfg.get("center_alpha", 0.65)), 0.0, 1.0))
    size_alpha = float(np.clip(float(cfg.get("size_alpha", 0.55)), 0.0, 1.0))
    max_center_step_ratio = max(float(cfg.get("max_center_step_ratio", 0.85)), 0.0)
    max_size_delta_ratio = float(np.clip(float(cfg.get("max_size_delta_ratio", 0.35)), 0.0, 0.95))
    min_box_diag = max(float(cfg.get("min_box_diag", 0.08)), 1e-6)

    out = dict(boxes)
    applied_parts = 0
    clipped_center_steps = 0
    jump_before: list[float] = []
    jump_after: list[float] = []

    for name, box in boxes.items():
        if not box.valid:
            continue
        row = prev_boxes.get(name)
        row_map = row if isinstance(row, Mapping) else None
        if row_map is None or not bool(row_map.get("valid", False)):
            continue

        prev_center_raw = row_map.get("center_xyz")
        prev_size_raw = row_map.get("size_xyz")
        if (
            not isinstance(prev_center_raw, (list, tuple))
            or not isinstance(prev_size_raw, (list, tuple))
            or len(prev_center_raw) != 3
            or len(prev_size_raw) != 3
        ):
            continue

        prev_center = np.asarray(prev_center_raw, dtype=np.float64)
        prev_size = np.maximum(np.asarray(prev_size_raw, dtype=np.float64), 1e-6)
        curr_center = np.asarray(box.center_xyz, dtype=np.float64)
        curr_size = np.maximum(np.asarray(box.size_xyz, dtype=np.float64), 1e-6)

        base_diag = max(float(np.linalg.norm(prev_size)), float(np.linalg.norm(curr_size)), min_box_diag)
        delta = curr_center - prev_center
        jump = float(np.linalg.norm(delta))
        step_limit = max_center_step_ratio * base_diag
        if step_limit > 0.0 and jump > step_limit:
            delta = delta * (step_limit / max(jump, 1e-9))
            clipped_center_steps += 1
        target_center = prev_center + delta
        new_center = prev_center + center_alpha * (target_center - prev_center)

        ratio = curr_size / np.maximum(prev_size, 1e-6)
        ratio = np.clip(ratio, 1.0 - max_size_delta_ratio, 1.0 + max_size_delta_ratio)
        target_size = prev_size * ratio
        new_size = np.maximum(prev_size + size_alpha * (target_size - prev_size), 1e-6)

        new_min = new_center - 0.5 * new_size
        new_max = new_center + 0.5 * new_size
        out[name] = _build_box_from_bounds(
            name,
            num_points=box.num_points,
            min_xyz=new_min,
            max_xyz=new_max,
            template=box,
        )
        applied_parts += 1
        jump_before.append(jump / base_diag)
        jump_after.append(float(np.linalg.norm(new_center - prev_center)) / base_diag)

    meta = {
        "enabled": True,
        "applied_parts": int(applied_parts),
        "clipped_center_steps": int(clipped_center_steps),
        "jump_ratio_before_mean": float(np.mean(jump_before)) if jump_before else 0.0,
        "jump_ratio_after_mean": float(np.mean(jump_after)) if jump_after else 0.0,
    }
    return out, meta


def _serialize_part_boxes_state(boxes: Mapping[str, PartBox]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for name, box in boxes.items():
        out[name] = {
            "valid": bool(box.valid),
            "center_xyz": [float(v) for v in np.asarray(box.center_xyz, dtype=np.float64).tolist()],
            "size_xyz": [float(v) for v in np.asarray(box.size_xyz, dtype=np.float64).tolist()],
            "min_xyz": [float(v) for v in np.asarray(box.min_xyz, dtype=np.float64).tolist()],
            "max_xyz": [float(v) for v in np.asarray(box.max_xyz, dtype=np.float64).tolist()],
        }
    return out


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
    box_post_cfg = parser_cfg.get("box_postprocess", {}) if isinstance(parser_cfg.get("box_postprocess"), Mapping) else {}
    box_post_enabled = bool(box_post_cfg.get("enabled", True))
    allow_small_parts = bool(box_post_cfg.get("allow_small_parts", True))
    small_part_min_size = float(box_post_cfg.get("small_part_min_size", 0.025))
    box_robust_quantile = float(box_post_cfg.get("robust_quantile", 0.03))
    box_contact_mode = str(box_post_cfg.get("contact_mode", "mixed"))
    limb_shape_cfg = box_post_cfg.get("limb_shape", {}) if isinstance(box_post_cfg.get("limb_shape"), Mapping) else {}
    hard_contact_cfg = box_post_cfg.get("hard_contact", {}) if isinstance(box_post_cfg.get("hard_contact"), Mapping) else {}
    contact_margin = float(box_post_cfg.get("contact_margin", 0.003))
    forbidden_margin = float(box_post_cfg.get("forbidden_margin", 0.003))

    y_norm, y_min, y_max = _normalize_y(points)
    torso_cfg = parser_cfg.get("torso", {}) if isinstance(parser_cfg.get("torso"), Mapping) else {}
    torso_center, torso_half_width, torso_half_depth = _torso_anchor(points, y_norm, torso_cfg)

    head_cfg_raw = parser_cfg.get("head")
    head_cfg = head_cfg_raw if isinstance(head_cfg_raw, Mapping) else {}
    head_y = head_cfg.get("y_range", [0.8, 1.02])
    head_x_scale = float(head_cfg.get("x_scale", 1.25))
    head_z_scale = float(head_cfg.get("z_scale", 1.6))
    head_center_keep_quantile = float(head_cfg.get("center_keep_quantile", 0.65))
    head_max_radius = float(head_cfg.get("max_radius", 1.15))
    head_top_band_scale = float(head_cfg.get("top_band_scale", 0.12))
    head_keep_drop_scale = float(head_cfg.get("keep_drop_scale", 0.30))
    head_core_x_scale = float(head_cfg.get("core_x_scale", 2.2))
    head_core_z_scale = float(head_cfg.get("core_z_scale", 2.4))
    head_core_min_keep_ratio = float(head_cfg.get("core_min_keep_ratio", 0.35))

    torso_y = torso_cfg.get("y_range", [0.46, 0.78])
    torso_limb_cfg = torso_cfg.get("limb_exclusion", {}) if isinstance(torso_cfg.get("limb_exclusion"), Mapping) else {}
    torso_limb_excl_enabled = bool(torso_limb_cfg.get("enabled", True))
    torso_limb_assign_to_torso = bool(torso_limb_cfg.get("assign_to_torso", True))
    torso_limb_y_pad = float(torso_limb_cfg.get("y_pad", 0.08))
    torso_limb_x_scale = float(torso_limb_cfg.get("x_scale", 1.55))
    torso_limb_z_scale = float(torso_limb_cfg.get("z_scale", 1.30))
    torso_arm_salvage_cfg = (
        parser_cfg.get("torso_arm_salvage", {})
        if isinstance(parser_cfg.get("torso_arm_salvage"), Mapping)
        else {}
    )

    arm_cfg = parser_cfg.get("arm", {}) if isinstance(parser_cfg.get("arm"), Mapping) else {}
    arm_y = arm_cfg.get("y_range", [0.38, 0.82])
    arm_side_min = float(arm_cfg.get("side_min_scale", 0.65))

    leg_cfg = parser_cfg.get("leg", {}) if isinstance(parser_cfg.get("leg"), Mapping) else {}
    leg_y = leg_cfg.get("y_range", [0.0, 0.44])
    leg_side_min = float(leg_cfg.get("side_min_scale", 0.15))
    limb_cfg = parser_cfg.get("limb_assignment", {}) if isinstance(parser_cfg.get("limb_assignment"), Mapping) else {}
    shoulder_quantile = float(limb_cfg.get("shoulder_quantile", 0.80))
    hip_quantile = float(limb_cfg.get("hip_quantile", 0.20))
    arm_vertical_penalty = float(limb_cfg.get("arm_vertical_penalty", 0.90))
    leg_vertical_penalty = float(limb_cfg.get("leg_vertical_penalty", 0.90))
    limb_side_min = float(limb_cfg.get("min_side_scale", 0.08))
    y_prior_weight = float(limb_cfg.get("y_prior_weight", 0.70))
    top_arm_seed_fraction = float(limb_cfg.get("top_arm_seed_fraction", 0.12))
    bottom_leg_seed_fraction = float(limb_cfg.get("bottom_leg_seed_fraction", 0.18))
    cluster_eps = float(limb_cfg.get("cluster_eps", 0.085))
    cluster_min_points = int(limb_cfg.get("cluster_min_points", 6))
    seed_conf_fraction = float(limb_cfg.get("seed_conf_fraction", 0.10))
    geodesic_k = int(limb_cfg.get("geodesic_k", 10))
    geodesic_radius_scale = float(limb_cfg.get("geodesic_radius_scale", 1.8))
    geodesic_score_weight = float(limb_cfg.get("geodesic_score_weight", 0.35))
    arm_absence_cfg = (
        parser_cfg.get("arm_absence_guard", {})
        if isinstance(parser_cfg.get("arm_absence_guard"), Mapping)
        else {}
    )
    split_cfg = parser_cfg.get("segment_split", {}) if isinstance(parser_cfg.get("segment_split"), Mapping) else {}
    arm_prox_quantile = float(split_cfg.get("arm_prox_quantile", 0.55))
    leg_prox_quantile = float(split_cfg.get("leg_prox_quantile", 0.50))
    arm_prox_reclaim_cfg = (
        split_cfg.get("arm_prox_reclaim", {})
        if isinstance(split_cfg.get("arm_prox_reclaim"), Mapping)
        else {}
    )
    boundary_refine_cfg = (
        parser_cfg.get("boundary_refine", {})
        if isinstance(parser_cfg.get("boundary_refine"), Mapping)
        else {}
    )
    part_group_cfg = parser_cfg.get("part_grouping", {}) if isinstance(parser_cfg.get("part_grouping"), Mapping) else {}
    part_group_enabled = bool(part_group_cfg.get("enabled", True))
    part_group_eps = float(part_group_cfg.get("cluster_eps", 0.070))
    part_group_min_points = int(part_group_cfg.get("cluster_min_points", 6))
    part_group_noise_attach_radius = float(part_group_cfg.get("noise_attach_radius", 0.045))
    part_group_min_keep_ratio = float(part_group_cfg.get("min_keep_ratio", 0.50))
    part_group_secondary_center_radius = float(part_group_cfg.get("secondary_center_radius", 0.10))
    part_group_proximal_min_keep_ratio = float(part_group_cfg.get("proximal_min_keep_ratio", part_group_min_keep_ratio))
    part_group_proximal_max_anchor_quantile = float(part_group_cfg.get("proximal_max_anchor_quantile", 0.88))
    part_group_distal_min_keep_ratio = float(part_group_cfg.get("distal_min_keep_ratio", 0.35))
    part_group_distal_secondary_center_radius = float(part_group_cfg.get("distal_secondary_center_radius", 0.06))
    part_group_distal_noise_attach_radius = float(part_group_cfg.get("distal_noise_attach_radius", 0.025))
    part_group_distal_fallback_to_input = bool(part_group_cfg.get("distal_fallback_to_input", False))
    part_group_distal_max_anchor_quantile = float(part_group_cfg.get("distal_max_anchor_quantile", 0.90))
    hierarchy_cfg = parser_cfg.get("hierarchy", {}) if isinstance(parser_cfg.get("hierarchy"), Mapping) else {}
    head_arm_guard_cfg = (
        hierarchy_cfg.get("head_arm_guard", {})
        if isinstance(hierarchy_cfg.get("head_arm_guard"), Mapping)
        else {}
    )
    head_arm_guard_enabled = bool(head_arm_guard_cfg.get("enabled", True))
    head_arm_guard_y_pad = float(head_arm_guard_cfg.get("y_pad", 0.03))
    head_arm_guard_rx = float(head_arm_guard_cfg.get("radius_scale_x", 1.25))
    head_arm_guard_ry = float(head_arm_guard_cfg.get("radius_scale_y", 1.15))
    head_arm_guard_rz = float(head_arm_guard_cfg.get("radius_scale_z", 1.25))
    arm_torso_guard_cfg = (
        hierarchy_cfg.get("arm_torso_guard", {})
        if isinstance(hierarchy_cfg.get("arm_torso_guard"), Mapping)
        else {}
    )
    arm_torso_guard_enabled = bool(arm_torso_guard_cfg.get("enabled", True))
    arm_torso_guard_y_pad = float(arm_torso_guard_cfg.get("y_pad", 0.05))
    arm_torso_guard_x_scale = float(arm_torso_guard_cfg.get("x_scale", 1.45))
    arm_torso_guard_z_scale = float(arm_torso_guard_cfg.get("z_scale", 1.65))
    arm_torso_guard_shoulder_keep = float(arm_torso_guard_cfg.get("shoulder_keep_radius_scale", 0.95))
    arm_torso_guard_outward_min = float(arm_torso_guard_cfg.get("shoulder_outward_min", 0.015))
    arm_torso_guard_min_keep_ratio = float(arm_torso_guard_cfg.get("min_keep_ratio", 0.45))
    temporal_box_cfg = (
        parser_cfg.get("temporal_box_stabilization", {})
        if isinstance(parser_cfg.get("temporal_box_stabilization"), Mapping)
        else {}
    )

    torso_mask = (
        (y_norm >= float(torso_y[0]))
        & (y_norm <= float(torso_y[1]))
        & (np.abs(points[:, 0] - torso_center[0]) <= torso_half_width)
        & (np.abs(points[:, 2] - torso_center[2]) <= torso_half_depth)
    )
    torso_points = points[torso_mask]
    if torso_points.shape[0] < min_points_per_part:
        relaxed_torso_mask = (
            (y_norm >= float(torso_y[0]))
            & (y_norm <= float(torso_y[1]))
            & (np.abs(points[:, 2] - torso_center[2]) <= torso_half_depth * 1.35)
        )
        torso_points = points[relaxed_torso_mask]
        torso_mask = relaxed_torso_mask
    if torso_points.shape[0] < min_points_per_part:
        torso_mask = (y_norm >= float(torso_y[0])) & (y_norm <= float(torso_y[1]))
        torso_points = points[torso_mask]
    if torso_points.shape[0] > 0:
        torso_center = np.mean(torso_points, axis=0)
        abs_x_torso = np.abs(torso_points[:, 0] - torso_center[0])
        abs_z_torso = np.abs(torso_points[:, 2] - torso_center[2])
        torso_half_width = max(float(np.percentile(abs_x_torso, 85.0)), 0.05)
        torso_half_depth = max(float(np.percentile(abs_z_torso, 85.0)), 0.04)

    head_candidate_mask = (y_norm >= float(head_y[0])) & (y_norm <= float(head_y[1]))
    head_candidate_indices = np.flatnonzero(head_candidate_mask)
    head_mask = np.zeros((points.shape[0],), dtype=bool)
    if head_candidate_indices.size > 0:
        head_candidates = points[head_candidate_indices]
        rx = max(torso_half_width * max(head_x_scale, 0.25), 0.05)
        rz = max(torso_half_depth * max(head_z_scale, 0.25), 0.04)
        radial = np.sqrt(
            ((head_candidates[:, 0] - torso_center[0]) / rx) ** 2
            + ((head_candidates[:, 2] - torso_center[2]) / rz) ** 2
        )
        quantile = float(np.clip(head_center_keep_quantile, 0.2, 1.0))
        radial_thresh = min(float(np.quantile(radial, quantile)), max(head_max_radius, 0.1))
        keep_local = radial <= (radial_thresh + 1e-9)
        min_keep = min(max(min_points_per_part, 1), int(head_candidate_indices.size))
        if int(np.count_nonzero(keep_local)) < min_keep:
            nearest_idx = np.argpartition(radial, min_keep - 1)[:min_keep]
            keep_local = np.zeros_like(keep_local, dtype=bool)
            keep_local[nearest_idx] = True
        head_mask[head_candidate_indices[keep_local]] = True
    head_points = points[head_mask]
    if head_points.shape[0] > 0:
        head_points_refined = _refine_head_points_from_top_core(
            head_points,
            y_min=y_min,
            y_max=y_max,
            torso_half_width=torso_half_width,
            torso_half_depth=torso_half_depth,
            cfg={
                "top_band_scale": head_top_band_scale,
                "keep_drop_scale": head_keep_drop_scale,
                "core_x_scale": head_core_x_scale,
                "core_z_scale": head_core_z_scale,
                "core_min_keep_ratio": head_core_min_keep_ratio,
            },
            min_points=min_points_per_part,
        )
        if head_points_refined.shape[0] != head_points.shape[0]:
            refined_lookup = _point_lookup_exact(head_points_refined)
            refined_mask = np.zeros((points.shape[0],), dtype=bool)
            for point_idx, row in enumerate(points):
                key = (float(row[0]), float(row[1]), float(row[2]))
                if refined_lookup.get(key):
                    refined_mask[point_idx] = True
            head_mask = refined_mask
            head_points = points[head_mask]

    torso_mask = torso_mask & (~head_mask)
    torso_points = points[torso_mask]
    if torso_points.shape[0] < min_points_per_part:
        fallback_torso_mask = ((y_norm >= float(torso_y[0])) & (y_norm <= float(torso_y[1])) & (~head_mask))
        torso_points = points[fallback_torso_mask]
        torso_mask = fallback_torso_mask

    torso_arm_salvage_mask = np.zeros((points.shape[0],), dtype=bool)
    if bool(torso_arm_salvage_cfg.get("enabled", True)) and torso_points.shape[0] > 0:
        salvage_y_margin = max(float(torso_arm_salvage_cfg.get("y_margin", 0.03)), 0.0)
        salvage_edge_percentile = float(np.clip(float(torso_arm_salvage_cfg.get("edge_percentile", 0.72)), 0.50, 0.98))
        salvage_min_lateral = max(float(torso_arm_salvage_cfg.get("min_lateral", 0.055)), 0.0)
        salvage_max_ratio = float(np.clip(float(torso_arm_salvage_cfg.get("max_ratio", 0.28)), 0.05, 0.80))
        salvage_min_points = max(int(torso_arm_salvage_cfg.get("min_points", 12)), 1)
        salvage_max_shoulder_dist = max(float(torso_arm_salvage_cfg.get("max_shoulder_dist", 0.42)), 0.05)

        torso_idx = np.flatnonzero(torso_mask)
        if torso_idx.size >= salvage_min_points:
            left_shoulder_seed = _estimate_side_anchor(
                torso_points,
                torso_center,
                torso_half_width,
                left=True,
                left_is_positive_x=left_is_positive_x,
                top=True,
                y_quantile=shoulder_quantile,
            )
            right_shoulder_seed = _estimate_side_anchor(
                torso_points,
                torso_center,
                torso_half_width,
                left=False,
                left_is_positive_x=left_is_positive_x,
                top=True,
                y_quantile=shoulder_quantile,
            )
            torso_abs_x = np.abs(points[torso_idx, 0] - torso_center[0])
            lateral_thr = max(float(np.quantile(torso_abs_x, salvage_edge_percentile)), salvage_min_lateral)
            y_lo_arm = float(arm_y[0]) - salvage_y_margin
            y_hi_arm = float(arm_y[1]) + salvage_y_margin
            salvage_candidate = (
                torso_mask
                & (~head_mask)
                & (y_norm >= y_lo_arm)
                & (y_norm <= y_hi_arm)
                & (np.abs(points[:, 0] - torso_center[0]) >= lateral_thr)
            )
            cand_idx = np.flatnonzero(salvage_candidate)
            if cand_idx.size > 0:
                cand_points = points[cand_idx]
                d_left = np.linalg.norm(cand_points - left_shoulder_seed[None, :], axis=1)
                d_right = np.linalg.norm(cand_points - right_shoulder_seed[None, :], axis=1)
                near_shoulder = np.minimum(d_left, d_right) <= salvage_max_shoulder_dist
                cand_idx = cand_idx[near_shoulder]
            if cand_idx.size > 0:
                max_take = max(1, int(np.floor(torso_idx.size * salvage_max_ratio)))
                if cand_idx.size > max_take:
                    order = np.argsort(np.abs(points[cand_idx, 0] - torso_center[0]))[::-1]
                    cand_idx = cand_idx[order[:max_take]]
                torso_arm_salvage_mask[cand_idx] = True
                torso_mask[cand_idx] = False
                torso_points = points[torso_mask]
                if torso_points.shape[0] > 0:
                    torso_center = np.mean(torso_points, axis=0)
                    abs_x_torso = np.abs(torso_points[:, 0] - torso_center[0])
                    abs_z_torso = np.abs(torso_points[:, 2] - torso_center[2])
                    torso_half_width = max(float(np.percentile(abs_x_torso, 85.0)), 0.05)
                    torso_half_depth = max(float(np.percentile(abs_z_torso, 85.0)), 0.04)

    torso_limb_extra_mask = np.zeros((points.shape[0],), dtype=bool)
    if torso_limb_excl_enabled:
        y_lo = float(torso_y[0]) - max(0.0, torso_limb_y_pad)
        y_hi = float(torso_y[1]) + max(0.0, torso_limb_y_pad)
        torso_limb_candidate = (
            (y_norm >= y_lo)
            & (y_norm <= y_hi)
            & (np.abs(points[:, 0] - torso_center[0]) <= torso_half_width * max(torso_limb_x_scale, 1.0))
            & (np.abs(points[:, 2] - torso_center[2]) <= torso_half_depth * max(torso_limb_z_scale, 1.0))
            & (~head_mask)
            & (~torso_arm_salvage_mask)
        )
        torso_limb_extra_mask = torso_limb_candidate & (~torso_mask)
        if torso_limb_assign_to_torso and int(np.count_nonzero(torso_limb_extra_mask)) > 0:
            torso_mask = torso_mask | torso_limb_extra_mask
            torso_points = points[torso_mask]
            if torso_points.shape[0] > 0:
                torso_center = np.mean(torso_points, axis=0)
                abs_x_torso = np.abs(torso_points[:, 0] - torso_center[0])
                abs_z_torso = np.abs(torso_points[:, 2] - torso_center[2])
                torso_half_width = max(float(np.percentile(abs_x_torso, 85.0)), 0.05)
                torso_half_depth = max(float(np.percentile(abs_z_torso, 85.0)), 0.04)
        else:
            torso_limb_extra_mask = torso_limb_candidate & (~torso_mask)
    torso_limb_block_mask = torso_mask | torso_limb_extra_mask

    limb_y_lo = min(float(arm_y[0]), float(leg_y[0]))
    limb_y_hi = max(float(arm_y[1]), float(leg_y[1]))
    limb_mask = (y_norm >= limb_y_lo) & (y_norm <= limb_y_hi) & (~head_mask) & (~torso_limb_block_mask)
    limb_points = points[limb_mask]
    limb_y_norm = y_norm[limb_mask]
    left_side_mask = _side_selector(limb_points, torso_center[0], left=True, left_is_positive_x=left_is_positive_x)
    right_side_mask = ~left_side_mask
    left_limb_all = limb_points[left_side_mask]
    right_limb_all = limb_points[right_side_mask]
    left_limb_y = limb_y_norm[left_side_mask]
    right_limb_y = limb_y_norm[right_side_mask]

    left_shoulder_anchor = _estimate_side_anchor(
        torso_points,
        torso_center,
        torso_half_width,
        left=True,
        left_is_positive_x=left_is_positive_x,
        top=True,
        y_quantile=shoulder_quantile,
    )
    right_shoulder_anchor = _estimate_side_anchor(
        torso_points,
        torso_center,
        torso_half_width,
        left=False,
        left_is_positive_x=left_is_positive_x,
        top=True,
        y_quantile=shoulder_quantile,
    )
    left_hip_anchor = _estimate_side_anchor(
        torso_points,
        torso_center,
        torso_half_width,
        left=True,
        left_is_positive_x=left_is_positive_x,
        top=False,
        y_quantile=hip_quantile,
    )
    right_hip_anchor = _estimate_side_anchor(
        torso_points,
        torso_center,
        torso_half_width,
        left=False,
        left_is_positive_x=left_is_positive_x,
        top=False,
        y_quantile=hip_quantile,
    )

    left_arm_all, left_leg_all = _split_arm_leg_by_anchors(
        left_limb_all,
        left_limb_y,
        left_shoulder_anchor,
        left_hip_anchor,
        arm_vertical_penalty,
        leg_vertical_penalty,
        (float(arm_y[0]), float(arm_y[1])),
        (float(leg_y[0]), float(leg_y[1])),
        y_prior_weight,
        top_arm_seed_fraction,
        bottom_leg_seed_fraction,
        cluster_eps,
        cluster_min_points,
        seed_conf_fraction,
        geodesic_k,
        geodesic_radius_scale,
        geodesic_score_weight,
    )
    right_arm_all, right_leg_all = _split_arm_leg_by_anchors(
        right_limb_all,
        right_limb_y,
        right_shoulder_anchor,
        right_hip_anchor,
        arm_vertical_penalty,
        leg_vertical_penalty,
        (float(arm_y[0]), float(arm_y[1])),
        (float(leg_y[0]), float(leg_y[1])),
        y_prior_weight,
        top_arm_seed_fraction,
        bottom_leg_seed_fraction,
        cluster_eps,
        cluster_min_points,
        seed_conf_fraction,
        geodesic_k,
        geodesic_radius_scale,
        geodesic_score_weight,
    )

    side_thresh_arm = torso_half_width * max(arm_side_min, limb_side_min)
    side_thresh_leg = torso_half_width * max(leg_side_min, limb_side_min)
    left_arm_all = left_arm_all[np.abs(left_arm_all[:, 0] - torso_center[0]) >= side_thresh_arm]
    right_arm_all = right_arm_all[np.abs(right_arm_all[:, 0] - torso_center[0]) >= side_thresh_arm]
    left_leg_all = left_leg_all[np.abs(left_leg_all[:, 0] - torso_center[0]) >= side_thresh_leg]
    right_leg_all = right_leg_all[np.abs(right_leg_all[:, 0] - torso_center[0]) >= side_thresh_leg]

    if arm_torso_guard_enabled:
        left_arm_all = _exclude_torso_zone_from_arms(
            left_arm_all,
            left_shoulder_anchor,
            torso_center,
            torso_half_width,
            torso_half_depth,
            y_range=(float(arm_y[0]), float(arm_y[1])),
            y_min=y_min,
            y_max=y_max,
            y_pad=arm_torso_guard_y_pad,
            x_scale=arm_torso_guard_x_scale,
            z_scale=arm_torso_guard_z_scale,
            shoulder_keep_radius_scale=arm_torso_guard_shoulder_keep,
            shoulder_outward_min=arm_torso_guard_outward_min,
            min_keep_ratio=arm_torso_guard_min_keep_ratio,
        )
        right_arm_all = _exclude_torso_zone_from_arms(
            right_arm_all,
            right_shoulder_anchor,
            torso_center,
            torso_half_width,
            torso_half_depth,
            y_range=(float(arm_y[0]), float(arm_y[1])),
            y_min=y_min,
            y_max=y_max,
            y_pad=arm_torso_guard_y_pad,
            x_scale=arm_torso_guard_x_scale,
            z_scale=arm_torso_guard_z_scale,
            shoulder_keep_radius_scale=arm_torso_guard_shoulder_keep,
            shoulder_outward_min=arm_torso_guard_outward_min,
            min_keep_ratio=arm_torso_guard_min_keep_ratio,
        )

    if head_arm_guard_enabled:
        left_arm_all = _exclude_head_zone_from_limbs(
            left_arm_all,
            head_points,
            torso_half_width=torso_half_width,
            torso_half_depth=torso_half_depth,
            y_pad=head_arm_guard_y_pad,
            radius_scale_x=head_arm_guard_rx,
            radius_scale_y=head_arm_guard_ry,
            radius_scale_z=head_arm_guard_rz,
        )
        right_arm_all = _exclude_head_zone_from_limbs(
            right_arm_all,
            head_points,
            torso_half_width=torso_half_width,
            torso_half_depth=torso_half_depth,
            y_pad=head_arm_guard_y_pad,
            radius_scale_x=head_arm_guard_rx,
            radius_scale_y=head_arm_guard_ry,
            radius_scale_z=head_arm_guard_rz,
        )

    left_arm_all, left_leg_all, left_arm_absent = _apply_arm_absence_guard(
        left_arm_all,
        left_leg_all,
        shoulder_anchor=left_shoulder_anchor,
        side_total_points=int(left_limb_all.shape[0]),
        y_min=y_min,
        y_max=y_max,
        cfg=arm_absence_cfg,
    )
    right_arm_all, right_leg_all, right_arm_absent = _apply_arm_absence_guard(
        right_arm_all,
        right_leg_all,
        shoulder_anchor=right_shoulder_anchor,
        side_total_points=int(right_limb_all.shape[0]),
        y_min=y_min,
        y_max=y_max,
        cfg=arm_absence_cfg,
    )

    left_upper, left_lower = _assign_proximal_distal_by_anchor(
        left_arm_all,
        anchor=left_shoulder_anchor,
        quantile=arm_prox_quantile,
        prev_prox=_prev_center(prev_state, "left_upper_arm") if continuity_enabled else None,
        prev_dist=_prev_center(prev_state, "left_lower_arm") if continuity_enabled else None,
    )
    right_upper, right_lower = _assign_proximal_distal_by_anchor(
        right_arm_all,
        anchor=right_shoulder_anchor,
        quantile=arm_prox_quantile,
        prev_prox=_prev_center(prev_state, "right_upper_arm") if continuity_enabled else None,
        prev_dist=_prev_center(prev_state, "right_lower_arm") if continuity_enabled else None,
    )
    left_thigh, left_shin = _assign_proximal_distal_by_anchor(
        left_leg_all,
        anchor=left_hip_anchor,
        quantile=leg_prox_quantile,
        prev_prox=_prev_center(prev_state, "left_thigh") if continuity_enabled else None,
        prev_dist=_prev_center(prev_state, "left_shin") if continuity_enabled else None,
    )
    right_thigh, right_shin = _assign_proximal_distal_by_anchor(
        right_leg_all,
        anchor=right_hip_anchor,
        quantile=leg_prox_quantile,
        prev_prox=_prev_center(prev_state, "right_thigh") if continuity_enabled else None,
        prev_dist=_prev_center(prev_state, "right_shin") if continuity_enabled else None,
    )

    if part_group_enabled:
        left_upper = _filter_part_component_by_anchor(
            left_upper,
            left_shoulder_anchor,
            eps=part_group_eps,
            min_points=part_group_min_points,
            noise_attach_radius=part_group_noise_attach_radius,
            min_keep_ratio=part_group_min_keep_ratio,
            secondary_center_radius=part_group_secondary_center_radius,
        )
        right_upper = _filter_part_component_by_anchor(
            right_upper,
            right_shoulder_anchor,
            eps=part_group_eps,
            min_points=part_group_min_points,
            noise_attach_radius=part_group_noise_attach_radius,
            min_keep_ratio=part_group_min_keep_ratio,
            secondary_center_radius=part_group_secondary_center_radius,
        )
        left_thigh = _filter_part_component_by_anchor(
            left_thigh,
            left_hip_anchor,
            eps=part_group_eps,
            min_points=part_group_min_points,
            noise_attach_radius=part_group_noise_attach_radius,
            min_keep_ratio=part_group_min_keep_ratio,
            secondary_center_radius=part_group_secondary_center_radius,
        )
        right_thigh = _filter_part_component_by_anchor(
            right_thigh,
            right_hip_anchor,
            eps=part_group_eps,
            min_points=part_group_min_points,
            noise_attach_radius=part_group_noise_attach_radius,
            min_keep_ratio=part_group_min_keep_ratio,
            secondary_center_radius=part_group_secondary_center_radius,
        )
        left_upper = _trim_far_points_from_anchor(
            left_upper,
            left_shoulder_anchor,
            max_anchor_quantile=part_group_proximal_max_anchor_quantile,
            min_keep_ratio=part_group_proximal_min_keep_ratio,
            min_points=part_group_min_points,
        )
        right_upper = _trim_far_points_from_anchor(
            right_upper,
            right_shoulder_anchor,
            max_anchor_quantile=part_group_proximal_max_anchor_quantile,
            min_keep_ratio=part_group_proximal_min_keep_ratio,
            min_points=part_group_min_points,
        )
        left_thigh = _trim_far_points_from_anchor(
            left_thigh,
            left_hip_anchor,
            max_anchor_quantile=part_group_proximal_max_anchor_quantile,
            min_keep_ratio=part_group_proximal_min_keep_ratio,
            min_points=part_group_min_points,
        )
        right_thigh = _trim_far_points_from_anchor(
            right_thigh,
            right_hip_anchor,
            max_anchor_quantile=part_group_proximal_max_anchor_quantile,
            min_keep_ratio=part_group_proximal_min_keep_ratio,
            min_points=part_group_min_points,
        )
        left_lower_anchor = (
            np.mean(left_upper, axis=0)
            if left_upper.shape[0] > 0
            else left_shoulder_anchor
        )
        right_lower_anchor = (
            np.mean(right_upper, axis=0)
            if right_upper.shape[0] > 0
            else right_shoulder_anchor
        )
        left_shin_anchor = (
            np.mean(left_thigh, axis=0)
            if left_thigh.shape[0] > 0
            else left_hip_anchor
        )
        right_shin_anchor = (
            np.mean(right_thigh, axis=0)
            if right_thigh.shape[0] > 0
            else right_hip_anchor
        )
        left_lower = _filter_part_component_by_anchor(
            left_lower,
            left_lower_anchor,
            eps=part_group_eps,
            min_points=part_group_min_points,
            noise_attach_radius=part_group_distal_noise_attach_radius,
            min_keep_ratio=part_group_distal_min_keep_ratio,
            secondary_center_radius=part_group_distal_secondary_center_radius,
            fallback_to_input=part_group_distal_fallback_to_input,
        )
        right_lower = _filter_part_component_by_anchor(
            right_lower,
            right_lower_anchor,
            eps=part_group_eps,
            min_points=part_group_min_points,
            noise_attach_radius=part_group_distal_noise_attach_radius,
            min_keep_ratio=part_group_distal_min_keep_ratio,
            secondary_center_radius=part_group_distal_secondary_center_radius,
            fallback_to_input=part_group_distal_fallback_to_input,
        )
        left_shin = _filter_part_component_by_anchor(
            left_shin,
            left_shin_anchor,
            eps=part_group_eps,
            min_points=part_group_min_points,
            noise_attach_radius=part_group_noise_attach_radius,
            min_keep_ratio=part_group_min_keep_ratio,
            secondary_center_radius=part_group_secondary_center_radius,
        )
        right_shin = _filter_part_component_by_anchor(
            right_shin,
            right_shin_anchor,
            eps=part_group_eps,
            min_points=part_group_min_points,
            noise_attach_radius=part_group_noise_attach_radius,
            min_keep_ratio=part_group_min_keep_ratio,
            secondary_center_radius=part_group_secondary_center_radius,
        )
        left_lower = _trim_far_points_from_anchor(
            left_lower,
            left_lower_anchor,
            max_anchor_quantile=part_group_distal_max_anchor_quantile,
            min_keep_ratio=part_group_distal_min_keep_ratio,
            min_points=part_group_min_points,
        )
        right_lower = _trim_far_points_from_anchor(
            right_lower,
            right_lower_anchor,
            max_anchor_quantile=part_group_distal_max_anchor_quantile,
            min_keep_ratio=part_group_distal_min_keep_ratio,
            min_points=part_group_min_points,
        )

    left_upper, left_lower = _reclaim_arm_proximal_points(
        left_upper,
        left_lower,
        shoulder_anchor=left_shoulder_anchor,
        cfg=arm_prox_reclaim_cfg,
    )
    right_upper, right_lower = _reclaim_arm_proximal_points(
        right_upper,
        right_lower,
        shoulder_anchor=right_shoulder_anchor,
        cfg=arm_prox_reclaim_cfg,
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
    point_labels = _assign_labels_from_part_points(points, part_points)
    point_labels_refined, boundary_meta = _refine_part_boundaries(
        points=points,
        y_norm=y_norm,
        labels=point_labels,
        part_points=part_points,
        torso_center=torso_center,
        torso_half_width=torso_half_width,
        torso_half_depth=torso_half_depth,
        arm_y=(float(arm_y[0]), float(arm_y[1])),
        leg_y=(float(leg_y[0]), float(leg_y[1])),
        head_y=(float(head_y[0]), float(head_y[1])),
        left_is_positive_x=left_is_positive_x,
        cfg=boundary_refine_cfg,
    )
    part_points = _rebuild_part_points_from_labels(points, point_labels_refined)
    head_points = part_points["head"]
    torso_points = part_points["torso"]
    left_upper = part_points["left_upper_arm"]
    left_lower = part_points["left_lower_arm"]
    right_upper = part_points["right_upper_arm"]
    right_lower = part_points["right_lower_arm"]
    left_thigh = part_points["left_thigh"]
    left_shin = part_points["left_shin"]
    right_thigh = part_points["right_thigh"]
    right_shin = part_points["right_shin"]

    arm_points = (
        np.concatenate([left_upper, left_lower, right_upper, right_lower], axis=0)
        if (left_upper.size or left_lower.size or right_upper.size or right_lower.size)
        else _empty_points()
    )
    leg_points = (
        np.concatenate([left_thigh, left_shin, right_thigh, right_shin], axis=0)
        if (left_thigh.size or left_shin.size or right_thigh.size or right_shin.size)
        else _empty_points()
    )

    boxes: Dict[str, PartBox] = {}
    for name in DEFAULT_PART_ORDER:
        pts = part_points.get(name, _empty_points())
        box = PartBox.from_points(
            name,
            pts,
            min_points=min_points_per_part,
            allow_small=allow_small_parts,
            min_size=small_part_min_size,
            robust_quantile=box_robust_quantile,
        )
        prev_center = _prev_center(prev_state, name) if continuity_enabled else None
        if box.valid and prev_center is not None and 0.0 < blend_prev < 1.0:
            blend_center = (1.0 - blend_prev) * box.center_xyz + blend_prev * prev_center
            center_delta = blend_center - box.center_xyz
            box = PartBox(
                name=box.name,
                valid=box.valid,
                num_points=box.num_points,
                min_xyz=box.min_xyz,
                max_xyz=box.max_xyz,
                center_xyz=blend_center,
                size_xyz=box.size_xyz,
                obb_center_xyz=box.obb_center_xyz + center_delta,
                obb_size_xyz=box.obb_size_xyz,
                obb_axes_xyz=box.obb_axes_xyz,
                obb_corners_xyz=box.obb_corners_xyz + center_delta[None, :],
            )
        boxes[name] = box
    continuity_boxes = dict(boxes)

    if box_post_enabled:
        boxes = _enforce_limb_shape_boxes(
            boxes,
            part_points,
            cfg=limb_shape_cfg,
            min_points=min_points_per_part,
        )

    if box_post_enabled:
        boxes = _enforce_box_hierarchy(
            boxes,
            contact_margin=max(contact_margin, 0.0),
            forbidden_margin=max(forbidden_margin, 0.0),
            contact_mode=box_contact_mode,
            hard_contact_cfg=hard_contact_cfg,
        )
    if continuity_enabled:
        boxes, temporal_box_meta = _stabilize_boxes_temporal(
            boxes,
            prev_state=prev_state,
            cfg=temporal_box_cfg,
        )
    else:
        temporal_box_meta = {"enabled": bool(temporal_box_cfg.get("enabled", True)), "applied_parts": 0, "reason": "continuity_disabled"}
    if box_post_enabled:
        boxes = _enforce_box_hierarchy(
            boxes,
            contact_margin=max(contact_margin, 0.0),
            forbidden_margin=max(forbidden_margin, 0.0),
            contact_mode=box_contact_mode,
            hard_contact_cfg=hard_contact_cfg,
        )

    part_centers: Dict[str, np.ndarray] = {}
    for name in DEFAULT_PART_ORDER:
        box = boxes.get(name, continuity_boxes.get(name))
        if box is None:
            continue
        if box.valid:
            part_centers[name] = box.center_xyz

    info = {
        "input_points": int(points.shape[0]),
        "valid_parts": int(sum(1 for box in boxes.values() if box.valid)),
        "torso_center": [float(v) for v in torso_center.tolist()],
        "torso_half_width": float(torso_half_width),
        "torso_half_depth": float(torso_half_depth),
        "head_points": int(head_points.shape[0]),
        "torso_points": int(torso_points.shape[0]),
        "arm_points": int(arm_points.shape[0]),
        "leg_points": int(leg_points.shape[0]),
        "anchors": {
            "left_shoulder": [float(v) for v in left_shoulder_anchor.tolist()],
            "right_shoulder": [float(v) for v in right_shoulder_anchor.tolist()],
            "left_hip": [float(v) for v in left_hip_anchor.tolist()],
            "right_hip": [float(v) for v in right_hip_anchor.tolist()],
        },
        "y_min": float(y_min),
        "y_max": float(y_max),
        "left_is_positive_x": left_is_positive_x,
        "continuity_enabled": continuity_enabled,
        "box_postprocess_enabled": box_post_enabled,
        "box_contact_mode": box_contact_mode,
        "hard_contact_enabled": bool(hard_contact_cfg.get("enabled", True)),
        "hard_contact_mode": str(hard_contact_cfg.get("mode", "shift")),
        "hard_contact_min_overlap": float(hard_contact_cfg.get("min_overlap", 0.0)),
        "hard_contact_projection_contact": bool(hard_contact_cfg.get("projection_contact", True)),
        "hard_contact_projection_max_shift": float(hard_contact_cfg.get("projection_max_shift", 0.12)),
        "limb_shape_enabled": bool(limb_shape_cfg.get("enabled", True)),
        "limb_shape_align_to_links": bool(limb_shape_cfg.get("align_to_links", True)),
        "limb_shape_use_torso_secondary_legs": bool(limb_shape_cfg.get("use_torso_secondary_legs", False)),
        "limb_shape_leg_direction_blend_scale": float(limb_shape_cfg.get("leg_direction_blend_scale", 0.55)),
        "limb_shape_min_aspect_ratio": float(limb_shape_cfg.get("min_aspect_ratio", 1.2)),
        "allow_small_parts": allow_small_parts,
        "box_robust_quantile": float(box_robust_quantile),
        "torso_limb_exclusion_enabled": torso_limb_excl_enabled,
        "torso_limb_assign_to_torso": torso_limb_assign_to_torso,
        "torso_limb_extra_points": int(np.count_nonzero(torso_limb_extra_mask)),
        "torso_arm_salvage_enabled": bool(torso_arm_salvage_cfg.get("enabled", True)),
        "torso_arm_salvage_points": int(np.count_nonzero(torso_arm_salvage_mask)),
        "head_top_band_scale": float(head_top_band_scale),
        "head_keep_drop_scale": float(head_keep_drop_scale),
        "head_arm_guard_enabled": head_arm_guard_enabled,
        "arm_torso_guard_enabled": arm_torso_guard_enabled,
        "arm_absence_guard_enabled": bool(arm_absence_cfg.get("enabled", True)),
        "left_arm_absent_guard": bool(left_arm_absent),
        "right_arm_absent_guard": bool(right_arm_absent),
        "arm_prox_reclaim_enabled": bool(arm_prox_reclaim_cfg.get("enabled", True)),
        "boundary_refine": boundary_meta,
        "part_grouping_enabled": part_group_enabled,
        "part_group_proximal_min_keep_ratio": float(part_group_proximal_min_keep_ratio),
        "part_group_proximal_max_anchor_quantile": float(part_group_proximal_max_anchor_quantile),
        "part_group_distal_min_keep_ratio": float(part_group_distal_min_keep_ratio),
        "part_group_distal_secondary_center_radius": float(part_group_distal_secondary_center_radius),
        "part_group_distal_noise_attach_radius": float(part_group_distal_noise_attach_radius),
        "part_group_distal_fallback_to_input": bool(part_group_distal_fallback_to_input),
        "part_group_distal_max_anchor_quantile": float(part_group_distal_max_anchor_quantile),
        "temporal_box_stabilization": temporal_box_meta,
    }
    next_state = {
        "part_centers": {k: [float(c) for c in v.tolist()] for k, v in part_centers.items()},
        "part_boxes": _serialize_part_boxes_state(boxes),
    }
    return boxes, info, part_points, next_state
