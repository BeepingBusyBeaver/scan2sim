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
            "head": {
                "y_range": [0.80, 1.02],
                "x_scale": 1.25,
                "z_scale": 1.6,
                "center_keep_quantile": 0.65,
                "max_radius": 1.15,
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
            "segment_split": {
                "arm_prox_quantile": 0.55,
                "leg_prox_quantile": 0.50,
            },
            "part_grouping": {
                "enabled": True,
                "cluster_eps": 0.070,
                "cluster_min_points": 6,
                "noise_attach_radius": 0.045,
                "min_keep_ratio": 0.50,
                "secondary_center_radius": 0.10,
            },
            "hierarchy": {
                "head_arm_guard": {
                    "enabled": True,
                    "y_pad": 0.03,
                    "radius_scale_x": 1.25,
                    "radius_scale_y": 1.15,
                    "radius_scale_z": 1.25,
                }
            },
            "box_postprocess": {
                "enabled": True,
                "allow_small_parts": True,
                "small_part_min_size": 0.025,
                "robust_quantile": 0.03,
                "contact_margin": 0.003,
                "forbidden_margin": 0.003,
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
        return points
    return selected


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


def _touch_child_to_parent(child: PartBox, parent: PartBox, margin: float) -> PartBox:
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


def _enforce_box_hierarchy(
    boxes: Dict[str, PartBox],
    *,
    contact_margin: float,
    forbidden_margin: float,
) -> Dict[str, PartBox]:
    out = dict(boxes)
    for _ in range(2):
        for child_name, parent_name in _PARENT_CHILD_LINKS:
            child = out.get(child_name)
            parent = out.get(parent_name)
            if child is None or parent is None:
                continue
            out[child_name] = _touch_child_to_parent(child, parent, margin=contact_margin)
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

    torso_y = torso_cfg.get("y_range", [0.46, 0.78])
    torso_limb_cfg = torso_cfg.get("limb_exclusion", {}) if isinstance(torso_cfg.get("limb_exclusion"), Mapping) else {}
    torso_limb_excl_enabled = bool(torso_limb_cfg.get("enabled", True))
    torso_limb_assign_to_torso = bool(torso_limb_cfg.get("assign_to_torso", True))
    torso_limb_y_pad = float(torso_limb_cfg.get("y_pad", 0.08))
    torso_limb_x_scale = float(torso_limb_cfg.get("x_scale", 1.55))
    torso_limb_z_scale = float(torso_limb_cfg.get("z_scale", 1.30))

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
    split_cfg = parser_cfg.get("segment_split", {}) if isinstance(parser_cfg.get("segment_split"), Mapping) else {}
    arm_prox_quantile = float(split_cfg.get("arm_prox_quantile", 0.55))
    leg_prox_quantile = float(split_cfg.get("leg_prox_quantile", 0.50))
    part_group_cfg = parser_cfg.get("part_grouping", {}) if isinstance(parser_cfg.get("part_grouping"), Mapping) else {}
    part_group_enabled = bool(part_group_cfg.get("enabled", True))
    part_group_eps = float(part_group_cfg.get("cluster_eps", 0.070))
    part_group_min_points = int(part_group_cfg.get("cluster_min_points", 6))
    part_group_noise_attach_radius = float(part_group_cfg.get("noise_attach_radius", 0.045))
    part_group_min_keep_ratio = float(part_group_cfg.get("min_keep_ratio", 0.50))
    part_group_secondary_center_radius = float(part_group_cfg.get("secondary_center_radius", 0.10))
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

    torso_mask = torso_mask & (~head_mask)
    torso_points = points[torso_mask]
    if torso_points.shape[0] < min_points_per_part:
        fallback_torso_mask = ((y_norm >= float(torso_y[0])) & (y_norm <= float(torso_y[1])) & (~head_mask))
        torso_points = points[fallback_torso_mask]
        torso_mask = fallback_torso_mask

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
            noise_attach_radius=part_group_noise_attach_radius,
            min_keep_ratio=part_group_min_keep_ratio,
            secondary_center_radius=part_group_secondary_center_radius,
        )
        right_lower = _filter_part_component_by_anchor(
            right_lower,
            right_lower_anchor,
            eps=part_group_eps,
            min_points=part_group_min_points,
            noise_attach_radius=part_group_noise_attach_radius,
            min_keep_ratio=part_group_min_keep_ratio,
            secondary_center_radius=part_group_secondary_center_radius,
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

    if box_post_enabled:
        boxes = _enforce_box_hierarchy(
            boxes,
            contact_margin=max(contact_margin, 0.0),
            forbidden_margin=max(forbidden_margin, 0.0),
        )

    part_centers: Dict[str, np.ndarray] = {}
    for name in DEFAULT_PART_ORDER:
        box = boxes[name]
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
        "allow_small_parts": allow_small_parts,
        "box_robust_quantile": float(box_robust_quantile),
        "torso_limb_exclusion_enabled": torso_limb_excl_enabled,
        "torso_limb_assign_to_torso": torso_limb_assign_to_torso,
        "torso_limb_extra_points": int(np.count_nonzero(torso_limb_extra_mask)),
        "head_arm_guard_enabled": head_arm_guard_enabled,
        "part_grouping_enabled": part_group_enabled,
    }
    next_state = {"part_centers": {k: [float(c) for c in v.tolist()] for k, v in part_centers.items()}}
    return boxes, info, part_points, next_state
