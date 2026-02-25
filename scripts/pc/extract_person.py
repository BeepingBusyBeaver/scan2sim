# scripts/pc/extract_person.py
from __future__ import annotations

import argparse
import copy
import csv
import json
import time
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Tuple, Any

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from scripts.common.pc_io import (
    apply_pcd_transform_if_needed,
    expand_inputs,
    read_point_cloud_any,
)
from scripts.common.io_paths import (
    DATA_REAL_HUMAN_DIR,
    DATA_REAL_RAW_EXAMPLE,
)


"""
Extract person points from raw Ouster LiDAR scans.

Pipeline (fast + robust):
  1) (optional) background subtraction using a static background scan
  2) ROI crop (distance + XYZ bounds)
  3) ground plane removal (RANSAC)
  4) outlier removal
  5) DBSCAN clustering -> pick best person cluster

Examples:

python -c "import scripts.pc.extract_person as m; print(m.__file__)"

python -m scripts.pc.extract_person \
  --in "data/real/raw/SN001_*.pcd" \
  --bg data/real/raw/SN001_bg.pcd \
  --out_dir data/real/human \
  --max_range 4.0 \
  --bg_icp --bg_icp_voxel 0.06 --bg_icp_max_corr 0.12 \
  --bg_thresh 0.05 \
  --post_bg_thresh 0.0 \
  --cluster_eps 0.012 --cluster_min_points 10 \
  --person_xy_max 1.1 --person_area_max 0.9 --person_planarity_max 0.8 \
  --vertical_plane_dist 0.01 --vertical_plane_min_inliers 300 \
  --post_rad_radius 0.08 --post_rad_min_points 8 \
  --debug_dir data/interim/debug_extract \
  --no_pcd_transform \
  --y_min -0.9 --y_max 0.6

"""


def to_numpy(pcd: o3d.geometry.PointCloud) -> np.ndarray:
    return np.asarray(pcd.points, dtype=np.float64)


def from_numpy(pts: np.ndarray) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd


def subtract_background(
    src_pts: np.ndarray, bg_pts: np.ndarray, thresh: float
) -> Tuple[np.ndarray, int]:
    if bg_pts.size == 0:
        return src_pts, 0
    tree = cKDTree(bg_pts)
    dists, _ = tree.query(src_pts, k=1, workers=-1)
    keep = dists > thresh
    removed = int((~keep).sum())
    return src_pts[keep], removed


def _pcd_from_numpy(pts: np.ndarray) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(pts, dtype=np.float64))
    return pcd


def _crop_pcd_by_range(
    pcd: o3d.geometry.PointCloud,
    r_min: Optional[float],
    r_max: Optional[float],
) -> o3d.geometry.PointCloud:
    if len(pcd.points) == 0 or (r_min is None and r_max is None):
        return pcd
    pts = np.asarray(pcd.points)
    r = np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2)
    m = np.ones(len(pts), dtype=bool)
    if r_min is not None:
        m &= r >= r_min
    if r_max is not None:
        m &= r <= r_max
    return _pcd_from_numpy(pts[m])


def align_bg_to_src_icp(
    bg_pcd: o3d.geometry.PointCloud,
    src_pcd: o3d.geometry.PointCloud,
    voxel: float,
    max_corr: float,
    point_to_plane: bool,
    r_min: Optional[float],
    r_max: Optional[float],
) -> Tuple[np.ndarray, dict]:
    """
    Align background -> source using ICP (downsampled for speed), then return aligned BG points for KD-tree.
    """
    if len(bg_pcd.points) == 0 or len(src_pcd.points) == 0:
        return np.empty((0, 3), dtype=np.float64), {"used": False, "reason": "empty"}

    bg_ds = bg_pcd.voxel_down_sample(voxel) if voxel > 0 else copy.deepcopy(bg_pcd)
    src_ds = src_pcd.voxel_down_sample(voxel) if voxel > 0 else copy.deepcopy(src_pcd)

    bg_ds = _crop_pcd_by_range(bg_ds, r_min, r_max)
    src_ds = _crop_pcd_by_range(src_ds, r_min, r_max)

    if len(bg_ds.points) < 30 or len(src_ds.points) < 30:
        return np.asarray(bg_ds.points), {"used": False, "reason": "too_few_points"}

    if point_to_plane:
        rad = max(voxel * 2.0, 0.10)
        bg_ds.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=rad, max_nn=30))
        src_ds.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=rad, max_nn=30))
        est = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    else:
        est = o3d.pipelines.registration.TransformationEstimationPointToPoint()

    reg = o3d.pipelines.registration.registration_icp(
        source=bg_ds,
        target=src_ds,
        max_correspondence_distance=max_corr,
        init=np.eye(4),
        estimation_method=est,
    )
    bg_aligned = copy.deepcopy(bg_ds)
    bg_aligned.transform(reg.transformation)
    info = {
        "used": True,
        "fitness": float(getattr(reg, "fitness", -1.0)),
        "inlier_rmse": float(getattr(reg, "inlier_rmse", -1.0)),
    }
    return np.asarray(bg_aligned.points), info


def pca_planarity(pts: np.ndarray) -> float:
    """Planarity in [0,1]. Larger => more planar (walls/boards)."""
    if pts.shape[0] < 10:
        return 0.0
    x = pts - pts.mean(axis=0, keepdims=True)
    cov = (x.T @ x) / max(len(x) - 1, 1)
    w = np.linalg.eigvalsh(cov)
    w = np.sort(w)[::-1] + 1e-12  # λ1>=λ2>=λ3
    plan = (w[1] - w[2]) / w[0]
    return float(np.clip(plan, 0.0, 1.0))


def apply_roi(
    pts: np.ndarray,
    x_min: Optional[float],
    x_max: Optional[float],
    y_min: Optional[float],
    y_max: Optional[float],
    z_min: Optional[float],
    z_max: Optional[float],
    r_min: Optional[float],
    r_max: Optional[float],
) -> Tuple[np.ndarray, int]:
    mask = np.ones(len(pts), dtype=bool)

    if x_min is not None:
        mask &= pts[:, 0] >= x_min
    if x_max is not None:
        mask &= pts[:, 0] <= x_max
    if y_min is not None:
        mask &= pts[:, 1] >= y_min
    if y_max is not None:
        mask &= pts[:, 1] <= y_max
    if z_min is not None:
        mask &= pts[:, 2] >= z_min
    if z_max is not None:
        mask &= pts[:, 2] <= z_max

    if r_min is not None or r_max is not None:
        r = np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2)
        if r_min is not None:
            mask &= r >= r_min
        if r_max is not None:
            mask &= r <= r_max

    removed = int((~mask).sum())
    return pts[mask], removed


def remove_ground_plane(
    pcd: o3d.geometry.PointCloud,
    dist: float,
    ransac_n: int,
    num_iter: int,
    min_inliers: int,
    fit_z_max: float = 0.25,        # 바닥 후보 점: z <= 25cm
    remove_z_max: float = 0.35,     # 바닥 제거 적용: z <= 35cm
    # remove_z_max: float = 0.45,     # 바닥 제거 적용: z <= 35cm
    normal_z_min: float = 0.85,     # |nz| >= 0.85 만 바닥으로 인정
) -> Tuple[o3d.geometry.PointCloud, int, dict]:
    if len(pcd.points) < max(ransac_n, 50):
        return pcd, 0, {"used": False, "reason": "too_few_points"}

    pts = np.asarray(pcd.points)
    low_idx = np.where(pts[:, 2] <= fit_z_max)[0]
    if low_idx.size < min_inliers:
        return pcd, 0, {"used": False, "reason": "not_enough_low_z"}

    low = pcd.select_by_index(low_idx)
    plane_model, inliers = low.segment_plane(
        distance_threshold=dist, ransac_n=ransac_n, num_iterations=num_iter
    )
    if len(inliers) < min_inliers:
        return pcd, 0, {"used": False, "reason": "min_inliers_fail"}

    a, b, c, d = plane_model
    n0 = np.array([a, b, c], dtype=np.float64)
    n_norm = float(np.linalg.norm(n0) + 1e-12)
    n = n0 / n_norm
    d_hat = float(d) / n_norm
    if abs(n[2]) < normal_z_min:
        return pcd, 0, {"used": False, "reason": "normal_not_ground", "normal": n.tolist()}

    dist_to_plane = np.abs(pts @ n + d_hat)
    keep = ~((dist_to_plane <= dist) & (pts[:, 2] <= remove_z_max))
    removed = int((~keep).sum())
    return from_numpy(pts[keep]), removed, {"used": True, "normal": n.tolist(), "inliers_low": int(len(inliers))}


def remove_outliers(
    pcd: o3d.geometry.PointCloud,
    method: str,
    nb_neighbors: int,
    std_ratio: float,
    radius: float,
    min_points: int,
) -> Tuple[o3d.geometry.PointCloud, int]:
    if method == "none" or len(pcd.points) == 0:
        return pcd, 0
    if method == "stat":
        filtered, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    elif method == "radius":
        filtered, ind = pcd.remove_radius_outlier(nb_points=min_points, radius=radius)
    else:
        raise ValueError(f"Unknown outlier method: {method}")
    removed = len(pcd.points) - len(ind)
    return filtered, removed


def remove_vertical_planes(
    pcd: o3d.geometry.PointCloud,
    dist: float,
    num_iter: int,
    min_inliers: int,
    z_abs_max: float,
    rounds: int,
) -> Tuple[o3d.geometry.PointCloud, int]:
    removed_total = 0
    cur = pcd
    for _ in range(max(0, rounds)):
        if len(cur.points) < 50:
            break
        model, inliers = cur.segment_plane(distance_threshold=dist, ransac_n=3, num_iterations=num_iter)
        if len(inliers) < min_inliers:
            break
        a, b, c, d = model
        n = np.array([a, b, c], dtype=np.float64)
        n /= (np.linalg.norm(n) + 1e-12)
        if abs(n[2]) <= z_abs_max:
            removed_total += len(inliers)
            cur = cur.select_by_index(inliers, invert=True)
        else:
            break
    return cur, removed_total


def cluster_dbscan(
    pcd: o3d.geometry.PointCloud,
    eps: float,
    min_points: int,
) -> np.ndarray:
    if len(pcd.points) == 0:
        return np.array([], dtype=np.int32)
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
    return labels


def select_person_cluster(
    pcd: o3d.geometry.PointCloud,
    labels: np.ndarray,
    min_points: int,
    height_min: float,
    height_max: float,
    xy_max: float,
    area_max: float,
    planarity_max: float,
    score_area_w: float,
    score_planarity_w: float,
) -> Tuple[o3d.geometry.PointCloud, dict]:
    if labels.size == 0 or labels.max(initial=-1) < 0:
        return pcd, {"selected": -1, "reason": "no_clusters"}

    best_id = -1
    best_score = -1.0
    best_info: dict[str, Any] = {}

    for cid in range(labels.max() + 1):
        idx = np.where(labels == cid)[0]
        if idx.size == 0:
            continue

        cluster = pcd.select_by_index(idx)
        pts = to_numpy(cluster)
        zmin, zmax = float(pts[:, 2].min()), float(pts[:, 2].max())
        height = zmax - zmin
        count = int(len(pts))

        ext_xy = np.ptp(pts[:, :2], axis=0)
        dx, dy = float(ext_xy[0]), float(ext_xy[1])
        area = dx * dy
        plan = pca_planarity(pts)

        valid = (
            (count >= min_points)
            and (height >= height_min)
            and (height <= height_max)
            and (max(dx, dy) <= xy_max)
            and (area <= area_max)
            and (plan <= planarity_max)
        )

        score = float(count) - float(score_area_w) * float(area) - float(score_planarity_w) * float(plan) * float(count)
        if not valid:
            score *= 0.1

        if score > best_score:
            best_score = score
            best_id = cid
            best_info = {
                "cluster_id": int(cid),
                "count": count,
                "height": float(height),
                "dx": float(dx),
                "dy": float(dy),
                "area": float(area),
                "planarity": float(plan),
                "valid": bool(valid),
                "score": float(score),
            }

    if best_id < 0:
        return pcd, {"selected": -1, "reason": "no_valid_cluster"}

    sel = pcd.select_by_index(np.where(labels == best_id)[0])
    best_info["selected"] = int(best_id)
    return sel, best_info


def merge_nearby_clusters(
    pcd: o3d.geometry.PointCloud,
    labels: np.ndarray,
    selected_id: int,
    merge_dist: float,
    min_points: int,
) -> Tuple[o3d.geometry.PointCloud, dict]:
    if selected_id < 0 or merge_dist <= 0:
        # selected_id<0이면 병합 의미 없음
        return pcd, {"merged_ids": []}

    pts_all = to_numpy(pcd)
    max_id = labels.max(initial=-1)
    if max_id < 0:
        return pcd, {"merged_ids": []}

    centroids: dict[int, np.ndarray] = {}
    counts: dict[int, int] = {}
    for cid in range(max_id + 1):
        idx = np.where(labels == cid)[0]
        if idx.size == 0:
            continue
        counts[cid] = int(idx.size)
        centroids[cid] = pts_all[idx].mean(axis=0)

    if selected_id not in centroids:
        return pcd, {"merged_ids": []}

    sel_c = centroids[selected_id]
    merged_ids: list[int] = []
    for cid, c in centroids.items():
        if counts.get(cid, 0) < min_points:
            continue
        if float(np.linalg.norm(c - sel_c)) <= merge_dist:
            merged_ids.append(cid)

    if not merged_ids:
        merged_ids = [selected_id]

    merge_idx = np.where(np.isin(labels, merged_ids))[0]
    merged = pcd.select_by_index(merge_idx)
    return merged, {"merged_ids": merged_ids}


def save_pcd(path: Path, pcd: o3d.geometry.PointCloud, ascii: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    ok = o3d.io.write_point_cloud(str(path), pcd, write_ascii=ascii)
    if not ok:
        raise RuntimeError(f"Failed to write: {path}")


@dataclass
class FrameStats:
    input_path: str
    output_path: str = ""
    ok: bool = False
    reason: str = ""

    raw_points: int = 0
    bg_removed: int = 0
    roi_removed: int = 0
    after_roi_points: int = 0

    after_voxel_points: int = 0
    ground_removed: int = 0
    after_ground_points: int = 0

    outliers_removed: int = 0
    after_outlier_points: int = 0

    clusters: int = 0
    noise_points: int = 0

    selected_cluster: int = -1
    selected_points: int = 0
    selected_valid: Optional[bool] = None
    selected_score: Optional[float] = None
    selected_reason: str = ""

    merged_ids: list[int] = None  # type: ignore[assignment]

    post_bg_removed: int = 0
    vertical_removed: int = 0
    post_radius_removed: int = 0
    final_points: int = 0

    icp_used: bool = False
    icp_fitness: Optional[float] = None
    icp_rmse: Optional[float] = None

    elapsed_s: float = 0.0


def _fmt_int(x: int) -> str:
    return f"{x:,}"


def _safe_mean(xs: list[float]) -> float:
    return float(np.mean(xs)) if xs else float("nan")


def _write_summary_json(path: Path, stats: list[FrameStats], meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": meta,
        "frames": [asdict(s) for s in stats],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_summary_csv(path: Path, stats: list[FrameStats]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(s) for s in stats]
    # list 필드는 csv-friendly 하게 문자열로
    for r in rows:
        if isinstance(r.get("merged_ids"), list):
            r["merged_ids"] = ",".join(map(str, r["merged_ids"]))
    fieldnames = list(rows[0].keys()) if rows else [f.name for f in FrameStats.__dataclass_fields__.values()]  # type: ignore[attr-defined]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def process_one(
    in_path: Path,
    root: Path,
    args: argparse.Namespace,
    out_dir: Path,
    output_name: Optional[str],
    debug_dir: Optional[Path],
    bg_pts_static: Optional[np.ndarray],
    bg_pcd_raw: Optional[o3d.geometry.PointCloud],
) -> FrameStats:
    t0 = time.time()
    st = FrameStats(input_path=str(in_path))
    try:
        def vlog(msg: str) -> None:
            if args.verbose:
                print(msg)

        pcd = read_point_cloud_any(in_path)
        pcd = apply_pcd_transform_if_needed(pcd, in_path, args.apply_pcd_transform)
        pts = to_numpy(pcd)
        st.raw_points = int(len(pts))
        vlog(f"[INFO] Processing: {in_path} | raw={st.raw_points}")

        # BG ICP (optional) -> per-frame bg_pts
        bg_pts_frame = bg_pts_static
        if bg_pcd_raw is not None and args.bg_icp:
            bg_pts_frame, icp_info = align_bg_to_src_icp(
                bg_pcd=bg_pcd_raw,
                src_pcd=pcd,
                voxel=args.bg_icp_voxel,
                max_corr=args.bg_icp_max_corr,
                point_to_plane=args.bg_icp_point_to_plane,
                r_min=args.min_range,
                r_max=args.max_range,
            )
            st.icp_used = bool(icp_info.get("used", False))
            if st.icp_used:
                st.icp_fitness = float(icp_info.get("fitness", -1.0))
                st.icp_rmse = float(icp_info.get("inlier_rmse", -1.0))
            vlog(f"[INFO] BG ICP: {icp_info}")

        # background subtraction
        if bg_pts_frame is not None and len(bg_pts_frame) > 0:
            pts, removed = subtract_background(pts, bg_pts_frame, args.bg_thresh)
            st.bg_removed = int(removed)

        # ROI crop
        pts, removed_roi = apply_roi(
            pts,
            args.x_min,
            args.x_max,
            args.y_min,
            args.y_max,
            args.z_min,
            args.z_max,
            args.min_range,
            args.max_range,
        )
        st.roi_removed = int(removed_roi)
        st.after_roi_points = int(len(pts))

        pcd = from_numpy(pts)
        if args.voxel and args.voxel > 0:
            pcd = pcd.voxel_down_sample(args.voxel)
        st.after_voxel_points = int(len(pcd.points))
        if debug_dir:
            save_pcd(debug_dir / f"{in_path.stem}_01_bg_roi.ply", pcd, ascii=args.ascii)

        # ground removal
        pcd, removed_ground, ground_info = remove_ground_plane(
            pcd,
            dist=args.ground_dist,
            ransac_n=args.ground_ransac_n,
            num_iter=args.ground_iter,
            min_inliers=args.ground_min_inliers,
        )
        st.ground_removed = int(removed_ground)
        st.after_ground_points = int(len(pcd.points))
        vlog(f"[INFO] Ground: removed={st.ground_removed}, info={ground_info}")
        if debug_dir:
            save_pcd(debug_dir / f"{in_path.stem}_02_no_ground.ply", pcd, ascii=args.ascii)

        # outlier removal
        pcd, removed_out = remove_outliers(
            pcd,
            method=args.outlier,
            nb_neighbors=args.stat_nb,
            std_ratio=args.stat_std,
            radius=args.rad_radius,
            min_points=args.rad_min_points,
        )
        st.outliers_removed = int(removed_out)
        st.after_outlier_points = int(len(pcd.points))
        if debug_dir:
            save_pcd(debug_dir / f"{in_path.stem}_03_no_outliers.ply", pcd, ascii=args.ascii)

        # clustering
        labels = cluster_dbscan(pcd, eps=args.cluster_eps, min_points=args.cluster_min_points)
        if labels.size:
            st.noise_points = int((labels < 0).sum())
            st.clusters = int(labels.max(initial=-1) + 1)
        else:
            st.noise_points = 0
            st.clusters = 0

        # select person
        if labels.size == 0 or labels.max(initial=-1) < 0:
            person_pcd = pcd
            info = {"selected": -1, "reason": "no_clusters"}
        else:
            person_pcd, info = select_person_cluster(
                pcd,
                labels,
                min_points=args.person_points_min,
                height_min=args.person_height_min,
                height_max=args.person_height_max,
                xy_max=args.person_xy_max,
                area_max=args.person_area_max,
                planarity_max=args.person_planarity_max,
                score_area_w=args.score_area_w,
                score_planarity_w=args.score_planarity_w,
            )

        st.selected_cluster = int(info.get("selected", -1))
        st.selected_points = int(len(person_pcd.points))
        st.selected_valid = info.get("valid", None)
        st.selected_score = info.get("score", None)
        st.selected_reason = str(info.get("reason", ""))

        # merge nearby clusters (optional) - only if selected exists
        st.merged_ids = []
        if args.merge_cluster_dist and args.merge_cluster_dist > 0 and st.selected_cluster >= 0 and labels.size:
            merged_pcd, merge_info = merge_nearby_clusters(
                pcd,
                labels,
                selected_id=st.selected_cluster,
                merge_dist=float(args.merge_cluster_dist),
                min_points=int(args.merge_cluster_min_points),
            )
            mids = merge_info.get("merged_ids", []) or []
            st.merged_ids = list(map(int, mids))
            if st.merged_ids:
                person_pcd = merged_pcd
                st.selected_points = int(len(person_pcd.points))

        if args.save_all_clusters and labels.size and labels.max(initial=-1) >= 0:
            for cid in range(labels.max() + 1):
                idx = np.where(labels == cid)[0]
                if idx.size == 0:
                    continue
                cluster = pcd.select_by_index(idx)
                save_pcd(out_dir / f"{in_path.stem}_cluster_{cid}.ply", cluster, ascii=args.ascii)

        if debug_dir:
            save_pcd(debug_dir / f"{in_path.stem}_04_selected_raw.ply", person_pcd, ascii=args.ascii)

        # ---- post-processing for precision ----
        if len(person_pcd.points) > 0:
            if args.post_bg_thresh and args.post_bg_thresh > 0 and bg_pts_frame is not None and len(bg_pts_frame) > 0:
                ppts = to_numpy(person_pcd)
                ppts2, removed_post_bg = subtract_background(ppts, bg_pts_frame, args.post_bg_thresh)
                person_pcd = from_numpy(ppts2)
                st.post_bg_removed = int(removed_post_bg)

            if args.rm_vertical_planes and len(person_pcd.points) > 0:
                person_pcd, removed_v = remove_vertical_planes(
                    person_pcd,
                    dist=args.vertical_plane_dist,
                    num_iter=args.vertical_plane_iter,
                    min_inliers=args.vertical_plane_min_inliers,
                    z_abs_max=args.vertical_plane_z_abs_max,
                    rounds=args.vertical_plane_rounds,
                )
                st.vertical_removed = int(removed_v)

            if args.post_radius and len(person_pcd.points) > 0:
                filtered, ind = person_pcd.remove_radius_outlier(
                    nb_points=args.post_rad_min_points, radius=args.post_rad_radius
                )
                removed_post = len(person_pcd.points) - len(ind)
                person_pcd = filtered
                st.post_radius_removed = int(removed_post)

        st.final_points = int(len(person_pcd.points))
        if debug_dir:
            save_pcd(debug_dir / f"{in_path.stem}_05_post_processed.ply", person_pcd, ascii=args.ascii)

        # output
        if args.out is not None:
            out_path = (root / args.out).resolve()
        elif output_name:
            out_path = out_dir / output_name
        else:
            out_path = out_dir / f"{in_path.stem}_person.ply"
        save_pcd(out_path, person_pcd, ascii=args.ascii)
        st.output_path = str(out_path)

        st.ok = True
        st.reason = "ok"
        vlog(f"[OK] Wrote: {out_path} | final={st.final_points}")

    except Exception as e:
        st.ok = False
        st.reason = f"{type(e).__name__}: {e}"
        if args.verbose:
            print(f"[ERROR] {in_path}: {st.reason}")
            traceback.print_exc()
    finally:
        st.elapsed_s = float(time.time() - t0)
    return st


def print_run_summary(
    stats: list[FrameStats],
    out_dir: Path,
    debug_dir: Optional[Path],
    args: argparse.Namespace,
) -> None:
    n = len(stats)
    ok = sum(1 for s in stats if s.ok)
    fail = n - ok

    # Totals (all frames)
    total_raw = sum(s.raw_points for s in stats)
    total_final = sum(s.final_points for s in stats if s.ok)

    total_bg_rm = sum(s.bg_removed for s in stats)
    total_roi_rm = sum(s.roi_removed for s in stats)
    total_ground_rm = sum(s.ground_removed for s in stats)
    total_out_rm = sum(s.outliers_removed for s in stats)
    total_post_bg_rm = sum(s.post_bg_removed for s in stats)
    total_v_rm = sum(s.vertical_removed for s in stats)
    total_post_rad_rm = sum(s.post_radius_removed for s in stats)

    # Clustering stats
    frames_with_clusters = sum(1 for s in stats if (s.clusters or 0) > 0)
    frames_no_clusters = sum(1 for s in stats if (s.clusters or 0) == 0)
    avg_clusters = _safe_mean([float(s.clusters) for s in stats if s.ok])
    avg_noise = _safe_mean([float(s.noise_points) for s in stats if s.ok])

    # ICP stats
    icp_used = [s for s in stats if s.icp_used]
    icp_fitness = [float(s.icp_fitness) for s in icp_used if s.icp_fitness is not None]
    icp_rmse = [float(s.icp_rmse) for s in icp_used if s.icp_rmse is not None]

    # Reasons for failures (or selection reasons)
    fail_reasons: dict[str, int] = {}
    for s in stats:
        if not s.ok:
            fail_reasons[s.reason] = fail_reasons.get(s.reason, 0) + 1
    sel_reasons: dict[str, int] = {}
    for s in stats:
        if s.ok and s.selected_cluster < 0:
            r = s.selected_reason or "no_selection"
            sel_reasons[r] = sel_reasons.get(r, 0) + 1

    total_time = sum(s.elapsed_s for s in stats)
    avg_time = (total_time / n) if n else 0.0

    # ---- single consolidated print ----
    print("\n========== Extract Person: Run Summary ==========")
    print(f"Inputs: {_fmt_int(n)} files | Success: {_fmt_int(ok)} | Fail: {_fmt_int(fail)}")
    print(f"Output dir: {out_dir}")
    if debug_dir:
        print(f"Debug dir:  {debug_dir}")
    print(f"Time: total {total_time:.2f}s | avg/file {avg_time:.3f}s")

    print("\n[Points]")
    print(f"  Raw total:          {_fmt_int(total_raw)}")
    if ok:
        print(f"  Final total(ok):    {_fmt_int(total_final)} | avg(ok): {_fmt_int(int(total_final / ok))}")

    print("\n[Removed totals]")
    print(f"  BG removed:         {_fmt_int(total_bg_rm)}")
    print(f"  ROI removed:        {_fmt_int(total_roi_rm)}")
    print(f"  Ground removed:     {_fmt_int(total_ground_rm)}")
    print(f"  Outliers removed:   {_fmt_int(total_out_rm)}")
    print(f"  Post-BG removed:    {_fmt_int(total_post_bg_rm)}")
    print(f"  Vertical removed:   {_fmt_int(total_v_rm)}")
    print(f"  Post-radius removed:{_fmt_int(total_post_rad_rm)}")

    print("\n[Clustering]")
    print(f"  Frames w/ clusters: {_fmt_int(frames_with_clusters)} | no clusters: {_fmt_int(frames_no_clusters)}")
    if ok:
        print(f"  Avg clusters(ok):   {avg_clusters:.2f} | avg noise(ok): {avg_noise:.2f}")

    if icp_used:
        print("\n[BG ICP]")
        print(f"  ICP used frames:    {_fmt_int(len(icp_used))}")
        if icp_fitness:
            print(f"  Mean fitness:       {_safe_mean(icp_fitness):.4f}")
        if icp_rmse:
            print(f"  Mean inlier_rmse:   {_safe_mean(icp_rmse):.4f}")

    if sel_reasons:
        print("\n[No selection reasons (ok frames)]")
        for k, v in sorted(sel_reasons.items(), key=lambda kv: kv[1], reverse=True)[:10]:
            print(f"  - {k}: {v}")

    if fail_reasons:
        print("\n[Failures]")
        for k, v in sorted(fail_reasons.items(), key=lambda kv: kv[1], reverse=True)[:10]:
            print(f"  - {k}: {v}")

    # Optional: show a few outputs for convenience (still single summary block)
    outs = [s.output_path for s in stats if s.ok and s.output_path]
    if outs:
        show = outs[: min(5, len(outs))]
        print("\n[Sample outputs]")
        for p in show:
            print(f"  - {p}")
        if len(outs) > len(show):
            print(f"  ... +{len(outs) - len(show)} more")
    print("===============================================\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in",
        dest="inputs",
        type=str,
        nargs="+",
        default=[DATA_REAL_RAW_EXAMPLE],
        help="Input .pcd/.ply path(s), directory, or glob pattern(s).",
    )
    ap.add_argument("--bg", type=str, default=None, help="Background scan (.pcd/.ply).")
    ap.add_argument("--out", type=str, default=None, help="Single output .ply path.")
    ap.add_argument("--out_dir", type=str, default=DATA_REAL_HUMAN_DIR, help="Output directory for batch mode.")
    ap.add_argument(
        "--output_mode",
        type=str,
        default="human_index",
        choices=["human_index", "input_stem"],
        help="Batch output naming mode: sequential human_### or <input_stem>_person.",
    )
    ap.add_argument("--output_prefix", type=str, default="human_", help="Filename prefix for output_mode=human_index.")
    ap.add_argument("--output_start", type=int, default=1, help="Start index for output_mode=human_index.")
    ap.add_argument("--output_digits", type=int, default=3, help="Zero-padding digits for output_mode=human_index.")
    ap.add_argument("--ascii", action="store_true", help="Write output in ASCII.")

    ap.add_argument(
        "--no_pcd_transform",
        action="store_false",
        dest="apply_pcd_transform",
        help="Disable mm->m + axis conversion for .pcd (default: apply).",
    )

    # NEW: logging/summaries
    ap.add_argument("--verbose", action="store_true", help="Print per-file logs (debug).")
    ap.add_argument("--summary_json", type=str, default=None, help="Write per-file stats + meta to JSON.")
    ap.add_argument("--summary_csv", type=str, default=None, help="Write per-file stats to CSV.")

    # background subtraction
    ap.add_argument("--bg_voxel", type=float, default=0.03, help="Background downsample voxel (m).")
    ap.add_argument("--bg_thresh", type=float, default=0.05, help="Background removal distance (m).")
    ap.add_argument("--post_bg_thresh", type=float, default=0.0, help="After selecting person, remove points close to BG (m). 0=off.")

    # optional BG ICP alignment
    ap.add_argument("--bg_icp", action="store_true", help="Align BG->frame using ICP before BG subtraction.")
    ap.add_argument("--bg_icp_voxel", type=float, default=0.06, help="ICP downsample voxel (m).")
    ap.add_argument("--bg_icp_max_corr", type=float, default=0.12, help="ICP max correspondence distance (m).")
    ap.add_argument("--bg_icp_point_to_plane", action="store_true", help="Use point-to-plane ICP (requires normals).")

    # ROI
    ap.add_argument("--x_min", type=float, default=None)
    ap.add_argument("--x_max", type=float, default=None)
    ap.add_argument("--y_min", type=float, default=-0.9)
    ap.add_argument("--y_max", type=float, default=0.6)
    ap.add_argument("--z_min", type=float, default=None)
    ap.add_argument("--z_max", type=float, default=None)
    ap.add_argument("--min_range", type=float, default=None)
    ap.add_argument("--max_range", type=float, default=4.0)
    ap.add_argument("--voxel", type=float, default=0.0, help="Optional voxel downsample before ground/outlier/cluster.")

    # ground
    ap.add_argument("--ground_dist", type=float, default=0.02, help="RANSAC plane distance threshold (m).")
    ap.add_argument("--ground_ransac_n", type=int, default=3)
    ap.add_argument("--ground_iter", type=int, default=200)
    ap.add_argument("--ground_min_inliers", type=int, default=200)

    # outliers
    ap.add_argument("--outlier", type=str, default="stat", choices=["none", "stat", "radius"])
    ap.add_argument("--stat_nb", type=int, default=20)
    ap.add_argument("--stat_std", type=float, default=2.0)
    ap.add_argument("--rad_radius", type=float, default=0.05)
    ap.add_argument("--rad_min_points", type=int, default=10)

    # clustering
    ap.add_argument("--cluster_eps", type=float, default=0.012, help="DBSCAN eps (m).")
    ap.add_argument("--cluster_min_points", type=int, default=10)
    ap.add_argument("--person_height_min", type=float, default=0.9)
    ap.add_argument("--person_height_max", type=float, default=2.3)
    ap.add_argument("--person_points_min", type=int, default=200)
    ap.add_argument("--person_xy_max", type=float, default=1.10)
    ap.add_argument("--person_area_max", type=float, default=0.90)
    ap.add_argument("--person_planarity_max", type=float, default=0.80)
    ap.add_argument("--score_area_w", type=float, default=300.0)
    ap.add_argument("--score_planarity_w", type=float, default=0.20)
    ap.add_argument("--merge_cluster_dist", type=float, default=0.0, help="Merge nearby clusters by centroid distance (m).")
    ap.add_argument("--merge_cluster_min_points", type=int, default=20, help="Min points for merge candidates.")
    ap.add_argument("--save_all_clusters", action="store_true", help="Save every cluster (debug).")

    # post-processing
    ap.add_argument("--rm_vertical_planes", action="store_true", help="Remove near-vertical planes from selected cluster.")
    ap.add_argument("--vertical_plane_dist", type=float, default=0.01)
    ap.add_argument("--vertical_plane_iter", type=int, default=200)
    ap.add_argument("--vertical_plane_min_inliers", type=int, default=300)
    ap.add_argument("--vertical_plane_z_abs_max", type=float, default=0.30)
    ap.add_argument("--vertical_plane_rounds", type=int, default=2)

    ap.add_argument("--post_radius", action="store_true", help="Apply radius outlier removal AFTER selecting person cluster.")
    ap.add_argument("--post_rad_radius", type=float, default=0.08)
    ap.add_argument("--post_rad_min_points", type=int, default=8)

    ap.add_argument("--debug_dir", type=str, default=None, help="Save intermediate stages here.")

    args = ap.parse_args()
    root = Path(__file__).resolve().parents[2]

    in_paths = expand_inputs(root, args.inputs, exts=[".pcd", ".ply"])
    if not in_paths:
        raise ValueError("No input files matched.")
    if args.out is not None and len(in_paths) != 1:
        raise ValueError("--out can be used only with a single input. Use --out_dir for batch.")

    out_dir = (root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    debug_dir = None
    if args.debug_dir:
        debug_dir = (root / args.debug_dir).resolve()
        debug_dir.mkdir(parents=True, exist_ok=True)

    # background load once
    bg_pts_static: Optional[np.ndarray] = None
    bg_pcd_raw: Optional[o3d.geometry.PointCloud] = None
    if args.bg:
        bg_path = Path(args.bg)
        if not bg_path.is_absolute():
            bg_path = root / bg_path
        if not bg_path.exists():
            raise ValueError(f"Background file not found: {bg_path}")
        bg_pcd_raw = read_point_cloud_any(bg_path)
        bg_pcd_raw = apply_pcd_transform_if_needed(bg_pcd_raw, bg_path, args.apply_pcd_transform)
        bg_pcd = bg_pcd_raw.voxel_down_sample(args.bg_voxel)
        bg_pts_static = to_numpy(bg_pcd)

    # ---- run (no per-file prints by default) ----
    if args.verbose:
        print(f"[INFO] Matched {len(in_paths)} file(s). Processing...")

    run_t0 = time.time()
    stats: list[FrameStats] = []
    for file_index, in_path in enumerate(in_paths):
        output_name: Optional[str] = None
        if args.out is None:
            if args.output_mode == "human_index":
                frame_no = int(args.output_start) + int(file_index)
                output_name = f"{args.output_prefix}{frame_no:0{int(args.output_digits)}d}.ply"
            else:
                output_name = f"{in_path.stem}_person.ply"
        stats.append(
            process_one(
                in_path=in_path,
                root=root,
                args=args,
                out_dir=out_dir,
                output_name=output_name,
                debug_dir=debug_dir,
                bg_pts_static=bg_pts_static,
                bg_pcd_raw=bg_pcd_raw,
            )
        )

    # ---- single consolidated output ----
    print_run_summary(stats, out_dir=out_dir, debug_dir=debug_dir, args=args)

    # optional artifacts
    meta = {
        "n_inputs": len(in_paths),
        "out_dir": str(out_dir),
        "debug_dir": str(debug_dir) if debug_dir else None,
        "elapsed_s": float(time.time() - run_t0),
        "bg_used": bool(args.bg),
        "bg_icp": bool(args.bg_icp),
    }
    if args.summary_json:
        _write_summary_json((root / args.summary_json).resolve(), stats, meta)
    if args.summary_csv:
        _write_summary_csv((root / args.summary_csv).resolve(), stats)


if __name__ == "__main__":
    main()
