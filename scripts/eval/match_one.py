# scripts/eval/match_one.py
from __future__ import annotations

import argparse
import json
import math
import sys
import re
import glob
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import open3d as o3d
import pandas as pd
import yaml
from rich.console import Console
from rich.table import Table

from scripts.common.io_paths import (
    CONFIG_AXIS_STEPS_YAML,
    DATA_REAL_HUMAN_EXAMPLE,
    DATASET_POSES_324_JSONL,
    DATA_VIRTUAL_POSE324_GLOB,
    OUTPUT_MATCH_ONE_DIR,
    resolve_repo_path,
)


"""
python -m scripts.eval.match_one \
  --real "data/real/human/human_*.ply" \
  --virtual-glob "data/virtual/pose324_10_deg45/virtual_*.ply" \
  --poses-jsonl "dataset/324_LowerBody_10set.jsonl" \
  --out-root "data/match_one" \
  --voxel 0.05 \
  --icp-dist-mul 1.5 \
  --ransac-dist-mul 2.0 \
  --topk 3
"""

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
console = Console()
_VIRTUAL_TOKEN_RE = re.compile(r"^(?P<idx>\d+)(?:-(?P<occ>\d+))?$")


def repo_root() -> Path:
    """
    Resolve repository root from scripts/eval/match_one.py:
    .../scan2sim/scripts/eval/match_one.py -> parents[2] == .../scan2sim
    """
    return Path(__file__).resolve().parents[2]


@dataclass
class RegMetrics:
    virtual_path: str
    virtual_idx: Optional[int]
    virtual_occ: Optional[int]

    # RANSAC (global)
    ransac_fitness: float
    ransac_inlier_rmse: float
    ransac_correspondences: int

    # ICP (refine)
    icp_fitness: float
    icp_inlier_rmse: float
    icp_correspondences: int

    # Final pick score (you can change)
    score: float

    # timing (sec)
    t_preprocess: float
    t_ransac: float
    t_icp: float


def parse_virtual_id(path: Path) -> Tuple[Optional[int], Optional[int]]:
    """
    Parse virtual id from filenames like:
      - virtual_000.ply   -> (0, None)
      - virtual_34-6.ply  -> (34, 6)   # idx=34, occ=6
    또한 'xxx_yyy_34-6.ply'처럼 underscore가 더 있어도 마지막 토큰 기준으로 파싱.
    """
    stem = path.stem  # suffix 제거된 파일명 :contentReference[oaicite:2]{index=2}
    parts = stem.split("_")
    if len(parts) < 2:
        return None, None

    m = _VIRTUAL_TOKEN_RE.fullmatch(parts[-1])
    if not m:
        return None, None

    idx = int(m.group("idx"))
    occ = int(m.group("occ")) if m.group("occ") is not None else None
    return idx, occ

def parse_virtual_idx(path: Path) -> Optional[int]:
    # 기존 함수 시그니처 유지 (외부에서 쓰고 있을 수 있으니)
    idx, _ = parse_virtual_id(path)
    return idx


def _maybe_int(x: object) -> Optional[int]:
    if x is None:
        return None
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, float) and math.isfinite(x):
        return int(x)
    return None


def load_axis_steps_order(path: Path) -> List[str]:
    """
    Load axis order from configs/axis_steps.yaml (e.g., ["Rx", "Rz", "Lx", "Lz", "Rk", "Lk"]).
    """
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    order = cfg.get("order")
    if not isinstance(order, list) or not order:
        raise ValueError(f"Invalid axis_steps.yaml (missing 'order' list): {path}")
    return [str(k) for k in order]


def load_pose_steps_jsonl(path: Path) -> Dict[int, Dict[str, int]]:
    """
    Read pose jsonl and return {idx: {axis: step_int}}.
    Expected per-line JSON with fields like:
      {"idx": 0, "step": {"Rx": -1, "Rz": 0, ...}, ...}
    """
    out: Dict[int, Dict[str, int]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {e}") from e

            idx = _maybe_int(obj.get("idx"))
            step = obj.get("step")
            if idx is None or not isinstance(step, dict):
                continue

            step_ints: Dict[str, int] = {}
            for k, v in step.items():
                v_int = _maybe_int(v)
                if v_int is None:
                    continue
                step_ints[str(k)] = int(v_int)
            out[int(idx)] = step_ints

    return out


def format_step_label(step: Dict[str, int], axis_order: List[str]) -> str:
    parts: List[str] = []
    for axis in axis_order:
        v = step.get(axis)
        parts.append(f"{axis}{'?' if v is None else int(v)}")
    return " ".join(parts)


def load_point_cloud(path: Path) -> o3d.geometry.PointCloud:
    # 읽을 때 바로 NaN/Inf 제거 (가장 깔끔)
    pcd = o3d.io.read_point_cloud(
        str(path),
        remove_nan_points=True,
        remove_infinite_points=True,
    )
    if pcd.is_empty():
        raise ValueError(f"Empty point cloud: {path}")

    # 혹시 남아있을 수 있는 비정상값 방어 (언팩 금지!)
    pcd = pcd.remove_non_finite_points(remove_nan=True, remove_infinite=True)
    if pcd.is_empty():
        raise ValueError(f"Point cloud became empty after filtering non-finite points: {path}")
    return pcd


def preprocess_for_fpfh(
    pcd: o3d.geometry.PointCloud,
    voxel_size: float,
    max_nn_normal: int = 30,
    max_nn_feature: int = 100,
) -> Tuple[o3d.geometry.PointCloud, o3d.pipelines.registration.Feature]:
    """
    Open3D-style preprocessing:
      - voxel downsample
      - estimate normals with radius = 2*voxel
      - compute FPFH with radius = 5*voxel
    """
    pcd_down = pcd.voxel_down_sample(voxel_size)

    if pcd_down.is_empty():
        raise ValueError("Downsampled point cloud is empty. Decrease voxel_size or check input.")

    radius_normal = voxel_size * 2.0
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=max_nn_normal)
    )

    radius_feature = voxel_size * 5.0
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=max_nn_feature),
    )
    return pcd_down, fpfh


def execute_ransac_global(
    source_down: o3d.geometry.PointCloud,
    target_down: o3d.geometry.PointCloud,
    source_fpfh: o3d.pipelines.registration.Feature,
    target_fpfh: o3d.pipelines.registration.Feature,
    voxel_size: float,
    distance_multiplier: float = 1.5,
    ransac_n: int = 4,
    max_iteration: int = 100000,
    confidence: float = 0.999,
) -> o3d.pipelines.registration.RegistrationResult:
    """
    Global registration via RANSAC-based feature matching.
    """
    distance_threshold = voxel_size * distance_multiplier

    checker_edge = o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9)
    checker_dist = o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)

    # Point-to-point estimation for global
    estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint(False)

    criteria = o3d.pipelines.registration.RANSACConvergenceCriteria(max_iteration, confidence)

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
        True,  # mutual filter
        distance_threshold,
        estimation,
        ransac_n,
        [checker_edge, checker_dist],
        criteria,
    )
    return result


def execute_icp_refine(
    source_down: o3d.geometry.PointCloud,
    target_down: o3d.geometry.PointCloud,
    init_trans: np.ndarray,
    voxel_size: float,
    distance_multiplier: float = 0.4,
    point_to_plane: bool = True,
    max_iteration: int = 50,
) -> o3d.pipelines.registration.RegistrationResult:
    """
    Local registration refinement via ICP (point-to-plane by default).
    """
    distance_threshold = voxel_size * distance_multiplier

    if point_to_plane:
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    else:
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint()

    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration)

    result = o3d.pipelines.registration.registration_icp(
        source_down,
        target_down,
        distance_threshold,
        init_trans,
        estimation,
        criteria,
    )
    return result


def draw_alignment(source: o3d.geometry.PointCloud, target: o3d.geometry.PointCloud, trans: np.ndarray) -> None:
    """
    Visualize alignment in Open3D viewer.
    """
    src_vis = o3d.geometry.PointCloud(source)
    tgt_vis = o3d.geometry.PointCloud(target)
    src_vis.paint_uniform_color([1.0, 0.706, 0.0])  # yellow
    tgt_vis.paint_uniform_color([0.0, 0.651, 0.929])  # blue
    src_vis.transform(trans)
    o3d.visualization.draw_geometries([src_vis, tgt_vis])


def compute_score(icp_fitness: float, icp_rmse: float) -> float:
    """
    Simple ranking score:
      - prefer high fitness
      - prefer low rmse
    """
    return float(icp_fitness) - float(icp_rmse)


def match_one_real_to_virtual_library(
    real_path: Path,
    virtual_paths: List[Path],
    voxel_size: float,
    ransac_distance_multiplier: float,
    icp_distance_multiplier: float,
    ransac_n: int,
    ransac_max_iteration: int,
    ransac_confidence: float,
    icp_max_iteration: int,
    icp_point_to_plane: bool,
    skip_icp_if_ransac_fails: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Returns:
      - DataFrame metrics for all virtual candidates
      - dict: virtual_path -> 4x4 transform (final ICP if available else RANSAC)
    """
    import time

    real = load_point_cloud(real_path)

    t0 = time.perf_counter()
    real_down, real_fpfh = preprocess_for_fpfh(real, voxel_size=voxel_size)
    t_real_pre = time.perf_counter() - t0

    metrics: List[RegMetrics] = []
    transforms: Dict[str, np.ndarray] = {}

    for vp in virtual_paths:
        vp = vp.resolve()
        v_idx, v_occ = parse_virtual_id(vp)

        try:
            v_full = load_point_cloud(vp)
        except Exception as e:
            console.print(f"[red]Failed to load {vp}: {e}[/red]")
            continue

        t1 = time.perf_counter()
        v_down, v_fpfh = preprocess_for_fpfh(v_full, voxel_size=voxel_size)
        t_pre = time.perf_counter() - t1

        # RANSAC
        t2 = time.perf_counter()
        ransac = execute_ransac_global(
            source_down=v_down,
            target_down=real_down,
            source_fpfh=v_fpfh,
            target_fpfh=real_fpfh,
            voxel_size=voxel_size,
            distance_multiplier=ransac_distance_multiplier,
            ransac_n=ransac_n,
            max_iteration=ransac_max_iteration,
            confidence=ransac_confidence,
        )
        t_ransac = time.perf_counter() - t2

        # ICP refine
        icp_fitness = 0.0
        icp_rmse = float("inf")
        icp_corr = 0
        final_T = ransac.transformation

        do_icp = True
        if skip_icp_if_ransac_fails and (ransac.fitness is None or float(ransac.fitness) <= 0.0):
            do_icp = False

        t_icp = 0.0
        if do_icp:
            t3 = time.perf_counter()
            icp = execute_icp_refine(
                source_down=v_down,
                target_down=real_down,
                init_trans=ransac.transformation,
                voxel_size=voxel_size,
                distance_multiplier=icp_distance_multiplier,
                point_to_plane=icp_point_to_plane,
                max_iteration=icp_max_iteration,
            )
            t_icp = time.perf_counter() - t3
            icp_fitness = float(icp.fitness)
            icp_rmse = float(icp.inlier_rmse)
            icp_corr = int(len(icp.correspondence_set))
            final_T = icp.transformation

        score = compute_score(icp_fitness, icp_rmse) if math.isfinite(icp_rmse) else float(ransac.fitness or 0.0)

        transforms[str(vp)] = np.asarray(final_T, dtype=np.float64)

        metrics.append(
            RegMetrics(
                virtual_path=str(vp),
                virtual_idx=v_idx,
                virtual_occ=v_occ,
                ransac_fitness=float(ransac.fitness),
                ransac_inlier_rmse=float(ransac.inlier_rmse),
                ransac_correspondences=int(len(ransac.correspondence_set)),
                icp_fitness=icp_fitness,
                icp_inlier_rmse=icp_rmse if math.isfinite(icp_rmse) else float("inf"),
                icp_correspondences=icp_corr,
                score=float(score),
                t_preprocess=float(t_pre),
                t_ransac=float(t_ransac),
                t_icp=float(t_icp),
            )
        )

    df = pd.DataFrame([asdict(m) for m in metrics])
    # Include real preprocess time for reference (not per virtual)
    df.attrs["real_preprocess_sec"] = float(t_real_pre)
    return df, transforms


def build_virtual_list(repo: Path, virtual_glob: str) -> List[Path]:
    """
    Resolve virtual point-cloud candidates from:
      - a glob pattern (relative to repo or absolute)
      - a directory (lists *.ply inside)
      - a single file path
    """
    has_glob = any(ch in virtual_glob for ch in ["*", "?", "["])

    base = Path(virtual_glob)
    base_resolved = base if base.is_absolute() else (repo / base)

    if has_glob:
        return sorted([Path(p) for p in glob.glob(str(base_resolved)) if Path(p).is_file()])

    if base_resolved.is_dir():
        return sorted([p for p in base_resolved.glob("*.ply") if p.is_file()])
    if base_resolved.is_file():
        return [base_resolved]
    return []


def build_real_list(repo: Path, real_args: List[str]) -> List[Path]:
    """
    Resolve real point-cloud inputs from one or more args:
      - glob patterns (supports *, ?, [], **)
      - directory (lists *.ply inside)
      - single file
    """
    out: List[Path] = []

    for real_arg in real_args:
        has_glob = any(ch in real_arg for ch in ["*", "?", "["])

        p = Path(real_arg)
        p_resolved = p if p.is_absolute() else (repo / p)

        matches: List[Path] = []
        if has_glob:
            matches = sorted(Path(x) for x in glob.glob(str(p_resolved), recursive=True))
        elif p_resolved.is_dir():
            matches = sorted([x for x in p_resolved.glob("*.ply") if x.is_file()])
        elif p_resolved.is_file():
            matches = [p_resolved]

        for m in matches:
            if m.is_file():
                out.append(m.resolve())

    # deduplicate while preserving order
    uniq: List[Path] = []
    seen: set[str] = set()
    for p in out:
        s = str(p)
        if s in seen:
            continue
        seen.add(s)
        uniq.append(p)
    return uniq


def ensure_outdir(repo: Path, out_root: str, real_path: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_base = resolve_repo_path(repo, out_root)
    outdir = out_base / real_path.stem / ts
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def save_results(
    outdir: Path,
    real_path: Path,
    df: pd.DataFrame,
    transforms: Dict[str, np.ndarray],
    topk: int,
    pose_steps_by_idx: Optional[Dict[int, Dict[str, int]]] = None,
    axis_order: Optional[List[str]] = None,
) -> None:
    # Sort: best first
    df_sorted = df.sort_values(
        by=["score", "icp_fitness", "ransac_fitness", "icp_inlier_rmse"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)

    csv_path = outdir / "results.csv"
    df_sorted.to_csv(csv_path, index=False)

    # Save all transforms
    tf_path = outdir / "transforms.json"
    tf_out = {k: v.tolist() for k, v in transforms.items()}
    with tf_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "real_path": str(real_path.resolve()),
                "created_at": datetime.now().isoformat(),
                "transforms": tf_out,
            },
            f,
            indent=2,
        )

    # Save best top-k summary
    topk_path = outdir / "topk.json"
    top_df = df_sorted.head(topk)
    top_records = top_df.to_dict(orient="records")
    if pose_steps_by_idx is not None and axis_order is not None:
        for rec in top_records:
            idx = _maybe_int(rec.get("virtual_idx"))
            if idx is None:
                rec["virtual_step"] = None
                rec["virtual_step_str"] = None
                continue
            step = pose_steps_by_idx.get(int(idx))
            rec["virtual_step"] = step
            rec["virtual_step_str"] = format_step_label(step, axis_order) if step is not None else None
    with topk_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "real_path": str(real_path.resolve()),
                "created_at": datetime.now().isoformat(),
                "topk": top_records,
            },
            f,
            indent=2,
        )

    # console.print(f"[green]Saved:[/green] {csv_path}")
    # console.print(f"[green]Saved:[/green] {tf_path}")
    # console.print(f"[green]Saved:[/green] {topk_path}")


def print_topk(
    df: pd.DataFrame,
    topk: int,
    pose_steps_by_idx: Optional[Dict[int, Dict[str, int]]] = None,
    axis_order: Optional[List[str]] = None,
) -> None:
    if df.empty:
        console.print("[red]No results.[/red]")
        return

    df_sorted = df.sort_values(
        by=["score", "icp_fitness", "ransac_fitness", "icp_inlier_rmse"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)

    table = Table(title=f"Top-{topk} matches")
    table.add_column("rank", justify="right")
    table.add_column("virtual_id", justify="right")
    table.add_column("step", overflow="fold")
    table.add_column("icp_fit", justify="right")
    table.add_column("icp_rmse", justify="right")
    table.add_column("ransac_fit", justify="right")
    table.add_column("path", overflow="fold")

    for i in range(min(topk, len(df_sorted))):
        r = df_sorted.iloc[i]
        idx = r["virtual_idx"]
        occ = r["virtual_occ"] if "virtual_occ" in r else np.nan
        
        if pd.notna(idx):
            vid = f"{int(idx)}-{int(occ)}" if pd.notna(occ) else str(int(idx))
        else:
            vid = "-"    

        step_str = "-"
        if pose_steps_by_idx is not None and axis_order is not None:
            idx_int = _maybe_int(idx)
            if idx_int is not None:
                step = pose_steps_by_idx.get(int(idx_int))
                if step is not None:
                    step_str = format_step_label(step, axis_order)

        table.add_row(
            str(i + 1),
            vid,
            step_str,
            f'{float(r["icp_fitness"]):.4f}',
            f'{float(r["icp_inlier_rmse"]):.5f}' if math.isfinite(float(r["icp_inlier_rmse"])) else "inf",
            f'{float(r["ransac_fitness"]):.4f}',
            str(r["virtual_path"]),
        )

    console.print(table)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Match ONE real LiDAR scan to a virtual pose library via global registration (FPFH+RANSAC) + ICP."
    )
    parser.add_argument(
        "--real",
        type=str, nargs="+",
        default=[DATA_REAL_HUMAN_EXAMPLE],
        help="One or more real scan path(s) or glob(s), relative to repo root unless absolute.",
    )
    parser.add_argument(
        "--virtual-glob",
        type=str,
        default=DATA_VIRTUAL_POSE324_GLOB,
        help='Glob for virtual scans, e.g. "data/virtual/pose324/virtual_*.ply"',
    )
    parser.add_argument(
        "--poses-jsonl",
        type=str,
        default=DATASET_POSES_324_JSONL,
        help='Pose definition jsonl (idx -> step label), e.g. "dataset/poses_324.jsonl".',
    )
    parser.add_argument(
        "--axis-steps-yaml",
        type=str,
        default=CONFIG_AXIS_STEPS_YAML,
        help='Axis order yaml for step label formatting, e.g. "configs/axis_steps.yaml".',
    )
    parser.add_argument(
        "--out-root",
        type=str,
        default=OUTPUT_MATCH_ONE_DIR,
        help='Output root directory for match results, e.g. "data/match_one".',
    )
    parser.add_argument("--voxel", type=float, default=0.03, help="Voxel size for downsampling (meters).")
    parser.add_argument("--topk", type=int, default=10, help="Show/save top-k matches.")
    parser.add_argument("--visualize", action="store_true", help="Visualize the best alignment (Open3D window).")
    parser.add_argument(
        "--visualize-topk",
        type=int,
        default=0,
        help="If >0, visualize top-k (one-by-one). Use with care.",
    )

    # RANSAC params
    parser.add_argument("--ransac-dist-mul", type=float, default=1.5, help="RANSAC max corr dist = voxel * mul.")
    parser.add_argument("--ransac-n", type=int, default=4, help="RANSAC n points.")
    parser.add_argument("--ransac-iter", type=int, default=100000, help="RANSAC max iterations.")
    parser.add_argument("--ransac-conf", type=float, default=0.999, help="RANSAC confidence.")

    # ICP params
    parser.add_argument("--icp-dist-mul", type=float, default=0.4, help="ICP max corr dist = voxel * mul.")
    parser.add_argument("--icp-iter", type=int, default=50, help="ICP max iterations.")
    parser.add_argument("--icp-point-to-point", action="store_true", help="Use point-to-point ICP (default is point-to-plane).")

    args = parser.parse_args()

    repo = repo_root()

    real_paths = build_real_list(repo, args.real)
    if not real_paths:
        console.print(f"[red]No real files found for:[/red] {args.real}")
        return 2

    virtual_paths = build_virtual_list(repo, args.virtual_glob)
    if not virtual_paths:
        console.print(f"[red]No virtual files found for glob:[/red] {args.virtual_glob}")
        return 2

    pose_steps_by_idx: Optional[Dict[int, Dict[str, int]]] = None
    axis_order: Optional[List[str]] = None
    poses_path = Path(args.poses_jsonl)
    if not poses_path.is_absolute():
        poses_path = repo / poses_path
    axis_path = Path(args.axis_steps_yaml)
    if not axis_path.is_absolute():
        axis_path = repo / axis_path
    if poses_path.exists() and axis_path.exists():
        try:
            axis_order = load_axis_steps_order(axis_path)
            pose_steps_by_idx = load_pose_steps_jsonl(poses_path)
        except Exception as e:
            console.print(f"[yellow]Warning:[/yellow] failed to load pose labels: {e}")
            pose_steps_by_idx = None
            axis_order = None

    console.print(f"[cyan]Real candidates:[/cyan] {len(real_paths)} files")
    console.print(f"[cyan]Virtual candidates:[/cyan] {len(virtual_paths)} files")
    console.print(f"[cyan]Voxel:[/cyan] {args.voxel}")

    processed = 0
    for i, real_path in enumerate(real_paths, start=1):
        console.rule(f"[bold cyan]Real {i}/{len(real_paths)}: {real_path.name}[/bold cyan]")
        console.print(f"[cyan]Real:[/cyan] {real_path}")

        try:
            df, transforms = match_one_real_to_virtual_library(
                real_path=real_path,
                virtual_paths=virtual_paths,
                voxel_size=float(args.voxel),
                ransac_distance_multiplier=float(args.ransac_dist_mul),
                icp_distance_multiplier=float(args.icp_dist_mul),
                ransac_n=int(args.ransac_n),
                ransac_max_iteration=int(args.ransac_iter),
                ransac_confidence=float(args.ransac_conf),
                icp_max_iteration=int(args.icp_iter),
                icp_point_to_plane=not bool(args.icp_point_to_point),
                skip_icp_if_ransac_fails=True,
            )
        except Exception as e:
            console.print(f"[red]Failed:[/red] {real_path} -> {e}")
            continue

        outdir = ensure_outdir(repo, args.out_root, real_path)
        save_results(
            outdir,
            real_path,
            df,
            transforms,
            topk=int(args.topk),
            pose_steps_by_idx=pose_steps_by_idx,
            axis_order=axis_order,
        )
        print_topk(df, topk=int(args.topk), pose_steps_by_idx=pose_steps_by_idx, axis_order=axis_order)

        # visualization (per real)
        if args.visualize or (args.visualize_topk and args.visualize_topk > 0):
            df_sorted = df.sort_values(
                by=["score", "icp_fitness", "ransac_fitness", "icp_inlier_rmse"],
                ascending=[False, False, False, True],
            ).reset_index(drop=True)

            real_full = load_point_cloud(real_path)

            k = 1 if args.visualize else int(args.visualize_topk)
            k = max(1, min(k, len(df_sorted)))

            for j in range(k):
                vp = Path(df_sorted.iloc[j]["virtual_path"])
                T = np.asarray(transforms[str(vp)], dtype=np.float64)
                virt_full = load_point_cloud(vp)
                console.print(f"[yellow]Visualize rank {j+1}[/yellow]: {vp.name}")
                draw_alignment(virt_full, real_full, T)

        processed += 1

    if processed == 0:
        console.print("[red]No real files processed successfully.[/red]")
        return 1
    console.print(f"[green]Done:[/green] {processed}/{len(real_paths)} real files processed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
