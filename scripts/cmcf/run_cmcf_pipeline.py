# scripts/cmcf/run_cmcf_pipeline.py
from __future__ import annotations

import argparse
import fnmatch
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import yaml

from scripts.common.io_paths import resolve_repo_path
from scripts.common.json_utils import write_json
from scripts.common.pc_io import expand_inputs, read_point_cloud_any
from scripts.parser.body_frame import BodyFrame, estimate_body_frame
from scripts.parser.label_decoder import decode_labels, head_names_from_decoder, load_decoder_config
from scripts.parser.part_parser import DEFAULT_PART_ORDER, load_part_parser_config, parse_parts_from_points
from scripts.parser.preprocess import preprocess_human_points
from scripts.parser.relation_features import compute_relation_features
from scripts.parser.run_parser_decoder import _write_part_visualization
from scripts.parser.types import PartBox

"""
# virtual canonical/prototype

WAYLAND_DISPLAY= DISPLAY=:0 XDG_SESSION_TYPE=x11 \
python -m scripts run-cmcf-pipeline \
  --input "data/virtual/lunge/*.ply" \
  --input-pattern "*.ply" \
  --part-config configs/parser/part_parser_rules_virtual.yaml \
  --marker-config configs/cmcf/marker_field_virtual.yaml \
  --decoder-config configs/cmcf/decoder_rules_12head.yaml \
  --viz-dir outputs/virtual_lunge/cmcf/viz \
  --viz-write-lines \
  --build-playback \
  --play-gui \
  --play-fps 2 \
  --play-loop \
  --play-log-every 30 \
  --output-dir outputs/virtual_lunge/cmcf

# real query feature

python -m scripts run-cmcf-pipeline \
  --input data/real/SQUAT/human \
  --input-pattern "human_*.ply" \
  --part-config configs/parser/part_parser_rules_v1.yaml \
  --marker-config configs/cmcf/marker_field_virtual.yaml \
  --decoder-config configs/cmcf/decoder_rules_12head.yaml \
  --output-dir outputs/SQUAT/cmcf

# prototype mapping

python -m scripts cmcf-map \
  --prototype-bank outputs/virtual_pdbr/cmcf/prototype_bank.json \
  --query outputs/SQUAT/cmcf \
  --query-pattern "cmcf_*.json" \
  --output outputs/SQUAT/cmcf/prototype_mapping.jsonl \
  --topk 3
"""


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def command_text(cmd: Sequence[str]) -> str:
    return " ".join(str(v) for v in cmd)


def run_step(cmd: Sequence[str], dry_run: bool) -> None:
    print(f"$ {command_text(cmd)}")
    if dry_run:
        return
    subprocess.run([str(v) for v in cmd], check=True)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _collect_input_paths(root: Path, items: Iterable[str], pattern: str) -> List[Path]:
    raw_paths = expand_inputs(root, items, exts=[".ply", ".pcd"])
    if not pattern or pattern in {"*", "*.*"}:
        return raw_paths
    filtered = [path for path in raw_paths if fnmatch.fnmatch(path.name, pattern)]
    return sorted(filtered, key=lambda path: str(path))


def _numeric_suffix(stem: str) -> int | None:
    matched = re.search(r"(\d+)$", stem)
    if not matched:
        return None
    return int(matched.group(1))


def _sequence_key(path: Path, mode: str, regex: re.Pattern[str] | None) -> str:
    if mode == "all":
        return "all"
    if mode == "parent":
        return str(path.parent)
    if mode == "regex":
        if regex is None:
            return path.stem
        matched = regex.search(str(path))
        if matched is None:
            return path.stem
        if matched.groups():
            return matched.group(1)
        return matched.group(0)
    return path.name


def _farthest_point_sampling(points: np.ndarray, count: int) -> np.ndarray:
    if points.shape[0] == 0 or count <= 0:
        return np.zeros((0, 3), dtype=np.float64)
    if points.shape[0] <= count:
        if points.shape[0] == count:
            return points.copy()
        pad_idx = np.zeros((count - points.shape[0],), dtype=np.int64)
        return np.concatenate([points, points[pad_idx]], axis=0)

    points_arr = np.asarray(points, dtype=np.float64)
    selected_idx = np.zeros((count,), dtype=np.int64)
    center = np.mean(points_arr, axis=0)
    d_center = np.linalg.norm(points_arr - center[None, :], axis=1)
    selected_idx[0] = int(np.argmax(d_center))
    min_dist = np.linalg.norm(points_arr - points_arr[selected_idx[0]][None, :], axis=1)

    for i in range(1, count):
        next_idx = int(np.argmax(min_dist))
        selected_idx[i] = next_idx
        d = np.linalg.norm(points_arr - points_arr[next_idx][None, :], axis=1)
        min_dist = np.minimum(min_dist, d)
    return points_arr[selected_idx]


def _allocate_marker_counts(
    *,
    part_points: Mapping[str, np.ndarray],
    marker_count: int,
    min_markers_per_part: int,
    part_names: Sequence[str],
    part_weights: Mapping[str, Any] | None = None,
) -> Dict[str, int]:
    available = [name for name in part_names if np.asarray(part_points.get(name, np.zeros((0, 3))), dtype=np.float64).shape[0] > 0]
    if not available:
        raise ValueError("No non-empty parts available for canonical marker sampling.")

    marker_total = max(int(marker_count), len(available))
    min_per = max(int(min_markers_per_part), 0)
    weights: Dict[str, float] = {}
    for name in available:
        pts = np.asarray(part_points.get(name), dtype=np.float64)
        base = float(max(pts.shape[0], 1))
        mul = 1.0
        if isinstance(part_weights, Mapping):
            mul = max(_safe_float(part_weights.get(name, 1.0), 1.0), 0.0)
        weights[name] = float(base * mul)
    weight_sum = max(sum(weights.values()), 1e-9)

    counts = {name: min_per for name in available}
    remaining = marker_total - (min_per * len(available))
    if remaining < 0:
        remaining = 0

    fractional: List[Tuple[float, str]] = []
    for name in available:
        raw = remaining * (weights[name] / weight_sum)
        add_int = int(np.floor(raw))
        counts[name] += add_int
        fractional.append((raw - add_int, name))
    left = marker_total - sum(counts.values())
    fractional.sort(key=lambda item: item[0], reverse=True)
    for i in range(left):
        counts[fractional[i % len(fractional)][1]] += 1

    return counts


def _build_canonical_markers(
    *,
    canonical_part_points: Mapping[str, np.ndarray],
    marker_count: int,
    min_markers_per_part: int,
    part_names: Sequence[str],
    part_weights: Mapping[str, Any] | None = None,
    attachment_default: str = "surface_nn",
    attachment_by_part: Mapping[str, Any] | None = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, List[int]]]:
    counts = _allocate_marker_counts(
        part_points=canonical_part_points,
        marker_count=marker_count,
        min_markers_per_part=min_markers_per_part,
        part_names=part_names,
        part_weights=part_weights,
    )
    markers: List[Dict[str, Any]] = []
    part_to_marker_ids: Dict[str, List[int]] = {name: [] for name in part_names}
    next_id = 0
    for part_name in part_names:
        k = counts.get(part_name, 0)
        if k <= 0:
            continue
        points = np.asarray(canonical_part_points.get(part_name, np.zeros((0, 3))), dtype=np.float64)
        if points.shape[0] == 0:
            continue
        sampled = _farthest_point_sampling(points, k)
        attachment = str(attachment_default)
        if isinstance(attachment_by_part, Mapping) and part_name in attachment_by_part:
            attachment = str(attachment_by_part.get(part_name))
        for point in sampled:
            marker = {
                "marker_id": int(next_id),
                "part_id": str(part_name),
                "attachment": attachment,
                "canonical_xyz": [float(point[0]), float(point[1]), float(point[2])],
            }
            markers.append(marker)
            part_to_marker_ids[part_name].append(int(next_id))
            next_id += 1
    if not markers:
        raise ValueError("Canonical marker sampling failed: no markers created.")
    return markers, part_to_marker_ids


def _assign_markers_to_candidates(markers_xyz: np.ndarray, candidate_xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    m = int(markers_xyz.shape[0])
    if m == 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0,), dtype=np.float64)
    if candidate_xyz.shape[0] == 0:
        return markers_xyz.copy(), np.zeros((m,), dtype=np.float64)

    markers = np.asarray(markers_xyz, dtype=np.float64)
    candidates = np.asarray(candidate_xyz, dtype=np.float64)
    n = int(candidates.shape[0])
    diff = markers[:, None, :] - candidates[None, :, :]
    dist = np.linalg.norm(diff, axis=2)

    assigned = np.full((m,), -1, dtype=np.int64)
    used = np.zeros((n,), dtype=bool)
    flat_order = np.argsort(dist, axis=None)
    for flat_idx in flat_order.tolist():
        marker_idx = int(flat_idx // n)
        candidate_idx = int(flat_idx % n)
        if assigned[marker_idx] >= 0 or used[candidate_idx]:
            continue
        assigned[marker_idx] = candidate_idx
        used[candidate_idx] = True
        if int(np.count_nonzero(assigned >= 0)) >= min(m, n):
            break
    for marker_idx in range(m):
        if assigned[marker_idx] >= 0:
            continue
        assigned[marker_idx] = int(np.argmin(dist[marker_idx]))

    mapped = candidates[assigned]
    mapped_dist = dist[np.arange(m), assigned]
    return mapped, mapped_dist


def _build_marker_groups(
    *,
    markers: Sequence[Mapping[str, Any]],
    marker_xyz: np.ndarray,
    part_names: Sequence[str],
) -> Dict[str, np.ndarray]:
    grouped: Dict[str, List[np.ndarray]] = {name: [] for name in part_names}
    for idx, marker in enumerate(markers):
        part_id = str(marker.get("part_id", ""))
        if part_id not in grouped:
            continue
        grouped[part_id].append(np.asarray(marker_xyz[idx], dtype=np.float64))
    out: Dict[str, np.ndarray] = {}
    for part_name in part_names:
        pts = grouped.get(part_name, [])
        if not pts:
            out[part_name] = np.zeros((0, 3), dtype=np.float64)
            continue
        out[part_name] = np.stack(pts, axis=0).astype(np.float64)
    return out


def _build_boxes_from_groups(
    marker_groups: Mapping[str, np.ndarray],
    *,
    part_names: Sequence[str],
) -> Dict[str, PartBox]:
    boxes: Dict[str, PartBox] = {}
    for part_name in part_names:
        pts = np.asarray(marker_groups.get(part_name, np.zeros((0, 3))), dtype=np.float64)
        if pts.shape[0] == 0:
            boxes[part_name] = PartBox.empty(part_name)
            continue
        boxes[part_name] = PartBox.from_points(
            part_name,
            pts,
            min_points=1,
            allow_small=True,
            min_size=0.01,
            robust_quantile=0.0,
        )
    return boxes


def _write_marker_visualization(
    *,
    point_path: Path,
    body_frame: BodyFrame,
    markers: Sequence[Mapping[str, Any]],
    marker_xyz_body: np.ndarray,
    viz_dir: Path,
    lineset_path: Path | None = None,
    glyph_radius: float = 0.012,
) -> Dict[str, Any]:
    import open3d as o3d

    radius = max(float(glyph_radius), 1e-5)
    viz_dir.mkdir(parents=True, exist_ok=True)
    stem = point_path.stem
    marker_world = body_frame.body_to_world(np.asarray(marker_xyz_body, dtype=np.float64))
    marker_color = np.array([1.0, 0.0, 1.0], dtype=np.float64)  # magenta: lidar/part 색과 구분
    colors = np.tile(marker_color[None, :], (max(int(len(markers)), 0), 1))
    offsets = np.array(
        [
            [0.0, 0.0, 0.0],
            [radius, 0.0, 0.0],
            [-radius, 0.0, 0.0],
            [0.0, radius, 0.0],
            [0.0, -radius, 0.0],
            [0.0, 0.0, radius],
            [0.0, 0.0, -radius],
        ],
        dtype=np.float64,
    )
    marker_glyph_world = (marker_world[:, None, :] + offsets[None, :, :]).reshape(-1, 3)
    marker_glyph_colors = np.repeat(colors, offsets.shape[0], axis=0)

    marker_cloud = o3d.geometry.PointCloud()
    marker_cloud.points = o3d.utility.Vector3dVector(marker_glyph_world.astype(np.float64))
    marker_cloud.colors = o3d.utility.Vector3dVector(marker_glyph_colors.astype(np.float64))
    marker_path = (viz_dir / f"{stem}_markers.ply").resolve()
    o3d.io.write_point_cloud(str(marker_path), marker_cloud, write_ascii=False)

    overlay_applied = False
    if lineset_path is not None and lineset_path.exists():
        try:
            line_cloud = o3d.io.read_point_cloud(str(lineset_path))
            line_pts = np.asarray(line_cloud.points, dtype=np.float64)
            line_cols = np.asarray(line_cloud.colors, dtype=np.float64)
            if line_pts.ndim != 2 or line_pts.shape[1] != 3:
                line_pts = np.zeros((0, 3), dtype=np.float64)
            if line_cols.shape != line_pts.shape:
                line_cols = np.tile(np.array([[0.85, 0.85, 0.85]], dtype=np.float64), (line_pts.shape[0], 1))
            merged_pts = np.concatenate([line_pts, marker_glyph_world], axis=0)
            merged_cols = np.concatenate([line_cols, marker_glyph_colors], axis=0)
            merged_cloud = o3d.geometry.PointCloud()
            merged_cloud.points = o3d.utility.Vector3dVector(merged_pts.astype(np.float64))
            merged_cloud.colors = o3d.utility.Vector3dVector(merged_cols.astype(np.float64))
            o3d.io.write_point_cloud(str(lineset_path), merged_cloud, write_ascii=False)
            overlay_applied = True
        except Exception:
            overlay_applied = False

    return {
        "markers_ply": str(marker_path),
        "marker_count": int(len(markers)),
        "marker_color_rgb": [float(v) for v in marker_color.tolist()],
        "marker_glyph_radius": float(radius),
        "markers_overlaid_on_lineset": bool(overlay_applied),
    }


def _write_raw_visualization(
    *,
    point_path: Path,
    points_world: np.ndarray,
    colors_world: np.ndarray | None,
    viz_dir: Path,
) -> Dict[str, Any]:
    import open3d as o3d

    viz_dir.mkdir(parents=True, exist_ok=True)
    stem = point_path.stem
    raw_cloud = o3d.geometry.PointCloud()
    points = np.asarray(points_world, dtype=np.float64)
    raw_cloud.points = o3d.utility.Vector3dVector(points)

    use_input_colors = (
        isinstance(colors_world, np.ndarray)
        and colors_world.ndim == 2
        and colors_world.shape == points.shape
        and colors_world.shape[0] > 0
    )
    if use_input_colors:
        raw_cloud.colors = o3d.utility.Vector3dVector(np.asarray(colors_world, dtype=np.float64))
    else:
        default_color = np.tile(np.array([[0.70, 0.70, 0.70]], dtype=np.float64), (points.shape[0], 1))
        raw_cloud.colors = o3d.utility.Vector3dVector(default_color)

    raw_path = (viz_dir / f"{stem}_raw.ply").resolve()
    o3d.io.write_point_cloud(str(raw_path), raw_cloud, write_ascii=False)
    return {
        "raw_ply": str(raw_path),
        "raw_point_count": int(points.shape[0]),
        "raw_uses_input_colors": bool(use_input_colors),
    }


def _parse_gt_jsonl(gt_jsonl_path: Path, head_names: Sequence[str]) -> Dict[int, Dict[str, Any]]:
    gt_map: Dict[int, Dict[str, Any]] = {}
    with gt_jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if not isinstance(row, Mapping):
                continue
            frame_raw = row.get("Frame")
            step_raw = row.get("step")
            if frame_raw is None or step_raw is None:
                continue
            frame = int(frame_raw)
            if isinstance(step_raw, str):
                values = [item.strip() for item in step_raw.strip("{}").split(",") if item.strip()]
            elif isinstance(step_raw, Sequence):
                values = list(step_raw)
            else:
                continue
            if len(values) != len(head_names):
                continue
            parsed = []
            for value in values:
                if isinstance(value, str):
                    value = value.strip()
                if isinstance(value, (int, float)):
                    parsed.append(int(value) if float(value).is_integer() else float(value))
                else:
                    try:
                        v = float(value)
                        parsed.append(int(v) if v.is_integer() else v)
                    except Exception:
                        parsed.append(value)
            gt_map[frame] = {head_names[i]: parsed[i] for i in range(len(head_names))}
    return gt_map


def _infer_frame_offset(pred_frames: Sequence[int], gt_frames: Sequence[int]) -> int:
    pred_set = set(pred_frames)
    gt_set = set(gt_frames)
    best_offset = 0
    best_overlap = -1
    for offset in (0, 1):
        overlap = sum(1 for frame in gt_set if (frame + offset) in pred_set)
        if overlap > best_overlap:
            best_overlap = overlap
            best_offset = offset
    return int(best_offset)


def _print_eval(
    *,
    predictions: Sequence[Tuple[int, Dict[str, Any]]],
    gt_map: Mapping[int, Mapping[str, Any]],
    head_names: Sequence[str],
    frame_offset: int,
) -> None:
    total = 0
    correct = 0
    sample_total = 0
    sample_exact = 0
    per_head_total = {name: 0 for name in head_names}
    per_head_correct = {name: 0 for name in head_names}
    for frame_num, labels in predictions:
        gt = gt_map.get(int(frame_num - frame_offset))
        if gt is None:
            continue
        sample_total += 1
        row_ok = True
        for head_name in head_names:
            if head_name not in gt:
                continue
            pred_val = labels.get(head_name)
            gt_val = gt.get(head_name)
            per_head_total[head_name] += 1
            total += 1
            if pred_val == gt_val:
                correct += 1
                per_head_correct[head_name] += 1
            else:
                row_ok = False
        if row_ok:
            sample_exact += 1
    if total <= 0:
        print("[cmcf][eval] no matched GT samples.")
        return
    print(f"[cmcf][eval] label_accuracy={100.0 * correct / total:.2f}% ({correct}/{total})")
    print(
        f"[cmcf][eval] sample_exact_accuracy={100.0 * sample_exact / max(sample_total, 1):.2f}% "
        f"({sample_exact}/{sample_total})"
    )
    for head_name in head_names:
        h_total = per_head_total[head_name]
        h_correct = per_head_correct[head_name]
        acc = (100.0 * h_correct / h_total) if h_total else 0.0
        print(f"[cmcf][eval] head={head_name} acc={acc:.2f}% ({h_correct}/{h_total})")


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "CMCF pipeline: canonical marker sampling + correspondence transport + "
            "feature-only 12-head label decoding."
        )
    )
    parser.add_argument("--input", type=str, nargs="+", required=True, help="Input virtual/real point cloud files or globs.")
    parser.add_argument("--input-pattern", type=str, default="*.ply")
    parser.add_argument("--part-config", type=str, default="configs/parser/part_parser_rules_virtual.yaml")
    parser.add_argument("--marker-config", type=str, default="configs/cmcf/marker_field_virtual.yaml")
    parser.add_argument("--decoder-config", type=str, default="configs/cmcf/decoder_rules_12head.yaml")
    parser.add_argument("--decoder-profile", type=str, default="global")
    parser.add_argument("--output-dir", type=str, default="outputs/cmcf")
    parser.add_argument("--output-prefix", type=str, default="cmcf_")
    parser.add_argument("--marker-count", type=int, default=None)
    parser.add_argument("--canonical-index", type=int, default=None)
    parser.add_argument("--min-markers-per-part", type=int, default=None)
    parser.add_argument(
        "--sequence-mode",
        type=str,
        default="single",
        choices=["single", "parent", "regex", "all"],
        help="Temporal state grouping mode for relation delta / decoder temporal.",
    )
    parser.add_argument("--sequence-regex", type=str, default=None, help="Regex for --sequence-mode regex (group1 = key).")
    parser.add_argument("--gt-jsonl", type=str, default=None)
    parser.add_argument("--gt-frame-offset", type=str, default="auto", choices=["auto", "0", "1"])
    parser.add_argument(
        "--viz-dir",
        type=str,
        default=None,
        help="Optional CMCF visualization output directory (colored parts/boxes + markers).",
    )
    parser.add_argument(
        "--viz-write-lines",
        action="store_true",
        help="Also export line-set PLY for part boxes when --viz-dir is set.",
    )
    parser.add_argument(
        "--build-playback",
        action="store_true",
        help="After CMCF run, build compressed playback NPZ from viz PLY sequence.",
    )
    parser.add_argument(
        "--playback-output",
        type=str,
        default=None,
        help="Playback NPZ path. Default: <output-dir>/cmcf_playback.npz",
    )
    parser.add_argument(
        "--playback-no-lines",
        action="store_true",
        help="Exclude *_part_boxes_lineset.ply when building playback NPZ.",
    )
    parser.add_argument(
        "--play-gui",
        action="store_true",
        help="Play playback NPZ using Open3D viewer after build (or existing NPZ).",
    )
    parser.add_argument("--play-fps", type=float, default=10.0, help="Playback FPS for --play-gui.")
    parser.add_argument("--play-loop", action="store_true", help="Use --loop for --play-gui.")
    parser.add_argument(
        "--play-log-every",
        type=int,
        default=10,
        help="Frame log interval passed to play-parser-playback (--log-every).",
    )
    parser.add_argument("--python", type=str, default=sys.executable, help="Python executable used for playback build/play commands.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)
    root = repo_root()
    input_paths = _collect_input_paths(root, args.input, args.input_pattern)
    if not input_paths:
        raise FileNotFoundError("No input point clouds found for CMCF pipeline.")

    output_dir = resolve_repo_path(root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = resolve_repo_path(root, args.viz_dir) if args.viz_dir else None
    if viz_dir is not None:
        viz_dir.mkdir(parents=True, exist_ok=True)

    part_config_path = resolve_repo_path(root, args.part_config)
    marker_config_path = resolve_repo_path(root, args.marker_config)
    decoder_config_path = resolve_repo_path(root, args.decoder_config)

    part_cfg = load_part_parser_config(part_config_path if part_config_path.exists() else None)
    marker_cfg: Dict[str, Any] = {}
    if marker_config_path.exists():
        with marker_config_path.open("r", encoding="utf-8") as handle:
            marker_cfg = yaml.safe_load(handle) or {}
            if not isinstance(marker_cfg, Mapping):
                marker_cfg = {}

    decoder_cfg = load_decoder_config(
        decoder_config_path if decoder_config_path.exists() else None,
        profile=str(args.decoder_profile),
    )
    head_names = head_names_from_decoder(decoder_cfg)
    part_names = list(DEFAULT_PART_ORDER)

    marker_count = int(
        args.marker_count
        if args.marker_count is not None
        else marker_cfg.get("marker_count", 100)
    )
    min_markers_per_part = int(
        args.min_markers_per_part
        if args.min_markers_per_part is not None
        else marker_cfg.get("min_markers_per_part", 2)
    )
    canonical_index = int(
        np.clip(
            int(
                args.canonical_index
                if args.canonical_index is not None
                else marker_cfg.get("canonical_index", 0)
            ),
            0,
            max(len(input_paths) - 1, 0),
        )
    )
    part_weights = marker_cfg.get("part_weights", {}) if isinstance(marker_cfg.get("part_weights"), Mapping) else {}
    attachment_default = str(marker_cfg.get("attachment", "surface_nn"))
    attachment_by_part = (
        marker_cfg.get("attachment_by_part", {})
        if isinstance(marker_cfg.get("attachment_by_part"), Mapping)
        else {}
    )

    preprocess_cfg = part_cfg.get("preprocess", {}) if isinstance(part_cfg, Mapping) else {}
    body_frame_cfg = part_cfg.get("body_frame", {}) if isinstance(part_cfg, Mapping) else {}
    if not isinstance(preprocess_cfg, Mapping):
        preprocess_cfg = {}
    if not isinstance(body_frame_cfg, Mapping):
        body_frame_cfg = {}

    print(f"[cmcf] samples={len(input_paths)}")
    print(f"[cmcf] canonical_index={canonical_index} marker_count={marker_count}")
    print(f"[cmcf] part_config={part_config_path}")
    print(f"[cmcf] decoder_config={decoder_config_path} profile={decoder_cfg.get('selected_profile')}")

    canonical_path = input_paths[canonical_index]
    canonical_cloud = read_point_cloud_any(canonical_path)
    canonical_world = np.asarray(canonical_cloud.points, dtype=np.float64)
    canonical_pre, canonical_pre_meta = preprocess_human_points(canonical_world, preprocess_cfg)
    canonical_frame, canonical_body_meta = estimate_body_frame(canonical_pre, body_frame_cfg)
    canonical_body = canonical_frame.world_to_body(canonical_pre)
    _, canonical_parse_meta, canonical_part_points, _ = parse_parts_from_points(canonical_body, part_cfg, prev_state=None)

    markers, part_to_marker_ids = _build_canonical_markers(
        canonical_part_points=canonical_part_points,
        marker_count=marker_count,
        min_markers_per_part=min_markers_per_part,
        part_names=part_names,
        part_weights=part_weights,
        attachment_default=attachment_default,
        attachment_by_part=attachment_by_part,
    )
    markers_sorted = sorted(markers, key=lambda row: int(row["marker_id"]))
    canonical_xyz = np.array([row["canonical_xyz"] for row in markers_sorted], dtype=np.float64)

    canonical_payload = {
        "meta": {
            "method": "CMCF",
            "marker_count": int(len(markers_sorted)),
            "canonical_index": int(canonical_index),
            "canonical_file": str(canonical_path),
            "part_config": str(part_config_path),
            "marker_config": str(marker_config_path),
            "marker_part_weights": part_weights,
            "marker_attachment_default": attachment_default,
            "decoder_config": str(decoder_config_path),
            "decoder_profile": str(decoder_cfg.get("selected_profile", "unknown")),
        },
        "markers": markers_sorted,
        "part_to_marker_ids": {k: v for k, v in part_to_marker_ids.items() if v},
        "canonical_preprocess_meta": canonical_pre_meta,
        "canonical_body_meta": canonical_body_meta,
        "canonical_parse_meta": canonical_parse_meta,
    }
    write_json(output_dir / "canonical_markers.json", canonical_payload, ascii_flag=False, compact=False)

    relation_state: Dict[str, Any] | None = None
    decoder_state: Dict[str, Any] | None = None
    sequence_regex = re.compile(args.sequence_regex) if args.sequence_mode == "regex" and args.sequence_regex else None
    prev_sequence_key: str | None = None

    prototype_rows: List[Dict[str, Any]] = []
    pred_rows: List[Tuple[int, Dict[str, Any]]] = []
    track_jsonl_path = output_dir / "cmcf_markers.jsonl"
    feature_jsonl_path = output_dir / "cmcf_features.jsonl"
    with track_jsonl_path.open("w", encoding="utf-8") as track_handle, feature_jsonl_path.open("w", encoding="utf-8") as feat_handle:
        for frame_idx, point_path in enumerate(input_paths):
            sequence_key = _sequence_key(point_path, args.sequence_mode, sequence_regex)
            if prev_sequence_key is None:
                prev_sequence_key = sequence_key
            elif sequence_key != prev_sequence_key:
                relation_state = None
                decoder_state = None
                prev_sequence_key = sequence_key

            cloud = read_point_cloud_any(point_path)
            points_world = np.asarray(cloud.points, dtype=np.float64)
            colors_world = np.asarray(cloud.colors, dtype=np.float64)
            points_pre, preprocess_meta = preprocess_human_points(points_world, preprocess_cfg)
            body_frame, body_meta = estimate_body_frame(points_pre, body_frame_cfg)
            points_body = body_frame.world_to_body(points_pre)
            _, parse_meta, parsed_part_points, _ = parse_parts_from_points(points_body, part_cfg, prev_state=None)

            transported_xyz = np.zeros_like(canonical_xyz, dtype=np.float64)
            transport_dist = np.zeros((canonical_xyz.shape[0],), dtype=np.float64)
            for part_name in part_names:
                marker_ids = part_to_marker_ids.get(part_name, [])
                if not marker_ids:
                    continue
                marker_idx = np.asarray(marker_ids, dtype=np.int64)
                marker_ref = canonical_xyz[marker_idx]
                candidates = np.asarray(parsed_part_points.get(part_name, np.zeros((0, 3))), dtype=np.float64)
                if candidates.shape[0] == 0:
                    candidates = points_body
                assigned_xyz, assigned_dist = _assign_markers_to_candidates(marker_ref, candidates)
                transported_xyz[marker_idx] = assigned_xyz
                transport_dist[marker_idx] = assigned_dist

            marker_groups = _build_marker_groups(
                markers=markers_sorted,
                marker_xyz=transported_xyz,
                part_names=part_names,
            )
            marker_boxes = _build_boxes_from_groups(marker_groups, part_names=part_names)
            relation_obj, relation_state = compute_relation_features(
                marker_boxes,
                decoder_cfg,
                prev_state=relation_state,
            )
            feature_map = relation_obj.get("flat_features", {})
            labels, decode_debug, decoder_state = decode_labels(
                feature_map if isinstance(feature_map, Mapping) else {},
                decoder_cfg,
                temporal_state=decoder_state,
            )

            marker_rows = []
            for idx, marker in enumerate(markers_sorted):
                xyz = transported_xyz[idx]
                marker_rows.append(
                    {
                        "marker_id": int(marker["marker_id"]),
                        "part_id": str(marker["part_id"]),
                        "xyz": [float(xyz[0]), float(xyz[1]), float(xyz[2])],
                        "transport_dist": float(transport_dist[idx]),
                    }
                )
            payload = {
                "meta": {
                    "method": "CMCF",
                    "source_file": str(point_path),
                    "frame_index": int(frame_idx),
                    "sequence_key": str(sequence_key),
                    "decoder_profile": str(decoder_cfg.get("selected_profile", "unknown")),
                },
                "preprocess_meta": preprocess_meta,
                "body_frame_meta": body_meta,
                "parse_meta": parse_meta,
                "markers": marker_rows,
                "marker_groups": {
                    name: marker_groups[name].tolist() if marker_groups[name].shape[0] > 0 else []
                    for name in part_names
                },
                "parts_body": {name: marker_boxes[name].to_jsonable() for name in part_names},
                "relations": relation_obj.get("pair_features", {}),
                "derived_features": relation_obj.get("derived_features", {}),
                "feature_map": feature_map,
                "labels": labels,
                "decode_debug": decode_debug,
            }
            if viz_dir is not None:
                viz_stub = point_path.with_name(f"{point_path.stem}_cmcf{point_path.suffix}")
                viz_artifacts = _write_part_visualization(
                    point_path=viz_stub,
                    points_body=points_body,
                    body_frame=body_frame,
                    parts=marker_boxes,
                    part_points=None,
                    viz_dir=viz_dir,
                    write_lines=bool(args.viz_write_lines),
                )
                viz_artifacts.update(
                    _write_marker_visualization(
                        point_path=viz_stub,
                        body_frame=body_frame,
                        markers=markers_sorted,
                        marker_xyz_body=transported_xyz,
                        viz_dir=viz_dir,
                    )
                )
                viz_artifacts.update(
                    _write_raw_visualization(
                        point_path=viz_stub,
                        points_world=points_world,
                        colors_world=colors_world,
                        viz_dir=viz_dir,
                    )
                )
                viz_artifacts["part_color_source"] = "cmcf_boxes"
                viz_artifacts["part_box_source"] = "cmcf_marker_groups"
                payload["visualization"] = viz_artifacts
            output_name = f"{args.output_prefix}{point_path.stem}.json"
            write_json(output_dir / output_name, payload, ascii_flag=False, compact=False)

            track_handle.write(
                json.dumps(
                    {
                        "Frame": int(frame_idx),
                        "File": str(point_path),
                        "markers": marker_rows,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            feat_handle.write(
                json.dumps(
                    {
                        "Frame": int(frame_idx),
                        "File": str(point_path),
                        "labels": labels,
                        "feature_map": feature_map,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

            prototype_rows.append(
                {
                    "prototype_id": int(frame_idx),
                    "source_file": str(point_path),
                    "frame_index": int(frame_idx),
                    "sequence_key": str(sequence_key),
                    "labels": labels,
                    "feature_map": feature_map,
                }
            )
            numeric = _numeric_suffix(point_path.stem)
            if numeric is not None:
                pred_rows.append((numeric, dict(labels)))

    write_json(
        output_dir / "prototype_bank.json",
        {
            "meta": {
                "method": "CMCF",
                "samples": int(len(prototype_rows)),
                "decoder_profile": str(decoder_cfg.get("selected_profile", "unknown")),
                "feature_source": "marker_group_relations",
            },
            "prototypes": prototype_rows,
        },
        ascii_flag=False,
        compact=False,
    )

    if args.gt_jsonl:
        gt_path = resolve_repo_path(root, args.gt_jsonl)
        if gt_path.exists():
            gt_map = _parse_gt_jsonl(gt_path, head_names)
            if gt_map:
                if args.gt_frame_offset == "auto":
                    offset = _infer_frame_offset([row[0] for row in pred_rows], list(gt_map.keys()))
                else:
                    offset = int(args.gt_frame_offset)
                print(f"[cmcf][eval] gt={gt_path}")
                print(f"[cmcf][eval] frame_offset={offset}")
                _print_eval(
                    predictions=pred_rows,
                    gt_map=gt_map,
                    head_names=head_names,
                    frame_offset=offset,
                )

    playback_output = (
        resolve_repo_path(root, args.playback_output)
        if args.playback_output
        else (output_dir / "cmcf_playback.npz")
    )
    if args.build_playback:
        if viz_dir is None:
            raise ValueError("--build-playback requires --viz-dir (source of *_parts_colored.ply).")
        playback_cmd: List[str] = [
            str(args.python),
            "-m",
            "scripts",
            "build-parser-playback",
            "--input-dir",
            str(viz_dir),
            "--parts-pattern",
            "*_cmcf_parts_colored.ply",
            "--raw-pattern",
            "*_cmcf_raw.ply",
            "--lines-pattern",
            "*_cmcf_part_boxes_lineset.ply",
            "--markers-pattern",
            "*_cmcf_markers.ply",
            "--output",
            str(playback_output),
        ]
        if args.playback_no_lines:
            playback_cmd.append("--no-lines")
        run_step(playback_cmd, dry_run=bool(args.dry_run))

    if args.play_gui:
        if not playback_output.exists() and not bool(args.dry_run):
            raise FileNotFoundError(
                f"Playback NPZ not found: {playback_output} "
                "(run with --build-playback or set --playback-output)."
            )
        play_cmd: List[str] = [
            str(args.python),
            "-m",
            "scripts",
            "play-parser-playback",
            "--input",
            str(playback_output),
            "--fps",
            str(float(args.play_fps)),
            "--log-every",
            str(max(0, int(args.play_log_every))),
        ]
        if args.play_loop:
            play_cmd.append("--loop")
        run_step(play_cmd, dry_run=bool(args.dry_run))

    print(
        f"[cmcf] done. markers={len(markers_sorted)} frames={len(input_paths)} "
        f"output={output_dir}"
    )


if __name__ == "__main__":
    main()
