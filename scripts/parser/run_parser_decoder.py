# scripts/parser/run_parser_decoder.py
from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import open3d as o3d

from scripts.common.io_paths import resolve_repo_path
from scripts.common.json_utils import write_json
from scripts.common.path_utils import natural_key_path
from scripts.common.pc_io import read_point_cloud_any
from scripts.feature.heads import load_head_specs
from scripts.parser.body_frame import BodyFrame, estimate_body_frame
from scripts.parser.label_decoder import decode_labels, load_decoder_config
from scripts.parser.part_parser import DEFAULT_PART_ORDER, load_part_parser_config, parse_parts_from_points
from scripts.parser.preprocess import preprocess_human_points
from scripts.parser.relation_features import compute_relation_features


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _has_glob_chars(text: str) -> bool:
    return any(char in text for char in ("*", "?", "["))


def collect_input_paths(root: Path, items: Sequence[str], input_pattern: str) -> List[Path]:
    out: List[Path] = []
    seen: set[Path] = set()
    for raw in items:
        raw_path = Path(raw)
        abs_raw = raw_path if raw_path.is_absolute() else (root / raw_path)
        abs_raw = abs_raw.resolve()

        candidates: List[Path] = []
        if _has_glob_chars(str(raw)):
            candidates.extend(Path(match).resolve() for match in glob.glob(str(root / raw)))
        elif abs_raw.is_file():
            candidates.append(abs_raw)
        elif abs_raw.is_dir():
            candidates.extend(path.resolve() for path in abs_raw.glob(input_pattern))
        else:
            raise FileNotFoundError(f"Input path not found: {raw}")

        for path in candidates:
            if not path.is_file():
                continue
            if path.suffix.lower() not in (".pcd", ".ply"):
                continue
            if path not in seen:
                seen.add(path)
                out.append(path)
    out.sort(key=natural_key_path)
    if not out:
        raise ValueError("No .pcd/.ply files found.")
    return out


def _numeric_suffix(text: str) -> int | None:
    match = re.search(r"(\d+)$", text)
    if match is None:
        return None
    return int(match.group(1))


def _normalize_label_value(value: Any) -> Any:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if float(value).is_integer() else value
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "":
            return value
        try:
            parsed = float(stripped)
            return int(parsed) if parsed.is_integer() else parsed
        except ValueError:
            return value
    return value


def _parse_step_vector(raw_step: Any, *, expected_len: int, source: str) -> List[Any]:
    values: List[Any]
    if isinstance(raw_step, str):
        text = raw_step.strip()
        if text.startswith("{") and text.endswith("}"):
            text = text[1:-1]
        values = [token.strip() for token in text.split(",") if token.strip()] if text else []
    elif isinstance(raw_step, (list, tuple)):
        values = list(raw_step)
    else:
        raise TypeError(f"Unsupported step type in {source}: {type(raw_step).__name__}")
    if len(values) != expected_len:
        raise ValueError(f"`step` length mismatch in {source}: expected {expected_len}, got {len(values)}")
    return [_normalize_label_value(v) for v in values]


def _load_gt_from_jsonl(gt_jsonl_path: Path, head_names: Sequence[str]) -> Dict[int, Dict[str, Any]]:
    gt_map: Dict[int, Dict[str, Any]] = {}
    with gt_jsonl_path.open("r", encoding="utf-8") as handle:
        for line_num, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"JSONL row must be object: {gt_jsonl_path}:{line_num}")
            frame_raw = row.get("Frame")
            if frame_raw is None:
                raise KeyError(f"Missing `Frame` in {gt_jsonl_path}:{line_num}")
            frame = int(frame_raw)
            step_vals = _parse_step_vector(
                row.get("step"),
                expected_len=len(head_names),
                source=f"{gt_jsonl_path}:{line_num}",
            )
            gt_map[frame] = {head_name: step_vals[idx] for idx, head_name in enumerate(head_names)}
    if not gt_map:
        raise ValueError(f"No valid GT rows in {gt_jsonl_path}")
    return gt_map


def _infer_frame_offset(pred_numeric_keys: Sequence[int], gt_frames: Sequence[int]) -> int:
    pred_set = set(pred_numeric_keys)
    gt_set = set(gt_frames)
    best_offset = 0
    best_overlap = -1
    for offset in (0, 1):
        overlap = sum(1 for frame in gt_set if (frame + offset) in pred_set)
        if overlap > best_overlap:
            best_overlap = overlap
            best_offset = offset
    return best_offset


def _evaluate_predictions(
    *,
    pred_rows: Sequence[Tuple[int, Dict[str, Any]]],
    gt_jsonl_path: Path,
    head_names: Sequence[str],
    frame_offset: str,
) -> None:
    gt_map = _load_gt_from_jsonl(gt_jsonl_path, head_names)
    pred_keys = [key for key, _ in pred_rows]
    offset = _infer_frame_offset(pred_keys, list(gt_map.keys())) if frame_offset == "auto" else int(frame_offset)

    per_head_correct = {head_name: 0 for head_name in head_names}
    per_head_total = {head_name: 0 for head_name in head_names}
    sample_exact = 0
    sample_compared = 0
    missing_gt = 0

    for pred_key, pred_labels in pred_rows:
        frame = pred_key - offset
        gt_labels = gt_map.get(frame)
        if gt_labels is None:
            missing_gt += 1
            continue
        sample_compared += 1
        all_match = True
        for head_name in head_names:
            gt_value = _normalize_label_value(gt_labels[head_name])
            pred_value = _normalize_label_value(pred_labels[head_name])
            per_head_total[head_name] += 1
            if pred_value == gt_value:
                per_head_correct[head_name] += 1
            else:
                all_match = False
        if all_match:
            sample_exact += 1

    total_correct = sum(per_head_correct.values())
    total_labels = sum(per_head_total.values())
    label_acc = (100.0 * total_correct / total_labels) if total_labels else 0.0
    sample_acc = (100.0 * sample_exact / sample_compared) if sample_compared else 0.0

    print(f"[parser-decoder][eval] gt={gt_jsonl_path}")
    print(
        f"[parser-decoder][eval] matched_samples={sample_compared}/{len(pred_rows)} "
        f"(missing_gt={missing_gt}) frame_offset={offset}"
    )
    print(
        f"[parser-decoder][eval] label_accuracy={label_acc:.2f}% "
        f"({total_correct}/{total_labels if total_labels else 0})"
    )
    print(
        f"[parser-decoder][eval] sample_exact_accuracy={sample_acc:.2f}% "
        f"({sample_exact}/{sample_compared if sample_compared else 0})"
    )
    for head_name in head_names:
        correct = per_head_correct[head_name]
        total = per_head_total[head_name]
        acc = (100.0 * correct / total) if total else 0.0
        print(f"[parser-decoder][eval] head={head_name} acc={acc:.2f}% ({correct}/{total})")


def _coerce_to_allowed_class(value: Any, class_values: Sequence[Any], default_label: Any) -> Any:
    normalized = _normalize_label_value(value)
    if normalized in class_values:
        return normalized
    if not class_values:
        return default_label

    numeric_candidates = [v for v in class_values if isinstance(v, (int, float))]
    if isinstance(normalized, (int, float)) and numeric_candidates:
        target = float(normalized)
        return min(numeric_candidates, key=lambda item: abs(float(item) - target))
    return class_values[0]


def _body_to_world_point(frame: BodyFrame, point_body: np.ndarray) -> List[float]:
    world = frame.body_to_world(point_body.reshape(1, 3))[0]
    return [float(v) for v in world.tolist()]


def _part_world_centers(parts: Mapping[str, Any], frame: BodyFrame) -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = {}
    for name, part_box in parts.items():
        if not getattr(part_box, "valid", False):
            continue
        center = getattr(part_box, "center_xyz", None)
        if center is None:
            continue
        out[name] = _body_to_world_point(frame, np.asarray(center, dtype=np.float64))
    return out


PART_COLORS: Dict[str, Tuple[float, float, float]] = {
    "head": (0.96, 0.76, 0.17),  # yellow
    "torso": (0.24, 0.60, 0.96),  # blue
    "left_upper_arm": (0.88, 0.33, 0.33),  # red
    "left_lower_arm": (0.76, 0.16, 0.16),  # dark red
    "right_upper_arm": (1.00, 0.55, 0.00),  # vivid orange
    "right_lower_arm": (0.82, 0.35, 0.00),  # burnt orange
    "left_thigh": (0.24, 0.72, 0.36),  # green
    "left_shin": (0.14, 0.58, 0.24),  # dark green
    "right_thigh": (0.65, 0.44, 0.95),  # violet
    "right_shin": (0.43, 0.28, 0.82),  # indigo
}


def _box_corners(min_xyz: np.ndarray, max_xyz: np.ndarray) -> np.ndarray:
    x0, y0, z0 = [float(v) for v in min_xyz.tolist()]
    x1, y1, z1 = [float(v) for v in max_xyz.tolist()]
    return np.array(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ],
        dtype=np.float64,
    )


_BOX_LINES = np.array(
    [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 4],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ],
    dtype=np.int32,
)


def _sample_edge_points(p0: np.ndarray, p1: np.ndarray, samples: int) -> np.ndarray:
    num = max(int(samples), 2)
    t = np.linspace(0.0, 1.0, num=num, endpoint=True, dtype=np.float64)[:, None]
    return p0[None, :] * (1.0 - t) + p1[None, :] * t


def _assign_colors_from_part_points(
    points_body: np.ndarray,
    parts: Mapping[str, Any],
    part_points: Mapping[str, np.ndarray],
) -> np.ndarray:
    color_default = np.array([0.60, 0.60, 0.60], dtype=np.float64)
    colors = np.tile(color_default[None, :], (points_body.shape[0], 1))
    fallback_max_distance = 0.08

    lookup: Dict[Tuple[float, float, float], List[int]] = {}
    for idx, row in enumerate(points_body):
        key = (float(row[0]), float(row[1]), float(row[2]))
        lookup.setdefault(key, []).append(idx)

    candidate_parts: List[List[str]] = [[] for _ in range(points_body.shape[0])]
    for name in DEFAULT_PART_ORDER:
        pts = part_points.get(name)
        if pts is None or pts.size == 0:
            continue
        for row in np.asarray(pts, dtype=np.float64):
            key = (float(row[0]), float(row[1]), float(row[2]))
            indices = lookup.get(key)
            if not indices:
                continue
            for idx in indices:
                if name not in candidate_parts[idx]:
                    candidate_parts[idx].append(name)

    for idx, names in enumerate(candidate_parts):
        if not names:
            continue
        if len(names) == 1:
            picked = names[0]
        else:
            point = points_body[idx]
            best_name = names[0]
            best_dist = np.inf
            for name in names:
                part_box = parts.get(name)
                if part_box is None or not getattr(part_box, "valid", False):
                    continue
                center_xyz = np.asarray(getattr(part_box, "center_xyz"), dtype=np.float64)
                dist = float(np.linalg.norm(point - center_xyz))
                if dist < best_dist:
                    best_dist = dist
                    best_name = name
            picked = best_name
        colors[idx] = np.array(PART_COLORS.get(picked, (0.95, 0.95, 0.95)), dtype=np.float64)

    fallback_boxes: List[Tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
    for name in DEFAULT_PART_ORDER:
        part_box = parts.get(name)
        if part_box is None or not getattr(part_box, "valid", False):
            continue
        min_xyz = np.asarray(getattr(part_box, "min_xyz"), dtype=np.float64)
        max_xyz = np.asarray(getattr(part_box, "max_xyz"), dtype=np.float64)
        center_xyz = np.asarray(getattr(part_box, "center_xyz"), dtype=np.float64)
        fallback_boxes.append((name, min_xyz, max_xyz, center_xyz))

    if fallback_boxes:
        for idx, names in enumerate(candidate_parts):
            if names:
                continue
            point = points_body[idx]
            best_name: str | None = None
            best_score = np.inf
            for name, min_xyz, max_xyz, center_xyz in fallback_boxes:
                outside = np.maximum(np.maximum(min_xyz - point, point - max_xyz), 0.0)
                dist_out = float(np.linalg.norm(outside))
                if dist_out > fallback_max_distance:
                    continue
                center_dist = float(np.linalg.norm(point - center_xyz))
                size = float(np.linalg.norm(max_xyz - min_xyz))
                score = dist_out + 0.04 * center_dist / max(size, 1e-6)
                if score < best_score:
                    best_score = score
                    best_name = name
            if best_name is not None:
                colors[idx] = np.array(PART_COLORS.get(best_name, (0.95, 0.95, 0.95)), dtype=np.float64)
    return colors


def _write_part_visualization(
    *,
    point_path: Path,
    points_body: np.ndarray,
    body_frame: BodyFrame,
    parts: Mapping[str, Any],
    part_points: Mapping[str, np.ndarray] | None,
    viz_dir: Path,
    write_lines: bool,
) -> Dict[str, Any]:
    viz_dir.mkdir(parents=True, exist_ok=True)
    stem = point_path.stem

    points_world = body_frame.body_to_world(points_body)
    if part_points is not None:
        colors = _assign_colors_from_part_points(points_body, parts, part_points)
    else:
        color_default = np.array([0.60, 0.60, 0.60], dtype=np.float64)
        colors = np.tile(color_default[None, :], (points_body.shape[0], 1))
    best_dist = np.full((points_body.shape[0],), np.inf, dtype=np.float64) if part_points is None else None

    boxes_json: Dict[str, Any] = {}
    wire_points: List[np.ndarray] = []
    wire_colors: List[np.ndarray] = []
    wire_samples_per_edge = 16

    for name, part_box in parts.items():
        if not getattr(part_box, "valid", False):
            continue

        color = np.array(PART_COLORS.get(name, (0.95, 0.95, 0.95)), dtype=np.float64)
        min_xyz = np.asarray(getattr(part_box, "min_xyz"), dtype=np.float64)
        max_xyz = np.asarray(getattr(part_box, "max_xyz"), dtype=np.float64)
        center_xyz = np.asarray(getattr(part_box, "center_xyz"), dtype=np.float64)
        obb_center_xyz = np.asarray(getattr(part_box, "obb_center_xyz", center_xyz), dtype=np.float64)

        if part_points is None:
            assert best_dist is not None
            inside = np.all((points_body >= (min_xyz[None, :] - 1e-9)) & (points_body <= (max_xyz[None, :] + 1e-9)), axis=1)
            if np.any(inside):
                indices = np.flatnonzero(inside)
                dist = np.linalg.norm(points_body[indices] - center_xyz[None, :], axis=1)
                better = dist < best_dist[indices]
                if np.any(better):
                    selected = indices[better]
                    colors[selected] = color[None, :]
                    best_dist[selected] = dist[better]

        corners_body_raw = getattr(part_box, "obb_corners_xyz", None)
        if corners_body_raw is None:
            corners_body = _box_corners(min_xyz, max_xyz)
        else:
            corners_body = np.asarray(corners_body_raw, dtype=np.float64)
            if corners_body.shape != (8, 3):
                corners_body = _box_corners(min_xyz, max_xyz)
        corners_world = body_frame.body_to_world(corners_body)
        boxes_json[name] = {
            "color_rgb": [float(v) for v in color.tolist()],
            "center_world": _body_to_world_point(body_frame, obb_center_xyz),
            "corners_world": [[float(v) for v in row] for row in corners_world.tolist()],
        }

        if write_lines:
            for edge in _BOX_LINES:
                p0 = corners_world[int(edge[0])]
                p1 = corners_world[int(edge[1])]
                sampled = _sample_edge_points(p0, p1, samples=wire_samples_per_edge)
                wire_points.append(sampled)
                wire_colors.append(np.tile(color[None, :], (sampled.shape[0], 1)))

    colored_cloud = o3d.geometry.PointCloud()
    colored_cloud.points = o3d.utility.Vector3dVector(points_world.astype(np.float64))
    colored_cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    colored_path = (viz_dir / f"{stem}_parts_colored.ply").resolve()
    o3d.io.write_point_cloud(str(colored_path), colored_cloud, write_ascii=False)

    boxes_path = (viz_dir / f"{stem}_part_boxes.json").resolve()
    write_json(boxes_path, boxes_json, ascii_flag=False, compact=False)

    artifacts: Dict[str, Any] = {
        "parts_colored_ply": str(colored_path),
        "part_boxes_json": str(boxes_path),
        "lineset_ply": None,
    }

    if write_lines and wire_points:
        lines_path = (viz_dir / f"{stem}_part_boxes_lineset.ply").resolve()
        try:
            wire_cloud = o3d.geometry.PointCloud()
            wire_cloud.points = o3d.utility.Vector3dVector(np.vstack(wire_points).astype(np.float64))
            wire_cloud.colors = o3d.utility.Vector3dVector(np.vstack(wire_colors).astype(np.float64))
            o3d.io.write_point_cloud(str(lines_path), wire_cloud, write_ascii=False)
            artifacts["lineset_ply"] = str(lines_path)
            artifacts["lineset_format"] = "wire_points_ply"
        except Exception as exc:
            artifacts["lineset_error"] = str(exc)

    return artifacts


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run rule-based parser-decoder: "
            "point cloud -> 10 body-part boxes -> relation features -> 15 labels"
        )
    )
    parser.add_argument("--input", type=str, nargs="+", required=True, help="Input .pcd/.ply file(s), directory, or glob(s).")
    parser.add_argument("--input-pattern", type=str, default="human_*.ply", help="Pattern used when --input item is a directory.")
    parser.add_argument("--output-dir", type=str, default="outputs/parser_label", help="Output directory.")
    parser.add_argument("--output-prefix", type=str, default="parser_label_", help="Output filename prefix.")
    parser.add_argument(
        "--part-config",
        type=str,
        default="configs/parser/part_parser_rules.yaml",
        help="Part parser rules YAML.",
    )
    parser.add_argument(
        "--decoder-config",
        type=str,
        default="configs/parser/label_decoder_rules.yaml",
        help="Label decoder rules YAML.",
    )
    parser.add_argument(
        "--head-rules",
        type=str,
        default="configs/feature/label_rules.json",
        help="Head class values JSON for output coercion and evaluation order.",
    )
    parser.add_argument(
        "--gt-jsonl",
        type=str,
        default=None,
        help="Optional GT jsonl for post-inference accuracy report (Frame + step).",
    )
    parser.add_argument(
        "--gt-frame-offset",
        type=str,
        default="auto",
        choices=["auto", "0", "1"],
        help="GT Frame to file index mapping: file_index = Frame + offset.",
    )
    parser.add_argument(
        "--viz-dir",
        type=str,
        default=None,
        help="Optional visualization output directory (colored points + part boxes).",
    )
    parser.add_argument(
        "--viz-write-lines",
        action="store_true",
        help="Also export part box line-set PLY when --viz-dir is set.",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)
    root = repo_root()
    input_paths = collect_input_paths(root, args.input, args.input_pattern)

    part_config_path = resolve_repo_path(root, args.part_config)
    decoder_config_path = resolve_repo_path(root, args.decoder_config)
    head_rules_path = resolve_repo_path(root, args.head_rules)
    output_dir = resolve_repo_path(root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = resolve_repo_path(root, args.viz_dir) if args.viz_dir else None
    if viz_dir is not None:
        viz_dir.mkdir(parents=True, exist_ok=True)

    part_cfg = load_part_parser_config(part_config_path if part_config_path.exists() else None)
    decoder_cfg = load_decoder_config(
        decoder_config_path if decoder_config_path.exists() else None,
        profile="global",
    )
    profile_source = "fixed_global"
    head_specs = load_head_specs(head_rules_path)
    head_class_values = {head.name: list(head.class_values) for head in head_specs}
    head_names = [head.name for head in head_specs]

    print(f"[parser-decoder] samples={len(input_paths)}")
    print(f"[parser-decoder] part_config={part_config_path}")
    print(f"[parser-decoder] decoder_config={decoder_config_path}")
    print(f"[parser-decoder] profile={decoder_cfg.get('selected_profile', 'unknown')}")
    print(f"[parser-decoder] profile_source={profile_source}")

    parser_state: Dict[str, Any] | None = None
    relation_state: Dict[str, Any] | None = None
    decoder_state: Dict[str, Any] | None = None
    prev_body_frame: BodyFrame | None = None
    pred_rows: List[Tuple[int, Dict[str, Any]]] = []
    for point_path in input_paths:
        cloud = read_point_cloud_any(point_path)
        points_world = np.asarray(cloud.points, dtype=np.float64)

        preprocess_cfg = part_cfg.get("preprocess", {}) if isinstance(part_cfg, Mapping) else {}
        points_pre, preprocess_meta = preprocess_human_points(points_world, preprocess_cfg)

        body_frame_cfg = part_cfg.get("body_frame", {}) if isinstance(part_cfg, Mapping) else {}
        if not isinstance(body_frame_cfg, Mapping):
            body_frame_cfg = {}
        body_frame_cfg_run: Dict[str, Any] = dict(body_frame_cfg)
        if prev_body_frame is not None:
            continuity_cfg_raw = body_frame_cfg_run.get("continuity", {})
            continuity_cfg: Dict[str, Any]
            if isinstance(continuity_cfg_raw, Mapping):
                continuity_cfg = dict(continuity_cfg_raw)
            else:
                continuity_cfg = {}
            if bool(continuity_cfg.get("enabled", True)):
                continuity_cfg["prev_axes_world"] = prev_body_frame.axes_world
                body_frame_cfg_run["continuity"] = continuity_cfg
        body_frame, body_meta = estimate_body_frame(points_pre, body_frame_cfg_run)
        prev_body_frame = body_frame
        points_body = body_frame.world_to_body(points_pre)

        parts, parse_meta, parsed_part_points, parser_state = parse_parts_from_points(
            points_body,
            part_cfg,
            prev_state=parser_state,
        )
        relation_obj, relation_state = compute_relation_features(
            parts,
            decoder_cfg,
            prev_state=relation_state,
        )
        feature_map = relation_obj.get("flat_features", {})
        if not isinstance(feature_map, Mapping):
            raise TypeError("Relation extraction must provide flat_features object.")
        labels_raw, decode_debug, decoder_state = decode_labels(
            feature_map,
            decoder_cfg,
            temporal_state=decoder_state,
        )

        labels: Dict[str, Any] = {}
        for head_name in head_names:
            labels[head_name] = _coerce_to_allowed_class(
                labels_raw.get(head_name, 0),
                class_values=head_class_values.get(head_name, []),
                default_label=0,
            )

        part_world_centers = _part_world_centers(parts, body_frame)
        step_vector = [labels.get(head_name, 0) for head_name in head_names]
        visualization = None
        if viz_dir is not None:
            visualization = _write_part_visualization(
                point_path=point_path,
                points_body=points_body,
                body_frame=body_frame,
                parts=parts,
                part_points=parsed_part_points,
                viz_dir=viz_dir,
                write_lines=bool(args.viz_write_lines),
            )

        payload = {
            "meta": {
                "model": "parser_decoder_rules_v2",
                "point_path": str(point_path),
                "part_config": str(part_config_path),
                "decoder_config": str(decoder_config_path),
                "decoder_profile": str(decoder_cfg.get("selected_profile", "unknown")),
                "decoder_profile_source": profile_source,
                "head_rules": str(head_rules_path),
                "norm_scale": relation_obj.get("norm_scale", 1.0),
                "pipeline": {
                    "stage_a_preprocess": True,
                    "stage_b_body_frame": True,
                    "stage_c_constrained_parser": True,
                    "stage_d_part_boxes": True,
                    "stage_e_relation_features": True,
                    "stage_f_rule_decoder": True,
                    "stage_g_temporal_stabilization": bool(
                        isinstance(decoder_cfg, Mapping)
                        and isinstance(decoder_cfg.get("temporal"), Mapping)
                        and decoder_cfg.get("temporal", {}).get("enabled", False)
                    ),
                },
                "visualization": visualization,
            },
            "preprocess_meta": preprocess_meta,
            "body_frame": {
                **body_frame.to_jsonable(),
                "estimation_meta": body_meta,
            },
            "parts_body": {name: parts[name].to_jsonable() for name in sorted(parts.keys())},
            "parts_world_centers": part_world_centers,
            "relations": relation_obj.get("pair_features", {}),
            "derived_features": relation_obj.get("derived_features", {}),
            "feature_map": feature_map,
            "decode_debug": decode_debug,
            "labels": labels,
            "step": step_vector,
            "parse_meta": parse_meta,
        }
        output_name = f"{args.output_prefix}{point_path.stem}.json"
        output_path = output_dir / output_name
        write_json(output_path, payload, ascii_flag=False, compact=False)
        print(f"[parser-decoder] wrote {output_path}")

        key = _numeric_suffix(point_path.stem)
        if key is not None:
            pred_rows.append((key, labels))

    if args.gt_jsonl:
        gt_jsonl_path = resolve_repo_path(root, args.gt_jsonl)
        if not pred_rows:
            raise ValueError("No numeric prediction keys found for GT evaluation.")
        _evaluate_predictions(
            pred_rows=pred_rows,
            gt_jsonl_path=gt_jsonl_path,
            head_names=head_names,
            frame_offset=args.gt_frame_offset,
        )


if __name__ == "__main__":
    main()
