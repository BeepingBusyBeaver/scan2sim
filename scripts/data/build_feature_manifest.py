# scripts/data/build_feature_manifest.py
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np

from scripts.common.io_paths import resolve_repo_path
from scripts.common.json_utils import read_json
from scripts.common.path_utils import natural_key_path
from scripts.feature.heads import load_head_specs


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _key_from_stem(stem: str) -> str:
    match = re.search(r"(\d+)$", stem)
    if match:
        return match.group(1)
    return stem


def _collect_map(base_dir: Path, pattern: str, suffix: str, *, required: bool = True) -> Dict[str, Path]:
    paths = sorted(
        [path for path in base_dir.glob(pattern) if path.is_file() and path.suffix.lower() == suffix],
        key=natural_key_path,
    )
    if not paths and required:
        raise ValueError(f"No files in {base_dir} with pattern='{pattern}' and suffix='{suffix}'")
    out: Dict[str, Path] = {}
    for path in paths:
        key = _key_from_stem(path.stem)
        if key in out:
            raise ValueError(f"Duplicate key '{key}' for files: {out[key].name}, {path.name}")
        out[key] = path.resolve()
    return out


def _sorted_keys(keys: Iterable[str]) -> List[str]:
    return sorted(
        keys,
        key=lambda text: (0, int(text)) if text.isdigit() else (1, text),
    )


def _load_jsonl(path: Path) -> List[dict[str, Any]]:
    rows: List[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_num, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise TypeError(f"JSONL row must be object: {path}:{line_num}")
            rows.append(obj)
    return rows


def _build_numeric_key_map(point_map: Dict[str, Path]) -> Dict[int, str]:
    out: Dict[int, str] = {}
    for key in point_map:
        if not key.isdigit():
            continue
        numeric = int(key)
        if numeric in out and out[numeric] != key:
            raise ValueError(f"Ambiguous numeric key in point files: '{out[numeric]}' vs '{key}'")
        out[numeric] = key
    return out


def _parse_step_vector(raw_step: Any) -> List[float]:
    if isinstance(raw_step, str):
        text = raw_step.strip()
        if text.startswith("{") and text.endswith("}"):
            text = text[1:-1]
        if not text:
            return []
        return [float(token.strip()) for token in text.split(",") if token.strip()]
    if isinstance(raw_step, (list, tuple)):
        return [float(value) for value in raw_step]
    raise TypeError(f"Unsupported step vector type: {type(raw_step).__name__}")


def _step_vector_to_labels(step_values: List[float], required_heads: List[str], *, source: str) -> Dict[str, Any]:
    expected = len(required_heads)
    if len(step_values) != expected:
        raise ValueError(f"`step` length mismatch in {source}: expected {expected}, got {len(step_values)}")
    out: Dict[str, Any] = {}
    for index, head_name in enumerate(required_heads):
        value = float(step_values[index])
        out[head_name] = int(value) if float(value).is_integer() else value
    return out


def _labels_from_jsonl(points_map: Dict[str, Path], jsonl_path: Path) -> Dict[str, Dict[str, Any]]:
    numeric_key_map = _build_numeric_key_map(points_map)
    if not numeric_key_map:
        raise ValueError("Point file keys are not numeric. JSONL label matching requires numeric point suffixes.")
    rows = _load_jsonl(jsonl_path)
    frames: List[int] = []
    for index, row in enumerate(rows):
        frame_raw = row.get("Frame")
        if frame_raw is None:
            raise KeyError(f"JSONL row[{index}] missing `Frame`: {jsonl_path}")
        frames.append(int(frame_raw))

    point_numeric_keys = set(numeric_key_map.keys())
    offset_candidates = [0, 1]
    best_offset = 0
    best_overlap = -1
    for offset in offset_candidates:
        overlap = sum(1 for frame_idx in frames if (frame_idx + offset) in point_numeric_keys)
        if overlap > best_overlap:
            best_overlap = overlap
            best_offset = offset
    if best_overlap <= 0:
        raise ValueError(f"Could not infer Frame->point index offset for {jsonl_path.name}")

    row_by_point_key: Dict[str, Dict[str, Any]] = {}
    for frame_idx, row in zip(frames, rows):
        selected_numeric = frame_idx + best_offset
        if selected_numeric not in numeric_key_map:
            raise KeyError(
                f"Could not match Frame={frame_idx} with inferred offset={best_offset} "
                f"to point keys in {jsonl_path.name}"
            )
        selected_key = numeric_key_map[selected_numeric]
        if selected_key in row_by_point_key:
            raise ValueError(
                f"Duplicate mapped label for point key '{selected_key}' "
                f"(Frame={frame_idx}) in {jsonl_path.name}"
            )
        row_by_point_key[selected_key] = row
    return row_by_point_key


def _as_repo_path_text(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return str(path.resolve())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build point-cloud + multi-head label manifest JSONL.")
    parser.add_argument("--points-dir", type=str, required=True, help="Directory containing .ply/.pcd point clouds.")
    parser.add_argument("--point-pattern", type=str, default="human_*.ply", help="Point cloud filename pattern.")
    parser.add_argument("--labels-dir", type=str, default=None, help="Directory containing label JSON files.")
    parser.add_argument("--label-pattern", type=str, default="*.json", help="Label filename pattern.")
    parser.add_argument(
        "--labels-jsonl",
        type=str,
        default=None,
        help="Single JSONL label file. Supports `Frame` + `step` vector format.",
    )
    parser.add_argument(
        "--head-rules",
        type=str,
        default="configs/feature/label_rules.json",
        help="Head rules JSON. Used to validate required head keys.",
    )
    parser.add_argument("--output", type=str, default="data/feature/manifest.jsonl", help="Output manifest JSONL path.")
    parser.add_argument(
        "--split-mode",
        type=str,
        default="random",
        choices=["none", "random"],
        help="'none': all train, 'random': split by val-ratio.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation ratio for random split.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = repo_root()
    points_dir = resolve_repo_path(root, args.points_dir)
    labels_dir = resolve_repo_path(root, args.labels_dir) if args.labels_dir else None
    labels_jsonl_path = resolve_repo_path(root, args.labels_jsonl) if args.labels_jsonl else None
    output_path = resolve_repo_path(root, args.output)
    head_rules_path = resolve_repo_path(root, args.head_rules)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if labels_dir is None and labels_jsonl_path is None:
        raise ValueError("One of --labels-dir or --labels-jsonl is required.")

    head_specs = load_head_specs(head_rules_path)
    required_heads = [head.name for head in head_specs]

    point_map = _collect_map(points_dir, args.point_pattern, ".ply", required=False)
    if not point_map:
        point_map = _collect_map(points_dir, args.point_pattern, ".pcd")

    label_file_map: Dict[str, Path] = {}
    label_jsonl_row_map: Dict[str, Dict[str, Any]] = {}
    if labels_jsonl_path is not None:
        label_jsonl_row_map = _labels_from_jsonl(point_map, labels_jsonl_path)
    elif labels_dir is not None:
        label_file_map = _collect_map(labels_dir, args.label_pattern, ".json")

    label_keys = set(label_jsonl_row_map.keys()) if label_jsonl_row_map else set(label_file_map.keys())
    common_keys = _sorted_keys(set(point_map.keys()) & label_keys)
    if not common_keys:
        raise ValueError("No matched point/label pairs by numeric suffix.")

    split_by_key: Dict[str, str] = {}
    if args.split_mode == "random":
        if not (0.0 < float(args.val_ratio) < 1.0):
            raise ValueError("--val-ratio must be in (0,1) for random split.")
        rng = np.random.default_rng(int(args.seed))
        perm = np.array(common_keys, dtype=object)
        rng.shuffle(perm)
        val_count = max(1, int(round(len(common_keys) * float(args.val_ratio))))
        val_keys = set(perm[:val_count].tolist())
        for key in common_keys:
            split_by_key[key] = "val" if key in val_keys else "train"
    else:
        split_by_key = {key: "train" for key in common_keys}

    lines: List[str] = []
    for key in common_keys:
        row_labels: Dict[str, Any]
        if label_jsonl_row_map:
            label_row = label_jsonl_row_map[key]
            step_values = _parse_step_vector(label_row.get("step"))
            row_labels = _step_vector_to_labels(
                step_values,
                required_heads,
                source=f"{labels_jsonl_path}:{key}",
            )
        else:
            label_obj = read_json(label_file_map[key])
            if not isinstance(label_obj, dict):
                raise TypeError(f"Label JSON must be object: {label_file_map[key]}")
            labels = label_obj.get("step")
            if not isinstance(labels, dict):
                raise KeyError(f"Label JSON missing `step` object: {label_file_map[key]}")
            row_labels = {}
            for head_name in required_heads:
                if head_name not in labels:
                    raise KeyError(f"Label JSON '{label_file_map[key].name}' missing head '{head_name}'.")
                row_labels[head_name] = labels[head_name]

        row = {
            "sample_id": key,
            "point_path": _as_repo_path_text(root, point_map[key]),
            "labels": row_labels,
            "split": split_by_key[key],
        }
        lines.append(json.dumps(row, ensure_ascii=False))

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    label_count = len(label_jsonl_row_map) if label_jsonl_row_map else len(label_file_map)
    print(f"[build-manifest] points={len(point_map)} labels={label_count} paired={len(common_keys)}")
    if label_jsonl_row_map:
        print(f"[build-manifest] labels_jsonl={labels_jsonl_path}")
    elif labels_dir is not None:
        print(f"[build-manifest] labels_dir={labels_dir}")
    print(f"[build-manifest] wrote {output_path}")


if __name__ == "__main__":
    main()
