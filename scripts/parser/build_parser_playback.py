# scripts/parser/build_parser_playback.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import open3d as o3d

from scripts.common.io_paths import resolve_repo_path
from scripts.common.path_utils import natural_key_path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _frame_key(path: Path) -> str:
    stem = path.stem
    for suffix in ("_parts_colored", "_raw", "_part_boxes_lineset", "_markers"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _collect_files(base_dir: Path, pattern: str) -> List[Path]:
    files = [path.resolve() for path in base_dir.glob(pattern) if path.is_file() and path.suffix.lower() == ".ply"]
    files.sort(key=natural_key_path)
    return files


def _load_cloud(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    cloud = o3d.io.read_point_cloud(str(path))
    points = np.asarray(cloud.points, dtype=np.float32)
    colors = np.asarray(cloud.colors, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)
    if colors.shape != points.shape:
        colors = np.tile(np.array([[0.6, 0.6, 0.6]], dtype=np.float32), (points.shape[0], 1))
    return points, colors


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build compressed time-series playback file from parser viz PLY frames.",
    )
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing *_parts_colored.ply frames.")
    parser.add_argument("--parts-pattern", type=str, default="*_parts_colored.ply")
    parser.add_argument("--raw-pattern", type=str, default="*_raw.ply")
    parser.add_argument("--lines-pattern", type=str, default="*_part_boxes_lineset.ply")
    parser.add_argument("--markers-pattern", type=str, default="*_markers.ply")
    parser.add_argument("--output", type=str, default="outputs/parser_playback/parser_playback.npz")
    parser.add_argument("--meta-output", type=str, default=None, help="Optional metadata JSON output path.")
    parser.add_argument("--max-frames", type=int, default=0, help="Limit processed frames (0 means all).")
    parser.add_argument("--no-lines", action="store_true", help="Exclude *_part_boxes_lineset.ply points.")
    parser.add_argument("--no-raw", action="store_true", help="Exclude *_raw.ply points.")
    parser.add_argument("--no-markers", action="store_true", help="Exclude *_markers.ply points.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    root = repo_root()
    input_dir = resolve_repo_path(root, args.input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input dir not found: {input_dir}")

    parts_files = _collect_files(input_dir, args.parts_pattern)
    if not parts_files:
        raise ValueError(f"No parts PLY files found in {input_dir} with pattern '{args.parts_pattern}'")

    lines_map: Dict[str, Path] = {}
    if not args.no_lines:
        for line_path in _collect_files(input_dir, args.lines_pattern):
            lines_map[_frame_key(line_path)] = line_path
    raw_map: Dict[str, Path] = {}
    if not args.no_raw:
        for raw_path in _collect_files(input_dir, args.raw_pattern):
            raw_map[_frame_key(raw_path)] = raw_path
    marker_map: Dict[str, Path] = {}
    if not args.no_markers:
        for marker_path in _collect_files(input_dir, args.markers_pattern):
            marker_map[_frame_key(marker_path)] = marker_path

    max_frames = max(0, int(args.max_frames))
    selected = parts_files[:max_frames] if max_frames > 0 else parts_files

    all_points: List[np.ndarray] = []
    all_colors: List[np.ndarray] = []
    frame_offsets: List[int] = [0]
    frame_names: List[str] = []
    frame_part_counts: List[int] = []
    frame_raw_counts: List[int] = []
    frame_line_counts: List[int] = []
    frame_marker_counts: List[int] = []
    source_parts: List[str] = []
    source_raw: List[str] = []
    source_lines: List[str] = []
    source_markers: List[str] = []

    for part_path in selected:
        frame_key = _frame_key(part_path)
        p_pts, p_cols = _load_cloud(part_path)
        merged_pts = p_pts
        merged_cols = p_cols
        frame_part_counts.append(int(p_pts.shape[0]))
        raw_count = 0
        line_count = 0
        marker_count = 0
        raw_path = raw_map.get(frame_key)
        if raw_path is not None:
            r_pts, r_cols = _load_cloud(raw_path)
            if r_pts.shape[0] > 0:
                merged_pts = np.concatenate([merged_pts, r_pts], axis=0)
                merged_cols = np.concatenate([merged_cols, r_cols], axis=0)
                raw_count = int(r_pts.shape[0])
        line_path = lines_map.get(frame_key)
        if line_path is not None:
            l_pts, l_cols = _load_cloud(line_path)
            if l_pts.shape[0] > 0:
                merged_pts = np.concatenate([merged_pts, l_pts], axis=0)
                merged_cols = np.concatenate([merged_cols, l_cols], axis=0)
                line_count = int(l_pts.shape[0])
        marker_path = marker_map.get(frame_key)
        if marker_path is not None:
            m_pts, m_cols = _load_cloud(marker_path)
            if m_pts.shape[0] > 0:
                merged_pts = np.concatenate([merged_pts, m_pts], axis=0)
                merged_cols = np.concatenate([merged_cols, m_cols], axis=0)
                marker_count = int(m_pts.shape[0])
        frame_raw_counts.append(int(raw_count))
        frame_line_counts.append(int(line_count))
        frame_marker_counts.append(int(marker_count))
        all_points.append(merged_pts.astype(np.float32, copy=False))
        all_colors.append(merged_cols.astype(np.float32, copy=False))
        frame_offsets.append(int(frame_offsets[-1] + merged_pts.shape[0]))
        frame_names.append(part_path.stem)
        source_parts.append(str(part_path))
        source_raw.append(str(raw_path) if raw_path is not None else "")
        source_lines.append(str(line_path) if line_path is not None else "")
        source_markers.append(str(marker_path) if marker_path is not None else "")

    if all_points:
        points = np.concatenate(all_points, axis=0)
        colors = np.concatenate(all_colors, axis=0)
    else:
        points = np.zeros((0, 3), dtype=np.float32)
        colors = np.zeros((0, 3), dtype=np.float32)

    output_path = resolve_repo_path(root, args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        points=points,
        colors=colors,
        frame_offsets=np.asarray(frame_offsets, dtype=np.int64),
        frame_names=np.asarray(frame_names, dtype=np.str_),
        frame_part_counts=np.asarray(frame_part_counts, dtype=np.int64),
        frame_raw_counts=np.asarray(frame_raw_counts, dtype=np.int64),
        frame_line_counts=np.asarray(frame_line_counts, dtype=np.int64),
        frame_marker_counts=np.asarray(frame_marker_counts, dtype=np.int64),
        source_parts=np.asarray(source_parts, dtype=np.str_),
        source_raw=np.asarray(source_raw, dtype=np.str_),
        source_lines=np.asarray(source_lines, dtype=np.str_),
        source_markers=np.asarray(source_markers, dtype=np.str_),
    )

    meta_path = resolve_repo_path(root, args.meta_output) if args.meta_output else output_path.with_suffix(".json")
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_payload = {
        "input_dir": str(input_dir),
        "parts_pattern": args.parts_pattern,
        "raw_pattern": args.raw_pattern if not args.no_raw else None,
        "lines_pattern": args.lines_pattern if not args.no_lines else None,
        "markers_pattern": args.markers_pattern if not args.no_markers else None,
        "frames": int(len(frame_names)),
        "total_points": int(points.shape[0]),
        "output_npz": str(output_path),
    }
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(meta_payload, handle, ensure_ascii=False, indent=2)

    print(f"[build-parser-playback] frames={len(frame_names)} total_points={points.shape[0]}")
    print(f"[build-parser-playback] wrote {output_path}")
    print(f"[build-parser-playback] wrote {meta_path}")


if __name__ == "__main__":
    main()
