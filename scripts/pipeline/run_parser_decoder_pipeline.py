from __future__ import annotations

import argparse
import glob
import subprocess
import sys
from pathlib import Path
from typing import List

from scripts.common.io_paths import resolve_repo_path


"""
python -m scripts run-parser-pipeline \
  --extract-person \
  --input "data/real/SQUAT/raw/*.pcd" \
  --extract-bg data/real/SQUAT/raw/SQUAT_bg.pcd \
  --output-dir outputs/SQUAT/parser_label \
  --viz-dir outputs/SQUAT/parser_viz \
  --viz-write-lines \
  --gt-jsonl data/feature/SQUAT_label_GT.jsonl
"""

def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def command_text(cmd: List[str]) -> str:
    return " ".join(cmd)


def run_step(cmd: List[str], dry_run: bool) -> None:
    print(f"$ {command_text(cmd)}")
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def _has_glob_chars(text: str) -> bool:
    return any(ch in text for ch in ("*", "?", "["))


def _resolve_bg_case_insensitive(path: Path) -> Path | None:
    if path.exists() and path.is_file():
        return path.resolve()
    parent = path.parent
    if not parent.exists() or not parent.is_dir():
        return None
    target_name = path.name.lower()
    matches = [candidate.resolve() for candidate in parent.iterdir() if candidate.is_file() and candidate.name.lower() == target_name]
    if len(matches) == 1:
        return matches[0]
    return None


def _candidate_input_dirs(root: Path, inputs: List[str]) -> List[Path]:
    dirs: List[Path] = []
    seen: set[Path] = set()
    for raw in inputs:
        if _has_glob_chars(raw):
            parent = (root / Path(raw).parent).resolve()
            if parent.is_dir() and parent not in seen:
                seen.add(parent)
                dirs.append(parent)
            continue
        path = resolve_repo_path(root, raw)
        if path.is_file():
            parent = path.parent.resolve()
            if parent not in seen:
                seen.add(parent)
                dirs.append(parent)
        elif path.is_dir():
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                dirs.append(resolved)
    return dirs


def _auto_find_bg_pcd(root: Path, inputs: List[str]) -> Path | None:
    candidates: List[Path] = []
    seen: set[Path] = set()
    for base_dir in _candidate_input_dirs(root, inputs):
        for path in base_dir.iterdir():
            if not path.is_file():
                continue
            name_lower = path.name.lower()
            if not name_lower.endswith(".pcd"):
                continue
            if "_bg.pcd" not in name_lower and not name_lower.endswith("bg.pcd"):
                continue
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                candidates.append(resolved)
    if len(candidates) == 1:
        return candidates[0]
    return None


def _expand_inputs_without_bg(root: Path, inputs: List[str], bg_path: Path | None) -> List[str]:
    if bg_path is None:
        return list(inputs)
    bg_resolved = bg_path.resolve()
    filtered: List[str] = []
    seen: set[Path] = set()
    for raw in inputs:
        if _has_glob_chars(raw):
            for match in glob.glob(str(root / raw)):
                candidate = Path(match).resolve()
                if not candidate.is_file():
                    continue
                if candidate.suffix.lower() not in (".pcd", ".ply"):
                    continue
                if candidate == bg_resolved:
                    continue
                if candidate not in seen:
                    seen.add(candidate)
                    filtered.append(str(candidate))
            continue
        candidate = resolve_repo_path(root, raw)
        if candidate.is_file():
            resolved = candidate.resolve()
            if resolved == bg_resolved:
                continue
            filtered.append(str(resolved))
        elif candidate.is_dir():
            filtered.append(str(candidate.resolve()))
        else:
            filtered.append(raw)
    return filtered


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run rule-based parser-decoder pipeline: "
            "point cloud -> 10 body-part boxes -> relation features -> 15 labels"
        )
    )
    parser.add_argument("--input", type=str, nargs="+", required=True, help="Input .pcd/.ply files, directory, or glob.")
    parser.add_argument("--input-pattern", type=str, default="human_*.ply")
    parser.add_argument("--output-dir", type=str, default="outputs/parser_label")
    parser.add_argument("--output-prefix", type=str, default="parser_label_")
    parser.add_argument("--part-config", type=str, default="configs/parser/part_parser_rules.yaml")
    parser.add_argument("--decoder-config", type=str, default="configs/parser/label_decoder_rules.yaml")
    parser.add_argument("--head-rules", type=str, default="configs/feature/label_rules.json")
    parser.add_argument("--gt-jsonl", type=str, default=None)
    parser.add_argument("--gt-frame-offset", type=str, default="auto", choices=["auto", "0", "1"])
    parser.add_argument("--viz-dir", type=str, default=None, help="Optional visualization output directory.")
    parser.add_argument("--viz-write-lines", action="store_true", help="Write part box line-set PLY in visualization output.")
    parser.add_argument(
        "--extract-person",
        action="store_true",
        help="Run scripts extract-person first, then use its output for parser input.",
    )
    parser.add_argument("--extract-bg", type=str, default=None, help="Background point cloud path for extract-person.")
    parser.add_argument("--extract-out-dir", type=str, default="data/interim/parser_human", help="Human cloud output dir for extract-person.")
    parser.add_argument("--extract-output-pattern", type=str, default="human_*.ply", help="Parser input pattern after extract-person.")
    parser.add_argument("--extract-max-range", type=float, default=4.0)
    parser.add_argument("--extract-bg-thresh", type=float, default=0.05)
    parser.add_argument("--extract-post-bg-thresh", type=float, default=0.0)
    parser.add_argument("--extract-cluster-eps", type=float, default=0.012)
    parser.add_argument("--extract-cluster-min-points", type=int, default=10)
    parser.add_argument("--extract-person-xy-max", type=float, default=1.10)
    parser.add_argument("--extract-person-area-max", type=float, default=0.90)
    parser.add_argument("--extract-person-planarity-max", type=float, default=0.80)
    parser.add_argument("--extract-vertical-plane-dist", type=float, default=0.01)
    parser.add_argument("--extract-vertical-plane-min-inliers", type=int, default=300)
    parser.add_argument("--extract-bg-icp-voxel", type=float, default=0.06)
    parser.add_argument("--extract-bg-icp-max-corr", type=float, default=0.12)
    parser.add_argument("--extract-y-min", type=float, default=-0.9)
    parser.add_argument("--extract-y-max", type=float, default=0.6)
    parser.add_argument("--extract-z-min", type=float, default=-0.73)
    parser.add_argument("--extract-debug-dir", type=str, default="data/interim/debug_extract_person")
    parser.add_argument("--extract-bg-icp", action="store_true", help="Enable BG ICP in extract-person.")
    parser.add_argument("--extract-no-pcd-transform", action="store_true", help="Disable transform stage in extract-person.")
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)
    root = repo_root()

    parser_inputs = list(args.input)
    parser_input_pattern = args.input_pattern
    if args.extract_person:
        extract_out_dir = resolve_repo_path(root, args.extract_out_dir)
        bg_path = None
        if args.extract_bg:
            requested = resolve_repo_path(root, args.extract_bg)
            bg_path = _resolve_bg_case_insensitive(requested)
            if bg_path is None:
                raise ValueError(
                    f"Background file not found: {requested} "
                    "(case mismatch possible)."
                )
        else:
            bg_path = _auto_find_bg_pcd(root, args.input)

        extract_inputs = _expand_inputs_without_bg(root, args.input, bg_path)
        extract_cmd = [
            args.python,
            "-m",
            "scripts",
            "extract-person",
            "--in",
            *extract_inputs,
            "--out_dir",
            str(extract_out_dir),
            "--output_mode",
            "human_index",
            "--output_prefix",
            "human_",
            "--output_start",
            "1",
            "--output_digits",
            "3",
            "--max_range",
            str(float(args.extract_max_range)),
            "--bg_thresh",
            str(float(args.extract_bg_thresh)),
            "--post_bg_thresh",
            str(float(args.extract_post_bg_thresh)),
            "--cluster_eps",
            str(float(args.extract_cluster_eps)),
            "--cluster_min_points",
            str(int(args.extract_cluster_min_points)),
            "--person_xy_max",
            str(float(args.extract_person_xy_max)),
            "--person_area_max",
            str(float(args.extract_person_area_max)),
            "--person_planarity_max",
            str(float(args.extract_person_planarity_max)),
            "--vertical_plane_dist",
            str(float(args.extract_vertical_plane_dist)),
            "--vertical_plane_min_inliers",
            str(int(args.extract_vertical_plane_min_inliers)),
            "--bg_icp_voxel",
            str(float(args.extract_bg_icp_voxel)),
            "--bg_icp_max_corr",
            str(float(args.extract_bg_icp_max_corr)),
            "--y_min",
            str(float(args.extract_y_min)),
            "--y_max",
            str(float(args.extract_y_max)),
            "--z_min",
            str(float(args.extract_z_min)),
        ]
        if bg_path is not None:
            extract_cmd.extend(["--bg", str(bg_path)])
        if args.extract_bg_icp:
            extract_cmd.append("--bg_icp")
        if args.extract_debug_dir:
            extract_cmd.extend(["--debug_dir", str(resolve_repo_path(root, args.extract_debug_dir))])
        if args.extract_no_pcd_transform:
            extract_cmd.append("--no_pcd_transform")
        run_step(extract_cmd, dry_run=bool(args.dry_run))
        parser_inputs = [str(extract_out_dir)]
        parser_input_pattern = args.extract_output_pattern

    cmd = [
        args.python,
        "-m",
        "scripts",
        "run-parser-decoder",
        "--input",
        *parser_inputs,
        "--input-pattern",
        parser_input_pattern,
        "--output-dir",
        str(resolve_repo_path(root, args.output_dir)),
        "--output-prefix",
        args.output_prefix,
        "--part-config",
        str(resolve_repo_path(root, args.part_config)),
        "--decoder-config",
        str(resolve_repo_path(root, args.decoder_config)),
        "--head-rules",
        str(resolve_repo_path(root, args.head_rules)),
        "--gt-frame-offset",
        args.gt_frame_offset,
    ]
    if args.viz_dir:
        cmd.extend(["--viz-dir", str(resolve_repo_path(root, args.viz_dir))])
    if args.viz_write_lines:
        cmd.append("--viz-write-lines")
    if args.gt_jsonl:
        cmd.extend(["--gt-jsonl", str(resolve_repo_path(root, args.gt_jsonl))])

    print(f"[parser-pipeline] root={root}")
    run_step(cmd, dry_run=bool(args.dry_run))
    print("[parser-pipeline] done")


if __name__ == "__main__":
    main()
