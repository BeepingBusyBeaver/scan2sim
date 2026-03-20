# scripts/pipeline/run_parser_decoder_pipeline.py
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from scripts.common.io_paths import resolve_repo_path


"""
python -m scripts run-parser-pipeline \
  --extract-person \
  --input "data/real/RMmotion/raw/*.pcd" \
  --extract-bg data/real/RMmotion/raw/RMmotion_bg.pcd \
  --output-dir outputs/RMmotion/parser_label \
  --viz-dir outputs/RMmotion/parser_viz \
  --viz-write-lines \
  --build-playback \
  --playback-output outputs/RMmotion/parser_playback.npz \
  --no-playback-open-windows \
&& WAYLAND_DISPLAY= DISPLAY=:0 XDG_SESSION_TYPE=x11 \
python -m scripts play-parser-playback \
  --input outputs/RMmotion/parser_playback.npz \
  --fps 2 \
  --loop \
  --log-every 30

hm      1 6 10 14 17 21 24 27 31 1
frame   0 5 9 13 16 20 23 26 30 0


python -m scripts run-parser-pipeline \
  --input "data/virtual/pdbr_25421_lidar/*.ply" \
  --input-pattern "*.ply" \
  --part-config configs/parser/part_parser_rules_virtual.yaml \
  --output-dir outputs/virtual_pdbr/parser_label \
  --viz-dir outputs/virtual_pdbr/parser_viz \
  --viz-write-lines \
  --sequence-mode regex \
  --sequence-regex "[^/]+\\.ply$" \
  --no-track-parts \
  --build-playback \
  --playback-output outputs/virtual_pdbr/parser_playback.npz \
  --no-playback-open-windows \
&& WAYLAND_DISPLAY= DISPLAY=:0 XDG_SESSION_TYPE=x11 \
python -m scripts play-parser-playback \
  --input outputs/virtual_pdbr/parser_playback.npz \
  --fps 2 \
  --loop \
  --log-every 30


  --viz-write-lines
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


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _median(values: List[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2 == 1:
        return float(ordered[mid])
    return float(0.5 * (ordered[mid - 1] + ordered[mid]))


def _box_volume(part: Dict[str, Any]) -> float:
    size = part.get("size_xyz")
    if not isinstance(size, list) or len(size) != 3:
        return 0.0
    sx = max(_safe_float(size[0]), 0.0)
    sy = max(_safe_float(size[1]), 0.0)
    sz = max(_safe_float(size[2]), 0.0)
    return float(sx * sy * sz)


def _axis_gap(child: Dict[str, Any], parent: Dict[str, Any]) -> float:
    child_min = child.get("min_xyz")
    child_max = child.get("max_xyz")
    parent_min = parent.get("min_xyz")
    parent_max = parent.get("max_xyz")
    if not (
        isinstance(child_min, list)
        and isinstance(child_max, list)
        and isinstance(parent_min, list)
        and isinstance(parent_max, list)
        and len(child_min) == len(child_max) == len(parent_min) == len(parent_max) == 3
    ):
        return 0.0
    sep = []
    for axis in range(3):
        a_min = _safe_float(child_min[axis])
        a_max = _safe_float(child_max[axis])
        b_min = _safe_float(parent_min[axis])
        b_max = _safe_float(parent_max[axis])
        d1 = max(a_min - b_max, 0.0)
        d2 = max(b_min - a_max, 0.0)
        sep.append(max(d1, d2))
    return float((sep[0] ** 2 + sep[1] ** 2 + sep[2] ** 2) ** 0.5)


def _collect_parser_report(output_dir: Path, output_prefix: str, since_ts: float) -> Dict[str, Any]:
    pattern = f"{output_prefix}*.json"
    all_files = sorted(path for path in output_dir.glob(pattern) if path.is_file())
    recent_files = [path for path in all_files if path.stat().st_mtime >= (since_ts - 0.2)]
    files = recent_files if recent_files else all_files

    stage_b_ortho: List[float] = []
    stage_b_up: List[float] = []
    stage_b_flip_frames = 0

    stage_c_valid_parts: List[float] = []
    stage_c_critical_miss = 0
    stage_c_left_right_arm_gap: List[float] = []
    stage_c_left_right_leg_gap: List[float] = []

    stage_d_parent_gap: List[float] = []
    stage_d_limb_torso_ratio: List[float] = []

    ref_enabled = 0
    ref_applied = 0
    ref_reverted_parts: List[float] = []
    ref_reason_counts: Dict[str, int] = {}
    ref_score_delta: List[float] = []

    parent_links = [
        ("left_lower_arm", "left_upper_arm"),
        ("right_lower_arm", "right_upper_arm"),
        ("left_upper_arm", "torso"),
        ("right_upper_arm", "torso"),
        ("left_shin", "left_thigh"),
        ("right_shin", "right_thigh"),
        ("left_thigh", "torso"),
        ("right_thigh", "torso"),
        ("head", "torso"),
    ]
    critical_parts = ["torso", "left_upper_arm", "right_upper_arm", "left_thigh", "right_thigh"]
    limb_parts = ["left_upper_arm", "left_lower_arm", "right_upper_arm", "right_lower_arm", "left_thigh", "left_shin", "right_thigh", "right_shin"]

    for path in files:
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception:
            continue

        body_frame = payload.get("body_frame", {})
        estimation_meta = body_frame.get("estimation_meta", {}) if isinstance(body_frame, dict) else {}
        stage_b_ortho.append(abs(_safe_float(estimation_meta.get("orthogonality_error"), 0.0)))
        stage_b_up.append(abs(_safe_float(estimation_meta.get("up_alignment_cosine"), 0.0)))
        flips = estimation_meta.get("continuity_axis_flips", {})
        if isinstance(flips, dict) and any(bool(flips.get(k, False)) for k in ("x", "y", "z")):
            stage_b_flip_frames += 1

        parts = payload.get("parts_body", {})
        if not isinstance(parts, dict):
            parts = {}
        valid_parts = [name for name, part in parts.items() if isinstance(part, dict) and bool(part.get("valid", False))]
        stage_c_valid_parts.append(float(len(valid_parts)))
        if any(not bool(parts.get(name, {}).get("valid", False)) for name in critical_parts):
            stage_c_critical_miss += 1

        left_arm = _safe_float(parts.get("left_upper_arm", {}).get("num_points"), 0.0) + _safe_float(parts.get("left_lower_arm", {}).get("num_points"), 0.0)
        right_arm = _safe_float(parts.get("right_upper_arm", {}).get("num_points"), 0.0) + _safe_float(parts.get("right_lower_arm", {}).get("num_points"), 0.0)
        left_leg = _safe_float(parts.get("left_thigh", {}).get("num_points"), 0.0) + _safe_float(parts.get("left_shin", {}).get("num_points"), 0.0)
        right_leg = _safe_float(parts.get("right_thigh", {}).get("num_points"), 0.0) + _safe_float(parts.get("right_shin", {}).get("num_points"), 0.0)
        stage_c_left_right_arm_gap.append(abs(left_arm - right_arm))
        stage_c_left_right_leg_gap.append(abs(left_leg - right_leg))

        for child_name, parent_name in parent_links:
            child = parts.get(child_name)
            parent = parts.get(parent_name)
            if not isinstance(child, dict) or not isinstance(parent, dict):
                continue
            if not bool(child.get("valid", False)) or not bool(parent.get("valid", False)):
                continue
            stage_d_parent_gap.append(_axis_gap(child, parent))

        torso = parts.get("torso", {})
        torso_vol = _box_volume(torso if isinstance(torso, dict) else {})
        if torso_vol > 1e-9:
            for limb_name in limb_parts:
                part = parts.get(limb_name)
                if not isinstance(part, dict) or (not bool(part.get("valid", False))):
                    continue
                stage_d_limb_torso_ratio.append(_box_volume(part) / torso_vol)

        parse_meta = payload.get("parse_meta", {})
        refiner = parse_meta.get("refiner", {}) if isinstance(parse_meta, dict) else {}
        if isinstance(refiner, dict):
            if bool(refiner.get("enabled", False)):
                ref_enabled += 1
            if bool(refiner.get("applied", False)):
                ref_applied += 1
            if "guard_reverted_parts" in refiner:
                ref_reverted_parts.append(_safe_float(refiner.get("guard_reverted_parts"), 0.0))
            if "score_before" in refiner and "score_after" in refiner:
                ref_score_delta.append(_safe_float(refiner.get("score_after")) - _safe_float(refiner.get("score_before")))
            reason = refiner.get("reason")
            if isinstance(reason, str) and reason:
                ref_reason_counts[reason] = int(ref_reason_counts.get(reason, 0) + 1)

    return {
        "files_total": int(len(all_files)),
        "files_recent": int(len(recent_files)),
        "files_used": int(len(files)),
        "stage_b_ortho_mean": _mean(stage_b_ortho),
        "stage_b_up_mean": _mean(stage_b_up),
        "stage_b_flip_frames": int(stage_b_flip_frames),
        "stage_c_valid_parts_mean": _mean(stage_c_valid_parts),
        "stage_c_valid_parts_min": int(min(stage_c_valid_parts)) if stage_c_valid_parts else 0,
        "stage_c_critical_miss": int(stage_c_critical_miss),
        "stage_c_arm_gap_median": _median(stage_c_left_right_arm_gap),
        "stage_c_leg_gap_median": _median(stage_c_left_right_leg_gap),
        "stage_d_parent_gap_mean": _mean(stage_d_parent_gap),
        "stage_d_parent_gap_p95": float(sorted(stage_d_parent_gap)[max(0, int(0.95 * (len(stage_d_parent_gap) - 1)))] if stage_d_parent_gap else 0.0),
        "stage_d_limb_torso_ratio_p95": float(sorted(stage_d_limb_torso_ratio)[max(0, int(0.95 * (len(stage_d_limb_torso_ratio) - 1)))] if stage_d_limb_torso_ratio else 0.0),
        "ref_enabled_frames": int(ref_enabled),
        "ref_applied_frames": int(ref_applied),
        "ref_reverted_parts_mean": _mean(ref_reverted_parts),
        "ref_score_delta_mean": _mean(ref_score_delta),
        "ref_reason_counts": ref_reason_counts,
    }


def _print_parser_report(report: Dict[str, Any]) -> None:
    used = int(report.get("files_used", 0))
    total = int(report.get("files_total", 0))
    recent = int(report.get("files_recent", 0))
    print(
        f"[parser-pipeline][report] files_used={used} "
        f"(recent={recent}, total_in_dir={total})"
    )
    if used <= 0:
        print("[parser-pipeline][report] no parser json files found for report.")
        return
    print(
        "[parser-pipeline][report][B] "
        f"up_align_mean={_safe_float(report.get('stage_b_up_mean')):.4f} "
        f"ortho_err_mean={_safe_float(report.get('stage_b_ortho_mean')):.2e} "
        f"axis_flip_frames={int(report.get('stage_b_flip_frames', 0))}/{used}"
    )
    print(
        "[parser-pipeline][report][C] "
        f"valid_parts_mean={_safe_float(report.get('stage_c_valid_parts_mean')):.2f} "
        f"valid_parts_min={int(report.get('stage_c_valid_parts_min', 0))} "
        f"critical_miss_frames={int(report.get('stage_c_critical_miss', 0))}/{used} "
        f"arm_gap_median_pts={_safe_float(report.get('stage_c_arm_gap_median')):.1f} "
        f"leg_gap_median_pts={_safe_float(report.get('stage_c_leg_gap_median')):.1f}"
    )
    print(
        "[parser-pipeline][report][D] "
        f"parent_gap_mean={_safe_float(report.get('stage_d_parent_gap_mean')):.4f} "
        f"parent_gap_p95={_safe_float(report.get('stage_d_parent_gap_p95')):.4f} "
        f"limb_torso_ratio_p95={_safe_float(report.get('stage_d_limb_torso_ratio_p95')):.3f}"
    )
    reasons = report.get("ref_reason_counts", {})
    reason_text = ", ".join(f"{k}:{v}" for k, v in sorted(reasons.items())) if isinstance(reasons, dict) and reasons else "none"
    print(
        "[parser-pipeline][report][Refiner] "
        f"enabled={int(report.get('ref_enabled_frames', 0))}/{used} "
        f"applied={int(report.get('ref_applied_frames', 0))}/{used} "
        f"reverted_parts_mean={_safe_float(report.get('ref_reverted_parts_mean')):.2f} "
        f"score_delta_mean={_safe_float(report.get('ref_score_delta_mean')):.4f} "
        f"reasons={reason_text}"
    )


def _is_wsl() -> bool:
    if os.environ.get("WSL_INTEROP") or os.environ.get("WSL_DISTRO_NAME"):
        return True
    version_path = Path("/proc/version")
    if version_path.exists():
        text = version_path.read_text(encoding="utf-8", errors="ignore").lower()
        if "microsoft" in text:
            return True
    return False


def _wsl_to_windows_path(path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["wslpath", "-w", str(path.resolve())],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    out = result.stdout.strip()
    return out or None


def _powershell_quote(text: str) -> str:
    return "'" + text.replace("'", "''") + "'"


def _build_windows_playback_launch_cmd(
    *,
    root: Path,
    playback_npz: Path,
    windows_python_cmd: str,
    fps: float,
    loop: bool,
) -> List[str] | None:
    if shutil.which("powershell.exe") is None:
        return None
    root_win = _wsl_to_windows_path(root)
    playback_win = _wsl_to_windows_path(playback_npz)
    if not root_win or not playback_win:
        return None

    play_cmd = (
        f"{windows_python_cmd} -m scripts play-parser-playback "
        f"--input {_powershell_quote(playback_win)} --fps {float(fps):.6f}"
    )
    if loop:
        play_cmd += " --loop"
    ps_script = (
        f"Set-Location -LiteralPath {_powershell_quote(root_win)}; "
        f"{play_cmd}"
    )
    return [
        "powershell.exe",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-Command",
        ps_script,
    ]


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
    parser.add_argument(
        "--decoder-profile",
        type=str,
        default="global",
        help="Decoder profile name in decoder-config.",
    )
    parser.add_argument("--head-rules", type=str, default="configs/feature/label_rules.json")
    parser.add_argument("--gt-jsonl", type=str, default=None)
    parser.add_argument("--gt-frame-offset", type=str, default="auto", choices=["auto", "0", "1"])
    parser.add_argument(
        "--sequence-mode",
        type=str,
        default="parent",
        choices=["auto", "single", "parent", "prefix", "regex"],
        help="How to split parser stream into temporal sequences. Default assumes one continuous sequence per parent dir.",
    )
    parser.add_argument("--sequence-regex", type=str, default=None, help="Regex for --sequence-mode regex.")
    parser.add_argument(
        "--sequence-gap-reset",
        type=int,
        default=1,
        help="Reset temporal state on numeric suffix discontinuity (>N). Default=1 for continuous time-series inputs.",
    )
    parser.add_argument("--track-max-missed", type=int, default=3)
    parser.add_argument("--track-velocity-alpha", type=float, default=0.45)
    parser.add_argument("--track-parts", dest="track_parts", action="store_true", help="Enable semantic part tracker.")
    parser.add_argument("--no-track-parts", dest="track_parts", action="store_false", help="Disable semantic part tracker.")
    parser.add_argument("--viz-dir", type=str, default=None, help="Optional visualization output directory.")
    parser.add_argument("--viz-write-lines", action="store_true", help="Write part box line-set PLY in visualization output.")
    parser.add_argument(
        "--build-playback",
        action="store_true",
        help="After parser run, build compressed playback NPZ from viz PLY sequence.",
    )
    parser.add_argument(
        "--playback-output",
        type=str,
        default="outputs/parser_playback/parser_playback.npz",
        help="Output NPZ path used when --build-playback is enabled.",
    )
    parser.add_argument(
        "--playback-no-lines",
        action="store_true",
        help="Exclude *_part_boxes_lineset.ply when building playback NPZ.",
    )
    parser.add_argument(
        "--playback-open-windows",
        dest="playback_open_windows",
        action="store_true",
        help="After --build-playback, launch Windows GUI playback via powershell.exe.",
    )
    parser.add_argument(
        "--no-playback-open-windows",
        dest="playback_open_windows",
        action="store_false",
        help="Disable auto launch of Windows GUI playback.",
    )
    parser.add_argument(
        "--windows-python-cmd",
        type=str,
        default="py -3.11",
        help="Windows-side python launcher command used by --playback-open-windows.",
    )
    parser.add_argument(
        "--playback-open-fps",
        type=float,
        default=10.0,
        help="FPS used when auto-launching Windows playback.",
    )
    parser.add_argument(
        "--playback-open-loop",
        dest="playback_open_loop",
        action="store_true",
        help="Use --loop for auto-launched Windows playback.",
    )
    parser.add_argument(
        "--no-playback-open-loop",
        dest="playback_open_loop",
        action="store_false",
        help="Disable --loop in auto-launched Windows playback.",
    )
    parser.add_argument(
        "--extract-person",
        action="store_true",
        help="Run scripts extract-person first, then use its output for parser input.",
    )
    parser.add_argument("--extract-bg", type=str, default=None, help="Background point cloud path for extract-person.")
    parser.add_argument("--extract-out-dir", type=str, default="data/interim/parser_human", help="Human cloud output dir for extract-person.")
    parser.add_argument("--extract-output-pattern", type=str, default="human_*.ply", help="Parser input pattern after extract-person.")
    parser.add_argument("--extract-max-range", type=float, default=5.0)
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
    parser.add_argument("--extract-bg-icp", dest="extract_bg_icp", action="store_true", help="Enable BG ICP in extract-person.")
    parser.add_argument("--no-extract-bg-icp", dest="extract_bg_icp", action="store_false", help="Disable BG ICP in extract-person.")
    parser.add_argument("--extract-no-pcd-transform", action="store_true", help="Disable transform stage in extract-person.")
    parser.add_argument(
        "--report-checks",
        dest="report_checks",
        action="store_true",
        help="Print staged parser quality report (B/C/D + refiner) after run-parser-decoder.",
    )
    parser.add_argument(
        "--no-report-checks",
        dest="report_checks",
        action="store_false",
        help="Disable staged parser quality report.",
    )
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    parser.set_defaults(extract_bg_icp=True)
    parser.set_defaults(track_parts=True)
    parser.set_defaults(playback_open_windows=True)
    parser.set_defaults(playback_open_loop=True)
    parser.set_defaults(report_checks=True)
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
        "--decoder-profile",
        str(args.decoder_profile),
        "--head-rules",
        str(resolve_repo_path(root, args.head_rules)),
        "--gt-frame-offset",
        args.gt_frame_offset,
        "--sequence-mode",
        args.sequence_mode,
        "--sequence-gap-reset",
        str(int(args.sequence_gap_reset)),
        "--track-max-missed",
        str(int(args.track_max_missed)),
        "--track-velocity-alpha",
        str(float(args.track_velocity_alpha)),
    ]
    if args.sequence_regex:
        cmd.extend(["--sequence-regex", args.sequence_regex])
    if args.track_parts:
        cmd.append("--track-parts")
    else:
        cmd.append("--no-track-parts")
    if args.viz_dir:
        cmd.extend(["--viz-dir", str(resolve_repo_path(root, args.viz_dir))])
    if args.viz_write_lines:
        cmd.append("--viz-write-lines")
    if args.gt_jsonl:
        cmd.extend(["--gt-jsonl", str(resolve_repo_path(root, args.gt_jsonl))])

    print(f"[parser-pipeline] root={root}")
    parser_output_dir = resolve_repo_path(root, args.output_dir)
    run_started_ts = time.time()
    run_step(cmd, dry_run=bool(args.dry_run))
    if args.report_checks and (not bool(args.dry_run)):
        report = _collect_parser_report(
            output_dir=parser_output_dir,
            output_prefix=str(args.output_prefix),
            since_ts=run_started_ts,
        )
        _print_parser_report(report)
    if args.build_playback:
        if not args.viz_dir:
            raise ValueError("--build-playback requires --viz-dir (source of *_parts_colored.ply).")
        playback_output = resolve_repo_path(root, args.playback_output)
        playback_cmd = [
            args.python,
            "-m",
            "scripts",
            "build-parser-playback",
            "--input-dir",
            str(resolve_repo_path(root, args.viz_dir)),
            "--output",
            str(playback_output),
        ]
        if args.playback_no_lines:
            playback_cmd.append("--no-lines")
        run_step(playback_cmd, dry_run=bool(args.dry_run))
        if args.playback_open_windows:
            if not _is_wsl():
                print("[parser-pipeline] skip Windows playback launch: not running in WSL.")
            else:
                launch_cmd = _build_windows_playback_launch_cmd(
                    root=root,
                    playback_npz=playback_output,
                    windows_python_cmd=str(args.windows_python_cmd),
                    fps=float(args.playback_open_fps),
                    loop=bool(args.playback_open_loop),
                )
                if launch_cmd is None:
                    print("[parser-pipeline] skip Windows playback launch: powershell/wslpath unavailable.")
                else:
                    run_step(launch_cmd, dry_run=bool(args.dry_run))
    print("[parser-pipeline] done")


if __name__ == "__main__":
    main()
