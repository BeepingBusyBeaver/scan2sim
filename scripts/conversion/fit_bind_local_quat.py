# scripts/conversion/fit_bind_local_quat.py
import argparse
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from scripts.common.json_utils import read_json, write_json
from scripts.common.livehps_defaults import DEFAULT_UNITY_BIND_LOCAL_QUAT_JSON
from scripts.common.path_utils import natural_key_path
from scripts.common.io_paths import resolve_repo_path


DEFAULT_SMPL_24_JOINT_NAMES = [
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hand", "right_hand",
]


def parse_joint_names(joint_names_arg: str | None) -> List[str]:
    if joint_names_arg is None:
        return list(DEFAULT_SMPL_24_JOINT_NAMES)
    names = [x.strip() for x in joint_names_arg.split(",") if x.strip()]
    if len(names) != 24:
        raise ValueError(f"--joint-names must contain exactly 24 names, got {len(names)}")
    return names


def _normalize_quat_xyzw(q: np.ndarray) -> np.ndarray:
    qq = np.asarray(q, dtype=np.float64).reshape(4)
    n = float(np.linalg.norm(qq))
    if n <= 1e-12:
        raise ValueError("Zero-norm quaternion encountered.")
    return (qq / n).astype(np.float64)


def _canonicalize_w_positive(q: np.ndarray) -> np.ndarray:
    qq = _normalize_quat_xyzw(q)
    if qq[3] < 0.0:
        qq = -qq
    return qq


def _quat_inverse_xyzw(q: np.ndarray) -> np.ndarray:
    qq = _normalize_quat_xyzw(q)
    return np.array([-qq[0], -qq[1], -qq[2], qq[3]], dtype=np.float64)


def _quat_multiply_xyzw(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = _normalize_quat_xyzw(q1)
    x2, y2, z2, w2 = _normalize_quat_xyzw(q2)
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    return _normalize_quat_xyzw(np.array([x, y, z, w], dtype=np.float64))


def _quat_angle_deg(q1: np.ndarray, q2: np.ndarray) -> float:
    a = _normalize_quat_xyzw(q1)
    b = _normalize_quat_xyzw(q2)
    dot = float(np.clip(abs(np.dot(a, b)), 0.0, 1.0))
    return float(np.degrees(2.0 * math.acos(dot)))


def _extract_frame_joint_quat_dict(payload: Dict[str, Any], frame_idx: int) -> Dict[str, Any]:
    if "joint_quat_xyzw" in payload:
        frame = payload["joint_quat_xyzw"]
        if not isinstance(frame, dict):
            raise TypeError("joint_quat_xyzw must be an object.")
        return frame

    if "joint_quat_xyzw_seq" in payload:
        seq = payload["joint_quat_xyzw_seq"]
        if not isinstance(seq, list) or not seq:
            raise TypeError("joint_quat_xyzw_seq must be a non-empty list.")
        if frame_idx < 0 or frame_idx >= len(seq):
            raise IndexError(f"frame_idx out of range: {frame_idx} (frames={len(seq)})")
        frame = seq[frame_idx]
        if not isinstance(frame, dict):
            raise TypeError("joint_quat_xyzw_seq frame must be an object.")
        return frame

    return payload


def _read_joint_quat_frame(path: Path, frame_idx: int) -> Dict[str, Any]:
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object.")
    frame = _extract_frame_joint_quat_dict(payload, frame_idx=frame_idx)
    if not isinstance(frame, dict):
        raise TypeError(f"{path} must contain joint quaternion map.")
    return frame


def _read_joint_quat(frame: Dict[str, Any], joint: str, src_label: str) -> np.ndarray:
    if joint not in frame:
        raise KeyError(f"Missing joint '{joint}' in {src_label}")
    entry = frame[joint]
    if not isinstance(entry, dict):
        raise TypeError(f"Joint '{joint}' in {src_label} must be an object.")
    for key in ("x", "y", "z", "w"):
        if key not in entry:
            raise KeyError(f"Joint '{joint}' in {src_label} missing key '{key}'")
    q = np.array([float(entry["x"]), float(entry["y"]), float(entry["z"]), float(entry["w"])], dtype=np.float64)
    return _canonicalize_w_positive(q)


def _collect_json_paths(input_path: Path, pattern: str) -> List[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() != ".json":
            raise ValueError(f"Expected .json file: {input_path}")
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path not found: {input_path}")
    paths = sorted(
        [p for p in input_path.glob(pattern) if p.is_file() and p.suffix.lower() == ".json"],
        key=natural_key_path,
    )
    if not paths:
        raise ValueError(f"No .json found in {input_path} with pattern={pattern}")
    return paths


def _pair_key(path: Path) -> str:
    stem = path.stem
    m = re.search(r"(\d+)$", stem)
    return m.group(1) if m else stem


def _pair_paths(source_paths: List[Path], target_paths: List[Path]) -> List[Tuple[Path, Path]]:
    if len(source_paths) == 1 and len(target_paths) == 1:
        return [(source_paths[0], target_paths[0])]

    source_map: Dict[str, Path] = {}
    for path in source_paths:
        key = _pair_key(path)
        if key in source_map:
            raise ValueError(f"Duplicate source pairing key '{key}'")
        source_map[key] = path

    target_map: Dict[str, Path] = {}
    for path in target_paths:
        key = _pair_key(path)
        if key in target_map:
            raise ValueError(f"Duplicate target pairing key '{key}'")
        target_map[key] = path

    common_keys = sorted(set(source_map.keys()) & set(target_map.keys()), key=lambda x: (len(x), x))
    if not common_keys:
        raise ValueError("No matching source/target file pairs by numeric suffix or stem.")

    return [(source_map[key], target_map[key]) for key in common_keys]


def _average_quaternions(quats: List[np.ndarray]) -> np.ndarray:
    if not quats:
        raise ValueError("No quaternion samples for averaging.")
    ref = _canonicalize_w_positive(quats[0])
    aligned: List[np.ndarray] = []
    for quat in quats:
        q = _canonicalize_w_positive(quat)
        if float(np.dot(ref, q)) < 0.0:
            q = -q
        aligned.append(q)
    mean_q = np.mean(np.stack(aligned, axis=0), axis=0)
    return _canonicalize_w_positive(mean_q)


def fit_bind_local_quat(
    pairs: List[Tuple[Path, Path]],
    joint_names: List[str],
    source_frame_idx: int,
    target_frame_idx: int,
    compose_order: str,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, float]]]:
    samples: Dict[str, List[np.ndarray]] = {joint: [] for joint in joint_names}

    for source_path, target_path in pairs:
        source_frame = _read_joint_quat_frame(source_path, frame_idx=source_frame_idx)
        target_frame = _read_joint_quat_frame(target_path, frame_idx=target_frame_idx)
        for joint in joint_names:
            q_source = _read_joint_quat(source_frame, joint, str(source_path))
            q_target = _read_joint_quat(target_frame, joint, str(target_path))
            if compose_order == "bind_mul_pose":
                q_bind = _quat_multiply_xyzw(q_target, _quat_inverse_xyzw(q_source))
            elif compose_order == "pose_mul_bind":
                q_bind = _quat_multiply_xyzw(_quat_inverse_xyzw(q_source), q_target)
            else:
                raise ValueError(f"Unsupported compose order: {compose_order}")
            samples[joint].append(q_bind)

    bind_map = {joint: _average_quaternions(samples[joint]) for joint in joint_names}

    stats: Dict[str, Dict[str, float]] = {}
    for joint in joint_names:
        errs: List[float] = []
        q_bind = bind_map[joint]
        for source_path, target_path in pairs:
            source_frame = _read_joint_quat_frame(source_path, frame_idx=source_frame_idx)
            target_frame = _read_joint_quat_frame(target_path, frame_idx=target_frame_idx)
            q_source = _read_joint_quat(source_frame, joint, str(source_path))
            q_target = _read_joint_quat(target_frame, joint, str(target_path))
            if compose_order == "bind_mul_pose":
                q_pred = _quat_multiply_xyzw(q_bind, q_source)
            else:
                q_pred = _quat_multiply_xyzw(q_source, q_bind)
            errs.append(_quat_angle_deg(q_pred, q_target))
        err_arr = np.asarray(errs, dtype=np.float64)
        stats[joint] = {
            "mean_deg": float(np.mean(err_arr)),
            "max_deg": float(np.max(err_arr)),
        }

    return bind_map, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit per-joint bind local quaternions from paired source/Unity quaternion JSON."
    )
    parser.add_argument("--source", required=True, type=str, help="Source quaternion JSON file or directory.")
    parser.add_argument("--target", required=True, type=str, help="Unity localRotation quaternion JSON file or directory.")
    parser.add_argument("--source-pattern", type=str, default="livehps_quaternion_*.json", help="Source glob pattern in directory mode.")
    parser.add_argument("--target-pattern", type=str, default="*.json", help="Target glob pattern in directory mode.")
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_UNITY_BIND_LOCAL_QUAT_JSON,
        help=f"Output bind quaternion JSON path. default={DEFAULT_UNITY_BIND_LOCAL_QUAT_JSON}",
    )
    parser.add_argument(
        "--compose-order",
        type=str,
        default="bind_mul_pose",
        choices=["bind_mul_pose", "pose_mul_bind"],
        help="Composition model used by npz2quat.",
    )
    parser.add_argument("--joint-names", type=str, default=None, help="Comma-separated 24 joint names in pose order.")
    parser.add_argument("--source-frame-idx", type=int, default=0, help="Frame index used when source has *_seq.")
    parser.add_argument("--target-frame-idx", type=int, default=0, help="Frame index used when target has *_seq.")
    parser.add_argument("--ascii", action="store_true", help="JSON ASCII escape")
    parser.add_argument("--compact", action="store_true", help="Compact JSON (no pretty indent)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]

    source_path = resolve_repo_path(repo_root, args.source)
    target_path = resolve_repo_path(repo_root, args.target)
    output_path = resolve_repo_path(repo_root, args.output)
    joint_names = parse_joint_names(args.joint_names)

    source_paths = _collect_json_paths(source_path, args.source_pattern)
    target_paths = _collect_json_paths(target_path, args.target_pattern)
    pairs = _pair_paths(source_paths, target_paths)

    bind_map, stats = fit_bind_local_quat(
        pairs=pairs,
        joint_names=joint_names,
        source_frame_idx=int(args.source_frame_idx),
        target_frame_idx=int(args.target_frame_idx),
        compose_order=args.compose_order,
    )

    output_payload = {
        "meta": {
            "fit_pairs": len(pairs),
            "compose_order": args.compose_order,
            "source_frame_idx": int(args.source_frame_idx),
            "target_frame_idx": int(args.target_frame_idx),
            "formula": "q_target = bind * q_source" if args.compose_order == "bind_mul_pose" else "q_target = q_source * bind",
        },
        "joint_quat_xyzw": {
            joint: {
                "x": float(bind_map[joint][0]),
                "y": float(bind_map[joint][1]),
                "z": float(bind_map[joint][2]),
                "w": float(bind_map[joint][3]),
            }
            for joint in joint_names
        },
        "fit_error_deg": stats,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, output_payload, ascii_flag=args.ascii, compact=args.compact)

    mean_err = float(np.mean([stats[j]["mean_deg"] for j in joint_names]))
    max_err = float(np.max([stats[j]["max_deg"] for j in joint_names]))
    print(f"Saved bind local quaternion JSON: {output_path}")
    print(f"pairs={len(pairs)}, compose_order={args.compose_order}, joints={len(joint_names)}")
    print(f"fit_error_deg_mean={mean_err:.6f}, fit_error_deg_max={max_err:.6f}")


if __name__ == "__main__":
    main()
