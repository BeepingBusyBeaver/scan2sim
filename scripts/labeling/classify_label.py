# scripts/labeling/classify_label.py
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from scripts.common.angle_utils import wrap180
from scripts.common.json_utils import write_json
from scripts.common.path_utils import natural_key_path, natural_key_text
from scripts.common.io_paths import (
    DATA_INTEROP_EULER_JSON,
    DATA_INTEROP_LABEL_DIR,
    DATA_INTEROP_LABEL_JSON,
    DATA_INTEROP_QUAT_JSON,
    DATASET_LABEL_RULES_JSON,
    resolve_repo_path,
)


"""
Single mode:
python -m scripts.labeling.classify_label \
  --idx 183 \
  --frame 0 \
  --euler-json outputs/euler/livehps_unity_virtual.json \
  --quat-json outputs/quat/livehps_quaternion_virtual.json \
  --rules dataset/label_rules.json \
  --output outputs/label/livehps_label_virtual.json \
  --compact \
  --joints rightHip,leftHip,rightKnee,leftKnee

Batch mode:
python -m scripts.labeling.classify_label \
  --batch_per_file \
  --euler-json outputs/euler \
  --quat-json outputs/quat \
  --rules dataset/label_rules.json \
  --output-dir outputs/label \
  --euler-pattern "livehps_unity_*.json" \
  --quat-pattern "livehps_quaternion_*.json" \
  --output-prefix "livehps_label_" \
  --joints rightHip,leftHip,rightKnee,leftKnee,rightShoulder,leftShoulder,rightElbow,leftElbow \
  --frame 0 \
  --compact

예시 매핑:
  outputs/euler/livehps_unity_123.json + outputs/quat/livehps_quaternion_123.json
  -> outputs/label/livehps_label_123.json
"""

# -----------------------------
# Joint name helpers
# -----------------------------
ALIAS_TO_SNAKE = {
    "pelvis": "pelvis",
    "leftHip": "left_hip",
    "rightHip": "right_hip",
    "spine1": "spine1",
    "leftKnee": "left_knee",
    "rightKnee": "right_knee",
    "spine2": "spine2",
    "leftAnkle": "left_ankle",
    "rightAnkle": "right_ankle",
    "spine3": "spine3",
    "leftFoot": "left_foot",
    "rightFoot": "right_foot",
    "neck": "neck",
    "leftCollar": "left_collar",
    "rightCollar": "right_collar",
    "head": "head",
    "leftShoulder": "left_shoulder",
    "rightShoulder": "right_shoulder",
    "leftElbow": "left_elbow",
    "rightElbow": "right_elbow",
    "leftWrist": "left_wrist",
    "rightWrist": "right_wrist",
    "leftHand": "left_hand",
    "rightHand": "right_hand",
}


def natural_key_str(s: str):
    return natural_key_text(s)


def snake_to_lower_camel(name: str) -> str:
    parts = name.split("_")
    if not parts:
        return name
    return parts[0] + "".join(p.capitalize() for p in parts[1:])


def snake_to_upper_camel(name: str) -> str:
    parts = [p for p in name.split("_") if p]
    if not parts:
        return name
    return "".join(p.capitalize() for p in parts)


def lower_camel_to_snake(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def normalize_joint_token_to_snake(token: str) -> str:
    token = token.strip()
    if not token:
        raise ValueError("Empty joint token.")

    if token in ALIAS_TO_SNAKE:
        return ALIAS_TO_SNAKE[token]

    # already snake_case?
    if "_" in token:
        return token.lower()

    # maybe lowerCamelCase
    return lower_camel_to_snake(token)


def resolve_joint_key_in_frame(frame_dict: Dict[str, Any], joint_name: str) -> str:
    """
    Resolve joint key in frame dict robustly:
      - exact key
      - snake <-> lowerCamel
    """
    cands = [joint_name]
    if "_" in joint_name:
        cands.append(snake_to_lower_camel(joint_name))
    else:
        cands.append(lower_camel_to_snake(joint_name))

    for c in cands:
        if c in frame_dict:
            return c
    raise KeyError(f"Joint '{joint_name}' not found in frame. candidates={cands}")


def classify_value_to_label(value: float, bins: List[Dict[str, Any]]) -> Optional[Any]:
    # interval: [min, max)
    for b in bins:
        bmin = float(b["min"])
        bmax = float(b["max"])
        if bmin <= value < bmax:
            return b["label"]
    return None


def side_letter_from_joint(joint_snake: str) -> str:
    if joint_snake.startswith("right_"):
        return "R"
    if joint_snake.startswith("left_"):
        return "L"
    return joint_snake[:1].upper() if joint_snake else "U"


def make_step_joint_prefix(joint_snake: str) -> str:
    # right_hip -> RightHip
    return snake_to_upper_camel(joint_snake)


# -----------------------------
# JSON frame extraction
# -----------------------------
def extract_frame_from_euler_json(obj: Dict[str, Any], frame_idx: int) -> Tuple[Dict[str, Any], int]:
    if "joint_euler_xyz_deg_seq" in obj:
        seq = obj["joint_euler_xyz_deg_seq"]
        if not isinstance(seq, list):
            raise TypeError("joint_euler_xyz_deg_seq must be a list.")
        n = len(seq)
        if n == 0:
            raise ValueError("joint_euler_xyz_deg_seq is empty.")
        if frame_idx < 0 or frame_idx >= n:
            raise IndexError(f"frame={frame_idx} out of range [0,{n-1}] for euler seq.")
        frame = seq[frame_idx]
        if not isinstance(frame, dict):
            raise TypeError("Each frame in joint_euler_xyz_deg_seq must be an object.")
        return frame, n

    if "joint_euler_xyz_deg" in obj:
        frame = obj["joint_euler_xyz_deg"]
        if not isinstance(frame, dict):
            raise TypeError("joint_euler_xyz_deg must be an object.")
        if frame_idx != 0:
            raise IndexError("Euler JSON is single-frame. Use --frame 0.")
        return frame, 1

    raise KeyError("Missing 'joint_euler_xyz_deg' or 'joint_euler_xyz_deg_seq' in euler JSON.")


def extract_frame_from_quat_json(obj: Dict[str, Any], frame_idx: int) -> Tuple[Dict[str, Any], int]:
    if "joint_quat_xyzw_seq" in obj:
        seq = obj["joint_quat_xyzw_seq"]
        if not isinstance(seq, list):
            raise TypeError("joint_quat_xyzw_seq must be a list.")
        n = len(seq)
        if n == 0:
            raise ValueError("joint_quat_xyzw_seq is empty.")
        if frame_idx < 0 or frame_idx >= n:
            raise IndexError(f"frame={frame_idx} out of range [0,{n-1}] for quat seq.")
        frame = seq[frame_idx]
        if not isinstance(frame, dict):
            raise TypeError("Each frame in joint_quat_xyzw_seq must be an object.")
        return frame, n

    if "joint_quat_xyzw" in obj:
        frame = obj["joint_quat_xyzw"]
        if not isinstance(frame, dict):
            raise TypeError("joint_quat_xyzw must be an object.")
        if frame_idx != 0:
            raise IndexError("Quaternion JSON is single-frame. Use --frame 0.")
        return frame, 1

    raise KeyError("Missing 'joint_quat_xyzw' or 'joint_quat_xyzw_seq' in quaternion JSON.")


# -----------------------------
# Batch pairing helpers
# -----------------------------
def extract_pair_key(path: Path, prefix_hint: str) -> str:
    """
    livehps_unity_123.json (prefix_hint=livehps_unity_) -> "123"
    livehps_quaternion_123.json (prefix_hint=livehps_quaternion_) -> "123"
    fallback: trailing digits, 없으면 stem 전체
    """
    stem = path.stem
    if prefix_hint and stem.startswith(prefix_hint):
        tail = stem[len(prefix_hint):]
        return tail if tail else stem

    m = re.search(r"(\d+)$", stem)
    if m:
        return m.group(1)

    return stem


def collect_json_map(input_dir: Path, pattern: str, prefix_hint: str) -> Dict[str, Path]:
    if not input_dir.is_dir():
        raise ValueError(f"Expected directory: {input_dir}")
    paths = sorted(
        [p for p in input_dir.glob(pattern) if p.is_file() and p.suffix.lower() == ".json"],
        key=natural_key_path,
    )
    if not paths:
        raise ValueError(f"No JSON files in {input_dir} with pattern='{pattern}'")

    out: Dict[str, Path] = {}
    for p in paths:
        k = extract_pair_key(p, prefix_hint=prefix_hint)
        if k in out:
            raise ValueError(
                f"Duplicate pair key '{k}' in directory {input_dir}. "
                f"Conflicts: {out[k].name}, {p.name}"
            )
        out[k] = p
    return out


def make_output_name(key: str, output_prefix: str) -> str:
    return f"{output_prefix}{key}.json"


# -----------------------------
# Core classify
# -----------------------------
def classify_payload(
    *,
    idx: int,
    frame: int,
    euler_obj: Dict[str, Any],
    quat_obj: Dict[str, Any],
    rules_obj: Dict[str, Any],
    joints_csv: str,
    on_out_of_range_arg: str,
    out_of_range_label: int,
) -> Dict[str, Any]:
    euler_frame, _ = extract_frame_from_euler_json(euler_obj, frame)
    quat_frame, _ = extract_frame_from_quat_json(quat_obj, frame)

    # meta
    meta = rules_obj.get("meta", {})
    normalize_mode = str(meta.get("normalize", "wrap180"))
    rule_policy = str(meta.get("on_out_of_range", "error"))

    if on_out_of_range_arg == "rule":
        on_oob = rule_policy
    else:
        on_oob = on_out_of_range_arg

    rules = rules_obj.get("rules", [])
    if not isinstance(rules, list):
        raise TypeError("rules must be a list in rules JSON.")

    # 1) Parse --joints once (order-preserving), dedupe by normalized snake name
    joint_tokens = [x.strip() for x in joints_csv.split(",") if x.strip()]
    if not joint_tokens:
        raise ValueError("--joints is empty.")

    selected_joints: List[Tuple[str, str, str]] = []  # (original_token, snake, step_prefix)
    seen_snakes = set()
    for tok in joint_tokens:
        snake = normalize_joint_token_to_snake(tok)
        if snake in seen_snakes:
            continue
        seen_snakes.add(snake)
        selected_joints.append((tok, snake, make_step_joint_prefix(snake)))

    selected_snake_set = {snake for _, snake, _ in selected_joints}

    # 2) classify only selected joints for step
    # key: (joint_snake, axis) -> label
    step_label_map: Dict[Tuple[str, str], Any] = {}
    # joint별 axis 등장 순서(룰 순서 보존)
    axis_order_by_joint: Dict[str, List[str]] = {}

    for rule in rules:
        rule_id = str(rule["id"])  # kept for diagnostics
        joints = rule.get("joints", [])
        axis = str(rule["axis"]).lower()
        bins = rule.get("bins", [])

        if axis not in ("x", "y", "z"):
            raise ValueError(f"Invalid axis in rule '{rule_id}': {axis}")

        for joint in joints:
            joint_snake = normalize_joint_token_to_snake(str(joint))
            # --joints에 없는 관절은 step에서 제외
            if joint_snake not in selected_snake_set:
                continue
            frame_joint_key = resolve_joint_key_in_frame(euler_frame, joint_snake)

            jdata = euler_frame[frame_joint_key]
            if not isinstance(jdata, dict) or axis not in jdata:
                raise KeyError(f"Euler frame joint '{frame_joint_key}' missing axis '{axis}'.")

            val = float(jdata[axis])
            if normalize_mode == "wrap180":
                val = wrap180(val)

            label = classify_value_to_label(val, bins)
            if label is None:
                if on_oob == "error":
                    raise ValueError(
                        f"Out-of-range value for rule='{rule_id}', joint='{joint_snake}', "
                        f"axis='{axis}', val={val}"
                    )
                if on_oob == "label":
                    label = out_of_range_label
                else:
                    raise ValueError(f"Unknown on_out_of_range policy: {on_oob}")

            step_label_map[(joint_snake, axis)] = label
            if joint_snake not in axis_order_by_joint:
                axis_order_by_joint[joint_snake] = []
            if axis not in axis_order_by_joint[joint_snake]:
                axis_order_by_joint[joint_snake].append(axis)

    # 3) build step in --joints order and per-joint axis order
    step: Dict[str, Any] = {}
    for _, joint_snake, step_prefix in selected_joints:
        for axis in axis_order_by_joint.get(joint_snake, []):
            step_key = f"{step_prefix}{axis.upper()}"   # e.g. RightHipX
            step[step_key] = step_label_map[(joint_snake, axis)]

    payload: Dict[str, Any] = {
        "idx": int(idx),
        "step": step,
    }

    for tok, snake, _ in selected_joints:
        qkey = resolve_joint_key_in_frame(quat_frame, snake)
        qd = quat_frame[qkey]

        if not isinstance(qd, dict):
            raise TypeError(f"Quaternion for joint '{qkey}' must be an object.")

        for comp in ("x", "y", "z", "w"):
            if comp not in qd:
                raise KeyError(f"Quaternion for joint '{qkey}' missing '{comp}'.")

        # output key name: keep user token 그대로 사용
        payload[tok] = {
            "x": float(qd["x"]),
            "y": float(qd["y"]),
            "z": float(qd["z"]),
            "w": float(qd["w"]),
        }

    return payload


def dump_json(payload: Dict[str, Any], out_path: Path, compact: bool, ascii_mode: bool) -> None:
    write_json(out_path, payload, ascii_flag=ascii_mode, compact=compact)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Classify labels from Unity-Euler JSON + Quaternion JSON using label_rules."
    )

    # Single / batch 공통 입력
    p.add_argument("--frame", type=int, default=0, help="Frame index to classify (for *_seq inputs).")
    p.add_argument("--euler-json", type=str, default=DATA_INTEROP_EULER_JSON)
    p.add_argument("--quat-json", type=str, default=DATA_INTEROP_QUAT_JSON)
    p.add_argument("--rules", type=str, default=DATASET_LABEL_RULES_JSON)

    # Single mode options
    p.add_argument("--idx", type=int, default=None, help="Single mode idx value (required in single mode).")
    p.add_argument("--output", type=str, default=DATA_INTEROP_LABEL_JSON, help="Single mode output path.")

    # Batch mode options
    p.add_argument("--batch_per_file", action="store_true", help="Enable directory batch processing.")
    p.add_argument("--euler-pattern", type=str, default="livehps_unity_*.json")
    p.add_argument("--quat-pattern", type=str, default="livehps_quaternion_*.json")
    p.add_argument("--euler-prefix", type=str, default="livehps_unity_")
    p.add_argument("--quat-prefix", type=str, default="livehps_quaternion_")
    p.add_argument("--output-dir", type=str, default=DATA_INTEROP_LABEL_DIR)
    p.add_argument("--output-prefix", type=str, default="livehps_label_")
    p.add_argument(
        "--pair_policy",
        choices=["error", "skip"],
        default="error",
        help="How to handle unpaired keys between euler/quat sets.",
    )

    # idx policy in batch
    p.add_argument(
        "--idx-mode",
        choices=["tail", "seq", "fixed"],
        default="tail",
        help="tail: use numeric pair key if possible, else seq. seq: incremental. fixed: same idx for all.",
    )
    p.add_argument("--idx-start", type=int, default=0, help="Start idx for seq mode (or tail fallback).")
    p.add_argument("--idx-fixed", type=int, default=0, help="Fixed idx value when idx-mode=fixed.")

    p.add_argument(
        "--joints",
        type=str,
        default="rightHip,leftHip,rightKnee,leftKnee,rightShoulder,leftShoulder,rightElbow,leftElbow",
        help="Comma-separated output joints (e.g. rightHip,leftHip,rightKnee,leftKnee).",
    )

    # out-of-range handling
    p.add_argument(
        "--on_out_of_range",
        choices=["rule", "error", "label"],
        default="rule",
        help="rule: follow rules.meta.on_out_of_range, error: force error, label: force fallback label.",
    )
    p.add_argument(
        "--out_of_range_label",
        type=int,
        default=-1,
        help="Fallback label value used when on_out_of_range=label.",
    )

    p.add_argument("--compact", dest="compact", action="store_true", help="Write compact one-line JSON.")
    p.add_argument("--pretty", dest="compact", action="store_false", help="Write pretty JSON.")
    p.add_argument("--ascii", action="store_true", help="Use ensure_ascii=True in JSON dump.")
    p.set_defaults(compact=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]

    rules_path = resolve_repo_path(repo_root, args.rules)
    if not rules_path.exists():
        raise FileNotFoundError(f"Rules JSON not found: {rules_path}")

    with rules_path.open("r", encoding="utf-8") as f:
        rules_obj = json.load(f)

    # -------------------------
    # Batch mode
    # -------------------------
    if args.batch_per_file:
        euler_dir = resolve_repo_path(repo_root, args.euler_json)
        quat_dir = resolve_repo_path(repo_root, args.quat_json)
        out_dir = resolve_repo_path(repo_root, args.output_dir)

        if not euler_dir.exists():
            raise FileNotFoundError(f"Euler dir not found: {euler_dir}")
        if not quat_dir.exists():
            raise FileNotFoundError(f"Quat dir not found: {quat_dir}")

        euler_map = collect_json_map(euler_dir, args.euler_pattern, args.euler_prefix)
        quat_map = collect_json_map(quat_dir, args.quat_pattern, args.quat_prefix)

        euler_keys = set(euler_map.keys())
        quat_keys = set(quat_map.keys())
        common_keys = sorted(euler_keys & quat_keys, key=natural_key_str)

        only_euler = sorted(euler_keys - quat_keys, key=natural_key_str)
        only_quat = sorted(quat_keys - euler_keys, key=natural_key_str)

        if (only_euler or only_quat) and args.pair_policy == "error":
            msg = ["Unpaired files detected."]
            if only_euler:
                msg.append(f"  only in euler ({len(only_euler)}): {only_euler[:10]}")
            if only_quat:
                msg.append(f"  only in quat  ({len(only_quat)}): {only_quat[:10]}")
            raise ValueError("\n".join(msg))

        if not common_keys:
            raise ValueError("No matched euler/quat file pairs found.")

        out_dir.mkdir(parents=True, exist_ok=True)

        total = len(common_keys)
        for seq_i, k in enumerate(common_keys):
            euler_path = euler_map[k]
            quat_path = quat_map[k]

            with euler_path.open("r", encoding="utf-8") as f:
                euler_obj = json.load(f)
            with quat_path.open("r", encoding="utf-8") as f:
                quat_obj = json.load(f)

            # idx 결정
            if args.idx_mode == "fixed":
                idx_value = int(args.idx_fixed)
            elif args.idx_mode == "seq":
                idx_value = int(args.idx_start + seq_i)
            else:  # tail
                if re.fullmatch(r"\d+", k):
                    idx_value = int(k)
                else:
                    m = re.search(r"(\d{3,})", k)  # 3자리 이상 숫자 덩어리(예: 000554)
                    if m:
                        idx_value = int(m.group(1))
                    else:
                        idx_value = int(args.idx_start + seq_i)

            payload = classify_payload(
                idx=idx_value,
                frame=args.frame,
                euler_obj=euler_obj,
                quat_obj=quat_obj,
                rules_obj=rules_obj,
                joints_csv=args.joints,
                on_out_of_range_arg=args.on_out_of_range,
                out_of_range_label=args.out_of_range_label,
            )

            out_name = make_output_name(k, args.output_prefix)
            out_path = out_dir / out_name
            dump_json(payload, out_path, compact=args.compact, ascii_mode=args.ascii)

            print(
                f"[{seq_i + 1:04d}/{total:04d}] "
                f"{euler_path.name} + {quat_path.name} -> {out_path.name} (idx={idx_value})"
            )

        print(f"Done. processed={total}, output_dir={out_dir}")
        return

    # -------------------------
    # Single mode
    # -------------------------
    euler_path = resolve_repo_path(repo_root, args.euler_json)
    quat_path = resolve_repo_path(repo_root, args.quat_json)

    if not euler_path.exists():
        raise FileNotFoundError(f"Euler JSON not found: {euler_path}")
    if not quat_path.exists():
        raise FileNotFoundError(f"Quaternion JSON not found: {quat_path}")

    if args.idx is None:
        raise ValueError("--idx is required in single mode.")

    with euler_path.open("r", encoding="utf-8") as f:
        euler_obj = json.load(f)
    with quat_path.open("r", encoding="utf-8") as f:
        quat_obj = json.load(f)

    payload = classify_payload(
        idx=int(args.idx),
        frame=args.frame,
        euler_obj=euler_obj,
        quat_obj=quat_obj,
        rules_obj=rules_obj,
        joints_csv=args.joints,
        on_out_of_range_arg=args.on_out_of_range,
        out_of_range_label=args.out_of_range_label,
    )

    # write / print
    if args.output:
        out_path = resolve_repo_path(repo_root, args.output)
        dump_json(payload, out_path, compact=args.compact, ascii_mode=args.ascii)
        print(f"saved: {out_path}")

    print(json.dumps(payload, ensure_ascii=args.ascii, separators=(",", ":") if args.compact else None))


if __name__ == "__main__":
    main()
