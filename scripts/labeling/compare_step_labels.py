# scripts/labeling/compare_step_labels.py
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from scripts.common.angle_utils import wrap180
from scripts.common.path_utils import extract_numeric_suffix, natural_key_path, natural_key_text
from scripts.common.io_paths import (
    DATA_INTEROP_EULER_DIR,
    DATA_INTEROP_LABEL_DIR,
    DATASET_LABEL_RULES_JSON,
    DATASET_REAL_IMU_GT_JSONL,
    resolve_repo_path,
)


"""
기본 사용:
python -m scripts.labeling.compare_step_labels \
  --pred-dir outputs/label \
  --pred-pattern 'livehps_label_*.json' \
  --gt-jsonl dataset/real_IMU_GT.jsonl

mismatch 거리 계산(기본값 경로 사용):
  - unity: outputs/euler/livehps_unity_*.json
  - rules: dataset/label_rules.json
즉, 기본 커맨드 그대로 실행해도 동작하도록 해둠.

필요시 명시:
python -m scripts.labeling.compare_step_labels \
  --pred-dir outputs/label \
  --pred-pattern 'livehps_label_*.json' \
  --gt-jsonl dataset/real_IMU_GT.jsonl \
  --unity-dir outputs/euler \
  --unity-pattern 'livehps_unity_*.json' \
  --rules-json dataset/label_rules.json \
  --show-key-order \
  --show-confusion \
  --dump-mismatch-rows \
  --show-gt-angle-percentiles

OX 문자열의 key 순서까지 함께 보고 싶다면:
python -m scripts.labeling.compare_step_labels \
  --pred-dir outputs/label \
  --pred-pattern 'livehps_label_*.json' \
  --gt-jsonl dataset/real_IMU_GT.jsonl \
  --show-key-order
"""

# (joint, axis) -> {label: [(min, max), ...]}
RuleLookup = Dict[Tuple[str, str], Dict[int, List[Tuple[float, float]]]]

def to_int_label(value: Any) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return None


def load_gt_map(gt_path: Path) -> Dict[int, Dict[str, Any]]:
    gt_map: Dict[int, Dict[str, Any]] = {}
    with gt_path.open("r", encoding="utf-8") as handle:
        for line_num, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            idx = obj.get("idx")
            step = obj.get("step")
            if idx is None or not isinstance(step, dict):
                raise ValueError(f"Invalid GT row at line {line_num}: requires idx + step object")
            idx_int = int(idx)
            if idx_int in gt_map:
                raise ValueError(f"Duplicate GT idx detected: {idx_int}")
            gt_map[idx_int] = step
    if not gt_map:
        raise ValueError(f"No valid GT rows found in {gt_path}")
    return gt_map


def load_pred_payload(pred_path: Path) -> Tuple[int, Dict[str, Any]]:
    with pred_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Prediction JSON must be an object")
    idx = payload.get("idx")
    step = payload.get("step")
    if idx is None or not isinstance(step, dict):
        raise ValueError("Prediction JSON requires idx + step object")
    return int(idx), step


def load_unity_joint_angles(unity_path: Path) -> Dict[str, Dict[str, float]]:
    with unity_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Unity JSON must be an object")
    joints = payload.get("joint_euler_xyz_deg")
    if not isinstance(joints, dict):
        raise ValueError("Unity JSON requires joint_euler_xyz_deg object")
    # 값은 float로 강제하지 않고, 실제 사용할 때 변환
    return joints  # type: ignore[return-value]


def load_rules_lookup(rules_path: Path) -> RuleLookup:
    with rules_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Rules JSON must be an object")

    rules = payload.get("rules")
    if not isinstance(rules, list):
        raise ValueError("Rules JSON requires 'rules' list")

    lookup: RuleLookup = {}
    for rule in rules:
        if not isinstance(rule, dict):
            raise ValueError("Each rule must be an object")

        axis = str(rule.get("axis", "")).lower()
        joints = rule.get("joints")
        bins = rule.get("bins")

        if axis not in {"x", "y", "z"}:
            raise ValueError(f"Invalid axis in rule: {rule}")
        if not isinstance(joints, list) or not all(isinstance(j, str) for j in joints):
            raise ValueError(f"Invalid joints in rule: {rule}")
        if not isinstance(bins, list):
            raise ValueError(f"Invalid bins in rule: {rule}")

        parsed_bins: List[Tuple[int, float, float]] = []
        for b in bins:
            if not isinstance(b, dict):
                raise ValueError(f"Invalid bin: {b}")
            label = to_int_label(b.get("label"))
            if label is None:
                raise ValueError(f"Invalid bin label: {b}")
            try:
                mn = float(b.get("min"))
                mx = float(b.get("max"))
            except Exception:
                raise ValueError(f"Invalid bin range: {b}")
            parsed_bins.append((label, mn, mx))

        for joint in joints:
            key = (joint, axis)
            by_label = lookup.setdefault(key, {})
            for label, mn, mx in parsed_bins:
                by_label.setdefault(label, []).append((mn, mx))

    return lookup


def step_key_to_joint_axis(step_key: str) -> Optional[Tuple[str, str]]:
    """
    step key 예:
      RightHipX, LeftShoulderZ, RightElbowY ...
    -> (right_hip, x), (left_shoulder, z), (right_elbow, y)
    """
    m = re.fullmatch(r"(Left|Right)([A-Za-z]+)([XYZ])", step_key)
    if not m:
        return None
    side, part, axis = m.groups()
    joint = f"{side.lower()}_{part.lower()}"
    return joint, axis.lower()


def signed_distance_point_to_interval(x: float, mn: float, mx: float) -> float:
    """
    x와 [mn, mx]의 signed distance.
    - x < mn : 음수 (GT 구간보다 작은 쪽으로 벗어남)
    - x > mx : 양수 (GT 구간보다 큰 쪽으로 벗어남)
    - inside : 0
    """
    if x < mn:
        return x - mn
    if x > mx:
        return x - mx
    return 0.0


def signed_distance_point_to_intervals(x: float, intervals: List[Tuple[float, float]]) -> float:
    """
    여러 interval 중 |signed distance|가 최소인 interval 기준 signed distance 반환.
    tie일 때는 (abs(d), d) 기준으로 선택(재현성 확보).
    """
    if not intervals:
        raise ValueError("intervals must not be empty")

    best_d: Optional[float] = None
    best_abs = math.inf
    for mn, mx in intervals:
        d = signed_distance_point_to_interval(x, mn, mx)
        abs_d = abs(d)
        if best_d is None or abs_d < best_abs or (abs_d == best_abs and d < best_d):
            best_d = d
            best_abs = abs_d

    return float(best_d if best_d is not None else 0.0)


def format_intervals(intervals: List[Tuple[float, float]]) -> str:
    return ", ".join(f"[{mn:.2f},{mx:.2f})" for mn, mx in intervals)

def percentile_linear(values: List[float], q: float) -> float:
    """
    Linear interpolation percentile (similar to numpy.percentile default 'linear').
    q: 0~100
    """
    if not values:
        raise ValueError("values must not be empty")
    xs = sorted(values)
    n = len(xs)
    if n == 1:
        return float(xs[0])
    pos = (n - 1) * (q / 100.0)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(xs[lo])
    w = pos - lo
    return float(xs[lo] * (1.0 - w) + xs[hi] * w)

def get_unity_angle_for_step_key(
    step_key: str,
    unity_angles: Optional[Dict[str, Dict[str, float]]],
) -> Tuple[Optional[float], str]:
    if unity_angles is None:
        return None, "missing_unity_pair"
    mapped = step_key_to_joint_axis(step_key)
    if mapped is None:
        return None, "unknown_step_key_format"
    joint, axis = mapped
    joint_angles = unity_angles.get(joint)
    if not isinstance(joint_angles, dict):
        return None, f"missing_joint:{joint}"
    if axis not in joint_angles:
        return None, f"missing_axis:{joint}.{axis}"
    try:
        raw_angle = float(joint_angles[axis])  # type: ignore[arg-type]
    except Exception:
        return None, f"invalid_angle:{joint}.{axis}"
    return wrap180(raw_angle), ""

def get_gt_intervals_for_step_key(
    step_key: str,
    gt_label_raw: Any,
    rule_lookup: RuleLookup,
) -> Tuple[Optional[int], Optional[List[Tuple[float, float]]], str]:
    mapped = step_key_to_joint_axis(step_key)
    if mapped is None:
        return None, None, "unknown_step_key_format"
    joint, axis = mapped
    gt_label = to_int_label(gt_label_raw)
    if gt_label is None:
        return None, None, "invalid_gt_label"
    by_label = rule_lookup.get((joint, axis))
    if by_label is None:
        return gt_label, None, f"missing_rule:{joint}.{axis}"
    intervals = by_label.get(gt_label)
    if not intervals:
        return gt_label, None, f"missing_gt_bin:{joint}.{axis}:{gt_label}"
    return gt_label, intervals, ""


def compare_step_dicts(
    pred_step: Dict[str, Any], gt_step: Dict[str, Any]
) -> Tuple[int, int, List[Tuple[str, bool, Any, Any]], List[str]]:
    """
    GT step에 있는 key만 비교.
    Returns:
      correct: 일치한 key 수
      total: 비교한 전체 key 수 (GT key 기준)
      details: [(key, is_match, pred_val, gt_val), ...] (key 정렬 순서)
      ignored_pred_only_keys: pred에만 있고 GT에는 없는 key들(점수 제외)
    """
    # GT key 기준
    all_keys = sorted(gt_step.keys(), key=natural_key_text)

    details: List[Tuple[str, bool, Any, Any]] = []
    correct = 0

    for key in all_keys:
        pred_val = pred_step.get(key, "<MISSING>")
        gt_val = gt_step[key]
        is_match = pred_val == gt_val
        if is_match:
            correct += 1
        details.append((key, is_match, pred_val, gt_val))

    ignored_pred_only_keys = sorted(set(pred_step.keys()) - set(gt_step.keys()), key=natural_key_text)
    return correct, len(all_keys), details, ignored_pred_only_keys


def parse_score_key(score_key: str) -> Tuple[int, int]:
    # "7/9" -> (7, 9)
    try:
        c_str, t_str = score_key.split("/")
        return int(c_str), int(t_str)
    except Exception:
        return -1, -1


def score_sort_key(item: Tuple[str, int]):
    # 높은 정답률 -> 높은 correct -> 작은 total 순
    score_key, _count = item
    c, t = parse_score_key(score_key)
    ratio = (c / t) if t > 0 else -1.0
    return (-ratio, -c, t, score_key)


def compute_mismatch_distance_to_gt(
    step_key: str,
    gt_label_raw: Any,
    unity_angles: Optional[Dict[str, Dict[str, float]]],
    rule_lookup: RuleLookup,
) -> Tuple[Optional[float], str]:
    angle, r1 = get_unity_angle_for_step_key(step_key, unity_angles)
    if angle is None:
        return None, r1
    _gt_label, intervals, r2 = get_gt_intervals_for_step_key(step_key, gt_label_raw, rule_lookup)
    if intervals is None:
        return None, r2
    return signed_distance_point_to_intervals(angle, intervals), ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare step labels in livehps_label_*.json against GT jsonl by matching idx."
    )
    parser.add_argument("--pred-dir", type=str, default=DATA_INTEROP_LABEL_DIR)
    parser.add_argument("--pred-pattern", type=str, default="livehps_label_*.json")
    parser.add_argument("--gt-jsonl", type=str, default=DATASET_REAL_IMU_GT_JSONL)

    # 추가: mismatch 거리 계산용 입력
    parser.add_argument("--unity-dir", type=str, default=DATA_INTEROP_EULER_DIR)
    parser.add_argument("--unity-pattern", type=str, default="livehps_unity_*.json")
    parser.add_argument("--rules-json", type=str, default=DATASET_LABEL_RULES_JSON)

    # 아래 2개 옵션은 하위호환용(더 이상 사용 안 함)
    parser.add_argument(
        "--show-mismatches",
        action="store_true",
        help="[deprecated] ignored. OX line is always shown.",
    )
    parser.add_argument(
        "--max-mismatches",
        type=int,
        default=12,
        help="[deprecated] ignored.",
    )

    # OX 문자열의 key 순서를 같이 보고 싶을 때
    parser.add_argument(
        "--show-key-order",
        action="store_true",
        help="Print key order used to build OX string for each file.",
    )

    # 파일별 mismatch 거리 평균(선택 출력)
    parser.add_argument(
        "--show-file-mismatch-distance",
        action="store_true",
        help="Print per-file signed mismatch distance summary (+/-/zero) to GT label intervals.",
    )

    parser.add_argument(
        "--show-confusion",
        action="store_true",
        help="Print per-step (gt_label, pred_label) count table.",
    )
    parser.add_argument(
        "--dump-mismatch-rows",
        action="store_true",
        help="Dump mismatch rows: idx, step_key, gt, pred, unity_angle, dist, gt_interval.",
    )
    parser.add_argument(
        "--show-gt-angle-percentiles",
        action="store_true",
        help="Print p10/p50/p90 of unity_angle grouped by (step_key, GT label).",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]

    pred_dir = resolve_repo_path(repo_root, args.pred_dir)
    gt_path = resolve_repo_path(repo_root, args.gt_jsonl)
    unity_dir = resolve_repo_path(repo_root, args.unity_dir)
    rules_path = resolve_repo_path(repo_root, args.rules_json)

    if not pred_dir.exists():
        raise FileNotFoundError(f"Prediction dir not found: {pred_dir}")
    if not gt_path.exists():
        raise FileNotFoundError(f"GT jsonl not found: {gt_path}")
    if not unity_dir.exists():
        raise FileNotFoundError(f"Unity dir not found: {unity_dir}")
    if not rules_path.exists():
        raise FileNotFoundError(f"Rules json not found: {rules_path}")

    if args.show_mismatches or args.max_mismatches != 12:
        print("[INFO] --show-mismatches / --max-mismatches are deprecated and ignored.")

    gt_map = load_gt_map(gt_path)
    rule_lookup = load_rules_lookup(rules_path)

    pred_paths = sorted(pred_dir.glob(args.pred_pattern), key=natural_key_path)
    if not pred_paths:
        raise FileNotFoundError(f"No files matched: {pred_dir / args.pred_pattern}")

    unity_paths = sorted(unity_dir.glob(args.unity_pattern), key=natural_key_path)
    unity_pair_map: Dict[int, Path] = {}
    for up in unity_paths:
        token = extract_numeric_suffix(up.name)
        if token is None:
            continue
        if token in unity_pair_map:
            raise ValueError(
                f"Duplicate unity numeric suffix detected: {token} "
                f"({unity_pair_map[token].name}, {up.name})"
            )
        unity_pair_map[token] = up

    print(f"[INFO] GT rows: {len(gt_map)} | Pred files: {len(pred_paths)}")
    print(f"[INFO] GT: {gt_path}")
    print(f"[INFO] Pred: {pred_dir / args.pred_pattern}")
    print(f"[INFO] Rules: {rules_path}")
    print(f"[INFO] Unity: {unity_dir / args.unity_pattern} (paired={len(unity_pair_map)})")
    print("-" * 120)

    total_files = len(pred_paths)
    compared_files = 0
    perfect_files = 0
    missing_gt_files = 0
    error_files = 0
    total_correct = 0
    total_labels = 0

    # 추가 집계
    score_counter: Counter[str] = Counter()  # 예: "7/9" -> n개
    step_correct: Dict[str, int] = defaultdict(int)
    step_total: Dict[str, int] = defaultdict(int)

    # mismatch signed 거리 집계 (+ / - / 0)
    mismatch_signed_pos_sum: Dict[str, float] = defaultdict(float)  # step_key별 +거리 합
    mismatch_signed_pos_count: Dict[str, int] = defaultdict(int)
    mismatch_signed_neg_sum: Dict[str, float] = defaultdict(float)  # step_key별 -거리 합(음수)
    mismatch_signed_neg_count: Dict[str, int] = defaultdict(int)
    mismatch_signed_zero_count: Dict[str, int] = defaultdict(int)  # step_key별 0 개수

    mismatch_dist_skip_reason: Counter[str] = Counter()
    total_mismatch_labels = 0

    total_mismatch_signed_pos_sum = 0.0
    total_mismatch_signed_pos_count = 0
    total_mismatch_signed_neg_sum = 0.0
    total_mismatch_signed_neg_count = 0
    total_mismatch_signed_zero_count = 0
    total_mismatch_signed_computed_count = 0

    # unity 파일 캐시
    unity_cache: Dict[Path, Dict[str, Dict[str, float]]] = {}

    # confusion: step_key -> Counter[(gt_str, pred_str)]
    confusion: Dict[str, Counter[Tuple[str, str]]] = defaultdict(Counter)

    # mismatch row dump lines
    mismatch_dump_lines: List[str] = []

    # angle percentiles: step_key -> gt_label(int) -> angles(list)
    angles_by_step_gt: Dict[str, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))

    for pred_path in pred_paths:
        name = pred_path.name
        try:
            idx, pred_step = load_pred_payload(pred_path)
        except Exception as exc:
            error_files += 1
            print(f"{name} | ERROR | {exc}")
            continue

        gt_step = gt_map.get(idx)
        if gt_step is None:
            missing_gt_files += 1
            print(f"{name} | idx={idx} | SKIP (GT idx not found)")
            continue

        # pair: 파일 suffix 기준 (예: 001)
        token = extract_numeric_suffix(name)
        unity_path = unity_pair_map.get(token if token is not None else -1)

        unity_angles: Optional[Dict[str, Dict[str, float]]] = None
        if unity_path is not None:
            if unity_path not in unity_cache:
                try:
                    unity_cache[unity_path] = load_unity_joint_angles(unity_path)
                except Exception as exc:
                    # 해당 파일은 거리 계산만 실패 처리, label 비교는 계속 진행
                    mismatch_dist_skip_reason[f"unity_read_error:{unity_path.name}:{exc}"] += 1
                    unity_cache[unity_path] = {}  # 빈 dict로 캐시
            cached = unity_cache[unity_path]
            if cached:
                unity_angles = cached
        else:
            # pair가 없으면 mismatch 거리 계산 불가
            unity_angles = None

        correct, label_count, details, ignored_pred_only_keys = compare_step_dicts(pred_step, gt_step)
        accuracy = (correct / label_count * 100.0) if label_count else 0.0
        status = "PASS" if correct == label_count else "FAIL"

        compared_files += 1
        total_correct += correct
        total_labels += label_count
        file_mismatches = label_count - correct
        total_mismatch_labels += file_mismatches

        if status == "PASS":
            perfect_files += 1

        # 점수 분포 카운트
        score_key = f"{correct}/{label_count}"
        score_counter[score_key] += 1

        # step별 정답률 집계 + mismatch signed 거리 집계
        file_signed_dist_values: List[float] = []
        for key, is_match, _pred_val, gt_val in details:
            # ---- confusion accumulation ----
            gt_i = to_int_label(gt_val)
            pred_i = to_int_label(_pred_val) if _pred_val != "<MISSING>" else None
            gt_s = str(gt_i) if gt_i is not None else str(gt_val)
            pred_s = "MISSING" if _pred_val == "<MISSING>" else (str(pred_i) if pred_i is not None else str(_pred_val))
            confusion[key][(gt_s, pred_s)] += 1

            # ---- angle accumulation per GT label ----
            if gt_i is not None:
                ang, ang_reason = get_unity_angle_for_step_key(key, unity_angles)
                if ang is not None:
                    angles_by_step_gt[key][gt_i].append(ang)

            # ---- step별 정확도 및 mismatch 거리 집계 ----
            step_total[key] += 1
            if is_match:
                step_correct[key] += 1
            else:
                # ---- mismatch distance + dump row data ----
                ang, r_ang = get_unity_angle_for_step_key(key, unity_angles)
                gt_label_i, intervals, r_int = get_gt_intervals_for_step_key(key, gt_val, rule_lookup)

                dist: Optional[float] = None
                reason = ""
                if ang is None:
                    reason = r_ang
                elif intervals is None:
                    reason = r_int
                else:
                    dist = signed_distance_point_to_intervals(ang, intervals)

                if dist is None:
                    mismatch_dist_skip_reason[reason] += 1
                else:
                    file_signed_dist_values.append(dist)
                    total_mismatch_signed_computed_count += 1

                    if dist > 0:
                        mismatch_signed_pos_sum[key] += dist
                        mismatch_signed_pos_count[key] += 1
                        total_mismatch_signed_pos_sum += dist
                        total_mismatch_signed_pos_count += 1
                    elif dist < 0:
                        mismatch_signed_neg_sum[key] += dist
                        mismatch_signed_neg_count[key] += 1
                        total_mismatch_signed_neg_sum += dist
                        total_mismatch_signed_neg_count += 1
                    else:
                        mismatch_signed_zero_count[key] += 1
                        total_mismatch_signed_zero_count += 1

                if args.dump_mismatch_rows:
                    pred_show = _pred_val
                    gt_show = gt_val
                    ang_show = f"{ang:.2f}" if ang is not None else "NA"
                    dist_show = f"{dist:.2f}" if dist is not None else "NA"
                    interval_show = format_intervals(intervals) if intervals is not None else "NA"
                    reason_show = reason if dist is None else ""
                    mismatch_dump_lines.append(
                        f"idx={idx} file={name} step={key} gt={gt_show} pred={pred_show} "
                        f"angle={ang_show} dist={dist_show} gt_interval={interval_show}"
                        + (f" reason={reason_show}" if reason_show else "")
                    )

        # OX 1줄
        ox_line = "".join("o" if is_match else "x" for _k, is_match, _pv, _gv in details)
        print(f"{name} | idx={idx} | {correct}/{label_count} ({accuracy:.1f}%) | {status} | OX={ox_line}")

        if args.show_key_order:
            key_order = ", ".join(key for key, _m, _pv, _gv in details)
            print(f"  key_order=[{key_order}]")

        if args.show_file_mismatch_distance and file_mismatches > 0:
            computed = len(file_signed_dist_values)
            if computed > 0:
                pos_vals = [d for d in file_signed_dist_values if d > 0]
                neg_vals = [d for d in file_signed_dist_values if d < 0]
                zero_n = computed - len(pos_vals) - len(neg_vals)
                unavailable_n = file_mismatches - computed

                pos_msg = (
                    f"+mean={sum(pos_vals)/len(pos_vals):.2f} deg ({len(pos_vals)}/{file_mismatches})"
                    if pos_vals
                    else f"+mean=(unavailable, 0/{file_mismatches})"
                )
                neg_msg = (
                    f"-mean={sum(neg_vals)/len(neg_vals):.2f} deg ({len(neg_vals)}/{file_mismatches})"
                    if neg_vals
                    else f"-mean=(unavailable, 0/{file_mismatches})"
                )
                print(
                    f"  mismatch-distance-signed: {pos_msg} | {neg_msg} | "
                    f"zero={zero_n}/{file_mismatches} | unavailable={unavailable_n}/{file_mismatches}"
                )
            else:
                print(
                    "  mismatch-distance-signed: "
                    f"+mean=(unavailable, 0/{file_mismatches}) | "
                    f"-mean=(unavailable, 0/{file_mismatches}) | "
                    f"zero=0/{file_mismatches} | unavailable={file_mismatches}/{file_mismatches}"
                )

        if ignored_pred_only_keys:
            print(f"  [INFO] ignored pred-only keys: {ignored_pred_only_keys}")

    print("-" * 120)
    field_accuracy = (total_correct / total_labels * 100.0) if total_labels else 0.0
    file_accuracy = (perfect_files / compared_files * 100.0) if compared_files else 0.0

    print(
        f"[SUMMARY] files={total_files}, compared={compared_files}, "
        f"perfect={perfect_files}, missing_gt={missing_gt_files}, errors={error_files}"
    )
    print(
        f"[SUMMARY] file-exact-accuracy={file_accuracy:.1f}% "
        f"({perfect_files}/{compared_files if compared_files else 0})"
    )
    print(
        f"[SUMMARY] label-accuracy={field_accuracy:.1f}% "
        f"({total_correct}/{total_labels if total_labels else 0})"
    )

    print("[SUMMARY] score-counts (correct/total):")
    if score_counter:
        for score_key, cnt in sorted(score_counter.items(), key=score_sort_key):
            print(f"  {score_key}: {cnt}")
    else:
        print("  (none)")

    print("[SUMMARY] per-step-accuracy:")
    if step_total:
        for step_key in sorted(step_total.keys(), key=natural_key_text):
            c = step_correct[step_key]
            t = step_total[step_key]
            pct = (c / t * 100.0) if t else 0.0
            print(f"  {step_key}: {pct:.1f}% ({c}/{t})")
    else:
        print("  (none)")

    # 신규 출력: mismatch signed 거리(정답 라벨 구간까지 최소거리)의 step별 +/- 분리 평균
    print("[SUMMARY] per-step-mismatch-distance-to-GT (signed deg, mismatches only):")
    any_mismatch_row = False
    for step_key in sorted(step_total.keys(), key=natural_key_text):
        mismatches = step_total[step_key] - step_correct[step_key]
        if mismatches <= 0:
            continue
        any_mismatch_row = True

        pos_n = mismatch_signed_pos_count.get(step_key, 0)
        neg_n = mismatch_signed_neg_count.get(step_key, 0)
        zero_n = mismatch_signed_zero_count.get(step_key, 0)
        computed_n = pos_n + neg_n + zero_n

        if pos_n > 0:
            pos_mean = mismatch_signed_pos_sum[step_key] / pos_n
            pos_msg = f"+mean={pos_mean:.2f} deg ({pos_n}/{mismatches})"
        else:
            pos_msg = f"+mean=(unavailable, 0/{mismatches})"

        if neg_n > 0:
            neg_mean = mismatch_signed_neg_sum[step_key] / neg_n
            neg_msg = f"-mean={neg_mean:.2f} deg ({neg_n}/{mismatches})"
        else:
            neg_msg = f"-mean=(unavailable, 0/{mismatches})"

        unavailable_n = mismatches - computed_n
        print(
            f"  {step_key}: {pos_msg} | {neg_msg} | "
            f"zero={zero_n}/{mismatches} | unavailable={unavailable_n}/{mismatches}"
        )

    if not any_mismatch_row:
        print("  (none)")

    if total_mismatch_labels > 0:
        pos_total_msg = (
            f"+mean={total_mismatch_signed_pos_sum/total_mismatch_signed_pos_count:.2f} deg "
            f"({total_mismatch_signed_pos_count}/{total_mismatch_labels})"
            if total_mismatch_signed_pos_count > 0
            else f"+mean=(unavailable, 0/{total_mismatch_labels})"
        )
        neg_total_msg = (
            f"-mean={total_mismatch_signed_neg_sum/total_mismatch_signed_neg_count:.2f} deg "
            f"({total_mismatch_signed_neg_count}/{total_mismatch_labels})"
            if total_mismatch_signed_neg_count > 0
            else f"-mean=(unavailable, 0/{total_mismatch_labels})"
        )

        unavailable_n = total_mismatch_labels - total_mismatch_signed_computed_count
        print(
            "[SUMMARY] mismatch-distance-signed-all: "
            f"{pos_total_msg} | {neg_total_msg} | "
            f"zero={total_mismatch_signed_zero_count}/{total_mismatch_labels} | "
            f"computed={total_mismatch_signed_computed_count}/{total_mismatch_labels} | "
            f"unavailable={unavailable_n}/{total_mismatch_labels}"
        )

        # 참고용: 부호 무시(abs) 평균도 함께 표시 (zero 제외)
        abs_n = total_mismatch_signed_pos_count + total_mismatch_signed_neg_count
        if abs_n > 0:
            abs_sum = total_mismatch_signed_pos_sum + abs(total_mismatch_signed_neg_sum)
            print(
                f"[SUMMARY] mismatch-distance-abs-mean-all={abs_sum/abs_n:.2f} deg "
                f"(computed {abs_n}/{total_mismatch_labels}, zero excluded)"
            )
        else:
            print(
                f"[SUMMARY] mismatch-distance-abs-mean-all=(unavailable) "
                f"(computed 0/{total_mismatch_labels})"
            )

    if mismatch_dist_skip_reason:
        print("[SUMMARY] mismatch-distance skipped reasons:")
        for reason, cnt in sorted(mismatch_dist_skip_reason.items(), key=lambda x: (-x[1], x[0])):
            print(f"  {reason}: {cnt}")

    # ---- per-step confusion output ----
    if args.show_confusion:
        print("[SUMMARY] per-step confusion counts (gt_label, pred_label) -> n:")
        for step_key in sorted(confusion.keys(), key=natural_key_text):
            cnt = confusion[step_key]
            if not cnt:
                continue
            print(f"  {step_key}:")
            # sort by gt int, pred int where possible
            def _sort_pair(item: Tuple[Tuple[str, str], int]):
                (g, p), n = item
                try:
                    gi = int(g)
                except Exception:
                    gi = 10**9
                try:
                    pi = int(p)
                except Exception:
                    pi = 10**9
                return (gi, pi, g, p)
            for (g, p), n in sorted(cnt.items(), key=_sort_pair):
                print(f"    (gt={g}, pred={p}): {n}")

    # ---- mismatch dump ----
    if args.dump_mismatch_rows:
        print("[SUMMARY] mismatch rows (idx, step_key, gt, pred, unity_angle, dist, gt_interval):")
        if mismatch_dump_lines:
            for line in mismatch_dump_lines:
                print(f"  {line}")
        else:
            print("  (none)")

    # ---- GT label별 unity_angle percentiles ----
    if args.show_gt_angle_percentiles:
        print("[SUMMARY] unity_angle percentiles by (step_key, GT label): p10/p50/p90 (wrap180)")
        for step_key in sorted(angles_by_step_gt.keys(), key=natural_key_text):
            by_gt = angles_by_step_gt[step_key]
            if not by_gt:
                continue
            for gt_label in sorted(by_gt.keys()):
                vals = by_gt[gt_label]
                if not vals:
                    continue
                p10 = percentile_linear(vals, 10.0)
                p50 = percentile_linear(vals, 50.0)
                p90 = percentile_linear(vals, 90.0)
                print(
                    f"  {step_key} | gt={gt_label} | n={len(vals)} | "
                    f"p10={p10:.2f} p50={p50:.2f} p90={p90:.2f}"
                )


if __name__ == "__main__":
    main()
