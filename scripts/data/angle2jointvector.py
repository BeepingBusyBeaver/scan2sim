# scripts/data/angle2jointvector.py
import argparse
import json
import sys
from dataclasses import dataclass
from typing import Dict, List

import pandas as pd


'''
python -m scripts.data.angle2jointvector \
  --csv data/labeling/001/8joints_02-06_16-49.csv \
  --pretty \
  --animframe 261
'''


@dataclass(frozen=True)
class Bin:
    label: int
    lo: float
    hi: float
    closed_hi: bool = False

    def contains(self, v: float) -> bool:
        # half-open: [lo, hi)  /  마지막 bin: [lo, hi]
        return (self.lo <= v <= self.hi) if self.closed_hi else (self.lo <= v < self.hi)


def pick_bin(v: float, bins: List[Bin], name: str) -> int:
    for b in bins:
        if b.contains(v):
            return b.label

    # out-of-range -> nearest snap + warn
    def dist_to_interval(x: float, lo: float, hi: float) -> float:
        if x < lo:
            return lo - x
        if x > hi:
            return x - hi
        return 0.0

    best = min(bins, key=lambda b: dist_to_interval(v, b.lo, b.hi))
    print(
        f"[warn] {name}: angle {v:.3f} out of range, snapping to {best.label} (range {best.lo}..{best.hi})",
        file=sys.stderr,
    )
    return best.label


# ===== mapping tables (from the image) =====
SHOULDER_X = [
    Bin(-1, -50.0, -20.0),
    Bin(0,  -20.0,  20.0),
    Bin(1,   20.0, 180.0, closed_hi=True),
]
SHOULDER_Z = [
    Bin(-1, -90.0, -10.0),
    Bin(0,   -10.0, 70.0),
    Bin(1,   70.0,   90.0, closed_hi=True),
]

HIP_X = [
    Bin(-1, -120.0, -20.0),
    Bin(0,   -20.0,  10.0),
    Bin(1,    10.0,  30.0, closed_hi=True),
]
HIP_Z = [
    Bin(1,  -45.0, -20.0, closed_hi=True),
    Bin(0,  -20.0,   0.0,),
]

ELBOW_Y = [
    Bin(0,   0.0,  25.0),
    Bin(1,  25.0, 140.0, closed_hi=True),
]
KNEE_X = [
    Bin(0,   0.0,  25.0),
    Bin(1,  25.0, 120.0, closed_hi=True),
]


JOINT_ORDER_12 = [
    "right_hip_x", "right_hip_z",
    "left_hip_x", "left_hip_z",
    "right_knee_x", "left_knee_x",
    "right_shoulder_x", "right_shoulder_z",
    "left_shoulder_x", "left_shoulder_z",
    "right_elbow_y", "left_elbow_y",
]


def f(row: pd.Series, key: str) -> float:
    """Fetch angle in degrees. Applies sign flip for specific keys before mapping."""
    v = float(row[key])
    if key in ("right_hip_z", "right_shoulder_z", "right_elbow_y"):
        return -v
    return v


def map_all_bins(row: pd.Series) -> Dict[str, int]:
    mapped: Dict[str, int] = {}

    # hips (x,z)
    mapped["right_hip_x"] = pick_bin(f(row, "right_hip_x"), HIP_X, "right_hip_x")
    mapped["right_hip_z"] = pick_bin(f(row, "right_hip_z"), HIP_Z, "right_hip_z (sign-flipped)")
    mapped["left_hip_x"] = pick_bin(f(row, "left_hip_x"), HIP_X, "left_hip_x")
    mapped["left_hip_z"] = pick_bin(f(row, "left_hip_z"), HIP_Z, "left_hip_z")

    # knees (x)
    mapped["right_knee_x"] = pick_bin(f(row, "right_knee_x"), KNEE_X, "right_knee_x")
    mapped["left_knee_x"] = pick_bin(f(row, "left_knee_x"), KNEE_X, "left_knee_x")

    # shoulders (x,z)
    mapped["right_shoulder_x"] = pick_bin(f(row, "right_shoulder_x"), SHOULDER_X, "right_shoulder_x")
    mapped["right_shoulder_z"] = pick_bin(f(row, "right_shoulder_z"), SHOULDER_Z, "right_shoulder_z (sign-flipped)")
    mapped["left_shoulder_x"] = pick_bin(f(row, "left_shoulder_x"), SHOULDER_X, "left_shoulder_x")
    mapped["left_shoulder_z"] = pick_bin(f(row, "left_shoulder_z"), SHOULDER_Z, "left_shoulder_z")

    # elbows (x)
    mapped["right_elbow_y"] = pick_bin(f(row, "right_elbow_y"), ELBOW_Y, "right_elbow_y (sign-flipped)")
    mapped["left_elbow_y"] = pick_bin(f(row, "left_elbow_y"), ELBOW_Y, "left_elbow_y")

    return mapped


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="8joints_02-05_11-14.csv", help="Path to CSV")
    ap.add_argument("--animframe", type=int, required=True, help="Target frame number (AnimFrame)")
    ap.add_argument("--pretty", action="store_true", help="Pretty-print JSON")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    if "AnimFrame" not in df.columns:
        print("[err] CSV has no 'AnimFrame' column", file=sys.stderr)
        return 2

    hit = df[df["AnimFrame"] == args.animframe]
    if hit.empty:
        frames = df["AnimFrame"].tolist()
        lo = int(min(frames)) if frames else None
        hi = int(max(frames)) if frames else None
        print(f"[err] animframe={args.animframe} not found. available range: {lo}..{hi}", file=sys.stderr)
        return 3

    row = hit.iloc[0]

    # raw angles (keep original CSV values)
    raw_angles_deg = {
        k: float(row[k])
        for k in df.columns
        if k not in ("AnimFrame", "AnimTime") and pd.notna(row[k])
    }

    mapped_all = map_all_bins(row)
    vec12 = [int(mapped_all[k]) for k in JOINT_ORDER_12]
    mapped_str = "{" + ",".join(str(v) for v in vec12) + "}"

    payload = {
        # "joint_order": JOINT_ORDER_12,
        "mapped": mapped_str,
        # "raw_angles_deg": raw_angles_deg,
    }

    if args.pretty:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
