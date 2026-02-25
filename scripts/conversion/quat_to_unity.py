# scripts/conversion/quat_to_unity.py
import argparse
import re
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from scipy.spatial.transform import Rotation as R
from scripts.common.angle_utils import wrap180
from scripts.common.json_utils import read_json, write_json
from scripts.common.path_utils import natural_key_path
from scripts.common.io_paths import (
    DATA_INTEROP_EULER_DIR,
    DATA_INTEROP_EULER_JSON,
    DATA_INTEROP_QUAT_JSON,
    resolve_repo_path,
)

"""
Single:
python -m scripts.conversion.quat_to_unity \
  --input outputs/quat/livehps_quaternion_virtual.json \
  --output outputs/euler/livehps_unity_virtual.json

Batch:
python -m scripts.conversion.quat_to_unity \
  --input outputs/quat \
  --batch_per_file \
  --input_pattern "livehps_quaternion_*.json" \
  --output_dir outputs/euler \
  --output_prefix "livehps_unity_"
"""

def quat_dict_to_unity_xyz_deg(qd: Dict[str, Any]) -> Dict[str, float]:
    """
    입력 quat은 xyzw.
    Unity Euler 순서(ZXY)에 맞춰 추출:
      zxy = as_euler("zxy", degrees=True) -> [z, x, y]
      출력 xyz는 각각 [x=zxy[1], y=zxy[2], z=zxy[0]]
    """
    required = ("x", "y", "z", "w")
    if not isinstance(qd, dict):
        raise TypeError(f"Quaternion entry must be dict, got {type(qd)}")
    for k in required:
        if k not in qd:
            raise KeyError(f"Quaternion dict missing key '{k}'")

    q = np.array([qd["x"], qd["y"], qd["z"], qd["w"]], dtype=np.float64)
    n = float(np.linalg.norm(q))
    if n <= 1e-12:
        raise ValueError("Invalid quaternion: norm is zero.")
    q /= n  # 안전하게 정규화

    zxy = R.from_quat(q).as_euler("zxy", degrees=True)
    x_deg = wrap180(float(zxy[1]))
    y_deg = wrap180(float(zxy[2]))
    z_deg = wrap180(float(zxy[0]))

    return {"x": x_deg, "y": y_deg, "z": z_deg}


def convert_one_frame(joint_quat_xyzw: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    if not isinstance(joint_quat_xyzw, dict):
        raise TypeError(f"Frame must be dict[joint_name -> quat], got {type(joint_quat_xyzw)}")

    out: Dict[str, Dict[str, float]] = {}
    for jname, qd in joint_quat_xyzw.items():
        out[jname] = quat_dict_to_unity_xyz_deg(qd)
    return out


def convert_payload(data: Dict[str, Any], keep_pcd_files: bool = True) -> Dict[str, Any]:
    """
    입력 JSON에서
      - joint_quat_xyzw_seq (다중 프레임) 또는
      - joint_quat_xyzw     (단일 프레임)
    를 읽어, 유니티 euler xyz(deg)만 출력 payload로 구성.
    """
    if "joint_quat_xyzw_seq" in data:
        frames = data["joint_quat_xyzw_seq"]
        if not isinstance(frames, list):
            raise TypeError("joint_quat_xyzw_seq는 list여야 합니다.")
        payload: Dict[str, Any] = {
            "joint_euler_xyz_deg_seq": [convert_one_frame(frame) for frame in frames]
        }

    elif "joint_quat_xyzw" in data:
        payload = {
            "joint_euler_xyz_deg": convert_one_frame(data["joint_quat_xyzw"])
        }

    else:
        raise KeyError("joint_quat_xyzw 또는 joint_quat_xyzw_seq 키가 없습니다.")

    if keep_pcd_files and "pcd_files" in data:
        pcd_files = data["pcd_files"]
        if isinstance(pcd_files, list):
            payload["pcd_files"] = [str(x) for x in pcd_files]
        else:
            payload["pcd_files"] = [str(x) for x in list(pcd_files)]

    return payload


def collect_json_paths(input_dir: Path, pattern: str) -> List[Path]:
    if not input_dir.is_dir():
        raise ValueError("--batch_per_file 모드에서 --input은 디렉터리여야 합니다.")
    paths = sorted(
        [p for p in input_dir.glob(pattern) if p.is_file() and p.suffix.lower() == ".json"],
        key=natural_key_path,
    )
    if not paths:
        raise ValueError(f"No .json files found in {input_dir} with pattern={pattern}")
    return paths


def make_batch_output_name(input_json: Path, output_prefix: str) -> str:
    """
    livehps_quaternion_123.json -> livehps_unity_123.json (기본 동작)
    prefix가 다르면 숫자 suffix를 우선 사용, 없으면 stem 전체 사용
    """
    stem = input_json.stem
    if stem.startswith("livehps_quaternion_"):
        tail = stem[len("livehps_quaternion_"):]
    else:
        m = re.search(r"(\d+)$", stem)
        tail = m.group(1) if m else stem
    return f"{output_prefix}{tail}.json"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert LiveHPS quaternion JSON to Unity Euler XYZ(deg) JSON"
    )

    # Single mode I/O
    parser.add_argument(
        "--input",
        type=str,
        default=DATA_INTEROP_QUAT_JSON,
        help="Input .json file path OR directory (with --batch_per_file)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DATA_INTEROP_EULER_JSON,
        help="Output .json file path (single mode)",
    )

    # Batch mode
    parser.add_argument(
        "--batch_per_file",
        action="store_true",
        help="If set, process all matching files in input directory.",
    )
    parser.add_argument(
        "--input_pattern",
        type=str,
        default="livehps_quaternion_*.json",
        help="Glob pattern used in --batch_per_file mode.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DATA_INTEROP_EULER_DIR,
        help="Output directory in batch mode.",
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="livehps_unity_",
        help="Output filename prefix in batch mode.",
    )

    # JSON formatting
    parser.add_argument("--ascii", action="store_true", help="JSON ASCII escape")
    parser.add_argument("--compact", action="store_true", help="Compact JSON (no pretty indent)")

    # Payload options
    parser.add_argument(
        "--no_pcd_files",
        action="store_true",
        help="Do not copy 'pcd_files' from source JSON even if present.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]

    in_path = resolve_repo_path(repo_root, args.input)

    if args.batch_per_file:
        # batch mode
        if not in_path.exists():
            raise FileNotFoundError(f"Input directory not found: {in_path}")

        json_paths = collect_json_paths(in_path, args.input_pattern)

        out_dir = resolve_repo_path(repo_root, args.output_dir)

        out_dir.mkdir(parents=True, exist_ok=True)

        total = len(json_paths)
        for i, src in enumerate(json_paths, start=1):
            dst_name = make_batch_output_name(src, args.output_prefix)
            dst = out_dir / dst_name

            data = read_json(src)
            payload = convert_payload(data, keep_pcd_files=not args.no_pcd_files)
            write_json(dst, payload, ascii_flag=args.ascii, compact=args.compact)

            if "joint_euler_xyz_deg_seq" in payload:
                frames = len(payload["joint_euler_xyz_deg_seq"])
                out_key = "joint_euler_xyz_deg_seq"
            else:
                frames = 1
                out_key = "joint_euler_xyz_deg"

            print(f"[{i:04d}/{total:04d}] {src.name} -> {dst.name} (frames={frames}, key={out_key})")

        print(f"Done. processed={total}, output_dir={out_dir}")
        return

    # single mode
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")
    if in_path.is_dir():
        raise ValueError(
            "Single mode requires --input to be a JSON file. "
            "If input is directory, use --batch_per_file."
        )
    if in_path.suffix.lower() != ".json":
        raise ValueError(f"Input must be .json file, got: {in_path}")

    out_path = resolve_repo_path(repo_root, args.output)

    data = read_json(in_path)
    payload = convert_payload(data, keep_pcd_files=not args.no_pcd_files)
    write_json(out_path, payload, ascii_flag=args.ascii, compact=args.compact)

    print(f"saved: {out_path}")
    print(f"keys: {list(payload.keys())}")


if __name__ == "__main__":
    main()
