# scripts/conversion/quat_to_unity.py
import argparse
import re
from pathlib import Path
from typing import Any, Dict, List

from scripts.common.coord_utils import (
    FRAME_OUSTER_FLU_RHS,
    FRAME_UNITY_RUF_LHS,
    SUPPORTED_SOURCE_FRAMES,
    infer_source_frame_from_meta,
    quat_xyzw_to_inspector_xyz_deg,
    source_quat_xyzw_to_unity_inspector_xyz_deg,
)
from scripts.common.livehps_defaults import (
    PIPELINE_DEFAULT_SOURCE_FRAME,
    QUAT2UNITY_DEFAULT_BASIS_CONVERSION,
)
from scripts.common.json_utils import read_json, write_json
from scripts.common.path_utils import natural_key_path
from scripts.common.io_paths import (
    DATA_INTEROP_EULER_DIR,
    DATA_INTEROP_EULER_JSON,
    DATA_INTEROP_QUAT_JSON,
    resolve_repo_path,
)

"""

ROOT_JOINT_CANDIDATES = {"pelvis", "root", "hips"}
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

def quat_dict_to_xyzw(qd: Dict[str, Any]) -> List[float]:
    required = ("x", "y", "z", "w")
    if not isinstance(qd, dict):
        raise TypeError(f"Quaternion entry must be dict, got {type(qd)}")
    for k in required:
        if k not in qd:
            raise KeyError(f"Quaternion dict missing key '{k}'")
    return [float(qd["x"]), float(qd["y"]), float(qd["z"]), float(qd["w"])]


def quat_dict_to_inspector_xyz_deg(
    qd: Dict[str, Any],
    source_frame: str,
    apply_basis_conversion: bool,
) -> Dict[str, float]:
    """
    Source quaternion(xyzw) -> Unity 좌표계 변환 ->
    Unity Inspector Rotation XYZ(Local) 기준 각도.
    """
    quat_xyzw = quat_dict_to_xyzw(qd)
    if apply_basis_conversion:
        xyz_deg = source_quat_xyzw_to_unity_inspector_xyz_deg(
            quat_xyzw,
            source_frame=source_frame,
        )
    else:
        xyz_deg = quat_xyzw_to_inspector_xyz_deg(quat_xyzw)
    return {
        "x": float(xyz_deg[0]),
        "y": float(xyz_deg[1]),
        "z": float(xyz_deg[2]),
    }


def convert_one_frame(
    joint_quat_xyzw: Dict[str, Any],
    source_frame: str,
    basis_conversion: str,
) -> Dict[str, Dict[str, float]]:
    if not isinstance(joint_quat_xyzw, dict):
        raise TypeError(f"Frame must be dict[joint_name -> quat], got {type(joint_quat_xyzw)}")
    if basis_conversion not in {"none", "root_only", "all"}:
        raise ValueError(f"Unsupported basis_conversion: {basis_conversion}")

    out: Dict[str, Dict[str, float]] = {}
    for jname, qd in joint_quat_xyzw.items():
        if basis_conversion == "none":
            use_basis_conversion = False
        elif basis_conversion == "all":
            use_basis_conversion = True
        else:
            use_basis_conversion = str(jname).strip().lower() in ROOT_JOINT_CANDIDATES
        out[jname] = quat_dict_to_inspector_xyz_deg(
            qd,
            source_frame=source_frame,
            apply_basis_conversion=use_basis_conversion,
        )
    return out


def infer_source_frame(data: Dict[str, Any], source_frame_arg: str) -> str:
    if source_frame_arg != "auto":
        return source_frame_arg
    return infer_source_frame_from_meta(data.get("meta"), fallback=FRAME_OUSTER_FLU_RHS)


def convert_payload(
    data: Dict[str, Any],
    source_frame: str,
    basis_conversion: str,
    keep_pcd_files: bool = True,
) -> Dict[str, Any]:
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
            "joint_euler_xyz_deg_seq": [
                convert_one_frame(
                    frame,
                    source_frame=source_frame,
                    basis_conversion=basis_conversion,
                )
                for frame in frames
            ]
        }

    elif "joint_quat_xyzw" in data:
        payload = {
            "joint_euler_xyz_deg": convert_one_frame(
                data["joint_quat_xyzw"],
                source_frame=source_frame,
                basis_conversion=basis_conversion,
            )
        }

    else:
        raise KeyError("joint_quat_xyzw 또는 joint_quat_xyzw_seq 키가 없습니다.")

    payload["meta"] = {
        "source_frame": source_frame,
        "target_frame": FRAME_UNITY_RUF_LHS,
        "euler_order": "zxy",
        "euler_range_deg": "[0, 360)",
        "rotation_target": "unity_inspector_rotation_xyz_local",
        "basis_conversion": basis_conversion,
    }

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
    parser.add_argument(
        "--source-frame",
        type=str,
        default=PIPELINE_DEFAULT_SOURCE_FRAME,
        choices=["auto", *SUPPORTED_SOURCE_FRAMES],
        help=f"Quaternion source frame. 'auto' reads payload meta and falls back to {FRAME_OUSTER_FLU_RHS}.",
    )
    parser.add_argument(
        "--basis-conversion",
        type=str,
        default=QUAT2UNITY_DEFAULT_BASIS_CONVERSION,
        choices=["none", "root_only", "all"],
        help=(
            "Before Unity Inspector Euler conversion, apply source->Unity basis conversion to "
            "none (recommended for SMPL local joint rotations), root_only, or all joints."
        ),
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
            source_frame = infer_source_frame(data, source_frame_arg=args.source_frame)
            payload = convert_payload(
                data,
                source_frame=source_frame,
                basis_conversion=args.basis_conversion,
                keep_pcd_files=not args.no_pcd_files,
            )
            write_json(dst, payload, ascii_flag=args.ascii, compact=args.compact)

            if "joint_euler_xyz_deg_seq" in payload:
                frames = len(payload["joint_euler_xyz_deg_seq"])
                out_key = "joint_euler_xyz_deg_seq"
            else:
                frames = 1
                out_key = "joint_euler_xyz_deg"

            print(
                f"[{i:04d}/{total:04d}] {src.name} -> {dst.name} "
                f"(frames={frames}, key={out_key}, source_frame={source_frame}, basis={args.basis_conversion})"
            )

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
    source_frame = infer_source_frame(data, source_frame_arg=args.source_frame)
    payload = convert_payload(
        data,
        source_frame=source_frame,
        basis_conversion=args.basis_conversion,
        keep_pcd_files=not args.no_pcd_files,
    )
    write_json(out_path, payload, ascii_flag=args.ascii, compact=args.compact)

    print(f"saved: {out_path}")
    print(f"keys: {list(payload.keys())}")
    print(f"source_frame={source_frame} -> target_frame={FRAME_UNITY_RUF_LHS}")
    print(f"basis_conversion={args.basis_conversion}")


if __name__ == "__main__":
    main()
