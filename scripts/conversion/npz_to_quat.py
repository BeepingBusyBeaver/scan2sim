# scripts/conversion/npz_to_quat.py
import argparse
import re
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

import numpy as np
from scipy.spatial.transform import Rotation as R
from scripts.common.json_utils import write_json
from scripts.common.path_utils import natural_key_path
from scripts.common.io_paths import (
    DATA_INTEROP_QUAT_DIR,
    resolve_repo_path,
)

"""
Single:
python -m scripts.conversion.npz_to_quat \
  --input outputs/smpl/livehps_smpl_virtual.npz \
  --output outputs/quat/livehps_quaternion_virtual.json \
  --coord-system unity \
  --source-handedness lhs \
  --rotation-format quat \
  --pre_flip_yz

Batch:
python -m scripts.conversion.npz_to_quat \
  --input outputs/smpl \
  --batch_per_file \
  --input_pattern "livehps_smpl_*.npz" \
  --output_dir outputs/quat \
  --output_prefix "livehps_quaternion_" \
  --coord-system unity \
  --source-handedness lhs \
  --rotation-format quat \
  --pre_flip_yz
"""

DEFAULT_SMPL_24_JOINT_NAMES = [
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hand", "right_hand",
]


def pick_key(data, key, fallback=None):
    if key in data.files:
        return key
    if fallback:
        for k in fallback:
            if k in data.files:
                return k
    return None


def parse_joint_names(joint_names_arg: Optional[str]):
    if joint_names_arg is None:
        return DEFAULT_SMPL_24_JOINT_NAMES
    names = [x.strip() for x in joint_names_arg.split(",") if x.strip()]
    if len(names) != 24:
        raise ValueError(f"--joint-names must contain exactly 24 names, got {len(names)}.")
    return names


def pose_to_rotvec_seq(pose_array: np.ndarray) -> np.ndarray:
    """
    입력 pose를 [T,24,3] (radian rotvec)으로 정규화
    허용:
      - [72]
      - [24,3]
      - [T,72]
      - [T,24,3]
    """
    arr = np.asarray(pose_array, dtype=np.float32)

    if arr.ndim == 1:
        if arr.shape[0] != 72:
            raise ValueError(f"1D pose must have 72 values, got shape={arr.shape}")
        return arr.reshape(1, 24, 3)

    if arr.ndim == 2:
        if arr.shape == (24, 3):
            return arr.reshape(1, 24, 3)
        if arr.shape[1] == 72:
            return arr.reshape(arr.shape[0], 24, 3)
        raise ValueError(f"2D pose must be [24,3] or [T,72], got shape={arr.shape}")

    if arr.ndim == 3:
        if arr.shape[1:] == (24, 3):
            return arr
        raise ValueError(f"3D pose must be [T,24,3], got shape={arr.shape}")

    raise ValueError(f"Unsupported pose shape: {arr.shape}")


def trans_to_seq(trans_array: np.ndarray) -> np.ndarray:
    """
    trans를 [T,3]으로 정규화
    허용:
      - [3]
      - [T,3]
    """
    arr = np.asarray(trans_array, dtype=np.float32)

    if arr.ndim == 1:
        if arr.shape[0] != 3:
            raise ValueError(f"1D trans must have 3 values, got shape={arr.shape}")
        return arr.reshape(1, 3)

    if arr.ndim == 2:
        if arr.shape[1] == 3:
            return arr
        raise ValueError(f"2D trans must be [T,3], got shape={arr.shape}")

    raise ValueError(f"Unsupported trans shape: {arr.shape}")


def rhs_to_unity_vec3(vec_seq: np.ndarray) -> np.ndarray:
    """
    RHS -> Unity(LHS) 변환 (X,Y 유지, Z 반전)
    """
    out = np.asarray(vec_seq, dtype=np.float32).copy()
    out[..., 2] *= -1.0
    return out


def rhs_to_unity_rotvec_seq(rotvec_seq: np.ndarray) -> np.ndarray:
    """
    회전벡터(rh) -> 회전행렬 -> Unity basis로 유사변환 -> 회전벡터(lh)
    R_u = S * R_rh * S,  S = diag(1,1,-1)
    """
    rotvec_seq = np.asarray(rotvec_seq, dtype=np.float32)
    orig_shape = rotvec_seq.shape  # [T,24,3]

    mats = R.from_rotvec(rotvec_seq.reshape(-1, 3)).as_matrix().astype(np.float32)  # [M,3,3]
    S = np.diag([1.0, 1.0, -1.0]).astype(np.float32)

    mats_u = np.einsum("ab,mbc,cd->mad", S, mats, S)
    rotvec_u = R.from_matrix(mats_u).as_rotvec().astype(np.float32)

    return rotvec_u.reshape(orig_shape)


def flip_yz_vec3(vec_seq: np.ndarray) -> np.ndarray:
    """
    성분 기반 Y/Z 부호 반전
      (x, y, z) -> (x, -y, -z)
    """
    out = np.asarray(vec_seq, dtype=np.float32).copy()
    out[..., 1] *= -1.0
    out[..., 2] *= -1.0
    return out


def flip_yz_rotvec_seq(rotvec_seq: np.ndarray) -> np.ndarray:
    """
    rotvec에 Y/Z 반전 좌표계 변환 적용.
    회전 일관성을 위해 행렬 유사변환으로 처리:
      R' = C R C^T,  C = diag(1, -1, -1)
    """
    rv = np.asarray(rotvec_seq, dtype=np.float32)
    if rv.shape[-1] != 3:
        raise ValueError(f"Expected rotvec last dim=3, got shape={rv.shape}")

    orig_shape = rv.shape
    mats = R.from_rotvec(rv.reshape(-1, 3)).as_matrix().astype(np.float32)
    C = np.diag([1.0, -1.0, -1.0]).astype(np.float32)
    mats2 = np.einsum("ab,mbc,cd->mad", C, mats, C.T)
    rv2 = R.from_matrix(mats2).as_rotvec().astype(np.float32)
    return rv2.reshape(orig_shape)


def flip_x_vec3(vec_seq: np.ndarray) -> np.ndarray:
    """
    성분 기반 X 부호 반전 (x 값만 반전)
      (x, y, z) -> (-x, y, z)
    """
    out = np.asarray(vec_seq, dtype=np.float32).copy()
    out[..., 0] *= -1.0
    return out


def flip_x_rotvec_seq(rotvec_seq: np.ndarray) -> np.ndarray:
    """
    성분 기반 rotvec X 부호 반전
      (..., x, y, z) -> (..., -x, y, z)
    """
    rv = np.asarray(rotvec_seq, dtype=np.float32).copy()
    if rv.shape[-1] != 3:
        raise ValueError(f"Expected rotvec last dim=3, got shape={rv.shape}")
    rv[..., 0] *= -1.0
    return rv


def rotvec_seq_to_quat_seq(
    rotvec_seq: np.ndarray,
    canonical: bool = True,
    continuous: bool = True,
) -> np.ndarray:
    """
    [T,24,3] rotvec(rad) -> [T,24,4] quat(x,y,z,w)
    - canonical: w>=0 정규화(가능한 경우)
    - continuous: 프레임간 부호 튐 완화(q와 -q는 동일 회전)
    """
    T, J, _ = rotvec_seq.shape
    r = R.from_rotvec(rotvec_seq.reshape(-1, 3))

    # SciPy 버전에 따라 canonical 인자 지원 여부가 다를 수 있어 fallback 처리
    try:
        quat = r.as_quat(canonical=canonical).astype(np.float32)
    except TypeError:
        quat = r.as_quat().astype(np.float32)
        if canonical:
            sign = np.where(quat[:, 3:4] < 0.0, -1.0, 1.0).astype(np.float32)
            quat *= sign

    quat = quat.reshape(T, J, 4)

    # q와 -q 동치 특성 때문에 시퀀스에서 부호 점프 방지
    if continuous and T > 1:
        for j in range(J):
            for t in range(1, T):
                if np.dot(quat[t - 1, j], quat[t, j]) < 0.0:
                    quat[t, j] *= -1.0

    return quat.astype(np.float32)


def build_joint_xyz_deg(rotvec_seq: np.ndarray, joint_names: List[str]):
    """
    rotvec_seq: [T,24,3] (radian)
    반환:
      - 길이 T의 리스트
      - 각 프레임은 {joint_name: {"x":..,"y":..,"z":..}} (degree)
    """
    rotvec_deg_seq = np.degrees(rotvec_seq).astype(np.float32)
    out = []

    T = rotvec_deg_seq.shape[0]
    for t in range(T):
        frame = {}
        for j, name in enumerate(joint_names):
            x, y, z = rotvec_deg_seq[t, j]
            frame[name] = {
                "x": float(x),
                "y": float(y),
                "z": float(z),
            }
        out.append(frame)

    return out


def build_joint_quat_xyzw(quat_seq: np.ndarray, joint_names: List[str]):
    """
    quat_seq: [T,24,4] (x,y,z,w)
    반환:
      - 길이 T의 리스트
      - 각 프레임은 {joint_name: {"x":..,"y":..,"z":..,"w":..}}
    """
    out = []
    T = quat_seq.shape[0]
    for t in range(T):
        frame = {}
        for j, name in enumerate(joint_names):
            x, y, z, w = quat_seq[t, j]
            frame[name] = {
                "x": float(x),
                "y": float(y),
                "z": float(z),
                "w": float(w),
            }
        out.append(frame)
    return out


def seq_to_pose_array(rotvec_seq: np.ndarray):
    T = rotvec_seq.shape[0]
    flat = rotvec_seq.reshape(T, 72).astype(np.float32)
    return flat[0] if T == 1 else flat


def seq_to_trans_array(trans_seq: np.ndarray):
    T = trans_seq.shape[0]
    out = trans_seq.astype(np.float32)
    return out[0] if T == 1 else out


def make_batch_output_name(input_npz: Path, output_prefix: str) -> str:
    """
    livehps_smpl_123.npz -> livehps_quaternion_123.json (default prefix 기준)
    """
    stem = input_npz.stem
    if stem.startswith("livehps_smpl_"):
        tail = stem[len("livehps_smpl_"):]
    else:
        # 끝 숫자가 있으면 그걸 tail로, 없으면 stem 전체 사용
        m = re.search(r"(\d+)$", stem)
        tail = m.group(1) if m else stem
    return f"{output_prefix}{tail}.json"


def collect_npz_paths(input_path: Path, pattern: str) -> List[Path]:
    """
    --batch_per_file 모드용: 디렉토리에서 패턴에 맞는 .npz 수집
    """
    if not input_path.is_dir():
        raise ValueError("--batch_per_file requires --input to be a directory.")
    paths = sorted(
        [p for p in input_path.glob(pattern) if p.is_file() and p.suffix.lower() == ".npz"],
        key=natural_key_path,
    )
    if not paths:
        raise ValueError(f"No .npz files found in {input_path} with pattern={pattern}")
    return paths


def convert_npz_to_payload(
    in_path: Path,
    allow_pickle: bool,
    joint_names: List[str],
    coord_system: str,
    source_handedness: str,
    pre_mirror_x: bool,
    pre_flip_yz: bool,
    rotation_format: str,
    trans_key_option: str,
    no_quat_canonical: bool,
    no_quat_continuity: bool,
) -> Tuple[Dict[str, Any], int]:
    with np.load(in_path, allow_pickle=allow_pickle) as data:
        pose_k = pick_key(data, "pose", fallback=["pose_seq"])
        betas_k = pick_key(data, "betas")

        if trans_key_option == "auto":
            trans_k = pick_key(data, "trans_world", fallback=["trans_world_seq", "trans", "trans_seq"])
        elif trans_key_option == "trans":
            trans_k = pick_key(data, "trans", fallback=["trans_seq"])
        else:  # trans_world
            trans_k = pick_key(data, "trans_world", fallback=["trans_world_seq"])

        missing = []
        if pose_k is None:
            missing.append("pose (or pose_seq)")
        if betas_k is None:
            missing.append("betas")
        if trans_k is None:
            missing.append("trans/trans_seq or trans_world/trans_world_seq")
        if missing:
            raise KeyError(f"Missing required keys: {', '.join(missing)} / available={data.files}")

        pose_arr = np.asarray(data[pose_k], dtype=np.float32)
        betas_arr = np.asarray(data[betas_k], dtype=np.float32)
        trans_arr = np.asarray(data[trans_k], dtype=np.float32)

        rotvec_seq = pose_to_rotvec_seq(pose_arr)   # [T,24,3], rad
        trans_seq = trans_to_seq(trans_arr)         # [T,3]

        T = rotvec_seq.shape[0]
        if trans_seq.shape[0] == 1 and T > 1:
            trans_seq = np.repeat(trans_seq, T, axis=0)
        elif trans_seq.shape[0] != T:
            raise ValueError(
                f"Frame mismatch: pose has T={T}, trans has T={trans_seq.shape[0]}. "
                f"Check --trans-key or input npz."
            )

        # 좌표계 변환
        coord_meta = "source"
        if coord_system == "unity":
            if source_handedness == "rhs":
                rotvec_seq = rhs_to_unity_rotvec_seq(rotvec_seq)
                trans_seq = rhs_to_unity_vec3(trans_seq)
                coord_meta = "unity_left_handed (from_rhs: z_flipped)"
            else:
                coord_meta = "unity_left_handed (source_already_lhs)"

        # 선택: pre_flip_yz (y/z 성분 반전)
        if pre_flip_yz:
            rotvec_seq = flip_yz_rotvec_seq(rotvec_seq)
            trans_seq = flip_yz_vec3(trans_seq)
            coord_meta = f"{coord_meta} + pre_flip_yz(yz_sign)"

        # 선택: pre_mirror_x (x 성분 부호만 반전)
        if pre_mirror_x:
            rotvec_seq = flip_x_rotvec_seq(rotvec_seq)
            trans_seq = flip_x_vec3(trans_seq)
            coord_meta = f"{coord_meta} + pre_mirror_x(x_sign_only)"

        pose_out = seq_to_pose_array(rotvec_seq)      # [72] or [T,72], rad
        trans_out = seq_to_trans_array(trans_seq)     # [3] or [T,3]

        payload = {
            "pose": pose_out.tolist(),
            "betas": betas_arr.tolist(),
            "trans": trans_out.tolist(),
            "meta": {
                "coord_system": coord_meta,
                "rotation_vector_unit": "radian",
                "rotation_format": rotation_format,
                "quaternion_order": "xyzw" if rotation_format in ("quat", "both") else None,
                "pose_key_used": pose_k,
                "trans_key_used": trans_k,
                "pre_mirror_x": bool(pre_mirror_x),
                "pre_mirror_x_mode": "x_sign_only",
                "pre_flip_yz": bool(pre_flip_yz),
                "pre_flip_yz_mode": "y_z_sign_flip",
            },
        }

        # 선택: pcd_files 전달
        if "pcd_files" in data.files:
            pcd_files = np.asarray(data["pcd_files"])
            payload["pcd_files"] = [str(x) for x in pcd_files.tolist()]

        # joint rotvec(deg)
        if rotation_format in ("rotvec_deg", "both"):
            joint_xyz_deg_seq = build_joint_xyz_deg(rotvec_seq, joint_names)
            if T == 1:
                payload["joint_xyz_deg"] = joint_xyz_deg_seq[0]
            else:
                payload["joint_xyz_deg_seq"] = joint_xyz_deg_seq

        # joint quaternion(x,y,z,w)
        if rotation_format in ("quat", "both"):
            quat_seq = rotvec_seq_to_quat_seq(
                rotvec_seq,
                canonical=not no_quat_canonical,
                continuous=not no_quat_continuity,
            )
            joint_quat_seq = build_joint_quat_xyzw(quat_seq, joint_names)
            if T == 1:
                payload["joint_quat_xyzw"] = joint_quat_seq[0]
            else:
                payload["joint_quat_xyzw_seq"] = joint_quat_seq

        return payload, T


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract pose/betas/trans and joint rotations from LiveHPS .npz (Unity/quaternion supported)"
    )

    # Input/Output
    parser.add_argument("--input", required=True, help="Input .npz file path OR directory (with --batch_per_file)")
    parser.add_argument("--output", default=None, help="Output .json path (single mode only)")
    parser.add_argument("--allow-pickle", action="store_true", help="np.load(..., allow_pickle=True)")
    parser.add_argument("--ascii", action="store_true", help="JSON ASCII escape")
    parser.add_argument("--compact", action="store_true", help="Compact JSON (no pretty indent)")

    # Batch options
    parser.add_argument(
        "--batch_per_file",
        action="store_true",
        help="If set, convert all matching .npz files in --input directory to one .json per file.",
    )
    parser.add_argument(
        "--input_pattern",
        type=str,
        default="livehps_smpl_*.npz",
        help="Glob pattern in --batch_per_file mode.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=(
            "Output directory in --batch_per_file mode. "
            "If omitted: use outputs/quat."
        ),
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="livehps_quaternion_",
        help="Output filename prefix in --batch_per_file mode.",
    )

    # Joint names
    parser.add_argument(
        "--joint-names",
        type=str,
        default=None,
        help="Comma-separated 24 joint names in pose order",
    )

    # Coordinate / format options
    parser.add_argument(
        "--coord-system",
        type=str,
        choices=["unity", "source"],
        default="unity",
        help="Output coordinate system. default=unity",
    )
    parser.add_argument(
        "--source-handedness",
        type=str,
        choices=["rhs", "lhs"],
        default="lhs",
        help="Handedness of input pose/trans. default=lhs",
    )
    parser.add_argument(
        "--pre_mirror_x",
        action="store_true",
        help="Apply X-sign flip only (x -> -x) to rotvec/trans before JSON export.",
    )
    parser.add_argument(
        "--pre_flip_yz",
        action="store_true",
        help="Apply Y/Z-sign flip (y,z -> -y,-z) to rotvec/trans before JSON export.",
    )
    parser.add_argument(
        "--rotation-format",
        type=str,
        choices=["rotvec_deg", "quat", "both"],
        default="quat",
        help="Joint rotation output format",
    )
    parser.add_argument(
        "--trans-key",
        type=str,
        choices=["auto", "trans", "trans_world"],
        default="auto",
        help="Which translation key to use from npz",
    )
    parser.add_argument(
        "--no-quat-canonical",
        action="store_true",
        help="Do not force quaternion w >= 0",
    )
    parser.add_argument(
        "--no-quat-continuity",
        action="store_true",
        help="Do not fix sign jumps across frames",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]

    input_path = resolve_repo_path(repo_root, args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    joint_names = parse_joint_names(args.joint_names)

    # -------------------------
    # Batch mode
    # -------------------------
    if args.batch_per_file:
        npz_paths = collect_npz_paths(input_path, args.input_pattern)

        if args.output_dir is not None:
            out_dir = resolve_repo_path(repo_root, args.output_dir)
        else:
            out_dir = resolve_repo_path(repo_root, DATA_INTEROP_QUAT_DIR)

        out_dir.mkdir(parents=True, exist_ok=True)

        total = len(npz_paths)
        for i, in_npz in enumerate(npz_paths, start=1):
            out_name = make_batch_output_name(in_npz, args.output_prefix)
            out_json = out_dir / out_name

            payload, T = convert_npz_to_payload(
                in_path=in_npz,
                allow_pickle=args.allow_pickle,
                joint_names=joint_names,
                coord_system=args.coord_system,
                source_handedness=args.source_handedness,
                pre_mirror_x=args.pre_mirror_x,
                pre_flip_yz=args.pre_flip_yz,
                rotation_format=args.rotation_format,
                trans_key_option=args.trans_key,
                no_quat_canonical=args.no_quat_canonical,
                no_quat_continuity=args.no_quat_continuity,
            )

            write_json(
                path=out_json,
                payload=payload,
                ascii_flag=args.ascii,
                compact=args.compact,
            )

            print(
                f"[{i:04d}/{total:04d}] {in_npz.name} -> {out_json.name} "
                f"(frames={T}, rotation_format={args.rotation_format})"
            )

        print(f"Done. processed={total}, output_dir={out_dir}")
        return

    # -------------------------
    # Single-file mode
    # -------------------------
    if input_path.is_dir():
        raise ValueError(
            "Single-file mode requires --input to be a .npz file. "
            "If you passed a directory, use --batch_per_file."
        )

    if input_path.suffix.lower() != ".npz":
        raise ValueError(f"--input must be a .npz file, got: {input_path}")

    if args.output is None:
        raise ValueError("--output is required in single-file mode.")

    out_path = resolve_repo_path(repo_root, args.output)

    payload, T = convert_npz_to_payload(
        in_path=input_path,
        allow_pickle=args.allow_pickle,
        joint_names=joint_names,
        coord_system=args.coord_system,
        source_handedness=args.source_handedness,
        pre_mirror_x=args.pre_mirror_x,
        pre_flip_yz=args.pre_flip_yz,
        rotation_format=args.rotation_format,
        trans_key_option=args.trans_key,
        no_quat_canonical=args.no_quat_canonical,
        no_quat_continuity=args.no_quat_continuity,
    )

    write_json(
        path=out_path,
        payload=payload,
        ascii_flag=args.ascii,
        compact=args.compact,
    )

    print(f"Saved JSON: {out_path}")
    print(f"coord_system={args.coord_system}, source_handedness={args.source_handedness}")
    print(f"rotation_format={args.rotation_format}, frames={T}")
    print(f"pre_mirror_x={args.pre_mirror_x} (mode=x_sign_only)")
    print(f"pre_flip_yz={args.pre_flip_yz} (mode=y_z_sign_flip)")


if __name__ == "__main__":
    main()
