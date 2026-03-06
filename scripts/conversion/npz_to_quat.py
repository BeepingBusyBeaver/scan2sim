# scripts/conversion/npz_to_quat.py
import argparse
import re
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

import numpy as np
from scipy.spatial.transform import Rotation as R
from scripts.common.coord_utils import (
    FRAME_OUSTER_FLU_RHS,
    FRAME_UNITY_RUF_LHS,
    SUPPORTED_SOURCE_FRAMES,
    convert_rotvec_to_unity,
    convert_vec3_to_unity,
)
from scripts.common.livehps_defaults import (
    DEFAULT_UNITY_BIND_LOCAL_QUAT_JSON,
    NPZ2QUAT_DEFAULT_BIND_COMPOSE_ORDER,
    NPZ2QUAT_DEFAULT_COORD_SYSTEM,
    NPZ2QUAT_DEFAULT_UNITY_POSE_MODE,
    PIPELINE_DEFAULT_SOURCE_FRAME,
)
from scripts.common.json_utils import read_json, write_json
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
  --source-frame ouster_flu_rhs \
  --unity-pose-mode root_only \
  --rotation-format quat \

Batch:
python -m scripts.conversion.npz_to_quat \
  --input outputs/smpl \
  --batch_per_file \
  --input_pattern "livehps_smpl_*.npz" \
  --output_dir outputs/quat \
  --output_prefix "livehps_quaternion_" \
  --coord-system unity \
  --source-frame ouster_flu_rhs \
  --unity-pose-mode root_only \
  --rotation-format quat \
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


def read_npz_scalar_str(data, key: str) -> Optional[str]:
    if key not in data.files:
        return None
    arr = np.asarray(data[key])
    if arr.size == 0:
        return None
    return str(arr.reshape(-1)[0])


def parse_joint_names(joint_names_arg: Optional[str]):
    if joint_names_arg is None:
        return DEFAULT_SMPL_24_JOINT_NAMES
    names = [x.strip() for x in joint_names_arg.split(",") if x.strip()]
    if len(names) != 24:
        raise ValueError(f"--joint-names must contain exactly 24 names, got {len(names)}.")
    return names


def infer_source_frame_for_npz(
    data: Any,
    source_frame_option: str,
    source_handedness: str,
) -> str:
    if source_frame_option != "auto":
        return source_frame_option

    source_frame = read_npz_scalar_str(data, "meta_source_frame")
    if source_frame in SUPPORTED_SOURCE_FRAMES:
        return source_frame

    legacy_frame = read_npz_scalar_str(data, "meta_legacy_frame_name")
    if legacy_frame in SUPPORTED_SOURCE_FRAMES:
        return legacy_frame

    if source_handedness == "lhs":
        return FRAME_UNITY_RUF_LHS
    return FRAME_OUSTER_FLU_RHS


def convert_pose_seq_to_unity(rotvec_seq: np.ndarray, source_frame: str, unity_pose_mode: str) -> np.ndarray:
    rv = np.asarray(rotvec_seq, dtype=np.float32).copy()
    if source_frame == FRAME_UNITY_RUF_LHS:
        return rv

    if source_frame != FRAME_OUSTER_FLU_RHS:
        raise ValueError(f"Unsupported source frame for Unity conversion: {source_frame}")

    if unity_pose_mode == "all_joints":
        return convert_rotvec_to_unity(rv, source_frame=source_frame).astype(np.float32)

    if unity_pose_mode == "root_only":
        rv[:, 0, :] = convert_rotvec_to_unity(rv[:, 0, :], source_frame=source_frame).astype(np.float32)
        return rv

    raise ValueError(f"Unsupported --unity-pose-mode: {unity_pose_mode}")


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
    - canonical: w>=0 정규화 + (w==0일 때 x/y/z 사전식 tie-break)
    - continuous: 프레임간 부호 튐 완화(q와 -q는 동일 회전)
    """

    def canonicalize_sign_with_tie_break(quat_xyzw: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """
        Deterministic canonical sign:
          1) w > 0 우선
          2) w == 0이면 x, y, z 순서로 첫 non-zero가 양수가 되도록 선택
        """
        q = np.asarray(quat_xyzw, dtype=np.float32).copy()
        flat = q.reshape(-1, 4)

        w = flat[:, 3]
        x = flat[:, 0]
        y = flat[:, 1]
        z = flat[:, 2]

        w_neg = w < -eps
        w_zero = np.abs(w) <= eps

        x_neg = x < -eps
        x_zero = np.abs(x) <= eps
        y_neg = y < -eps
        y_zero = np.abs(y) <= eps
        z_neg = z < -eps

        tie_flip = w_zero & (
            x_neg
            | (x_zero & y_neg)
            | (x_zero & y_zero & z_neg)
        )

        flip_mask = w_neg | tie_flip
        flat[flip_mask] *= -1.0
        return flat.reshape(q.shape)

    T, J, _ = rotvec_seq.shape
    r = R.from_rotvec(rotvec_seq.reshape(-1, 3))

    # SciPy 버전에 따라 canonical 인자 지원 여부가 다를 수 있어 fallback 처리
    try:
        quat = r.as_quat(canonical=canonical).astype(np.float32)
    except TypeError:
        quat = r.as_quat().astype(np.float32)
    if canonical:
        quat = canonicalize_sign_with_tie_break(quat)

    quat = quat.reshape(T, J, 4)

    # q와 -q 동치 특성 때문에 시퀀스에서 부호 점프 방지
    if continuous and T > 1:
        eps = 1e-8
        for j in range(J):
            for t in range(1, T):
                dot = float(np.dot(quat[t - 1, j], quat[t, j]))
                if dot < -eps:
                    quat[t, j] *= -1.0
                elif canonical and abs(dot) <= eps:
                    quat[t, j] = canonicalize_sign_with_tie_break(quat[t, j])

    return quat.astype(np.float32)


def quat_multiply_xyzw(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Hamilton product for xyzw quaternions.
    """
    q1 = np.asarray(q1, dtype=np.float32)
    q2 = np.asarray(q2, dtype=np.float32)

    x1, y1, z1, w1 = np.moveaxis(q1, -1, 0)
    x2, y2, z2, w2 = np.moveaxis(q2, -1, 0)

    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    return np.stack([x, y, z, w], axis=-1).astype(np.float32)


def normalize_quat_xyzw_array(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float32)
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(n <= 1e-12):
        raise ValueError("Invalid quaternion norm (<= 1e-12) found while normalizing.")
    return (q / n).astype(np.float32)


def parse_joint_quat_dict(quat_dict: Dict[str, Any], joint_names: List[str]) -> np.ndarray:
    quat_arr = np.zeros((len(joint_names), 4), dtype=np.float32)
    for j, name in enumerate(joint_names):
        if name not in quat_dict:
            raise KeyError(f"bind quaternion json missing joint: {name}")
        entry = quat_dict[name]
        if not isinstance(entry, dict):
            raise TypeError(f"bind quaternion entry for {name} must be dict, got {type(entry)}")
        for key in ("x", "y", "z", "w"):
            if key not in entry:
                raise KeyError(f"bind quaternion entry for {name} missing key: {key}")
        quat_arr[j] = [
            float(entry["x"]),
            float(entry["y"]),
            float(entry["z"]),
            float(entry["w"]),
        ]
    return normalize_quat_xyzw_array(quat_arr)


def load_bind_local_quat_array(bind_json_path: Path, joint_names: List[str]) -> np.ndarray:
    payload = read_json(bind_json_path)
    if not isinstance(payload, dict):
        raise TypeError("bind quaternion json must be an object/dict.")

    if "joint_quat_xyzw" in payload and isinstance(payload["joint_quat_xyzw"], dict):
        payload = payload["joint_quat_xyzw"]

    return parse_joint_quat_dict(payload, joint_names)


def compose_bind_local_quat(
    pose_quat_seq: np.ndarray,
    bind_local_quat: np.ndarray,
    compose_order: str,
) -> np.ndarray:
    pose = np.asarray(pose_quat_seq, dtype=np.float32)
    bind = np.asarray(bind_local_quat, dtype=np.float32)

    if pose.ndim != 3 or pose.shape[-1] != 4:
        raise ValueError(f"pose_quat_seq must be [T,J,4], got {pose.shape}")
    if bind.shape != (pose.shape[1], 4):
        raise ValueError(
            f"bind_local_quat shape mismatch: expected {(pose.shape[1], 4)}, got {bind.shape}"
        )

    bind_seq = np.broadcast_to(bind[None, :, :], pose.shape).astype(np.float32)
    if compose_order == "bind_mul_pose":
        out = quat_multiply_xyzw(bind_seq, pose)
    elif compose_order == "pose_mul_bind":
        out = quat_multiply_xyzw(pose, bind_seq)
    else:
        raise ValueError(f"Unsupported bind compose order: {compose_order}")

    return normalize_quat_xyzw_array(out)


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
    source_frame_option: str,
    source_handedness: str,
    unity_pose_mode: str,
    pre_mirror_x: bool,
    pre_flip_yz: bool,
    rotation_format: str,
    trans_key_option: str,
    no_quat_canonical: bool,
    no_quat_continuity: bool,
    bind_local_quat: Optional[np.ndarray],
    bind_compose_order: str,
    bind_source_label: Optional[str],
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
        source_frame = infer_source_frame_for_npz(
            data=data,
            source_frame_option=source_frame_option,
            source_handedness=source_handedness,
        )

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
        payload_source_frame = source_frame
        if coord_system == "unity":
            rotvec_seq = convert_pose_seq_to_unity(
                rotvec_seq=rotvec_seq,
                source_frame=source_frame,
                unity_pose_mode=unity_pose_mode,
            )
            trans_seq = convert_vec3_to_unity(trans_seq, source_frame=source_frame).astype(np.float32)
            payload_source_frame = FRAME_UNITY_RUF_LHS
            coord_meta = (
                "unity_left_handed "
                f"(from={source_frame}, unity_pose_mode={unity_pose_mode})"
            )

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

        meta: Dict[str, Any] = {
            "coord_system": coord_meta,
            "source_frame": payload_source_frame,
            "source_frame_input": source_frame,
            "rotation_vector_unit": "radian",
            "rotation_format": rotation_format,
            "bind_local_quat_applied": bind_local_quat is not None,
        }
        if rotation_format in ("quat", "both"):
            meta["quaternion_order"] = "xyzw"
        if coord_system == "unity":
            meta["unity_pose_mode"] = unity_pose_mode
        if pre_mirror_x:
            meta["pre_mirror_x_mode"] = "x_sign_only"
        if pre_flip_yz:
            meta["pre_flip_yz_mode"] = "y_z_sign_flip"
        if bind_local_quat is not None:
            meta["bind_compose_order"] = bind_compose_order
            if bind_source_label is not None:
                meta["bind_local_quat_json"] = bind_source_label

        payload = {
            "pose": pose_out.tolist(),
            "betas": betas_arr.tolist(),
            "trans": trans_out.tolist(),
            "meta": meta,
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
            if bind_local_quat is not None:
                quat_seq = compose_bind_local_quat(
                    pose_quat_seq=quat_seq,
                    bind_local_quat=bind_local_quat,
                    compose_order=bind_compose_order,
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
        default=NPZ2QUAT_DEFAULT_COORD_SYSTEM,
        help="Output coordinate system. default=unity",
    )
    parser.add_argument(
        "--source-frame",
        type=str,
        default=PIPELINE_DEFAULT_SOURCE_FRAME,
        choices=["auto", *SUPPORTED_SOURCE_FRAMES],
        help=(
            "Input pose/trans coordinate frame. "
            "'auto' reads meta_source_frame/meta_legacy_frame_name from npz."
        ),
    )
    parser.add_argument(
        "--source-handedness",
        type=str,
        choices=["rhs", "lhs"],
        default="rhs",
        help="Fallback handedness when --source-frame auto and frame metadata is missing.",
    )
    parser.add_argument(
        "--unity-pose-mode",
        type=str,
        default=NPZ2QUAT_DEFAULT_UNITY_POSE_MODE,
        choices=["root_only", "all_joints"],
        help=(
            "When --coord-system unity, convert pose as root_only (default; global orientation only) "
            "or all_joints."
        ),
    )
    parser.add_argument(
        "--bind-local-quat-json",
        type=str,
        default=None,
        help=(
            "Optional JSON path for Unity bind local rotations by joint name. "
            "If provided, output quaternion is composed as bind * pose (default order). "
            f"If omitted and '{DEFAULT_UNITY_BIND_LOCAL_QUAT_JSON}' exists, it is auto-applied."
        ),
    )
    parser.add_argument(
        "--bind-compose-order",
        type=str,
        default=NPZ2QUAT_DEFAULT_BIND_COMPOSE_ORDER,
        choices=["bind_mul_pose", "pose_mul_bind"],
        help="Quaternion composition order when --bind-local-quat-json is used.",
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
    bind_local_quat = None
    bind_source_label = None
    if args.bind_local_quat_json:
        bind_json_path = resolve_repo_path(repo_root, args.bind_local_quat_json)
    else:
        default_bind_json = resolve_repo_path(repo_root, DEFAULT_UNITY_BIND_LOCAL_QUAT_JSON)
        bind_json_path = default_bind_json if default_bind_json.exists() else None

    if bind_json_path is not None:
        bind_local_quat = load_bind_local_quat_array(bind_json_path, joint_names)
        bind_source_label = str(bind_json_path)

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
                source_frame_option=args.source_frame,
                source_handedness=args.source_handedness,
                unity_pose_mode=args.unity_pose_mode,
                pre_mirror_x=args.pre_mirror_x,
                pre_flip_yz=args.pre_flip_yz,
                rotation_format=args.rotation_format,
                trans_key_option=args.trans_key,
                no_quat_canonical=args.no_quat_canonical,
                no_quat_continuity=args.no_quat_continuity,
                bind_local_quat=bind_local_quat,
                bind_compose_order=args.bind_compose_order,
                bind_source_label=bind_source_label,
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
        source_frame_option=args.source_frame,
        source_handedness=args.source_handedness,
        unity_pose_mode=args.unity_pose_mode,
        pre_mirror_x=args.pre_mirror_x,
        pre_flip_yz=args.pre_flip_yz,
        rotation_format=args.rotation_format,
        trans_key_option=args.trans_key,
        no_quat_canonical=args.no_quat_canonical,
        no_quat_continuity=args.no_quat_continuity,
        bind_local_quat=bind_local_quat,
        bind_compose_order=args.bind_compose_order,
        bind_source_label=bind_source_label,
    )

    write_json(
        path=out_path,
        payload=payload,
        ascii_flag=args.ascii,
        compact=args.compact,
    )

    print(f"Saved JSON: {out_path}")
    print(
        f"coord_system={args.coord_system}, source_frame={args.source_frame}, "
        f"unity_pose_mode={args.unity_pose_mode}"
    )
    print(f"rotation_format={args.rotation_format}, frames={T}")
    print(f"bind_local_quat_json={bind_source_label}, bind_compose_order={args.bind_compose_order}")
    print(f"pre_mirror_x={args.pre_mirror_x} (mode=x_sign_only)")
    print(f"pre_flip_yz={args.pre_flip_yz} (mode=y_z_sign_flip)")


if __name__ == "__main__":
    main()
