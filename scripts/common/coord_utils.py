# scripts/common/coord_utils.py
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from scipy.spatial.transform import Rotation as R


FRAME_OUSTER_FLU_RHS = "ouster_flu_rhs"
FRAME_UNITY_RUF_LHS = "unity_ruf_lhs"

SUPPORTED_SOURCE_FRAMES = (
    FRAME_OUSTER_FLU_RHS,
    FRAME_UNITY_RUF_LHS,
)

# Ouster FLU (RHS): x=forward, y=left, z=up
# Unity  RUF (LHS): x=right,   y=up,   z=forward
# Mapping:
#   x_u = -y_flu
#   y_u =  z_flu
#   z_u =  x_flu
FLU_TO_UNITY_BASIS = np.array(
    [
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)


def _require_last_dim(arr: np.ndarray, dim: int, name: str) -> None:
    if arr.ndim < 1 or arr.shape[-1] != dim:
        raise ValueError(f"{name} must have last dim={dim}, got shape={arr.shape}")


def _canonicalize_quat_w_positive(quat_xyzw: np.ndarray) -> np.ndarray:
    out = np.asarray(quat_xyzw, dtype=np.float64).copy()
    flip = out[..., 3] < 0.0
    out[flip] *= -1.0
    return out


def normalize_quat_xyzw(quat_xyzw: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat_xyzw, dtype=np.float64)
    _require_last_dim(quat, 4, name="quat_xyzw")
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    if np.any(norm <= 1e-12):
        raise ValueError("Invalid quaternion: norm is zero.")
    return quat / norm


def change_basis_rotvec(rotvec: np.ndarray, basis: np.ndarray) -> np.ndarray:
    rv = np.asarray(rotvec, dtype=np.float64)
    _require_last_dim(rv, 3, name="rotvec")
    mats = R.from_rotvec(rv.reshape(-1, 3)).as_matrix()
    b = np.asarray(basis, dtype=np.float64).reshape(3, 3)
    mats_new = np.einsum("ab,mbc,cd->mad", b, mats, b.T)
    rv_new = R.from_matrix(mats_new).as_rotvec()
    return rv_new.reshape(rv.shape)


def change_basis_quat_xyzw(quat_xyzw: np.ndarray, basis: np.ndarray) -> np.ndarray:
    quat = normalize_quat_xyzw(quat_xyzw)
    mats = R.from_quat(quat.reshape(-1, 4)).as_matrix()
    b = np.asarray(basis, dtype=np.float64).reshape(3, 3)
    mats_new = np.einsum("ab,mbc,cd->mad", b, mats, b.T)
    quat_new = R.from_matrix(mats_new).as_quat().reshape(quat.shape)
    return _canonicalize_quat_w_positive(quat_new)


def convert_vec3_to_unity(vec3: np.ndarray, source_frame: str) -> np.ndarray:
    vec = np.asarray(vec3, dtype=np.float64)
    _require_last_dim(vec, 3, name="vec3")
    if source_frame == FRAME_UNITY_RUF_LHS:
        return vec.copy()
    if source_frame != FRAME_OUSTER_FLU_RHS:
        raise ValueError(f"Unsupported source frame: {source_frame}")

    out = np.empty_like(vec)
    out[..., 0] = -vec[..., 1]
    out[..., 1] = vec[..., 2]
    out[..., 2] = vec[..., 0]
    return out


def convert_rotvec_to_unity(rotvec: np.ndarray, source_frame: str) -> np.ndarray:
    rv = np.asarray(rotvec, dtype=np.float64)
    _require_last_dim(rv, 3, name="rotvec")
    if source_frame == FRAME_UNITY_RUF_LHS:
        return rv.copy()
    if source_frame != FRAME_OUSTER_FLU_RHS:
        raise ValueError(f"Unsupported source frame: {source_frame}")
    return change_basis_rotvec(rv, FLU_TO_UNITY_BASIS)


def convert_quat_xyzw_to_unity(quat_xyzw: np.ndarray, source_frame: str) -> np.ndarray:
    quat = normalize_quat_xyzw(quat_xyzw)
    if source_frame == FRAME_UNITY_RUF_LHS:
        return _canonicalize_quat_w_positive(quat)
    if source_frame != FRAME_OUSTER_FLU_RHS:
        raise ValueError(f"Unsupported source frame: {source_frame}")
    return change_basis_quat_xyzw(quat, FLU_TO_UNITY_BASIS)


def _wrap360_vec(values_deg: np.ndarray) -> np.ndarray:
    values = np.asarray(values_deg, dtype=np.float64)
    wrapped = np.mod(values, 360.0)
    wrapped[np.isclose(wrapped, 360.0, atol=1e-9)] = 0.0
    return wrapped


def quat_xyzw_to_inspector_xyz_deg(quat_xyzw: np.ndarray) -> np.ndarray:
    quat = normalize_quat_xyzw(quat_xyzw)
    zxy = R.from_quat(quat.reshape(-1, 4)).as_euler("zxy", degrees=True)
    xyz = np.stack([zxy[:, 1], zxy[:, 2], zxy[:, 0]], axis=-1)
    xyz = _wrap360_vec(xyz)
    return xyz.reshape(quat.shape[:-1] + (3,))


def unity_quat_xyzw_to_inspector_xyz_deg(quat_unity_xyzw: np.ndarray) -> np.ndarray:
    return quat_xyzw_to_inspector_xyz_deg(quat_unity_xyzw)


def source_quat_xyzw_to_unity_inspector_xyz_deg(
    quat_source_xyzw: np.ndarray, source_frame: str
) -> np.ndarray:
    quat_unity = convert_quat_xyzw_to_unity(quat_source_xyzw, source_frame=source_frame)
    return quat_xyzw_to_inspector_xyz_deg(quat_unity)


def infer_source_frame_from_meta(
    meta: Mapping[str, Any] | None,
    fallback: str = FRAME_OUSTER_FLU_RHS,
) -> str:
    if not isinstance(meta, Mapping):
        return fallback

    source_frame = meta.get("source_frame")
    if isinstance(source_frame, str) and source_frame in SUPPORTED_SOURCE_FRAMES:
        return source_frame

    legacy_frame_name = meta.get("legacy_frame_name")
    if isinstance(legacy_frame_name, str) and legacy_frame_name in SUPPORTED_SOURCE_FRAMES:
        return legacy_frame_name

    coord_system = meta.get("coord_system")
    if isinstance(coord_system, str) and "unity" in coord_system.lower():
        return FRAME_UNITY_RUF_LHS

    return fallback
