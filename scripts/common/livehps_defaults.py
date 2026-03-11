# scripts/common/livehps_defaults.py
from scripts.common.coord_utils import FRAME_OUSTER_FLU_RHS


LIVEHPS_TRANSFORM_PROFILE = "ouster_source_to_unity_root_only"

PIPELINE_DEFAULT_SOURCE_FRAME = FRAME_OUSTER_FLU_RHS
DEFAULT_UNITY_BIND_LOCAL_QUAT_JSON = "configs/unity/smplx_bind_local_quat.json"

HUMAN2SMPL_DEFAULT_EXPORT_COORDS = "source"
HUMAN2SMPL_DEFAULT_LEGACY_AS = "source"

NPZ2QUAT_DEFAULT_COORD_SYSTEM = "unity"
NPZ2QUAT_DEFAULT_UNITY_POSE_MODE = "root_only"
NPZ2QUAT_DEFAULT_BIND_COMPOSE_ORDER = "bind_mul_pose"

QUAT2UNITY_DEFAULT_BASIS_CONVERSION = "none"
