# scripts/common/io_paths.py
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Union


StrPath = Union[str, Path]
_PathEntries = Mapping[str, tuple[str, ...]]


def _path(*parts: str) -> str:
    return Path(*parts).as_posix()


def _prefix(*parts: str) -> tuple[str, ...]:
    return tuple(parts)


def _define_paths(scope: dict[str, str], prefix: tuple[str, ...], entries: _PathEntries) -> None:
    for name, suffix in entries.items():
        scope[name] = _path(*prefix, *suffix)


_SCOPE = globals()
_DATA = _prefix("data")
_OUTPUTS = _prefix("outputs")
_DATASET = _prefix("dataset")
_CONFIGS = _prefix("configs")

_define_paths(
    _SCOPE,
    (*_DATA, "background"),
    {
        "DATA_BACKGROUND_DIR": (),
        "DATA_BACKGROUND_PCD": ("background.pcd",),
        "DATA_BACKGROUND_PCD_GLOB": ("pcd_out_*.pcd",),
    },
)

_define_paths(
    _SCOPE,
    (*_DATA, "interim"),
    {
        "DATA_INTERIM_DIR": (),
        "DATA_INTERIM_PERSON_DIR": ("person",),
        "DATA_INTERIM_DEBUG_EXTRACT_DIR": ("debug_extract",),
        "DATA_INTERIM_PCD2PLY_DIR": ("pcd2ply",),
        "DATA_INTERIM_CENTERED_REALITY": ("centered", "reality_centered.ply"),
        "DATA_INTERIM_CENTERED_VIRTUAL": ("centered", "virtual_centered.ply"),
    },
)

_define_paths(
    _SCOPE,
    (*_DATA, "labeling"),
    {
        "DATA_LABELING_DIR": (),
        "DATA_LABELING_PCAP_FRAMES_DIR": ("pcap_frames",),
        "DATA_LABELING_001_DIR": ("001",),
        "DATA_LABELING_001_PCAP": ("001", "simpleMotion_wonjin.pcap"),
        "DATA_LABELING_001_META_JSON": ("001", "simpleMotion_wonjin.json"),
        "DATA_LABELING_001_PCAP_FRAMES_TEST1": ("001", "pcap_frames", "test1"),
        "DATA_LABELING_001_ANGLE_CSV": ("001", "8joints_02-06_16-49.csv"),
    },
)

_define_paths(
    _SCOPE,
    _OUTPUTS,
    {
        "DATA_INTEROP_DIR": (),
        "DATA_INTEROP_SMPL_DIR": ("smpl",),
        "DATA_INTEROP_QUAT_DIR": ("quat",),
        "DATA_INTEROP_EULER_DIR": ("euler",),
        "DATA_INTEROP_LABEL_DIR": ("label",),
        "DATA_INTEROP_OBJ_DIR": ("obj",),
        "DATA_INTEROP_FBX_DIR": ("fbx",),
        "DATA_INTEROP_SMPL_JSON": ("smpl", "livehps_smpl.npz"),
        "DATA_INTEROP_QUAT_JSON": ("quat", "livehps_quaternion.json"),
        "DATA_INTEROP_EULER_JSON": ("euler", "livehps_unity.json"),
        "DATA_INTEROP_LABEL_JSON": ("label", "livehps_label.json"),
        "DATA_INTEROP_OBJ_MESH": ("obj", "livehps_mesh.obj"),
        "DATA_INTEROP_FBX_MOTION": ("fbx", "livehps_rigged.fbx"),
    },
)

_define_paths(
    _SCOPE,
    (*_DATA, "real"),
    {
        "DATA_REAL_DIR": (),
        "DATA_REAL_RAW_DIR": ("raw",),
        "DATA_REAL_RAW_EXAMPLE": ("raw", "SN001_000861.pcd"),
        "DATA_REAL_RAW_GLOB": ("raw", "SN001_*.pcd"),
        "DATA_REAL_HUMAN_DIR": ("human",),
        "DATA_REAL_HUMAN_EXAMPLE": ("human", "human_001.ply"),
        "DATA_REAL_HUMAN_GLOB": ("human", "human_*.ply"),
        "DATA_REAL_VAR_DIR": ("VAR",),
        "DATA_REAL_VARMOTION_DIR": ("VARmotion",),
    },
)

for alias_name, canonical_name in {
    "DATA_REAL_001_RAW_DIR": "DATA_REAL_RAW_DIR",
    "DATA_REAL_001_RAW_EXAMPLE": "DATA_REAL_RAW_EXAMPLE",
    "DATA_REAL_001_RAW_GLOB": "DATA_REAL_RAW_GLOB",
    "DATA_REAL_002_NOBG_DEG45_EXAMPLE": "DATA_REAL_RAW_EXAMPLE",
    "DATA_REAL_002_NOBG_DEG45_GLOB": "DATA_REAL_RAW_GLOB",
    "DATA_REAL_003_PROCESSED_DEG0_EXAMPLE": "DATA_REAL_HUMAN_EXAMPLE",
    "DATA_REAL_003_PROCESSED_DEG45_EXAMPLE": "DATA_REAL_HUMAN_EXAMPLE",
    "DATA_REAL_003_PROCESSED_DEG45_DIR": "DATA_REAL_HUMAN_DIR",
    "DATA_REAL_004_HUMAN_DEG0_DIR": "DATA_REAL_HUMAN_DIR",
    "DATA_REAL_RAW_DEG45_EXAMPLE": "DATA_REAL_RAW_EXAMPLE",
    "DATA_REAL_RAW_DEG45_GLOB": "DATA_REAL_RAW_GLOB",
    "DATA_REAL_PROCESSED_DEG0_EXAMPLE": "DATA_REAL_HUMAN_EXAMPLE",
    "DATA_REAL_PROCESSED_DEG45_EXAMPLE": "DATA_REAL_HUMAN_EXAMPLE",
    "DATA_REAL_PROCESSED_DEG45_DIR": "DATA_REAL_HUMAN_DIR",
}.items():
    _SCOPE[alias_name] = _SCOPE[canonical_name]

_define_paths(
    _SCOPE,
    (*_DATA, "virtual"),
    {
        "DATA_VIRTUAL_POSE324_DIR": ("pose324",),
        "DATA_VIRTUAL_POSE324_EXAMPLE": ("pose324", "virtual_1-0.ply"),
        "DATA_VIRTUAL_POSE324_GLOB": ("pose324", "virtual_*.ply"),
        "DATA_VIRTUAL_POSE324_10_DEG45_GLOB": ("pose324_10_deg45", "virtual_*.ply"),
    },
)

_define_paths(
    _SCOPE,
    _DATASET,
    {
        "DATASET_POSES_324_JSONL": ("poses_324.jsonl",),
        "DATASET_324_LOWERBODY_10SET_JSONL": ("324_LowerBody_10set.jsonl",),
        "DATASET_REAL_IMU_GT_JSONL": ("real_IMU_GT.jsonl",),
        "DATASET_LABEL_RULES_JSON": ("label_rules.json",),
    },
)

_define_paths(
    _SCOPE,
    _CONFIGS,
    {
        "CONFIG_AXIS_STEPS_YAML": ("axis_steps.yaml",),
    },
)

_define_paths(
    _SCOPE,
    _DATA,
    {
        "OUTPUT_MATCH_ONE_DIR": ("match_one",),
    },
)


def resolve_repo_path(repo_root: Path, raw_path: StrPath) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()
