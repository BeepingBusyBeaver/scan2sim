# scripts/conversion/npz_to_fbx.py
from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
try:
    from scripts.common.io_paths import (
        DATA_INTEROP_FBX_DIR,
        DATA_INTEROP_FBX_MOTION,
        DATA_INTEROP_SMPL_DIR,
        resolve_repo_path,
    )
    from scripts.common.path_utils import natural_key_path
except ModuleNotFoundError:
    _repo_root = Path(__file__).resolve().parents[2]
    if str(_repo_root) not in sys.path:
        sys.path.insert(0, str(_repo_root))
    from scripts.common.io_paths import (
        DATA_INTEROP_FBX_DIR,
        DATA_INTEROP_FBX_MOTION,
        DATA_INTEROP_SMPL_DIR,
        resolve_repo_path,
    )
    from scripts.common.path_utils import natural_key_path


SCAN2SIM_ROOT = Path(__file__).resolve().parents[2]


SMPL_JOINT_NAMES = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hand",
    "right_hand",
]

SMPL_PARENTS = [
    -1,
    0,
    0,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    9,
    9,
    12,
    13,
    14,
    16,
    17,
    18,
    19,
    20,
    21,
]


def _to_windows_path(p: Path) -> str:
    # WSL -> Windows 경로(UNC 또는 드라이브 경로)로 변환
    return subprocess.check_output(["wslpath", "-w", str(p)], text=True).strip()

def _resolve_livehps_smpl_paths() -> Tuple[Path, Path]:
    livehps_root = (SCAN2SIM_ROOT.parent / "LiveHPS").resolve()
    smpl_module_dir = livehps_root / "smpl"
    if not smpl_module_dir.exists():
        raise FileNotFoundError(f"SMPL directory not found: {smpl_module_dir}")
    if str(smpl_module_dir) not in sys.path:
        sys.path.insert(0, str(smpl_module_dir))

    vibe_data_dir = (smpl_module_dir / "data" / "vibe_data").resolve()
    if not vibe_data_dir.exists():
        raise FileNotFoundError(f"VIBE data directory not found: {vibe_data_dir}")
    return smpl_module_dir, vibe_data_dir


def choose_device(device_arg: str):
    import torch

    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no CUDA device is available.")
    return torch.device(device_arg)


def _has_glob_magic(text: str) -> bool:
    return any(ch in text for ch in ("*", "?", "["))


def collect_npz_paths(repo_root: Path, input_value: str, input_pattern: str) -> List[Path]:
    src = resolve_repo_path(repo_root, input_value)

    if _has_glob_magic(input_value):
        base = Path(input_value).expanduser()
        if not base.is_absolute():
            base = (repo_root / base).resolve()
        paths = sorted(
            [Path(p).resolve() for p in glob.glob(str(base), recursive=False) if Path(p).suffix.lower() == ".npz"],
            key=natural_key_path,
        )
        if not paths:
            raise ValueError(f"No .npz files matched: {input_value}")
        return paths

    if src.is_file():
        if src.suffix.lower() != ".npz":
            raise ValueError(f"--input must be .npz in single mode: {src}")
        return [src.resolve()]

    if src.is_dir():
        paths = sorted(
            [path.resolve() for path in src.glob(input_pattern) if path.is_file() and path.suffix.lower() == ".npz"],
            key=natural_key_path,
        )
        if not paths:
            raise ValueError(f"No .npz files found in {src} with pattern={input_pattern}")
        return paths

    raise ValueError(
        "Single mode requires --input to be a .npz path, glob, or directory."
    )


def pick_npz_key(data: np.lib.npyio.NpzFile, keys: Sequence[str]) -> str:
    for key in keys:
        if key in data.files:
            return key
    raise KeyError(f"None of keys {list(keys)} found. available={list(data.files)}")


def normalize_pose(pose_arr: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose_arr, dtype=np.float32)
    if pose.ndim == 1:
        if pose.shape[0] != 72:
            raise ValueError(f"1D pose must be [72], got {pose.shape}")
        pose = pose.reshape(1, 72)
    if pose.ndim != 2 or pose.shape[1] != 72:
        raise ValueError(f"Pose must be [T,72], got {pose.shape}")
    return pose


def normalize_trans(trans_arr: np.ndarray, frames: int) -> np.ndarray:
    trans = np.asarray(trans_arr, dtype=np.float32)
    if trans.ndim == 1:
        if trans.shape[0] != 3:
            raise ValueError(f"1D trans must be [3], got {trans.shape}")
        trans = trans.reshape(1, 3)
    if trans.ndim != 2 or trans.shape[1] != 3:
        raise ValueError(f"Trans must be [T,3], got {trans.shape}")
    if trans.shape[0] == 1 and frames > 1:
        trans = np.repeat(trans, repeats=frames, axis=0)
    if trans.shape[0] != frames:
        raise ValueError(f"Frame mismatch: pose={frames}, trans={trans.shape[0]}")
    return trans


def normalize_betas(betas_arr: np.ndarray) -> np.ndarray:
    betas = np.asarray(betas_arr, dtype=np.float32)
    if betas.ndim == 1:
        if betas.shape[0] != 10:
            raise ValueError(f"1D betas must be [10], got {betas.shape}")
        betas = betas.reshape(1, 10)
    if betas.ndim != 2 or betas.shape[1] != 10:
        raise ValueError(f"Betas must be [B,10], got {betas.shape}")
    return betas


def load_livehps_params(npz_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(npz_path, allow_pickle=True)
    pose_key = pick_npz_key(data, ["pose_seq_source", "pose_source", "pose_seq", "pose"])
    trans_key = pick_npz_key(
        data,
        [
            "trans_world_seq_source",
            "trans_world_source",
            "trans_world_seq",
            "trans_world",
            "trans_seq_source",
            "trans_source",
            "trans_seq",
            "trans",
        ],
    )
    betas_key = pick_npz_key(data, ["betas"])
    pose = normalize_pose(data[pose_key])
    trans = normalize_trans(data[trans_key], pose.shape[0])
    betas = normalize_betas(data[betas_key])
    return pose, betas, trans


def merge_livehps_params(npz_paths: Sequence[Path]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    if len(npz_paths) == 0:
        raise ValueError("npz_paths is empty.")

    pose_list: List[np.ndarray] = []
    trans_list: List[np.ndarray] = []
    betas_list: List[np.ndarray] = []
    for npz_path in npz_paths:
        pose, betas, trans = load_livehps_params(npz_path)
        pose_list.append(pose)
        trans_list.append(trans)
        betas_list.append(betas[0])

    pose_seq = np.concatenate(pose_list, axis=0).astype(np.float32)
    trans_seq = np.concatenate(trans_list, axis=0).astype(np.float32)
    betas_stack = np.stack(betas_list, axis=0).astype(np.float32)
    betas_ref = betas_stack[0:1].copy()
    betas_max_abs_diff = float(np.max(np.abs(betas_stack - betas_ref)))
    return pose_seq, betas_ref, trans_seq, betas_max_abs_diff


def _setup_smpl_imports() -> None:
    import importlib

    _, vibe_data_dir = _resolve_livehps_smpl_paths()
    core_config = importlib.import_module("core.config")
    core_config.VIBE_DATA_DIR = str(vibe_data_dir)


def build_blender_payload(npz_paths: Sequence[Path], device) -> Tuple[Path, dict]:
    import torch

    _setup_smpl_imports()
    try:
        from smpl.smpl import SMPL, SMPL_MODEL_DIR
    except ImportError:
        from smpl import SMPL, SMPL_MODEL_DIR

    pose_seq, betas, trans_seq, betas_max_abs_diff = merge_livehps_params(npz_paths)

    smpl = SMPL(SMPL_MODEL_DIR, create_transl=False).to(device)
    weights = smpl.lbs_weights.detach().cpu().numpy().astype(np.float32)
    faces = smpl.faces.astype(np.int32)

    with torch.no_grad():
        zero_pose = torch.zeros((1, 72), dtype=torch.float32, device=device)
        zero_trans = torch.zeros((1, 3), dtype=torch.float32, device=device)
        betas_tensor = torch.from_numpy(betas).to(device=device, dtype=torch.float32)
        rest_out = smpl(
            betas=betas_tensor,
            body_pose=zero_pose[:, 3:],
            global_orient=zero_pose[:, :3],
            transl=zero_trans,
        )

    rest_vertices = rest_out.vertices[0].detach().cpu().numpy().astype(np.float32)
    rest_joints = rest_out.joints[0].detach().cpu().numpy().astype(np.float32)

    payload = {
        "rest_vertices": rest_vertices,
        "rest_joints": rest_joints,
        "weights": weights,
        "faces": faces,
        "pose_seq": pose_seq.astype(np.float32),
        "trans_seq": trans_seq.astype(np.float32),
        "joint_names": np.array(SMPL_JOINT_NAMES, dtype="<U32"),
        "joint_parents": np.array(SMPL_PARENTS, dtype=np.int32),
    }

    temp = tempfile.NamedTemporaryFile(prefix="scan2sim_smpl_payload_", suffix=".npz", delete=False)
    temp_path = Path(temp.name)
    temp.close()
    np.savez(temp_path, **payload)

    meta = {
        "npz_files": [str(path) for path in npz_paths],
        "input_count": int(len(npz_paths)),
        "frames": int(pose_seq.shape[0]),
        "vertices": int(rest_vertices.shape[0]),
        "joints": int(rest_joints.shape[0]),
        "betas_merge": "first_file",
        "betas_max_abs_diff": float(betas_max_abs_diff),
    }
    return temp_path, meta


def run_blender_export(
    blender_exe: str,
    payload_path: Path,
    output_fbx: Path,
    fps: int,
    start_frame: int,
    axis_forward: str,
    axis_up: str,
    global_scale: float,
) -> None:
    script_path = Path(__file__).resolve()

    # WSL(Linux)에서 Windows Blender(.exe)를 호출하는 경우 경로 변환
    is_windows_blender = (sys.platform != "win32") and str(blender_exe).lower().endswith(".exe")
    if is_windows_blender:
        script_arg = _to_windows_path(script_path)
        payload_arg = _to_windows_path(payload_path)
        output_arg = _to_windows_path(output_fbx)
    else:
        script_arg = str(script_path)
        payload_arg = str(payload_path)
        output_arg = str(output_fbx)

    cmd = [
        blender_exe,
        "--background",
        "--python",
        script_arg,
        "--",
        "--blender-worker",
        "--payload",
        payload_arg,
        "--output",
        output_arg,
        "--fps",
        str(int(fps)),
        "--start-frame",
        str(int(start_frame)),
        f"--axis-forward={axis_forward}",
        f"--axis-up={axis_up}",
        "--global-scale",
        str(float(global_scale)),
    ]
    subprocess.run(cmd, check=True)


def parse_python_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert LiveHPS SMPL NPZ sequence to a single rigged FBX (SMPL armature + skin weights)."
    )
    parser.add_argument("--input", type=str, default=DATA_INTEROP_SMPL_DIR, help="Input .npz path, glob, or directory.")
    parser.add_argument("--output", type=str, default=DATA_INTEROP_FBX_MOTION, help="Output .fbx path.")
    parser.add_argument("--batch_per_file", action="store_true", help="Deprecated. Ignored (FBX is always single sequence).")
    parser.add_argument("--input_pattern", type=str, default="livehps_smpl_*.npz", help="Glob when --input is directory.")
    parser.add_argument("--output_dir", type=str, default=DATA_INTEROP_FBX_DIR, help=argparse.SUPPRESS)
    parser.add_argument("--output_prefix", type=str, default="livehps_rigged_", help=argparse.SUPPRESS)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Device for SMPL precompute.")
    parser.add_argument("--blender", type=str, default="blender", help="Blender executable path.")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--start-frame", type=int, default=1)
    parser.add_argument("--axis-forward", type=str, default="-Z", choices=["X", "Y", "Z", "-X", "-Y", "-Z"])
    parser.add_argument("--axis-up", type=str, default="Y", choices=["X", "Y", "Z", "-X", "-Y", "-Z"])
    parser.add_argument("--global-scale", type=float, default=1.0)
    parser.add_argument("--keep-temp", action="store_true", help="Keep temporary payload npz files.")
    return parser.parse_args(list(argv))


def _rotvec_to_quaternion_mathutils(rotvec) -> "Quaternion":
    from mathutils import Quaternion, Vector

    rv = np.asarray(rotvec, dtype=np.float64)
    angle = float(np.linalg.norm(rv))
    if angle < 1e-10:
        return Quaternion((1.0, 0.0, 0.0, 0.0))
    axis = Vector((float(rv[0] / angle), float(rv[1] / angle), float(rv[2] / angle)))
    return Quaternion(axis, angle)


def run_blender_worker(argv: Sequence[str]) -> None:
    import bpy
    from mathutils import Vector

    parser = argparse.ArgumentParser(description="Blender worker for npz_to_fbx")
    parser.add_argument("--blender-worker", action="store_true")
    parser.add_argument("--payload", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--start-frame", type=int, default=1)
    parser.add_argument("--axis-forward", type=str, default="-Z")
    parser.add_argument("--axis-up", type=str, default="Y")
    parser.add_argument("--global-scale", type=float, default=1.0)
    args = parser.parse_args(list(argv))

    data = np.load(args.payload, allow_pickle=False)
    rest_vertices = data["rest_vertices"].astype(np.float32)
    rest_joints = data["rest_joints"].astype(np.float32)
    weights = data["weights"].astype(np.float32)
    faces = data["faces"].astype(np.int32)
    pose_seq = data["pose_seq"].astype(np.float32).reshape(-1, 24, 3)
    trans_seq = data["trans_seq"].astype(np.float32).reshape(-1, 3)
    joint_names = [str(x) for x in data["joint_names"].tolist()]
    joint_parents = data["joint_parents"].astype(np.int32).tolist()

    if len(joint_names) != 24 or len(joint_parents) != 24:
        raise RuntimeError("Expected 24-joint SMPL payload.")
    if weights.shape[1] != 24:
        raise RuntimeError(f"weights shape mismatch: {weights.shape}")

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    scene = bpy.context.scene
    scene.render.fps = int(args.fps)
    scene.frame_start = int(args.start_frame)
    scene.frame_end = int(args.start_frame + pose_seq.shape[0] - 1)

    mesh = bpy.data.meshes.new("SMPLMesh")
    mesh.from_pydata(rest_vertices.tolist(), [], faces.tolist())
    mesh.update()
    mesh_obj = bpy.data.objects.new("SMPLMesh", mesh)
    scene.collection.objects.link(mesh_obj)

    arm_data = bpy.data.armatures.new("SMPLArmature")
    arm_obj = bpy.data.objects.new("SMPLArmature", arm_data)
    scene.collection.objects.link(arm_obj)
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode="EDIT")

    children = {idx: [] for idx in range(len(joint_names))}
    for idx, parent in enumerate(joint_parents):
        if parent >= 0:
            children[parent].append(idx)

    edit_bones = {}
    for idx, name in enumerate(joint_names):
        bone = arm_data.edit_bones.new(name)
        head = Vector(rest_joints[idx].tolist())
        child_ids = children.get(idx, [])
        if child_ids:
            child_head = Vector(rest_joints[child_ids[0]].tolist())
            direction = child_head - head
        elif joint_parents[idx] >= 0:
            parent_head = Vector(rest_joints[joint_parents[idx]].tolist())
            direction = head - parent_head
        else:
            direction = Vector((0.0, 1.0, 0.0))
        if direction.length < 1e-8:
            direction = Vector((0.0, 1.0, 0.0))
        direction.normalize()
        tail = head + direction * 0.05
        bone.head = head
        bone.tail = tail
        bone.roll = 0.0
        edit_bones[idx] = bone

    for idx, parent in enumerate(joint_parents):
        if parent >= 0:
            edit_bones[idx].parent = edit_bones[parent]
            edit_bones[idx].use_connect = False

    bpy.ops.object.mode_set(mode="OBJECT")

    for j, joint_name in enumerate(joint_names):
        group = mesh_obj.vertex_groups.new(name=joint_name)
        nz = np.where(weights[:, j] > 1e-8)[0]
        for vid in nz.tolist():
            group.add([int(vid)], float(weights[vid, j]), "REPLACE")

    modifier = mesh_obj.modifiers.new(name="Armature", type="ARMATURE")
    modifier.object = arm_obj
    mesh_obj.parent = arm_obj

    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode="POSE")
    for name in joint_names:
        pbone = arm_obj.pose.bones[name]
        pbone.rotation_mode = "QUATERNION"
        pbone.location = (0.0, 0.0, 0.0)

    for frame_idx in range(pose_seq.shape[0]):
        frame_no = int(args.start_frame + frame_idx)
        scene.frame_set(frame_no)
        arm_obj.location = tuple(float(x) for x in trans_seq[frame_idx])
        arm_obj.keyframe_insert(data_path="location", frame=frame_no)
        for j, name in enumerate(joint_names):
            pbone = arm_obj.pose.bones[name]
            pbone.rotation_quaternion = _rotvec_to_quaternion_mathutils(pose_seq[frame_idx, j])
            pbone.keyframe_insert(data_path="rotation_quaternion", frame=frame_no)

    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")
    arm_obj.select_set(True)
    mesh_obj.select_set(True)
    bpy.context.view_layer.objects.active = arm_obj

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.fbx(
        filepath=str(out_path),
        use_selection=True,
        object_types={"MESH", "ARMATURE"},
        use_armature_deform_only=True,
        add_leaf_bones=False,
        bake_anim=True,
        bake_anim_use_all_bones=True,
        bake_anim_use_nla_strips=False,
        bake_anim_use_all_actions=False,
        bake_anim_force_startend_keying=True,
        bake_anim_step=1.0,
        axis_forward=args.axis_forward,
        axis_up=args.axis_up,
        global_scale=float(args.global_scale),
    )
    print(f"[done] fbx={out_path.resolve()}")


def run_python_mode(argv: Sequence[str]) -> None:
    args = parse_python_args(argv)
    repo_root = Path(__file__).resolve().parents[2]
    if args.batch_per_file:
        print("[warn] --batch_per_file is deprecated and ignored. Exporting one time-series FBX.")
    npz_paths = collect_npz_paths(
        repo_root=repo_root,
        input_value=args.input,
        input_pattern=args.input_pattern,
    )

    device = choose_device(args.device)

    out_path = resolve_repo_path(repo_root, args.output)
    payload_path, meta = build_blender_payload(npz_paths, device=device)
    try:
        run_blender_export(
            blender_exe=args.blender,
            payload_path=payload_path,
            output_fbx=out_path,
            fps=args.fps,
            start_frame=args.start_frame,
            axis_forward=args.axis_forward,
            axis_up=args.axis_up,
            global_scale=args.global_scale,
        )
    finally:
        if not args.keep_temp:
            payload_path.unlink(missing_ok=True)

    print(json.dumps({"output": str(out_path), **meta}, ensure_ascii=False))


def _extract_tool_args(argv: Sequence[str]) -> List[str]:
    if "--" in argv:
        return list(argv[argv.index("--") + 1 :])
    return list(argv[1:])


def main() -> None:
    tool_args = _extract_tool_args(sys.argv)
    if "--blender-worker" in tool_args:
        run_blender_worker(tool_args)
    else:
        run_python_mode(tool_args)


if __name__ == "__main__":
    main()
