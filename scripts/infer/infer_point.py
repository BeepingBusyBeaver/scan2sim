# scripts/infer/infer_point.py
import argparse
import inspect
import sys
from functools import wraps
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R
from scripts.common.path_utils import natural_key_path
from scripts.common.io_paths import (
    DATA_INTEROP_SMPL_DIR,
    DATA_INTEROP_SMPL_JSON,
    resolve_repo_path,
)

SCAN2SIM_ROOT = Path(__file__).resolve().parents[2]
LIVEHPS_ROOT = (SCAN2SIM_ROOT.parent / "LiveHPS").resolve()
if str(LIVEHPS_ROOT) not in sys.path:
    sys.path.insert(0, str(LIVEHPS_ROOT))

from dataset.Livehps_dataset import farthest_point_sample
from models import ModelNet_shape, load_GPUS
from Model.livehps import CPO, PBT, SMPL_trans


"""
python -m scripts.infer.infer_point \
  --input data/real/human \
  --batch_per_file \
  --input_pattern "human_*.ply" \
  --output_dir outputs/smpl \
  --output_prefix "livehps_smpl_" \
  --weights ../LiveHPS/save_models/livehps.t7 \
  --device auto \
  --source_frame ouster_flu_rhs \
  --export_coords both \
  --legacy_as source \
  --unity_pose_mode root_only


python -m scripts.infer.infer_point \
  --input data/real/human/human_001.ply \
  --output outputs/smpl/livehps_smpl_001.npz \
  --weights ../LiveHPS/save_models/livehps.t7 \
  --device auto \
  --source_frame ouster_flu_rhs \
  --export_coords both \
  --legacy_as source \
  --unity_pose_mode root_only

"""


# -----------------------------
# Coordinate conversion constants
# -----------------------------
# Ouster FLU (RHS): x=forward, y=left, z=up
# Unity  RUF (LHS): x=right,   y=up,   z=forward
# Mapping:
#   x_u = -y_flu
#   y_u =  z_flu
#   z_u =  x_flu
FLU_TO_UNITY_C = np.array(
    [
        [0.0, -1.0, 0.0],  # x_u
        [0.0,  0.0, 1.0],  # y_u
        [1.0,  0.0, 0.0],  # z_u
    ],
    dtype=np.float32,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Infer SMPL parameters from person point clouds (.ply/.pcd) using LiveHPS."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to a .ply/.pcd file or a directory containing .ply/.pcd files.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DATA_INTEROP_SMPL_JSON,
        help="Output npz file path.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="../LiveHPS/save_models/livehps.t7",
        help="Path to LiveHPS pretrained weights.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Inference device.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=None,
        help="Target sequence length. If omitted, use all input frames as-is.",
    )
    parser.add_argument(
        "--num_points",
        type=int,
        default=256,
        help="Number of sampled points per frame.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for farthest point sampling initialization.",
    )

    # ---- NEW: Batch-per-file options ----
    parser.add_argument(
        "--batch_per_file",
        action="store_true",
        help=(
            "If --input is a directory, run inference per file and write one npz per input "
            "(e.g. human_0001.ply -> livehps_smpl_0001.npz)."
        ),
    )
    parser.add_argument(
        "--input_pattern",
        type=str,
        default="human_*.ply",
        help="Glob pattern used in --batch_per_file mode.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=(
            "Output directory in --batch_per_file mode. "
            "If omitted, uses outputs/smpl."
        ),
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="livehps_smpl_",
        help="Output filename prefix in --batch_per_file mode.",
    )

    # ---- Coordinate handling ----
    parser.add_argument(
        "--source_frame",
        type=str,
        default="ouster_flu_rhs",
        choices=["ouster_flu_rhs", "unity_ruf_lhs"],
        help="Coordinate frame of input point clouds.",
    )
    parser.add_argument(
        "--export_coords",
        type=str,
        default="both",
        choices=["source", "unity", "both"],
        help="Which coordinate versions to store in npz.",
    )
    parser.add_argument(
        "--legacy_as",
        type=str,
        default="source",
        choices=["source", "unity"],
        help=(
            "Which coordinate version to write into legacy keys "
            "(pose, trans, trans_world, pose_seq, trans_seq, trans_world_seq, centers, centers_seq)."
        ),
    )
    parser.add_argument(
        "--unity_pose_mode",
        type=str,
        default="root_only",
        choices=["root_only", "all_joints"],
        help=(
            "How to convert pose to Unity when source_frame=ouster_flu_rhs. "
            "'root_only' (recommended): convert pelvis/global orient only. "
            "'all_joints': convert all 24 joints (legacy behavior)."
        ),
    )

    return parser.parse_args()


def collect_point_paths(input_path: Path, pattern: str = None):
    exts = {".pcd", ".ply"}

    if input_path.is_file():
        if input_path.suffix.lower() not in exts:
            raise ValueError(f"Input file must be one of {exts}: {input_path}")
        return [input_path]

    if input_path.is_dir():
        if pattern is None:
            candidates = [p for p in input_path.iterdir() if p.is_file()]
        else:
            candidates = [p for p in input_path.glob(pattern) if p.is_file()]

        paths = sorted(
            [p for p in candidates if p.suffix.lower() in exts],
            key=natural_key_path,
        )
        if not paths:
            raise ValueError(
                f"No .pcd/.ply files found in directory: {input_path} (pattern={pattern})"
            )
        return paths

    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def select_frame_paths(paths, target_frames):
    if target_frames is None:
        return paths
    if target_frames <= 0:
        raise ValueError("--frames must be a positive integer.")
    if len(paths) == target_frames:
        return paths

    indices = np.linspace(0, len(paths) - 1, num=target_frames)
    indices = np.round(indices).astype(np.int32)
    return [paths[i] for i in indices]


def load_points_from_cloud(path: Path):
    try:
        import open3d as o3d
    except Exception as exc:
        raise ImportError(
            "open3d is required to read .pcd/.ply files. Install with `pip install open3d`."
        ) from exc

    pcd = o3d.io.read_point_cloud(str(path))
    points = np.asarray(pcd.points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        raise ValueError(f"Invalid or empty point cloud: {path}")
    return points


def preprocess_points(points: np.ndarray, num_points: int):
    center = points.mean(axis=0, dtype=np.float32)
    centered_points = points - center
    sampled_points = farthest_point_sample(centered_points, num_points).astype(np.float32)
    return sampled_points, center.astype(np.float32)


def rotation_6d_to_matrix(rot_6d: torch.Tensor):
    a1 = rot_6d[..., 0:3]
    a2 = rot_6d[..., 3:6]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def matrix_to_axis_angle_torch(rot_mats: torch.Tensor):
    original_shape = rot_mats.shape[:-2]
    flat_rot_mats = rot_mats.reshape(-1, 3, 3).detach().cpu().numpy()
    rot_vec = R.from_matrix(flat_rot_mats).as_rotvec().astype(np.float32)
    return torch.from_numpy(rot_vec).reshape(*original_shape, 3)


def configure_transformer_encoder_defaults() -> None:
    encoder_cls = torch.nn.TransformerEncoder
    original_init = encoder_cls.__init__
    if getattr(original_init, "_scan2sim_nested_tensor_patched", False):
        return

    try:
        signature = inspect.signature(original_init)
    except (TypeError, ValueError):
        return
    if "enable_nested_tensor" not in signature.parameters:
        return

    @wraps(original_init)
    def patched_init(self, *args, **kwargs):
        try:
            bound = signature.bind_partial(None, *args, **kwargs)
        except TypeError:
            bound = None
        if bound is not None and "enable_nested_tensor" not in bound.arguments:
            encoder_layer = bound.arguments.get("encoder_layer")
            self_attn = getattr(encoder_layer, "self_attn", None)
            kwargs["enable_nested_tensor"] = bool(getattr(self_attn, "batch_first", False))
        return original_init(self, *args, **kwargs)

    setattr(patched_init, "_scan2sim_nested_tensor_patched", True)
    encoder_cls.__init__ = patched_init


def build_livehps_model(weights_path: Path):
    configure_transformer_encoder_defaults()
    num_joints = 24
    model_kp = PBT(k=num_joints * 3, channel=3)
    model_trans = SMPL_trans(in_c=3 + 1024)
    model_kp_refine = CPO(in_c=1024 + 72, hidden=256 + 3 + 1024, out_c=3, kp_num=num_joints)
    model_rot = CPO(in_c=1024 + 72, hidden=256 + 3 + 1024, out_c=6, kp_num=num_joints)
    model_shape = CPO(
        in_c=1024 + 72,
        hidden=256 + 3 + 1024,
        out_c=6,
        kp_num=num_joints,
        shape=True,
    )
    model = ModelNet_shape(model_kp, model_trans, model_kp_refine, model_rot, model_shape)
    load_GPUS(model, str(weights_path))
    return model


def choose_device(device_arg: str):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no CUDA device is available.")
    return torch.device(device_arg)


# -----------------------------
# Coordinate conversion helpers
# -----------------------------
def flu_to_unity_vec(vec_seq: np.ndarray) -> np.ndarray:
    """
    vec_seq: [..., 3] in FLU
    return : [..., 3] in Unity RUF
    """
    v = np.asarray(vec_seq, dtype=np.float32)
    out = np.empty_like(v)
    out[..., 0] = -v[..., 1]  # x_u = -y_flu
    out[..., 1] =  v[..., 2]  # y_u =  z_flu
    out[..., 2] =  v[..., 0]  # z_u =  x_flu
    return out


def change_basis_rotvec(rotvec: np.ndarray, C: np.ndarray) -> np.ndarray:
    """
    Generic basis change for axis-angle rotations.
    rotvec: [..., 3] axis-angle (radian)
    C:      [3,3] basis mapping matrix (v_new = C @ v_old)
    """
    rv = np.asarray(rotvec, dtype=np.float32)
    mats = R.from_rotvec(rv.reshape(-1, 3)).as_matrix().astype(np.float32)  # [M,3,3]
    mats_new = np.einsum("ab,mbc,cd->mad", C.astype(np.float32), mats, C.astype(np.float32).T)
    rv_new = R.from_matrix(mats_new).as_rotvec().astype(np.float32)
    return rv_new.reshape(rv.shape)


def flu_to_unity_rotvec_any(rotvec: np.ndarray) -> np.ndarray:
    """
    rotvec: [...,3] in FLU basis
    return: [...,3] in Unity basis
    """
    return change_basis_rotvec(rotvec, FLU_TO_UNITY_C)


def convert_pose_seq72_to_unity(
    pose_seq72: np.ndarray,
    source_frame: str,
    unity_pose_mode: str = "root_only",
) -> np.ndarray:
    """
    pose_seq72: [T,72] axis-angle (radian)

    unity_pose_mode:
      - root_only : convert only joint 0 (pelvis/global orient) from FLU -> Unity
      - all_joints: convert all 24 joints (legacy behavior)
    """
    pose_seq72 = np.asarray(pose_seq72, dtype=np.float32)
    if pose_seq72.ndim != 2 or pose_seq72.shape[1] != 72:
        raise ValueError(f"Expected pose_seq [T,72], got {pose_seq72.shape}")

    if source_frame == "unity_ruf_lhs":
        return pose_seq72.copy()

    # source is ouster_flu_rhs
    rv24 = pose_seq72.reshape(-1, 24, 3)

    if unity_pose_mode == "all_joints":
        rv24_u = flu_to_unity_rotvec_any(rv24)
    elif unity_pose_mode == "root_only":
        rv24_u = rv24.copy()
        # pelvis/global orientation only
        rv24_u[:, 0, :] = flu_to_unity_rotvec_any(rv24[:, 0, :])
        # body local joints remain unchanged
    else:
        raise ValueError(f"Unknown unity_pose_mode: {unity_pose_mode}")

    return rv24_u.reshape(-1, 72).astype(np.float32)


def convert_vec_seq_to_unity(vec_seq: np.ndarray, source_frame: str) -> np.ndarray:
    vec_seq = np.asarray(vec_seq, dtype=np.float32)
    if vec_seq.ndim != 2 or vec_seq.shape[1] != 3:
        raise ValueError(f"Expected vec_seq [T,3], got {vec_seq.shape}")

    if source_frame == "unity_ruf_lhs":
        return vec_seq.copy()

    # source is ouster_flu_rhs
    return flu_to_unity_vec(vec_seq)


def pack_single_or_seq(seq: np.ndarray) -> np.ndarray:
    seq = np.asarray(seq, dtype=np.float32)
    if seq.ndim >= 1 and seq.shape[0] == 1:
        return seq[0]
    return seq


def make_batch_output_name(input_file: Path, output_prefix: str) -> str:
    """
    human_0001.ply -> livehps_smpl_0001.npz (default)
    non-human prefix files -> livehps_smpl_<stem>.npz
    """
    stem = input_file.stem
    if stem.startswith("human_"):
        tail = stem[len("human_"):]
    else:
        tail = stem
    return f"{output_prefix}{tail}.npz"


def run_inference_and_save(
    selected_paths,
    output_path: Path,
    args,
    model,
    device,
    verbose: bool = True,
):
    if len(selected_paths) == 0:
        raise ValueError("selected_paths is empty.")

    sampled_frames = []
    centers = []
    for pc_path in selected_paths:
        points = load_points_from_cloud(pc_path)
        sampled_points, center = preprocess_points(points, args.num_points)
        sampled_frames.append(sampled_points)
        centers.append(center)

    seq_points = np.stack(sampled_frames, axis=0).astype(np.float32)  # [T,N,3]
    centers_seq_src = np.stack(centers, axis=0).astype(np.float32)    # [T,3]
    t = seq_points.shape[0]

    with torch.no_grad():
        seq_tensor = torch.from_numpy(seq_points).unsqueeze(0).to(device)  # [1,T,N,3]
        _, rot_6d, betas, trans = model(seq_tensor.float())
        rot_6d = rot_6d.reshape(1, t, 24, 6)
        pose = matrix_to_axis_angle_torch(rotation_6d_to_matrix(rot_6d)).reshape(1, t, 72)

    # source-frame outputs from model
    pose_seq_src = pose.squeeze(0).cpu().numpy().astype(np.float32)                      # [T,72]
    betas = betas.squeeze(0).cpu().numpy().astype(np.float32)                             # [10]
    trans_seq_src = trans.reshape(1, t, 3).squeeze(0).cpu().numpy().astype(np.float32)   # [T,3]
    trans_world_seq_src = (trans_seq_src + centers_seq_src).astype(np.float32)           # [T,3]

    # unity-frame converted outputs
    pose_seq_unity = convert_pose_seq72_to_unity(
        pose_seq_src, args.source_frame, args.unity_pose_mode
    )                                                                                      # [T,72]
    trans_seq_unity = convert_vec_seq_to_unity(trans_seq_src, args.source_frame)          # [T,3]
    centers_seq_unity = convert_vec_seq_to_unity(centers_seq_src, args.source_frame)      # [T,3]
    trans_world_seq_unity = convert_vec_seq_to_unity(
        trans_world_seq_src, args.source_frame
    )                                                                                      # [T,3]

    # choose legacy alias
    if args.legacy_as == "source":
        pose_seq_legacy = pose_seq_src
        trans_seq_legacy = trans_seq_src
        centers_seq_legacy = centers_seq_src
        trans_world_seq_legacy = trans_world_seq_src
        legacy_frame_name = args.source_frame
    else:
        pose_seq_legacy = pose_seq_unity
        trans_seq_legacy = trans_seq_unity
        centers_seq_legacy = centers_seq_unity
        trans_world_seq_legacy = trans_world_seq_unity
        legacy_frame_name = "unity_ruf_lhs"

    # build save dict
    save_dict = {
        "betas": betas.astype(np.float32),
        "pcd_files": np.array([str(path) for path in selected_paths]),

        # metadata
        "meta_source_frame": np.array(args.source_frame),
        "meta_target_frame_unity": np.array("unity_ruf_lhs"),
        "meta_export_coords": np.array(args.export_coords),
        "meta_legacy_as": np.array(args.legacy_as),
        "meta_legacy_frame_name": np.array(legacy_frame_name),
        "meta_rotation_vector_unit": np.array("radian"),
        "meta_flu_to_unity_C": FLU_TO_UNITY_C.astype(np.float32),
        "meta_unity_pose_mode": np.array(args.unity_pose_mode),
        "num_frames": np.array([t], dtype=np.int32),
        "num_points_per_frame": np.array([args.num_points], dtype=np.int32),
    }

    # source keys
    if args.export_coords in ("source", "both"):
        save_dict.update({
            "pose_seq_source": pose_seq_src.astype(np.float32),
            "trans_seq_source": trans_seq_src.astype(np.float32),
            "centers_seq_source": centers_seq_src.astype(np.float32),
            "trans_world_seq_source": trans_world_seq_src.astype(np.float32),

            "pose_source": pack_single_or_seq(pose_seq_src),
            "trans_source": pack_single_or_seq(trans_seq_src),
            "centers_source": pack_single_or_seq(centers_seq_src),
            "trans_world_source": pack_single_or_seq(trans_world_seq_src),
        })

    # unity keys
    if args.export_coords in ("unity", "both"):
        save_dict.update({
            "pose_seq_unity": pose_seq_unity.astype(np.float32),
            "trans_seq_unity": trans_seq_unity.astype(np.float32),
            "centers_seq_unity": centers_seq_unity.astype(np.float32),
            "trans_world_seq_unity": trans_world_seq_unity.astype(np.float32),

            "pose_unity": pack_single_or_seq(pose_seq_unity),
            "trans_unity": pack_single_or_seq(trans_seq_unity),
            "centers_unity": pack_single_or_seq(centers_seq_unity),
            "trans_world_unity": pack_single_or_seq(trans_world_seq_unity),
        })

    # legacy keys
    save_dict.update({
        "pose_seq": pose_seq_legacy.astype(np.float32),
        "trans_seq": trans_seq_legacy.astype(np.float32),
        "centers_seq": centers_seq_legacy.astype(np.float32),
        "trans_world_seq": trans_world_seq_legacy.astype(np.float32),

        "pose": pack_single_or_seq(pose_seq_legacy),
        "trans": pack_single_or_seq(trans_seq_legacy),
        "centers": pack_single_or_seq(centers_seq_legacy),
        "trans_world": pack_single_or_seq(trans_world_seq_legacy),
    })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **save_dict)

    if verbose:
        print(f"Saved output: {output_path}")
        print(f"frames={t}, num_points={args.num_points}, device={device.type}")
        print(
            f"source_frame={args.source_frame}, export_coords={args.export_coords}, "
            f"legacy_as={args.legacy_as}, unity_pose_mode={args.unity_pose_mode}"
        )
        print(f"legacy pose shape: {save_dict['pose'].shape}")
        print(f"betas shape: {betas.shape}")
        print(f"legacy trans shape: {save_dict['trans'].shape}")

    return t, betas.shape, save_dict["trans"].shape


def main():
    args = parse_args()
    np.random.seed(args.seed)
    repo_root = Path(__file__).resolve().parents[2]

    # argument consistency
    if args.export_coords == "source" and args.legacy_as == "unity":
        raise ValueError("--legacy_as unity requires --export_coords unity/both")
    if args.export_coords == "unity" and args.legacy_as == "source":
        raise ValueError("--legacy_as source requires --export_coords source/both")

    input_path = resolve_repo_path(repo_root, args.input)
    output_path = resolve_repo_path(repo_root, args.output)
    weights_path = resolve_repo_path(repo_root, args.weights)

    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    device = choose_device(args.device)
    model = build_livehps_model(weights_path).to(device)
    model.eval()

    # --- NEW: batch per file mode ---
    if args.batch_per_file:
        if not input_path.is_dir():
            raise ValueError("--batch_per_file requires --input to be a directory.")

        if args.frames not in (None, 1):
            print(
                "[WARN] --frames is ignored in --batch_per_file mode "
                "(each file is treated as one frame)."
            )

        point_paths = collect_point_paths(input_path, pattern=args.input_pattern)

        if args.output_dir is not None:
            batch_out_dir = resolve_repo_path(repo_root, args.output_dir)
        else:
            batch_out_dir = resolve_repo_path(repo_root, DATA_INTEROP_SMPL_DIR)

        batch_out_dir.mkdir(parents=True, exist_ok=True)

        total = len(point_paths)
        for i, pc_path in enumerate(point_paths, start=1):
            out_name = make_batch_output_name(pc_path, args.output_prefix)
            out_path = batch_out_dir / out_name

            t, betas_shape, trans_shape = run_inference_and_save(
                [pc_path], out_path, args, model, device, verbose=False
            )
            print(
                f"[{i:04d}/{total:04d}] {pc_path.name} -> {out_path.name} "
                f"(frames={t}, betas={betas_shape}, trans={trans_shape})"
            )

        print(f"Done. processed={total}, output_dir={batch_out_dir}")
        return

    # --- 기존 동작: 단일 파일 or 디렉터리 전체를 시퀀스로 ---
    point_paths = collect_point_paths(input_path)
    selected_paths = select_frame_paths(point_paths, args.frames)
    run_inference_and_save(selected_paths, output_path, args, model, device, verbose=True)

    sampled_frames = []
    centers = []
    for pc_path in selected_paths:
        points = load_points_from_cloud(pc_path)
        sampled_points, center = preprocess_points(points, args.num_points)
        sampled_frames.append(sampled_points)
        centers.append(center)

    seq_points = np.stack(sampled_frames, axis=0).astype(np.float32)  # [T,N,3]
    centers_seq_src = np.stack(centers, axis=0).astype(np.float32)    # [T,3]
    t = seq_points.shape[0]

    device = choose_device(args.device)
    model = build_livehps_model(weights_path).to(device)
    model.eval()

    with torch.no_grad():
        seq_tensor = torch.from_numpy(seq_points).unsqueeze(0).to(device)  # [1,T,N,3]
        _, rot_6d, betas, trans = model(seq_tensor.float())
        rot_6d = rot_6d.reshape(1, t, 24, 6)
        pose = matrix_to_axis_angle_torch(rotation_6d_to_matrix(rot_6d)).reshape(1, t, 72)

    # source-frame outputs from model
    pose_seq_src = pose.squeeze(0).cpu().numpy().astype(np.float32)                      # [T,72]
    betas = betas.squeeze(0).cpu().numpy().astype(np.float32)                             # [10]
    trans_seq_src = trans.reshape(1, t, 3).squeeze(0).cpu().numpy().astype(np.float32)   # [T,3]
    trans_world_seq_src = (trans_seq_src + centers_seq_src).astype(np.float32)           # [T,3]

    # unity-frame converted outputs
    pose_seq_unity = convert_pose_seq72_to_unity(
        pose_seq_src, args.source_frame, args.unity_pose_mode
    )                                                                                      # [T,72]
    trans_seq_unity = convert_vec_seq_to_unity(trans_seq_src, args.source_frame)          # [T,3]
    centers_seq_unity = convert_vec_seq_to_unity(centers_seq_src, args.source_frame)      # [T,3]
    trans_world_seq_unity = convert_vec_seq_to_unity(
        trans_world_seq_src, args.source_frame
    )                                                                                      # [T,3]

    # choose legacy alias
    if args.legacy_as == "source":
        pose_seq_legacy = pose_seq_src
        trans_seq_legacy = trans_seq_src
        centers_seq_legacy = centers_seq_src
        trans_world_seq_legacy = trans_world_seq_src
        legacy_frame_name = args.source_frame
    else:
        pose_seq_legacy = pose_seq_unity
        trans_seq_legacy = trans_seq_unity
        centers_seq_legacy = centers_seq_unity
        trans_world_seq_legacy = trans_world_seq_unity
        legacy_frame_name = "unity_ruf_lhs"

    # build save dict
    save_dict = {
        "betas": betas.astype(np.float32),
        "pcd_files": np.array([str(path) for path in selected_paths]),

        # metadata
        "meta_source_frame": np.array(args.source_frame),
        "meta_target_frame_unity": np.array("unity_ruf_lhs"),
        "meta_export_coords": np.array(args.export_coords),
        "meta_legacy_as": np.array(args.legacy_as),
        "meta_legacy_frame_name": np.array(legacy_frame_name),
        "meta_rotation_vector_unit": np.array("radian"),
        "meta_flu_to_unity_C": FLU_TO_UNITY_C.astype(np.float32),
        "meta_unity_pose_mode": np.array(args.unity_pose_mode),
        "num_frames": np.array([t], dtype=np.int32),
        "num_points_per_frame": np.array([args.num_points], dtype=np.int32),
    }

    # source keys
    if args.export_coords in ("source", "both"):
        save_dict.update({
            "pose_seq_source": pose_seq_src.astype(np.float32),
            "trans_seq_source": trans_seq_src.astype(np.float32),
            "centers_seq_source": centers_seq_src.astype(np.float32),
            "trans_world_seq_source": trans_world_seq_src.astype(np.float32),

            "pose_source": pack_single_or_seq(pose_seq_src),
            "trans_source": pack_single_or_seq(trans_seq_src),
            "centers_source": pack_single_or_seq(centers_seq_src),
            "trans_world_source": pack_single_or_seq(trans_world_seq_src),
        })

    # unity keys
    if args.export_coords in ("unity", "both"):
        save_dict.update({
            "pose_seq_unity": pose_seq_unity.astype(np.float32),
            "trans_seq_unity": trans_seq_unity.astype(np.float32),
            "centers_seq_unity": centers_seq_unity.astype(np.float32),
            "trans_world_seq_unity": trans_world_seq_unity.astype(np.float32),

            "pose_unity": pack_single_or_seq(pose_seq_unity),
            "trans_unity": pack_single_or_seq(trans_seq_unity),
            "centers_unity": pack_single_or_seq(centers_seq_unity),
            "trans_world_unity": pack_single_or_seq(trans_world_seq_unity),
        })

    # legacy keys (for backward compatibility / downstream scripts)
    save_dict.update({
        "pose_seq": pose_seq_legacy.astype(np.float32),
        "trans_seq": trans_seq_legacy.astype(np.float32),
        "centers_seq": centers_seq_legacy.astype(np.float32),
        "trans_world_seq": trans_world_seq_legacy.astype(np.float32),

        "pose": pack_single_or_seq(pose_seq_legacy),
        "trans": pack_single_or_seq(trans_seq_legacy),
        "centers": pack_single_or_seq(centers_seq_legacy),
        "trans_world": pack_single_or_seq(trans_world_seq_legacy),
    })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **save_dict)

    print(f"Saved output: {output_path}")
    print(f"frames={t}, num_points={args.num_points}, device={device.type}")
    print(
        f"source_frame={args.source_frame}, export_coords={args.export_coords}, "
        f"legacy_as={args.legacy_as}, unity_pose_mode={args.unity_pose_mode}"
    )
    print(f"legacy pose shape: {save_dict['pose'].shape}")
    print(f"betas shape: {betas.shape}")
    print(f"legacy trans shape: {save_dict['trans'].shape}")


if __name__ == "__main__":
    main()
