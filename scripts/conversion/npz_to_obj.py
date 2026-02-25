# scripts/conversion/npz_to_obj.py
import argparse
import sys
import glob
from pathlib import Path
import importlib

import numpy as np
import torch
from scripts.common.io_paths import (
    DATA_INTEROP_OBJ_MESH,
    resolve_repo_path,
)


SCAN2SIM_ROOT = Path(__file__).resolve().parents[2]
LIVEHPS_ROOT = (SCAN2SIM_ROOT.parent / "LiveHPS").resolve()
SMPL_MODULE_DIR = LIVEHPS_ROOT / "smpl"
if not SMPL_MODULE_DIR.exists():
    raise FileNotFoundError(f"SMPL directory not found: {SMPL_MODULE_DIR}")

# LiveHPS smpl/smpl.py가 내부적으로 `from core ...`를 쓰는 구조를 고려해
# smpl 디렉터리 자체를 sys.path 맨 앞에 둔다.
if str(SMPL_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(SMPL_MODULE_DIR))

VIBE_DATA_DIR = (SMPL_MODULE_DIR / "data" / "vibe_data").resolve()
if not VIBE_DATA_DIR.exists():
    raise FileNotFoundError(f"VIBE data directory not found: {VIBE_DATA_DIR}")

core_config = importlib.import_module("core.config")
core_config.VIBE_DATA_DIR = str(VIBE_DATA_DIR)

try:
    from smpl.smpl import SMPL, SMPL_MODEL_DIR
except ImportError:
    from smpl import SMPL, SMPL_MODEL_DIR

"""
Single:
python -m scripts.conversion.npz_to_obj \
  --input outputs/smpl/livehps_smpl.npz \
  --output outputs/obj/livehps_mesh.obj \
  --device auto

Batch (glob):
python -m scripts.conversion.npz_to_obj \
  --input "outputs/smpl/VAR_livehps_smpl_*.npz" \
  --output outputs/obj \
  --device auto

Batch (directory):
python -m scripts.conversion.npz_to_obj \
  --input outputs/smpl \
  --output outputs/obj \
  --device auto
"""

def parse_args():
    parser = argparse.ArgumentParser(
        description="Export SMPL mesh OBJ files from LiveHPS inference NPZ(s)."
    )
    parser.add_argument(
        "--input",
        type=str,
        nargs="+",
        required=True,
        help=(
            "Input NPZ path(s), a directory, or a glob pattern. "
            "Examples: outputs/smpl/livehps_smpl.npz, "
            "outputs/smpl, outputs/smpl/livehps_smpl_*.npz"
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DATA_INTEROP_OBJ_MESH,
        help=(
            "Output path. Single input: .obj path recommended (same as before). "
            "Batch input: treat as output directory (if .obj is given, its parent dir is used). "
            "Multi-frame NPZ: if output is .obj, files will be exported as <stem>_0000.obj, <stem>_0001.obj ..."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device for SMPL forward.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="frame",
        help="Filename prefix when exporting multiple OBJ files (multi-frame output dir case).",
    )
    return parser.parse_args()


def choose_device(device_arg: str):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no CUDA device is available.")
    return torch.device(device_arg)


def get_npz_array(data, keys):
    for key in keys:
        if key in data.files:
            return data[key], key
    available = ", ".join(sorted(data.files))
    raise KeyError(f"None of keys {keys} found in npz. Available keys: {available}")


def normalize_pose(pose: np.ndarray):
    pose = np.asarray(pose, dtype=np.float32)
    if pose.ndim == 1:
        pose = pose.reshape(1, -1)
    if pose.ndim != 2 or pose.shape[1] != 72:
        raise ValueError(f"Pose must have shape [T,72] or [72], got {pose.shape}")
    return pose


def normalize_trans(trans: np.ndarray, num_frames: int):
    trans = np.asarray(trans, dtype=np.float32)
    if trans.ndim == 1:
        trans = trans.reshape(1, -1)
    if trans.ndim != 2 or trans.shape[1] != 3:
        raise ValueError(f"Translation must have shape [T,3] or [3], got {trans.shape}")
    if trans.shape[0] == 1 and num_frames > 1:
        trans = np.repeat(trans, repeats=num_frames, axis=0)
    if trans.shape[0] != num_frames:
        raise ValueError(
            f"Translation frame count mismatch: pose={num_frames}, trans={trans.shape[0]}"
        )
    return trans


def normalize_betas(betas: np.ndarray, num_frames: int):
    betas = np.asarray(betas, dtype=np.float32)
    if betas.ndim == 1:
        betas = betas.reshape(1, -1)
    if betas.ndim != 2 or betas.shape[1] != 10:
        raise ValueError(f"Betas must have shape [T,10] or [10], got {betas.shape}")
    if betas.shape[0] == 1 and num_frames > 1:
        betas = np.repeat(betas, repeats=num_frames, axis=0)
    if betas.shape[0] != num_frames:
        raise ValueError(f"Betas frame count mismatch: pose={num_frames}, betas={betas.shape[0]}")
    return betas


def write_obj(path: Path, vertices: np.ndarray, faces: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for vertex in vertices:
            handle.write(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
        for face in faces:
            handle.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")


def resolve_output_paths(output: Path, num_frames: int, prefix: str):
    if num_frames == 1:
        if output.suffix.lower() == ".obj":
            return [output]
        return [output / f"{prefix}_0000.obj"]

    if output.suffix.lower() == ".obj":
        out_dir = output.parent
        frame_prefix = output.stem
    else:
        out_dir = output
        frame_prefix = prefix
    return [out_dir / f"{frame_prefix}_{index:04d}.obj" for index in range(num_frames)]


def load_params(npz_path: Path):
    data = np.load(npz_path, allow_pickle=True)

    # source 우선
    pose_raw, pose_key = get_npz_array(
        data, ["pose_seq_source", "pose_source", "pose_seq", "pose"]
    )
    trans_raw, trans_key = get_npz_array(
        data,
        [
            "trans_world_seq_source", "trans_seq_source",
            "trans_world_seq", "trans_seq", "trans_world", "trans"
        ],
    )
    betas_raw, betas_key = get_npz_array(data, ["betas"])

    pose = normalize_pose(pose_raw)
    num_frames = pose.shape[0]
    trans = normalize_trans(trans_raw, num_frames)
    betas = normalize_betas(betas_raw, num_frames)

    return pose, betas, trans, pose_key, betas_key, trans_key


def _has_glob_magic(s: str) -> bool:
    return any(ch in s for ch in ("*", "?", "["))


def collect_input_files(repo_root: Path, input_tokens):
    """
    input_tokens: list[str]
      - file path: .../livehps_smpl_123.npz
      - dir path:  .../outputs/smpl  -> livehps_smpl_*.npz
      - glob:      .../outputs/smpl/livehps_smpl_*.npz
    """
    files = []

    for token in input_tokens:
        token_path = Path(token).expanduser()
        token_str = str(token_path)

        # glob pattern
        if _has_glob_magic(token_str):
            if not token_path.is_absolute():
                token_path = repo_root / token_path
            # glob은 tilde(~) 확장을 안 하므로 expanduser() 후 문자열로 넘김
            matches = glob.glob(str(token_path), recursive=False)
            for m in matches:
                p = Path(m).resolve()
                if p.is_file() and p.suffix.lower() == ".npz":
                    files.append(p)
            continue

        p = resolve_repo_path(repo_root, token_path)
        if p.is_dir():
            for f in sorted(p.glob("livehps_smpl_*.npz")):
                files.append(f.resolve())
        else:
            files.append(p)

    # dedup + sort
    uniq = []
    seen = set()
    for p in files:
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        uniq.append(rp)
    return sorted(uniq)


def derive_mesh_obj_name(npz_path: Path) -> str:
    """
    livehps_smpl_123.npz -> livehps_mesh_123.obj
    livehps_smpl.npz     -> livehps_mesh.obj
    기타                 -> <stem>.obj (fallback)
    """
    stem = npz_path.stem
    if stem == "livehps_smpl":
        return "livehps_mesh.obj"
    if stem.startswith("livehps_smpl_"):
        suffix = stem[len("livehps_smpl_"):]
        return f"livehps_mesh_{suffix}.obj"
    if stem.startswith("livehps_smpl"):
        return stem.replace("livehps_smpl", "livehps_mesh", 1) + ".obj"
    return stem + ".obj"


def export_one(npz_path: Path, output_path: Path, prefix: str, smpl: SMPL, device: torch.device):
    if not npz_path.exists():
        raise FileNotFoundError(f"Input npz not found: {npz_path}")

    pose, betas, trans, pose_key, betas_key, trans_key = load_params(npz_path)
    num_frames = pose.shape[0]

    with torch.no_grad():
        pose_tensor = torch.from_numpy(pose).to(device=device, dtype=torch.float32)
        betas_tensor = torch.from_numpy(betas).to(device=device, dtype=torch.float32)
        trans_tensor = torch.from_numpy(trans).to(device=device, dtype=torch.float32)
        smpl_output = smpl(
            betas=betas_tensor,
            body_pose=pose_tensor[:, 3:],
            global_orient=pose_tensor[:, :3],
            transl=trans_tensor,
        )

    vertices = smpl_output.vertices.detach().cpu().numpy().astype(np.float32)
    faces = smpl.faces.astype(np.int32)

    output_paths = resolve_output_paths(output_path, num_frames, prefix)
    for frame_index, path in enumerate(output_paths):
        write_obj(path, vertices[frame_index], faces)

    return num_frames, output_paths, (pose_key, betas_key, trans_key)


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]

    input_paths = collect_input_files(repo_root, args.input)
    if len(input_paths) == 0:
        raise FileNotFoundError(f"No input npz found from: {args.input}")

    output_path_arg = resolve_repo_path(repo_root, args.output)
    device = choose_device(args.device)

    smpl = SMPL(SMPL_MODEL_DIR, create_transl=False).to(device)

    # 단일/일괄 모드 자동 판별: 입력이 2개 이상이면 일괄
    batch_mode = len(input_paths) > 1

    total_frames = 0
    exported = 0

    if not batch_mode:
        npz_path = input_paths[0]
        num_frames, output_paths, keys = export_one(
            npz_path=npz_path,
            output_path=output_path_arg,
            prefix=args.prefix,
            smpl=smpl,
            device=device,
        )
        total_frames += num_frames
        exported += 1

        pose_key, betas_key, trans_key = keys
        print(f"Input: {npz_path}")
        print(f"Loaded keys: pose={pose_key}, betas={betas_key}, trans={trans_key}")
        print(f"Frames exported: {num_frames}")
        if num_frames == 1:
            print(f"OBJ saved: {output_paths[0]}")
        else:
            print(f"OBJ directory: {output_paths[0].parent}")
            print(f"First OBJ: {output_paths[0].name}")
        return

    # batch: output은 디렉터리처럼 취급
    out_dir = output_path_arg if output_path_arg.suffix.lower() != ".obj" else output_path_arg.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Batch mode: {len(input_paths)} file(s)")
    print(f"Output dir: {out_dir}")

    for npz_path in input_paths:
        out_obj = out_dir / derive_mesh_obj_name(npz_path)
        num_frames, output_paths, keys = export_one(
            npz_path=npz_path,
            output_path=out_obj,
            prefix=args.prefix,
            smpl=smpl,
            device=device,
        )
        total_frames += num_frames
        exported += 1

        if num_frames == 1:
            print(f"- {npz_path.name} -> {output_paths[0].name}")
        else:
            print(f"- {npz_path.name} -> {output_paths[0].name} ... ({num_frames} frames)")

    print(f"Done. Exported files: {exported}, total frames: {total_frames}")


if __name__ == "__main__":
    main()
