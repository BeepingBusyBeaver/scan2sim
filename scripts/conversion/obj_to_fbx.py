# scripts/conversion/obj_to_fbx.py
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable, List, Sequence

try:
    import bpy
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "This script must be executed with Blender.\n"
        "Example:\n"
        "  blender --background --python scripts/conversion/obj_to_fbx.py -- "
        "--input-glob 'outputs/obj/*.obj' --output outputs/motion.fbx"
    ) from exc


"""
BLENDER="/mnt/c/Program Files/Blender Foundation/Blender 4.3/blender.exe"
"$BLENDER" --background --python scripts/conversion/obj_to_fbx.py -- \
  --input-glob "outputs/ARMmotion/obj/*.obj" \
  --output "outputs/ARMmotion/obj/livehps_motion.fbx" \
  --fps 30
"""

AXIS_TO_WM_ENUM = {
    "X": "X",
    "Y": "Y",
    "Z": "Z",
    "-X": "NEGATIVE_X",
    "-Y": "NEGATIVE_Y",
    "-Z": "NEGATIVE_Z",
}


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    parser = argparse.ArgumentParser(
        description="Convert single-frame OBJ sequence to animated FBX (shape key animation)."
    )
    parser.add_argument(
        "--input-glob",
        required=True,
        help="Glob for per-frame OBJ files (e.g. outputs/obj/livehps_mesh_*.obj)",
    )
    parser.add_argument("--output", required=True, help="Output FBX path")
    parser.add_argument("--fps", type=int, default=30, help="Animation FPS")
    parser.add_argument(
        "--start-frame", type=int, default=1, help="Start frame index in FBX timeline"
    )
    parser.add_argument(
        "--axis-forward",
        default="-Z",
        choices=["X", "Y", "Z", "-X", "-Y", "-Z"],
        help="FBX forward axis",
    )
    parser.add_argument(
        "--axis-up",
        default="Y",
        choices=["X", "Y", "Z", "-X", "-Y", "-Z"],
        help="FBX up axis",
    )
    parser.add_argument(
        "--global-scale", type=float, default=1.0, help="Global export scale"
    )
    return parser.parse_args(list(argv))


def natural_key(path: Path) -> List[object]:
    return [int(s) if s.isdigit() else s.lower() for s in re.split(r"(\d+)", path.name)]


def resolve_inputs(pattern: str) -> List[Path]:
    files = sorted(Path().glob(pattern), key=natural_key)
    if not files:
        raise FileNotFoundError(f"No OBJ files matched: {pattern}")
    return files


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def import_obj(filepath: Path, axis_forward: str, axis_up: str) -> List[bpy.types.Object]:
    before_names = set(bpy.data.objects.keys())
    if hasattr(bpy.ops.wm, "obj_import"):
        bpy.ops.wm.obj_import(
            filepath=str(filepath),
            forward_axis=AXIS_TO_WM_ENUM[axis_forward],
            up_axis=AXIS_TO_WM_ENUM[axis_up],
        )
    elif hasattr(bpy.ops.import_scene, "obj"):
        bpy.ops.import_scene.obj(
            filepath=str(filepath),
            axis_forward=axis_forward,
            axis_up=axis_up,
        )
    else:  # pragma: no cover
        raise RuntimeError("OBJ importer is unavailable in current Blender build.")
    return [obj for obj in bpy.data.objects if obj.name not in before_names]


def join_mesh_objects(objects: Iterable[bpy.types.Object], name: str) -> bpy.types.Object:
    mesh_objects = [obj for obj in objects if obj.type == "MESH"]
    if not mesh_objects:
        raise RuntimeError("No mesh object found in imported OBJ.")
    bpy.ops.object.select_all(action="DESELECT")
    if len(mesh_objects) == 1:
        obj = mesh_objects[0]
        bpy.context.view_layer.objects.active = obj
        obj.name = name
        return obj
    for obj in mesh_objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = mesh_objects[0]
    bpy.ops.object.join()
    joined = bpy.context.view_layer.objects.active
    joined.name = name
    return joined


def add_shape_key_frame(
    base_obj: bpy.types.Object, frame_obj: bpy.types.Object, key_name: str
) -> None:
    if len(base_obj.data.vertices) != len(frame_obj.data.vertices):
        raise RuntimeError(
            f"Topology mismatch: base vertices={len(base_obj.data.vertices)} "
            f"vs frame vertices={len(frame_obj.data.vertices)} ({frame_obj.name})"
        )
    key_block = base_obj.shape_key_add(name=key_name, from_mix=False)
    for idx, vert in enumerate(frame_obj.data.vertices):
        key_block.data[idx].co = vert.co


def keyframe_shape_keys(base_obj: bpy.types.Object, start_frame: int, frame_count: int) -> None:
    if base_obj.data.shape_keys is None:
        return
    key_blocks = base_obj.data.shape_keys.key_blocks
    if len(key_blocks) <= 1:
        return
    end_frame = start_frame + frame_count - 1
    for key_block in key_blocks[1:]:
        key_block.value = 0.0
        key_block.keyframe_insert(data_path="value", frame=start_frame)
        key_block.keyframe_insert(data_path="value", frame=end_frame)
    for idx, key_block in enumerate(key_blocks[1:], start=1):
        target_frame = start_frame + idx
        prev_frame = max(start_frame, target_frame - 1)
        next_frame = min(end_frame, target_frame + 1)
        key_block.value = 0.0
        key_block.keyframe_insert(data_path="value", frame=prev_frame)
        key_block.value = 1.0
        key_block.keyframe_insert(data_path="value", frame=target_frame)
        key_block.value = 0.0
        key_block.keyframe_insert(data_path="value", frame=next_frame)


def export_fbx(
    base_obj: bpy.types.Object,
    out_path: Path,
    axis_forward: str,
    axis_up: str,
    global_scale: float,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action="DESELECT")
    base_obj.select_set(True)
    bpy.context.view_layer.objects.active = base_obj
    bpy.ops.export_scene.fbx(
        filepath=str(out_path),
        use_selection=True,
        object_types={"MESH"},
        add_leaf_bones=False,
        bake_anim=True,
        bake_anim_use_nla_strips=False,
        bake_anim_use_all_actions=False,
        bake_anim_force_startend_keying=True,
        bake_anim_step=1.0,
        axis_forward=axis_forward,
        axis_up=axis_up,
        global_scale=global_scale,
    )


def main(argv: Sequence[str] | None = None) -> None:
    argv = sys.argv if argv is None else list(argv)
    args = parse_args(argv)
    input_files = resolve_inputs(args.input_glob)
    clear_scene()

    scene = bpy.context.scene
    scene.render.fps = args.fps
    scene.frame_start = args.start_frame
    scene.frame_end = args.start_frame + len(input_files) - 1

    first_imported = import_obj(
        input_files[0], axis_forward=args.axis_forward, axis_up=args.axis_up
    )
    base_obj = join_mesh_objects(first_imported, name="motion_mesh")
    base_obj.shape_key_add(name="Basis", from_mix=False)

    for frame_idx, obj_path in enumerate(input_files[1:], start=1):
        imported = import_obj(obj_path, axis_forward=args.axis_forward, axis_up=args.axis_up)
        frame_obj = join_mesh_objects(imported, name=f"_tmp_frame_{frame_idx:06d}")
        add_shape_key_frame(
            base_obj=base_obj,
            frame_obj=frame_obj,
            key_name=f"frame_{frame_idx + args.start_frame:06d}",
        )
        bpy.data.objects.remove(frame_obj, do_unlink=True)

    keyframe_shape_keys(
        base_obj=base_obj, start_frame=args.start_frame, frame_count=len(input_files)
    )
    export_fbx(
        base_obj=base_obj,
        out_path=Path(args.output),
        axis_forward=args.axis_forward,
        axis_up=args.axis_up,
        global_scale=args.global_scale,
    )
    print(f"[done] input_frames={len(input_files)} output={Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
