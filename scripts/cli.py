# scripts/cli.py
from __future__ import annotations

import importlib
import sys
from typing import Dict, Tuple


COMMANDS: Dict[str, Tuple[str, str]] = {
    "pcap-to-pcd": ("scripts.ouster.pcap_to_pcd", "main"),
    "pcap2pcd": ("scripts.ouster.pcap_to_pcd", "main"),
    "extract-person": ("scripts.pc.extract_person", "main"),
    "pcd-to-human": ("scripts.pc.extract_person", "main"),
    "pcd2human": ("scripts.pc.extract_person", "main"),
    "infer-smpl": ("scripts.infer.infer_point", "main"),
    "human-to-smpl": ("scripts.infer.infer_point", "main"),
    "human2smpl": ("scripts.infer.infer_point", "main"),
    "run-livehps": ("scripts.pipeline.run_livehps_pipeline", "main"),
    "run-feature-pipeline": ("scripts.pipeline.run_feature_pipeline", "main"),
    "run-parser-pipeline": ("scripts.pipeline.run_parser_decoder_pipeline", "main"),
    "run-parser-decoder": ("scripts.parser.run_parser_decoder", "main"),
    "run-parser": ("scripts.parser.run_parser_decoder", "main"),
    "build-parser-playback": ("scripts.parser.build_parser_playback", "main"),
    "build-playback": ("scripts.parser.build_parser_playback", "main"),
    "play-parser-playback": ("scripts.parser.play_parser_playback", "main"),
    "play-playback": ("scripts.parser.play_parser_playback", "main"),
    "run-feature": ("scripts.infer.infer_feature_classifier", "main"),
    "train-feature": ("scripts.train.train_feature_classifier", "main"),
    "train-pointnet2": ("scripts.train.train_feature_classifier", "main"),
    "infer-feature": ("scripts.infer.infer_feature_classifier", "main"),
    "infer-pointnet2": ("scripts.infer.infer_feature_classifier", "main"),
    "build-feature-manifest": ("scripts.data.build_feature_manifest", "main"),
    "feature-manifest": ("scripts.data.build_feature_manifest", "main"),
    "match-one": ("scripts.eval.match_one", "main"),
    "npz-to-obj": ("scripts.conversion.npz_to_obj", "main"),
    "npz2obj": ("scripts.conversion.npz_to_obj", "main"),
    "pcd-to-ply": ("scripts.conversion.pcd_to_ply", "main"),
    "pcd2ply": ("scripts.conversion.pcd_to_ply", "main"),
    "npz-to-fbx": ("scripts.conversion.npz_to_fbx", "main"),
    "npz2fbx": ("scripts.conversion.npz_to_fbx", "main"),
    "npz-to-quat": ("scripts.conversion.npz_to_quat", "main"),
    "npz2quat": ("scripts.conversion.npz_to_quat", "main"),
    "fit-bind-local": ("scripts.conversion.fit_bind_local_quat", "main"),
    "fit-bind-local-quat": ("scripts.conversion.fit_bind_local_quat", "main"),
    "fitbind": ("scripts.conversion.fit_bind_local_quat", "main"),
    "quat-to-unity": ("scripts.conversion.quat_to_unity", "main"),
    "quat2unity": ("scripts.conversion.quat_to_unity", "main"),
    "label-classify": ("scripts.labeling.classify_label", "main"),
    "unity-to-label": ("scripts.labeling.classify_label", "main"),
    "unity2label": ("scripts.labeling.classify_label", "main"),
    "label-compare": ("scripts.labeling.compare_step_labels", "main"),
    "label-step-jsonl": ("scripts.labeling.export_step_jsonl", "main"),
    "label2jsonl": ("scripts.labeling.export_step_jsonl", "main"),
    "workspace-init": ("scripts.workspace.workspace_layout", "main"),
    "workspace-doctor": ("scripts.workspace.workspace_layout", "main"),
    "workspace-layout": ("scripts.workspace.workspace_layout", "main"),
}


def print_help() -> None:
    command_list = "\n".join(f"  - {name}" for name in sorted(COMMANDS.keys()))
    print("Usage: python -m scripts <command> [args...]")
    print("Commands:")
    print(command_list)
    print("\nPipeline shortcuts:")
    print("  pcap2pcd -> pcd2human -> human2smpl -> npz2quat/npz2obj/npz2fbx -> quat2unity -> unity2label")
    print("  run-livehps (LiveHPS end-to-end)")
    print("  run-feature-pipeline (PointNet++ manifest/train/infer)")
    print("  run-feature (PointNet++ feature-only inference)")
    print("  run-parser-pipeline (rule-based parser-decoder)")
    print("  run-parser-decoder (point cloud -> part boxes -> labels)")
    print("  build-parser-playback (PLY sequence -> compressed playback NPZ)")
    print("  play-parser-playback (NPZ playback in Open3D viewer)")


def main() -> None:
    if len(sys.argv) < 2:
        print_help()
        return

    command = sys.argv[1].strip()
    target_spec = COMMANDS.get(command)
    if target_spec is None:
        print(f"Unknown command: {command}")
        print_help()
        raise SystemExit(2)

    module_name, function_name = target_spec
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        missing = getattr(exc, "name", None) or str(exc)
        print(f"Failed to load command '{command}': missing dependency '{missing}'.")
        raise SystemExit(3) from exc
    target = getattr(module, function_name)
    forwarded_args = sys.argv[2:]
    if command == "workspace-init":
        sys.argv = [f"scripts.{command}", "init"] + forwarded_args
    elif command == "workspace-doctor":
        sys.argv = [f"scripts.{command}", "doctor"] + forwarded_args
    elif command == "workspace-layout":
        sys.argv = [f"scripts.{command}", "layout"] + forwarded_args
    else:
        sys.argv = [f"scripts.{command}"] + forwarded_args
    target()


if __name__ == "__main__":
    main()
