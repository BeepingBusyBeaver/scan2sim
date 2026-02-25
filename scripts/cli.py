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
    "pipeline-livehps": ("scripts.pipeline.run_livehps_pipeline", "main"),
    "run-pipeline": ("scripts.pipeline.run_livehps_pipeline", "main"),
    "match-one": ("scripts.eval.match_one", "main"),
    "npz-to-obj": ("scripts.conversion.npz_to_obj", "main"),
    "npz2obj": ("scripts.conversion.npz_to_obj", "main"),
    "npz-to-quat": ("scripts.conversion.npz_to_quat", "main"),
    "npz2quat": ("scripts.conversion.npz_to_quat", "main"),
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
    print("  pcap2pcd -> pcd2human -> human2smpl -> npz2quat/npz2obj -> quat2unity -> unity2label")
    print("  run-pipeline (or pipeline-livehps) for end-to-end execution")


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
