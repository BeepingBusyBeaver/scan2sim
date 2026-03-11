# scripts/pipeline/run_feature_pipeline.py
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any, List

from scripts.common.io_paths import resolve_repo_path


"""
1) head change (freeze backbone)
python -m scripts run-feature-pipeline \
  --points-dir data/real/LUNGE/human \
  --labels-jsonl data/feature/LUNGE_label_GT.jsonl \
  --skip-infer \
  --freeze-backbone \
  --init-backbone weights/pointnet2/yanx27/modelnet40_cls/pointnet2_ssg_wo_normals_best_model.pth \
  --train-output-dir outputs/feature_models/pointnet2_15head_lunge_head

2) inference
python -m scripts run-feature \
  --checkpoint outputs/feature_models/pointnet2_15head_lunge_head/checkpoint_best.pt \
  --input data/real/WBmotion/human \
  --input-pattern "human_*.ply" \
  --output-dir outputs/WBmotion/feature_pred \
  --output-prefix feature_pred_ \
  --gt-jsonl data/feature/WBmotion_label_GT.jsonl \
  --gt-frame-offset 1
"""

def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def command_text(cmd: List[str]) -> str:
    return " ".join(cmd)


def run_step(cmd: List[str], dry_run: bool) -> None:
    print(f"$ {command_text(cmd)}")
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("yaml") from exc
    with path.open("r", encoding="utf-8") as handle:
        obj = yaml.safe_load(handle) or {}
    if not isinstance(obj, dict):
        raise TypeError(f"Config root must be object: {path}")
    return obj


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run PointNet++ feature pipeline: "
            "build-feature-manifest -> train-feature -> run-feature"
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/feature/pointnet2_multitask.yaml",
        help="Feature pipeline config YAML.",
    )
    parser.add_argument(
        "--head-rules",
        type=str,
        default=None,
        help="Head rules JSON override. default: config paths.head_rules",
    )
    parser.add_argument(
        "--manifest-output",
        type=str,
        default=None,
        help="Manifest output JSONL path override. default: config paths.manifest",
    )
    parser.add_argument(
        "--train-output-dir",
        type=str,
        default=None,
        help="Train output dir override. default: config paths.output_dir",
    )

    parser.add_argument("--points-dir", type=str, default=None, help="Directory with input point clouds (.ply/.pcd).")
    parser.add_argument("--point-pattern", type=str, default="human_*.ply")
    parser.add_argument(
        "--labels-dir",
        type=str,
        default=None,
        help="Directory with supervision label json files (project/user labels, not necessarily LiveHPS outputs).",
    )
    parser.add_argument(
        "--labels-jsonl",
        type=str,
        default=None,
        help="Single label JSONL path with Frame/step vector format (e.g., data/feature/SQUAT_label_GT.jsonl).",
    )
    parser.add_argument("--label-pattern", type=str, default="*.json")
    parser.add_argument("--split-mode", type=str, default="random", choices=["none", "random"])
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--epochs", type=int, default=None, help="Train epochs override.")
    parser.add_argument("--device", type=str, default=None, choices=["auto", "cuda", "cpu"], help="Train/infer device override.")
    parser.add_argument("--resume", type=str, default=None, help="Train resume checkpoint path.")
    parser.add_argument(
        "--init-backbone",
        type=str,
        default=None,
        help="Original PointNet++ checkpoint for encoder initialization in train stage.",
    )
    parser.add_argument(
        "--auto-init-backbone",
        dest="auto_init_backbone",
        action="store_true",
        help="Enable auto backbone init from weights/pointnet2 in train stage (default: on).",
    )
    parser.add_argument(
        "--no-auto-init-backbone",
        dest="auto_init_backbone",
        action="store_false",
        help="Disable auto backbone init in train stage.",
    )
    parser.add_argument(
        "--freeze-backbone",
        dest="freeze_backbone",
        action="store_true",
        help="Freeze PointNet++ encoder and train only head layers.",
    )
    parser.add_argument(
        "--no-freeze-backbone",
        dest="freeze_backbone",
        action="store_false",
        help="Fine-tune full model in train stage.",
    )

    parser.add_argument("--infer-input", type=str, nargs="+", default=None, help="Infer input file/dir/glob. default: --points-dir")
    parser.add_argument("--infer-input-pattern", type=str, default="*.ply", help="Infer pattern when infer input is a directory.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Inference checkpoint. default: <train-output-dir>/checkpoint_best.pt")
    parser.add_argument("--infer-output", type=str, default=None, help="Single output JSON path for single input.")
    parser.add_argument("--infer-output-dir", type=str, default="outputs/feature_pred")
    parser.add_argument("--infer-output-prefix", type=str, default="feature_pred_")
    parser.add_argument("--infer-num-points", type=int, default=None)
    parser.add_argument("--infer-topk", type=int, default=3)
    parser.add_argument("--infer-no-normalize", action="store_true")

    parser.add_argument("--skip-manifest", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-infer", action="store_true")
    parser.add_argument("--python", type=str, default=sys.executable, help="Python executable for subprocess steps.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands only.")
    parser.set_defaults(auto_init_backbone=True, freeze_backbone=None)
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)
    root = repo_root()
    config_path = resolve_repo_path(root, args.config)
    config_obj = load_yaml(config_path)

    paths_obj = config_obj.get("paths", {}) or {}
    if not isinstance(paths_obj, dict):
        raise TypeError("config.paths must be an object.")

    manifest_path = resolve_repo_path(root, args.manifest_output or paths_obj.get("manifest", "data/feature/manifest.jsonl"))
    head_rules_path = resolve_repo_path(root, args.head_rules or paths_obj.get("head_rules", "configs/feature/label_rules.json"))
    train_output_dir = resolve_repo_path(
        root,
        args.train_output_dir or paths_obj.get("output_dir", "outputs/feature_models/pointnet2_15head"),
    )
    points_dir = resolve_repo_path(root, args.points_dir) if args.points_dir else None
    labels_dir = resolve_repo_path(root, args.labels_dir) if args.labels_dir else None
    labels_jsonl_path = resolve_repo_path(root, args.labels_jsonl) if args.labels_jsonl else None

    steps: List[List[str]] = []

    if not args.skip_manifest:
        if points_dir is None:
            raise ValueError("--points-dir is required unless --skip-manifest is set.")
        if labels_dir is None and labels_jsonl_path is None:
            raise ValueError("One of --labels-dir or --labels-jsonl is required unless --skip-manifest is set.")

        manifest_cmd = [
            args.python,
            "-m",
            "scripts",
            "build-feature-manifest",
            "--points-dir",
            str(points_dir),
            "--point-pattern",
            args.point_pattern,
            "--head-rules",
            str(head_rules_path),
            "--output",
            str(manifest_path),
            "--split-mode",
            args.split_mode,
            "--val-ratio",
            str(float(args.val_ratio)),
            "--seed",
            str(int(args.seed)),
        ]
        if labels_jsonl_path is not None:
            manifest_cmd.extend(["--labels-jsonl", str(labels_jsonl_path)])
        elif labels_dir is not None:
            manifest_cmd.extend(["--labels-dir", str(labels_dir), "--label-pattern", args.label_pattern])
        steps.append(manifest_cmd)

    if not args.skip_train:
        train_cmd = [
            args.python,
            "-m",
            "scripts",
            "train-feature",
            "--config",
            str(config_path),
            "--manifest",
            str(manifest_path),
            "--head-rules",
            str(head_rules_path),
            "--output-dir",
            str(train_output_dir),
        ]
        if args.epochs is not None:
            train_cmd.extend(["--epochs", str(int(args.epochs))])
        if args.device is not None:
            train_cmd.extend(["--device", args.device])
        if args.resume:
            train_cmd.extend(["--resume", str(resolve_repo_path(root, args.resume))])
        if args.init_backbone:
            train_cmd.extend(["--init-backbone", str(resolve_repo_path(root, args.init_backbone))])
        if not bool(args.auto_init_backbone):
            train_cmd.append("--no-auto-init-backbone")
        if args.freeze_backbone is True:
            train_cmd.append("--freeze-backbone")
        elif args.freeze_backbone is False:
            train_cmd.append("--no-freeze-backbone")
        steps.append(train_cmd)

    if not args.skip_infer:
        infer_inputs = args.infer_input if args.infer_input else ([str(points_dir)] if points_dir is not None else None)
        if not infer_inputs:
            raise ValueError("--infer-input (or --points-dir) is required unless --skip-infer is set.")

        checkpoint_path: Path | None
        if args.checkpoint:
            checkpoint_path = resolve_repo_path(root, args.checkpoint)
            if not checkpoint_path.exists():
                raise FileNotFoundError(
                    f"--checkpoint not found: {checkpoint_path}\n"
                    "Use a valid scan2sim checkpoint (.pt with model_state_dict), "
                    "or omit --checkpoint to let run-feature auto-search."
                )
        else:
            default_checkpoint = (train_output_dir / "checkpoint_best.pt").resolve()
            checkpoint_path = default_checkpoint if default_checkpoint.exists() else None

        infer_cmd = [
            args.python,
            "-m",
            "scripts",
            "run-feature",
            "--input",
            *infer_inputs,
            "--input-pattern",
            args.infer_input_pattern,
            "--topk",
            str(int(args.infer_topk)),
        ]
        if checkpoint_path is not None:
            infer_cmd.extend(["--checkpoint", str(checkpoint_path)])
        infer_cmd.extend(["--head-rules", str(head_rules_path)])
        if args.device is not None:
            infer_cmd.extend(["--device", args.device])
        if args.infer_num_points is not None:
            infer_cmd.extend(["--num-points", str(int(args.infer_num_points))])
        if args.infer_no_normalize:
            infer_cmd.append("--no-normalize")
        if args.infer_output:
            infer_cmd.extend(["--output", str(resolve_repo_path(root, args.infer_output))])
        else:
            infer_cmd.extend(
                [
                    "--output-dir",
                    str(resolve_repo_path(root, args.infer_output_dir)),
                    "--output-prefix",
                    args.infer_output_prefix,
                ]
            )
        steps.append(infer_cmd)

    print(f"[feature-pipeline] root={root}")
    print(f"[feature-pipeline] config={config_path}")
    print(f"[feature-pipeline] manifest={manifest_path}")
    print(f"[feature-pipeline] train_output_dir={train_output_dir}")
    for cmd in steps:
        run_step(cmd, dry_run=bool(args.dry_run))
    print("[feature-pipeline] done")


if __name__ == "__main__":
    main()
