# scripts/train/train_feature_classifier.py
from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from scripts.common.io_paths import resolve_repo_path
from scripts.feature.heads import HeadSpec, head_specs_to_jsonable, load_head_specs
from scripts.feature.losses import compute_multitask_loss
from scripts.feature.manifest_dataset import (
    PointCloudHeadDataset,
    load_manifest_jsonl,
    split_manifest_samples,
)
from scripts.feature.pointnet2_multitask import build_model_from_config


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def choose_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_yaml_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    if not isinstance(cfg, dict):
        raise TypeError(f"Config root must be object: {path}")
    return cfg


def _load_yanx27_backbone_into_model(model: torch.nn.Module, checkpoint_path: Path) -> tuple[int, int]:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        source_state = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
    else:
        raise TypeError("Unsupported checkpoint payload type.")
    if not isinstance(source_state, dict):
        raise TypeError("Backbone checkpoint does not contain state_dict.")

    target_state = model.state_dict()
    loaded = 0
    scanned = 0

    conv_target_indices = [0, 3, 6]
    bn_target_indices = [1, 4, 7]
    bn_fields = ["weight", "bias", "running_mean", "running_var", "num_batches_tracked"]

    for stage in (1, 2):
        stage_idx = stage - 1
        for layer_idx in (0, 1, 2):
            scanned += 1
            source_conv = f"sa{stage}.mlp_convs.{layer_idx}.weight"
            target_conv = f"encoder.sa_layers.{stage_idx}.mlp.{conv_target_indices[layer_idx]}.weight"
            if source_conv in source_state and target_conv in target_state:
                if tuple(source_state[source_conv].shape) == tuple(target_state[target_conv].shape):
                    target_state[target_conv] = source_state[source_conv]
                    loaded += 1

            source_bn_prefix = f"sa{stage}.mlp_bns.{layer_idx}"
            target_bn_prefix = f"encoder.sa_layers.{stage_idx}.mlp.{bn_target_indices[layer_idx]}"
            for field in bn_fields:
                scanned += 1
                source_bn = f"{source_bn_prefix}.{field}"
                target_bn = f"{target_bn_prefix}.{field}"
                if source_bn in source_state and target_bn in target_state:
                    if tuple(source_state[source_bn].shape) == tuple(target_state[target_bn].shape):
                        target_state[target_bn] = source_state[source_bn]
                        loaded += 1

    if loaded == 0:
        raise ValueError("No compatible encoder weights were found in the provided PointNet++ checkpoint.")

    model.load_state_dict(target_state, strict=False)
    return loaded, scanned


def _auto_find_backbone_checkpoint(root: Path) -> Path | None:
    candidates = [
        root / "weights/pointnet2/yanx27/modelnet40_cls/pointnet2_ssg_wo_normals_best_model.pth",
        root / "weights/pointnet2/yanx27/modelnet40_cls/pointnet2_msg_normals_best_model.pth",
    ]
    for path in candidates:
        if path.exists() and path.is_file():
            return path.resolve()
    return None


def _set_encoder_trainable(model: torch.nn.Module, trainable: bool) -> tuple[int, int]:
    if not hasattr(model, "encoder"):
        raise AttributeError("Model has no 'encoder' module for backbone freezing.")
    encoder = getattr(model, "encoder")
    if not isinstance(encoder, torch.nn.Module):
        raise TypeError("Model encoder is not a torch.nn.Module.")
    total = 0
    changed = 0
    for param in encoder.parameters():
        total += 1
        desired = bool(trainable)
        if param.requires_grad != desired:
            param.requires_grad = desired
            changed += 1
    return changed, total


def evaluate_epoch(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    heads: Sequence[HeadSpec],
    device: torch.device,
) -> Tuple[float, Dict[str, float], Dict[str, float]]:
    model.eval()
    total_count = 0
    total_loss = 0.0
    head_loss_sums = defaultdict(float)
    head_acc_sums = defaultdict(float)

    with torch.no_grad():
        for points, labels in loader:
            points = points.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(points)
            loss, head_losses, head_acc = compute_multitask_loss(logits, labels, heads)
            batch_size = int(points.shape[0])
            total_count += batch_size
            total_loss += float(loss.detach().cpu().item()) * batch_size
            for name, value in head_losses.items():
                head_loss_sums[name] += float(value) * batch_size
            for name, value in head_acc.items():
                head_acc_sums[name] += float(value) * batch_size

    if total_count == 0:
        raise ValueError("Validation dataloader produced no batches.")

    mean_loss = total_loss / total_count
    mean_head_loss = {name: value / total_count for name, value in head_loss_sums.items()}
    mean_head_acc = {name: value / total_count for name, value in head_acc_sums.items()}
    return mean_loss, mean_head_loss, mean_head_acc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PointNet++ feature-only multi-head classifier.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/feature/pointnet2_multitask.yaml",
        help="Training config YAML.",
    )
    parser.add_argument("--manifest", type=str, default=None, help="Override data manifest JSONL path.")
    parser.add_argument("--head-rules", type=str, default=None, help="Override head rules JSON path.")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory.")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs from config.")
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cuda", "cpu"],
        default=None,
        help="Override device from config.",
    )
    parser.add_argument("--resume", type=str, default=None, help="Optional checkpoint path to resume from.")
    parser.add_argument(
        "--init-backbone",
        type=str,
        default=None,
        help=(
            "Original PointNet++ checkpoint (.pth/.pt) for encoder initialization only. "
            "Head layers remain newly initialized."
        ),
    )
    parser.add_argument(
        "--auto-init-backbone",
        dest="auto_init_backbone",
        action="store_true",
        help="Auto-load default PointNet++ weights from weights/pointnet2 if available (default: on).",
    )
    parser.add_argument(
        "--no-auto-init-backbone",
        dest="auto_init_backbone",
        action="store_false",
        help="Disable automatic PointNet++ backbone initialization.",
    )
    parser.add_argument(
        "--freeze-backbone",
        dest="freeze_backbone",
        action="store_true",
        help="Freeze PointNet++ encoder and train only newly attached head layers.",
    )
    parser.add_argument(
        "--no-freeze-backbone",
        dest="freeze_backbone",
        action="store_false",
        help="Fine-tune full model (encoder + heads).",
    )
    parser.set_defaults(auto_init_backbone=True, freeze_backbone=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = repo_root()
    config_path = resolve_repo_path(root, args.config)
    cfg = load_yaml_config(config_path)

    exp_cfg = cfg.get("experiment", {}) or {}
    path_cfg = cfg.get("paths", {}) or {}
    data_cfg = cfg.get("data", {}) or {}
    model_cfg = cfg.get("model", {}) or {}
    train_cfg = cfg.get("train", {}) or {}
    split_cfg = data_cfg.get("split", {}) or {}
    augment_cfg = data_cfg.get("augment", {}) or {}

    seed = int(exp_cfg.get("seed", 42))
    seed_everything(seed)

    manifest_path = resolve_repo_path(root, args.manifest or path_cfg.get("manifest", "data/feature/manifest.jsonl"))
    head_rules_path = resolve_repo_path(root, args.head_rules or path_cfg.get("head_rules", "configs/feature/label_rules.json"))
    output_dir = resolve_repo_path(root, args.output_dir or path_cfg.get("output_dir", "outputs/feature_models/pointnet2_15head"))
    output_dir.mkdir(parents=True, exist_ok=True)

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}\n"
            "Create it first with build-feature-manifest using your own label JSON directory, e.g.\n"
            "  python -m scripts build-feature-manifest --points-dir <hm_ply_dir> --labels-dir <label_json_dir>"
        )

    head_specs = load_head_specs(head_rules_path)
    samples = load_manifest_jsonl(manifest_path, root)
    train_samples, val_samples = split_manifest_samples(
        samples,
        mode=str(split_cfg.get("mode", "field")),
        train_value=str(split_cfg.get("train_value", "train")),
        val_value=str(split_cfg.get("val_value", "val")),
        val_ratio=float(split_cfg.get("val_ratio", 0.2)),
        seed=seed,
    )

    num_points = int(data_cfg.get("num_points", 1024))
    normalize = bool(data_cfg.get("normalize", True))
    cache_in_memory = bool(data_cfg.get("cache_in_memory", False))
    train_dataset = PointCloudHeadDataset(
        samples=train_samples,
        head_specs=head_specs,
        num_points=num_points,
        normalize=normalize,
        augment=bool(augment_cfg.get("enabled", True)),
        jitter_std=float(augment_cfg.get("jitter_std", 0.005)),
        jitter_clip=float(augment_cfg.get("jitter_clip", 0.02)),
        cache_in_memory=cache_in_memory,
    )
    val_dataset = PointCloudHeadDataset(
        samples=val_samples,
        head_specs=head_specs,
        num_points=num_points,
        normalize=normalize,
        augment=False,
        jitter_std=0.0,
        jitter_clip=0.0,
        cache_in_memory=cache_in_memory,
    )

    batch_size = int(train_cfg.get("batch_size", 12))
    num_workers = int(train_cfg.get("num_workers", 0))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    device_arg = args.device or str(train_cfg.get("device", "auto"))
    device = choose_device(device_arg)
    model = build_model_from_config(model_cfg, head_specs).to(device)
    freeze_backbone = (
        bool(args.freeze_backbone)
        if args.freeze_backbone is not None
        else bool(train_cfg.get("freeze_backbone", True))
    )

    total_epochs = int(args.epochs if args.epochs is not None else train_cfg.get("epochs", 60))
    log_interval = max(1, int(train_cfg.get("log_interval", 10)))
    save_every = max(1, int(train_cfg.get("save_every", 5)))

    start_epoch = 1
    best_val_loss = float("inf")
    init_backbone_path_used: Path | None = None
    if args.resume:
        resume_path = resolve_repo_path(root, args.resume)
        ckpt = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val_loss = float(ckpt.get("best_val_loss", best_val_loss))
        print(f"[train-feature] resumed from {resume_path} (start_epoch={start_epoch})")
    else:
        if args.init_backbone:
            init_backbone_path_used = resolve_repo_path(root, args.init_backbone)
            if not init_backbone_path_used.exists():
                raise FileNotFoundError(f"--init-backbone not found: {init_backbone_path_used}")
        elif bool(args.auto_init_backbone):
            init_backbone_path_used = _auto_find_backbone_checkpoint(root)

        if init_backbone_path_used is not None:
            loaded_count, scanned_count = _load_yanx27_backbone_into_model(model, init_backbone_path_used)
            print(
                f"[train-feature] backbone initialized from {init_backbone_path_used} "
                f"(loaded={loaded_count}/{scanned_count}, heads=new)"
            )
        else:
            print("[train-feature] backbone init skipped (random init).")
        if freeze_backbone and init_backbone_path_used is None:
            raise ValueError(
                "freeze_backbone=True but no pretrained backbone checkpoint was loaded. "
                "Set --init-backbone (or place default checkpoint under weights/pointnet2), "
                "or pass --no-freeze-backbone."
            )

    changed_count, total_count = _set_encoder_trainable(model, trainable=not freeze_backbone)
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters found. Check freeze settings.")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(train_cfg.get("lr", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    if args.resume:
        resume_path = resolve_repo_path(root, args.resume)
        ckpt = torch.load(resume_path, map_location="cpu")
        if "optimizer_state_dict" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except ValueError as exc:
                print(
                    "[train-feature] skipped optimizer state restore due to param-group mismatch "
                    f"(freeze_backbone={freeze_backbone}): {exc}"
                )

    with (output_dir / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)

    print(f"[train-feature] manifest={manifest_path}")
    print(f"[train-feature] head_rules={head_rules_path}")
    print(f"[train-feature] train={len(train_dataset)} val={len(val_dataset)}")
    print(f"[train-feature] device={device} epochs={total_epochs}")
    print(
        f"[train-feature] freeze_backbone={freeze_backbone} "
        f"(encoder_requires_grad_changed={changed_count}/{total_count}) "
        f"trainable_tensors={len(trainable_params)}"
    )

    for epoch in range(start_epoch, total_epochs + 1):
        model.train()
        if freeze_backbone:
            model.encoder.eval()
        total_count = 0
        total_loss = 0.0

        for batch_index, (points, labels) in enumerate(train_loader, start=1):
            points = points.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(points)
            loss, _, _ = compute_multitask_loss(logits, labels, head_specs)
            loss.backward()
            optimizer.step()

            batch_size_now = int(points.shape[0])
            total_count += batch_size_now
            total_loss += float(loss.detach().cpu().item()) * batch_size_now

            if batch_index % log_interval == 0:
                print(
                    f"[train-feature] epoch={epoch}/{total_epochs} "
                    f"step={batch_index}/{len(train_loader)} loss={float(loss.detach().cpu().item()):.6f}"
                )

        if total_count == 0:
            raise ValueError("Training dataloader produced no batches.")
        train_loss = total_loss / total_count

        val_loss, val_head_loss, val_head_acc = evaluate_epoch(
            model=model,
            loader=val_loader,
            heads=head_specs,
            device=device,
        )

        head_acc_mean = float(np.mean(list(val_head_acc.values()))) if val_head_acc else 0.0
        print(
            f"[train-feature] epoch={epoch} train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} val_head_acc_mean={head_acc_mean:.4f}"
        )

        checkpoint = {
            "epoch": epoch,
            "best_val_loss": min(best_val_loss, val_loss),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "head_specs": head_specs_to_jsonable(head_specs),
            "config": cfg,
            "metrics": {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_head_loss": val_head_loss,
                "val_head_acc": val_head_acc,
            },
        }
        torch.save(checkpoint, output_dir / "checkpoint_last.pt")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, output_dir / "checkpoint_best.pt")
            print(f"[train-feature] best checkpoint updated (val_loss={best_val_loss:.6f})")

        if epoch % save_every == 0:
            torch.save(checkpoint, output_dir / f"checkpoint_epoch_{epoch:03d}.pt")

    print(f"[train-feature] done. best_val_loss={best_val_loss:.6f}")


if __name__ == "__main__":
    main()
