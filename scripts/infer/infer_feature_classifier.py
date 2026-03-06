from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import yaml

from scripts.common.io_paths import resolve_repo_path
from scripts.common.json_utils import read_json, write_json
from scripts.common.path_utils import natural_key_path
from scripts.common.pc_io import read_point_cloud_any
from scripts.feature.heads import HeadSpec, load_head_specs, load_head_specs_from_object
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


def load_yaml_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    if not isinstance(cfg, dict):
        raise TypeError(f"Config root must be object: {path}")
    return cfg


def _candidate_checkpoint_paths(root: Path, path_cfg: Dict[str, Any]) -> List[Path]:
    candidates: List[Path] = []

    output_dir_raw = path_cfg.get("output_dir")
    if output_dir_raw:
        output_dir = resolve_repo_path(root, str(output_dir_raw))
        candidates.extend(
            [
                output_dir / "checkpoint_best.pt",
                output_dir / "checkpoint_last.pt",
            ]
        )
        epoch_ckpts = sorted(output_dir.glob("checkpoint_epoch_*.pt"), reverse=True)
        candidates.extend(epoch_ckpts)

    default_output_dir = resolve_repo_path(root, "outputs/feature_models/pointnet2_15head")
    candidates.extend(
        [
            default_output_dir / "checkpoint_best.pt",
            default_output_dir / "checkpoint_last.pt",
        ]
    )
    candidates.extend(sorted(default_output_dir.glob("checkpoint_epoch_*.pt"), reverse=True))

    weights_dir = resolve_repo_path(root, "weights/pointnet2")
    candidates.extend(
        [
            weights_dir / "checkpoint_best.pt",
            weights_dir / "checkpoint_last.pt",
        ]
    )
    candidates.extend(sorted(weights_dir.rglob("*.pt")))
    candidates.extend(sorted(weights_dir.rglob("*.pth")))

    dedup: List[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        dedup.append(resolved)
    return dedup


def _load_scan2sim_checkpoint(path: Path) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise ValueError("checkpoint payload must be a dict.")
    if "model_state_dict" not in ckpt:
        raise ValueError("missing 'model_state_dict' key.")
    return ckpt


def resolve_checkpoint_path(root: Path, requested: str | None, path_cfg: Dict[str, Any]) -> tuple[Path, Dict[str, Any]]:
    if requested:
        path = resolve_repo_path(root, requested)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        try:
            ckpt = _load_scan2sim_checkpoint(path)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(
                f"Checkpoint '{path}' is not a scan2sim feature checkpoint "
                "(expected dict with 'model_state_dict')."
            ) from exc
        return path, ckpt

    checked: List[str] = []
    for candidate in _candidate_checkpoint_paths(root, path_cfg):
        if not candidate.exists() or not candidate.is_file():
            continue
        try:
            ckpt = _load_scan2sim_checkpoint(candidate)
            print(f"[infer-feature] auto checkpoint: {candidate}")
            return candidate, ckpt
        except Exception as exc:  # noqa: BLE001
            checked.append(f"{candidate} ({exc})")
            continue

    detail = "\n".join(f"  - {line}" for line in checked[:12])
    raise FileNotFoundError(
        "No compatible scan2sim feature checkpoint found.\n"
        "Pass --checkpoint explicitly, or place checkpoint_best.pt under "
        "'outputs/feature_models/pointnet2_15head'.\n"
        f"Checked candidates:\n{detail if detail else '  - (no candidate files found)'}"
    )


def _has_glob_chars(text: str) -> bool:
    return any(char in text for char in ("*", "?", "["))


def collect_input_paths(root: Path, items: Sequence[str], input_pattern: str) -> List[Path]:
    out: List[Path] = []
    seen: set[Path] = set()
    for raw in items:
        raw_path = Path(raw)
        abs_raw = raw_path if raw_path.is_absolute() else (root / raw_path)
        abs_raw = abs_raw.resolve()

        candidates: List[Path] = []
        if _has_glob_chars(str(raw)):
            candidates.extend(Path(match).resolve() for match in glob.glob(str(root / raw)))
        elif abs_raw.is_file():
            candidates.append(abs_raw)
        elif abs_raw.is_dir():
            candidates.extend(path.resolve() for path in abs_raw.glob(input_pattern))
        else:
            raise FileNotFoundError(f"Input path not found: {raw}")

        for path in candidates:
            if not path.is_file():
                continue
            if path.suffix.lower() not in (".pcd", ".ply"):
                continue
            if path not in seen:
                seen.add(path)
                out.append(path)
    out.sort(key=natural_key_path)
    if not out:
        raise ValueError("No .pcd/.ply files found for inference.")
    return out


def _sample_points_deterministic(points: np.ndarray, num_points: int) -> np.ndarray:
    num_points = int(num_points)
    if points.shape[0] == num_points:
        return points
    if points.shape[0] > num_points:
        idx = np.linspace(0, points.shape[0] - 1, num=num_points)
        idx = np.round(idx).astype(np.int64)
        return points[idx]
    reps = int(np.ceil(num_points / points.shape[0]))
    tiled = np.tile(points, (reps, 1))
    return tiled[:num_points]


def _normalize_points(points: np.ndarray) -> np.ndarray:
    centered = points - points.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(centered, axis=1)
    max_norm = float(np.max(norms))
    if max_norm > 1e-8:
        centered = centered / max_norm
    return centered.astype(np.float32)


def _predict_one(
    *,
    model: torch.nn.Module,
    heads: Sequence[HeadSpec],
    point_path: Path,
    num_points: int,
    normalize: bool,
    device: torch.device,
    topk: int,
) -> Dict[str, Any]:
    cloud = read_point_cloud_any(point_path)
    points = np.asarray(cloud.points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        raise ValueError(f"Invalid point cloud: {point_path}")

    sampled = _sample_points_deterministic(points, num_points)
    if normalize:
        sampled = _normalize_points(sampled)
    tensor = torch.from_numpy(sampled).unsqueeze(0).to(device)

    with torch.no_grad():
        logits_by_head = model(tensor)

    result_heads: Dict[str, Any] = {}
    for head in heads:
        logits = logits_by_head[head.name][0]
        probs = torch.softmax(logits, dim=-1)
        pred_idx = int(torch.argmax(probs).item())
        k = max(1, min(int(topk), probs.shape[0]))
        top_prob, top_idx = torch.topk(probs, k=k, dim=-1)
        result_heads[head.name] = {
            "pred_index": pred_idx,
            "pred_label": head.class_values[pred_idx],
            "class_values": list(head.class_values),
            "probs": [float(value) for value in probs.detach().cpu().tolist()],
            "logits": [float(value) for value in logits.detach().cpu().tolist()],
            "topk": [
                {
                    "index": int(index.item()),
                    "label": head.class_values[int(index.item())],
                    "prob": float(prob.item()),
                }
                for prob, index in zip(top_prob, top_idx)
            ],
        }

    return {
        "point_path": str(point_path),
        "heads": result_heads,
    }


def _numeric_suffix(text: str) -> int | None:
    match = re.search(r"(\d+)$", text)
    if match is None:
        return None
    return int(match.group(1))


def _normalize_label_value(value: Any) -> Any:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if float(value).is_integer() else value
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "":
            return value
        try:
            parsed = float(stripped)
            return int(parsed) if parsed.is_integer() else parsed
        except ValueError:
            return value
    return value


def _parse_step_vector(raw_step: Any, *, expected_len: int, source: str) -> List[Any]:
    values: List[Any]
    if isinstance(raw_step, str):
        text = raw_step.strip()
        if text.startswith("{") and text.endswith("}"):
            text = text[1:-1]
        values = [token.strip() for token in text.split(",") if token.strip()] if text else []
    elif isinstance(raw_step, (list, tuple)):
        values = list(raw_step)
    else:
        raise TypeError(f"Unsupported step type in {source}: {type(raw_step).__name__}")
    if len(values) != expected_len:
        raise ValueError(f"`step` length mismatch in {source}: expected {expected_len}, got {len(values)}")
    return [_normalize_label_value(v) for v in values]


def _load_gt_from_jsonl(gt_jsonl_path: Path, head_names: Sequence[str]) -> Dict[int, Dict[str, Any]]:
    gt_map: Dict[int, Dict[str, Any]] = {}
    with gt_jsonl_path.open("r", encoding="utf-8") as handle:
        for line_num, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception as exc:  # noqa: BLE001
                raise ValueError(f"Invalid JSONL row at {gt_jsonl_path}:{line_num}") from exc
            if not isinstance(row, dict):
                raise TypeError(f"JSONL row must be object: {gt_jsonl_path}:{line_num}")
            frame_raw = row.get("Frame")
            if frame_raw is None:
                raise KeyError(f"Missing `Frame` in {gt_jsonl_path}:{line_num}")
            frame = int(frame_raw)
            step_vals = _parse_step_vector(
                row.get("step"),
                expected_len=len(head_names),
                source=f"{gt_jsonl_path}:{line_num}",
            )
            gt_map[frame] = {head_name: step_vals[idx] for idx, head_name in enumerate(head_names)}
    if not gt_map:
        raise ValueError(f"No valid GT rows in {gt_jsonl_path}")
    return gt_map


def _extract_pred_labels_from_file(pred_json_path: Path, head_names: Sequence[str]) -> Tuple[int, Dict[str, Any]]:
    payload = read_json(pred_json_path)
    if not isinstance(payload, dict):
        raise TypeError(f"Prediction JSON root must be object: {pred_json_path}")
    result = payload.get("result")
    if not isinstance(result, dict):
        raise KeyError(f"Prediction JSON missing `result`: {pred_json_path}")
    point_path_raw = result.get("point_path")
    if not isinstance(point_path_raw, str) or not point_path_raw.strip():
        raise KeyError(f"Prediction JSON missing `result.point_path`: {pred_json_path}")
    numeric_key = _numeric_suffix(Path(point_path_raw).stem)
    if numeric_key is None:
        raise ValueError(f"Could not parse numeric suffix from point_path='{point_path_raw}' in {pred_json_path}")
    heads_obj = result.get("heads")
    if not isinstance(heads_obj, dict):
        raise KeyError(f"Prediction JSON missing `result.heads`: {pred_json_path}")
    labels: Dict[str, Any] = {}
    for head_name in head_names:
        head_obj = heads_obj.get(head_name)
        if not isinstance(head_obj, dict):
            raise KeyError(f"Missing head '{head_name}' in {pred_json_path}")
        labels[head_name] = _normalize_label_value(head_obj.get("pred_label"))
    return numeric_key, labels


def _infer_frame_offset(pred_numeric_keys: Sequence[int], gt_frames: Sequence[int]) -> int:
    pred_set = set(pred_numeric_keys)
    gt_set = set(gt_frames)
    best_offset = 0
    best_overlap = -1
    for offset in (0, 1):
        overlap = sum(1 for frame in gt_set if (frame + offset) in pred_set)
        if overlap > best_overlap:
            best_overlap = overlap
            best_offset = offset
    return best_offset


def _evaluate_prediction_files(
    *,
    pred_json_paths: Sequence[Path],
    gt_jsonl_path: Path,
    head_names: Sequence[str],
    frame_offset: str,
) -> None:
    gt_map = _load_gt_from_jsonl(gt_jsonl_path, head_names)
    pred_rows: List[Tuple[int, Dict[str, Any]]] = []
    for pred_path in pred_json_paths:
        pred_rows.append(_extract_pred_labels_from_file(pred_path, head_names))

    pred_keys = [key for key, _ in pred_rows]
    offset = _infer_frame_offset(pred_keys, list(gt_map.keys())) if frame_offset == "auto" else int(frame_offset)

    per_head_correct = {head_name: 0 for head_name in head_names}
    per_head_total = {head_name: 0 for head_name in head_names}
    sample_exact = 0
    sample_compared = 0
    missing_gt = 0

    for pred_key, pred_labels in pred_rows:
        frame = pred_key - offset
        gt_labels = gt_map.get(frame)
        if gt_labels is None:
            missing_gt += 1
            continue
        sample_compared += 1
        all_match = True
        for head_name in head_names:
            gt_value = _normalize_label_value(gt_labels[head_name])
            pred_value = _normalize_label_value(pred_labels[head_name])
            per_head_total[head_name] += 1
            if pred_value == gt_value:
                per_head_correct[head_name] += 1
            else:
                all_match = False
        if all_match:
            sample_exact += 1

    total_correct = sum(per_head_correct.values())
    total_labels = sum(per_head_total.values())
    label_acc = (100.0 * total_correct / total_labels) if total_labels else 0.0
    sample_acc = (100.0 * sample_exact / sample_compared) if sample_compared else 0.0

    print(f"[infer-feature][eval] gt={gt_jsonl_path}")
    print(
        f"[infer-feature][eval] matched_samples={sample_compared}/{len(pred_rows)} "
        f"(missing_gt={missing_gt}) frame_offset={offset}"
    )
    print(
        f"[infer-feature][eval] label_accuracy={label_acc:.2f}% "
        f"({total_correct}/{total_labels if total_labels else 0})"
    )
    print(
        f"[infer-feature][eval] sample_exact_accuracy={sample_acc:.2f}% "
        f"({sample_exact}/{sample_compared if sample_compared else 0})"
    )
    for head_name in head_names:
        correct = per_head_correct[head_name]
        total = per_head_total[head_name]
        acc = (100.0 * correct / total) if total else 0.0
        print(f"[infer-feature][eval] head={head_name} acc={acc:.2f}% ({correct}/{total})")


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PointNet++ multi-head feature-only inference.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=(
            "Trained checkpoint path (.pt). "
            "If omitted, auto-searches outputs/feature_models and weights/pointnet2."
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/feature/pointnet2_multitask.yaml",
        help="Model config YAML. Used when checkpoint does not store config.",
    )
    parser.add_argument(
        "--head-rules",
        type=str,
        default=None,
        help="Head rules JSON override. If omitted, uses checkpoint head_specs or config paths.head_rules.",
    )
    parser.add_argument("--input", type=str, nargs="+", required=True, help="Input .pcd/.ply file(s), directory, or glob(s).")
    parser.add_argument("--input-pattern", type=str, default="*.ply", help="Pattern used when --input item is a directory.")
    parser.add_argument("--output", type=str, default=None, help="Single output JSON path. Use only when one input sample.")
    parser.add_argument("--output-dir", type=str, default="outputs/feature_pred", help="Output directory for batch mode.")
    parser.add_argument("--output-prefix", type=str, default="feature_pred_", help="Output filename prefix for batch mode.")
    parser.add_argument("--num-points", type=int, default=None, help="Override data.num_points.")
    parser.add_argument("--no-normalize", action="store_true", help="Disable normalization before inference.")
    parser.add_argument("--topk", type=int, default=3, help="Top-k predictions saved per head.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument(
        "--gt-jsonl",
        type=str,
        default=None,
        help="Optional GT jsonl for post-inference accuracy report (expects Frame + 15-step labels).",
    )
    parser.add_argument(
        "--gt-frame-offset",
        type=str,
        default="auto",
        choices=["auto", "0", "1"],
        help="GT Frame to file index mapping: file_index = Frame + offset.",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)
    root = repo_root()
    config_path = resolve_repo_path(root, args.config)
    cfg = load_yaml_config(config_path)

    path_cfg = cfg.get("paths", {}) or {}
    checkpoint_path, ckpt = resolve_checkpoint_path(root, args.checkpoint, path_cfg)

    ckpt_cfg = ckpt.get("config")
    if isinstance(ckpt_cfg, dict):
        cfg = ckpt_cfg
        path_cfg = cfg.get("paths", {}) or {}

    data_cfg = cfg.get("data", {}) or {}
    model_cfg = cfg.get("model", {}) or {}

    if "head_specs" in ckpt and isinstance(ckpt["head_specs"], list):
        heads = load_head_specs_from_object({"heads": ckpt["head_specs"], "defaults": {"weight": 1.0}})
    else:
        head_rules_path = resolve_repo_path(
            root,
            args.head_rules or path_cfg.get("head_rules", "configs/feature/label_rules.json"),
        )
        heads = load_head_specs(head_rules_path)

    model = build_model_from_config(model_cfg, heads)
    model.load_state_dict(ckpt["model_state_dict"])
    device = choose_device(args.device)
    model = model.to(device)
    model.eval()

    input_paths = collect_input_paths(root, args.input, args.input_pattern)
    num_points = int(args.num_points if args.num_points is not None else data_cfg.get("num_points", 1024))
    normalize = not bool(args.no_normalize)
    topk = max(1, int(args.topk))

    print(f"[infer-feature] checkpoint={checkpoint_path}")
    print(f"[infer-feature] device={device} samples={len(input_paths)}")

    if args.output:
        if len(input_paths) != 1:
            raise ValueError("--output can be used only with a single input sample.")
        output_path = resolve_repo_path(root, args.output)
        result = _predict_one(
            model=model,
            heads=heads,
            point_path=input_paths[0],
            num_points=num_points,
            normalize=normalize,
            device=device,
            topk=topk,
        )
        payload = {
            "meta": {
                "model": "pointnet2_multitask",
                "checkpoint": str(checkpoint_path),
                "num_points": num_points,
                "normalize": normalize,
            },
            "result": result,
        }
        write_json(output_path, payload, ascii_flag=False, compact=False)
        print(f"[infer-feature] wrote {output_path}")
        if args.gt_jsonl:
            gt_jsonl_path = resolve_repo_path(root, args.gt_jsonl)
            _evaluate_prediction_files(
                pred_json_paths=[output_path],
                gt_jsonl_path=gt_jsonl_path,
                head_names=[head.name for head in heads],
                frame_offset=args.gt_frame_offset,
            )
        return

    output_dir = resolve_repo_path(root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written_paths: List[Path] = []
    for point_path in input_paths:
        result = _predict_one(
            model=model,
            heads=heads,
            point_path=point_path,
            num_points=num_points,
            normalize=normalize,
            device=device,
            topk=topk,
        )
        output_name = f"{args.output_prefix}{point_path.stem}.json"
        payload = {
            "meta": {
                "model": "pointnet2_multitask",
                "checkpoint": str(checkpoint_path),
                "num_points": num_points,
                "normalize": normalize,
            },
            "result": result,
        }
        output_path = output_dir / output_name
        write_json(output_path, payload, ascii_flag=False, compact=False)
        print(f"[infer-feature] wrote {output_path}")
        written_paths.append(output_path)

    if args.gt_jsonl:
        gt_jsonl_path = resolve_repo_path(root, args.gt_jsonl)
        _evaluate_prediction_files(
            pred_json_paths=written_paths,
            gt_jsonl_path=gt_jsonl_path,
            head_names=[head.name for head in heads],
            frame_offset=args.gt_frame_offset,
        )


if __name__ == "__main__":
    main()
