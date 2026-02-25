# scripts/labeling/export_step_jsonl.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from scripts.common.json_utils import read_json
from scripts.common.path_utils import natural_key_path
from scripts.common.io_paths import DATA_INTEROP_LABEL_DIR, resolve_repo_path


"""
Directory mode (step-label-only JSONL):
python -m scripts.labeling.export_step_jsonl \
  --input outputs/VAR/label \
  --pattern "livehps_label_*.json" \
  --mode step \
  --output outputs/VAR/label/livehps_label_VAR_step.jsonl

Directory mode (full JSONL):
python -m scripts.labeling.export_step_jsonl \
  --input outputs/VAR/label \
  --pattern "livehps_label_*.json" \
  --mode full \
  --output outputs/VAR/label/livehps_label_VAR_full.jsonl

File mode:
python -m scripts.labeling.export_step_jsonl \
  --input outputs/VAR/label/livehps_label_001.json \
  --mode step \
  --output outputs/VAR/label/livehps_label_001_step.jsonl
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export idx + step-label-only records from livehps_label_*.json to JSONL."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=DATA_INTEROP_LABEL_DIR,
        help="Input label json file or directory.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="livehps_label_*.json",
        help="Glob pattern in directory mode.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Output JSONL path. If empty, uses '<input_dir>/livehps_label_steps.jsonl'.",
    )
    parser.add_argument(
        "--ascii",
        action="store_true",
        help="Write JSONL with ensure_ascii=True.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["step", "full"],
        default="step",
        help="step: export {idx, step labels only}. full: export full original JSON per line.",
    )
    return parser.parse_args()


def collect_input_paths(input_path: Path, pattern: str) -> List[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() != ".json":
            raise ValueError(f"Input file must be .json: {input_path}")
        return [input_path]

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    paths = sorted(
        [path for path in input_path.glob(pattern) if path.is_file() and path.suffix.lower() == ".json"],
        key=natural_key_path,
    )
    if not paths:
        raise ValueError(f"No JSON files in {input_path} with pattern={pattern}")
    return paths


def normalize_step_labels(step: Any) -> List[Any]:
    if isinstance(step, dict):
        return [value for value in step.values()]
    if isinstance(step, list):
        return step
    raise TypeError(f"'step' must be dict or list, got {type(step)}")


def to_step_record(payload: Dict[str, Any], src_path: Path) -> Dict[str, Any]:
    if "idx" not in payload:
        raise KeyError(f"Missing 'idx' in {src_path}")
    if "step" not in payload:
        raise KeyError(f"Missing 'step' in {src_path}")

    idx = int(payload["idx"])
    step_labels = normalize_step_labels(payload["step"])
    return {
        "idx": idx,
        "step": step_labels,
    }


def write_jsonl(path: Path, rows: List[Any], ascii_mode: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=ascii_mode, separators=(",", ":")))
            handle.write("\n")


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]

    input_path = resolve_repo_path(repo_root, args.input)
    json_paths = collect_input_paths(input_path, args.pattern)

    if args.output:
        output_path = resolve_repo_path(repo_root, args.output)
    else:
        output_dir = input_path if input_path.is_dir() else input_path.parent
        output_path = output_dir / "livehps_label_steps.jsonl"

    rows: List[Any] = []
    for path in json_paths:
        payload = read_json(path)
        if args.mode == "full":
            rows.append(payload)
        else:
            if not isinstance(payload, dict):
                raise TypeError(f"Top-level JSON must be object: {path}")
            rows.append(to_step_record(payload, path))

    write_jsonl(output_path, rows, ascii_mode=args.ascii)
    print(f"saved: {output_path}")
    print(f"files={len(json_paths)}, records={len(rows)}, mode={args.mode}")


if __name__ == "__main__":
    main()
