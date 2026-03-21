# scripts/cmcf/map_prototypes.py
from __future__ import annotations

import argparse
import fnmatch
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from scripts.common.io_paths import resolve_repo_path
from scripts.common.pc_io import expand_inputs


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _collect_query_paths(root: Path, items: Iterable[str], pattern: str) -> List[Path]:
    raw = expand_inputs(root, items, exts=[".json"])
    if not pattern or pattern in {"*", "*.*"}:
        return raw
    return [path for path in raw if fnmatch.fnmatch(path.name, pattern)]


def _load_prototypes(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("prototypes", []) if isinstance(payload, Mapping) else []
    if not isinstance(rows, list):
        return []
    out: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        fmap = row.get("feature_map", {})
        labels = row.get("labels", {})
        if not isinstance(fmap, Mapping) or not isinstance(labels, Mapping):
            continue
        out.append(
            {
                "prototype_id": int(row.get("prototype_id", len(out))),
                "source_file": str(row.get("source_file", "")),
                "labels": dict(labels),
                "feature_map": {str(k): _safe_float(v) for k, v in fmap.items()},
            }
        )
    return out


def _feature_distance(
    query_features: Mapping[str, Any],
    proto_features: Mapping[str, Any],
    *,
    min_shared: int,
) -> Tuple[float, int]:
    shared = [key for key in query_features.keys() if key in proto_features]
    if len(shared) < min_shared:
        return float("inf"), int(len(shared))
    q = np.array([_safe_float(query_features[key]) for key in shared], dtype=np.float64)
    p = np.array([_safe_float(proto_features[key]) for key in shared], dtype=np.float64)
    dist = float(np.sqrt(np.mean((q - p) ** 2)))
    return dist, int(len(shared))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CMCF prototype mapper (real/virtual query -> nearest virtual prototype labels).")
    parser.add_argument("--prototype-bank", type=str, required=True)
    parser.add_argument("--query", type=str, nargs="+", required=True, help="CMCF frame json files/dirs/globs.")
    parser.add_argument("--query-pattern", type=str, default="cmcf_*.json")
    parser.add_argument("--output", type=str, default="outputs/cmcf/prototype_mapping.jsonl")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--min-shared-features", type=int, default=20)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    root = repo_root()
    prototype_path = resolve_repo_path(root, args.prototype_bank)
    if not prototype_path.exists():
        raise FileNotFoundError(f"Prototype bank not found: {prototype_path}")
    prototypes = _load_prototypes(prototype_path)
    if not prototypes:
        raise ValueError(f"No valid prototypes in: {prototype_path}")

    query_paths = _collect_query_paths(root, args.query, args.query_pattern)
    if not query_paths:
        raise FileNotFoundError("No query CMCF json files found.")

    output_path = resolve_repo_path(root, args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    topk = max(int(args.topk), 1)
    min_shared = max(int(args.min_shared_features), 1)

    mapped_count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for query_path in query_paths:
            payload = json.loads(query_path.read_text(encoding="utf-8"))
            feature_map = payload.get("feature_map", {}) if isinstance(payload, Mapping) else {}
            if not isinstance(feature_map, Mapping):
                continue
            rows: List[Dict[str, Any]] = []
            for proto in prototypes:
                dist, shared = _feature_distance(
                    feature_map,
                    proto.get("feature_map", {}),
                    min_shared=min_shared,
                )
                if not np.isfinite(dist):
                    continue
                rows.append(
                    {
                        "prototype_id": int(proto.get("prototype_id", -1)),
                        "source_file": str(proto.get("source_file", "")),
                        "distance": float(dist),
                        "shared_features": int(shared),
                        "labels": dict(proto.get("labels", {})),
                    }
                )
            if not rows:
                continue
            rows.sort(key=lambda row: float(row["distance"]))
            best = rows[0]
            out_row = {
                "query_file": str(query_path),
                "best_prototype_id": int(best["prototype_id"]),
                "best_distance": float(best["distance"]),
                "predicted_labels": dict(best["labels"]),
                "topk": rows[:topk],
            }
            handle.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            mapped_count += 1

    print(
        f"[cmcf-map] queries={len(query_paths)} mapped={mapped_count} "
        f"prototypes={len(prototypes)} output={output_path}"
    )


if __name__ == "__main__":
    main()
