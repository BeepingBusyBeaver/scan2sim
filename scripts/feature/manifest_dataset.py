# scripts/feature/manifest_dataset.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from scripts.common.pc_io import read_point_cloud_any
from scripts.feature.heads import HeadSpec


@dataclass(frozen=True)
class ManifestSample:
    sample_id: str
    point_path: Path
    labels: Dict[str, Any]
    split: str


def _to_label_index(value: Any, mapping: Mapping[Any, int], head_name: str) -> int:
    if value in mapping:
        return int(mapping[value])
    if isinstance(value, float) and float(value).is_integer():
        as_int = int(value)
        if as_int in mapping:
            return int(mapping[as_int])
    if isinstance(value, int):
        as_float = float(value)
        if as_float in mapping:
            return int(mapping[as_float])
    as_str = str(value)
    if as_str in mapping:
        return int(mapping[as_str])
    expected = list(mapping.keys())
    raise ValueError(f"Label '{value}' is not valid for head '{head_name}'. expected one of {expected}")


def _manifest_point_path(row: Mapping[str, Any]) -> str:
    for key in ("point_path", "point", "pc_path", "input_path"):
        if key in row:
            return str(row[key])
    raise KeyError("Manifest row missing point cloud path key. expected one of: point_path, point, pc_path, input_path")


def _manifest_labels(row: Mapping[str, Any]) -> Dict[str, Any]:
    labels = row.get("labels")
    if labels is None:
        labels = row.get("step")
    if not isinstance(labels, dict):
        raise TypeError("Manifest row must provide object labels in `labels` or `step`.")
    return dict(labels)


def _manifest_sample_id(row: Mapping[str, Any], point_path: Path) -> str:
    for key in ("sample_id", "id", "idx"):
        if key in row:
            return str(row[key])
    return point_path.stem


def load_manifest_jsonl(manifest_path: Path, repo_root: Path) -> List[ManifestSample]:
    rows: List[ManifestSample] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_index, raw in enumerate(handle, start=1):
            text = raw.strip()
            if not text:
                continue
            row = json.loads(text)
            if not isinstance(row, dict):
                raise TypeError(f"manifest line {line_index}: expected object JSON.")

            point_raw = _manifest_point_path(row)
            point_path = Path(point_raw)
            if not point_path.is_absolute():
                point_path = (repo_root / point_path).resolve()
            labels = _manifest_labels(row)
            sample_id = _manifest_sample_id(row, point_path)
            split = str(row.get("split", "train")).strip().lower() or "train"
            rows.append(
                ManifestSample(
                    sample_id=sample_id,
                    point_path=point_path,
                    labels=labels,
                    split=split,
                )
            )

    if not rows:
        raise ValueError(f"Manifest is empty: {manifest_path}")
    return rows


def split_manifest_samples(
    samples: Sequence[ManifestSample],
    *,
    mode: str,
    train_value: str,
    val_value: str,
    val_ratio: float,
    seed: int,
) -> Tuple[List[ManifestSample], List[ManifestSample]]:
    mode_normalized = str(mode).strip().lower()
    if mode_normalized == "field":
        train_samples = [sample for sample in samples if sample.split == train_value]
        val_samples = [sample for sample in samples if sample.split == val_value]
        if not train_samples:
            raise ValueError(f"No training rows for split='{train_value}'.")
        if not val_samples:
            raise ValueError(
                f"No validation rows for split='{val_value}'. "
                "Set split.mode=random in config if manifest has no split field."
            )
        return train_samples, val_samples

    if mode_normalized == "random":
        if not (0.0 < val_ratio < 1.0):
            raise ValueError(f"split.val_ratio must be in (0,1), got {val_ratio}")
        indices = np.arange(len(samples))
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

        val_count = max(1, int(round(len(samples) * val_ratio)))
        val_indices = set(indices[:val_count].tolist())
        train_samples = [sample for index, sample in enumerate(samples) if index not in val_indices]
        val_samples = [sample for index, sample in enumerate(samples) if index in val_indices]
        if not train_samples or not val_samples:
            raise ValueError("Random split failed. adjust val_ratio or dataset size.")
        return train_samples, val_samples

    raise ValueError(f"Unsupported split.mode='{mode}'. Use 'field' or 'random'.")


def _sample_points(points: np.ndarray, num_points: int) -> np.ndarray:
    if points.shape[0] == num_points:
        return points
    replace = points.shape[0] < num_points
    idx = np.random.choice(points.shape[0], size=num_points, replace=replace)
    return points[idx]


def _normalize_points(points: np.ndarray) -> np.ndarray:
    centered = points - points.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(centered, axis=1)
    max_norm = float(np.max(norms))
    if max_norm > 1e-8:
        centered = centered / max_norm
    return centered.astype(np.float32)


def _augment_points(points: np.ndarray, jitter_std: float, jitter_clip: float) -> np.ndarray:
    jitter = np.random.normal(loc=0.0, scale=jitter_std, size=points.shape).astype(np.float32)
    if jitter_clip > 0.0:
        np.clip(jitter, -jitter_clip, jitter_clip, out=jitter)
    return (points + jitter).astype(np.float32)


class PointCloudHeadDataset(Dataset):
    def __init__(
        self,
        *,
        samples: Sequence[ManifestSample],
        head_specs: Sequence[HeadSpec],
        num_points: int,
        normalize: bool,
        augment: bool,
        jitter_std: float,
        jitter_clip: float,
        cache_in_memory: bool,
    ) -> None:
        super().__init__()
        if num_points <= 0:
            raise ValueError("num_points must be > 0")
        if not head_specs:
            raise ValueError("head_specs cannot be empty")

        self.samples = list(samples)
        self.head_specs = list(head_specs)
        self.num_points = int(num_points)
        self.normalize = bool(normalize)
        self.augment = bool(augment)
        self.jitter_std = float(jitter_std)
        self.jitter_clip = float(jitter_clip)
        self.cache_in_memory = bool(cache_in_memory)
        self._point_cache: Dict[Path, np.ndarray] = {}

        head_maps = [head.class_to_index() for head in self.head_specs]
        self._label_rows: List[np.ndarray] = []
        for sample in self.samples:
            label_indices = np.zeros((len(self.head_specs),), dtype=np.int64)
            for head_index, (head, class_map) in enumerate(zip(self.head_specs, head_maps)):
                if head.name not in sample.labels:
                    raise KeyError(f"Sample '{sample.sample_id}' missing label for head '{head.name}'.")
                raw_label = sample.labels[head.name]
                label_indices[head_index] = _to_label_index(raw_label, class_map, head.name)
            self._label_rows.append(label_indices)

    def __len__(self) -> int:
        return len(self.samples)

    def _load_points(self, point_path: Path) -> np.ndarray:
        if self.cache_in_memory and point_path in self._point_cache:
            return self._point_cache[point_path].copy()

        if not point_path.exists():
            raise FileNotFoundError(f"Point cloud not found: {point_path}")
        cloud = read_point_cloud_any(point_path)
        points = np.asarray(cloud.points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
            raise ValueError(f"Invalid point cloud (need Nx3 and non-empty): {point_path}")

        if self.cache_in_memory:
            self._point_cache[point_path] = points.copy()
        return points

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        points = self._load_points(sample.point_path)
        points = _sample_points(points, self.num_points)
        if self.normalize:
            points = _normalize_points(points)
        if self.augment:
            points = _augment_points(points, self.jitter_std, self.jitter_clip)
        labels = self._label_rows[index]
        return torch.from_numpy(points.astype(np.float32)), torch.from_numpy(labels.copy())


def collect_point_paths(paths: Iterable[Path]) -> List[Path]:
    out = [path.resolve() for path in paths if path.is_file()]
    out.sort(key=lambda path: path.name)
    return out
