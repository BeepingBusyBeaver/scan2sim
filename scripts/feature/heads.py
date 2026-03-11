# scripts/feature/heads.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from scripts.common.json_utils import read_json


@dataclass(frozen=True)
class BoundaryRelaxConfig:
    enabled: bool
    mode: str
    epsilon: float


@dataclass(frozen=True)
class HeadSpec:
    name: str
    class_values: Tuple[Any, ...]
    weight: float
    boundary_relax: BoundaryRelaxConfig

    def class_to_index(self) -> Dict[Any, int]:
        return {value: index for index, value in enumerate(self.class_values)}


def _parse_boundary_relax(raw: Mapping[str, Any], default: BoundaryRelaxConfig) -> BoundaryRelaxConfig:
    enabled = bool(raw.get("enabled", default.enabled))
    mode = str(raw.get("mode", default.mode)).strip().lower()
    epsilon = float(raw.get("epsilon", default.epsilon))
    if mode != "adjacent":
        raise ValueError(f"Unsupported boundary_relax.mode='{mode}'. Use 'adjacent'.")
    if not (0.0 <= epsilon < 1.0):
        raise ValueError(f"boundary_relax.epsilon must be in [0,1), got {epsilon}")
    return BoundaryRelaxConfig(enabled=enabled, mode=mode, epsilon=epsilon)


def _unique(values: Sequence[Any]) -> bool:
    return len(set(values)) == len(values)


def _parse_head_specs_obj(obj: Mapping[str, Any]) -> List[HeadSpec]:
    if not isinstance(obj, dict):
        raise TypeError("Head rules JSON must be an object.")

    defaults_obj = obj.get("defaults", {})
    if defaults_obj is None:
        defaults_obj = {}
    if not isinstance(defaults_obj, dict):
        raise TypeError("`defaults` must be an object.")

    default_weight = float(defaults_obj.get("weight", 1.0))
    if default_weight <= 0.0:
        raise ValueError("defaults.weight must be > 0.")

    default_relax = _parse_boundary_relax(
        defaults_obj.get("boundary_relax", {}) or {},
        BoundaryRelaxConfig(enabled=False, mode="adjacent", epsilon=0.0),
    )

    heads_raw = obj.get("heads")
    if not isinstance(heads_raw, list) or not heads_raw:
        raise ValueError("Head rules JSON must contain non-empty `heads` list.")

    names_seen: set[str] = set()
    out: List[HeadSpec] = []
    for index, item in enumerate(heads_raw):
        if not isinstance(item, dict):
            raise TypeError(f"heads[{index}] must be an object.")

        name = str(item.get("name", "")).strip()
        if not name:
            raise ValueError(f"heads[{index}] missing non-empty `name`.")
        if name in names_seen:
            raise ValueError(f"Duplicate head name: '{name}'.")
        names_seen.add(name)

        class_values = item.get("class_values")
        if class_values is None:
            class_values = item.get("classes")
        if class_values is None:
            class_values = item.get("labels")
        if not isinstance(class_values, list):
            raise TypeError(f"heads[{index}].class_values must be a list.")
        if len(class_values) not in (2, 3):
            raise ValueError(f"heads[{index}] '{name}' must have 2 or 3 classes, got {len(class_values)}.")
        if not _unique(class_values):
            raise ValueError(f"heads[{index}] '{name}' class_values must be unique.")

        weight = float(item.get("weight", default_weight))
        if weight <= 0.0:
            raise ValueError(f"heads[{index}] '{name}' weight must be > 0.")

        relax = _parse_boundary_relax(item.get("boundary_relax", {}) or {}, default_relax)

        out.append(
            HeadSpec(
                name=name,
                class_values=tuple(class_values),
                weight=weight,
                boundary_relax=relax,
            )
        )

    return out


def load_head_specs(path: Path) -> List[HeadSpec]:
    obj = read_json(path)
    return _parse_head_specs_obj(obj)


def load_head_specs_from_object(obj: Mapping[str, Any]) -> List[HeadSpec]:
    return _parse_head_specs_obj(obj)


def head_specs_to_jsonable(heads: Iterable[HeadSpec]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for head in heads:
        rows.append(
            {
                "name": head.name,
                "class_values": list(head.class_values),
                "weight": float(head.weight),
                "boundary_relax": {
                    "enabled": bool(head.boundary_relax.enabled),
                    "mode": head.boundary_relax.mode,
                    "epsilon": float(head.boundary_relax.epsilon),
                },
            }
        )
    return rows
