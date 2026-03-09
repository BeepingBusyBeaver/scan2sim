from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence, Tuple

import yaml


@dataclass(frozen=True)
class BinRule:
    label: Any
    min_value: float
    max_value: float
    closed_max: bool

    def contains(self, value: float) -> bool:
        if self.closed_max:
            return self.min_value <= value <= self.max_value
        return self.min_value <= value < self.max_value

    def distance(self, value: float) -> float:
        if self.contains(value):
            return 0.0
        if value < self.min_value:
            return self.min_value - value
        return value - self.max_value


@dataclass(frozen=True)
class FeatureTerm:
    feature: str
    weight: float
    min_value: float | None
    max_value: float | None
    target: float | None
    tolerance: float
    use_abs: bool
    clip: float | None
    default: float


@dataclass(frozen=True)
class ClassRule:
    label: Any
    bias: float
    terms: Tuple[FeatureTerm, ...]


@dataclass(frozen=True)
class HeadDecodeRule:
    name: str
    mode: str
    fallback: Any
    feature: str | None
    bins: Tuple[BinRule, ...]
    classes: Tuple[ClassRule, ...]


def _default_decoder_config() -> Dict[str, Any]:
    return {
        "default_profile": "global",
        "decoder": {
            "default_label": 0,
            "heads": [],
        },
        "temporal": {
            "enabled": True,
            "hysteresis_margin": 0.03,
            "window_size": 3,
            "majority_min_count": 2,
        },
        "swap_validation": {
            "enabled": True,
            "min_improvement": 0.02,
            "prefer_previous": True,
        },
        "consistency": {
            "enabled": True,
            "weight": 1.0,
            "temporal_stickiness": 0.10,
            "search_topk": 3,
            "max_iter": 4,
            "rules": [],
        },
        "head_label_bias": {},
    }


def _deep_merge_dict(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_decoder_config(path: Path | None, profile: str | None = None) -> Dict[str, Any]:
    cfg = _default_decoder_config()
    if path is None:
        cfg["selected_profile"] = str(profile or cfg.get("default_profile", "squat"))
        return cfg

    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise TypeError(f"Decoder config root must be object: {path}")

    loaded_profiles = loaded.get("profiles")
    base_loaded = {key: value for key, value in loaded.items() if key != "profiles"}
    cfg = _deep_merge_dict(cfg, base_loaded)

    selected_profile = str(profile or cfg.get("default_profile", "squat"))
    if isinstance(loaded_profiles, Mapping) and selected_profile in loaded_profiles:
        profile_obj = loaded_profiles.get(selected_profile)
        if not isinstance(profile_obj, Mapping):
            raise TypeError(f"profiles.{selected_profile} must be object: {path}")
        cfg = _deep_merge_dict(cfg, profile_obj)
    elif profile is not None and isinstance(loaded_profiles, Mapping) and profile not in loaded_profiles:
        available = ", ".join(sorted(str(key) for key in loaded_profiles.keys()))
        raise KeyError(f"Unknown decoder profile '{profile}'. Available: {available}")

    cfg["selected_profile"] = selected_profile
    return cfg


def _parse_feature_term(head_name: str, class_label: Any, term_obj: Mapping[str, Any]) -> FeatureTerm:
    feature = str(term_obj.get("feature", "")).strip()
    if not feature:
        raise ValueError(f"decoder head '{head_name}' class '{class_label}' term requires 'feature'.")
    min_value = term_obj.get("min")
    max_value = term_obj.get("max")
    target = term_obj.get("target")
    if target is None and min_value is None and max_value is None:
        raise ValueError(
            f"decoder head '{head_name}' class '{class_label}' term '{feature}' requires "
            "either target/tol or min/max."
        )
    return FeatureTerm(
        feature=feature,
        weight=float(term_obj.get("weight", 1.0)),
        min_value=None if min_value is None else float(min_value),
        max_value=None if max_value is None else float(max_value),
        target=None if target is None else float(target),
        tolerance=max(float(term_obj.get("tol", term_obj.get("tolerance", 1.0))), 1e-6),
        use_abs=bool(term_obj.get("use_abs", False)),
        clip=None if term_obj.get("clip") is None else float(term_obj.get("clip")),
        default=float(term_obj.get("default", 0.0)),
    )


def _parse_head_rules(config: Mapping[str, Any]) -> List[HeadDecodeRule]:
    decoder_obj = config.get("decoder", {})
    if not isinstance(decoder_obj, Mapping):
        raise TypeError("decoder section must be object.")
    default_label = decoder_obj.get("default_label", 0)
    heads_raw = decoder_obj.get("heads", [])
    if not isinstance(heads_raw, list) or not heads_raw:
        raise ValueError("decoder.heads must be non-empty list.")

    rules: List[HeadDecodeRule] = []
    for item in heads_raw:
        if not isinstance(item, Mapping):
            raise TypeError("decoder.heads item must be object.")
        name = str(item.get("name", "")).strip()
        if not name:
            raise ValueError("decoder head requires name.")
        fallback = item.get("fallback", default_label)

        classes_raw = item.get("classes")
        if isinstance(classes_raw, list):
            classes: List[ClassRule] = []
            for class_obj in classes_raw:
                if not isinstance(class_obj, Mapping):
                    raise TypeError(f"decoder head '{name}' classes item must be object.")
                label = class_obj.get("label")
                terms_raw = class_obj.get("terms", [])
                if not isinstance(terms_raw, list) or not terms_raw:
                    raise ValueError(f"decoder head '{name}' class '{label}' terms must be non-empty list.")
                terms = tuple(
                    _parse_feature_term(name, label, term_obj)
                    for term_obj in terms_raw
                    if isinstance(term_obj, Mapping)
                )
                if not terms:
                    raise ValueError(f"decoder head '{name}' class '{label}' has no valid terms.")
                classes.append(
                    ClassRule(
                        label=label,
                        bias=float(class_obj.get("bias", 0.0)),
                        terms=terms,
                    )
                )
            rules.append(
                HeadDecodeRule(
                    name=name,
                    mode="classes",
                    fallback=fallback,
                    feature=None,
                    bins=tuple(),
                    classes=tuple(classes),
                )
            )
            continue

        feature = str(item.get("feature", "")).strip()
        bins_raw = item.get("bins", [])
        if not feature:
            raise ValueError(f"decoder head '{name}' requires feature (bins mode).")
        if not isinstance(bins_raw, list) or not bins_raw:
            raise ValueError(f"decoder head '{name}' bins must be non-empty list.")
        bins: List[BinRule] = []
        for bin_obj in bins_raw:
            if not isinstance(bin_obj, Mapping):
                raise TypeError(f"decoder head '{name}' bin must be object.")
            bins.append(
                BinRule(
                    label=bin_obj.get("label"),
                    min_value=float(bin_obj.get("min")),
                    max_value=float(bin_obj.get("max")),
                    closed_max=bool(bin_obj.get("closed_max", bin_obj.get("closed_hi", False))),
                )
            )
        rules.append(
            HeadDecodeRule(
                name=name,
                mode="bins",
                fallback=fallback,
                feature=feature,
                bins=tuple(bins),
                classes=tuple(),
            )
        )
    return rules


def _to_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, (float, int)):
        return float(value)
    try:
        return float(value)
    except Exception:
        return float(default)


def _lookup_label_bias(head_bias: Mapping[str, Any] | None, label: Any) -> float:
    if not isinstance(head_bias, Mapping):
        return 0.0
    if label in head_bias:
        return _to_float(head_bias[label], 0.0)
    key = str(label)
    if key in head_bias:
        return _to_float(head_bias[key], 0.0)
    return 0.0


def _term_penalty(term: FeatureTerm, feature_map: Mapping[str, float]) -> Tuple[float, float]:
    value = _to_float(feature_map.get(term.feature, term.default), term.default)
    if term.use_abs:
        value = abs(value)

    penalty = 0.0
    if term.target is not None:
        penalty = abs(value - term.target) / max(term.tolerance, 1e-6)
    else:
        min_value = -1e18 if term.min_value is None else float(term.min_value)
        max_value = 1e18 if term.max_value is None else float(term.max_value)
        if value < min_value:
            dist = min_value - value
        elif value > max_value:
            dist = value - max_value
        else:
            dist = 0.0
        span = max(max_value - min_value, 1e-6) if (term.min_value is not None and term.max_value is not None) else 1.0
        penalty = dist / span

    penalty *= max(term.weight, 0.0)
    if term.clip is not None:
        penalty = min(penalty, max(term.clip, 0.0))
    return float(penalty), float(value)


def _evaluate_bins_head(
    rule: HeadDecodeRule,
    feature_map: Mapping[str, float],
    head_bias: Mapping[str, Any] | None,
) -> Tuple[Any, float, float, Dict[Any, float], Dict[str, Any]]:
    if rule.feature is None:
        raise ValueError(f"Head '{rule.name}' missing feature for bins mode.")
    value = _to_float(feature_map.get(rule.feature, 0.0), 0.0)
    penalties: Dict[Any, float] = {}
    for bin_rule in rule.bins:
        dist = float(bin_rule.distance(value)) + _lookup_label_bias(head_bias, bin_rule.label)
        prev = penalties.get(bin_rule.label)
        penalties[bin_rule.label] = dist if prev is None else min(prev, dist)

    if not penalties:
        return rule.fallback, 1e9, 0.0, {rule.fallback: 1e9}, {"mode": "bins", "value": value}

    ordered = sorted(penalties.items(), key=lambda item: item[1])
    best_label, best_penalty = ordered[0]
    second_penalty = ordered[1][1] if len(ordered) > 1 else (best_penalty + 1.0)
    confidence = float(max(second_penalty - best_penalty, 0.0))
    return (
        best_label,
        float(best_penalty),
        confidence,
        penalties,
        {
            "mode": "bins",
            "feature": rule.feature,
            "value": value,
            "penalties": {str(label): float(score) for label, score in penalties.items()},
        },
    )


def _evaluate_classes_head(
    rule: HeadDecodeRule,
    feature_map: Mapping[str, float],
    head_bias: Mapping[str, Any] | None,
) -> Tuple[Any, float, float, Dict[Any, float], Dict[str, Any]]:
    penalties: Dict[Any, float] = {}
    class_debug: Dict[str, Any] = {}

    for class_rule in rule.classes:
        class_penalty = float(class_rule.bias) + _lookup_label_bias(head_bias, class_rule.label)
        terms_debug = []
        for term in class_rule.terms:
            term_penalty, value = _term_penalty(term, feature_map)
            class_penalty += term_penalty
            terms_debug.append(
                {
                    "feature": term.feature,
                    "value": value,
                    "penalty": float(term_penalty),
                }
            )
        penalties[class_rule.label] = float(class_penalty)
        class_debug[str(class_rule.label)] = {
            "penalty": float(class_penalty),
            "terms": terms_debug,
        }

    if not penalties:
        return rule.fallback, 1e9, 0.0, {rule.fallback: 1e9}, {"mode": "classes"}

    ordered = sorted(penalties.items(), key=lambda item: item[1])
    best_label, best_penalty = ordered[0]
    second_penalty = ordered[1][1] if len(ordered) > 1 else (best_penalty + 1.0)
    confidence = float(max(second_penalty - best_penalty, 0.0))
    return (
        best_label,
        float(best_penalty),
        confidence,
        penalties,
        {
            "mode": "classes",
            "classes": class_debug,
        },
    )


def _evaluate_head_rule(
    rule: HeadDecodeRule,
    feature_map: Mapping[str, float],
    head_bias: Mapping[str, Any] | None,
) -> Tuple[Any, float, float, Dict[Any, float], Dict[str, Any]]:
    if rule.mode == "classes":
        return _evaluate_classes_head(rule, feature_map, head_bias=head_bias)
    return _evaluate_bins_head(rule, feature_map, head_bias=head_bias)


def _majority_label(history: Sequence[Any], min_count: int) -> Any | None:
    if not history:
        return None
    counts: Dict[Any, int] = {}
    for item in history:
        counts[item] = counts.get(item, 0) + 1
    label = max(counts, key=lambda key: counts[key])
    if counts[label] >= min_count:
        return label
    return None


def _rule_penalty(rule: Mapping[str, Any], labels: Mapping[str, Any]) -> float:
    rule_type = str(rule.get("type", "")).strip()
    heads = rule.get("heads", [])
    if not isinstance(heads, list) or len(heads) < 2:
        return 0.0
    head_a = str(heads[0])
    head_b = str(heads[1])
    if head_a not in labels or head_b not in labels:
        return 0.0

    value_a = labels[head_a]
    value_b = labels[head_b]
    penalty = float(rule.get("penalty", 0.0))
    if penalty <= 0.0:
        return 0.0

    if rule_type == "abs_diff_le":
        if not isinstance(value_a, (int, float)) or not isinstance(value_b, (int, float)):
            return 0.0
        max_diff = float(rule.get("max_diff", 0.0))
        diff = abs(float(value_a) - float(value_b))
        if diff <= max_diff:
            return 0.0
        scale = float(rule.get("scale", 1.0))
        return penalty + max(diff - max_diff, 0.0) * max(scale, 0.0)

    if rule_type == "equal":
        return 0.0 if value_a == value_b else penalty

    if rule_type == "pair_forbidden":
        pairs = rule.get("pairs", [])
        if not isinstance(pairs, list):
            return 0.0
        for pair in pairs:
            if isinstance(pair, (list, tuple)) and len(pair) == 2 and value_a == pair[0] and value_b == pair[1]:
                return penalty
        return 0.0

    if rule_type == "pair_preferred":
        pairs = rule.get("pairs", [])
        if not isinstance(pairs, list) or not pairs:
            return 0.0
        for pair in pairs:
            if isinstance(pair, (list, tuple)) and len(pair) == 2 and value_a == pair[0] and value_b == pair[1]:
                return 0.0
        return penalty

    if rule_type == "imply":
        if_value = rule.get("if")
        then_in = rule.get("then_in", [])
        if not isinstance(then_in, list) or not then_in:
            return 0.0
        if value_a == if_value and value_b not in then_in:
            return penalty
        return 0.0

    return 0.0


def _consistency_penalty(consistency_cfg: Mapping[str, Any], labels: Mapping[str, Any]) -> float:
    rules = consistency_cfg.get("rules", [])
    if not isinstance(rules, list):
        return 0.0
    total = 0.0
    for rule in rules:
        if isinstance(rule, Mapping):
            total += _rule_penalty(rule, labels)
    return float(total)


def _objective_score(
    labels: Mapping[str, Any],
    per_head_penalties: Mapping[str, Mapping[Any, float]],
    temporal_labels: Mapping[str, Any],
    consistency_cfg: Mapping[str, Any],
) -> float:
    missing_penalty = 1000.0
    base = 0.0
    for head_name, label in labels.items():
        head_penalties = per_head_penalties.get(head_name, {})
        base += float(head_penalties.get(label, missing_penalty))

    stickiness = float(consistency_cfg.get("temporal_stickiness", 0.0))
    if stickiness > 0.0:
        for head_name, label in labels.items():
            prev_label = temporal_labels.get(head_name, label)
            if label != prev_label:
                base += stickiness

    weight = float(consistency_cfg.get("weight", 1.0))
    if weight > 0.0:
        base += weight * _consistency_penalty(consistency_cfg, labels)
    return float(base)


def _optimize_with_consistency(
    *,
    labels_init: Mapping[str, Any],
    per_head_penalties: Mapping[str, Mapping[Any, float]],
    temporal_labels: Mapping[str, Any],
    head_order: Sequence[str],
    consistency_cfg: Mapping[str, Any],
) -> Tuple[Dict[str, Any], float, Dict[str, Any]]:
    if not bool(consistency_cfg.get("enabled", False)):
        labels = dict(labels_init)
        score = _objective_score(labels, per_head_penalties, temporal_labels, consistency_cfg)
        return labels, score, {"enabled": False, "changed_heads": [], "iterations": 0}

    search_topk = max(int(consistency_cfg.get("search_topk", 3)), 1)
    max_iter = max(int(consistency_cfg.get("max_iter", 4)), 1)

    candidates: Dict[str, List[Any]] = {}
    for head_name in head_order:
        penalties = per_head_penalties.get(head_name, {})
        ordered = [label for label, _ in sorted(penalties.items(), key=lambda item: item[1])]
        if not ordered:
            ordered = [labels_init.get(head_name, 0)]
        ordered = ordered[:search_topk]
        init_label = labels_init.get(head_name)
        if init_label not in ordered:
            ordered.append(init_label)
        candidates[head_name] = ordered

    labels = dict(labels_init)
    best_score = _objective_score(labels, per_head_penalties, temporal_labels, consistency_cfg)
    changed_heads: set[str] = set()
    iteration_count = 0
    for _ in range(max_iter):
        iteration_count += 1
        improved = False
        for head_name in head_order:
            current_label = labels.get(head_name)
            local_best_label = current_label
            local_best_score = best_score
            for candidate in candidates.get(head_name, []):
                if candidate == current_label:
                    continue
                trial = dict(labels)
                trial[head_name] = candidate
                trial_score = _objective_score(trial, per_head_penalties, temporal_labels, consistency_cfg)
                if trial_score + 1e-9 < local_best_score:
                    local_best_score = trial_score
                    local_best_label = candidate
            if local_best_label != current_label:
                labels[head_name] = local_best_label
                best_score = local_best_score
                changed_heads.add(head_name)
                improved = True
        if not improved:
            break

    return labels, float(best_score), {
        "enabled": True,
        "changed_heads": sorted(changed_heads),
        "iterations": iteration_count,
    }


def _swap_left_right_key(key: str) -> str:
    swapped = str(key)
    swapped = swapped.replace("left_", "__TMP_LEFT_U__")
    swapped = swapped.replace("right_", "left_")
    swapped = swapped.replace("__TMP_LEFT_U__", "right_")

    swapped = swapped.replace("_left", "__TMP_LEFT_L__")
    swapped = swapped.replace("_right", "_left")
    swapped = swapped.replace("__TMP_LEFT_L__", "_right")

    swapped = swapped.replace("Left", "__TMP_LEFT_C__")
    swapped = swapped.replace("Right", "Left")
    swapped = swapped.replace("__TMP_LEFT_C__", "Right")
    return swapped


def _swap_feature_map_left_right(feature_map: Mapping[str, float]) -> Dict[str, float]:
    swapped: Dict[str, float] = {}
    for key, value in feature_map.items():
        swapped[_swap_left_right_key(key)] = _to_float(value, 0.0)
    return swapped


def _decode_once(
    feature_map: Mapping[str, float],
    config: Mapping[str, Any],
    temporal_state: MutableMapping[str, Any] | None,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], float]:
    rules = _parse_head_rules(config)
    temporal_cfg = config.get("temporal", {}) if isinstance(config, Mapping) else {}
    consistency_cfg = config.get("consistency", {}) if isinstance(config, Mapping) else {}
    head_bias_cfg = config.get("head_label_bias", {}) if isinstance(config, Mapping) else {}
    if not isinstance(temporal_cfg, Mapping):
        temporal_cfg = {}
    if not isinstance(consistency_cfg, Mapping):
        consistency_cfg = {}
    if not isinstance(head_bias_cfg, Mapping):
        head_bias_cfg = {}

    temporal_enabled = bool(temporal_cfg.get("enabled", False))
    hysteresis_margin = float(temporal_cfg.get("hysteresis_margin", 0.03))
    window_size = max(int(temporal_cfg.get("window_size", 3)), 1)
    majority_min_count = max(int(temporal_cfg.get("majority_min_count", 2)), 1)

    prev_labels: Dict[str, Any] = {}
    prev_history: Dict[str, List[Any]] = {}
    if temporal_state is not None:
        if isinstance(temporal_state.get("labels"), Mapping):
            prev_labels = dict(temporal_state.get("labels"))
        if isinstance(temporal_state.get("history"), Mapping):
            prev_history = {
                key: list(value)
                for key, value in temporal_state.get("history").items()
                if isinstance(value, list)
            }

    raw_labels: Dict[str, Any] = {}
    temporal_labels: Dict[str, Any] = {}
    debug: Dict[str, Any] = {}
    next_history: Dict[str, List[Any]] = {}
    per_head_penalties: Dict[str, Dict[Any, float]] = {}
    head_order = [rule.name for rule in rules]
    for rule in rules:
        head_bias = head_bias_cfg.get(rule.name, {})
        if not isinstance(head_bias, Mapping):
            head_bias = {}
        raw_label, base_penalty, confidence, penalties, eval_debug = _evaluate_head_rule(
            rule,
            feature_map,
            head_bias=head_bias,
        )
        per_head_penalties[rule.name] = dict(penalties)
        raw_labels[rule.name] = raw_label
        label = raw_label

        if temporal_enabled and rule.name in prev_labels:
            prev_label = prev_labels[rule.name]
            if label != prev_label and confidence < hysteresis_margin:
                label = prev_label

        history = list(prev_history.get(rule.name, []))
        history.append(label)
        if len(history) > window_size:
            history = history[-window_size:]

        if temporal_enabled:
            majority_label = _majority_label(history, min_count=min(majority_min_count, len(history)))
            if majority_label is not None:
                label = majority_label
                history[-1] = label

        temporal_labels[rule.name] = label
        next_history[rule.name] = history
        debug[rule.name] = {
            **eval_debug,
            "raw_label": raw_label,
            "label": label,
            "base_penalty": float(base_penalty),
            "confidence": float(confidence),
            "history": history,
        }

    labels, objective_after, consistency_debug = _optimize_with_consistency(
        labels_init=temporal_labels,
        per_head_penalties=per_head_penalties,
        temporal_labels=temporal_labels,
        head_order=head_order,
        consistency_cfg=consistency_cfg,
    )
    objective_before = _objective_score(temporal_labels, per_head_penalties, temporal_labels, consistency_cfg)

    for head_name, history in next_history.items():
        if not history:
            next_history[head_name] = [labels[head_name]]
            continue
        history[-1] = labels[head_name]
        if len(history) > window_size:
            history = history[-window_size:]
        next_history[head_name] = history
        if head_name in debug:
            debug[head_name]["label"] = labels[head_name]

    debug["_consistency"] = {
        **consistency_debug,
        "objective_before": float(objective_before),
        "objective_after": float(objective_after),
    }
    next_state = {"labels": labels, "history": next_history}
    return labels, debug, next_state, float(objective_after)


def decode_labels(
    feature_map: Mapping[str, float],
    config: Mapping[str, Any],
    temporal_state: MutableMapping[str, Any] | None = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    normal_labels, normal_debug, normal_state, normal_objective = _decode_once(
        feature_map,
        config,
        temporal_state=temporal_state,
    )

    swap_cfg = config.get("swap_validation", {}) if isinstance(config, Mapping) else {}
    if not isinstance(swap_cfg, Mapping) or not bool(swap_cfg.get("enabled", False)):
        normal_debug["_swap_validation"] = {
            "enabled": False,
            "selected": "normal",
            "objective_normal": float(normal_objective),
        }
        normal_state["swap_applied"] = False
        return normal_labels, normal_debug, normal_state

    swapped_map = _swap_feature_map_left_right(feature_map)
    swapped_labels, swapped_debug, swapped_state, swapped_objective = _decode_once(
        swapped_map,
        config,
        temporal_state=temporal_state,
    )

    min_improvement = float(swap_cfg.get("min_improvement", 0.0))
    prefer_previous = bool(swap_cfg.get("prefer_previous", True))
    prev_swap = bool(
        isinstance(temporal_state, Mapping)
        and temporal_state is not None
        and temporal_state.get("swap_applied", False)
    )

    use_swapped = False
    if swapped_objective + min_improvement < normal_objective:
        use_swapped = True
    elif normal_objective + min_improvement < swapped_objective:
        use_swapped = False
    else:
        if prefer_previous:
            use_swapped = prev_swap
        else:
            use_swapped = swapped_objective < normal_objective

    if use_swapped:
        swapped_debug["_swap_validation"] = {
            "enabled": True,
            "selected": "swapped",
            "objective_normal": float(normal_objective),
            "objective_swapped": float(swapped_objective),
            "min_improvement": float(min_improvement),
            "prefer_previous": bool(prefer_previous),
            "previous_swap": bool(prev_swap),
        }
        swapped_state["swap_applied"] = True
        return swapped_labels, swapped_debug, swapped_state

    normal_debug["_swap_validation"] = {
        "enabled": True,
        "selected": "normal",
        "objective_normal": float(normal_objective),
        "objective_swapped": float(swapped_objective),
        "min_improvement": float(min_improvement),
        "prefer_previous": bool(prefer_previous),
        "previous_swap": bool(prev_swap),
    }
    normal_state["swap_applied"] = False
    return normal_labels, normal_debug, normal_state


def head_names_from_decoder(config: Mapping[str, Any]) -> List[str]:
    return [rule.name for rule in _parse_head_rules(config)]
