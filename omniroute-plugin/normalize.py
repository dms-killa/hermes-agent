"""Reproduction of OmniRoute's combo write-path normalization, where it can be reproduced from source.

Every rule below cites OmniRoute at the SHA in docs/CAPABILITIES.md (rows O-WRITE-*):

- ``normalize_combo_step`` / ``normalize_combo_models``: port of ``normalizeComboStep`` /
  ``normalizeComboModels`` (src/lib/combos/steps.ts:272-457).
- ``normalize_combo_record``: port of ``normalizeComboRecord`` (steps.ts:459-471), which the
  repository applies to the whole merged record on every update (sqliteComboRepository.ts:252).
- ``strip_dead_config_keys``: port of ``stripDeadComboConfigKeys`` (src/lib/combos/deadConfigKeys.ts).
- ``WRITABLE_KEYS`` / ``NULLABLE_KEYS``: the top-level keys of ``updateComboSchema``
  (src/shared/validation/schemas/combo.ts:424-455); zod strips every other key.

- ``server_config``: the zod ``.transform`` on ``comboRuntimeConfigSchema`` (zero-latency flag
  promotion, schemas/combo.ts:301-319), then the dead-key strip.

NOT reproduced (named so plans and read-back can account for it): zod coercion, trimming and
defaults inside ``config`` (``comboRuntimeConfigSchema`` coerces numbers and fills ``weights``
defaults), zod range/length checks, ``validateCompositeTiersConfig``, model-family invariants, and
the server-set ``updatedAt``.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Dict, Iterable, List, Optional

COMBO_SCHEMA_VERSION = 2
QUOTA_MODEL_PREFIX = "qtSd/"                      # src/lib/quota/quotaModelNaming.ts:12
DEAD_COMBO_CONFIG_KEYS = ("pipelineConcurrency", "resetAwareEnabled", "resetAwareWindow")
MAX_COMBO_DEPTH, MAX_COMBO_DEPTH_HARD_CAP = 3, 10   # open-sse/services/combo/comboPredicates.ts:115-119

WRITABLE_KEYS = frozenset({
    "name", "description", "displayName", "models", "strategy", "config", "isActive", "isHidden",
    "allowedProviders", "allowedModelFamilies", "overrideAllowedProviders", "system_message",
    "tool_filter_regex", "context_cache_protection", "context_length", "compressionOverride",
    "dimensions",
})
NULLABLE_KEYS = frozenset({
    "description", "displayName", "allowedProviders", "allowedModelFamilies", "system_message",
    "tool_filter_regex", "context_cache_protection", "context_length", "dimensions",
})
# Writable, but outside this plugin's scope: rename, and two legacy/implicit rewrite switches.
OUT_OF_SCOPE_KEYS = frozenset({"name", "overrideAllowedProviders", "compressionOverride"})
PLANNABLE_KEYS = WRITABLE_KEYS - OUT_OF_SCOPE_KEYS
# Set by the server on every write; never compared.
SERVER_MANAGED_KEYS = frozenset({"updatedAt"})
# Nested config objects whose zod schema fills defaults: compare the keys that were sent.
CONFIG_SUBSET_COMPARED = frozenset({"weights"})


def _trimmed(value: Any) -> Optional[str]:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _js_number(value: float) -> Any:
    return int(value) if float(value).is_integer() else value


def _weight(value: Any) -> Any:
    if isinstance(value, bool):
        parsed = 0.0
    elif isinstance(value, (int, float)):
        parsed = float(value)
    elif isinstance(value, str) and value.strip():
        try:
            parsed = float(value)
        except ValueError:
            parsed = math.nan
    else:
        parsed = 0.0
    if not math.isfinite(parsed):
        return 0
    return _js_number(max(0.0, min(100.0, parsed)))


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "step"


def _step_id(kind: str, combo_name: Optional[str], index: int, seed: str) -> str:
    kind_slug = "ref" if kind == "combo-ref" else "wildcard" if kind == "provider-wildcard" else "model"
    return "-".join([_slugify(combo_name or "combo"), kind_slug, str(index + 1), _slugify(seed)])[:200]


def _provider_of(model: str) -> Optional[str]:
    model = model.strip()
    slash = model.find("/")
    if slash <= 0:
        return None
    return model[:slash].strip() or None


def _wildcard(target: str):
    target = target.strip()
    slash = target.find("/")
    if slash <= 0:
        return None
    provider, pattern = target[:slash].strip(), target[slash + 1:].strip()
    if not provider or "*" not in pattern:
        return None
    return provider, pattern


def _full_model(model: str, provider: Optional[str]) -> str:
    model = model.strip()
    if "/" in model:
        return model
    return f"{provider}/{model}" if _trimmed(provider) else model


def _treat_as_combo_ref(target: str, provider: Optional[str], names: Iterable[str], combo_name: Optional[str]) -> bool:
    if provider:
        return False
    return target in set(names) or target == _trimmed(combo_name)


def _conn_ids(value: Any) -> Optional[List[str]]:
    if not isinstance(value, list):
        return None
    return [t for t in (_trimmed(v) for v in value) if t]


def normalize_combo_step(value: Any, *, combo_name: Optional[str], index: int,
                         names: Iterable[str]) -> Optional[Dict[str, Any]]:
    names = list(names)
    combo_name = _trimmed(combo_name)
    if isinstance(value, str):
        target = _trimmed(value)
        if not target:
            return None
        if _treat_as_combo_ref(target, None, names, combo_name):
            return {"id": _step_id("combo-ref", combo_name, index, target), "kind": "combo-ref",
                    "comboName": target, "weight": 0}
        wild = _wildcard(target)
        if wild:
            provider, pattern = wild
            return {"id": _step_id("provider-wildcard", combo_name, index, f"{provider}/{pattern}"),
                    "kind": "provider-wildcard", "providerId": provider, "modelPattern": pattern, "weight": 0}
        provider = _provider_of(target)
        step = {"id": _step_id("model", combo_name, index, target), "kind": "model", "model": target}
        if provider:
            step["providerId"] = provider
        step["weight"] = 0
        return step
    if not isinstance(value, dict):
        return None
    explicit_id = _trimmed(value.get("id"))
    weight = _weight(value.get("weight"))
    label = _trimmed(value.get("label"))
    prompt = _trimmed(value.get("prompt"))
    fallback_only = value.get("fallbackOnlyOnQuotaExhaustion") is True

    if value.get("kind") == "combo-ref":
        ref = _trimmed(value.get("comboName"))
        if not ref:
            return None
        step = {"id": explicit_id or _step_id("combo-ref", combo_name, index, ref), "kind": "combo-ref",
                "comboName": ref, "weight": weight}
        if label:
            step["label"] = label
        if fallback_only:
            step["fallbackOnlyOnQuotaExhaustion"] = True
        return step

    def wildcard_step(provider: str, pattern: str) -> Dict[str, Any]:
        conn = None if value.get("connectionId") is None and "connectionId" in value else (
            _trimmed(value.get("connectionId")) or _ABSENT)
        allowed = _conn_ids(value.get("allowedConnectionIds"))
        seed = f"{provider}/{pattern}:{conn}" if conn not in (None, _ABSENT) else f"{provider}/{pattern}"
        step = {"id": explicit_id or _step_id("provider-wildcard", combo_name, index, seed),
                "kind": "provider-wildcard", "providerId": provider, "modelPattern": pattern}
        if conn is not _ABSENT:
            step["connectionId"] = conn
        step["weight"] = weight
        if label:
            step["label"] = label
        if allowed:
            step["allowedConnectionIds"] = allowed
        return step

    if value.get("kind") == "provider-wildcard":
        provider = (_trimmed(value.get("providerId")) or _trimmed(value.get("provider"))
                    or _provider_of(_trimmed(value.get("model")) or ""))
        if not provider:
            return None
        return wildcard_step(provider, _trimmed(value.get("modelPattern")) or "*")

    raw_model = _trimmed(value.get("model"))
    if not raw_model:
        return None
    explicit_model = value.get("kind") == "model"
    provider = _trimmed(value.get("providerId")) or _trimmed(value.get("provider")) or _provider_of(raw_model)
    wild = _wildcard(raw_model)
    if wild:
        return wildcard_step(*wild)
    if not explicit_model and _treat_as_combo_ref(raw_model, provider, names, combo_name):
        step = {"id": explicit_id or _step_id("combo-ref", combo_name, index, raw_model), "kind": "combo-ref",
                "comboName": raw_model, "weight": weight}
        if label:
            step["label"] = label
        return step
    model = _full_model(raw_model, provider)
    conn = None if value.get("connectionId") is None and "connectionId" in value else (
        _trimmed(value.get("connectionId")) or _ABSENT)
    tags = [t for t in (_trimmed(v) for v in value["tags"]) if t] if isinstance(value.get("tags"), list) else None
    allowed = _conn_ids(value.get("allowedConnectionIds"))
    seed = f"{model}:{conn}" if conn not in (None, _ABSENT) else model
    step = {"id": explicit_id or _step_id("model", combo_name, index, seed), "kind": "model", "model": model}
    if provider:
        step["providerId"] = provider
    if conn is not _ABSENT:
        step["connectionId"] = conn
    step["weight"] = weight
    if label:
        step["label"] = label
    if prompt:
        step["prompt"] = prompt
    if tags:
        step["tags"] = tags
    if allowed:
        step["allowedConnectionIds"] = allowed
    if fallback_only:
        step["fallbackOnlyOnQuotaExhaustion"] = True
    return step


_ABSENT = object()


def normalize_combo_models(models: Any, *, combo_name: Optional[str], names: Iterable[str]) -> List[Dict[str, Any]]:
    names = list(names)
    out = []
    for index, value in enumerate(models if isinstance(models, list) else []):
        step = normalize_combo_step(value, combo_name=combo_name, index=index, names=names)
        if step is not None:
            out.append(step)
    return out


def normalize_combo_record(combo: Dict[str, Any], names: Iterable[str]) -> Dict[str, Any]:
    name = _trimmed(combo.get("name"))
    return {**combo, "version": COMBO_SCHEMA_VERSION,
            "models": normalize_combo_models(combo.get("models"), combo_name=name, names=names)}


def strip_dead_config_keys(config: Any) -> Any:
    if not isinstance(config, dict):
        return config
    return {k: v for k, v in config.items() if k not in DEAD_COMBO_CONFIG_KEYS}


def needs_zero_latency_promotion(config: Any) -> bool:
    """``comboRuntimeConfigSchema``'s ``.transform`` (schemas/combo.ts:301-319): a config with a
    zero-latency feature (hedging, predictiveTtftMs > 0, fallbackCompressionMode not "off") but without
    ``zeroLatencyOptimizationsEnabled: true`` gets that flag added on every config write."""
    if not isinstance(config, dict) or config.get("zeroLatencyOptimizationsEnabled") is True:
        return False
    ttft = config.get("predictiveTtftMs")
    fallback = config.get("fallbackCompressionMode")
    return (config.get("hedging") is True
            or (isinstance(ttft, (int, float)) and not isinstance(ttft, bool) and ttft > 0)
            or (bool(fallback) and fallback != "off"))


def server_config(config: Any) -> Any:
    """The config the server stores for a ``config`` in the PUT body: zod transform, then dead keys
    stripped (route.ts:124-126). Zod coercion and sub-schema defaults are not reproduced."""
    if not isinstance(config, dict):
        return config
    if needs_zero_latency_promotion(config):
        config = {**config, "zeroLatencyOptimizationsEnabled": True}
    return strip_dead_config_keys(config)


def clamp_combo_depth(value: Any) -> int:
    try:
        n = math.floor(float(value))
    except (TypeError, ValueError):
        return MAX_COMBO_DEPTH
    if not math.isfinite(n) or n < 1:
        return MAX_COMBO_DEPTH
    return min(int(n), MAX_COMBO_DEPTH_HARD_CAP)


def content_hash(combo: Dict[str, Any]) -> str:
    """Optimistic-concurrency token: SHA-256 over the canonical JSON of the full stored combo.
    It includes ``updatedAt``, which the server rewrites on every update, so any write changes it."""
    data = json.dumps(combo, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(data.encode("utf-8")).hexdigest()


def diff_paths(before: Any, after: Any, path: str = "", ignore: Iterable[str] = ()) -> List[Dict[str, Any]]:
    """Field-level differences as ``[{path, before, after}]``; top-level keys in ``ignore`` skipped.
    A missing key is reported with the marker ``"<absent>"``."""
    ignore = set(ignore)
    out: List[Dict[str, Any]] = []
    if isinstance(before, dict) and isinstance(after, dict):
        for key in sorted(set(before) | set(after)):
            if not path and key in ignore:
                continue
            sub = f"{path}.{key}" if path else key
            if key not in before:
                out.append({"path": sub, "before": "<absent>", "after": after[key]})
            elif key not in after:
                out.append({"path": sub, "before": before[key], "after": "<absent>"})
            else:
                out.extend(diff_paths(before[key], after[key], sub))
        return out
    if isinstance(before, list) and isinstance(after, list) and len(before) == len(after):
        for i, (b, a) in enumerate(zip(before, after)):
            out.extend(diff_paths(b, a, f"{path}[{i}]"))
        return out
    if not _same_scalar(before, after):
        out.append({"path": path or "<root>", "before": before, "after": after})
    return out


def _same_scalar(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):   # Python: True == 1; JSON: they differ
        return type(a) is type(b) and a == b
    return a == b
