"""``hermes_combo_check``: do the Hermes profile's model settings name OmniRoute combos, and do
those combos' legs accept what each Hermes task sends?

The integration model is "Hermes uses combos, not backends": every model setting that points at
the OmniRoute base URL should name a combo, so backends are swapped in the combo and not in
Hermes config. Settings read (Hermes source in docs/CAPABILITIES.md, rows H-MODEL-*):

- ``model`` (string, or mapping with ``default``/``model``, ``provider``, ``base_url``)
- ``providers.<name>.base_url``: a named provider that ``model.provider`` (or a task's
  ``provider``) refers to
- ``auxiliary.<task>.{provider, model, base_url, timeout}``; ``provider: auto`` with no model
  inherits the main model
- ``delegation.{provider, model, base_url}``
- ``fallback_providers`` / ``fallback_model`` entries
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .audit import combo_view
from .findings import finding, summarize
from .model import Combo, effective_target_timeout
from .settings import SettingsError, normalize_base_url

# What each Hermes task sends, from Hermes source (CAPABILITIES.md H-MODEL-5, H-MODEL-6).
STRUCTURED_OUTPUT_TASKS = frozenset({"auxiliary.title_generation"})
TOOL_CALLING_TASKS = frozenset({"model", "delegation", "auxiliary.review"})
HERMES_DEFAULT_API_TIMEOUT_S = 1800.0   # agent/chat_completion_helpers.py, HERMES_API_TIMEOUT default


@dataclass
class ModelSetting:
    task: str               # "model", "auxiliary.vision", "delegation", "fallback_providers[0]"
    model: str
    provider: str
    base_url: str
    timeout_s: Optional[float]
    timeout_source: str
    inherits_main: bool = False


def _rec(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _str(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _provider_timeout(config: Dict[str, Any], provider: str, model: str) -> Tuple[Optional[float], str]:
    """Mirror hermes_cli/timeouts.py: per-model timeout_seconds, then request_timeout_seconds."""
    pcfg = _rec(_rec(config.get("providers")).get(provider))
    per_model = _float(_rec(_rec(pcfg.get("models")).get(model)).get("timeout_seconds"))
    if per_model:
        return per_model, f"providers.{provider}.models.{model}.timeout_seconds"
    per_provider = _float(pcfg.get("request_timeout_seconds"))
    if per_provider:
        return per_provider, f"providers.{provider}.request_timeout_seconds"
    return None, ""


def _resolve_base_url(config: Dict[str, Any], provider: str, base_url: str) -> str:
    if base_url:
        return base_url
    named = _rec(_rec(config.get("providers")).get(provider))
    return _str(named.get("base_url"))


def collect_model_settings(config: Dict[str, Any]) -> List[ModelSetting]:
    out: List[ModelSetting] = []
    raw_model = config.get("model")
    if isinstance(raw_model, dict):
        main_model = _str(raw_model.get("default")) or _str(raw_model.get("model"))
        main_provider = _str(raw_model.get("provider"))
        main_base = _resolve_base_url(config, main_provider, _str(raw_model.get("base_url")))
    else:
        main_model, main_provider, main_base = _str(raw_model), "", ""
    if main_model:
        timeout, source = _provider_timeout(config, main_provider, main_model)
        out.append(ModelSetting("model", main_model, main_provider, main_base, timeout, source))

    for task, block in _rec(config.get("auxiliary")).items():
        block = _rec(block)
        if not block:
            continue
        provider, model = _str(block.get("provider")), _str(block.get("model"))
        base = _resolve_base_url(config, provider, _str(block.get("base_url")))
        timeout = _float(block.get("timeout"))
        source = f"auxiliary.{task}.timeout" if timeout else ""
        if (not provider or provider == "auto") and not model and not base:
            if main_model:
                out.append(ModelSetting(f"auxiliary.{task}", main_model, main_provider, main_base,
                                        timeout, source, inherits_main=True))
            continue
        if model or base:
            out.append(ModelSetting(f"auxiliary.{task}", model or main_model, provider, base,
                                    timeout, source))

    delegation = _rec(config.get("delegation"))
    if _str(delegation.get("model")) or _str(delegation.get("base_url")):
        provider = _str(delegation.get("provider"))
        model = _str(delegation.get("model")) or main_model
        base = _resolve_base_url(config, provider, _str(delegation.get("base_url")))
        timeout, source = _provider_timeout(config, provider, model)
        out.append(ModelSetting("delegation", model, provider, base, timeout, source))

    fallbacks: Iterable[Any] = config.get("fallback_providers") or []
    raw_fallback_model = config.get("fallback_model")
    if isinstance(raw_fallback_model, dict):
        raw_fallback_model = [raw_fallback_model]
    for label, entries in (("fallback_providers", fallbacks), ("fallback_model", raw_fallback_model or [])):
        if not isinstance(entries, list):
            continue
        for i, entry in enumerate(entries):
            entry = _rec(entry)
            provider, model = _str(entry.get("provider")), _str(entry.get("model"))
            base = _resolve_base_url(config, provider, _str(entry.get("base_url")))
            if model:
                timeout, source = _provider_timeout(config, provider, model)
                out.append(ModelSetting(f"{label}[{i}]", model, provider, base, timeout, source))
    return out


def points_at_gateway(setting: ModelSetting, gateway_base: str) -> bool:
    if not setting.base_url:
        return False
    try:
        return normalize_base_url(setting.base_url) == gateway_base
    except SettingsError:
        return False


def _traits(task: str) -> List[str]:
    traits = []
    root = task.split("[", 1)[0]
    if task in STRUCTURED_OUTPUT_TASKS:
        traits.append("structured_output")
    if task in TOOL_CALLING_TASKS or root in ("fallback_providers", "fallback_model"):
        traits.append("tool_calling")
    return traits


def run_combo_check(config: Dict[str, Any], gateway_base: str, combos: List[Combo],
                    combo_defaults: Dict[str, Any], *, profile: str = "") -> Dict[str, Any]:
    by_name = {c.name: c for c in combos}
    settings = collect_model_settings(config)
    findings: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []
    for setting in settings:
        if not points_at_gateway(setting, gateway_base):
            rows.append({"task": setting.task, "model": setting.model, "points_at_omniroute": False})
            continue
        traits = _traits(setting.task)
        combo = by_name.get(setting.model)
        row: Dict[str, Any] = {"task": setting.task, "model": setting.model, "points_at_omniroute": True,
                               "inherits_main": setting.inherits_main, "traits": traits,
                               "combo_found": combo is not None}
        rows.append(row)
        subject = {"task": setting.task, "model": setting.model}
        if combo is None:
            direct = "/" in setting.model
            findings.append(finding(
                "COMBO_NOT_FOUND", "L10", "warning" if direct else "error",
                (f"{setting.task} sends model '{setting.model}' to OmniRoute but no combo has that "
                 "name. " + ("It looks like a direct provider/model id, so Hermes is coupled to a "
                             "backend and swaps need a Hermes config change." if direct else
                             "OmniRoute will answer 404 (body code model_not_found), which Hermes "
                             "classifies as model_not_found and falls back.")),
                action="Create a combo with this name, or point the setting at an existing combo.",
                cite=["H-MODEL-1", "O-COMBO-1", "L10"], subject=subject))
            continue
        row["combo"] = combo_view(combo, combo_defaults)
        if not combo.active:
            findings.append(finding(
                "COMBO_INACTIVE", "", "warning",
                f"{setting.task} uses combo '{combo.name}', which is marked inactive.",
                cite=["O-COMBO-1"], subject=subject))
        gated = sorted({f"{leg.index}:{c.name}" for leg in combo.legs for c in leg.connections
                        if c.l1_gated})
        if "structured_output" in traits and gated:
            findings.append(finding(
                "STRUCTURED_OUTPUT_ON_L1_GATED_LEG", "L1", "warning",
                f"{setting.task} sends response_format json_schema, and combo '{combo.name}' has "
                f"legs on L1-gated connections ({', '.join(gated)}). On those legs OmniRoute "
                f"downgrades it to json_object, so the schema is not enforced.",
                action="Move these legs to a local-provider id, or accept prompt-only JSON on them.",
                cite=["L1", "O-L1-1", "H-MODEL-5"], subject={**subject, "legs": gated}))
        if "tool_calling" in traits and combo.tool_filter_regex:
            findings.append(finding(
                "TOOLS_FILTERED_BY_COMBO", "", "warning",
                f"{setting.task} sends tools, and combo '{combo.name}' has tool_filter_regex "
                f"{combo.tool_filter_regex!r}: OmniRoute drops every tool whose name does not match "
                f"before forwarding.",
                action="Clear tool_filter_regex on this combo or widen it to Hermes' tool names.",
                cite=["O-COMBO-3", "H-MODEL-6"], subject=subject))
        if "tool_calling" in traits and combo.system_message:
            findings.append(finding(
                "SYSTEM_PROMPT_REPLACED_BY_COMBO", "", "warning",
                f"{setting.task} relies on Hermes' system prompt (tool guidance, memory, skills), "
                f"and combo '{combo.name}' has a system_message: OmniRoute removes every system "
                f"message in the request and injects its own.",
                action="Remove system_message from combos that Hermes' agent loop uses.",
                cite=["O-COMBO-3", "H-MODEL-6"], subject=subject))
        if setting.timeout_s:
            per_call_ms = int(setting.timeout_s * 1000)
            for leg in combo.legs:
                if leg.kind == "combo-ref":
                    continue
                pinned = leg.connections[0] if leg.pinned_connection_id and leg.connections else None
                leg_ms = effective_target_timeout(combo, combo_defaults, pinned)["ms"]
                if leg_ms < per_call_ms:
                    findings.append(finding(
                        "L2_LEG_TIMEOUT_BELOW_CALL_TIMEOUT", "L2", "warning",
                        f"{setting.task}: Hermes waits up to {per_call_ms} ms per call "
                        f"({setting.timeout_source}), but OmniRoute abandons leg {leg.index} of "
                        f"combo '{combo.name}' after {leg_ms} ms. Both are per-request timeouts.",
                        action=f"Raise the combo's config.targetTimeoutMs to at least {per_call_ms}.",
                        cite=["L2", "O-COMBO-4", "H-MODEL-4"],
                        subject={**subject, "leg": leg.index, "leg_timeout_ms": leg_ms,
                                 "per_call_timeout_ms": per_call_ms}))
        elif setting.task == "model":
            row["per_call_timeout_note"] = (
                f"no explicit per-call timeout; Hermes uses HERMES_API_TIMEOUT "
                f"(default {int(HERMES_DEFAULT_API_TIMEOUT_S)} s) for the main model")
    return {
        "report": "hermes_combo_check", "gateway": gateway_base, "profile": profile,
        "summary": summarize(findings), "findings": findings, "settings": rows,
        "checked_settings": len(settings),
        "pointing_at_omniroute": sum(1 for r in rows if r["points_at_omniroute"]),
    }
