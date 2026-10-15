"""``omniroute_audit``: a read-only report on connections and combos (lessons L1-L4).

Inputs are the decoded JSON of four allowlisted GETs (connections, provider nodes, combos,
combo defaults) plus optional caller limits. The only thresholds are OmniRoute's own defaults
(``model.DEFAULT_COMBO_TARGET_TIMEOUT_MS``, ``model.DEFAULT_FETCH_TIMEOUT_MS``) and the numbers
the caller passes in.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional

from .findings import finding, summarize
from .model import (LOCAL_PROVIDER_IDS, Combo, Connection, combo_defaults_of,
                    effective_target_timeout, parse_combos, parse_connections, parse_nodes)

P2C = "p2c"


def build_snapshot(connections_raw: Any, nodes_raw: Any, combos_raw: Any,
                   combo_defaults_raw: Any) -> Dict[str, Any]:
    nodes = parse_nodes(nodes_raw)
    connections = parse_connections(connections_raw, nodes)
    combos = parse_combos(combos_raw, connections, nodes)
    return {"nodes": nodes, "connections": connections, "combos": combos,
            "combo_defaults": combo_defaults_of(combo_defaults_raw)}


def leg_view(combo: Combo, combo_defaults: Dict[str, Any]) -> List[Dict[str, Any]]:
    legs = []
    for leg in combo.legs:
        pinned = leg.connections[0] if leg.pinned_connection_id and leg.connections else None
        legs.append({
            "index": leg.index, "step_id": leg.step_id, "kind": leg.kind,
            "model": leg.model or None, "combo_ref": leg.combo_ref or None,
            "pinned_connection_id": leg.pinned_connection_id,
            "pinned_connection_found": bool(pinned) if leg.pinned_connection_id else None,
            "connections": [{"id": c.id, "name": c.name, "provider": c.provider,
                             "max_concurrent": c.max_concurrent, "l1_gated": c.l1_gated,
                             "host_class": c.host_class} for c in leg.connections],
            "target_timeout": effective_target_timeout(combo, combo_defaults, pinned),
        })
    return legs


def combo_view(combo: Combo, combo_defaults: Dict[str, Any]) -> Dict[str, Any]:
    return {"id": combo.id, "name": combo.name, "strategy": combo.strategy, "active": combo.active,
            "system_message_set": bool(combo.system_message),
            "tool_filter_regex": combo.tool_filter_regex or None,
            "legs": leg_view(combo, combo_defaults)}


def _l1(connections: Dict[str, Connection], combos: List[Combo]) -> List[Dict[str, Any]]:
    out = []
    used_by: Dict[str, List[str]] = defaultdict(list)
    for combo in combos:
        for leg in combo.legs:
            for conn in leg.connections:
                if combo.name not in used_by[conn.id]:
                    used_by[conn.id].append(combo.name)
    for conn in connections.values():
        if conn.l1_gated and conn.host_class:
            out.append(finding(
                "L1_JSON_SCHEMA_DOWNGRADE", "L1", "warning",
                f"Connection '{conn.name}' uses provider id '{conn.provider}', which OmniRoute's "
                f"executor rewrites from response_format json_schema to json_object (schema pasted "
                f"into the system prompt), but its base URL is a {conn.host_class} host, i.e. a "
                f"self-hosted server that could enforce the schema natively. Grammar enforcement "
                f"is lost silently.",
                action=("Register this backend with a local-provider id instead of a generic "
                        "openai-compatible node, e.g. one of: " + ", ".join(LOCAL_PROVIDER_IDS[:6])
                        + ". The provider type is a correctness setting."),
                cite=["L1", "O-L1-1", "O-L1-2", "O-CONN-3"],
                subject={"connection_id": conn.id, "provider": conn.provider,
                         "host_class": conn.host_class, "base_url_source": conn.base_url_source,
                         "combos": used_by.get(conn.id, [])}))
    return out


def _l2(combo: Combo, combo_defaults: Dict[str, Any], per_call_timeout_ms: Optional[int],
        source: str = "caller argument") -> List[Dict[str, Any]]:
    if not per_call_timeout_ms:
        return []
    out = []
    for leg in combo.legs:
        if leg.kind == "combo-ref":
            continue
        pinned = leg.connections[0] if leg.pinned_connection_id and leg.connections else None
        timeout = effective_target_timeout(combo, combo_defaults, pinned)
        if timeout["ms"] < per_call_timeout_ms:
            out.append(finding(
                "L2_LEG_TIMEOUT_BELOW_CALL_TIMEOUT", "L2", "warning",
                f"Combo '{combo.name}' leg {leg.index} ({leg.model}) is abandoned by OmniRoute "
                f"after {timeout['ms']} ms ({timeout['source']}, ceiling {timeout['ceiling_source']}), "
                f"but one call from the caller may legitimately take up to {per_call_timeout_ms} ms "
                f"({source}). Both numbers are per single request, so they compare directly.",
                action=(f"Set the combo's config.targetTimeoutMs to at least {per_call_timeout_ms} "
                        "(do not compare it with a multi-call operation budget). For streaming "
                        "requests the leg timeout only bounds time-to-first-headers."),
                cite=["L2", "O-COMBO-4", "O-COMBO-5", "O-COMBO-6"],
                subject={"combo": combo.name, "leg": leg.index, "leg_timeout_ms": timeout["ms"],
                         "per_call_timeout_ms": per_call_timeout_ms}))
    return out


def _l3(combo: Combo, concurrency: Optional[int]) -> List[Dict[str, Any]]:
    if combo.strategy != P2C:
        return []
    expects_parallel = bool(concurrency and concurrency > 1)
    return [finding(
        "L3_P2C_NOT_INFLIGHT_BALANCED", "L3", "warning" if expects_parallel else "info",
        f"Combo '{combo.name}' uses p2c, which picks between two random legs by historical success "
        f"rate and latency. It does not count requests in flight"
        + (f", so {concurrency} concurrent requests are not spread by load." if expects_parallel
           else "."),
        action=("For ordered spill-over under concurrency, use strategy 'priority' with each leg "
                "pinned to a connection and a max_concurrent on each connection. Live in-flight "
                "balancing exists only in the internal quota-share strategy."),
        cite=["L3", "O-ROUTE-1", "O-ROUTE-2", "O-ROUTE-3"],
        subject={"combo": combo.name, "strategy": combo.strategy, "concurrency": concurrency})]


def _l4(combos: List[Combo], concurrency: Optional[int]) -> List[Dict[str, Any]]:
    out = []
    shared: Dict[str, List[str]] = defaultdict(list)
    names: Dict[str, Connection] = {}
    for combo in combos:
        for leg in combo.legs:
            for conn in leg.connections:
                if conn.max_concurrent and combo.name not in shared[conn.id]:
                    shared[conn.id].append(combo.name)
                    names[conn.id] = conn
    for conn_id, combo_names in shared.items():
        if len(combo_names) > 1:
            conn = names[conn_id]
            out.append(finding(
                "L4_SHARED_CONNECTION_CAP", "L4", "info",
                f"Connection '{conn.name}' has max_concurrent={conn.max_concurrent} and is used by "
                f"{len(combo_names)} combos ({', '.join(sorted(combo_names))}). The cap belongs to "
                f"the connection and is shared by all of them; it is not a per-combo limit.",
                action="Size max_concurrent for the sum of traffic across these combos.",
                cite=["L4", "O-CONN-2", "O-ROUTE-4"],
                subject={"connection_id": conn_id, "max_concurrent": conn.max_concurrent,
                         "combos": sorted(combo_names)}))
    if not concurrency:
        return out
    for combo in combos:
        model_legs = [leg for leg in combo.legs if leg.kind != "combo-ref"]
        if not model_legs:
            continue
        caps, uncapped = {}, []
        for leg in model_legs:
            if not leg.connections:
                uncapped.append(leg.index)
            for conn in leg.connections:
                if conn.max_concurrent:
                    caps[conn.id] = conn.max_concurrent
                else:
                    uncapped.append(leg.index)
        if uncapped:
            continue  # at least one leg has no per-connection cap: spill-over never runs dry
        capacity = sum(caps.values())
        if concurrency > capacity:
            out.append(finding(
                "L4_CONCURRENCY_EXCEEDS_CAPS", "L4", "warning",
                f"Combo '{combo.name}': every leg's connection is capped (sum of max_concurrent = "
                f"{capacity}, counting each connection once), but the caller runs {concurrency} "
                f"requests at once. When all caps are full, OmniRoute skips each leg before dispatch "
                f"and the request fails; other combos using the same connections reduce the "
                f"headroom further.",
                action=(f"Lower the caller's concurrency to {capacity} or less, raise max_concurrent "
                        "on the connections, or add a leg with spare capacity."),
                cite=["L4", "O-CONN-2", "O-ROUTE-4"],
                subject={"combo": combo.name, "capacity": capacity, "concurrency": concurrency,
                         "caps": caps}))
    return out


def run_audit(snapshot: Dict[str, Any], *, per_call_timeout_ms: Optional[int] = None,
              concurrency: Optional[int] = None, combo_names: Optional[List[str]] = None,
              gateway: str = "") -> Dict[str, Any]:
    connections: Dict[str, Connection] = snapshot["connections"]
    all_combos: List[Combo] = snapshot["combos"]
    combo_defaults = snapshot["combo_defaults"]
    combos = [c for c in all_combos if not combo_names or c.name in combo_names]
    if combo_names:
        used = {conn.id for c in combos for leg in c.legs for conn in leg.connections}
        findings = _l1({k: v for k, v in connections.items() if k in used}, combos)
    else:
        findings = _l1(connections, all_combos)
    for combo in combos:
        findings += _l2(combo, combo_defaults, per_call_timeout_ms)
        findings += _l3(combo, concurrency)
    findings += _l4(all_combos if not combo_names else combos, concurrency)
    missing = sorted({leg.pinned_connection_id for c in combos for leg in c.legs
                      if leg.pinned_connection_id and not leg.connections})
    for conn_id in missing:
        findings.append(finding(
            "PINNED_CONNECTION_MISSING", "L4", "warning",
            f"A combo leg is pinned to connection id '{conn_id}', which GET /api/providers did not "
            f"return; its max_concurrent and base URL cannot be checked.",
            cite=["O-COMBO-2", "O-CONN-1"], subject={"connection_id": conn_id}))
    unknown = [n for n in (combo_names or []) if n not in {c.name for c in all_combos}]
    return {
        "report": "omniroute_audit", "gateway": gateway,
        "inputs": {"per_call_timeout_ms": per_call_timeout_ms, "concurrency": concurrency,
                   "combos": combo_names or "all"},
        "summary": summarize(findings),
        "findings": findings,
        "connections": [c.public() for c in connections.values()],
        "combos": [combo_view(c, combo_defaults) for c in combos],
        "unknown_combos": unknown,
        "not_checked": [
            "FETCH_TIMEOUT_MS server env override (assumed default 600000 ms when a leg has no "
            "connection timeoutMs)",
            "resilience.comboCooldownWait, which can raise the default leg timeout floor",
            "per-provider settings.providerOverrides (not applied by combo setup at this SHA)",
        ],
    }
