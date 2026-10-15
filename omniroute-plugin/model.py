"""Normalize OmniRoute management responses into the few fields the reports need.

Shapes follow OmniRoute source at the SHA in docs/CAPABILITIES.md:
- ``GET /api/providers`` -> ``{connections: [...], total}`` (O-CONN-1); a connection carries
  ``provider``, ``maxConcurrent`` (column ``max_concurrent``, O-CONN-2) and a sanitized
  ``providerSpecificData`` that may hold ``baseUrl`` and ``timeoutMs``.
- ``GET /api/provider-nodes`` -> ``{nodes: [...]}`` with ``id``/``type``/``prefix``/``baseUrl``
  (O-CONN-3); compatible connections take their base URL from the node when not stamped.
- ``GET /api/combos`` -> ``{combos: [...], total}``; steps are ``ComboStep`` (O-COMBO-2).

Secrets never leave this module: only an explicit field allowlist is copied out of a connection
(``apiKey`` may be returned unmasked when the operator enabled key reveal, O-CONN-1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .netclass import classify_url

# OmniRoute's own defaults (O-COMBO-4, O-COMBO-6). Not tunable here on purpose.
DEFAULT_COMBO_TARGET_TIMEOUT_MS = 120_000
DEFAULT_FETCH_TIMEOUT_MS = 600_000
DEFAULT_STRATEGY = "priority"

# json_schema -> json_object downgrade gate (L1): open-sse/executors/default.ts applyJsonSchemaFallback
L1_GATED_PREFIX = "openai-compatible-"
L1_GATED_EXACT = frozenset({"kilocode"})

# Local-provider registry ids (src/shared/constants/providers/local.ts) that are NOT gated.
LOCAL_PROVIDER_IDS = ("ollama-local", "llama-cpp", "lm-studio", "vllm", "lemonade", "llamafile",
                      "triton", "docker-model-runner", "xinference", "oobabooga", "mlx-gemma",
                      "mlx-qwen")


def is_l1_gated(provider_id: str) -> bool:
    provider_id = provider_id or ""
    return provider_id.startswith(L1_GATED_PREFIX) or provider_id in L1_GATED_EXACT


def _rec(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any, *keys: str) -> List[Any]:
    if isinstance(value, list):
        return value
    for key in keys:
        inner = _rec(value).get(key)
        if isinstance(inner, list):
            return inner
    return []


def _positive_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


@dataclass
class Node:
    id: str
    type: str
    prefix: str
    name: str
    base_url: str


@dataclass
class Connection:
    id: str
    provider: str
    name: str
    active: bool
    max_concurrent: Optional[int]
    base_url: str
    base_url_source: str
    timeout_ms: Optional[int]
    host_class: Optional[str]

    @property
    def l1_gated(self) -> bool:
        return is_l1_gated(self.provider)

    def public(self) -> Dict[str, Any]:
        return {"id": self.id, "provider": self.provider, "name": self.name, "active": self.active,
                "max_concurrent": self.max_concurrent, "base_url": self.base_url or None,
                "base_url_source": self.base_url_source, "host_class": self.host_class,
                "l1_gated": self.l1_gated}


@dataclass
class Leg:
    index: int
    step_id: str
    kind: str
    model: str
    provider_token: str
    pinned_connection_id: Optional[str]
    allowed_connection_ids: List[str]
    weight: float
    combo_ref: str = ""
    connections: List[Connection] = field(default_factory=list)


@dataclass
class Combo:
    id: str
    name: str
    strategy: str
    active: bool
    config: Dict[str, Any]
    system_message: str
    tool_filter_regex: str
    legs: List[Leg]


def parse_nodes(raw: Any) -> Dict[str, Node]:
    nodes: Dict[str, Node] = {}
    for item in _list(raw, "nodes"):
        item = _rec(item)
        node_id = str(item.get("id") or "")
        if not node_id:
            continue
        nodes[node_id] = Node(id=node_id, type=str(item.get("type") or ""),
                              prefix=str(item.get("prefix") or ""), name=str(item.get("name") or ""),
                              base_url=str(item.get("baseUrl") or ""))
    return nodes


def parse_connections(raw: Any, nodes: Dict[str, Node]) -> Dict[str, Connection]:
    out: Dict[str, Connection] = {}
    for item in _list(raw, "connections"):
        item = _rec(item)
        conn_id = str(item.get("id") or "")
        if not conn_id:
            continue
        provider = str(item.get("provider") or "")
        psd = _rec(item.get("providerSpecificData"))
        base_url, source = str(psd.get("baseUrl") or ""), "connection.providerSpecificData.baseUrl"
        if not base_url and provider in nodes and nodes[provider].base_url:
            base_url, source = nodes[provider].base_url, "provider_node.baseUrl"
        if not base_url:
            source = "unknown"
        active = item.get("isActive")
        out[conn_id] = Connection(
            id=conn_id, provider=provider,
            name=str(item.get("displayName") or item.get("name") or item.get("email") or conn_id),
            active=active is not False, max_concurrent=_positive_int(item.get("maxConcurrent")),
            base_url=base_url, base_url_source=source, timeout_ms=_positive_int(psd.get("timeoutMs")),
            host_class=classify_url(base_url))
    return out


def _resolve_provider_ids(token: str, nodes: Dict[str, Node]) -> List[str]:
    """Provider ids a step's provider token can mean: itself, a node whose prefix/id matches, or the
    sole node of that generic type (mirrors the unambiguous-type fallback in providerNodeSelect)."""
    if not token:
        return []
    ids = {token}
    ids.update(n.id for n in nodes.values() if n.prefix == token)
    of_type = [n.id for n in nodes.values() if n.type == token]
    if len(of_type) == 1:
        ids.add(of_type[0])
    return sorted(ids)


def parse_combos(raw: Any, connections: Dict[str, Connection], nodes: Dict[str, Node]) -> List[Combo]:
    combos: List[Combo] = []
    for item in _list(raw, "combos"):
        item = _rec(item)
        config = _rec(item.get("config"))
        legs: List[Leg] = []
        for index, step in enumerate(_list(item.get("models"))):
            if isinstance(step, str):
                step = {"kind": "model", "model": step}
            step = _rec(step)
            kind = str(step.get("kind") or "model")
            model = str(step.get("model") or "")
            token = str(step.get("providerId") or (model.split("/", 1)[0] if "/" in model else ""))
            pinned = step.get("connectionId") or None
            allowed = [str(c) for c in (step.get("allowedConnectionIds") or []) if c]
            leg = Leg(index=index, step_id=str(step.get("id") or f"step-{index}"), kind=kind,
                      model=model or str(step.get("modelPattern") or ""), provider_token=token,
                      pinned_connection_id=str(pinned) if pinned else None,
                      allowed_connection_ids=allowed, weight=float(step.get("weight") or 0),
                      combo_ref=str(step.get("comboName") or ""))
            if kind != "combo-ref":
                if leg.pinned_connection_id:
                    conn = connections.get(leg.pinned_connection_id)
                    leg.connections = [conn] if conn else []
                elif allowed:
                    leg.connections = [connections[c] for c in allowed if c in connections]
                else:
                    ids = set(_resolve_provider_ids(token, nodes))
                    leg.connections = [c for c in connections.values() if c.provider in ids]
            legs.append(leg)
        active = item.get("isActive", item.get("enabled", True))
        combos.append(Combo(
            id=str(item.get("id") or ""), name=str(item.get("name") or ""),
            strategy=str(item.get("strategy") or config.get("strategy") or DEFAULT_STRATEGY),
            active=active is not False, config=config,
            system_message=str(item.get("system_message") or ""),
            tool_filter_regex=str(item.get("tool_filter_regex") or ""), legs=legs))
    return combos


def effective_target_timeout(combo: Combo, combo_defaults: Dict[str, Any],
                             connection: Optional[Connection]) -> Dict[str, Any]:
    """Per-leg timeout the way OmniRoute resolves it (O-COMBO-4..6): combo.config.targetTimeoutMs,
    else settings.comboDefaults.targetTimeoutMs, else 120000; a non-positive value counts as unset;
    then capped by the pinned connection's providerSpecificData.timeoutMs, else FETCH_TIMEOUT_MS
    (600000 by default; the server's env override is not readable over the API)."""
    configured = _positive_int(combo.config.get("targetTimeoutMs"))
    source = "combo.config.targetTimeoutMs"
    if configured is None:
        configured = _positive_int(_rec(combo_defaults).get("targetTimeoutMs"))
        source = "settings.comboDefaults.targetTimeoutMs"
    if configured is None:
        configured, source = DEFAULT_COMBO_TARGET_TIMEOUT_MS, "OmniRoute default"
    if connection is not None and connection.timeout_ms:
        ceiling, ceiling_source = connection.timeout_ms, "connection.providerSpecificData.timeoutMs"
    else:
        ceiling, ceiling_source = DEFAULT_FETCH_TIMEOUT_MS, "FETCH_TIMEOUT_MS default (assumed)"
    return {"ms": min(configured, ceiling), "configured_ms": configured, "source": source,
            "ceiling_ms": ceiling, "ceiling_source": ceiling_source}


def combo_defaults_of(raw: Any) -> Dict[str, Any]:
    return _rec(_rec(raw).get("comboDefaults"))
