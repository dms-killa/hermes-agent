"""Inspection reports the OmniRoute MCP server does not provide (docs/DESIGN.md, section 3):
connections with base URL / max_concurrent, combos with full legs, call logs filtered by exact
combo name with a request-body signature, and per-combo health."""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional

from . import signatures
from .audit import combo_view
from .model import Combo, Connection

CallLogFetcher = Callable[[str], Any]


def connections_report(connections: Dict[str, Connection]) -> Dict[str, Any]:
    rows = [c.public() for c in connections.values()]
    return {"report": "omniroute_connections", "count": len(rows), "connections": rows,
            "cite": ["docs/CAPABILITIES.md#o-conn-1", "docs/CAPABILITIES.md#o-conn-2"]}


def combos_report(combos: List[Combo], combo_defaults: Dict[str, Any],
                  name: Optional[str] = None) -> Dict[str, Any]:
    selected = [c for c in combos if not name or c.name == name]
    return {"report": "omniroute_combos", "count": len(selected),
            "combos": [combo_view(c, combo_defaults) for c in selected],
            "not_found": bool(name) and not selected,
            "cite": ["docs/CAPABILITIES.md#o-combo-1", "docs/CAPABILITIES.md#o-combo-2"]}


_ROW_FIELDS = ("id", "timestamp", "method", "path", "status", "model", "requestedModel", "provider",
               "connectionId", "duration", "ttft", "comboName", "comboStepId", "requestType",
               "error", "hasRequestBody", "detailState", "correlationId")


def call_logs_report(rows: Any, *, combo_name: Optional[str], limit: int,
                     detail_fetcher: Optional[CallLogFetcher] = None,
                     extra_probe_prompts: Iterable[str] = ()) -> Dict[str, Any]:
    """Filter by exact ``comboName`` on the client: the server's ``combo`` filter is a presence
    check, not a name match (L5, O-LOGS-2). With ``detail_fetcher`` set, fetch each row's stored
    request body and attach a signature (L6, O-LOGS-3)."""
    items = rows if isinstance(rows, list) else []
    if combo_name:
        items = [r for r in items if isinstance(r, dict) and r.get("comboName") == combo_name]
    items = items[:limit]
    out: List[Dict[str, Any]] = []
    probe_count = real_count = unknown_count = 0
    for row in items:
        if not isinstance(row, dict):
            continue
        entry = {k: row.get(k) for k in _ROW_FIELDS if k in row}
        if detail_fetcher is not None:
            body = None
            if row.get("hasRequestBody") and row.get("id"):
                detail = detail_fetcher(str(row["id"]))
                body = detail.get("requestBody") if isinstance(detail, dict) else None
            sig = signatures.body_signature(body)
            probe = signatures.classify_probe(sig, extra_probe_prompts)
            entry["body_signature"] = signatures.public(sig)
            entry["probe"] = probe
            if sig is None:
                entry["traffic"] = "unknown"
                unknown_count += 1
            elif probe:
                entry["traffic"] = "probe"
                probe_count += 1
            else:
                entry["traffic"] = "real"
                real_count += 1
        out.append(entry)
    report: Dict[str, Any] = {
        "report": "omniroute_call_logs", "combo_name": combo_name, "count": len(out), "rows": out,
        "note": ("Call logs record proxied /v1 inference traffic only, not /api management reads. "
                 "The combo filter is applied here by exact comboName."),
        "cite": ["docs/CAPABILITIES.md#l5", "docs/CAPABILITIES.md#o-logs-1",
                 "docs/CAPABILITIES.md#o-logs-2"],
    }
    if detail_fetcher is not None:
        report["traffic"] = {"real": real_count, "probe": probe_count, "unknown": unknown_count}
        report["probe_signatures"] = signatures.builtin_probe_list()
        report["cite"] += ["docs/CAPABILITIES.md#l6", "docs/CAPABILITIES.md#o-logs-3",
                           "docs/CAPABILITIES.md#o-logs-4"]
    return report


def combo_health_report(combo: Optional[Combo], health: Any, metrics: Any, *,
                        range_: str) -> Dict[str, Any]:
    entry = None
    if combo is not None and isinstance(health, dict):
        for item in health.get("combos") or []:
            if isinstance(item, dict) and (item.get("comboId") == combo.id or item.get("comboName") == combo.name):
                entry = item
                break
    return {"report": "omniroute_combo_health", "combo": combo.name if combo else None,
            "found": combo is not None, "range": range_, "health": entry,
            "live_metrics": (metrics or {}).get("metrics") if isinstance(metrics, dict) else None,
            "note": ("Health and metrics include synthetic probe traffic; on a low-traffic combo "
                     "check omniroute_call_logs with include_body_signature before trusting them."),
            "cite": ["docs/CAPABILITIES.md#o-health-1", "docs/CAPABILITIES.md#o-health-2",
                     "docs/CAPABILITIES.md#l6"]}
