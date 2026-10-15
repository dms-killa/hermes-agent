"""Model-facing JSON schemas. Read tools say "Read-only"; the two write tools say "WRITE"."""

from __future__ import annotations

TOOLSET = "omniroute"            # read tools (and the read-only combo planner)
WRITE_TOOLSET = "omniroute_write"  # apply / rollback: enable separately with `hermes tools`

_TIMEOUT = {"type": "integer", "minimum": 1,
            "description": "The caller's timeout for ONE request, in milliseconds (not a multi-call budget)."}
_CONCURRENCY = {"type": "integer", "minimum": 1,
                "description": "How many requests the caller sends to a combo at the same time."}
_COMBOS = {"type": "array", "items": {"type": "string"}, "maxItems": 50,
           "description": "Limit the per-combo sections to these combo names (default: all)."}

OMNIROUTE_AUDIT = {
    "name": "omniroute_audit",
    "description": (
        "Read-only audit of the OmniRoute gateway's connections and combos. Flags self-hosted "
        "backends registered as generic openai-compatible nodes (json_schema silently downgraded), "
        "lists each combo's strategy, legs and pinned connections with max_concurrent, and when you "
        "pass per_call_timeout_ms / concurrency compares them at matching scope with each leg's "
        "effective targetTimeoutMs and the connections' max_concurrent. Notes p2c combos (no "
        "in-flight balancing). Makes only allowlisted GET requests."),
    "parameters": {"type": "object", "additionalProperties": False, "properties": {
        "per_call_timeout_ms": _TIMEOUT, "concurrency": _CONCURRENCY, "combos": _COMBOS}},
}

HERMES_COMBO_CHECK = {
    "name": "hermes_combo_check",
    "description": (
        "Read-only cross-check between this Hermes profile and OmniRoute: for every main, "
        "auxiliary, delegation and fallback model setting whose base_url is the OmniRoute gateway, "
        "confirm a combo of that name exists, list its legs, and flag mismatches: structured-output "
        "tasks on legs that downgrade json_schema, tool-calling tasks on combos that filter tools or "
        "replace the system prompt, and per-call timeouts longer than a leg's timeout."),
    "parameters": {"type": "object", "additionalProperties": False, "properties": {}},
}

OMNIROUTE_CONNECTIONS = {
    "name": "omniroute_connections",
    "description": ("Read-only list of OmniRoute provider connections: provider id, resolved base "
                    "URL and where it came from, host class, max_concurrent, active flag. Never "
                    "returns keys or tokens."),
    "parameters": {"type": "object", "additionalProperties": False, "properties": {
        "provider": {"type": "string", "maxLength": 200, "description": "Only this provider id."}}},
}

OMNIROUTE_COMBOS = {
    "name": "omniroute_combos",
    "description": ("Read-only list of OmniRoute combos with strategy, every leg, pinned connection, "
                    "the connection's max_concurrent and the effective per-leg targetTimeoutMs."),
    "parameters": {"type": "object", "additionalProperties": False, "properties": {
        "name": {"type": "string", "maxLength": 100, "description": "Only the combo with this exact name."}}},
}

OMNIROUTE_CALL_LOGS = {
    "name": "omniroute_call_logs",
    "description": (
        "Read-only recent OmniRoute call logs (proxied /v1 inference only), filtered by exact combo "
        "name on the client. With include_body_signature=true, fetches each row's stored request "
        "body and reports its shape and whether it matches a known synthetic probe, so probe "
        "traffic can be told from real traffic. That detail fetch is opt-in: OmniRoute's detail "
        "route may mark a row whose stored artifact is missing."),
    "parameters": {"type": "object", "additionalProperties": False, "properties": {
        "combo_name": {"type": "string", "maxLength": 100},
        "status": {"type": "string", "enum": ["ok", "error"]},
        "limit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 20},
        "scan_limit": {"type": "integer", "minimum": 1, "maximum": 1000, "default": 200,
                       "description": "Rows fetched from OmniRoute before the client-side filter."},
        "include_body_signature": {"type": "boolean", "default": False},
        "probe_prompts": {"type": "array", "items": {"type": "string", "maxLength": 500}, "maxItems": 20,
                          "description": "Extra exact prompt texts your own health probes send."}}},
}

OMNIROUTE_COMBO_HEALTH = {
    "name": "omniroute_combo_health",
    "description": ("Read-only health for one OmniRoute combo: per-target success rate, latency and "
                    "quota from /api/usage/combo-health plus live in-memory combo metrics."),
    "parameters": {"type": "object", "additionalProperties": False, "required": ["combo_name"],
                   "properties": {
                       "combo_name": {"type": "string", "maxLength": 100},
                       "range": {"type": "string", "enum": ["1h", "24h", "7d", "30d"], "default": "24h"}}},
}

OMNIROUTE_COMBO_PLAN = {
    "name": "omniroute_combo_plan",
    "description": (
        "Read-only. Plan a change to ONE existing OmniRoute combo: computes the field-level diff the "
        "server will store, a freshness token, a reversibility verdict, findings about the proposed "
        "state (L1-L4, unpinned legs, effects on Hermes tasks) and what the server may still reject. "
        "Writes nothing to OmniRoute. `changes` holds top-level combo fields (models, strategy, config, "
        "isActive, isHidden, description, displayName, allowedProviders, allowedModelFamilies, "
        "system_message, tool_filter_regex, context_cache_protection, context_length, dimensions); "
        "`changes.config` is a key-by-key patch where null removes a key; null on a nullable "
        "top-level field resets it. Load skill omniroute:omniroute-routing before planning."),
    "parameters": {"type": "object", "additionalProperties": False, "required": ["combo", "changes"],
                   "properties": {
                       "combo": {"type": "string", "maxLength": 200, "description": "Combo id or exact name."},
                       "changes": {"type": "object"},
                       "caller_limits": {"type": "object", "additionalProperties": False, "properties": {
                           "per_call_timeout_ms": _TIMEOUT, "concurrency": _CONCURRENCY}},
                       "ttl_seconds": {"type": "integer", "minimum": 60, "maximum": 3600, "default": 900}}},
}

OMNIROUTE_COMBO_APPLY = {
    "name": "omniroute_combo_apply",
    "description": (
        "WRITE. Apply a plan from omniroute_combo_plan with one PUT to that combo. Refuses an unknown, "
        "expired, already-used or irreversible plan, and a plan whose combo changed since planning. "
        "Snapshots the combo first, verifies the stored result against the plan, and restores the "
        "snapshot automatically if verification fails. Returns a rollback_id."),
    "parameters": {"type": "object", "additionalProperties": False, "required": ["plan_id"],
                   "properties": {"plan_id": {"type": "string", "maxLength": 64}}},
}

OMNIROUTE_COMBO_ROLLBACK = {
    "name": "omniroute_combo_rollback",
    "description": (
        "WRITE. Restore a combo to the snapshot taken by omniroute_combo_apply, by sending the complete "
        "snapshot (with explicit resets for fields the apply added) and verifying the read-back. "
        "Refuses if the combo changed after the apply unless force=true; a forced rollback overwrites "
        "those changes and is reported as forced."),
    "parameters": {"type": "object", "additionalProperties": False, "required": ["rollback_id"],
                   "properties": {"rollback_id": {"type": "string", "maxLength": 64},
                                  "force": {"type": "boolean", "default": False}}},
}

READ_TOOLS = (OMNIROUTE_AUDIT, HERMES_COMBO_CHECK, OMNIROUTE_CONNECTIONS, OMNIROUTE_COMBOS,
              OMNIROUTE_CALL_LOGS, OMNIROUTE_COMBO_HEALTH, OMNIROUTE_COMBO_PLAN)
WRITE_TOOLS = (OMNIROUTE_COMBO_APPLY, OMNIROUTE_COMBO_ROLLBACK)
WRITE_TOOL_NAMES = frozenset(t["name"] for t in WRITE_TOOLS)
ALL = READ_TOOLS + WRITE_TOOLS
