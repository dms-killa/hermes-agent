"""Tool handlers. Each returns a JSON string and never raises into Hermes' tool loop; errors are
``{"error": ...}`` with messages that never contain the API key."""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, Optional

from . import audit, combo_check, inspect_tools
from .combo_write import ComboWriter, PlanRefused
from .client import OmniRouteClient, OmniRouteError
from .endpoints import EndpointNotAllowed
from .settings import SettingsError, load_settings
from .write_store import MemoryState, WriteStore

logger = logging.getLogger(__name__)

ClientFactory = Callable[[], OmniRouteClient]
ConfigLoader = Callable[[], Dict[str, Any]]


def default_client_factory() -> OmniRouteClient:
    return OmniRouteClient(load_settings())


def default_config_loader() -> Dict[str, Any]:
    from hermes_cli.config import load_config_readonly   # inside Hermes only
    config = load_config_readonly()
    return config if isinstance(config, dict) else {}


def _profile_name() -> str:
    try:
        from hermes_constants import get_hermes_home
        return get_hermes_home().name
    except Exception:
        return ""


def _int(args: Dict[str, Any], key: str, default: Optional[int] = None, *, lo: int = 1,
         hi: Optional[int] = None) -> Optional[int]:
    value = args.get(key, default)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or int(value) != value:
        raise ValueError(f"{key} must be an integer")
    value = int(value)
    if value < lo or (hi is not None and value > hi):
        raise ValueError(f"{key} must be between {lo} and {hi if hi is not None else 'any'}")
    return value


def _str(args: Dict[str, Any], key: str, limit: int = 200) -> Optional[str]:
    value = args.get(key)
    if value is None or value == "":
        return None
    if not isinstance(value, str) or len(value) > limit:
        raise ValueError(f"{key} must be a string of at most {limit} characters")
    return value.strip()


def _safe(fn: Callable[[Dict[str, Any]], Dict[str, Any]], name: str) -> Callable[..., str]:
    def handler(args: Optional[Dict[str, Any]] = None, **_kwargs: Any) -> str:
        args = args if isinstance(args, dict) else {}
        try:
            return json.dumps(fn(args), default=str)
        except (ValueError, SettingsError, EndpointNotAllowed) as exc:
            return json.dumps({"error": f"{name}: {exc}"})
        except OmniRouteError as exc:
            return json.dumps({"error": f"{name}: {exc}", "status": exc.status})
        except Exception as exc:  # never raise into the tool loop; never echo the exception text
            logger.warning("%s failed: %s", name, type(exc).__name__)
            return json.dumps({"error": f"{name}: internal error ({type(exc).__name__})"})
    handler.__name__ = name
    return handler


def _snapshot(client: OmniRouteClient) -> Dict[str, Any]:
    return audit.build_snapshot(client.get("connections"), client.get("provider_nodes"),
                                client.get("combos"), client.get("combo_defaults"))


def make_handlers(client_factory: ClientFactory = default_client_factory,
                  config_loader: ConfigLoader = default_config_loader,
                  store: Optional[WriteStore] = None) -> Dict[str, Callable[..., str]]:
    store = store or WriteStore(MemoryState())

    def omniroute_audit(args: Dict[str, Any]) -> Dict[str, Any]:
        per_call = _int(args, "per_call_timeout_ms")
        concurrency = _int(args, "concurrency")
        names = args.get("combos")
        if names is not None and (not isinstance(names, list) or not all(isinstance(n, str) for n in names)):
            raise ValueError("combos must be a list of combo names")
        client = client_factory()
        return audit.run_audit(_snapshot(client), per_call_timeout_ms=per_call,
                               concurrency=concurrency, combo_names=names or None,
                               gateway=client.base_url)

    def hermes_combo_check(args: Dict[str, Any]) -> Dict[str, Any]:
        client = client_factory()
        snap = _snapshot(client)
        return combo_check.run_combo_check(config_loader(), client.base_url, snap["combos"],
                                           snap["combo_defaults"], profile=_profile_name())

    def omniroute_connections(args: Dict[str, Any]) -> Dict[str, Any]:
        provider = _str(args, "provider")
        client = client_factory()
        snap = audit.build_snapshot(client.get("connections", query={"provider": provider}),
                                    client.get("provider_nodes"), [], {})
        return inspect_tools.connections_report(snap["connections"])

    def omniroute_combos(args: Dict[str, Any]) -> Dict[str, Any]:
        snap = _snapshot(client_factory())
        return inspect_tools.combos_report(snap["combos"], snap["combo_defaults"], _str(args, "name", 100))

    def omniroute_call_logs(args: Dict[str, Any]) -> Dict[str, Any]:
        combo_name = _str(args, "combo_name", 100)
        limit = _int(args, "limit", 20, hi=100)
        scan = _int(args, "scan_limit", 200, hi=1000)
        status = _str(args, "status")
        if status not in (None, "ok", "error"):
            raise ValueError("status must be 'ok' or 'error'")
        prompts = args.get("probe_prompts") or []
        if not isinstance(prompts, list) or not all(isinstance(p, str) for p in prompts):
            raise ValueError("probe_prompts must be a list of strings")
        client = client_factory()
        rows = client.get("call_logs", query={"combo": "1" if combo_name else None, "status": status,
                                              "limit": scan})
        fetcher = None
        if args.get("include_body_signature") is True:
            fetcher = lambda log_id: client.get("call_log_detail", path_params={"id": log_id})
        return inspect_tools.call_logs_report(rows, combo_name=combo_name, limit=limit,
                                              detail_fetcher=fetcher, extra_probe_prompts=prompts)

    def omniroute_combo_health(args: Dict[str, Any]) -> Dict[str, Any]:
        name = _str(args, "combo_name", 100)
        if not name:
            raise ValueError("combo_name is required")
        range_ = _str(args, "range") or "24h"
        if range_ not in ("1h", "24h", "7d", "30d"):
            raise ValueError("range must be one of 1h, 24h, 7d, 30d")
        client = client_factory()
        combos = audit.build_snapshot([], [], client.get("combos"), {})["combos"]
        combo = next((c for c in combos if c.name == name), None)
        if combo is None:
            return inspect_tools.combo_health_report(None, None, None, range_=range_)
        health = client.get("combo_health", query={"range": range_, "comboId": combo.id or None})
        metrics = client.get("combo_metrics", query={"combo": combo.name})
        return inspect_tools.combo_health_report(combo, health, metrics, range_=range_)

    def writer() -> ComboWriter:
        return ComboWriter(client_factory(), store, config_loader)

    def omniroute_combo_plan(args: Dict[str, Any]) -> Dict[str, Any]:
        combo = _str(args, "combo", 200)
        if not combo:
            raise ValueError("combo is required (id or exact name)")
        changes = args.get("changes")
        limits = args.get("caller_limits") or {}
        if not isinstance(limits, dict):
            raise ValueError("caller_limits must be an object")
        _int(limits, "per_call_timeout_ms")
        _int(limits, "concurrency")
        ttl = _int(args, "ttl_seconds", 900, lo=60, hi=3600)
        try:
            return writer().plan(combo, changes, limits, ttl)
        except PlanRefused as exc:
            return {"refused": True, "code": "plan_refused", "message": str(exc)}

    def omniroute_combo_apply(args: Dict[str, Any]) -> Dict[str, Any]:
        plan_id = _str(args, "plan_id", 64)
        if not plan_id:
            raise ValueError("plan_id is required")
        return writer().apply(plan_id)

    def omniroute_combo_rollback(args: Dict[str, Any]) -> Dict[str, Any]:
        rollback_id = _str(args, "rollback_id", 64)
        if not rollback_id:
            raise ValueError("rollback_id is required")
        force = args.get("force", False)
        if not isinstance(force, bool):
            raise ValueError("force must be a boolean")
        return writer().rollback(rollback_id, force=force)

    fns = {"omniroute_audit": omniroute_audit, "hermes_combo_check": hermes_combo_check,
           "omniroute_combo_plan": omniroute_combo_plan, "omniroute_combo_apply": omniroute_combo_apply,
           "omniroute_combo_rollback": omniroute_combo_rollback,
           "omniroute_connections": omniroute_connections, "omniroute_combos": omniroute_combos,
           "omniroute_call_logs": omniroute_call_logs, "omniroute_combo_health": omniroute_combo_health}
    return {name: _safe(fn, name) for name, fn in fns.items()}
