"""The ONLY OmniRoute endpoints this plugin may call. Hard-coded, one module.

Every endpoint is GET-only except one: ``/api/combos/{id}``, which also allows ``PUT`` (the
combo write route, O-WRITE-1). It is the plugin's single write route, used only by
``omniroute_combo_apply`` and ``omniroute_combo_rollback``.

Every entry was read in OmniRoute source at the SHA recorded in docs/CAPABILITIES.md and checked
for side effects: no DB writes, no outbound calls, no spawns on the GET path. GET alone is not
proof of safety (``/api/monitoring/health?deep=1`` makes an outbound probe; the call-log detail
route can rewrite a row), so each entry also fixes which query parameters may be sent.

``client.OmniRouteClient`` refuses any name not in ``ALLOWLIST`` and any query key not listed
for that entry, and any method other than the entry's own. The strict test mock fails on any
request outside this table (method included).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Mapping, Optional, Tuple
from urllib.parse import quote, urlencode

METHOD = "GET"
WRITE_METHOD = "PUT"


@dataclass(frozen=True)
class Endpoint:
    name: str
    template: str                       # "/api/usage/call-logs/{id}"
    params: FrozenSet[str] = frozenset()
    cite: str = ""                      # CAPABILITIES.md row id
    side_effect: str = ""               # "" = none found in source
    opt_in: bool = False                # only called when a tool argument asks for it
    path_params: Tuple[str, ...] = field(default=())
    methods: FrozenSet[str] = frozenset({METHOD})
    reserved_values: FrozenSet[str] = frozenset()   # path-param values that name a sibling static route


_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,200}$")

ALLOWLIST: Dict[str, Endpoint] = {e.name: e for e in (
    Endpoint("connections", "/api/providers", frozenset({"provider", "limit", "offset"}), "O-CONN-1"),
    Endpoint("provider_nodes", "/api/provider-nodes", frozenset({"limit", "offset"}), "O-CONN-3"),
    Endpoint("combos", "/api/combos", frozenset({"limit", "offset"}), "O-COMBO-1"),
    Endpoint("combo_defaults", "/api/settings/combo-defaults", frozenset(), "O-COMBO-5"),
    Endpoint("combo_metrics", "/api/combos/metrics", frozenset({"combo"}), "O-HEALTH-1"),
    Endpoint("combo_health", "/api/usage/combo-health", frozenset({"range", "comboId"}), "O-HEALTH-2"),
    Endpoint("call_logs", "/api/usage/call-logs",
             frozenset({"status", "model", "provider", "combo", "limit", "offset", "excludeTests"}),
             "O-LOGS-1"),
    Endpoint("call_log_detail", "/api/usage/call-logs/{id}", frozenset(), "O-LOGS-3",
             side_effect=("when the stored artifact is missing or corrupt, the handler clears the "
                          "artifact pointer / marks the row corrupt (bookkeeping on call_logs only)"),
             opt_in=True, path_params=("id",)),
    # The single write route. GET on it is read-only (route.ts:38-55); PUT merges into one stored
    # combo (O-WRITE-1..6). PATCH is the same handler upstream and is never sent.
    Endpoint("combo", "/api/combos/{id}", frozenset(), "O-WRITE-1", path_params=("id",),
             methods=frozenset({METHOD, WRITE_METHOD}),
             reserved_values=frozenset({"auto", "builder", "duplicate", "metrics", "reorder", "test"}),
             side_effect="PUT only: stores the merged combo, may clear session model pins, backs up "
                         "the DB and triggers cloud sync (O-WRITE-6)"),
)}


class EndpointNotAllowed(PermissionError):
    pass


def build_path(name: str, path_params: Optional[Mapping[str, str]] = None,
               query: Optional[Mapping[str, object]] = None, method: str = METHOD) -> str:
    """Return ``path?query`` for an allowlisted endpoint and method, or raise ``EndpointNotAllowed``."""
    endpoint = ALLOWLIST.get(name)
    if endpoint is None:
        raise EndpointNotAllowed(f"endpoint {name!r} is not in the allowlist")
    if method not in endpoint.methods:
        raise EndpointNotAllowed(f"method {method} is not allowed for {name!r}")
    path_params = dict(path_params or {})
    if set(path_params) != set(endpoint.path_params):
        raise EndpointNotAllowed(f"endpoint {name!r} takes path params {endpoint.path_params}")
    path = endpoint.template
    for key, value in path_params.items():
        text = str(value)
        if not _ID_RE.match(text):
            raise EndpointNotAllowed(f"path param {key!r} has unexpected characters")
        if text in endpoint.reserved_values:
            raise EndpointNotAllowed(f"path param {key!r} names a different route ({text!r})")
        path = path.replace("{" + key + "}", quote(text, safe=""))
    clean = {k: v for k, v in (query or {}).items() if v is not None and v != ""}
    extra = set(clean) - endpoint.params
    if extra:
        raise EndpointNotAllowed(f"query params {sorted(extra)} not allowed for {name!r}")
    if not clean:
        return path
    return path + "?" + urlencode(sorted((k, str(v)) for k, v in clean.items()))


def allowed_paths() -> Tuple[str, ...]:
    """Path templates, for tests and the mock server."""
    return tuple(e.template for e in ALLOWLIST.values())
