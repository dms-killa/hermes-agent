"""A fixture-backed, strict stand-in for OmniRoute's management API, with a stateful combo store.

Strict: any request outside the plugin's allowlist (path, query key, or method) is recorded as a
violation and answered with an error. The ``omniroute`` pytest fixture fails the test on violations
it did not consume via ``take_violations()``. The only accepted non-GET is ``PUT /api/combos/{id}``
(and ``PATCH``, its upstream alias). Every connection/provider route is GET-only, as are all others.

Simulated server behaviour for ``PUT /api/combos/{id}``, mirroring OmniRoute at the SHA in
docs/CAPABILITIES.md (rows O-WRITE-*):

- SIMULATED: update-schema key stripping (unknown top-level keys dropped), nullability (null on a
  non-nullable key -> 400 COMBO_002), "no valid fields" -> 400, the config zod transform
  (zero-latency flag promotion), ``models`` non-empty with only
  string / model-step / combo-ref entries (a stored provider-wildcard object -> 400), 404 for an
  unknown id, 409 COMBO_006 for the quota-share prefix (or ids in ``force_409``), dead config keys
  stripped, models normalized, nested-combo cycle/depth -> 400 COMBO_005, the merge
  ``{...current, ...body}``, keys sent as null deleted, the whole record re-normalized
  (version 2), ``updatedAt`` rewritten, the stored combo returned.
- NOT SIMULATED: zod coercion/ranges inside ``config``, composite tiers, model-family invariants,
  cloud sync, DB backup, session-pin clearing.

The normalization reuses the plugin's own port (``hermes_omniroute.normalize``). These tests
therefore prove the write plumbing, verification and restore logic, not the port's fidelity to
OmniRoute. Fidelity is covered by ``test_normalize.py``, whose expected values were derived by hand
from ``src/lib/combos/steps.ts``.
"""

from __future__ import annotations

import copy
import itertools
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import parse_qs, urlsplit

FIXTURES = Path(__file__).resolve().parent / "fixtures"
TEST_KEY = "test-placeholder-manage-key"     # placeholder, not a real key
_TS = itertools.count(1)


def _route_regexes(endpoints_mod):
    routes = []
    for ep in endpoints_mod.ALLOWLIST.values():
        pattern = "^" + re.sub(r"\\\{[a-z_]+\\\}", "([^/]+)", re.escape(ep.template)) + "$"
        routes.append((re.compile(pattern), ep))
    return routes


class ServerError(Exception):
    def __init__(self, status: int, code: str, details: Optional[dict] = None):
        super().__init__(code)
        self.status, self.code, self.details = status, code, details or {}

    def body(self) -> dict:
        return {"error": {"code": self.code, "message": self.code, "category": "combo",
                          "details": self.details}}


class ComboStore:
    """The simulated server-side combo table and the PUT handler logic."""

    def __init__(self, norm, combos: List[Dict[str, Any]]):
        self.norm = norm
        names = [c["name"] for c in combos]
        self.combos: Dict[str, Dict[str, Any]] = {}
        for i, combo in enumerate(combos):
            stored = norm.normalize_combo_record(dict(combo), names)
            stored.setdefault("sortOrder", i + 1)
            stored.setdefault("createdAt", "2026-01-01T00:00:00.000Z")
            stored.setdefault("updatedAt", "2026-01-01T00:00:00.000Z")
            stored.pop("computed_context_length", None)
            self.combos[stored["id"]] = stored
        self.force_409: set = set()
        self.after_store: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None  # server-side surprise
        self.lock = threading.Lock()

    def names(self) -> List[str]:
        return [c["name"] for c in self.combos.values()]

    def list(self) -> Dict[str, Any]:
        return {"combos": [copy.deepcopy(c) for c in self.combos.values()], "total": len(self.combos)}

    def get(self, combo_id: str) -> Optional[Dict[str, Any]]:
        combo = self.combos.get(combo_id)
        return copy.deepcopy(combo) if combo else None

    def _validate(self, raw: Any) -> Dict[str, Any]:
        norm = self.norm
        if not isinstance(raw, dict):
            raise ServerError(400, "COMBO_002", {"firstField": "body"})
        body = {k: v for k, v in raw.items() if k in norm.WRITABLE_KEYS}        # zod strips the rest
        if not body:
            raise ServerError(400, "COMBO_002", {"firstMessage": "No valid fields to update"})
        for key, value in body.items():
            if value is None and key not in norm.NULLABLE_KEYS:
                raise ServerError(400, "COMBO_002", {"firstField": key, "firstMessage": "not nullable"})
        if "models" in body:
            models = body["models"]
            if not isinstance(models, list) or not models:
                raise ServerError(400, "COMBO_002", {"firstField": "models"})
            for i, entry in enumerate(models):
                if isinstance(entry, str):
                    continue
                kind = entry.get("kind") if isinstance(entry, dict) else "?"
                ok = isinstance(entry, dict) and (
                    (kind in (None, "model") and isinstance(entry.get("model"), str)) or
                    (kind == "combo-ref" and isinstance(entry.get("comboName"), str)))
                weight = entry.get("weight", 0) if isinstance(entry, dict) else 0
                if not ok or isinstance(weight, bool) or not isinstance(weight, (int, float)) or not 0 <= weight <= 100:
                    raise ServerError(400, "COMBO_002", {"firstField": f"models.{i}"})
        if "config" in body and not isinstance(body["config"], dict):
            raise ServerError(400, "COMBO_002", {"firstField": "config"})
        return body

    def _dag(self, name: str, graph: Dict[str, Dict[str, Any]], seen: tuple, depth: int, max_depth: int):
        if depth > max_depth:
            raise ServerError(400, "COMBO_005", {"comboName": name, "reason": "max-depth-exceeded"})
        if name in seen:
            raise ServerError(400, "COMBO_005", {"comboName": name, "reason": "cycle-detected"})
        for step in graph.get(name, {}).get("models") or []:
            ref = step.get("comboName") if isinstance(step, dict) and step.get("kind") == "combo-ref" else None
            if ref and ref in graph:
                self._dag(ref, graph, seen + (name,), depth + 1, max_depth)

    def put(self, combo_id: str, raw: Any) -> Dict[str, Any]:
        norm = self.norm
        with self.lock:
            body = self._validate(raw)
            current = self.combos.get(combo_id)
            if current is None:
                raise ServerError(404, "COMBO_007", {"id": combo_id})
            if current["name"].startswith(norm.QUOTA_MODEL_PREFIX) or combo_id in self.force_409:
                raise ServerError(409, "COMBO_006", {"name": current["name"], "source": "quota-share"})
            names = self.names()
            server_body = dict(body)
            if isinstance(server_body.get("config"), dict):
                server_body["config"] = norm.server_config(server_body["config"])
            if server_body.get("models"):
                server_body["models"] = norm.normalize_combo_models(
                    server_body["models"], combo_name=current["name"], names=names)
                graph = {c["name"]: c for c in self.combos.values()}
                graph[current["name"]] = {**current, **server_body}
                max_depth = norm.clamp_combo_depth(
                    ({**current, **server_body}.get("config") or {}).get("maxComboDepth"))
                self._dag(current["name"], graph, (), 0, max_depth)
            merged = {**current, **server_body, "sortOrder": current.get("sortOrder"),
                      "updatedAt": f"2026-01-01T00:00:{next(_TS):06d}.000Z"}
            for key, value in body.items():
                if value is None:
                    merged.pop(key, None)
            stored = norm.normalize_combo_record(merged, names)
            if self.after_store is not None:
                stored = self.after_store(copy.deepcopy(stored))
            self.combos[combo_id] = stored
            return copy.deepcopy(stored)


class MockOmniRoute:
    def __init__(self, endpoints_mod, overrides: Optional[Dict[str, object]] = None, norm=None):
        self._routes = _route_regexes(endpoints_mod)
        self.overrides = dict(overrides or {})      # endpoint name -> JSON body
        self.requests: List[Dict[str, object]] = []
        self.violations: List[str] = []
        self._lock = threading.Lock()
        if norm is None:
            import hermes_omniroute.normalize as norm   # the plugin's port; see module docstring
        self.store = ComboStore(norm, json.loads((FIXTURES / "combos.json").read_text())["combos"])
        mock = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # silence
                pass

            def _reply(self, status, body):
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _handle(self):
                with mock._lock:
                    mock.requests.append({"method": self.command, "path": self.path,
                                          "auth": self.headers.get("Authorization")})
                parts = urlsplit(self.path)
                for regex, ep in mock._routes:
                    match = regex.match(parts.path)
                    if match:
                        break
                else:
                    mock._violate(f"path outside allowlist: {self.command} {self.path}")
                    return self._reply(404, {"error": "not allowlisted"})
                writable = ep.name == "combo" and self.command in ("PUT", "PATCH")
                if self.command != "GET" and not writable:
                    mock._violate(f"non-GET request: {self.command} {self.path}")
                    return self._reply(405, {"error": "method not allowed"})
                query = parse_qs(parts.query)
                extra = set(query) - set(ep.params)
                if extra:
                    mock._violate(f"query keys outside allowlist: {sorted(extra)} on {parts.path}")
                    return self._reply(400, {"error": "bad query"})
                if self.headers.get("Authorization") != f"Bearer {TEST_KEY}":
                    return self._reply(401, {"error": {"message": "Authentication required"}})
                if "api_key" in parts.query.lower() or TEST_KEY in self.path:
                    mock._violate("credential in URL")
                if writable:
                    length = int(self.headers.get("Content-Length") or 0)
                    try:
                        raw = json.loads(self.rfile.read(length) or b"null")
                    except json.JSONDecodeError:
                        return self._reply(400, ServerError(400, "COMBO_001").body())
                    try:
                        return self._reply(200, mock.store.put(match.group(1), raw))
                    except ServerError as exc:
                        return self._reply(exc.status, exc.body())
                body = mock.body_for(ep.name, match.groups(), query)
                if body is None:
                    return self._reply(404, {"error": "not found"})
                return self._reply(200, body)

            do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = do_HEAD = _handle

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def _violate(self, text: str) -> None:
        with self._lock:
            self.violations.append(text)

    def take_violations(self) -> List[str]:
        with self._lock:
            out, self.violations = self.violations, []
        return out

    def writes(self) -> List[Dict[str, object]]:
        return [r for r in self.requests if r["method"] != "GET"]

    def body_for(self, name, groups, query):
        if name in self.overrides:
            return self.overrides[name]
        if name == "combos":
            return self.store.list()
        if name == "combo":
            return self.store.get(groups[0])
        if name == "call_log_detail":
            path = FIXTURES / f"call_log_detail_{groups[0]}.json"
            return json.loads(path.read_text()) if path.exists() else None
        if name == "call_logs":
            rows = json.loads((FIXTURES / "call_logs.json").read_text())
            if query.get("combo"):       # OmniRoute semantics: presence check, not a name match
                rows = [r for r in rows if r.get("comboName") is not None]
            return rows
        files = {"connections": "connections.json", "provider_nodes": "provider_nodes.json",
                 "combo_defaults": "combo_defaults.json", "combo_metrics": "combo_metrics.json",
                 "combo_health": "combo_health.json"}
        return json.loads((FIXTURES / files[name]).read_text())

    @property
    def url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._server.shutdown()
        self._server.server_close()
