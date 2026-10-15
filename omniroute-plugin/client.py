"""OmniRoute management client: stdlib ``urllib``, allowlisted endpoints and methods only.

GET for every read; PUT only where the allowlist entry permits it (the combo route).

The key travels in the ``Authorization: Bearer`` header only. OmniRoute's management auth never
accepts a URL-borne credential (CAPABILITIES.md O-AUTH-1), and neither does this client.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Callable, Mapping, Optional

from .endpoints import METHOD, WRITE_METHOD, build_path
from .settings import Settings

USER_AGENT = "hermes-omniroute-plugin/0.2"
_MAX_BODY_BYTES = 8 * 1024 * 1024


class OmniRouteError(RuntimeError):
    """HTTP or transport failure. ``str()`` never contains the API key or a response body dump."""

    def __init__(self, message: str, status: Optional[int] = None, endpoint: str = ""):
        super().__init__(message)
        self.status = status
        self.endpoint = endpoint
        self.server_error: Optional[dict] = None


def _hint(status: int) -> str:
    if status == 401:
        return "authentication required or key invalid"
    if status == 403:
        return "key rejected; management routes need an API key with 'manage' scope"
    if status == 404:
        return "not found"
    return "request failed"


class OmniRouteClient:
    def __init__(self, settings: Settings,
                 opener: Optional[Callable[..., Any]] = None):
        self._settings = settings
        self._open = opener or urllib.request.urlopen

    def __repr__(self) -> str:
        return f"OmniRouteClient({self._settings!r})"

    @property
    def base_url(self) -> str:
        return self._settings.base_url

    def get(self, endpoint: str, *, path_params: Optional[Mapping[str, str]] = None,
            query: Optional[Mapping[str, object]] = None) -> Any:
        return self._request(METHOD, endpoint, path_params=path_params, query=query)

    def put(self, endpoint: str, body: Mapping[str, Any], *,
            path_params: Optional[Mapping[str, str]] = None) -> Any:
        """The single write verb, for allowlist entries that permit it (only ``combo``)."""
        return self._request(WRITE_METHOD, endpoint, path_params=path_params, body=body)

    def _request(self, method: str, endpoint: str, *, path_params: Optional[Mapping[str, str]] = None,
                 query: Optional[Mapping[str, object]] = None,
                 body: Optional[Mapping[str, Any]] = None) -> Any:
        path = build_path(endpoint, path_params, query, method=method)   # raises if not allowlisted
        headers = {"Authorization": f"Bearer {self._settings.api_key}", "Accept": "application/json",
                   "User-Agent": USER_AGENT}
        data = None
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(self._settings.base_url + path, method=method, headers=headers,
                                         data=data)
        bare = path.split("?")[0]
        try:
            with self._open(request, timeout=self._settings.timeout_s) as response:
                raw = response.read(_MAX_BODY_BYTES + 1)
        except urllib.error.HTTPError as exc:
            err = OmniRouteError(f"{method} {bare} -> HTTP {exc.code}: {_hint(exc.code)}",
                                 status=exc.code, endpoint=endpoint)
            err.server_error = _server_error(exc)
            raise err from None
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            reason = getattr(exc, "reason", exc)
            raise OmniRouteError(f"{method} {bare} failed: {type(reason).__name__}",
                                 endpoint=endpoint) from None
        if len(raw) > _MAX_BODY_BYTES:
            raise OmniRouteError(f"{method} {bare}: response exceeds {_MAX_BODY_BYTES} bytes",
                                 endpoint=endpoint)
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise OmniRouteError(f"{method} {bare}: response is not JSON", endpoint=endpoint) from None


def _server_error(exc: urllib.error.HTTPError) -> Optional[dict]:
    """The structured error OmniRoute returns for combo writes, ``{code, details}`` only
    (src/lib/api/comboErrorResponse.ts). Never the raw body."""
    try:
        payload = json.loads(exc.read(65536).decode("utf-8"))
    except Exception:
        return None
    error = payload.get("error") if isinstance(payload, dict) else None
    if not isinstance(error, dict):
        return None
    details = error.get("details") if isinstance(error.get("details"), dict) else {}
    keep = {k: details[k] for k in ("reason", "firstField", "firstMessage", "field", "source") if k in details}
    return {"code": error.get("code"), "details": keep}
