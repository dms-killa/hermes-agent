"""Connection settings: ``OMNIROUTE_BASE_URL`` and ``OMNIROUTE_API_KEY``.

These are the names OmniRoute's own MCP server reads (open-sse/mcp-server/server.ts), so one
``.env`` serves both. The key must carry the ``manage`` (or ``admin``) scope: management routes
reject a plain ``/v1`` proxy key with 403 (CAPABILITIES.md O-AUTH-2).

Secrets are read through Hermes' profile secret scope when it is importable (the same path
``agent/auxiliary_client.py::_scoped_key_env`` uses), else from the process environment. The key
is never logged, printed, put in a URL or included in ``repr``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlsplit, urlunsplit

ENV_BASE_URL = "OMNIROUTE_BASE_URL"
ENV_API_KEY = "OMNIROUTE_API_KEY"


class SettingsError(ValueError):
    """Missing or malformed configuration. Messages never contain the key."""


def _read_secret(name: str) -> str:
    try:
        from agent.secret_scope import UnscopedSecretError, get_secret
    except ImportError:  # outside Hermes (tests, CLI use)
        return (os.environ.get(name) or "").strip()
    try:
        return (get_secret(name) or "").strip()
    except UnscopedSecretError:
        return (os.environ.get(name) or "").strip()


def normalize_base_url(raw: str) -> str:
    """``scheme://host[:port][/prefix]`` without a trailing slash, query or fragment.

    A trailing ``/v1`` or ``/api/v1`` is stripped: management routes live at the gateway root
    (``/api/...``), while Hermes model settings usually point at ``<root>/v1``.
    """
    value = (raw or "").strip()
    if not value:
        raise SettingsError(f"{ENV_BASE_URL} is not set")
    parts = urlsplit(value)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise SettingsError(f"{ENV_BASE_URL} must be an http(s) URL with a host")
    if parts.username or parts.password:
        raise SettingsError(f"{ENV_BASE_URL} must not embed credentials")
    path = parts.path.rstrip("/")
    for suffix in ("/api/v1", "/v1"):
        if path.endswith(suffix):
            path = path[: -len(suffix)]
            break
    return urlunsplit((parts.scheme, parts.netloc, path, "", ""))


@dataclass(frozen=True)
class Settings:
    base_url: str
    api_key: str
    timeout_s: float = 20.0

    def __repr__(self) -> str:  # never leak the key through logs or tracebacks
        return f"Settings(base_url={self.base_url!r}, api_key=<redacted>, timeout_s={self.timeout_s})"

    __str__ = __repr__


def load_settings(base_url: Optional[str] = None, api_key: Optional[str] = None,
                  timeout_s: float = 20.0) -> Settings:
    raw_url = base_url if base_url is not None else _read_secret(ENV_BASE_URL)
    key = api_key if api_key is not None else _read_secret(ENV_API_KEY)
    url = normalize_base_url(raw_url)
    if not key:
        raise SettingsError(f"{ENV_API_KEY} is not set (needs an API key with 'manage' scope)")
    return Settings(base_url=url, api_key=key, timeout_s=timeout_s)


def is_configured() -> bool:
    """check_fn for tool availability: both variables present (values are not validated here)."""
    try:
        return bool(_read_secret(ENV_BASE_URL)) and bool(_read_secret(ENV_API_KEY))
    except Exception:  # a broken secret scope must hide the tools, not crash discovery
        return False
