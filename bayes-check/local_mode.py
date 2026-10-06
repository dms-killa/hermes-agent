"""Local-vs-cloud classification from ``pre_api_request``'s ``provider`` + ``base_url``."""

from __future__ import annotations

import ipaddress
from typing import Optional
from urllib.parse import urlparse

AUTO, FORCE_LOCAL, FORCE_CLOUD = "auto", "force_local", "force_cloud"
LOCAL_MODES = (AUTO, FORCE_LOCAL, FORCE_CLOUD)

LOCAL_PROVIDERS = frozenset({
    "ollama", "llama.cpp", "llamacpp", "llama-cpp", "lmstudio", "lm-studio", "lm_studio",
    "vllm", "custom_openai",
})
_LOCAL_HOSTNAMES = frozenset({"localhost", "0.0.0.0", "host.docker.internal"})


def _host_is_local(host: str) -> bool:
    host = host.strip("[]").lower()
    if not host:
        return False
    if host in _LOCAL_HOSTNAMES or host.endswith(".localhost"):
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    return ip.is_loopback or ip.is_unspecified


def base_url_is_local(base_url: str) -> bool:
    raw = (base_url or "").strip()
    if not raw:
        return False
    lowered = raw.lower()
    if lowered.startswith(("unix:", "http+unix:", "unix+http:")) or lowered.startswith("/"):
        return True
    parsed = urlparse(raw if "://" in raw else f"http://{raw}")
    return _host_is_local(parsed.hostname or "")


def provider_is_local(provider: str) -> bool:
    return (provider or "").strip().lower() in LOCAL_PROVIDERS


def detect_local(provider: str, base_url: str) -> bool:
    return provider_is_local(provider) or base_url_is_local(base_url)


def resolve_local(setting: str, detected: Optional[bool]) -> bool:
    """Apply the tri-state override; ``auto`` with nothing observed yet is conservative-cloud."""
    if setting == FORCE_LOCAL:
        return True
    if setting == FORCE_CLOUD:
        return False
    return bool(detected)
