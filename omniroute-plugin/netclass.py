"""Classify a base URL's host as self-hosted (loopback, private range, link-local, or a
``.lan`` / ``.local`` / ``.localhost`` name). Pure, no DNS lookups: a hostname that is not an IP
literal is classified by its suffix only."""

from __future__ import annotations

import ipaddress
from typing import Optional
from urllib.parse import urlsplit

SELF_HOSTED_SUFFIXES = (".lan", ".local", ".localhost")
LOOPBACK_NAMES = ("localhost",)


def host_of(url: Optional[str]) -> str:
    if not url or not isinstance(url, str):
        return ""
    try:
        return (urlsplit(url.strip()).hostname or "").lower()
    except ValueError:
        return ""


def classify_host(host: str) -> Optional[str]:
    """Return ``"loopback"``, ``"private"``, ``"link-local"``, ``"local-name"`` or ``None``."""
    host = (host or "").strip().lower().rstrip(".")
    if not host:
        return None
    if host in LOOPBACK_NAMES:
        return "loopback"
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return "local-name" if host.endswith(SELF_HOSTED_SUFFIXES) else None
    if getattr(ip, "ipv4_mapped", None):
        ip = ip.ipv4_mapped
    if ip.is_loopback:
        return "loopback"
    if ip.is_link_local:
        return "link-local"
    if ip.is_private:
        return "private"
    return None


def classify_url(url: Optional[str]) -> Optional[str]:
    return classify_host(host_of(url))
