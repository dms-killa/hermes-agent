"""Allowlist, GET-only client, and the strict mock's rejection of anything else."""

from __future__ import annotations

import urllib.error
import urllib.request

import pytest

from hermes_omniroute import endpoints
from hermes_omniroute.client import OmniRouteClient, OmniRouteError
from hermes_omniroute.settings import Settings, SettingsError, load_settings, normalize_base_url
from mock_server import TEST_KEY


def test_allowlist_is_small_and_get_only_except_the_combo_route():
    assert endpoints.METHOD == "GET" and endpoints.WRITE_METHOD == "PUT"
    assert set(endpoints.allowed_paths()) == {
        "/api/providers", "/api/provider-nodes", "/api/combos", "/api/settings/combo-defaults",
        "/api/combos/metrics", "/api/usage/combo-health", "/api/usage/call-logs",
        "/api/usage/call-logs/{id}", "/api/combos/{id}"}
    writable = {e.name: e.methods for e in endpoints.ALLOWLIST.values() if e.methods != {"GET"}}
    assert writable == {"combo": frozenset({"GET", "PUT"})}          # one write route, PUT only
    flagged = sorted(e.name for e in endpoints.ALLOWLIST.values() if e.side_effect)
    assert flagged == ["call_log_detail", "combo"] and endpoints.ALLOWLIST["call_log_detail"].opt_in
    assert all(e.cite for e in endpoints.ALLOWLIST.values())


def test_write_method_refused_everywhere_else():
    for name in ("connections", "provider_nodes", "combos", "combo_defaults", "combo_metrics"):
        with pytest.raises(endpoints.EndpointNotAllowed):
            endpoints.build_path(name, method="PUT")
    with pytest.raises(endpoints.EndpointNotAllowed):
        endpoints.build_path("combo", {"id": "x"}, method="DELETE")
    with pytest.raises(endpoints.EndpointNotAllowed):
        endpoints.build_path("combo", {"id": "metrics"}, method="PUT")   # a sibling static route


@pytest.mark.parametrize("name,kwargs", [
    ("settings", {}),                                               # not allowlisted
    ("connections", {"query": {"reveal": "1"}}),                    # query key not allowed
    ("call_log_detail", {}),                                        # missing path param
    ("call_log_detail", {"path_params": {"id": "../combos"}}),      # traversal characters
    ("combos", {"path_params": {"id": "x"}}),                       # unexpected path param
])
def test_build_path_refuses(name, kwargs):
    with pytest.raises(endpoints.EndpointNotAllowed):
        endpoints.build_path(name, kwargs.get("path_params"), kwargs.get("query"))


def test_build_path_encodes_and_sorts_query():
    assert endpoints.build_path("call_logs", query={"limit": 5, "combo": "1", "status": None}) == \
        "/api/usage/call-logs?combo=1&limit=5"
    assert endpoints.build_path("call_log_detail", {"id": "log-0001"}) == "/api/usage/call-logs/log-0001"


def test_client_sends_get_with_bearer_header_only(client, omniroute):
    data = client.get("combos")
    assert data["total"] == 5
    req = omniroute.requests[-1]
    assert req["method"] == "GET" and req["auth"] == f"Bearer {TEST_KEY}"
    assert TEST_KEY not in req["path"]


def test_client_refuses_before_sending(client, omniroute):
    with pytest.raises(endpoints.EndpointNotAllowed):
        client.get("resilience")
    assert omniroute.requests == []


def test_mock_rejects_non_get_and_non_allowlisted(omniroute):
    """The mock is the test oracle: prove it fails closed on what the plugin must never send."""
    def send(method, path):
        req = urllib.request.Request(omniroute.url + path, method=method,
                                     headers={"Authorization": f"Bearer {TEST_KEY}"},
                                     data=b"{}" if method in ("POST", "PUT", "PATCH") else None)
        try:
            urllib.request.urlopen(req, timeout=5)
        except urllib.error.HTTPError as exc:
            return exc.code
        return 200

    assert send("POST", "/api/combos") == 405
    assert send("DELETE", "/api/combos/metrics") == 405
    assert send("PATCH", "/api/settings/combo-defaults") == 405
    assert send("GET", "/api/resilience") == 404
    assert send("GET", "/api/monitoring/health?deep=1") == 404
    assert send("GET", "/api/combos?reveal=1") == 400
    violations = omniroute.take_violations()
    assert len(violations) == 6
    assert sum(v.startswith("non-GET") for v in violations) == 3
    assert sum(v.startswith("path outside allowlist") for v in violations) == 2
    assert sum(v.startswith("query keys outside allowlist") for v in violations) == 1


def test_wrong_key_gets_403_hint_without_leaking(omniroute):
    secret = "wrong-placeholder-key-123"
    bad = OmniRouteClient(Settings(base_url=omniroute.url, api_key=secret))
    with pytest.raises(OmniRouteError) as info:
        bad.get("combos")
    assert info.value.status == 401
    assert secret not in str(info.value) and secret not in repr(bad)


def test_settings_never_show_key_and_normalize_url():
    s = load_settings(base_url="https://gw.example.com/v1/", api_key="placeholder-secret")
    assert s.base_url == "https://gw.example.com"
    assert "placeholder-secret" not in repr(s) and "placeholder-secret" not in str(s)
    assert normalize_base_url("http://gw.example.com/api/v1") == "http://gw.example.com"
    assert normalize_base_url("http://gw.example.com/prefix") == "http://gw.example.com/prefix"
    for bad in ("", "ftp://x", "http://user:pw@gw.example.com"):
        with pytest.raises(SettingsError):
            normalize_base_url(bad)
    with pytest.raises(SettingsError) as info:
        load_settings(base_url="http://gw.example.com", api_key="")
    assert "manage" in str(info.value)


def test_settings_read_from_environment(monkeypatch):
    from hermes_omniroute import settings
    monkeypatch.setenv(settings.ENV_BASE_URL, "http://gw.example.com")
    monkeypatch.setenv(settings.ENV_API_KEY, "placeholder")
    assert settings.is_configured()
    monkeypatch.delenv(settings.ENV_API_KEY)
    assert not settings.is_configured()


def test_transport_error_is_sanitized():
    c = OmniRouteClient(Settings(base_url="http://127.0.0.1:9", api_key=TEST_KEY, timeout_s=2))
    with pytest.raises(OmniRouteError) as info:
        c.get("combos")
    assert "failed" in str(info.value) and TEST_KEY not in str(info.value)
