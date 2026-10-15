"""Tool handlers end to end over HTTP against the strict mock (no Hermes needed)."""

from __future__ import annotations

import json

import pytest

from hermes_omniroute.client import OmniRouteClient
from hermes_omniroute.settings import load_settings
from hermes_omniroute.tools import make_handlers
from mock_server import TEST_KEY


@pytest.fixture
def handlers(omniroute):
    cfg = {"model": {"default": "structured-local", "provider": "custom",
                     "base_url": omniroute.url + "/v1"}}
    return make_handlers(
        client_factory=lambda: OmniRouteClient(load_settings(base_url=omniroute.url, api_key=TEST_KEY)),
        config_loader=lambda: cfg)


def call(handlers, tool, **args):
    return json.loads(handlers[tool](args))


def test_audit_tool(handlers, omniroute):
    report = call(handlers, "omniroute_audit", per_call_timeout_ms=300000, concurrency=4)
    lessons = {f["lesson"] for f in report["findings"]}
    assert {"L1", "L2", "L3", "L4"} <= lessons
    assert report["gateway"] == omniroute.url
    assert {r["path"].split("?")[0] for r in omniroute.requests} == {
        "/api/providers", "/api/provider-nodes", "/api/combos", "/api/settings/combo-defaults"}
    assert all(r["method"] == "GET" for r in omniroute.requests)


def test_combo_check_tool(handlers):
    report = call(handlers, "hermes_combo_check")
    assert report["pointing_at_omniroute"] == 1
    assert report["settings"][0]["combo"]["name"] == "structured-local"
    assert report["summary"]["status"] == "clean"   # main model: tools, no filter/system_message


def test_connections_and_combos_tools(handlers, omniroute):
    conns = call(handlers, "omniroute_connections")
    assert conns["count"] == 6 and "PLACEHOLDER-MASKED-KEY" not in json.dumps(conns)
    one = call(handlers, "omniroute_connections", provider="llama-cpp")
    assert any("provider=llama-cpp" in r["path"] for r in omniroute.requests)
    assert one["count"] == 6   # the mock ignores the filter; the query reached it as allowed
    combos = call(handlers, "omniroute_combos", name="spill")
    assert combos["count"] == 1 and combos["combos"][0]["legs"][1]["connections"][0]["max_concurrent"] == 2


def test_call_logs_tool_with_and_without_bodies(handlers, omniroute):
    plain = call(handlers, "omniroute_call_logs", combo_name="structured-local")
    assert plain["count"] == 3
    assert not any("/api/usage/call-logs/" in r["path"] for r in omniroute.requests)
    signed = call(handlers, "omniroute_call_logs", combo_name="structured-local",
                  include_body_signature=True, limit=5)
    assert signed["traffic"] == {"real": 1, "probe": 1, "unknown": 1}
    assert any(r["path"] == "/api/usage/call-logs/log-0001" for r in omniroute.requests)


def test_combo_health_tool(handlers, omniroute):
    report = call(handlers, "omniroute_combo_health", combo_name="structured-local", range="24h")
    assert report["found"] and report["health"]["comboName"] == "structured-local"
    paths = [r["path"] for r in omniroute.requests]
    assert any(p.startswith("/api/usage/combo-health?comboId=") for p in paths)
    assert "/api/combos/metrics?combo=structured-local" in paths
    assert call(handlers, "omniroute_combo_health", combo_name="nope")["found"] is False


@pytest.mark.parametrize("name,args,needle", [
    ("omniroute_audit", {"concurrency": "four"}, "concurrency must be an integer"),
    ("omniroute_call_logs", {"limit": 0}, "limit must be between"),
    ("omniroute_call_logs", {"status": "weird"}, "status must be"),
    ("omniroute_combo_health", {}, "combo_name is required"),
    ("omniroute_combo_health", {"combo_name": "x", "range": "2y"}, "range must be"),
])
def test_argument_errors_are_returned_not_raised(handlers, name, args, needle):
    assert needle in call(handlers, name, **args)["error"]


def test_unconfigured_and_rejected_key(omniroute, monkeypatch):
    monkeypatch.delenv("OMNIROUTE_BASE_URL", raising=False)
    monkeypatch.delenv("OMNIROUTE_API_KEY", raising=False)
    out = json.loads(make_handlers()["omniroute_audit"]({}))
    assert "OMNIROUTE_BASE_URL is not set" in out["error"]
    monkeypatch.setenv("OMNIROUTE_BASE_URL", omniroute.url)
    monkeypatch.setenv("OMNIROUTE_API_KEY", "not-the-placeholder-key")
    out = json.loads(make_handlers()["omniroute_combos"]({}))
    assert out["status"] == 401 and "not-the-placeholder-key" not in json.dumps(out)
