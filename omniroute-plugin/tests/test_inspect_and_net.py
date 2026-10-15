"""Call-log filtering and probe signatures (L5, L6), combo health, and host classification."""

from __future__ import annotations

import json

import pytest

from hermes_omniroute import signatures
from hermes_omniroute.inspect_tools import call_logs_report, combo_health_report
from hermes_omniroute.netclass import classify_url


def test_combo_filter_is_exact_name_on_client(fixture_json):
    rows = fixture_json("call_logs.json")
    report = call_logs_report(rows, combo_name="structured-local", limit=10)
    assert [r["id"] for r in report["rows"]] == ["log-0001", "log-0002", "log-0005"]
    assert call_logs_report(rows, combo_name="spill", limit=10)["count"] == 1
    assert "body_signature" not in report["rows"][0]          # detail fetch is opt-in


def test_body_signature_tells_probe_from_real(fixture_json):
    rows = fixture_json("call_logs.json")
    fetched = []

    def fetch(log_id):
        fetched.append(log_id)
        return fixture_json(f"call_log_detail_{log_id}.json")

    report = call_logs_report(rows, combo_name="structured-local", limit=10, detail_fetcher=fetch)
    by_id = {r["id"]: r for r in report["rows"]}
    assert by_id["log-0001"]["probe"] == "omniroute-combo-test" and by_id["log-0001"]["traffic"] == "probe"
    assert by_id["log-0002"]["traffic"] == "real"
    assert by_id["log-0002"]["body_signature"]["response_format"] == "json_schema"
    assert by_id["log-0005"]["traffic"] == "unknown"           # no stored body
    assert fetched == ["log-0001", "log-0002"]                 # rows without a body are not fetched
    assert report["traffic"] == {"real": 1, "probe": 1, "unknown": 1}
    # message text never appears in the output, only a hash and a length
    assert "Synthetic conversation text" not in json.dumps(report)


def test_operator_probe_prompts(fixture_json):
    rows = fixture_json("call_logs.json")
    fetch = lambda i: fixture_json(f"call_log_detail_{i}.json")
    plain = call_logs_report(rows, combo_name="spill", limit=5, detail_fetcher=fetch)
    assert plain["rows"][0]["traffic"] == "real"                # unknown probe text: not guessed
    tagged = call_logs_report(rows, combo_name="spill", limit=5, detail_fetcher=fetch,
                              extra_probe_prompts=["Say ok"])
    assert tagged["rows"][0]["probe"] == "operator-probe"


def test_probe_needs_exact_body_shape():
    base = {"messages": [{"role": "user", "content": "Reply with exactly: pong"}], "max_tokens": 64}
    assert signatures.classify_probe(signatures.body_signature(base)) == "omniroute-combo-test"
    assert signatures.classify_probe(signatures.body_signature(dict(base, max_tokens=65))) is None
    assert signatures.classify_probe(signatures.body_signature(dict(base, tools=[{}]))) is None
    with_system = {"messages": [{"role": "system", "content": "x"}] + base["messages"], "max_tokens": 64}
    assert signatures.classify_probe(signatures.body_signature(with_system)) is None
    assert signatures.body_signature("not json") is None


def test_combo_health_report(snapshot, fixture_json):
    combo = next(c for c in snapshot["combos"] if c.name == "structured-local")
    report = combo_health_report(combo, fixture_json("combo_health.json"),
                                 fixture_json("combo_metrics.json"), range_="24h")
    assert report["health"]["targetHealth"][0]["connectionId"] == "conn-loop"
    assert report["live_metrics"]["totalRequests"] == 40
    assert combo_health_report(None, None, None, range_="24h")["found"] is False


@pytest.mark.parametrize("url,expected", [
    ("http://127.0.0.1:8080/v1", "loopback"), ("http://localhost:1234", "loopback"),
    ("http://[::1]:8000", "loopback"), ("http://10.1.2.3", "private"), ("http://192.168.0.9", "private"),
    ("http://172.16.5.5", "private"), ("http://169.254.1.1", "link-local"),
    ("http://box.example.lan", "local-name"), ("http://box.local:11434", "local-name"),
    ("http://app.localhost", "local-name"), ("https://api.example.com/v1", None),
    ("http://8.8.8.8", None), ("", None), ("not a url", None),
])
def test_classify_url(url, expected):
    assert classify_url(url) == expected
