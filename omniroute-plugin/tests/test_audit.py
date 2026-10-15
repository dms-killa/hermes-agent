"""omniroute_audit against fixtures: one test per lesson finding (L1-L4) plus a clean case."""

from __future__ import annotations

import json

from hermes_omniroute.audit import run_audit
from hermes_omniroute.model import effective_target_timeout


def _codes(report):
    return [f["code"] for f in report["findings"]]


def _by_code(report, code):
    return [f for f in report["findings"] if f["code"] == code]


def test_l1_flags_gated_self_hosted_connections_only(snapshot):
    report = run_audit(snapshot)
    l1 = _by_code(report, "L1_JSON_SCHEMA_DOWNGRADE")
    flagged = {f["subject"]["connection_id"]: f["subject"] for f in l1}
    # loopback via node baseUrl, .lan name, private range: flagged
    assert set(flagged) == {"conn-loop", "conn-lan", "conn-priv"}
    assert flagged["conn-loop"]["host_class"] == "loopback"
    assert flagged["conn-loop"]["base_url_source"] == "provider_node.baseUrl"
    assert flagged["conn-lan"]["host_class"] == "local-name"
    assert flagged["conn-priv"]["host_class"] == "private"
    assert sorted(flagged["conn-loop"]["combos"]) == ["spill", "structured-local"]
    # gated but public host, and a native llama-cpp id on loopback: not flagged
    assert "conn-cloud-compat" not in flagged and "conn-llamacpp" not in flagged
    assert all(f["severity"] == "warning" for f in l1)
    assert all(any(c.endswith("#l1") for c in f["cite"]) for f in l1)


def test_l2_compares_per_call_timeout_with_leg_timeout(snapshot):
    report = run_audit(snapshot, per_call_timeout_ms=300_000)
    l2 = {(f["subject"]["combo"], f["subject"]["leg"]): f["subject"]
          for f in _by_code(report, "L2_LEG_TIMEOUT_BELOW_CALL_TIMEOUT")}
    # structured-local has no targetTimeoutMs: OmniRoute default 120000 < 300000 on both legs
    assert l2[("structured-local", 0)]["leg_timeout_ms"] == 120_000
    assert ("structured-local", 1) in l2
    # fanout-p2c sets exactly 300000: equal is fine; spill/clean-cloud set more: fine
    assert not any(k[0] in ("fanout-p2c", "spill", "clean-cloud") for k in l2)


def test_l2_leg_timeout_resolution_follows_omniroute(snapshot):
    combos = {c.name: c for c in snapshot["combos"]}
    defaults = snapshot["combo_defaults"]
    t = effective_target_timeout(combos["structured-local"], defaults, None)
    assert (t["ms"], t["source"]) == (120_000, "OmniRoute default")
    t = effective_target_timeout(combos["spill"], defaults, None)
    assert (t["ms"], t["source"]) == (600_000, "combo.config.targetTimeoutMs")
    # combo-level default from settings.comboDefaults wins over the built-in default
    t = effective_target_timeout(combos["structured-local"], {"targetTimeoutMs": 200_000}, None)
    assert (t["ms"], t["source"]) == (200_000, "settings.comboDefaults.targetTimeoutMs")
    # capped by the pinned connection's own timeoutMs
    priv = snapshot["connections"]["conn-priv"]
    combos["clean-cloud"].config["targetTimeoutMs"] = 1_200_000
    t = effective_target_timeout(combos["clean-cloud"], defaults, priv)
    assert (t["ms"], t["ceiling_source"]) == (900_000, "connection.providerSpecificData.timeoutMs")
    # 0 is treated as unset (normalizePositiveTimeoutMs), so the default applies
    combos["clean-cloud"].config["targetTimeoutMs"] = 0
    assert effective_target_timeout(combos["clean-cloud"], defaults, None)["ms"] == 120_000


def test_l3_p2c_info_and_warning_under_concurrency(snapshot):
    info = _by_code(run_audit(snapshot), "L3_P2C_NOT_INFLIGHT_BALANCED")
    assert [f["subject"]["combo"] for f in info] == ["fanout-p2c"] and info[0]["severity"] == "info"
    warn = _by_code(run_audit(snapshot, concurrency=4), "L3_P2C_NOT_INFLIGHT_BALANCED")
    assert warn[0]["severity"] == "warning" and "4 concurrent" in warn[0]["message"]


def test_l4_shared_cap_and_concurrency_over_caps(snapshot):
    report = run_audit(snapshot, concurrency=4)
    shared = _by_code(report, "L4_SHARED_CONNECTION_CAP")
    assert sorted((f["subject"]["connection_id"], f["subject"]["combos"]) for f in shared) == [
        ("conn-lan", ["fanout-p2c", "spill"]), ("conn-loop", ["spill", "structured-local"])]
    over = {f["subject"]["combo"]: f["subject"] for f in _by_code(report, "L4_CONCURRENCY_EXCEEDS_CAPS")}
    # spill: conn-loop (1) + conn-lan (2) = 3 < 4 ; structured-local: conn-loop (1) + conn-llamacpp (1) = 2
    assert over["spill"]["capacity"] == 3
    assert over["structured-local"]["capacity"] == 2
    # fanout-p2c has an uncapped leg (conn-priv) and clean-cloud an uncapped connection: no finding
    assert "fanout-p2c" not in over and "clean-cloud" not in over
    assert not _by_code(run_audit(snapshot, concurrency=3, combo_names=["spill"]), "L4_CONCURRENCY_EXCEEDS_CAPS")


def test_combo_section_joins_connection_max_concurrent(snapshot):
    report = run_audit(snapshot)
    spill = next(c for c in report["combos"] if c["name"] == "spill")
    assert spill["strategy"] == "priority"
    legs = [(l["pinned_connection_id"], l["connections"][0]["max_concurrent"]) for l in spill["legs"]]
    assert legs == [("conn-loop", 1), ("conn-lan", 2)]
    p2c = next(c for c in report["combos"] if c["name"] == "fanout-p2c")
    assert [l["connections"][0]["id"] for l in p2c["legs"]] == ["conn-lan", "conn-priv"]


def test_clean_case(snapshot):
    report = run_audit(snapshot, per_call_timeout_ms=300_000, concurrency=2, combo_names=["clean-cloud"])
    assert report["summary"] == {"status": "clean", "error": 0, "warning": 0, "info": 0}
    assert report["findings"] == []
    assert report["combos"][0]["legs"][0]["target_timeout"]["ms"] == 600_000  # capped by FETCH default


def test_every_finding_cites_capabilities_and_no_secret_leaks(snapshot):
    report = run_audit(snapshot, per_call_timeout_ms=300_000, concurrency=4)
    assert report["findings"]
    for f in report["findings"]:
        assert f["cite"] and all(c.startswith("docs/CAPABILITIES.md#") for c in f["cite"])
    text = json.dumps(report)
    assert "PLACEHOLDER-MASKED-KEY" not in text and "apiKey" not in text


def test_missing_pinned_connection_is_reported(snapshot, fixture_json):
    from hermes_omniroute.audit import build_snapshot
    combos = fixture_json("combos.json")
    combos["combos"][2]["models"][1]["connectionId"] = "conn-deleted"
    snap = build_snapshot(fixture_json("connections.json"), fixture_json("provider_nodes.json"),
                          combos, fixture_json("combo_defaults.json"))
    assert _by_code(run_audit(snap), "PINNED_CONNECTION_MISSING")[0]["subject"] == {"connection_id": "conn-deleted"}
