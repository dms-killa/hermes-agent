"""plan -> apply -> read-back -> rollback against the stateful mock combo store."""

from __future__ import annotations

import json

import pytest

from hermes_omniroute.approval import make_approval_hook
from hermes_omniroute.combo_write import ComboWriter, PlanRefused, compare_stored
from hermes_omniroute.write_store import MemoryState, WriteStore

SPILL = "00000000-0000-4000-8000-00000000c003"
STRUCTURED = "00000000-0000-4000-8000-00000000c001"
FANOUT = "00000000-0000-4000-8000-00000000c002"


class Clock:
    def __init__(self):
        self.t = 1_000_000.0

    def __call__(self):
        return self.t


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def store(clock):
    return WriteStore(MemoryState(), clock=clock)


@pytest.fixture
def writer(client, store):
    return ComboWriter(client, store)


def combo_route_writes(omniroute):
    return [(r["method"], r["path"]) for r in omniroute.writes()]


# ── plan ─────────────────────────────────────────────────────────────────────────────────────────

def test_plan_makes_no_writes_and_returns_the_contract(writer, omniroute):
    plan = writer.plan("spill", {"config": {"targetTimeoutMs": 300000}})
    assert omniroute.writes() == []
    assert {"plan_id", "token", "expires_at", "diff", "applicable", "irreversible_reasons", "findings",
            "server_may_reject", "side_effects"} <= set(plan)
    assert plan["diff"] == [{"path": "config.targetTimeoutMs", "before": 600000, "after": 300000}]
    assert plan["applicable"] and plan["reversible"] and plan["token"].startswith("sha256:")
    assert any("cloud sync" in s for s in plan["side_effects"])
    assert plan["put_body_keys"] == ["config"]
    assert {r["check"].split(" (")[0] for r in plan["server_may_reject"]} >= {"name collision",
                                                                               "quota-share managed combo"}


def test_findings_l1_to_l4_are_information_not_a_block(writer, omniroute):
    plan = writer.plan("structured-local", {"strategy": "p2c"},
                       caller_limits={"per_call_timeout_ms": 300000, "concurrency": 4})
    codes = {f["code"] for f in plan["findings"]}
    assert {"L1_JSON_SCHEMA_DOWNGRADE", "L2_LEG_TIMEOUT_BELOW_CALL_TIMEOUT",
            "L3_P2C_NOT_INFLIGHT_BALANCED", "L4_CONCURRENCY_EXCEEDS_CAPS",
            "L4_SHARED_CONNECTION_CAP"} <= codes
    assert all(c.startswith("docs/CAPABILITIES.md#") for f in plan["findings"] for c in f["cite"])
    l3 = next(f for f in plan["findings"] if f["code"] == "L3_P2C_NOT_INFLIGHT_BALANCED")
    assert l3["new_with_this_change"] is True                       # p2c is introduced by this change
    l1 = next(f for f in plan["findings"] if f["code"] == "L1_JSON_SCHEMA_DOWNGRADE")
    assert l1["new_with_this_change"] is False                      # pre-existing
    result = writer.apply(plan["plan_id"])
    assert result["applied"] is True and result["verification"] == "passed"


def test_unpinned_leg_finding(writer):
    plan = writer.plan("fanout-p2c", {"description": "two unpinned legs"})
    assert [f["subject"]["leg"] for f in plan["findings"] if f["code"] == "UNPINNED_LEG"] == [0, 1]


@pytest.mark.parametrize("changes,needle", [
    ({"name": "renamed"}, "out of scope"),
    ({"overrideAllowedProviders": True}, "out of scope"),
    ({"secretField": 1}, "silently drop"),
    ({"strategy": None}, "cannot be reset with null"),
    ({"config": None}, "cannot be reset with null"),
    ({"config": "x"}, "must be an object"),
    ({"strategy": "priority"}, "empty diff"),
    ({}, "non-empty"),
])
def test_plan_refusals(writer, omniroute, changes, needle):
    with pytest.raises(PlanRefused) as info:
        writer.plan("spill", changes)
    assert needle in str(info.value)
    assert omniroute.writes() == []


def test_plan_refuses_unknown_and_quota_share_combos(writer, omniroute):
    with pytest.raises(PlanRefused, match="no existing combo"):
        writer.plan("does-not-exist", {"description": "x"})
    q = dict(omniroute.store.get(SPILL), id="00000000-0000-4000-8000-0000000000f1", name="qtSd/pool")
    omniroute.store.combos[q["id"]] = q
    with pytest.raises(PlanRefused, match="quota-share"):
        writer.plan("qtSd/pool", {"description": "x"})


# ── apply refusals ───────────────────────────────────────────────────────────────────────────────

def test_apply_refuses_unknown_expired_stale_irreversible_and_reused(writer, omniroute, clock):
    assert writer.apply("plan-nope")["code"] == "unknown_plan"

    expired = writer.plan("spill", {"description": "a"}, ttl_seconds=60)
    clock.t += 61
    assert writer.apply(expired["plan_id"])["code"] == "plan_expired"

    stale = writer.plan("spill", {"description": "b"})
    omniroute.store.put(SPILL, {"description": "third-party edit"})
    refusal = writer.apply(stale["plan_id"])
    assert refusal["code"] == "plan_stale" and any(c["path"] == "description" for c in refusal["changed"])

    # isActive is absent on the stored combo and not nullable: rollback could never remove it again.
    irreversible = writer.plan("spill", {"isActive": False})
    assert irreversible["applicable"] is False
    assert "not nullable" in irreversible["irreversible_reasons"][0]
    assert writer.apply(irreversible["plan_id"])["code"] == "plan_irreversible"

    ok = writer.plan("spill", {"description": "c"})
    assert writer.apply(ok["plan_id"])["applied"] is True
    assert writer.apply(ok["plan_id"])["code"] == "plan_consumed"
    # Only the third-party edit (made directly in the store) and the one good apply wrote anything.
    assert combo_route_writes(omniroute) == [("PUT", f"/api/combos/{SPILL}")]


def test_plan_refuses_irreversible_dead_keys_and_wildcards(writer, omniroute):
    omniroute.store.combos[SPILL]["config"]["resetAwareEnabled"] = True       # legacy stored row
    plan = writer.plan("spill", {"description": "x"})
    assert not plan["applicable"] and "strips on every config write" in plan["irreversible_reasons"][0]
    omniroute.store.combos[SPILL]["config"].pop("resetAwareEnabled")
    omniroute.store.combos[SPILL]["models"].append(
        {"id": "w", "kind": "provider-wildcard", "providerId": "openai", "modelPattern": "*", "weight": 0})
    plan = writer.plan("spill", {"description": "x"})
    assert not plan["applicable"] and "provider-wildcard" in plan["irreversible_reasons"][0]
    omniroute.store.combos[SPILL]["models"].pop()
    omniroute.store.combos[SPILL]["config"]["hedging"] = True        # stored before the gate existed
    plan = writer.plan("spill", {"description": "x"})
    assert not plan["applicable"] and "zero-latency" in plan["irreversible_reasons"][0]


def test_plan_reproduces_zero_latency_promotion(writer, omniroute):
    plan = writer.plan("spill", {"config": {"hedging": True}})
    assert {d["path"] for d in plan["diff"]} == {"config.hedging", "config.zeroLatencyOptimizationsEnabled"}
    applied = writer.apply(plan["plan_id"])
    assert applied["applied"] and applied["verification"] == "passed"
    assert writer.rollback(applied["rollback_id"])["rolled_back"] is True


# ── round trip ───────────────────────────────────────────────────────────────────────────────────

def test_round_trip_ends_equal_to_snapshot_and_removes_added_key(writer, omniroute):
    before = omniroute.store.get(SPILL)
    assert "description" not in before
    changes = {
        "description": "pinned spill-over",                          # a key the apply ADDS
        "config": {"targetTimeoutMs": 300000, "maxRetries": 2},
        "models": ["openai/example-model", *before["models"]],       # string leg: server-normalized
    }
    plan = writer.plan("spill", changes)
    assert {d["path"] for d in plan["diff"]} >= {"description", "config.targetTimeoutMs", "config.maxRetries"}
    applied = writer.apply(plan["plan_id"])
    assert applied["applied"] and applied["verification"] == "passed"
    after = omniroute.store.get(SPILL)
    assert after["description"] == "pinned spill-over"
    assert after["models"][0] == {"id": "spill-model-1-openai-example-model", "kind": "model",
                                  "model": "openai/example-model", "providerId": "openai", "weight": 0}
    rolled = writer.rollback(applied["rollback_id"])
    assert rolled["rolled_back"] and rolled["verification"] == "passed" and rolled["forced"] is False
    final = omniroute.store.get(SPILL)
    assert compare_stored(before, final) == []                       # equal up to server-set updatedAt
    assert "description" not in final                                # reset with an explicit null
    assert combo_route_writes(omniroute) == [("PUT", f"/api/combos/{SPILL}")] * 2
    assert writer.rollback(applied["rollback_id"])["code"] == "already_rolled_back"


def test_rollback_sends_complete_snapshot_with_explicit_resets(writer, omniroute, client, store):
    plan = writer.plan("spill", {"description": "x", "tool_filter_regex": "^a"})
    applied = writer.apply(plan["plan_id"])
    sent = []
    real_put = client.put
    client.put = lambda ep, body, **kw: (sent.append(body), real_put(ep, body, **kw))[1]
    writer.rollback(applied["rollback_id"])
    body = sent[0]
    assert body["description"] is None and body["tool_filter_regex"] is None     # explicit resets
    assert {"models", "strategy", "config"} <= set(body) and "name" not in body  # complete, no rename


# ── verification failure, server rejects ─────────────────────────────────────────────────────────

def test_readback_mismatch_restores_automatically(writer, omniroute):
    before = omniroute.store.get(SPILL)
    calls = {"n": 0}

    def surprise(stored):          # the server changes a field the plan did not touch, once
        calls["n"] += 1
        if calls["n"] == 1:
            stored["strategy"] = "random"
        return stored

    omniroute.store.after_store = surprise
    plan = writer.plan("spill", {"config": {"targetTimeoutMs": 300000}})
    result = writer.apply(plan["plan_id"])
    assert result["applied"] is False and result["verification"] == "failed"
    bad = {m["path"]: m for m in result["mismatch"]}
    assert bad["strategy"]["planned_field"] is False
    assert result["restore"]["restored"] is True
    assert compare_stored(before, omniroute.store.get(SPILL)) == []
    assert len(omniroute.writes()) == 2                              # the PUT and the restore PUT


def test_server_dag_reject_leaves_combo_unchanged(writer, omniroute):
    omniroute.store.put(FANOUT, {"models": [{"kind": "combo-ref", "comboName": "spill"}]})
    before = omniroute.store.get(SPILL)
    plan = writer.plan("spill", {"models": before["models"] + [{"kind": "combo-ref", "comboName": "fanout-p2c"}]})
    dag = next(r for r in plan["server_may_reject"] if r["check"].startswith("nested-combo DAG"))
    assert dag["predicted"] is True and "cycle" in dag["note"]
    assert plan["applicable"] is True                                # a predicted reject is information
    result = writer.apply(plan["plan_id"])
    assert result["applied"] is False and result["status"] == 400
    assert result["server_error"]["code"] == "COMBO_005"
    assert result["server_error"]["details"]["reason"] == "cycle-detected"
    assert result["combo_unchanged"] is True and omniroute.store.get(SPILL) == before


def test_server_quota_409_leaves_combo_unchanged(writer, omniroute):
    before = omniroute.store.get(SPILL)
    plan = writer.plan("spill", {"description": "x"})
    omniroute.store.force_409.add(SPILL)        # server-side quota-share ownership the plan cannot see
    result = writer.apply(plan["plan_id"])
    assert result["status"] == 409 and result["server_error"]["code"] == "COMBO_006"
    assert result["combo_unchanged"] is True and omniroute.store.get(SPILL) == before


# ── forced vs unforced rollback ──────────────────────────────────────────────────────────────────

def test_rollback_after_third_party_edit_needs_force(writer, omniroute):
    before = omniroute.store.get(SPILL)
    applied = writer.apply(writer.plan("spill", {"config": {"targetTimeoutMs": 300000}})["plan_id"])
    omniroute.store.put(SPILL, {"system_message": "edited elsewhere"})
    refused = writer.rollback(applied["rollback_id"])
    assert refused["code"] == "changed_since_apply"
    assert [c["path"] for c in refused["changed"]] == ["system_message"]
    forced = writer.rollback(applied["rollback_id"], force=True)
    assert forced["rolled_back"] and forced["forced"] is True
    assert [c["path"] for c in forced["overwrote_third_party_changes"]] == ["system_message"]
    assert compare_stored(before, omniroute.store.get(SPILL)) == []


# ── mock strictness and the approval hook ────────────────────────────────────────────────────────

def test_mock_keeps_rejecting_other_writes(omniroute):
    import urllib.error
    import urllib.request
    from mock_server import TEST_KEY
    for method, path in [("PUT", "/api/providers"), ("POST", "/api/combos"), ("DELETE", f"/api/combos/{SPILL}"),
                         ("PUT", "/api/settings/combo-defaults"), ("PUT", "/api/combos/metrics")]:
        req = urllib.request.Request(omniroute.url + path, method=method, data=b"{}",
                                     headers={"Authorization": f"Bearer {TEST_KEY}"})
        with pytest.raises(urllib.error.HTTPError) as info:
            urllib.request.urlopen(req, timeout=5)
        assert info.value.code == 405
    assert len(omniroute.take_violations()) == 5


def test_approval_hook_escalates_only_write_tools(store, writer):
    hook = make_approval_hook(store)
    plan = writer.plan("spill", {"description": "x"})
    directive = hook(tool_name="omniroute_combo_apply", args={"plan_id": plan["plan_id"]})
    assert directive["action"] == "approve" and directive["rule_key"] == "omniroute_combo_apply"
    assert "spill" in directive["message"] and "description" in directive["message"]
    rb = hook(tool_name="omniroute_combo_rollback", args={"rollback_id": "rb-x", "force": True})
    assert rb["action"] == "approve"
    for read_tool in ("omniroute_combo_plan", "omniroute_audit", "terminal"):
        assert hook(tool_name=read_tool, args={}) is None
    json.dumps(directive)
