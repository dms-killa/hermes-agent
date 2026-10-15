"""Golden tests for the normalization port. Expected values were derived by hand from OmniRoute
src/lib/combos/steps.ts (normalizeComboStep / buildStepId / slugify / toWeight) at the SHA in
docs/CAPABILITIES.md, not produced by running the port."""

from __future__ import annotations

import pytest

from hermes_omniroute import normalize as n


def step(value, index=0, combo="My Combo", names=()):
    return n.normalize_combo_step(value, combo_name=combo, index=index, names=names)


def test_string_model_step():
    assert step("openai/gpt-x") == {"id": "my-combo-model-1-openai-gpt-x", "kind": "model",
                                    "model": "openai/gpt-x", "providerId": "openai", "weight": 0}


def test_string_combo_ref_and_wildcard():
    assert step("other", index=1, names=["other"]) == {
        "id": "my-combo-ref-2-other", "kind": "combo-ref", "comboName": "other", "weight": 0}
    assert step("prov/*") == {"id": "my-combo-wildcard-1-prov", "kind": "provider-wildcard",
                              "providerId": "prov", "modelPattern": "*", "weight": 0}


def test_object_model_step_full_model_pin_and_weight_clamp():
    assert step({"model": "small", "providerId": "p", "connectionId": " c1 ", "weight": "150"},
                combo="x") == {"id": "x-model-1-p-small-c1", "kind": "model", "model": "p/small",
                               "providerId": "p", "connectionId": "c1", "weight": 100}


def test_null_connection_is_kept_and_default_combo_slug():
    assert step({"model": "m", "connectionId": None}, combo=None) == {
        "id": "combo-model-1-m", "kind": "model", "model": "m", "connectionId": None, "weight": 0}


def test_explicit_id_tags_and_empty_allowlist():
    out = step({"id": "keep-me", "kind": "model", "model": "a/b", "tags": [" t ", ""],
                "allowedConnectionIds": [], "fallbackOnlyOnQuotaExhaustion": False, "weight": 5})
    assert out == {"id": "keep-me", "kind": "model", "model": "a/b", "providerId": "a", "weight": 5,
                   "tags": ["t"]}


def test_combo_ref_object():
    assert step({"kind": "combo-ref", "comboName": "z", "label": "L"}, combo="x") == {
        "id": "x-ref-1-z", "kind": "combo-ref", "comboName": "z", "weight": 0, "label": "L"}


def test_non_explicit_object_matching_a_combo_name_becomes_ref():
    assert step({"model": "z"}, combo="x", names=["z"])["kind"] == "combo-ref"
    assert step({"kind": "model", "model": "z"}, combo="x", names=["z"])["kind"] == "model"


def test_normalization_is_idempotent_on_stored_steps():
    once = n.normalize_combo_models(["openai/gpt-x", {"model": "m", "providerId": "p", "connectionId": "c"}],
                                    combo_name="x", names=[])
    assert n.normalize_combo_models(once, combo_name="x", names=[]) == once


def test_record_dead_keys_depth_hash_and_diff():
    rec = n.normalize_combo_record({"name": "x", "models": ["a/b"]}, [])
    assert rec["version"] == 2 and rec["models"][0]["id"] == "x-model-1-a-b"
    assert n.strip_dead_config_keys({"resetAwareEnabled": True, "maxRetries": 1}) == {"maxRetries": 1}
    assert [n.clamp_combo_depth(v) for v in (None, 0, 2.9, 50)] == [3, 3, 2, 10]
    assert n.content_hash({"a": 1, "b": [1, 2]}) == n.content_hash({"b": [1, 2], "a": 1})
    assert n.diff_paths({"x": True}, {"x": 1}) == [{"path": "x", "before": True, "after": 1}]
    assert n.diff_paths({"a": {"b": 1}}, {"a": {"b": 1}, "c": 2}) == [{"path": "c", "before": "<absent>", "after": 2}]


@pytest.mark.parametrize("key", sorted(n.NULLABLE_KEYS))
def test_nullable_keys_are_writable(key):
    assert key in n.WRITABLE_KEYS and key not in n.OUT_OF_SCOPE_KEYS


def test_zero_latency_promotion_matches_schema_transform():
    # schemas/combo.ts:301-319: any zero-latency feature without the gate promotes the gate.
    assert n.server_config({"hedging": True}) == {"hedging": True, "zeroLatencyOptimizationsEnabled": True}
    assert n.server_config({"predictiveTtftMs": 5}) == {"predictiveTtftMs": 5, "zeroLatencyOptimizationsEnabled": True}
    assert n.server_config({"fallbackCompressionMode": "off"}) == {"fallbackCompressionMode": "off"}
    assert n.server_config({"hedging": True, "zeroLatencyOptimizationsEnabled": True}) == {
        "hedging": True, "zeroLatencyOptimizationsEnabled": True}
    assert n.server_config({"predictiveTtftMs": 0, "pipelineConcurrency": 2}) == {"predictiveTtftMs": 0}
