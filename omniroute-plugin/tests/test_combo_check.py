"""hermes_combo_check against fixtures: settings collection, combo lookup, and each mismatch."""

from __future__ import annotations

from hermes_omniroute.combo_check import collect_model_settings, run_combo_check

GATEWAY = "http://gateway.example.com:20128"


def _config(**over):
    cfg = {
        "model": {"default": "agent-main", "provider": "omniroute"},
        "providers": {"omniroute": {"base_url": GATEWAY + "/v1", "request_timeout_seconds": 300}},
        "auxiliary": {
            "title_generation": {"provider": "omniroute", "model": "structured-local", "timeout": 30},
            "compression": {"base_url": GATEWAY + "/v1", "model": "missing-combo", "timeout": 120},
            "vision": {"provider": "auto", "model": "", "base_url": ""},          # inherits main
            "approval": {"provider": "openrouter", "model": "some/cloud-model"},   # not OmniRoute
        },
        "delegation": {"model": "openai/example-model", "base_url": GATEWAY + "/v1"},
        "fallback_providers": [{"provider": "omniroute", "model": "clean-cloud"}],
    }
    cfg.update(over)
    return cfg


def _check(snapshot, cfg):
    return run_combo_check(cfg, GATEWAY, snapshot["combos"], snapshot["combo_defaults"], profile="t")


def _codes(report):
    return sorted((f["code"], f["subject"].get("task")) for f in report["findings"])


def test_collects_main_aux_delegation_fallback_settings():
    tasks = {s.task: s for s in collect_model_settings(_config())}
    assert set(tasks) == {"model", "auxiliary.title_generation", "auxiliary.compression",
                          "auxiliary.vision", "auxiliary.approval", "delegation", "fallback_providers[0]"}
    assert tasks["auxiliary.vision"].inherits_main and tasks["auxiliary.vision"].model == "agent-main"
    assert tasks["model"].timeout_s == 300 and tasks["model"].timeout_source.endswith("request_timeout_seconds")
    assert tasks["auxiliary.title_generation"].base_url == GATEWAY + "/v1"   # via providers.omniroute


def test_findings_for_each_mismatch(snapshot):
    report = _check(snapshot, _config())
    codes = _codes(report)
    # title_generation sends json_schema; structured-local has a leg on an L1-gated loopback node
    assert ("STRUCTURED_OUTPUT_ON_L1_GATED_LEG", "auxiliary.title_generation") in codes
    # main agent loop (tools + system prompt) on a combo that filters tools and replaces the prompt
    assert ("TOOLS_FILTERED_BY_COMBO", "model") in codes
    assert ("SYSTEM_PROMPT_REPLACED_BY_COMBO", "model") in codes
    assert ("TOOLS_FILTERED_BY_COMBO", "auxiliary.vision") not in codes   # vision is not tool-calling
    # compression names a combo that does not exist -> error; delegation uses a direct id -> warning
    missing = {f["subject"]["task"]: f["severity"] for f in report["findings"] if f["code"] == "COMBO_NOT_FOUND"}
    assert missing == {"auxiliary.compression": "error", "delegation": "warning"}
    # approval points at another provider: listed, not checked
    rows = {r["task"]: r for r in report["settings"]}
    assert rows["auxiliary.approval"]["points_at_omniroute"] is False
    assert rows["model"]["combo"]["legs"][0]["model"] == "openai/example-model"
    assert report["summary"]["status"] == "attention"


def test_l2_uses_hermes_per_call_timeout(snapshot):
    cfg = _config(auxiliary={"title_generation": {"provider": "omniroute", "model": "structured-local",
                                                  "timeout": 300}})
    l2 = [f for f in _check(snapshot, cfg)["findings"] if f["code"] == "L2_LEG_TIMEOUT_BELOW_CALL_TIMEOUT"]
    assert {f["subject"]["leg"] for f in l2} == {0, 1}
    assert l2[0]["subject"]["leg_timeout_ms"] == 120_000 and l2[0]["subject"]["per_call_timeout_ms"] == 300_000
    # 30 s per call is below the 120 s leg timeout: no L2 finding
    assert not [f for f in _check(snapshot, _config())["findings"]
                if f["code"] == "L2_LEG_TIMEOUT_BELOW_CALL_TIMEOUT"]


def test_clean_profile(snapshot):
    cfg = {"model": {"default": "clean-cloud", "provider": "custom", "base_url": GATEWAY + "/v1"},
           "auxiliary": {"vision": {"provider": "auto", "model": ""}}}
    report = _check(snapshot, cfg)
    assert report["summary"] == {"status": "clean", "error": 0, "warning": 0, "info": 0}
    assert report["pointing_at_omniroute"] == 2
    assert all(r["combo_found"] for r in report["settings"])


def test_model_string_without_base_url_is_not_claimed():
    assert collect_model_settings({"model": "anthropic/some-model"})[0].base_url == ""
