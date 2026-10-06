import json

from bayes_check import ledger
from bayes_check.commands import make_command
from bayes_check.config import Settings, load
from bayes_check.hooks import Hooks
from bayes_check.tools import make_handler


def _cmd(store, **over):
    return make_command(lambda: Settings(**over), lambda: store)


def test_on_off_cli_session(store, monkeypatch):
    monkeypatch.setenv("HERMES_SESSION_ID", "cli-1")
    bayes = _cmd(store)
    assert "ON" in bayes("on") and ledger.is_enabled("cli-1")
    assert "OFF" in bayes("off") and not ledger.is_enabled("cli-1")


def test_on_without_any_session_is_explained(store):
    assert "no active session" in _cmd(store)("on")


def test_status_reports_mode_local_and_ledger(store, monkeypatch):
    monkeypatch.setenv("HERMES_SESSION_ID", "cli-1")
    ledger.set_local("cli-1", True)
    out = _cmd(store, local_mode="auto")("status")
    assert "mode=annotate" in out and "→ local" in out and "ledger: 0 claim(s)" in out
    assert "→ cloud" in _cmd(store, local_mode="force_cloud")("status")


def test_correct_and_refit_roundtrip(store, monkeypatch):
    monkeypatch.setenv("HERMES_SESSION_ID", "cli-1")
    ledger.enable("cli-1")
    handler = make_handler(lambda: Settings(), lambda: store)
    hooks = Hooks(lambda: Settings(), lambda: store)
    res = handler({"claims": [{"id": "c1", "text": "Water boils at 100 C at sea level.", "claim_type": "fact",
                               "prior": 0.8, "evidence": []}]}, session_id="cli-1")
    hooks.post_tool_call(tool_name="bayes_score", result=res, status="ok", session_id="cli-1", turn_id="t1")
    bayes = _cmd(store)
    assert "TRUE" in bayes("correct c1 true")                       # model id via last scored turn
    h = json.loads(res)["results"][0]["hash"]
    assert "FALSE" in bayes(f"correct {h} false")                   # footer hash
    assert "no scored claim" in bayes("correct zzzzzz true")
    assert "usage" in bayes("correct c1 maybe")
    assert "refit over 1 labelled" in bayes("refit")


def test_unknown_verb_prints_usage(store):
    assert _cmd(store)("dance").startswith("usage:")


def test_settings_validation_falls_back_to_defaults():
    vals = {"mode": "loud", "local_mode": "force_local", "min_posterior_plain": 0.3,
            "min_posterior_hedge": 0.6, "block_delegation_local": "yes"}
    s = load(lambda k, d: vals.get(k, d))
    assert s.mode == "annotate" and s.local_mode == "force_local"
    assert (s.min_posterior_plain, s.min_posterior_hedge) == (0.9, 0.5)
    assert s.block_delegation_local is True
