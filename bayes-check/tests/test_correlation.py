"""Turn ledger + hook flow: opt-in, correlation by turn_id, the gate, subagents, loud failure."""

import json

import pytest

from bayes_check import ledger
from bayes_check.config import Settings
from bayes_check.footer import has_footer
from bayes_check.tools import make_handler

ANSWER = "The Eiffel Tower is 330 metres tall. It was completed in 1889. Paris is lovely."


def _score(store, session_id="s1"):
    handler = make_handler(lambda: Settings(), lambda: store)
    args = {"claims": [{"id": "c1", "text": "The Eiffel Tower is 330 metres tall.", "claim_type": "statistic",
                        "prior": 0.7, "evidence": [{"source": "toureiffel.paris", "source_type": "official_docs"}]}]}
    return handler(args, session_id=session_id)


def test_not_opted_in_is_a_noop(make_hooks):
    hooks = make_hooks()
    assert hooks.pre_llm_call(session_id="s1", turn_id="t1", user_message="hi") is None
    assert hooks.transform_llm_output(response_text=ANSWER, session_id="s1", turn_id="t1") is None


def test_full_scored_turn(make_hooks, store):
    hooks = make_hooks()
    ledger.enable("s1")
    ctx = hooks.pre_llm_call(session_id="s1", turn_id="t1", user_message="how tall is the eiffel tower")
    assert "bayes_score" in ctx["context"] and "bayes-check:bayes-check" in ctx["context"]
    result = _score(store)
    hooks.post_tool_call(tool_name="bayes_score", result=result, status="ok", session_id="s1", turn_id="t1")
    out = hooks.transform_llm_output(response_text=ANSWER, session_id="s1", turn_id="t1")
    assert out.startswith(ANSWER) and "bayes-checked:" in out and "c1·" in out
    assert "unscored" not in out
    assert store._connect().execute("SELECT turn_id FROM claims").fetchone()[0] == "t1"
    hooks.post_llm_call(session_id="s1", turn_id="t1", assistant_response=out)
    assert ledger.gate_skips("s1") == 0


@pytest.mark.parametrize("status", ["error", "blocked", "cancelled"])
def test_non_ok_tool_status_does_not_satisfy_gate(make_hooks, store, status):
    hooks = make_hooks()
    ledger.enable("s1")
    hooks.post_tool_call(tool_name="bayes_score", result=_score(store), status=status, session_id="s1", turn_id="t1")
    out = hooks.transform_llm_output(response_text=ANSWER, session_id="s1", turn_id="t1")
    assert "⚠ bayes-check: unscored" in out


def test_scoring_in_another_turn_does_not_leak(make_hooks, store):
    hooks = make_hooks()
    ledger.enable("s1")
    hooks.post_tool_call(tool_name="bayes_score", result=_score(store), status="ok", session_id="s1", turn_id="t1")
    out = hooks.transform_llm_output(response_text=ANSWER, session_id="s1", turn_id="t2")
    assert "unscored" in out


def test_other_tools_ignored_and_forged_call_id_rejected(make_hooks):
    hooks = make_hooks()
    ledger.enable("s1")
    hooks.post_tool_call(tool_name="web_search", result="{}", status="ok", session_id="s1", turn_id="t1")
    hooks.post_tool_call(tool_name="bayes_score", result=json.dumps({"call_id": "forged"}), status="ok",
                         session_id="s1", turn_id="t1")
    assert not ledger.turn("t1").scored


def test_annotate_lists_candidates_nudge_only_marks(make_hooks):
    ledger.enable("s1")
    annotated = make_hooks(mode="annotate").transform_llm_output(response_text=ANSWER, session_id="s1", turn_id="a")
    assert "330 metres" in annotated and "Paris is lovely" not in annotated.split("---")[-1]
    nudged = make_hooks(mode="nudge").transform_llm_output(response_text=ANSWER, session_id="s1", turn_id="b")
    assert "unscored" in nudged and "330 metres" not in nudged.split("---")[-1]


def test_mode_off_disables_even_opted_in_sessions(make_hooks):
    ledger.enable("s1")
    hooks = make_hooks(mode="off")
    assert hooks.pre_llm_call(session_id="s1", turn_id="t") is None
    assert hooks.transform_llm_output(response_text=ANSWER, session_id="s1", turn_id="t") is None


def test_transform_is_idempotent_on_already_annotated_text(make_hooks):
    ledger.enable("s1")
    hooks = make_hooks()
    once = hooks.transform_llm_output(response_text=ANSWER, session_id="s1", turn_id="t")
    assert hooks.transform_llm_output(response_text=once, session_id="s1", turn_id="t") is None


def test_internal_error_fails_loudly_not_silently(make_hooks, monkeypatch):
    ledger.enable("s1")
    hooks = make_hooks()
    monkeypatch.setattr("bayes_check.hooks.candidate_claims", lambda _t: 1 / 0)
    out = hooks.transform_llm_output(response_text=ANSWER, session_id="s1", turn_id="t")
    assert out.startswith(ANSWER) and "gate error (ZeroDivisionError)" in out and "NOT checked" in out


def test_skipped_gate_is_detected_after_the_fact(make_hooks, store):
    """Host timeout abandons the transform: post_llm_call sees an opted-in, untransformed turn."""
    ledger.enable("s1")
    hooks = make_hooks()
    hooks.pre_llm_call(session_id="s1", turn_id="t1", user_message="q")
    hooks.post_llm_call(session_id="s1", turn_id="t1", assistant_response=ANSWER)
    assert ledger.gate_skips("s1") == 1 and store.stats()["gate_skips"] == 1


def test_session_key_opt_in_pins_concrete_session(make_hooks, monkeypatch):
    """Gateway commands know only HERMES_SESSION_KEY; hooks see session ids (DESIGN D1)."""
    ledger.enable("agent:main:telegram:dm:42")
    monkeypatch.setenv("HERMES_SESSION_KEY", "agent:main:telegram:dm:42")
    hooks = make_hooks()
    assert hooks.pre_llm_call(session_id="sess-9", turn_id="t1", user_message="q") is not None
    monkeypatch.delenv("HERMES_SESSION_KEY")
    assert ledger.is_enabled("sess-9")
    assert ledger.session_for_key("agent:main:telegram:dm:42") == "sess-9"


def test_auto_enable_on_research(make_hooks):
    hooks = make_hooks(auto_enable_on_research=True)
    ctx = hooks.pre_llm_call(session_id="s1", turn_id="t", user_message="What percentage of EVs are sold in Norway, with sources?")
    assert ctx and "enabled automatically" in ctx["context"]
    assert hooks.pre_llm_call(session_id="s2", turn_id="u", user_message="thanks!") is None


# --- subagents --------------------------------------------------------------------------------

def test_child_inherits_opt_in_and_unverified_child_is_reported(make_hooks, store):
    ledger.enable("parent")
    hooks = make_hooks()
    hooks.subagent_start(parent_session_id="parent", parent_turn_id="pt", child_session_id="kid-a")
    hooks.subagent_start(parent_session_id="parent", parent_turn_id="pt", child_session_id="kid-b")
    assert ledger.is_enabled("kid-a")
    # kid-a scores in its own turn; kid-b does not
    hooks.post_tool_call(tool_name="bayes_score", result=_score(store, "kid-a"), status="ok",
                         session_id="kid-a", turn_id="kid-a-t1")
    hooks.subagent_stop(child_session_id="kid-a", child_status="completed",
                        tool_call_history=[{"tool_name": "bayes_score", "status": "ok"}])
    hooks.subagent_stop(child_session_id="kid-b", child_status="completed", tool_call_history=[])
    out = hooks.transform_llm_output(response_text=ANSWER, session_id="parent", turn_id="pt")
    assert "unverified subagent output from 1 child session(s): kid-b" in out


def test_children_of_non_opted_parent_are_not_tracked(make_hooks):
    hooks = make_hooks()
    hooks.subagent_start(parent_session_id="p", parent_turn_id="pt", child_session_id="kid")
    assert not ledger.is_enabled("kid") and ledger.unverified_children("pt") == []


def test_background_completion_closes_ledger_on_next_parent_turn(make_hooks):
    """subagent_stop for a background child fires after the dispatching turn (DESIGN D5)."""
    ledger.enable("parent")
    hooks = make_hooks()
    hooks.subagent_start(parent_session_id="parent", parent_turn_id="t1", child_session_id="bg")
    hooks.subagent_stop(child_session_id="bg", child_status="completed", tool_call_history=[],
                        parent_turn_id="t5")   # live parent turn id at completion time: ignored
    ctx = hooks.pre_llm_call(session_id="parent", turn_id="t6", user_message="[async delegation result]")
    assert "were NOT bayes-checked" in ctx["context"]
    assert ledger.open_children("parent") == []


def test_turn_ledger_is_bounded():
    for i in range(ledger.MAX_TURNS + 50):
        ledger.turn(f"t{i}", "s")
    assert len(ledger._turn_ledger) == ledger.MAX_TURNS and ledger.peek_turn("t0") is None
