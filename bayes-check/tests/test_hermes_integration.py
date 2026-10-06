"""Real-runtime checks: the plugin is installed into a temp HERMES_HOME, enabled in config.yaml,
loaded by Hermes' own discovery, and driven through Hermes' own dispatch/hook code paths.

These are the GOAL.md VERIFY-IN-DEV checks (plugin-tool post_tool_call with turn_id; subagent_stop
on the background finalizer) plus the gate, veto, command, skill and prompt section end to end.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PLUGIN_DIR = Path(__file__).resolve().parents[1]

hermes_plugins = pytest.importorskip("hermes_cli.plugins", reason="needs a Hermes checkout/env")


@pytest.fixture
def hermes(_isolated):
    home: Path = _isolated
    shutil.copytree(PLUGIN_DIR, home / "plugins" / "bayes-check",
                    ignore=shutil.ignore_patterns("tests", "__pycache__", "*.pyc"))
    (home / "config.yaml").write_text("plugins:\n  enabled:\n    - bayes-check\n", encoding="utf-8")
    hermes_plugins._reset_plugin_managers_for_tests()
    hermes_plugins.discover_plugins(force=True)
    manager = hermes_plugins.get_plugin_manager()
    loaded = manager._plugins.get("bayes-check")
    assert loaded is not None and loaded.enabled and not loaded.error, getattr(loaded, "error", "not loaded")
    mod = next(m for name, m in sys.modules.items()
               if name.startswith("hermes_plugins.") and name.endswith(".ledger") and "bayes" in name)
    yield SimpleNamespace(manager=manager, ledger=mod, home=home)
    hermes_plugins._reset_plugin_managers_for_tests()


CLAIMS = {"claims": [{"id": "c1", "text": "The Moon is about 384,400 km from Earth.", "claim_type": "statistic",
                      "prior": 0.8, "evidence": [{"source": "nasa.gov", "source_type": "official_docs"}]}]}


def _agent(session_id, turn_id):
    return SimpleNamespace(session_id=session_id, _current_turn_id=turn_id, _current_api_request_id="",
                           model="test-model", platform="cli", _persist_disabled=False)


def test_registrations_visible_to_hermes(hermes):
    from tools.registry import registry
    assert registry.get_entry("bayes_score", scope=hermes.manager.scope_key) is not None
    assert hermes_plugins.get_plugin_command_handler("bayes") is not None
    assert hermes.manager.find_plugin_skill("bayes-check:bayes-check") is not None      # DESIGN D2
    sections = hermes_plugins.render_system_prompt_sections(
        {"session_id": "s", "model": "m", "provider": "p", "platform": "cli", "profile_name": "default", "cwd": "/"})
    assert any("bayes_score" in getattr(s, "content", str(s)) for s in sections)


def test_verify_1_plugin_tool_emits_post_tool_call_with_turn_id_model_tools_path(hermes):
    """VERIFY-IN-DEV 1, dispatcher path: model_tools.handle_function_call -> registry -> post hook."""
    import model_tools
    hermes.ledger.enable("sess-A")
    raw = model_tools.handle_function_call("bayes_score", CLAIMS, task_id="task", session_id="sess-A",
                                           turn_id="sess-A:task:turn0001", tool_call_id="call-1")
    assert "call_id" in json.loads(raw)
    entry = hermes.ledger.peek_turn("sess-A:task:turn0001")
    assert entry is not None and entry.scored and entry.results[0]["id"] == "c1"


def test_verify_1_executor_terminal_emitter_carries_agent_turn_id(hermes):
    """VERIFY-IN-DEV 1, agent-loop path: the executor suppresses the inner hook and emits the terminal
    post_tool_call from agent._current_turn_id (agent/inline_tool_executors.py)."""
    import model_tools
    from agent.inline_tool_executors import emit_terminal_post_tool_call
    with model_tools.suppress_post_tool_call_hook():
        raw = model_tools.handle_function_call("bayes_score", CLAIMS, task_id="task", session_id="sess-B")
    assert hermes.ledger.peek_turn("sess-B:task:turn0002") is None
    emit_terminal_post_tool_call(_agent("sess-B", "sess-B:task:turn0002"), function_name="bayes_score",
                                 function_args=CLAIMS, result=raw, effective_task_id="task", tool_call_id="call-2")
    assert hermes.ledger.peek_turn("sess-B:task:turn0002").scored


def test_gate_through_hermes_transform_seam(hermes):
    import model_tools
    from agent.turn_finalizer import apply_llm_output_transform
    hermes.ledger.enable("sess-C")
    model_tools.handle_function_call("bayes_score", CLAIMS, task_id="t", session_id="sess-C", turn_id="tc1")
    agent = _agent("sess-C", "tc1")
    text, transformed, original = apply_llm_output_transform(agent, "The Moon is about 384,400 km away.", turn_id="tc1")
    assert transformed and original and "bayes-checked:" in text
    # Idempotent per turn: a second seam gets the recorded outcome, no second footer.
    again, _, _ = apply_llm_output_transform(agent, text, turn_id="tc1")
    assert again.count("bayes-checked:") == 1
    # Not opted in -> untouched.
    plain, transformed2, _ = apply_llm_output_transform(_agent("other", "x1"), "hello", turn_id="x1")
    assert plain == "hello" and not transformed2


def test_pre_tool_call_veto_through_hermes_resolver(hermes):
    hermes.ledger.enable("sess-D")
    hermes_plugins.invoke_hook("pre_api_request", session_id="sess-D", provider="ollama",
                               base_url="http://127.0.0.1:11434/v1")
    msg = hermes_plugins.resolve_pre_tool_block("delegate_task", {"goal": "x"}, session_id="sess-D")
    assert msg and "blocked" in msg
    assert hermes_plugins.resolve_pre_tool_block("delegate_task", {"goal": "x"}, session_id="cloud-sess") is None
    assert hermes_plugins.resolve_pre_tool_block("read_file", {"path": "x"}, session_id="sess-D") is None


def test_verify_2_subagent_stop_fires_from_background_finalizer(hermes):
    """VERIFY-IN-DEV 2: background units run _execute_and_aggregate -> _finalize_child_results ->
    _fire_subagent_stop_hooks (tools/delegate_tool_dispatch.py). Drive that shared finalizer."""
    from tools.delegate_tool_results import _finalize_child_results
    hermes.ledger.enable("parent")
    hermes_plugins.invoke_hook("subagent_start", parent_session_id="parent", parent_turn_id="pt-dispatch",
                               child_session_id="child-1", child_subagent_id="sa1", child_role="leaf", child_goal="g")
    parent = SimpleNamespace(session_id="parent", _current_turn_id="pt-LATER", _memory_manager=None)
    child = SimpleNamespace(session_id="child-1")
    results = [{"task_index": 0, "status": "completed", "summary": "done", "duration_seconds": 1.0,
                "tool_trace": [{"tool": "web_search", "status": "ok"}]}]
    _finalize_child_results(results, [{"goal": "g"}], [(0, {"goal": "g"}, child)], parent)
    entry = hermes.ledger._subagent_ledger["child-1"]
    assert entry.stopped and not entry.scored and entry.parent_turn_id == "pt-dispatch"   # DESIGN D5
    assert hermes.ledger.unverified_children("pt-dispatch") == ["child-1"]


def test_command_handler_runs_under_hermes_lookup(hermes, monkeypatch):
    monkeypatch.setenv("HERMES_SESSION_ID", "cli-sess")
    handler = hermes_plugins.get_plugin_command_handler("bayes")
    assert "ON" in hermes_plugins.resolve_plugin_command_result(handler("on"))
    assert hermes.ledger.is_enabled("cli-sess")
    assert "ledger:" in handler("status")


def test_ledger_db_lives_in_profile_plugin_data(hermes):
    import model_tools
    model_tools.handle_function_call("bayes_score", CLAIMS, task_id="t", session_id="s", turn_id="t9")
    assert (hermes.home / "plugin-data" / "bayes-check" / "data.db").exists()


def test_host_timeout_on_transform_is_detected_loudly(hermes, monkeypatch, caplog):
    """The host abandons a transform that overruns plugins.hook_callback_timeout and delivers raw text.
    The plugin cannot write a marker from an abandoned worker, so post_llm_call must flag it."""
    import time
    from agent.turn_finalizer import apply_llm_output_transform
    with (hermes.home / "config.yaml").open("a", encoding="utf-8") as fh:
        fh.write("  hook_callback_timeout: 0.3\n")
    hooks_mod = next(m for name, m in sys.modules.items()
                     if name.startswith("hermes_plugins.") and name.endswith(".hooks") and "bayes" in name)
    monkeypatch.setattr(hooks_mod, "unscored_footer", lambda *a, **k: time.sleep(1.5) or "late")
    hermes.ledger.enable("sess-T")
    hermes_plugins.invoke_hook("pre_llm_call", session_id="sess-T", turn_id="tt1", user_message="q")
    text, transformed, _ = apply_llm_output_transform(_agent("sess-T", "tt1"), "Raw answer 42.", turn_id="tt1")
    assert text == "Raw answer 42." and not transformed           # host failed open
    with caplog.at_level("WARNING"):
        hermes_plugins.invoke_hook("post_llm_call", session_id="sess-T", turn_id="tt1", assistant_response=text)
    assert hermes.ledger.gate_skips("sess-T") == 1
    assert any("gate did NOT run" in r.getMessage() for r in caplog.records)
