"""Real-runtime checks: install into a temp HERMES_HOME, enable in config.yaml, load through Hermes'
own discovery, and dispatch tools through Hermes' own dispatcher (which runs the pre_tool_call
approval gate) against the stateful mock."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from mock_server import TEST_KEY

pytestmark = pytest.mark.hermes
hermes_plugins = pytest.importorskip("hermes_cli.plugins", reason="needs a Hermes checkout/env")

PLUGIN_DIR = Path(__file__).resolve().parents[1]
READ = ("omniroute_audit", "hermes_combo_check", "omniroute_connections", "omniroute_combos",
        "omniroute_call_logs", "omniroute_combo_health", "omniroute_combo_plan")
WRITE = ("omniroute_combo_apply", "omniroute_combo_rollback")
SPILL = "00000000-0000-4000-8000-00000000c003"


@pytest.fixture
def hermes(tmp_path, monkeypatch, omniroute):
    home = tmp_path / "hermes-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("OMNIROUTE_BASE_URL", omniroute.url)
    monkeypatch.setenv("OMNIROUTE_API_KEY", TEST_KEY)
    for var in ("HERMES_INTERACTIVE", "HERMES_GATEWAY_SESSION", "HERMES_EXEC_ASK", "HERMES_YOLO_MODE"):
        monkeypatch.delenv(var, raising=False)
    shutil.copytree(PLUGIN_DIR, home / "plugins" / "omniroute",
                    ignore=shutil.ignore_patterns("tests", "docs", "examples", ".git*", "__pycache__"))
    (home / "config.yaml").write_text(
        "plugins:\n  enabled:\n    - omniroute\n"
        f"model:\n  default: structured-local\n  provider: custom\n  base_url: {omniroute.url}/v1\n",
        encoding="utf-8")
    hermes_plugins._reset_plugin_managers_for_tests()
    hermes_plugins.discover_plugins(force=True)
    manager = hermes_plugins.get_plugin_manager()
    loaded = manager._plugins.get("omniroute")
    assert loaded is not None and loaded.enabled and not loaded.error, getattr(loaded, "error", "not loaded")
    yield manager
    from tools import terminal_tool
    terminal_tool.set_approval_callback(None)
    hermes_plugins._reset_plugin_managers_for_tests()


@pytest.fixture
def human(monkeypatch):
    """Stand-in for the person at the CLI prompt: the same callback slot Hermes's CLI registers
    (tools.terminal_tool.set_approval_callback) under HERMES_INTERACTIVE."""
    from tools import terminal_tool
    seen = []

    def answer(choice):
        monkeypatch.setenv("HERMES_INTERACTIVE", "1")

        def cb(command, description, **_kw):
            seen.append({"command": command, "description": description})
            return choice
        terminal_tool.set_approval_callback(cb)
        return seen
    return answer


def call(name, args):
    import model_tools
    return json.loads(model_tools.handle_function_call(name, args, task_id="t", session_id="s"))


def test_toolsets_skill_and_hook_registered(hermes):
    from tools.registry import registry
    toolsets = {key: desc for key, _label, desc in hermes_plugins.get_plugin_toolsets()}
    assert {"omniroute", "omniroute_write"} <= set(toolsets)
    for name in READ:
        assert registry.get_entry(name, scope=hermes.scope_key).toolset == "omniroute"
    for name in WRITE:
        assert registry.get_entry(name, scope=hermes.scope_key).toolset == "omniroute_write"
    assert hermes.find_plugin_skill("omniroute:omniroute-routing") is not None
    hooks = {name for name, cbs in hermes._hooks.items() for cb in cbs
             if "omniroute" in (getattr(cb, "__module__", "") or "")}
    assert hooks == {"pre_tool_call"}            # approval only; no error-classification hook (D4)


def test_write_toolset_follows_hermes_toolset_config(hermes):
    from hermes_cli.tools_config import _get_platform_tools
    default = _get_platform_tools({}, "cli")
    assert {"omniroute", "omniroute_write"} <= default     # plugin toolsets default ON (H-GATE-1 gap)
    disabled = _get_platform_tools({"platform_toolsets": {"cli": ["hermes-cli", "omniroute"]},
                                    "known_plugin_toolsets": {"cli": ["omniroute", "omniroute_write"]}}, "cli")
    assert "omniroute" in disabled and "omniroute_write" not in disabled


def test_apply_without_a_human_fails_closed(hermes, omniroute):
    plan = call("omniroute_combo_plan", {"combo": "spill", "changes": {"description": "x"}})
    out = call("omniroute_combo_apply", {"plan_id": plan["plan_id"]})
    assert "BLOCKED" in json.dumps(out) and omniroute.writes() == []


def test_human_deny_blocks_and_plan_needs_no_approval(hermes, omniroute, human):
    seen = human("deny")
    plan = call("omniroute_combo_plan", {"combo": "spill", "changes": {"description": "x"}})
    assert plan["applicable"] and seen == []                     # read-only planner: no prompt
    out = call("omniroute_combo_apply", {"plan_id": plan["plan_id"]})
    assert "BLOCKED" in json.dumps(out) and omniroute.writes() == []
    assert len(seen) == 1 and "spill" in seen[0]["description"]


def test_plan_apply_rollback_through_hermes_dispatcher(hermes, omniroute, human):
    seen = human("once")
    before = omniroute.store.get(SPILL)
    plan = call("omniroute_combo_plan", {"combo": "spill", "changes": {"config": {"targetTimeoutMs": 300000}},
                                         "caller_limits": {"per_call_timeout_ms": 300000, "concurrency": 4}})
    assert plan["diff"] == [{"path": "config.targetTimeoutMs", "before": 600000, "after": 300000}]
    applied = call("omniroute_combo_apply", {"plan_id": plan["plan_id"]})
    assert applied["applied"] and applied["verification"] == "passed"
    rolled = call("omniroute_combo_rollback", {"rollback_id": applied["rollback_id"]})
    assert rolled["rolled_back"] and rolled["verification"] == "passed"
    after = omniroute.store.get(SPILL)
    assert {k: v for k, v in after.items() if k != "updatedAt"} == {k: v for k, v in before.items()
                                                                    if k != "updatedAt"}
    assert len(seen) == 2                                         # apply and rollback each asked once
    writes = omniroute.writes()
    assert [(w["method"], w["path"]) for w in writes] == [("PUT", f"/api/combos/{SPILL}")] * 2
    # plans and rollback records live in Hermes plugin storage for this profile
    states = list((Path(hermes_plugins.get_hermes_home()) / "plugin-data").glob("*omniroute*/state.json"))
    assert len(states) == 1 and plan["plan_id"] in states[0].read_text()


def test_combo_check_reads_real_hermes_config(hermes):
    report = call("hermes_combo_check", {})
    rows = {r["task"]: r for r in report["settings"]}
    assert rows["model"]["points_at_omniroute"] and rows["model"]["combo_found"]
    assert any(f["code"] == "STRUCTURED_OUTPUT_ON_L1_GATED_LEG" for f in report["findings"])
