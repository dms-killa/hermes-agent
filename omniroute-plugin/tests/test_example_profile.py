"""The shipped example profile parses, names combos (not backends) on every model setting, and its
MCP include-list holds only OmniRoute tools whose declared scopes are all read:* (needs Hermes'
YAML dependency)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.hermes
ruamel = pytest.importorskip("ruamel.yaml", reason="Hermes environment provides ruamel.yaml")

from hermes_omniroute.combo_check import collect_model_settings, points_at_gateway  # noqa: E402

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "hermes-profile.yaml"

# Read-only OmniRoute MCP tools: every declared scope is read:* in
# OmniRoute open-sse/mcp-server/schemas/tools.ts at the SHA in docs/CAPABILITIES.md.
READ_ONLY_MCP_TOOLS = {
    "omniroute_get_health", "omniroute_list_combos", "omniroute_get_combo_metrics",
    "omniroute_check_quota", "omniroute_cost_report", "omniroute_list_models_catalog",
    "omniroute_simulate_route", "omniroute_get_provider_metrics", "omniroute_best_combo_for_task",
    "omniroute_explain_route", "omniroute_get_session_snapshot", "omniroute_cache_stats",
    "omniroute_compression_status", "omniroute_list_compression_combos",
    "omniroute_compression_combo_stats",
}


@pytest.fixture
def example():
    return ruamel.YAML(typ="safe").load(EXAMPLE.read_text())


def test_every_model_setting_names_a_combo_at_the_gateway(example):
    settings = collect_model_settings(example)
    tasks = {s.task for s in settings}
    assert {"model", "auxiliary.title_generation", "auxiliary.compression", "delegation",
            "fallback_providers[0]"} <= tasks
    for s in settings:
        assert points_at_gateway(s, "https://omniroute.example.com"), s.task
        assert "/" not in s.model, f"{s.task} names a backend id, not a combo"


def test_mcp_include_list_is_read_only(example):
    server = example["mcp_servers"]["omniroute"]
    include = set(server["tools"]["include"])
    assert include and include <= READ_ONLY_MCP_TOOLS
    assert server["url"].endswith("/api/mcp/stream")
    assert server["headers"]["Authorization"] == "Bearer ${OMNIROUTE_API_KEY}"
