"""Optional live smoke test against a real OmniRoute. Runs only when OMNIROUTE_BASE_URL and
OMNIROUTE_API_KEY are both set; skipped otherwise and never required by CI. Read-only: it uses the
same allowlisted GETs as the tools and does not fetch call-log bodies."""

from __future__ import annotations

import json
import os

import pytest

pytestmark = [pytest.mark.live, pytest.mark.skipif(
    not (os.environ.get("OMNIROUTE_BASE_URL") and os.environ.get("OMNIROUTE_API_KEY")),
    reason="set OMNIROUTE_BASE_URL and OMNIROUTE_API_KEY (manage scope) to run")]


@pytest.fixture(autouse=True)
def _keep_ambient_proxy_rules():
    """Override conftest's loopback NO_PROXY only for the mock; a live gateway uses the env as-is."""
    yield


def test_live_audit_and_inventory():
    from hermes_omniroute.tools import make_handlers
    handlers = make_handlers(config_loader=lambda: {})
    audit = json.loads(handlers["omniroute_audit"]({}))
    assert "error" not in audit, audit.get("error")
    assert audit["report"] == "omniroute_audit" and isinstance(audit["combos"], list)
    conns = json.loads(handlers["omniroute_connections"]({}))
    assert "error" not in conns and os.environ["OMNIROUTE_API_KEY"] not in json.dumps(conns)
