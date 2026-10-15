"""Load the plugin directory as package ``hermes_omniroute`` (the directory name is not a valid
module name) and keep it off sys.path, so its ``tools.py`` never shadows Hermes' ``tools``."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

PLUGIN_DIR = Path(__file__).resolve().parents[1]
TESTS_DIR = Path(__file__).resolve().parent
PKG = "hermes_omniroute"

sys.path[:] = [p for p in sys.path if Path(p or os.getcwd()).resolve() != PLUGIN_DIR]
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))


def _load_package():
    if PKG in sys.modules:
        return sys.modules[PKG]
    spec = importlib.util.spec_from_file_location(
        PKG, PLUGIN_DIR / "__init__.py", submodule_search_locations=[str(PLUGIN_DIR)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[PKG] = module
    spec.loader.exec_module(module)
    return module


plugin = _load_package()

from mock_server import FIXTURES, TEST_KEY, MockOmniRoute  # noqa: E402


@pytest.fixture(autouse=True)
def _no_proxy_for_loopback(monkeypatch):
    # The mock server is on 127.0.0.1; never route it through an ambient HTTP(S) proxy.
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")


@pytest.fixture
def omniroute():
    """A running strict mock. Fails the test on any recorded violation it did not consume."""
    from hermes_omniroute import endpoints
    with MockOmniRoute(endpoints) as mock:
        yield mock
        leftover = mock.take_violations()
    assert not leftover, f"mock OmniRoute recorded violations: {leftover}"


@pytest.fixture
def client(omniroute):
    from hermes_omniroute.client import OmniRouteClient
    from hermes_omniroute.settings import load_settings
    return OmniRouteClient(load_settings(base_url=omniroute.url, api_key=TEST_KEY, timeout_s=5))


@pytest.fixture
def fixture_json():
    return lambda name: json.loads((FIXTURES / name).read_text())


@pytest.fixture
def snapshot(fixture_json):
    from hermes_omniroute.audit import build_snapshot
    return build_snapshot(fixture_json("connections.json"), fixture_json("provider_nodes.json"),
                          fixture_json("combos.json"), fixture_json("combo_defaults.json"))
