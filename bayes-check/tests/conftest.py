"""Load the plugin directory as package ``bayes_check`` (its dir name is not importable, and its
``tools.py`` must never shadow Hermes' ``tools`` package on sys.path)."""

from __future__ import annotations

import importlib.util
import os
import sqlite3
import sys
from pathlib import Path

import pytest

PLUGIN_DIR = Path(__file__).resolve().parents[1]
PKG = "bayes_check"

# Never let the plugin dir (or cwd == plugin dir) shadow Hermes' own top-level packages.
sys.path[:] = [p for p in sys.path if Path(p or os.getcwd()).resolve() != PLUGIN_DIR]


def _load_package():
    if PKG in sys.modules:
        return sys.modules[PKG]
    spec = importlib.util.spec_from_file_location(
        PKG, PLUGIN_DIR / "__init__.py", submodule_search_locations=[str(PLUGIN_DIR)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[PKG] = module
    spec.loader.exec_module(module)
    return module


bayes_check = _load_package()


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    """Fresh in-process ledger and an isolated HERMES_HOME for every test (never ~/.hermes)."""
    home = tmp_path / "hermes-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    for var in ("HERMES_SESSION_ID", "HERMES_SESSION_KEY"):
        monkeypatch.delenv(var, raising=False)
    from bayes_check import ledger
    ledger.reset()
    yield home
    ledger.reset()


@pytest.fixture
def store(tmp_path):
    from bayes_check.state import Store
    path = tmp_path / "ledger.db"
    s = Store(opener=lambda: sqlite3.connect(path, check_same_thread=False))
    yield s
    s.close()


class FakeSettings:
    def __init__(self, **overrides):
        from bayes_check.config import Settings
        self.value = Settings(**overrides)

    def __call__(self):
        return self.value


@pytest.fixture
def make_hooks(store):
    from bayes_check.hooks import Hooks

    def _make(**overrides):
        return Hooks(FakeSettings(**overrides), lambda: store)
    return _make
