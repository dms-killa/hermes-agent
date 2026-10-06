import pytest

from bayes_check import ledger
from bayes_check.local_mode import base_url_is_local, detect_local, provider_is_local, resolve_local


@pytest.mark.parametrize("url", [
    "http://localhost:11434/v1", "http://127.0.0.1:8080", "http://0.0.0.0:1234/v1", "http://[::1]:8000",
    "127.0.0.2:5000", "unix:/run/llama.sock", "http+unix://%2Frun%2Fx.sock/v1", "/var/run/vllm.sock",
    "http://box.localhost/v1",
])
def test_local_base_urls(url):
    assert base_url_is_local(url)


@pytest.mark.parametrize("url", [
    "https://api.openai.com/v1", "https://openrouter.ai/api/v1", "http://192.168.1.20:11434", "", "not a url",
])
def test_cloud_or_lan_base_urls(url):
    assert not base_url_is_local(url)


@pytest.mark.parametrize("provider", ["ollama", "llama.cpp", "LMStudio", "vllm", "custom_openai"])
def test_local_providers(provider):
    assert provider_is_local(provider)


def test_provider_or_url_is_enough():
    assert detect_local("ollama", "http://gpu-box.lan:11434")      # proxied ollama: provider wins
    assert detect_local("openai", "http://localhost:8000/v1")       # local server speaking openai
    assert not detect_local("anthropic", "https://api.anthropic.com")


@pytest.mark.parametrize("setting,detected,expected", [
    ("auto", True, True), ("auto", False, False), ("auto", None, False),
    ("force_local", False, True), ("force_local", None, True),
    ("force_cloud", True, False),
])
def test_override_both_directions(setting, detected, expected):
    assert resolve_local(setting, detected) is expected


def test_pre_api_request_caches_classification_per_session(make_hooks):
    hooks = make_hooks()
    hooks.pre_api_request(session_id="s1", provider="custom", base_url="http://127.0.0.1:1234/v1")
    hooks.pre_api_request(session_id="s2", provider="openrouter", base_url="https://openrouter.ai/api/v1")
    assert ledger.get_local("s1") is True and ledger.get_local("s2") is False
