"""L10 evidence: Hermes' own classifier on OmniRoute's exact 404 body (needs a Hermes env).

At the recorded SHAs Hermes already classifies OmniRoute's combo-exhausted 404 as
model_not_found (fallback fires) because OmniRoute's body carries ``code: "model_not_found"``
and Hermes matches that token. So this plugin ships no transform_api_error_classification hook
(docs/DESIGN.md D4). These tests pin that verdict: if either side changes, they fail and the hook
decision must be revisited.
"""

from __future__ import annotations

import json

import pytest

from mock_server import FIXTURES

pytestmark = pytest.mark.hermes
classifier = pytest.importorskip("agent.error_classifier", reason="needs a Hermes checkout/env")
httpx = pytest.importorskip("httpx")
openai = pytest.importorskip("openai")


def _sdk_error(body):
    request = httpx.Request("POST", "http://gateway.invalid/v1/chat/completions")
    response = httpx.Response(404, request=request, json=body)
    return openai.NotFoundError(message=f"Error code: 404 - {json.dumps(body)}",
                                response=response, body=body)


@pytest.fixture
def exact_body():
    return json.loads((FIXTURES / "omniroute_404_combo_exhausted.json").read_text())


def test_exact_omniroute_404_is_model_not_found(exact_body):
    result = classifier.classify_api_error(_sdk_error(exact_body), provider="custom", model="some-combo")
    assert result.reason == classifier.FailoverReason.model_not_found
    assert result.retryable is False and result.should_fallback is True


def test_message_alone_would_not_match(exact_body):
    """Documents WHY it works: the message text itself ('model — Not Found (HTTP 404)') matches no
    Hermes pattern; the body's code field does. Without the code the verdict is unknown."""
    body = {"error": {k: v for k, v in exact_body["error"].items() if k != "code"}}
    result = classifier.classify_api_error(_sdk_error(body), provider="custom", model="some-combo")
    assert result.reason == classifier.FailoverReason.unknown


def test_no_error_classification_hook_declared():
    import hermes_omniroute
    from pathlib import Path
    manifest = Path(hermes_omniroute.__file__).with_name("plugin.yaml").read_text()
    assert "transform_api_error_classification" not in manifest
    hooks = manifest.split("provides_hooks:", 1)[1]
    assert [h.strip(" -") for h in hooks.strip().splitlines()] == ["pre_tool_call"]   # approval only (v0.2)
