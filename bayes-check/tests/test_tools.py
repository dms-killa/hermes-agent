import json

import pytest

from bayes_check import ledger
from bayes_check.config import Settings
from bayes_check.schemas import MAX_CLAIMS, MAX_TEXT
from bayes_check.tools import make_handler


def claim(**over):
    base = {"id": "c1", "text": "Python 3.11 was released in October 2022.", "claim_type": "date",
            "prior": 0.6, "evidence": [{"source": "python.org", "source_type": "official_docs"}]}
    base.update(over)
    return base


@pytest.fixture
def handler(store):
    return make_handler(lambda: Settings(), lambda: store)


def call(handler, args, **kw):
    return json.loads(handler(args, session_id=kw.pop("session_id", "s1"), **kw))


@pytest.mark.parametrize("args,needle", [
    ("not a dict", "arguments must be an object"),
    ({}, "claims must be a non-empty array"),
    ({"claims": []}, "non-empty"),
    ({"claims": "{oops"}, "not valid JSON"),
    ({"claims": [1]}, "claims[0] must be an object"),
    ({"claims": [claim(id="")]}, "claims[0].id"),
    ({"claims": [claim(), claim()]}, "duplicated"),
    ({"claims": [claim(prior=1.5)]}, "within [0, 1]"),
    ({"claims": [claim(prior=True)]}, "must be a number"),
    ({"claims": [claim(evidence="docs")]}, "evidence must be an array"),
    ({"claims": [claim(evidence=[{"source": "x"}])]}, "source_type"),
    ({"claims": [claim(evidence=[{"source": "x", "source_type": "web_search", "stance": "meh"}])]}, "stance"),
    ({"claims": [claim(evidence=[{"source": "x", "source_type": "web_search", "lr_override": 0}])]}, "> 0"),
    ({"claims": [claim(text="x" * (MAX_TEXT + 1))]}, "exceeds"),
    ({"claims": [claim(id=f"c{i}") for i in range(MAX_CLAIMS + 1)]}, "too many claims"),
])
def test_malformed_input_returns_error_json_never_raises(handler, args, needle):
    out = call(handler, args)
    assert "error" in out and needle in out["error"]
    assert ledger._pending == {}


def test_valid_call_scores_stashes_and_persists(handler, store):
    out = call(handler, {"claims": [claim(), claim(id="c2", prior=None, evidence=[])]})
    assert out["call_id"] and [r["id"] for r in out["results"]] == ["c1", "c2"]
    r1, r2 = out["results"]
    assert r1["posterior"] == pytest.approx(0.6 * 3 / (0.6 * 3 + 0.4), abs=1e-4)
    assert r2["prior"] == 0.5 and r2["verdict"] == "uncertain"
    assert out["call_id"] in ledger._pending
    assert store.stats()["claims"] == 2


def test_double_encoded_claims_string_is_accepted(handler):
    out = call(handler, {"claims": json.dumps([claim()])})
    assert out["results"][0]["id"] == "c1"


def test_learned_base_rate_used_when_prior_omitted(store):
    rows = [{"id": f"c{i}", "hash": f"{i:08x}", "text": "t", "claim_type": "date", "prior": .5,
             "posterior": .5, "verdict": "uncertain", "evidence": []} for i in range(4)]
    store.append_claims("s", "c", rows)
    for i in range(4):
        store.label(f"{i:08x}", True)
    store.refit()
    out = call(make_handler(lambda: Settings(), lambda: store), {"claims": [claim(prior=None, evidence=[])]})
    assert out["results"][0]["prior"] == pytest.approx(5 / 6, abs=1e-4)


def test_store_failure_degrades_to_error_json_not_exception():
    class Broken:
        def lr_table(self, d):
            raise RuntimeError("disk gone")
    out = json.loads(make_handler(lambda: Settings(), lambda: Broken())({"claims": [claim()]}))
    assert "internal error" in out["error"]


def test_works_without_a_store():
    out = json.loads(make_handler(lambda: Settings(), lambda: None)({"claims": [claim()]}, session_id="s"))
    assert out["results"][0]["verdict"] in {"supported", "uncertain", "unsupported"}
