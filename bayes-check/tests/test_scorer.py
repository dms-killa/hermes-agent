import math

import pytest

from bayes_check.scorer import (
    DEFAULT_LR_TABLE, MAX_LOG_SHIFT, Evidence, LabelledClaim, Thresholds, beta_mean, beta_update,
    refit_base_rates, refit_lrs, score_claim, sigmoid, verdict_for,
)

T = Thresholds()


def ev(st, stance="supports", lr=None):
    return Evidence(source="s", source_type=st, stance=stance, lr_override=lr)


def test_zero_evidence_returns_prior():
    s = score_claim("c1", 0.7, [], T)
    assert s.posterior == pytest.approx(0.7)
    assert s.verdict == "uncertain" and s.action == "qualify"


def test_log_odds_update_matches_bayes_rule():
    # prior odds 1:1, LR 4 -> posterior 0.8 exactly
    s = score_claim("c1", 0.5, [ev("primary_source")], T)
    assert s.posterior == pytest.approx(4 / 5)


def test_independent_types_multiply():
    s = score_claim("c1", 0.5, [ev("primary_source"), ev("official_docs")], T)
    assert s.posterior == pytest.approx(12 / 13)
    assert s.verdict == "supported" and s.action == "state_plainly"


def test_same_type_evidence_is_discounted():
    once = score_claim("c", 0.5, [ev("web_search")], T).posterior
    twice = score_claim("c", 0.5, [ev("web_search"), ev("web_search")], T).posterior
    naive = sigmoid(2 * math.log(DEFAULT_LR_TABLE["web_search"]))
    assert once < twice < naive


def test_contradicting_evidence_lowers_posterior():
    s = score_claim("c", 0.5, [ev("peer_reviewed", stance="contradicts")], T)
    assert s.posterior == pytest.approx(1 / 4.5)
    assert s.verdict == "unsupported" and s.action == "remove"


def test_contradictory_evidence_cancels():
    s = score_claim("c", 0.6, [ev("official_docs"), ev("code_or_data", stance="contradicts")], T)
    assert s.posterior == pytest.approx(0.6)


@pytest.mark.parametrize("prior,expected", [(0.0, 0.01), (1.0, 0.99), (-3, 0.01), (7, 0.99)])
def test_prior_clamped_at_bounds(prior, expected):
    s = score_claim("c", prior, [], T)
    assert s.prior == expected and s.posterior == pytest.approx(expected)
    assert any("clamped" in n for n in s.notes)


def test_lr_override_clamped_and_shift_capped():
    s = score_claim("c", 0.5, [ev("anecdotal", lr=1e9)], T)
    assert any("clamped" in n for n in s.notes)
    many = [ev(t) for t in DEFAULT_LR_TABLE] + [ev("x", lr=20)] * 5
    capped = score_claim("c", 0.5, many, T)
    assert capped.log_odds_shift == pytest.approx(MAX_LOG_SHIFT)


def test_unknown_source_type_counts_as_no_evidence():
    s = score_claim("c", 0.4, [ev("vibes")], T)
    assert s.posterior == pytest.approx(0.4)
    assert any("unknown source_type" in n for n in s.notes)


def test_verdict_thresholds_boundaries():
    assert verdict_for(0.9, T) == "supported"
    assert verdict_for(0.8999, T) == "uncertain"
    assert verdict_for(0.5, T) == "uncertain"
    assert verdict_for(0.4999, T) == "unsupported"
    with pytest.raises(ValueError):
        Thresholds(plain=0.4, hedge=0.6)


def test_beta_update_and_mean():
    a, b = beta_update(1, 1, 7, 3)
    assert (a, b) == (8, 4) and beta_mean(a, b) == pytest.approx(2 / 3)
    with pytest.raises(ValueError):
        beta_update(0, 1, 1, 1)


def test_refit_learns_reliable_and_unreliable_sources():
    rows = ([LabelledClaim("stat", True, ("official_docs",))] * 8
            + [LabelledClaim("stat", False, ("official_docs",))] * 1
            + [LabelledClaim("stat", True, ("web_search",))] * 3
            + [LabelledClaim("stat", False, ("web_search",))] * 6)
    lrs = refit_lrs(rows)
    assert lrs["official_docs"]["lr"] > 1.5 > lrs["web_search"]["lr"]
    assert lrs["web_search"]["lr"] < 1.0


def test_refit_requires_minimum_labels():
    assert refit_lrs([LabelledClaim("x", True, ("tool_output",))] * 4) == {}


def test_refit_base_rates_are_beta_posteriors():
    out = refit_base_rates([LabelledClaim("date", True, ())] * 3 + [LabelledClaim("date", False, ())])
    assert out["date"]["alpha"] == 4 and out["date"]["beta"] == 2
    assert out["date"]["mean"] == pytest.approx(4 / 6)
