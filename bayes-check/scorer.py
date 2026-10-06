"""Pure Bayesian scoring: log-odds updating, Beta posteriors, verdicts, refit estimators.

No I/O and no model calls — every number the plugin reports comes from here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

PRIOR_MIN, PRIOR_MAX = 0.01, 0.99
LR_MIN, LR_MAX = 0.05, 20.0
# Cap on the total |log-odds| shift evidence may apply to one claim (DESIGN N4).
MAX_LOG_SHIFT = math.log(100.0)
# Geometric discount for repeated evidence of the same source type (correlated sources).
SAME_TYPE_DISCOUNT = 0.5

# Supporting-direction likelihood ratios P(E|claim true) / P(E|claim false) (DESIGN N3).
DEFAULT_LR_TABLE: Dict[str, float] = {
    "primary_source": 4.0,
    "peer_reviewed": 3.5,
    "official_docs": 3.0,
    "code_or_data": 3.0,
    "tool_output": 2.5,
    "reputable_secondary": 2.0,
    "web_search": 1.5,
    "user_provided": 1.5,
    "model_knowledge": 1.2,
    "anecdotal": 1.1,
    "none": 1.0,
}

SUPPORTED, UNCERTAIN, UNSUPPORTED = "supported", "uncertain", "unsupported"
ACTIONS = {SUPPORTED: "state_plainly", UNCERTAIN: "qualify", UNSUPPORTED: "remove"}


@dataclass(frozen=True)
class Thresholds:
    plain: float = 0.9   # posterior >= plain  -> supported / state_plainly
    hedge: float = 0.5   # posterior <  hedge  -> unsupported / remove

    def __post_init__(self) -> None:
        if not 0.0 < self.hedge <= self.plain < 1.0:
            raise ValueError("thresholds must satisfy 0 < min_posterior_hedge <= min_posterior_plain < 1")


@dataclass(frozen=True)
class Evidence:
    source: str
    source_type: str
    stance: str = "supports"           # supports | contradicts  (DESIGN N2)
    lr_override: Optional[float] = None


@dataclass
class ClaimScore:
    id: str
    posterior: float
    verdict: str
    action: str
    prior: float
    log_odds_shift: float
    notes: List[str] = field(default_factory=list)


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def evidence_lr(item: Evidence, table: Mapping[str, float]) -> Tuple[float, Optional[str]]:
    """Directional LR for one evidence item plus an optional note (unknown type, clamped override)."""
    note = None
    if item.lr_override is not None:
        lr = clamp(float(item.lr_override), LR_MIN, LR_MAX)
        if lr != item.lr_override:
            note = f"lr_override {item.lr_override} clamped to {lr}"
    elif item.source_type in table:
        lr = clamp(float(table[item.source_type]), LR_MIN, LR_MAX)
    else:
        lr, note = 1.0, f"unknown source_type {item.source_type!r} counted as no evidence"
    # An override already states the direction-specific ratio; a table LR is the supporting one.
    if item.stance == "contradicts" and item.lr_override is None:
        lr = 1.0 / lr
    return lr, note


def score_claim(
    claim_id: str, prior: float, evidence: Sequence[Evidence], thresholds: Thresholds,
    table: Mapping[str, float] = DEFAULT_LR_TABLE,
) -> ClaimScore:
    """Posterior via log-odds: logit(prior) + Σ discounted log(LR_i), capped at ±MAX_LOG_SHIFT."""
    p0 = clamp(float(prior), PRIOR_MIN, PRIOR_MAX)
    notes: List[str] = []
    if p0 != prior:
        notes.append(f"prior {prior} clamped to {p0}")
    seen: Dict[str, int] = {}
    shift = 0.0
    for item in evidence:
        lr, note = evidence_lr(item, table)
        if note:
            notes.append(note)
        k = seen.get(item.source_type, 0)
        seen[item.source_type] = k + 1
        shift += math.log(lr) * (SAME_TYPE_DISCOUNT ** k)
    if abs(shift) > MAX_LOG_SHIFT:
        notes.append("evidence shift capped")
        shift = math.copysign(MAX_LOG_SHIFT, shift)
    posterior = sigmoid(logit(p0) + shift)
    verdict = verdict_for(posterior, thresholds)
    return ClaimScore(claim_id, posterior, verdict, ACTIONS[verdict], p0, shift, notes)


def verdict_for(posterior: float, thresholds: Thresholds) -> str:
    if posterior >= thresholds.plain:
        return SUPPORTED
    if posterior < thresholds.hedge:
        return UNSUPPORTED
    return UNCERTAIN


# --- Beta posteriors ---------------------------------------------------------------------------

def beta_update(alpha: float, beta: float, successes: int, failures: int) -> Tuple[float, float]:
    if alpha <= 0 or beta <= 0 or successes < 0 or failures < 0:
        raise ValueError("Beta parameters must be positive and counts non-negative")
    return alpha + successes, beta + failures


def beta_mean(alpha: float, beta: float) -> float:
    return alpha / (alpha + beta)


# --- Refit (CPU-only, DESIGN N10) --------------------------------------------------------------

REFIT_MIN_LABELLED = 5
REFIT_LR_MIN, REFIT_LR_MAX = 0.2, 10.0


@dataclass(frozen=True)
class LabelledClaim:
    claim_type: str
    outcome: bool
    supporting_types: Tuple[str, ...]   # source types of supporting evidence on this claim


def refit_lrs(labelled: Iterable[LabelledClaim]) -> Dict[str, Dict[str, float]]:
    """Beta-smoothed LR per source type from labelled outcomes:
    ((s_true+1)/(T+2)) / ((s_false+1)/(F+2)); only types seen on >= REFIT_MIN_LABELLED claims."""
    rows = list(labelled)
    total_true = sum(1 for r in rows if r.outcome)
    total_false = len(rows) - total_true
    counts: Dict[str, List[int]] = {}
    for r in rows:
        for st in set(r.supporting_types):
            c = counts.setdefault(st, [0, 0])
            c[0 if r.outcome else 1] += 1
    fitted: Dict[str, Dict[str, float]] = {}
    for st, (s_true, s_false) in counts.items():
        if s_true + s_false < REFIT_MIN_LABELLED:
            continue
        lr = ((s_true + 1) / (total_true + 2)) / ((s_false + 1) / (total_false + 2))
        fitted[st] = {"lr": clamp(lr, REFIT_LR_MIN, REFIT_LR_MAX), "n": float(s_true + s_false)}
    return fitted


def refit_base_rates(labelled: Iterable[LabelledClaim]) -> Dict[str, Dict[str, float]]:
    """Per claim_type Beta(1+true, 1+false) posterior over 'claims of this type turn out true'."""
    counts: Dict[str, List[int]] = {}
    for r in labelled:
        c = counts.setdefault(r.claim_type, [0, 0])
        c[0 if r.outcome else 1] += 1
    out: Dict[str, Dict[str, float]] = {}
    for ct, (t, f) in counts.items():
        a, b = beta_update(1.0, 1.0, t, f)
        out[ct] = {"alpha": a, "beta": b, "mean": beta_mean(a, b)}
    return out
