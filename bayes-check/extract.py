"""Deterministic claim-candidate extraction for the ``annotate`` fallback (no model call).

Used only when the model did not call ``bayes_score``: it lists sentences that look checkable so
the unscored footer names what went unverified. Heuristic by design, bounded in work.
"""

from __future__ import annotations

import re
from typing import List

MAX_SCAN_CHARS = 20_000
MAX_CANDIDATES = 8
MAX_CANDIDATE_CHARS = 160

_FENCE = re.compile(r"```.*?```", re.DOTALL)
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")
_CUES = (
    re.compile(r"\d"),  # numbers, dates, percentages, versions
    re.compile(r"\b(always|never|all|none|every|only|first|largest|smallest|most|least|best|worst)\b", re.I),
    re.compile(r"\b(causes?|caused|leads? to|results? in|proves?|shows?|according to|studies|research)\b", re.I),
    re.compile(r"\b(is|are|was|were) (the|a|an) \w+", re.I),
)
_SKIP_PREFIX = re.compile(r"^\s*(#|>|[-*]\s*\[|\||<)")


def candidate_claims(text: str) -> List[str]:
    """Up to MAX_CANDIDATES checkable-looking sentences, code blocks and headings excluded."""
    body = _FENCE.sub(" ", (text or "")[:MAX_SCAN_CHARS])
    out: List[str] = []
    for raw in _SENTENCE.split(body):
        sentence = raw.strip().lstrip("-*• ").strip()
        if len(sentence) < 12 or sentence.endswith("?") or _SKIP_PREFIX.match(sentence):
            continue
        if not any(cue.search(sentence) for cue in _CUES):
            continue
        if len(sentence) > MAX_CANDIDATE_CHARS:
            sentence = sentence[: MAX_CANDIDATE_CHARS - 1].rstrip() + "…"
        out.append(sentence)
        if len(out) >= MAX_CANDIDATES:
            break
    return out


_RESEARCH_CUES = re.compile(
    r"\b(research|evidence|studies|study|source|sources|cite|citation|statistic|statistics|data|"
    r"compare|comparison|pros and cons|should i|recommend|which is better|is it true|fact[- ]check|"
    r"how many|how much|what percentage|historically|according to)\b",
    re.I,
)


def looks_like_research(user_message: str) -> bool:
    """Heuristic advisory/research detector for ``auto_enable_on_research`` (DESIGN N11)."""
    msg = (user_message or "")[:4000]
    if len(msg.strip()) < 15:
        return False
    return bool(_RESEARCH_CUES.search(msg))
