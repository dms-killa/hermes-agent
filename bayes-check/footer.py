"""Footer rendering. Every footer starts with FOOTER_RULE so ``post_llm_call`` can detect it."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Iterable, List, Mapping, Sequence

FOOTER_RULE = "\n\n---\n"
FOOTER_TAG = "bayes-check"
_GLYPH = {"supported": "✓", "uncertain": "~", "unsupported": "✗"}
MAX_LINES = 20


def claim_hash(text: str) -> str:
    return hashlib.sha256(" ".join((text or "").split()).lower().encode("utf-8")).hexdigest()[:8]


def results_hash(results: Sequence[Mapping[str, Any]]) -> str:
    canonical = json.dumps(
        [{k: r.get(k) for k in ("hash", "posterior", "verdict")} for r in results],
        sort_keys=True, separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def has_footer(text: str) -> bool:
    return isinstance(text, str) and f"{FOOTER_TAG}" in text and FOOTER_RULE in text


def _claim_lines(results: Iterable[Mapping[str, Any]]) -> List[str]:
    lines = []
    for r in list(results)[:MAX_LINES]:
        snippet = " ".join(str(r.get("text", "")).split())
        if len(snippet) > 90:
            snippet = snippet[:89] + "…"
        lines.append(
            f"- {_GLYPH.get(r['verdict'], '?')} {r['verdict']} p={r['posterior']:.2f} "
            f"[{r.get('id', '?')}·{r['hash']}] {snippet}"
        )
    return lines


def scored_footer(results: Sequence[Mapping[str, Any]], unverified_children: Sequence[str] = ()) -> str:
    counts: Dict[str, int] = {}
    for r in results:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    summary = ", ".join(f"{counts[v]} {v}" for v in ("supported", "uncertain", "unsupported") if v in counts)
    parts = [f"{FOOTER_RULE}**{FOOTER_TAG}** `bayes-checked:{results_hash(results)}` — {summary or 'no claims'}"]
    parts.extend(_claim_lines(results))
    if len(results) > MAX_LINES:
        parts.append(f"- … {len(results) - MAX_LINES} more claim(s) in the ledger")
    parts.extend(_children_lines(unverified_children))
    return "\n".join(parts)


def unscored_footer(candidates: Sequence[str], mode: str, unverified_children: Sequence[str] = ()) -> str:
    parts = [f"{FOOTER_RULE}**{FOOTER_TAG}** ⚠ bayes-check: unscored — bayes_score was not called this turn; "
             "claims below are unverified (posterior = prior)."]
    if mode == "annotate":
        for c in candidates:
            parts.append(f"- ~ uncertain p=prior [{claim_hash(c)}] {c}")
        if not candidates:
            parts.append("- no checkable-looking claims detected")
    parts.extend(_children_lines(unverified_children))
    return "\n".join(parts)


def error_footer(exc: BaseException) -> str:
    return (f"{FOOTER_RULE}**{FOOTER_TAG}** ⚠ bayes-check: gate error ({type(exc).__name__}) — "
            "this answer was NOT checked.")


def _children_lines(children: Sequence[str]) -> List[str]:
    if not children:
        return []
    shown = ", ".join(c[:24] for c in children[:5])
    more = f" (+{len(children) - 5})" if len(children) > 5 else ""
    return [f"- ⚠ unverified subagent output from {len(children)} child session(s): {shown}{more}"]
