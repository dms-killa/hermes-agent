"""``bayes_score`` handler: validates its own args (the registry does not), never raises."""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple

from . import ledger
from .footer import claim_hash
from .schemas import MAX_CLAIMS, MAX_EVIDENCE, MAX_TEXT
from .scorer import DEFAULT_LR_TABLE, Evidence, Thresholds, score_claim

logger = logging.getLogger(__name__)

STANCES = ("supports", "contradicts")


class ArgError(ValueError):
    pass


def _str(obj: Dict[str, Any], key: str, where: str, *, required: bool = True, limit: int = MAX_TEXT) -> str:
    value = obj.get(key)
    if value is None and not required:
        return ""
    if not isinstance(value, str) or not value.strip():
        raise ArgError(f"{where}.{key} must be a non-empty string")
    if len(value) > limit:
        raise ArgError(f"{where}.{key} exceeds {limit} characters")
    return value.strip()


def _number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ArgError(f"{where} must be a number")
    return float(value)


def _evidence(raw: Any, where: str) -> Evidence:
    if not isinstance(raw, dict):
        raise ArgError(f"{where} must be an object")
    stance = raw.get("stance", "supports")
    if stance not in STANCES:
        raise ArgError(f"{where}.stance must be one of {', '.join(STANCES)}")
    lr = raw.get("lr_override")
    if lr is not None:
        lr = _number(lr, f"{where}.lr_override")
        if lr <= 0:
            raise ArgError(f"{where}.lr_override must be > 0")
    return Evidence(source=_str(raw, "source", where), source_type=_str(raw, "source_type", where, limit=64),
                    stance=stance, lr_override=lr)


def parse_claims(args: Any) -> List[Dict[str, Any]]:
    """Validated, normalized claims. Accepts ``claims`` as a list or a JSON string of one."""
    if not isinstance(args, dict):
        raise ArgError("arguments must be an object")
    claims = args.get("claims")
    if isinstance(claims, str):  # weaker local models often double-encode
        try:
            claims = json.loads(claims)
        except ValueError as exc:
            raise ArgError(f"claims is a string but not valid JSON: {exc}") from None
    if not isinstance(claims, list) or not claims:
        raise ArgError("claims must be a non-empty array")
    if len(claims) > MAX_CLAIMS:
        raise ArgError(f"too many claims ({len(claims)} > {MAX_CLAIMS}); score the most load-bearing ones")
    out, ids = [], set()
    for i, raw in enumerate(claims):
        where = f"claims[{i}]"
        if not isinstance(raw, dict):
            raise ArgError(f"{where} must be an object")
        cid = _str(raw, "id", where, limit=64)
        if cid in ids:
            raise ArgError(f"{where}.id {cid!r} is duplicated")
        ids.add(cid)
        prior = raw.get("prior")
        if prior is not None:
            prior = _number(prior, f"{where}.prior")
            if not 0.0 <= prior <= 1.0:
                raise ArgError(f"{where}.prior must be within [0, 1]")
        evidence_raw = raw.get("evidence", [])
        if not isinstance(evidence_raw, list):
            raise ArgError(f"{where}.evidence must be an array")
        if len(evidence_raw) > MAX_EVIDENCE:
            raise ArgError(f"{where}.evidence has more than {MAX_EVIDENCE} items")
        out.append({
            "id": cid, "text": _str(raw, "text", where), "claim_type": _str(raw, "claim_type", where, limit=64),
            "prior": prior,
            "evidence": [_evidence(e, f"{where}.evidence[{j}]") for j, e in enumerate(evidence_raw)],
        })
    return out


def score(claims: List[Dict[str, Any]], thresholds: Thresholds, lr_table: Dict[str, float],
          base_rate: Callable[[str], Optional[float]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    results = []
    for c in claims:
        prior = c["prior"]
        prior_source = "model"
        if prior is None:
            learned = base_rate(c["claim_type"])
            prior, prior_source = (learned, "learned") if learned is not None else (0.5, "default")
        s = score_claim(c["id"], prior, c["evidence"], thresholds, lr_table)
        results.append({
            "id": c["id"], "hash": claim_hash(c["text"]), "text": c["text"], "claim_type": c["claim_type"],
            "prior": round(s.prior, 4), "prior_source": prior_source, "posterior": round(s.posterior, 4),
            "verdict": s.verdict, "action": s.action, "notes": s.notes,
            "evidence": [{"source_type": e.source_type, "stance": e.stance} for e in c["evidence"]],
        })
    counts: Dict[str, int] = {}
    for r in results:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    summary = {
        "counts": counts,
        "instruction": "Rewrite the answer so every claim honors its action: state_plainly as-is, qualify "
                       "with explicit uncertainty, remove (or mark unverified). The final answer is annotated "
                       "with these verdicts automatically.",
        "thresholds": {"min_posterior_plain": thresholds.plain, "min_posterior_hedge": thresholds.hedge},
    }
    return results, summary


def make_handler(settings: Callable, store_getter: Callable) -> Callable[..., str]:
    def bayes_score(args: dict, **kwargs: Any) -> str:
        try:
            claims = parse_claims(args)
        except ArgError as exc:
            return json.dumps({"error": f"bayes_score: {exc}"})
        try:
            cfg = settings()
            store = store_getter()
            lr_table = DEFAULT_LR_TABLE
            base_rate: Callable[[str], Optional[float]] = lambda _ct: None
            if store is not None:
                lr_table = store.lr_table(DEFAULT_LR_TABLE)
                base_rate = store.base_rate
            results, summary = score(claims, cfg.thresholds, lr_table, base_rate)
            session_id = str(kwargs.get("session_id") or "")
            # Handlers never receive turn_id (DESIGN D7): stash under a call id that post_tool_call,
            # which does carry turn_id and this result, binds to the turn.
            call_id = uuid.uuid4().hex[:16]
            ledger.stash_pending(call_id, session_id, results)
            if store is not None:
                store.append_claims(session_id, call_id, results)   # commits before returning
            public = [{k: r[k] for k in ("id", "hash", "posterior", "verdict", "action", "prior", "notes")}
                      for r in results]
            return json.dumps({"call_id": call_id, "results": public, "summary": summary})
        except Exception as exc:  # never raise into the tool loop
            logger.warning("bayes_score failed", exc_info=True)
            return json.dumps({"error": f"bayes_score internal error: {type(exc).__name__}: {exc}"})

    return bayes_score

