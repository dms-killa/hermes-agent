"""``/bayes on|off|status|correct <claim> <true|false>|refit`` — handler gets ``raw_args`` only (DESIGN D1)."""

from __future__ import annotations

import logging
from typing import Callable, Tuple

from . import ledger
from .hooks import current_session_id, session_key
from .local_mode import resolve_local

logger = logging.getLogger(__name__)

USAGE = ("usage: /bayes on | off | status | correct <claim-hash-or-id> <true|false> | refit\n"
         "  on/off   opt this session in/out of Bayesian claim checking\n"
         "  status   mode, local/cloud, scored turns, skipped gates, ledger size\n"
         "  correct  record ground truth for a scored claim (the learning signal)\n"
         "  refit    recalibrate likelihood ratios from labelled claims (CPU only, no model calls)")
_TRUE, _FALSE = {"true", "t", "yes", "y", "1", "correct"}, {"false", "f", "no", "n", "0", "wrong"}


def _identity() -> Tuple[str, str]:
    """(opt-in key, concrete session id): gateway binds a session key, the CLI a session id."""
    key, sid = session_key(), current_session_id()
    if key and not sid:
        sid = ledger.session_for_key(key)
    return key or sid, sid


def make_command(settings: Callable, store_getter: Callable) -> Callable[[str], str]:
    def bayes(raw_args: str = "") -> str:
        parts = (raw_args or "").split()
        verb = parts[0].lower() if parts else "status"
        try:
            handler = _VERBS.get(verb)
            if handler is None:
                return USAGE
            return handler(parts[1:], settings, store_getter)
        except Exception as exc:
            logger.warning("bayes-check /bayes %s failed", verb, exc_info=True)
            return f"bayes-check: /bayes {verb} failed: {type(exc).__name__}: {exc}"

    return bayes


def _on(_args, settings, _store) -> str:
    key, sid = _identity()
    if not key:
        return "bayes-check: no active session yet — send a message first, then run /bayes on."
    ledger.enable(key)
    if sid:
        ledger.enable(sid)
    mode = settings().mode
    warn = " (note: plugin mode is 'off' in config, so nothing will be checked)" if mode == "off" else ""
    return f"bayes-check: ON for this session (mode={mode}){warn}. Claims will be scored and annotated."


def _off(_args, _settings, _store) -> str:
    key, sid = _identity()
    ledger.disable(*(k for k in (key, sid) if k))
    return "bayes-check: OFF for this session."


def _status(_args, settings, store_getter) -> str:
    cfg = settings()
    key, sid = _identity()
    on = ledger.is_enabled(key, sid)
    detected = ledger.get_local(sid) if sid else None
    is_local = resolve_local(cfg.local_mode, detected)
    turns = ledger.session_turns(sid) if sid else []
    scored = sum(1 for t in turns if t.scored)
    lines = [
        f"bayes-check: {'ON' if on else 'off'} for this session · mode={cfg.mode}",
        f"  local mode: {cfg.local_mode} → {'local' if is_local else 'cloud'}"
        f" (detected: {'unknown' if detected is None else ('local' if detected else 'cloud')})"
        f" · delegation veto when local: {cfg.block_delegation_local}",
        f"  thresholds: plain ≥ {cfg.min_posterior_plain}, remove < {cfg.min_posterior_hedge}",
        f"  this session: {scored}/{len(turns)} tracked turn(s) scored · gate skips: {ledger.gate_skips(sid)}",
    ]
    store = store_getter()
    if store is not None:
        s = store.stats()
        lines.append(f"  ledger: {s['claims']} claim(s), {s['labelled']} labelled, {s['gate_skips']} skipped gate(s)")
    return "\n".join(lines)


def _correct(args, _settings, store_getter) -> str:
    if len(args) != 2 or args[1].lower() not in _TRUE | _FALSE:
        return "usage: /bayes correct <claim-hash-or-id> <true|false>"
    store = store_getter()
    if store is None:
        return "bayes-check: ledger unavailable; cannot record the outcome."
    _key, sid = _identity()
    outcome = args[1].lower() in _TRUE
    n = store.label(args[0], outcome, session_id=sid, turn_id=ledger.last_scored_turn(sid) or "" if sid else "")
    if not n:
        return f"bayes-check: no scored claim matches {args[0]!r} (use the 8-hex hash shown in the footer)."
    return f"bayes-check: recorded claim {args[0]} as {'TRUE' if outcome else 'FALSE'}. Run /bayes refit to recalibrate."


def _refit(_args, _settings, store_getter) -> str:
    store = store_getter()
    if store is None:
        return "bayes-check: ledger unavailable; cannot refit."
    out = store.refit()
    lrs = ", ".join(f"{k}={v['lr']:.2f} (n={int(v['n'])})" for k, v in sorted(out["lrs"].items())) or "none yet"
    return (f"bayes-check: refit over {out['labelled']} labelled claim(s). Calibrated LRs: {lrs}. "
            f"Claim-type base rates: {len(out['base_rates'])}.")


_VERBS = {"on": _on, "enable": _on, "off": _off, "disable": _off, "status": _status,
          "correct": _correct, "refit": _refit}
