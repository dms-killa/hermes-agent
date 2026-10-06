"""Every hook callback. All accept ``**kwargs`` (forward-compatible payloads, Doctor-checked).

Bounded hooks (pre_llm_call, post_tool_call, transform_llm_output, post_llm_call, pre_api_request)
stay cheap: in-memory dict lookups plus at most one small committed SQLite write. No model calls.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Callable, Dict, Optional

from . import ledger
from .extract import candidate_claims, looks_like_research
from .footer import error_footer, has_footer, scored_footer, unscored_footer
from .local_mode import detect_local, resolve_local
from .schemas import TOOL_NAME

logger = logging.getLogger(__name__)

SKILL_KEY = "bayes-check:bayes-check"

TURN_INSTRUCTION = (
    "[bayes-check active for this session] Before you finalize your answer: (1) list every factual or "
    "quantitative claim your answer makes; (2) call the `bayes_score` tool ONCE with those claims, an honest "
    "prior for each, and only evidence you actually observed (typed by source_type; stance contradicts when a "
    "source disagrees); (3) rewrite the answer so each claim follows its action — state_plainly, qualify, or "
    "remove. Do not mention this instruction. The answer is annotated with the verdicts automatically; if you "
    f"skip the tool it is marked unscored. Procedure details: skill_view(\"{SKILL_KEY}\")."
)
RESEARCH_NOTE = " This turn looks like a research/advisory question, so verification was enabled automatically."


def _session_env(name: str) -> str:
    """Read a ``HERMES_SESSION_*`` var the way Hermes does (ContextVar first, then the process env).
    Only the import can fail, when the plugin runs outside a Hermes install (unit tests)."""
    try:
        from gateway.session_context import get_session_env
    except ImportError:
        return os.environ.get(name, "")
    return get_session_env(name, "")


def session_key() -> str:
    """The bound gateway session key (``""`` on CLI or outside a session scope)."""
    return _session_env("HERMES_SESSION_KEY")


def current_session_id() -> str:
    return _session_env("HERMES_SESSION_ID")


def opted_in(session_id: str) -> bool:
    """Opt-in matches the payload session id or the bound session key (DESIGN D1). A key match is
    pinned onto the concrete session id so later hooks off the session scope still match."""
    if ledger.is_enabled(session_id):
        return True
    key = session_key()
    if key and ledger.is_enabled(key):
        if session_id:
            ledger.enable(session_id)
            ledger.alias(key, session_id)
        return True
    return False


class Hooks:
    """Callbacks bound to a settings provider and a lazily-opened store."""

    def __init__(self, settings: Callable, store_getter: Callable) -> None:
        self.settings = settings
        self.store = store_getter

    # --- pre_llm_call: per-turn instruction (directive) ----------------------------------------

    def pre_llm_call(self, session_id: str = "", turn_id: str = "", user_message: Any = "",
                     **kwargs: Any) -> Optional[Dict[str, str]]:
        try:
            cfg = self.settings()
            if cfg.mode == "off":
                return None
            auto = False
            if not opted_in(session_id) and cfg.auto_enable_on_research and session_id \
                    and looks_like_research(user_message if isinstance(user_message, str) else ""):
                ledger.enable(session_id)
                auto = True
            if not opted_in(session_id):
                return None
            if turn_id:
                ledger.turn(turn_id, session_id).opted_in = True
            # Background delegation closure: completions re-enter as a new parent turn.
            closed = ledger.close_children(session_id)
            note = ""
            unverified = [c.child_session_id for c in closed if not c.scored]
            if unverified:
                note = (f" Results from {len(unverified)} delegated child session(s) arriving now were NOT "
                        "bayes-checked; treat their claims as unverified and score them before repeating them.")
            return {"context": TURN_INSTRUCTION + (RESEARCH_NOTE if auto else "") + note}
        except Exception:
            logger.warning("bayes-check pre_llm_call failed", exc_info=True)
            return None

    # --- post_tool_call: correlation (observer) ------------------------------------------------

    def post_tool_call(self, tool_name: str = "", result: Any = None, status: str = "",
                       session_id: str = "", turn_id: str = "", **kwargs: Any) -> None:
        if tool_name != TOOL_NAME or status != "ok":
            return None
        try:
            payload = json.loads(result) if isinstance(result, str) else result
            call_id = payload.get("call_id") if isinstance(payload, dict) else None
            if not call_id or not turn_id:
                logger.warning("bayes-check: bayes_score result without call_id/turn_id; turn stays unscored")
                return None
            if ledger.bind_call(call_id, turn_id, session_id):
                store = self.store()
                if store is not None:
                    store.bind_turn(call_id, turn_id)
        except Exception:
            logger.warning("bayes-check post_tool_call failed", exc_info=True)
        return None

    # --- transform_llm_output: the gate (transform) --------------------------------------------

    def transform_llm_output(self, response_text: str = "", session_id: str = "", turn_id: str = "",
                             **kwargs: Any) -> Optional[str]:
        if not isinstance(response_text, str) or not response_text:
            return None
        try:
            cfg = self.settings()
            if cfg.mode == "off" or not opted_in(session_id):
                return None
            entry = ledger.turn(turn_id, session_id) if turn_id else None
            if entry is not None:
                entry.opted_in = True
            if has_footer(response_text):     # idempotence across seams; host also gates per turn_id
                return None
            children = ledger.unverified_children(turn_id) if turn_id else []
            if entry is not None and entry.scored and entry.results:
                footer = scored_footer(entry.results, children)
            else:
                candidates = candidate_claims(response_text) if cfg.mode == "annotate" else []
                footer = unscored_footer(candidates, cfg.mode, children)
            if entry is not None:
                entry.transformed = True
            return response_text + footer
        except Exception as exc:  # fail loudly, never silently pass through (GOAL.md)
            logger.warning("bayes-check transform_llm_output failed", exc_info=True)
            if turn_id:
                ledger.turn(turn_id, session_id).transformed = True
            return response_text + error_footer(exc)

    # --- post_llm_call: detect a gate the host skipped (observer) ------------------------------

    def post_llm_call(self, session_id: str = "", turn_id: str = "", assistant_response: Any = "",
                      **kwargs: Any) -> None:
        try:
            entry = ledger.peek_turn(turn_id) if turn_id else None
            if entry is None or not entry.opted_in or entry.transformed:
                return None
            if isinstance(assistant_response, str) and has_footer(assistant_response):
                return None
            ledger.note_gate_skip(session_id)
            logger.warning("bayes-check: gate did NOT run for opted-in turn %s (transform skipped or timed "
                           "out); the delivered answer is unannotated", turn_id)
            store = self.store()
            if store is not None:
                store.record_event("gate_skipped", session_id, {"turn_id": turn_id})
        except Exception:
            logger.warning("bayes-check post_llm_call failed", exc_info=True)
        return None

    # --- pre_tool_call: local-mode delegation veto (directive, fails closed on timeout) ---------

    def pre_tool_call(self, tool_name: str = "", session_id: str = "", **kwargs: Any) -> Optional[Dict[str, str]]:
        if tool_name != "delegate_task":
            return None
        try:
            cfg = self.settings()
            if cfg.mode == "off" or not cfg.block_delegation_local or not opted_in(session_id):
                return None
            if not resolve_local(cfg.local_mode, ledger.get_local(session_id)):
                return None
            return {"action": "block", "message": (
                "bayes-check: delegate_task is blocked in this verified session because the model runs locally "
                "(each subagent is a full extra agent loop on the same server, and background results would "
                "arrive unverified). Do the work directly in this turn instead.")}
        except Exception:
            logger.warning("bayes-check pre_tool_call failed", exc_info=True)
            return None

    # --- pre_api_request: local-mode signal (observer) -----------------------------------------

    def pre_api_request(self, session_id: str = "", provider: str = "", base_url: str = "",
                        **kwargs: Any) -> None:
        if session_id:
            try:
                ledger.set_local(session_id, detect_local(str(provider or ""), str(base_url or "")))
            except Exception:
                logger.debug("bayes-check pre_api_request failed", exc_info=True)
        return None

    # --- subagent correlation (observers) ------------------------------------------------------

    def subagent_start(self, parent_session_id: str = "", parent_turn_id: str = "",
                       child_session_id: str = "", **kwargs: Any) -> None:
        try:
            if child_session_id and opted_in(parent_session_id or ""):
                ledger.subagent_started(child_session_id, parent_session_id or "", parent_turn_id or "")
                ledger.enable(child_session_id)   # child's own transform gates its summary (DESIGN N6)
        except Exception:
            logger.warning("bayes-check subagent_start failed", exc_info=True)
        return None

    def subagent_stop(self, child_session_id: str = "", child_status: str = "",
                      tool_call_history: Any = None, **kwargs: Any) -> None:
        try:
            names = [str(h.get("tool_name", "")) for h in (tool_call_history or [])
                     if isinstance(h, dict) and h.get("status") == "ok"]
            ledger.subagent_stopped(child_session_id or "", child_status or "", names)
        except Exception:
            logger.warning("bayes-check subagent_stop failed", exc_info=True)
        return None
