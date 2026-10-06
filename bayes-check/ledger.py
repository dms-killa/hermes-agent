"""In-process correlation state shared by every hook (same process, per plugin load).

All hooks run in one process — CLI, gateway, and delegated children on worker threads — so a
lock-guarded module dict is the correlation channel; SQLite holds only the durable learning data.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

MAX_TURNS = 512
_lock = threading.RLock()

_enabled_sessions: Set[str] = set()
_turn_ledger: "OrderedDict[str, TurnEntry]" = OrderedDict()
# child_session_id -> SubagentEntry (keyed by child: subagent_stop's parent_turn_id is read live, DESIGN D5)
_subagent_ledger: Dict[str, "SubagentEntry"] = {}
_local_sessions: Dict[str, bool] = {}
# session_id -> most recent scored turn_id (for /bayes correct by model claim id)
_last_scored_turn: Dict[str, str] = {}
_gate_skips: Dict[str, int] = {}
_turns_seen: Dict[str, List[str]] = {}
# call_id -> (session_id, results): scored by the handler, bound to a turn by post_tool_call (DESIGN D7)
_pending: "OrderedDict[str, tuple]" = OrderedDict()
MAX_PENDING = 256
# gateway session_key -> the concrete session_id it was last seen with (commands only know the key)
_key_sessions: Dict[str, str] = {}


@dataclass
class TurnEntry:
    turn_id: str
    session_id: str = ""
    scored: bool = False             # set by post_tool_call(status == ok)
    results: List[Dict[str, Any]] = field(default_factory=list)   # set by the bayes_score handler
    transformed: bool = False
    opted_in: bool = False


@dataclass
class SubagentEntry:
    child_session_id: str
    parent_session_id: str = ""
    parent_turn_id: str = ""
    scored: bool = False
    stopped: bool = False
    status: str = ""


def reset() -> None:
    """Drop all in-process state (plugin unload and tests)."""
    with _lock:
        _enabled_sessions.clear()
        _turn_ledger.clear()
        _subagent_ledger.clear()
        _local_sessions.clear()
        _last_scored_turn.clear()
        _gate_skips.clear()
        _turns_seen.clear()
        _pending.clear()
        _key_sessions.clear()


# --- opt-in -------------------------------------------------------------------------------------

def enable(key: str) -> None:
    with _lock:
        _enabled_sessions.add(key)


def disable(*keys: str) -> None:
    with _lock:
        for key in keys:
            _enabled_sessions.discard(key)


def is_enabled(*keys: str) -> bool:
    with _lock:
        return any(k and k in _enabled_sessions for k in keys)


def alias(key: str, session_id: str) -> None:
    with _lock:
        _key_sessions[key] = session_id


def session_for_key(key: str) -> str:
    with _lock:
        return _key_sessions.get(key, "")


def enabled_count() -> int:
    with _lock:
        return len(_enabled_sessions)


# --- turns --------------------------------------------------------------------------------------

def turn(turn_id: str, session_id: str = "") -> TurnEntry:
    """Get-or-create the entry for ``turn_id`` (LRU-bounded at MAX_TURNS)."""
    with _lock:
        entry = _turn_ledger.get(turn_id)
        if entry is None:
            entry = TurnEntry(turn_id=turn_id, session_id=session_id)
            _turn_ledger[turn_id] = entry
            if session_id:
                _turns_seen.setdefault(session_id, []).append(turn_id)
            while len(_turn_ledger) > MAX_TURNS:
                _turn_ledger.popitem(last=False)
        elif session_id and not entry.session_id:
            entry.session_id = session_id
        _turn_ledger.move_to_end(turn_id)
        return entry


def peek_turn(turn_id: str) -> Optional[TurnEntry]:
    with _lock:
        return _turn_ledger.get(turn_id)


def stash_pending(call_id: str, session_id: str, results: List[Dict[str, Any]]) -> None:
    """Called by the tool handler, which knows the session but not the turn."""
    with _lock:
        _pending[call_id] = (session_id, results)
        while len(_pending) > MAX_PENDING:
            _pending.popitem(last=False)


def bind_call(call_id: str, turn_id: str, session_id: str) -> bool:
    """Called by post_tool_call(status == ok): move a stashed call's results onto its turn and
    mark the turn scored. False when the call id is unknown (evicted or forged)."""
    with _lock:
        pending = _pending.pop(call_id, None)
        if pending is None:
            return False
        record_results(turn_id, session_id or pending[0], pending[1])
        mark_scored(turn_id, session_id or pending[0])
        return True


def record_results(turn_id: str, session_id: str, results: List[Dict[str, Any]]) -> None:
    """Append scored results to the turn."""
    with _lock:
        entry = turn(turn_id, session_id)
        entry.results.extend(results)
        if session_id:
            _last_scored_turn[session_id] = turn_id


def mark_scored(turn_id: str, session_id: str) -> None:
    """Called by post_tool_call on a status == ok bayes_score call."""
    with _lock:
        entry = turn(turn_id, session_id)
        entry.scored = True
        child = _subagent_ledger.get(session_id)
        if child is not None:
            child.scored = True


def last_scored_turn(session_id: str) -> Optional[str]:
    with _lock:
        return _last_scored_turn.get(session_id)


def session_turns(session_id: str) -> List[TurnEntry]:
    with _lock:
        return [e for t in _turns_seen.get(session_id, []) if (e := _turn_ledger.get(t)) is not None]


def note_gate_skip(session_id: str) -> None:
    with _lock:
        _gate_skips[session_id] = _gate_skips.get(session_id, 0) + 1


def gate_skips(session_id: str) -> int:
    with _lock:
        return _gate_skips.get(session_id, 0)


# --- subagents ----------------------------------------------------------------------------------

def subagent_started(child_session_id: str, parent_session_id: str, parent_turn_id: str) -> None:
    with _lock:
        _subagent_ledger[child_session_id] = SubagentEntry(
            child_session_id=child_session_id, parent_session_id=parent_session_id,
            parent_turn_id=parent_turn_id)


def subagent_stopped(child_session_id: str, status: str, tool_names: List[str]) -> Optional[SubagentEntry]:
    with _lock:
        entry = _subagent_ledger.get(child_session_id)
        if entry is None:
            return None
        entry.stopped = True
        entry.status = status or ""
        # tool_call_history is metadata-only; it can confirm the tool ran, never which claims.
        if "bayes_score" in tool_names:
            entry.scored = True
        return entry


def unverified_children(parent_turn_id: str) -> List[str]:
    """Children dispatched from ``parent_turn_id`` with no scored bayes_score call."""
    with _lock:
        return sorted(e.child_session_id for e in _subagent_ledger.values()
                      if e.parent_turn_id == parent_turn_id and not e.scored)


def open_children(parent_session_id: str) -> List[SubagentEntry]:
    with _lock:
        return [e for e in _subagent_ledger.values() if e.parent_session_id == parent_session_id]


def close_children(parent_session_id: str) -> List[SubagentEntry]:
    """Close stopped children of ``parent_session_id`` (background delivery turn) and return them."""
    with _lock:
        closed = [e for e in _subagent_ledger.values() if e.parent_session_id == parent_session_id and e.stopped]
        for e in closed:
            _subagent_ledger.pop(e.child_session_id, None)
        return closed


# --- local mode ---------------------------------------------------------------------------------

def set_local(session_id: str, is_local: bool) -> None:
    with _lock:
        _local_sessions[session_id] = is_local


def get_local(session_id: str) -> Optional[bool]:
    with _lock:
        return _local_sessions.get(session_id)
