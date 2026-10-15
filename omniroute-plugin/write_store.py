"""Plan and rollback records in plugin storage (``ctx.state``: profile-scoped JSON under
``$HERMES_HOME/plugin-data/<plugin>/state.json``, H-PLUG-7). Expiry is enforced here mechanically."""

from __future__ import annotations

import time
import uuid
from typing import Any, Callable, Dict, Optional

PLAN_TTL_SECONDS = 900
MAX_PLAN_TTL_SECONDS = 3600
MAX_ROLLBACKS = 50
_PLANS, _ROLLBACKS = "combo_plans", "combo_rollbacks"


class MemoryState:
    """ctx.state stand-in for tests and for use outside Hermes (same get/set contract)."""

    def __init__(self) -> None:
        self._data: Dict[str, Any] = {}

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        import json
        self._data[key] = json.loads(json.dumps(value))   # same JSON round trip as PluginState


class WriteStore:
    def __init__(self, state: Any, clock: Callable[[], float] = time.time):
        self._state = state
        self._clock = clock

    def now(self) -> float:
        return self._clock()

    # plans --------------------------------------------------------------------------------------
    def save_plan(self, plan: Dict[str, Any], ttl_seconds: int = PLAN_TTL_SECONDS) -> Dict[str, Any]:
        now = self.now()
        plans = {k: v for k, v in (self._state.get(_PLANS, {}) or {}).items()
                 if v.get("expires_at", 0) > now}                         # prune expired
        plan = dict(plan, plan_id="plan-" + uuid.uuid4().hex[:16], created_at=now,
                    expires_at=now + max(60, min(int(ttl_seconds), MAX_PLAN_TTL_SECONDS)))
        plans[plan["plan_id"]] = plan
        self._state.set(_PLANS, plans)
        return plan

    def get_plan(self, plan_id: str) -> Optional[Dict[str, Any]]:
        return (self._state.get(_PLANS, {}) or {}).get(plan_id)

    def mark_plan(self, plan_id: str, **fields: Any) -> None:
        plans = dict(self._state.get(_PLANS, {}) or {})
        if plan_id in plans:
            plans[plan_id] = dict(plans[plan_id], **fields)
            self._state.set(_PLANS, plans)

    # rollbacks ----------------------------------------------------------------------------------
    def save_rollback(self, record: Dict[str, Any]) -> Dict[str, Any]:
        rollbacks = dict(self._state.get(_ROLLBACKS, {}) or {})
        record = dict(record, rollback_id=record.get("rollback_id") or "rb-" + uuid.uuid4().hex[:16],
                      created_at=record.get("created_at") or self.now())
        rollbacks[record["rollback_id"]] = record
        if len(rollbacks) > MAX_ROLLBACKS:
            for key in sorted(rollbacks, key=lambda k: rollbacks[k].get("created_at", 0))[:-MAX_ROLLBACKS]:
                rollbacks.pop(key)
        self._state.set(_ROLLBACKS, rollbacks)
        return record

    def get_rollback(self, rollback_id: str) -> Optional[Dict[str, Any]]:
        return (self._state.get(_ROLLBACKS, {}) or {}).get(rollback_id)
