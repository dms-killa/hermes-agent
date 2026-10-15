"""Route every call of the write tools to Hermes's human-approval gate (H-GATE-2).

Hermes documents this as the entry point for plugin tools: a ``pre_tool_call`` callback returning
``{"action": "approve", "message", "rule_key"}`` asks the same human gate as dangerous shell
commands, and fails closed when no human is present (``tools/approval.py:request_tool_approval``).

The hook decides nothing: it always escalates the two write tools and never the others. The
message only describes the pending action so the human can review it. ``rule_key`` is the tool name,
so a human's "always" answer means "always for this tool", Hermes's own default grain.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from .schemas import WRITE_TOOL_NAMES
from .write_store import WriteStore


def _describe_plan(store: WriteStore, plan_id: Any) -> str:
    plan = store.get_plan(plan_id) if isinstance(plan_id, str) else None
    if not plan:
        return f"apply OmniRoute combo plan {plan_id!r} (unknown to this profile; the tool will refuse it)"
    fields = ", ".join(plan.get("planned_fields", [])[:12]) or "(none)"
    return (f"apply plan {plan_id} to OmniRoute combo '{plan.get('combo_name')}': one PUT changing "
            f"{fields}. Review the plan output (diff, findings, server-may-reject) before approving.")


def _describe_rollback(store: WriteStore, rollback_id: Any, force: Any) -> str:
    record = store.get_rollback(rollback_id) if isinstance(rollback_id, str) else None
    if not record:
        return f"roll back {rollback_id!r} (unknown to this profile; the tool will refuse it)"
    forced = " FORCED: overwrites any change made after the apply." if force is True else ""
    return (f"roll back OmniRoute combo '{record.get('combo_name')}' to the snapshot taken before plan "
            f"{record.get('plan_id')}.{forced}")


def make_approval_hook(store: WriteStore) -> Callable[..., Optional[Dict[str, Any]]]:
    def pre_tool_call(tool_name: str = "", args: Optional[Dict[str, Any]] = None, **_: Any):
        if tool_name not in WRITE_TOOL_NAMES:
            return None
        args = args if isinstance(args, dict) else {}
        try:
            if tool_name == "omniroute_combo_apply":
                message = _describe_plan(store, args.get("plan_id"))
            else:
                message = _describe_rollback(store, args.get("rollback_id"), args.get("force"))
        except Exception:  # a broken store must not skip the gate
            message = f"{tool_name} (details unavailable)"
        return {"action": "approve", "message": message, "rule_key": tool_name}
    return pre_tool_call
