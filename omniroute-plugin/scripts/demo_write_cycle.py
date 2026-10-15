"""Run one plan -> apply -> rollback cycle through Hermes's own tool dispatcher
(model_tools.handle_function_call), which runs the plugin's pre_tool_call approval gate.

    HERMES_HOME=... python scripts/demo_write_cycle.py <combo> '<changes json>' ['<caller_limits json>']

Run it with the Python environment Hermes is installed in. With HERMES_INTERACTIVE=1 Hermes asks for
approval on the terminal before apply and before rollback; without a human it blocks (fail-closed).
"""

from __future__ import annotations

import json
import sys

import model_tools
from hermes_cli import plugins as hermes_plugins


def call(name, args):
    print(f"\n>>> {name} {json.dumps(args)}")
    out = json.loads(model_tools.handle_function_call(name, args, task_id="demo", session_id="demo"))
    print(json.dumps(out, indent=2))
    return out


def main() -> int:
    hermes_plugins.discover_plugins(force=True)
    combo, changes = sys.argv[1], json.loads(sys.argv[2])
    limits = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
    plan = call("omniroute_combo_plan", {"combo": combo, "changes": changes, "caller_limits": limits})
    if not plan.get("plan_id") or not plan.get("applicable"):
        return 1
    applied = call("omniroute_combo_apply", {"plan_id": plan["plan_id"]})
    if not applied.get("rollback_id"):
        return 2
    rolled = call("omniroute_combo_rollback", {"rollback_id": applied["rollback_id"]})
    return 0 if rolled.get("rolled_back") else 3


if __name__ == "__main__":
    sys.exit(main())
