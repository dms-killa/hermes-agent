"""hermes-omniroute — OmniRoute integration for Hermes Agent.

register(ctx) wires:
- read tools in toolset ``omniroute`` (v0.1's six plus the read-only ``omniroute_combo_plan``);
- write tools ``omniroute_combo_apply`` / ``omniroute_combo_rollback`` in toolset
  ``omniroute_write``, which an operator enables or disables with ``hermes tools`` (H-GATE-1);
- one ``pre_tool_call`` hook that sends every call of the two write tools to Hermes's human
  approval gate (H-GATE-2). The hook holds no policy: it always asks, and the human decides;
- the bundled skill ``omniroute:omniroute-routing``, which holds the routing guidance.

No error-classification hook (docs/DESIGN.md D4).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .approval import make_approval_hook
from .schemas import READ_TOOLS, TOOLSET, WRITE_TOOLS, WRITE_TOOLSET
from .settings import ENV_API_KEY, ENV_BASE_URL, is_configured
from .tools import make_handlers
from .write_store import WriteStore

__version__ = "0.2.0"

SKILL_NAME = "omniroute-routing"
SKILL_PATH = Path(__file__).parent / "skills" / SKILL_NAME / "SKILL.md"
_EMOJI = {"omniroute_audit": "🧭", "hermes_combo_check": "🔗", "omniroute_combo_plan": "📝",
          "omniroute_combo_apply": "✍️", "omniroute_combo_rollback": "↩️"}


class _LazyState:
    """Resolve ``ctx.state`` on first use: Hermes's validation probe runs ``register()`` with a stub
    context that has no state facade, and registration must not need storage."""

    def __init__(self, ctx: Any) -> None:
        self._ctx = ctx

    def get(self, key: str, default: Any = None) -> Any:
        return self._ctx.state.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._ctx.state.set(key, value)


def register(ctx: Any) -> None:
    store = WriteStore(_LazyState(ctx))
    handlers = make_handlers(store=store)
    for toolset, schemas in ((TOOLSET, READ_TOOLS), (WRITE_TOOLSET, WRITE_TOOLS)):
        for schema in schemas:
            name = schema["name"]
            ctx.register_tool(name=name, toolset=toolset, schema=schema, handler=handlers[name],
                              check_fn=is_configured, requires_env=[ENV_BASE_URL, ENV_API_KEY],
                              emoji=_EMOJI.get(name, "🔎"))
    ctx.register_hook("pre_tool_call", make_approval_hook(store))
    ctx.register_skill(SKILL_NAME, SKILL_PATH,
                       description="OmniRoute routing semantics and the plan/apply/rollback workflow.")


__all__ = ["register", "__version__"]
