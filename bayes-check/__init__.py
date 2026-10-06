"""bayes-check — deterministic Bayesian claim scoring + forced confidence annotation for Hermes.

register(ctx) wires: the ``bayes_score`` tool, eight hooks, the ``/bayes`` command, the bundled
skill, and one cache-safe system-prompt section. The plugin makes zero LLM calls.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Optional

from . import config, ledger
from .commands import USAGE, make_command
from .hooks import SKILL_KEY, Hooks
from .schemas import BAYES_SCORE, TOOL_NAME, TOOLSET
from .state import Store
from .tools import make_handler

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_RULE = (
    "bayes-check (Bayesian claim verification) is installed. When a turn tells you bayes-check is active for "
    "this session, you must, before finalizing: extract each factual or quantitative claim in your draft, call "
    "the `bayes_score` tool once with an honest prior and only evidence you actually observed, and rewrite so "
    "every claim follows its returned action (state_plainly / qualify / remove). Never invent evidence to raise "
    "a posterior. Your final answer is annotated with the computed verdicts automatically; skipping the tool "
    f"marks the answer unscored. Full procedure: skill_view(\"{SKILL_KEY}\")."
)


class _LazyStore:
    """Open the SQLite store on first use (Doctor runs register() with a throwaway home)."""

    def __init__(self) -> None:
        self._store: Optional[Store] = None
        self._failed = False
        self._lock = threading.Lock()

    def __call__(self) -> Optional[Store]:
        if self._store is not None or self._failed:
            return self._store
        with self._lock:
            if self._store is None and not self._failed:
                try:
                    self._store = Store()
                    self._store._connect()
                except Exception:
                    # Scoring and annotation keep working; only learning/persistence degrade.
                    logger.warning("bayes-check: ledger database unavailable; scoring continues without it",
                                   exc_info=True)
                    self._store, self._failed = None, True
        return self._store

    def close(self) -> None:
        if self._store is not None:
            self._store.close()
        self._store, self._failed = None, False


def register(ctx: Any) -> None:
    settings = config.SettingsCache(lambda key, default: ctx.get_config(key, default))
    store = _LazyStore()
    hooks = Hooks(settings, store)

    ctx.register_tool(name=TOOL_NAME, toolset=TOOLSET, schema=BAYES_SCORE,
                      handler=make_handler(settings, store), emoji="🎲")

    for name in ("pre_llm_call", "post_tool_call", "transform_llm_output", "post_llm_call",
                 "pre_tool_call", "pre_api_request", "subagent_start", "subagent_stop"):
        ctx.register_hook(name, getattr(hooks, name))

    ctx.register_command("bayes", make_command(settings, store),
                         description="Bayesian claim checking: on | off | status | correct | refit",
                         args_hint="on|off|status|correct|refit")

    ctx.register_skill("bayes-check", Path(__file__).parent / "skills" / "bayes-check" / "SKILL.md",
                       description="Procedure for Bayesian claim verification with the bayes_score tool.")

    ctx.register_system_prompt_section("bayes-check.verification-rule", SYSTEM_PROMPT_RULE,
                                       position="after_memory", max_chars=4000)

    def _unload() -> None:
        store.close()
        ledger.reset()

    ctx.on_unload(_unload)


__all__ = ["register", "USAGE"]
