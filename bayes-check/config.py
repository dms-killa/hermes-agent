"""Typed plugin settings (``plugins.entries.bayes-check.settings``) with spec defaults."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

from .local_mode import AUTO, LOCAL_MODES
from .scorer import Thresholds

logger = logging.getLogger(__name__)

MODES = ("off", "annotate", "nudge")
_TTL_SECONDS = 5.0  # hooks are hot paths: re-read config.yaml at most this often


@dataclass(frozen=True)
class Settings:
    mode: str = "annotate"
    local_mode: str = AUTO
    min_posterior_plain: float = 0.9
    min_posterior_hedge: float = 0.5
    block_delegation_local: bool = True
    auto_enable_on_research: bool = False

    @property
    def thresholds(self) -> Thresholds:
        return Thresholds(plain=self.min_posterior_plain, hedge=self.min_posterior_hedge)


def _pick(get: Callable[[str, Any], Any], key: str, default: Any, ok: Callable[[Any], bool]) -> Any:
    value = get(key, default)
    if value is default or ok(value):
        return value
    logger.warning("bayes-check: invalid setting %s=%r; using default %r", key, value, default)
    return default


def _is_prob(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and 0.0 < float(v) < 1.0


def load(get: Callable[[str, Any], Any]) -> Settings:
    d = Settings()
    plain = float(_pick(get, "min_posterior_plain", d.min_posterior_plain, _is_prob))
    hedge = float(_pick(get, "min_posterior_hedge", d.min_posterior_hedge, _is_prob))
    if hedge > plain:
        logger.warning("bayes-check: min_posterior_hedge %s > min_posterior_plain %s; using defaults", hedge, plain)
        plain, hedge = d.min_posterior_plain, d.min_posterior_hedge
    return Settings(
        mode=_pick(get, "mode", d.mode, lambda v: v in MODES),
        local_mode=_pick(get, "local_mode", d.local_mode, lambda v: v in LOCAL_MODES),
        min_posterior_plain=plain,
        min_posterior_hedge=hedge,
        block_delegation_local=_pick(get, "block_delegation_local", d.block_delegation_local,
                                     lambda v: isinstance(v, bool)),
        auto_enable_on_research=_pick(get, "auto_enable_on_research", d.auto_enable_on_research,
                                      lambda v: isinstance(v, bool)),
    )


class SettingsCache:
    def __init__(self, get: Callable[[str, Any], Any]) -> None:
        self._get = get
        self._value: Optional[Settings] = None
        self._at = 0.0

    def __call__(self) -> Settings:
        now = time.monotonic()
        if self._value is None or now - self._at > _TTL_SECONDS:
            try:
                self._value = load(self._get)
            except Exception:
                logger.warning("bayes-check: settings read failed; using defaults", exc_info=True)
                self._value = Settings()
            self._at = now
        return self._value

    def invalidate(self) -> None:
        self._value = None
