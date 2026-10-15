"""Request-body signatures for telling synthetic probe traffic from real traffic (lesson L6).

Probe and real requests land in the same call logs and combo metrics. Timing alone cannot tell
them apart; the request body can. The built-in signatures are the exact bodies OmniRoute itself
sends (CAPABILITIES.md O-LOGS-4). Operators add their own probe prompts via ``extra_prompts``.

A signature is computed from a stored request body only; nothing else is inferred.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Iterable, List, Optional

# (label, prompt text, max_tokens or None) — from OmniRoute source:
#   src/lib/combos/testHealth.ts      COMBO_TEST_PROMPT / COMBO_TEST_MAX_TOKENS (combo + model test)
#   src/lib/warmupScheduler.ts        WARMUP_MESSAGES, max_tokens: 1 (sent straight to the provider;
#                                     listed for completeness, normally not in /v1 call logs)
#   src/app/api/provider-nodes/validate/route.ts   "ping" (node validation)
#   src/shared/constants/quotaAutoPing.ts           "Reply with OK." (quota auto-ping)
BUILTIN_PROBES = (
    ("omniroute-combo-test", "Reply with exactly: pong", 64),
    ("omniroute-warmup", "hi", 1), ("omniroute-warmup", "hello", 1),
    ("omniroute-warmup", "ping", 1), ("omniroute-warmup", "ready", 1),
    ("omniroute-node-validate", "ping", None),
    ("omniroute-quota-autoping", "Reply with OK.", None),
)


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict)
                       and isinstance(p.get("text"), str))
    return ""


def body_signature(body: Any) -> Optional[Dict[str, Any]]:
    """Shape of a chat request body: no message text is returned, only a short hash of it."""
    if isinstance(body, str):
        try:
            body = json.loads(body)
        except json.JSONDecodeError:
            return None
    if not isinstance(body, dict):
        return None
    messages = body.get("messages") if isinstance(body.get("messages"), list) else []
    roles = [m.get("role") for m in messages if isinstance(m, dict)]
    user_texts = [_text(m.get("content")) for m in messages
                  if isinstance(m, dict) and m.get("role") == "user"]
    last_user = user_texts[-1] if user_texts else ""
    rf = body.get("response_format")
    max_tokens = body.get("max_tokens", body.get("max_completion_tokens"))
    return {
        "message_count": len(messages),
        "roles": roles[:8],
        "has_system": "system" in roles,
        "tools": len(body["tools"]) if isinstance(body.get("tools"), list) else 0,
        "response_format": rf.get("type") if isinstance(rf, dict) else None,
        "max_tokens": max_tokens if isinstance(max_tokens, int) else None,
        "stream": body.get("stream") is True,
        "last_user_chars": len(last_user),
        "last_user_sha256_12": hashlib.sha256(last_user.encode("utf-8")).hexdigest()[:12],
        "_last_user": last_user,          # stripped before output; used for probe matching only
    }


def classify_probe(sig: Optional[Dict[str, Any]], extra_prompts: Iterable[str] = ()) -> Optional[str]:
    """Return the probe label when the body matches a known probe exactly, else None.

    A probe is a single user message with no system prompt, no tools and no response_format whose
    text equals a known probe prompt (and whose max_tokens equals the probe's, when it sets one)."""
    if not sig or sig["message_count"] != 1 or sig["roles"] != ["user"]:
        return None
    if sig["tools"] or sig["response_format"]:
        return None
    text = sig["_last_user"].strip()
    for label, prompt, max_tokens in BUILTIN_PROBES:
        if text == prompt and (max_tokens is None or sig["max_tokens"] == max_tokens):
            return label
    for prompt in extra_prompts:
        if isinstance(prompt, str) and prompt.strip() and text == prompt.strip():
            return "operator-probe"
    return None


def public(sig: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if sig is None:
        return None
    return {k: v for k, v in sig.items() if not k.startswith("_")}


def builtin_probe_list() -> List[Dict[str, Any]]:
    return [{"label": label, "prompt": prompt, "max_tokens": mt} for label, prompt, mt in BUILTIN_PROBES]
