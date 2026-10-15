"""Finding records. Every finding cites rows of docs/CAPABILITIES.md (row ids like ``O-L1-1`` and
lesson ids ``L1``..``L11``), so a reader can check the claim against OmniRoute source."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

CAPABILITIES_DOC = "docs/CAPABILITIES.md"
SEVERITIES = ("error", "warning", "info")


def finding(code: str, lesson: str, severity: str, message: str, *, cite: Iterable[str],
            action: str = "", subject: Dict[str, Any] | None = None) -> Dict[str, Any]:
    if severity not in SEVERITIES:
        raise ValueError(severity)
    rows = list(cite)
    if not rows:
        raise ValueError("a finding must cite at least one CAPABILITIES.md row")
    return {"code": code, "lesson": lesson, "severity": severity, "message": message,
            "action": action, "subject": subject or {},
            "cite": [f"{CAPABILITIES_DOC}#{r.lower()}" for r in rows]}


def summarize(findings: List[Dict[str, Any]]) -> Dict[str, Any]:
    counts = {s: sum(1 for f in findings if f["severity"] == s) for s in SEVERITIES}
    status = "clean" if counts["error"] == 0 and counts["warning"] == 0 else "attention"
    return {"status": status, **counts}
