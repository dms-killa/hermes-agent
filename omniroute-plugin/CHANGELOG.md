# Changelog

## 0.2.0 (unreleased)

Combo write path with mechanical invariants. Judgment stays with the agent and with Hermes.

- New tools: `omniroute_combo_plan` (read-only, toolset `omniroute`), `omniroute_combo_apply` and
  `omniroute_combo_rollback` (toolset `omniroute_write`).
- One write route added to the allowlist: `PUT /api/combos/{id}` (plus `GET` on the same route).
- Mechanical invariants: one existing combo per apply, update-schema keys only, no rename,
  create, delete or quota-share combos; content-hash freshness and plan expiry; refusal of
  irreversible changes; read-back verification with automatic restore; complete-snapshot rollback
  with explicit resets and a `force` flag for third-party edits.
- Hermes-native gating: a separate toolset for writes, and a `pre_tool_call` hook that sends every
  write call to Hermes's human approval gate (fail-closed without a human). Documented gap: Hermes
  enables plugin toolsets by default.
- Bundled skill `omniroute:omniroute-routing` (`ctx.register_skill`), with every claim citing
  docs/CAPABILITIES.md.
- docs/CAPABILITIES.md: combo write surface (O-WRITE-1..7), Hermes gating (H-GATE-1..5), correction
  of a v0.1 design note (H-SKILL-1), and a re-confirmation table for the write-surface facts. The
  claim that combos have no `updatedAt` came back CHANGED.
- Tests: a stateful mock combo store (merge, null-deletes, normalization, DAG/409 rejects), round
  trip, automatic restore, server rejects, forced and unforced rollback, irreversibility,
  normalization golden tests, and the Hermes approval gate through the real dispatcher.
- Upstream SHAs unchanged: OmniRoute `fc5e2bcc`, hermes-agent `bcf29fa8`.

## 0.1.0 (unreleased)

First version. Read-only.

- Tools: `omniroute_audit`, `hermes_combo_check`, `omniroute_connections`, `omniroute_combos`,
  `omniroute_call_logs`, `omniroute_combo_health`.
- Hard-coded allowlist of eight GET endpoints (`endpoints.py`). The one with a side effect, the
  call-log detail route, is opt-in.
- No hooks: the planned error-classification hook was dropped because Hermes already classifies
  OmniRoute's 404 as `model_not_found` (lesson L10 CHANGED, docs/DESIGN.md D4). Pinned by tests.
- docs/CAPABILITIES.md (code-derived, with commit SHAs) and docs/DESIGN.md (MCP vs native split).
- Fixture-based tests with a strict mock gateway; optional live smoke test.
- Verified against OmniRoute `fc5e2bcc` and hermes-agent `bcf29fa8`.
