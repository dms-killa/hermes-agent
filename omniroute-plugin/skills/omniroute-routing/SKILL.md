---
name: omniroute-routing
description: How OmniRoute combos actually route requests, and how to change one safely with omniroute_combo_plan / apply / rollback. Load before editing any combo, before choosing a strategy, and before blaming a model for bad structured output.
---

# OmniRoute routing semantics and the combo write workflow

Hermes sends each request to an OmniRoute **combo by name**. OmniRoute picks provider, host and model
per request. You change behaviour by editing the combo, not Hermes config. Every claim below cites a
row of the plugin's `docs/CAPABILITIES.md` (e.g. `O-ROUTE-2`); read the row when unsure.

The tools hold no policy. They refuse only mechanical problems (stale plan, irreversible change, wrong
route). Whether a change is wise is **your** judgment, guided by this page. Whether it may run is the
operator's: the write tools sit in their own toolset and every call goes to Hermes's approval gate
(`H-GATE-1`, `H-GATE-2`).

## Decision guidance (L1–L11)

1. **Bad JSON from a small model? Check the provider type first (L1, O-L1-1).** A backend registered as a
   generic `openai-compatible-*` node (or `kilocode`) gets `response_format: json_schema` rewritten to
   `json_object` plus a pasted schema. Grammar enforcement is gone. Do **not** respond by pointing the
   combo at a bigger model. The fix is registering the server under a local-provider id
   (`O-L1-2`), which is a connection change and outside these tools; tell the operator.
2. **Compare timeouts per request (L2, O-COMBO-4..6).** `config.targetTimeoutMs` (default 120000 ms)
   bounds one attempt on one leg. Compare it with the caller's timeout for one call, never with a
   multi-call budget. `0` means "unset", not "no timeout".
3. **p2c does not balance live load (L3, O-ROUTE-2, O-ROUTE-3).** It picks the better of two random legs
   by past success rate and latency, and counts nothing in flight. If you need concurrent requests to
   spill over in order, use `priority` with every leg pinned to a connection and a `max_concurrent`
   on each connection (`O-ROUTE-4`). Never pick p2c "to spread load".
4. **`max_concurrent` and connection edits have blast radius (L4, O-CONN-2, O-ROUTE-4).** The cap belongs
   to the connection and is shared by every combo and request using it. Before pinning a leg to a
   connection, look at the plan's `L4_SHARED_CONNECTION_CAP` findings: other combos already lean on
   it. Repointing several combos onto one host multiplies its load; do it one plan at a time and
   check each plan's findings.
5. **Unpinned legs do not spill over (O-COMBO-2, O-ROUTE-4).** Only a pinned leg is skipped when its
   connection is full; the plan flags unpinned legs (`UNPINNED_LEG`).
6. **Call logs and metrics (L5, L6, O-LOGS-1..4, O-HEALTH-1..2).** Logs are /v1 traffic only; filter by
   exact combo name. Health probes look like traffic: use `omniroute_call_logs` with
   `include_body_signature=true` before trusting health numbers on a quiet combo.
7. **Prove the body before changing models or limits (L7, O-LOGS-3).** Fetch the stored request body
   (body signature) or ask the operator for a direct-vs-gateway replay. Change model, context or
   limits only once you know what reached the backend.
8. **Combo fields that change Hermes itself (O-COMBO-3).** `system_message` replaces every system
   message (Hermes's prompt, tool guidance, memory). `tool_filter_regex` drops tools whose names do
   not match. Never set either on a combo Hermes's agent loop uses unless the operator asked.
9. **Management needs a `manage` key in a header (L8, O-AUTH-1, O-AUTH-2).**
10. **Read-only checks belong to OmniRoute's MCP server where it covers them (L9, O-MCP-5, O-MCP-6).**
11. **404 from the gateway falls back already (L10, O-ERR-2).** No special handling needed.
12. **Stickiness and compression (L11, O-ROUTE-5).** A conversation can stay on one leg by design;
    enabling `config.compressionMode` changes prompt bytes and defeats prefix caches on self-hosted
    servers.

## When to plan

Plan whenever you intend to change a combo, even to "just check". `omniroute_combo_plan` writes nothing
to OmniRoute (`O-WRITE-1`; the plan only reads). One plan covers one existing combo. Rename, create,
delete, and every connection, provider or settings change are outside these tools.

`changes` uses the server's top-level fields (`O-WRITE-2`). `changes.config` is a key-by-key patch
(`null` removes a key): the server replaces `config` as a whole, so the plugin sends the full result
(`O-WRITE-3`). `null` on a nullable top-level field resets it (`O-WRITE-4`).

## How to read a plan

- `diff`: what the stored combo will look like, after the server's normalization (`O-WRITE-3`).
- `applicable` / `irreversible_reasons`: the plan is refused when rollback could not restore today's
  state through the merge route (`O-WRITE-4`). Re-plan differently; do not work around it.
- `findings`: the audit on the **proposed** state. `new_with_this_change: true` marks problems this
  change introduces. They never block apply. Read them, decide, and say why in your reply.
- `server_may_reject`: checks only the server can make (`O-WRITE-5`). A rejected PUT leaves the combo
  unchanged and the tool reports the server's code.
- `side_effects`: cloud sync, DB backup, and cleared session pins when `models` is sent (`O-WRITE-6`).

## Apply, verify, roll back

1. Show the operator the plan (diff + findings) and say what you expect to improve.
2. `omniroute_combo_apply(plan_id)` goes through Hermes's approval prompt. It refuses a stale plan:
   if someone changed the combo since you planned, plan again rather than forcing anything.
3. Apply verifies the stored result. On a mismatch it restores the snapshot itself and reports both.
4. If the result is wrong in practice, **prefer `omniroute_combo_rollback(rollback_id)` over a
   corrective re-edit.** Rollback returns the combo to the exact snapshot; a second edit stacks a
   new unknown on top of the first. Then re-diagnose from the restored state.
5. Rollback refuses if the combo changed after your apply. `force=true` overwrites those changes:
   use it only when the operator confirms the later change should go.
