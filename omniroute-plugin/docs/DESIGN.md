# Design: hermes-omniroute v0.1

Row ids (`O-…`, `H-…`, `L…`) refer to [CAPABILITIES.md](CAPABILITIES.md).

## Integration model

Hermes sends every request to an OmniRoute **combo by name**: the combo name is the `model`
string, and `base_url` is the gateway (H-MODEL-1..3). OmniRoute picks provider, host and model per
request (O-COMBO-1..2, O-ROUTE-*). Backends are swapped inside the combo and never in Hermes
config. This plugin does not route anything itself. It checks that this arrangement is wired
correctly and helps diagnose it, read-only.

## D1. Can Hermes consume OmniRoute's MCP server with read-only scopes?

**Yes, with two caveats.**

- Hermes speaks Streamable HTTP and stdio MCP, sends static headers, and takes an
  include-allowlist of tool names (H-MCP-1, H-MCP-2). OmniRoute serves Streamable HTTP at
  `/api/mcp/stream` and authenticates a Bearer key with `manage`, `admin` or `mcp:connect`
  (O-MCP-1, O-MCP-2, O-AUTH-4). stdio (`omniroute --mcp`) also works when Hermes runs on the
  gateway host, because the stdio server opens OmniRoute's own database (O-MCP-3).
- **Caveat 1: `read:*` scopes do not protect anything by default.** Scope enforcement ships off
  (O-MCP-4), so an MCP connection can call every tool, `write:*` ones included. Read-only use is
  achieved in two layers:
  1. **Hermes-side `tools.include`** limited to the read tools (always effective). The example
     profile does this.
  2. **OmniRoute-side** `OMNIROUTE_MCP_ENFORCE_SCOPES=true` plus a key whose scopes are only
     `mcp:connect` + `read:*`. Optional hardening; it is the operator's switch, not the plugin's.
- **Caveat 2:** no OmniRoute tool carries `readOnlyHint` (O-MCP-8). Under Hermes
  `trust: untrusted`, every call would therefore need approval. With the include-list in place,
  `trust: full` (the default) is the practical setting.

The example include-list contains only tools whose declared scopes are all `read:*`
(`OR:open-sse/mcp-server/schemas/tools.ts`). Their handlers were spot-checked, not audited the way
this plugin's own allowlist was. For example, `omniroute_check_quota` reads
`/api/usage/quota`, which is in-memory only.

**Decision:** the plugin does **not** reimplement what the MCP server provides. It ships a
documented MCP config ([examples/hermes-profile.yaml](../examples/hermes-profile.yaml)) for the
MCP-covered capabilities. It adds native tools **only** for data the MCP server does not expose
(next section).

## D2. Why native read-only tools are still needed

The checks this integration needs depend on fields no MCP tool returns (O-MCP-6):

- connection `provider` id, resolved base URL and `maxConcurrent` (`/api/providers`,
  `/api/provider-nodes`); none is exposed over MCP. They drive L1 and L4.
- per-leg pinned `connectionId`, combo `config.targetTimeoutMs`, `system_message` and
  `tool_filter_regex`. `omniroute_list_combos` strips them. They drive L2–L4 and the Hermes
  mismatch checks.
- `settings.comboDefaults`, the middle layer of the timeout cascade (O-COMBO-5).
- call logs and per-combo health (`/api/usage/call-logs[/id]`, `/api/usage/combo-health`). No MCP
  tool reads them. They drive L5 and L6.
- the Hermes profile's own model settings. Only an in-process Hermes plugin can read them, and
  only it can compare them with the gateway.

Calling the MCP server from the plugin (`ctx.call_mcp`, H-PLUG-9) would not help, because the
data is not there. Read-only safety is enforced in one module, `endpoints.py`: a hard-coded
allowlist of eight GET paths with fixed query keys. Each was read for side effects (see D5).

## D3. Capability split

| Capability | Where |
|---|---|
| Gateway health, uptime, circuit breakers, rate limits | **provided by OmniRoute MCP** (`omniroute_get_health`) |
| Combo list with strategy and legs (provider/model only) | **provided by OmniRoute MCP** (`omniroute_list_combos`, lossy: O-MCP-6) |
| Live per-combo metrics | **provided by OmniRoute MCP** in principle (`omniroute_get_combo_metrics`), but it sends the wrong query parameter at this SHA (O-MCP-7). Also returned by this plugin's `omniroute_combo_health`. |
| Per-provider metrics, quota, cost, route simulation/explanation | **provided by OmniRoute MCP** (`omniroute_get_provider_metrics`, `omniroute_check_quota`, `omniroute_cost_report`, `omniroute_simulate_route`, `omniroute_explain_route`) |
| Connections: provider id, base URL (+ source), host class, max_concurrent | **provided by this plugin** (`omniroute_connections`) |
| Combos with full legs, pinned connection, joined max_concurrent, effective per-leg timeout | **provided by this plugin** (`omniroute_combos`) |
| L1–L4 audit (gated self-hosted backends, timeout scope, p2c, connection caps) | **provided by this plugin** (`omniroute_audit`) |
| Hermes settings ↔ combo names cross-check, task/leg mismatches | **provided by this plugin** (`hermes_combo_check`) |
| Call logs filtered by exact combo name, request-body signature, probe detection | **provided by this plugin** (`omniroute_call_logs`) |
| Per-combo health (`/api/usage/combo-health`) | **provided by this plugin** (`omniroute_combo_health`) |
| Error classification of OmniRoute's 404 | **not needed**: Hermes already classifies it as `model_not_found` (L10 CHANGED, D4) |
| Combo decision trace by invocation id | **not available** in v0.1: needs the `X-OmniRoute-Combo-Trace` response header, which Hermes does not surface to plugins (O-HEALTH-3) |
| Plan a change to one existing combo: expected stored diff, freshness token, reversibility, findings on the proposed state, server-may-reject list | **provided by this plugin** (`omniroute_combo_plan`, read-only, toolset `omniroute`; D7) |
| Apply a plan (one PUT, read-back verification, automatic restore) and roll it back | **provided by this plugin** (`omniroute_combo_apply`, `omniroute_combo_rollback`, toolset `omniroute_write`, every call through Hermes's approval gate; D7) |
| Combo writes via MCP (`omniroute_switch_combo`, `omniroute_set_routing_strategy`, `omniroute_create_combo`) | **not used**: no plan, freshness, verification or restore; excluded by the example include-list |
| Combo create, delete, rename; any connection, provider, settings, budget, cache, compression or resilience write; metric reset | **not available** by design |
| Routing guidance for the agent (L1–L11 as decisions, plan/apply/rollback workflow) | **provided by this plugin**: bundled skill `omniroute:omniroute-routing` (`ctx.register_skill`, H-SKILL-1) |
| OmniRoute `skills/*/SKILL.md` | **not bundled.** They document write endpoints too (O-SKILL-1). Operators may install them into Hermes separately. |

## D4. Error-classification hook: skipped

The spec makes this hook conditional on L10 being CONFIRMED. It came back **CHANGED**:

- OmniRoute's combo-exhausted 404 body is
  `{"error": {"message": "<model>: model — Not Found (HTTP 404)", "type": "invalid_request_error", "code": "model_not_found"}, …}`
  (O-ERR-1, O-ERR-2).
- Hermes's classifier lowercases `str(error)` (the OpenAI SDK includes the JSON body there) and
  matches the `model_not_found` pattern, giving `model_not_found`, `retryable=False`,
  `should_fallback=True`.
- Executed against Hermes at the recorded SHA in `tests/test_l10_classifier.py`. A second test
  shows that the message text alone would still be `unknown`, so the match depends on the `code`
  field.

A hook would duplicate a verdict Hermes already reaches. The tests pin the current behaviour, so
a future change on either side fails CI and reopens the decision. `plugin.yaml` declares
`provides_hooks: []`, and `tests/test_hermes_integration.py` asserts that no hook is registered.

## D5. Read-only guarantees

- `endpoints.ALLOWLIST` is the only place a path can come from. `client.OmniRouteClient` sends
  `GET` only, with the key in `Authorization: Bearer` (never in the URL, O-AUTH-1), and refuses
  unknown names, unknown query keys and path-param characters outside `[A-Za-z0-9_.:-]`.
- Side-effect review of each allowlisted GET handler at the recorded SHA: no `.run(`, INSERT,
  UPDATE, DELETE, fetch, sync or spawn on the GET path. The single exception is
  `GET /api/usage/call-logs/{id}`, which can clear a dangling artifact pointer or mark a row
  corrupt (O-LOGS-3). It is **opt-in** (`include_body_signature=true`) and its tool description
  says so. Excluded on purpose: `/api/monitoring/health` (the `deep=1` outbound probe),
  `/api/resilience` and every `/api/mcp/*`, `/api/providers/*/test|sync-models|refresh` route.
- Connection records are copied through a field allowlist (`model.Connection.public`), so a
  revealed `apiKey` (O-CONN-1) never reaches tool output.
- The strict mock server in `tests/` fails every test that makes a non-GET or off-allowlist
  request.

## D6. Judgment calls not dictated by the spec

1. **L2 rule direction.** The rule follows L2 as stated: leg timeout ≥ caller per-call timeout,
   compared only at matching scope (both per single request). `omniroute_audit` takes the
   per-call timeout as an argument. `hermes_combo_check` reads it from Hermes config
   (`auxiliary.<task>.timeout`, `providers.<id>[.models.<m>].timeout_seconds` /
   `request_timeout_seconds`). When only Hermes's `HERMES_API_TIMEOUT` default (1800 s) applies to
   the main model, the plugin reports a note instead of a warning, because that is not an
   operator choice.
2. **"Self-hosted" host test** (L1): loopback, RFC 1918/ULA private, link-local, `localhost`, and
   `.lan` / `.local` / `.localhost` names. No DNS lookups.
3. **Tool-calling mismatch:** OmniRoute stores no tool-capability record for generic or local
   nodes. The checkable, source-backed mismatches are a combo's `tool_filter_regex` and
   `system_message` (O-COMBO-3), so those are what is flagged.
4. **Probe detection** matches only exact bodies from OmniRoute's own probes (O-LOGS-4), plus
   prompts the operator passes. Nothing is guessed from size or timing.
5. **No dependency beyond the standard library**, so tests run without Hermes and the plugin adds
   nothing to Hermes's pinned dependency set.

## D7. Write-safety model (v0.2)

**Principle.** A tool does one deterministic thing; the model decides whether to call it; Hermes
decides whether it may run. The write tools therefore enforce only mechanical invariants. Judgment
lives in the bundled skill, and permission lives in Hermes.

**Mechanical invariants (enforced by the tools).**

| Invariant | Where |
|---|---|
| Only `PUT /api/combos/{id}`, the single write route in the allowlist; never PATCH, POST, DELETE; `{id}` cannot name a sibling static route | `endpoints.py` (O-WRITE-1) |
| One existing combo per apply; no rename, create or delete; no quota-share (`qtSd/`) combos | `combo_write.plan` (O-WRITE-5) |
| Only keys the update schema accepts (unknown keys would be silently stripped); `null` only on nullable keys | `combo_write.plan` (O-WRITE-2, O-WRITE-4) |
| Freshness: SHA-256 of the full stored combo at plan time must equal the live hash at apply time; plans expire (15 min default, 1 h max) and apply once | `normalize.content_hash`, `write_store` (O-WRITE-7) |
| Reversibility: refuse (mark `applicable: false`) a plan whose prior state rollback could not restore through the merge route | `combo_write.irreversibility` (O-WRITE-4) |
| Read-back: the PUT response **and** a fresh GET must equal the plan's expected stored result (server-set `updatedAt` excluded); any difference fails, inside or outside the planned fields | `combo_write.compare_stored` |
| Automatic restore on a failed verification or an unexplained post-error state, then verification of the restore | `ComboWriter._restore` |
| Rollback sends the complete snapshot (every writable key it has, explicit `null` for nullable keys the apply added), refuses if the combo changed since the apply unless `force=true`, and verifies the read-back | `ComboWriter.rollback` |

**Expected stored result.** Reproduced from source: the zod config transform (zero-latency flag),
the dead-key strip, `normalizeComboStep` for `models`, the shallow merge, null-deletes and the
whole-record re-normalization (O-WRITE-3). Not reproduced, and named in every plan: zod coercion
and defaults inside `config` (`config.weights` is compared on the keys sent), zod range/length/enum
checks, composite tiers, the model-family invariant, and the server-set `updatedAt`.

**Refused as irreversible.**
1. Adding a non-nullable top-level key that is absent today (`isActive`, `isHidden`, `strategy`,
   `config`, `models`): the schema has no way to make it absent again.
2. Any plan on a combo whose stored `models` contain a provider-wildcard step: the rollback's
   complete `models` would be rejected by the schema.
3. Any plan on a combo whose stored `config` holds dead keys, or enables a zero-latency feature
   without its gate: the rollback's complete `config` would be rewritten by the server.
4. Any plan on a combo with no models (the schema rejects an empty array).

**Not refused (information only).** L1–L4 findings, unpinned legs, Hermes task effects
(`system_message`, `tool_filter_regex`, structured output on downgrading legs), a predicted DAG
reject. A server reject leaves the combo unchanged; apply verifies that by hash and reports the
server's error code.

**Permission (Hermes, not the plugin).**
- **Toolsets (H-GATE-1).** Read tools and the planner are in toolset `omniroute`; apply and rollback
  are in `omniroute_write`, which an operator enables or disables per platform with `hermes tools`.
  **Gap:** Hermes turns plugin toolsets on by default, and a plugin cannot change that. The README
  therefore tells operators to run `hermes tools disable omniroute_write` after install.
- **Approval (H-GATE-2).** A `pre_tool_call` hook returns `{"action": "approve"}` for every call
  of the two write tools, so Hermes asks a human (once / session / always / deny). With no human
  present it blocks (fail-closed). The hook carries no policy: it always asks, and only describes
  the pending change. `approvals.mode: off` or yolo bypass it, as they bypass every Hermes approval.
  That is the operator's setting.
- **Not used:** `capabilities:` (H-GATE-3) has no id for this. A `trust: untrusted` MCP shim
  (H-GATE-4) would give real per-call approval, but nothing H-GATE-2 does not already give; not
  built.

**Known limits.** OmniRoute writes are not conditional (O-WRITE-7). A write by someone else
between apply's hash check and its PUT is not detected before the PUT. It is caught afterwards
only if it touched a planned field (read-back mismatch, then restore) or any field (the next
rollback refuses without `force`). Side effects that a rollback cannot undo: cloud sync and the
DB backup already ran, and session model pins cleared when `models` was sent (O-WRITE-6).

## Deferred decisions

- Ship the OmniRoute MCP connection with the plugin instead of a documented snippet. Corrected in
  v0.2 (H-SKILL-1): native plugins **do** bundle skills (`ctx.register_skill`, now used for
  `omniroute:omniroute-routing`); only MCP server *definitions* from a package (`mcp.json`) are wired
  to portable Agent Plugin manifests. `capabilities:` is unrelated (consent for privileged host
  surfaces, H-GATE-3). The MCP snippet stays: becoming a portable package would drop the native
  `register(ctx)` tools and hook.
- `omniroute_combo_trace` once Hermes exposes response headers to plugins.
- Evaluate `tool_filter_regex` against Hermes's actual tool names (JavaScript and Python regex
  dialects differ, so v0.1 reports the regex rather than simulating it).
- Read `resilience.comboCooldownWait` (it raises the default leg-timeout floor). Not allowlisted
  in v0.1 because that GET handler takes no request argument and exposes settings wholesale.
