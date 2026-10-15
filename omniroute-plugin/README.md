# hermes-omniroute

A read-only [Hermes Agent](https://github.com/NousResearch/hermes-agent) plugin for people who run
Hermes behind an [OmniRoute](https://github.com/diegosouzapw/OmniRoute) gateway.

The model it supports: **Hermes sends every request to an OmniRoute combo by name. OmniRoute
decides which provider, host and model serves it.** Hermes never names a backend. Backends are
swapped in the combo, not in Hermes config. The plugin checks that this wiring is correct and
helps diagnose it. Since v0.2 it can also change **one existing combo at a time**, through a
plan → apply → rollback path that is reversible, freshness-checked and verified by read-back.

- Seven read-only tools (toolset `omniroute`) and two write tools (toolset `omniroute_write`).
- One `pre_tool_call` hook: every call of the two write tools goes to Hermes's approval prompt.
- A bundled skill, `omniroute:omniroute-routing`, that teaches the agent OmniRoute's routing
  semantics. The tools hold no policy; the skill holds the guidance.
- Calls a hard-coded allowlist of nine endpoints, defined in [`endpoints.py`](endpoints.py). All
  are `GET`-only except `/api/combos/{id}`, which also allows `PUT`: the single write route.
- Standard library only. Tests run without a gateway and without Hermes.
- Every finding cites a row of [docs/CAPABILITIES.md](docs/CAPABILITIES.md), which in turn cites
  OmniRoute and Hermes source at recorded commits.

## Tools

| Tool | What it answers |
|---|---|
| `omniroute_audit` | Which self-hosted backends are registered as generic `openai-compatible-*` nodes (structured output silently downgraded)? For each combo: strategy, legs, pinned connection, that connection's `max_concurrent`, and the effective per-leg timeout. Given your per-call timeout and concurrency, where do they not fit? Which combos use `p2c` while you expect load spreading? |
| `hermes_combo_check` | Does every model setting of this Hermes profile that points at the gateway (main, auxiliary, delegation, fallbacks) name an existing combo? Do the combo's legs accept what that task sends (json_schema on downgrading legs, tools filtered by `tool_filter_regex`, system prompt replaced by `system_message`, per-call timeout longer than the leg timeout)? |
| `omniroute_connections` | Connections with provider id, resolved base URL (and where it came from), host class and `max_concurrent`. Keys are never returned. |
| `omniroute_combos` | Combos with every leg, the pinned connection, `max_concurrent` and the effective `targetTimeoutMs`. |
| `omniroute_call_logs` | Recent call logs filtered by **exact** combo name (OmniRoute's own `combo` filter only checks presence). With `include_body_signature=true`, each row gets the shape of its stored request body and is marked probe or real. |
| `omniroute_combo_health` | Per-target success rate, latency and quota for one combo, plus live combo metrics. |
| `omniroute_combo_plan` | **Read-only.** For one change to one existing combo: the exact field-level diff the server will store, a freshness token, a reversibility verdict, findings on the proposed state (L1–L4, unpinned legs, effects on Hermes tasks), what the server may still reject, and side effects. |
| `omniroute_combo_apply` | **Write** (toolset `omniroute_write`, approval required). Applies a plan with one PUT, verifies the stored result, and restores the snapshot automatically if verification fails. Returns a `rollback_id`. |
| `omniroute_combo_rollback` | **Write** (toolset `omniroute_write`, approval required). Restores the pre-apply snapshot and verifies it. Refuses if the combo changed after the apply, unless `force=true`. |

## Install

The plugin is a plain directory. Put it in the profile's plugin directory and enable it:

```bash
# HERMES_HOME is the active profile's home (default ~/.hermes)
git clone <this repo> "$HERMES_HOME/plugins/omniroute"      # or copy / symlink the directory
hermes plugins enable omniroute
hermes plugins doctor omniroute        # expect: registrations: 9 tool(s), 1 hook(s)
hermes tools disable omniroute_write   # keep combo writes off until you want them (see below)
```

Or add `omniroute` under `plugins.enabled` in `config.yaml`. The tools are available from the
next session.

To see the registered tool names and run one tool through Hermes's own dispatcher, use the
Python environment Hermes runs in:

```bash
HERMES_HOME=... python scripts/verify_install.py omniroute_audit '{"per_call_timeout_ms": 300000}'
```

## Configuration

| Variable | Meaning |
|---|---|
| `OMNIROUTE_BASE_URL` | Gateway root, e.g. `https://omniroute.example.com`. A trailing `/v1` or `/api/v1` is stripped. |
| `OMNIROUTE_API_KEY` | An OmniRoute API key with the **`manage`** (or `admin`) scope. |

These are the same names OmniRoute's own MCP server uses. Put them in the profile's `.env`.
Hermes reads them through the profile secret scope, so each profile can point at its own gateway.
The tools stay hidden until both are set.

**Scope.** OmniRoute's management routes (`/api/*`) accept a key only in the
`Authorization: Bearer` header and only with `manage` or `admin` scope. A plain `/v1` inference
key gets HTTP 403. The plugin never puts the key in a URL and never logs or prints it. The key is
read-only *in effect* because the plugin only sends allowlisted GETs, but OmniRoute has no
read-only management scope, so treat the key as an admin credential.

## Changing combos (v0.2)

### Who decides what

| Question | Decided by |
|---|---|
| Is this change wise? | The agent, using the plan's findings and the bundled skill |
| May the write tools be offered at all? | The operator: toolset `omniroute_write` in `hermes tools` |
| May this particular apply / rollback run? | A human, at Hermes's approval prompt (once / session / always / deny) |
| Is the plan still valid, reversible, and did the write land as planned? | The tools, mechanically |

### Gating, and its gap

- **Toolset.** The write tools are in their own toolset, `omniroute_write`. **Hermes enables a
  plugin's toolsets by default, and a plugin cannot change that** (the default-off list is
  hard-coded in Hermes core). Run `hermes tools disable omniroute_write` after installing, and
  `hermes tools enable omniroute_write` when you want writes.
- **Approval.** Every `omniroute_combo_apply` and `omniroute_combo_rollback` call is sent to
  Hermes's human approval gate through a `pre_tool_call` hook, the mechanism Hermes provides for
  plugin tools. With no interactive user or gateway present it **blocks** (fail-closed).
  `approvals.mode: off` and yolo mode bypass it, as they bypass every Hermes approval. Answering
  "always" approves that tool from then on. Choose once/session if you want to see each write.
- Hermes has no per-tool "approval required" flag at registration, and `capabilities:` has no id
  for outbound writes. A `trust: untrusted` MCP shim would also give real approval, but nothing
  more than the hook already does, so v0.2 does not build one.

### What the tools guarantee

- Only `PUT /api/combos/{id}`; one existing combo per apply; no rename, create or delete; never
  quota-share (`qtSd/`) combos.
- The plan is refused when its change could not be undone through OmniRoute's update route, for
  example adding `isActive` to a combo that has none. See [DESIGN.md D7](docs/DESIGN.md).
- Apply refuses an unknown, expired, used or irreversible plan, and a plan whose combo changed since
  planning (content hash of the full stored combo).
- Apply snapshots the combo, sends one PUT, compares both the response and a fresh read with the
  plan's expected result, and restores the snapshot on any difference.
- Rollback sends the complete snapshot, including explicit resets for fields the apply added, and
  verifies the result.

### Example operator flow

1. You ask the agent to give the `spill` combo a 300 s leg timeout. It loads
   `omniroute:omniroute-routing` and calls `omniroute_combo_plan`:
   ```json
   {"combo": "spill", "changes": {"config": {"targetTimeoutMs": 300000}},
    "caller_limits": {"per_call_timeout_ms": 300000, "concurrency": 4}}
   ```
   It shows you the diff (`config.targetTimeoutMs: 600000 → 300000`) and the findings: here an L4
   warning that four concurrent requests exceed the legs' combined `max_concurrent` of 3.
2. You review. If you want the write, enable it: `hermes tools enable omniroute_write`.
3. The agent calls `omniroute_combo_apply` with the `plan_id`. Hermes prompts you:
   *"Tool 'omniroute_combo_apply' requires approval (apply plan … to OmniRoute combo 'spill': one
   PUT changing config.targetTimeoutMs …)"*. You answer `once`.
4. The tool reports `verification: passed` and a `rollback_id`.
5. If traffic behaves worse, ask for `omniroute_combo_rollback` with that id. You approve again;
   the combo returns to the snapshot. Prefer this over a corrective re-edit. If someone else edited
   the combo in between, the rollback refuses and shows their change; `force=true` overwrites it.
6. Turn writes off again when done: `hermes tools disable omniroute_write`.

## What the MCP server does, and what this plugin adds

OmniRoute ships its own MCP server (health, combos, quota, cost, route simulation and more).
Hermes can consume it, so this plugin does not duplicate it. Use both:

- **OmniRoute MCP** (configure it in `mcp_servers`, see
  [examples/hermes-profile.yaml](examples/hermes-profile.yaml)): gateway health and circuit
  breakers, combo list, quota, cost, route simulation and explanation.
- **This plugin:** everything that needs data the MCP tools do not return. That covers connection
  base URLs and `max_concurrent`, pinned legs, combo timeouts, `system_message` and
  `tool_filter_regex`, call logs and combo health, plus the Hermes-side cross-check that only an
  in-process plugin can do.

Two cautions for the MCP connection, both from OmniRoute source:

1. OmniRoute's MCP scope enforcement is **off by default**, so `read:*` scopes alone do not make
   the connection read-only. The example limits it with Hermes's `tools.include`.
2. OmniRoute's MCP tools carry no `readOnlyHint`. Hermes `trust: untrusted` would therefore ask
   approval for every call. With the include-list, the default `trust: full` is fine.

The full split is in [docs/DESIGN.md](docs/DESIGN.md#d3-capability-split).

## Operator guidance: lessons from running this stack

Each lesson was checked against source at the commits in CAPABILITIES.md. The verdicts are in
[CAPABILITIES.md §3](docs/CAPABILITIES.md#3-lessons-l1l11-verdicts-at-the-recorded-shas).

1. **Provider type is a correctness setting (L1).** OmniRoute rewrites
   `response_format: json_schema` into `json_object` plus a schema pasted into the system prompt
   for any provider id starting with `openai-compatible-` (and for `kilocode`). A self-hosted
   llama.cpp or Ollama registered as a generic openai-compatible node therefore loses grammar
   enforcement silently. Small models then return bare arrays or fenced JSON, and the usual
   response of moving to bigger models treats the symptom. Register such servers with a
   local-provider id (`llama-cpp`, `ollama-local`, `lm-studio`, `vllm`, …). `omniroute_audit`
   lists the affected connections.
2. **Compare timeouts at the same scope (L2).** A combo's `targetTimeoutMs` (default 120000 ms)
   bounds one attempt on one leg. Compare it with the caller's timeout for one request, never
   with a budget spanning many calls. If one call may legitimately take 300 s, the leg timeout
   must be at least 300000. For streaming requests it bounds only time-to-first-headers, which
   still includes prompt prefill.
3. **p2c is not load balancing (L3).** p2c picks the better of two random legs by historical
   success rate and latency. It does not count requests in flight. For ordered spill-over under
   concurrency, use `priority`, pin each leg to a connection, and set `max_concurrent` on each
   connection. A full connection is skipped before dispatch and the next leg is tried.
4. **`max_concurrent` belongs to the connection (L4).** It is shared by every combo and every
   request that uses that connection. Size it for the total.
5. **Read call logs with care (L5).** They record proxied `/v1` inference traffic, not management
   reads. OmniRoute's `combo` filter only checks that a combo was involved, so filter by combo
   name on the client (`omniroute_call_logs` does). Per-combo data lives at
   `/api/combos/metrics?combo=<name>` and under `/api/usage/combo-*`.
6. **Probes look like traffic (L6).** Health checks and test requests land in the same logs and
   metrics as real requests. On a low-traffic combo they can dominate the health numbers. Tell
   them apart by request body, not by timing. `omniroute_call_logs` with
   `include_body_signature=true` recognises OmniRoute's own probe bodies and any prompts you pass
   in `probe_prompts`.
7. **Prove the body first (L7).** Before changing model, context size, batch size or retries to
   fix a structured-output failure, confirm what body reached the backend. Use the stored request
   body (the body signature above), or replay the same request directly and through the gateway.
8. **Management needs a `manage` key in a header (L8).** See Configuration.
9. **Use OmniRoute's MCP server for what it covers (L9)**, with the include-list above.
10. **Gateway 404s fall back correctly (L10).** At the recorded commits, Hermes classifies
    OmniRoute's combo-exhausted 404 as `model_not_found` and falls back. It matches the body's
    `code: "model_not_found"`, not the message text. This plugin therefore ships no
    error-classification hook. `tests/test_l10_classifier.py` fails if that changes.
11. **Stickiness and compression interact with caches (L11).** Session stickiness and
    prompt-cache affinity can keep a conversation on one leg (by design, for provider prompt
    caches). Prompt compression is off by default. When enabled on a combo, it changes prompt
    bytes and so defeats prefix-cache reuse on llama.cpp-style servers.

## Development

```bash
python -m pytest -q tests                      # standard library + pytest; Hermes tests skip
# with a Hermes checkout installed in the same environment, the integration tests run too
OMNIROUTE_BASE_URL=... OMNIROUTE_API_KEY=... python -m pytest -q tests/test_live_smoke.py
python scripts/serve_mock.py                    # fixture-backed strict mock for manual trials
HERMES_HOME=... python scripts/demo_write_cycle.py spill '{"description": "x"}'   # plan/apply/rollback
```

- `tests/mock_server.py` is a strict stand-in for the gateway with a stateful combo store that
  simulates OmniRoute's update route (merge, null-deletes, reproducible normalization, DAG and
  quota-share rejects). Any other non-GET request, a path outside
  the allowlist, or an unexpected query key fails the test.
- Fixtures are synthetic and shaped after OmniRoute source. See
  [tests/fixtures/PROVENANCE.md](tests/fixtures/PROVENANCE.md).

## License

MIT, the same license as hermes-agent ([LICENSE](LICENSE)).
