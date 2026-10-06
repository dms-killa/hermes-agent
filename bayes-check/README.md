# bayes-check

A Hermes Agent plugin that puts an answer's factual claims through a deterministic Bayesian
check and stamps the answer with a calibrated-confidence footer before it is saved.

- **Opt-in per session:** `/bayes on`. Sessions you don't turn on are untouched.
- **Zero extra LLM calls:** all the math runs in Python. Claims are extracted by the model you
  are already talking to, in the same turn, through the `bayes_score` tool.
- **Footer is enforced:** a `transform_llm_output` hook adds the footer whether or not the model
  cooperated. A turn without a scoring call gets `⚠ bayes-check: unscored`.
- **Works with local models:** it detects local providers, blocks request-multiplying delegation
  on them, and accepts the sloppier tool arguments small models produce.

It does not make prose true. It separates what was checked from what wasn't, so an answer can be
audited.

## What an annotated answer looks like

```
The Moon is about 384,400 km from Earth on average.

---
**bayes-check** `bayes-checked:3f9a0c1d2b4e` — 1 supported
- ✓ supported p=0.92 [c1·5be1a0d2] The Moon is about 384,400 km from Earth.
```

Verdicts use `min_posterior_plain` (0.9) and `min_posterior_hedge` (0.5):
`supported` → state plainly, `uncertain` → qualify, `unsupported` → remove or mark unverified.

## Install

The plugin is a standalone directory. Copy or symlink it into the profile's plugins dir and
enable it:

```bash
# HERMES_HOME is the active profile's home (default ~/.hermes)
ln -s "$PWD/bayes-check" "$HERMES_HOME/plugins/bayes-check"
hermes plugins enable bayes-check
hermes plugins doctor bayes-check --ci     # or: cd bayes-check && hermes plugins doctor . --ci
```

Or add it under `plugins.enabled` in `config.yaml`. Hooks go live as soon as the plugin loads.
The tool, skill and system-prompt section take effect from the next session.

## Use

| Command | Effect |
|---|---|
| `/bayes on` / `/bayes off` | Opt this session in or out |
| `/bayes status` | Mode, local/cloud classification, scored turns, skipped gates, ledger size |
| `/bayes correct <hash\|id> true\|false` | Record ground truth for a footer line. This is the learning signal. |
| `/bayes refit` | Recalibrate likelihood ratios and claim-type base rates from the labels (CPU only) |

On an opted-in turn:
1. `pre_llm_call` adds one short instruction to the turn.
2. The model calls `bayes_score` with its claims, priors and typed evidence.
3. The handler scores the claims and records them.
4. `post_tool_call` ties that call to the turn by `turn_id`.
5. `transform_llm_output` appends the footer.

The bundled skill (`skill_view("bayes-check:bayes-check")`) documents the procedure and the
likelihood-ratio table.

## Settings (`plugins.entries.bayes-check.settings`)

| Key | Default | Meaning |
|---|---|---|
| `mode` | `annotate` | `off` (inert) · `annotate` (an unscored footer lists checkable-looking sentences) · `nudge` (an unscored footer only marks the answer) |
| `local_mode` | `auto` | `auto` · `force_local` · `force_cloud` |
| `min_posterior_plain` | `0.9` | at or above → supported |
| `min_posterior_hedge` | `0.5` | below → unsupported |
| `block_delegation_local` | `true` | block `delegate_task` on opted-in local sessions |
| `auto_enable_on_research` | `false` | opt a session in when a message looks like a research or advisory question (keyword heuristic) |

## Local-mode behaviour

`auto` classifies each session from the `provider` and `base_url` that `pre_api_request` carries:
- **Local hosts:** localhost, loopback (`127.x`, `::1`), `0.0.0.0`, `*.localhost`, unix sockets.
- **Local providers:** ollama, llama.cpp, lmstudio, vllm, custom_openai.

If nothing has been seen yet, `auto` treats the session as cloud. `force_local` and `force_cloud`
override detection in either direction; use them for a local server behind a reverse proxy.

On an opted-in local session, `delegate_task` is blocked with a clear message. Every model-issued
delegation runs in the background, and each subagent is a full extra agent loop on the same GPU.

## State

- **In-process:** module dicts hold the opt-in set, the turn ledger (capped at 512 turns), the
  subagent ledger and the local/cloud map. They are lost on restart, which is intentional for
  opt-in.
- **Durable:** `plugin_db("bayes-check")` lives at
  `<HERMES_HOME>/plugin-data/bayes-check/data.db` (SQLite WAL), so each profile has its own.
  - It stores every scored claim, the evidence types behind it, labels, calibrated LRs and
    gate-skip events.
  - When the file passes 50 MB, the oldest *unlabelled* claims are deleted. Labels are kept.
  - Every write commits before returning.

## Non-goals (v1)

- No `ctx.llm` calls, no LLM-based claim extraction, no multi-sample calibration.
- No oracle I/O (HTTP, DB, subprocess) in the scorer or the gate. Evidence is what the model
  reports, weighted by the likelihood-ratio table. Oracles are a v2 item.
- No MCP backend and no state shared across profiles.
- No true block-before-finalize gate for prose. That needs a core change to `apply_stop_gates`
  and is an upstream feature request. This plugin annotates; it cannot veto.
- No background task that touches a model. Refit runs only when you call `/bayes refit`.

## Honest limits

- **A host timeout is detected after the fact, not prevented.**
  - If the transform overruns `plugins.hook_callback_timeout` (30 s by default), Hermes
    abandons it and delivers the raw text.
  - The plugin can't write a marker from an abandoned worker. Instead, `post_llm_call` logs a
    WARNING, records a `gate_skipped` event, and `/bayes status` shows the count.
  - The transform does no I/O and caps its own work, so it should stay well under the bound.
- **Evidence is self-reported.** The model chooses the evidence it lists. The table only limits
  how much weight it can claim: same-type sources are discounted, any single LR is clamped, and
  the total log-odds shift per claim is capped. `/bayes correct` and `/bayes refit` calibrate the
  table over time.
- **Subagents:** each child of an opted-in parent is opted in too, so its own transform annotates
  the summary it returns. The parent's footer lists children that never scored. Background
  results that arrive in a later parent turn are flagged as unverified in that turn's injected
  context.

## Verify-in-dev results

Each item was checked in the code and with runtime tests that load the plugin through Hermes'
real discovery into a temporary `HERMES_HOME` (`tests/test_hermes_integration.py`).

1. **Plugin-registered tools emit `post_tool_call` with the correct `turn_id`: CONFIRMED, with
   one correction.**
   - Both emission paths carry the turn id:
     - `model_tools._emit_post_tool_call_hook` on the dispatcher path.
     - `agent.inline_tool_executors.emit_terminal_post_tool_call` on the agent-loop executor
       path. The executor suppresses the inner hook, and the terminal emitter reads
       `agent._current_turn_id`.
   - Tests: `test_verify_1_*`.
   - **Correction:** the tool *handler* does not receive `turn_id`. `model_tools._execute_tool`
     forwards only `task_id`, `session_id` and `user_task`. So the handler stamps a `call_id`
     into its result, and `post_tool_call`, which has `turn_id` and the result, binds the call
     to the turn (DESIGN.md D7).
2. **`subagent_stop` fires for background-dispatched children: CONFIRMED, with one caveat.**
   - Path: background runner `_execute_and_aggregate` → `_finalize_child_results` →
     `_fire_subagent_stop_hooks`, in `tools/delegate_tool_dispatch.py` and
     `delegate_tool_results.py`.
   - **Caveat:** its `parent_turn_id` is the parent's turn *when the child finishes*, not the
     turn that dispatched it. The ledger is therefore keyed by `child_session_id`, and the
     dispatching turn comes from `subagent_start`.
   - Test: `test_verify_2_subagent_stop_fires_from_background_finalizer`.
3. **`pre_tool_call` block shape: `{"action": "block", "message": str}`.** A block with no
   message is ignored (`_get_pre_tool_call_directive_details`). Test:
   `test_pre_tool_call_veto_through_hermes_resolver`.
4. **Skill key: `bayes-check:bayes-check`**, not `plugin:bayes-check`. `register_skill`
   qualifies it as `<plugin name>:<skill name>`. Test: `test_registrations_visible_to_hermes`.

No upstream issue is needed. Neither item failed; both corrections are handled inside the
plugin.

## Development

```bash
cd bayes-check
../.venv/bin/pytest tests -q            # 102 tests, no network, no LLM
hermes plugins doctor . --ci            # exits 0
```

`tests/pytest.ini` makes `tests/` the rootdir, so pytest never imports the plugin root as a
package. `tests/conftest.py` loads the plugin as package `bayes_check` and keeps the plugin
directory off `sys.path`, so its `tools.py` cannot shadow Hermes' `tools` package. The
integration tests skip themselves outside a Hermes environment.

See `DESIGN.md` for which decisions came verbatim from the spec, where the code disagreed with
the spec, and where new judgment was needed.
