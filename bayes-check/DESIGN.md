# bayes-check — DESIGN

Written before code, per GOAL.md. GOAL.md is the design contract; this file records the
implementation plan, which decisions are taken **verbatim** from the spec, which needed
**new judgment**, and every place where the **code disagreed with the spec** (code wins).

Source read before writing this: `hermes_cli/plugins.py` (PluginContext, pre_tool_call
directive parsing, command dispatch), `hermes_cli/plugins_content.py` (register_skill),
`hermes_cli/plugins_dispatch.py` (bounded hooks), `hermes_cli/plugin_dev.py` (Plugin Doctor),
`hermes_cli/plugins_manifest.py` (manifest fields / config_schema types),
`plugins/plugin_storage.py`, `model_tools.py` (post_tool_call emission),
`agent/inline_tool_executors.py`, `agent/tool_executor.py`, `agent/turn_context.py`
(pre_llm_call), `tools/delegate_tool*.py` (subagent hooks, background dispatch),
`cli.py` / `gateway/run_inbound.py` (plugin slash-command dispatch),
`gateway/session_context.py` (session env).

## 1. Plan

```
bayes-check/
  plugin.yaml      manifest_version 2, api_version 1, hooks/tools/config_schema
  __init__.py      register(ctx): tool, hooks, command, skill, system-prompt section
  schemas.py       JSON Schema for bayes_score (model-facing description lives here)
  scorer.py        pure math: log-odds update, Beta posteriors, verdicts, refit estimators
  state.py         SQLite store (plugin_db): claims ledger, outcomes, calibrated tables, rotation
  ledger.py        in-process module state: opt-in set, turn ledger, subagent ledger, local map
  local_mode.py    provider/base_url classifier + tri-state override
  extract.py       deterministic heuristic claim-candidate extraction (annotate fallback only)
  footer.py        footer rendering + bayes-checked hash
  tools.py         bayes_score handler (self-validating, never raises)
  hooks.py         every hook callback (all accept **kwargs)
  commands.py      /bayes on|off|status|correct|refit
  config.py        typed settings read through ctx.get_config with spec defaults
  skills/bayes-check/SKILL.md
  tests/           pytest, no LLM, no network
  README.md  DESIGN.md  PROGRESS.md
```

Data flow for an opted-in turn:

1. `pre_llm_call` → returns `{"context": <bounded mechanical instruction>}`.
2. Model calls `bayes_score` → handler validates, scores via `scorer`, writes the turn ledger
   (module dict keyed by `turn_id`) and appends to SQLite, `commit()` before return.
3. `post_tool_call` (`tool_name == "bayes_score"`, `status == "ok"`) → marks
   `(session_id, turn_id)` scored.
4. `transform_llm_output` → reads the ledger only (no I/O beyond an in-memory dict) and
   appends the footer. Never calls a model, never does oracle I/O.
5. `post_llm_call` → checks the persisted answer actually carries a bayes-check footer; if not
   (host skipped the transform on timeout), logs WARNING and records a `gate_skipped` event so
   `/bayes status` shows it.

## 2. Decisions taken verbatim from GOAL.md

- Plugin shape (tool + hooks + command + skill + system-prompt section), manifest_version 2,
  api_version 1, `provides_tools: [bayes_score]`.
- Zero `ctx.llm` calls; no multi-sample calibration; no background task that hits a model.
- Bounded hooks stay cheap; heavy work lives in the `bayes_score` handler.
- `local_mode` tri-state `auto | force_local | force_cloud`, auto-detected from `provider` +
  `base_url` on `pre_api_request`; localhost / 127.0.0.1 / 0.0.0.0 / ::1 / unix sockets;
  providers ollama, llama.cpp, lmstudio, vllm, custom_openai.
- `bayes_score` input `{claims:[{id,text,claim_type,prior,evidence:[{source,source_type,
  lr_override?}]}]}`, output `{results:[{id,posterior,verdict,action}],summary}`; verdicts
  `supported|uncertain|unsupported`; actions `state_plainly|qualify|remove`.
- Thresholds: `min_posterior_plain` 0.9 (≥ → supported / state_plainly),
  `min_posterior_hedge` 0.5 (< → unsupported / remove), between → uncertain / qualify.
- Transform behaviour: not opted in → `None`; scored → `bayes-checked:<hash>` footer with per-claim
  verdicts; not scored → `⚠ bayes-check: unscored` (annotate mode also lists heuristic claim
  candidates at their prior, i.e. `uncertain`).
- State tiers: module dicts (`_enabled_sessions`, `_turn_ledger`, `_subagent_ledger`,
  `_local_sessions`), `ctx.state` for small values, `plugin_db("bayes-check")` for posteriors and
  the append-only outcome ledger with ~50 MB self-rotation; always `commit()`.
- Commands `/bayes on|off|status|correct <claim_id> <true|false>|refit`; refit CPU-only.
- Skill is documentation; enforcement lives in the transform.
- System-prompt section (`max_chars=4000`, `position="after_memory"`) carries the standing rule;
  `pre_llm_call` carries the per-turn dynamic instruction.
- Out of scope: MCP backend, cross-profile state, oracle I/O, a core pre-finalize prose gate.

## 3. Code vs spec discrepancies (code wins)

| # | Spec said | Code says | What the plugin does |
|---|---|---|---|
| D1 | `/bayes on` adds "the current `session_id`" | `register_command` handlers receive only `raw_args: str` (`hermes_cli/plugins.py::register_command`, `cli.py::_run_plugin_slash_command`, `gateway/run_inbound.py`). The gateway binds `HERMES_SESSION_KEY` (not a session id — no entry exists yet) around the handler; the CLI has `HERMES_SESSION_ID` in the process env. | Commands resolve the session via `gateway.session_context.get_session_env` (`HERMES_SESSION_KEY`, then `HERMES_SESSION_ID`). The opt-in set holds those keys; hooks match on the payload `session_id` **or** the bound `HERMES_SESSION_KEY` and, on a key match, also add the concrete `session_id`. |
| D2 | Skill loads as `skill_view("plugin:bayes-check")` | `register_skill` qualifies as `<plugin_name>:<skill_name>` (`plugins_content.py`). | Key is `bayes-check:bayes-check`; used verbatim in the injected instruction and docs. |
| D3 | Block `delegate_task` with `mode="background"`, optionally all | `delegate_task` has no `mode` argument. Model-issued top-level delegations are **always** background (`tools/delegate_tool.py::_model_background_value`); only orchestrator children (depth > 0) run synchronously. | With `block_delegation_local: true` (default) every `delegate_task` call on an opted-in local session is blocked. `false` allows delegation; children inherit opt-in (see N6) so their own transform still gates their output. |
| D4 | `pre_tool_call` block shape unconfirmed | `_get_pre_tool_call_directive_details`: `{"action": "block", "message": str}`; a block **without** a non-empty message is ignored. | Always returns a non-empty message. |
| D5 | Subagent ledger keyed by `parent_turn_id` (from `subagent_stop`) | `subagent_stop` reads `parent_agent._current_turn_id` **when the child finishes** (`delegate_tool_results.py::_fire_subagent_stop_hooks`). For background children this is whatever turn the parent is in by then, not the dispatching turn. | The subagent ledger is keyed by `child_session_id` (unique); the dispatching `parent_turn_id` is taken from `subagent_start` and stored on the entry. |
| D6 | Documented `transform_llm_output` payload lacks `turn_id` | Call site passes `turn_id` (`agent/turn_finalizer.py::apply_llm_output_transform`). | Callbacks take `**kwargs` and read `turn_id`. |

## 4. New judgment (not covered by the spec)

- **N1 Location.** The plugin lives at repo-root `bayes-check/`, a standalone plugin directory,
  not under `plugins/` (in-tree policy in `plugins/AGENTS.md` keeps new feature plugins out of
  core). Install = copy or symlink into `$HERMES_HOME/plugins/bayes-check`. Doctor is run as
  `cd bayes-check && hermes plugins doctor . --ci`.
- **N2 Evidence direction.** The spec's evidence item has no direction, but a source that
  contradicts a claim must push the posterior down. Added optional `stance:
  supports|contradicts` (default `supports`); `contradicts` uses `1/LR`. Without it the scorer
  could only ever raise posteriors.
- **N3 Likelihood-ratio table.** Default LRs per `source_type` (supporting direction):
  `primary_source` 4.0, `peer_reviewed` 3.5, `official_docs` 3.0, `code_or_data` 3.0,
  `reputable_secondary` 2.0, `web_search` 1.5, `tool_output` 2.5, `user_provided` 1.5,
  `model_knowledge` 1.2, `anecdotal` 1.1, `none` 1.0; unknown types → 1.0 (no evidence) and are
  reported. `lr_override` wins but is clamped to [0.05, 20] so one item cannot dominate.
  Calibrated LRs from `/bayes refit` replace defaults per source type once enough outcomes exist.
- **N4 Correlated evidence.** Sources are often correlated (GOAL.md Q1). Repeated evidence of
  the same `source_type` on one claim is discounted geometrically: the k-th item contributes
  `log(LR) * 0.5**(k-1)`. Total |log-odds shift| per claim is capped at `log(100)`.
- **N5 Prior.** Claim `prior` clamped to [0.01, 0.99]; when omitted, the per-`claim_type` Beta
  posterior mean from the store (Beta(1,1) base) is used, else 0.5.
- **N6 Child opt-in inheritance.** `subagent_start` with an opted-in parent session opts the
  child session in, so the child's own `transform_llm_output` gates the summary the parent
  receives. The parent's footer lists children without a scored entry as
  "unverified subagent output".
- **N7 Loud failure.** Every hook body is wrapped; an internal error in the transform returns the
  original text plus `⚠ bayes-check: gate error (<type>)` — never a silent pass-through. The host
  timeout (30 s default) abandons a callback before it can write anything, which the plugin
  cannot catch from inside; it is detected after the fact by `post_llm_call` (footer missing on an
  opted-in turn → WARNING + `gate_skipped` row + visible in `/bayes status`). The transform also
  self-limits work (claim candidates capped, text scan capped) so it stays far below the bound.
  `post_llm_call` is therefore added to `provides_hooks` (not in the spec list).
- **N8 Annotate vs nudge without a scored call.** `annotate`: deterministic sentence heuristics
  (numbers, dates, percentages, superlatives, "is/was/causes" factual cues) pick ≤ 8 candidate
  claims, scored with empty evidence (→ prior → `uncertain`) and listed. `nudge`: marker only.
  `off`: plugin inert even for opted-in sessions.
- **N9 Footer hash.** `bayes-checked:<12 hex>` = sha256 of the canonical JSON of the turn's
  scored results; each claim line carries its 8-hex claim hash, which `/bayes correct` accepts
  (or the model's claim id, resolved against the session's most recent scored turn).
- **N10 Refit estimator.** Per source type, Beta-smoothed
  `LR = ((s_true+1)/(T+2)) / ((s_false+1)/(F+2))` where `s_true`/`s_false` count supporting items
  on claims later marked true/false and `T`/`F` the labelled totals; requires ≥ 5 labelled claims
  carrying that source type, clamped to [0.2, 10]. Per claim_type base rate Beta(1+true, 1+false).
  `/bayes refit` only — no periodic `spawn_task` (the CLI has no running loop; a manual command is
  sufficient and keeps the "no surprise work" posture).
- **N11 Research classifier.** `auto_enable_on_research` uses keyword heuristics on
  `user_message` (question forms + research/advisory cue words); on match the session is opted
  in and the turn's injected context says why. No model call.
- **N12 Ledger growth.** Turn ledger entries older than the 512 most recent turns are evicted
  (in-process dict, long gateway processes).
- **N13 Rotation.** When the DB file exceeds `max_db_bytes` (50 MB), the oldest unlabelled claim
  rows are deleted in chunks (labelled rows are the learning signal and are kept) and the DB is
  vacuumed.

## 5. Verify-in-dev plan

1. Plugin-tool `post_tool_call` with correct `turn_id`: runtime test loads the plugin through the
   real `PluginManager`, dispatches `bayes_score` through `model_tools.handle_function_call` and
   through `agent.inline_tool_executors.emit_terminal_post_tool_call` (the executor's terminal
   emitter), and asserts the hook received the `turn_id`.
2. `subagent_stop` for background children: code trace (background runner →
   `_execute_and_aggregate` → `_finalize_child_results` → `_fire_subagent_stop_hooks`) plus a
   runtime test invoking the shared finalizer with the plugin loaded.
3. `pre_tool_call` block shape → D4. 4. Skill key → D2.

Results go to README.md § Verify-in-dev results.
