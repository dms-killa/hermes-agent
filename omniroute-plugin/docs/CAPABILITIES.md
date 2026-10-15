# Capabilities (derived from code)

Every row cites `file:line` in one of two checkouts at the commits below. Nothing here comes from
documentation alone. Where a docs page is cited too, the code citation is the authority. Row ids
(`O-…` for OmniRoute, `H-…` for Hermes, `L1`–`L11` for lessons) are what the plugin's findings
cite, as `docs/CAPABILITIES.md#<id>`.

## Recorded commits

| Repo | Checkout | HEAD SHA | Commit date | Default branch |
|---|---|---|---|---|
| OmniRoute | `dms-killa/OmniRoute` (fork) | `fc5e2bccd4f70fecf5aab94dfb8136c74ab5a21b` | 2026-09-30T00:53:30-03:00 | `main` (HEAD == `origin/main`) |
| hermes-agent | `dms-killa/hermes-agent` (fork of `NousResearch/hermes-agent`) | `bcf29fa8da7011eb0706f3e7d85704655ff1cea6` | 2026-10-05T20:57:03-05:00 | `main` (HEAD == `origin/main`) |

Both checkouts are on branch `claude/intelligent-mendel-9sgsth`, which points at the same commit as
`main`. Both are shallow clones of forks. The hermes-agent fork carries five fork-only commits
above upstream `56f79864` (a `bayes-check/` plugin and an uploaded `GOAL.md`). None of them
touches a file cited below.

**v0.2 re-check (2026-10-15):** both upstream checkouts are still at the SHAs above, so every v0.1
row stands as recorded. The plugin repository was at `878646e` (v0.1) when v0.2 started. v0.2 adds
§1.11 (combo write surface), §2.4 (Hermes gating for plugin tools), §2.5 (correction of a v0.1
design note) and §4 (re-confirmation of the write-surface facts the v0.2 spec supplied).

Paths are relative to each repo root. `OR:` = OmniRoute, `HA:` = hermes-agent.

---

## 1. OmniRoute

### 1.1 Authentication for management routes

| Id | Capability | Evidence |
|---|---|---|
| <a id="o-auth-1"></a>O-AUTH-1 | Management routes call `requireManagementAuth`. The API-key branch reads the key from the `Authorization: Bearer` header only. URL-borne credentials are refused (`allowUrl: false`). | `OR:src/lib/api/requireManagementAuth.ts:49`, `:119`; `OR:src/sse/services/auth.ts:3571`, `:3577` |
| <a id="o-auth-2"></a>O-AUTH-2 | A valid key without `manage` or `admin` scope gets **403** "API key lacks 'manage' scope". A missing credential gets 401. | `OR:src/lib/api/requireManagementAuth.ts:154`; scopes `OR:src/shared/constants/managementScopes.ts:21` |
| <a id="o-auth-3"></a>O-AUTH-3 | All `/api/*` paths go through the central authz pipeline (`src/proxy.ts`), which classifies them as `MANAGEMENT`. Some handlers (`/api/provider-nodes`, `/api/usage/combo-health`) rely on that pipeline and do not call the guard themselves. | `OR:src/proxy.ts:24`; `OR:src/server/authz/classify.ts:124`; `OR:src/app/api/provider-nodes/route.ts:86`; `OR:src/app/api/usage/combo-health/route.ts:14` |
| <a id="o-auth-4"></a>O-AUTH-4 | `/api/mcp/*` is LOCAL_ONLY. A non-loopback caller passes only with a `manage`/`admin` key or the narrow `mcp:connect` scope. | `OR:src/server/authz/routeGuard.ts:33-34`, `:152`; `OR:src/shared/constants/managementScopes.ts:33` |

### 1.2 Connections and providers

| Id | Capability | Evidence |
|---|---|---|
| <a id="o-conn-1"></a>O-CONN-1 | `GET /api/providers` → `{connections, total}`. Optional `provider`, `limit`, `offset`. Tokens are removed. `apiKey` is masked unless the operator enabled key reveal, so clients must never echo it. `providerSpecificData` is sanitized (credentials and cookies deleted; `baseUrl` and `timeoutMs` kept). GET path has no writes. | `OR:src/app/api/providers/route.ts:115`, `:143`, `:147`, `:168`; `OR:src/lib/providers/requestDefaults.ts:327` |
| <a id="o-conn-2"></a>O-CONN-2 | `max_concurrent` is a column of `provider_connections` (`maxConcurrent` in JSON). It is a **connection** field. | `OR:src/lib/db/core.ts:257`; `OR:src/shared/validation/schemas/provider.ts:540`; `OR:src/lib/db/providers.ts:872` |
| <a id="o-conn-3"></a>O-CONN-3 | `GET /api/provider-nodes` → `{nodes, total, …}`. Nodes carry `id`, `type`, `prefix`, `baseUrl`. A compatible connection's provider id is its node id (`openai-compatible-chat-<uuid>`). The executor reads `providerSpecificData.baseUrl`, and a missing value is re-hydrated from the node at request time. | `OR:src/app/api/provider-nodes/route.ts:86-109`; `OR:open-sse/executors/default.ts:235`; `OR:open-sse/config/providerRegistry.ts:341`; `OR:src/sse/services/compatibleNodeBaseUrl.ts:34`; id example `OR:tests/unit/10085-compatible-generic-vs-uuid-credential.test.ts:31` |
| <a id="o-conn-4"></a>O-CONN-4 | Per-connection `providerSpecificData.timeoutMs` becomes a combo leg's timeout **ceiling** when the leg's connection is known. | `OR:open-sse/services/combo.ts:588`; `OR:open-sse/handlers/chatCore/upstreamTimeouts.ts:96` |

### 1.3 Combos

| Id | Capability | Evidence |
|---|---|---|
| <a id="o-combo-1"></a>O-COMBO-1 | `GET /api/combos` → `{combos, total}`, all combos when no `limit`. `GET /api/combos/{id}` → one combo. The record fields are `name`, `models` (ordered legs), `strategy` (default `priority`), `config`, `system_message`, `tool_filter_regex`, `isActive`. GET paths have no writes; cloud sync is on POST/PUT/DELETE only. | `OR:src/app/api/combos/route.ts:19`, `:41`; `OR:src/app/api/combos/[id]/route.ts:38`; `OR:src/shared/validation/schemas/combo.ts:370-375` |
| <a id="o-combo-2"></a>O-COMBO-2 | A leg (`ComboModelStep`) has `id`, `kind`, `model` (`"<provider>/<model>"`), `providerId`, an optional pinned `connectionId`, `allowedConnectionIds` and `weight`. Other step kinds are `combo-ref` and `provider-wildcard`. | `OR:src/lib/combos/steps.ts:12-51`; input schema `OR:src/shared/validation/schemas/combo.ts:31-32` |
| <a id="o-combo-3"></a>O-COMBO-3 | Combo agent middleware runs on every combo request. `system_message` **removes every system message** and injects its own. `tool_filter_regex` **drops every tool whose name does not match**. | `OR:open-sse/services/combo/comboSetup.ts:100`; `OR:open-sse/services/comboAgentMiddleware.ts:122-127`, `:135-155` |
| <a id="o-combo-4"></a>O-COMBO-4 | `targetTimeoutMs` lives in combo `config`, not on the leg. It bounds **one attempt on one leg** (default 120000 ms, `DEFAULT_COMBO_TARGET_TIMEOUT_MS`). For streaming it bounds only time-to-first-headers. A non-positive value is treated as unset. | `OR:src/shared/validation/schemas/combo.ts:178`; `OR:open-sse/services/comboConfig.ts:19-28`, `:241-270` |
| <a id="o-combo-5"></a>O-COMBO-5 | Effective combo config cascade: built-in defaults → `settings.comboDefaults` → combo `config`. Combo setup applies no provider override. `GET /api/settings/combo-defaults` returns `{comboDefaults, providerOverrides}` and has no writes. | `OR:open-sse/services/comboConfig.ts:294-322`, `:372-383`; `OR:open-sse/services/combo/comboSetup.ts:110-116`; `OR:src/app/api/settings/combo-defaults/route.ts:38` |
| <a id="o-combo-6"></a>O-COMBO-6 | The leg timeout is capped by the upstream ceiling: `FETCH_TIMEOUT_MS` (default 600000) or the leg connection's `timeoutMs`. If the cooldown-wait feature is enabled, the default floor rises to `budgetMs + 10000`. | `OR:open-sse/services/comboConfig.ts:69-81`, `:247-270`; `OR:src/shared/utils/runtimeTimeouts.ts:9` |

### 1.4 Routing strategies

| Id | Capability | Evidence |
|---|---|---|
| <a id="o-route-1"></a>O-ROUTE-1 | Public strategies: `priority, weighted, round-robin, context-relay, fill-first, p2c, random, least-used, cost-optimized, reset-aware, reset-window, headroom, quota-weighted, strict-random, auto, lkgp, context-optimized, cache-optimized, fusion, pipeline`. `quota-share` is internal-only and cannot be selected. | `OR:src/shared/constants/routingStrategies.ts:1-22`, `:32` |
| <a id="o-route-2"></a>O-ROUTE-2 | `p2c` draws two random legs and keeps the one with the better score. The score is `successRate/100 + 1/log10(avgLatencyMs+10)`, with a half-open breaker penalty and open breaker = −∞. **No in-flight count.** | `OR:open-sse/services/combo/applyStrategyOrdering.ts:150-152`; `OR:open-sse/services/combo/targetSorters.ts:126-156` |
| <a id="o-route-3"></a>O-ROUTE-3 | Live in-flight counting exists only in the quota-share engine: a per-connection lease counter, plus `maxConcurrent` gating at ordering time. | `OR:open-sse/services/combo/quotaShareInflight.ts:1-40`; `OR:open-sse/services/combo/applyStrategyOrdering.ts:271` |
| <a id="o-route-4"></a>O-ROUTE-4 | For every strategy, a leg pinned to a connection is **skipped before dispatch** when that connection's semaphore is at `maxConcurrent` (`decision: skipped_before_dispatch`, `reason: concurrency_cap`). The next leg is tried, so ordered strategies spill over. The semaphore key is `provider + connection`, which is shared by every combo and request that uses the connection. | `OR:open-sse/services/combo/executeTargetGates.ts:404-421`; `OR:open-sse/services/combo/concurrencyCaps.ts:27`; `OR:open-sse/services/accountSemaphore.ts:365-375` |
| <a id="o-route-5"></a>O-ROUTE-5 | Session stickiness pins a conversation to one connection, keyed by a hash of the first user message (opt out with `config.disableSessionStickiness`). Round-robin also has `stickyRoundRobinLimit`. Prompt-cache affinity reorders targets by prefix. | `OR:open-sse/services/combo/sessionStickiness.ts:1-30`; `OR:src/shared/validation/schemas/combo.ts:190`, `:196`; `OR:open-sse/services/combo/promptCacheAffinity.ts:25` |

### 1.5 Per-combo health, metrics and trace

| Id | Capability | Evidence |
|---|---|---|
| <a id="o-health-1"></a>O-HEALTH-1 | `GET /api/combos/metrics?combo=<name>` returns live in-memory metrics for one combo, keyed by **name**. Without the parameter it returns all of them. `DELETE` resets them (not used). | `OR:src/app/api/combos/metrics/route.ts:11-28`; `OR:open-sse/services/comboMetrics.ts:434-474` |
| <a id="o-health-2"></a>O-HEALTH-2 | `GET /api/usage/combo-health?range=1h\|24h\|7d\|30d[&comboId=<uuid>]` → `ComboHealthResponse` with per-target `requests`, `successRate`, `avgLatencyMs`, `connectionId`, quota and skew. `range` is required. No writes. | `OR:src/app/api/usage/combo-health/route.ts:6-36`; `OR:src/shared/types/utilization.ts:36-91` |
| <a id="o-health-3"></a>O-HEALTH-3 | `GET /api/usage/combo-trace/{id}` returns a decision trace by invocation id (`combo-…`, carried in the `X-OmniRoute-Combo-Trace` response header). Traces live in memory for 30 minutes. | `OR:src/app/api/usage/combo-trace/[id]/route.ts:12-24`; `OR:open-sse/services/combo.ts:629`; `OR:open-sse/services/combo/decisionTrace.ts:106`, `:314` |
| <a id="o-health-4"></a>O-HEALTH-4 | Other per-combo GET routes exist: `/api/usage/combo-health-dashboard`, `/combo-forecast`, `/combo-health-autopilot`, `/combo-scoring-inspector`. Latency per model: `/api/usage/model-latency-stats`. | `OR:src/app/api/usage/combo-health-dashboard/route.ts:15`; `OR:src/app/api/usage/model-latency-stats/route.ts:20`; directory listing of `OR:src/app/api/usage/` |
| <a id="o-health-5"></a>O-HEALTH-5 | Gateway health and circuit breakers: `GET /api/monitoring/health` (full view requires auth). `?deep=1` makes an **outbound** probe. `DELETE` resets breakers. | `OR:src/app/api/monitoring/health/route.ts:47`, `:122`, `:345` |

### 1.6 Call logs

| Id | Capability | Evidence |
|---|---|---|
| <a id="o-logs-1"></a>O-LOGS-1 | `GET /api/usage/call-logs` returns a JSON array of rows (`comboName`, `comboStepId`, `connectionId`, `status`, `duration`, `ttft`, `hasRequestBody`, …). Default limit 200. `excludeTests=1` keeps `/v1/*` and `/api/v1/*` paths only. Rows are written for proxied inference, not for `/api/*` management reads. | `OR:src/app/api/usage/call-logs/route.ts:228-277`; `OR:src/lib/usage/callLogs.ts:467-523`, `:1000-1007`, `:1034` |
| <a id="o-logs-2"></a>O-LOGS-2 | The `combo` filter is a **presence check**, not a name match, in both the SQL clause and the in-memory pass. | `OR:src/lib/usage/callLogs.ts:998`; `OR:src/app/api/usage/call-logs/route.ts:81-83` |
| <a id="o-logs-3"></a>O-LOGS-3 | `GET /api/usage/call-logs/{id}` returns the row plus the stored `requestBody` / `responseBody` when an artifact exists. **Side effect:** a missing artifact clears the pointer, and a corrupt one sets `detail_state='corrupt'`. The list's `requestSummary` covers search requests only. | `OR:src/app/api/usage/call-logs/[id]/route.ts:5-17`; `OR:src/lib/usage/callLogs.ts:1044-1086`; `OR:src/lib/usage/callLogs/format.ts:151` |
| <a id="o-logs-4"></a>O-LOGS-4 | OmniRoute's own synthetic probes send fixed bodies. Combo/model test: one user message `"Reply with exactly: pong"`, `max_tokens: 64`. Warmup: `hi`/`hello`/`ping`/`ready`, `max_tokens: 1`. Node validation: `"ping"`. Quota auto-ping: `"Reply with OK."`. | `OR:src/lib/combos/testHealth.ts:3-5`, `:114-135`; `OR:src/lib/api/modelTestRunner.ts:542`; `OR:src/lib/warmupScheduler.ts:30`, `:282`; `OR:src/app/api/provider-nodes/validate/route.ts:137`; `OR:src/shared/constants/quotaAutoPing.ts:54` |

### 1.7 The JSON-schema downgrade gate and the local-provider registry

| Id | Capability | Evidence |
|---|---|---|
| <a id="o-l1-1"></a>O-L1-1 | `applyJsonSchemaFallback` runs only for provider ids starting with `openai-compatible-` or equal to `kilocode`. It rewrites `response_format: json_schema` to `json_object` and appends the schema to the system prompt. It is applied on every request built by `DefaultExecutor`. | `OR:open-sse/executors/default.ts:722-726`, `:737`, `:770`, `:814` |
| <a id="o-l1-2"></a>O-L1-2 | Local-provider registry ids (not gated): `mlx-gemma, mlx-qwen, ollama-local, lm-studio, vllm, lemonade, llamafile, llama-cpp, triton, docker-model-runner, xinference, oobabooga` (+ image `sdwebui`, `comfyui`). | `OR:src/shared/constants/providers/local.ts:7-200` (e.g. `:35` ollama-local, `:105` llama-cpp) |

### 1.8 Error bodies

| Id | Capability | Evidence |
|---|---|---|
| <a id="o-err-1"></a>O-ERR-1 | When every leg of a combo fails, the client message is `"<model>: <kind> — <reason> (HTTP <status>)"` per leg, joined by `; `. `kind` is `model` for a 4xx that is not auth or rate-limit. A homogeneous model-class 4xx keeps that status (404). | `OR:open-sse/services/combo/comboErrorAggregation.ts:41`, `:111-133`, `:160-176`; `OR:open-sse/services/combo/comboAttemptLoop.ts:593-600`, `:698` |
| <a id="o-err-2"></a>O-ERR-2 | The body is built by `buildErrorBody`: `{error: {message, type, code}, diagnostics, …}`. For 404 the default `code` is **`model_not_found`**, type `invalid_request_error`. | `OR:open-sse/utils/error.ts:381-401`, `:586-605`; `OR:open-sse/config/errorConfig.ts:29` |
| <a id="o-err-3"></a>O-ERR-3 | A combo leg whose provider has no active credentials yields a 404 "No active credentials for provider: …" inside combos. Single-model requests remap it to 401/503. | `OR:src/sse/handlers/chatHelpers.ts:910-926` |

### 1.9 MCP server

| Id | Capability | Evidence |
|---|---|---|
| <a id="o-mcp-1"></a>O-MCP-1 | Transports: stdio (`omniroute --mcp`), Streamable HTTP at `/api/mcp/stream` and SSE at `/api/mcp/sse`, both inside the Next.js process. HTTP requires the dashboard setting `mcpEnabled` and `mcpTransport = streamable-http`. | `OR:bin/omniroute.mjs:339`; `OR:open-sse/mcp-server/httpTransport.ts:1-10`; `OR:src/app/api/mcp/stream/route.ts:15-33`, `:35-40` |
| <a id="o-mcp-2"></a>O-MCP-2 | The HTTP transport authenticates with `requireManagementAuth(…, {acceptMcpConnectScope: true})`: a Bearer key with `manage`, `admin` or `mcp:connect`. | `OR:src/app/api/mcp/stream/route.ts:35-37` |
| <a id="o-mcp-3"></a>O-MCP-3 | Environment for the stdio server and its internal fetches: `OMNIROUTE_BASE_URL` (fallbacks `BASE_URL`, `NEXT_PUBLIC_BASE_URL`, then `PORT`), `OMNIROUTE_API_KEY`, `OMNIROUTE_MCP_SCOPES`, `OMNIROUTE_MCP_ENFORCE_SCOPES`. The stdio server opens OmniRoute's own SQLite DB, so it runs on the gateway host. | `OR:open-sse/mcp-server/server.ts:93`, `:104-110`, `:130`, `:195-203`; `OR:src/shared/utils/resolveOmniRouteBaseUrl.ts:18-27` |
| <a id="o-mcp-4"></a>O-MCP-4 | Scopes: `read:health, read:combos, read:quota, read:usage, read:models, read:radar, read:cache, read:compression, read:proxies, write:*, execute:completions, execute:search, pricing:write`. **Enforcement ships off** (`OMNIROUTE_MCP_ENFORCE_SCOPES` default `false`), so every tool is callable by any authenticated caller unless the operator turns it on. Over HTTP, caller scopes are the API key's own scopes. | `OR:src/shared/constants/mcpScopes.ts:11-29`; `OR:src/shared/constants/featureFlagDefinitions.ts:483-490`; `OR:open-sse/mcp-server/scopeEnforcement.ts:72-133`; `OR:open-sse/mcp-server/httpAuthContext.ts:64` |
| <a id="o-mcp-5"></a>O-MCP-5 | Core catalog: 36 tools in `MCP_TOOLS` (counted). Further modules (memory, skills, pools, plugins, Notion, Obsidian, …) bring the README's stated total to 110. Read tools relevant here: `omniroute_get_health` (`/api/monitoring/health`, `/api/resilience`, `/api/rate-limits`), `omniroute_list_combos`, `omniroute_get_combo_metrics`, `omniroute_get_provider_metrics`, `omniroute_check_quota`, `omniroute_cost_report`, `omniroute_simulate_route`, `omniroute_explain_route` (`/api/routing/decisions/{id}`). | `OR:open-sse/mcp-server/schemas/tools.ts:103-1562`; `OR:open-sse/mcp-server/server.ts:283-292`, `:379-421`; `OR:open-sse/mcp-server/tools/advancedTools.ts:626-628`, `:762`; `OR:open-sse/mcp-server/README.md:3` |
| <a id="o-mcp-6"></a>O-MCP-6 | `omniroute_list_combos` is **lossy**: each leg is reduced to `{provider, model, priority}`. It drops `connectionId`, `allowedConnectionIds`, combo `config` (`targetTimeoutMs`), `system_message` and `tool_filter_regex`. No MCP tool reads `/api/providers` (connections, base URLs, `maxConcurrent`), `/api/provider-nodes`, `/api/usage/call-logs`, or `/api/usage/combo-health`. | `OR:open-sse/mcp-server/server.ts:175-190`, `:379-411`; full tool→endpoint list in the two files above |
| <a id="o-mcp-7"></a>O-MCP-7 | `omniroute_get_combo_metrics` sends `?comboId=`, but the route reads `?combo=` (a name), so it returns all combos' metrics. `list_combos(includeMetrics)` indexes the `{metrics: …}` envelope by combo id, so its metrics are always `null`. | `OR:open-sse/mcp-server/server.ts:408`, `:426`; `OR:src/app/api/combos/metrics/route.ts:17`, `:28` |
| <a id="o-mcp-8"></a>O-MCP-8 | No OmniRoute MCP tool declares a `readOnlyHint` annotation. | `grep -rn readOnlyHint OR:open-sse/mcp-server` (no matches outside tests) |

### 1.10 Skills directory

| Id | Capability | Evidence |
|---|---|---|
| <a id="o-skill-1"></a>O-SKILL-1 | `skills/` holds 46 generated `SKILL.md` directories (`omni-*`, `cli-*`, …) with `name`/`description` frontmatter. They document REST endpoints, write endpoints included. | `OR:skills/omni-combos-routing/SKILL.md:1-6`, `:17-46`; generator note `:6` (`src/lib/agentSkills/generator.ts`) |


### 1.11 Combo write surface (v0.2)

| Id | Capability | Evidence |
|---|---|---|
| <a id="o-write-1"></a>O-WRITE-1 | **Route.** `PUT /api/combos/{id}` updates one existing combo behind `requireManagementAuth`. `PATCH` calls the same handler. `GET /api/combos/{id}` returns the stored combo and has no writes. `{id}` is a dynamic segment beside static siblings (`auto`, `builder`, `duplicate`, `metrics`, `reorder`, `test`). | `OR:src/app/api/combos/[id]/route.ts:38-55`, `:57-61`, `:255-259`; directory `OR:src/app/api/combos/` |
| <a id="o-write-2"></a>O-WRITE-2 | **Request schema.** `updateComboSchema` (zod, not `.strict()`): top-level keys `name, description, displayName, models, strategy, config, isActive, isHidden, allowedProviders, allowedModelFamilies, overrideAllowedProviders, system_message, tool_filter_regex, context_cache_protection, context_length, compressionOverride, dimensions`. **Unknown keys are silently stripped.** At least one key is required ("No valid fields"). `models` needs ≥1 entry, each a string, a model step or a combo-ref (a stored `provider-wildcard` object matches neither and is rejected). `config` is `comboRuntimeConfigSchema`: `.passthrough()` keeps unknown config keys, numbers are coerced, `weights` gets defaults, and a `.transform` adds `zeroLatencyOptimizationsEnabled: true` when hedging, `predictiveTtftMs > 0` or a non-`off` `fallbackCompressionMode` is set without it. | `OR:src/shared/validation/schemas/combo.ts:424-486`; `:26-53`; `:170`, `:225`, `:300-319` |
| <a id="o-write-3"></a>O-WRITE-3 | **Merge and normalization.** Route: zod first (unknown keys stripped, config transform, O-WRITE-2); the legacy `compressionOverride` is rewritten into `config.compressionMode`; `config` goes through `stripDeadComboConfigKeys` (drops `pipelineConcurrency`, `resetAwareEnabled`, `resetAwareWindow`); `models` goes through `normalizeComboModels` (step ids, `provider/model` strings, combo-ref detection, weight clamp 0–100). Repository: `merged = {...current, ...body, sortOrder, updatedAt: now}` (a **shallow** merge, so `config` and `models` are replaced as a whole); then keys sent as `null` are deleted; then the whole record is re-normalized (`normalizeComboRecord`: `version: 2`, models again) before it is stored. The route's own `{...currentCombo, ...body}` (`:167-171`) is a validation copy, not the stored value. | `OR:src/app/api/combos/[id]/route.ts:106-140`, `:167-171`; `OR:src/lib/db/repositories/sqliteComboRepository.ts:220-269`; `OR:src/lib/combos/steps.ts:269-471`; `OR:src/lib/combos/deadConfigKeys.ts:8-28` |
| <a id="o-write-4"></a>O-WRITE-4 | **Resetting a field.** A top-level key sent as `null` is deleted from the stored combo, but only where the schema allows `null`: `description, displayName, allowedProviders, allowedModelFamilies, system_message, tool_filter_regex, context_cache_protection, context_length, dimensions`. `models, strategy, config, isActive, isHidden` (and `name`) cannot be reset: once present they can only be overwritten. A key **inside** `config` is reset by sending a `config` object without it (shallow merge). Consequences for rollback: adding a non-nullable key that was absent cannot be undone; dead config keys in a stored row are lost on any config write; a stored config with an un-gated zero-latency feature gains the gate on any config write; a stored provider-wildcard step cannot be re-sent. | `OR:src/shared/validation/schemas/combo.ts:424-455` (nullable comment at `:447-449`); `OR:src/lib/db/repositories/sqliteComboRepository.ts:241-246` |
| <a id="o-write-5"></a>O-WRITE-5 | **Server-only checks** (4xx, nothing written). Zod failure `COMBO_002`; quota-only combo-ref execute mode (`COMBO_002`); composite tiers `COMBO_003`; name collision `COMBO_004` (only when `name` is sent); nested-combo DAG over **all** combos (cycle or depth > `clampComboDepth(config.maxComboDepth)`, default 3, cap 10) `COMBO_005`; quota-share prefix `qtSd/` → **409** `COMBO_006`; unknown id `COMBO_007`; allowed providers/model-family invariant `COMBO_008`. Error body: `{error: {code, message, category, details}}`. | `OR:src/app/api/combos/[id]/route.ts:76-103`, `:172-234`, `:247-249`; `OR:open-sse/services/combo/comboStructure.ts:372-399`; `OR:open-sse/services/combo/comboPredicates.ts:115-119`, `:157-161`; `OR:src/lib/combos/invariants.ts:34-74`; `OR:src/lib/quota/quotaModelNaming.ts:12`; `OR:src/lib/api/comboErrorResponse.ts:7-49` |
| <a id="o-write-6"></a>O-WRITE-6 | **Response and side effects.** The response is the stored combo, plus an optional `warning` when the name collides with a real model id. Side effects of a successful update: session model pins for the combo are cleared when `models` was sent; the DB read cache is invalidated; a pre-write DB backup is taken; cloud sync runs if enabled (errors swallowed). | `OR:src/app/api/combos/[id]/route.ts:236-245`, `:296-308`; `OR:src/lib/db/combos.ts:47-61` |
| <a id="o-write-7"></a>O-WRITE-7 | **Concurrency.** No ETag, `If-Match` or version counter (`version: 2` is a schema constant). The repository rewrites `updatedAt` (ISO time) in the stored JSON and the `updated_at` column on **every** update, so any write changes the stored object. Writes are not conditional: a check-then-PUT client has a race window, and a concurrent write to other keys is preserved by the shallow merge. | `OR:src/lib/db/repositories/sqliteComboRepository.ts:235-240`, `:260-268`; `OR:src/lib/combos/steps.ts:459-471` |

---

## 2. Hermes Agent

### 2.1 Plugins

| Id | Capability | Evidence |
|---|---|---|
| <a id="h-plug-1"></a>H-PLUG-1 | Discovery sources: bundled `<repo>/plugins/`, user `$HERMES_HOME/plugins/<name>/`, project `./.hermes/plugins/` (needs `HERMES_ENABLE_PROJECT_PLUGINS`), and pip entry points. A directory plugin needs `plugin.yaml` plus `__init__.py` exposing `register(ctx)`. | `HA:hermes_cli/plugins.py:1-9`; `HA:hermes_cli/plugins_discovery.py:209-217` |
| <a id="h-plug-2"></a>H-PLUG-2 | Plugins are opt-in through `plugins.enabled` in `config.yaml`. `plugins.disabled` wins. CLI: `hermes plugins list / enable / disable / show / doctor / install`. | `HA:hermes_cli/plugins_discovery.py:112-131`; CLI observed: `hermes plugins --help` at this SHA |
| <a id="h-plug-3"></a>H-PLUG-3 | Manifest fields include `name, version, description, author, requires_env, provides_tools, provides_hooks, manifest_version, api_version, license, tags, config_schema`. `requires_env` only prompts at install time. | `HA:hermes_cli/plugins_manifest.py:342-386`; `HA:hermes_cli/plugins_cmd_install.py:165` |
| <a id="h-plug-4"></a>H-PLUG-4 | `ctx.register_tool(name, toolset, schema, handler, check_fn=None, requires_env=None, is_async=False, description="", emoji="", override=False)`. The handler is `handler(args: dict, **kwargs) -> str`; Hermes passes `task_id`, `session_id` and `user_task`, not `turn_id`. `check_fn` gates availability. | `HA:hermes_cli/plugins.py:456-499`; `HA:tools/registry.py:666-736`; handler kwargs per `HA:bayes-check/README.md` "Verify-in-dev" item 1 and `HA:bayes-check/tools.py:127` |
| <a id="h-plug-5"></a>H-PLUG-5 | Hooks: `ctx.register_hook(name, cb)` against `VALID_HOOKS` (tool, LLM, API-request, aux-call, session, subagent, gateway, approval, kanban, … ). Other registrations: `register_command`, `register_cli_command`, `register_skill`, `register_system_prompt_section`, `register_middleware`, `on_unload`. | `HA:hermes_cli/plugins.py:109-204`, `:633`, `:647`, `:933`, `:957`, `:409` |
| <a id="h-plug-6"></a>H-PLUG-6 | `transform_api_error_classification` runs once per failed API call **before** the built-in classifier. It returns `{"reason": <FailoverReason>, …}` or `None`; the first valid result wins. It is plugin-only (`SHELL_UNSUPPORTED_HOOKS`). | `HA:hermes_cli/plugins.py:126-132`, `:208`, `:2115-2170`; `HA:agent/error_classifier.py:702-717`; test `HA:tests/plugins/test_transform_api_error_classification_hook.py:66` |
| <a id="h-plug-7"></a>H-PLUG-7 | Storage: `ctx.state` / plugin data under `$HERMES_HOME/plugin-data/<plugin>/` (per profile). `ctx.get_config(key)` reads `plugins.entries.<id>.settings`. | `HA:hermes_cli/plugins.py:266`, `:283`; `HA:hermes_cli/plugins_state.py:143` |
| <a id="h-plug-8"></a>H-PLUG-8 | Plugins can bundle skills: `ctx.register_skill(name, path)` registers a read-only skill named `<plugin>:<name>`, which is not copied into `~/.hermes/skills/`. | `HA:hermes_cli/plugins_content.py:23-36` |
| <a id="h-plug-9"></a>H-PLUG-9 | Plugins can call configured MCP servers (`ctx.call_mcp`, behind an allowlist). Only *portable* Agent-Plugin packages can ship their own MCP server definitions. | `HA:hermes_cli/plugins.py:509`, `:1481-1508`; `HA:hermes_cli/agent_plugins.py:381` |

### 2.2 MCP servers for a profile

| Id | Capability | Evidence |
|---|---|---|
| <a id="h-mcp-1"></a>H-MCP-1 | `mcp_servers.<name>` in the profile's `config.yaml`. stdio: `command/args/env`. HTTP: `url`, `headers` (Streamable HTTP by default, `transport: sse` optional). `${VAR}` references resolve from the profile's `.env` or secret scope. | `HA:hermes_cli/config.py:1004`; `HA:tools/mcp_tool_transport.py:117`; `HA:tools/mcp_tool_server_run.py:283-292`; reference `HA:website/docs/reference/mcp-config-reference.md:18-50` |
| <a id="h-mcp-2"></a>H-MCP-2 | `tools.include` / `tools.exclude` select server-native tool names (exact or glob); include wins. | `HA:tools/mcp_tool_registration.py:216-228` |
| <a id="h-mcp-3"></a>H-MCP-3 | `trust: untrusted` requires approval for every tool without `readOnlyHint: true`. The default trust is `full`. | `HA:tools/mcp_tool.py:479-480`; `HA:tools/mcp_tool_handlers.py:52` |

### 2.3 Model settings (where a combo name sits)

| Id | Capability | Evidence |
|---|---|---|
| <a id="h-model-1"></a>H-MODEL-1 | Main model: `model` is a string, or a mapping with `default` (or `model`), `provider` and `base_url`. A named provider `providers.<name>.base_url` can carry the endpoint. With `base_url` set, the model string is sent verbatim, so a **combo name works as the model**. | `HA:hermes_cli/config_defaults.py:34-35`; `HA:cli-config.yaml.example:72-112`, `:178-215`; `HA:hermes_cli/runtime_provider.py:982`; `HA:hermes_cli/runtime_provider_custom.py:191` |
| <a id="h-model-2"></a>H-MODEL-2 | Auxiliary tasks: `auxiliary.<task>.{provider, model, base_url, api_key, timeout, extra_body}`. `provider: auto` with an empty model inherits the main model. A bare `base_url` means a custom endpoint. Tasks include `vision, compression, skills_hub, approval, review, mcp, title_generation, memory_query_rewrite, curator, …`. | `HA:hermes_cli/config_defaults.py:20-30`, `:738-800`; `HA:agent/auxiliary_task_config.py:16`; `HA:agent/auxiliary_client.py:6182-6250` |
| <a id="h-model-3"></a>H-MODEL-3 | Delegation and fallbacks: `delegation.{model, provider, base_url}` and top-level `fallback_providers` (+ legacy `fallback_model`) entries `{provider, model, base_url}`. | `HA:hermes_cli/config_defaults.py:36`, `:1331-1340`; `HA:hermes_cli/config.py:1003` |
| <a id="h-model-4"></a>H-MODEL-4 | Per-call timeouts: main and delegation use `providers.<id>.models.<model>.timeout_seconds`, then `providers.<id>.request_timeout_seconds`, then env `HERMES_API_TIMEOUT` (1800 s). Aux tasks use `auxiliary.<task>.timeout` (seconds, one request). | `HA:hermes_cli/timeouts.py:12-35`; `HA:agent/chat_completion_helpers.py:3026-3033`; `HA:hermes_cli/config_defaults.py:26` |
| <a id="h-model-5"></a>H-MODEL-5 | Structured output: `title_generation` sends `response_format: json_schema`, the only built-in aux task that does at this SHA (search: `task=` plus `response_format` in `agent/`, `tools/`, `hermes_cli/`, `gateway/`). | `HA:agent/title_generator.py:119-125`, `:515`, `:523` |
| <a id="h-model-6"></a>H-MODEL-6 | Tool-calling: the main agent loop, delegated subagents and `auxiliary.review` (a full subagent) send `tools` and depend on Hermes's system prompt. | `HA:hermes_cli/config_defaults.py:768-771`; `HA:model_tools.py` (dispatch); `HA:bayes-check/README.md` "Verify-in-dev" (agent-loop tool path) |
| <a id="h-model-7"></a>H-MODEL-7 | Secrets are read through the profile secret scope (`get_secret`), falling back to `os.environ` when unscoped. | `HA:agent/secret_scope.py:219-247`; `HA:agent/auxiliary_client.py:1114-1128` |


### 2.4 Native gating for plugin tools (v0.2)

| Id | Capability | Evidence |
|---|---|---|
| <a id="h-gate-1"></a>H-GATE-1 | **(a) Own toolset: yes.** A plugin chooses the `toolset` of each tool it registers; Hermes groups plugin tools by toolset and lists each one under "Plugin toolsets" in `hermes tools list`. The operator enables or disables a plugin toolset per platform (`hermes tools enable/disable`, `platform_toolsets.<platform>`). **Gap: a plugin toolset is ON by default.** Default-off is a hard-coded core set (`_DEFAULT_OFF_TOOLSETS`) with no plugin API, so a plugin cannot ship its write toolset disabled. | `HA:hermes_cli/plugins.py:456-499`, `:2237-2264`; `HA:hermes_cli/tools_config.py:110`, `:526-533`, `:598-613`; `HA:hermes_cli/tools_config_mcp.py:185-206` |
| <a id="h-gate-2"></a>H-GATE-2 | **(b) Approval for plugin tools: yes, through `pre_tool_call`.** A `pre_tool_call` callback returning `{"action": "approve", "message", "rule_key"}` sends the call to the same human gate as dangerous shell commands (`request_tool_approval`). It offers once/session/always/deny, honours `approvals.mode: off` and yolo, and **fails closed** when no interactive user or gateway is present. A `block` from any plugin outranks an `approve`. The tool dispatcher runs it on every call. No smart (LLM guardian) step is applied to plugin approvals. **There is no registration-time flag** (`register_tool` has no read-only or approval parameter). | `HA:hermes_cli/plugins.py:1971-2020`, `:2051-2080`; `HA:tools/approval.py:1104-1127`, `:951-1021`; `HA:model_tools.py:768-790`; `HA:hermes_cli/plugins.py:456-460` |
| <a id="h-gate-3"></a>H-GATE-3 | **(c) `capabilities:` does not cover this.** It is consent for a closed set of privileged host surfaces (`tools.override`, five `llm.*` overrides, `gateway.platform_actions`). Unknown ids are dropped with a warning. Nothing covers "tool makes outbound writes". | `HA:hermes_cli/plugin_capabilities.py:28-51`, `:58-88` |
| <a id="h-gate-4"></a>H-GATE-4 | **MCP `trust: untrusted` shim (not built).** On an untrusted MCP server, every tool without `readOnlyHint: true` asks `request_elicitation_consent` before the RPC: real per-call approval, fail-closed. Compared with H-GATE-2 it is a different prompt path, with no session/always allowlist and no yolo/`approvals.mode` handling. A shim would therefore give real approval, but nothing H-GATE-2 does not already give, at the cost of a separate MCP server. | `HA:tools/mcp_tool_handlers.py:59-84`; `HA:tools/approval_prompt.py:288-303`; `HA:tools/mcp_tool.py:478-489` |
| <a id="h-gate-5"></a>H-GATE-5 | **Approval in practice.** The CLI registers its prompt in a per-thread callback slot (`terminal_tool.set_approval_callback`) under `HERMES_INTERACTIVE`; without a callback the prompt falls back to `input()`. Gateways route it to the chat platform. | `HA:tools/terminal_tool.py:102-104`; `HA:tools/approval_context.py:48-51`, `:185-193`; `HA:tools/approval_prompt.py:20-58` |

### 2.5 Correction to v0.1 (bundled skills and MCP)

| Id | Capability | Evidence |
|---|---|---|
| <a id="h-skill-1"></a>H-SKILL-1 | v0.1 DESIGN.md's deferred-decision note grouped bundled MCP and skills under "portable packages". From code: **native plugins bundle skills** with `ctx.register_skill(name, path)` (documented as "Bundle skills"; v0.2 does so). **Shipping MCP server definitions** from a package (`mcp.json`) is wired only for portable Agent Plugin manifests (`manifest.portable`; `load_agent_plugin` → `_discover_mcp`). A native plugin can *call* configured MCP servers (`ctx.call_mcp`). `capabilities:` is the consent list for privileged host surfaces (H-GATE-3), unrelated to skills or MCP. H-PLUG-8 and H-PLUG-9 were correct as written. | `HA:hermes_cli/plugins_content.py:23-36`; `HA:website/docs/developer-guide/plugins/index.md:833-873`, `:254-262`, `:48-95`; `HA:hermes_cli/plugins_loader.py:467`, `:630-631`; `HA:hermes_cli/plugins.py:1481-1508`; `HA:hermes_cli/agent_plugins.py:381`, `:422-428` |

---

## 3. Lessons L1–L11: verdicts at the recorded SHAs

CONFIRMED = holds at HEAD (line numbers may have moved). CHANGED = the claim no longer holds as
stated. GONE = the mechanism no longer exists.

| Lesson | Verdict | What the code says now | Evidence |
|---|---|---|---|
| <a id="l1"></a>**L1** JSON-schema downgrade for `openai-compatible-*` / `kilocode`; local ids not gated | **CONFIRMED** | Same gate, same rewrite. Now at `default.ts:722-770` (was ~722-770). `json_object` passes through for compatible providers; `kilocode` strips `response_format` entirely. `llama-cpp` and `ollama-local` do not match the gate. | O-L1-1, O-L1-2 |
| <a id="l2"></a>**L2** `targetTimeoutMs` (default 120000) covers one request to one leg; compare at matching scope | **CONFIRMED** (refined) | Default 120000 per leg attempt, in combo `config` (not per leg). Cascade: defaults → `settings.comboDefaults` → combo. Capped by `FETCH_TIMEOUT_MS` (600000) or the leg connection's `timeoutMs`. For streaming it bounds only time-to-first-headers. `0` counts as unset. | O-COMBO-4, O-COMBO-5, O-COMBO-6, O-CONN-4 |
| <a id="l3"></a>**L3** p2c ranks by history, not in-flight; in-flight only in quota-share; priority + pinned + max_concurrent ≈ ordered spill-over | **CONFIRMED** | p2c score = success rate + latency (and breaker). The in-flight lease counter is quota-share only, and quota-share is internal-only. The pre-dispatch `concurrency_cap` skip applies to pinned legs under every strategy. | O-ROUTE-1, O-ROUTE-2, O-ROUTE-3, O-ROUTE-4 |
| <a id="l4"></a>**L4** `max_concurrent` belongs to a connection, across all combos | **CONFIRMED** | Column on `provider_connections`. The semaphore key is `provider + connectionId`. There is no combo-level field. | O-CONN-2, O-ROUTE-4 |
| <a id="l5"></a>**L5** Call logs = /v1 traffic; combo filter is presence-only; no `/api/combos/metrics`; per-combo data under `/api/usage/combo-*` | **CHANGED** | Logs and the presence-only filter are confirmed (`route.ts:81-83`, `callLogs.ts:998`). **`GET /api/combos/metrics` now exists** (in-memory, keyed by name, `?combo=<name>`). Per-combo routes: `combo-health`, `combo-trace/[id]`, `combo-health-dashboard`, `combo-forecast`, `combo-health-autopilot`, `combo-scoring-inspector`. | O-LOGS-1, O-LOGS-2, O-HEALTH-1..4 |
| <a id="l6"></a>**L6** Probe traffic pollutes logs/metrics; separate by body signature | **CONFIRMED** | Combo tests go through the normal path. Nothing marks them as probes in the log row (no flag field; `excludeTests` filters by path only). Bodies are retrievable through the detail route when stored. | O-LOGS-1, O-LOGS-3, O-LOGS-4 |
| <a id="l7"></a>**L7** Prove what body reached the backend before tuning | **CONFIRMED** (method) | The tools for it exist: the stored `requestBody` (client side) and `pipelinePayloads` (provider side) are on the call-log detail route, and the L1 rewrite is the documented example of a body change. | O-LOGS-3, O-L1-1 |
| <a id="l8"></a>**L8** `/api/*` needs a `manage`-scope key, in a header only; a /v1 key gets 403 | **CONFIRMED** | `allowUrl: false`, Bearer header, `manage`/`admin` scopes, 403 message. Now at `requireManagementAuth.ts:119-156` (was ~117-156). | O-AUTH-1, O-AUTH-2 |
| <a id="l9"></a>**L9** MCP server (~110 tools, read/write/execute scopes), skills, `OMNIROUTE_BASE_URL`/`OMNIROUTE_API_KEY` | **CONFIRMED** (refined) | As stated, with these additions: scope enforcement is off by default; HTTP transport is opt-in in settings; the stdio server must run on the gateway host; `list_combos` is lossy and `get_combo_metrics` sends the wrong parameter. | O-MCP-1..8, O-SKILL-1 |
| <a id="l10"></a>**L10** Hermes misses OmniRoute's 404 `"…: model — Not Found (HTTP 404)"` → retries instead of falling back | **CHANGED** | At these SHAs Hermes classifies the exact OmniRoute body as **`model_not_found`, `retryable=False`, `should_fallback=True`**. The message text still matches no pattern, but OmniRoute's body carries `code: "model_not_found"` (404 default), and Hermes's lowered message includes `str(error)` with that code. Without the `code` the verdict would be `unknown`. Executed and pinned by `tests/test_l10_classifier.py`. The hook seam is confirmed to exist. | O-ERR-1, O-ERR-2; `HA:agent/error_classifier.py:273-279`, `:604-607`, `:1035-1045`, `:1340-1354`; H-PLUG-6 |
| <a id="l11"></a>**L11** Round-robin stickiness and prompt-cache affinity can pin to one leg; compression changes prompt bytes | **CONFIRMED** (refined) | Session stickiness (first-message hash, opt-out `disableSessionStickiness`), `stickyRoundRobinLimit`, and prompt-cache affinity all exist. Compression is **off by default** (`DEFAULT_COMPRESSION_CONFIG.enabled=false`, `defaultMode "off"`) and opt-in per combo through `config.compressionMode`. The effect on llama.cpp prefix-cache reuse follows from the byte change and is not otherwise testable from code. | O-ROUTE-5; `OR:src/shared/validation/schemas/combo.ts:212`; `OR:open-sse/services/compression/types.ts:421-423` |

---

## 4. Write-surface facts supplied by the v0.2 spec, re-confirmed at HEAD

| Claim | Verdict | Evidence at the recorded SHA |
|---|---|---|
| `PUT /api/combos/[id]`, `PATCH` an alias, behind `requireManagementAuth` (~57-61, ~255-259) | **CONFIRMED** | `OR:src/app/api/combos/[id]/route.ts:57-61`, `:255-259` |
| PUT merges: stored combo becomes `{...current, ...body}` (~167-171); omitted fields keep their value | **CONFIRMED (refined)** | `:167-171` is the validation copy. The stored merge is in `OR:src/lib/db/repositories/sqliteComboRepository.ts:235-246`, which also deletes keys sent as `null` (O-WRITE-3, O-WRITE-4) |
| Server normalizes: `normalizeComboModels` on `models`, `stripDeadComboConfigKeys` on `config` (~124-140) | **CONFIRMED (refined)** | `route.ts:124-140`; also legacy `compressionOverride` rewrite `:106-123`, whole-record `normalizeComboRecord` in the repository (`sqliteComboRepository.ts:252`), and zod coercion, defaults and the zero-latency `.transform` (`schemas/combo.ts:300-319`, O-WRITE-2) |
| Name collision via DB (~197-208) | **CONFIRMED** | `route.ts:197-208` (only when `name` is in the body) |
| `validateComboDAG` over all combos (~210-233) | **CONFIRMED** | `route.ts:210-234`; `OR:open-sse/services/combo/comboStructure.ts:372-399` |
| 409 for the quota-share prefix (~96-103) | **CONFIRMED** | `route.ts:96-103`; prefix `qtSd/` at `OR:src/lib/quota/quotaModelNaming.ts:12` |
| Response returns the stored combo (~236-245) | **CONFIRMED (refined)** | `route.ts:236-245`; may add a `warning` field |
| `syncToCloudIfEnabled` after every successful update | **CONFIRMED** | `route.ts:239`, `:299-308` (errors logged, not returned) |
| No updatedAt, ETag or version counter; `version: 2` is a constant | **CHANGED** | **`updatedAt` exists:** the repository stamps it on every update (`sqliteComboRepository.ts:239`, column at `:262-267`). No ETag or version counter (O-WRITE-7). A content hash still works as the freshness token, and `updatedAt` makes every write change that hash |
| `targetTimeoutMs: 0` treated as unset (comboConfig.ts ~241-270) | **CONFIRMED** (SHA unchanged since v0.1) | `OR:open-sse/services/comboConfig.ts:241-270` |
