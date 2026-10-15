# Fixture provenance

All fixtures are produced by `generate_fixtures.py` and contain synthetic values only.

- Ids are placeholder UUIDs (`00000000-0000-4000-8000-…`).
- Hosts are `127.0.0.1` (OmniRoute's own llama-cpp default), one RFC 1918 placeholder
  (`10.20.30.40`), `gpu-node.example.lan` and `api.example.com` (RFC 2606 reserved).
- No key is real. The mock's key is `test-placeholder-manage-key`.

Shapes follow OmniRoute source at `fc5e2bccd4f70fecf5aab94dfb8136c74ab5a21b`:

| Fixture | Endpoint | Shape source |
|---|---|---|
| `connections.json` | `GET /api/providers` | `src/app/api/providers/route.ts:141-168` (`{connections, total}`, masked `apiKey`, tokens removed); row fields `src/lib/db/providers.ts:855-878` |
| `provider_nodes.json` | `GET /api/provider-nodes` | `src/app/api/provider-nodes/route.ts:103-109`; node fields `src/lib/db/providers/nodes.ts:96-110`; compatible id form `tests/unit/10085-compatible-generic-vs-uuid-credential.test.ts:31` |
| `combos.json` | `GET /api/combos` | `src/app/api/combos/route.ts:35-41`; step shape `src/lib/combos/steps.ts:12-38`; combo fields `src/shared/validation/schemas/combo.ts:366-375` |
| `combo_defaults.json` | `GET /api/settings/combo-defaults` | `src/app/api/settings/combo-defaults/route.ts:38-63` (the built-in default object returned when nothing is stored) |
| `call_logs.json` | `GET /api/usage/call-logs` | row mapping `src/lib/usage/callLogs.ts:467-523`; route returns a bare array `src/app/api/usage/call-logs/route.ts:276-277` |
| `call_log_detail_*.json` | `GET /api/usage/call-logs/{id}` | `src/lib/usage/callLogs.ts:1060-1076` (`requestBody` / `responseBody` added). `log-0001` uses OmniRoute's combo test body, `src/lib/combos/testHealth.ts:3-5`, `:114-135` |
| `combo_health.json` | `GET /api/usage/combo-health` | `src/shared/types/utilization.ts:36-91` |
| `combo_metrics.json` | `GET /api/combos/metrics?combo=` | `src/app/api/combos/metrics/route.ts:19-24`; view fields `open-sse/services/comboMetrics.ts:434-465` |
| `omniroute_404_combo_exhausted.json` | terminal combo error | message `open-sse/services/combo/comboErrorAggregation.ts:111-133`; body `open-sse/utils/error.ts:381-401`, `:586-605`; `code` for 404 `open-sse/config/errorConfig.ts:29`; diagnostics `open-sse/utils/error.ts:464-479` |

The mock (`tests/mock_server.py`) reproduces one behaviour on top of the static files: the `combo`
query on call logs is a presence filter, as in `src/lib/usage/callLogs.ts:997-998`.
