"""Regenerate the JSON fixtures. Shapes mirror OmniRoute source at the SHA in
docs/CAPABILITIES.md (see PROVENANCE.md for file:line per fixture). All values are synthetic:
ids are placeholder UUIDs, hosts are 127.0.0.1 / a 10.x placeholder / *.example.lan /
api.example.com, and no key is real.

Run:  python tests/fixtures/generate_fixtures.py
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
NA = "openai-compatible-chat-00000000-0000-4000-8000-0000000000a1"
NB = "openai-compatible-chat-00000000-0000-4000-8000-0000000000b2"
NC = "openai-compatible-chat-00000000-0000-4000-8000-0000000000c3"
ND = "openai-compatible-chat-00000000-0000-4000-8000-0000000000d4"
TS = "2026-01-01T00:00:00.000Z"
LOOPBACK_URL = "http://127.0.0.1:8080/v1"          # placeholder: llama-cpp default in local.ts
LAN_URL = "http://gpu-node.example.lan:11434/v1"   # placeholder .lan name
PRIVATE_URL = "http://10.20.30.40:8000/v1"         # placeholder RFC 1918 address
CLOUD_URL = "https://api.example.com/v1"           # RFC 2606 reserved name


def w(name, obj):
    (HERE / name).write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def node(node_id, prefix, name, base):
    return {"id": node_id, "type": "openai-compatible-chat", "name": name, "prefix": prefix,
            "apiType": "chat", "baseUrl": base, "chatPath": None, "modelsPath": None,
            "createdAt": TS, "updatedAt": TS}


def conn(cid, provider, name, max_c, psd=None, active=True):
    return {"id": cid, "provider": provider, "authType": "apikey", "name": name, "displayName": None,
            "email": None, "priority": 1, "isActive": active, "testStatus": "active",
            "apiKey": "PLACEHOLDER-MASKED-KEY-****", "accessToken": None, "refreshToken": None,
            "idToken": None, "maxConcurrent": max_c, "providerSpecificData": psd,
            "createdAt": TS, "updatedAt": TS}


def step(sid, model, provider=None, connection=None):
    s = {"id": sid, "kind": "model", "model": model, "weight": 0}
    if provider:
        s["providerId"] = provider
    if connection:
        s["connectionId"] = connection
    return s


def row(log_id, combo, status=200, has_body=True, model="loopnode/small-model"):
    return {"id": log_id, "timestamp": "2026-01-01T00:00:00Z", "method": "POST",
            "path": "/v1/chat/completions", "status": status, "model": "small-model",
            "requestedModel": model, "provider": NA, "providerDisplay": "Loopback node",
            "account": "loop-small", "connectionId": "conn-loop", "duration": 850, "ttft": 120,
            "tokens": {"in": 12, "out": 3, "cacheRead": None, "cacheWrite": None,
                       "reasoning": None, "compressed": None},
            "cacheSource": "upstream", "requestType": "chat", "sourceFormat": "openai",
            "targetFormat": "openai", "apiKeyId": "key-1", "apiKeyName": "hermes",
            "comboName": combo, "comboStepId": "s1", "comboExecutionKey": "s1", "error": None,
            "detailState": "ready" if has_body else "none", "requestSummary": None,
            "hasRequestBody": has_body, "hasResponseBody": has_body, "hasPipelineDetails": False,
            "correlationId": None, "active": False, "completed": True}


def main():
    w("provider_nodes.json", {"nodes": [
        node(NA, "loopnode", "Loopback node", LOOPBACK_URL),
        node(NB, "lannode", "LAN node", LAN_URL),
        node(NC, "privnode", "Private-range node", PRIVATE_URL),
        node(ND, "cloudnode", "Hosted compatible API", CLOUD_URL),
    ], "total": 4, "ccCompatibleProviderEnabled": False})
    w("connections.json", {"connections": [
        conn("conn-loop", NA, "loop-small", 1),            # base URL comes from the node
        conn("conn-lan", NB, "lan-medium", 2, {"baseUrl": LAN_URL}),
        conn("conn-priv", NC, "private-large", None, {"baseUrl": PRIVATE_URL, "timeoutMs": 900000}),
        conn("conn-cloud-compat", ND, "hosted-compat", None, {"baseUrl": CLOUD_URL}),
        conn("conn-llamacpp", "llama-cpp", "llamacpp-native", 1, {"baseUrl": "http://127.0.0.1:8081/v1"}),
        conn("conn-openai", "openai", "openai-key", None, None),
    ], "total": 6})
    w("combos.json", {"combos": [
        {"id": "00000000-0000-4000-8000-00000000c001", "name": "structured-local", "strategy": "priority",
         "models": [step("s1", "loopnode/small-model", "loopnode", "conn-loop"),
                    step("s2", "llama-cpp/small-model", "llama-cpp", "conn-llamacpp")],
         "config": {}, "version": 2, "computed_context_length": 32768},
        {"id": "00000000-0000-4000-8000-00000000c002", "name": "fanout-p2c", "strategy": "p2c",
         "models": [step("p1", "lannode/medium-model", "lannode"),
                    step("p2", "privnode/large-model", "privnode")],
         "config": {"targetTimeoutMs": 300000}, "version": 2},
        {"id": "00000000-0000-4000-8000-00000000c003", "name": "spill", "strategy": "priority",
         "models": [step("q1", "loopnode/small-model", "loopnode", "conn-loop"),
                    step("q2", "lannode/medium-model", "lannode", "conn-lan")],
         "config": {"targetTimeoutMs": 600000}, "version": 2},
        {"id": "00000000-0000-4000-8000-00000000c004", "name": "clean-cloud", "strategy": "priority",
         "models": [step("r1", "openai/example-model", "openai")],
         "config": {"targetTimeoutMs": 900000}, "version": 2},
        {"id": "00000000-0000-4000-8000-00000000c005", "name": "agent-main", "strategy": "priority",
         "models": [step("t1", "openai/example-model", "openai")],
         "config": {"targetTimeoutMs": 900000}, "version": 2,
         "system_message": "You are a terse assistant.", "tool_filter_regex": "^web_"},
    ], "total": 5})
    w("combo_defaults.json", {"comboDefaults": {
        "strategy": "priority", "maxRetries": 1, "retryDelayMs": 2000, "fallbackDelayMs": 0,
        "handoffThreshold": 0.85, "handoffModel": "", "maxMessagesForSummary": 30,
        "maxComboDepth": 3, "trackMetrics": True, "reasoningTokenBufferEnabled": True,
        "zeroLatencyOptimizationsEnabled": False}, "providerOverrides": {}})
    w("call_logs.json", [row("log-0001", "structured-local"), row("log-0002", "structured-local"),
                         row("log-0003", "spill"), row("log-0004", None, has_body=False),
                         row("log-0005", "structured-local", has_body=False)])
    w("call_log_detail_log-0001.json", dict(row("log-0001", "structured-local"), requestBody={
        "model": "structured-local", "max_tokens": 64, "stream": False,
        "messages": [{"role": "user", "content": "Reply with exactly: pong"}]},
        responseBody={"choices": [{"message": {"role": "assistant", "content": "pong"}}]}))
    w("call_log_detail_log-0002.json", dict(row("log-0002", "structured-local"), requestBody={
        "model": "structured-local", "max_tokens": 200, "stream": False,
        "messages": [{"role": "system", "content": "Generate a title."},
                     {"role": "user", "content": "Synthetic conversation text for the fixture."}],
        "response_format": {"type": "json_schema",
                            "json_schema": {"name": "session_title", "schema": {"type": "object"}}}},
        responseBody=None))
    w("call_log_detail_log-0003.json", dict(row("log-0003", "spill"), requestBody={
        "model": "spill", "max_tokens": 4, "messages": [{"role": "user", "content": "Say ok"}]},
        responseBody=None))
    w("combo_health.json", {"timeRange": "24h", "combos": [{
        "comboId": "00000000-0000-4000-8000-00000000c001", "comboName": "structured-local",
        "strategy": "priority", "models": ["loopnode/small-model", "llama-cpp/small-model"],
        "targetHealth": [{"executionKey": "s1", "stepId": "s1", "model": "loopnode/small-model",
                          "provider": NA, "connectionId": "conn-loop", "label": None, "requests": 40,
                          "successRate": 97.5, "avgLatencyMs": 900, "lastStatus": "ok",
                          "lastUsedAt": TS, "quotaRemainingPct": None, "quotaIsExhausted": None,
                          "quotaTrend": None, "quotaScope": "none"}],
        "quotaHealth": {"providers": [], "worstRemainingPct": None},
        "usageSkew": {"modelDistribution": [{"model": "loopnode/small-model", "requestShare": 1,
                                             "tokenShare": 1}], "giniCoefficient": 0},
        "performance": {"avgLatencyMs": 900, "successRate": 97.5, "totalRequests": 40}}]})
    w("combo_metrics.json", {"metrics": {
        "strategy": "priority", "totalRequests": 40, "totalSuccesses": 39, "totalFallbacks": 1,
        "totalLatencyMs": 36000, "productionTraffic": True, "persistedSkipBypassed": False,
        "avgLatencyMs": 900, "successRate": 98, "fallbackRate": 3, "intentCounts": {},
        "byModel": {"loopnode/small-model": {"requests": 40, "successRate": 98, "avgLatencyMs": 900}},
        "byTarget": {}}})
    # Terminal combo 404 when every target failed with a model-class 404:
    # comboAttemptLoop.ts -> formatComboOutcomes -> errorResponseWithComboDiagnostics -> buildErrorBody
    w("omniroute_404_combo_exhausted.json", {
        "error": {"message": "loopnode/small-model: model — Not Found (HTTP 404)",
                  "type": "invalid_request_error", "code": "model_not_found"},
        "diagnostics": {"poolSize": 1, "attempted": 1, "excluded": [],
                        "attemptOrder": [{"provider": NA, "model": "small-model"}],
                        "terminalReason": "Not Found"}})


if __name__ == "__main__":
    main()
