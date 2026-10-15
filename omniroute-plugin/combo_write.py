"""omniroute_combo_plan / omniroute_combo_apply / omniroute_combo_rollback.

The tools enforce mechanical invariants only:
- one existing combo per apply, through the single allowlisted write route (O-WRITE-1);
- only keys the server's update schema accepts and this plugin's scope allows (no rename);
- plan freshness: a content hash of the full stored combo, and an expiry;
- reversibility: a plan whose prior state rollback could not restore through the merge route is
  refused (O-WRITE-4);
- read-back verification against the expected stored result, with automatic restore.

Judgment is not encoded here. Findings (L1-L4, unpinned legs, Hermes task effects) are returned as
information; whether to apply is the agent's call, and whether the call may run is Hermes's
(toolset enablement and the approval gate, H-GATE-*).
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Dict, List, Optional, Tuple

from . import audit, combo_check
from .client import OmniRouteClient, OmniRouteError
from .findings import finding
from .normalize import (CONFIG_SUBSET_COMPARED, DEAD_COMBO_CONFIG_KEYS, NULLABLE_KEYS, OUT_OF_SCOPE_KEYS,
                        QUOTA_MODEL_PREFIX, SERVER_MANAGED_KEYS, WRITABLE_KEYS,
                        clamp_combo_depth, content_hash, diff_paths, needs_zero_latency_promotion,
                        normalize_combo_models, normalize_combo_record, server_config)
from .write_store import PLAN_TTL_SECONDS, WriteStore

ConfigLoader = Callable[[], Dict[str, Any]]


class PlanRefused(ValueError):
    """A change outside the mechanical invariants (scope, schema, reversibility)."""


def _rec(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _combo_list(raw: Any) -> List[Dict[str, Any]]:
    items = raw.get("combos") if isinstance(raw, dict) else raw
    return [c for c in (items or []) if isinstance(c, dict)]


def _names(combos: List[Dict[str, Any]], extra: str = "") -> List[str]:
    names = [c["name"].strip() for c in combos if isinstance(c.get("name"), str) and c["name"].strip()]
    return names + ([extra] if extra else [])


# ── body and expected result ─────────────────────────────────────────────────────────────────────

def build_put_body(current: Dict[str, Any], changes: Dict[str, Any]) -> Dict[str, Any]:
    """Translate the agent's ``changes`` into the PUT body. ``changes.config`` is a key-by-key patch
    (a null value removes the key); because the server replaces ``config`` as a whole (O-WRITE-3),
    the body carries the complete resulting config."""
    body: Dict[str, Any] = {}
    for key, value in changes.items():
        if key == "config":
            if value is None:
                raise PlanRefused("config cannot be null (updateComboSchema: not nullable); "
                                  "remove individual keys with config.<key>: null")
            if not isinstance(value, dict):
                raise PlanRefused("config must be an object of keys to set (null removes a key)")
            merged = dict(_rec(current.get("config")))
            for ckey, cval in value.items():
                if cval is None:
                    merged.pop(ckey, None)
                else:
                    merged[ckey] = copy.deepcopy(cval)
            body["config"] = merged
        else:
            body[key] = copy.deepcopy(value)
    return body


def expected_stored(current: Dict[str, Any], body: Dict[str, Any], names: List[str]) -> Dict[str, Any]:
    """What the server will store, reproducing route + repository steps (O-WRITE-2, O-WRITE-3):
    dead config keys stripped; body models normalized; ``{...current, ...body}``; keys sent as
    null deleted; whole record re-normalized (version 2, models). ``updatedAt`` is server-set."""
    combo_name = current.get("name")
    server_body = dict(body)
    if isinstance(server_body.get("config"), dict):
        server_body["config"] = server_config(server_body["config"])
    if server_body.get("models"):
        server_body["models"] = normalize_combo_models(server_body["models"], combo_name=combo_name, names=names)
    merged = {**current, **server_body}
    for key, value in body.items():
        if value is None:
            merged.pop(key, None)
    return normalize_combo_record(merged, names)


def compare_stored(expected: Dict[str, Any], actual: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Differences between an expected and a returned stored combo. ``updatedAt`` is server-set and
    skipped; ``config.weights`` is compared on the keys that were sent (zod fills defaults);
    a ``warning`` field the route may add to the response is not part of the stored combo."""
    actual = {k: v for k, v in actual.items() if k != "warning"}
    expected = dict(expected)
    if isinstance(expected.get("config"), dict) and isinstance(actual.get("config"), dict):
        exp_cfg, act_cfg = dict(expected["config"]), dict(actual["config"])
        for key in CONFIG_SUBSET_COMPARED:
            if isinstance(exp_cfg.get(key), dict) and isinstance(act_cfg.get(key), dict):
                act_cfg[key] = {k: v for k, v in act_cfg[key].items() if k in exp_cfg[key]}
        expected["config"], actual = exp_cfg, dict(actual, config=act_cfg)
    return diff_paths(expected, actual, ignore=SERVER_MANAGED_KEYS)


def rollback_body(snapshot: Dict[str, Any], live: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """The complete snapshot as a PUT body: every writable key the snapshot has, explicit null for
    every nullable key the live combo has and the snapshot lacks. Returns ``(body, unresettable)``
    where ``unresettable`` lists live-only keys the schema cannot null."""
    body: Dict[str, Any] = {}
    unresettable: List[str] = []
    for key in sorted(WRITABLE_KEYS - OUT_OF_SCOPE_KEYS):
        if key in snapshot:
            body[key] = copy.deepcopy(snapshot[key])
        elif key in live:
            if key in NULLABLE_KEYS:
                body[key] = None
            else:
                unresettable.append(key)
    return body, unresettable


# ── checks ───────────────────────────────────────────────────────────────────────────────────────

def irreversibility(current: Dict[str, Any], body: Dict[str, Any]) -> List[str]:
    """Reasons a rollback could not restore ``current`` through the merge route (O-WRITE-4)."""
    reasons = []
    for key, value in body.items():
        if key not in current and key not in NULLABLE_KEYS and value is not None:
            reasons.append(f"'{key}' is absent today and the update schema cannot reset it to absent "
                           "(not nullable)")
    for i, step in enumerate(current.get("models") or []):
        if isinstance(step, dict) and step.get("kind") == "provider-wildcard":
            reasons.append(f"models[{i}] is a provider-wildcard step; the update schema rejects that "
                           "stored shape, so the rollback's complete models array would be refused")
            break
    if not current.get("models"):
        reasons.append("the combo has no models; the update schema rejects an empty models array")
    dead = sorted(set(_rec(current.get("config"))) & set(DEAD_COMBO_CONFIG_KEYS))
    if dead:
        reasons.append(f"config holds keys the server strips on every config write ({', '.join(dead)}); "
                       "the rollback's complete config could not restore them")
    if needs_zero_latency_promotion(current.get("config")):
        reasons.append("config enables a zero-latency feature without zeroLatencyOptimizationsEnabled; the "
                       "server adds that flag on every config write, so the rollback could not restore "
                       "the config as stored")
    return reasons


def predicted_rejects(target_name: str, expected: Dict[str, Any], combos: List[Dict[str, Any]],
                      body: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Server-only checks: reproduced where possible ("predicted"), named otherwise."""
    out: List[Dict[str, Any]] = [
        {"check": "name collision (COMBO_004)", "predicted": False,
         "note": "not applicable: this plugin never sends 'name'"},
    ]
    if "models" in body:
        cycle = _dag_problem(target_name, expected, combos)
        out.append({"check": "nested-combo DAG over all combos (COMBO_005)", "predicted": bool(cycle),
                    "note": cycle or "approximate check over the combos read at plan time found no "
                                     "cycle or depth overflow; the server re-checks against live data"})
    out += [
        {"check": "quota-share managed combo (COMBO_006, 409)", "predicted": False,
         "note": "names starting with the quota-share prefix are refused at plan time; the server "
                 "re-checks the live name"},
        {"check": "update schema ranges/lengths/enums, config sub-schemas (COMBO_002)", "predicted": False,
         "note": "not reproduced (zod); only key names and nullability are checked here"},
        {"check": "composite tiers / quota-only combo-ref execute mode (COMBO_002/003)", "predicted": False,
         "note": "not reproduced"},
        {"check": "allowedProviders / allowedModelFamilies invariant (COMBO_008)", "predicted": False,
         "note": "not reproduced"},
    ]
    return out


def _dag_problem(target_name: str, expected: Dict[str, Any], combos: List[Dict[str, Any]]) -> str:
    graph = {c.get("name"): c for c in combos}
    graph[target_name] = expected
    max_depth = clamp_combo_depth(_rec(expected.get("config")).get("maxComboDepth"))

    def refs(combo: Dict[str, Any]) -> List[str]:
        out = []
        for step in combo.get("models") or []:
            name = step.get("comboName") if isinstance(step, dict) and step.get("kind") == "combo-ref" else (
                step.get("model") if isinstance(step, dict) else step)
            if isinstance(name, str) and name in graph:
                out.append(name)
        return out

    def walk(name: str, seen: Tuple[str, ...], depth: int) -> str:
        if depth > max_depth:
            return f"nesting depth exceeds {max_depth} at '{name}'"
        if name in seen:
            return f"cycle through '{name}'"
        for child in refs(graph.get(name) or {}):
            problem = walk(child, seen + (name,), depth + 1)
            if problem:
                return problem
        return ""

    return walk(target_name, (), 0)


def _leg_findings(expected: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    for i, step in enumerate(expected.get("models") or []):
        if isinstance(step, dict) and step.get("kind") == "model" and not step.get("connectionId"):
            out.append(finding(
                "UNPINNED_LEG", "L3", "info",
                f"Leg {i} ({step.get('model')}) is not pinned to a connection. OmniRoute's pre-dispatch "
                f"max_concurrent skip applies only to pinned legs, so this leg cannot take part in "
                f"ordered spill-over.",
                cite=["O-COMBO-2", "O-ROUTE-4", "L3"], subject={"leg": i, "model": step.get("model")}))
    return out


def _involving(findings: List[Dict[str, Any]], name: str) -> List[Dict[str, Any]]:
    def hit(subject: Dict[str, Any]) -> bool:
        return subject.get("combo") == name or name in (subject.get("combos") or [])
    return [f for f in findings if hit(f.get("subject") or {})]


def _finding_key(f: Dict[str, Any]) -> Tuple[str, str]:
    import json
    return f["code"], json.dumps(f["subject"], sort_keys=True, default=str)


# ── the three operations ─────────────────────────────────────────────────────────────────────────

class ComboWriter:
    def __init__(self, client: OmniRouteClient, store: WriteStore,
                 config_loader: Optional[ConfigLoader] = None):
        self.client = client
        self.store = store
        self.config_loader = config_loader

    def _live(self, combo_id: str) -> Dict[str, Any]:
        return self.client.get("combo", path_params={"id": combo_id})

    def _resolve(self, combos: List[Dict[str, Any]], ref: str) -> Dict[str, Any]:
        for combo in combos:
            if combo.get("id") == ref:
                return combo
        matches = [c for c in combos if c.get("name") == ref]
        if not matches:
            raise PlanRefused(f"no existing combo with id or name {ref!r} (this plugin never creates combos)")
        return matches[0]

    # plan ---------------------------------------------------------------------------------------
    def plan(self, combo_ref: str, changes: Dict[str, Any], caller_limits: Optional[Dict[str, Any]] = None,
             ttl_seconds: int = PLAN_TTL_SECONDS) -> Dict[str, Any]:
        if not isinstance(changes, dict) or not changes:
            raise PlanRefused("changes must be a non-empty object of combo fields")
        unknown = sorted(set(changes) - WRITABLE_KEYS)
        if unknown:
            raise PlanRefused(f"fields {unknown} are not in OmniRoute's update schema; the server would "
                              "silently drop them (zod strips unknown keys)")
        scoped_out = sorted(set(changes) & OUT_OF_SCOPE_KEYS)
        if scoped_out:
            raise PlanRefused(f"fields {scoped_out} are out of scope (no rename; no legacy rewrite switches)")
        bad_null = sorted(k for k, v in changes.items() if v is None and k not in NULLABLE_KEYS)
        if bad_null:
            raise PlanRefused(f"fields {bad_null} cannot be reset with null (not nullable in the update schema)")

        combos_raw = self.client.get("combos")
        combos = _combo_list(combos_raw)
        target = self._resolve(combos, combo_ref)
        current = self._live(str(target.get("id")))
        name = str(current.get("name") or "")
        if name.startswith(QUOTA_MODEL_PREFIX):
            raise PlanRefused(f"combo {name!r} is managed by quota-share (prefix {QUOTA_MODEL_PREFIX!r}); "
                              "out of scope")
        connections_raw = self.client.get("connections")
        nodes_raw = self.client.get("provider_nodes")
        defaults_raw = self.client.get("combo_defaults")

        names = _names(combos, name)
        body = build_put_body(current, changes)
        expected = expected_stored(current, body, names)
        diff = diff_paths(current, expected, ignore=SERVER_MANAGED_KEYS)
        if not diff:
            raise PlanRefused("the change would leave the stored combo as it is (empty diff)")
        reasons = irreversibility(current, body)

        limits = _rec(caller_limits)
        per_call, concurrency = limits.get("per_call_timeout_ms"), limits.get("concurrency")
        proposed = [expected if c.get("id") == current.get("id") else c for c in combos]
        before_snap = audit.build_snapshot(connections_raw, nodes_raw, combos, defaults_raw)
        after_snap = audit.build_snapshot(connections_raw, nodes_raw, proposed, defaults_raw)
        # Audit ALL combos (so connection sharing across combos is visible, L4), keep what involves this one.
        before = _involving(audit.run_audit(before_snap, per_call_timeout_ms=per_call,
                                            concurrency=concurrency)["findings"], name)
        after = _involving(audit.run_audit(after_snap, per_call_timeout_ms=per_call,
                                           concurrency=concurrency)["findings"], name)
        after += _leg_findings(expected)
        hermes_note = None
        if self.config_loader is not None:
            try:
                config = self.config_loader()
                check = combo_check.run_combo_check(config, self.client.base_url, after_snap["combos"],
                                                    after_snap["combo_defaults"])
                after += [f for f in check["findings"] if f["subject"].get("model") == name]
            except Exception as exc:  # outside Hermes: the cross-check is skipped, and said so
                hermes_note = f"Hermes settings cross-check skipped ({type(exc).__name__})"
        before_keys = {_finding_key(f) for f in before}
        for f in after:
            f["new_with_this_change"] = _finding_key(f) not in before_keys

        plan = {
            "combo_id": current.get("id"), "combo_name": name, "changes": changes, "body": body,
            "token": content_hash(current), "expected": expected,
            "planned_fields": sorted({d["path"] for d in diff}),
            "applicable": not reasons, "irreversible_reasons": reasons,
        }
        saved = self.store.save_plan(plan, ttl_seconds)
        return {
            "report": "omniroute_combo_plan", "plan_id": saved["plan_id"], "combo": name,
            "combo_id": current.get("id"), "applicable": not reasons,
            "reversible": not reasons, "irreversible_reasons": reasons,
            "token": saved["token"], "expires_at": saved["expires_at"],
            "diff": diff,
            "put_body_keys": sorted(body),
            "server_normalized": [
                "models: normalizeComboStep (reproduced)", "config: zero-latency flag promotion and dead-key strip (reproduced)",
                "record: version=2 and models re-normalized (reproduced)",
                "config: zod coercion/defaults (NOT reproduced; config.weights compared on sent keys)",
                "updatedAt: set by the server (not compared)"],
            "server_may_reject": predicted_rejects(name, expected, combos, body),
            "findings": after,
            "side_effects": [
                "cloud sync may fire after a successful update (if cloud sync is enabled)",
                "OmniRoute takes a pre-write DB backup",
            ] + (["sending models clears this combo's session model pins"] if "models" in body else []),
            "notes": [n for n in (hermes_note,) if n],
        }

    # apply --------------------------------------------------------------------------------------
    def apply(self, plan_id: str) -> Dict[str, Any]:
        plan = self.store.get_plan(plan_id) if isinstance(plan_id, str) else None
        if plan is None:
            return _refusal("unknown_plan", f"no plan {plan_id!r} (plans expire and are profile-local)")
        if plan.get("consumed"):
            return _refusal("plan_consumed", "this plan was already applied; plan again")
        if self.store.now() >= plan.get("expires_at", 0):
            return _refusal("plan_expired", "the plan expired; plan again")
        if not plan.get("applicable"):
            return _refusal("plan_irreversible", "the plan was marked irreversible",
                            reasons=plan.get("irreversible_reasons"))
        combo_id = str(plan["combo_id"])
        snapshot = self._live(combo_id)
        if content_hash(snapshot) != plan["token"]:
            return _refusal("plan_stale", "the combo changed since the plan was made; plan again",
                            changed=diff_paths(plan["expected"], snapshot, ignore=SERVER_MANAGED_KEYS)[:20])
        self.store.mark_plan(plan_id, consumed=True)
        try:
            returned = self.client.put("combo", plan["body"], path_params={"id": combo_id})
        except OmniRouteError as exc:
            after = self._live(combo_id)
            unchanged = content_hash(after) == plan["token"]
            out = {"report": "omniroute_combo_apply", "applied": False, "plan_id": plan_id,
                   "error": str(exc), "status": exc.status, "server_error": exc.server_error,
                   "combo_unchanged": unchanged}
            if not unchanged:
                out["restore"] = self._restore(combo_id, snapshot, after)
            return out
        stored = _rec(returned)
        mismatch = compare_stored(plan["expected"], stored)
        readback = self._live(combo_id)
        mismatch += [dict(m, source="read-back") for m in compare_stored(plan["expected"], readback)
                     if m["path"] not in {x["path"] for x in mismatch}]
        if mismatch:
            planned = set(plan["planned_fields"])
            return {"report": "omniroute_combo_apply", "applied": False, "plan_id": plan_id,
                    "verification": "failed",
                    "mismatch": [dict(m, planned_field=_under(m["path"], planned)) for m in mismatch],
                    "restore": self._restore(combo_id, snapshot, readback)}
        record = self.store.save_rollback({
            "combo_id": combo_id, "combo_name": plan["combo_name"], "plan_id": plan_id,
            "snapshot": snapshot, "post_apply": readback, "post_apply_hash": content_hash(readback),
            "status": "applied"})
        return {"report": "omniroute_combo_apply", "applied": True, "verification": "passed",
                "plan_id": plan_id, "rollback_id": record["rollback_id"], "combo": plan["combo_name"],
                "verified_diff": diff_paths(snapshot, readback, ignore=SERVER_MANAGED_KEYS)}

    def _restore(self, combo_id: str, snapshot: Dict[str, Any], live: Dict[str, Any]) -> Dict[str, Any]:
        body, unresettable = rollback_body(snapshot, live)
        try:
            self.client.put("combo", body, path_params={"id": combo_id})
        except OmniRouteError as exc:
            return {"restored": False, "error": str(exc), "server_error": exc.server_error}
        after = self._live(combo_id)
        remaining = compare_stored(snapshot, after)
        return {"restored": not remaining, "remaining_differences": remaining,
                "unresettable_fields": unresettable}

    # rollback -----------------------------------------------------------------------------------
    def rollback(self, rollback_id: str, force: bool = False) -> Dict[str, Any]:
        record = self.store.get_rollback(rollback_id) if isinstance(rollback_id, str) else None
        if record is None:
            return _refusal("unknown_rollback", f"no rollback {rollback_id!r}")
        if record.get("status") == "rolled_back":
            return _refusal("already_rolled_back", "this rollback was already performed")
        combo_id = str(record["combo_id"])
        live = self._live(combo_id)
        third_party = diff_paths(record["post_apply"], live, ignore=SERVER_MANAGED_KEYS)
        changed_since = content_hash(live) != record["post_apply_hash"]
        if changed_since and not force:
            return _refusal("changed_since_apply",
                            "the combo changed after the apply; rollback would overwrite that change. "
                            "Pass force=true to overwrite it.", changed=third_party[:20])
        body, unresettable = rollback_body(record["snapshot"], live)
        try:
            self.client.put("combo", body, path_params={"id": combo_id})
        except OmniRouteError as exc:
            after = self._live(combo_id)
            return {"report": "omniroute_combo_rollback", "rolled_back": False, "error": str(exc),
                    "status": exc.status, "server_error": exc.server_error,
                    "combo_unchanged": content_hash(after) == content_hash(live)}
        readback = self._live(combo_id)
        remaining = compare_stored(record["snapshot"], readback)
        record = dict(record, status="rolled_back" if not remaining else "rollback_incomplete",
                      forced=bool(force and changed_since))
        self.store.save_rollback(record)
        return {"report": "omniroute_combo_rollback", "rolled_back": not remaining,
                "verification": "passed" if not remaining else "failed",
                "forced": bool(force and changed_since), "overwrote_third_party_changes": third_party
                if changed_since else [], "remaining_differences": remaining,
                "unresettable_fields": unresettable, "combo": record.get("combo_name"),
                "sent_keys": sorted(body)}


def _under(path: str, planned: set) -> bool:
    return any(path == p or path.startswith(p + ".") or path.startswith(p + "[") for p in planned)


def _refusal(code: str, message: str, **extra: Any) -> Dict[str, Any]:
    return {"refused": True, "code": code, "message": message, **extra}
