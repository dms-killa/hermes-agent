import sqlite3

from bayes_check.scorer import DEFAULT_LR_TABLE
from bayes_check.state import Store


def _row(i, outcome_type="official_docs", claim_type="stat"):
    return {"id": f"c{i}", "hash": f"{i:08x}", "text": f"claim {i}", "claim_type": claim_type,
            "prior": 0.5, "posterior": 0.8, "verdict": "uncertain",
            "evidence": [{"source_type": outcome_type, "stance": "supports"}]}


def test_writes_are_committed_and_visible_to_a_second_connection(tmp_path):
    path = tmp_path / "db.sqlite"
    writer = Store(opener=lambda: sqlite3.connect(path, check_same_thread=False))
    writer.append_claims("s1", "call1", [_row(1)])
    reader = sqlite3.connect(path)
    assert reader.execute("SELECT COUNT(*) FROM claims").fetchone()[0] == 1
    writer.bind_turn("call1", "turn-1")
    assert reader.execute("SELECT turn_id FROM claims").fetchone()[0] == "turn-1"
    writer.close()


def test_label_by_hash_prefix_and_by_model_id(store):
    store.append_claims("s", "call", [_row(1), _row(2)])
    store.bind_turn("call", "t1")
    assert store.label("00000001", True) == 1
    assert store.label("c2", False, turn_id="t1") == 1
    assert store.label("nope", True) == 0
    assert store.stats()["labelled"] == 2


def test_refit_persists_calibrated_lrs(store):
    rows = [_row(i) for i in range(10)]
    store.append_claims("s", "call", rows)
    for i in range(10):
        store.label(f"{i:08x}", i < 9)
    out = store.refit()
    assert out["labelled"] == 10 and "official_docs" in out["lrs"]
    table = store.lr_table(DEFAULT_LR_TABLE)
    assert table["official_docs"] == out["lrs"]["official_docs"]["lr"]
    assert table["web_search"] == DEFAULT_LR_TABLE["web_search"]
    assert store.base_rate("stat") is not None


def test_rotation_drops_oldest_unlabelled_and_keeps_labels(tmp_path):
    path = tmp_path / "rot.sqlite"
    s = Store(opener=lambda: sqlite3.connect(path, check_same_thread=False), max_db_bytes=10**12)
    big = [dict(_row(i), text="x" * 2000) for i in range(3000)]
    s.append_claims("s", "c", big)
    s.label(f"{0:08x}", True)   # oldest row is labelled: must survive
    s.max_db_bytes = 2 * 1024 * 1024
    removed = s.rotate_if_needed()
    assert removed > 0
    conn = sqlite3.connect(path)
    assert conn.execute("SELECT COUNT(*) FROM claims WHERE outcome IS NOT NULL").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM evidence WHERE claim_row NOT IN (SELECT id FROM claims)").fetchone()[0] == 0
    assert s._db_bytes() <= 2 * 1024 * 1024
    s.close()


def test_gate_skip_events_are_counted(store):
    store.record_event("gate_skipped", "s", {"turn_id": "t"})
    assert store.stats()["gate_skips"] == 1
