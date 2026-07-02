"""
evaluation/reward.py
====================
Backend-agnostic execution-accuracy (loose-EX) reward for RLVR / GRPO.

Reuses the SAME SQL extractor as the eval harness (evaluation.evaluate.extract_sql)
so the post-</think> / markdown-fence / plaintext-recovery behaviour is identical and
the known extraction/contamination fixes are NOT re-introduced. evaluation/evaluate.py
is imported, never modified.

What this module ADDS over evaluate.execution_match (which has no timeout):
  - a per-query wall-clock TIMEOUT via a threading.Timer -> sqlite3 Connection.interrupt()
    watchdog (interrupt() is the one sqlite3 call documented safe from another thread), so a
    hanging / cartesian-exploding rollout deterministically scores 0.0 and never stalls the
    trainer;
  - a defensive ROW-CAP (fetchmany) so a cross-join that materialises rows for the full
    timeout window cannot spike memory.

Reward = binary loose-EX (result-set match, row & column order ignored, column count must
match) vs gold, 1.0 / 0.0. Any failure mode (non-exec, mismatch, gold-failed, timeout,
exception, row-cap exceeded) -> 0.0. No reward shaping (paper: final-answer exact-match).

Exposes:
  - score_sql(completion, gold_sql, schema, timeout_s, row_cap) -> float   (the core)
  - compute_score(...)                                   verl reward-manager adapter
  - make_reward_fn(timeout_s, row_cap) / reward_fn       TRL GRPOTrainer reward-callable adapter
"""

import os
import sqlite3
import sys
import threading
from pathlib import Path

from evaluation.evaluate import extract_sql  # single source of truth for SQL extraction

DEFAULT_TIMEOUT_S = 5.0
DEFAULT_ROW_CAP = 100_000

# Tokenizer for the think-token count (trace-collapse axis). Loaded once, fail-safe:
# any Qwen3 tokenizer works; override with env REWARD_TOKENIZER. If it can't load
# (no transformers / no cache), fall back to whitespace word count so reward never breaks.
_TOKENIZER = "unset"


def _get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER == "unset":
        try:
            from transformers import AutoTokenizer
            _TOKENIZER = AutoTokenizer.from_pretrained(
                os.environ.get("REWARD_TOKENIZER", "Qwen/Qwen3-4B"), trust_remote_code=True)
        except Exception:
            _TOKENIZER = None
    return _TOKENIZER


def _think_token_count(completion):
    """Tokens of reasoning before </think> (collapse/loop watch). If never closed, count the
    whole completion (a cap/loop signal). Uses a cached Qwen3 tokenizer; word-count fallback."""
    text = (completion or "").split("</think>")[0]
    tok = _get_tokenizer()
    try:
        if tok is not None:
            return len(tok(text, add_special_tokens=False).input_ids)
    except Exception:
        pass
    return len(text.split())


def _run_sql_timed(sql, schema_ddl, timeout_s=DEFAULT_TIMEOUT_S, row_cap=DEFAULT_ROW_CAP):
    """Execute `sql` against an in-memory SQLite built from `schema_ddl`.

    Mirrors evaluation.evaluate.execute_sql_on_schema, but adds (a) a wall-clock timeout via
    Connection.interrupt() from a watchdog thread, and (b) a fetchmany row-cap. The watchdog
    covers both the execute() and the fetch (interrupt() aborts any op on the connection).

    Returns (ok: bool, rows: list[tuple] | None). ok=False on any error / timeout / row-cap.
    """
    conn = None
    timer = None
    try:
        conn = sqlite3.connect(":memory:")
        cur = conn.cursor()
        # Build the schema. Statements that fail are skipped — identical to the eval harness
        # (evaluation/evaluate.py:92-98), so "sets up" means the same thing everywhere.
        for stmt in schema_ddl.split(";"):
            stmt = stmt.strip()
            if stmt:
                try:
                    cur.execute(stmt)
                except sqlite3.Error:
                    pass
        conn.commit()
        # Arm the watchdog AFTER schema build so we only bound the query itself.
        timer = threading.Timer(timeout_s, conn.interrupt)
        timer.daemon = True
        timer.start()
        cur.execute(sql)
        rows = cur.fetchmany(row_cap + 1)  # bounded pull; +1 detects overflow
        if len(rows) > row_cap:            # cross-join blow-up guard -> treat as failure
            return False, None
        return True, rows
    except Exception:
        return False, None
    finally:
        if timer is not None:
            timer.cancel()
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def _loose_key(rows):
    # SAME canonicalisation as evaluation/evaluate.py:151-153 (loose EX): row AND column order
    # ignored, column count preserved. Keep in sync with that line if it ever changes.
    return sorted(tuple(sorted(str(v) for v in row)) for row in rows)


def score_sql(completion, gold_sql, schema, timeout_s=DEFAULT_TIMEOUT_S, row_cap=DEFAULT_ROW_CAP):
    """Binary loose-EX reward for ONE rollout.

    Returns 1.0 iff the extracted SQL's result set loose-matches the gold's, with both
    executed within the timeout/row-cap; 0.0 on any failure (non-exec, mismatch, gold-failed,
    timeout, exception, empty extraction). Never raises.
    """
    try:
        pred_sql = extract_sql(completion or "")
        if not pred_sql or not pred_sql.strip():
            return 0.0
        gold_ok, gold_rows = _run_sql_timed(gold_sql, schema, timeout_s, row_cap)
        if not gold_ok:
            return 0.0  # gold itself failed -> no signal (such prompts are filtered upstream)
        pred_ok, pred_rows = _run_sql_timed(pred_sql, schema, timeout_s, row_cap)
        if not pred_ok:
            return 0.0
        return 1.0 if _loose_key(gold_rows) == _loose_key(pred_rows) else 0.0
    except Exception:
        return 0.0


# --- verl reward-manager adapter -------------------------------------------------------
def compute_score(data_source=None, solution_str=None, ground_truth=None, extra_info=None, **kw):
    """verl reward contract: ground_truth = gold_sql; extra_info carries {'schema': ...} and
    optionally {'reward_timeout_s', 'row_cap'}.

    Returns a DICT — verl's NaiveRewardManager uses dict["score"] as the reward and forwards
    every other key into reward_extra_info, which verl reduces into VAL metrics as
    `val-aux/<data_source>/<key>/mean@1` (process_validation_metrics). So on the held-out val set:
      - val-core/sql_exec/acc/mean@1  = held-out weak-category EX (the score itself)
      - val-aux/sql_exec/think_tokens/mean@1 = trace-collapse axis (mean reasoning tokens)
      - val-aux/sql_exec/truncated/mean@1    = fraction of rollouts that never closed </think>
      - val-aux/sql_exec/passed/mean@1       = pass rate
    (Per-TRAIN-step truncation/variance come from NATIVE verl metrics — response_length/clip_ratio
    and critic/advantages/{max,min}≠0 ⟺ intra-group variance>0 — no callback needed.)
    """
    extra_info = extra_info or {}
    schema = extra_info.get("schema", "")
    timeout_s = float(extra_info.get("reward_timeout_s", DEFAULT_TIMEOUT_S))
    row_cap = int(extra_info.get("row_cap", DEFAULT_ROW_CAP))
    reward = score_sql(solution_str, ground_truth, schema, timeout_s, row_cap)
    return {
        "score": reward,
        "think_tokens": _think_token_count(solution_str),
        "passed": float(reward == 1.0),
        "truncated": float("</think>" not in (solution_str or "")),  # no clean close = truncated/looped
    }


# --- TRL GRPOTrainer reward-callable adapter -------------------------------------------
def make_reward_fn(timeout_s=DEFAULT_TIMEOUT_S, row_cap=DEFAULT_ROW_CAP):
    def reward_fn(prompts=None, completions=None, **cols):
        """TRL contract: `completions` is a list (len = num_generations * num_prompts); the
        dataset's non-prompt columns (`schema`, `gold_sql`/`sql`, ...) arrive as same-length
        list kwargs. Returns one float per completion."""
        schemas = cols.get("schema")
        golds = cols.get("gold_sql") or cols.get("sql")
        out = []
        for i, comp in enumerate(completions or []):
            if isinstance(comp, list):  # chat-style completion -> take last message content
                comp = comp[-1].get("content", "") if comp else ""
            schema = schemas[i] if schemas else ""
            gold = golds[i] if golds else ""
            out.append(score_sql(comp, gold, schema, timeout_s, row_cap))
        return out
    return reward_fn


reward_fn = make_reward_fn()


# --- self-test -------------------------------------------------------------------------
def _selftest():
    schema = "CREATE TABLE t (a INT, b INT); INSERT INTO t VALUES (1,2),(3,4);"
    gold = "SELECT a, b FROM t ORDER BY a;"
    assert score_sql("```sql\nSELECT b, a FROM t;\n```", gold, schema) == 1.0  # col reorder = loose match
    assert score_sql("<think>reason here</think>\nSELECT a,b FROM t;", gold, schema) == 1.0  # thinking trace
    assert score_sql("SELECT a FROM t;", gold, schema) == 0.0                  # wrong (col count)
    assert score_sql("not sql at all", gold, schema) == 0.0                    # unparseable
    assert score_sql("", gold, schema) == 0.0                                  # empty
    assert score_sql("SELECT * FROM t x, t y, t z;", gold, schema) in (0.0, 1.0)  # bounded cross-join -> float
    # TRL adapter shape
    rf = make_reward_fn()
    r = rf(completions=["SELECT b,a FROM t;", "SELECT a FROM t;"],
           schema=[schema, schema], gold_sql=[gold, gold])
    assert r == [1.0, 0.0], r
    # verl adapter shape (dict with score + reward_extra_info keys)
    d = compute_score(solution_str="<think>reason</think>\nSELECT b,a FROM t;", ground_truth=gold,
                      extra_info={"schema": schema})
    assert d["score"] == 1.0 and d["passed"] == 1.0 and d["truncated"] == 0.0, d
    assert d["think_tokens"] >= 1, d
    dt = compute_score(solution_str="reasoning without a close", ground_truth=gold,
                       extra_info={"schema": schema})
    assert dt["score"] == 0.0 and dt["truncated"] == 1.0, dt   # no </think> -> truncated
    print("reward.py self-test OK")


if __name__ == "__main__":
    _selftest()
