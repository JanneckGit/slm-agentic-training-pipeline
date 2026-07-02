"""
sdg_pipeline/blocks/sql_complexity_filter.py
============================================
Custom SDG Hub block that scores the quality of generated Text-to-SQL examples.

Registered as "SQLComplexityFilterBlock" in SDG Hub's BlockRegistry.
It computes a quality_score (0.0–1.0) for each example based on:
  - SQL syntactic checks (heuristic, no DB needed)
  - Whether the upgraded SQL is actually more complex than original
  - Completeness: all required fields were parsed successfully
  - Reasoning trace quality (length / completeness)

Examples below the min_score_threshold are filtered out by the flow.
"""

import re
from typing import Any

from datasets import Dataset

# SDG Hub imports – these are available when sdg-hub is installed
try:
    from sdg_hub.core.blocks.base import BaseBlock
    from sdg_hub.core.blocks.registry import BlockRegistry
    SDG_HUB_AVAILABLE = True
except ImportError:
    # Fallback for local testing without sdg_hub installed
    SDG_HUB_AVAILABLE = False
    BaseBlock = object
    BlockRegistry = None


# SQL keywords that indicate increasing complexity
COMPLEXITY_MARKERS = {
    "basic":        [],
    "join":         ["JOIN"],
    "aggregation":  ["GROUP BY", "HAVING", "COUNT(", "SUM(", "AVG(", "MAX(", "MIN("],
    "subquery":     ["SELECT", "(SELECT"],   # nested SELECT
    "window":       ["OVER (", "OVER(", "ROW_NUMBER(", "RANK(", "LAG(", "LEAD(", "NTILE("],
    "cte":          ["WITH ", "AS ("],
    "set_ops":      ["EXCEPT", "INTERSECT", "UNION"],
}


def count_sql_complexity(sql: str) -> int:
    """
    Returns a complexity score (0–7) for a SQL string.
    Higher = more complex SQL features used.
    """
    if not sql:
        return 0
    sql_upper = sql.upper()
    score = 0
    for level, markers in COMPLEXITY_MARKERS.items():
        if any(m in sql_upper for m in markers):
            score += 1
    return score


def is_valid_sql_heuristic(sql: str) -> bool:
    """
    Lightweight SQL validity check without a database connection.
    Checks for common structural issues.
    """
    if not sql or len(sql.strip()) < 10:
        return False
    sql_upper = sql.upper().strip()
    # Must start with a SQL statement keyword
    if not any(sql_upper.startswith(kw) for kw in ["SELECT", "WITH", "INSERT", "UPDATE", "DELETE"]):
        return False
    # Balanced parentheses
    if sql.count("(") != sql.count(")"):
        return False
    # No obvious placeholder text
    bad_patterns = ["<your", "TODO", "FIXME", "...", "YOUR_SQL"]
    if any(p in sql for p in bad_patterns):
        return False
    return True


def score_reasoning_trace(trace: str) -> float:
    """
    Scores a chain-of-thought reasoning trace for quality.
    Returns 0.0–1.0.
    """
    if not trace or len(trace.strip()) < 50:
        return 0.0

    score = 0.0
    # Reward length (up to a point)
    score += min(0.3, len(trace) / 1000)
    # Reward mentions of SQL keywords (it's actually talking about SQL)
    sql_mentions = len(re.findall(r'\b(SELECT|WHERE|JOIN|GROUP BY|HAVING|ORDER BY|WITH|SUBQUERY)\b',
                                   trace, re.IGNORECASE))
    score += min(0.4, sql_mentions * 0.08)
    # Reward step indicators
    step_indicators = len(re.findall(r'(\bstep\b|\bfirst\b|\bnext\b|\bthen\b|\bfinally\b|\d+\.)',
                                      trace, re.IGNORECASE))
    score += min(0.3, step_indicators * 0.06)

    return min(1.0, score)


def compute_quality_score(
    original_sql: str,
    upgraded_sql: str,
    variant_sql: str,
    upgraded_question: str,
    variant_question: str,
    reasoning_trace: str,
) -> float:
    """
    Computes an overall quality score for a generated example.
    Returns 0.0–1.0. Below ~0.7 the example should be dropped.
    """
    scores = []

    # 1. Upgraded SQL: valid + actually harder than original
    if is_valid_sql_heuristic(upgraded_sql):
        orig_complexity = count_sql_complexity(original_sql)
        upgraded_complexity = count_sql_complexity(upgraded_sql)
        if upgraded_complexity > orig_complexity:
            scores.append(1.0)
        elif upgraded_complexity == orig_complexity:
            scores.append(0.6)  # Not harder, but at least valid
        else:
            scores.append(0.2)  # Got simpler – bad
    else:
        scores.append(0.0)

    # 2. Variant SQL: valid
    if is_valid_sql_heuristic(variant_sql):
        scores.append(0.9)
    else:
        scores.append(0.1)

    # 3. Questions: non-empty and non-trivial
    q_score = 0.0
    if upgraded_question and len(upgraded_question) > 15:
        q_score += 0.5
    if variant_question and len(variant_question) > 15:
        q_score += 0.5
    scores.append(q_score)

    # 4. Reasoning trace quality
    scores.append(score_reasoning_trace(reasoning_trace))

    # Weighted average
    weights = [0.35, 0.25, 0.15, 0.25]
    final = sum(s * w for s, w in zip(scores, weights))
    return round(final, 3)


# =============================================================================
# SDG Hub Block Registration
# =============================================================================

if SDG_HUB_AVAILABLE and BlockRegistry is not None:
    @BlockRegistry.register(
        "SQLComplexityFilterBlock",
        "filtering",
        "Scores and filters Text-to-SQL generated examples by quality and complexity gain"
    )
    class SQLComplexityFilterBlock(BaseBlock):
        """
        Custom SDG Hub block for scoring Text-to-SQL synthetic examples.

        Computes a quality_score for each row and sets it to -1.0 for rows
        that should be filtered out (below min_score_threshold).
        The flow then drops those rows.
        """

        def __init__(self, min_score_threshold: float = 0.7, **kwargs):
            super().__init__(**kwargs)
            self.min_score_threshold = min_score_threshold

        def generate(self, samples: Dataset, **kwargs: Any) -> Dataset:
            def score_row(row):
                score = compute_quality_score(
                    original_sql=row.get("sql", ""),
                    upgraded_sql=row.get("upgraded_sql", ""),
                    variant_sql=row.get("variant_sql", ""),
                    upgraded_question=row.get("upgraded_question", ""),
                    variant_question=row.get("variant_question", ""),
                    reasoning_trace=row.get("reasoning_trace", ""),
                )
                # Mark low-quality examples for filtering
                if score < self.min_score_threshold:
                    score = -1.0
                row["quality_score"] = score
                return row

            return samples.map(score_row)

else:
    # Stub for environments without SDG Hub (e.g. local testing)
    class SQLComplexityFilterBlock:
        """Stub class – SDG Hub not installed."""
        def __init__(self, *args, **kwargs):
            pass

        def generate(self, samples, **kwargs):
            return samples


# =============================================================================
# Standalone usage for testing
# =============================================================================
if __name__ == "__main__":
    # Quick smoke test
    test_cases = [
        {
            "original": "SELECT name FROM users WHERE age > 30",
            "upgraded": "SELECT u.name, COUNT(o.id) as order_count FROM users u LEFT JOIN orders o ON u.id = o.user_id WHERE u.age > 30 GROUP BY u.name HAVING COUNT(o.id) > 5 ORDER BY order_count DESC",
            "expected": "high quality (upgraded is harder)",
        },
        {
            "original": "SELECT * FROM products",
            "upgraded": "SELECT * FROM products",  # Same – not harder
            "expected": "medium quality (not harder)",
        },
        {
            "original": "SELECT id FROM users",
            "upgraded": "SELECT",  # Invalid
            "expected": "low quality (invalid SQL)",
        },
    ]

    for tc in test_cases:
        score = compute_quality_score(
            original_sql=tc["original"],
            upgraded_sql=tc["upgraded"],
            variant_sql="SELECT COUNT(*) FROM orders GROUP BY user_id",
            upgraded_question="How many orders has each user placed?",
            variant_question="Show total orders per user",
            reasoning_trace="Step 1: Identify tables. Step 2: JOIN users with orders. Step 3: GROUP BY user_id.",
        )
        print(f"Score: {score:.3f} | Expected: {tc['expected']}")
