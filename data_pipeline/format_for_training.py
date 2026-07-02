"""
data_pipeline/format_for_training.py
=========================================
Converts the mixed dataset (data/final/train.jsonl) into the chat-format
JSONL that Training Hub / Unsloth expects.

Training Hub expects one of these formats:
  - "messages" format (preferred for instruct models):
      {"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}
  - "text" format (raw string, for SFT on completions):
      {"text": "<full prompt + completion>"}

We use the "messages" format because:
  - Qwen2.5-7B-Instruct is a chat model
  - It preserves the system prompt structure
  - Reasoning traces go into the assistant message as <think> blocks
    (compatible with Qwen2.5's native thinking format)

Usage:
    python data_pipeline/format_for_training.py --config config/pipeline_config.yaml
    python data_pipeline/format_for_training.py --input data/final/train.jsonl --output data/final/train_chat.jsonl
"""

import argparse
import json
import logging
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt for the Text-to-SQL task
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert SQL query writer. Given a natural language question and a database schema, write the correct SQL query that answers the question.

Rules:
- Output ONLY the SQL query, nothing else
- The SQL must be syntactically valid
- Use only tables and columns that exist in the provided schema
- Use standard SQL compatible with SQLite and PostgreSQL
- Do not add explanations or markdown formatting around the SQL"""

SYSTEM_PROMPT_WITH_THINKING = """You are an expert SQL query writer. Given a natural language question and a database schema, write the correct SQL query that answers the question.

Think through the problem step by step before writing the SQL:
1. Identify which tables are needed
2. Determine what joins are required
3. Figure out what filters, aggregations, or ordering to apply
4. Write the final SQL

Output your reasoning in <think>...</think> tags, then the SQL query."""

# ---------------------------------------------------------------------------
# Formatting functions
# ---------------------------------------------------------------------------

def format_user_message(question: str, schema: str) -> str:
    """Formats the user turn: schema + question."""
    return (
        f"Database schema:\n"
        f"{schema}\n\n"
        f"Question: {question}"
    )


def format_assistant_message(sql: str, thinking: str = "") -> str:
    """
    Formats the assistant turn.
    If a reasoning trace is available, wraps it in <think> tags
    (Qwen2.5's native chain-of-thought format).
    """
    if thinking and len(thinking.strip()) > 20:
        return f"<think>\n{thinking.strip()}\n</think>\n\n{sql.strip()}"
    return sql.strip()


def to_chat_format(example: dict, strip_thinking: bool = False) -> dict | None:
    """
    Converts a raw example to chat-format training example.
    Returns None if the example is invalid.

    strip_thinking=True forces the NON-THINKING target: the reasoning trace is
    discarded, the assistant turn is SQL only, and the non-thinking system prompt
    is used — regardless of whether a `thinking` field is present. This is the
    Qwen2.5 (non-thinking student) target. With strip_thinking=False the existing
    thinking path is untouched.
    """
    question = example.get("question", "").strip()
    schema = example.get("schema", "").strip()
    sql = example.get("sql", "").strip()
    thinking = "" if strip_thinking else example.get("thinking", "").strip()

    # Basic validation
    if not question or not schema or not sql:
        return None
    if len(sql) < 10:
        return None

    has_thinking = bool(thinking and len(thinking) > 20)
    system_prompt = SYSTEM_PROMPT_WITH_THINKING if has_thinking else SYSTEM_PROMPT

    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": format_user_message(question, schema)},
            {"role": "assistant", "content": format_assistant_message(sql, thinking)},
        ],
        # Metadata (not used in training, but useful for debugging)
        "_meta": {
            "complexity": example.get("complexity", "unknown"),
            "domain": example.get("domain", "unknown"),
            "source": example.get("source", "unknown"),
            "has_thinking": has_thinking,
        },
    }


def format_dataset(input_path: Path, output_path: Path, strip_thinking: bool = False) -> dict:
    """
    Reads raw JSONL, converts to chat format, writes output JSONL.
    Returns statistics dict.

    strip_thinking=True produces the non-thinking (Qwen2.5) target: SQL only,
    reasoning traces discarded.
    """
    stats = {"total": 0, "converted": 0, "skipped": 0, "with_thinking": 0}

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(input_path) as fin, open(output_path, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            stats["total"] += 1

            try:
                example = json.loads(line)
                chat_example = to_chat_format(example, strip_thinking=strip_thinking)

                if chat_example is None:
                    stats["skipped"] += 1
                    continue

                if chat_example["_meta"]["has_thinking"]:
                    stats["with_thinking"] += 1

                fout.write(json.dumps(chat_example, ensure_ascii=False) + "\n")
                stats["converted"] += 1

            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Skipping malformed example: {e}")
                stats["skipped"] += 1

    return stats


def main():
    parser = argparse.ArgumentParser(description="Format dataset for Training Hub")
    parser.add_argument("--config", default="config/pipeline_config.yaml")
    parser.add_argument("--input", default=None, help="Override input JSONL path")
    parser.add_argument("--output", default=None, help="Override output JSONL path")
    parser.add_argument("--no-thinking", action="store_true",
                        help="Non-thinking (Qwen2.5) target: discard reasoning traces, "
                             "assistant turn is SQL only, no <think> block.")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    final_dir = Path(config["data"]["final_dir"])

    splits = [
        ("train", final_dir / "train.jsonl", final_dir / "train_chat.jsonl"),
        ("eval", final_dir / "eval.jsonl", final_dir / "eval_chat.jsonl"),
        ("test", final_dir / "test.jsonl", final_dir / "test_chat.jsonl"),
    ]

    # Override if explicit paths given
    if args.input and args.output:
        splits = [("custom", Path(args.input), Path(args.output))]

    for split_name, in_path, out_path in splits:
        if not in_path.exists():
            logger.warning(f"Skipping {split_name}: {in_path} not found")
            continue

        logger.info(f"Formatting {split_name}: {in_path} → {out_path}"
                    f"{' [non-thinking/SQL-only]' if args.no_thinking else ''}")
        stats = format_dataset(in_path, out_path, strip_thinking=args.no_thinking)

        logger.info(f"  {split_name}: {stats['converted']}/{stats['total']} converted "
                    f"({stats['skipped']} skipped, {stats['with_thinking']} with reasoning)")

    logger.info("\n✅ Formatting complete. Next step: python training_pipeline/train.py")


if __name__ == "__main__":
    main()
