"""
serving/query_model.py
======================
Sendet Text-to-SQL Anfragen an den laufenden vLLM Server.

Usage:
    # Einfache Anfrage:
    python3 serving/query_model.py \
        --schema "CREATE TABLE employees (id INT, name VARCHAR(50), salary DECIMAL, department VARCHAR(50));" \
        --question "What is the average salary per department?"

    # Interaktiver Modus:
    python3 serving/query_model.py --interactive

    # Gegen einen anderen Port:
    python3 serving/query_model.py --port 8001 --interactive
"""

import argparse
import json
import sys

import requests


SYSTEM_PROMPT = (
    "You are an expert SQL query writer. "
    "Given a database schema and a natural language question, "
    "write the correct SQL query. Output ONLY the SQL query, nothing else."
)


def query_vllm(schema: str, question: str, host: str = "localhost", port: int = 8000) -> str:
    """Sendet eine Text-to-SQL Anfrage an den vLLM Server."""
    url = f"http://{host}:{port}/v1/chat/completions"

    payload = {
        "model": "text2sql",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Database schema:\n{schema}\n\nQuestion: {question}"
            },
        ],
        "max_tokens": 256,
        "temperature": 0.1,
    }

    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()
    except requests.exceptions.ConnectionError:
        print(f"Fehler: Kein vLLM Server auf {host}:{port} erreichbar.")
        print("Starte den Server mit: ./serving/deploy_vllm.sh <checkpoint>")
        sys.exit(1)
    except Exception as e:
        print(f"Fehler: {e}")
        sys.exit(1)


def interactive_mode(host: str, port: int):
    """Interaktiver Modus für mehrere Anfragen."""
    print("=" * 60)
    print("  Text-to-SQL – Interaktiver Modus")
    print(f"  Server: {host}:{port}")
    print("  Beenden: Ctrl+C oder 'exit'")
    print("=" * 60)

    # Standard-Schema für schnelle Tests
    default_schema = (
        "CREATE TABLE employees (id INT, name VARCHAR(50), "
        "salary DECIMAL(10,2), department VARCHAR(50), hire_date DATE);\n"
        "CREATE TABLE departments (id INT, name VARCHAR(50), budget DECIMAL(15,2));"
    )

    while True:
        print("\n--- Neue Anfrage ---")

        schema_input = input(f"Schema [Enter für Beispiel-Schema]: ").strip()
        schema = schema_input if schema_input else default_schema
        if schema == default_schema:
            print(f"  (Nutze Beispiel-Schema)")

        question = input("Frage: ").strip()
        if question.lower() in ("exit", "quit", ""):
            break

        print("\nGeneriere SQL...")
        sql = query_vllm(schema, question, host, port)
        print(f"\n→ SQL:\n{sql}\n")


def main():
    parser = argparse.ArgumentParser(description="Text-to-SQL vLLM Client")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--schema", default=None)
    parser.add_argument("--question", default=None)
    parser.add_argument("--interactive", action="store_true",
                        help="Interaktiver Modus für mehrere Anfragen")
    args = parser.parse_args()

    if args.interactive:
        interactive_mode(args.host, args.port)
    elif args.schema and args.question:
        sql = query_vllm(args.schema, args.question, args.host, args.port)
        print(sql)
    else:
        # Health check
        try:
            r = requests.get(f"http://{args.host}:{args.port}/health", timeout=5)
            print(f"Server Status: {r.status_code}")
            r2 = requests.get(f"http://{args.host}:{args.port}/v1/models", timeout=5)
            print(f"Verfügbare Modelle: {json.dumps(r2.json(), indent=2)}")
        except Exception as e:
            print(f"Server nicht erreichbar: {e}")
            parser.print_help()


if __name__ == "__main__":
    main()
