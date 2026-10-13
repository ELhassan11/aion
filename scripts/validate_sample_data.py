#!/usr/bin/env python3
"""Validate sample data against the Pydantic contracts (backend/models).

Usage: python scripts/validate_sample_data.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.models import CustomerProfile, Ticket  # noqa: E402

SAMPLE = ROOT / "data" / "sample"


def load_json(path: Path) -> list:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON array")
    return data


def main() -> None:
    customers = [CustomerProfile(**c) for c in load_json(SAMPLE / "customers.json")]
    tickets = [Ticket(**t) for t in load_json(SAMPLE / "tickets.json")]

    customer_ids = {c.id for c in customers}
    orphan = [t.id for t in tickets if t.customer_id not in customer_ids]
    if orphan:
        raise ValueError(f"tickets with unknown customer_id: {orphan}")

    docs = list((SAMPLE / "documents").glob("*.md"))
    if not docs:
        raise ValueError("no sample documents found")

    print(f"customers OK: {len(customers)}")
    print(f"tickets OK: {len(tickets)} (all linked to known customers)")
    print(f"documents OK: {len(docs)} markdown files")
    print("sample data valid")


if __name__ == "__main__":
    main()
