"""Customer 360 builder (P1-T05).

Joins raw customer records with their tickets into one unified
CustomerProfile per customer, including computed aggregates
(ticket counts, channels used). Output is validated against the
backend.models contracts and written as JSONL.

Usage:
    python -m ingestion.customer_360 \\
        --customers data/sample/customers.json \\
        --tickets data/sample/tickets.json \\
        --output out/customer_360.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.models import CustomerProfile, Ticket  # noqa: E402

_OPEN_STATUSES = {"open", "in_progress"}


def build_customer_360(
    customers_path: Path, tickets_path: Path
) -> list[CustomerProfile]:
    """Build one validated CustomerProfile per customer."""
    customers = json.loads(customers_path.read_text(encoding="utf-8"))
    tickets = [Ticket(**t) for t in json.loads(tickets_path.read_text(encoding="utf-8"))]

    by_customer: dict[str, list[Ticket]] = {}
    for ticket in tickets:
        by_customer.setdefault(ticket.customer_id, []).append(ticket)

    profiles: list[CustomerProfile] = []
    for raw in customers:
        cust_tickets = by_customer.get(raw["id"], [])
        profiles.append(
            CustomerProfile(
                **raw,
                ticket_count=len(cust_tickets),
                open_ticket_count=sum(
                    1 for t in cust_tickets if t.status.value in _OPEN_STATUSES
                ),
                channels=sorted({t.channel for t in cust_tickets}),
            )
        )
    return profiles


def main() -> None:
    ap = argparse.ArgumentParser(description="Build Customer 360 profiles (P1-T05)")
    ap.add_argument("--customers", required=True, help="customers JSON path")
    ap.add_argument("--tickets", required=True, help="tickets JSON path")
    ap.add_argument("--output", required=True, help="output JSONL path")
    args = ap.parse_args()

    profiles = build_customer_360(Path(args.customers), Path(args.tickets))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        for profile in profiles:
            f.write(profile.model_dump_json() + "\n")

    # Validate the round-trip.
    with open(output, encoding="utf-8") as f:
        for line in f:
            CustomerProfile.model_validate_json(line)

    print(f"wrote {len(profiles)} customer-360 profiles to {args.output}")


if __name__ == "__main__":
    main()
