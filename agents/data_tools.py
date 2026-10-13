"""Data tools for agents (P3-T02): structured access to customers & tickets.

Each Tool has a name, an Arabic/English description for the LLM, a JSON
parameter schema, and a Python function. The planner (P3-T03) will call
these by name with JSON arguments.

Usage:
    python -m agents.data_tools --list
    python -m agents.data_tools --call customer_lookup --params '{"customer_id": "cust-001"}'
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    func: Callable[..., str]


def _load_json(name: str):
    path = ROOT / "data" / "sample" / name
    return json.loads(path.read_text(encoding="utf-8"))


def customer_lookup(customer_id: str) -> str:
    """Full Customer 360 profile: customer record + their tickets."""
    customers = _load_json("customers.json")
    tickets = _load_json("tickets.json")
    customer = next((c for c in customers if c["id"] == customer_id), None)
    if not customer:
        return json.dumps({"error": f"unknown customer_id: {customer_id}"},
                          ensure_ascii=False)
    own_tickets = [t for t in tickets if t["customer_id"] == customer_id]
    profile = dict(customer)
    profile["tickets"] = [
        {k: t[k] for k in ("id", "subject", "status", "priority", "channel")}
        for t in own_tickets
    ]
    return json.dumps(profile, ensure_ascii=False, indent=2)


def customer_search(query: str) -> str:
    """Find customers by name fragment or segment."""
    customers = _load_json("customers.json")
    q = query.strip()
    hits = [
        c for c in customers
        if q in c["name"] or q == c["segment"]
    ]
    brief = [
        {k: c[k] for k in ("id", "name", "segment", "lifetime_value", "churn_risk")}
        for c in hits
    ]
    return json.dumps(brief, ensure_ascii=False, indent=2)


def ticket_lookup(ticket_id: str) -> str:
    """Full ticket details including messages and AI summary."""
    tickets = _load_json("tickets.json")
    ticket = next((t for t in tickets if t["id"] == ticket_id), None)
    if not ticket:
        return json.dumps({"error": f"unknown ticket_id: {ticket_id}"},
                          ensure_ascii=False)
    return json.dumps(ticket, ensure_ascii=False, indent=2)


def customer_stats() -> str:
    """Aggregate stats: counts, avg lifetime value, churn distribution, open tickets."""
    customers = _load_json("customers.json")
    tickets = _load_json("tickets.json")
    n = len(customers)
    avg_ltv = sum(c["lifetime_value"] for c in customers) / max(n, 1)
    churn = {}
    for c in customers:
        churn[c["churn_risk"]] = churn.get(c["churn_risk"], 0) + 1
    segments = {}
    for c in customers:
        segments[c["segment"]] = segments.get(c["segment"], 0) + 1
    open_tickets = sum(1 for t in tickets if t["status"] in ("open", "مفتوحة"))
    return json.dumps(
        {
            "customers": n,
            "avg_lifetime_value": round(avg_ltv, 2),
            "churn_risk_distribution": churn,
            "segments": segments,
            "tickets": len(tickets),
            "open_tickets": open_tickets,
        },
        ensure_ascii=False,
        indent=2,
    )


TOOLS: dict[str, Tool] = {
    "customer_lookup": Tool(
        name="customer_lookup",
        description="Get a customer's full profile (Customer 360) by ID, including their tickets.",
        parameters={"customer_id": "string, e.g. cust-001"},
        func=customer_lookup,
    ),
    "customer_search": Tool(
        name="customer_search",
        description="Search customers by name fragment or segment (نشط/مميز/جديد/خامل).",
        parameters={"query": "string"},
        func=customer_search,
    ),
    "ticket_lookup": Tool(
        name="ticket_lookup",
        description="Get a support ticket's full details by ID.",
        parameters={"ticket_id": "string, e.g. tkt-001"},
        func=ticket_lookup,
    ),
    "customer_stats": Tool(
        name="customer_stats",
        description="Aggregate customer statistics (counts, avg lifetime value, churn risk).",
        parameters={},
        func=customer_stats,
    ),
}


def call_tool(name: str, **kwargs) -> str:
    tool = TOOLS.get(name)
    if not tool:
        return json.dumps({"error": f"unknown tool: {name}. "
                                    f"Available: {sorted(TOOLS)}"},
                          ensure_ascii=False)
    try:
        return tool.func(**kwargs)
    except TypeError as e:
        return json.dumps({"error": f"bad parameters for {name}: {e}. "
                                    f"Expected: {tool.parameters}"},
                          ensure_ascii=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="Agent data tools (P3-T02)")
    ap.add_argument("--list", action="store_true", help="list available tools")
    ap.add_argument("--call", dest="call", default=None, help="tool name to call")
    ap.add_argument("--params", default="{}", help="JSON parameters")
    args = ap.parse_args()

    if args.list:
        for t in TOOLS.values():
            print(f"- {t.name}: {t.description}")
            print(f"  params: {json.dumps(t.parameters, ensure_ascii=False)}")
        return
    if args.call:
        print(call_tool(args.call, **json.loads(args.params)))
        return
    ap.print_help()


if __name__ == "__main__":
    main()
