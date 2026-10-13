"""Enriched Customer 360 profiles (P4-T01).

Builds on P1-T05's basic join: adds rule-based intelligence per customer —
churn scoring, engagement tier, ticket digest, and a recommended action.
Deterministic (no LLM calls) so it runs offline.

Usage:
    python -m customer_intelligence.profiles \\
        --customers data/sample/customers.json \\
        --tickets data/sample/tickets.json \\
        --output out/customer_360_enriched.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_OPEN_STATUSES = {"open", "in_progress", "مفتوحة"}
_URGENT = {"urgent", "عاجلة"}


def _days_since(iso: str, now: datetime) -> int:
    try:
        dt = datetime.fromisoformat(iso)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return max((now - dt).days, 0)
    except (ValueError, TypeError):
        return 999


def churn_score(customer: dict, open_tickets: int, days_since: int) -> float:
    """Rule-based churn score 0..1."""
    score = float(customer.get("churn_risk", 0.5))
    if days_since > 30:
        score += 0.20
    elif days_since > 14:
        score += 0.10
    if open_tickets > 0:
        score += 0.15
    if customer.get("segment") == "خامل":
        score += 0.10
    return round(min(max(score, 0.0), 1.0), 2)


def engagement_tier(lifetime_value: float) -> str:
    if lifetime_value >= 10000:
        return "high"
    if lifetime_value >= 5000:
        return "medium"
    return "low"


def recommended_action(customer: dict, tickets: list[dict],
                       score: float) -> str:
    urgent_open = [t for t in tickets
                   if t.get("priority") in _URGENT
                   and t.get("status") in _OPEN_STATUSES]
    if urgent_open:
        return "متابعة فورية: تذكرة عاجلة مفتوحة"
    if score >= 0.7:
        return "استعادة: عرض خاص لإعادة التفاعل"
    if customer.get("segment") == "جديد":
        return "تأهيل: رسالة ترحيب ودليل البدء"
    if customer.get("segment") == "مميز":
        return "ولاء: مكافأة VIP"
    return "متابعة دورية"


def enrich(customer: dict, tickets: list[dict], now: datetime) -> dict:
    own = [t for t in tickets if t.get("customer_id") == customer["id"]]
    open_t = [t for t in own if t.get("status") in _OPEN_STATUSES]
    days = _days_since(customer.get("last_interaction_at", ""), now)
    score = churn_score(customer, len(open_t), days)
    profile = dict(customer)
    profile.update({
        "days_since_last_interaction": days,
        "churn_score": score,
        "churn_level": ("critical" if score >= 0.7
                        else "watch" if score >= 0.4 else "healthy"),
        "engagement_tier": engagement_tier(
            float(customer.get("lifetime_value", 0))),
        "ticket_count": len(own),
        "open_ticket_count": len(open_t),
        "ticket_subjects": [t.get("subject", "") for t in own],
        "recommended_action": recommended_action(customer, own, score),
    })
    return profile


def main() -> None:
    ap = argparse.ArgumentParser(description="Enriched Customer 360 (P4-T01)")
    ap.add_argument("--customers", required=True)
    ap.add_argument("--tickets", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    customers = json.loads(Path(args.customers).read_text(encoding="utf-8"))
    tickets = json.loads(Path(args.tickets).read_text(encoding="utf-8"))
    now = datetime.now(timezone.utc)

    profiles = [enrich(c, tickets, now) for c in customers]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(profiles, ensure_ascii=False, indent=2),
                   encoding="utf-8")

    critical = sum(1 for p in profiles if p["churn_level"] == "critical")
    print(f"enriched {len(profiles)} profiles -> {out} "
          f"({critical} critical churn)")


if __name__ == "__main__":
    main()
