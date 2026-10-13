"""Churn signals & next best action (P4-T04).

Reads enriched profiles (+ optional conversation insights) and produces,
per customer: detected churn signals with weights, and a ranked next-best
action (what to do, through which channel, why). Deterministic rules.

Usage:
    python -m customer_intelligence.churn \\
        --profiles out/customer_360_enriched.json \\
        --conversations out/conversation_insights.json \\
        --output out/churn_actions.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def detect_signals(profile: dict,
                   conversations: dict[str, dict]) -> list[dict]:
    """Return churn signals with weights for one customer."""
    signals = []
    days = profile.get("days_since_last_interaction", 0)
    if days > 30:
        signals.append({"signal": "dormant_30d",
                        "weight": 0.25,
                        "detail": f"آخر تفاعل منذ {days} يوم"})
    elif days > 14:
        signals.append({"signal": "dormant_14d",
                        "weight": 0.12,
                        "detail": f"آخر تفاعل منذ {days} يوم"})

    if profile.get("open_ticket_count", 0) > 0:
        signals.append({"signal": "open_complaints",
                        "weight": 0.15,
                        "detail": f"{profile['open_ticket_count']} تذاكر مفتوحة"})

    conv = conversations.get(profile["id"], {})
    if conv.get("overall_sentiment") == "negative":
        signals.append({"signal": "negative_sentiment",
                        "weight": 0.18,
                        "detail": "مشاعر سلبية في المحادثات"})
    if conv.get("needs_human"):
        signals.append({"signal": "escalation_needed",
                        "weight": 0.20,
                        "detail": "يحتاج تدخل بشري"})

    if profile.get("engagement_tier") == "low":
        signals.append({"signal": "low_engagement",
                        "weight": 0.10,
                        "detail": "تفاعل منخفض"})

    ltv = float(profile.get("lifetime_value", 0))
    if ltv >= 10000 and profile.get("churn_level") in ("watch", "critical"):
        signals.append({"signal": "high_value_at_risk",
                        "weight": 0.15,
                        "detail": f"قيمة عمرية مرتفعة ({ltv:,.0f}) معرضة للخطر"})
    return signals


def next_best_action(profile: dict, signals: list[dict]) -> dict:
    """Map signals to one prioritized action."""
    names = {s["signal"] for s in signals}
    name = profile.get("name", profile["id"])

    if "escalation_needed" in names:
        return {"action": "تدخل بشري فوري",
                "channel": "اتصال هاتفي", "priority": 1,
                "reason": "إشارات تصعيد في المحادثة"}
    if "open_complaints" in names and profile.get("churn_level") == "critical":
        return {"action": "متابعة الشكوى المفتوحة مع عرض تعويض",
                "channel": "واتساب", "priority": 1,
                "reason": "شكوى مفتوحة + خطر تسرب حرج"}
    if "high_value_at_risk" in names:
        return {"action": "عرض ولاء حصري",
                "channel": "واتساب", "priority": 2,
                "reason": f"عميل عالي القيمة ({name}) معرض للخطر"}
    if "open_complaints" in names:
        return {"action": "متابعة الشكاوى المفتوحة وإغلاقها",
                "channel": "واتساب", "priority": 2,
                "reason": "تذاكر مفتوحة تحتاج إغلاق"}
    if "dormant_30d" in names:
        return {"action": "حملة استعادة بعرض خاص",
                "channel": "واتساب", "priority": 2,
                "reason": "خمول أكثر من 30 يوم"}
    if "dormant_14d" in names:
        return {"action": "تذكير ودود بالعروض الجديدة",
                "channel": "واتساب", "priority": 3,
                "reason": "خمول أكثر من 14 يوم"}
    if "negative_sentiment" in names:
        return {"action": "اعتذار ومراجعة التجربة",
                "channel": "ماسنجر", "priority": 3,
                "reason": "مشاعر سلبية مسجلة"}
    if profile.get("segment") == "جديد":
        return {"action": "رسالة تأهيل ودليل البدء",
                "channel": "واتساب", "priority": 4,
                "reason": "عميل جديد"}
    return {"action": "متابعة دورية",
            "channel": "واتساب", "priority": 5,
            "reason": "لا إشارات خطر"}


def analyze(profiles: list[dict],
            conversations: dict[str, dict]) -> list[dict]:
    results = []
    for p in profiles:
        signals = detect_signals(p, conversations)
        nba = next_best_action(p, signals)
        results.append({
            "customer_id": p["id"],
            "name": p.get("name"),
            "churn_score": p.get("churn_score"),
            "churn_level": p.get("churn_level"),
            "signals": signals,
            "next_best_action": nba,
        })
    # highest risk first, then highest priority action
    results.sort(key=lambda r: (-(r["churn_score"] or 0),
                                r["next_best_action"]["priority"]))
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="Churn signals + NBA (P4-T04)")
    ap.add_argument("--profiles", required=True)
    ap.add_argument("--conversations", default=None,
                    help="conversation_insights.json (optional)")
    ap.add_argument("--tickets", default=None,
                    help="tickets.json to map insights to customers")
    ap.add_argument("--output", default=None)
    ap.add_argument("--customer-id", default=None)
    args = ap.parse_args()

    profiles = json.loads(Path(args.profiles).read_text(encoding="utf-8"))
    conversations: dict[str, dict] = {}
    if args.conversations and args.tickets:
        items = json.loads(Path(args.conversations).read_text(
            encoding="utf-8"))
        tickets = json.loads(Path(args.tickets).read_text(encoding="utf-8"))
        t2c = {t["id"]: t.get("customer_id") for t in tickets}
        for item in items:
            cid = t2c.get(item.get("ticket_id"))
            if cid:
                conversations[cid] = item

    if args.customer_id:
        profiles = [p for p in profiles if p["id"] == args.customer_id]
        if not profiles:
            print(f"unknown customer: {args.customer_id}")
            return

    results = analyze(profiles, conversations)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        print(f"analyzed {len(results)} customer(s) -> {out}")

    for r in results:
        nba = r["next_best_action"]
        print(f"{r['name']} [{r['churn_level']}] "
              f"score={r['churn_score']}")
        if r["signals"]:
            print("  signals: " + ", ".join(
                s["signal"] for s in r["signals"]))
        print(f"  → {nba['action']} عبر {nba['channel']} "
              f"(أولوية {nba['priority']}): {nba['reason']}")


if __name__ == "__main__":
    main()
