"""Conversation intelligence (P4-T03): analyze support ticket conversations.

Per ticket: rule-based Arabic sentiment per message, sentiment trajectory,
escalation signals, and an LLM summary. Batch mode writes aggregate
insights JSON.

Usage:
    python -m customer_intelligence.conversation \\
        --tickets data/sample/tickets.json --output out/conversation_insights.json
    python -m customer_intelligence.conversation --ticket-id tkt-001 \\
        --tickets data/sample/tickets.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ingestion.normalize import normalize_arabic  # noqa: E402
from rag.llm import generate  # noqa: E402

_POSITIVE = ["شكرا", "ممتاز", "رائع", "سعيد", "تمام", "كويس", "حلو",
             "مبسوط", "عظيم", "هايل"]
_NEGATIVE = ["سيء", "زفت", "غاضب", "مشكلة", "عيب", "بايظ", "خربان",
             "متضايق", "زعلان", "قرف", "فاشل", "تأخر", "زفت"]


def sentiment(text: str) -> str:
    """Rule-based Arabic sentiment: positive | negative | neutral."""
    t = normalize_arabic(text)
    pos = sum(1 for w in _POSITIVE if w in t)
    neg = sum(1 for w in _NEGATIVE if w in t)
    if pos > neg:
        return "positive"
    if neg > pos:
        return "negative"
    return "neutral"


def escalation_signals(messages: list[dict]) -> list[str]:
    """Detect signals that a conversation needs human attention."""
    signals = []
    texts = " ".join(m.get("text", "") for m in messages
                     if m.get("role") == "customer")
    t = normalize_arabic(texts)
    if any(w in t for w in ["غاضب", "زعلان", "متضايق", "قرف"]):
        signals.append("angry_customer")
    neg_count = sum(1 for m in messages
                    if m.get("role") == "customer"
                    and sentiment(m.get("text", "")) == "negative")
    if neg_count >= 2:
        signals.append("repeated_complaints")
    if any(w in t for w in ["مدير", "مسؤول", "اشتكي"]):
        signals.append("escalation_request")
    return signals


def summarize(ticket: dict) -> str:
    """LLM summary of the conversation."""
    convo = "\n".join(
        f"{m.get('role')}: {m.get('text', '')}"
        for m in ticket.get("messages", []))
    prompt = (
        "لخص محادثة الدعم التالية في سطرين بالعربية: ما المشكلة وما تم.\n\n"
        f"التذكرة: {ticket.get('subject')}\n{convo}"
    )
    return generate(prompt)


def analyze_ticket(ticket: dict, with_summary: bool = True) -> dict:
    messages = ticket.get("messages", [])
    sentiments = [sentiment(m.get("text", "")) for m in messages
                  if m.get("role") == "customer"]
    customer_msgs = [m for m in messages if m.get("role") == "customer"]
    overall = ("negative" if "negative" in sentiments
               else "positive" if "positive" in sentiments else "neutral")
    result = {
        "ticket_id": ticket.get("id"),
        "subject": ticket.get("subject"),
        "status": ticket.get("status"),
        "message_count": len(messages),
        "customer_sentiments": sentiments,
        "overall_sentiment": overall,
        "escalation_signals": escalation_signals(messages),
        "needs_human": bool(escalation_signals(messages)),
    }
    if with_summary and customer_msgs:
        result["summary"] = summarize(ticket)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Conversation intelligence (P4-T03)")
    ap.add_argument("--tickets", required=True)
    ap.add_argument("--ticket-id", default=None)
    ap.add_argument("--output", default=None)
    ap.add_argument("--no-summary", action="store_true",
                    help="skip LLM summaries (faster, offline)")
    args = ap.parse_args()

    tickets = json.loads(Path(args.tickets).read_text(encoding="utf-8"))
    if args.ticket_id:
        tickets = [t for t in tickets if t["id"] == args.ticket_id]
        if not tickets:
            print(f"unknown ticket: {args.ticket_id}")
            return

    results = [analyze_ticket(t, with_summary=not args.no_summary)
               for t in tickets]

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        print(f"analyzed {len(results)} ticket(s) -> {out}")

    neg = sum(1 for r in results if r["overall_sentiment"] == "negative")
    esc = sum(1 for r in results if r["needs_human"])
    print(f"tickets: {len(results)} | negative: {neg} | need human: {esc}")
    for r in results:
        print(f"  {r['ticket_id']}: {r['overall_sentiment']}"
              + (f" ⚠️{','.join(r['escalation_signals'])}"
                 if r["escalation_signals"] else ""))
        if r.get("summary"):
            print(f"    ملخص: {r['summary'][:120]}")


if __name__ == "__main__":
    main()
