"""Inbox API: real tickets as conversations, real AI replies.

Backs the React support dashboard (frontend/inbox-ui). Tickets from
data/sample/tickets.json become conversations; replies go through the
customer service agent (customer_intelligence.service_agent), so the
"AI reply" button produces a real cited answer, not canned text.

Runtime messages are appended to out/inbox_runtime.json so they
survive server restarts (the sample data files stay untouched).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

DATA = Path("data/sample/tickets.json")
CUSTOMERS = Path("data/sample/customers.json")
PROFILES = Path("out/customer_360_enriched.json")
RUNTIME = Path("out/inbox_runtime.json")

STATUS_MAP = {"open": "Open", "in_progress": "Pending", "resolved": "Resolved"}
_COLORS = ["rose", "sand", "peach", "lilac", "mint",
           "blue", "slate", "pink", "tan"]


def _color_for(tid: str) -> str:
    return _COLORS[abs(hash(tid)) % len(_COLORS)]


class InboxStore:
    def __init__(self) -> None:
        self.tickets: dict[str, dict] = {
            t["id"]: t for t in
            json.loads(DATA.read_text(encoding="utf-8"))
        }
        self.customers: dict[str, dict] = {
            c["id"]: c for c in
            json.loads(CUSTOMERS.read_text(encoding="utf-8"))
        }
        self.profiles: dict[str, dict] = {}
        if PROFILES.exists():
            self.profiles = {
                p["id"]: p for p in
                json.loads(PROFILES.read_text(encoding="utf-8"))
            }
        if RUNTIME.exists():
            runtime = json.loads(RUNTIME.read_text(encoding="utf-8"))
            for tid, msgs in runtime.items():
                if tid in self.tickets:
                    self.tickets[tid].setdefault("messages",
                                                 []).extend(msgs)

    def _persist(self, tid: str, msg: dict) -> None:
        runtime = (json.loads(RUNTIME.read_text(encoding="utf-8"))
                   if RUNTIME.exists() else {})
        runtime.setdefault(tid, []).append(msg)
        RUNTIME.parent.mkdir(parents=True, exist_ok=True)
        RUNTIME.write_text(json.dumps(runtime, ensure_ascii=False, indent=2),
                           encoding="utf-8")

    def list_conversations(self) -> list[dict]:
        out = []
        for t in self.tickets.values():
            cust = self.customers.get(t["customer_id"], {})
            msgs = t.get("messages", [])
            last = msgs[-1] if msgs else {}
            name = cust.get("name", t["customer_id"])
            out.append({
                "id": t["id"],
                "customer_id": t["customer_id"],
                "name": name,
                "initials": "".join(
                    w[0] for w in name.split()[:2] if w),
                "color": _color_for(t["id"]),
                "subject": t.get("subject", ""),
                "preview": (last.get("text", "") or "")[:70],
                "time": (last.get("created_at", "") or "")[:16].replace(
                    "T", " "),
                "channel": t.get("channel", "web"),
                "status": STATUS_MAP.get(t.get("status"), "Open"),
                "priority": t.get("priority", ""),
                "unread": (1 if t.get("status") == "open"
                           and last.get("role") == "customer" else 0),
            })
        # open first, then by recency
        order = {"Open": 0, "Pending": 1, "Resolved": 2}
        out.sort(key=lambda c: (order.get(c["status"], 3), c["time"]),
                 reverse=False)
        return out

    def get_conversation(self, tid: str) -> dict | None:
        t = self.tickets.get(tid)
        if not t:
            return None
        cust = self.customers.get(t["customer_id"], {})
        prof = self.profiles.get(t["customer_id"], {})
        return {
            "id": t["id"],
            "customer_id": t["customer_id"],
            "subject": t.get("subject"),
            "status": STATUS_MAP.get(t.get("status"), "Open"),
            "channel": t.get("channel"),
            "priority": t.get("priority"),
            "ai_summary": t.get("ai_summary", ""),
            "messages": [
                {"role": m.get("role"), "text": m.get("text", ""),
                 "time": (m.get("created_at", "") or "")[11:16]}
                for m in t.get("messages", [])
            ],
            "customer": {
                "id": cust.get("id"),
                "name": cust.get("name"),
                "initials": "".join(
                    w[0] for w in cust.get("name", "?").split()[:2]
                    if w),
                "color": _color_for(t["id"]),
                "email": cust.get("email", ""),
                "phone": cust.get("phone", ""),
                "segment": cust.get("segment", ""),
                "tags": cust.get("tags", []),
                "lifetime_value": cust.get("lifetime_value"),
                "churn_score": prof.get("churn_score"),
                "churn_level": prof.get("churn_level"),
                "open_tickets": prof.get("open_ticket_count"),
            },
        }

    def add_message(self, tid: str, role: str, text: str) -> dict | None:
        t = self.tickets.get(tid)
        if not t:
            return None
        msg = {"role": role, "text": text,
               "created_at": datetime.now(timezone.utc).isoformat()}
        t.setdefault("messages", []).append(msg)
        self._persist(tid, msg)
        return msg

    def close(self, tid: str) -> bool:
        t = self.tickets.get(tid)
        if not t:
            return False
        t["status"] = "resolved"
        return True


_STORE: InboxStore | None = None


def store() -> InboxStore:
    global _STORE
    if _STORE is None:
        _STORE = InboxStore()
    return _STORE
