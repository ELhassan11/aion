"""Human approvals (P3-T05): the agent pauses before consequential actions.

An ApprovalGate keeps a persistent JSON queue (survives restarts — the
"checkpoint" idea): the agent requests approval, stops, and resumes once
a human approves or rejects.

Risk policy: high-risk actions always need approval; medium needs it
unless --auto-medium; low never does.

Usage:
    python -m agents.approvals --request "send WhatsApp to cust-001" \\
        --details '{"message": "..."}' --risk high
    python -m agents.approvals --list
    python -m agents.approvals --approve apr-1a2b3c
    python -m agents.approvals --reject apr-1a2b3c --reason "too risky"
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEFAULT_QUEUE = Path("out/approvals.json")


@dataclass
class ApprovalRequest:
    id: str
    action: str
    details: str
    risk: str  # low | medium | high
    status: str = "pending"  # pending | approved | rejected
    created_at: str = ""
    decided_by: str = ""
    decided_at: str = ""
    reason: str = ""


def needs_approval(risk: str, auto_medium: bool = False) -> bool:
    """Policy: high always, medium unless auto-approved, low never."""
    if risk == "high":
        return True
    if risk == "medium":
        return not auto_medium
    return False


class ApprovalGate:
    def __init__(self, queue_path: Path = DEFAULT_QUEUE):
        self.queue_path = Path(queue_path)
        self.requests: dict[str, ApprovalRequest] = {}
        self._load()

    def _load(self) -> None:
        if self.queue_path.exists():
            data = json.loads(self.queue_path.read_text(encoding="utf-8"))
            for item in data:
                self.requests[item["id"]] = ApprovalRequest(**item)

    def _save(self) -> None:
        self.queue_path.parent.mkdir(parents=True, exist_ok=True)
        self.queue_path.write_text(
            json.dumps([asdict(r) for r in self.requests.values()],
                       ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def request(self, action: str, details: str = "",
                risk: str = "high") -> ApprovalRequest:
        req = ApprovalRequest(
            id=f"apr-{uuid.uuid4().hex[:6]}",
            action=action,
            details=details,
            risk=risk,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        self.requests[req.id] = req
        self._save()
        return req

    def pending(self) -> list[ApprovalRequest]:
        return [r for r in self.requests.values() if r.status == "pending"]

    def decide(self, request_id: str, approved: bool,
               by: str = "human", reason: str = "") -> ApprovalRequest | None:
        req = self.requests.get(request_id)
        if not req or req.status != "pending":
            return None
        req.status = "approved" if approved else "rejected"
        req.decided_by = by
        req.decided_at = datetime.now(timezone.utc).isoformat()
        req.reason = reason
        self._save()
        return req

    def wait_for(self, request_id: str, timeout: int = 300,
                 poll: int = 5) -> ApprovalRequest | None:
        """Block until the request is decided or timeout (agent pause/resume)."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            self._load()  # re-read: human may decide in another process
            req = self.requests.get(request_id)
            if req and req.status != "pending":
                return req
            time.sleep(poll)
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Human approval gate (P3-T05)")
    ap.add_argument("--queue", default=str(DEFAULT_QUEUE))
    ap.add_argument("--request", dest="request", default=None,
                    help="action description to request approval for")
    ap.add_argument("--details", default="")
    ap.add_argument("--risk", choices=["low", "medium", "high"], default="high")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--approve", default=None, metavar="ID")
    ap.add_argument("--reject", default=None, metavar="ID")
    ap.add_argument("--reason", default="")
    args = ap.parse_args()

    gate = ApprovalGate(Path(args.queue))

    if args.request:
        if not needs_approval(args.risk):
            print(f"risk={args.risk}: no approval needed, proceeding")
            return
        req = gate.request(args.request, args.details, args.risk)
        print(f"approval requested: {req.id}")
        print(f"  action: {req.action} (risk={req.risk})")
        print(f"  decide: python -m agents.approvals --approve {req.id}  |  "
              f"--reject {req.id}")
        return
    if args.list:
        pend = gate.pending()
        print(f"{len(pend)} pending request(s)")
        for r in pend:
            print(f"  {r.id} [{r.risk}] {r.action}")
            if r.details:
                print(f"    details: {r.details[:120]}")
        return
    if args.approve:
        req = gate.decide(args.approve, True, reason=args.reason)
        print(f"{args.approve}: {'approved ✅' if req else 'not found/already decided'}")
        return
    if args.reject:
        req = gate.decide(args.reject, False, reason=args.reason)
        print(f"{args.reject}: {'rejected ❌' if req else 'not found/already decided'}"
              + (f" ({args.reason})" if req and args.reason else ""))
        return
    ap.print_help()


if __name__ == "__main__":
    main()
