"""AION dashboard server (P4-T06). Standard library only.

Serves the Arabic RTL dashboard and a JSON API over the pipeline outputs
(customers, churn actions, conversations, approvals) plus a live chat
endpoint backed by the customer service agent.

Usage:
    python -m frontend.serve --port 8000
    # open http://localhost:8000
"""

from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CONFIG = {"index": "out/index.npz", "embedder": "gemini"}

_MIME = {".html": "text/html", ".js": "text/javascript",
         ".css": "text/css", ".json": "application/json",
         ".png": "image/png", ".svg": "image/svg+xml",
         ".ico": "image/x-icon", ".webp": "image/webp",
         ".woff2": "font/woff2", ".woff": "font/woff"}


def _read_json(name: str):
    path = Path("out") / name
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


class Handler(BaseHTTPRequestHandler):
    server_version = "AION-Dashboard/1.0"

    def _send(self, body: bytes, ctype: str = "application/json"):
        self.send_response(200)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _serve_static(self, rel: str) -> bool:
        """Serve a file from the React dist (inbox UI)."""
        dist = ROOT / "frontend" / "inbox-ui" / "dist"
        target = (dist / rel).resolve()
        if not str(target).startswith(str(dist.resolve())) or not target.is_file():
            return False
        self._send(target.read_bytes(),
                   _MIME.get(target.suffix.lower(), "application/octet-stream"))
        return True

    def _send_json(self, obj):
        self._send(json.dumps(obj, ensure_ascii=False).encode("utf-8"))

    def _not_found(self):
        self.send_response(404)
        self.end_headers()
        self.wfile.write(b"not found")

    def do_GET(self):  # noqa: N802
        path = urlparse(self.path).path
        if path == "/":
            html = (ROOT / "frontend" / "dashboard.html").read_bytes()
            self._send(html, "text/html")
        elif path == "/api/customers":
            self._send_json(_read_json("customer_360_enriched.json"))
        elif path == "/api/actions":
            self._send_json(_read_json("churn_actions.json"))
        elif path == "/api/conversations":
            self._send_json(_read_json("conversation_insights.json"))
        elif path == "/api/approvals":
            items = _read_json("approvals.json")
            if isinstance(items, dict):
                items = list(items.values())
            self._send_json([i for i in items
                             if i.get("status") == "pending"])
        elif path == "/api/tickets":
            qs = parse_qs(urlparse(self.path).query)
            cid = qs.get("customer_id", [""])[0]
            tpath = Path("data/sample/tickets.json")
            tickets = (json.loads(tpath.read_text(encoding="utf-8"))
                       if tpath.exists() else [])
            self._send_json([t for t in tickets
                             if t.get("customer_id") == cid])
        elif path == "/api/followups":
            self._send_json(_read_json("followups.json"))
        elif path == "/api/inbox":
            from frontend.inbox_api import store
            self._send_json(store().list_conversations())
        elif path.startswith("/api/inbox/"):
            from frontend.inbox_api import store
            tid = path[len("/api/inbox/"):]
            conv = store().get_conversation(tid)
            if conv:
                self._send_json(conv)
            else:
                self._not_found()
        elif path == "/inbox" or path == "/inbox/":
            if not self._serve_static("index.html"):
                self._send(
                    b"<h1>Inbox UI not built</h1><p>Run: npm install && npm run build "
                    b"inside frontend/inbox-ui</p>", "text/html")
        elif path.startswith("/inbox/"):
            if not self._serve_static(path[len("/inbox/"):]):
                self._not_found()
        else:
            self._not_found()

    def do_POST(self):  # noqa: N802
        path = urlparse(self.path).path
        length = int(self.headers.get("Content-Length", 0))
        try:
            data = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            data = {}

        if path == "/api/approvals":
            from agents.approvals import ApprovalGate
            gate = ApprovalGate(Path("out/approvals.json"))
            req = gate.decide(data.get("id", ""),
                              bool(data.get("approved")),
                              reason=data.get("reason", ""))
            self._send_json({"ok": req is not None,
                             "status": req.status if req else None})
        elif path == "/api/chat":
            from customer_intelligence.service_agent import serve
            try:
                out = serve(data.get("customer_id", ""),
                            data.get("message", ""),
                            CONFIG["index"], CONFIG["embedder"])
                out["verification"] = {
                    "passed": out["verification"]["passed"],
                    "issues": out["verification"]["issues"],
                }
                self._send_json(out)
            except Exception as exc:  # noqa: BLE001
                self._send_json({"error": str(exc)})
        elif path.startswith("/api/inbox/") and path.endswith("/reply"):
            from frontend.inbox_api import store
            tid = path[len("/api/inbox/"):-len("/reply")]
            msg = store().add_message(tid, "agent", data.get("text", ""))
            self._send_json({"ok": msg is not None,
                             "conversation": store().get_conversation(tid)})
        elif path.startswith("/api/inbox/") and path.endswith("/ai-reply"):
            from customer_intelligence.service_agent import serve
            from frontend.inbox_api import store
            tid = path[len("/api/inbox/"):-len("/ai-reply")]
            conv = store().get_conversation(tid)
            if not conv:
                self._send_json({"error": "unknown conversation"})
                return
            cust_msgs = [m for m in conv["messages"]
                         if m["role"] == "customer"]
            if not cust_msgs:
                self._send_json({"error": "no customer message yet"})
                return
            try:
                res = serve(conv["customer_id"],
                            cust_msgs[-1]["text"],
                            CONFIG["index"], CONFIG["embedder"])
                store().add_message(tid, "agent", res["reply"])
                self._send_json({
                    "ok": True,
                    "reply": res["reply"],
                    "citations": res["citations"],
                    "verification": {
                        "passed": res["verification"]["passed"],
                        "issues": res["verification"]["issues"],
                    },
                    "needs_approval": res["needs_approval"],
                    "conversation": store().get_conversation(tid),
                })
            except Exception as exc:  # noqa: BLE001
                self._send_json({"error": str(exc)})
        elif path.startswith("/api/inbox/") and path.endswith("/close"):
            from frontend.inbox_api import store
            tid = path[len("/api/inbox/"):-len("/close")]
            self._send_json({"ok": store().close(tid)})
        else:
            self._not_found()

    def do_OPTIONS(self):  # noqa: N802
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def log_message(self, *args):  # quieter logs
        pass


def main() -> None:
    ap = argparse.ArgumentParser(description="AION dashboard (P4-T06)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--index", default="out/index.npz")
    ap.add_argument("--embedder", default="gemini",
                    choices=["api", "local", "gemini"])
    args = ap.parse_args()
    CONFIG["index"] = args.index
    CONFIG["embedder"] = args.embedder

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"AION dashboard -> http://localhost:{args.port}")
    print("Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
