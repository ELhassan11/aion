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
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CONFIG = {"index": "out/index.npz", "embedder": "gemini"}


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
        self.end_headers()
        self.wfile.write(body)

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
        else:
            self._not_found()

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
