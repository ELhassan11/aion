"""Customer service agent (P4-T02): answers customer messages in Arabic.

Flow: identify customer -> gather profile + tickets -> retrieve policy
docs -> draft reply (LLM, cited) -> verify -> risk check (refunds and
promises need human approval via the P3-T05 gate).

Usage:
    python -m customer_intelligence.service_agent \\
        --customer-id cust-001 --message "الشاحن بايظ عايز استبدله" \\
        --index out/index.npz --embedder gemini
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.verify import verify as verify_answer  # noqa: E402
from rag.citations import extract_citations  # noqa: E402
from rag.llm import generate  # noqa: E402

_RISK_WORDS = ["استرداد", "استرجاع المبلغ", "تعويض", "خصم", "مجانًا",
               "أعتذر نيابة", "نضمن"]


def load_profile(customer_id: str) -> dict | None:
    path = Path("out/customer_360_enriched.json")
    if not path.exists():
        return None
    for p in json.loads(path.read_text(encoding="utf-8")):
        if p["id"] == customer_id:
            return p
    return None


def retrieve_docs(query: str, index_path: str, embedder_name: str,
                  k: int = 3) -> list[tuple[str, str]]:
    from rag.embeddings import ApiEmbedder, GeminiEmbedder, LocalEmbedder
    from rag.hybrid import BM25, rrf_fuse, tokenize
    from rag.index import VectorIndex

    if embedder_name == "local":
        embedder = LocalEmbedder("BAAI/bge-m3")
    elif embedder_name == "gemini":
        embedder = GeminiEmbedder("gemini-embedding-001")
    else:
        embedder = ApiEmbedder("openai/text-embedding-3-small")

    index = VectorIndex.load(Path(index_path))
    bm25 = BM25([tokenize(t) for t in index.texts])
    by_id = dict(zip(index.ids, index.texts, strict=True))
    (qvec,) = embedder.embed([query])
    vr = [c for c, _, _ in index.search(qvec, k=10)]
    kr = bm25.rank(query, index.ids, 10)
    fused = rrf_fuse([vr, kr])[:k]
    return [(c, by_id[c]) for c, _ in fused if c in by_id]


def draft_reply(customer: dict, message: str,
                docs: list[tuple[str, str]]) -> str:
    context = "\n\n".join(f"[{c}]\n{t[:600]}" for c, t in docs)
    prompt = (
        "أنت وكيل خدمة عملاء متجر النيل. رد على رسالة العميل بالعربية "
        "(فصحى مبسطة ودودة). استخدم بيانات العميل والمستندات فقط. "
        "ضع معرف المصدر بين قوسين مربعين بعد كل معلومة، مثال [doc-xxx]. "
        "لا تعد باسترداد أو خصم — اقترح الخطوة التالية فقط.\n\n"
        f"العميل: {customer.get('name')} "
        f"(الشريحة: {customer.get('segment')}, "
        f"التذاكر: {customer.get('ticket_subjects', [])})\n"
        f"رسالة العميل: {message}\n\n"
        f"المستندات:\n{context}"
    )
    return generate(prompt)


def assess_risk(reply: str) -> str:
    """High risk if the draft promises money-related actions."""
    return "high" if any(w in reply for w in _RISK_WORDS) else "low"


def serve(customer_id: str, message: str, index_path: str,
          embedder_name: str) -> dict:
    customer = load_profile(customer_id) or {"id": customer_id,
                                             "name": customer_id}
    docs = retrieve_docs(message, index_path, embedder_name)
    reply = draft_reply(customer, message, docs)
    evidence = {c: t for c, t in docs}
    report = verify_answer(reply, evidence)
    risk = assess_risk(reply)
    return {
        "customer": customer.get("name"),
        "reply": reply,
        "citations": extract_citations(reply),
        "verification": report,
        "risk": risk,
        "needs_approval": risk == "high",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Customer service agent (P4-T02)")
    ap.add_argument("--customer-id", required=True)
    ap.add_argument("--message", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--embedder", choices=["api", "local", "gemini"],
                    default="gemini")
    args = ap.parse_args()

    out = serve(args.customer_id, args.message, args.index, args.embedder)
    print(f"العميل: {out['customer']}\n")
    print("الرد:")
    print(out["reply"])
    print(f"\nالتحقق: {'✅ سليم' if out['verification']['passed'] else '❌ فيه مشاكل'}")
    for issue in out["verification"]["issues"]:
        print(f"  - {issue}")
    print(f"المخاطر: {out['risk']}"
          + (" — يحتاج موافقة بشرية ⚠️" if out["needs_approval"] else ""))


if __name__ == "__main__":
    main()
